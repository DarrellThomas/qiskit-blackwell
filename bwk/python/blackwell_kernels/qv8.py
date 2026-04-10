# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell
#
# QV-8: Quantum Volume 8-qubit simulation.
#
# Custom CUDA kernel wrapper + PyTorch reference + circuit generator.

import math
import os
import torch
import numpy as np

N_QUBITS = 8
STATE_SIZE = 1 << N_QUBITS  # 256

_cuda_mod = None


def _load_cuda_module():
    """Load the CUDA extension, JIT-compiling if the pre-built .so is missing."""
    global _cuda_mod
    if _cuda_mod is not None:
        return _cuda_mod
    try:
        from blackwell_kernels._C import qv8_simulate as _fn
        import blackwell_kernels._C as mod
        _cuda_mod = mod
        return mod
    except ImportError:
        pass
    # JIT compile from source
    from torch.utils.cpp_extension import load
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(pkg_dir))
    src = os.path.join(project_root, "csrc", "qv8", "qv8_sim_sm120.cu")
    inc = os.path.join(project_root, "csrc", "common")
    _cuda_mod = load(
        name="qv8_cuda",
        sources=[src],
        extra_include_paths=[inc],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math",
            "-gencode", "arch=compute_120a,code=sm_120a",
            "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
        ],
        verbose=False,
    )
    return _cuda_mod


def _random_haar_u4_batch(rng, count):
    """Generate *count* Haar-random U(4) matrices via batched QR.

    Returns complex64 array of shape [count, 4, 4].
    Uses batched np.linalg.qr — orders of magnitude faster than
    looping with per-gate QR decompositions.

    We produce U(4) (not SU(4)): the global phase from det != 1
    has no effect on measurement probabilities, so the SU(4)
    normalization is unnecessary for quantum volume evaluation.
    """
    z = (rng.standard_normal((count, 4, 4))
         + 1j * rng.standard_normal((count, 4, 4))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    # Haar phase correction: Q → Q * diag(d / |d|)
    d = np.diagonal(r, axis1=-2, axis2=-1)          # [count, 4]
    ph = d / np.abs(d)                               # unit-phase diag
    q = q * ph[:, np.newaxis, :]                     # broadcast over rows
    return q.astype(np.complex64)


def generate_qv8_circuits(num_circuits, num_layers=N_QUBITS, seed=42):
    """Generate random QV-8 circuits (batched, fast).

    Each layer applies N_QUBITS/2 = 4 random Haar-U(4) gates to random
    qubit pairs (a permutation of all 8 qubits, split into pairs).

    Returns:
        gate_matrices: [C, G, 4, 4, 2] float32 tensor (real/imag last dim)
        gate_qubits:   [C, G, 2] int32 tensor
        num_gates_per_circuit: int (= num_layers * 4)
    """
    rng = np.random.default_rng(seed)
    num_gates = num_layers * (N_QUBITS // 2)

    # Qubit permutations per layer (shared across circuits, standard for QV)
    layer_perms = [rng.permutation(N_QUBITS) for _ in range(num_layers)]

    # Batched Haar-random U(4) generation — single call replaces C*G python loops
    total = num_circuits * num_gates
    u_batch = _random_haar_u4_batch(rng, total)       # [total, 4, 4] complex64
    u_batch = u_batch.reshape(num_circuits, num_gates, 4, 4)

    # Pack real/imag into [C, G, 4, 4, 2]
    all_matrices = np.stack([u_batch.real, u_batch.imag], axis=-1)

    # Build qubit index array
    all_qubits = np.zeros((num_circuits, num_gates, 2), dtype=np.int32)
    g_idx = 0
    for layer in range(num_layers):
        perm = layer_perms[layer]
        for pair in range(N_QUBITS // 2):
            all_qubits[:, g_idx, 0] = int(perm[2 * pair])
            all_qubits[:, g_idx, 1] = int(perm[2 * pair + 1])
            g_idx += 1

    gate_matrices = torch.from_numpy(np.ascontiguousarray(all_matrices))
    gate_qubits = torch.from_numpy(np.ascontiguousarray(all_qubits))
    return gate_matrices, gate_qubits, num_gates


def _gram_schmidt_u4_gpu(v_r, v_i):
    """Batched Gram-Schmidt on 4 complex vectors of length 4, using real arithmetic.

    Args:
        v_r: [N, 4, 4] float32 CUDA — real parts (v_r[:, col, row])
        v_i: [N, 4, 4] float32 CUDA — imag parts

    Returns:
        q_r, q_i: [N, 4, 4] float32 CUDA — orthonormalized columns
    """
    q_r = torch.empty_like(v_r)
    q_i = torch.empty_like(v_i)
    for j in range(4):
        # Start with column j
        u_r = v_r[:, j]   # [N, 4]
        u_i = v_i[:, j]
        # Subtract projections onto previous orthonormal columns
        for k in range(j):
            qk_r = q_r[:, k]  # [N, 4]
            qk_i = q_i[:, k]
            # Complex inner product <q_k, u> = sum(qk_r*u_r + qk_i*u_i) + i*sum(qk_r*u_i - qk_i*u_r)
            dot_r = (qk_r * u_r + qk_i * u_i).sum(-1, keepdim=True)  # [N, 1]
            dot_i = (qk_r * u_i - qk_i * u_r).sum(-1, keepdim=True)
            # u -= dot * q_k  (complex multiply: (a+ib)(c+id) = (ac-bd)+i(ad+bc))
            u_r = u_r - (dot_r * qk_r - dot_i * qk_i)
            u_i = u_i - (dot_r * qk_i + dot_i * qk_r)
        # Normalize
        norm = (u_r * u_r + u_i * u_i).sum(-1, keepdim=True).sqrt()  # [N, 1]
        q_r[:, j] = u_r / norm
        q_i[:, j] = u_i / norm
    return q_r, q_i


def generate_qv8_circuits_gpu(num_circuits, num_layers=N_QUBITS, seed=42, device='cuda'):
    """Generate random QV-8 circuits entirely on GPU.

    Uses Gram-Schmidt orthogonalization of random Gaussian vectors in real
    arithmetic — avoids CPU bottleneck and broken PyTorch complex CUDA JIT.

    Returns:
        gate_matrices: [C, G, 4, 4, 2] float32 CUDA tensor
        gate_qubits:   [C, G, 2] int32 CUDA tensor
        num_gates_per_circuit: int
    """
    num_gates = num_layers * (N_QUBITS // 2)
    total = num_circuits * num_gates

    # Generate random complex Gaussian vectors on GPU
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    v_r = torch.randn(total, 4, 4, device=device, generator=gen)
    v_i = torch.randn(total, 4, 4, device=device, generator=gen)

    # Gram-Schmidt → Haar-random U(4) in real arithmetic
    q_r, q_i = _gram_schmidt_u4_gpu(v_r, v_i)

    # Reshape and pack: columns are stored as q[:, col, row], need [C, G, row, col, 2]
    q_r = q_r.reshape(num_circuits, num_gates, 4, 4).permute(0, 1, 3, 2)
    q_i = q_i.reshape(num_circuits, num_gates, 4, 4).permute(0, 1, 3, 2)
    gate_matrices = torch.stack([q_r, q_i], dim=-1).contiguous()

    # Qubit permutations (tiny — CPU is fine)
    rng = np.random.default_rng(seed)
    layer_perms = [rng.permutation(N_QUBITS) for _ in range(num_layers)]
    gate_qubits = torch.zeros(num_circuits, num_gates, 2, dtype=torch.int32, device=device)
    g_idx = 0
    for layer in range(num_layers):
        perm = layer_perms[layer]
        for pair in range(N_QUBITS // 2):
            gate_qubits[:, g_idx, 0] = int(perm[2 * pair])
            gate_qubits[:, g_idx, 1] = int(perm[2 * pair + 1])
            g_idx += 1

    return gate_matrices, gate_qubits, num_gates


def qv8_simulate(gate_matrices, gate_qubits, num_circuits):
    """Run fused CUDA QV-8 simulation.

    Args:
        gate_matrices: [C, G, 4, 4, 2] float32 CUDA tensor
        gate_qubits:   [C, G, 2] int32 CUDA tensor
        num_circuits:  number of circuits

    Returns:
        probs: [C, 256] float32 tensor of output probabilities
    """
    mod = _load_cuda_module()
    return mod.qv8_simulate(gate_matrices, gate_qubits, num_circuits)


def qv8_simulate_ref(gate_matrices, gate_qubits, num_circuits):
    """PyTorch reference QV-8 simulation (batched, vectorized).

    Same interface as qv8_simulate but uses PyTorch operations.
    """
    device = gate_matrices.device
    C = num_circuits
    G = gate_matrices.size(1)
    N = STATE_SIZE

    # Build complex gate matrices: [C, G, 4, 4]
    gates_complex = torch.complex(gate_matrices[..., 0], gate_matrices[..., 1])

    # Initialize |0⟩^⊗8
    state = torch.zeros(C, N, dtype=torch.complex64, device=device)
    state[:, 0] = 1.0 + 0.0j

    for g in range(G):
        q0 = gate_qubits[0, g, 0].item()  # same qubit layout across batch
        q1 = gate_qubits[0, g, 1].item()

        mask_q0 = 1 << q0
        mask_q1 = 1 << q1

        # Build index arrays for the 4 sub-populations
        all_idx = torch.arange(N, device=device)
        bases = all_idx[((all_idx & mask_q0) == 0) & ((all_idx & mask_q1) == 0)]

        i00 = bases
        i01 = bases | mask_q0
        i10 = bases | mask_q1
        i11 = bases | mask_q0 | mask_q1

        # Gather: [C, N/4, 4]
        amps = torch.stack([state[:, i00], state[:, i01],
                            state[:, i10], state[:, i11]], dim=-1)

        # Per-circuit gate: [C, 4, 4]
        gate = gates_complex[:, g]

        # Apply: [C, N/4, 4] @ [C, 4, 4]^T → [C, N/4, 4]
        new_amps = torch.bmm(amps, gate.transpose(-1, -2))

        # Scatter back
        state[:, i00] = new_amps[..., 0]
        state[:, i01] = new_amps[..., 1]
        state[:, i10] = new_amps[..., 2]
        state[:, i11] = new_amps[..., 3]

    # Return probabilities
    return (state.real ** 2 + state.imag ** 2)
