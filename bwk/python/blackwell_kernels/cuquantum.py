# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell
#
# Python wrapper for quantum state vector gate application kernel.

import torch
from blackwell_kernels._C import apply_gate as _apply_gate
from blackwell_kernels._C import apply_gates_fused as _apply_gates_fused
from blackwell_kernels._C import apply_gates_reg_tiled as _apply_gates_reg_tiled
from blackwell_kernels._C import apply_gate_2q as _apply_gate_2q
from blackwell_kernels._C import apply_diagonal as _apply_diagonal
from blackwell_kernels._C import apply_diagonal_phase as _apply_diagonal_phase
from blackwell_kernels._C import apply_diagonal_2q as _apply_diagonal_2q
from blackwell_kernels._C import state_init as _state_init
from blackwell_kernels._C import measure_qubit as _measure_qubit
from blackwell_kernels._C import measure_probs as _measure_probs
from blackwell_kernels._C import sample as _sample
from blackwell_kernels._C import expectation_z as _expectation_z
from blackwell_kernels._C import expectation_pauli as _expectation_pauli
from blackwell_kernels._C import apply_mcgate as _apply_mcgate
from blackwell_kernels._C import apply_mc2qgate as _apply_mc2qgate
from blackwell_kernels._C import apply_depolarizing as _apply_depolarizing
from blackwell_kernels._C import apply_amplitude_damping as _apply_amplitude_damping
from blackwell_kernels._C import apply_dephasing as _apply_dephasing
from blackwell_kernels._C import apply_gate_batched as _apply_gate_batched
from blackwell_kernels._C import apply_gates_batched_fused as _apply_gates_batched_fused
from blackwell_kernels._C import state_init_batched as _state_init_batched
from blackwell_kernels._C import renormalize as _renormalize
# Tier 3: Chebyshev Hamiltonian evolution (experimental/beta)
from blackwell_kernels._C import spmv_csr as _spmv_csr
from blackwell_kernels._C import chebyshev_step as _chebyshev_step
from blackwell_kernels._C import chebyshev_accum as _chebyshev_accum
from blackwell_kernels._C import chebyshev_step_accum as _chebyshev_step_accum


def apply_gate(state: torch.Tensor, gate: torch.Tensor, target_qubit: int) -> torch.Tensor:
    """Apply a single-qubit gate to a state vector.

    Args:
        state: complex64 tensor of shape [2^n], modified in-place
        gate: complex64 tensor of shape [2, 2], unitary matrix
        target_qubit: which qubit to apply the gate to (0-indexed)

    Returns:
        state (modified in-place)
    """
    return _apply_gate(state, gate, target_qubit)


def apply_gate_2q(state: torch.Tensor, gate: torch.Tensor, qubit0: int, qubit1: int) -> torch.Tensor:
    """Apply a two-qubit gate to a state vector.

    Args:
        state: complex64 tensor of shape [2^n], modified in-place
        gate: complex64 tensor of shape [4, 4], unitary matrix
        qubit0: first qubit (0-indexed)
        qubit1: second qubit (0-indexed, must differ from qubit0)

    Returns:
        state (modified in-place)
    """
    return _apply_gate_2q(state, gate, qubit0, qubit1)


def apply_gates_fused(state: torch.Tensor, gates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Apply multiple gates fused in a single kernel launch (tile-based).

    Args:
        state: complex64 tensor of shape [2^n], modified in-place
        gates: complex64 tensor of shape [N, 2, 2], N gate matrices
        targets: int tensor of shape [N], target qubits for each gate

    Returns:
        state (modified in-place)
    """
    return _apply_gates_fused(state, gates, targets.to(torch.int32))


def apply_gates_reg_tiled(state: torch.Tensor, gates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Apply multiple low-qubit gates fused in registers (Phase 9a).

    Each thread loads a tile of amplitudes into registers, applies ALL gates
    without round-tripping through L2/DRAM, writes back once. Only works for
    gates targeting qubits 0-4 (max target < 5).

    Args:
        state: complex64 tensor of shape [2^n], modified in-place
        gates: complex64 tensor of shape [N, 2, 2], N gate matrices (contiguous)
        targets: int32 tensor of shape [N], target qubits (all must be < 5, contiguous, CUDA)

    Returns:
        state (modified in-place)
    """
    return _apply_gates_reg_tiled(state, gates, targets)


class CircuitGraph:
    """CUDA graph-captured circuit: N gate applications replayed with near-zero launch overhead."""

    def __init__(self, state: torch.Tensor, gates: torch.Tensor, targets):
        """Capture a circuit into a CUDA graph.

        Args:
            state: complex64 tensor [2^n] — MUST be the same tensor for all replays
            gates: complex64 tensor [N, 2, 2]
            targets: list of ints or int tensor, target qubits
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

        # Warmup (populate caches, JIT)
        for i, t in enumerate(targets):
            apply_gate(state, gates[i], int(t))
        torch.cuda.synchronize()

        # Capture graph
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            for i, t in enumerate(targets):
                apply_gate(state, gates[i], int(t))

        self._state = state

    def replay(self):
        """Replay the captured circuit (near-zero CPU overhead)."""
        self._graph.replay()


def hadamard_gate(device="cuda"):
    """Return the Hadamard gate as a 2x2 complex64 tensor."""
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=device) / (2 ** 0.5)
    return h


def random_unitary(device="cuda"):
    """Return a random 2x2 unitary matrix (Haar-distributed).
    Generated on CPU to avoid PyTorch complex abs() JIT issues on sm_120a."""
    z = torch.randn(2, 2, dtype=torch.complex64)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    ph = d / d.abs()
    q = q * ph.unsqueeze(0)
    return q.to(device)


def cnot_gate(device="cuda"):
    """Return the CNOT (CX) gate as a 4x4 complex64 tensor.
    |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>"""
    g = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                     dtype=torch.complex64, device=device)
    return g


def cz_gate(device="cuda"):
    """Return the CZ gate as a 4x4 complex64 tensor.
    Diagonal: phases |11> by -1."""
    g = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],
                     dtype=torch.complex64, device=device)
    return g


def swap_gate(device="cuda"):
    """Return the SWAP gate as a 4x4 complex64 tensor.
    Swaps |01> and |10>."""
    g = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
                     dtype=torch.complex64, device=device)
    return g


def apply_diagonal(state: torch.Tensor, diag: torch.Tensor, target_qubit: int) -> torch.Tensor:
    """Apply a diagonal single-qubit gate to a state vector.

    Args:
        state: complex64 tensor of shape [2^n], modified in-place
        diag: complex64 tensor of shape [2], diagonal entries
        target_qubit: which qubit to apply the gate to (0-indexed)

    Returns:
        state (modified in-place)
    """
    return _apply_diagonal(state, diag, target_qubit)


def apply_diagonal_phase(state: torch.Tensor, phase: complex, target_qubit: int) -> torch.Tensor:
    """Apply a phase-type diagonal gate — half the memory traffic.

    Only modifies amplitudes where bit target_qubit is 1. For Z, T, S, Phase gates.
    Phase is the complex scalar to multiply bit=1 amplitudes by (diag[1]).
    Pass a Python complex directly — no GPU tensor needed.
    """
    return _apply_diagonal_phase(state, phase.real, phase.imag, target_qubit)


def apply_diagonal_2q(state: torch.Tensor, diag: torch.Tensor, qubit0: int, qubit1: int) -> torch.Tensor:
    """Apply a diagonal two-qubit gate to a state vector.

    Args:
        state: complex64 tensor of shape [2^n], modified in-place
        diag: complex64 tensor of shape [4], diagonal entries
        qubit0: first qubit (0-indexed)
        qubit1: second qubit (0-indexed)

    Returns:
        state (modified in-place)
    """
    return _apply_diagonal_2q(state, diag, qubit0, qubit1)


def state_init(n_qubits: int) -> torch.Tensor:
    """Initialize |0>^n state vector: state[0] = 1+0j, all others = 0.

    Returns:
        complex64 tensor of shape [2^n] on CUDA
    """
    return _state_init(n_qubits)


def rz_gate_diag(theta, device="cuda"):
    """Rz(theta) diagonal: [exp(-i*theta/2), exp(i*theta/2)]."""
    import cmath
    d0 = cmath.exp(-1j * theta / 2)
    d1 = cmath.exp(1j * theta / 2)
    return torch.tensor([d0, d1], dtype=torch.complex64, device=device)


def phase_gate_diag(theta, device="cuda"):
    """Phase(theta) diagonal: [1, exp(i*theta)]."""
    import cmath
    return torch.tensor([1.0, cmath.exp(1j * theta)], dtype=torch.complex64, device=device)


def t_gate_diag(device="cuda"):
    """T gate diagonal: [1, exp(i*pi/4)]."""
    import cmath, math
    return torch.tensor([1.0, cmath.exp(1j * math.pi / 4)], dtype=torch.complex64, device=device)


def s_gate_diag(device="cuda"):
    """S gate diagonal: [1, i]."""
    return torch.tensor([1.0, 1j], dtype=torch.complex64, device=device)


def z_gate_diag(device="cuda"):
    """Z gate diagonal: [1, -1]."""
    return torch.tensor([1.0, -1.0], dtype=torch.complex64, device=device)


def cz_gate_diag(device="cuda"):
    """CZ gate diagonal: [1, 1, 1, -1]."""
    return torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.complex64, device=device)


def measure_qubit(state: torch.Tensor, target_qubit: int, random_val: float) -> int:
    """Measure target_qubit. Collapse state vector in-place. Return 0 or 1.

    Args:
        state: complex64 tensor [2^n], modified in-place (collapsed)
        target_qubit: which qubit to measure (0-indexed)
        random_val: float in [0, 1) — pre-generated random number for determinism

    Returns:
        int (0 or 1): measurement outcome
    """
    return int(_measure_qubit(state, target_qubit, float(random_val)))


def measure_probs(state: torch.Tensor) -> torch.Tensor:
    """Compute probability vector from state vector.

    Args:
        state: complex64 tensor [2^n] — NOT modified

    Returns:
        float32 tensor [2^n]: prob[i] = |state[i]|^2
    """
    return _measure_probs(state)


def sample(state: torch.Tensor, n_shots: int) -> torch.Tensor:
    """Sample n_shots bitstrings from |state|^2 distribution.

    Args:
        state: complex64 tensor [2^n] — NOT modified
        n_shots: number of samples to draw

    Returns:
        int64 tensor [n_shots]: sampled bitstring indices
    """
    rand_vals = torch.rand(n_shots, dtype=torch.float32, device=state.device)
    return _sample(state, rand_vals)


def apply_mcgate(state: torch.Tensor, gate: torch.Tensor,
                 controls, target_qubit: int) -> torch.Tensor:
    """Apply a multi-controlled 2x2 gate.

    The gate is applied to target_qubit only when all control qubits are |1>.

    Args:
        state: complex64 tensor [2^n], modified in-place
        gate: complex64 tensor [2, 2], 2x2 unitary gate
        controls: list of ints or int tensor, control qubit indices (0-7 qubits)
        target_qubit: target qubit index

    Returns:
        state (modified in-place)
    """
    # Controls are used only for CPU-side index computation — keep on CPU.
    if not isinstance(controls, torch.Tensor):
        controls = torch.tensor(controls, dtype=torch.int64)  # CPU tensor
    else:
        controls = controls.to(torch.int64).cpu()
    return _apply_mcgate(state, gate, controls, target_qubit)


def apply_mc2qgate(state: torch.Tensor, gate: torch.Tensor,
                    controls, target0: int, target1: int) -> torch.Tensor:
    """Apply a multi-controlled 4x4 gate (Phase 10).

    The 4x4 gate is applied to (target0, target1) only when all control qubits are |1>.
    Covers CSWAP/Fredkin (1 control + SWAP), CC-SWAP, controlled-iSWAP, etc.

    Args:
        state: complex64 tensor [2^n], modified in-place
        gate: complex64 tensor [4, 4], 4x4 unitary gate
        controls: list of ints or int tensor, control qubit indices (0-7 qubits)
        target0, target1: target qubit indices
    """
    if not isinstance(controls, torch.Tensor):
        controls = torch.tensor(controls, dtype=torch.int64)
    else:
        controls = controls.to(torch.int64).cpu()
    return _apply_mc2qgate(state, gate, controls, target0, target1)


def swap_gate(device="cuda"):
    """SWAP gate matrix (4x4)."""
    g = torch.zeros(4, 4, dtype=torch.complex64, device=device)
    g[0, 0] = 1; g[1, 2] = 1; g[2, 1] = 1; g[3, 3] = 1
    return g


def cswap_gate(device="cuda"):
    """Alias for SWAP gate — use with apply_mc2qgate(state, swap, [control], t0, t1)."""
    return swap_gate(device)


def apply_depolarizing(state: torch.Tensor, target_qubit: int,
                        p: float, random_val: float = None) -> int:
    """Apply depolarizing noise channel (Phase 11).

    With probability (1-p): identity. With probability p/3 each: Pauli X, Y, or Z.

    Args:
        state: complex64 tensor [2^n], modified in-place
        target_qubit: qubit to apply noise to
        p: error probability [0, 1]
        random_val: pre-generated uniform [0, 1). If None, generated automatically.

    Returns:
        int: which Pauli was applied (0=I, 1=X, 2=Y, 3=Z)
    """
    if random_val is None:
        random_val = torch.rand(1).item()
    return _apply_depolarizing(state, target_qubit, p, random_val)


def apply_amplitude_damping(state: torch.Tensor, target_qubit: int,
                             gamma: float, random_val: float = None) -> int:
    """Apply amplitude damping (T1 decay) noise channel (Phase 11).

    Models energy relaxation: |1⟩ decays to |0⟩ with probability γ.
    γ = 1 - exp(-t/T1).

    Args:
        state: complex64 tensor [2^n], modified in-place
        target_qubit: qubit to apply noise to
        gamma: decay probability [0, 1]
        random_val: pre-generated uniform [0, 1). If None, generated automatically.

    Returns:
        int: which branch was taken (0=no decay, 1=decay)
    """
    if random_val is None:
        random_val = torch.rand(1).item()
    return _apply_amplitude_damping(state, target_qubit, gamma, random_val)


def apply_dephasing(state: torch.Tensor, target_qubit: int,
                     lam: float, random_val: float = None) -> int:
    """Apply dephasing (T2 phase decoherence) noise channel (Phase 11).

    Models phase decoherence: off-diagonal elements of density matrix decay.
    λ = 1 - exp(-t/T2).

    Args:
        state: complex64 tensor [2^n], modified in-place
        target_qubit: qubit to apply noise to
        lam: dephasing probability [0, 1]
        random_val: pre-generated uniform [0, 1). If None, generated automatically.

    Returns:
        int: which branch was taken (0=no error, 1=phase error)
    """
    if random_val is None:
        random_val = torch.rand(1).item()
    return _apply_dephasing(state, target_qubit, lam, random_val)


def apply_gate_batched(states: torch.Tensor, gate: torch.Tensor, target_qubit: int) -> torch.Tensor:
    """Apply a single-qubit gate to N independent state vectors.

    Args:
        states: complex64 tensor of shape [N, 2^n], modified in-place
        gate: complex64 tensor of shape [2, 2], unitary matrix
        target_qubit: which qubit to apply the gate to (0-indexed)

    Returns:
        states (modified in-place)
    """
    return _apply_gate_batched(states, gate, target_qubit)


def apply_gates_batched_fused(states: torch.Tensor, gates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Apply multiple gates to N batched states in one fused kernel launch (Phase 9c).

    Combines Phase 9a register-tiled fusion with Phase 8 batch grid.

    Args:
        states: complex64 tensor [N, 2^n], modified in-place
        gates: complex64 tensor [K, 2, 2], K gate matrices (contiguous)
        targets: int32 tensor [K], target qubits (all < 5, contiguous, CUDA)

    Returns:
        states (modified in-place)
    """
    return _apply_gates_batched_fused(states, gates, targets)


def state_init_batched(n_states: int, n_qubits: int) -> torch.Tensor:
    """Initialize N state vectors to |0>^n.

    Returns:
        complex64 tensor of shape [N, 2^n] on CUDA
    """
    return _state_init_batched(n_states, n_qubits)


def pauli_x_gate(device="cuda"):
    """Return the Pauli-X (NOT) gate as a 2x2 complex64 tensor."""
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)


def expectation_z(state: torch.Tensor, z_mask: int) -> float:
    """Compute Pauli Z-string expectation value <psi|Z_z_mask|psi>.

    Args:
        state: complex64 tensor [2^n] — NOT modified
        z_mask: bitmask of qubits with Z operator (others = identity)
                e.g. z_mask=0b101 applies Z to qubits 0 and 2

    Returns:
        float: expectation value in [-1, +1]
    """
    return float(_expectation_z(state, z_mask))


def expectation_pauli(state: torch.Tensor, z_mask: int, xy_qubit: int, op: int) -> float:
    """Compute expectation value for a Pauli string with one X or Y qubit.

    Computes <psi| (Z on z_mask qubits) * (op on xy_qubit) |psi>.
    For pure Z strings (no X/Y), use expectation_z() instead.

    Args:
        state: complex64 tensor [2^n] — NOT modified
        z_mask: bitmask of qubits with Z operator (should not include xy_qubit)
        xy_qubit: qubit index with X or Y operator
        op: 0=I/Z (redirects to expectation_z), 1=X, 2=Y

    Returns:
        float: expectation value
    """
    return float(_expectation_pauli(state, z_mask, xy_qubit, op))


def random_unitary_4x4(device="cuda"):
    """Return a random 4x4 unitary matrix (Haar-distributed).
    Generated on CPU to avoid PyTorch complex abs() JIT issues on sm_120a."""
    z = torch.randn(4, 4, dtype=torch.complex64)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    ph = d / d.abs()
    q = q * ph.unsqueeze(0)
    return q.to(device)


def renormalize(state: torch.Tensor) -> torch.Tensor:
    """Renormalize state vector to unit norm.

    Corrects accumulated floating-point drift from repeated gate application.
    Two-pass GPU kernel: streaming reduction for ||psi||^2, then scale by 1/sqrt(norm).

    Args:
        state: complex64 tensor of shape [2^n], modified in-place

    Returns:
        state (modified in-place)
    """
    return _renormalize(state)


# ─── Tier 3: Chebyshev Hamiltonian evolution (experimental/beta) ─────

def spmv_csr(row_ptr, col_idx, values, x, y,
             alpha_re=1.0, alpha_im=0.0, beta_re=0.0, beta_im=0.0):
    """Sparse matrix-vector multiply: y = alpha * A * x + beta * y.

    EXPERIMENTAL (beta). Part of the Chebyshev evolution engine.

    Args:
        row_ptr: int32 [n_rows+1], col_idx: int32 [nnz], values: complex64 [nnz]
        x: complex64 [n_rows] input, y: complex64 [n_rows] output (in-place)
        alpha_re, alpha_im: complex scalar for A*x term
        beta_re, beta_im: complex scalar for y term
    """
    _spmv_csr(row_ptr, col_idx, values, x, y,
              alpha_re, alpha_im, beta_re, beta_im)


def chebyshev_step(spmv_result, t_prev, t_next):
    """Chebyshev recurrence: t_next = 2 * spmv_result - t_prev.

    EXPERIMENTAL (beta).
    """
    _chebyshev_step(spmv_result, t_prev, t_next)


def chebyshev_accum(accum, t_k, c_re, c_im):
    """Chebyshev accumulation: accum += (c_re + i*c_im) * t_k.

    EXPERIMENTAL (beta).
    """
    _chebyshev_accum(accum, t_k, c_re, c_im)


def chebyshev_step_accum(spmv_result, t_prev, t_next, accum, c_re, c_im):
    """Fused Chebyshev step + accumulation. Saves one memory pass.

    EXPERIMENTAL (beta).
    """
    _chebyshev_step_accum(spmv_result, t_prev, t_next, accum, c_re, c_im)
