# Copyright (c) 2026 Darrell Thomas. MIT License.
"""QV-4: Quantum Volume 4-qubit simulation — Python wrapper + reference."""

import numpy as np
import torch
from scipy.stats import unitary_group

import blackwell_kernels._C as _C

# ── Qubit pair encoding ────────────────────────────────────────────────────────
# Map (qa, qb) with qa < qb to pair index 0-5.
PAIR_MAP = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}

# The 3 possible pair assignments per layer (how 4 qubits split into 2 pairs)
PAIR_ASSIGNMENTS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
]


def generate_qv4_circuits(n_circuits, rng=None):
    """Generate n_circuits random QV-4 circuits.

    Returns:
        gate_data: float32 ndarray [n_circuits, 8, 32] — 8 gates, each 16 re + 16 im
        pair_ids:  int32 ndarray [n_circuits, 8] — qubit pair index per gate
    """
    if rng is None:
        rng = np.random.default_rng()

    gate_data = np.empty((n_circuits, 8, 32), dtype=np.float32)
    pair_ids = np.empty((n_circuits, 8), dtype=np.int32)

    for c in range(n_circuits):
        for layer in range(4):
            # Random permutation → determines pair assignment
            perm = rng.permutation(4)
            qa0, qb0 = sorted([perm[0], perm[1]])
            qa1, qb1 = sorted([perm[2], perm[3]])

            for sub, (qa, qb) in enumerate([(qa0, qb0), (qa1, qb1)]):
                gidx = layer * 2 + sub
                # Random SU(4) matrix from Haar measure
                u = unitary_group.rvs(4, random_state=rng).astype(np.complex64)
                gate_data[c, gidx, :16] = u.real.ravel()
                gate_data[c, gidx, 16:] = u.imag.ravel()
                pair_ids[c, gidx] = PAIR_MAP[(qa, qb)]

    return gate_data, pair_ids


def qv4_simulate_cuda(gate_data, pair_ids):
    """Run batched QV-4 simulation on GPU.

    Args:
        gate_data: float32 tensor or ndarray [n_circuits, 8, 32]
        pair_ids:  int32 tensor or ndarray [n_circuits, 8]

    Returns:
        probs: float32 tensor [n_circuits, 16] — output probabilities
    """
    if isinstance(gate_data, np.ndarray):
        gate_data = torch.from_numpy(gate_data).cuda()
    if isinstance(pair_ids, np.ndarray):
        pair_ids = torch.from_numpy(pair_ids).cuda()

    n_circuits = gate_data.shape[0]
    n_gates = gate_data.shape[1]
    return _C.qv4_simulate(gate_data.contiguous(), pair_ids.contiguous(),
                           n_circuits, n_gates)


def qv4_simulate_numpy(gate_data_np, pair_ids_np):
    """Reference QV-4 simulation in NumPy. Returns probs [n_circuits, 16]."""
    # Index table matching the CUDA constant memory
    idx_table = [
        [0, 1, 2, 3,   4, 5, 6, 7,   8, 9,10,11,  12,13,14,15],
        [0, 1, 4, 5,   2, 3, 6, 7,   8, 9,12,13,  10,11,14,15],
        [0, 1, 8, 9,   2, 3,10,11,   4, 5,12,13,   6, 7,14,15],
        [0, 2, 4, 6,   1, 3, 5, 7,   8,10,12,14,   9,11,13,15],
        [0, 2, 8,10,   1, 3, 9,11,   4, 6,12,14,   5, 7,13,15],
        [0, 4, 8,12,   1, 5, 9,13,   2, 6,10,14,   3, 7,11,15],
    ]

    n_circuits = gate_data_np.shape[0]
    n_gates = gate_data_np.shape[1]
    probs = np.empty((n_circuits, 16), dtype=np.float32)

    for c in range(n_circuits):
        # State vector: |0000⟩
        state = np.zeros(16, dtype=np.complex64)
        state[0] = 1.0

        for g in range(n_gates):
            gre = gate_data_np[c, g, :16].reshape(4, 4)
            gim = gate_data_np[c, g, 16:].reshape(4, 4)
            gate = gre + 1j * gim

            pid = pair_ids_np[c, g]
            idx = idx_table[pid]

            for grp in range(4):
                ii = idx[grp * 4: grp * 4 + 4]
                vec = state[ii]
                state[ii] = gate @ vec

        probs[c] = np.abs(state) ** 2

    return probs


def heavy_output_probability(probs):
    """Compute heavy output probability for each circuit.

    For each circuit, find the median probability over 16 outcomes.
    Heavy outputs are those with probability > median.
    Return the fraction of probability mass on heavy outputs.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    median = np.median(probs, axis=1, keepdims=True)
    heavy_mask = probs > median
    return (probs * heavy_mask).sum(axis=1)
