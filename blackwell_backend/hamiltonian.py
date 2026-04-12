# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell
#
# EXPERIMENTAL (beta) — Hamiltonian CSR representation for Chebyshev evolution.

"""GPU-resident CSR Hamiltonian for Chebyshev time evolution.

Converts Qiskit's SparsePauliOp into compressed sparse row (CSR) format
on GPU, ready for the SpMV kernel in the Chebyshev recurrence loop.
"""

import numpy as np
import torch


class HamiltonianCSR:
    """GPU-resident CSR representation of a Hamiltonian.

    EXPERIMENTAL (beta). Part of the Chebyshev evolution engine.

    Attributes:
        row_ptr: int32 tensor [n_rows + 1] on CUDA
        col_idx: int32 tensor [nnz] on CUDA
        values: complex64 tensor [nnz] on CUDA
        n_qubits: number of qubits
        n_rows: 2^n_qubits (dimension of Hilbert space)
        nnz: number of nonzero entries
        spectral_bound: upper bound on max |eigenvalue|
    """

    def __init__(self, row_ptr, col_idx, values, n_qubits, spectral_bound):
        self.row_ptr = row_ptr
        self.col_idx = col_idx
        self.values = values
        self.n_qubits = n_qubits
        self.n_rows = 1 << n_qubits
        self.nnz = col_idx.shape[0]
        self.spectral_bound = spectral_bound

    @staticmethod
    def from_sparse_pauli_op(spo, device="cuda"):
        """Convert a Qiskit SparsePauliOp to CSR on GPU.

        EXPERIMENTAL (beta).

        Args:
            spo: qiskit.quantum_info.SparsePauliOp
            device: torch device string

        Returns:
            HamiltonianCSR
        """
        # Convert to scipy sparse CSR (CPU)
        mat = spo.to_matrix(sparse=True).tocsr()

        # Transfer to GPU
        row_ptr = torch.tensor(mat.indptr, dtype=torch.int32, device=device)
        col_idx = torch.tensor(mat.indices, dtype=torch.int32, device=device)
        values = torch.tensor(mat.data, dtype=torch.complex64, device=device)

        # Spectral bound: sum of absolute coefficients (guaranteed upper bound)
        spectral_bound = float(np.sum(np.abs(spo.coeffs)))

        return HamiltonianCSR(row_ptr, col_idx, values,
                              spo.num_qubits, spectral_bound)

    def memory_bytes(self):
        """Total GPU memory used by CSR arrays."""
        return (self.row_ptr.nbytes + self.col_idx.nbytes + self.values.nbytes)

    def __repr__(self):
        return (f"HamiltonianCSR(n_qubits={self.n_qubits}, "
                f"dim={self.n_rows}, nnz={self.nnz}, "
                f"spectral_bound={self.spectral_bound:.4f}, "
                f"memory={self.memory_bytes()/1024:.1f} KB)")
