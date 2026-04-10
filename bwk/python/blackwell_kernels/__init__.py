# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell

"""Custom CUDA kernels optimized for RTX 5090 (sm_120)."""

__version__ = "0.1.0"

from blackwell_kernels.cuquantum import (
    apply_gate, apply_gate_2q, apply_gates_fused, apply_gates_reg_tiled,
    apply_diagonal, apply_diagonal_phase, apply_diagonal_2q, state_init,
    measure_qubit, measure_probs, sample,
    expectation_z, expectation_pauli,
    apply_mcgate, apply_mc2qgate, swap_gate, cswap_gate,
    apply_depolarizing, apply_amplitude_damping, apply_dephasing, pauli_x_gate,
    apply_gate_batched, apply_gates_batched_fused, state_init_batched,
    CircuitGraph, hadamard_gate, random_unitary,
    cnot_gate, cz_gate, swap_gate, random_unitary_4x4,
    rz_gate_diag, phase_gate_diag, t_gate_diag, s_gate_diag, z_gate_diag, cz_gate_diag,
)

from blackwell_kernels.qv8 import (
    qv8_simulate, qv8_simulate_ref, generate_qv8_circuits, generate_qv8_circuits_gpu,
)

from blackwell_kernels.qv4 import (
    qv4_simulate_cuda, qv4_simulate_numpy, generate_qv4_circuits, heavy_output_probability,
)
