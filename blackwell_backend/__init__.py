# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell

"""Qiskit backend powered by custom Blackwell (SM_120) CUDA kernels."""

from blackwell_backend.simulator import BlackwellSimulator
from blackwell_backend.hybrid import HybridSimulator, RoutingPolicy, analyze_circuit

__all__ = ["BlackwellSimulator", "HybridSimulator", "RoutingPolicy", "analyze_circuit"]
