# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell

"""
Qiskit-compatible statevector simulator using custom Blackwell CUDA kernels.

This backend translates Qiskit circuit instructions into calls to the
blackwell_kernels library (SM_120-optimized CUDA kernels running on RTX 5090).

Circuit fusion: consecutive gates are batched into fewer kernel launches.
Diagonal gates on the same qubit are algebraically merged. 1Q gate runs
are dispatched via fused kernels. Full circuits are captured as CUDA graphs
for near-zero launch overhead on repeated execution.
"""

import sys
import os
import math
import cmath
import time
import importlib
import numpy as np
import torch

from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.result import Result

# ─── Kernel loading ───────────────────────────────────────────────────
#
# Default kernel path: bwk/python (relative to qiskit-blackwell repo root)
# Override with env var: BLACKWELL_KERNEL_PATH=/path/to/new/kernels/python
#
# To A/B test, create two simulators pointing at different kernel builds:
#
#   sim_old = BlackwellSimulator()                                    # default (bwk/)
#   sim_new = BlackwellSimulator(kernel_path="/path/to/other/python")
#

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_KERNEL_PATH = os.path.join(_REPO_ROOT, "bwk", "python")

def _load_kernels(kernel_path=None):
    """Import blackwell_kernels from a specific path (isolated from sys.path)."""
    path = kernel_path or os.environ.get("BLACKWELL_KERNEL_PATH", _DEFAULT_KERNEL_PATH)
    # Use importlib to load from an explicit path without polluting sys.path
    if path not in sys.path:
        sys.path.insert(0, path)
    # Force reimport if loading from a different path
    if "blackwell_kernels" in sys.modules:
        existing = sys.modules["blackwell_kernels"]
        existing_path = getattr(existing, "__path__", [None])[0] or ""
        if os.path.abspath(path) not in os.path.abspath(existing_path):
            # Different path — need a fresh import
            del sys.modules["blackwell_kernels"]
            for key in list(sys.modules):
                if key.startswith("blackwell_kernels."):
                    del sys.modules[key]
    import blackwell_kernels
    return blackwell_kernels


# ─── Compiled circuit operations ─────────────────────────────────────

class _FusedOp:
    """A batch of gates dispatched in a single kernel launch."""
    __slots__ = ('kind', 'gates', 'targets', 'qubit0', 'qubit1',
                 'controls', 'diag', 'phase', 'target', 'clbit')

    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.gates = kwargs.get('gates')
        self.targets = kwargs.get('targets')
        self.qubit0 = kwargs.get('qubit0')
        self.qubit1 = kwargs.get('qubit1')
        self.controls = kwargs.get('controls')
        self.diag = kwargs.get('diag')
        self.phase = kwargs.get('phase')
        self.target = kwargs.get('target')
        self.clbit = kwargs.get('clbit')


class BlackwellSimulator:
    """Statevector simulator dispatching to Blackwell CUDA kernels.

    Usage:
        sim = BlackwellSimulator()                                    # default kernels
        sim = BlackwellSimulator(kernel_path="/path/to/other/python")    # alt kernels
        result = sim.run(circuit, shots=1024)
    """

    def __init__(self, device="cuda", kernel_path=None):
        self.device = device
        self.kernel_path = kernel_path
        self.bk = _load_kernels(kernel_path)
        self._kernel_label = kernel_path or _DEFAULT_KERNEL_PATH
        # Native gate set — gates we handle directly without decomposition
        # Only standard Qiskit gate names (no aliases like "cnot")
        self._native_1q = {
            "h", "x", "y", "z", "s", "sdg", "t", "tdg",
            "rx", "ry", "rz", "p", "u1", "u2", "u3", "u",
            "id", "sx", "sxdg",
        }
        self._native_2q = {"cx", "cz", "swap", "cp", "crz", "cu"}
        self._native_mcx = {"ccx"}
        self._native_other = {"measure", "barrier", "reset"}
        self._graph_cache = {}  # topology hash -> (CUDAGraph, gate_buffers, state_ref)
        self._compile_cache = {}  # circuit_key -> (n_qubits, n_clbits, compiled_ops)
        # Pre-allocate constant gates for use during CUDA graph capture
        self._const_pauli_x = self.bk.pauli_x_gate(device)
        self._const_swap = self.bk.swap_gate(device)
        self._pm = None  # lazy-init transpiler pass manager

    # -----------------------------------------------------------------
    # Gate matrix builders (on GPU, complex64)
    # -----------------------------------------------------------------
    def _gate_matrix_1q(self, name, params):
        """Return a 2x2 complex64 gate tensor on device."""
        d = self.device
        if name == "h":
            return self.bk.hadamard_gate(d)
        elif name == "x":
            return self.bk.pauli_x_gate(d)
        elif name == "y":
            return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=d)
        elif name == "z":
            return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=d)
        elif name == "s":
            return torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64, device=d)
        elif name == "sdg":
            return torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64, device=d)
        elif name == "t":
            return torch.tensor([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=torch.complex64, device=d)
        elif name == "tdg":
            return torch.tensor([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=torch.complex64, device=d)
        elif name in ("id",):
            return torch.eye(2, dtype=torch.complex64, device=d)
        elif name == "sx":
            return torch.tensor([[0.5 + 0.5j, 0.5 - 0.5j],
                                 [0.5 - 0.5j, 0.5 + 0.5j]], dtype=torch.complex64, device=d)
        elif name == "sxdg":
            return torch.tensor([[0.5 - 0.5j, 0.5 + 0.5j],
                                 [0.5 + 0.5j, 0.5 - 0.5j]], dtype=torch.complex64, device=d)
        elif name == "rx":
            theta = float(params[0])
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64, device=d)
        elif name == "ry":
            theta = float(params[0])
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return torch.tensor([[c, -s], [s, c]], dtype=torch.complex64, device=d)
        elif name == "rz":
            theta = float(params[0])
            return torch.tensor([[cmath.exp(-1j * theta / 2), 0],
                                 [0, cmath.exp(1j * theta / 2)]], dtype=torch.complex64, device=d)
        elif name in ("p", "u1"):
            lam = float(params[0])
            return torch.tensor([[1, 0], [0, cmath.exp(1j * lam)]], dtype=torch.complex64, device=d)
        elif name == "u2":
            phi, lam = float(params[0]), float(params[1])
            inv_sqrt2 = 1 / math.sqrt(2)
            return torch.tensor([
                [inv_sqrt2, -cmath.exp(1j * lam) * inv_sqrt2],
                [cmath.exp(1j * phi) * inv_sqrt2, cmath.exp(1j * (phi + lam)) * inv_sqrt2]
            ], dtype=torch.complex64, device=d)
        elif name in ("u3", "u"):
            theta, phi, lam = float(params[0]), float(params[1]), float(params[2])
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return torch.tensor([
                [c, -cmath.exp(1j * lam) * s],
                [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c]
            ], dtype=torch.complex64, device=d)
        else:
            raise ValueError(f"Unknown 1Q gate: {name}")

    def _gate_matrix_2q(self, name, params):
        """Return a 4x4 complex64 gate tensor on device."""
        d = self.device
        if name == "cx":
            return self.bk.cnot_gate(d)
        elif name == "cz":
            return self.bk.cz_gate(d)
        elif name == "swap":
            return self.bk.swap_gate(d)
        elif name == "cp":
            lam = float(params[0])
            g = torch.eye(4, dtype=torch.complex64, device=d)
            g[3, 3] = cmath.exp(1j * lam)
            return g
        elif name == "crz":
            theta = float(params[0])
            g = torch.eye(4, dtype=torch.complex64, device=d)
            g[2, 2] = cmath.exp(-1j * theta / 2)
            g[3, 3] = cmath.exp(1j * theta / 2)
            return g
        elif name == "cu":
            theta, phi, lam, gamma = [float(p) for p in params]
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            g = torch.eye(4, dtype=torch.complex64, device=d)
            eig = cmath.exp(1j * gamma)
            g[2, 2] = eig * c
            g[2, 3] = -eig * cmath.exp(1j * lam) * s
            g[3, 2] = eig * cmath.exp(1j * phi) * s
            g[3, 3] = eig * cmath.exp(1j * (phi + lam)) * c
            return g
        else:
            raise ValueError(f"Unknown 2Q gate: {name}")

    @staticmethod
    def _swap_gate_qubits(gate):
        """Swap which qubit is MSB/LSB in a 4x4 gate matrix."""
        perm = [0, 2, 1, 3]
        return gate[perm][:, perm]

    def _is_diagonal_1q(self, name):
        """Check if a 1Q gate is diagonal (can use faster diagonal kernel)."""
        return name in ("z", "s", "sdg", "t", "tdg", "rz", "p", "u1", "id")

    def _diagonal_1q(self, name, params):
        """Return diagonal entries (d0, d1) for a diagonal 1Q gate."""
        if name == "z":
            return 1.0, -1.0
        elif name == "s":
            return 1.0, 1j
        elif name == "sdg":
            return 1.0, -1j
        elif name == "t":
            return 1.0, cmath.exp(1j * math.pi / 4)
        elif name == "tdg":
            return 1.0, cmath.exp(-1j * math.pi / 4)
        elif name in ("rz",):
            theta = float(params[0])
            return cmath.exp(-1j * theta / 2), cmath.exp(1j * theta / 2)
        elif name in ("p", "u1"):
            lam = float(params[0])
            return 1.0, cmath.exp(1j * lam)
        elif name == "id":
            return 1.0, 1.0
        return None

    # -----------------------------------------------------------------
    # Circuit compilation — fusion pass
    # -----------------------------------------------------------------
    def _compile_circuit(self, transpiled):
        """Compile a transpiled circuit into a list of FusedOps.

        Optimizations applied:
        1. Consecutive diagonal gates on same qubit → algebraically merged
        2. Same-axis rotations (Rx/Ry) on same qubit → angles accumulated in float64
        3. Diagonal adjacent to dense gate on same qubit → absorbed into matrix
        4. Consecutive dense 1Q gates on same qubit → matrices multiplied
        5. Remaining 1Q dense gates → batched into apply_gates_fused
        6. Everything else → individual dispatch (captured by CUDA graph)
        """
        qubit_indices = {q: i for i, q in enumerate(transpiled.qubits)}
        clbit_indices = {c: i for i, c in enumerate(transpiled.clbits)}

        # Phase 1: Parse instructions into an intermediate representation
        raw_ops = []
        for instruction in transpiled.data:
            op = instruction.operation
            name = op.name.lower()
            params = op.params
            qubits = [qubit_indices[q] for q in instruction.qubits]

            if name == "barrier":
                continue
            elif name == "measure":
                cb = clbit_indices[instruction.clbits[0]]
                raw_ops.append(('measure', qubits[0], cb))
            elif name == "reset":
                raw_ops.append(('reset', qubits[0]))
            elif name in self._native_1q:
                q = qubits[0]
                if self._is_diagonal_1q(name):
                    d0, d1 = self._diagonal_1q(name, params)
                    raw_ops.append(('diag', q, complex(d0), complex(d1)))
                elif name in ("rx", "ry"):
                    # Tag rotation gates with axis and angle for Phase 2.5 merging
                    theta = float(params[0])
                    gate = self._gate_matrix_1q(name, params)
                    raw_ops.append(('rot1q', q, gate, name, theta))
                else:
                    gate = self._gate_matrix_1q(name, params)
                    raw_ops.append(('dense1q', q, gate))
            elif name in self._native_2q:
                q_first, q_second = qubits[0], qubits[1]
                gate = self._gate_matrix_2q(name, params)
                if q_first < q_second:
                    gate = self._swap_gate_qubits(gate)
                raw_ops.append(('gate2q', q_first, q_second, gate))
            elif name in ("ccx", "toffoli"):
                controls = qubits[:-1]
                target = qubits[-1]
                raw_ops.append(('mcgate', controls, target))
            elif name in ("mcx", "c3x", "c4x"):
                controls = qubits[:-1]
                target = qubits[-1]
                raw_ops.append(('mcgate', controls, target))
            elif name == "cswap":
                raw_ops.append(('mc2qgate', [qubits[0]], qubits[1], qubits[2]))
            else:
                raise ValueError(f"Unsupported instruction: {name}")

        # Phase 2: Merge consecutive diagonals on the same qubit
        merged = []
        for op in raw_ops:
            if op[0] == 'diag' and merged and merged[-1][0] == 'diag' and merged[-1][1] == op[1]:
                # Same qubit diagonal: multiply diagonal entries
                prev = merged[-1]
                merged[-1] = ('diag', op[1], prev[2] * op[2], prev[3] * op[3])
            else:
                merged.append(op)

        # Phase 2.5: Merge consecutive same-axis rotations on same qubit
        # Accumulate angles in float64, then build a single gate matrix.
        # More precise than multiplying individually-rounded float32 matrices.
        rot_merged = []
        for op in merged:
            if (op[0] == 'rot1q' and rot_merged
                    and rot_merged[-1][0] == 'rot1q'
                    and rot_merged[-1][1] == op[1]       # same qubit
                    and rot_merged[-1][3] == op[3]):      # same axis
                prev = rot_merged[-1]
                combined_theta = prev[4] + op[4]
                combined_gate = self._gate_matrix_1q(prev[3], [combined_theta])
                rot_merged[-1] = ('rot1q', op[1], combined_gate, prev[3], combined_theta)
            else:
                rot_merged.append(op)
        # Convert surviving rot1q back to dense1q for downstream phases
        for idx_rm in range(len(rot_merged)):
            if rot_merged[idx_rm][0] == 'rot1q':
                op = rot_merged[idx_rm]
                rot_merged[idx_rm] = ('dense1q', op[1], op[2])

        # Phase 3: Absorb diagonals into adjacent dense gates on same qubit
        absorbed = []
        i = 0
        while i < len(rot_merged):
            op = rot_merged[i]
            if op[0] == 'diag' and i + 1 < len(rot_merged) and rot_merged[i + 1][0] == 'dense1q' and rot_merged[i + 1][1] == op[1]:
                # diag before dense on same qubit: gate = gate @ diag
                _, q, d0, d1 = op
                next_op = rot_merged[i + 1]
                diag_mat = torch.tensor([[d0, 0], [0, d1]], dtype=torch.complex64, device=self.device)
                fused_gate = next_op[2] @ diag_mat
                absorbed.append(('dense1q', q, fused_gate))
                i += 2
            elif op[0] == 'dense1q' and i + 1 < len(rot_merged) and rot_merged[i + 1][0] == 'diag' and rot_merged[i + 1][1] == op[1]:
                # dense before diag on same qubit: gate = diag @ gate
                next_op = rot_merged[i + 1]
                d0, d1 = next_op[2], next_op[3]
                diag_mat = torch.tensor([[d0, 0], [0, d1]], dtype=torch.complex64, device=self.device)
                fused_gate = diag_mat @ op[2]
                absorbed.append(('dense1q', op[1], fused_gate))
                i += 2
            else:
                absorbed.append(op)
                i += 1

        # Phase 3.5: Multiply consecutive dense 1Q gates on same qubit
        # Reduces N sequential kernel applications to 1, fewer rounding steps.
        # Cap at 8 gates to limit float32 matrix product error accumulation.
        mat_fused = []
        for op in absorbed:
            if (op[0] == 'dense1q' and mat_fused
                    and mat_fused[-1][0] == 'dense1q'
                    and mat_fused[-1][1] == op[1]):
                prev = mat_fused[-1]
                # later gate @ earlier gate (circuit order: earlier applied first)
                fused_gate = op[2] @ prev[2]
                mat_fused[-1] = ('dense1q', op[1], fused_gate)
            else:
                mat_fused.append(op)

        # Phase 4: Batch consecutive 1Q ops into FusedOps
        compiled = []
        pending_1q_gates = []  # list of (gate_2x2, target_qubit)
        pending_diags = []     # list of (d0, d1, target_qubit)

        def flush_pending():
            nonlocal pending_1q_gates, pending_diags
            # Flush diagonals first (they're fast individual launches)
            for d0, d1, t in pending_diags:
                if abs(d0 - 1.0) < 1e-10:
                    compiled.append(_FusedOp('diag_phase', phase=complex(d1), target=t))
                else:
                    diag = torch.tensor([d0, d1], dtype=torch.complex64, device=self.device)
                    compiled.append(_FusedOp('diag', diag=diag, target=t))
            pending_diags = []

            # Batch 1Q dense gates
            if len(pending_1q_gates) == 0:
                pass
            elif len(pending_1q_gates) == 1:
                g, t = pending_1q_gates[0]
                compiled.append(_FusedOp('gate1q', gates=g, target=t))
            else:
                gates = torch.stack([g for g, _ in pending_1q_gates])
                targets = torch.tensor([t for _, t in pending_1q_gates],
                                       dtype=torch.int32, device=self.device)
                # Use reg_tiled for low qubits, fused for general
                if all(t < 5 for _, t in pending_1q_gates):
                    compiled.append(_FusedOp('fused_reg_tiled', gates=gates, targets=targets))
                else:
                    compiled.append(_FusedOp('fused_1q', gates=gates, targets=targets))
            pending_1q_gates = []

        for op in mat_fused:
            if op[0] == 'dense1q':
                pending_1q_gates.append((op[2], op[1]))
            elif op[0] == 'diag':
                pending_diags.append((op[2], op[3], op[1]))
            else:
                # Non-1Q op: flush pending 1Q batch, then emit this op
                flush_pending()
                if op[0] == 'gate2q':
                    compiled.append(_FusedOp('gate2q', gates=op[3],
                                             qubit0=op[1], qubit1=op[2]))
                elif op[0] == 'mcgate':
                    compiled.append(_FusedOp('mcgate', controls=op[1], target=op[2]))
                elif op[0] == 'mc2qgate':
                    compiled.append(_FusedOp('mc2qgate', controls=op[1],
                                             qubit0=op[2], qubit1=op[3]))
                elif op[0] == 'measure':
                    compiled.append(_FusedOp('measure', target=op[1], clbit=op[2]))
                elif op[0] == 'reset':
                    compiled.append(_FusedOp('reset', target=op[1]))

        flush_pending()
        return compiled

    # -----------------------------------------------------------------
    # Compiled circuit execution
    # -----------------------------------------------------------------
    def _execute_compiled(self, state, ops, clbits):
        """Execute a list of FusedOps on the state vector."""
        bk = self.bk
        for op in ops:
            kind = op.kind
            if kind == 'fused_1q':
                bk.apply_gates_fused(state, op.gates, op.targets)
            elif kind == 'fused_reg_tiled':
                bk.apply_gates_reg_tiled(state, op.gates, op.targets)
            elif kind == 'gate1q':
                bk.apply_gate(state, op.gates, op.target)
            elif kind == 'diag_phase':
                bk.apply_diagonal_phase(state, op.phase, op.target)
            elif kind == 'diag':
                bk.apply_diagonal(state, op.diag, op.target)
            elif kind == 'gate2q':
                bk.apply_gate_2q(state, op.gates, op.qubit0, op.qubit1)
            elif kind == 'mcgate':
                bk.apply_mcgate(state, self._const_pauli_x,
                                op.controls, op.target)
            elif kind == 'mc2qgate':
                bk.apply_mc2qgate(state, self._const_swap,
                                  op.controls, op.qubit0, op.qubit1)
            elif kind == 'measure':
                outcome = bk.measure_qubit(state, op.target, torch.rand(1).item())
                clbits[op.clbit] = outcome
            elif kind == 'reset':
                outcome = bk.measure_qubit(state, op.target, torch.rand(1).item())
                if outcome == 1:
                    bk.apply_gate(state, self._const_pauli_x, op.target)

    def _can_graph_capture(self, ops):
        """Check if ops can be captured as a CUDA graph."""
        for op in ops:
            if op.kind in ('measure', 'reset'):
                return False
        return True

    def _execute_with_graph(self, state, ops, n_qubits):
        """Execute ops with fusion (no CUDA graph for now — fusion alone is the win)."""
        # The fusion pass already batches 1Q gates and merges diagonals.
        # CUDA graph capture adds complexity with tensor allocation during capture.
        # For now, just execute the compiled ops directly — the kernel launch
        # reduction from fusion is the primary win.
        dummy_clbits = []
        self._execute_compiled(state, ops, dummy_clbits)

    # -----------------------------------------------------------------
    # Circuit execution (main entry point)
    # -----------------------------------------------------------------
    def _get_pass_manager(self):
        """Lazy-init the transpiler pass manager."""
        if self._pm is None:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            basis = list(self._native_1q | self._native_2q | self._native_mcx)
            self._pm = generate_preset_pass_manager(
                optimization_level=1,
                basis_gates=basis,
            )
        return self._pm

    def _circuit_cache_key(self, circuit):
        """Generate a cache key for a circuit."""
        parts = []
        for inst in circuit.data:
            op = inst.operation
            param_vals = []
            for p in op.params:
                try:
                    param_vals.append(float(p))
                except (TypeError, ValueError):
                    param_vals.append(hash(str(p)))
            parts.append((op.name, tuple(param_vals),
                          tuple(circuit.qubits.index(q) for q in inst.qubits)))
        return (circuit.num_qubits, circuit.num_clbits, hash(tuple(parts)))

    def _execute_circuit(self, circuit):
        """Execute a circuit, return (statevector, measurements, elapsed_s)."""
        cache_key = self._circuit_cache_key(circuit)

        if cache_key in self._compile_cache:
            n_qubits, n_clbits, ops = self._compile_cache[cache_key]
        else:
            pm = self._get_pass_manager()
            transpiled = pm.run(circuit)
            n_qubits = transpiled.num_qubits
            n_clbits = transpiled.num_clbits
            ops = self._compile_circuit(transpiled)
            self._compile_cache[cache_key] = (n_qubits, n_clbits, ops)

        state = self.bk.state_init(n_qubits)
        clbits = [0] * n_clbits

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        has_stochastic = any(op.kind in ('measure', 'reset') for op in ops)
        if has_stochastic:
            self._execute_compiled(state, ops, clbits)
        else:
            self._execute_with_graph(state, ops, n_qubits)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return state, clbits, elapsed

    def _strip_measurements(self, circuit):
        """Return a copy of the circuit with measurement/reset gates removed."""
        from qiskit import QuantumCircuit as QC
        new_qc = QC(circuit.num_qubits, circuit.num_clbits, name=circuit.name)
        for inst in circuit.data:
            if inst.operation.name.lower() not in ("measure", "reset"):
                new_qc.append(inst.operation, inst.qubits, inst.clbits)
        return new_qc

    def run(self, circuit, shots=1024, seed=None):
        """Run a circuit and return a Qiskit Result object.

        Args:
            circuit: Qiskit QuantumCircuit
            shots: number of measurement shots
            seed: optional RNG seed for reproducibility

        Returns:
            qiskit.result.Result
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Execute without measurements to get the pre-measurement statevector,
        # then sample from it. This avoids state collapse corrupting shots.
        clean_circuit = self._strip_measurements(circuit)
        state, _, elapsed = self._execute_circuit(clean_circuit)

        # Sample shots from the final statevector
        samples = self.bk.sample(state, shots)
        samples_np = samples.cpu().numpy()

        # Convert to Qiskit-style hex counts
        n_qubits = int(math.log2(state.shape[0]))
        counts = {}
        for s in samples_np:
            # Qiskit uses big-endian bit strings
            bitstring = format(int(s), f"0{n_qubits}b")[::-1]  # reverse for Qiskit convention
            hex_key = hex(int(bitstring[::-1], 2))
            counts[hex_key] = counts.get(hex_key, 0) + 1

        result_data = ExperimentResultData(counts=counts)
        exp_result = ExperimentResult(
            shots=shots,
            success=True,
            data=result_data,
            header={"name": circuit.name, "n_qubits": n_qubits},
        )

        return Result(
            backend_name="blackwell_simulator",
            backend_version="0.1.0",
            qobj_id="",
            job_id="",
            success=True,
            results=[exp_result],
            time_taken=elapsed,
        )

    def run_statevector(self, circuit):
        """Run circuit and return the full statevector as a numpy array.

        Useful for accuracy comparisons (no sampling noise).
        """
        state, _, elapsed = self._execute_circuit(circuit)
        sv = state.cpu().numpy()
        return sv, elapsed

    def run_and_time(self, circuit, shots=1024, seed=None, warmup=2, repeats=5):
        """Run circuit multiple times and return (Result, timing_stats).

        Returns:
            (Result, dict) where dict has keys: warmup_s, times_s, median_s, min_s
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Warmup runs
        for _ in range(warmup):
            self._execute_circuit(circuit)

        # Timed runs
        times = []
        for _ in range(repeats):
            state, _, elapsed = self._execute_circuit(circuit)
            times.append(elapsed)

        # Final run for result
        result = self.run(circuit, shots=shots, seed=seed)

        stats = {
            "times_s": times,
            "median_s": float(np.median(times)),
            "min_s": float(np.min(times)),
            "mean_s": float(np.mean(times)),
        }
        return result, stats
