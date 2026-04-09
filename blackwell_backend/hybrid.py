"""
Hybrid simulator: routes each circuit to the best available backend.

Default policy:
  - Blackwell custom kernels for most circuits (faster on native gates)
  - Aer GPU (cuQuantum) for circuits with high non-native gate density
    (e.g. Quantum Volume), where Python-side overhead in Blackwell dominates

The routing table is configurable and designed to be updated as Blackwell
kernels improve (gate fusion, CUDA graph batching, etc.).
"""

import os
import sys
import time
import logging
import numpy as np

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from blackwell_backend.simulator import BlackwellSimulator

logger = logging.getLogger(__name__)


# ─── Circuit analysis ─────────────────────────────────────────────────

# Gates the Blackwell backend handles natively (no decomposition overhead)
_BLACKWELL_NATIVE = {
    # 1Q
    "h", "x", "y", "z", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "p", "u1", "u2", "u3", "u",
    "id", "sx", "sxdg",
    # 2Q
    "cx", "cz", "swap", "cp", "crz", "cu",
    # Multi-controlled
    "ccx",
    # Non-gate
    "measure", "barrier", "reset",
}


def analyze_circuit(circuit):
    """Return a dict of circuit properties used for routing decisions."""
    total_gates = 0
    native_gates = 0
    non_native_gates = 0
    n_1q = 0
    n_2q = 0
    n_mcx = 0
    gate_names = set()

    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name in ("barrier",):
            continue
        total_gates += 1
        gate_names.add(name)
        n_qubits = len(inst.qubits)

        if name in _BLACKWELL_NATIVE:
            native_gates += 1
        else:
            non_native_gates += 1

        if n_qubits == 1:
            n_1q += 1
        elif n_qubits == 2:
            n_2q += 1
        else:
            n_mcx += 1

    non_native_ratio = non_native_gates / max(total_gates, 1)

    return {
        "num_qubits": circuit.num_qubits,
        "total_gates": total_gates,
        "native_gates": native_gates,
        "non_native_gates": non_native_gates,
        "non_native_ratio": non_native_ratio,
        "n_1q": n_1q,
        "n_2q": n_2q,
        "n_mcx": n_mcx,
        "gate_names": gate_names,
    }


# ─── Routing policy ──────────────────────────────────────────────────

class RoutingPolicy:
    """Decides which backend to use for a given circuit.

    Override thresholds or add per-circuit-name rules as Blackwell improves.
    """

    def __init__(
        self,
        non_native_threshold=0.3,
        force_blackwell=None,
        force_cuquantum=None,
    ):
        """
        Args:
            non_native_threshold: if non-native gate ratio exceeds this,
                route to cuQuantum. Set to 1.0 to always prefer Blackwell.
            force_blackwell: set of circuit names to always route to Blackwell
            force_cuquantum: set of circuit names to always route to cuQuantum
        """
        self.non_native_threshold = non_native_threshold
        self.force_blackwell = force_blackwell or set()
        self.force_cuquantum = force_cuquantum or set()

    def choose(self, circuit, analysis=None):
        """Return 'blackwell' or 'cuquantum'."""
        name = getattr(circuit, "name", "")

        # Explicit overrides
        if name in self.force_blackwell:
            return "blackwell"
        if name in self.force_cuquantum:
            return "cuquantum"

        if analysis is None:
            analysis = analyze_circuit(circuit)

        # High non-native gate ratio → cuQuantum handles decomposition better
        if analysis["non_native_ratio"] > self.non_native_threshold:
            return "cuquantum"

        return "blackwell"


# ─── Hybrid simulator ────────────────────────────────────────────────

class HybridSimulator:
    """Routes circuits to the best available backend automatically.

    Usage:
        sim = HybridSimulator()
        result = sim.run(circuit, shots=1024)

        # Force a specific backend:
        result = sim.run(circuit, shots=1024, backend='cuquantum')

        # Check what would be chosen:
        choice, analysis = sim.analyze(circuit)
    """

    def __init__(self, policy=None, aer_device="GPU", kernel_path=None):
        """
        Args:
            policy: RoutingPolicy instance (or None for defaults)
            aer_device: 'GPU' or 'CPU' for the Aer backend
            kernel_path: path to Blackwell kernel build (None = default)
        """
        self.policy = policy or RoutingPolicy()
        self._blackwell = BlackwellSimulator(kernel_path=kernel_path)
        self._aer = AerSimulator(method="statevector", device=aer_device)
        self._aer_device = aer_device
        self._route_log = []

    def analyze(self, circuit):
        """Return (chosen_backend, analysis_dict) without running."""
        analysis = analyze_circuit(circuit)
        choice = self.policy.choose(circuit, analysis)
        return choice, analysis

    def run(self, circuit, shots=1024, seed=None, backend=None):
        """Run circuit on the best (or specified) backend.

        Args:
            circuit: Qiskit QuantumCircuit
            shots: number of measurement shots
            seed: optional RNG seed
            backend: force 'blackwell' or 'cuquantum' (None = auto)

        Returns:
            qiskit.result.Result with extra metadata:
                result.metadata['backend_used'] = 'blackwell' | 'cuquantum'
                result.metadata['analysis'] = {...}
        """
        analysis = analyze_circuit(circuit)
        chosen = backend or self.policy.choose(circuit, analysis)

        log_entry = {
            "circuit_name": getattr(circuit, "name", "unnamed"),
            "num_qubits": circuit.num_qubits,
            "total_gates": analysis["total_gates"],
            "non_native_ratio": analysis["non_native_ratio"],
            "backend_used": chosen,
        }

        if chosen == "blackwell":
            result = self._run_blackwell(circuit, shots, seed)
        else:
            result = self._run_cuquantum(circuit, shots, seed)

        log_entry["success"] = result.success
        self._route_log.append(log_entry)

        # Attach routing metadata
        if not hasattr(result, "metadata"):
            result.metadata = {}
        result.metadata["backend_used"] = chosen
        result.metadata["analysis"] = analysis

        logger.info(
            "Routed %s (%dq, %d gates, %.0f%% non-native) → %s",
            log_entry["circuit_name"],
            log_entry["num_qubits"],
            log_entry["total_gates"],
            log_entry["non_native_ratio"] * 100,
            chosen,
        )

        return result

    def run_statevector(self, circuit, backend=None):
        """Run circuit and return (statevector_np, elapsed_s, backend_used)."""
        analysis = analyze_circuit(circuit)
        chosen = backend or self.policy.choose(circuit, analysis)

        if chosen == "blackwell":
            sv, elapsed = self._blackwell.run_statevector(circuit)
        else:
            sv, elapsed = self._run_cuquantum_statevector(circuit)

        return sv, elapsed, chosen

    def run_both(self, circuit, shots=1024, seed=None):
        """Run on BOTH backends and return comparison results.

        Returns:
            dict with keys: blackwell_result, cuquantum_result,
                            blackwell_sv, cuquantum_sv, fidelity,
                            blackwell_time, cuquantum_time, speedup
        """
        # Blackwell
        sv_bw, t_bw = self._blackwell.run_statevector(circuit)

        # cuQuantum
        sv_cq, t_cq = self._run_cuquantum_statevector(circuit)

        # Fidelity
        sv_bw_n = sv_bw / np.linalg.norm(sv_bw)
        sv_cq_n = sv_cq / np.linalg.norm(sv_cq)
        fidelity = float(np.abs(np.vdot(sv_bw_n, sv_cq_n)) ** 2)

        return {
            "blackwell_sv": sv_bw,
            "cuquantum_sv": sv_cq,
            "fidelity": fidelity,
            "blackwell_time_ms": t_bw * 1000,
            "cuquantum_time_ms": t_cq * 1000,
            "speedup": t_cq / max(t_bw, 1e-9),
            "blackwell_faster": t_bw < t_cq,
        }

    @property
    def route_log(self):
        """Return the list of routing decisions made so far."""
        return list(self._route_log)

    def print_route_summary(self):
        """Print a summary of all routing decisions."""
        if not self._route_log:
            print("No circuits routed yet.")
            return
        bw_count = sum(1 for e in self._route_log if e["backend_used"] == "blackwell")
        cq_count = len(self._route_log) - bw_count
        print(f"Routing summary: {len(self._route_log)} circuits")
        print(f"  Blackwell: {bw_count}  |  cuQuantum: {cq_count}")
        for e in self._route_log:
            status = "OK" if e["success"] else "FAIL"
            print(f"  {e['circuit_name']:30s} {e['num_qubits']:3d}q "
                  f"{e['total_gates']:4d}g  "
                  f"nn={e['non_native_ratio']:.0%}  "
                  f"→ {e['backend_used']:10s} [{status}]")

    # ── Internal backend dispatch ─────────────────────────────────────

    def _run_blackwell(self, circuit, shots, seed):
        """Dispatch to Blackwell backend."""
        return self._blackwell.run(circuit, shots=shots, seed=seed)

    def _run_cuquantum(self, circuit, shots, seed):
        """Dispatch to Aer GPU (cuQuantum)."""
        if seed is not None:
            self._aer.set_options(seed_simulator=seed)
        return self._aer.run(circuit, shots=shots).result()

    def _run_cuquantum_statevector(self, circuit):
        """Run on Aer GPU and return (statevector_np, elapsed_s)."""
        qc = circuit.copy()
        qc.save_statevector()
        t0 = time.perf_counter()
        result = self._aer.run(qc, shots=0).result()
        elapsed = time.perf_counter() - t0
        sv = np.array(result.data()["statevector"])
        return sv, elapsed
