#!/usr/bin/env python3
"""
A/B Test Harness: Aer (cuQuantum GPU) vs Blackwell Custom Kernels

Compares accuracy (statevector fidelity) and performance (wall-clock time)
across a suite of quantum circuits at varying qubit counts and depths.

Usage:
    python tests/ab_test.py                    # full suite
    python tests/ab_test.py --qubits 10 15 20  # specific qubit counts
    python tests/ab_test.py --quick             # fast smoke test
"""

import sys
import os
import time
import argparse
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    QFT, EfficientSU2, QuantumVolume, RealAmplitudes,
)
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator

from blackwell_backend import BlackwellSimulator


# ─── Circuit Generators ──────────────────────────────────────────────

def make_ghz(n):
    """GHZ state: H on q0, then CNOT cascade."""
    qc = QuantumCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def make_qft(n):
    """Quantum Fourier Transform."""
    qc = QFT(n, name=f"qft_{n}")
    return qc.decompose()


def make_random_clifford(n, depth=20, seed=42):
    """Random Clifford-like circuit (H, S, CNOT)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"clifford_{n}_d{depth}")
    for _ in range(depth):
        gate = rng.choice(["h", "s", "cx"])
        if gate == "h":
            qc.h(rng.integers(n))
        elif gate == "s":
            qc.s(rng.integers(n))
        elif gate == "cx" and n >= 2:
            q0, q1 = rng.choice(n, size=2, replace=False)
            qc.cx(int(q0), int(q1))
    return qc


def make_parametric(n, reps=2):
    """EfficientSU2 ansatz with random parameters (VQE-style)."""
    qc = EfficientSU2(n, reps=reps, name=f"su2_{n}_r{reps}")
    params = np.random.default_rng(42).uniform(0, 2 * np.pi, qc.num_parameters)
    qc = qc.assign_parameters(params)
    return qc.decompose()


def make_quantum_volume(n, depth=None, seed=42):
    """Quantum Volume circuit."""
    if depth is None:
        depth = n
    qc = QuantumVolume(n, depth=depth, seed=seed)
    qc.name = f"qv_{n}_d{depth}"
    return qc.decompose()


def make_toffoli_chain(n):
    """Chain of Toffoli gates (tests multi-controlled path)."""
    if n < 3:
        raise ValueError("Need at least 3 qubits for Toffoli chain")
    qc = QuantumCircuit(n, name=f"toffoli_{n}")
    # Put all qubits in superposition first
    for i in range(n):
        qc.h(i)
    # Toffoli chain
    for i in range(n - 2):
        qc.ccx(i, i + 1, i + 2)
    return qc


def make_deep_rz(n, depth=50, seed=42):
    """Deep circuit of Rz rotations + entangling CX layers (tests diagonal kernel)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"deep_rz_{n}_d{depth}")
    for _ in range(depth):
        for q in range(n):
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(0, n - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            qc.cx(q, q + 1)
    return qc


# ─── Backends ─────────────────────────────────────────────────────────

def _aer_gpu_available():
    """Check if Aer GPU simulation is supported."""
    try:
        sim = AerSimulator(method="statevector", device="GPU")
        qc = QuantumCircuit(1)
        qc.save_statevector()
        sim.run(qc, shots=0).result()
        return True
    except Exception:
        return False

_GPU_OK = None

def get_aer_statevector(circuit):
    """Run on Aer GPU and return (statevector_np, elapsed_s)."""
    global _GPU_OK
    if _GPU_OK is None:
        _GPU_OK = _aer_gpu_available()
    if not _GPU_OK:
        return get_aer_statevector_cpu(circuit)

    sim = AerSimulator(method="statevector", device="GPU")
    qc = circuit.copy()
    qc.save_statevector()

    t0 = time.perf_counter()
    result = sim.run(qc, shots=0).result()
    elapsed = time.perf_counter() - t0

    sv = np.array(result.data()["statevector"])
    return sv, elapsed


def get_aer_statevector_cpu(circuit):
    """Run on Aer CPU as a reference (highest trust)."""
    sim = AerSimulator(method="statevector", device="CPU")
    qc = circuit.copy()
    qc.save_statevector()

    t0 = time.perf_counter()
    result = sim.run(qc, shots=0).result()
    elapsed = time.perf_counter() - t0

    sv = np.array(result.data()["statevector"])
    return sv, elapsed


def get_blackwell_statevector(circuit):
    """Run on Blackwell backend and return (statevector_np, elapsed_s)."""
    sim = BlackwellSimulator()
    sv, elapsed = sim.run_statevector(circuit)
    return sv, elapsed


def get_aer_shots(circuit, shots=8192):
    """Run on Aer GPU (or CPU fallback) and return (counts_dict, elapsed_s)."""
    global _GPU_OK
    if _GPU_OK is None:
        _GPU_OK = _aer_gpu_available()
    device = "GPU" if _GPU_OK else "CPU"
    sim = AerSimulator(method="statevector", device=device)
    t0 = time.perf_counter()
    result = sim.run(circuit, shots=shots).result()
    elapsed = time.perf_counter() - t0
    return result.get_counts(), elapsed


def get_blackwell_shots(circuit, shots=8192):
    """Run on Blackwell backend and return (counts_dict, elapsed_s)."""
    sim = BlackwellSimulator()
    qc = circuit.copy()
    qc.measure_all()
    t0 = time.perf_counter()
    result = sim.run(qc, shots=shots)
    elapsed = time.perf_counter() - t0
    return result.get_counts(), elapsed


# ─── Comparison Utilities ─────────────────────────────────────────────

def compute_fidelity(sv1, sv2):
    """State fidelity between two statevector arrays."""
    # Normalize
    sv1 = sv1 / np.linalg.norm(sv1)
    sv2 = sv2 / np.linalg.norm(sv2)
    # Fidelity = |<sv1|sv2>|^2
    overlap = np.abs(np.vdot(sv1, sv2)) ** 2
    return float(overlap)


def compare_distributions(counts1, counts2, n_qubits):
    """Compare two shot-count distributions. Returns TVD and KL divergence."""
    # Build full probability vectors
    n_states = 2 ** n_qubits
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    p1 = np.zeros(n_states)
    p2 = np.zeros(n_states)

    for key, val in counts1.items():
        idx = int(key, 16) if key.startswith("0x") else int(key, 2)
        p1[idx] = val / total1
    for key, val in counts2.items():
        idx = int(key, 16) if key.startswith("0x") else int(key, 2)
        p2[idx] = val / total2

    tvd = 0.5 * np.sum(np.abs(p1 - p2))
    # KL divergence (with smoothing)
    eps = 1e-10
    p1s = p1 + eps
    p2s = p2 + eps
    p1s /= p1s.sum()
    p2s /= p2s.sum()
    kl = np.sum(p1s * np.log(p1s / p2s))

    return float(tvd), float(kl)


# ─── Main Test Runner ─────────────────────────────────────────────────

def run_accuracy_test(name, circuit, verbose=True):
    """Compare statevectors from Aer CPU (reference), Aer GPU, and Blackwell."""
    n = circuit.num_qubits
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {name}  ({n} qubits, {circuit.size()} gates)")
        print(f"{'='*70}")

    results = {}

    # Reference: Aer CPU
    try:
        sv_ref, t_ref = get_aer_statevector_cpu(circuit)
        results["aer_cpu"] = {"sv": sv_ref, "time": t_ref}
        if verbose:
            print(f"  Aer CPU:       {t_ref*1000:8.2f} ms  (reference)")
    except Exception as e:
        print(f"  Aer CPU:       FAILED ({e})")
        return None

    # Aer GPU (or CPU fallback)
    try:
        global _GPU_OK
        if _GPU_OK is None:
            _GPU_OK = _aer_gpu_available()
        aer_label = "Aer GPU" if _GPU_OK else "Aer CPU*"
        sv_gpu, t_gpu = get_aer_statevector(circuit)
        fid_gpu = compute_fidelity(sv_ref, sv_gpu)
        results["aer_gpu"] = {"sv": sv_gpu, "time": t_gpu, "fidelity": fid_gpu}
        if verbose:
            print(f"  {aer_label:12s} {t_gpu*1000:8.2f} ms  fidelity={fid_gpu:.10f}"
                  f"  speedup={t_ref/t_gpu:.2f}x")
    except Exception as e:
        print(f"  Aer GPU:       FAILED ({e})")

    # Blackwell
    try:
        sv_bw, t_bw = get_blackwell_statevector(circuit)
        fid_bw = compute_fidelity(sv_ref, sv_bw)
        results["blackwell"] = {"sv": sv_bw, "time": t_bw, "fidelity": fid_bw}
        speedup_vs_gpu = results.get("aer_gpu", {}).get("time", t_bw) / t_bw
        if verbose:
            print(f"  Blackwell:     {t_bw*1000:8.2f} ms  fidelity={fid_bw:.10f}"
                  f"  speedup={t_ref/t_bw:.2f}x vs CPU"
                  f"  {speedup_vs_gpu:.2f}x vs Aer GPU")
    except Exception as e:
        print(f"  Blackwell:     FAILED ({e})")

    return results


def run_performance_test(name, circuit, shots=8192, warmup=3, repeats=5, verbose=True):
    """Benchmark sampling performance: Aer GPU vs Blackwell."""
    n = circuit.num_qubits
    if verbose:
        print(f"\n{'─'*70}")
        print(f"  PERF: {name}  ({n} qubits, {shots} shots, {repeats} repeats)")
        print(f"{'─'*70}")

    # Aer timing (GPU if available, else CPU)
    global _GPU_OK
    if _GPU_OK is None:
        _GPU_OK = _aer_gpu_available()
    aer_device = "GPU" if _GPU_OK else "CPU"
    aer_label = f"Aer {aer_device}"
    aer_sim = AerSimulator(method="statevector", device=aer_device)
    aer_circuit = circuit.copy()
    aer_circuit.measure_all()

    # warmup
    for _ in range(warmup):
        aer_sim.run(aer_circuit, shots=shots).result()

    aer_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        aer_sim.run(aer_circuit, shots=shots).result()
        aer_times.append(time.perf_counter() - t0)

    # Blackwell timing
    bw_sim = BlackwellSimulator()
    bw_circuit = circuit.copy()
    bw_circuit.measure_all()

    # warmup
    for _ in range(warmup):
        bw_sim.run(bw_circuit, shots=shots)

    bw_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        bw_sim.run(bw_circuit, shots=shots)
        bw_times.append(time.perf_counter() - t0)

    aer_med = np.median(aer_times) * 1000
    bw_med = np.median(bw_times) * 1000
    speedup = np.median(aer_times) / np.median(bw_times)

    if verbose:
        print(f"  {aer_label}:  median={aer_med:8.2f} ms  "
              f"[{min(aer_times)*1000:.2f}, {max(aer_times)*1000:.2f}]")
        print(f"  Blackwell:  median={bw_med:8.2f} ms  "
              f"[{min(bw_times)*1000:.2f}, {max(bw_times)*1000:.2f}]")
        print(f"  Speedup:    {speedup:.2f}x {'(Blackwell faster)' if speedup > 1 else f'({aer_label} faster)'}")

    return {
        "aer_gpu_ms": aer_med,
        "blackwell_ms": bw_med,
        "speedup": speedup,
    }


# ─── Test Suites ──────────────────────────────────────────────────────

def build_circuit_suite(qubits_list, depths=None):
    """Generate the full circuit suite for given qubit counts."""
    if depths is None:
        depths = {"clifford": 30, "rz": 50}

    suite = []
    for n in qubits_list:
        suite.append((f"GHZ-{n}", make_ghz(n)))
        suite.append((f"QFT-{n}", make_qft(n)))
        suite.append((f"Clifford-{n}", make_random_clifford(n, depth=depths["clifford"])))
        suite.append((f"EfficientSU2-{n}", make_parametric(n)))
        suite.append((f"DeepRz-{n}", make_deep_rz(n, depth=depths["rz"])))
        if n >= 3:
            suite.append((f"Toffoli-{n}", make_toffoli_chain(n)))
        if n >= 4:
            suite.append((f"QV-{n}", make_quantum_volume(n)))
    return suite


def main():
    parser = argparse.ArgumentParser(description="A/B Test: Aer GPU vs Blackwell Kernels")
    parser.add_argument("--qubits", nargs="+", type=int, default=None,
                        help="Qubit counts to test (default: 4 8 12 16 20)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (small circuits only)")
    parser.add_argument("--perf-only", action="store_true",
                        help="Skip accuracy, run performance only")
    parser.add_argument("--accuracy-only", action="store_true",
                        help="Skip performance, run accuracy only")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Number of shots for performance test")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of timing repeats")
    args = parser.parse_args()

    if args.quick:
        qubits_list = [4, 8]
        depths = {"clifford": 10, "rz": 10}
    elif args.qubits:
        qubits_list = args.qubits
        depths = {"clifford": 30, "rz": 50}
    else:
        qubits_list = [4, 8, 12, 16, 20]
        depths = {"clifford": 30, "rz": 50}

    suite = build_circuit_suite(qubits_list, depths)

    print(f"\n{'#'*70}")
    print(f"  A/B Test: Aer (cuQuantum GPU) vs Blackwell Custom Kernels")
    print(f"  Circuits: {len(suite)}  |  Qubits: {qubits_list}")
    print(f"{'#'*70}")

    # ── Accuracy tests ──
    if not args.perf_only:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: ACCURACY (Statevector Fidelity)")
        print(f"{'='*70}")

        accuracy_results = {}
        for name, circuit in suite:
            res = run_accuracy_test(name, circuit)
            if res:
                accuracy_results[name] = res

        # Summary
        print(f"\n\n{'='*70}")
        print(f"  ACCURACY SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Circuit':<25} {'Aer GPU Fid':>14} {'Blackwell Fid':>14} {'Match?':>8}")
        print(f"  {'─'*25} {'─'*14} {'─'*14} {'─'*8}")
        for name, res in accuracy_results.items():
            fid_gpu = res.get("aer_gpu", {}).get("fidelity", float("nan"))
            fid_bw = res.get("blackwell", {}).get("fidelity", float("nan"))
            match = "PASS" if fid_bw > 0.999999 else "WARN" if fid_bw > 0.9999 else "FAIL"
            print(f"  {name:<25} {fid_gpu:>14.10f} {fid_bw:>14.10f} {match:>8}")

    # ── Performance tests ──
    if not args.accuracy_only:
        print(f"\n\n{'='*70}")
        print(f"  PHASE 2: PERFORMANCE (Wall-Clock Sampling)")
        print(f"{'='*70}")

        perf_results = {}
        for name, circuit in suite:
            res = run_performance_test(name, circuit, shots=args.shots, repeats=args.repeats)
            perf_results[name] = res

        # Summary
        print(f"\n\n{'='*70}")
        print(f"  PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Circuit':<25} {'Aer GPU (ms)':>13} {'Blackwell (ms)':>15} {'Speedup':>9}")
        print(f"  {'─'*25} {'─'*13} {'─'*15} {'─'*9}")
        for name, res in perf_results.items():
            print(f"  {name:<25} {res['aer_gpu_ms']:>13.2f} {res['blackwell_ms']:>15.2f} "
                  f"{res['speedup']:>8.2f}x")

    print(f"\n{'#'*70}")
    print(f"  Done.")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
