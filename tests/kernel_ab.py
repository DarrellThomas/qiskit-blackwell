#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell
"""
A/B test between two Blackwell kernel builds.

Compare the current (stable) kernels against a new build to verify
accuracy and measure speedup before promoting the new kernels.

Usage:
    # Compare default kernels (bwk/python) vs a new build:
    python tests/kernel_ab.py /path/to/new/kernels/python

    # Compare two explicit paths:
    python tests/kernel_ab.py /path/to/kernels_a/python /path/to/kernels_b/python

    # Quick mode (small circuits only):
    python tests/kernel_ab.py --quick /path/to/new/kernels/python
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume
from qiskit_aer import AerSimulator

from blackwell_backend.simulator import BlackwellSimulator


def ref_statevector(circuit):
    """Aer CPU reference (float64)."""
    sim = AerSimulator(method="statevector", device="CPU")
    qc = circuit.copy()
    qc.save_statevector()
    return np.array(sim.run(qc, shots=0).result().data()["statevector"])


def fidelity(sv1, sv2):
    sv1 = sv1 / np.linalg.norm(sv1)
    sv2 = sv2 / np.linalg.norm(sv2)
    return float(np.abs(np.vdot(sv1, sv2)) ** 2)


def make_circuits(qubit_counts, quick=False):
    """Generate test circuits."""
    circuits = []

    for n in qubit_counts:
        # GHZ
        qc = QuantumCircuit(n, name=f"GHZ-{n}")
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        circuits.append(qc)

        # Random Clifford
        rng = np.random.default_rng(42)
        depth = 10 if quick else 30
        qc2 = QuantumCircuit(n, name=f"Clifford-{n}")
        for _ in range(depth):
            g = rng.choice(["h", "s", "cx"])
            if g == "h":
                qc2.h(rng.integers(n))
            elif g == "s":
                qc2.s(rng.integers(n))
            elif g == "cx" and n >= 2:
                q0, q1 = rng.choice(n, 2, replace=False)
                qc2.cx(int(q0), int(q1))
        circuits.append(qc2)

        # Deep Rz (diagonal kernel path)
        depth_rz = 10 if quick else 50
        qc3 = QuantumCircuit(n, name=f"DeepRz-{n}")
        for _ in range(depth_rz):
            for q in range(n):
                qc3.rz(rng.uniform(0, 2 * np.pi), q)
            for q in range(0, n - 1, 2):
                qc3.cx(q, q + 1)
        circuits.append(qc3)

        # Quantum Volume (the circuit type we're trying to improve)
        if n >= 4:
            qc4 = quantum_volume(n, seed=42)
            qc4.name = f"QV-{n}"
            circuits.append(qc4.decompose())

    return circuits


def run_comparison(sim_a, sim_b, label_a, label_b, circuits, warmup=2, repeats=5):
    """Run circuits on both simulators, compare accuracy and speed."""
    print(f"\n{'='*75}")
    print(f"  Kernel A/B Comparison")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"{'='*75}")

    results = []

    for circuit in circuits:
        name = circuit.name
        n = circuit.num_qubits
        n_gates = circuit.size()

        # Accuracy: compare both against Aer CPU reference
        sv_ref = ref_statevector(circuit)
        sv_a, _ = sim_a.run_statevector(circuit)
        sv_b, _ = sim_b.run_statevector(circuit)

        fid_a = fidelity(sv_ref, sv_a)
        fid_b = fidelity(sv_ref, sv_b)
        fid_ab = fidelity(sv_a, sv_b)

        # Speed: timed runs
        for _ in range(warmup):
            sim_a.run_statevector(circuit)
            sim_b.run_statevector(circuit)

        times_a = []
        times_b = []
        for _ in range(repeats):
            _, t = sim_a.run_statevector(circuit)
            times_a.append(t)
            _, t = sim_b.run_statevector(circuit)
            times_b.append(t)

        med_a = np.median(times_a) * 1000
        med_b = np.median(times_b) * 1000
        speedup = np.median(times_a) / np.median(times_b)

        acc_ok = fid_a > 0.99999 and fid_b > 0.99999
        winner = "B faster" if speedup > 1 else "A faster"

        print(f"\n  {name} ({n}q, {n_gates} gates)")
        print(f"    Accuracy:  A={fid_a:.10f}  B={fid_b:.10f}  A↔B={fid_ab:.10f}  "
              f"{'OK' if acc_ok else 'WARN'}")
        print(f"    Speed:     A={med_a:.2f}ms  B={med_b:.2f}ms  → {speedup:.2f}x ({winner})")

        results.append({
            "name": name, "n_qubits": n, "gates": n_gates,
            "fid_a": fid_a, "fid_b": fid_b, "fid_ab": fid_ab,
            "ms_a": med_a, "ms_b": med_b, "speedup": speedup,
        })

    # Summary table
    print(f"\n\n{'='*75}")
    print(f"  SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Circuit':<20} {'Fid A':>10} {'Fid B':>10} {'ms A':>8} {'ms B':>8} {'B/A':>7} {'Winner':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*7} {'─'*10}")
    for r in results:
        winner = "B" if r["speedup"] > 1.05 else "A" if r["speedup"] < 0.95 else "tie"
        print(f"  {r['name']:<20} {r['fid_a']:>10.8f} {r['fid_b']:>10.8f} "
              f"{r['ms_a']:>8.2f} {r['ms_b']:>8.2f} {r['speedup']:>6.2f}x {winner:>10}")

    # Promote/reject recommendation
    all_accurate = all(r["fid_b"] > 0.99999 for r in results)
    wins = sum(1 for r in results if r["speedup"] > 1.05)
    losses = sum(1 for r in results if r["speedup"] < 0.95)

    print(f"\n  Accuracy: {'ALL PASS' if all_accurate else 'FAILURES DETECTED'}")
    print(f"  Speed:    B wins {wins}/{len(results)}, A wins {losses}/{len(results)}")
    if all_accurate and wins > losses:
        print(f"  Recommendation: PROMOTE B → replace A")
    elif not all_accurate:
        print(f"  Recommendation: REJECT B — accuracy regression")
    else:
        print(f"  Recommendation: HOLD — no clear improvement")

    return results


def main():
    parser = argparse.ArgumentParser(description="A/B test two Blackwell kernel builds")
    parser.add_argument("paths", nargs="+",
                        help="One path = compare vs default. Two paths = compare against each other.")
    parser.add_argument("--quick", action="store_true", help="Small circuits only")
    parser.add_argument("--qubits", nargs="+", type=int, default=None,
                        help="Qubit counts (default: 4 8 12 for quick, 4 8 12 16 20 for full)")
    args = parser.parse_args()

    if len(args.paths) == 1:
        path_a = None  # default
        path_b = args.paths[0]
        label_a = "default (bwk/python)"
        label_b = args.paths[0]
    else:
        path_a = args.paths[0]
        path_b = args.paths[1]
        label_a = args.paths[0]
        label_b = args.paths[1]

    if args.qubits:
        qubit_counts = args.qubits
    elif args.quick:
        qubit_counts = [4, 8, 12]
    else:
        qubit_counts = [4, 8, 12, 16, 20]

    sim_a = BlackwellSimulator(kernel_path=path_a)
    sim_b = BlackwellSimulator(kernel_path=path_b)

    circuits = make_circuits(qubit_counts, quick=args.quick)
    run_comparison(sim_a, sim_b, label_a, label_b, circuits)


if __name__ == "__main__":
    main()
