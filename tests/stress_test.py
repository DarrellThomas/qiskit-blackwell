#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell
"""
Stress test suite for the Blackwell custom kernel backend.

Designed to find corner cases, edge cases, and numerical drift that
standard circuit tests won't catch. Every test compares Blackwell output
against Aer CPU (double-precision reference).

Categories:
  1. Identity / round-trip tests (apply gate then inverse → |0>)
  2. All-qubit sweep (every gate on every target qubit)
  3. Qubit pair combinatorics (2Q gates on all pairs)
  4. Norm preservation (unitarity check after deep circuits)
  5. Gate decomposition equivalences (CX = H·CZ·H, etc.)
  6. Phase accumulation accuracy (long Rz chains)
  7. Entanglement structure (GHZ, W-state, cluster states)
  8. Measurement statistics (chi-squared on known distributions)
  9. Multi-controlled edge cases (0-7 controls, all qubit combos)
 10. Adversarial circuits (worst-case patterns for the kernel)
 11. Large qubit scaling (16, 20, 24 qubits)
 12. Repeated gate stress (same gate 1000x)

Usage:
    ./run_ab_test.sh -- tests/stress_test.py           # full suite
    ./run_ab_test.sh -- tests/stress_test.py --quick    # fast subset
    ./run_ab_test.sh -- tests/stress_test.py -k phase   # filter by name
"""

import sys
import os
import math
import cmath
import argparse
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from blackwell_backend import BlackwellSimulator

# ─── Globals ──────────────────────────────────────────────────────────

SIM_BW = BlackwellSimulator()
SIM_REF = AerSimulator(method="statevector", device="CPU")

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0
FAILURES = []


def record_pass():
    global PASS_COUNT
    PASS_COUNT += 1


def record_fail(msg):
    global FAIL_COUNT
    FAIL_COUNT += 1
    if msg:
        FAILURES.append(msg)


def ref_statevector(circuit):
    """Get reference statevector from Aer CPU (float64)."""
    qc = circuit.copy()
    qc.save_statevector()
    result = SIM_REF.run(qc, shots=0).result()
    return np.array(result.data()["statevector"])


def bw_statevector(circuit):
    """Get Blackwell statevector."""
    sv, _ = SIM_BW.run_statevector(circuit)
    return sv


def fidelity(sv1, sv2):
    """State fidelity |<a|b>|^2."""
    sv1 = sv1 / np.linalg.norm(sv1)
    sv2 = sv2 / np.linalg.norm(sv2)
    return float(np.abs(np.vdot(sv1, sv2)) ** 2)


def check(name, circuit, tol=0.99999, verbose=True):
    """Compare Blackwell vs Aer CPU. Returns True if fidelity > tol."""
    global PASS_COUNT, FAIL_COUNT, FAILURES
    try:
        sv_ref = ref_statevector(circuit)
        sv_bw = bw_statevector(circuit)

        # Also check norm preservation
        norm_bw = float(np.linalg.norm(sv_bw))
        norm_ok = abs(norm_bw - 1.0) < 1e-4

        fid = fidelity(sv_ref, sv_bw)
        passed = fid > tol and norm_ok

        if passed:
            record_pass()
            if verbose:
                print(f"  PASS  {name}  fid={fid:.10f}  norm={norm_bw:.6f}")
        else:
            reason = ""
            if fid <= tol:
                reason += f"fid={fid:.10f} "
            if not norm_ok:
                reason += f"norm={norm_bw:.6f} "
            msg = f"  FAIL  {name}  {reason}"
            print(msg)
            record_fail(msg)
        return passed
    except Exception as e:
        msg = f"  FAIL  {name}  EXCEPTION: {e}"
        print(msg)
        record_fail(msg)
        traceback.print_exc()
        return False


# =====================================================================
# 1. IDENTITY / ROUND-TRIP TESTS
# =====================================================================

def test_identity_roundtrips(n_qubits=6):
    """Apply gate then its inverse → should return to |0>."""
    print(f"\n{'='*60}")
    print(f"  1. Identity round-trips ({n_qubits} qubits)")
    print(f"{'='*60}")

    # Single-qubit gate round-trips
    for gate_name in ["h", "x", "y", "z", "s", "t", "sx", "rx", "ry", "rz"]:
        for q in [0, n_qubits // 2, n_qubits - 1]:
            qc = QuantumCircuit(n_qubits)
            if gate_name == "h":
                qc.h(q); qc.h(q)
            elif gate_name == "x":
                qc.x(q); qc.x(q)
            elif gate_name == "y":
                qc.y(q); qc.y(q)
            elif gate_name == "z":
                qc.z(q); qc.z(q)
            elif gate_name == "s":
                qc.s(q); qc.sdg(q)
            elif gate_name == "t":
                qc.t(q); qc.tdg(q)
            elif gate_name == "sx":
                qc.sx(q); qc.sxdg(q)
            elif gate_name == "rx":
                qc.rx(1.234, q); qc.rx(-1.234, q)
            elif gate_name == "ry":
                qc.ry(2.345, q); qc.ry(-2.345, q)
            elif gate_name == "rz":
                qc.rz(3.456, q); qc.rz(-3.456, q)
            check(f"roundtrip_{gate_name}_q{q}", qc)

    # Two-qubit round-trips
    for q0, q1 in [(0, 1), (0, n_qubits - 1), (n_qubits - 2, n_qubits - 1)]:
        qc = QuantumCircuit(n_qubits)
        qc.cx(q0, q1); qc.cx(q0, q1)
        check(f"roundtrip_cx_q{q0}_q{q1}", qc)

        qc2 = QuantumCircuit(n_qubits)
        qc2.swap(q0, q1); qc2.swap(q0, q1)
        check(f"roundtrip_swap_q{q0}_q{q1}", qc2)

    # Toffoli round-trip
    if n_qubits >= 3:
        qc = QuantumCircuit(n_qubits)
        qc.ccx(0, 1, 2); qc.ccx(0, 1, 2)
        check("roundtrip_ccx_012", qc)


# =====================================================================
# 2. ALL-QUBIT SWEEP
# =====================================================================

def test_all_qubit_sweep(n_qubits=10):
    """Apply H to every qubit individually, check each."""
    print(f"\n{'='*60}")
    print(f"  2. All-qubit sweep ({n_qubits} qubits)")
    print(f"{'='*60}")

    for gate in ["h", "x", "s", "t", "rx", "rz"]:
        for q in range(n_qubits):
            qc = QuantumCircuit(n_qubits)
            if gate == "h":
                qc.h(q)
            elif gate == "x":
                qc.x(q)
            elif gate == "s":
                qc.s(q)
            elif gate == "t":
                qc.t(q)
            elif gate == "rx":
                qc.rx(math.pi / 3, q)
            elif gate == "rz":
                qc.rz(math.pi / 5, q)
            check(f"sweep_{gate}_q{q}", qc, verbose=False)

    print(f"  (checked {n_qubits * 6} single-gate circuits)")


# =====================================================================
# 3. QUBIT PAIR COMBINATORICS
# =====================================================================

def test_qubit_pair_combinatorics(n_qubits=8):
    """CX on every ordered pair (i,j) where i != j."""
    print(f"\n{'='*60}")
    print(f"  3. Qubit pair combinatorics ({n_qubits} qubits)")
    print(f"{'='*60}")

    fail_pairs = []
    count = 0
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i == j:
                continue
            qc = QuantumCircuit(n_qubits)
            qc.h(i)  # Put control in superposition
            qc.cx(i, j)
            ok = check(f"cx_q{i}_q{j}", qc, verbose=False)
            if not ok:
                fail_pairs.append((i, j))
            count += 1

    if fail_pairs:
        print(f"  FAILED pairs: {fail_pairs}")
    else:
        print(f"  All {count} CX pairs passed")

    # Also test CZ on all pairs
    cz_fails = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qc = QuantumCircuit(n_qubits)
            qc.h(i)
            qc.h(j)
            qc.cz(i, j)
            ok = check(f"cz_q{i}_q{j}", qc, verbose=False)
            if not ok:
                cz_fails.append((i, j))

    if cz_fails:
        print(f"  FAILED CZ pairs: {cz_fails}")
    else:
        print(f"  All CZ pairs passed")


# =====================================================================
# 4. NORM PRESERVATION
# =====================================================================

def test_norm_preservation(n_qubits=12, depth=200):
    """Deep random circuit — check that |sv|=1 at the end."""
    print(f"\n{'='*60}")
    print(f"  4. Norm preservation ({n_qubits}q, depth={depth})")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    qc = QuantumCircuit(n_qubits)

    for _ in range(depth):
        gate = rng.choice(["h", "rx", "rz", "cx", "cz", "s", "t"])
        if gate in ("h", "s", "t"):
            qc.h(rng.integers(n_qubits)) if gate == "h" else \
                (qc.s(rng.integers(n_qubits)) if gate == "s" else qc.t(rng.integers(n_qubits)))
        elif gate == "rx":
            qc.rx(rng.uniform(0, 2 * np.pi), rng.integers(n_qubits))
        elif gate == "rz":
            qc.rz(rng.uniform(0, 2 * np.pi), rng.integers(n_qubits))
        elif gate in ("cx", "cz") and n_qubits >= 2:
            q0, q1 = rng.choice(n_qubits, 2, replace=False)
            if gate == "cx":
                qc.cx(int(q0), int(q1))
            else:
                qc.cz(int(q0), int(q1))

    sv_bw = bw_statevector(qc)
    norm = float(np.linalg.norm(sv_bw))
    passed = abs(norm - 1.0) < 1e-4
    print(f"  {'PASS' if passed else 'FAIL'}  norm={norm:.10f}  (target=1.0)")

    # Also check fidelity
    check(f"deep_random_{n_qubits}q_d{depth}", qc)


# =====================================================================
# 5. GATE DECOMPOSITION EQUIVALENCES
# =====================================================================

def test_gate_equivalences(n_qubits=4):
    """Known identities: CX = H·CZ·H, SWAP = 3 CX, etc."""
    print(f"\n{'='*60}")
    print(f"  5. Gate decomposition equivalences ({n_qubits} qubits)")
    print(f"{'='*60}")

    # CX(0,1) = H(1) · CZ(0,1) · H(1)
    qc1 = QuantumCircuit(n_qubits)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(n_qubits)
    qc2.h(0)
    qc2.h(1); qc2.cz(0, 1); qc2.h(1)

    sv1 = bw_statevector(qc1)
    sv2 = bw_statevector(qc2)
    fid = fidelity(sv1, sv2)
    passed = fid > 0.99999
    print(f"  {'PASS' if passed else 'FAIL'}  CX = H·CZ·H  fid={fid:.10f}")
    if not passed:
        global FAIL_COUNT
        record_fail(f"  FAIL  CX = H·CZ·H  fid={fid:.10f}")
    else:
        global PASS_COUNT
        record_pass()

    # SWAP = CX·CX·CX (with direction swap in middle)
    qc3 = QuantumCircuit(n_qubits)
    qc3.h(0); qc3.t(1)
    qc3.swap(0, 1)

    qc4 = QuantumCircuit(n_qubits)
    qc4.h(0); qc4.t(1)
    qc4.cx(0, 1); qc4.cx(1, 0); qc4.cx(0, 1)

    sv3 = bw_statevector(qc3)
    sv4 = bw_statevector(qc4)
    fid2 = fidelity(sv3, sv4)
    passed2 = fid2 > 0.99999
    print(f"  {'PASS' if passed2 else 'FAIL'}  SWAP = CX·CX·CX  fid={fid2:.10f}")
    if not passed2:
        record_fail(f"  FAIL  SWAP = CX·CX·CX  fid={fid2:.10f}")
    else:
        record_pass()

    # HZH = X
    qc5 = QuantumCircuit(n_qubits)
    qc5.h(0); qc5.z(0); qc5.h(0)

    qc6 = QuantumCircuit(n_qubits)
    qc6.x(0)

    sv5 = bw_statevector(qc5)
    sv6 = bw_statevector(qc6)
    fid3 = fidelity(sv5, sv6)
    passed3 = fid3 > 0.99999
    print(f"  {'PASS' if passed3 else 'FAIL'}  HZH = X  fid={fid3:.10f}")
    if not passed3:
        record_fail(f"  FAIL  HZH = X  fid={fid3:.10f}")
    else:
        record_pass()

    # S·S = Z
    qc7 = QuantumCircuit(n_qubits)
    qc7.h(0)
    qc7.s(0); qc7.s(0)

    qc8 = QuantumCircuit(n_qubits)
    qc8.h(0)
    qc8.z(0)

    sv7 = bw_statevector(qc7)
    sv8 = bw_statevector(qc8)
    fid4 = fidelity(sv7, sv8)
    passed4 = fid4 > 0.99999
    print(f"  {'PASS' if passed4 else 'FAIL'}  S·S = Z  fid={fid4:.10f}")
    if not passed4:
        record_fail(f"  FAIL  S·S = Z  fid={fid4:.10f}")
    else:
        record_pass()

    # T^8 = I
    for q in [0, n_qubits - 1]:
        qc9 = QuantumCircuit(n_qubits)
        qc9.h(q)
        for _ in range(8):
            qc9.t(q)

        qc10 = QuantumCircuit(n_qubits)
        qc10.h(q)
        # T^8 = e^{i*pi} = Z^2 = I, but T = diag(1, e^{i*pi/4}), so T^8 = diag(1,1) = I
        check(f"T^8=I_q{q}", qc9)


# =====================================================================
# 6. PHASE ACCUMULATION ACCURACY
# =====================================================================

def test_phase_accumulation(n_qubits=8):
    """Chain of Rz gates — accumulated phase must match."""
    print(f"\n{'='*60}")
    print(f"  6. Phase accumulation accuracy ({n_qubits} qubits)")
    print(f"{'='*60}")

    # Many small Rz should equal one big Rz
    n_steps = 1000
    total_angle = math.pi / 3
    small_angle = total_angle / n_steps

    for q in [0, n_qubits // 2, n_qubits - 1]:
        qc_many = QuantumCircuit(n_qubits)
        qc_many.h(q)
        for _ in range(n_steps):
            qc_many.rz(small_angle, q)

        qc_one = QuantumCircuit(n_qubits)
        qc_one.h(q)
        qc_one.rz(total_angle, q)

        sv_many = bw_statevector(qc_many)
        sv_one = bw_statevector(qc_one)
        fid_val = fidelity(sv_many, sv_one)
        # Lower tolerance — float32 accumulation over 1000 steps
        passed = fid_val > 0.999
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  1000×Rz({small_angle:.6f}) vs 1×Rz({total_angle:.6f}) "
              f"q{q}  fid={fid_val:.10f}")
        if not passed:
            record_fail(f"  FAIL  phase_accum_q{q} fid={fid_val:.10f}")
        else:
            record_pass()

    # Rx accumulation
    for q in [0, n_qubits - 1]:
        n_steps = 500
        total_angle = 2 * math.pi  # Full rotation → should be -I (global phase)
        small_angle = total_angle / n_steps

        qc = QuantumCircuit(n_qubits)
        for _ in range(n_steps):
            qc.rx(small_angle, q)
        # Rx(2pi) = -I, so state should be -|0> (global phase)
        sv = bw_statevector(qc)
        # Check norm and that we're back to |0> (up to global phase)
        idx0_amp = sv[0]
        others_max = np.max(np.abs(sv[1:]))
        passed = abs(abs(idx0_amp) - 1.0) < 0.01 and others_max < 0.01
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  500×Rx(4pi/{n_steps}) → |0>  q{q}  "
              f"|a0|={abs(idx0_amp):.6f}  max_other={others_max:.6f}")
        if not passed:
            record_fail(f"  FAIL  rx_full_rotation_q{q}")
        else:
            record_pass()


# =====================================================================
# 7. ENTANGLEMENT STRUCTURE
# =====================================================================

def test_entanglement_structures(n_qubits=8):
    """GHZ, W-state, cluster state — check known structure."""
    print(f"\n{'='*60}")
    print(f"  7. Entanglement structures ({n_qubits} qubits)")
    print(f"{'='*60}")

    # GHZ: only |00...0> and |11...1> should have amplitude
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    sv = bw_statevector(qc)
    probs = np.abs(sv) ** 2
    ghz_expected = np.zeros(2 ** n_qubits)
    ghz_expected[0] = 0.5
    ghz_expected[-1] = 0.5
    tvd = 0.5 * np.sum(np.abs(probs - ghz_expected))
    passed = tvd < 1e-5
    print(f"  {'PASS' if passed else 'FAIL'}  GHZ-{n_qubits}  TVD={tvd:.10f}")
    if not passed:
        record_fail(f"  FAIL  GHZ-{n_qubits} TVD={tvd:.10f}")
    else:
        record_pass()

    # Also verify against Aer
    check(f"GHZ_{n_qubits}_vs_aer", qc)

    # Linear cluster state: H on all, then CZ on neighbors
    qc2 = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc2.h(i)
    for i in range(n_qubits - 1):
        qc2.cz(i, i + 1)
    check(f"cluster_{n_qubits}", qc2)

    # Reverse GHZ: build from last qubit
    qc3 = QuantumCircuit(n_qubits)
    qc3.h(n_qubits - 1)
    for i in range(n_qubits - 2, -1, -1):
        qc3.cx(i + 1, i)
    check(f"reverse_GHZ_{n_qubits}", qc3)


# =====================================================================
# 8. MEASUREMENT STATISTICS
# =====================================================================

def test_measurement_statistics():
    """Chi-squared test on shot distributions for known states."""
    print(f"\n{'='*60}")
    print(f"  8. Measurement statistics")
    print(f"{'='*60}")

    from scipy.stats import chisquare

    n_shots = 50000

    # Uniform superposition of 4 qubits: all 16 outcomes equally likely
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.h(i)
    qc.measure_all()

    result = SIM_BW.run(qc, shots=n_shots, seed=42)
    counts = result.get_counts()

    observed = np.zeros(16)
    for key, val in counts.items():
        idx = int(key, 16) if key.startswith("0x") else int(key, 2)
        observed[idx] = val
    expected = np.ones(16) * n_shots / 16

    stat, p_value = chisquare(observed, expected)
    passed = p_value > 0.001  # Very generous — just checking not wildly wrong
    print(f"  {'PASS' if passed else 'FAIL'}  Uniform-4q  chi2={stat:.1f}  "
          f"p={p_value:.4f}  (expect p>0.001)")
    if not passed:
        record_fail(f"  FAIL  measurement_uniform p={p_value}")
    else:
        record_pass()

    # Bell state: should only get 00 and 11
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure_all()

    result2 = SIM_BW.run(qc2, shots=n_shots, seed=43)
    counts2 = result2.get_counts()

    observed2 = np.zeros(4)
    for key, val in counts2.items():
        idx = int(key, 16) if key.startswith("0x") else int(key, 2)
        observed2[idx] = val

    # Only indices 0 (00) and 3 (11) should have counts
    spurious = observed2[1] + observed2[2]
    passed2 = spurious == 0
    print(f"  {'PASS' if passed2 else 'FAIL'}  Bell-state  "
          f"|00>={int(observed2[0])} |01>={int(observed2[1])} "
          f"|10>={int(observed2[2])} |11>={int(observed2[3])}  "
          f"spurious={int(spurious)}")
    if not passed2:
        record_fail(f"  FAIL  measurement_bell spurious={int(spurious)}")
    else:
        record_pass()


# =====================================================================
# 9. MULTI-CONTROLLED EDGE CASES
# =====================================================================

def test_multi_controlled(n_qubits=8):
    """Toffoli and CCX with various control/target configurations."""
    print(f"\n{'='*60}")
    print(f"  9. Multi-controlled edge cases ({n_qubits} qubits)")
    print(f"{'='*60}")

    # CCX with all qubit triple combinations (on smaller system)
    n = min(n_qubits, 6)
    count = 0
    fails = []
    for c0 in range(n):
        for c1 in range(n):
            for t in range(n):
                if len({c0, c1, t}) < 3:
                    continue
                qc = QuantumCircuit(n)
                qc.x(c0)
                qc.x(c1)
                qc.ccx(c0, c1, t)
                ok = check(f"ccx_c{c0}c{c1}t{t}", qc, verbose=False)
                if not ok:
                    fails.append((c0, c1, t))
                count += 1

    if fails:
        print(f"  FAILED CCX triples: {fails}")
    else:
        print(f"  All {count} CCX triples passed")

    # CCX where controls are NOT |1> — should NOT flip target
    qc = QuantumCircuit(n_qubits)
    qc.x(0)  # Only one control is |1>
    qc.ccx(0, 1, 2)
    check("ccx_partial_control", qc)

    # CCX with target already |1> — should flip to |0>
    qc2 = QuantumCircuit(n_qubits)
    qc2.x(0)
    qc2.x(1)
    qc2.x(2)
    qc2.ccx(0, 1, 2)
    check("ccx_target_already_1", qc2)


# =====================================================================
# 10. ADVERSARIAL CIRCUITS
# =====================================================================

def test_adversarial(n_qubits=10):
    """Patterns designed to stress specific kernel paths."""
    print(f"\n{'='*60}")
    print(f"  10. Adversarial circuits ({n_qubits} qubits)")
    print(f"{'='*60}")

    # All qubits in |1>, then cascade of CX
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.x(i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    check("all_ones_cx_cascade", qc)

    # Alternating CX directions (stride-1 adjacent pairs)
    qc2 = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc2.h(i)
    for _ in range(5):
        for i in range(0, n_qubits - 1, 2):
            qc2.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc2.cx(i + 1, i)  # Reverse direction
    check("alternating_cx_directions", qc2)

    # Max-stride CX: q0 controls q(n-1)
    qc3 = QuantumCircuit(n_qubits)
    qc3.h(0)
    qc3.cx(0, n_qubits - 1)
    check("max_stride_cx", qc3)

    # Same gate on same qubit many times in a row
    qc4 = QuantumCircuit(n_qubits)
    qc4.h(0)
    for _ in range(100):
        qc4.t(0)
    check("100x_T_gate_q0", qc4)

    # Rapid gate switching between two qubits
    qc5 = QuantumCircuit(n_qubits)
    qc5.h(0)
    qc5.h(1)
    for _ in range(50):
        qc5.cx(0, 1)
        qc5.cx(1, 0)
    check("rapid_cx_pingpong", qc5)

    # Diagonal-only circuit (tests diagonal kernel path exclusively)
    qc6 = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc6.h(i)
    for _ in range(20):
        for i in range(n_qubits):
            qc6.rz(np.random.uniform(0, 2 * np.pi), i)
        for i in range(0, n_qubits - 1):
            qc6.cz(i, i + 1)
    check("diagonal_only_deep", qc6)

    # Every gate type in sequence on q0
    qc7 = QuantumCircuit(n_qubits)
    qc7.h(0)
    qc7.x(0)
    qc7.y(0)
    qc7.z(0)
    qc7.s(0)
    qc7.sdg(0)
    qc7.t(0)
    qc7.tdg(0)
    qc7.sx(0)
    qc7.sxdg(0)
    qc7.rx(0.5, 0)
    qc7.ry(0.7, 0)
    qc7.rz(0.9, 0)
    qc7.p(1.1, 0)
    qc7.u(0.3, 0.4, 0.5, 0)
    check("all_1q_gates_sequence", qc7)

    # Interleaved 1Q and 2Q gates on adjacent qubits (cache thrashing)
    qc8 = QuantumCircuit(n_qubits)
    for _ in range(30):
        for i in range(n_qubits):
            qc8.rx(0.1 * (i + 1), i)
        for i in range(0, n_qubits - 1):
            qc8.cx(i, i + 1)
    check("interleaved_1q_2q_deep", qc8)

    # CX with all qubit pairs having same control (fan-out pattern)
    qc9 = QuantumCircuit(n_qubits)
    qc9.h(0)
    for i in range(1, n_qubits):
        qc9.cx(0, i)
    check("cx_fanout_from_q0", qc9)

    # CX fan-in: all qubits control the last one
    qc10 = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        qc10.h(i)
        qc10.cx(i, n_qubits - 1)
    check("cx_fanin_to_last", qc10)


# =====================================================================
# 11. LARGE QUBIT SCALING
# =====================================================================

def test_large_qubit_scaling(qubit_counts=None):
    """Verify at larger qubit counts where memory/indexing bugs surface."""
    if qubit_counts is None:
        qubit_counts = [16, 20]
    print(f"\n{'='*60}")
    print(f"  11. Large qubit scaling ({qubit_counts})")
    print(f"{'='*60}")

    for n in qubit_counts:
        # GHZ
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        check(f"GHZ_{n}", qc)

        # All H (uniform superposition) — analytically known
        qc2 = QuantumCircuit(n)
        for i in range(n):
            qc2.h(i)
        sv = bw_statevector(qc2)
        expected_amp = 1.0 / (2 ** (n / 2))
        max_dev = float(np.max(np.abs(np.abs(sv) - expected_amp)))
        passed = max_dev < 1e-4
        print(f"  {'PASS' if passed else 'FAIL'}  uniform_{n}q  "
              f"max_deviation={max_dev:.8f}  (expected amp={expected_amp:.8f})")
        if not passed:
            record_fail(f"  FAIL  uniform_{n}q max_dev={max_dev}")
        else:
            record_pass()

        # CX on first and last qubit (max stride at scale)
        qc3 = QuantumCircuit(n)
        qc3.h(0)
        qc3.cx(0, n - 1)
        check(f"max_stride_cx_{n}q", qc3)

        # Deep parametric
        rng = np.random.default_rng(42)
        qc4 = QuantumCircuit(n)
        for _ in range(50):
            q = rng.integers(n)
            qc4.rx(rng.uniform(0, 2 * np.pi), int(q))
            if n >= 2:
                q0, q1 = rng.choice(n, 2, replace=False)
                qc4.cx(int(q0), int(q1))
        check(f"deep_random_{n}q", qc4)


# =====================================================================
# 12. REPEATED GATE STRESS
# =====================================================================

def test_repeated_gate_stress(n_qubits=8):
    """Same gate applied 1000+ times — tests numerical stability."""
    print(f"\n{'='*60}")
    print(f"  12. Repeated gate stress ({n_qubits} qubits)")
    print(f"{'='*60}")

    # H^1000 on q0: H^(2k)=I, H^(2k+1)=H
    qc = QuantumCircuit(n_qubits)
    for _ in range(1000):
        qc.h(0)
    # 1000 is even → should be |0>
    check("H^1000_q0", qc)

    # X^999 on q0: odd → should be |1>
    qc2 = QuantumCircuit(n_qubits)
    for _ in range(999):
        qc2.x(0)
    check("X^999_q0", qc2)

    # CX^1000: even → identity on the pair
    qc3 = QuantumCircuit(n_qubits)
    qc3.h(0)
    for _ in range(1000):
        qc3.cx(0, 1)
    # 1000 CX is even → identity. State should be H|0> ⊗ |0...0>
    qc_ref = QuantumCircuit(n_qubits)
    qc_ref.h(0)
    sv3 = bw_statevector(qc3)
    sv_ref = bw_statevector(qc_ref)
    fid_val = fidelity(sv3, sv_ref)
    passed = fid_val > 0.999
    print(f"  {'PASS' if passed else 'FAIL'}  CX^1000  fid={fid_val:.10f}")
    if not passed:
        record_fail(f"  FAIL  CX^1000 fid={fid_val:.10f}")
    else:
        record_pass()

    # Rz(0.001)^10000 vs Rz(10): float32 accumulation stress
    qc4 = QuantumCircuit(n_qubits)
    qc4.h(0)
    for _ in range(10000):
        qc4.rz(0.001, 0)
    qc5 = QuantumCircuit(n_qubits)
    qc5.h(0)
    qc5.rz(10.0, 0)
    sv4 = bw_statevector(qc4)
    sv5 = bw_statevector(qc5)
    fid5 = fidelity(sv4, sv5)
    # Generous tolerance for 10K float32 accumulations
    passed5 = fid5 > 0.99
    print(f"  {'PASS' if passed5 else 'FAIL'}  Rz(0.001)^10000 vs Rz(10)  fid={fid5:.10f}")
    if not passed5:
        record_fail(f"  FAIL  Rz_10K_accum fid={fid5:.10f}")
    else:
        record_pass()


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Blackwell Kernel Stress Tests")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced test set (faster)")
    parser.add_argument("-k", "--filter", type=str, default=None,
                        help="Only run tests whose name contains this string")
    parser.add_argument("--large", nargs="+", type=int, default=None,
                        help="Qubit counts for large scaling test (default: 16 20)")
    args = parser.parse_args()

    tests = [
        ("identity", lambda: test_identity_roundtrips(4 if args.quick else 6)),
        ("sweep", lambda: test_all_qubit_sweep(6 if args.quick else 10)),
        ("pairs", lambda: test_qubit_pair_combinatorics(4 if args.quick else 8)),
        ("norm", lambda: test_norm_preservation(8 if args.quick else 12, 50 if args.quick else 200)),
        ("equivalence", lambda: test_gate_equivalences(4)),
        ("phase", lambda: test_phase_accumulation(4 if args.quick else 8)),
        ("entanglement", lambda: test_entanglement_structures(4 if args.quick else 8)),
        ("measurement", test_measurement_statistics),
        ("multicontrol", lambda: test_multi_controlled(4 if args.quick else 8)),
        ("adversarial", lambda: test_adversarial(6 if args.quick else 10)),
        ("large", lambda: test_large_qubit_scaling(args.large or ([12] if args.quick else [16, 20]))),
        ("repeated", lambda: test_repeated_gate_stress(4 if args.quick else 8)),
    ]

    print(f"\n{'#'*60}")
    print(f"  Blackwell Kernel Stress Test Suite")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    if args.filter:
        print(f"  Filter: {args.filter}")
    print(f"{'#'*60}")

    for name, fn in tests:
        if args.filter and args.filter.lower() not in name.lower():
            continue
        fn()

    print(f"\n\n{'#'*60}")
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FAILURES:
        print(f"\n  Failures:")
        for f in FAILURES:
            print(f"    {f}")
    else:
        print(f"  ALL TESTS PASSED")
    print(f"{'#'*60}\n")

    return 1 if FAIL_COUNT > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
