# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
#
# qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
# Licensed under the MIT License. See LICENSE file in the project root.
# https://github.com/DarrellThomas/qiskit-blackwell
#
# EXPERIMENTAL (beta) — Chebyshev polynomial expansion for Hamiltonian evolution.

"""Chebyshev polynomial time evolution: exp(-iHt)|psi>.

Evaluates the matrix exponential directly as a polynomial in H using
Chebyshev expansion with Bessel function coefficients. Exponential
convergence in polynomial degree, rigorous error bounds, numerically
stable three-term recurrence.

See ROADMAP.md for theory and motivation.
"""

import time
import numpy as np
import torch
from scipy.special import jv as bessel_j


def chebyshev_coefficients(t, spectral_bound, precision=1e-10):
    """Compute Chebyshev expansion coefficients for exp(-iHt).

    EXPERIMENTAL (beta).

    exp(-iHt)|psi> = sum_{k=0}^{K} c_k * T_k(H_norm)|psi>

    where H_norm = H / spectral_bound (rescaled to [-1, 1])
    and c_k = (2 - delta_{k,0}) * (-i)^k * J_k(spectral_bound * |t|)

    Args:
        t: evolution time
        spectral_bound: upper bound on max |eigenvalue| of H
        precision: target truncation error (tail sum of |c_k|)

    Returns:
        coeffs: complex128 array [K+1] of Chebyshev coefficients
        K: polynomial degree used
        tail_bound: estimated truncation error
    """
    alpha = spectral_bound * abs(t)

    # Estimate required degree: K ~ alpha + C*log(1/precision)
    K_est = int(alpha + 2.0 * np.log(1.0 / precision) + 20)
    K_est = max(K_est, 10)

    # Compute coefficients
    coeffs = np.zeros(K_est + 1, dtype=np.complex128)
    for k in range(K_est + 1):
        prefactor = 2.0 if k > 0 else 1.0
        phase = (-1j) ** k
        coeffs[k] = prefactor * phase * bessel_j(k, alpha)

    # Find actual truncation point: where cumulative tail < precision
    abs_coeffs = np.abs(coeffs)
    cumtail = np.cumsum(abs_coeffs[::-1])[::-1]
    # Find smallest K where tail sum < precision
    K = K_est
    for k in range(K_est, 0, -1):
        if cumtail[k] < precision:
            K = k
        else:
            break

    K = max(K, 2)  # need at least T_0 and T_1
    coeffs = coeffs[:K + 1]

    # Adjust sign for negative time: exp(-iH(-|t|)) = exp(iH|t|) = conj(exp(-iH|t|))
    if t < 0:
        coeffs = np.conj(coeffs)

    tail_bound = float(cumtail[min(K, len(cumtail) - 1)])
    return coeffs, K, tail_bound


def chebyshev_evolve(state, hamiltonian_csr, t, precision=1e-10, bk=None):
    """Evolve state under Hamiltonian using Chebyshev expansion.

    EXPERIMENTAL (beta). Computes: result = exp(-i * H * t) |state>

    Uses the three-term Chebyshev recurrence:
        T_0|psi> = |psi>
        T_1|psi> = H_norm|psi>
        T_{k+1}|psi> = 2 * H_norm * T_k|psi> - T_{k-1}|psi>

    with accumulation: result = sum_k c_k * T_k|psi>

    Args:
        state: complex64 tensor [2^n] on CUDA — the initial state
        hamiltonian_csr: HamiltonianCSR object
        t: evolution time (float)
        precision: target truncation error for Chebyshev expansion
        bk: blackwell_kernels module (auto-imported if None)

    Returns:
        result: complex64 tensor [2^n] — the evolved state
        info: dict with metadata (degree, spectral_bound, tail_bound, elapsed)
    """
    if bk is None:
        import blackwell_kernels as bk

    t_start = time.perf_counter()

    n = state.shape[0]
    hcsr = hamiltonian_csr
    scale = 1.0 / hcsr.spectral_bound  # rescale H to [-1, 1]

    # Compute Bessel coefficients (CPU, microseconds)
    coeffs, K, tail_bound = chebyshev_coefficients(t, hcsr.spectral_bound, precision)

    # Allocate workspace: triple buffer + accumulator + spmv scratch
    buf = [
        state.clone(),                                    # buf[0] = T_0 = |psi>
        torch.zeros(n, dtype=torch.complex64, device=state.device),  # buf[1]
        torch.zeros(n, dtype=torch.complex64, device=state.device),  # buf[2]
    ]
    accum = torch.zeros(n, dtype=torch.complex64, device=state.device)
    spmv_out = torch.zeros(n, dtype=torch.complex64, device=state.device)

    # Accumulate T_0 term: accum += c_0 * T_0
    c0 = coeffs[0]
    bk.chebyshev_accum(accum, buf[0], float(c0.real), float(c0.imag))

    # Compute T_1 = H_norm * T_0 = scale * H * buf[0]
    bk.spmv_csr(hcsr.row_ptr, hcsr.col_idx, hcsr.values,
                buf[0], buf[1],
                scale, 0.0,   # alpha = scale (real)
                0.0, 0.0)     # beta = 0

    # Accumulate T_1 term: accum += c_1 * T_1
    c1 = coeffs[1]
    bk.chebyshev_accum(accum, buf[1], float(c1.real), float(c1.imag))

    # Chebyshev recurrence loop: T_{k+1} = 2*H_norm*T_k - T_{k-1}
    prev_idx, curr_idx, next_idx = 0, 1, 2
    for k in range(2, K + 1):
        # SpMV: spmv_out = scale * H * buf[curr_idx]
        bk.spmv_csr(hcsr.row_ptr, hcsr.col_idx, hcsr.values,
                    buf[curr_idx], spmv_out,
                    scale, 0.0,
                    0.0, 0.0)

        # Fused step + accumulate:
        #   buf[next_idx] = 2 * spmv_out - buf[prev_idx]
        #   accum += c_k * buf[next_idx]
        ck = coeffs[k]
        bk.chebyshev_step_accum(spmv_out, buf[prev_idx], buf[next_idx],
                                 accum, float(ck.real), float(ck.imag))

        # Rotate buffers: prev <- curr, curr <- next, next <- prev
        prev_idx, curr_idx, next_idx = curr_idx, next_idx, prev_idx

    elapsed = time.perf_counter() - t_start

    info = {
        "engine": "chebyshev-beta",
        "experimental": True,
        "chebyshev_degree": K,
        "spectral_bound": hcsr.spectral_bound,
        "truncation_error_bound": tail_bound,
        "spmv_count": K,
        "workspace_vectors": 5,
        "workspace_bytes": 5 * n * 8,
        "elapsed_s": elapsed,
    }

    return accum, info
