# Roadmap

## Where We Are (v0.2)

We used our CUDA factory to improve twelve CUDA kernels covering the full statevector simulation
primitive set: single/two-qubit gates, multi-controlled gates, diagonal gates,
measurement, sampling, expectation values, noise channels, circuit fusion, and
batched simulation. All kernels use compensated floating-point arithmetic
(TwoProduct + Kahan summation) to achieve float64-class fidelity at float32
throughput. 2-547x faster than Aer GPU (cuQuantum) across all tested circuits.

See [README.md](README.md) for benchmarks, correctness results, and architecture details.

## What's Next: Chebyshev Polynomial Expansion for Hamiltonian Simulation

### The Problem

Gate-by-gate simulation applies unitaries one at a time. Each gate application
reads the full state vector, transforms it, and writes it back. For a circuit
with N gates, that's N full read-write passes over the state vector -- even when
the circuit represents a single physical operation like time evolution under a
Hamiltonian.

Our compensated arithmetic reduces per-gate rounding error to ~1-2 ULP, but
errors still accumulate across gates. A 220-gate DeepRz circuit reaches
fidelity residuals of ~1e-5. For deep variational circuits (VQE, QAOA) or
Hamiltonian simulation with fine Trotter steps, that error budget gets tight.

There's a deeper issue: Trotterization itself introduces **algorithmic error**
on top of floating-point error. Decomposing `exp(-iHt)` into a product of
single/two-qubit gates is an approximation. Reducing the Trotter step size
improves algorithmic accuracy but increases gate count, which increases
floating-point error. These two error sources fight each other.

### The Solution: Chebyshev Expansion

Instead of decomposing the evolution operator into gates, evaluate it directly
as a polynomial in the Hamiltonian:

```
exp(-iHt)|psi> = sum_{k=0}^{K} c_k(t) * T_k(H_norm)|psi>
```

where `T_k` are Chebyshev polynomials of the first kind, `H_norm` is the
Hamiltonian rescaled to [-1, 1], and `c_k` are analytically known coefficients
(Bessel functions of the first kind, scaled by powers of -i).

The recurrence relation makes this GPU-friendly:

```
T_0|psi> = |psi>
T_1|psi> = H_norm|psi>
T_{k+1}|psi> = 2 * H_norm * T_k|psi> - T_{k-1}|psi>
```

Each iteration is one sparse matrix-vector product (SpMV) plus a vector update
-- exactly the kind of bandwidth-bound operation our kernels are optimized for.

### Why This Is Better

**Exponential convergence.** Chebyshev expansion converges exponentially in the
polynomial degree K, not polynomially like Trotter. For a given target precision
epsilon, you need approximately `K ~ ||H||*t + log(1/epsilon)` terms. A 220-gate
Trotter circuit might be replaced by 20-30 Chebyshev iterations at equal or
better accuracy.

**Rigorous error bound.** The truncation error is bounded by the tail of the
Bessel function series -- known analytically before you run anything. You choose
K to hit your fidelity target, and the bound is guaranteed. No hoping that
floating-point errors cancel.

**Numerical stability.** Chebyshev polynomials are bounded on [-1, 1] by
definition. The recurrence `T_{k+1} = 2x*T_k - T_{k-1}` does not amplify
errors the way naive Taylor expansion or power iteration would. Combined with
our compensated arithmetic on the SpMV, this should push fidelity to the float32
machine limit across deep evolutions.

**Fewer memory passes.** K Chebyshev iterations = K SpMV operations. Each SpMV
reads the state vector and the (sparse) Hamiltonian, writes one vector. For a
typical quantum chemistry Hamiltonian with K=30 terms to reach 1e-10 precision,
that's 30 memory passes -- compared to hundreds or thousands of gate applications
for equivalent Trotter accuracy.

### What This Looks Like as a Kernel

```
chebyshev_evolve(state, hamiltonian_csr, time, precision)
```

**Inputs:**
- State vector: complex64, same format as existing kernels
- Hamiltonian: sparse CSR (compressed sparse row), complex64 values
- Evolution time: float64 (high precision for coefficient computation)
- Target precision: determines polynomial degree K automatically

**Kernel structure:**
1. Compute Bessel coefficients `c_k` on CPU (trivial, microseconds)
2. Rescale Hamiltonian spectral range to [-1, 1] (one-time SpMV for spectral bound estimate)
3. Chebyshev recurrence loop (K iterations on GPU):
   - SpMV: `T_{k+1} = 2 * H_norm * T_k - T_{k-1}` (bandwidth-bound, our wheelhouse)
   - Accumulate: `result += c_k * T_k` (vector scale-and-add)
4. Three state-sized vectors in flight: `T_{k-1}`, `T_k`, accumulator

**Memory budget:** Three state vectors. At Q=20 that's 24 MB -- fits in L2 with
room for the Hamiltonian. At Q=24 it's 384 MB -- streaming regime, same
optimization playbook as our existing gate kernels (vectorized loads, cp.async
prefetch, L2 tiling).

**SpMV optimization on sm_120a:** The Hamiltonian is typically very sparse for
physically local interactions (nearest-neighbor, Heisenberg, Hubbard). CSR SpMV
on the 5090's 1792 GB/s GDDR7 with 96 MB L2 should achieve >80% of peak for
the state vector access pattern. The Hamiltonian matrix itself may fit entirely
in L2 for systems up to ~20 qubits.

### Where It Applies

Chebyshev expansion is the right tool when the circuit **is** a Hamiltonian
evolution -- either explicitly or through Trotterization:

| Use case | Current approach | Chebyshev benefit |
|----------|-----------------|-------------------|
| Quantum chemistry (VQE) | Trotter circuit, 100s of gates | 20-30 iterations, better fidelity |
| Condensed matter (Hubbard, Heisenberg) | Trotter or direct simulation | Exponential convergence in K |
| QAOA with problem Hamiltonian | Gate decomposition | Direct evolution, fewer passes |
| Time-dependent simulation | Small Trotter steps | Adaptive K per time slice |
| Spectral analysis (DOS, Green's functions) | Gate-based phase estimation | Native Chebyshev moment expansion |

For **arbitrary gate circuits** (random circuits, QV, Clifford-heavy), the
existing gate-by-gate path with compensated arithmetic remains optimal. Chebyshev
adds a second simulation mode, not a replacement.

### Estimated Effort

This is a substantial kernel -- SpMV is a different access pattern from our
existing strided gate kernels. Rough breakdown:

1. **CSR SpMV kernel for sm_120a** -- the core. Vectorized row processing,
   L2 exploitation for the matrix, coalesced vector access. This is the new
   optimization target.
2. **Chebyshev recurrence driver** -- orchestrates the K iterations. CUDA
   graphs for the recurrence loop (identical structure every iteration).
3. **Coefficient computation** -- Bessel functions, spectral bounds. CPU-side,
   straightforward.
4. **Qiskit integration** -- accept a `SparsePauliOp` or Hamiltonian, convert
   to CSR, dispatch to the Chebyshev kernel.

### Open Questions

- **Spectral bound estimation.** The rescaling to [-1, 1] requires knowing
  (or bounding) the spectral range of H. Lanczos iteration gives a cheap
  estimate, but adds complexity. Alternative: use known bounds from the
  Hamiltonian structure (e.g., sum of absolute coefficient values).
- **Time-dependent Hamiltonians.** The basic Chebyshev expansion assumes
  time-independent H. Extending to H(t) requires splitting the evolution
  into small time slices with separate expansions -- still better than Trotter
  for smooth time dependence.
- **Mixed mode.** Some circuits mix Hamiltonian evolution with non-Hamiltonian
  operations (mid-circuit measurement, reset, classically controlled gates).
  The Chebyshev path handles the evolution segments; the existing gate path
  handles the rest.

## Beyond Chebyshev

Ideas on the horizon, not yet committed:

- **Multi-GPU statevector sharding** -- distribute state vectors across both
  5090s for Q=30+ (64 GB combined). NVLink-less, so PCIe 5.0 x16 bisection
  bandwidth (64 GB/s) is the constraint. Worth it only for Q>=26 where single-GPU
  memory is the bottleneck.
- **Density matrix simulation** -- for mixed states and noise-heavy circuits.
  Squares the memory requirement (2^n x 2^n), but reuses the same SpMV machinery
  that Chebyshev introduces.
- **Automatic differentiation through kernels** -- gradient computation for
  variational circuits (parameter-shift rule or adjoint method). Enables
  on-GPU VQE optimization loops without round-tripping through Python.
