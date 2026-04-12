# Quantum Simulation on a Gaming GPU

A primer for the curious reader.

---

## 1. What Is a Qubit?

A classical bit is 0 or 1. A qubit is both at once.

That's the popular version, and it's not wrong, but it skips the part that matters: a qubit is a pair of complex numbers called *amplitudes* that describe the probability of measuring 0 or 1. If the amplitudes are *a* and *b*, then:

- Probability of measuring 0 = |*a*|^2
- Probability of measuring 1 = |*b*|^2
- |*a*|^2 + |*b*|^2 = 1 (always)

Before you measure, the qubit exists as both possibilities simultaneously, weighted by these amplitudes. After you measure, it collapses to one outcome. The amplitudes are gone. You get a single bit.

This is not a metaphor. The math does not describe hidden information or uncertainty about which value the qubit "really" has. It describes a system that genuinely has no definite value until observed. That's the part of quantum mechanics that took physicists decades to accept.

## 2. More Qubits, Exponentially More Amplitudes

One qubit needs 2 amplitudes. Two qubits need 4. Three need 8.

In general, *n* qubits need 2^*n* amplitudes. This is the state vector -- a list of complex numbers, one for every possible combination of qubit values. A 10-qubit system has 1,024 amplitudes. A 20-qubit system has about a million. A 30-qubit system has a billion.

This exponential growth is both why quantum computers are interesting and why simulating them on classical hardware is hard. Every qubit you add doubles the memory required to represent the state.

## 3. Gates: How You Manipulate Qubits

Quantum computation works by applying *gates* to qubits. A gate is a mathematical operation -- specifically, a unitary matrix -- that transforms the amplitudes.

A single-qubit gate is a 2x2 matrix. It takes the two amplitudes of one qubit and produces two new amplitudes:

```
[new_a]   [g00  g01] [a]
[new_b] = [g10  g11] [b]
```

Common single-qubit gates include:

- **H** (Hadamard): puts a qubit into equal superposition
- **X** (NOT): flips 0 and 1 (like a classical NOT)
- **Rz(theta)**: rotates the phase by angle theta

A two-qubit gate is a 4x4 matrix that operates on four amplitudes (all combinations of two qubits being 0 or 1). The most important is **CNOT** (controlled-NOT): if the first qubit is 1, flip the second. This is how you create *entanglement* -- correlations between qubits that have no classical equivalent.

Gates must be *unitary*, meaning they preserve the total probability (the state vector stays normalized). This is the quantum version of "information is conserved."

## 4. A Quantum Circuit

A quantum circuit is a sequence of gates applied to a set of qubits. You start with all qubits in state |0>, apply gates in order, and then measure. Here's a simple circuit that creates a Bell pair -- two qubits that are maximally entangled:

```
q0: ─── H ─── CNOT ─── Measure
q1: ────────── CNOT ─── Measure
```

After the H gate, q0 is in superposition: equal chance of 0 and 1. The CNOT entangles q1 with q0. The result is a state where both qubits are always measured the same -- both 0 or both 1, with equal probability. There is no state where one is 0 and the other is 1.

## 5. What Is a Quantum Simulator?

A quantum simulator is a classical program that tracks all 2^*n* amplitudes and applies gate operations as matrix multiplications. It answers the question: "if I ran this circuit on a real quantum computer, what would the probabilities look like?"

This project -- qiskit-blackwell -- is a quantum simulator. It runs on a GPU, not a quantum computer. The qubits are not physical objects; they're entries in an array of complex numbers stored in video memory.

### Simulators vs. Real Quantum Computers

| | Simulator | Real Quantum Computer |
|---|---|---|
| **Qubits** | Numbers in RAM/VRAM | Physical objects (superconducting circuits, trapped ions, photons) |
| **Errors** | Only floating-point rounding | Decoherence, cross-talk, gate infidelity, measurement errors |
| **Scaling** | 2^n memory, limited to ~30-35 qubits | Hundreds to thousands of physical qubits (but noisy) |
| **Speed** | Fast for small circuits, exponential wall | Constant time per gate, independent of state size |
| **Repeatability** | Deterministic (same seed, same result) | Inherently probabilistic |
| **Access** | Your laptop or GPU | Cloud access, multi-million dollar hardware |

The key difference: a simulator is *exact* (up to floating-point precision) but hits a memory wall. A real quantum computer doesn't have the memory problem but introduces errors at every step.

Simulators are essential tools for:
- Developing and debugging quantum algorithms before running on real hardware
- Verifying the correctness of quantum hardware (does the real computer match the ideal?)
- Exploring quantum chemistry and physics when you need exact answers at small system sizes
- Education and research

## 6. The Exponential Wall

Every qubit doubles the memory. At 8 bytes per complex number (float32 real + float32 imaginary):

| Qubits | Amplitudes | Memory |
|--------|-----------|--------|
| 10 | 1,024 | 8 KB |
| 20 | 1,048,576 | 8 MB |
| 25 | 33,554,432 | 256 MB |
| 30 | 1,073,741,824 | 8 GB |
| 33 | 8,589,934,592 | 64 GB |
| 40 | ~1 trillion | 8 TB |
| 50 | ~1 quadrillion | 8 PB |

Our RTX 5090 has 32 GB of VRAM. That means we max out around 31-32 qubits for the state vector alone. At 50 qubits -- where quantum computers start doing things classical computers struggle with -- you'd need more memory than exists in most data centers.

This is the fundamental limit of classical simulation. It's not a software problem. It's not an optimization problem. It's the physics of exponential growth meeting the engineering of finite memory.

## 7. Why a GPU?

A GPU is not a faster CPU. It's a fundamentally different machine -- thousands of simple cores designed to do the same operation on many data points simultaneously.

Quantum simulation is a natural fit because:

1. **The core operation is embarrassingly parallel.** When you apply a single-qubit gate, every pair of amplitudes can be updated independently. A 20-qubit system has 512K pairs. A GPU can process all of them at once.

2. **It's memory-bandwidth bound.** The gate matrix is tiny (32 bytes for a 2x2 complex matrix). The state vector is large (8 MB at 20 qubits). The bottleneck is reading and writing the state vector, not computing the matrix multiply. GPUs have massive memory bandwidth -- the RTX 5090 delivers 1,792 GB/s.

3. **The access pattern is regular.** For a gate on qubit *t*, pairs of amplitudes are separated by a stride of 2^*t*. This is predictable and can be optimized for hardware cache behavior.

The RTX 5090 (NVIDIA Blackwell architecture, sm_120a) has specific characteristics we exploit:

- **96 MB L2 cache**: state vectors up to 22 qubits fit entirely in cache. Gates can read and write without touching main memory.
- **170 streaming multiprocessors**: enough parallel execution units to keep the memory bus saturated.
- **32 GB GDDR7**: enough for 31-32 qubits in a single state vector.

## 8. How This Simulator Works

When you tell our simulator to run a circuit, four things happen:

### Step 1: Transpile

Qiskit circuits can use any gate. Our GPU kernels support a fixed set of gates (H, X, Rx, Ry, Rz, CNOT, CZ, Toffoli, etc.). The transpiler decomposes arbitrary gates into this native set. For example, a SWAP gate becomes three CNOT gates.

### Step 2: Compile (Fusion)

Naive execution would launch one GPU kernel per gate. Each kernel launch has overhead -- 2-3 microseconds of CPU-GPU synchronization. For a 200-gate circuit, that's 400-600 microseconds of pure overhead.

Our 6-phase compiler eliminates this:

1. **Diagonal merging**: Rz(0.1) followed by Rz(0.2) on the same qubit becomes Rz(0.3). One kernel instead of two.
2. **Rotation merging**: 10 consecutive Rx gates on the same qubit accumulate their angles and become a single gate.
3. **Diagonal absorption**: A diagonal gate next to a dense gate on the same qubit gets folded into the dense gate's matrix.
4. **Matrix fusion**: Any consecutive single-qubit gates on the same qubit are multiplied into one 2x2 matrix.
5. **Batching**: Remaining single-qubit gates are dispatched in a single kernel call.
6. **Caching**: Compiled circuits are stored so repeated runs skip steps 1-5.

A 200-gate circuit might compile to 30-40 actual kernel launches.

### Step 3: Execute

Each compiled operation dispatches to a CUDA kernel optimized for the RTX 5090:

- **Single-qubit gate**: Each GPU thread reads two amplitudes (separated by stride 2^*t*), applies the 2x2 matrix, writes them back. 256 threads per block, grid sized to cover all pairs.
- **Two-qubit gate**: Each thread reads four amplitudes, applies a 4x4 matrix. Same approach, more complex indexing.
- **Diagonal gate**: Each thread reads one amplitude and multiplies by a scalar. Half the memory traffic of a general gate.

### Step 4: Sample

After all gates are applied, the state vector contains 2^*n* amplitudes. To simulate measurement, we compute probabilities (|amplitude|^2 for each entry) and draw random samples according to that distribution. 8,192 shots is standard -- enough to estimate probabilities to about 1% accuracy.

## 9. The Precision Problem

Here's a problem most quantum simulation tutorials don't mention: floating-point arithmetic introduces errors, and those errors accumulate.

A complex number in float32 has about 7 decimal digits of precision. Every time you multiply two complex numbers, you lose a fraction of a digit. Over a 500-gate circuit, those fractions add up.

Consider the operation `a*b + c*d` in float32:

```
a * b = 1.2345679...  (rounded from 1.23456789...)
c * d = 0.9876543...  (rounded from 0.98765432...)
sum   = 2.2222222...  (rounded again)
```

Each rounding is tiny -- maybe 1 part in 10 million. But after 500 gates, each touching amplitudes that other gates already rounded, the errors compound. The state vector drifts away from the mathematically exact answer.

This matters because quantum algorithms are sensitive to small errors. Quantum error correction -- one of the most important applications of quantum simulation -- requires tracking amplitudes to many digits of precision.

### How We Fix It: Compensated Arithmetic

The FMA (fused multiply-add) instruction on modern GPUs computes `a*b + c` in a single operation with a single rounding at the end, rather than rounding `a*b` and then rounding the addition separately. But we can go further.

The **TwoProduct** technique uses FMA to compute the *exact rounding error* of a multiplication:

```c
float p = a * b;              // rounded product
float err = fmaf(a, b, -p);   // exact error: true(a*b) - p
```

This works because `fmaf(a, b, -p)` computes `a*b - p` without any intermediate rounding -- and since `a*b` rounded to `p`, the difference is exactly the rounding error. We fold this error back into the final result, recovering almost all the lost precision.

For two-qubit gates, we go even further with **Kahan summation** -- maintaining a running correction variable that tracks accumulated addition errors across an 8-term dot product.

The result: our float32 kernels achieve fidelity comparable to float64 (double-precision) computation. We get the precision of 64-bit arithmetic at the speed of 32-bit arithmetic, because the extra FMA instructions are hidden behind the memory latency that dominates kernel execution time. Zero performance cost.

## 10. Chebyshev Expansion: A Different Approach

The gate-by-gate method works well for arbitrary quantum circuits. But many important problems in quantum chemistry and physics are naturally expressed as *Hamiltonian evolution*: given a Hamiltonian operator H (representing the energy of a physical system), compute the state after time t:

```
|psi(t)> = exp(-iHt) |psi(0)>
```

The traditional approach (Trotterization) decomposes this into a product of simple gates -- essentially approximating the continuous evolution with discrete steps. More steps means better accuracy but more gates, more kernel launches, and more accumulated floating-point error.

Chebyshev expansion takes a fundamentally different approach. Instead of decomposing into gates, it evaluates exp(-iHt) directly as a polynomial in H:

```
exp(-iHt)|psi> = sum_{k=0}^{K} c_k * T_k(H_norm)|psi>
```

where T_k are Chebyshev polynomials (a well-known family from approximation theory) and c_k are analytically known coefficients (Bessel functions).

The recurrence relation makes this GPU-friendly:

```
T_0|psi> = |psi>
T_1|psi> = H_norm|psi>
T_{k+1}|psi> = 2 * H_norm * T_k|psi> - T_{k-1}|psi>
```

Each step is one sparse matrix-vector multiply (SpMV) plus a vector update. The polynomial degree K scales as roughly `K ~ ||H||*t + log(1/epsilon)` -- meaning for a given time and precision target, you know exactly how many iterations you need. And the convergence is *exponential* in K, not polynomial like Trotter.

In our tests, a 12-qubit Heisenberg chain that scipy solves in 4 seconds via dense matrix exponentiation runs in 0.4 milliseconds on the Chebyshev engine -- **8,000x faster** -- with fidelity at 1.0000000004.

## 11. The Hardware-Software Dance

The performance gains in this project don't come from algorithmic cleverness alone. They come from understanding the specific hardware and writing code that matches its strengths.

### Memory Hierarchy Exploitation

The RTX 5090's 96 MB L2 cache is unusually large. A 20-qubit state vector is 8 MB -- it fits in L2 with room to spare. A 22-qubit state vector is 32 MB -- still fits. When the state vector lives in L2, gate application becomes limited by L2 bandwidth (~12 TB/s effective) rather than GDDR7 bandwidth (1.8 TB/s). That's a 6-7x difference.

We design kernel launch patterns to keep the state vector resident in L2 across multiple gate applications. The circuit fusion engine helps here -- fewer kernel launches means fewer opportunities for the state vector to be evicted from cache.

### Vectorized Memory Access

GPU memory controllers operate on 128-bit (16-byte) transactions. A complex64 number is 8 bytes. Two of them -- which is exactly what a single-qubit gate needs to read -- pack into one 128-bit load.

For gates on qubit 0 (the least significant bit), paired amplitudes are adjacent in memory. We use float4 (128-bit) loads to grab both at once, achieving maximum memory bandwidth utilization.

### Warp-Level Cooperation

A GPU warp is a group of 32 threads that execute in lockstep. When a gate targets a low-numbered qubit (0-4), the amplitude pairs needed by different threads are within the same warp. Instead of going through memory, threads can exchange data via **warp shuffle** instructions -- register-to-register transfers at essentially zero cost.

Our register-tiled fusion kernel exploits this: for sequences of gates all targeting qubits 0-4, the data never leaves the register file. No memory reads, no memory writes, just shuffles between thread registers.

### Diagonal Gate Optimization

Gates like Rz, S, T, and Z are diagonal -- they scale each amplitude independently without mixing pairs. This means:

- No pair coupling: each thread processes one amplitude instead of two
- Half the memory traffic: for phase gates (where one diagonal entry is 1), we skip amplitudes that don't change
- Trivially parallel: no indexing complexity

## 12. Testing Philosophy

Quantum simulation has a trust problem. The outputs are complex probability distributions over exponentially many states. How do you know they're correct?

Our approach is defense in depth:

1. **Analytical tests**: For circuits with known analytical solutions (GHZ states, Bell pairs, Rz chains), we verify exact amplitudes.

2. **Cross-reference tests**: For general circuits, we compare against Qiskit Aer's CPU simulator running in float64 (double precision). This is a completely independent implementation -- different code, different algorithms, different hardware. Agreement to 7+ decimal places is strong evidence.

3. **Property tests**: Some bugs don't affect individual circuits but violate physical properties. We test norm preservation (unitarity), gate decomposition equivalences (CNOT = H-CZ-H), measurement statistics (chi-squared against known distributions), and multi-controlled gate edge cases.

4. **Adversarial tests**: Near-zero amplitudes, near-uniform superpositions, maximum-stride qubit targets, repeated-gate stress tests. These catch numerical instability and indexing bugs that normal circuits don't trigger.

5. **Scaling tests**: Running at 16 and 20 qubits verifies that kernels work beyond the L2-resident regime. Some bugs only appear when the state vector must be streamed from main memory.

Total test count: 444 tests across 4 test suites, all passing.

## 13. Where Simulation Fits in the Quantum Ecosystem

Quantum simulation is not a replacement for quantum computing. It's a development tool.

Today's quantum computers have tens to hundreds of noisy qubits. They can run circuits that are difficult to simulate classically, but the noise limits their practical utility. The field is in a phase called NISQ (Noisy Intermediate-Scale Quantum) where the primary challenge is making hardware reliable enough to run useful algorithms.

Simulators serve this effort by:

- **Algorithm development**: Design and test quantum algorithms at small scale before committing expensive QPU time.
- **Error analysis**: Compare ideal (simulated) results against noisy (hardware) results to characterize and correct errors.
- **Education**: Learn quantum computing without needing access to hardware.
- **Benchmarking**: Quantum Volume, a standard benchmark for quantum computers, is defined relative to the ideal (simulated) circuit outcome.

As quantum hardware improves and qubit counts grow past 50-100, simulators will become less able to verify results directly. But they remain essential for small-scale validation, algorithm prototyping, and understanding the theory.

## 14. Further Reading

This primer covered the basics -- qubits, gates, circuits, simulation, and the hardware and numerical techniques that make this project fast. For deeper study:

- **IBM Qiskit Tutorials**: Comprehensive, interactive lessons from the basics through advanced algorithms.
  https://www.ibm.com/quantum/qiskit#tutorials

- **Qiskit Textbook**: A full open-source textbook covering quantum computing from linear algebra through quantum error correction.
  https://github.com/Qiskit/textbook

- **Nielsen & Chuang**: *Quantum Computation and Quantum Information* -- the standard graduate reference. Dense but definitive.

- **Preskill's Lecture Notes**: John Preskill's Caltech lecture notes on quantum computation. Freely available and excellent for building mathematical intuition.
  http://theory.caltech.edu/~preskill/ph219/

- **This repository**: The source code, benchmarks, and roadmap for the Blackwell-optimized quantum simulation kernels.
  https://github.com/DarrellThomas/qiskit-blackwell

---

*This primer is part of the qiskit-blackwell project. MIT License. Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC.*
