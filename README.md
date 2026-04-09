# qiskit-blackwell

Custom CUDA quantum simulation kernels for the NVIDIA RTX 5090 (Blackwell, sm_120a), integrated as a drop-in Qiskit backend.

Blackwell beats Aer GPU (cuQuantum) on every circuit type tested -- 2-5x faster, with full statevector fidelity verified against float64 reference.

## Correctness First

Performance claims without rigorous testing are noise. This project ships with 112 kernel-level tests, 12 stress tests, and a full A/B accuracy suite that compares every result against Aer CPU (float64) as ground truth.

### Statevector Fidelity

Every circuit is tested by computing the full statevector on three backends -- Aer CPU (float64 reference), Aer GPU (cuQuantum, float32), and Blackwell (float32) -- and comparing fidelity. All 14 test circuits pass with fidelity > 0.9999999:

| Circuit | Qubits | Aer GPU Fidelity | Blackwell Fidelity | Status |
|---------|--------|-----------------|-------------------|--------|
| GHZ | 4 | 1.0000000000 | 1.0000001344 | PASS |
| QFT | 4 | 1.0000000000 | 1.0000000000 | PASS |
| Clifford | 4 | 1.0000000000 | 1.0000000000 | PASS |
| EfficientSU2 | 4 | 1.0000000000 | 1.0000000317 | PASS |
| DeepRz | 4 | 1.0000000000 | 1.0000000642 | PASS |
| Toffoli | 4 | 1.0000000000 | 1.0000000000 | PASS |
| Quantum Volume | 4 | 1.0000000000 | 1.0000000619 | PASS |
| GHZ | 8 | 1.0000000000 | 1.0000001344 | PASS |
| QFT | 8 | 1.0000000000 | 1.0000000000 | PASS |
| Clifford | 8 | 1.0000000000 | 1.0000000000 | PASS |
| EfficientSU2 | 8 | 1.0000000000 | 1.0000001196 | PASS |
| DeepRz | 8 | 1.0000000000 | 1.0000001097 | PASS |
| Toffoli | 8 | 1.0000000000 | 1.0000000000 | PASS |
| Quantum Volume | 8 | 1.0000000000 | 1.0000001106 | PASS |

### Stress Tests

Beyond fidelity, we test properties that break subtly when kernels have edge-case bugs:

| Test | What It Catches |
|------|----------------|
| **Identity roundtrips** | Apply H-H, X-X, CNOT-CNOT pairs and verify state returns to \|0>. Catches indexing bugs that silently permute amplitudes. |
| **All-qubit sweep** | Apply a gate to every qubit index (0 through n-1) on a 10-qubit system. Catches stride miscalculations at boundary qubits. |
| **Qubit pair combinatorics** | Test all 28 two-qubit pairs on 8 qubits. Catches insert-zeros indexing errors that only appear for specific (q0, q1) combinations. |
| **Norm preservation** | Apply 200 random gates and verify \|\|state\|\| = 1.0 after every gate. Catches numerical drift, missing renormalization, and off-by-one amplitude updates. |
| **Gate equivalences** | Verify Rx(pi)=iX, Ry(pi)=iY, Rz(pi)=iZ, H=Rx(pi)Rz(pi)/i, CNOT=H-CZ-H. Catches wrong gate matrix construction. |
| **Phase accumulation** | Chain 100 Rz gates and verify the accumulated phase matches the analytic result. Catches phase sign errors and floating-point drift. |
| **Entanglement structures** | Build GHZ, W, and Bell states, verify exact amplitudes. Catches two-qubit gate errors that don't show up in single-qubit tests. |
| **Measurement statistics** | Measure 10,000 shots from known states, chi-squared test against expected distribution. Catches sampling bias and collapse errors. |
| **Multi-controlled gates** | Toffoli and CSWAP with 1-4 control qubits, all target positions. Catches control-bit masking errors. |
| **Adversarial inputs** | Near-zero amplitudes, near-uniform superpositions, maximally entangled states. Catches numerical instability at distribution edges. |
| **Large qubit scaling** | Run at Q=10, 14, 18, 22 to verify kernels work beyond L2-resident sizes. Catches assumptions that break when state vectors exceed cache. |
| **Repeated gate stress** | Apply the same gate 1,000 times, verify final state matches analytic prediction. Catches accumulating rounding errors. |

Run the full suite yourself:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_ab_test.sh        # A/B accuracy + performance
python tests/stress_test.py                         # 12 stress tests
```

## Performance

With correctness established, here's the speed. Benchmarked against Qiskit Aer with cuQuantum GPU acceleration. 8,192 shots per circuit, median of 5 runs after 3 warmup runs.

| Circuit | Qubits | Gates | Aer GPU (ms) | Blackwell (ms) | Speedup |
|---------|--------|-------|-------------|---------------|---------|
| GHZ | 4 | 4 | 16.4 | 3.9 | **4.2x** |
| QFT | 4 | 12 | 16.9 | 3.9 | **4.3x** |
| Clifford | 4 | 10 | 17.7 | 5.1 | **3.5x** |
| EfficientSU2 | 4 | 30 | 16.1 | 4.2 | **3.8x** |
| DeepRz | 4 | 70 | 16.7 | 4.6 | **3.7x** |
| Toffoli | 4 | 6 | 17.3 | 3.9 | **4.4x** |
| Quantum Volume | 4 | 8 | 18.8 | 5.2 | **3.6x** |
| GHZ | 8 | 8 | 16.1 | 4.0 | **4.0x** |
| QFT | 8 | 40 | 19.1 | 4.6 | **4.2x** |
| Clifford | 8 | 10 | 19.4 | 4.1 | **4.7x** |
| EfficientSU2 | 8 | 62 | 20.0 | 4.4 | **4.6x** |
| DeepRz | 8 | 150 | 20.8 | 5.0 | **4.1x** |
| Toffoli | 8 | 14 | 18.6 | 4.1 | **4.5x** |
| Quantum Volume | 8 | 32 | 18.9 | 9.0 | **2.1x** |

### Per-Kernel Benchmarks

Raw kernel performance at Q=20 (1M amplitudes, 8 MB state vector), compared against equivalent PyTorch operations:

| Kernel | Time (us) | vs PyTorch | Notes |
|--------|-----------|------------|-------|
| Single-qubit gate | 4.1 | 26.9x | L2-resident, 100% occupancy |
| Two-qubit gate | 6.2 | 37.2x | Warp-cooperative indexing |
| Diagonal gate | 4.1 | 34.0x | No pair coupling, element-wise |
| Diagonal (phase-only) | 2.1 | 60.0x | Half memory traffic |
| Multi-controlled gate | 5.0 | 27.6x | Insert-zeros pattern |
| Measurement + collapse | 24.7 | 5.0x | Two-pass streaming reduction |
| Sampling (1000 shots) | 20.5 | 17.8x | CUB prefix-sum + binary search |
| 10-gate fused circuit | 10.3 | 4.0x vs sequential | Register-tiled, data stays in L2 |

## What's Inside

### 12 CUDA Kernels (`bwk/csrc/`)

Purpose-built for sm_120a. No cuBLAS, no cuStateVec, no CUTLASS.

| Kernel | Operation |
|--------|-----------|
| `apply_gate_sm120` | Single-qubit gate (H, X, Rx, Ry, etc.) |
| `apply_gate2q_sm120` | Two-qubit gate (CNOT, CZ, SWAP) |
| `apply_diagonal_sm120` | Diagonal gates (Rz, S, T, Z, Phase) |
| `apply_mcgate_sm120` | Multi-controlled single-qubit gate (Toffoli) |
| `apply_mc2qgate_sm120` | Multi-controlled two-qubit gate (CSWAP) |
| `measure_sm120` | Single-qubit measurement + state collapse |
| `sample_sm120` | Multi-shot sampling from statevector |
| `expectation_sm120` | Pauli Z expectation values |
| `batch_sm120` | Batched multi-circuit simulation |
| `noise_sm120` | Noise channels (depolarizing, amplitude damping, dephasing) |
| `qv4_sim_sm120` | Fused QV-4 circuit simulator |
| `qv8_sim_sm120` | Fused QV-8 circuit simulator |

### Circuit Fusion Engine

The Qiskit integration includes a circuit compilation pass that reduces kernel launches:

1. **Diagonal merging** -- consecutive Rz/S/T/Z gates on the same qubit are algebraically merged into a single diagonal gate
2. **Diagonal absorption** -- a diagonal gate adjacent to a dense gate on the same qubit is folded into the dense gate's matrix
3. **1Q gate batching** -- runs of single-qubit gates are dispatched as one fused kernel call instead of N individual launches
4. **Transpilation caching** -- compiled circuits are cached so repeated executions skip transpilation and gate matrix construction

A 76-gate QV-4 circuit compiles to 49 fused operations. A 296-gate QV-8 circuit compiles to 193. Each eliminated kernel launch saves 2-3 microseconds of CPU-GPU dispatch overhead.

## Requirements

- NVIDIA RTX 5090 (sm_120a)
- CUDA 13.x (set `CUDA_HOME` if not at `/usr/local/cuda`)
- Python 3.10+
- PyTorch 2.x with CUDA support
- Qiskit 2.x
- qiskit-aer (optional, for A/B testing against Aer GPU)

## Setup

```bash
git clone https://github.com/DarrellThomas/qiskit-blackwell.git
cd qiskit-blackwell

pip install qiskit torch

cd bwk
CUDA_HOME=/usr/local/cuda-13 python setup.py build_ext --inplace
cd ..
```

The simulator finds kernels from `bwk/python/` via relative paths. No environment variables needed.

## Quick Start

```python
from qiskit import QuantumCircuit
from blackwell_backend import BlackwellSimulator

# Create a circuit
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)  # 4-qubit GHZ state

# Run on Blackwell
sim = BlackwellSimulator()
result = sim.run(qc, shots=8192)
print(result.get_counts())
# {'0x0': 4096, '0xf': 4096}  (approximately)
```

### Statevector Access

```python
sv, elapsed_s = sim.run_statevector(qc)
print(f"Statevector computed in {elapsed_s*1e6:.0f} us")
print(f"|0000> amplitude: {sv[0]:.4f}")
print(f"|1111> amplitude: {sv[15]:.4f}")
```

### Benchmarking

```python
result, stats = sim.run_and_time(qc, shots=8192, warmup=3, repeats=10)
print(f"Median: {stats['median_s']*1000:.2f} ms")
print(f"Min:    {stats['min_s']*1000:.2f} ms")
```

### Quantum Volume

```python
from qiskit.circuit.library import quantum_volume

qv = quantum_volume(8, depth=8, seed=42)
result = sim.run(qv, shots=8192)
counts = result.get_counts()
print(f"Unique outcomes: {len(counts)}, total shots: {sum(counts.values())}")
```

## How It Works

When you call `sim.run(circuit)`:

1. **Transpile** to a native gate set (H, X, Rx, Ry, Rz, CX, CZ, CCX, etc.)
2. **Compile** through the fusion pass (merge diagonals, absorb into dense gates, batch 1Q gates)
3. **Execute** compiled operations on the GPU
4. **Sample** from the final statevector

### Why It's Fast

These kernels are tuned for the RTX 5090's specific hardware characteristics:

- **L2 persistence** -- state vectors up to 32 MB (Q<=22) stay fully resident in the 96 MB L2 cache across gate applications
- **100% occupancy** -- single-qubit gate kernels use only 26 registers and zero shared memory
- **Vectorized loads** -- float4 (128-bit) coalesced loads for stride-1 qubit targets
- **Warp-cooperative indexing** -- two-qubit gates use an insert-zeros pattern that distributes index computation across the warp
- **Circuit fusion** -- gate batching and diagonal merging cut kernel launch count by 30-50%

NVIDIA's cuStateVec targets datacenter GPUs with features like multi-GPU distribution. On a consumer 5090, it doesn't exploit L2 residency, doesn't fuse gates, and pays per-gate launch overhead that dominates at small qubit counts.

## Hardware

**Compatible with:** RTX 5090, 5080, 5070 Ti, 5070 (all sm_120a consumer Blackwell).

Developed and tested on:
- **GPU:** NVIDIA GeForce RTX 5090 (GB202, sm_120a, 170 SMs, 32 GB GDDR7)
- **Bandwidth:** 1792 GB/s (GDDR7)
- **L2 Cache:** 96 MB
- **Shared Memory:** 99 KB per block
- **CUDA:** 13.2

These kernels target sm_120a (consumer Blackwell) and should run on any GPU in the family -- RTX 5090, 5080, 5070 Ti, and 5070. Performance will scale with SM count and memory bandwidth, but correctness is architecture-level. They will not run on datacenter Blackwell (B200/GB200 use sm_100 with `tcgen05`, a different ISA) or previous GPU architectures.

## Project Structure

```
qiskit-blackwell/
├── blackwell_backend/                ← Qiskit integration
│   ├── simulator.py                  ← Circuit fusion + dispatch
│   └── hybrid.py                     ← Aer/Blackwell hybrid router
├── bwk/                              ← Kernel package
│   ├── setup.py                      ← Build from source
│   ├── csrc/                         ← 12 CUDA kernel source files
│   └── python/blackwell_kernels/     ← Python wrappers
├── tests/
│   ├── ab_test.py                    ← A/B accuracy + performance vs Aer GPU
│   ├── stress_test.py                ← 12 stress tests (norm, equivalence, adversarial)
│   └── kernel_ab.py                  ← Compare two kernel builds
├── run_ab_test.sh                    ← One-command benchmark
└── README.md
```

## License

MIT License. Copyright (c) 2026 Darrell Thomas.
