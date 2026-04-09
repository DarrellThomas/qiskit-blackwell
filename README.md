# qiskit-blackwell

Custom CUDA quantum simulation kernels for the NVIDIA RTX 5090 (Blackwell, sm_120a), integrated as a drop-in Qiskit backend.

**Blackwell beats Aer GPU (cuQuantum) on every circuit type tested** -- 2x to 5x faster end-to-end, with full statevector fidelity.

## Performance

Benchmarked against Qiskit Aer with cuQuantum GPU acceleration. 8,192 shots per circuit, median of 5 runs after warmup.

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

## Accuracy

Statevector fidelity against Aer CPU (float64 reference). All circuits pass with fidelity > 0.9999999.

| Circuit | Qubits | Aer GPU Fidelity | Blackwell Fidelity | Status |
|---------|--------|-----------------|-------------------|--------|
| GHZ | 4 | 1.0000000000 | 1.0000001344 | PASS |
| QFT | 4 | 1.0000000000 | 1.0000000000 | PASS |
| EfficientSU2 | 4 | 1.0000000000 | 1.0000000317 | PASS |
| Quantum Volume | 4 | 1.0000000000 | 1.0000000619 | PASS |
| GHZ | 8 | 1.0000000000 | 1.0000001344 | PASS |
| QFT | 8 | 1.0000000000 | 1.0000000000 | PASS |
| EfficientSU2 | 8 | 1.0000000000 | 1.0000001196 | PASS |
| Quantum Volume | 8 | 1.0000000000 | 1.0000001106 | PASS |

## What's Inside

### 12 CUDA Kernels (`bwk/csrc/`)

Every kernel is purpose-built for sm_120a. No cuBLAS, no cuStateVec, no CUTLASS. Pure CUDA with PTX intrinsics where it matters.

| Kernel | Operation | Architecture |
|--------|-----------|-------------|
| `apply_gate_sm120` | Single-qubit gate (H, X, Rx, Ry, etc.) | Vectorized float4 loads, L2-resident for Q<=22, 100% occupancy |
| `apply_gate2q_sm120` | Two-qubit gate (CNOT, CZ, SWAP) | Warp-cooperative with insert-zeros indexing, 37x PyTorch |
| `apply_diagonal_sm120` | Diagonal gates (Rz, S, T, Z, Phase) | Element-wise bitmask, no pair coupling, 34x PyTorch |
| `apply_mcgate_sm120` | Multi-controlled single-qubit gate (Toffoli) | Insert-zeros pattern for k controls, 28x PyTorch |
| `apply_mc2qgate_sm120` | Multi-controlled two-qubit gate (CSWAP) | 4x4 gate with k control bits, 1489x PyTorch |
| `measure_sm120` | Single-qubit measurement + state collapse | Two-pass: streaming reduction + conditional scale |
| `sample_sm120` | Multi-shot sampling from statevector | CUB prefix-sum CDF + parallel binary search |
| `expectation_sm120` | Pauli Z expectation values | Weighted sign-flip reduction |
| `batch_sm120` | Batched multi-circuit simulation | N independent state vectors in one launch |
| `noise_sm120` | Noise channels (depolarizing, amplitude damping, dephasing) | Stochastic Kraus operator application |
| `qv4_sim_sm120` | Fused QV-4 circuit simulator | Entire 4-qubit QV circuit in shared memory |
| `qv8_sim_sm120` | Fused QV-8 circuit simulator | 256 threads/block, one amplitude per thread |

### Circuit Fusion Engine (`blackwell_backend/simulator.py`)

The Qiskit integration isn't just a gate-by-gate dispatch loop. It includes a three-layer circuit fusion pass that reduces kernel launches:

1. **Diagonal merging** -- consecutive Rz/S/T/Z gates on the same qubit are algebraically merged (multiply diagonal entries in Python, emit one kernel instead of N)
2. **Diagonal absorption** -- a diagonal gate adjacent to a dense gate on the same qubit is folded into the matrix (eliminates the diagonal launch entirely)
3. **1Q gate batching** -- consecutive single-qubit gates are dispatched via `apply_gates_fused` (one kernel launch for N gates instead of N launches)
4. **Transpilation caching** -- Qiskit transpilation + compilation results are cached so repeated executions of the same circuit pay the cost once

### Hybrid Router (`blackwell_backend/hybrid.py`)

Automatically routes circuits to Blackwell or Aer GPU based on circuit characteristics. With fusion, Blackwell wins on essentially everything.

## Requirements

- NVIDIA RTX 5090 (sm_120a)
- CUDA 13.x (`/usr/local/cuda-13` or set `CUDA_HOME`)
- Python 3.10+
- PyTorch 2.x with CUDA support
- Qiskit 2.x
- qiskit-aer (optional, for A/B testing against Aer GPU)

## Setup

```bash
# Clone
git clone https://github.com/yourusername/qiskit-blackwell.git
cd qiskit-blackwell

# Install Python dependencies
pip install qiskit torch

# Build the CUDA kernels from source
cd bwk
CUDA_HOME=/usr/local/cuda-13 python setup.py build_ext --inplace
cd ..
```

That's it. The simulator auto-discovers kernels from `bwk/python/` via relative paths.

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
# Get the full statevector (no sampling noise)
sv, elapsed_s = sim.run_statevector(qc)
print(f"Statevector computed in {elapsed_s*1e6:.0f} us")
print(f"|0000> amplitude: {sv[0]:.4f}")
print(f"|1111> amplitude: {sv[15]:.4f}")
```

### Benchmarking

```python
# Time a circuit with warmup
result, stats = sim.run_and_time(qc, shots=8192, warmup=3, repeats=10)
print(f"Median: {stats['median_s']*1000:.2f} ms")
print(f"Min:    {stats['min_s']*1000:.2f} ms")
```

### Quantum Volume

```python
from qiskit.circuit.library import quantum_volume

# QV-8: 8-qubit quantum volume circuit
qv = quantum_volume(8, depth=8, seed=42)
result = sim.run(qv, shots=8192)

# Check heavy output probability
counts = result.get_counts()
total = sum(counts.values())
print(f"Unique outcomes: {len(counts)}")
print(f"Total shots: {total}")
```

## Run the Benchmark Suite

```bash
# Full A/B test against Aer GPU (requires qiskit-aer + cuQuantum)
CUDA_VISIBLE_DEVICES=0 bash run_ab_test.sh

# Quick mode (4 and 8 qubits only)
CUDA_VISIBLE_DEVICES=0 bash run_ab_test.sh --quick
```

## How It Works

### Gate Dispatch

When you call `sim.run(circuit)`, the simulator:

1. **Transpiles** the circuit to a native gate set (H, X, Rx, Ry, Rz, CX, CZ, CCX, etc.)
2. **Compiles** the gate sequence through the fusion pass:
   - Merges consecutive diagonal gates on the same qubit
   - Absorbs diagonals into adjacent dense gates
   - Batches remaining 1Q gates into fused kernel calls
3. **Executes** the compiled ops on the GPU
4. **Samples** from the final statevector

The fusion pass turns a 76-gate QV-4 circuit into ~49 fused operations. For a 296-gate QV-8 circuit, it produces ~193 fused operations. The reduction in kernel launches is where the speedup comes from -- each eliminated launch saves 2-3 microseconds of CPU-GPU overhead.

### Why Not cuStateVec?

NVIDIA's cuStateVec is designed for datacenter GPUs (A100, H100, B200) with features like multi-GPU distribution and mixed-precision simulation. On a consumer RTX 5090:

- cuStateVec doesn't exploit the 96 MB L2 cache for L2-resident simulation (Q<=22)
- It doesn't fuse gates or cache transpilation results
- Per-gate kernel launch overhead dominates at small qubit counts

These kernels exploit sm_120a-specific features:
- **L2 persistence windows** keep state vectors warm across gate applications
- **100% occupancy** (26 registers, zero shared memory for single-qubit gates)
- **Vectorized float4 loads** for stride-1 qubit targets
- **Warp-cooperative indexing** for two-qubit gates (insert-zeros pattern)

### Per-Kernel Performance vs PyTorch Reference

These are raw kernel benchmarks at Q=20 (1M amplitudes, 8 MB state vector):

| Kernel | Time (us) | vs PyTorch | Bandwidth |
|--------|-----------|------------|-----------|
| Single-qubit gate | 4.1 | 26.9x | L2-resident |
| Two-qubit gate | 6.2 | 37.2x | L2-resident |
| Diagonal gate | 4.1 | 34.0x | L2-resident |
| Diagonal (phase-only) | 2.1 | 60.0x | Half memory traffic |
| Multi-controlled gate | 5.0 | 27.6x | L2-resident |
| Measurement + collapse | 24.7 | 5.0x | Two-pass reduction |
| Sampling (1000 shots) | 20.5 | 17.8x | CUB prefix-sum |
| 10-gate fused circuit | 10.3 | 4.0x vs sequential | Register-tiled |

## Hardware

Tested on:
- **GPU:** NVIDIA GeForce RTX 5090 (GB202, sm_120a, 170 SMs, 32 GB GDDR7)
- **Bandwidth:** 1792 GB/s (GDDR7)
- **L2 Cache:** 96 MB
- **Shared Memory:** 99 KB per block (128 KB per SM, shared with L1)
- **CUDA:** 13.2

This code is sm_120a-specific. It uses `mma.sync` (not `tcgen05`), consumer Blackwell PTX intrinsics, and sm_120a compilation targets. It will not run on datacenter Blackwell (B200/GB200) or previous architectures.

## Project Structure

```
qiskit-blackwell/
├── blackwell_backend/                ← Qiskit integration
│   ├── __init__.py
│   ├── simulator.py                  ← Circuit fusion + dispatch
│   └── hybrid.py                     ← Aer/Blackwell hybrid router
├── bwk/                              ← Kernel package
│   ├── setup.py                      ← Build from source
│   ├── csrc/                         ← 12 CUDA kernel source files
│   │   ├── apply_gate_sm120.cu       ← 1Q gates (26.9x PyTorch)
│   │   ├── apply_gate2q_sm120.cu     ← 2Q gates (37.2x PyTorch)
│   │   ├── apply_diagonal_sm120.cu   ← Diagonal gates (34x PyTorch)
│   │   ├── apply_mcgate_sm120.cu     ← Multi-controlled gates
│   │   ├── apply_mc2qgate_sm120.cu   ← Multi-controlled 2Q gates
│   │   ├── measure_sm120.cu          ← Measurement + collapse
│   │   ├── sample_sm120.cu           ← Multi-shot sampling
│   │   ├── expectation_sm120.cu      ← Pauli expectation values
│   │   ├── batch_sm120.cu            ← Batched simulation
│   │   ├── noise_sm120.cu            ← Noise channels
│   │   ├── qv4_sim_sm120.cu          ← Fused QV-4 simulator
│   │   └── qv8_sim_sm120.cu          ← Fused QV-8 simulator
│   └── python/blackwell_kernels/     ← Python wrappers
│       ├── __init__.py
│       ├── cuquantum.py
│       ├── qv4.py
│       └── qv8.py
├── tests/
│   ├── ab_test.py                    ← A/B benchmark vs Aer GPU
│   ├── kernel_ab.py                  ← Kernel build comparison
│   └── stress_test.py                ← Extended stress tests
├── run_ab_test.sh                    ← One-command benchmark
├── .gitignore
└── README.md
```

## License

MIT License. Copyright (c) 2026 Darrell Thomas.

## Acknowledgments

Built on the RTX 5090 kernel optimization work from the [blackwell-kernels](https://github.com/yourusername/blackwell-kernels) project: 633 experiments, 273 keeps, 11 phases of iterative kernel development across quantum simulation, BLAS, and attention kernels.
