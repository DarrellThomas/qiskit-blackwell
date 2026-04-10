// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
//
// qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
// Licensed under the MIT License. See LICENSE file in the project root.
// https://github.com/DarrellThomas/qiskit-blackwell
//
// Quantum state vector: single-qubit gate application kernel for sm_120a.
//
// Applies a 2x2 complex unitary gate to target_qubit of a state vector.
// For each index i where bit target_qubit is 0:
//   j = i | (1 << target_qubit)
//   new[i] = gate[0][0]*state[i] + gate[0][1]*state[j]
//   new[j] = gate[1][0]*state[i] + gate[1][1]*state[j]
//
// This kernel is MEMORY-BANDWIDTH-BOUND. No tensor cores.
// Gate matrix (32 bytes) fits in registers. Optimization is purely
// about memory access patterns, coalescing, and cache exploitation.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// Complex multiply-add: result += a * b (all float2 = complex64)
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// Compensated complex multiply-accumulate: g0*a0 + g1*a1.
// Uses TwoProduct (FMA) to capture rounding errors of each multiply,
// then corrects the pairwise sum. ~2 ULP accuracy vs ~4 ULP for cmul+cadd.
// Zero extra registers vs FMA chain; high ILP — products and errors
// are independent, pairwise sums are partially independent.
__device__ __forceinline__ float2 cmac2(float2 g0, float2 a0, float2 g1, float2 a1) {
    // Real: g0.x*a0.x - g0.y*a0.y + g1.x*a1.x - g1.y*a1.y
    float p0 = g0.x * a0.x;
    float p1 = -g0.y * a0.y;
    float p2 = g1.x * a1.x;
    float p3 = -g1.y * a1.y;
    float re = (p0 + p1) + (p2 + p3);
    // TwoProduct error correction (all 4 FMAs are independent → ILP)
    re += fmaf(g0.x, a0.x, -p0) + fmaf(-g0.y, a0.y, -p1)
        + fmaf(g1.x, a1.x, -p2) + fmaf(-g1.y, a1.y, -p3);

    // Imag: g0.x*a0.y + g0.y*a0.x + g1.x*a1.y + g1.y*a1.x
    float q0 = g0.x * a0.y;
    float q1 = g0.y * a0.x;
    float q2 = g1.x * a1.y;
    float q3 = g1.y * a1.x;
    float im = (q0 + q1) + (q2 + q3);
    im += fmaf(g0.x, a0.y, -q0) + fmaf(g0.y, a0.x, -q1)
        + fmaf(g1.x, a1.y, -q2) + fmaf(g1.y, a1.x, -q3);

    return make_float2(re, im);
}

// v3: Specialized path for target_qubit==0 (stride=1, adjacent pairs).
// float4 load gets both amplitudes in one 16-byte transaction.
// Gate loaded from device memory via pointer (avoids CPU copy overhead).
__global__ void __launch_bounds__(256)
apply_gate_stride1(
    float4* __restrict__ state4,
    const float2* __restrict__ gate,
    const int num_pairs
) {
    // Load gate into registers (all warps read same 32 bytes → broadcast from L1)
    float2 g00 = gate[0], g01 = gate[1], g10 = gate[2], g11 = gate[3];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    float4 v = state4[tid];
    float2 a0 = make_float2(v.x, v.y);
    float2 a1 = make_float2(v.z, v.w);

    float2 new0 = cmac2(g00, a0, g01, a1);
    float2 new1 = cmac2(g10, a0, g11, a1);

    state4[tid] = make_float4(new0.x, new0.y, new1.x, new1.y);
}

// v3: General path — int32 indices, gate from device pointer.
__global__ void __launch_bounds__(256)
apply_gate_sm120(
    float2* __restrict__ state,
    const float2* __restrict__ gate,
    const int target_qubit,
    const int num_pairs
) {
    float2 g00 = gate[0], g01 = gate[1], g10 = gate[2], g11 = gate[3];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    int low_mask = (1 << target_qubit) - 1;
    int low_bits = tid & low_mask;
    int high_bits = (tid >> target_qubit) << (target_qubit + 1);
    int i0 = high_bits | low_bits;
    int i1 = i0 | (1 << target_qubit);

    float2 a0 = state[i0];
    float2 a1 = state[i1];

    float2 new0 = cmac2(g00, a0, g01, a1);
    float2 new1 = cmac2(g10, a0, g11, a1);

    state[i0] = new0;
    state[i1] = new1;
}



// Phase 2: Tile-based fused multi-gate kernel.
// Partitions state vector into tiles of size 2^(max_target+1). Within each tile,
// all gate pairs are self-contained. Block-local __syncthreads() between gates.
__global__ void __launch_bounds__(256)
apply_gates_tiled_kernel(
    float2* __restrict__ state,
    const float2* __restrict__ gates,   // N_GATES * 4 float2 (flattened 2x2 matrices)
    const int* __restrict__ targets,    // N_GATES target qubits
    const int n_gates,
    const int tile_bits,                // max_target + 1
    const int n_amplitudes
) {
    int tile_size = 1 << tile_bits;
    int n_tiles = n_amplitudes >> tile_bits;
    int pairs_per_tile = tile_size >> 1;

    for (int tile_idx = blockIdx.x; tile_idx < n_tiles; tile_idx += gridDim.x) {
        int tile_offset = tile_idx << tile_bits;

        for (int g = 0; g < n_gates; g++) {
            float2 g00 = gates[g * 4 + 0];
            float2 g01 = gates[g * 4 + 1];
            float2 g10 = gates[g * 4 + 2];
            float2 g11 = gates[g * 4 + 3];
            int t = targets[g];
            int low_mask = (1 << t) - 1;

            for (int p = threadIdx.x; p < pairs_per_tile; p += blockDim.x) {
                int low_bits = p & low_mask;
                int high_bits = (p >> t) << (t + 1);
                int i0 = tile_offset + (high_bits | low_bits);
                int i1 = i0 + (1 << t);

                float2 a0 = state[i0];
                float2 a1 = state[i1];

                state[i0] = cadd(cmul(g00, a0), cmul(g01, a1));
                state[i1] = cadd(cmul(g10, a0), cmul(g11, a1));
            }

            if (g < n_gates - 1) __syncthreads();
        }
    }
}


// Phase 9a: Warp-shuffle register-tiled multi-gate fusion.
//
// Two amplitudes per thread via float4 load/store. Perfectly coalesced.
// Thread lane l in a warp holds amplitudes at global indices 2*warp_base+2l
// and 2*warp_base+2l+1.
//
// Index bit layout within warp's 64-element block:
//   Bit 0: position within float4 (a0=0, a1=1)
//   Bits 1-5: thread lane
//
// Gate application:
//   Qubit 0: pairs (a0,a1) within same thread → no shuffles!
//   Qubit 1-4: pairs differ in lane bit (t-1) → warp shuffle
//   Qubit 5+: not supported (use individual gate launches)

__global__ void __launch_bounds__(256)
apply_gates_reg_tiled_kernel(
    float4* __restrict__ state4,        // state as float4 (2 complex64 per element)
    const float2* __restrict__ gates,   // N_GATES * 4 float2 (flattened 2x2)
    const int* __restrict__ targets,    // N_GATES target qubits (all < 5)
    const int n_gates,
    const int n_pairs                   // n_amplitudes / 2
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) return;
    int lane = threadIdx.x & 31;

    // Load 2 consecutive amplitudes as float4 (16B, coalesced)
    float4 v = state4[gid];
    float2 a0 = make_float2(v.x, v.y);  // amplitude at even index
    float2 a1 = make_float2(v.z, v.w);  // amplitude at odd index

    // Apply each gate (gates/targets in L1 cache after first block reads them)
    for (int g = 0; g < n_gates; g++) {
        float2 g00 = gates[g * 4 + 0];
        float2 g01 = gates[g * 4 + 1];
        float2 g10 = gates[g * 4 + 2];
        float2 g11 = gates[g * 4 + 3];
        int t = targets[g];

        if (t == 0) {
            // Qubit 0: pair is (a0, a1) within same thread — no shuffle!
            float2 new0 = cadd(cmul(g00, a0), cmul(g01, a1));
            float2 new1 = cadd(cmul(g10, a0), cmul(g11, a1));
            a0 = new0;
            a1 = new1;
        } else {
            // Qubit 1-4: pairs differ in lane bit (t-1)
            int xor_mask = 1 << (t - 1);

            // Shuffle both amplitudes with partner thread
            float2 p0, p1;
            p0.x = __shfl_xor_sync(0xFFFFFFFF, a0.x, xor_mask);
            p0.y = __shfl_xor_sync(0xFFFFFFFF, a0.y, xor_mask);
            p1.x = __shfl_xor_sync(0xFFFFFFFF, a1.x, xor_mask);
            p1.y = __shfl_xor_sync(0xFFFFFFFF, a1.y, xor_mask);

            // Branchless: select gate rows based on which half we hold
            // bit=0: new = g00*val + g01*partner
            // bit=1: new = g11*val + g10*partner
            bool is_low = (lane & xor_mask) == 0;
            float2 ga = is_low ? g00 : g11;  // coefficient for our value
            float2 gb = is_low ? g01 : g10;  // coefficient for partner's value
            a0 = cadd(cmul(ga, a0), cmul(gb, p0));
            a1 = cadd(cmul(ga, a1), cmul(gb, p1));
        }
    }

    // Store back (coalesced float4)
    state4[gid] = make_float4(a0.x, a0.y, a1.x, a1.y);
}


// Python-facing wrapper for register-tiled fusion (warp-shuffle, qubits 0-4)
torch::Tensor apply_gates_reg_tiled_cuda(
    torch::Tensor state,       // [2^n] complex64
    torch::Tensor gates,       // [N, 2, 2] complex64
    torch::Tensor targets      // [N] int (target qubits, all must be < 5)
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(gates.is_cuda(), "gates must be CUDA");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(gates.dtype() == torch::kComplexFloat, "gates must be complex64");

    int n_gates = (int)gates.size(0);
    int n_amplitudes = (int)state.size(0);
    int n_pairs = n_amplitudes / 2;

    auto* state4_ptr = reinterpret_cast<float4*>(state.data_ptr());
    auto* gates_ptr = reinterpret_cast<const float2*>(gates.data_ptr());
    auto* targets_ptr = targets.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int blocks = (n_pairs + threads - 1) / threads;
    apply_gates_reg_tiled_kernel<<<blocks, threads, 0, stream>>>(
        state4_ptr, gates_ptr, targets_ptr, n_gates, n_pairs);

    return state;
}


// Python-facing wrapper
torch::Tensor apply_gate_cuda(
    torch::Tensor state,       // [2^n] complex64
    torch::Tensor gate,        // [2, 2] complex64
    int64_t target_qubit
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(gate.dtype() == torch::kComplexFloat, "gate must be complex64");
    TORCH_CHECK(gate.size(0) == 2 && gate.size(1) == 2, "gate must be 2x2");

    long long n_amplitudes = state.size(0);
    int num_pairs = (int)(n_amplitudes / 2);

    // Gate stays on device — kernel reads it directly (32 bytes, L1 broadcast)
    auto gate_contig = gate.contiguous();
    auto* gate_ptr = reinterpret_cast<float2*>(gate_contig.data_ptr());

    int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (target_qubit == 0) {
        apply_gate_stride1<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4*>(state.data_ptr()),
            gate_ptr,
            num_pairs
        );
    } else {
        apply_gate_sm120<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float2*>(state.data_ptr()),
            gate_ptr,
            (int)target_qubit,
            num_pairs
        );
    }

    return state;
}


// Python-facing wrapper for fused multi-gate (tile-based)
torch::Tensor apply_gates_fused_cuda(
    torch::Tensor state,       // [2^n] complex64
    torch::Tensor gates,       // [N, 2, 2] complex64
    torch::Tensor targets      // [N] int (target qubits)
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(gates.is_cuda(), "gates must be CUDA");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(gates.dtype() == torch::kComplexFloat, "gates must be complex64");

    int n_gates = (int)gates.size(0);
    int n_amplitudes = (int)state.size(0);

    auto gates_c = gates.contiguous();
    auto targets_c = targets.to(torch::kInt32).contiguous();

    // Compute max_target on CPU to determine tile size
    auto targets_cpu = targets_c.cpu();
    auto* t_cpu = targets_cpu.data_ptr<int>();
    int max_target = 0;
    for (int i = 0; i < n_gates; i++) {
        if (t_cpu[i] > max_target) max_target = t_cpu[i];
    }
    int tile_bits = max_target + 1;
    int n_tiles = n_amplitudes >> tile_bits;

    auto* state_ptr = reinterpret_cast<float2*>(state.data_ptr());
    auto* gates_ptr = reinterpret_cast<const float2*>(gates_c.data_ptr());
    auto* targets_ptr = targets_c.data_ptr<int>();

    int threads = 256;
    int blocks = n_tiles;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    apply_gates_tiled_kernel<<<blocks, threads, 0, stream>>>(
        state_ptr, gates_ptr, targets_ptr, n_gates, tile_bits, n_amplitudes);

    return state;
}


// Forward declaration for register-tiled fusion (defined above)
torch::Tensor apply_gates_reg_tiled_cuda(torch::Tensor state, torch::Tensor gates,
                                          torch::Tensor targets);

// Forward declarations — defined in other .cu files
torch::Tensor apply_gate_2q_cuda(torch::Tensor state, torch::Tensor gate,
                                  int64_t qubit0, int64_t qubit1);
torch::Tensor apply_mcgate_cuda(torch::Tensor state, torch::Tensor gate,
                                 torch::Tensor controls, int64_t target_qubit);
torch::Tensor apply_mc2qgate_cuda(torch::Tensor state, torch::Tensor gate,
                                    torch::Tensor controls, int64_t target0, int64_t target1);
torch::Tensor apply_diagonal_cuda(torch::Tensor state, torch::Tensor diag,
                                   int64_t target_qubit);
torch::Tensor apply_diagonal_phase_cuda(torch::Tensor state, double phase_re,
                                         double phase_im, int64_t target_qubit);
torch::Tensor apply_diagonal_2q_cuda(torch::Tensor state, torch::Tensor diag,
                                      int64_t qubit0, int64_t qubit1);
torch::Tensor state_init_cuda(int64_t n_qubits);
int64_t measure_qubit_cuda(torch::Tensor state, int64_t target_qubit, double random_val);
torch::Tensor measure_probs_cuda(torch::Tensor state);
torch::Tensor sample_cuda(torch::Tensor state, torch::Tensor rand_vals);
double expectation_z_cuda(torch::Tensor state, int64_t z_mask);
double expectation_pauli_cuda(torch::Tensor state, int64_t z_mask, int64_t xy_qubit, int64_t op);
int apply_depolarizing_cuda(torch::Tensor state, int64_t target_qubit, double p, double random_val);
int apply_amplitude_damping_cuda(torch::Tensor state, int64_t target_qubit, double gamma, double random_val);
int apply_dephasing_cuda(torch::Tensor state, int64_t target_qubit, double lambda, double random_val);
torch::Tensor apply_gate_batched_cuda(torch::Tensor states, torch::Tensor gate, int64_t target_qubit);
torch::Tensor apply_gates_batched_fused_cuda(torch::Tensor states, torch::Tensor gates, torch::Tensor targets);
torch::Tensor state_init_batched_cuda(int64_t n_states, int64_t n_qubits);
torch::Tensor qv8_simulate(torch::Tensor gate_matrices, torch::Tensor gate_qubits, int64_t num_circuits);
torch::Tensor qv4_simulate(torch::Tensor gate_data, torch::Tensor pair_ids, int n_circuits, int n_gates);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_gate", &apply_gate_cuda,
          "Apply single-qubit gate to state vector (CUDA)",
          py::arg("state"), py::arg("gate"), py::arg("target_qubit"));
    m.def("apply_gates_fused", &apply_gates_fused_cuda,
          "Apply multiple gates fused in a single kernel (CUDA)",
          py::arg("state"), py::arg("gates"), py::arg("targets"));
    m.def("apply_gates_reg_tiled", &apply_gates_reg_tiled_cuda,
          "Apply multiple low-qubit gates fused in registers (CUDA)",
          py::arg("state"), py::arg("gates"), py::arg("targets"));
    m.def("apply_gate_2q", &apply_gate_2q_cuda,
          "Apply two-qubit gate to state vector (CUDA)",
          py::arg("state"), py::arg("gate"), py::arg("qubit0"), py::arg("qubit1"));
    m.def("apply_diagonal", &apply_diagonal_cuda,
          "Apply diagonal single-qubit gate (CUDA)",
          py::arg("state"), py::arg("diag"), py::arg("target_qubit"));
    m.def("apply_diagonal_phase", &apply_diagonal_phase_cuda,
          "Apply phase-type diagonal gate — half traffic (CUDA)",
          py::arg("state"), py::arg("phase_re"), py::arg("phase_im"), py::arg("target_qubit"));
    m.def("apply_diagonal_2q", &apply_diagonal_2q_cuda,
          "Apply diagonal two-qubit gate (CUDA)",
          py::arg("state"), py::arg("diag"), py::arg("qubit0"), py::arg("qubit1"));
    m.def("state_init", &state_init_cuda,
          "Initialize |0>^n state vector (CUDA)",
          py::arg("n_qubits"));
    m.def("measure_qubit", &measure_qubit_cuda,
          "Measure one qubit, collapse state in-place, return 0 or 1 (CUDA)",
          py::arg("state"), py::arg("target_qubit"), py::arg("random_val"));
    m.def("measure_probs", &measure_probs_cuda,
          "Compute probability vector: prob[i] = |state[i]|^2 (CUDA)",
          py::arg("state"));
    m.def("sample", &sample_cuda,
          "Sample n_shots bitstrings from |state|^2 distribution (CUDA)",
          py::arg("state"), py::arg("rand_vals"));
    m.def("expectation_z", &expectation_z_cuda,
          "Pauli Z-string expectation value <psi|Z_mask|psi> (CUDA)",
          py::arg("state"), py::arg("z_mask"));
    m.def("expectation_pauli", &expectation_pauli_cuda,
          "Single-Pauli expectation value: Z-string * one X or Y (CUDA)",
          py::arg("state"), py::arg("z_mask"), py::arg("xy_qubit"), py::arg("op"));
    m.def("apply_mcgate", &apply_mcgate_cuda,
          "Apply multi-controlled 2x2 gate (CUDA)",
          py::arg("state"), py::arg("gate"), py::arg("controls"), py::arg("target_qubit"));
    m.def("apply_mc2qgate", &apply_mc2qgate_cuda,
          "Apply multi-controlled 4x4 gate (CUDA)",
          py::arg("state"), py::arg("gate"), py::arg("controls"),
          py::arg("target0"), py::arg("target1"));
    m.def("apply_depolarizing", &apply_depolarizing_cuda,
          "Apply depolarizing noise channel (CUDA)",
          py::arg("state"), py::arg("target_qubit"), py::arg("p"), py::arg("random_val"));
    m.def("apply_amplitude_damping", &apply_amplitude_damping_cuda,
          "Apply amplitude damping noise channel (CUDA)",
          py::arg("state"), py::arg("target_qubit"), py::arg("gamma"), py::arg("random_val"));
    m.def("apply_dephasing", &apply_dephasing_cuda,
          "Apply dephasing noise channel (CUDA)",
          py::arg("state"), py::arg("target_qubit"), py::arg("lambda"), py::arg("random_val"));
    m.def("apply_gate_batched", &apply_gate_batched_cuda,
          "Apply single-qubit gate to N batched state vectors (CUDA)",
          py::arg("states"), py::arg("gate"), py::arg("target_qubit"));
    m.def("apply_gates_batched_fused", &apply_gates_batched_fused_cuda,
          "Apply multiple gates to N batched states in one fused launch (CUDA)",
          py::arg("states"), py::arg("gates"), py::arg("targets"));
    m.def("state_init_batched", &state_init_batched_cuda,
          "Initialize N state vectors to |0>^n (CUDA)",
          py::arg("n_states"), py::arg("n_qubits"));
    m.def("qv8_simulate", &qv8_simulate,
          "Fused QV-8 quantum volume simulation (batched circuits)",
          py::arg("gate_matrices"), py::arg("gate_qubits"), py::arg("num_circuits"));
    m.def("qv4_simulate", &qv4_simulate,
          "Batched QV-4 circuit simulation (CUDA)",
          py::arg("gate_data"), py::arg("pair_ids"),
          py::arg("n_circuits"), py::arg("n_gates"));
}
