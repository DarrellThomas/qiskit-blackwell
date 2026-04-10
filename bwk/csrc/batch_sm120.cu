// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
//
// qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
// Licensed under the MIT License. See LICENSE file in the project root.
// https://github.com/DarrellThomas/qiskit-blackwell
//
// Batched quantum state vector simulation for sm_120a.
//
// Applies a single-qubit gate to N independent state vectors simultaneously.
// States are stored as [N, 2^n] complex64. One kernel launch processes all N.
//
// Use case: VQE/QAOA shot estimation — run the same circuit on N fresh |0> states
// in one kernel launch, amortizing all launch overhead across N shots.
//
// Memory layout: states[batch_idx * 2^n + amp_idx]
// Grid: dim3(blocks_per_state, N) — blockIdx.y = batch, blockIdx.x = pair block
// Within each block: consecutive threads handle consecutive pairs in the same state.
// This gives fully coalesced access to the state vector.
//
// Scaling:
//   Q=20, N≤12 (96 MB ≤ L2): all states L2-resident → ~constant time
//   Q=20, N>12: DRAM-bound → time scales linearly with N
//   Target: near-linear time from N=1 to GPU saturation.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float2 cmul_b(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cadd_b(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// Compensated complex multiply-accumulate: g0*a0 + g1*a1 (TwoProduct pairwise)
__device__ __forceinline__ float2 cmac2_b(float2 g0, float2 a0, float2 g1, float2 a1) {
    float p0 = g0.x * a0.x, p1 = -g0.y * a0.y, p2 = g1.x * a1.x, p3 = -g1.y * a1.y;
    float re = (p0 + p1) + (p2 + p3);
    re += fmaf(g0.x, a0.x, -p0) + fmaf(-g0.y, a0.y, -p1)
        + fmaf(g1.x, a1.x, -p2) + fmaf(-g1.y, a1.y, -p3);
    float q0 = g0.x * a0.y, q1 = g0.y * a0.x, q2 = g1.x * a1.y, q3 = g1.y * a1.x;
    float im = (q0 + q1) + (q2 + q3);
    im += fmaf(g0.x, a0.y, -q0) + fmaf(g0.y, a0.x, -q1)
        + fmaf(g1.x, a1.y, -q2) + fmaf(g1.y, a1.x, -q3);
    return make_float2(re, im);
}

// Batched single-qubit gate: apply to all N state vectors.
// Grid: dim3(ceil(num_pairs/256), N)
__global__ void __launch_bounds__(256)
apply_gate_batched_sm120(
    float2* __restrict__ states,   // [N, 2^n] flat complex64
    const float2* __restrict__ gate,
    int target_qubit,
    int num_pairs,
    long long state_stride          // 2^n (amplitudes per state)
) {
    float2 g00 = gate[0], g01 = gate[1], g10 = gate[2], g11 = gate[3];

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_pairs) return;

    // blockIdx.y selects which state in the batch
    float2* state = states + (long long)blockIdx.y * state_stride;

    int low_mask = (1 << target_qubit) - 1;
    int low_bits = pid & low_mask;
    int high_bits = (pid >> target_qubit) << (target_qubit + 1);
    int i0 = high_bits | low_bits;
    int i1 = i0 | (1 << target_qubit);

    float2 a0 = state[i0], a1 = state[i1];
    state[i0] = cmac2_b(g00, a0, g01, a1);
    state[i1] = cmac2_b(g10, a0, g11, a1);
}

// stride-1 variant for target_qubit==0: float4 vectorized load
__global__ void __launch_bounds__(256)
apply_gate_batched_stride1_sm120(
    float4* __restrict__ states4,
    const float2* __restrict__ gate,
    int num_pairs,
    long long state_stride4         // 2^(n-1) float4 elements per state
) {
    float2 g00 = gate[0], g01 = gate[1], g10 = gate[2], g11 = gate[3];

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_pairs) return;

    float4* state4 = states4 + (long long)blockIdx.y * state_stride4;
    float4 v = state4[pid];

    float2 a0 = make_float2(v.x, v.y);
    float2 a1 = make_float2(v.z, v.w);
    float2 new0 = cmac2_b(g00, a0, g01, a1);
    float2 new1 = cmac2_b(g10, a0, g11, a1);

    state4[pid] = make_float4(new0.x, new0.y, new1.x, new1.y);
}


// Phase 9c: Batched + fused kernel — apply N_GATES gates to N_BATCH states
// in a single kernel launch. Combines Phase 9a register-tiled fusion with
// Phase 8 batch grid. Each thread processes 2 amplitudes from one state,
// applies ALL gates via warp shuffles (qubits 0-4), stores back.
// Grid: dim3(ceil(n_pairs/256), N_BATCH)
__global__ void __launch_bounds__(256)
apply_gates_batched_fused_sm120(
    float4* __restrict__ states4,       // [N, 2^n] as float4 (2 complex64 per element)
    const float2* __restrict__ gates,   // N_GATES * 4 float2 (flattened 2x2 matrices)
    const int* __restrict__ targets,    // N_GATES target qubits (all < 5)
    int n_gates,
    int n_pairs,                        // 2^(n-1)
    long long state_stride4             // 2^(n-1) float4 elements per state
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_pairs) return;
    int lane = threadIdx.x & 31;

    // Select this batch's state
    float4* state4 = states4 + (long long)blockIdx.y * state_stride4;

    // Load 2 amplitudes as float4 (coalesced)
    float4 v = state4[pid];
    float2 a0 = make_float2(v.x, v.y);
    float2 a1 = make_float2(v.z, v.w);

    // Apply each gate via warp shuffle (same as Phase 9a)
    for (int g = 0; g < n_gates; g++) {
        float2 g00 = gates[g * 4 + 0];
        float2 g01 = gates[g * 4 + 1];
        float2 g10 = gates[g * 4 + 2];
        float2 g11 = gates[g * 4 + 3];
        int t = targets[g];

        if (t == 0) {
            float2 new0 = cmac2_b(g00, a0, g01, a1);
            float2 new1 = cmac2_b(g10, a0, g11, a1);
            a0 = new0;
            a1 = new1;
        } else {
            int xor_mask = 1 << (t - 1);
            float2 p0, p1;
            p0.x = __shfl_xor_sync(0xFFFFFFFF, a0.x, xor_mask);
            p0.y = __shfl_xor_sync(0xFFFFFFFF, a0.y, xor_mask);
            p1.x = __shfl_xor_sync(0xFFFFFFFF, a1.x, xor_mask);
            p1.y = __shfl_xor_sync(0xFFFFFFFF, a1.y, xor_mask);

            bool is_low = (lane & xor_mask) == 0;
            float2 ga = is_low ? g00 : g11;
            float2 gb = is_low ? g01 : g10;
            a0 = cmac2_b(ga, a0, gb, p0);
            a1 = cmac2_b(ga, a1, gb, p1);
        }
    }

    state4[pid] = make_float4(a0.x, a0.y, a1.x, a1.y);
}


// Initialize N state vectors to |0>^n: states[b][0] = 1+0j, all others = 0.
__global__ void __launch_bounds__(256)
state_init_batched_sm120(
    float2* __restrict__ states,
    int n_amplitudes,
    long long state_stride
) {
    // Each thread zeroes one amplitude in one state
    int amp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    if (amp_idx >= n_amplitudes) return;

    float2* state = states + (long long)batch_idx * state_stride;
    if (amp_idx == 0)
        state[0] = make_float2(1.0f, 0.0f);
    else
        state[amp_idx] = make_float2(0.0f, 0.0f);
}


// Python-facing wrappers

torch::Tensor apply_gate_batched_cuda(
    torch::Tensor states,      // [N, 2^n] complex64
    torch::Tensor gate,        // [2, 2] complex64
    int64_t target_qubit
) {
    TORCH_CHECK(states.is_cuda(), "states must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(states.dtype() == torch::kComplexFloat, "states must be complex64");
    TORCH_CHECK(gate.dtype() == torch::kComplexFloat, "gate must be complex64");
    TORCH_CHECK(states.dim() == 2, "states must be [N, 2^n]");

    int N = (int)states.size(0);
    long long n_amplitudes = states.size(1);
    int num_pairs = (int)(n_amplitudes / 2);

    auto gate_c = gate.contiguous();
    auto* gate_ptr = reinterpret_cast<float2*>(gate_c.data_ptr());

    int threads = 256;
    int blocks_per_state = (num_pairs + threads - 1) / threads;
    dim3 grid(blocks_per_state, N);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (target_qubit == 0) {
        apply_gate_batched_stride1_sm120<<<grid, threads, 0, stream>>>(
            reinterpret_cast<float4*>(states.data_ptr()),
            gate_ptr,
            num_pairs,
            n_amplitudes / 2   // stride in float4 units
        );
    } else {
        apply_gate_batched_sm120<<<grid, threads, 0, stream>>>(
            reinterpret_cast<float2*>(states.data_ptr()),
            gate_ptr,
            (int)target_qubit,
            num_pairs,
            n_amplitudes
        );
    }

    return states;
}


torch::Tensor apply_gates_batched_fused_cuda(
    torch::Tensor states,      // [N, 2^n] complex64
    torch::Tensor gates,       // [K, 2, 2] complex64
    torch::Tensor targets      // [K] int32 target qubits (all < 5)
) {
    TORCH_CHECK(states.is_cuda() && gates.is_cuda() && targets.is_cuda());
    TORCH_CHECK(states.dtype() == torch::kComplexFloat);
    TORCH_CHECK(states.dim() == 2, "states must be [N, 2^n]");

    int N = (int)states.size(0);
    long long n_amplitudes = states.size(1);
    int n_pairs = (int)(n_amplitudes / 2);
    int n_gates = (int)gates.size(0);

    auto* states4_ptr = reinterpret_cast<float4*>(states.data_ptr());
    auto* gates_ptr = reinterpret_cast<const float2*>(gates.data_ptr());
    auto* targets_ptr = targets.data_ptr<int>();

    int threads = 256;
    int blocks_per_state = (n_pairs + threads - 1) / threads;
    dim3 grid(blocks_per_state, N);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    apply_gates_batched_fused_sm120<<<grid, threads, 0, stream>>>(
        states4_ptr, gates_ptr, targets_ptr, n_gates, n_pairs, n_amplitudes / 2);

    return states;
}


torch::Tensor state_init_batched_cuda(int64_t n_states, int64_t n_qubits) {
    long long n_amplitudes = 1LL << n_qubits;
    auto states = torch::empty({n_states, n_amplitudes},
                               torch::TensorOptions()
                                   .dtype(torch::kComplexFloat)
                                   .device(torch::kCUDA));

    int threads = 256;
    int blocks_per_state = (int)((n_amplitudes + threads - 1) / threads);
    dim3 grid(blocks_per_state, (int)n_states);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    state_init_batched_sm120<<<grid, threads, 0, stream>>>(
        reinterpret_cast<float2*>(states.data_ptr()),
        (int)n_amplitudes,
        n_amplitudes
    );

    return states;
}

// Binding is registered in apply_gate_sm120.cu via forward declaration.
