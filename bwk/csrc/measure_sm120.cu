// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Quantum state vector: measurement kernels for sm_120a.
//
// measure_qubit: single-qubit projective measurement (two-pass)
//   Pass 1 — streaming reduction: p0 = sum(|state[i]|^2 for bit_t(i)==0)
//   Pass 2 — conditional collapse: state[i] *= norm (bit==result) or = 0
//
// measure_probs: probability extraction
//   prob[i] = |state[i]|^2  (element-wise, float2 → float)
//
// Both kernels are bandwidth-bound. Target: >80% of peak BW.
// Same playbook as single-gate kernel (no tensor cores, float2 loads).

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// ─── Pass 1: Parallel reduction for p0 ─────────────────────────────────────
//
// Each thread processes multiple amplitudes (grid-stride loop).
// Warp-level reduction via __shfl_xor_sync, then block-level via shared mem,
// then global via atomicAdd.
// Pattern: same as dotproduct kernel (1605 GB/s, 89.4% of peak).

__global__ void __launch_bounds__(256)
measure_qubit_reduce_sm120(
    const float2* __restrict__ state,
    int target_qubit,
    int n,
    float* __restrict__ p0_out   // device scalar, zeroed before launch
) {
    float local = 0.0f;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride: each thread accumulates across its assigned amplitudes
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (!((i >> target_qubit) & 1)) {
            float2 a = state[i];
            local += a.x * a.x + a.y * a.y;
        }
    }

    // Warp-level reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        local += __shfl_xor_sync(0xffffffffu, local, mask);

    // Store warp totals in shared memory (8 warps per 256-thread block)
    __shared__ float smem[8];
    int lane  = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    if (lane == 0) smem[warpid] = local;
    __syncthreads();

    // First warp reduces the 8 warp totals
    if (warpid == 0) {
        local = (lane < 8) ? smem[lane] : 0.0f;
        for (int mask = 4; mask > 0; mask >>= 1)
            local += __shfl_xor_sync(0xffu, local, mask);
        if (lane == 0)
            atomicAdd(p0_out, local);
    }
}

// ─── Pass 2: Conditional collapse (v2: GPU-side decision) ──────────────────
//
// Reads p0 from device memory (L2 broadcast — already in L2 from pass 1).
// Computes result and norm on GPU: no CPU round-trip between passes.
// Each block independently reads p0 — produces the same result (p0 is uniform).
// All blocks write the same value to result_out (idempotent, any block wins).

__global__ void __launch_bounds__(256)
measure_qubit_collapse_sm120(
    float2* __restrict__ state,
    const float* __restrict__ p0_ptr,   // device scalar written by pass 1
    int target_qubit,
    float random_val,
    int n,
    int* __restrict__ result_out        // device scalar: write result for CPU read
) {
    // Every block reads p0 from device (L2-cached after pass 1)
    float p0     = *p0_ptr;
    int result   = (random_val < p0) ? 0 : 1;
    float p_res  = (result == 0) ? p0 : (1.0f - p0);
    float norm   = (p_res > 1e-15f) ? rsqrtf(p_res) : 0.0f;

    // Write result once (any block, any lane — all write same value)
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *result_out = result;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int bit = (idx >> target_qubit) & 1;
    if (bit == result) {
        float2 a = state[idx];
        state[idx] = make_float2(a.x * norm, a.y * norm);
    } else {
        state[idx] = make_float2(0.0f, 0.0f);
    }
}

// ─── measure_probs: |state[i]|^2 ────────────────────────────────────────────
//
// Reads complex64 (float2), writes float32. Traffic: 12 bytes per element.
// Vectorized: each thread processes 2 amplitudes via float4 load.
// Works only when n is a multiple of 2 (always true for quantum state vectors).

__global__ void __launch_bounds__(256)
measure_probs_sm120(
    const float4* __restrict__ state_f4,   // view state as float4 (2 complex64 per load)
    float2* __restrict__ probs_f2,          // view probs as float2 (2 floats per store)
    int n_half                              // n_amplitudes / 2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_half) return;

    // Load 2 complex64 (= 1 float4 = 16 bytes)
    float4 v = state_f4[idx];
    // v = (re0, im0, re1, im1)
    float p0 = v.x * v.x + v.y * v.y;
    float p1 = v.z * v.z + v.w * v.w;
    probs_f2[idx] = make_float2(p0, p1);
}

// ─── Python-facing wrappers ─────────────────────────────────────────────────

// ─── Persistent device/host buffers (allocated once) ───────────────────────
static float* s_p0_buf     = nullptr;  // device: p0 accumulator
static int*   s_result_buf = nullptr;  // device: measurement result
static int*   s_result_pin = nullptr;  // pinned host: for async D2H copy

static void ensure_measure_bufs() {
    if (s_p0_buf) return;
    cudaError_t err;
    err = cudaMalloc(&s_p0_buf, sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("measure: cudaMalloc p0 failed");
    err = cudaMalloc(&s_result_buf, sizeof(int));
    if (err != cudaSuccess) throw std::runtime_error("measure: cudaMalloc result failed");
    err = cudaMallocHost(&s_result_pin, sizeof(int));
    if (err != cudaSuccess) throw std::runtime_error("measure: cudaMallocHost result failed");
}

int64_t measure_qubit_cuda(
    torch::Tensor state,
    int64_t target_qubit,
    double random_val
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");

    int n = (int)state.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto* state_ptr = reinterpret_cast<float2*>(state.data_ptr());
    ensure_measure_bufs();

    // Zero p0 accumulator
    cudaMemsetAsync(s_p0_buf, 0, sizeof(float), stream);

    // Pass 1: streaming reduction → s_p0_buf
    int threads = 256;
    int blocks1 = std::min(4096, (n + threads - 1) / threads);
    measure_qubit_reduce_sm120<<<blocks1, threads, 0, stream>>>(
        state_ptr, (int)target_qubit, n, s_p0_buf
    );

    // Pass 2: GPU-side decision + collapse (reads s_p0_buf from device — NO CPU sync)
    int blocks2 = (n + threads - 1) / threads;
    measure_qubit_collapse_sm120<<<blocks2, threads, 0, stream>>>(
        state_ptr, s_p0_buf, (int)target_qubit, (float)random_val, n, s_result_buf
    );

    // Async copy result to pinned host memory, then sync stream once at end
    cudaMemcpyAsync(s_result_pin, s_result_buf, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return (int64_t)(*s_result_pin);
}

torch::Tensor measure_probs_cuda(torch::Tensor state) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");

    int n = (int)state.size(0);
    TORCH_CHECK((n & 1) == 0, "state size must be even");

    auto probs = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int n_half = n / 2;
    int blocks = (n_half + threads - 1) / threads;

    measure_probs_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float4*>(state.data_ptr()),
        reinterpret_cast<float2*>(probs.data_ptr()),
        n_half
    );

    return probs;
}
