// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Quantum state vector: sampling kernel for sm_120a.
//
// sample(state, n_shots):
//   Step 1 — probs[i] = |state[i]|^2  (element-wise kernel)
//   Step 2 — cdf[i] = prefix_sum(probs[0..i])  (CUB DeviceScan::InclusiveSum)
//   Step 3 — bitstring[k] = binary_search(cdf, u[k]) for each shot (parallel)
//
// All three steps run on the same CUDA stream for correctness.
// Random values u[k] ~ Uniform(0,1) are passed as a GPU tensor from Python.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Include CUB from CUDA 13 CCCL package
#include <cub/cub.cuh>

// ─── Step 1: probability extraction ─────────────────────────────────────────
// Vectorized: 2 complex64 per thread via float4 load → 2 float writes.
// Traffic: 12 bytes/amplitude (8B read + 4B write). Bandwidth-bound.

__global__ void __launch_bounds__(256)
sample_probs_sm120(
    const float4* __restrict__ state_f4,
    float* __restrict__ probs,
    int n_half
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_half) return;

    float4 v = state_f4[idx];
    probs[2 * idx]     = v.x * v.x + v.y * v.y;
    probs[2 * idx + 1] = v.z * v.z + v.w * v.w;
}

// ─── Step 3: parallel binary search ─────────────────────────────────────────
// Each thread handles one shot. Binary search on sorted CDF array.
// At Q=20 (2^20 = 1M amplitudes), binary search depth = 20 iterations.
// Fully coalesced random reads (sorted order) if u[k] is pre-sorted.

__global__ void __launch_bounds__(256)
sample_binary_search_sm120(
    const float* __restrict__ cdf,
    int n_amplitudes,
    const float* __restrict__ u,     // random values in [0,1), n_shots
    int n_shots,
    int64_t* __restrict__ results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_shots) return;

    float val = u[tid];

    // Binary search: find smallest i such that cdf[i] >= val
    int lo = 0, hi = n_amplitudes - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    results[tid] = (int64_t)lo;
}

// ─── Python-facing wrapper ───────────────────────────────────────────────────

torch::Tensor sample_cuda(
    torch::Tensor state,
    torch::Tensor rand_vals    // float32 [n_shots], pre-generated on GPU
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(rand_vals.is_cuda(), "rand_vals must be CUDA");
    TORCH_CHECK(rand_vals.dtype() == torch::kFloat32, "rand_vals must be float32");

    int n = (int)state.size(0);
    int n_shots = (int)rand_vals.size(0);
    TORCH_CHECK((n & 1) == 0, "state size must be even");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Allocate temporaries: probs and cdf
    auto probs = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto cdf   = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto results = torch::empty({n_shots}, torch::dtype(torch::kInt64).device(torch::kCUDA));

    float* probs_ptr   = probs.data_ptr<float>();
    float* cdf_ptr     = cdf.data_ptr<float>();
    float* u_ptr       = rand_vals.data_ptr<float>();
    int64_t* res_ptr   = results.data_ptr<int64_t>();

    int threads = 256;

    // Step 1: compute probs = |state|^2 (vectorized float4)
    int n_half = n / 2;
    int blocks1 = (n_half + threads - 1) / threads;
    sample_probs_sm120<<<blocks1, threads, 0, stream>>>(
        reinterpret_cast<const float4*>(state.data_ptr()),
        probs_ptr,
        n_half
    );

    // Step 2: CUB inclusive prefix sum: cdf[i] = sum(probs[0..i])
    {
        // Query temp storage size
        void*  d_temp  = nullptr;
        size_t temp_sz = 0;
        cub::DeviceScan::InclusiveSum(d_temp, temp_sz, probs_ptr, cdf_ptr, n, stream);

        // Allocate and run
        auto temp_buf = torch::empty(
            {(int64_t)temp_sz},
            torch::dtype(torch::kByte).device(torch::kCUDA)
        );
        d_temp = temp_buf.data_ptr();
        cub::DeviceScan::InclusiveSum(d_temp, temp_sz, probs_ptr, cdf_ptr, n, stream);
    }

    // Step 3: parallel binary search
    int blocks3 = (n_shots + threads - 1) / threads;
    sample_binary_search_sm120<<<blocks3, threads, 0, stream>>>(
        cdf_ptr, n, u_ptr, n_shots, res_ptr
    );

    return results;
}
