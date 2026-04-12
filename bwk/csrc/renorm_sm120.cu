// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
//
// qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
// Licensed under the MIT License. See LICENSE file in the project root.
// https://github.com/DarrellThomas/qiskit-blackwell
//
// State vector renormalization kernel for sm_120a.
//
// Two-pass design:
//   Pass 1: Streaming reduction to compute ||psi||^2 = sum |a_i|^2
//   Pass 2: Scale all amplitudes by 1/sqrt(||psi||^2)
//
// Corrects accumulated floating-point drift from repeated gate application.
// Pattern follows amp_damp_prob_sm120 from noise_sm120.cu.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Pass 1: Compute ||psi||^2 via streaming reduction.
// Grid-stride loop + block reduction + atomicAdd.
__global__ void __launch_bounds__(256)
norm_reduce_sm120(
    const float2* __restrict__ state,
    float* __restrict__ norm_out,       // output: scalar (device), must be zeroed before launch
    const int n_amplitudes
) {
    __shared__ float sdata[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Grid-stride accumulation — all amplitudes
    for (int i = tid; i < n_amplitudes; i += gridDim.x * blockDim.x) {
        float2 a = state[i];
        sum += a.x * a.x + a.y * a.y;
    }

    // Block reduction via shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(norm_out, sdata[0]);
}

// Pass 2: Scale all amplitudes by rsqrt(norm) to restore unit norm.
// Reads norm from device memory (broadcast from L2 on first access).
__global__ void __launch_bounds__(256)
renorm_scale_sm120(
    float2* __restrict__ state,
    const float* __restrict__ norm_ptr,  // device scalar: ||psi||^2
    const int n_amplitudes
) {
    float scale = rsqrtf(*norm_ptr);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_amplitudes) return;

    float2 a = state[idx];
    state[idx] = make_float2(a.x * scale, a.y * scale);
}


// ─── Python-facing wrapper ────────────────────────────────────────────

// Static device buffer for norm accumulation (allocated once, reused)
static float* s_norm_buf = nullptr;

torch::Tensor renormalize_cuda(torch::Tensor state) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");

    long long n = state.size(0);
    int threads = 256;
    int blocks = std::min((int)((n + threads - 1) / threads), 1024);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Allocate persistent device buffer for norm scalar
    if (s_norm_buf == nullptr) {
        cudaMalloc(&s_norm_buf, sizeof(float));
    }

    // Zero the accumulator
    cudaMemsetAsync(s_norm_buf, 0, sizeof(float), stream);

    // Pass 1: Compute ||psi||^2
    norm_reduce_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(state.data_ptr()),
        s_norm_buf,
        (int)n
    );

    // Pass 2: Scale by 1/sqrt(||psi||^2)
    int scale_blocks = ((int)n + threads - 1) / threads;
    renorm_scale_sm120<<<scale_blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(state.data_ptr()),
        s_norm_buf,
        (int)n
    );

    return state;
}
