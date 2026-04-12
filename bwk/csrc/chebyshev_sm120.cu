// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
//
// qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
// Licensed under the MIT License. See LICENSE file in the project root.
// https://github.com/DarrellThomas/qiskit-blackwell
//
// Chebyshev recurrence vector operations for Hamiltonian simulation.
//
// Three kernels:
//   1. chebyshev_step:  t_next = 2 * spmv_result - t_prev
//   2. chebyshev_accum: accum += c_k * t_k
//   3. chebyshev_step_accum: fused step + accumulation (saves 1 memory pass)
//
// All are bandwidth-bound element-wise vector operations.
//
// EXPERIMENTAL (beta) — part of the Chebyshev evolution engine.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Chebyshev recurrence step: t_next[i] = 2 * spmv_result[i] - t_prev[i]
__global__ void __launch_bounds__(256)
chebyshev_step_sm120(
    const float2* __restrict__ spmv_result,  // H_norm * T_k (from SpMV)
    const float2* __restrict__ t_prev,       // T_{k-1}
    float2* __restrict__ t_next,             // T_{k+1} output
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float2 s = spmv_result[idx];
    float2 p = t_prev[idx];

    // t_next = 2 * s - p
    t_next[idx] = make_float2(2.0f * s.x - p.x, 2.0f * s.y - p.y);
}

// Chebyshev accumulation: accum[i] += c_k * t_k[i]
// c_k is a complex Bessel coefficient.
__global__ void __launch_bounds__(256)
chebyshev_accum_sm120(
    float2* __restrict__ accum,
    const float2* __restrict__ t_k,
    const float2 c_k,              // complex Bessel coefficient
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float2 t = t_k[idx];
    float2 a = accum[idx];

    // accum += c_k * t_k (complex multiply + add)
    a.x += c_k.x * t.x - c_k.y * t.y;
    a.y += c_k.x * t.y + c_k.y * t.x;

    accum[idx] = a;
}

// Fused step + accumulation: saves one full memory pass.
// Computes: t_next = 2 * spmv_result - t_prev
//           accum += c_k * t_next
__global__ void __launch_bounds__(256)
chebyshev_step_accum_sm120(
    const float2* __restrict__ spmv_result,
    const float2* __restrict__ t_prev,
    float2* __restrict__ t_next,
    float2* __restrict__ accum,
    const float2 c_k,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float2 s = spmv_result[idx];
    float2 p = t_prev[idx];
    float2 a = accum[idx];

    // Step
    float2 tn = make_float2(2.0f * s.x - p.x, 2.0f * s.y - p.y);

    // Accumulate
    a.x += c_k.x * tn.x - c_k.y * tn.y;
    a.y += c_k.x * tn.y + c_k.y * tn.x;

    t_next[idx] = tn;
    accum[idx] = a;
}


// ─── Python-facing wrappers ──────────────────────────────────────────

void chebyshev_step_cuda(
    torch::Tensor spmv_result,
    torch::Tensor t_prev,
    torch::Tensor t_next
) {
    TORCH_CHECK(spmv_result.is_cuda() && t_prev.is_cuda() && t_next.is_cuda(),
                "all tensors must be CUDA");
    int n = (int)spmv_result.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    chebyshev_step_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(spmv_result.data_ptr()),
        reinterpret_cast<const float2*>(t_prev.data_ptr()),
        reinterpret_cast<float2*>(t_next.data_ptr()),
        n
    );
}

void chebyshev_accum_cuda(
    torch::Tensor accum,
    torch::Tensor t_k,
    double c_re, double c_im
) {
    TORCH_CHECK(accum.is_cuda() && t_k.is_cuda(), "tensors must be CUDA");
    int n = (int)accum.size(0);
    float2 c_k = make_float2((float)c_re, (float)c_im);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    chebyshev_accum_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(accum.data_ptr()),
        reinterpret_cast<const float2*>(t_k.data_ptr()),
        c_k,
        n
    );
}

void chebyshev_step_accum_cuda(
    torch::Tensor spmv_result,
    torch::Tensor t_prev,
    torch::Tensor t_next,
    torch::Tensor accum,
    double c_re, double c_im
) {
    TORCH_CHECK(spmv_result.is_cuda() && t_prev.is_cuda() &&
                t_next.is_cuda() && accum.is_cuda(), "tensors must be CUDA");
    int n = (int)spmv_result.size(0);
    float2 c_k = make_float2((float)c_re, (float)c_im);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    chebyshev_step_accum_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(spmv_result.data_ptr()),
        reinterpret_cast<const float2*>(t_prev.data_ptr()),
        reinterpret_cast<float2*>(t_next.data_ptr()),
        reinterpret_cast<float2*>(accum.data_ptr()),
        c_k,
        n
    );
}
