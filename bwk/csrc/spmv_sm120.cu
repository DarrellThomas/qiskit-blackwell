// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
//
// qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
// Licensed under the MIT License. See LICENSE file in the project root.
// https://github.com/DarrellThomas/qiskit-blackwell
//
// Sparse matrix-vector multiply (SpMV) for Chebyshev Hamiltonian simulation.
//
// CSR format: y = alpha * A * x + beta * y
//
// Quantum Hamiltonians from Pauli decomposition have uniform row lengths
// (each Pauli term contributes exactly one nonzero per row). This makes
// the access pattern predictable and amenable to row-per-thread processing.
//
// TwoProduct-compensated dot product within each row for precision.
//
// EXPERIMENTAL (beta) — part of the Chebyshev evolution engine.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// TwoProduct-compensated complex multiply-accumulate for SpMV inner loop.
// Accumulates val * x[col] into (sum_re, sum_im, comp_re, comp_im).
__device__ __forceinline__ void cmac_kahan(
    float2 val, float2 xj,
    float& sum_re, float& comp_re,
    float& sum_im, float& comp_im
) {
    // Real part: val.x*xj.x - val.y*xj.y
    float p0 = val.x * xj.x;
    float e0 = fmaf(val.x, xj.x, -p0);
    float p1 = -(val.y * xj.y);
    float e1 = fmaf(-val.y, xj.y, -p1);
    float term_re = p0 + p1;
    float corr_re = e0 + e1 + ((p0 - term_re) + p1);

    // Kahan accumulation for real
    float t_re = sum_re + term_re;
    comp_re += corr_re + ((sum_re - t_re) + term_re);
    sum_re = t_re;

    // Imag part: val.x*xj.y + val.y*xj.x
    float q0 = val.x * xj.y;
    float e2 = fmaf(val.x, xj.y, -q0);
    float q1 = val.y * xj.x;
    float e3 = fmaf(val.y, xj.x, -q1);
    float term_im = q0 + q1;
    float corr_im = e2 + e3 + ((q0 - term_im) + q1);

    // Kahan accumulation for imag
    float t_im = sum_im + term_im;
    comp_im += corr_im + ((sum_im - t_im) + term_im);
    sum_im = t_im;
}

// CSR SpMV kernel: y = alpha * A * x + beta * y
// One thread per row. For quantum Hamiltonians, rows are short (1-100 nnz)
// so row-per-thread is appropriate.
__global__ void __launch_bounds__(256)
spmv_csr_sm120(
    const int* __restrict__ row_ptr,      // [n_rows + 1]
    const int* __restrict__ col_idx,      // [nnz]
    const float2* __restrict__ values,    // [nnz] complex64
    const float2* __restrict__ x,         // [n_rows] input vector
    float2* __restrict__ y,               // [n_rows] output vector
    const float2 alpha,                   // complex scalar
    const float2 beta,                    // complex scalar
    const int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    // Kahan-compensated accumulation of A[row,:] * x
    float sum_re = 0.0f, comp_re = 0.0f;
    float sum_im = 0.0f, comp_im = 0.0f;

    for (int j = start; j < end; j++) {
        float2 v = values[j];
        float2 xj = x[col_idx[j]];
        cmac_kahan(v, xj, sum_re, comp_re, sum_im, comp_im);
    }

    float dot_re = sum_re + comp_re;
    float dot_im = sum_im + comp_im;

    // result = alpha * dot + beta * y[row]
    float2 y_old = y[row];

    // alpha * dot (complex multiply)
    float r_re = alpha.x * dot_re - alpha.y * dot_im;
    float r_im = alpha.x * dot_im + alpha.y * dot_re;

    // beta * y_old (complex multiply)
    r_re += beta.x * y_old.x - beta.y * y_old.y;
    r_im += beta.x * y_old.y + beta.y * y_old.x;

    y[row] = make_float2(r_re, r_im);
}


// ─── Python-facing wrapper ────────────────────────────────────────────

void spmv_csr_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x,
    torch::Tensor y,
    double alpha_re, double alpha_im,
    double beta_re, double beta_im
) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda() && values.is_cuda(),
                "CSR arrays must be CUDA tensors");
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "vectors must be CUDA tensors");
    TORCH_CHECK(values.dtype() == torch::kComplexFloat, "values must be complex64");
    TORCH_CHECK(x.dtype() == torch::kComplexFloat, "x must be complex64");
    TORCH_CHECK(y.dtype() == torch::kComplexFloat, "y must be complex64");

    int n_rows = (int)(row_ptr.size(0) - 1);
    float2 alpha = make_float2((float)alpha_re, (float)alpha_im);
    float2 beta_val = make_float2((float)beta_re, (float)beta_im);

    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    spmv_csr_sm120<<<blocks, threads, 0, stream>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        reinterpret_cast<const float2*>(values.data_ptr()),
        reinterpret_cast<const float2*>(x.data_ptr()),
        reinterpret_cast<float2*>(y.data_ptr()),
        alpha,
        beta_val,
        n_rows
    );
}
