// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Quantum state vector: diagonal gate application kernel for sm_120a.
//
// Diagonal gates have non-zero entries only on the diagonal.
// Each amplitude is independently scaled — no cross-terms, no pair coupling.
// state[i] *= diag[bit(i, target_qubit)]  (1Q)
// state[i] *= diag[2*bit(i,q1) + bit(i,q0)]  (2Q)
//
// This is SIMPLER than general gates: no gather/scatter, pure element-wise.
// Float4 vectorized loads work for ALL target qubits since amplitudes are independent.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Complex multiply: a * b
__device__ __forceinline__ float2 cmul_diag(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// v3: General diagonal gate. 1 amplitude per thread.
__global__ void __launch_bounds__(256)
apply_diagonal_sm120(
    float2* __restrict__ state,
    const float2* __restrict__ diag,
    const int target_qubit,
    const int n_amplitudes
) {
    float2 d0 = diag[0];
    float2 d1 = diag[1];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_amplitudes) return;

    float2 a = state[idx];
    float2 d = ((idx >> target_qubit) & 1) ? d1 : d0;
    state[idx] = cmul_diag(a, d);
}

// v3: Phase-type diagonal gate — only modifies amplitudes where bit=1.
// For gates where diag[0]=1+0j (Z, T, S, Phase, Rz): skip bit=0 amplitudes entirely.
// Traffic: read n/2 + write n/2 = 8MB at Q=20 (vs 16MB for general gate).
__global__ void __launch_bounds__(256)
apply_diagonal_phase_sm120(
    float2* __restrict__ state,
    const float2 phase,               // diag[1] (the non-trivial entry)
    const int target_qubit,
    const int n_pairs                  // n_amplitudes / 2
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    // Compute index where bit target_qubit is 1
    // Insert a 1-bit at position target_qubit
    int low_mask = (1 << target_qubit) - 1;
    int lo = tid & low_mask;
    int hi = tid >> target_qubit;
    int idx = (hi << (target_qubit + 1)) | (1 << target_qubit) | lo;

    float2 a = state[idx];
    state[idx] = cmul_diag(a, phase);
}


// v1: Two-qubit diagonal gate. 1 amplitude per thread.
// state[i] *= diag[2*bit(i,q1) + bit(i,q0)]
__global__ void __launch_bounds__(256)
apply_diagonal2q_sm120(
    float2* __restrict__ state,
    const float2* __restrict__ diag,  // [4] complex64: diagonal entries
    const int qubit0,
    const int qubit1,
    const int n_amplitudes
) {
    float2 d00 = diag[0], d01 = diag[1], d10 = diag[2], d11 = diag[3];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_amplitudes) return;

    int sel = (((idx >> qubit1) & 1) << 1) | ((idx >> qubit0) & 1);
    float2 a = state[idx];
    float2 d = (sel == 0) ? d00 : (sel == 1) ? d01 : (sel == 2) ? d10 : d11;
    state[idx] = cmul_diag(a, d);
}


// State initialization: set state[0] = 1+0j, all others = 0
__global__ void state_init_kernel(float2* __restrict__ state, int n_amplitudes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_amplitudes) return;
    state[idx] = (idx == 0) ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 0.0f);
}


// ─── Python-facing wrappers ───────────────────────────────────────────

torch::Tensor apply_diagonal_cuda(
    torch::Tensor state,
    torch::Tensor diag,
    int64_t target_qubit
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(diag.is_cuda(), "diag must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(diag.dtype() == torch::kComplexFloat, "diag must be complex64");
    TORCH_CHECK(diag.numel() == 2, "diag must have 2 elements");

    long long n = state.size(0);
    auto diag_c = diag.contiguous();

    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Always use general path — safe for CUDA graph capture.
    // Phase-optimized path available via apply_diagonal_phase_cuda().
    int blocks = ((int)n + threads - 1) / threads;
    apply_diagonal_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(state.data_ptr()),
        reinterpret_cast<float2*>(diag_c.data_ptr()),
        (int)target_qubit,
        (int)n
    );

    return state;
}

// Phase-optimized: only touches amplitudes where bit=1. Half the traffic.
// Phase real/imag passed as scalars — no device-to-host copy needed.
torch::Tensor apply_diagonal_phase_cuda(
    torch::Tensor state,
    double phase_re,          // real part of diag[1]
    double phase_im,          // imag part of diag[1]
    int64_t target_qubit
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");

    long long n = state.size(0);
    int n_pairs = (int)(n / 2);
    float2 phase_val = make_float2((float)phase_re, (float)phase_im);

    int threads = 256;
    int blocks = (n_pairs + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    apply_diagonal_phase_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(state.data_ptr()),
        phase_val,
        (int)target_qubit,
        n_pairs
    );

    return state;
}

torch::Tensor apply_diagonal_2q_cuda(
    torch::Tensor state,
    torch::Tensor diag,
    int64_t qubit0,
    int64_t qubit1
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(diag.is_cuda(), "diag must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(diag.dtype() == torch::kComplexFloat, "diag must be complex64");
    TORCH_CHECK(diag.numel() == 4, "diag must have 4 elements");

    int q0 = (int)std::min(qubit0, qubit1);
    int q1 = (int)std::max(qubit0, qubit1);
    TORCH_CHECK(q0 != q1, "qubit0 and qubit1 must differ");

    long long n = state.size(0);
    auto diag_c = diag.contiguous();

    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    apply_diagonal2q_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(state.data_ptr()),
        reinterpret_cast<float2*>(diag_c.data_ptr()),
        q0, q1,
        (int)n
    );

    return state;
}

torch::Tensor state_init_cuda(int64_t n_qubits) {
    long long n = 1LL << n_qubits;
    auto state = torch::zeros({n}, torch::dtype(torch::kComplexFloat).device(torch::kCUDA));

    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    state_init_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(state.data_ptr()),
        (int)n
    );

    return state;
}
