// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
//
// qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
// Licensed under the MIT License. See LICENSE file in the project root.
// https://github.com/DarrellThomas/qiskit-blackwell
//
// Quantum state vector: expectation value kernels for sm_120a.
//
// expectation_z: Pauli Z-string expectation value
//   <ψ|Z_{q0} ⊗ Z_{q1} ⊗ ... |ψ> = sum_i |ψ[i]|^2 * (-1)^popcount(i & z_mask)
//
//   Kernel: streaming reduction over all amplitudes. Each thread:
//     1. Loads amplitude (float2)
//     2. Computes probability |a|^2
//     3. Computes sign: __popc(i & z_mask) & 1 → +1 or -1
//     4. Accumulates signed probability
//   Block reduction via warp shuffle + shared memory, global via atomicAdd.
//
//   Same playbook as measure_qubit_reduce_sm120 (dotproduct variant with sign flip).
//   Bandwidth-bound: O(N) reads, O(1) write (scalar result).
//
// expectation_z_xy: General single-Pauli-per-qubit expectation value
//   For Pauli X on qubit q: pairs amplitudes at (i, i^(1<<q)), accumulates cross-terms
//   For Pauli Y on qubit q: same pairing, with imaginary phase (-i factor)
//   For Pauli Z on qubit q: uses sign flip (same as expectation_z)
//   For Identity: contributes factor 1 (can be ignored in reduction)
//
//   Single-qubit observable (1Q Pauli): handles X, Y, Z on one qubit.
//   General multi-Pauli string is built from these.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ─── Pauli Z-string expectation value ────────────────────────────────────────
//
// Result = sum_i |ψ[i]|^2 * (-1)^popcount(i & z_mask)
// When z_mask=0 (all identity), result = sum |ψ|^2 = 1 (norm check).
// When z_mask has 1 qubit set, result = P(q=0) - P(q=1) = 2*P(q=0) - 1.
//
// Pattern: identical to dotproduct streaming reduction at 89.4% DRAM BW.
// The sign computation (__popc) is 1 integer ALU instruction — negligible.

__global__ void __launch_bounds__(256)
expectation_z_sm120(
    const float2* __restrict__ state,
    uint32_t z_mask,
    int n,
    float* __restrict__ out   // device scalar, zeroed before launch
) {
    float local = 0.0f;
    int stride = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float2 a = state[i];
        float prob = a.x * a.x + a.y * a.y;
        // Sign: +1 if even number of 1-bits in (i & z_mask), -1 if odd
        int parity = __popc((unsigned)i & z_mask) & 1;
        local += prob * (1.0f - 2.0f * parity);
    }

    // Warp-level reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        local += __shfl_xor_sync(0xffffffffu, local, mask);

    __shared__ float smem[8];
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    if (lane == 0) smem[warpid] = local;
    __syncthreads();

    if (warpid == 0) {
        local = (lane < 8) ? smem[lane] : 0.0f;
        for (int mask = 4; mask > 0; mask >>= 1)
            local += __shfl_xor_sync(0xffu, local, mask);
        if (lane == 0)
            atomicAdd(out, local);
    }
}

// ─── Single-qubit Pauli expectation value ────────────────────────────────────
//
// For a single Pauli operator on qubit q, with Z-string on remaining qubits:
//   <ψ|Z_mask ⊗ P_q|ψ> where P_q ∈ {X, Y, Z, I}
//
// X on qubit q: <X_q> = 2 * Re(sum_{bit_q=0} ψ[i]* × ψ[i | stride])
//   (paired-amplitude reduction, like single-gate but accumulating)
//
// Y on qubit q: <Y_q> = -2 * Im(sum_{bit_q=0} ψ[i]* × ψ[i | stride])
//   (same pairing but Im part)
//
// Z on qubit q: handled by expectation_z_sm120 with z_mask |= (1<<q)
//
// This kernel handles a tensor product with exactly ONE non-Z, non-I qubit.
// For Pauli X or Y, the amplitude pairing structure requires accessing pairs.
//
// op: 0=I, 1=X, 2=Y, 3=Z (we only call this for X or Y, Z uses expectation_z)

__global__ void __launch_bounds__(256)
expectation_xy_sm120(
    const float2* __restrict__ state,
    uint32_t z_mask,    // Z operators on these qubits (other than the XY qubit)
    int xy_qubit,       // qubit with X or Y operator
    int op,             // 1=X, 2=Y
    int n,
    float* __restrict__ out
) {
    float local = 0.0f;
    int stride = gridDim.x * blockDim.x;
    int xy_stride = 1 << xy_qubit;

    // Only process amplitudes where bit xy_qubit == 0 (pairs both i and i^xy_stride)
    int n_pairs = n / 2;
    int lo_mask = xy_stride - 1;

    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < n_pairs; g += stride) {
        // Insert 0 at position xy_qubit to get the i with bit_q=0
        int lo = g & lo_mask;
        int hi = g >> xy_qubit;
        int i0 = (hi << (xy_qubit + 1)) | lo;
        int i1 = i0 | xy_stride;

        float2 a0 = state[i0];  // ψ[i0], bit_q=0
        float2 a1 = state[i1];  // ψ[i1], bit_q=1

        // Z-string sign from other qubits (i0 and i1 differ only at xy_qubit)
        int parity = __popc((unsigned)i0 & z_mask) & 1;
        float sign = 1.0f - 2.0f * parity;

        // Pauli X contribution: 2 * Re(a0* × a1) * sign
        // a0* × a1 = (a0.x - i*a0.y) × (a1.x + i*a1.y)
        //          = a0.x*a1.x + a0.y*a1.y + i*(a0.x*a1.y - a0.y*a1.x)
        float re_cross = a0.x * a1.x + a0.y * a1.y;
        float im_cross = a0.x * a1.y - a0.y * a1.x;

        if (op == 1) {
            // X: 2 * Re(a0* a1) = re_cross + (paired term is counted once)
            local += sign * 2.0f * re_cross;
        } else {
            // Y: -2 * Im(a0* a1) = im_cross
            local += sign * 2.0f * im_cross;
        }
    }

    // Warp-level reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        local += __shfl_xor_sync(0xffffffffu, local, mask);

    __shared__ float smem[8];
    int lane   = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    if (lane == 0) smem[warpid] = local;
    __syncthreads();

    if (warpid == 0) {
        local = (lane < 8) ? smem[lane] : 0.0f;
        for (int mask = 4; mask > 0; mask >>= 1)
            local += __shfl_xor_sync(0xffu, local, mask);
        if (lane == 0)
            atomicAdd(out, local);
    }
}

// ─── Persistent device buffer ────────────────────────────────────────────────
static float* s_exp_buf    = nullptr;  // device: expectation accumulator
static float* s_exp_pin    = nullptr;  // pinned host: for async D2H copy

static void ensure_exp_bufs() {
    if (s_exp_buf) return;
    cudaError_t err;
    err = cudaMalloc(&s_exp_buf, sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("expectation: cudaMalloc failed");
    err = cudaMallocHost(&s_exp_pin, sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("expectation: cudaMallocHost failed");
}

// ─── Python-facing wrappers ──────────────────────────────────────────────────

double expectation_z_cuda(
    torch::Tensor state,
    int64_t z_mask
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");

    int n = (int)state.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto* state_ptr = reinterpret_cast<float2*>(state.data_ptr());
    ensure_exp_bufs();

    cudaMemsetAsync(s_exp_buf, 0, sizeof(float), stream);

    int threads = 256;
    int blocks  = std::min(4096, (n + threads - 1) / threads);
    expectation_z_sm120<<<blocks, threads, 0, stream>>>(
        state_ptr, (uint32_t)z_mask, n, s_exp_buf
    );

    cudaMemcpyAsync(s_exp_pin, s_exp_buf, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return (double)(*s_exp_pin);
}

double expectation_pauli_cuda(
    torch::Tensor state,
    int64_t z_mask,
    int64_t xy_qubit,
    int64_t op       // 0=I (use expectation_z), 1=X, 2=Y
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(op >= 0 && op <= 2, "op must be 0=I/Z, 1=X, 2=Y");

    if (op == 0) {
        // Pure Z string (or identity): use expectation_z
        return expectation_z_cuda(state, z_mask);
    }

    int n = (int)state.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto* state_ptr = reinterpret_cast<float2*>(state.data_ptr());
    ensure_exp_bufs();

    cudaMemsetAsync(s_exp_buf, 0, sizeof(float), stream);

    int threads = 256;
    int n_half  = n / 2;
    int blocks  = std::min(4096, (n_half + threads - 1) / threads);
    expectation_xy_sm120<<<blocks, threads, 0, stream>>>(
        state_ptr, (uint32_t)z_mask, (int)xy_qubit, (int)op, n, s_exp_buf
    );

    cudaMemcpyAsync(s_exp_pin, s_exp_buf, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return (double)(*s_exp_pin);
}
