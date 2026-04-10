// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Darrell Thomas / Redshed Lab LLC
//
// qiskit-blackwell — Custom CUDA quantum simulation kernels for RTX 5090
// Licensed under the MIT License. See LICENSE file in the project root.
// https://github.com/DarrellThomas/qiskit-blackwell
//
// Multi-controlled two-qubit gate kernel for sm_120a (Phase 10).
//
// Applies a 4x4 complex unitary gate to (target0, target1) conditioned on
// k control qubits all being |1>. For each group of 4 amplitudes where all
// control bits are 1 and both target bits enumerate {00, 01, 10, 11}:
//
//   i00 = base | control_mask
//   i01 = i00 | (1 << target0)
//   i10 = i00 | (1 << target1)
//   i11 = i00 | (1 << target0) | (1 << target1)
//
//   new[i00] = gate[0]*s[i00] + gate[1]*s[i01] + gate[2]*s[i10] + gate[3]*s[i11]
//   new[i01] = gate[4]*s[i00] + gate[5]*s[i01] + gate[6]*s[i10] + gate[7]*s[i11]
//   new[i10] = gate[8]*s[i00] + gate[9]*s[i01] + gate[10]*s[i10] + gate[11]*s[i11]
//   new[i11] = gate[12]*s[i00] + gate[13]*s[i01] + gate[14]*s[i10] + gate[15]*s[i11]
//
// Uses "insert zeros" indexing: only 2^(n-k-2) threads, all doing useful work.
// No wasted threads (no check-and-skip).
//
// Memory characteristics:
//   Traffic: 2^(n-k-2) groups × 64B/group = 2^(n-k+4) bytes
//   For CSWAP (k=1, n=20): 2^17 groups × 64B = 8 MB
//   This kernel is MEMORY-BANDWIDTH-BOUND. No tensor cores.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float2 cmul_mc2(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cadd_mc2(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// Kahan-compensated 8-term dot product using TwoProduct (FMA)
__device__ __forceinline__ float comp_dot8_mc2(
    float a0, float b0, float a1, float b1, float a2, float b2, float a3, float b3,
    float a4, float b4, float a5, float b5, float a6, float b6, float a7, float b7) {
    float s = a0 * b0, c = fmaf(a0, b0, -s), p, e, t;
    #define KA(ai, bi) p=(ai)*(bi); e=fmaf((ai),(bi),-p); t=s+p; c+=e+((s-t)+p); s=t;
    KA(a1,b1); KA(a2,b2); KA(a3,b3); KA(a4,b4); KA(a5,b5); KA(a6,b6); KA(a7,b7);
    #undef KA
    return s + c;
}

// Compensated complex multiply-accumulate: g0*a0 + g1*a1 + g2*a2 + g3*a3
__device__ __forceinline__ float2 cmac4_mc2(
    float2 g0, float2 a0, float2 g1, float2 a1,
    float2 g2, float2 a2, float2 g3, float2 a3) {
    return make_float2(
        comp_dot8_mc2(g0.x,a0.x, -g0.y,a0.y, g1.x,a1.x, -g1.y,a1.y,
                      g2.x,a2.x, -g2.y,a2.y, g3.x,a3.x, -g3.y,a3.y),
        comp_dot8_mc2(g0.x,a0.y, g0.y,a0.x, g1.x,a1.y, g1.y,a1.x,
                      g2.x,a2.y, g2.y,a2.x, g3.x,a3.y, g3.y,a3.x));
}

// Kernel parameter struct — passed by value (fits in registers)
struct MC2QGateParams {
    int sorted_qubits[10];  // sorted qubit positions (controls + 2 targets)
    int n_affected;          // k + 2
    int target0;             // lower target qubit
    int target1;             // higher target qubit
    int control_mask;        // OR of (1 << c[j]) for all control qubits
    int num_groups;          // 2^(n - n_affected)
};

// Insert zeros at sorted positions sorted_qubits[0..n-1].
// Each insertion shifts bits above the position up by one, creating a 0 there.
__device__ __forceinline__ int insert_zeros(int g, const int* q, int n) {
    int r = g;
    for (int j = 0; j < n; j++) {
        int lo = r & ((1 << q[j]) - 1);
        int hi = r >> q[j];
        r = (hi << (q[j] + 1)) | lo;
    }
    return r;
}

__global__ void __launch_bounds__(256)
apply_mc2qgate_sm120(
    float2* __restrict__ state,
    const float2* __restrict__ gate,
    MC2QGateParams p
) {
    // Load all 16 gate elements into registers (4x4 complex = 32 floats)
    float2 g00 = gate[0],  g01 = gate[1],  g02 = gate[2],  g03 = gate[3];
    float2 g10 = gate[4],  g11 = gate[5],  g12 = gate[6],  g13 = gate[7];
    float2 g20 = gate[8],  g21 = gate[9],  g22 = gate[10], g23 = gate[11];
    float2 g30 = gate[12], g31 = gate[13], g32 = gate[14], g33 = gate[15];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= p.num_groups) return;

    // Insert zeros at all k+2 affected positions to get base index
    int base = insert_zeros(tid, p.sorted_qubits, p.n_affected);

    // Set all control bits to 1; both targets remain 0
    int i00 = base | p.control_mask;
    int i01 = i00 | (1 << p.target0);
    int i10 = i00 | (1 << p.target1);
    int i11 = i01 | (1 << p.target1);

    // Load 4 amplitudes
    float2 a00 = state[i00];
    float2 a01 = state[i01];
    float2 a10 = state[i10];
    float2 a11 = state[i11];

    // Apply 4x4 gate (Kahan-compensated)
    state[i00] = cmac4_mc2(g00, a00, g01, a01, g02, a10, g03, a11);
    state[i01] = cmac4_mc2(g10, a00, g11, a01, g12, a10, g13, a11);
    state[i10] = cmac4_mc2(g20, a00, g21, a01, g22, a10, g23, a11);
    state[i11] = cmac4_mc2(g30, a00, g31, a01, g32, a10, g33, a11);
}


// Python-facing wrapper
torch::Tensor apply_mc2qgate_cuda(
    torch::Tensor state,        // [2^n] complex64
    torch::Tensor gate,         // [4, 4] complex64
    torch::Tensor controls_t,   // [k] int64, control qubit indices (CPU)
    int64_t target0,
    int64_t target1
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(gate.dtype() == torch::kComplexFloat, "gate must be complex64");
    TORCH_CHECK(gate.size(0) == 4 && gate.size(1) == 4, "gate must be 4x4");
    TORCH_CHECK(target0 != target1, "target0 and target1 must differ");

    int k = (int)controls_t.size(0);
    TORCH_CHECK(k >= 0 && k <= 7, "controls must have 0-7 elements");

    int n_amplitudes = (int)state.size(0);
    int t0 = (int)std::min(target0, target1);
    int t1 = (int)std::max(target0, target1);

    // Build params on CPU
    MC2QGateParams p;
    p.n_affected = k + 2;
    p.target0 = t0;
    p.target1 = t1;
    p.control_mask = 0;
    p.num_groups = n_amplitudes >> p.n_affected;  // 2^(n - k - 2)

    // Collect all qubit indices: controls + 2 targets
    int all_qubits[10];
    auto ctrl_cpu = controls_t.to(torch::kInt32);
    auto* ctrl_ptr = ctrl_cpu.data_ptr<int>();
    for (int i = 0; i < k; i++) {
        all_qubits[i] = ctrl_ptr[i];
        p.control_mask |= (1 << ctrl_ptr[i]);
        TORCH_CHECK(ctrl_ptr[i] != t0 && ctrl_ptr[i] != t1,
                    "target qubits must not appear in controls");
    }
    all_qubits[k] = t0;
    all_qubits[k + 1] = t1;

    // Sort ascending (insertion sort for small k)
    for (int i = 1; i < p.n_affected; i++) {
        int val = all_qubits[i];
        int j = i - 1;
        while (j >= 0 && all_qubits[j] > val) {
            all_qubits[j + 1] = all_qubits[j];
            j--;
        }
        all_qubits[j + 1] = val;
    }
    for (int i = 0; i < p.n_affected; i++)
        p.sorted_qubits[i] = all_qubits[i];

    auto gate_contig = gate.contiguous();
    auto* gate_ptr = reinterpret_cast<float2*>(gate_contig.data_ptr());

    int threads = 256;
    int blocks = (p.num_groups + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    apply_mc2qgate_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(state.data_ptr()),
        gate_ptr,
        p
    );

    return state;
}
