// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Quantum state vector: two-qubit gate application kernel for sm_120a.
//
// Applies a 4x4 complex unitary gate to (qubit0, qubit1) of a state vector.
// For each index i where bits qubit0 AND qubit1 are both 0:
//   i00 = i
//   i01 = i | (1 << qubit0)
//   i10 = i | (1 << qubit1)
//   i11 = i | (1 << qubit0) | (1 << qubit1)
//
//   new[i00] = gate[0]*s[i00] + gate[1]*s[i01] + gate[2]*s[i10] + gate[3]*s[i11]
//   new[i01] = gate[4]*s[i00] + gate[5]*s[i01] + gate[6]*s[i10] + gate[7]*s[i11]
//   new[i10] = gate[8]*s[i00] + gate[9]*s[i01] + gate[10]*s[i10] + gate[11]*s[i11]
//   new[i11] = gate[12]*s[i00] + gate[13]*s[i01] + gate[14]*s[i10] + gate[15]*s[i11]
//
// This kernel is MEMORY-BANDWIDTH-BOUND. No tensor cores.
// Gate matrix (128 bytes) fits in 32 registers. Same total traffic as single-qubit.
// Optimization is purely about memory access patterns, coalescing, and cache exploitation.

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

// v1: Specialized path for q0=0 with float4 vectorization.
// When q0=0: i01=i00+1 and i11=i10+1, so pairs are adjacent.
// Two float4 loads (16B each) replace four float2 loads (8B each).
__global__ void __launch_bounds__(256)
apply_gate2q_q0zero_sm120(
    float2* __restrict__ state,
    const float2* __restrict__ gate,
    const int qubit1,                 // higher qubit index (q0=0 implied)
    const int num_groups
) {
    float2 g00 = gate[0],  g01 = gate[1],  g02 = gate[2],  g03 = gate[3];
    float2 g10 = gate[4],  g11 = gate[5],  g12 = gate[6],  g13 = gate[7];
    float2 g20 = gate[8],  g21 = gate[9],  g22 = gate[10], g23 = gate[11];
    float2 g30 = gate[12], g31 = gate[13], g32 = gate[14], g33 = gate[15];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_groups) return;

    // q0=0: insert zero at bit 0 → i00 = tid*2 with gap at qubit1
    // Step 1: insert zero at bit 0
    int tmp = tid << 1;  // insert 0 at position 0
    // Step 2: insert zero at qubit1
    int lo = tmp & ((1 << qubit1) - 1);
    int hi = tmp >> qubit1;
    int i00 = (hi << (qubit1 + 1)) | lo;

    // i01 = i00 + 1, i10 = i00 + stride1, i11 = i10 + 1
    int stride1 = 1 << qubit1;
    int i10 = i00 | stride1;

    // float4 load: gets (state[i00], state[i00+1]) as one 16B transaction
    float4 pair_lo = *reinterpret_cast<float4*>(&state[i00]);   // a00, a01
    float4 pair_hi = *reinterpret_cast<float4*>(&state[i10]);   // a10, a11

    float2 a00 = make_float2(pair_lo.x, pair_lo.y);
    float2 a01 = make_float2(pair_lo.z, pair_lo.w);
    float2 a10 = make_float2(pair_hi.x, pair_hi.y);
    float2 a11 = make_float2(pair_hi.z, pair_hi.w);

    float2 new00 = cadd(cadd(cmul(g00, a00), cmul(g01, a01)),
                        cadd(cmul(g02, a10), cmul(g03, a11)));
    float2 new01 = cadd(cadd(cmul(g10, a00), cmul(g11, a01)),
                        cadd(cmul(g12, a10), cmul(g13, a11)));
    float2 new10 = cadd(cadd(cmul(g20, a00), cmul(g21, a01)),
                        cadd(cmul(g22, a10), cmul(g23, a11)));
    float2 new11 = cadd(cadd(cmul(g30, a00), cmul(g31, a01)),
                        cadd(cmul(g32, a10), cmul(g33, a11)));

    // float4 store: writes (new00, new01) as one 16B transaction
    *reinterpret_cast<float4*>(&state[i00]) = make_float4(new00.x, new00.y, new01.x, new01.y);
    *reinterpret_cast<float4*>(&state[i10]) = make_float4(new10.x, new10.y, new11.x, new11.y);
}

// v3: Warp-cooperative loading — pair even/odd threads to halve loads per thread.
// Even thread loads a00,a01; odd loads a10,a11. Shuffle to exchange.
// Even stores new00,new01; odd stores new10,new11.
// 2 loads + 4 shuffles + 2 stores per thread (vs 4 loads + 4 stores in v1).
__global__ void __launch_bounds__(256)
apply_gate2q_sm120(
    float2* __restrict__ state,
    const float2* __restrict__ gate,
    const int qubit0,
    const int qubit1,
    const int num_groups
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int group_id = global_tid >> 1;
    int is_odd = global_tid & 1;

    // Each thread only loads the 2 gate rows it needs (16 floats, not 32).
    // Even: rows 0,1. Odd: rows 2,3.
    const float2* my_gate = gate + is_odd * 8;
    float2 gr0 = my_gate[0], gr1 = my_gate[1], gr2 = my_gate[2], gr3 = my_gate[3];
    float2 gs0 = my_gate[4], gs1 = my_gate[5], gs2 = my_gate[6], gs3 = my_gate[7];

    if (group_id >= num_groups) return;

    // Compute i00 for this group
    int lo = group_id & ((1 << qubit0) - 1);
    int hi = group_id >> qubit0;
    int tmp = (hi << (qubit0 + 1)) | lo;
    lo = tmp & ((1 << qubit1) - 1);
    hi = tmp >> qubit1;
    int i00 = (hi << (qubit1 + 1)) | lo;

    int stride0 = 1 << qubit0;
    int stride1 = 1 << qubit1;

    // Each thread loads 2 amplitudes based on even/odd.
    // When q0=0: adjacent pairs allow float4 vectorized loads (16B vs 2×8B).
    float2 my0, my1;
    if (qubit0 == 0) {
        // q0=0: i01=i00+1, i11=i10+1 — pairs are adjacent in memory
        float4 pair;
        if (is_odd == 0) {
            pair = *reinterpret_cast<float4*>(&state[i00]);         // a00, a01
        } else {
            pair = *reinterpret_cast<float4*>(&state[i00 | stride1]); // a10, a11
        }
        my0 = make_float2(pair.x, pair.y);
        my1 = make_float2(pair.z, pair.w);
    } else {
        if (is_odd == 0) {
            my0 = state[i00];                // a00
            my1 = state[i00 | stride0];      // a01
        } else {
            my0 = state[i00 | stride1];              // a10
            my1 = state[i00 | stride1 | stride0];    // a11
        }
    }

    // Shuffle to get partner's values (XOR with 1 swaps even<->odd)
    float2 partner0, partner1;
    partner0.x = __shfl_xor_sync(0xFFFFFFFF, my0.x, 1);
    partner0.y = __shfl_xor_sync(0xFFFFFFFF, my0.y, 1);
    partner1.x = __shfl_xor_sync(0xFFFFFFFF, my1.x, 1);
    partner1.y = __shfl_xor_sync(0xFFFFFFFF, my1.y, 1);

    // Even thread: a00=my0, a01=my1, a10=partner0, a11=partner1 → compute+store new00, new01
    // Odd thread: a00=partner0, a01=partner1, a10=my0, a11=my1 → compute+store new10, new11
    // Each thread computes only its 2 outputs (half the FLOPs of computing all 4).
    // Even: a00=my0, a01=my1, a10=partner0, a11=partner1
    // Odd:  a00=partner0, a01=partner1, a10=my0, a11=my1
    float2 a0, a1, a2, a3;
    if (is_odd == 0) {
        a0 = my0; a1 = my1; a2 = partner0; a3 = partner1;
    } else {
        a0 = partner0; a1 = partner1; a2 = my0; a3 = my1;
    }

    // Compute 2 outputs using this thread's 2 gate rows
    float2 out0 = cadd(cadd(cmul(gr0, a0), cmul(gr1, a1)),
                       cadd(cmul(gr2, a2), cmul(gr3, a3)));
    float2 out1 = cadd(cadd(cmul(gs0, a0), cmul(gs1, a1)),
                       cadd(cmul(gs2, a2), cmul(gs3, a3)));

    // Even stores to i00, i01; odd stores to i10, i11
    if (qubit0 == 0) {
        // float4 vectorized stores for adjacent pairs
        if (is_odd == 0) {
            *reinterpret_cast<float4*>(&state[i00]) = make_float4(out0.x, out0.y, out1.x, out1.y);
        } else {
            *reinterpret_cast<float4*>(&state[i00 | stride1]) = make_float4(out0.x, out0.y, out1.x, out1.y);
        }
    } else {
        if (is_odd == 0) {
            state[i00] = out0;
            state[i00 | stride0] = out1;
        } else {
            state[i00 | stride1] = out0;
            state[i00 | stride1 | stride0] = out1;
        }
    }
}


// Python-facing wrapper
torch::Tensor apply_gate_2q_cuda(
    torch::Tensor state,       // [2^n] complex64
    torch::Tensor gate,        // [4, 4] complex64
    int64_t qubit0,
    int64_t qubit1
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(gate.dtype() == torch::kComplexFloat, "gate must be complex64");
    TORCH_CHECK(gate.size(0) == 4 && gate.size(1) == 4, "gate must be 4x4");

    // Ensure q0 < q1
    int q0 = (int)std::min(qubit0, qubit1);
    int q1 = (int)std::max(qubit0, qubit1);
    TORCH_CHECK(q0 != q1, "qubit0 and qubit1 must differ");

    long long n_amplitudes = state.size(0);
    int num_groups = (int)(n_amplitudes / 4);

    auto gate_contig = gate.contiguous();
    auto* gate_ptr = reinterpret_cast<float2*>(gate_contig.data_ptr());

    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (q0 == 0) {
        // q0=0: float4 vectorized, 1 thread/group (best coalescing for stride0=1)
        int blocks = (num_groups + threads - 1) / threads;
        apply_gate2q_q0zero_sm120<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float2*>(state.data_ptr()),
            gate_ptr,
            q1,
            num_groups
        );
    } else {
        // General: cooperative loading, 2 threads/group (fixes low-qubit coalescing)
        int total_threads = num_groups * 2;
        int blocks = (total_threads + threads - 1) / threads;
        apply_gate2q_sm120<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float2*>(state.data_ptr()),
            gate_ptr,
            q0, q1,
            num_groups
        );
    }

    return state;
}
