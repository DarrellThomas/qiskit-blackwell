// Copyright (c) 2026 Darrell Thomas. MIT License.
// QV-4: Batched Quantum Volume 4-qubit circuit simulation for RTX 5090 (sm_120a).
//
// Each thread simulates one complete QV-4 circuit (4 layers, 2 SU(4) gates/layer).
// State vector (16 complex amplitudes) lives in shared memory with bank-conflict-free
// stride. Gates loaded from global memory. Output: 16 probabilities per circuit.

#include <torch/extension.h>
#include <cuda_runtime.h>

// 6 possible qubit pairs for 4 qubits, each with 4 groups × 4 indices.
// For pair (qa, qb), groups iterate over the 2^(n-2)=4 fixed-bit patterns
// of the other qubits. Within each group, 4 indices correspond to the
// 4 computational basis states of the (qa, qb) subspace.
__constant__ int c_qv4_idx_table[6][16] = {
    // pair 0: qubits (0,1)
    { 0, 1, 2, 3,   4, 5, 6, 7,   8, 9,10,11,  12,13,14,15},
    // pair 1: qubits (0,2)
    { 0, 1, 4, 5,   2, 3, 6, 7,   8, 9,12,13,  10,11,14,15},
    // pair 2: qubits (0,3)
    { 0, 1, 8, 9,   2, 3,10,11,   4, 5,12,13,   6, 7,14,15},
    // pair 3: qubits (1,2)
    { 0, 2, 4, 6,   1, 3, 5, 7,   8,10,12,14,   9,11,13,15},
    // pair 4: qubits (1,3)
    { 0, 2, 8,10,   1, 3, 9,11,   4, 6,12,14,   5, 7,13,15},
    // pair 5: qubits (2,3)
    { 0, 4, 8,12,   1, 5, 9,13,   2, 6,10,14,   3, 7,11,15},
};

// Shared memory stride per thread: 33 floats avoids bank conflicts.
// Thread t accessing sre[k] hits bank (t*33+k)%32 = (t+k)%32 — unique per warp.
#define QV4_SMEM_STRIDE 33

__global__ void qv4_simulate_kernel(
    const float* __restrict__ gate_data,  // [n_circuits, n_gates, 32]: 16 re + 16 im
    const int* __restrict__ pair_ids,     // [n_circuits, n_gates]: qubit pair index 0-5
    float* __restrict__ probs_out,        // [n_circuits, 16]: output probabilities
    int n_circuits,
    int n_gates
) {
    extern __shared__ float smem[];

    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_circuits) return;

    float* sre = smem + threadIdx.x * QV4_SMEM_STRIDE;
    float* sim = smem + blockDim.x * QV4_SMEM_STRIDE + threadIdx.x * QV4_SMEM_STRIDE;

    // Initialize to |0000⟩
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        sre[i] = (i == 0) ? 1.0f : 0.0f;
        sim[i] = 0.0f;
    }

    // Apply each gate
    for (int g = 0; g < n_gates; g++) {
        // Load 4×4 complex gate matrix from global memory
        const float* gptr = gate_data + (static_cast<long long>(cid) * n_gates + g) * 32;
        float gre[16], gim[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            gre[k] = gptr[k];
            gim[k] = gptr[16 + k];
        }

        int pid = pair_ids[cid * n_gates + g];
        const int* idx = c_qv4_idx_table[pid];

        // Apply unitary to each of the 4 subspace groups
        #pragma unroll
        for (int grp = 0; grp < 4; grp++) {
            int i0 = idx[grp * 4 + 0];
            int i1 = idx[grp * 4 + 1];
            int i2 = idx[grp * 4 + 2];
            int i3 = idx[grp * 4 + 3];

            // Load amplitudes before overwriting
            float r[4] = {sre[i0], sre[i1], sre[i2], sre[i3]};
            float m[4] = {sim[i0], sim[i1], sim[i2], sim[i3]};

            // Complex matrix-vector multiply
            float nr[4], ni[4];
            #pragma unroll
            for (int row = 0; row < 4; row++) {
                float acc_re = 0.0f, acc_im = 0.0f;
                #pragma unroll
                for (int col = 0; col < 4; col++) {
                    float gr = gre[row * 4 + col];
                    float gi = gim[row * 4 + col];
                    acc_re += gr * r[col] - gi * m[col];
                    acc_im += gr * m[col] + gi * r[col];
                }
                nr[row] = acc_re;
                ni[row] = acc_im;
            }

            // Write back
            sre[i0] = nr[0]; sre[i1] = nr[1]; sre[i2] = nr[2]; sre[i3] = nr[3];
            sim[i0] = ni[0]; sim[i1] = ni[1]; sim[i2] = ni[2]; sim[i3] = ni[3];
        }
    }

    // Output probabilities: |amplitude|^2
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float r = sre[i], m = sim[i];
        probs_out[cid * 16 + i] = r * r + m * m;
    }
}

// PyTorch binding
torch::Tensor qv4_simulate(
    torch::Tensor gate_data,
    torch::Tensor pair_ids,
    int n_circuits,
    int n_gates
) {
    TORCH_CHECK(gate_data.is_cuda(), "gate_data must be CUDA tensor");
    TORCH_CHECK(pair_ids.is_cuda(), "pair_ids must be CUDA tensor");
    TORCH_CHECK(gate_data.dtype() == torch::kFloat32, "gate_data must be float32");
    TORCH_CHECK(pair_ids.dtype() == torch::kInt32, "pair_ids must be int32");

    auto probs = torch::empty({n_circuits, 16}, gate_data.options());

    const int threads = 256;
    const int blocks = (n_circuits + threads - 1) / threads;
    const int smem_bytes = 2 * threads * QV4_SMEM_STRIDE * sizeof(float);

    // Must opt in to >48 KB dynamic shared memory
    cudaFuncSetAttribute(qv4_simulate_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    qv4_simulate_kernel<<<blocks, threads, smem_bytes>>>(
        gate_data.data_ptr<float>(),
        pair_ids.data_ptr<int>(),
        probs.data_ptr<float>(),
        n_circuits,
        n_gates
    );

    return probs;
}

// Binding registered in apply_gate_sm120.cu PYBIND11_MODULE block.
