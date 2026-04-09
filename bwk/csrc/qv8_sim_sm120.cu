// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// QV-8: Fused quantum volume circuit simulator for 8 qubits.
// Each thread block simulates one random circuit instance.
// 256 threads per block, one per amplitude of the 2^8 statevector.
// State lives in shared memory; all gates applied in a single kernel launch.
//
// Target: RTX 5090 (sm_120a, consumer Blackwell)

#include <cuda_runtime.h>
#include <torch/extension.h>

#define N_QUBITS 8
#define STATE_SIZE 256  // 2^8

// Complex multiply-accumulate: acc += a * b
__device__ __forceinline__ float2 cmac(float2 acc, float2 a, float2 b) {
    acc.x += a.x * b.x - a.y * b.y;
    acc.y += a.x * b.y + a.y * b.x;
    return acc;
}

// Batch-simulate QV-8 circuits.
//
// gate_matrices: [num_circuits, num_gates_per_circuit, 4, 4, 2]  (real/imag last)
//     Flattened as float2: [num_circuits * num_gates * 16]
// gate_qubits:   [num_circuits, num_gates_per_circuit, 2]
//     Packed as int: [num_circuits * num_gates * 2]
// out_probs:     [num_circuits, STATE_SIZE]   -- output probability vector
//
__global__ void qv8_batch_simulate_kernel(
    const float2* __restrict__ gate_matrices,  // [C * G * 16]
    const int*    __restrict__ gate_qubits,    // [C * G * 2]
    int           num_gates,
    float*        __restrict__ out_probs,      // [C * 256]
    int           num_circuits
) {
    const int cid = blockIdx.x;
    if (cid >= num_circuits) return;
    const int tid = threadIdx.x;  // 0..255

    // ── shared state vector ─────────────────────────────────────────────
    __shared__ float2 state[STATE_SIZE];

    // Initialize |00000000⟩
    state[tid] = (tid == 0) ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 0.0f);
    __syncthreads();

    // Pointers into this circuit's gate data
    const float2* my_gates  = gate_matrices + (long long)cid * num_gates * 16;
    const int*    my_qubits = gate_qubits   + (long long)cid * num_gates * 2;

    // ── apply gates ─────────────────────────────────────────────────────
    for (int g = 0; g < num_gates; g++) {
        const int q0 = my_qubits[g * 2 + 0];
        const int q1 = my_qubits[g * 2 + 1];
        const int mask_q0 = 1 << q0;
        const int mask_q1 = 1 << q1;

        // Which row of the 4×4 gate does this thread use?
        const int bit0 = (tid >> q0) & 1;
        const int bit1 = (tid >> q1) & 1;
        const int row  = (bit1 << 1) | bit0;

        // Four amplitude indices sharing this gate group
        const int base = tid & ~(mask_q0 | mask_q1);
        const float2 a0 = state[base];
        const float2 a1 = state[base | mask_q0];
        const float2 a2 = state[base | mask_q1];
        const float2 a3 = state[base | mask_q0 | mask_q1];

        // Load the row of the gate matrix from global (L1-cached)
        const float2* grow = my_gates + g * 16 + row * 4;
        const float2 g0 = grow[0];
        const float2 g1 = grow[1];
        const float2 g2 = grow[2];
        const float2 g3 = grow[3];

        // Complex matrix-vector product for this row
        float2 nv = make_float2(0.0f, 0.0f);
        nv = cmac(nv, g0, a0);
        nv = cmac(nv, g1, a1);
        nv = cmac(nv, g2, a2);
        nv = cmac(nv, g3, a3);

        __syncthreads();
        state[tid] = nv;
        __syncthreads();
    }

    // ── write probability vector ────────────────────────────────────────
    float2 amp = state[tid];
    out_probs[(long long)cid * STATE_SIZE + tid] = amp.x * amp.x + amp.y * amp.y;
}

// ── Torch binding ───────────────────────────────────────────────────────

torch::Tensor qv8_simulate(
    torch::Tensor gate_matrices,  // [C, G, 4, 4, 2] float32
    torch::Tensor gate_qubits,    // [C, G, 2] int32
    int64_t num_circuits
) {
    TORCH_CHECK(gate_matrices.is_cuda(), "gate_matrices must be CUDA");
    TORCH_CHECK(gate_qubits.is_cuda(),   "gate_qubits must be CUDA");
    TORCH_CHECK(gate_matrices.scalar_type() == torch::kFloat32, "float32 required");
    TORCH_CHECK(gate_qubits.scalar_type()   == torch::kInt32,   "int32 required");

    const int num_gates = gate_matrices.size(1);

    // Ensure contiguous
    gate_matrices = gate_matrices.contiguous();
    gate_qubits   = gate_qubits.contiguous();

    // Output: probability vectors [C, 256]
    auto out_probs = torch::empty({num_circuits, STATE_SIZE},
                                  torch::dtype(torch::kFloat32).device(gate_matrices.device()));

    // Launch: one block per circuit, 256 threads per block
    qv8_batch_simulate_kernel<<<num_circuits, STATE_SIZE>>>(
        reinterpret_cast<const float2*>(gate_matrices.data_ptr<float>()),
        gate_qubits.data_ptr<int>(),
        num_gates,
        out_probs.data_ptr<float>(),
        num_circuits
    );

    return out_probs;
}

// Binding registered in apply_gate_sm120.cu PYBIND11_MODULE block.
