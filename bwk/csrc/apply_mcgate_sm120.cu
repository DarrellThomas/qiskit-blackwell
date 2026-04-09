// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Multi-controlled single-qubit gate kernel for sm_120a.
//
// Applies a 2x2 unitary gate conditioned on k control qubits all being |1>.
// For each pair (i0, i1) where all control bits are 1 and target bit is (0, 1):
//   new[i0] = gate[0]*state[i0] + gate[1]*state[i1]
//   new[i1] = gate[2]*state[i0] + gate[3]*state[i1]
//
// Examples:
//   Toffoli (CCX): gate=[[0,1],[1,0]] (Pauli-X), controls=[c0,c1], target=t
//   CCZ:           gate=[[1,0],[0,-1]] (Pauli-Z), controls=[c0,c1], target=t
//   CX (CNOT):     gate=[[0,1],[1,0]], controls=[c0], target=t
//
// Uses "insert zeros" indexing: only 2^(n-k-1) threads, all doing work.
// No wasted threads (no check-and-skip).
//
// Memory characteristics:
//   Traffic: 2^(n-k-1) pairs × 32B/pair = 2^(n-k+4) bytes
//   For Toffoli (k=2, n=20): 2^17 pairs × 32B = 4 MB (vs 8 MB for 1Q gate)
//   This kernel is MEMORY-BANDWIDTH-BOUND. No tensor cores.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float2 cmul_mc(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cadd_mc(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// Kernel parameter struct — passed by value (fits in registers)
struct MCGateParams {
    int sorted_qubits[8];  // sorted qubit positions (controls + target)
    int n_affected;        // how many sorted_qubits are valid (k+1)
    int target_qubit;      // which qubit is the target
    int control_mask;      // OR of (1 << c[j]) for all control qubits
    int num_pairs;         // 2^(n - n_affected)
};

// Insert zeros at sorted positions sorted_qubits[0..n_affected-1] in the output.
// Each insertion shifts bits above the position up by one, creating a 0 there.
__device__ __forceinline__ int insert_zeros_mc(int g, const int* q, int n) {
    int r = g;
    for (int j = 0; j < n; j++) {
        int lo = r & ((1 << q[j]) - 1);
        int hi = r >> q[j];
        r = (hi << (q[j] + 1)) | lo;
    }
    return r;
}

__global__ void __launch_bounds__(256)
apply_mcgate_sm120(
    float2* __restrict__ state,
    const float2* __restrict__ gate,
    MCGateParams p
) {
    float2 g00 = gate[0], g01 = gate[1], g10 = gate[2], g11 = gate[3];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= p.num_pairs) return;

    // Insert zeros at all k+1 affected positions to get base index
    // (all affected bits = 0, all other bits from tid)
    int base = insert_zeros_mc(tid, p.sorted_qubits, p.n_affected);

    // Set all control bits to 1; target remains 0
    int i0 = base | p.control_mask;
    int i1 = i0 | (1 << p.target_qubit);

    float2 a0 = state[i0];
    float2 a1 = state[i1];

    state[i0] = cadd_mc(cmul_mc(g00, a0), cmul_mc(g01, a1));
    state[i1] = cadd_mc(cmul_mc(g10, a0), cmul_mc(g11, a1));
}


// Python-facing wrapper
torch::Tensor apply_mcgate_cuda(
    torch::Tensor state,        // [2^n] complex64
    torch::Tensor gate,         // [2, 2] complex64
    torch::Tensor controls_t,   // [k] int64, control qubit indices
    int64_t target_qubit
) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(state.dtype() == torch::kComplexFloat, "state must be complex64");
    TORCH_CHECK(gate.dtype() == torch::kComplexFloat, "gate must be complex64");
    TORCH_CHECK(gate.size(0) == 2 && gate.size(1) == 2, "gate must be 2x2");

    int k = (int)controls_t.size(0);
    TORCH_CHECK(k >= 0 && k <= 7, "controls must have 0-7 elements");

    int n_amplitudes = (int)state.size(0);

    // Build sorted_qubits and control_mask on CPU
    MCGateParams p;
    p.n_affected = k + 1;
    p.target_qubit = (int)target_qubit;
    p.control_mask = 0;

    // Collect all qubit indices: controls + target
    int all_qubits[8];
    // controls_t is already on CPU (guaranteed by Python wrapper)
    auto ctrl_cpu = controls_t.to(torch::kInt32);
    auto* ctrl_ptr = ctrl_cpu.data_ptr<int>();
    for (int i = 0; i < k; i++) {
        all_qubits[i] = ctrl_ptr[i];
        p.control_mask |= (1 << ctrl_ptr[i]);
        TORCH_CHECK(ctrl_ptr[i] != (int)target_qubit,
                    "target_qubit must not appear in controls");
    }
    all_qubits[k] = (int)target_qubit;

    // Sort all_qubits ascending (insertion sort for small k)
    for (int i = 1; i < p.n_affected; i++) {
        int key = all_qubits[i];
        int j = i - 1;
        while (j >= 0 && all_qubits[j] > key) {
            all_qubits[j + 1] = all_qubits[j];
            j--;
        }
        all_qubits[j + 1] = key;
    }

    for (int i = 0; i < p.n_affected; i++) p.sorted_qubits[i] = all_qubits[i];
    for (int i = p.n_affected; i < 8; i++) p.sorted_qubits[i] = 0;

    p.num_pairs = n_amplitudes >> p.n_affected;

    auto gate_c = gate.contiguous();
    auto* gate_ptr = reinterpret_cast<float2*>(gate_c.data_ptr());

    int threads = 256;
    int blocks = (p.num_pairs + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    apply_mcgate_sm120<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(state.data_ptr()), gate_ptr, p);

    return state;
}


// Binding is registered in apply_gate_sm120.cu via forward declaration.
