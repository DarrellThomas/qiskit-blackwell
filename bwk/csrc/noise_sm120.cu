// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Quantum noise channel kernels for sm_120a (Phase 11).
//
// Implements stochastic noise channels for state vector simulation:
//   1. Depolarizing: with probability p, apply random Pauli {X, Y, Z}
//   2. Amplitude damping: T1 decay — E0=[[1,0],[0,√(1-γ)]], E1=[[0,√γ],[0,0]]
//   3. Dephasing: T2 phase decoherence — diagonal scaling
//
// All channels preserve trace (||ψ||² = 1) after renormalization.
// Two-pass design: probability reduction + conditional application.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ─── Depolarizing Channel ─────────────────────────────────────────────────
//
// Fast path: no Kraus decomposition needed.
// With probability (1-p): identity (do nothing)
// With probability p/3 each: apply Pauli X, Y, or Z
//
// Pauli X on qubit t: swap state[i] and state[i ^ (1<<t)]
// Pauli Y on qubit t: swap with phase: i→-i·state[j], j→i·state[i]
// Pauli Z on qubit t: negate state[i] where bit t = 1
//
// The random_val input selects which operation:
//   [0, 1-p)       → identity
//   [1-p, 1-2p/3)  → Pauli X
//   [1-2p/3, 1-p/3)→ Pauli Y
//   [1-p/3, 1)     → Pauli Z

// Pauli X kernel: swap pairs at stride 2^t
__global__ void __launch_bounds__(256)
pauli_x_sm120(
    float2* __restrict__ state,
    const int target_qubit,
    const int n_amplitudes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_amplitudes >> 1;
    if (tid >= n_pairs) return;

    int low_mask = (1 << target_qubit) - 1;
    int lo = tid & low_mask;
    int hi = tid >> target_qubit;
    int i0 = (hi << (target_qubit + 1)) | lo;
    int i1 = i0 | (1 << target_qubit);

    float2 tmp = state[i0];
    state[i0] = state[i1];
    state[i1] = tmp;
}

// Pauli Y kernel: swap with phase factors (Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩)
// new[i0] = -i * state[i1] = (imag, -real) of state[i1]
// new[i1] =  i * state[i0] = (-imag, real) of state[i0]
__global__ void __launch_bounds__(256)
pauli_y_sm120(
    float2* __restrict__ state,
    const int target_qubit,
    const int n_amplitudes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_amplitudes >> 1;
    if (tid >= n_pairs) return;

    int low_mask = (1 << target_qubit) - 1;
    int lo = tid & low_mask;
    int hi = tid >> target_qubit;
    int i0 = (hi << (target_qubit + 1)) | lo;
    int i1 = i0 | (1 << target_qubit);

    float2 a0 = state[i0];
    float2 a1 = state[i1];
    // -i * a1 = (a1.y, -a1.x)
    state[i0] = make_float2(a1.y, -a1.x);
    // i * a0 = (-a0.y, a0.x)
    state[i1] = make_float2(-a0.y, a0.x);
}

// Pauli Z kernel: negate amplitudes where bit t = 1
__global__ void __launch_bounds__(256)
pauli_z_sm120(
    float2* __restrict__ state,
    const int target_qubit,
    const int n_amplitudes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_amplitudes) return;

    if ((tid >> target_qubit) & 1) {
        float2 a = state[tid];
        state[tid] = make_float2(-a.x, -a.y);
    }
}

// ─── Amplitude Damping Channel ────────────────────────────────────────────
//
// E0 = [[1, 0], [0, √(1-γ)]]   (no decay)
// E1 = [[0, √γ], [0, 0]]       (decay: |1⟩ → |0⟩)
//
// Probability of decay branch: p1 = γ * Σ|state[i]|² for i where bit t=1
//
// Pass 1: Compute p1 via streaming reduction (same as measure_qubit pass 1)
// Pass 2: Apply selected Kraus operator:
//   If no decay (u >= p1): state[i] *= √(1-γ) for bit t=1, renormalize by 1/√(1-p1)
//   If decay (u < p1):     state[i0] += √γ * state[i1], state[i1] = 0, renormalize by 1/√p1

// Pass 1: Compute probability of decay (sum |state[i]|^2 for bit t=1, multiply by gamma)
__global__ void __launch_bounds__(256)
amp_damp_prob_sm120(
    const float2* __restrict__ state,
    float* __restrict__ p_decay,    // output: scalar
    const int target_qubit,
    const float gamma,
    const int n_amplitudes
) {
    __shared__ float sdata[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Grid-stride accumulation
    for (int i = tid; i < n_amplitudes; i += gridDim.x * blockDim.x) {
        if ((i >> target_qubit) & 1) {
            float2 a = state[i];
            sum += a.x * a.x + a.y * a.y;
        }
    }

    // Block reduction
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(p_decay, sdata[0] * gamma);
}

// Pass 2a: No-decay branch — scale |1⟩ amplitudes by √(1-γ), renormalize
__global__ void __launch_bounds__(256)
amp_damp_no_decay_sm120(
    float2* __restrict__ state,
    const int target_qubit,
    const float scale_1,      // √(1-γ)
    const float renorm,       // 1/√(1-p1)
    const int n_amplitudes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_amplitudes) return;

    float2 a = state[tid];
    if ((tid >> target_qubit) & 1) {
        // |1⟩ component: multiply by √(1-γ) and renormalize
        float s = scale_1 * renorm;
        state[tid] = make_float2(a.x * s, a.y * s);
    } else {
        // |0⟩ component: just renormalize
        state[tid] = make_float2(a.x * renorm, a.y * renorm);
    }
}

// Pass 2b: Decay branch — transfer |1⟩→|0⟩, zero |1⟩, renormalize
__global__ void __launch_bounds__(256)
amp_damp_decay_sm120(
    float2* __restrict__ state,
    const int target_qubit,
    const float sqrt_gamma,
    const float renorm,       // 1/√p1
    const int n_amplitudes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_amplitudes >> 1;
    if (tid >= n_pairs) return;

    int low_mask = (1 << target_qubit) - 1;
    int lo = tid & low_mask;
    int hi = tid >> target_qubit;
    int i0 = (hi << (target_qubit + 1)) | lo;
    int i1 = i0 | (1 << target_qubit);

    // E1|ψ⟩: new[i0] = √γ * state[i1], new[i1] = 0
    float2 a1 = state[i1];
    float s = sqrt_gamma * renorm;
    float2 a0 = state[i0];
    state[i0] = make_float2(a0.x * renorm + a1.x * s,
                             a0.y * renorm + a1.y * s);
    state[i1] = make_float2(0.0f, 0.0f);
}

// ─── Dephasing (Phase Damping) Channel ────────────────────────────────────
//
// E0 = [[1, 0], [0, √(1-λ)]]   (no phase error)
// E1 = [[0, 0], [0, √λ]]       (phase error)
//
// Same structure as amplitude damping but E1 doesn't transfer amplitude.
// Probability of phase error: p1 = λ * Σ|state[i]|² for bit t=1
//
// No-error branch: scale |1⟩ by √(1-λ), renormalize
// Error branch: zero |0⟩ components (only |1⟩ survives), renormalize

// Pass 2b: Dephasing error branch — zero |0⟩ components, renormalize
__global__ void __launch_bounds__(256)
dephase_error_sm120(
    float2* __restrict__ state,
    const int target_qubit,
    const float sqrt_lambda,
    const float renorm,       // 1/√p1
    const int n_amplitudes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_amplitudes) return;

    if ((tid >> target_qubit) & 1) {
        // |1⟩: scale by √λ * renorm
        float s = sqrt_lambda * renorm;
        float2 a = state[tid];
        state[tid] = make_float2(a.x * s, a.y * s);
    } else {
        // |0⟩: zero out
        state[tid] = make_float2(0.0f, 0.0f);
    }
}


// ─── Python-facing wrappers ───────────────────────────────────────────────

// Depolarizing channel: apply random Pauli based on random_val and probability p
int apply_depolarizing_cuda(
    torch::Tensor state,
    int64_t target_qubit,
    double p,               // error probability
    double random_val       // pre-generated uniform [0, 1)
) {
    int n = (int)state.size(0);
    auto* state_ptr = reinterpret_cast<float2*>(state.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;

    // Determine which Pauli (or identity) to apply
    // [0, 1-p) → identity (return 0)
    // [1-p, 1-2p/3) → X (return 1)
    // [1-2p/3, 1-p/3) → Y (return 2)
    // [1-p/3, 1) → Z (return 3)
    if (random_val < 1.0 - p) {
        return 0;  // identity
    }

    double remaining = random_val - (1.0 - p);
    double third = p / 3.0;

    int n_pairs = n >> 1;
    int blocks_pair = (n_pairs + threads - 1) / threads;
    int blocks_full = (n + threads - 1) / threads;

    if (remaining < third) {
        // Pauli X
        pauli_x_sm120<<<blocks_pair, threads, 0, stream>>>(
            state_ptr, (int)target_qubit, n);
        return 1;
    } else if (remaining < 2 * third) {
        // Pauli Y
        pauli_y_sm120<<<blocks_pair, threads, 0, stream>>>(
            state_ptr, (int)target_qubit, n);
        return 2;
    } else {
        // Pauli Z
        pauli_z_sm120<<<blocks_full, threads, 0, stream>>>(
            state_ptr, (int)target_qubit, n);
        return 3;
    }
}

// Amplitude damping channel
int apply_amplitude_damping_cuda(
    torch::Tensor state,
    int64_t target_qubit,
    double gamma,           // decay probability (1 - exp(-t/T1))
    double random_val       // pre-generated uniform [0, 1)
) {
    int n = (int)state.size(0);
    auto* state_ptr = reinterpret_cast<float2*>(state.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;

    // Pass 1: compute probability of decay
    auto p_decay_t = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(state.device()));
    float* p_ptr = p_decay_t.data_ptr<float>();

    int blocks_reduce = std::min(1024, (n + threads - 1) / threads);
    amp_damp_prob_sm120<<<blocks_reduce, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(state.data_ptr()),
        p_ptr, (int)target_qubit, (float)gamma, n);

    // Read p1 back to CPU
    float p1;
    cudaMemcpyAsync(&p1, p_ptr, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Pass 2: apply selected branch
    int blocks_full = (n + threads - 1) / threads;
    int n_pairs = n >> 1;
    int blocks_pair = (n_pairs + threads - 1) / threads;

    if (random_val >= p1) {
        // No decay: E0
        float scale_1 = sqrtf(1.0f - (float)gamma);
        float renorm = 1.0f / sqrtf(1.0f - p1);
        amp_damp_no_decay_sm120<<<blocks_full, threads, 0, stream>>>(
            state_ptr, (int)target_qubit, scale_1, renorm, n);
        return 0;
    } else {
        // Decay: E1
        float sqrt_gamma = sqrtf((float)gamma);
        float renorm = 1.0f / sqrtf(p1);
        amp_damp_decay_sm120<<<blocks_pair, threads, 0, stream>>>(
            state_ptr, (int)target_qubit, sqrt_gamma, renorm, n);
        return 1;
    }
}

// Dephasing channel
int apply_dephasing_cuda(
    torch::Tensor state,
    int64_t target_qubit,
    double lambda,          // dephasing probability (1 - exp(-t/T2))
    double random_val
) {
    int n = (int)state.size(0);
    auto* state_ptr = reinterpret_cast<float2*>(state.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;

    // Pass 1: same as amplitude damping (probability from |1⟩ amplitudes)
    auto p_err_t = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(state.device()));
    float* p_ptr = p_err_t.data_ptr<float>();

    int blocks_reduce = std::min(1024, (n + threads - 1) / threads);
    amp_damp_prob_sm120<<<blocks_reduce, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(state.data_ptr()),
        p_ptr, (int)target_qubit, (float)lambda, n);

    float p1;
    cudaMemcpyAsync(&p1, p_ptr, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int blocks_full = (n + threads - 1) / threads;

    if (random_val >= p1) {
        // No error: E0 (same as amplitude damping no-decay)
        float scale_1 = sqrtf(1.0f - (float)lambda);
        float renorm = 1.0f / sqrtf(1.0f - p1);
        amp_damp_no_decay_sm120<<<blocks_full, threads, 0, stream>>>(
            state_ptr, (int)target_qubit, scale_1, renorm, n);
        return 0;
    } else {
        // Phase error: E1 (zero |0⟩, keep |1⟩ scaled)
        float sqrt_lambda = sqrtf((float)lambda);
        float renorm = 1.0f / sqrtf(p1);
        dephase_error_sm120<<<blocks_full, threads, 0, stream>>>(
            state_ptr, (int)target_qubit, sqrt_lambda, renorm, n);
        return 1;
    }
}
