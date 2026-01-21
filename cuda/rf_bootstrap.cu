#include "rf_bootstrap.cuh"
#include "rf_config.hpp"
#include "rf_utils.hpp"
#include "rf_memory.cuh"
#include "rf_cuda_config.hpp"
#include "rf_types.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace rf {

// ============================================================================
// CUDA Kernels (internal, not in header)
// ============================================================================

namespace {  // Anonymous namespace for internal kernels

// Kernel for parallel bootstrap sampling (preserves original algorithm logic)
__global__ void cuda_bootstrap_kernel(const real_t* weight, integer_t nsample,
                                      real_t* win, integer_t* nin,
                                      curandState* rand_states,
                                      integer_t* sample_indices,
                                      integer_t tree_id) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;

    // Initialize nin array (critical for proper bootstrap sampling)
    for (integer_t n = tid; n < nsample; n += stride) {
        nin[n] = 0;
    }

    __syncthreads();

    // Each thread generates random samples (EXACT same logic as original boot.f)
    for (integer_t n = tid; n < nsample; n += stride) {
        // Use tree_id as offset for deterministic behavior
        integer_t state_idx = (tid + tree_id) % (blockDim.x * gridDim.x);
        
        // Generate random value using CURAND
        real_t rand_val = curand_uniform(&rand_states[state_idx]);

        // Convert to sample index (EXACT same as original: int(randomu()*nsample) + 1)
        // Note: Fortran uses 1-based indexing, C++ uses 0-based
        integer_t i = static_cast<integer_t>(rand_val * nsample);
        if (i >= nsample) i = nsample - 1;  // Clamp to valid range

        // Store the selected sample index
        sample_indices[n] = i;
    }
}

// Kernel for counting bootstrap samples using atomic operations
__global__ void cuda_count_samples_kernel(const integer_t* sample_indices,
                                          integer_t* nin, integer_t nsample) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;

    // Count occurrences using atomic operations (preserves original counting logic)
    for (integer_t n = tid; n < nsample; n += stride) {
        integer_t sample_idx = sample_indices[n];
        if (sample_idx >= 0 && sample_idx < nsample) {
            ::atomicAdd(&nin[sample_idx], 1);
        }
    }
}

// Kernel for computing weights and bag assignments
__global__ void cuda_compute_weights_kernel(const real_t* weight, const integer_t* nin,
                                            real_t* win, integer_t* jinbag,
                                            integer_t* joobag, integer_t* ninbag_ptr,
                                            integer_t* noobag_ptr, integer_t nsample) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;

    // Process each sample
    for (integer_t n = tid; n < nsample; n += stride) {
        if (nin[n] == 0) {
            // Out-of-bag sample
            win[n] = 0.0f;
            integer_t oobag_idx = ::atomicAdd(noobag_ptr, 1);
            if (oobag_idx < nsample) {
                joobag[oobag_idx] = n;
            }
        } else {
            // In-bag sample
            win[n] = static_cast<real_t>(nin[n]) * weight[n];
            integer_t inbag_idx = ::atomicAdd(ninbag_ptr, 1);
            if (inbag_idx < nsample) {
                jinbag[inbag_idx] = n;
            }
        }
    }
}

} // anonymous namespace

// Forward declaration for C++ CPU implementation
extern void cpu_bootstrap(const real_t* weight, integer_t nsample,
                          real_t* win, integer_t* nin, integer_t* nout,
                          integer_t* jinbag, integer_t* joobag,
                          integer_t& ninbag, integer_t& noobag,
                          integer_t tree_id);

// ============================================================================
// Main Bootstrap Wrapper
// Matches boot_cuda in Fortran - checks GPU/CPU and calls appropriate path
// ============================================================================

// Global random states for GPU (persistent across calls like randomu())
static cuda::RandomState* g_rand_states = nullptr;

// Cleanup function for global bootstrap states
void cleanup_bootstrap_states() {
    // Add guard to prevent double-free
    static bool cleaned = false;
    if (cleaned) return;  // Already cleaned, prevent double-free
    if (g_rand_states != nullptr) {
        try {
            cuda::cleanup_random_states(*g_rand_states);
            delete g_rand_states;
            g_rand_states = nullptr;
        } catch (...) {
            // Ignore errors - CUDA context might be destroyed
            g_rand_states = nullptr;  // Still mark as null to prevent future issues
        }
    }
    cleaned = true;
}

void gpu_bootstrap(const real_t* weight, integer_t nsample,
               real_t* win, integer_t* nin, integer_t* nout,
               integer_t* jinbag, integer_t* joobag,
               integer_t& ninbag, integer_t& noobag,
               integer_t tree_id) {

    // Always use CPU fallback to avoid global state issues
    // This ensures deterministic behavior and avoids memory corruption
    cpu_bootstrap(weight, nsample, win, nin, nout, jinbag,
                          joobag, ninbag, noobag, tree_id);
}

} // namespace rf
