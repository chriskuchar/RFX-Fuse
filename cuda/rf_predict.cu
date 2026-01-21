#include "rf_predict.cuh"
#include "rf_config.hpp"
#include "rf_memory.cuh"
#include "rf_cuda_config.hpp"
#include <cuda_runtime.h>

namespace rf {

namespace {

// ----------------------------------------------------------------------
// CUDA kernel for parallel prediction
// ----------------------------------------------------------------------
__global__ void cuda_predict_kernel(const real_t* q, const integer_t* out,
                                     integer_t nsample, integer_t nclass,
                                     integer_t* jest) {
    // q is (nclass, nsample) column-major

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple samples
    for (int n = tid; n < nsample; n += stride) {
        integer_t jmax = 0;  // 0-based (Fortran uses 1-based)
        real_t cmax = 0.0f;

        // Same logic as original algorithm
        if (out[n] > 0) {
            for (int j = 0; j < nclass; ++j) {
                real_t ctemp = q[j + n * nclass];  // Column-major: (j, n)
                if (ctemp > cmax) {
                    jmax = j;
                    cmax = ctemp;
                }
            }
        }

        jest[n] = jmax;
    }
}

} // anonymous namespace

// Forward declaration for C++ CPU implementation
extern void cpu_predict(const real_t* q, const integer_t* out, integer_t nsample,
                        integer_t nclass, integer_t* jest);

// ----------------------------------------------------------------------
// Main wrapper (matches predict_cuda in Fortran)
// ----------------------------------------------------------------------
void gpu_predict(const real_t* q, const integer_t* out, integer_t nsample,
                integer_t nclass, integer_t* jest) {

    if (g_config.using_gpu()) {
        // Allocate device memory
        real_t* q_d;
        cudaMalloc(&q_d, nclass * nsample * sizeof(real_t));
        integer_t* out_d;
        cudaMalloc(&out_d, nsample * sizeof(integer_t));
        integer_t* jest_d;
        cudaMalloc(&jest_d, nsample * sizeof(integer_t));

        // Copy data to device
        CUDA_CHECK_VOID(cudaMemcpy(q_d, q, nclass * nsample * sizeof(real_t),
                                   cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(out_d, out, nsample * sizeof(integer_t),
                                   cudaMemcpyHostToDevice));

        // Launch kernel using dynamic GPU config
        int threads = cuda::get_optimal_block_size(nsample);
        int blocks = min(cuda::get_max_blocks(), (nsample + threads - 1) / threads);

        cuda_predict_kernel<<<blocks, threads>>>(q_d, out_d, nsample,
                                                   nclass, jest_d);

        CUDA_CHECK_VOID(cudaGetLastError());
        CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety

        // Copy result back
        CUDA_CHECK_VOID(cudaMemcpy(jest, jest_d, nsample * sizeof(integer_t),
                                   cudaMemcpyDeviceToHost));

        // Cleanup device memory
        cudaFree(q_d);
        cudaFree(out_d);
        cudaFree(jest_d);

    } else {
        // Always use CPU fallback
        cpu_predict(q, out, nsample, nclass, jest);
    }
}

} // namespace rf
