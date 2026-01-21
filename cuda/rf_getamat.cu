#include "rf_getamat.cuh"
#include "rf_config.hpp"
#include "rf_memory.cuh"
#include "rf_cuda_config.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace rf {

namespace {

// ----------------------------------------------------------------------
// CUDA kernel for parallel matrix copying (basic version)
// ----------------------------------------------------------------------
__global__ void cuda_getamat_kernel(const integer_t* asave, const integer_t* nin,
                                     const integer_t* cat, integer_t nsample,
                                     integer_t mdim, integer_t* amat) {
    // asave is (mdim, nsample) column-major
    // amat is (mdim, nsample) column-major

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple variables (columns) in parallel
    for (int m = tid; m < mdim; m += stride) {
        int nt = 0;  // 0-based index

        // Binary features (cat <= 2) use quantitative path
        if (cat[m] <= 2) {
            // Quantitative variable (including binary) - exact same logic as original
            for (int n = 0; n < nsample; ++n) {
                int idx = asave[m + n * mdim];  // Column-major access
                if (idx >= 0 && idx < nsample && nin[idx] >= 1) {
                    amat[m + nt * mdim] = idx;
                    ++nt;
                }
            }
        } else {
            // Categorical variable - exact same logic as original
            for (int n = 0; n < nsample; ++n) {
                amat[m + n * mdim] = asave[m + n * mdim];
            }
        }
    }
}

// ----------------------------------------------------------------------
// CUDA kernel for parallel matrix copying with shared memory optimization
// ----------------------------------------------------------------------
__global__ void cuda_getamat_optimized_kernel(const integer_t* asave, const integer_t* nin,
                                               const integer_t* cat, integer_t nsample,
                                               integer_t mdim, integer_t* amat) {
    // Shared memory for frequently accessed data
    __shared__ integer_t shared_nin[256];
    __shared__ integer_t shared_cat[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int local_tid = threadIdx.x;

    // Load data into shared memory (coalesced access)
    if (local_tid < min(blockDim.x, nsample)) {
        if (local_tid < nsample) shared_nin[local_tid] = nin[local_tid];
        if (local_tid < mdim) shared_cat[local_tid] = cat[local_tid];
    }

    __syncthreads();

    // Process variables with shared memory access
    for (int m = tid; m < mdim; m += stride) {
        int nt = 0;

        if (m < mdim && (m < 256 ? shared_cat[m] : cat[m]) == 1) {
            // Quantitative variable
            for (int n = 0; n < nsample; ++n) {
                int idx = asave[m + n * mdim];
                if (idx >= 0 && idx < nsample) {
                    int nin_val = (idx < 256) ? shared_nin[idx] : nin[idx];
                    if (nin_val >= 1) {
                        amat[m + nt * mdim] = idx;
                        ++nt;
                    }
                }
            }
        } else if (m < mdim) {
            // Categorical variable
            for (int n = 0; n < nsample; ++n) {
                amat[m + n * mdim] = asave[m + n * mdim];
            }
        }
    }
}

} // anonymous namespace

// Forward declaration for C++ CPU implementation
extern void cpu_getamat(const integer_t* asave, const integer_t* nin, const integer_t* cat,
                        integer_t nsample, integer_t mdim, integer_t* amat);

// ----------------------------------------------------------------------
// GPU GETAMAT IMPLEMENTATION WITH PARALLEL MATRIX OPERATIONS
// ----------------------------------------------------------------------
void gpu_getamat(const integer_t* asave, const integer_t* nin, const integer_t* cat,
                 integer_t nsample, integer_t mdim, integer_t* amat) {

    // For small matrices, CPU might be faster due to GPU overhead
    if (nsample * mdim < 10000) {
        cpu_getamat(asave, nin, cat, nsample, mdim, amat);
        return;
    }
    
    // std::cout << "GPU: Computing matrix operations with parallel copying..." << std::endl;
    
    // Allocate device memory
    integer_t* asave_d;
    integer_t* nin_d;
    integer_t* cat_d;
    integer_t* amat_d;
    
    CUDA_CHECK_VOID(cudaMalloc(&asave_d, mdim * nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nin_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&cat_d, mdim * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&amat_d, mdim * nsample * sizeof(integer_t)));
    
    // Copy data to device
    CUDA_CHECK_VOID(cudaMemcpy(asave_d, asave, mdim * nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nin_d, nin, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(cat_d, cat, mdim * sizeof(integer_t), cudaMemcpyHostToDevice));
    
    // Initialize amat to zero
    CUDA_CHECK_VOID(cudaMemset(amat_d, 0, mdim * nsample * sizeof(integer_t)));
    
    // Launch kernel for parallel matrix operations
    dim3 block_size(256);
    dim3 grid_size((mdim * nsample + block_size.x - 1) / block_size.x);
    
    cuda_getamat_kernel<<<grid_size, block_size>>>(
        asave_d, nin_d, cat_d, nsample, mdim, amat_d
    );
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Copy results back to host
    CUDA_CHECK_VOID(cudaMemcpy(amat, amat_d, mdim * nsample * sizeof(integer_t), cudaMemcpyDeviceToHost));
    
    // Cleanup device memory
    cudaFree(asave_d);
    cudaFree(nin_d);
    cudaFree(cat_d);
    cudaFree(amat_d);
    
    // std::cout << "GPU: Matrix operations completed." << std::endl;
}

} // namespace rf
