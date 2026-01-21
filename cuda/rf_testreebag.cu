#include "rf_testreebag.cuh"
#include "rf_config.hpp"
#include "rf_memory.cuh"
#include "rf_cuda_config.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace rf {

// ============================================================================
// CUDA Kernel (internal)
// ============================================================================

namespace {  // Anonymous namespace

// CUDA kernel for parallel tree testing
// Exact port of cuda_testreebag_kernel (lines 15-70)
__global__ void cuda_testreebag_kernel(const real_t* x, const real_t* xbestsplit,
                                       const integer_t* nin, const integer_t* treemap,
                                       const integer_t* bestvar, const integer_t* nodeclass,
                                       const integer_t* cat, const integer_t* nodestatus,
                                       const integer_t* catgoleft, integer_t nsample,
                                       integer_t mdim, integer_t nnode, integer_t maxcat,
                                       integer_t* jtr, integer_t* nodextr) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;

    // Each thread processes multiple samples in parallel
    for (integer_t n = tid; n < nsample; n += stride) {
        // ALL samples need nodextr for proximity computation
        // But only OOB samples (nin==0) update jtr for predictions

        // Pass nth case down the tree
        integer_t kt = 0;  // 0-based (root node)
        for (integer_t k = 0; k < nnode; k++) {
            if (nodestatus[kt] == -1) {
                // Node kt is where case n ends up: it's a terminal node
                nodextr[n] = kt;
                if (nin[n] == 0) {
                    // Only set prediction for OOB samples
                    jtr[n] = nodeclass[kt];
                }
                break;  // exit
            }

            integer_t m = bestvar[kt];
            // See whether case n goes right or left
            // Binary features (cat <= 2) use quantitative path
            if (cat[m] <= 2) {
                // Quantitative variable (including binary)
                // Row-major: x(n, m) - sample n, feature m
                if (x[n * mdim + m] <= xbestsplit[kt]) {
                    kt = treemap[0 + kt * 2];  // Column-major: treemap(1, kt)
                } else {
                    kt = treemap[1 + kt * 2];  // Column-major: treemap(2, kt)
                }
            } else {
                // Categorical variable
                integer_t jcat = static_cast<integer_t>(x[n * mdim + m] + 0.5f);  // nint
                // Column-major: catgoleft(jcat, kt)
                if (catgoleft[jcat + kt * maxcat] == 1) {
                    kt = treemap[0 + kt * 2];
                } else {
                    kt = treemap[1 + kt * 2];
                }
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Forward declarations for C++ CPU implementations
extern void cpu_testreebag(const real_t* x, const real_t* xbestsplit,
                             const integer_t* nin, const integer_t* treemap,
                             const integer_t* bestvar, const integer_t* nodeclass,
                             const integer_t* cat, const integer_t* nodestatus,
                             const integer_t* catgoleft, integer_t nsample,
                             integer_t mdim, integer_t nnode, integer_t maxcat,
                             integer_t* jtr, integer_t* nodextr);

// ============================================================================
// GPU TESTREEBAG IMPLEMENTATION WITH PARALLEL OOB TESTING
// ============================================================================

void gpu_testreebag(const real_t* x, const real_t* xbestsplit,
                   const integer_t* nin, const integer_t* treemap,
                   const integer_t* bestvar, const integer_t* nodeclass,
                   const integer_t* cat, const integer_t* nodestatus,
                   const integer_t* catgoleft, integer_t nsample,
                   integer_t mdim, integer_t nnode, integer_t maxcat,
                   integer_t* jtr, integer_t* nodextr) {
    
    // For small datasets, CPU might be faster due to GPU overhead
    if (nsample < 200) {
        cpu_testreebag(x, xbestsplit, nin, treemap, bestvar, nodeclass, cat,
                      nodestatus, catgoleft, nsample, mdim, nnode, maxcat,
                      jtr, nodextr);
        return;
    }
    
    // std::cout << "GPU: Computing tree bag testing with parallel OOB traversal..." << std::endl;
    
    // Allocate device memory
    real_t* x_d;
    real_t* xbestsplit_d;
    integer_t* nin_d;
    integer_t* treemap_d;
    integer_t* bestvar_d;
    integer_t* nodeclass_d;
    integer_t* cat_d;
    integer_t* nodestatus_d;
    integer_t* catgoleft_d;
    integer_t* jtr_d;
    integer_t* nodextr_d;
    
    CUDA_CHECK_VOID(cudaMalloc(&x_d, nsample * mdim * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&xbestsplit_d, nnode * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nin_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&treemap_d, 2 * nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&bestvar_d, nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nodeclass_d, nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&cat_d, mdim * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nodestatus_d, nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&catgoleft_d, maxcat * nnode * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&jtr_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nodextr_d, nsample * sizeof(integer_t)));
    
    // Copy data to device
    CUDA_CHECK_VOID(cudaMemcpy(x_d, x, nsample * mdim * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(xbestsplit_d, xbestsplit, nnode * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nin_d, nin, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(treemap_d, treemap, 2 * nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(bestvar_d, bestvar, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nodeclass_d, nodeclass, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(cat_d, cat, mdim * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nodestatus_d, nodestatus, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(catgoleft_d, catgoleft, maxcat * nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    
    // Initialize output arrays
    CUDA_CHECK_VOID(cudaMemset(jtr_d, 0, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMemset(nodextr_d, 0, nsample * sizeof(integer_t)));
    
    // Launch kernel for parallel tree testing
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    cuda_testreebag_kernel<<<grid_size, block_size>>>(
        x_d, xbestsplit_d, nin_d, treemap_d, bestvar_d, nodeclass_d,
        cat_d, nodestatus_d, catgoleft_d, nsample, mdim, nnode, maxcat,
        jtr_d, nodextr_d
    );
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Copy results back to host
    CUDA_CHECK_VOID(cudaMemcpy(jtr, jtr_d, nsample * sizeof(integer_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(nodextr, nodextr_d, nsample * sizeof(integer_t), cudaMemcpyDeviceToHost));
    
    // Cleanup device memory
    cudaFree(x_d);
    cudaFree(xbestsplit_d);
    cudaFree(nin_d);
    cudaFree(treemap_d);
    cudaFree(bestvar_d);
    cudaFree(nodeclass_d);
    cudaFree(cat_d);
    cudaFree(nodestatus_d);
    cudaFree(catgoleft_d);
    cudaFree(jtr_d);
    cudaFree(nodextr_d);
    
    // std::cout << "GPU: Tree bag testing completed." << std::endl;
}

} // namespace rf
