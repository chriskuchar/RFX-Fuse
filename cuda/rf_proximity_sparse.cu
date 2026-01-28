/**
 * @file rf_proximity_sparse.cu
 * @brief GPU sparse proximity implementation
 * 
 * Proximity uses terminal nodes (computed from sparse tree traversal).
 * The kernels themselves don't need sparse matrix access.
 */

#include "rf_proximity_sparse.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

// ============================================================================
// Full proximity update kernel
// ============================================================================

__global__ void gpu_update_proximity_kernel(
    const integer_t* nodextr,
    const integer_t* nin,
    const integer_t* nodestatus,
    integer_t nsample,
    integer_t nnode,
    bool use_casewise,
    dp_t* prox,
    integer_t* error_code
) {
    // 2D grid: each thread handles one (i, j) pair
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= nsample || j >= nsample) return;
    
    // Only process OOB sample i
    if (nin[i] != 0) return;
    
    // Get terminal nodes
    integer_t node_i = nodextr[i];
    integer_t node_j = nodextr[j];
    
    // Check bounds and terminal status
    if (node_i < 0 || node_i >= nnode || 
        node_j < 0 || node_j >= nnode) {
        return;
    }
    
    // Both must be in same terminal node
    if (node_i != node_j) return;
    if (nodestatus[node_i] != -1) return;  // Must be terminal
    
    // Update proximity
    dp_t weight = 1.0;
    if (use_casewise && nin[j] > 0) {
        // Could weight by nin[j] for casewise, but simplified here
        weight = 1.0;
    }
    
    // Row-major: prox[i * nsample + j]
    ::atomicAdd(&prox[i * nsample + j], weight);
}

// ============================================================================
// Low-rank proximity update kernel (QLoRA)
// ============================================================================

__global__ void gpu_update_lowrank_proximity_kernel(
    const integer_t* nodextr,
    const integer_t* nin,
    const integer_t* nodestatus,
    integer_t nsample,
    integer_t nnode,
    integer_t rank,
    integer_t tree_id,
    bool use_casewise,
    dp_t* A,
    dp_t* B,
    integer_t* error_code
) {
    // 2D grid: each thread handles one (i, j) pair
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= nsample || j >= nsample || i == j) return;
    
    // Only process OOB sample i, in-bag sample j
    if (nin[i] != 0) return;
    if (nin[j] == 0) return;  // j must be in-bag
    
    // Get terminal nodes
    integer_t node_i = nodextr[i];
    integer_t node_j = nodextr[j];
    
    // Bounds check
    if (node_i < 0 || node_i >= nnode || 
        node_j < 0 || node_j >= nnode) {
        return;
    }
    
    // Both must be in same terminal node
    if (node_i != node_j) return;
    if (nodestatus[node_i] != -1) return;
    
    // Column selection: round-robin based on tree_id
    integer_t col = tree_id % rank;
    
    // Weight
    dp_t weight = 1.0;
    if (use_casewise) {
        weight = static_cast<dp_t>(nin[j]);
    }
    
    // Update low-rank factors
    // Row-major: A[i * rank + col], B[j * rank + col]
    ::atomicAdd(&A[i * rank + col], weight);
    ::atomicAdd(&B[j * rank + col], weight);
}

// ============================================================================
// Finalize kernels
// ============================================================================

__global__ void gpu_finalize_proximity_kernel(
    dp_t* prox,
    integer_t nsample,
    integer_t ntree,
    integer_t* error_code
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t total = nsample * nsample;
    integer_t stride = blockDim.x * gridDim.x;
    
    dp_t inv_ntree = 1.0 / static_cast<dp_t>(ntree);
    
    for (integer_t idx = tid; idx < total; idx += stride) {
        prox[idx] *= inv_ntree;
    }
}

__global__ void gpu_finalize_lowrank_kernel(
    dp_t* A,
    dp_t* B,
    integer_t nsample,
    integer_t rank,
    integer_t ntree,
    integer_t* error_code
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t total = nsample * rank;
    integer_t stride = blockDim.x * gridDim.x;
    
    dp_t inv_ntree = 1.0 / static_cast<dp_t>(ntree);
    
    for (integer_t idx = tid; idx < total; idx += stride) {
        A[idx] *= inv_ntree;
        B[idx] *= inv_ntree;
    }
}

// ============================================================================
// Host function
// ============================================================================

integer_t gpu_proximity_update_sparse(
    const integer_t* d_nodextr,
    const integer_t* d_nin,
    const integer_t* d_nodestatus,
    integer_t nsample,
    integer_t nnode,
    bool use_casewise,
    bool use_lowrank,
    integer_t rank,
    integer_t tree_id,
    dp_t* d_prox,
    dp_t* d_A,
    dp_t* d_B,
    cudaStream_t stream
) {
    // Allocate error code
    integer_t* d_error;
    cudaMalloc(&d_error, sizeof(integer_t));
    cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
    
    // 2D grid for (i, j) pairs
    dim3 block_size(16, 16);  // 256 threads per block
    dim3 grid_size(
        (nsample + block_size.x - 1) / block_size.x,
        (nsample + block_size.y - 1) / block_size.y
    );
    
    if (use_lowrank) {
        gpu_update_lowrank_proximity_kernel<<<grid_size, block_size, 0, stream>>>(
            d_nodextr, d_nin, d_nodestatus,
            nsample, nnode, rank, tree_id, use_casewise,
            d_A, d_B, d_error
        );
    } else {
        gpu_update_proximity_kernel<<<grid_size, block_size, 0, stream>>>(
            d_nodextr, d_nin, d_nodestatus,
            nsample, nnode, use_casewise,
            d_prox, d_error
        );
    }
    
    // Sync and check error
    cudaStreamSynchronize(stream);
    integer_t h_error = 0;
    cudaMemcpy(&h_error, d_error, sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_error);
    return h_error;
}

} // namespace cuda
} // namespace rf

