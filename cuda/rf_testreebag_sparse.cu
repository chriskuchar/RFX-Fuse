/**
 * @file rf_testreebag_sparse.cu
 * @brief GPU sparse OOB testing - find terminal nodes using sparse matrix input
 * 
 * Matches CPU cpu_testreebag_sparse exactly.
 * Uses same traversal pattern as dense rf_testreebag.cu but with sparse access.
 * 
 * Key differences from dense:
 * - Uses CudaSparseMatrixCSR.get(row, col) instead of x[row * mdim + col]
 * - Input is already on GPU (CudaSparseMatrixCSR)
 * - No data upload needed per-tree
 */

#include "rf_testreebag_sparse.cuh"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

// ============================================================================
// CUDA Kernel: Find terminal nodes for sparse input
// ============================================================================

/**
 * GPU kernel for parallel tree testing with sparse matrix
 * Exact match of cuda_testreebag_kernel but using sparse access
 * 
 * Pattern: Each thread processes one sample, traverses tree to terminal node
 * Uses fixed-iteration loop to avoid warp divergence from while loops
 */
__global__ void gpu_find_terminal_node_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    const integer_t* samples,      // Sample indices to process (can be all samples or OOB only)
    integer_t n_samples,           // Number of samples to process
    const integer_t* treemap,      // Tree structure [2 * maxnode] - col-major: treemap(child, node)
    const integer_t* nodestatus,   // Node status (-1 = terminal, 1 = internal)
    const real_t* xbestsplit,      // Split values [maxnode]
    const integer_t* bestvar,      // Split variable indices [maxnode]
    const integer_t* nodeclass,    // Class predictions [maxnode]
    integer_t nnode,               // Number of nodes in tree
    integer_t max_depth,           // Maximum tree depth for fixed iteration
    integer_t* terminal_nodes,     // Output: terminal node index for each sample
    integer_t* predictions,        // Output: prediction (class) for each sample
    integer_t* error_code          // Output: error code (0 = success)
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple samples in grid-stride loop
    for (integer_t idx = tid; idx < n_samples; idx += stride) {
        // Get actual sample index
        integer_t n = samples ? samples[idx] : idx;
        
        // Bounds check
        if (n < 0 || n >= X_sparse.nrows) {
            ::atomicExch(error_code, CUDA_ERROR_INVALID_SAMPLE);
            continue;
        }
        
        // Traverse tree from root (node 0)
        integer_t kt = 0;
        bool reached_terminal = false;
        
        // Fixed iteration count - same as dense kernel's for loop over nnode
        // But we use max_depth as the limit since trees are typically log(n) deep
        for (integer_t depth = 0; depth < max_depth && !reached_terminal; ++depth) {
            // Check if terminal
            if (kt < 0 || kt >= nnode) {
                ::atomicExch(error_code, CUDA_ERROR_INVALID_NODE);
                break;
            }
            
            if (nodestatus[kt] == -1) {
                // Terminal node - record and exit
                terminal_nodes[idx] = kt;
                predictions[idx] = nodeclass[kt];
                reached_terminal = true;
                break;
            }
            
            // Get split variable and value
            integer_t m = bestvar[kt];
            if (m < 0 || m >= X_sparse.ncols) {
                // Invalid split variable - treat as terminal
                terminal_nodes[idx] = kt;
                predictions[idx] = nodeclass[kt];
                reached_terminal = true;
                break;
            }
            
            // Get feature value using sparse access
            // ROW-MAJOR: X.get(sample, feature) - matches CPU cpu_testreebag_sparse
            // n = sample index (row), m = feature index (col)
            real_t xmn = X_sparse.get(n, m);
            
            // Determine which child to go to
            // Quantitative split (cat check not needed for most RF - simplified)
            // treemap layout: [left_child_0, right_child_0, left_child_1, right_child_1, ...]
            if (xmn <= xbestsplit[kt]) {
                kt = treemap[kt * 2];      // Left child
            } else {
                kt = treemap[kt * 2 + 1];  // Right child
            }
        }
        
        // If we didn't reach terminal in max_depth iterations, record current node
        if (!reached_terminal) {
            terminal_nodes[idx] = kt;
            predictions[idx] = (kt >= 0 && kt < nnode) ? nodeclass[kt] : 0;
        }
    }
}

// ============================================================================
// CUDA Kernel: Accumulate OOB votes
// ============================================================================

/**
 * Accumulate OOB predictions into vote counts
 * For classification: votes per class per sample
 */
__global__ void gpu_accumulate_oob_votes_kernel(
    const integer_t* terminal_nodes,  // Terminal node for each sample
    const integer_t* nodeclass,       // Class at each node
    const integer_t* nin,             // Bootstrap frequency (0 = OOB)
    integer_t n_samples,
    integer_t nclass,
    real_t* oob_votes,                // Output: [nsample * nclass] vote counts (real_t for fractional weights)
    integer_t* oob_counts,            // Output: [nsample] number of OOB votes
    integer_t* error_code
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;

    for (integer_t n = tid; n < n_samples; n += stride) {
        // Only accumulate for OOB samples
        if (nin[n] == 0) {
            integer_t pred_class = nodeclass[terminal_nodes[n]];

            // Bounds check class
            if (pred_class >= 0 && pred_class < nclass) {
                // Atomic increment vote count (real_t for fractional casewise weights)
                ::atomicAdd(&oob_votes[n * nclass + pred_class], 1.0f);
                ::atomicAdd(&oob_counts[n], 1);
            }
        }
    }
}

// ============================================================================
// Host function: Run sparse testreebag
// ============================================================================

integer_t gpu_testreebag_sparse(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_treemap,
    const integer_t* d_nodestatus,
    const real_t* d_xbestsplit,
    const integer_t* d_bestvar,
    const integer_t* d_nodeclass,
    const integer_t* d_nin,
    integer_t nnode,
    integer_t nsample,
    integer_t nclass,
    integer_t max_depth,
    integer_t* d_terminal_nodes,
    real_t* d_oob_votes,
    integer_t* d_oob_counts,
    cudaStream_t stream
) {
    // Allocate error code on device
    integer_t* d_error;
    cudaMalloc(&d_error, sizeof(integer_t));
    cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
    
    // Launch kernel: find terminal nodes for all samples
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    // Temporary predictions array
    integer_t* d_predictions;
    cudaMalloc(&d_predictions, nsample * sizeof(integer_t));
    cudaMemsetAsync(d_predictions, 0, nsample * sizeof(integer_t), stream);
    
    // Find terminal nodes
    gpu_find_terminal_node_sparse_kernel<<<grid_size, block_size, 0, stream>>>(
        X_sparse,
        nullptr,              // Process all samples (indices 0..nsample-1)
        nsample,
        d_treemap,
        d_nodestatus,
        d_xbestsplit,
        d_bestvar,
        d_nodeclass,
        nnode,
        max_depth,
        d_terminal_nodes,
        d_predictions,
        d_error
    );
    
    // Check for errors after first kernel
    cudaStreamSynchronize(stream);
    integer_t h_error = 0;
    cudaMemcpy(&h_error, d_error, sizeof(integer_t), cudaMemcpyDeviceToHost);
    if (h_error != CUDA_OK) {
        cudaFree(d_error);
        cudaFree(d_predictions);
        return h_error;
    }
    
    // Accumulate OOB votes (only if output arrays provided)
    if (d_oob_votes != nullptr && d_oob_counts != nullptr) {
        cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
        
        gpu_accumulate_oob_votes_kernel<<<grid_size, block_size, 0, stream>>>(
            d_terminal_nodes,
            d_nodeclass,
            d_nin,
            nsample,
            nclass,
            d_oob_votes,
            d_oob_counts,
            d_error
        );
        
        // Final sync and error check
        cudaStreamSynchronize(stream);
        cudaMemcpy(&h_error, d_error, sizeof(integer_t), cudaMemcpyDeviceToHost);
    }
    
    // Cleanup
    cudaFree(d_error);
    cudaFree(d_predictions);
    
    return h_error;
}

// ============================================================================
// CUDA Kernel: Casewise vote adjustment (GPU-only, no CPU round-trip)
// ============================================================================

/**
 * Adjust OOB votes for casewise mode:
 * - Undo the +1 unweighted vote from gpu_accumulate_oob_votes_kernel
 * - Add weighted vote based on tnodewt (mean bootstrap freq in terminal node)
 * 
 * This replaces the CPU round-trip in rf_sparse_forest.cu
 */
__global__ void gpu_adjust_casewise_votes_kernel(
    const integer_t* terminal_nodes,  // Terminal node for each sample
    const integer_t* nodeclass,       // Class at each node
    const real_t* tnodewt,            // Node weights [nnode]
    const integer_t* nin,             // Bootstrap frequency (0 = OOB)
    integer_t nsample,
    integer_t nclass,
    integer_t nnode,
    real_t* oob_votes,                // [nsample * nclass] vote counts (modified in-place)
    integer_t* oob_counts             // [nsample] OOB counts (modified in-place)
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;
    
    for (integer_t n = tid; n < nsample; n += stride) {
        // Only adjust OOB samples
        if (nin[n] != 0) continue;
        
        integer_t kt = terminal_nodes[n];
        if (kt < 0 || kt >= nnode) continue;
        
        integer_t pred_class = nodeclass[kt];
        if (pred_class < 0 || pred_class >= nclass) continue;
        
        // Undo unweighted vote, add weighted vote
        ::atomicAdd(&oob_votes[n * nclass + pred_class], -1.0f);
        ::atomicSub(&oob_counts[n], 1);

        real_t vote_weight = tnodewt[kt];
        ::atomicAdd(&oob_votes[n * nclass + pred_class], vote_weight);

        integer_t weighted_count = static_cast<integer_t>(vote_weight + 0.5f);
        if (weighted_count < 1) weighted_count = 1;
        ::atomicAdd(&oob_counts[n], weighted_count);
    }
}

/**
 * Host function: Adjust casewise votes on GPU
 * Replaces the CPU round-trip in rf_sparse_forest.cu
 */
void gpu_adjust_casewise_votes(
    const integer_t* d_terminal_nodes,
    const integer_t* d_nodeclass,
    const real_t* d_tnodewt,
    const integer_t* d_nin,
    integer_t nsample,
    integer_t nclass,
    integer_t nnode,
    real_t* d_oob_votes,
    integer_t* d_oob_counts,
    cudaStream_t stream
) {
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    gpu_adjust_casewise_votes_kernel<<<grid_size, block_size, 0, stream>>>(
        d_terminal_nodes,
        d_nodeclass,
        d_tnodewt,
        d_nin,
        nsample,
        nclass,
        nnode,
        d_oob_votes,
        d_oob_counts
    );
}

} // namespace cuda
} // namespace rf

