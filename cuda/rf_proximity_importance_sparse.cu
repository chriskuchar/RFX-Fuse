/**
 * @file rf_proximity_importance_sparse.cu
 * @brief GPU sparse proximity importance implementation
 * 
 * Proximity importance = fraction of trees where permuting a feature
 * changes the terminal node for OOB samples WITH CORRECT PREDICTIONS.
 * 
 * This matches Breiman-Cutler methodology: only accumulate for samples
 * that were correctly classified OOB (same as local variable importance).
 * 
 * Key design:
 * - One kernel per tree (2D grid over samples x features)
 * - Each thread processes one (sample, feature) pair
 * - Uses sparse matrix access via CudaSparseMatrixCSR.get()
 * - Computes jtr on-the-fly: jtr = nodeclass[nodextr]
 * - Only accumulates when jtr[n] == cl[n] (correct prediction)
 * - Uses per-tree normalization (1.0 / noob) - same as local importance
 */

#include "rf_proximity_importance_sparse.cuh"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

// ============================================================================
// Device function: Traverse tree with permuted feature
// ============================================================================

/**
 * Traverse tree using sparse data, with one feature permuted.
 * 
 * For feature k: use random sample's value instead of original
 * For other features: use original sample's value
 * 
 * Returns terminal node reached.
 */
__device__ integer_t traverse_with_permuted_feature_sparse(
    const CudaSparseMatrixCSR& X_sparse,
    integer_t sample_orig,
    integer_t sample_perm,  // Random sample for permuted feature
    integer_t feature_k,
    integer_t mdim,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    integer_t nnode,
    integer_t max_depth
) {
    integer_t kt = 0;  // Start at root
    
    for (integer_t depth = 0; depth < max_depth; ++depth) {
        if (kt < 0 || kt >= nnode) break;
        
        // Terminal node?
        if (nodestatus[kt] == -1) {
            return kt;
        }
        
        // Get split variable
        integer_t m = bestvar[kt];
        if (m < 0 || m >= mdim) break;
        
        // Get feature value
        // ROW-MAJOR: X_sparse.get(sample, feature)
        real_t xmn;
        if (m == feature_k) {
            // Use permuted sample for this feature
            xmn = X_sparse.get(sample_perm, m);
        } else {
            // Use original sample
            xmn = X_sparse.get(sample_orig, m);
        }
        
        // Traverse (quantitative split)
        if (xmn <= xbestsplit[kt]) {
            kt = treemap[kt * 2];      // Left child
        } else {
            kt = treemap[kt * 2 + 1];  // Right child
        }
    }
    
    return kt;  // Return wherever we ended up
}

// ============================================================================
// Proximity importance kernel (ONE feature) - legacy, kept for compatibility
// ============================================================================

__global__ void gpu_proximity_importance_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    const integer_t* nodextr_orig,
    const integer_t* nin,
    const integer_t* cl,
    const integer_t* nodeclass,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t max_depth,
    integer_t feature_k,
    integer_t tree_id,
    integer_t noob,
    real_t* prox_imp,
    bool is_regression,
    integer_t* error_code
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;
    
    if (noob <= 0) return;
    
    real_t inv_noob = 1.0f / static_cast<real_t>(noob);
    
    // Each thread processes multiple samples
    for (integer_t i = tid; i < nsample; i += stride) {
        // Only process OOB samples
        if (nin[i] != 0) continue;
        
        integer_t node_orig = nodextr_orig[i];
        if (node_orig < 0 || node_orig >= nnode) continue;
        
        // Compute jtr on-the-fly from nodeclass
        // Only process samples with CORRECT predictions
        if (!is_regression) {
            integer_t jtr = nodeclass[node_orig];
            if (jtr != cl[i]) continue;  // Skip incorrect predictions
        }
        
        // Get donor sample using deterministic hash (matches GPU Dense for reproducibility)
        integer_t sample_perm = (i + feature_k * 31 + tree_id * 17) % nsample;
        if (sample_perm == i) sample_perm = (sample_perm + 1) % nsample;
        
        // Traverse with permuted feature
        integer_t node_perm = traverse_with_permuted_feature_sparse(
            X_sparse, i, sample_perm, feature_k,
            mdim, treemap, nodestatus, xbestsplit, bestvar,
            nnode, max_depth
        );
        
        // If terminal node changed, increment proximity importance
        // Use per-tree normalization (1/noob) - same as local variable importance
        if (node_orig != node_perm) {
            ::atomicAdd(&prox_imp[i * mdim + feature_k], inv_noob);
        }
    }
}

// ============================================================================
// OPTIMIZED: Proximity importance kernel (ALL features in one launch)
// Uses 2D grid: blockIdx.x = sample batches, blockIdx.y = feature batches
// With correctness check matching Breiman-Cutler methodology
// ============================================================================

__global__ void gpu_proximity_importance_sparse_kernel_2d(
    const CudaSparseMatrixCSR X_sparse,
    const integer_t* nodextr_orig,
    const integer_t* nin,
    const integer_t* cl,
    const integer_t* nodeclass,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t max_depth,
    integer_t tree_id,
    integer_t noob,
    real_t* prox_imp,
    bool is_regression
) {
    // 2D indexing: x = samples, y = features
    integer_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    integer_t feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (sample_idx >= nsample || feature_idx >= mdim) return;
    if (noob <= 0) return;
    
    // Only process OOB samples
    if (nin[sample_idx] != 0) return;
    
    // Get original terminal node
    integer_t node_orig = nodextr_orig[sample_idx];
    if (node_orig < 0 || node_orig >= nnode) return;
    
    // Compute jtr on-the-fly from nodeclass
    // Only process samples with CORRECT predictions
    if (!is_regression) {
        integer_t jtr = nodeclass[node_orig];
        if (jtr != cl[sample_idx]) return;  // Skip incorrect predictions
    }
    
    // Get donor sample using deterministic hash
    integer_t sample_perm = (sample_idx + feature_idx * 31 + tree_id * 17) % nsample;
    if (sample_perm == sample_idx) sample_perm = (sample_perm + 1) % nsample;
    
    // Traverse with permuted feature
    integer_t node_perm = traverse_with_permuted_feature_sparse(
        X_sparse, sample_idx, sample_perm, feature_idx,
        mdim, treemap, nodestatus, xbestsplit, bestvar,
        nnode, max_depth
    );
    
    // If terminal node changed, increment proximity importance
    // Use per-tree normalization (1/noob) - same as local variable importance
    if (node_orig != node_perm) {
        real_t inv_noob = 1.0f / static_cast<real_t>(noob);
        ::atomicAdd(&prox_imp[sample_idx * mdim + feature_idx], inv_noob);
    }
}

// ============================================================================
// Reduce kernel: local -> overall
// ============================================================================

__global__ void gpu_reduce_proximity_importance_kernel(
    const real_t* local_prox_imp,
    integer_t nsample,
    integer_t mdim,
    real_t* overall_prox_imp,
    integer_t* error_code
) {
    integer_t k = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (k >= mdim) return;
    
    // Sum across all samples for feature k
    real_t sum = 0.0f;
    integer_t count = 0;
    
    for (integer_t i = 0; i < nsample; i++) {
        real_t val = local_prox_imp[i * mdim + k];
        if (val > 0) {
            sum += val;
            count++;
        }
    }
    
    // Mean (handle zero count)
    overall_prox_imp[k] = (count > 0) ? sum / static_cast<real_t>(nsample) : 0.0f;
}

// ============================================================================
// Host functions
// ============================================================================

integer_t gpu_compute_proximity_importance_sparse(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_nodextr_orig,
    const integer_t* d_nin,
    const integer_t* d_cl,
    const integer_t* d_nodeclass,
    const integer_t* d_treemap,
    const integer_t* d_nodestatus,
    const real_t* d_xbestsplit,
    const integer_t* d_bestvar,
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t max_depth,
    integer_t tree_id,
    integer_t noob,
    real_t* d_local_prox_imp,
    bool is_regression,
    cudaStream_t stream
) {
    if (noob <= 0) return 0;  // No OOB samples, nothing to do
    
    // OPTIMIZED: Use 2D kernel - ONE launch per tree instead of mdim launches
    // Block: 16x16 = 256 threads (good occupancy)
    // Grid: ceil(nsample/16) x ceil(mdim/16)
    dim3 block_size(16, 16);  // 256 threads total
    dim3 grid_size(
        (nsample + block_size.x - 1) / block_size.x,
        (mdim + block_size.y - 1) / block_size.y
    );
    
    gpu_proximity_importance_sparse_kernel_2d<<<grid_size, block_size, 0, stream>>>(
            X_sparse,
            d_nodextr_orig,
            d_nin,
        d_cl,
        d_nodeclass,
            d_treemap,
            d_nodestatus,
            d_xbestsplit,
            d_bestvar,
            nsample,
            mdim,
            nnode,
            max_depth,
            tree_id,
        noob,
        d_local_prox_imp,
        is_regression
    );
    
    return 0;
}

integer_t gpu_finalize_proximity_importance_sparse(
    real_t* d_local_prox_imp,
    integer_t nsample,
    integer_t mdim,
    real_t* d_overall_prox_imp,
    cudaStream_t stream
) {
    // Allocate error code
    integer_t* d_error;
    cudaMalloc(&d_error, sizeof(integer_t));
    cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
    
    // Per-tree normalization is already done during accumulation (1/noob per tree)
    // No additional normalization needed - just compute overall (mean across samples)
    dim3 grid_reduce((mdim + 255) / 256);
    gpu_reduce_proximity_importance_kernel<<<grid_reduce, 256, 0, stream>>>(
        d_local_prox_imp,
        nsample,
        mdim,
        d_overall_prox_imp,
        d_error
    );
    
    // Sync and check error
    cudaStreamSynchronize(stream);
    integer_t h_error = 0;
    cudaMemcpy(&h_error, d_error, sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_error);
    return h_error;
}

} // namespace cuda
} // namespace rf

