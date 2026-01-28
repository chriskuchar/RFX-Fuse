/**
 * @file rf_proximity_importance.cu
 * @brief GPU DENSE proximity importance implementation
 * 
 * Proximity importance = fraction of trees where permuting a feature
 * changes the terminal node for OOB samples WITH CORRECT PREDICTIONS.
 * 
 * This matches Breiman-Cutler methodology for local variable importance:
 * - Only accumulate for samples that were correctly classified OOB
 * - Normalize per-tree by (1.0 / noob) - same as local importance
 * 
 * Key design:
 * - One kernel launch per tree (after tree is grown)
 * - Each thread processes one (sample, feature) pair
 * - Uses dense matrix access: x[sample * mdim + feature]
 * - Computes jtr on-the-fly: jtr[n] = nodeclass[nodextr[n]]
 * - Only accumulates when jtr[n] == cl[n] (correct prediction)
 * - Accumulates (1.0 / noob) per change, not raw counts
 */

#include "rf_proximity_importance.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

// ============================================================================
// Device function: Traverse tree with permuted feature (DENSE version)
// ============================================================================

/**
 * Traverse tree using dense data, with one feature permuted.
 * 
 * For feature k: use donor sample's value instead of original
 * For other features: use original sample's value
 * 
 * Returns terminal node reached.
 */
__device__ integer_t traverse_with_permuted_feature_dense(
    const real_t* x,           // Dense data [nsample x mdim] row-major
    integer_t sample_orig,
    integer_t sample_donor,    // Donor sample for permuted feature
    integer_t feature_k,
    integer_t mdim,
    integer_t nsample,
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
        
        // Terminal node? (nodestatus == -1 means terminal)
        if (nodestatus[kt] == -1) {
            return kt;
        }
        
        // Get split variable
        integer_t m = bestvar[kt];
        if (m < 0 || m >= mdim) break;
        
        // Get feature value - ROW-MAJOR: x[sample * mdim + feature]
        real_t xmn;
        if (m == feature_k) {
            // Use donor sample for this feature
            xmn = x[sample_donor * mdim + m];
        } else {
            // Use original sample
            xmn = x[sample_orig * mdim + m];
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
// Proximity importance kernel (processes all samples x all features)
// ============================================================================

/**
 * GPU Dense Proximity Importance Kernel
 * 
 * For each OOB sample n WITH CORRECT PREDICTION and feature k:
 *   - Traverse tree with permuted feature value (from donor sample)
 *   - If terminal node differs from original, accumulate (1.0 / noob)
 * 
 * This matches Breiman-Cutler local importance methodology:
 *   - Only count correct predictions
 *   - Normalize per-tree by noob (total OOB count for this tree)
 * 
 * jtr is computed on-the-fly: jtr[n] = nodeclass[nodextr[n]]
 * 
 * Grid: (num_samples * num_features + blockDim.x - 1) / blockDim.x
 * Each thread handles one (sample, feature) pair.
 */
__global__ void gpu_proximity_importance_dense_kernel(
    const real_t* x,           // Dense data [nsample x mdim]
    const integer_t* nodextr,  // Terminal node for each sample [nsample]
    const integer_t* nin,      // Bootstrap multiplicity [nsample] (0 = OOB)
    const integer_t* cl,       // True classes [nsample]
    const integer_t* nodeclass, // Class per node [maxnode] - for computing jtr
    const integer_t* treemap,  // Tree structure [2 * maxnode]
    const integer_t* nodestatus, // Node status [maxnode]
    const real_t* xbestsplit,  // Split points [maxnode]
    const integer_t* bestvar,  // Split variables [maxnode]
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t maxnode,
    integer_t max_depth,
    integer_t tree_id,         // For deterministic donor selection
    integer_t noob,            // Total OOB count for this tree (for normalization)
    real_t* prox_imp,          // Output [nsample x mdim] - atomically updated
    bool is_regression         // If true, skip correctness check
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t total_pairs = nsample * mdim;
    
    if (tid >= total_pairs) return;
    
    // Decode (sample, feature) from thread id
    integer_t n = tid / mdim;  // Sample index
    integer_t k = tid % mdim;  // Feature index
    
    // Only process OOB samples (nin == 0)
    if (nin[n] != 0) return;
    
    // Get original terminal node
    integer_t original_node = nodextr[n];
    if (original_node < 0 || original_node >= nnode) return;
    
    // Compute jtr (predicted class) on-the-fly
    // For classification/unsupervised: jtr = nodeclass[nodextr]
    // For regression: skip correctness check entirely
    if (!is_regression) {
        integer_t jtr = nodeclass[original_node];
        if (jtr != cl[n]) return;  // Skip incorrect predictions
    }
    
    // Get donor sample for permutation (deterministic hash for reproducibility)
    // Same formula as CPU: (n + k * 31 + tree_id * 17) % nsample
    integer_t donor = (n + k * 31 + tree_id * 17) % nsample;
    if (donor == n) donor = (donor + 1) % nsample;
    
    // Traverse tree with permuted feature value
    integer_t permuted_node = traverse_with_permuted_feature_dense(
        x, n, donor, k,
        mdim, nsample,
        treemap, nodestatus, xbestsplit, bestvar,
        nnode, max_depth
    );
    
    // If terminal node changed, increment proximity importance
    // Use per-tree normalization (1.0 / noob) - same as local variable importance
    if (permuted_node != original_node && noob > 0) {
        // Row-major: prox_imp[sample * mdim + feature]
        real_t contribution = 1.0f / static_cast<real_t>(noob);
        ::atomicAdd(&prox_imp[n * mdim + k], contribution);
    }
}

// ============================================================================
// OOB counting kernel (simple reduction)
// ============================================================================

/**
 * Count OOB samples (nin == 0) using parallel reduction
 */
__global__ void count_oob_kernel(
    const integer_t* nin,
    integer_t nsample,
    integer_t* result
) {
    extern __shared__ integer_t sdata[];
    
    integer_t tid = threadIdx.x;
    integer_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread counts its element
    sdata[tid] = (i < nsample && nin[i] == 0) ? 1 : 0;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (integer_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        ::atomicAdd(result, sdata[0]);
    }
}

// ============================================================================
// Host functions
// ============================================================================

integer_t gpu_count_oob_samples(
    const integer_t* d_nin,
    integer_t nsample,
    cudaStream_t stream
) {
    // Allocate device result
    integer_t* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(integer_t));
    cudaMemset(d_result, 0, sizeof(integer_t));
    
    // Launch reduction kernel
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    size_t shared_size = block_size.x * sizeof(integer_t);
    
    count_oob_kernel<<<grid_size, block_size, shared_size, stream>>>(
        d_nin, nsample, d_result
    );
    
    // Copy result back
    integer_t h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

integer_t gpu_compute_proximity_importance_dense(
    const real_t* d_x,
    const integer_t* d_nodextr,
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
    integer_t maxnode,
    integer_t max_depth,
    integer_t tree_id,
    integer_t noob,
    real_t* d_prox_imp,
    bool is_regression,
    cudaStream_t stream
) {
    if (noob <= 0) return 0;  // No OOB samples, nothing to do
    
    // Grid/block dimensions
    integer_t total_pairs = nsample * mdim;
    dim3 block_size(256);
    dim3 grid_size((total_pairs + block_size.x - 1) / block_size.x);
    
    // Launch kernel
    gpu_proximity_importance_dense_kernel<<<grid_size, block_size, 0, stream>>>(
        d_x, d_nodextr, d_nin, d_cl, d_nodeclass,
        d_treemap, d_nodestatus, d_xbestsplit, d_bestvar,
        nsample, mdim, nnode, maxnode, max_depth, tree_id, noob,
        d_prox_imp, is_regression
    );
    
    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        return 1;
    }
    
    return 0;
}

} // namespace cuda
} // namespace rf

