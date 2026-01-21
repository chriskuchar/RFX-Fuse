/**
 * @file rf_proximity_importance_sparse.cuh
 * @brief GPU sparse proximity importance header
 * 
 * Proximity importance measures which features affect sample proximity.
 * For each sample i and feature k (where OOB AND prediction was CORRECT):
 *   prox_imp[i, k] = sum of (1/noob_t) across trees where permuting feature k
 *                    changed the terminal node for sample i
 * 
 * This matches Breiman-Cutler methodology for local variable importance:
 * - Only accumulate for samples that were correctly classified OOB
 * - Normalize per-tree by (1.0 / noob) - same as local importance
 * 
 * Output:
 *   - Local: (n_samples, n_features) - per-sample importance
 *   - Overall: (n_features,) - mean across samples
 */

#ifndef RF_PROXIMITY_IMPORTANCE_SPARSE_CUH
#define RF_PROXIMITY_IMPORTANCE_SPARSE_CUH

#include "rf_types.hpp"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

/**
 * GPU kernel: Compute proximity importance for ONE feature
 * 
 * For each OOB sample WITH CORRECT PREDICTION, permute feature k and 
 * check if terminal node changes.
 * Accumulates (1.0 / noob) per change - same normalization as local variable importance.
 */
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
);

/**
 * GPU kernel: Reduce local proximity importance to overall
 * 
 * overall[k] = mean(local[:, k])
 */
__global__ void gpu_reduce_proximity_importance_kernel(
    const real_t* local_prox_imp,
    integer_t nsample,
    integer_t mdim,
    real_t* overall_prox_imp,
    integer_t* error_code
);

/**
 * Host function: Compute local proximity importance for all features (ONE tree)
 * 
 * Uses 2D kernel for efficiency. Accumulates (1/noob) per change.
 * Call this for EACH tree in the forest, then call gpu_finalize_proximity_importance_sparse.
 * 
 * @param X_sparse GPU sparse matrix
 * @param d_nodextr_orig Original terminal nodes for this tree
 * @param d_nin Bootstrap frequency for this tree
 * @param d_cl True classes [nsample]
 * @param d_nodeclass Node classes [maxnode] - for computing jtr on-the-fly
 * @param d_treemap Tree structure for this tree
 * @param d_nodestatus Node status for this tree
 * @param d_xbestsplit Split values for this tree
 * @param d_bestvar Split variables for this tree
 * @param nsample Number of samples
 * @param mdim Number of features
 * @param nnode Number of nodes
 * @param max_depth Maximum tree depth
 * @param tree_id Tree index for deterministic donor selection
 * @param noob Number of OOB samples in this tree (for normalization)
 * @param d_local_prox_imp Output: local prox imp [nsample * mdim] (accumulated via atomicAdd)
 * @param is_regression If true, skip correctness check
 * @param stream CUDA stream
 * @return Error code
 */
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
    bool is_regression = false,
    cudaStream_t stream = 0
);

/**
 * Host function: Finalize and reduce to overall importance
 * 
 * Since per-tree normalization is already done, this just computes
 * the mean across samples for overall importance.
 * 
 * @param d_local_prox_imp Local prox imp [nsample * mdim]
 * @param nsample Number of samples
 * @param mdim Number of features
 * @param d_overall_prox_imp Output: overall prox imp [mdim]
 * @param stream CUDA stream
 * @return Error code
 */
integer_t gpu_finalize_proximity_importance_sparse(
    real_t* d_local_prox_imp,
    integer_t nsample,
    integer_t mdim,
    real_t* d_overall_prox_imp,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace rf

#endif // RF_PROXIMITY_IMPORTANCE_SPARSE_CUH

