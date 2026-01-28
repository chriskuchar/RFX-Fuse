/**
 * @file rf_proximity_importance.cuh
 * @brief GPU DENSE proximity importance header
 */

#ifndef RF_PROXIMITY_IMPORTANCE_CUH
#define RF_PROXIMITY_IMPORTANCE_CUH

#include "rf_types.hpp"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

/**
 * Compute proximity importance for one tree (GPU Dense version)
 * 
 * For each OOB sample where prediction was CORRECT, and for each feature,
 * checks if permuting that feature changes the terminal node.
 * Accumulates (1.0 / noob) for each change - same normalization as local variable importance.
 * 
 * This matches Breiman-Cutler methodology: only accumulate importance
 * for samples that were correctly classified OOB.
 * 
 * jtr (predicted class) is computed on-the-fly as: nodeclass[nodextr[n]]
 * 
 * @param d_x          Device dense data [nsample x mdim]
 * @param d_nodextr    Device terminal nodes [nsample]
 * @param d_nin        Device bootstrap multiplicities [nsample] (0 = OOB)
 * @param d_cl         Device true classes [nsample]
 * @param d_nodeclass  Device class per node [maxnode] (to compute jtr = nodeclass[nodextr])
 * @param d_treemap    Device tree structure [2 * maxnode]
 * @param d_nodestatus Device node status [maxnode]
 * @param d_xbestsplit Device split points [maxnode]
 * @param d_bestvar    Device split variables [maxnode]
 * @param nsample      Number of samples
 * @param mdim         Number of features
 * @param nnode        Number of nodes in this tree
 * @param maxnode      Maximum nodes (for bounds checking)
 * @param max_depth    Maximum traversal depth (safety limit)
 * @param tree_id      Tree index (for deterministic donor selection)
 * @param noob         Number of OOB samples in this tree (for normalization)
 * @param d_prox_imp   Device proximity importance [nsample x mdim] - accumulated
 * @param is_regression If true, skip correctness check (always accumulate for OOB)
 * @param stream       CUDA stream
 * @return 0 on success
 */
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
    bool is_regression = false,
    cudaStream_t stream = 0
);

/**
 * Count OOB samples for a tree (simple reduction on host or GPU)
 * 
 * @param d_nin        Device bootstrap multiplicities [nsample]
 * @param nsample      Number of samples
 * @param stream       CUDA stream
 * @return Number of OOB samples (nin == 0)
 */
integer_t gpu_count_oob_samples(
    const integer_t* d_nin,
    integer_t nsample,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace rf

#endif // RF_PROXIMITY_IMPORTANCE_CUH

