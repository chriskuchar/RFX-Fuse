/**
 * @file rf_testreebag_sparse.cuh
 * @brief GPU sparse OOB testing - find terminal nodes for sparse input
 * 
 * Matches CPU cpu_testreebag_sparse exactly.
 * Uses level-synchronous traversal to avoid warp divergence.
 */

#ifndef RF_TESTREEBAG_SPARSE_CUH
#define RF_TESTREEBAG_SPARSE_CUH

#include "rf_types.hpp"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

/**
 * GPU kernel: Find terminal node for each sample using sparse matrix
 * 
 * Uses fixed-iteration traversal to avoid warp divergence.
 * All threads process same number of levels, inactive threads skip work.
 * 
 * @param X_sparse GPU sparse matrix
 * @param samples Sample indices to process
 * @param n_samples Number of samples
 * @param treemap Tree structure [2, maxnode] - left/right children
 * @param nodestatus Node status (-1 = terminal)
 * @param xbestsplit Split values
 * @param bestvar Split variable indices
 * @param nodeclass Class predictions for terminal nodes
 * @param nnode Number of nodes in tree
 * @param max_depth Maximum tree depth (for fixed iteration)
 * @param terminal_nodes Output: terminal node for each sample
 * @param predictions Output: prediction for each sample
 * @param error_code Output: error code (0 = success)
 */
__global__ void gpu_find_terminal_node_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    const integer_t* samples,
    integer_t n_samples,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    const integer_t* nodeclass,
    integer_t nnode,
    integer_t max_depth,
    integer_t* terminal_nodes,
    integer_t* predictions,
    integer_t* error_code
);

/**
 * GPU kernel: Compute OOB predictions and update counts
 * 
 * For classification: accumulate votes per class
 * 
 * @param terminal_nodes Terminal node for each sample
 * @param nodeclass Class at each terminal node
 * @param nin Bootstrap frequency (0 = OOB)
 * @param n_samples Number of samples
 * @param nclass Number of classes
 * @param oob_votes Output: [nsample, nclass] vote counts
 * @param oob_counts Output: [nsample] number of OOB votes
 * @param error_code Output: error code
 */
__global__ void gpu_accumulate_oob_votes_kernel(
    const integer_t* terminal_nodes,
    const integer_t* nodeclass,
    const integer_t* nin,
    integer_t n_samples,
    integer_t nclass,
    real_t* oob_votes,
    integer_t* oob_counts,
    integer_t* error_code
);

/**
 * Host function: Run OOB testing for sparse classification
 * 
 * @param X_sparse GPU sparse matrix
 * @param treemap Tree structure
 * @param nodestatus Node status array
 * @param xbestsplit Split values
 * @param bestvar Split variables
 * @param nodeclass Node class predictions
 * @param nin Bootstrap frequency
 * @param nnode Number of nodes
 * @param nsample Number of samples
 * @param nclass Number of classes
 * @param max_depth Maximum tree depth
 * @param terminal_nodes Output: terminal nodes
 * @param oob_votes Output: OOB vote accumulator
 * @param oob_counts Output: OOB count accumulator
 * @param stream CUDA stream
 * @return Error code
 */
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
);

/**
 * Host function: Adjust casewise votes on GPU (no CPU round-trip)
 * 
 * For casewise mode, adjusts votes by:
 * - Undoing the +1 unweighted vote from gpu_accumulate_oob_votes_kernel
 * - Adding weighted vote based on tnodewt (mean bootstrap freq in terminal node)
 * 
 * @param d_terminal_nodes Terminal node for each sample
 * @param d_nodeclass Class at each node
 * @param d_tnodewt Node weights [nnode]
 * @param d_nin Bootstrap frequency (0 = OOB)
 * @param nsample Number of samples
 * @param nclass Number of classes
 * @param nnode Number of nodes
 * @param d_oob_votes [nsample * nclass] vote counts (modified in-place)
 * @param d_oob_counts [nsample] OOB counts (modified in-place)
 * @param stream CUDA stream
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
);

} // namespace cuda
} // namespace rf

#endif // RF_TESTREEBAG_SPARSE_CUH

