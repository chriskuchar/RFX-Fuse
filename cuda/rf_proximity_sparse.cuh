/**
 * @file rf_proximity_sparse.cuh
 * @brief GPU sparse proximity computation header
 * 
 * Proximity uses terminal nodes (already computed from sparse data).
 * The proximity kernel itself doesn't need sparse matrix access.
 * 
 * Supports:
 * - Full proximity matrix (for small datasets)
 * - QLoRA low-rank approximation (for large datasets)
 */

#ifndef RF_PROXIMITY_SPARSE_CUH
#define RF_PROXIMITY_SPARSE_CUH

#include "rf_types.hpp"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

/**
 * GPU kernel: Update proximity matrix from terminal nodes
 * 
 * For each pair (i, j) where i is OOB and they share terminal node:
 *   prox[i, j] += weight
 * 
 * @param nodextr Terminal node for each sample [nsample]
 * @param nin Bootstrap frequency [nsample] - 0 = OOB
 * @param nodestatus Node status [nnode]
 * @param nsample Number of samples
 * @param nnode Number of nodes
 * @param use_casewise Use casewise weighting
 * @param prox Output: proximity matrix [nsample * nsample] (atomicAdd)
 * @param error_code Output: error code
 */
__global__ void gpu_update_proximity_kernel(
    const integer_t* nodextr,
    const integer_t* nin,
    const integer_t* nodestatus,
    integer_t nsample,
    integer_t nnode,
    bool use_casewise,
    dp_t* prox,
    integer_t* error_code
);

/**
 * GPU kernel: Update low-rank proximity factors (QLoRA)
 * 
 * For each OOB sample i sharing terminal node with in-bag sample j:
 *   A[i, col] += weight
 *   B[j, col] += weight
 * 
 * col = tree_id % rank (round-robin column selection)
 * 
 * @param nodextr Terminal node for each sample [nsample]
 * @param nin Bootstrap frequency [nsample]
 * @param nodestatus Node status [nnode]
 * @param nsample Number of samples
 * @param nnode Number of nodes
 * @param rank Low-rank dimension
 * @param tree_id Current tree ID (for column selection)
 * @param use_casewise Use casewise weighting
 * @param A Output: left factors [nsample * rank] (atomicAdd)
 * @param B Output: right factors [nsample * rank] (atomicAdd)
 * @param error_code Output: error code
 */
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
);

/**
 * GPU kernel: Finalize proximity (normalize by tree count)
 * 
 * prox[i, j] /= ntree
 */
__global__ void gpu_finalize_proximity_kernel(
    dp_t* prox,
    integer_t nsample,
    integer_t ntree,
    integer_t* error_code
);

/**
 * GPU kernel: Finalize low-rank factors
 * 
 * A /= ntree, B /= ntree
 */
__global__ void gpu_finalize_lowrank_kernel(
    dp_t* A,
    dp_t* B,
    integer_t nsample,
    integer_t rank,
    integer_t ntree,
    integer_t* error_code
);

/**
 * Host function: Update proximity after one tree (sparse path)
 * 
 * Uses terminal nodes computed from sparse tree traversal.
 */
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
    dp_t* d_prox,      // Full proximity (if !use_lowrank)
    dp_t* d_A,         // Left factors (if use_lowrank)
    dp_t* d_B,         // Right factors (if use_lowrank)
    cudaStream_t stream
);

} // namespace cuda
} // namespace rf

#endif // RF_PROXIMITY_SPARSE_CUH

