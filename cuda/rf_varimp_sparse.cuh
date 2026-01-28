/**
 * @file rf_varimp_sparse.cuh
 * @brief GPU sparse variable importance header
 * 
 * Matches CPU cpu_varimp_sparse exactly.
 * Uses CudaSparseMatrixCSR for data access.
 */

#ifndef RF_VARIMP_SPARSE_CUH
#define RF_VARIMP_SPARSE_CUH

#include "rf_types.hpp"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

/**
 * GPU kernel: Test tree with permuted feature (sparse version)
 * 
 * For each OOB sample, traverse tree using:
 * - Original values for all features except `mr`
 * - Permuted values for feature `mr`
 * 
 * Uses fixed-iteration traversal to avoid warp divergence.
 * 
 * @param X_sparse GPU sparse matrix
 * @param nsample Total number of samples
 * @param mdim Number of features
 * @param joob OOB sample indices [noob]
 * @param pjoob Permuted OOB indices [noob] - for accessing feature mr
 * @param noob Number of OOB samples
 * @param mr Feature being permuted
 * @param treemap Tree structure [2 * nnode]
 * @param nodestatus Node status (-1 = terminal)
 * @param xbestsplit Split values [nnode]
 * @param bestvar Split variables [nnode]
 * @param nodeclass Class at each node [nnode]
 * @param nnode Number of nodes
 * @param max_depth Maximum tree depth
 * @param jvr Output: prediction for each OOB sample [noob]
 * @param nodexvr Output: terminal node for each OOB sample [noob]
 * @param error_code Output: error code
 */
__global__ void gpu_testreeimp_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    integer_t nsample,
    integer_t mdim,
    const integer_t* joob,
    const integer_t* pjoob,
    integer_t noob,
    integer_t mr,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    const integer_t* nodeclass,
    integer_t nnode,
    integer_t max_depth,
    integer_t* jvr,
    integer_t* nodexvr,
    integer_t* error_code
);

/**
 * GPU kernel: Compute importance from permuted predictions
 * 
 * Compare original accuracy vs permuted accuracy.
 * Importance = decrease in accuracy when feature is permuted.
 * 
 * @param joob OOB sample indices [noob]
 * @param cl True class labels [nsample]
 * @param jtr Original predictions [nsample]
 * @param jvr Permuted predictions [noob]
 * @param nodextr Original terminal nodes [nsample]
 * @param nodexvr Permuted terminal nodes [noob]
 * @param tnodewt Node weights for casewise [nnode]
 * @param noob Number of OOB samples
 * @param mr Feature being measured
 * @param nnode Number of nodes
 * @param use_casewise Whether to use casewise weighting
 * @param avimp Output: accumulated importance [mdim] (atomicAdd)
 * @param qimpm Output: local importance [nsample * mdim] (atomicAdd, if impn=1)
 * @param impn Whether to compute local importance
 * @param error_code Output: error code
 */
__global__ void gpu_compute_importance_kernel(
    const integer_t* joob,
    const integer_t* cl,
    const integer_t* jtr,
    const integer_t* jvr,
    const integer_t* nodextr,
    const integer_t* nodexvr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t mr,
    integer_t mdim,
    integer_t nnode,
    bool use_casewise,
    real_t* avimp,
    real_t* qimpm,
    integer_t impn,
    integer_t* error_code
);

/**
 * Host function: Run sparse variable importance for one tree
 * 
 * Processes all features in parallel (one kernel per feature to avoid divergence).
 * 
 * @param X_sparse GPU sparse matrix
 * @param d_cl True class labels
 * @param d_jtr Original OOB predictions
 * @param d_nin Bootstrap frequency
 * @param d_nodextr Original terminal nodes
 * @param d_treemap Tree structure
 * @param d_nodestatus Node status
 * @param d_xbestsplit Split values
 * @param d_bestvar Split variables
 * @param d_nodeclass Node class predictions
 * @param d_tnodewt Node weights
 * @param nsample Number of samples
 * @param mdim Number of features
 * @param nnode Number of nodes
 * @param max_depth Maximum tree depth
 * @param use_casewise Use casewise weighting
 * @param impn Compute local importance
 * @param d_avimp Output: feature importance [mdim]
 * @param d_qimpm Output: local importance [nsample * mdim]
 * @param d_rng_states Random number generator states
 * @param stream CUDA stream
 * @return Error code
 */
integer_t gpu_varimp_sparse(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const integer_t* d_jtr,
    const integer_t* d_nin,
    const integer_t* d_nodextr,
    const integer_t* d_treemap,
    const integer_t* d_nodestatus,
    const real_t* d_xbestsplit,
    const integer_t* d_bestvar,
    const integer_t* d_nodeclass,
    const real_t* d_tnodewt,
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t max_depth,
    bool use_casewise,
    integer_t impn,
    real_t* d_avimp,
    real_t* d_qimp,      // Per-sample original correct weights (for localimp)
    real_t* d_qimpm,
    curandState* d_rng_states,
    cudaStream_t stream
);

/**
 * Host function: Run sparse variable importance for REGRESSION
 * 
 * MSE-based importance: measures INCREASE in MSE when feature is permuted.
 * Matches gpu_varimp_regression kernels used for dense regression.
 * 
 * @param X_sparse GPU sparse matrix
 * @param d_y_pred OOB predictions (tnodewt[nodextr[n]] for each sample)
 * @param d_y_true True regression targets
 * @param d_nin Bootstrap frequency
 * @param d_nodextr Original terminal nodes
 * @param d_treemap Tree structure
 * @param d_nodestatus Node status
 * @param d_xbestsplit Split values
 * @param d_bestvar Split variables
 * @param d_tnodewt Node weights (mean y for terminal nodes)
 * @param nsample Number of samples
 * @param mdim Number of features
 * @param nnode Number of nodes
 * @param max_depth Maximum tree depth
 * @param use_casewise Use casewise weighting
 * @param impn Compute local importance
 * @param d_avimp Output: feature importance [mdim]
 * @param d_qimp Output: per-sample original MSE [nsample]
 * @param d_qimpm Output: local importance [nsample * mdim]
 * @param d_rng_states Random number generator states
 * @param stream CUDA stream
 * @return Error code
 */
integer_t gpu_varimp_sparse_regression(
    const CudaSparseMatrixCSR& X_sparse,
    const real_t* d_y_pred,
    const real_t* d_y_true,
    const integer_t* d_nin,
    const integer_t* d_nodextr,
    const integer_t* d_treemap,
    const integer_t* d_nodestatus,
    const real_t* d_xbestsplit,
    const integer_t* d_bestvar,
    const real_t* d_nodepred,   // Terminal node predictions (mean y)
    const real_t* d_tnodewt,    // Terminal node weights (mean bootstrap weight)
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t max_depth,
    bool use_casewise,
    integer_t impn,
    real_t* d_avimp,
    real_t* d_qimp,
    real_t* d_qimpm,
    curandState* d_rng_states,
    cudaStream_t stream
);

} // namespace cuda
} // namespace rf

#endif // RF_VARIMP_SPARSE_CUH

