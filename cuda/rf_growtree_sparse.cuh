/**
 * @file rf_growtree_sparse.cuh
 * @brief GPU sparse tree growing header
 * 
 * Matches CPU cpu_growtree_sparse exactly.
 * Uses CudaSparseMatrixCSR for feature access during split finding.
 * 
 * Key differences from dense GPU growtree:
 * - Uses sparse matrix access for split evaluation
 * - Maintains row-major access pattern
 * - All other tree logic (splitting, partitioning) is identical
 */

#ifndef RF_GROWTREE_SPARSE_CUH
#define RF_GROWTREE_SPARSE_CUH

#include "rf_types.hpp"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

/**
 * GPU kernel: Find best split for a node (sparse version)
 * 
 * For each candidate feature, evaluates all possible split points
 * using sparse matrix access.
 * 
 * @param X_sparse GPU sparse matrix
 * @param sample_indices Indices of samples in this node
 * @param n_samples_node Number of samples in node
 * @param cl Class labels [nsample_total]
 * @param nin Bootstrap frequency [nsample_total]
 * @param mdim Number of features
 * @param nclass Number of classes
 * @param mtry Number of features to try
 * @param d_rng_states Random states for feature selection
 * @param best_feature Output: best split feature
 * @param best_threshold Output: best split threshold
 * @param best_impurity Output: best impurity reduction
 * @param error_code Output: error code
 */
__global__ void gpu_find_best_split_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    const integer_t* sample_indices,
    integer_t n_samples_node,
    const integer_t* cl,
    const real_t* win,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    curandState* d_rng_states,
    integer_t* best_feature,
    real_t* best_threshold,
    real_t* best_impurity,
    integer_t* error_code
);

/**
 * GPU kernel: Partition samples based on split (sparse version)
 * 
 * Splits samples into left/right children based on split decision.
 * 
 * @param X_sparse GPU sparse matrix
 * @param sample_indices Samples to partition [n_samples]
 * @param n_samples Number of samples
 * @param split_feature Feature used for split
 * @param split_threshold Split threshold
 * @param left_samples Output: samples going left
 * @param right_samples Output: samples going right
 * @param n_left Output: number going left
 * @param n_right Output: number going right
 * @param error_code Output: error code
 */
__global__ void gpu_partition_samples_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    const integer_t* sample_indices,
    integer_t n_samples,
    integer_t split_feature,
    real_t split_threshold,
    integer_t* left_samples,
    integer_t* right_samples,
    integer_t* n_left,
    integer_t* n_right,
    integer_t* error_code
);

/**
 * Host function: Grow a single tree using sparse data
 * 
 * Matches cpu_growtree_sparse algorithm exactly.
 * Uses breadth-first node processing to avoid warp divergence.
 * 
 * @param X_sparse GPU sparse matrix
 * @param d_cl Class labels [nsample]
 * @param d_nin Bootstrap frequency [nsample]
 * @param d_win Bootstrap weights [nsample]
 * @param nsample Number of samples
 * @param mdim Number of features
 * @param nclass Number of classes
 * @param mtry Features to try per split
 * @param maxnode Maximum nodes
 * @param min_node_size Minimum samples per node
 * @param d_rng_states Random number states
 * @param d_nodestatus Output: node status [maxnode]
 * @param d_bestvar Output: split variables [maxnode]
 * @param d_xbestsplit Output: split values [maxnode]
 * @param d_treemap Output: tree structure [2 * maxnode]
 * @param d_nodeclass Output: class at each node [maxnode]
 * @param d_tnodewt Output: node weights [maxnode]
 * @param nnode Output: actual number of nodes
 * @param stream CUDA stream
 * @return Error code
 */
integer_t gpu_growtree_sparse(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const integer_t* d_nin,
    const real_t* d_win,
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    curandState* d_rng_states,
    integer_t tree_id,  // Tree index for RNG state
    integer_t* d_nodestatus,
    integer_t* d_bestvar,
    real_t* d_xbestsplit,
    integer_t* d_treemap,
    integer_t* d_nodeclass,
    real_t* d_tnodewt,
    integer_t& nnode,
    cudaStream_t stream
);

/**
 * Host function: Grow multiple trees in parallel (sparse batch)
 * 
 * For maximum GPU utilization, grows multiple trees concurrently.
 * 
 * @param num_trees Number of trees to grow
 * @param X_sparse GPU sparse matrix (shared across trees)
 * @param ... (same parameters as single tree, batched)
 */
integer_t gpu_growtree_batch_sparse(
    integer_t num_trees,
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const integer_t* d_nin_batch,  // [num_trees * nsample]
    const real_t* d_win_batch,     // [num_trees * nsample]
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    curandState* d_rng_states,
    // Batched outputs
    integer_t* d_nodestatus_batch,
    integer_t* d_bestvar_batch,
    real_t* d_xbestsplit_batch,
    integer_t* d_treemap_batch,
    integer_t* d_nodeclass_batch,
    real_t* d_tnodewt_batch,
    integer_t* nnode_batch,
    cudaStream_t stream
);

/**
 * Host function: Grow all trees in parallel (sparse batch - matches GPU Dense architecture)
 * 
 * Uses single kernel launch for ALL trees - same architecture as GPU Dense.
 * Each CUDA block handles one complete tree.
 * No host-device synchronization during tree growing.
 * 
 * This is the FAST path for Netflix-scale sparse training.
 * 
 * @param task_type 0=classification (Gini), 1=regression (MSE)
 * @param y_regression Regression targets [nsample] (nullptr for classification)
 * @param d_tnodewt_all Output: node predictions for regression [num_trees * maxnode]
 */
integer_t gpu_growtree_sparse_parallel_batch(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const real_t* d_y_regression,  // Regression targets (nullptr for classification)
    const integer_t* d_nin_all,
    const real_t* d_win_all,
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    integer_t num_trees,
    integer_t task_type,  // 0=classification, 1=regression
    curandState* d_rng_states,
    integer_t* d_nodestatus_all,
    integer_t* d_bestvar_all,
    real_t* d_xbestsplit_all,
    integer_t* d_treemap_all,
    integer_t* d_nodeclass_all,
    real_t* d_tnodewt_all,  // Node predictions for regression
    integer_t* d_nnode_all,
    cudaStream_t stream
);

// ============================================================================
// PRE-SORTED INDICES FOR SPARSE (Eliminates O(n log n) sorting per node)
// ============================================================================

/**
 * Pre-sort all features once before tree growing.
 * sorted_indices[feature * nsample + rank] = sample with rank-th smallest value
 * 
 * @param X_sparse GPU sparse matrix
 * @param nsample Number of samples
 * @param mdim Number of features
 * @param sorted_indices Output: [mdim × nsample] pre-sorted indices
 * @param stream CUDA stream
 */
void gpu_presort_sparse_features(
    const CudaSparseMatrixCSR& X_sparse,
    integer_t nsample,
    integer_t mdim,
    integer_t* sorted_indices,
    cudaStream_t stream
);

// ============================================================================
// HISTOGRAM BINNING FOR SPARSE (O(256) split finding instead of O(n log n))
// ============================================================================

/**
 * Sparse histogram data structure
 * Stores binned values at same positions as CSR data array
 */
struct SparseHistogramData {
    uint8_t* d_X_binned_values;      // [nnz] binned non-zero values
    real_t* d_bin_edges_all;         // [mdim × (max_bins+1)] bin edges
    integer_t* d_n_bins_per_feature; // [mdim] actual bins per feature
    integer_t max_bins;              // Typically 256
    integer_t nnz;                   // Number of non-zeros
    integer_t mdim;                  // Number of features
    bool allocated;
    
    SparseHistogramData() : d_X_binned_values(nullptr), d_bin_edges_all(nullptr),
                            d_n_bins_per_feature(nullptr), max_bins(256), 
                            nnz(0), mdim(0), allocated(false) {}
    
    void allocate(integer_t nnz_, integer_t mdim_, integer_t max_bins_);
    void free();
};

/**
 * Compute quantile bin edges for sparse data
 * Uses only non-zero values for binning (zeros get bin 0)
 * 
 * @param X_sparse GPU sparse matrix
 * @param max_bins Maximum number of bins (typically 256)
 * @param hist_data Output: histogram data structure
 * @param stream CUDA stream
 */
void gpu_compute_sparse_bin_edges(
    const CudaSparseMatrixCSR& X_sparse,
    integer_t max_bins,
    SparseHistogramData& hist_data,
    cudaStream_t stream
);

/**
 * Bin sparse data values into histogram bins
 * Each non-zero value gets mapped to a bin [0, max_bins)
 * 
 * @param X_sparse GPU sparse matrix
 * @param hist_data Histogram data with bin edges
 * @param stream CUDA stream
 */
void gpu_bin_sparse_data(
    const CudaSparseMatrixCSR& X_sparse,
    SparseHistogramData& hist_data,
    cudaStream_t stream
);

/**
 * Extended parallel batch tree growing with pre-sort and histogram support
 * 
 * @param use_presort Use pre-sorted indices (can be nullptr if false)
 * @param sorted_indices Pre-sorted feature indices [mdim × nsample]
 * @param use_histogram Use histogram binning (can be nullptr if false)
 * @param hist_data Histogram data structure
 */
integer_t gpu_growtree_sparse_parallel_batch_v2(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const real_t* d_y_regression,
    const integer_t* d_nin_all,
    const real_t* d_win_all,
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    integer_t num_trees,
    integer_t task_type,
    curandState* d_rng_states,
    // Pre-sort parameters
    bool use_presort,
    const integer_t* sorted_indices,
    // Histogram parameters
    bool use_histogram,
    const SparseHistogramData* hist_data,
    // Outputs
    integer_t* d_nodestatus_all,
    integer_t* d_bestvar_all,
    real_t* d_xbestsplit_all,
    integer_t* d_treemap_all,
    integer_t* d_nodeclass_all,
    real_t* d_tnodewt_all,
    integer_t* d_nnode_all,
    cudaStream_t stream
);

} // namespace cuda
} // namespace rf

#endif // RF_GROWTREE_SPARSE_CUH

