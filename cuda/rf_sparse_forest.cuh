/**
 * @file rf_sparse_forest.cuh
 * @brief GPU sparse random forest orchestration
 * 
 * Main entry point for GPU sparse classification/regression/unsupervised.
 * Coordinates all sparse GPU kernels:
 * - Tree growing (rf_growtree_sparse)
 * - OOB testing (rf_testreebag_sparse)
 * - Variable importance (rf_varimp_sparse)
 * - Proximity importance (rf_proximity_importance_sparse)
 * 
 * NOTE: GPU sparse proximity matrix removed. Use leaf_assignments for
 * on-demand proximity computation - more memory efficient for Netflix-scale.
 */

#ifndef RF_SPARSE_FOREST_CUH
#define RF_SPARSE_FOREST_CUH

#include "rf_types.hpp"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include "rf_growtree_sparse.cuh"
#include "rf_testreebag_sparse.cuh"
#include "rf_varimp_sparse.cuh"
#include "rf_proximity_importance_sparse.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <functional>

namespace rf {
namespace cuda {

/**
 * GPU sparse forest configuration
 */
struct GpuSparseForestConfig {
    integer_t nsample;
    integer_t mdim;
    integer_t nclass;
    integer_t ntree;
    integer_t mtry;
    integer_t maxnode;
    integer_t min_node_size;
    integer_t rank;             // For QLoRA
    integer_t max_depth;        // For tree traversal safety
    integer_t task_type;        // 0=CLASSIFICATION, 1=REGRESSION, 2=UNSUPERVISED
    bool compute_importance;
    bool compute_local_importance;
    bool compute_proximity;
    bool compute_proximity_importance;
    bool compute_leaf_assignments;  // Store terminal node per sample per tree (for on-demand proximity)
    bool use_casewise;
    bool use_qlora;
    integer_t seed;
    // Histogram binning parameters (for XGBoost-like speed)
    bool use_histogram = false;     // Use histogram binning for split finding
    integer_t n_bins = 256;         // Number of bins for histogram (default: 256)
    // Batching parameters (for memory efficiency with large datasets)
    integer_t batch_size = 0;       // Trees per batch (0 = auto, -1 = all at once)
    // Unsupervised: number of real samples (0 = all real)
    integer_t n_real_samples = 0;
};

/**
 * Host function: Train sparse random forest on GPU
 * 
 * Complete training pipeline:
 * 1. Initialize CUDA resources
 * 2. Upload sparse matrix
 * 3. For each tree:
 *    a. Bootstrap sample
 *    b. Grow tree (sparse)
 *    c. Test OOB samples (sparse)
 *    d. Compute importance (if requested)
 *    e. Store leaf assignments (if requested)
 *    f. Compute proximity importance (if requested)
 * 4. Finalize (normalize importances)
 * 5. Download results
 * 
 * NOTE: GPU sparse proximity matrix removed. Use h_leaf_assignments for
 * on-demand proximity computation - more memory efficient for Netflix-scale.
 * 
 * @param h_X_sparse Host sparse matrix
 * @param h_y Class labels (classification) or values (regression)
 * @param config Configuration
 * @param h_nodestatus Output: node status [ntree * maxnode]
 * @param h_bestvar Output: split variables [ntree * maxnode]
 * @param h_xbestsplit Output: split values [ntree * maxnode]
 * @param h_treemap Output: tree structure [ntree * 2 * maxnode]
 * @param h_nodeclass Output: node classes [ntree * maxnode]
 * @param h_importance Output: feature importance [mdim]
 * @param h_local_importance Output: local importance [nsample * mdim]
 * @param h_prox_importance Output: proximity importance [nsample * mdim]
 * @param h_leaf_assignments Output: terminal node per sample per tree [ntree * nsample]
 * @return Error code
 */
integer_t train_sparse_forest_gpu(
    const SparseMatrixCSR& h_X_sparse,
    const integer_t* h_y,           // Class labels (classification) or scaled int (regression fallback)
    const real_t* h_y_regression,   // Regression targets (nullptr for classification)
    const GpuSparseForestConfig& config,
    // Tree structure outputs (pre-allocated by caller)
    integer_t* h_nodestatus,
    integer_t* h_bestvar,
    real_t* h_xbestsplit,
    integer_t* h_treemap,
    integer_t* h_nodeclass,
    integer_t* h_nnode,
    // OOB vote outputs (for compute_oob_predictions)
    real_t* h_q,           // [nsample * nclass] - OOB vote counts per class
    integer_t* h_nout,     // [nsample] - OOB count per sample
    real_t* h_oob_predictions_regression,  // [nsample] - Regression OOB predictions (sum, divide by nout)
    // Regression node predictions (for predict() on new data)
    real_t* h_nodepred,    // [ntree * maxnode] - Terminal node predictions (mean y) for regression
    // Importance outputs
    real_t* h_importance,
    real_t* h_qimp,        // [nsample] - Per-sample original correct weights (for localimp)
    real_t* h_local_importance,
    // Proximity importance outputs
    real_t* h_prox_importance,
    real_t* h_overall_prox_importance,
    // Leaf assignments (for on-demand proximity computation)
    // This replaces full proximity matrix - compute proximity on-demand from leaves
    int16_t* h_leaf_assignments,  // [ntree * nsample] - terminal node per sample per tree
    // OOB tracking for Breiman's proximity (optional, nullptr to skip)
    integer_t* h_nin_all = nullptr,  // [ntree * nsample] - bootstrap membership per tree (0 = OOB)
    // Progress callback (optional, nullptr to skip)
    std::function<void(integer_t, integer_t)> progress_callback = nullptr,  // (current_tree, total_trees)
    // Bootstrap weights (optional, nullptr = uniform bootstrap)
    const real_t* bootstrap_weights = nullptr
);

/**
 * Host function: Predict using sparse random forest on GPU
 * 
 * @param d_X_sparse Device sparse matrix (new data)
 * @param d_nodestatus Tree node status
 * @param d_bestvar Tree split variables
 * @param d_xbestsplit Tree split values
 * @param d_treemap Tree structure
 * @param d_nodeclass Tree node classes
 * @param nsample_new Number of new samples
 * @param ntree Number of trees
 * @param maxnode Maximum nodes per tree
 * @param d_predictions Output: predictions [nsample_new]
 * @param stream CUDA stream
 * @return Error code
 */
integer_t predict_sparse_gpu(
    const CudaSparseMatrixCSR& d_X_sparse,
    const integer_t* d_nodestatus,
    const integer_t* d_bestvar,
    const real_t* d_xbestsplit,
    const integer_t* d_treemap,
    const integer_t* d_nodeclass,
    integer_t nsample_new,
    integer_t ntree,
    integer_t maxnode,
    integer_t max_depth,
    integer_t* d_predictions,
    cudaStream_t stream
);

} // namespace cuda
} // namespace rf

#endif // RF_SPARSE_FOREST_CUH

