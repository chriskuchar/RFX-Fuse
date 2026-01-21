#ifndef RF_VARIMP_CUH
#define RF_VARIMP_CUH

#include "rf_types.hpp"

namespace rf {

// ============================================================================
// PRE-SHUFFLED PERMUTATION (Fisher-Yates on GPU) - OPTIMIZATION
// ============================================================================
// Pre-compute Fisher-Yates shuffle ONCE per tree, reuse for all features.
// Eliminates per-feature RNG overhead.

// GPU kernel to initialize pre-shuffled permutation array
// Single thread does Fisher-Yates (sequential algorithm, but done only ONCE)
__global__ void gpu_init_preshuffle_kernel(
    integer_t* preshuffle_indices,  // Output: [nsample] shuffled indices
    integer_t nsample,
    unsigned int seed
);

// Initialize pre-shuffle on GPU (call once before varimp)
void gpu_init_preshuffle(
    integer_t* d_preshuffle_indices,  // Device pointer
    integer_t nsample,
    unsigned int seed,
    cudaStream_t stream = 0
);

// Main wrapper - automatically selects GPU or CPU (matches varimp_cuda in Fortran)
void varimp(const real_t* x, integer_t nsample, integer_t mdim,
            const integer_t* cl, const integer_t* nin, const integer_t* jtr,
            integer_t impn, real_t* qimp, real_t* qimpm,
            real_t* avimp, real_t* sqsd,
            const integer_t* treemap, const integer_t* nodestatus,
            const real_t* xbestsplit, const integer_t* bestvar,
            const integer_t* nodeclass, integer_t nnode,
            const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
            integer_t maxcat, const integer_t* catgoleft,
            const real_t* tnodewt, const integer_t* nodextr,
            integer_t* joob, integer_t* pjoob, integer_t* iv);

// CPU fallback - exact copy of original algorithm (matches varimp_cpu_fallback in Fortran)
void varimp_cpu_fallback(const real_t* x, integer_t nsample, integer_t mdim,
                         const integer_t* cl, const integer_t* nin, const integer_t* jtr,
                         integer_t impn, real_t* qimp, real_t* qimpm,
                         real_t* avimp, real_t* sqsd,
                         const integer_t* treemap, const integer_t* nodestatus,
                         const real_t* xbestsplit, const integer_t* bestvar,
                         const integer_t* nodeclass, integer_t nnode,
                         const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
                         integer_t maxcat, const integer_t* catgoleft,
                         const real_t* tnodewt, const integer_t* nodextr,
                         integer_t* joob, integer_t* pjoob, integer_t* iv);

// Helper functions for CPU fallback
void testreeimp_cpu_fallback(const real_t* x, integer_t nsample, integer_t mdim,
                            const integer_t* cl, const integer_t* joob, integer_t* pjoob, integer_t noob,
                            integer_t mr, const integer_t* treemap, const integer_t* nodestatus,
                            const real_t* xbestsplit, const integer_t* bestvar,
                            const integer_t* nodeclass, integer_t nnode,
                            const integer_t* cat, integer_t maxcat,
                            const integer_t* catgoleft, integer_t* jvr, integer_t* nodexvr);

void permobmr_cpu_fallback(const integer_t* joob, integer_t* pjoob, integer_t noob);

// GPU Variable Importance Context (reusable memory allocation)
struct GPUVarImpContext;

GPUVarImpContext* gpu_varimp_alloc_context(integer_t nsample, integer_t mdim, integer_t maxnode, integer_t maxcat,
                                           integer_t max_ninbag, integer_t nclass, integer_t grid_size_x, integer_t block_size_x);

void gpu_varimp_free_context(GPUVarImpContext* ctx);

void gpu_varimp_with_context(GPUVarImpContext* ctx, const real_t* x, integer_t nsample, integer_t mdim,
                             const integer_t* cl, const integer_t* nin, const integer_t* jtr, integer_t impn,
                             real_t* qimp, real_t* qimpm, real_t* avimp, real_t* sqsd,
                             const integer_t* treemap, const integer_t* nodestatus, const real_t* xbestsplit,
                             const integer_t* bestvar, const integer_t* nodeclass, integer_t nnode,
                             const integer_t* cat, integer_t* jvr, integer_t* nodexvr, integer_t maxcat,
                             const integer_t* catgoleft, const real_t* tnodewt, const integer_t* nodextr,
                             const real_t* y_regression, const real_t* win, const integer_t* jinbag,
                             integer_t ninbag, integer_t nclass, integer_t task_type);

// BATCHED Variable Importance - Process ALL trees in parallel (no per-tree loops!)
// All data must already be on GPU. Accumulates importance directly on GPU.
void gpu_varimp_batch_all_trees(
    integer_t num_trees,
    integer_t nsample,
    integer_t mdim,
    integer_t maxnode,
    integer_t nclass,
    const real_t* x_gpu,               // Already on GPU [nsample * mdim]
    const integer_t* cl_gpu,           // Already on GPU [nsample]
    const integer_t* nin_all_gpu,      // Already on GPU [num_trees * nsample]
    const integer_t* jtr_all_gpu,      // Already on GPU [num_trees * nsample]
    const integer_t* nodextr_all_gpu,  // Already on GPU [num_trees * nsample]
    const integer_t* treemap_all_gpu,  // Already on GPU [num_trees * 2 * maxnode]
    const integer_t* nodestatus_all_gpu,
    const real_t* xbestsplit_all_gpu,
    const integer_t* bestvar_all_gpu,
    const integer_t* nodeclass_all_gpu,
    const integer_t* nnode_all_gpu,
    const integer_t* cat_gpu,
    integer_t maxcat,
    const real_t* tnodewt_all_gpu,     // May be nullptr
    real_t* avimp_gpu,                 // Output: [mdim] - accumulated importance
    real_t* qimp_gpu,                  // Output: [nsample] - original correct rate (or nullptr)
    real_t* qimpm_gpu,                 // Output: [nsample * mdim] - local importance (or nullptr)
    integer_t use_casewise,
    cudaStream_t stream
);

// BATCHED Regression Variable Importance - Process ALL trees in parallel
void gpu_varimp_regression_batch_all_trees(
    integer_t num_trees,
    integer_t nsample,
    integer_t mdim,
    integer_t maxnode,
    const real_t* x_gpu,               // Already on GPU
    const real_t* y_true_gpu,          // Already on GPU [nsample]
    const real_t* y_pred_all_gpu,      // Already on GPU [num_trees * nsample]
    const integer_t* nin_all_gpu,      // Already on GPU [num_trees * nsample]
    const integer_t* nodextr_all_gpu,  // Already on GPU [num_trees * nsample]
    const integer_t* treemap_all_gpu,
    const integer_t* nodestatus_all_gpu,
    const real_t* xbestsplit_all_gpu,
    const integer_t* bestvar_all_gpu,
    const real_t* nodepred_all_gpu,    // [num_trees * maxnode]
    const integer_t* nnode_all_gpu,
    const integer_t* cat_gpu,
    integer_t maxcat,
    const real_t* tnodewt_all_gpu,     // May be nullptr
    real_t* avimp_gpu,                 // Output: [mdim]
    real_t* qimp_gpu,                  // Output: [nsample] - original squared error (or nullptr)
    real_t* qimpm_gpu,                 // Output: [nsample * mdim] - local importance (or nullptr)
    integer_t use_casewise,
    cudaStream_t stream
);

} // namespace rf

// CUDA kernel declarations (must be outside namespace)
// GPU kernel to compute tnodewt (node weights) in parallel
// Each thread block handles one node, threads within block parallelize over samples
// For classification: tnodewt = sum(win) / sum(nin) for in-bag samples (matches CPU: tw/tn)
// For regression: tnodewt = mean(y) for samples in the node
// Matches CPU implementation: iterates through jinbag (in-bag samples) only
__global__ void gpu_compute_tnodewt_kernel(
    const rf::real_t* x, rf::integer_t nsample, rf::integer_t mdim, rf::integer_t nnode,
    const rf::integer_t* treemap, const rf::integer_t* nodestatus,
    const rf::real_t* xbestsplit, const rf::integer_t* bestvar,
    const rf::integer_t* cl, const rf::real_t* y_regression, const rf::real_t* win,
    const rf::integer_t* nin,  // Bootstrap frequency - needed for correct tnodewt computation
    const rf::integer_t* jinbag,  // In-bag sample indices (like CPU)
    rf::integer_t ninbag,  // Number of in-bag samples (like CPU)
    rf::integer_t nclass, rf::integer_t task_type,
    rf::real_t* tnodewt);

#endif // RF_VARIMP_CUH
