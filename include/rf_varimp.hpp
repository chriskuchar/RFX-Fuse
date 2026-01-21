#ifndef RF_VARIMP_HPP
#define RF_VARIMP_HPP

#include "rf_types.hpp"
#include <vector>

namespace rf {

// ============================================================================
// PRE-SHUFFLED PERMUTATION CACHE (Fisher-Yates optimization)
// ============================================================================
// Pre-compute shuffled indices for all samples ONCE, then reuse across all
// trees and all features. This eliminates per-tree/per-feature RNG overhead.

struct PreShuffledPermutation {
    std::vector<integer_t> shuffled_indices;  // Size: nsample - Fisher-Yates shuffled [0..nsample-1]
    integer_t nsample;
    
    PreShuffledPermutation() : nsample(0) {}
    
    // Initialize with Fisher-Yates shuffle of [0, nsample-1]
    void initialize(integer_t n, unsigned int seed = 42);
    
    // Get permuted index for sample i
    inline integer_t get_permuted(integer_t i) const {
        return shuffled_indices[i];
    }
};

// Global pre-shuffled permutation cache (shared across OpenMP threads)
extern PreShuffledPermutation g_preshuffle_cache;

// Initialize pre-shuffle cache for current thread
void init_preshuffle_cache(integer_t nsample, unsigned int seed = 42);

// ============================================================================
// BATCHED TREE TRAVERSAL (processes all OOB samples at once per feature)
// ============================================================================

// Batched version: traverse tree for ALL OOB samples with feature k permuted
// Returns predictions in jvr_batch[noob] and terminal nodes in nodexvr_batch[noob]
void cpu_testreeimp_batched(
    const real_t* x, integer_t nsample, integer_t mdim,
    const integer_t* joob, integer_t noob, integer_t mr,
    const PreShuffledPermutation& preshuffle,
    const integer_t* treemap, const integer_t* nodestatus,
    const real_t* xbestsplit, const integer_t* bestvar,
    const integer_t* nodeclass, integer_t nnode,
    const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
    integer_t* jvr, integer_t* nodexvr);

// ============================================================================
// ORIGINAL FUNCTIONS (kept for compatibility)
// ============================================================================

// CPU variable importance implementations
void cpu_permobmr(const integer_t* joob, integer_t* pjoob, integer_t noob);

void cpu_testreeimp(const real_t* x, integer_t nsample, integer_t mdim,
                   const integer_t* joob, integer_t noob, integer_t mr,
                   const integer_t* treemap, const integer_t* nodestatus,
                   const real_t* xbestsplit, const integer_t* bestvar,
                   const integer_t* nodeclass, integer_t nnode,
                   const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
                   integer_t* jvr, integer_t* nodexvr);

void cpu_varimp(const real_t* x, integer_t nsample, integer_t mdim,
               const integer_t* cl, const integer_t* nin, const integer_t* jtr,
               integer_t impn, real_t* qimp, real_t* qimpm,
               real_t* avimp, real_t* sqsd,
               const integer_t* treemap, const integer_t* nodestatus,
               const real_t* xbestsplit, const integer_t* bestvar,
               const integer_t* nodeclass, integer_t nnode,
               const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
               integer_t maxcat, const integer_t* catgoleft,
               const real_t* tnodewt, const integer_t* nodextr);

// Regression-specific variable importance (MSE-based, Breiman 2001)
void cpu_varimp_regression(const real_t* x, integer_t nsample, integer_t mdim,
                           const real_t* y, const integer_t* nin, 
                           const real_t* y_pred,  // OOB predictions (nodepred values)
                           integer_t impn, real_t* qimp, real_t* qimpm,
                           real_t* avimp, real_t* sqsd,
                           const integer_t* treemap, const integer_t* nodestatus,
                           const real_t* xbestsplit, const integer_t* bestvar,
                           const real_t* nodepred,  // Terminal node predictions (mean y)
                           const real_t* tnodewt,   // Terminal node weights (mean bootstrap weight)
                           integer_t nnode,
                           const integer_t* cat, integer_t maxcat, 
                           const integer_t* catgoleft,
                           const integer_t* nodextr);

// ============================================================================
// SPARSE MATRIX SUPPORT (CSR format)
// ============================================================================

void cpu_testreeimp_sparse(const SparseMatrixCSR& x_sparse, integer_t nsample, integer_t mdim,
                           const integer_t* joob, integer_t noob, integer_t mr,
                           const integer_t* treemap, const integer_t* nodestatus,
                           const real_t* xbestsplit, const integer_t* bestvar,
                           const integer_t* nodeclass, integer_t nnode,
                           const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
                           integer_t* jvr, integer_t* nodexvr);

void cpu_varimp_sparse(const SparseMatrixCSR& x_sparse, integer_t nsample, integer_t mdim,
                       const integer_t* cl, const integer_t* nin, const integer_t* jtr,
                       integer_t impn, real_t* qimp, real_t* qimpm,
                       real_t* avimp, real_t* sqsd,
                       const integer_t* treemap, const integer_t* nodestatus,
                       const real_t* xbestsplit, const integer_t* bestvar,
                       const integer_t* nodeclass, integer_t nnode,
                       const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
                       integer_t maxcat, const integer_t* catgoleft,
                       const real_t* tnodewt, const integer_t* nodextr);

void cpu_varimp_regression_sparse(const SparseMatrixCSR& x_sparse, integer_t nsample, integer_t mdim,
                                  const real_t* y, const integer_t* nin, 
                                  const real_t* y_pred,
                                  integer_t impn, real_t* qimp, real_t* qimpm,
                                  real_t* avimp, real_t* sqsd,
                                  const integer_t* treemap, const integer_t* nodestatus,
                                  const real_t* xbestsplit, const integer_t* bestvar,
                                  const real_t* nodepred,  // Terminal node predictions (mean y)
                                  const real_t* tnodewt,   // Terminal node weights (mean bootstrap weight)
                                  integer_t nnode,
                                  const integer_t* cat, integer_t maxcat, 
                                  const integer_t* catgoleft,
                                  const integer_t* nodextr);

// GPU variable importance implementations with parallel permutation testing
void gpu_permobmr(const integer_t* joob, integer_t* pjoob, integer_t noob);

void gpu_testreeimp(const real_t* x, integer_t nsample, integer_t mdim,
                   const integer_t* treemap, const integer_t* bestvar,
                   const integer_t* nodeclass, const integer_t* cat,
                   const integer_t* nodestatus, const integer_t* catgoleft,
                   const real_t* xbestsplit, integer_t nnode, integer_t maxcat,
                   const integer_t* cl, integer_t nclass, integer_t* nodextr,
                   real_t* qimpm);

void gpu_varimp(const real_t* x, integer_t nsample, integer_t mdim,
               const integer_t* cl, const integer_t* nin, const integer_t* jtr,
               integer_t impn, real_t* qimp, real_t* qimpm,
               real_t* avimp, real_t* sqsd,
               const integer_t* treemap, const integer_t* nodestatus,
               const real_t* xbestsplit, const integer_t* bestvar,
               const integer_t* nodeclass, integer_t nnode,
               const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
               integer_t maxcat, const integer_t* catgoleft,
               const real_t* tnodewt, const integer_t* nodextr,
               const real_t* y_regression, const real_t* win,
               const integer_t* jinbag, integer_t ninbag,  // In-bag samples (like CPU)
               integer_t nclass, integer_t task_type);

} // namespace rf

#endif // RF_VARIMP_HPP
