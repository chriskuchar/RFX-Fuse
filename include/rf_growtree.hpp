#ifndef RF_GROWTREE_HPP
#define RF_GROWTREE_HPP

#include "rf_types.hpp"
#include "rf_histogram.hpp"

namespace rf {

// Regression-specific tree growing
void cpu_growtree_regression(const real_t* x, const real_t* win, const real_t* y_true,
                              const integer_t* cat, const integer_t* ties, const integer_t* nin,
                              integer_t mdim, integer_t nsample, integer_t maxnode,
                              integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                              integer_t iseed, integer_t ncsplit, integer_t ncmax,
                              integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* xbestsplit,
                              integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
                              integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
                              integer_t* nodextr, integer_t& nnode, integer_t* idmove,
                              integer_t* igoleft, integer_t* igl, integer_t* nodestart,
                              integer_t* nodestop, integer_t* itemp, integer_t* itmp,
                              integer_t* icat, real_t* tcat, real_t* wr, real_t* wl,
                              real_t* classpop, real_t* tmpclass);

// CPU-specific tree growing
// C++ wrapper for cpu_growtree (defined in CUDA file)
void cpu_growtree_wrapper(const real_t* x, const real_t* win, const integer_t* cl,
                          const integer_t* cat, const integer_t* ties, const integer_t* nin,
                          integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
                          integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                          integer_t iseed, integer_t ncsplit, integer_t ncmax,
                          integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* nodepred,
                          real_t* xbestsplit,
                          integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
                          integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
                          integer_t* nodextr, integer_t& nnode, integer_t* idmove,
                          integer_t* igoleft, integer_t* igl, integer_t* nodestart,
                          integer_t* nodestop, integer_t* itemp, integer_t* itmp,
                          integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
                          real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass,
                          // Histogram binning (optional - pass nullptr to use sorting-based splits)
                          const HistogramData* histogram_data = nullptr);

// GPU-specific batch tree growing
#ifdef CUDA_FOUND
void gpu_growtree_batch(integer_t num_trees, const real_t* x, const real_t* win, const integer_t* cl,
                        integer_t task_type,  // 0=CLASSIFICATION, 1=REGRESSION, 2=UNSUPERVISED
                        const integer_t* cat, const integer_t* ties, integer_t* nin,
                        const real_t* y_regression,  // Original continuous y values for regression (nullptr for classification)
                        integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
                        integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                        const integer_t* seeds, integer_t ncsplit, integer_t ncmax,
                        integer_t* amat_all, integer_t* jinbag_all, real_t* tnodewt_all, real_t* xbestsplit_all,
                        integer_t* nodestatus_all, integer_t* bestvar_all, integer_t* treemap_all,
                        integer_t* catgoleft_all, integer_t* nodeclass_all, integer_t* nnode_all,
                        integer_t* jtr_all, integer_t* nodextr_all, integer_t* idmove_all,
                        integer_t* igoleft_all, integer_t* igl_all, integer_t* nodestart_all,
                        integer_t* nodestop_all, integer_t* itemp_all, integer_t* itmp_all,
                        integer_t* icat_all, real_t* tcat_all, real_t* classpop_all, real_t* wr_all,
                        real_t* wl_all, real_t* tclasscat_all, real_t* tclasspop_all, real_t* tmpclass_all, 
                        real_t* q_all, integer_t* nout_all, real_t* avimp_all,
                        real_t* qimp_all, real_t* qimpm_all, dp_t* proximity_all,
                        real_t* oob_predictions_all,  // For regression: accumulate OOB predictions
                        real_t* nodepred_all,  // For regression: terminal node predictions [ntree * maxnode]
                        void** lowrank_proximity_ptr_out,  // Output: LowRankProximityMatrix pointer for low-rank mode
                        real_t* prox_imp_all = nullptr,  // Proximity importance [nsample x mdim]
                        int16_t* leaf_assignments_all = nullptr,  // Leaf assignments [ntree * nsample]
                        // HISTOGRAM PARAMETERS (for XGBoost-like speed)
                        bool use_histogram = false,  // Enable histogram-based split finding
                        const uint8_t* X_binned = nullptr,  // Pre-binned feature data [nsample × mdim]
                        const real_t* bin_edges_all = nullptr,  // Bin edges [mdim × 257]
                        const integer_t* n_bins_per_feature = nullptr,  // Number of bins per feature [mdim]
                        integer_t max_bins = 256,  // Maximum bins
                        const real_t* bootstrap_weights = nullptr);  // Per-sample bootstrap draw probability
#endif

// Unified interface - automatically selects GPU or CPU
void growtree(const real_t* x, const real_t* win, const integer_t* cl,
              const integer_t* cat, const integer_t* ties, const integer_t* nin,
              integer_t mdim, integer_t nsample, integer_t nclass,
              integer_t maxnode, integer_t minndsize, integer_t mtry,
              integer_t ninbag, integer_t maxcat, integer_t iseed,
              integer_t ncsplit, integer_t ncmax,
              integer_t* amat, integer_t* jinbag,
              real_t* tnodewt, real_t* xbestsplit,
              integer_t* nodestatus, integer_t* bestvar,
              integer_t* treemap, integer_t* catgoleft,
              integer_t* nodeclass, integer_t* jtr, integer_t* nodextr,
              integer_t& nnode,
              integer_t* idmove, integer_t* igoleft, integer_t* igl,
              integer_t* nodestart, integer_t* nodestop, integer_t* itemp,
              integer_t* itmp, integer_t* icat, real_t* tcat,
              real_t* classpop, real_t* wr, real_t* wl,
              real_t* tclasscat, real_t* tclasspop, real_t* tmpclass);

namespace cpu {

// CPU implementation (exact port of original growtree.f)
void growtree_cpu(const real_t* x, const real_t* win, const integer_t* cl,
                  const integer_t* cat, const integer_t* ties, const integer_t* nin,
                  integer_t mdim, integer_t nsample, integer_t nclass,
                  integer_t maxnode, integer_t minndsize, integer_t mtry,
                  integer_t ninbag, integer_t maxcat, integer_t iseed,
                  integer_t ncsplit, integer_t ncmax,
                  integer_t* amat, integer_t* jinbag,
                  real_t* tnodewt, real_t* xbestsplit,
                  integer_t* nodestatus, integer_t* bestvar,
                  integer_t* treemap, integer_t* catgoleft,
                  integer_t* nodeclass, integer_t* jtr, integer_t* nodextr,
                  integer_t& nnode,
                  integer_t* idmove, integer_t* igoleft, integer_t* igl,
                  integer_t* nodestart, integer_t* nodestop, integer_t* itemp,
                  integer_t* itmp, integer_t* icat, real_t* tcat,
                  real_t* classpop, real_t* wr, real_t* wl,
                  real_t* tclasscat, real_t* tclasspop, real_t* tmpclass);

} // namespace cpu

namespace cuda {

// GPU implementation (currently calls CPU fallback like Fortran version)
void growtree_gpu(const real_t* x, const real_t* win, const integer_t* cl,
                  const integer_t* cat, const integer_t* ties, const integer_t* nin,
                  integer_t mdim, integer_t nsample, integer_t nclass,
                  integer_t maxnode, integer_t minndsize, integer_t mtry,
                  integer_t ninbag, integer_t maxcat, integer_t iseed,
                  integer_t ncsplit, integer_t ncmax,
                  integer_t* amat, integer_t* jinbag,
                  real_t* tnodewt, real_t* xbestsplit,
                  integer_t* nodestatus, integer_t* bestvar,
                  integer_t* treemap, integer_t* catgoleft,
                  integer_t* nodeclass, integer_t* jtr, integer_t* nodextr,
                  integer_t& nnode,
                  integer_t* idmove, integer_t* igoleft, integer_t* igl,
                  integer_t* nodestart, integer_t* nodestop, integer_t* itemp,
                  integer_t* itmp, integer_t* icat, real_t* tcat,
                  real_t* classpop, real_t* wr, real_t* wl,
                  real_t* tclasscat, real_t* tclasspop, real_t* tmpclass);

} // namespace cuda

// Helper: Move data based on best split (original movedata from growtree.f:502-587)
void movedata(integer_t mdim, integer_t nsample, integer_t msplit,
              integer_t nbest, integer_t ndstart, integer_t ndstop,
              integer_t maxcat, const integer_t* cat, const integer_t* igoleft,
              integer_t* amat, integer_t* jinbag, integer_t& ndstopl,
              integer_t* idmove, integer_t* itemp);

// Helper: Create node and check if terminal (original createnode from growtree.f:590-644)
void createnode(const real_t* win, integer_t inode, integer_t nsample,
                integer_t nclass, integer_t minndsize,
                const integer_t* nodestart, const integer_t* nodestop,
                const integer_t* jinbag, const integer_t* cl,
                real_t* classpop, integer_t* nodestatus);

// Helper: Find best split for current node
void findbestsplit(const real_t* tclasspop, const real_t* win, const integer_t* cl,
                   const integer_t* jinbag, const integer_t* cat,
                   const integer_t* amat, const integer_t* ties,
                   integer_t mdim, integer_t nsample, integer_t nclass,
                   integer_t maxcat, integer_t ndstart, integer_t ndstop,
                   integer_t mtry, integer_t ncmax, integer_t ncsplit,
                   integer_t& iseed, integer_t* igoleft,
                   integer_t& msplit, integer_t& nbest, integer_t& jstat,
                   real_t* tclasscat, real_t* tmpclass, real_t* tcat,
                   integer_t* icat, real_t* wr, real_t* wl,
                   integer_t* igl, integer_t* itmp);

// Helper: Find best categorical split (exhaustive search, Garside algorithm)
void cpu_catmax(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
            integer_t nnz, integer_t numcat, integer_t nclass,
            integer_t* igl, integer_t* itmp, real_t* tmpclass,
            const real_t* tcat, const integer_t* icat,
            real_t& critmax, integer_t* igoleft, integer_t& nhit);

// Helper: Find best categorical split (random search for high-cardinality)
void cpu_catmaxr(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
             integer_t numcat, integer_t nclass, integer_t* igl,
             integer_t ncsplit, real_t* tmpclass,
             real_t& critmax, integer_t* igoleft, integer_t& nhit);

// Helper: Calculate split criterion (Gini)
void calcs(const real_t* tmpclass, const real_t* tclasspop, real_t pdo,
           integer_t nclass, real_t& tdec);

// ============================================================================
// SPARSE MATRIX SUPPORT (CSR format)
// ============================================================================
// Native sparse tree growing - O(nnz) memory, no dense conversion
// For recommender systems with massive sparse user-item matrices
// ============================================================================

// Sparse CPU tree growing for classification
void cpu_growtree_sparse(const SparseMatrixCSR& x_sparse, const real_t* win, const integer_t* cl,
                         const integer_t* cat, const integer_t* ties, const integer_t* nin,
                         integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
                         integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                         integer_t iseed, integer_t ncsplit, integer_t ncmax,
                         integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* nodepred,
                         real_t* xbestsplit,
                         integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
                         integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
                         integer_t* nodextr, integer_t& nnode, integer_t* idmove,
                         integer_t* igoleft, integer_t* igl, integer_t* nodestart,
                         integer_t* nodestop, integer_t* itemp, integer_t* itmp,
                         integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
                         real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass);

// Sparse CPU tree growing for regression
void cpu_growtree_regression_sparse(const SparseMatrixCSR& x_sparse, const real_t* win, const real_t* y_true,
                                    const integer_t* cat, const integer_t* ties, const integer_t* nin,
                                    integer_t mdim, integer_t nsample, integer_t maxnode,
                                    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                                    integer_t iseed, integer_t ncsplit, integer_t ncmax,
                                    integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* xbestsplit,
                                    integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
                                    integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
                                    integer_t* nodextr, integer_t& nnode, integer_t* idmove,
                                    integer_t* igoleft, integer_t* igl, integer_t* nodestart,
                                    integer_t* nodestop, integer_t* itemp, integer_t* itmp,
                                    integer_t* icat, real_t* tcat, real_t* wr, real_t* wl,
                                    real_t* classpop, real_t* tmpclass);

#ifdef CUDA_FOUND
// Sparse GPU batch tree growing
void gpu_growtree_batch_sparse(integer_t num_trees, const SparseMatrixCSR& x_sparse, 
                               const real_t* win, const integer_t* cl,
                               integer_t task_type, const integer_t* cat, const integer_t* ties, integer_t* nin,
                               const real_t* y_regression, integer_t mdim, integer_t nsample, integer_t nclass,
                               integer_t maxnode, integer_t minndsize, integer_t mtry, integer_t ninbag,
                               integer_t maxcat, const integer_t* seeds, integer_t ncsplit, integer_t ncmax,
                               integer_t* amat_all, integer_t* jinbag_all, real_t* tnodewt_all, real_t* xbestsplit_all,
                               integer_t* nodestatus_all, integer_t* bestvar_all, integer_t* treemap_all,
                               integer_t* catgoleft_all, integer_t* nodeclass_all, integer_t* nnode_all,
                               integer_t* jtr_all, integer_t* nodextr_all, integer_t* idmove_all,
                               integer_t* igoleft_all, integer_t* igl_all, integer_t* nodestart_all,
                               integer_t* nodestop_all, integer_t* itemp_all, integer_t* itmp_all,
                               integer_t* icat_all, real_t* tcat_all, real_t* classpop_all, real_t* wr_all,
                               real_t* wl_all, real_t* tclasscat_all, real_t* tclasspop_all, real_t* tmpclass_all,
                               real_t* q_all, integer_t* nout_all, real_t* avimp_all,
                               real_t* qimp_all, real_t* qimpm_all, dp_t* proximity_all,
                               real_t* oob_predictions_all, void** lowrank_proximity_ptr_out);
#endif

} // namespace rf

#endif // RF_GROWTREE_HPP
