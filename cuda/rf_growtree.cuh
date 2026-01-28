#ifndef RF_GROWTREE_CUH
#define RF_GROWTREE_CUH

#include "rf_types.hpp"

namespace rf {

// Main wrapper - single tree, auto GPU/CPU selection (matches growtree_cuda in Fortran)
void growtree(const real_t* x, const real_t* win, const integer_t* cl,
              const integer_t* cat, const integer_t* ties, const integer_t* nin,
              integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
              integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
              integer_t iseed, integer_t ncsplit, integer_t ncmax,
              integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* xbestsplit,
              integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
              integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
              integer_t* nodextr, integer_t& nnode, integer_t* idmove,
              integer_t* igoleft, integer_t* igl, integer_t* nodestart,
              integer_t* nodestop, integer_t* itemp, integer_t* itmp,
              integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
              real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass);

// CPU implementation - exact copy of original algorithm (matches growtree_cpu_fallback in Fortran)
void growtree_cpu_fallback(const real_t* x, const real_t* win, const integer_t* cl,
                            const integer_t* cat, const integer_t* ties, const integer_t* nin,
                            integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
                            integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                            integer_t iseed, integer_t ncsplit, integer_t ncmax,
                            integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* xbestsplit,
                            integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
                            integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
                            integer_t* nodextr, integer_t& nnode, integer_t* idmove,
                            integer_t* igoleft, integer_t* igl, integer_t* nodestart,
                            integer_t* nodestop, integer_t* itemp, integer_t* itmp,
                            integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
                            real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass);

// GPU batch orchestration - grows multiple trees in parallel on GPU
// This is the recommended way to use GPU acceleration for Random Forest
void growtree_batch_gpu(
    integer_t num_trees, const real_t* x, const real_t* win, const integer_t* cl,
    const integer_t* cat, const integer_t* ties, const integer_t* nin,
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
    const integer_t* seeds, integer_t ncsplit, integer_t ncmax,
    // Per-tree input arrays (num_trees × size)
    const integer_t* amat_in, const integer_t* jinbag_in,
    // Per-tree output arrays (num_trees × size)
    real_t* tnodewt_out, real_t* xbestsplit_out, integer_t* nodestatus_out,
    integer_t* bestvar_out, integer_t* treemap_out, integer_t* catgoleft_out,
    integer_t* nodeclass_out, integer_t* nnode_out);

} // namespace rf

#endif // RF_GROWTREE_CUH
