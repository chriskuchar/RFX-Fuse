#include "rf_growtree.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_histogram.hpp"
#include <iostream>

namespace rf {

// Forward declaration of cpu_growtree (defined in CPU file)
extern void cpu_growtree(
    const real_t* x, const real_t* win, const integer_t* cl,
    const integer_t* cat, const integer_t* ties, const integer_t* nin,
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
    integer_t iseed, integer_t ncsplit, integer_t ncmax,
    integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* nodepred, real_t* xbestsplit,
    integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
    integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
    integer_t* nodextr, integer_t& nnode, integer_t* idmove,
    integer_t* igoleft, integer_t* igl, integer_t* nodestart,
    integer_t* nodestop, integer_t* itemp, integer_t* itmp,
    integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
    real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass,
    const HistogramData* histogram_data);

// C++ wrapper for cpu_growtree (defined in CPU file)
// This allows C++ code to call the CPU growtree function
void cpu_growtree_wrapper(
    const real_t* x, const real_t* win, const integer_t* cl,
    const integer_t* cat, const integer_t* ties, const integer_t* nin,
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
    integer_t iseed, integer_t ncsplit, integer_t ncmax,
    integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* nodepred, real_t* xbestsplit,
    integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
    integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
    integer_t* nodextr, integer_t& nnode, integer_t* idmove,
    integer_t* igoleft, integer_t* igl, integer_t* nodestart,
    integer_t* nodestop, integer_t* itemp, integer_t* itmp,
    integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
    real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass,
    const HistogramData* histogram_data) {

    // Call the actual cpu_growtree function defined in CPU file
    cpu_growtree(x, win, cl, cat, ties, nin, mdim, nsample, nclass, maxnode,
                 minndsize, mtry, ninbag, maxcat, iseed, ncsplit, ncmax,
                 amat, jinbag, tnodewt, nodepred, xbestsplit, nodestatus, bestvar, treemap,
                 catgoleft, nodeclass, jtr, nodextr, nnode, idmove, igoleft, igl,
                 nodestart, nodestop, itemp, itmp, icat, tcat, classpop, wr, wl,
                 tclasscat, tclasspop, tmpclass, histogram_data);
}

} // namespace rf
