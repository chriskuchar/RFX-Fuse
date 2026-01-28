#ifndef RF_TESTREEBAG_CUH
#define RF_TESTREEBAG_CUH

#include "rf_types.hpp"

namespace rf {

// Main wrapper - automatically selects GPU or CPU (matches testreebag_cuda in Fortran)
void testreebag(const real_t* x, const real_t* xbestsplit, const integer_t* nin,
                const integer_t* treemap, const integer_t* bestvar,
                const integer_t* nodeclass, const integer_t* cat,
                const integer_t* nodestatus, const integer_t* catgoleft,
                integer_t nsample, integer_t mdim, integer_t nnode, integer_t maxcat,
                integer_t* jtr, integer_t* nodextr);

// CPU fallback - exact copy of original algorithm (matches testreebag_cpu_fallback in Fortran)
void testreebag_cpu_fallback(const real_t* x, const real_t* xbestsplit,
                             const integer_t* nin, const integer_t* treemap,
                             const integer_t* bestvar, const integer_t* nodeclass,
                             const integer_t* cat, const integer_t* nodestatus,
                             const integer_t* catgoleft, integer_t nsample,
                             integer_t mdim, integer_t nnode, integer_t maxcat,
                             integer_t* jtr, integer_t* nodextr);

} // namespace rf

#endif // RF_TESTREEBAG_CUH
