#ifndef RF_TESTREEBAG_HPP
#define RF_TESTREEBAG_HPP

#include "rf_types.hpp"

namespace rf {

// CPU testreebag implementation
void cpu_testreebag(const real_t* x, const real_t* xbestsplit,
                   const integer_t* nin, const integer_t* treemap,
                   const integer_t* bestvar, const integer_t* nodeclass,
                   const integer_t* cat, const integer_t* nodestatus,
                   const integer_t* catgoleft, integer_t nsample,
                   integer_t mdim, integer_t nnode, integer_t maxcat,
                   integer_t* jtr, integer_t* nodextr);

// GPU testreebag implementation with parallel OOB testing
void gpu_testreebag(const real_t* x, const real_t* xbestsplit,
                   const integer_t* nin, const integer_t* treemap,
                   const integer_t* bestvar, const integer_t* nodeclass,
                   const integer_t* cat, const integer_t* nodestatus,
                   const integer_t* catgoleft, integer_t nsample,
                   integer_t mdim, integer_t nnode, integer_t maxcat,
                   integer_t* jtr, integer_t* nodextr);

// ============================================================================
// SPARSE MATRIX SUPPORT (CSR format)
// ============================================================================
// Native sparse testreebag - O(nnz) memory, no dense conversion
// ============================================================================

// CPU testreebag for sparse matrices
void cpu_testreebag_sparse(const SparseMatrixCSR& x_sparse, const real_t* xbestsplit,
                           const integer_t* nin, const integer_t* treemap,
                           const integer_t* bestvar, const integer_t* nodeclass,
                           const integer_t* cat, const integer_t* nodestatus,
                           const integer_t* catgoleft, integer_t nsample,
                           integer_t mdim, integer_t nnode, integer_t maxcat,
                           integer_t* jtr, integer_t* nodextr);

} // namespace rf

#endif // RF_TESTREEBAG_HPP
