#ifndef RF_GETAMAT_CUH
#define RF_GETAMAT_CUH

#include "rf_types.hpp"

namespace rf {

// Main wrapper - automatically selects GPU or CPU (matches getamat_cuda in Fortran)
void getamat(const integer_t* asave, const integer_t* nin, const integer_t* cat,
             integer_t nsample, integer_t mdim, integer_t* amat);

// CPU fallback - exact copy of original algorithm (matches getamat_cpu_fallback in Fortran)
void getamat_cpu_fallback(const integer_t* asave, const integer_t* nin, const integer_t* cat,
                          integer_t nsample, integer_t mdim, integer_t* amat);

} // namespace rf

#endif // RF_GETAMAT_CUH
