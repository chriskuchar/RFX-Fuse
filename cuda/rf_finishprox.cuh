#ifndef RF_FINISHPROX_CUH
#define RF_FINISHPROX_CUH

#include "rf_types.hpp"

namespace rf {

// Main wrapper - automatically selects GPU or CPU (matches finishprox_cuda in Fortran)
void finishprox(integer_t nsample, const integer_t* nout, dp_t* prox, dp_t* proxsym);

// CPU fallback - exact copy of original algorithm (matches finishprox_cpu_fallback in Fortran)
void finishprox_cpu_fallback(integer_t nsample, const integer_t* nout, dp_t* prox, dp_t* proxsym);

} // namespace rf

#endif // RF_FINISHPROX_CUH
