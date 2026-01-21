#ifndef RF_PREDICT_CUH
#define RF_PREDICT_CUH

#include "rf_types.hpp"

namespace rf {

// Main wrapper - automatically selects GPU or CPU (matches predict_cuda in Fortran)
void predict(const real_t* q, const integer_t* out, integer_t nsample,
             integer_t nclass, integer_t* jest);

// CPU fallback - exact copy of original algorithm (matches predict_cpu_fallback in Fortran)
void predict_cpu_fallback(const real_t* q, const integer_t* out, integer_t nsample,
                          integer_t nclass, integer_t* jest);

} // namespace rf

#endif // RF_PREDICT_CUH
