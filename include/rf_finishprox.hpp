#ifndef RF_FINISHPROX_HPP
#define RF_FINISHPROX_HPP

#include "rf_types.hpp"

namespace rf {

// CPU finishprox implementation
void cpu_finishprox(integer_t nsample, const integer_t* nout, dp_t* prox, dp_t* proxsym);

// GPU finishprox implementation with quantization kernels
void gpu_finishprox(integer_t nsample, const integer_t* nout, dp_t* prox, dp_t* proxsym);

} // namespace rf

#endif // RF_FINISHPROX_HPP
