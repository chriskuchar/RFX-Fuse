#include "rf_config.hpp"
#include <iostream>
#include <cstdlib>

namespace rf {

// Global configuration instance
RFConfig g_config;

// CUDA functions moved to cuda/rf_config_cuda.cu
// This file now only contains CPU-only functions

} // namespace rf
