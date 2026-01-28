#ifndef RF_BOOTSTRAP_CUH
#define RF_BOOTSTRAP_CUH

#include "rf_types.hpp"

namespace rf {

// CUDA kernel declaration moved to rf_growtree.cu to avoid C++ compilation issues
// __global__ void gpu_bootstrap_kernel(...) - declared where needed

// Main wrapper - automatically selects GPU or CPU (matches boot_cuda in Fortran)
void gpu_bootstrap(const real_t* weight, integer_t nsample,
               real_t* win, integer_t* nin, integer_t* nout,
               integer_t* jinbag, integer_t* joobag,
               integer_t& ninbag, integer_t& noobag,
               integer_t tree_id = 0);


// Cleanup function for global bootstrap states
void cleanup_bootstrap_states();

} // namespace rf

#endif // RF_BOOTSTRAP_CUH
