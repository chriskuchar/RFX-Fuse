// Helper function declaration for getting low-rank factors from CPU code
#ifndef RF_LOWRANK_HELPERS_HPP
#define RF_LOWRANK_HELPERS_HPP

#include "rf_types.hpp"
#include <vector>

namespace rf {
namespace cuda {

// Helper function to copy low-rank factors from GPU to host
// Implementation in rf_lowrank_helpers.cu
bool get_lowrank_factors_host(void* lowrank_proximity_ptr, 
                               dp_t** A_host, dp_t** B_host, 
                               integer_t* rank, integer_t nsamples);

// Helper function to compute MDS coordinates from low-rank factors
// Implementation in rf_lowrank_helpers.cu
// Returns empty vector on failure
std::vector<double> compute_mds_from_factors_host(void* lowrank_proximity_ptr, integer_t k);

// Helper function to compute 3D MDS coordinates from low-rank factors (convenience wrapper)
// Implementation in rf_lowrank_helpers.cu
// Returns empty vector on failure
inline std::vector<double> compute_mds_3d_from_factors_host(void* lowrank_proximity_ptr) {
    return compute_mds_from_factors_host(lowrank_proximity_ptr, 3);
}

// Helper function to reconstruct full proximity matrix from low-rank factors (with RF-GAP normalization if enabled)
// Implementation in rf_lowrank_helpers.cu
// Allocates output_prox on host - caller is responsible for freeing with delete[]
bool reconstruct_proximity_matrix_host(void* lowrank_proximity_ptr, 
                                       dp_t** output_prox, 
                                       integer_t nsamples);

} // namespace cuda
} // namespace rf

#endif // RF_LOWRANK_HELPERS_HPP

