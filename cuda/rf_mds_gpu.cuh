#ifndef RF_MDS_GPU_CUH
#define RF_MDS_GPU_CUH

#include "rf_types.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <stdexcept>

namespace rf {
namespace cuda {

/**
 * @brief Compute 3D MDS (Multi-Dimensional Scaling) coordinates from GPU proximity matrix (CUDA implementation)
 * 
 * This function performs classical MDS on GPU to embed proximity relationships into 3D space.
 * Algorithm:
 * 1. Convert proximity to distance matrix (1 - proximity)
 * 2. Double-center the distance matrix
 * 3. Compute eigendecomposition using cuSolver (get top 3 eigenvectors)
 * 4. Return 3D coordinates (n_samples × 3)
 * 
 * @param proximity_matrix_gpu Input proximity matrix on GPU (n_samples × n_samples), stored row-major
 * @param n_samples Number of samples
 * @param memory_check If true, check available GPU memory before computation
 * @return std::vector<double> 3D coordinates (n_samples × 3), stored row-major: [x0,y0,z0, x1,y1,z1, ...]
 * 
 * @throws std::runtime_error If GPU memory insufficient or computation fails
 */
std::vector<double> compute_mds_3d_gpu(
    const dp_t* proximity_matrix_gpu,
    integer_t n_samples,
    bool memory_check = true
);

/**
 * @brief Estimate GPU memory required for MDS computation
 * 
 * @param n_samples Number of samples
 * @return size_t Estimated memory in bytes
 */
size_t estimate_mds_gpu_memory(integer_t n_samples);

/**
 * @brief Check if GPU has sufficient memory for MDS computation
 * 
 * @param n_samples Number of samples
 * @return std::pair<bool, std::string> (can_compute, error_message)
 */
std::pair<bool, std::string> check_gpu_mds_memory(integer_t n_samples);

} // namespace cuda
} // namespace rf

#endif // RF_MDS_GPU_CUH

