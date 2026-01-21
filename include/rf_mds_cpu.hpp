#ifndef RF_MDS_CPU_HPP
#define RF_MDS_CPU_HPP

#include "rf_types.hpp"
#include <vector>
#include <stdexcept>

namespace rf {

/**
 * @brief Compute 3D MDS (Multi-Dimensional Scaling) coordinates from proximity matrix (CPU implementation)
 * 
 * This function performs classical MDS to embed proximity relationships into 3D space.
 * Algorithm:
 * 1. Convert proximity to distance matrix (1 - proximity)
 * 2. Double-center the distance matrix
 * 3. Compute eigendecomposition (get top 3 eigenvectors)
 * 4. Return 3D coordinates (n_samples × 3)
 * 
 * @param proximity_matrix Input proximity matrix (n_samples × n_samples), stored row-major
 * @param n_samples Number of samples
 * @param memory_check If true, check available memory before computation
 * @return std::vector<double> 3D coordinates (n_samples × 3), stored row-major: [x0,y0,z0, x1,y1,z1, ...]
 * 
 * @throws std::runtime_error If memory insufficient or computation fails
 */
std::vector<double> compute_mds_3d_cpu(
    const dp_t* proximity_matrix,
    integer_t n_samples,
    bool memory_check = true,
    const integer_t* oob_counts_rfgap = nullptr,  // OOB counts for RF-GAP normalization (|Si| for each sample i)
    bool use_rfgap = false  // Only normalize by OOB counts if RF-GAP is actually enabled
);

/**
 * @brief Estimate memory required for CPU MDS computation
 * 
 * @param n_samples Number of samples
 * @return size_t Estimated memory in bytes
 */
size_t estimate_mds_cpu_memory(integer_t n_samples);

/**
 * @brief Check if CPU has sufficient memory for MDS computation
 * 
 * @param n_samples Number of samples
 * @param available_memory_bytes Available CPU memory in bytes (0 = auto-detect)
 * @return std::pair<bool, std::string> (can_compute, error_message)
 */
std::pair<bool, std::string> check_cpu_mds_memory(integer_t n_samples, size_t available_memory_bytes = 0);

} // namespace rf

#endif // RF_MDS_CPU_HPP

