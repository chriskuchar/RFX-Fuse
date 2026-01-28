#include "rf_mds_cpu.hpp"
#include "rf_utils.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

// LAPACK function declarations
extern "C" {
    void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda,
                 double* w, double* work, const int* lwork, int* iwork, const int* liwork, int* info);
}

namespace rf {

namespace {
    // Helper to get available memory (simplified - can be enhanced)
    size_t get_available_memory() {
        // Try to read /proc/meminfo or use system call
        // For now, return a conservative estimate
        // In production, this should read actual available memory
        FILE* meminfo = fopen("/proc/meminfo", "r");
        if (meminfo) {
            char line[256];
            size_t available_kb = 0;
            while (fgets(line, sizeof(line), meminfo)) {
                if (sscanf(line, "MemAvailable: %zu kB", &available_kb) == 1) {
                    fclose(meminfo);
                    return available_kb * 1024; // Convert to bytes
                }
            }
            fclose(meminfo);
        }
        // Fallback: assume 4GB available if meminfo cannot be read
        return 4ULL * 1024 * 1024 * 1024;
    }
}

size_t estimate_mds_cpu_memory(integer_t n_samples) {
    // Memory requirements:
    // - Distance matrix: n² × sizeof(double)
    // - Centered matrix: n² × sizeof(double)  
    // - Eigenvectors (all): n² × sizeof(double) for LAPACK workspace
    // - Eigenvalues: n × sizeof(double)
    // - LAPACK workspace: ~n² × sizeof(double)
    // - Total: ~4 × n² × sizeof(double) + n × sizeof(double)
    
    size_t n = static_cast<size_t>(n_samples);
    size_t matrix_size = n * n * sizeof(double);
    size_t eigenvalues_size = n * sizeof(double);
    size_t workspace_size = (1 + 6 * n + 2 * n * n) * sizeof(double); // LAPACK workspace estimate
    size_t iwork_size = (3 + 5 * n) * sizeof(int); // LAPACK integer workspace
    
    return 2 * matrix_size + eigenvalues_size + workspace_size + iwork_size;
}

std::pair<bool, std::string> check_cpu_mds_memory(integer_t n_samples, size_t available_memory_bytes) {
    size_t required = estimate_mds_cpu_memory(n_samples);
    
    if (available_memory_bytes == 0) {
        available_memory_bytes = get_available_memory();
    }
    
    // Require at least 2x the estimated memory for safety
    size_t required_with_safety = required * 2;
    
    if (required_with_safety > available_memory_bytes) {
        double required_gb = required_with_safety / (1024.0 * 1024.0 * 1024.0);
        double available_gb = available_memory_bytes / (1024.0 * 1024.0 * 1024.0);
        
        std::string error_msg = 
            "CPU MDS would crash due to insufficient memory.\n"
            "  Required (with safety margin): " + std::to_string(required_gb) + " GB\n"
            "  Available: " + std::to_string(available_gb) + " GB\n"
            "  Dataset size: " + std::to_string(n_samples) + " samples\n"
            "\n"
            "SOLUTION: Use GPU mode for MDS computation:\n"
            "  - Set use_gpu=True when creating the model\n"
            "  - GPU MDS can handle much larger datasets\n"
            "  - Or reduce dataset size";
        
        return std::make_pair(false, error_msg);
    }
    
    return std::make_pair(true, "");
}

std::vector<double> compute_mds_3d_cpu(
    const dp_t* proximity_matrix,
    integer_t n_samples,
    bool memory_check,
    const integer_t* oob_counts_rfgap,
    bool use_rfgap
) {
    const int k = 3; // Fixed to 3 for 3D visualization
    
    if (n_samples < 2) {
        throw std::runtime_error("MDS requires at least 2 samples");
    }
    
    if (!proximity_matrix) {
        throw std::runtime_error("Proximity matrix is null");
    }
    
    // Memory check
    if (memory_check) {
        auto [can_compute, error_msg] = check_cpu_mds_memory(n_samples);
        if (!can_compute) {
            throw std::runtime_error(error_msg);
        }
    }
    
    size_t n = static_cast<size_t>(n_samples);
    size_t n2 = n * n;
    
    // Step 0: Normalize proximity matrix by OOB counts (for both RF-GAP and standard proximity)
    // Standard proximity: normalize each row i by nout[i] (matching original Fortran)
    // RF-GAP: normalize each row i by |Si| (already done in cpu_proximity_rfgap, but needed if reconstructed from low-rank)
    // Proximity matrix is stored in COLUMN-MAJOR format (prox[i + j * n])
    // Keep everything column-major throughout to avoid transpositions
    std::vector<double> normalized_proximity;
    const dp_t* prox_to_use = proximity_matrix;
    
    // Normalize by OOB counts (for both RF-GAP and standard proximity)
    if (use_rfgap && oob_counts_rfgap != nullptr) {
        // Use RF-GAP OOB counts
        normalized_proximity.resize(n2);
        for (size_t i = 0; i < n; i++) {
            integer_t si = oob_counts_rfgap[i];
            if (si > 0) {
                for (size_t j = 0; j < n; j++) {
                    // Column-major: prox[i + j * n] is element at row i, column j
                    normalized_proximity[i + j * n] = static_cast<double>(proximity_matrix[i + j * n]) / static_cast<double>(si);
                }
            } else {
                // Sample i is never OOB - keep original values (should be diagonal = 1.0)
                for (size_t j = 0; j < n; j++) {
                    normalized_proximity[i + j * n] = static_cast<double>(proximity_matrix[i + j * n]);
                }
            }
        }
        prox_to_use = reinterpret_cast<const dp_t*>(normalized_proximity.data());
        
        // After row-wise normalization, symmetrize (matching original Fortran)
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                double prox_ij = normalized_proximity[i + j * n];
                double prox_ji = normalized_proximity[j + i * n];
                double avg_prox = 0.5 * (prox_ij + prox_ji);
                normalized_proximity[i + j * n] = avg_prox;
                normalized_proximity[j + i * n] = avg_prox;
            }
        }
    } else if (oob_counts_rfgap != nullptr) {
        // Use standard OOB counts (for non-RF-GAP proximity)
        // This is needed when proximity matrix was reconstructed from low-rank factors
        normalized_proximity.resize(n2);
        for (size_t i = 0; i < n; i++) {
            integer_t nout_i = oob_counts_rfgap[i];  // Reuse parameter name for standard OOB counts
            if (nout_i > 0) {
                for (size_t j = 0; j < n; j++) {
                    // Column-major: prox[i + j * n] is element at row i, column j
                    normalized_proximity[i + j * n] = static_cast<double>(proximity_matrix[i + j * n]) / static_cast<double>(nout_i);
                }
            } else {
                // Sample i is never OOB - keep original values
                for (size_t j = 0; j < n; j++) {
                    normalized_proximity[i + j * n] = static_cast<double>(proximity_matrix[i + j * n]);
                }
            }
        }
        prox_to_use = reinterpret_cast<const dp_t*>(normalized_proximity.data());
        
        // After row-wise normalization, symmetrize (matching original Fortran)
        for (size_t i = 0; i < n; i++) {
            // Set diagonal to 1.0 explicitly
            normalized_proximity[i + i * n] = 1.0;
            for (size_t j = i + 1; j < n; j++) {
                double prox_ij = normalized_proximity[i + j * n];
                double prox_ji = normalized_proximity[j + i * n];
                double avg_prox = 0.5 * (prox_ij + prox_ji);
                normalized_proximity[i + j * n] = avg_prox;
                normalized_proximity[j + i * n] = avg_prox;
            }
        }
    }
    // Note: If proximity matrix has already been normalized by finishprox, oob_counts_rfgap may be nullptr
    // In that case, the proximity matrix is used as-is (already normalized and symmetrized)
    
    // Step 1: Convert proximity to distance matrix
    // Match rfviz backup implementation: use max_prox - proximity
    // This works for both normalized and unnormalized proximity matrices
    // Keep in column-major format: distance[i + j * n] = element at row i, column j
    std::vector<double> distance_matrix(n2);
    
    // Find maximum proximity value (for distance conversion)
    double max_prox = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double prox = static_cast<double>(prox_to_use[i + j * n]);
            if (prox > max_prox) max_prox = prox;
        }
    }
    
    // Convert proximity to distance: distance = max_prox - proximity
    // This matches rfviz backup: distance_matrix = proximity_max - proximity_matrix
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double prox = static_cast<double>(prox_to_use[i + j * n]);
            distance_matrix[i + j * n] = max_prox - prox;
        }
    }
    
    // Ensure diagonal is exactly 0 (same sample has distance 0)
    for (size_t i = 0; i < n; i++) {
        distance_matrix[i + i * n] = 0.0;
    }
    
    // Step 1b: Square the distance matrix (D²) for classical MDS
    // Classical MDS requires squared distances before double-centering
    std::vector<double> distance_squared(n2);
    for (size_t i = 0; i < n2; i++) {
        double d = distance_matrix[i];
        distance_squared[i] = d * d;  // D²
    }
    
    // Step 2: Double-centering of squared distance matrix (keep column-major)
    // Compute row means of D²
    std::vector<double> row_means(n, 0.0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            row_means[i] += distance_squared[i + j * n];  // Column-major: row i
        }
        row_means[i] /= static_cast<double>(n);
    }
    
    // Compute column means of D²
    std::vector<double> col_means(n, 0.0);
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            col_means[j] += distance_squared[i + j * n];  // Column-major: column j
        }
        col_means[j] /= static_cast<double>(n);
    }
    
    // Compute grand mean of D²
    double grand_mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        grand_mean += row_means[i];
    }
    grand_mean /= static_cast<double>(n);
    
    // Apply double-centering: B = -0.5 * (D² - row_means - col_means + grand_mean)
    // Keep in column-major format for LAPACK
    std::vector<double> centered_matrix(n2);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double val = distance_squared[i + j * n] - row_means[i] - col_means[j] + grand_mean;
            centered_matrix[i + j * n] = -0.5 * val;  // Column-major
        }
    }
    
    // Step 3: Eigendecomposition using LAPACK dsyevd
    // LAPACK expects column-major format, which is already provided
    std::vector<double> work_matrix = centered_matrix;  // Already column-major, no transpose needed
    
    // Query workspace size
    int n_int = static_cast<int>(n);
    char jobz = 'V'; // Compute eigenvectors
    char uplo = 'U'; // Upper triangular
    int lwork = -1;
    int liwork = -1;
    int info = 0;
    
    double work_query;
    int iwork_query;
    
    dsyevd_(&jobz, &uplo, &n_int, work_matrix.data(), &n_int,
            nullptr, &work_query, &lwork, &iwork_query, &liwork, &info);
    
    if (info != 0) {
        throw std::runtime_error("LAPACK workspace query failed");
    }
    
    lwork = static_cast<int>(work_query);
    liwork = iwork_query;
    
    std::vector<double> eigenvalues(n);
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
    
    // Compute eigendecomposition
    dsyevd_(&jobz, &uplo, &n_int, work_matrix.data(), &n_int,
            eigenvalues.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    
    if (info != 0) {
        throw std::runtime_error("LAPACK eigendecomposition failed with error code: " + std::to_string(info));
    }
    
    // Step 4: Extract top 3 eigenvectors (corresponding to largest POSITIVE eigenvalues)
    // LAPACK returns eigenvalues in ascending order
    // For classical MDS, we want the largest positive eigenvalues (negative eigenvalues are noise)
    std::vector<std::pair<double, int>> eigen_pairs;
    eigen_pairs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        // Only consider positive eigenvalues (negative ones are numerical noise or indicate non-Euclidean distances)
        if (eigenvalues[i] > 1e-10) {
            eigen_pairs.push_back(std::make_pair(eigenvalues[i], static_cast<int>(i)));
        }
    }
    
    if (eigen_pairs.size() < static_cast<size_t>(k)) {
        throw std::runtime_error("Not enough positive eigenvalues for MDS. Proximity matrix may not be Euclidean.");
    }
    
    // Sort by eigenvalue value (descending) - largest positive eigenvalues first
    std::sort(eigen_pairs.rbegin(), eigen_pairs.rend(), 
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first < b.first;  // Sort by value, not absolute value
              });
    
    // Extract top k=3 eigenvectors
    std::vector<double> coords_3d(n * k);
    for (int comp = 0; comp < k; comp++) {
        int eigen_idx = eigen_pairs[comp].second;
        double eigenval = eigen_pairs[comp].first;
        
        // Scale by sqrt(eigenvalue) for proper MDS coordinates
        // Eigenvalue should be positive, but add safety check
        double scale = (eigenval > 0) ? std::sqrt(std::abs(eigenval)) : 0.0;
        
        for (size_t i = 0; i < n; i++) {
            // work_matrix is column-major (LAPACK format), so eigenvector eigen_idx is stored in column eigen_idx
            // Element at row i, column eigen_idx is at index i + eigen_idx * n (column-major)
            // Output coords_3d is row-major: [x0,y0,z0, x1,y1,z1, ...]
            coords_3d[i * k + comp] = work_matrix[i + eigen_idx * n] * scale;
        }
    }
    
    return coords_3d;
}

} // namespace rf


