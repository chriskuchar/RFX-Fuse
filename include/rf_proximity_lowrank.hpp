#ifndef RF_PROXIMITY_LOWRANK_HPP
#define RF_PROXIMITY_LOWRANK_HPP

#include "rf_types.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // MUST be before cublas_v2.h
#include "rf_quantization_kernels.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <memory>

namespace rf {
namespace cuda {

/**
 * @brief Low-rank proximity matrix using AB' decomposition (QLoRA-style)
 * 
 * Stores proximity matrix P ≈ A × B' where:
 * - A: n_samples × rank (low-rank factor)
 * - B: n_samples × rank (low-rank factor)
 * - rank << n_samples (typically 100-1000)
 * 
 * Memory: O(2 × n_samples × rank) instead of O(n_samples²)
 * For 100k samples with rank=100: 160 MB vs 80 GB (500x reduction!)
 */
class LowRankProximityMatrix {
public:
    LowRankProximityMatrix(integer_t nsample, integer_t initial_rank = 100, 
                           QuantizationLevel quant_level = QuantizationLevel::FP16,
                           integer_t max_rank = 1000,
                           integer_t power_iterations = 30);
    ~LowRankProximityMatrix();
    
    /**
     * @brief Initialize GPU memory for low-rank factors
     * @return true if successful, false if insufficient GPU memory
     */
    bool initialize();
    
    /**
     * @brief Add tree contribution with incremental low-rank update (NO temp buffer!)
     * 
     * This method updates A and B factors directly without storing full temp buffer.
     * More memory efficient for large datasets, but slower (SVD per tree).
     * 
     * Strategy:
     * 1. Get rank-1 SVD of tree_proximity: P_tree ≈ u × v'
     * 2. Update: A_new = [A_old, u], B_new = [B_old, v]
     * 3. Periodically truncate to maintain rank
     * 
     * @param tree_proximity Temporary proximity matrix for this tree (n × n)
     * @param nsample Number of samples
     * @param use_incremental If true, use incremental update (no temp buffer)
     */
    void add_tree_contribution(const dp_t* tree_proximity, integer_t nsample, bool use_incremental = false);
    
    /**
     * @brief Add tree contribution using incremental low-rank update (memory efficient)
     * 
     * Updates A and B factors directly without temp buffer.
     * Use this for large datasets where temp buffer would be too large.
     * 
     * @param tree_proximity Temporary proximity matrix for this tree (n × n)
     * @param nsample Number of samples
     */
    void add_tree_contribution_incremental(const dp_t* tree_proximity, integer_t nsample);
    
    /**
     * @brief Add tree contribution directly from upper triangle (all quantization levels)
     * 
     * Ultra memory-efficient version that accepts pre-computed upper triangle in any quantization format.
     * This avoids the need for full matrix computation entirely!
     * 
     * Memory requirements:
     * - CPU: ~10 GB RAM (FP16), ~5 GB (INT8), ~2.5 GB (NF4)
     * - GPU: ~10 GB VRAM (FP16), ~5 GB (INT8), ~2.5 GB (NF4)
     * 
     * @param tree_proximity_upper Upper triangle of proximity matrix (packed row-wise)
     * @param nsample Number of samples
     * @param quant_level Quantization level of input (FP16, INT8, NF4)
     * @param int8_scale Scale parameter for INT8 quantization (required if quant_level == INT8)
     * @param int8_zero_point Zero point parameter for INT8 quantization (required if quant_level == INT8)
     */
    void add_tree_contribution_incremental_upper_triangle(
        const void* tree_proximity_upper,
        integer_t nsample,
        QuantizationLevel quant_level,
        float int8_scale = 1.0f / 127.0f,
        float int8_zero_point = 0.0f
    );
    
    /**
     * @brief Add tree contribution directly from upper triangle FP16 (ultra memory efficient!)
     * 
     * This version accepts pre-computed upper triangle FP16, avoiding the need for
     * full matrix computation. Use this when you can compute proximity directly
     * in upper triangle FP16 format (e.g., from GPU proximity kernel).
     * 
     * Memory requirements:
     * - CPU: ~10 GB RAM (upper triangle FP16)
     * - GPU: ~10 GB VRAM (upper triangle FP16)
     * 
     * @param tree_proximity_upper_fp16 Upper triangle of proximity matrix (packed row-wise, FP16)
     * @param nsample Number of samples
     */
    void add_tree_contribution_incremental_upper_triangle_fp16(const __half* tree_proximity_upper_fp16, integer_t nsample);
    
    /**
     * @brief Finalize accumulation and update quantized factors
     * 
     * Call this after all tree contributions have been added.
     * Performs SVD on accumulated proximity and updates quantized A and B factors.
     * After this, temp_prox_gpu_ can be freed (huge memory savings!).
     * 
     * @param n_trees_processed Number of trees processed (for logging)
     * @param final_call If true, quantize factors to save memory (only do this on the final call)
     */
    void finalize_accumulation(integer_t n_trees_processed = 0, bool final_call = false);
    
    /**
     * @brief Get accumulated proximity matrix (FP32) before factorization
     * 
     * Returns the full accumulated proximity matrix before SVD factorization.
     * Useful for compatibility with existing code that expects full matrix.
     * 
     * @param output_prox Output buffer (n_samples × n_samples)
     * @param nsample Number of samples
     */
    void get_accumulated_proximity(dp_t* output_prox, integer_t nsample);
    
    /**
     * @brief Reconstruct full proximity matrix (on-demand) with quantization support
     * 
     * Computes P = A × B' and stores in output buffer.
     * Supports quantized factors for memory efficiency.
     * If RF-GAP normalization is enabled (oob_counts_rfgap_ is set), normalizes each row i by |Si|.
     * 
     * @param output_prox Output buffer (n_samples × n_samples)
     * @param nsample Number of samples
     * @param quant_level Quantization level for factors (FP32, FP16, INT8, NF4)
     */
    void reconstruct_full_matrix(dp_t* output_prox, integer_t nsample, QuantizationLevel quant_level = QuantizationLevel::FP32);
    
    /**
     * @brief Quantize low-rank factors A and B
     * 
     * Quantizes A and B factors separately for memory efficiency.
     * Quantized factors can be used in GEMM operations with mixed precision.
     * 
     * @param quant_level Quantization level (FP16, INT8, NF4)
     * @note If factors are already quantized, this will re-quantize them
     */
    void quantize_factors(QuantizationLevel quant_level);
    
    /**
     * @brief Dequantize factors back to FP32 (if needed)
     * 
     * Dequantizes factors for operations that require full precision.
     * Use sparingly as it increases memory usage.
     */
    void dequantize_factors();
    
    /**
     * @brief Check if factors are currently quantized
     * @return true if factors are quantized
     */
    bool is_quantized() const { return current_quant_level_ != QuantizationLevel::FP32; }
    
    /**
     * @brief Get current quantization level
     * @return Current quantization level
     */
    QuantizationLevel get_quantization_level() const { return current_quant_level_; }
    
    /**
     * @brief Get low-rank factors A and B (for direct MDS computation)
     * 
     * Returns pointers to GPU memory for A and B factors.
     * Useful for computing distances directly from factors without reconstruction.
     * 
     * @param[out] A_gpu Pointer to A factor (n_samples × rank)
     * @param[out] B_gpu Pointer to B factor (n_samples × rank)
     * @param[out] rank Current rank
     */
    void get_factors(dp_t** A_gpu, dp_t** B_gpu, integer_t* rank);
    
    /**
     * @brief Compute pairwise distances from low-rank factors (for MDS)
     * 
     * Efficiently computes distance matrix D where:
     * D[i,j] = ||A[i,:] - A[j,:]||²
     * 
     * This avoids reconstructing the full proximity matrix.
     * 
     * @param output_distances Output buffer (n_samples × n_samples)
     * @param nsample Number of samples
     */
    void compute_distances_from_factors(dp_t* output_distances, integer_t nsample);
    
    /**
     * @brief Set OOB counts for RF-GAP normalization
     * 
     * For RF-GAP proximity, each row i needs to be normalized by |Si| (number of trees where sample i is OOB).
     * This method stores the OOB counts for normalization during MDS computation.
     * 
     * @param oob_counts Array of OOB counts per sample (length nsample_)
     */
    void set_oob_counts_for_rfgap(const integer_t* oob_counts);
    
    /**
     * @brief Compute MDS coordinates directly from low-rank factors (GPU ONLY)
     * 
     * Performs classical MDS on GPU without reconstructing full proximity matrix.
     * Algorithm:
     * 1. Reconstruct P = A × B^T (temporary, freed after use)
     * 2. If RF-GAP normalization is enabled, normalize each row i by |Si| (OOB count)
     * 3. Convert to distance matrix D = max_prox - P
     * 4. Double-center D
     * 5. Eigendecompose to get top k eigenvectors
     * 6. Return k-dimensional coordinates (n_samples × k)
     * 
     * Memory efficient: Only allocates O(n²) temporarily during computation.
     * 
     * NOTE: This method is GPU-only and requires CUDA. For CPU, use compute_mds_3d_cpu()
     * with the full proximity matrix.
     * 
     * @param k Number of dimensions (default: 3 for 3D visualization)
     * @return std::vector<double> k-dimensional coordinates (n_samples × k), row-major: [x0,y0,z0,..., x1,y1,z1,..., ...]
     * @throws std::runtime_error If GPU memory insufficient or computation fails
     */
    std::vector<double> compute_mds_from_factors(integer_t k = 3);
    
    /**
     * @brief Compute 3D MDS coordinates directly from low-rank factors (GPU ONLY)
     * 
     * Convenience method that calls compute_mds_from_factors(3).
     * 
     * @return std::vector<double> 3D coordinates (n_samples × 3), row-major: [x0,y0,z0, x1,y1,z1, ...]
     * @throws std::runtime_error If GPU memory insufficient or computation fails
     */
    std::vector<double> compute_mds_3d_from_factors() {
        return compute_mds_from_factors(3);
    }
    
    /**
     * @brief Auto-adjust rank based on variance captured
     * 
     * Performs SVD to determine optimal rank that captures target_variance
     * (e.g., 95% or 99% of variance).
     * 
     * @param target_variance Target variance to capture (0.95 = 95%)
     * @return New rank
     */
    integer_t auto_adjust_rank(double target_variance = 0.95);
    
    /**
     * @brief Get current memory usage
     * @return Memory usage in bytes
     */
    /**
     * @brief Get current memory usage
     * @return Memory usage in bytes (including quantized factors)
     */
    size_t get_memory_usage() const;
    
    /**
     * @brief Get memory usage if using quantized storage
     * @param quant_level Quantization level
     * @return Memory usage in bytes
     */
    size_t get_memory_usage(QuantizationLevel quant_level) const;
    
    /**
     * @brief Check if incremental updates are beneficial (no temp buffer)
     * 
     * Returns true if incremental A/B updates should be used instead of temp buffer.
     * Decision based on dataset size and available memory.
     * 
     * @param nsample Number of samples
     * @param available_memory Available GPU memory in bytes
     * @return true if incremental updates are beneficial
     */
    static bool should_use_incremental(integer_t nsample, size_t available_memory = 0);

private:
    integer_t nsample_;
    integer_t rank_;  // Current rank of low-rank approximation
    integer_t max_rank_;  // Maximum allowed rank
    integer_t power_iterations_;  // Power iteration count for SVD (30=99% convergence, 100=99.9%)
    
    // RF-GAP normalization: OOB counts per sample (|Si| for each sample i)
    std::vector<integer_t> oob_counts_rfgap_;  // Empty if RF-GAP not used
    /**
     * @brief Helper function: Perform rank-1 SVD from upper triangle FP16 matrix
     * 
     * Internal function that performs power iteration SVD on upper triangle FP16 matrix
     * and updates A and B factors.
     * 
     * @param tree_prox_upper_gpu Upper triangle of proximity matrix (packed row-wise, FP16, GPU)
     * @param nsample Number of samples
     */
    void perform_rank1_svd_from_upper_triangle_fp16(const __half* tree_prox_upper_gpu, integer_t nsample);
    
    /**
     * @brief Helper function: Symmetric matrix-vector multiply using upper triangle storage (FP16)
     * 
     * Computes y = P × x where P is symmetric and stored in upper triangle format as FP16.
     * Used for power iteration in rank-1 SVD with quantized temporary buffer.
     * 
     * @param P_upper_gpu Upper triangle of symmetric matrix (packed row-wise, FP16)
     * @param x_gpu Input vector (FP32)
     * @param y_gpu Output vector (FP32, y = P × x)
     * @param nsample Number of samples
     */
    void symmetric_matrix_vector_multiply_fp16(
        const __half* P_upper_gpu,
        const dp_t* x_gpu,
        dp_t* y_gpu,
        integer_t nsample
    );
    
    /**
     * @brief Helper function: Symmetric matrix-vector multiply using upper triangle storage
     * 
     * Computes y = P × x where P is symmetric and stored in upper triangle format.
     * Used for power iteration in rank-1 SVD.
     * 
     * @param P_upper_gpu Upper triangle of symmetric matrix (packed row-wise)
     * @param x_gpu Input vector
     * @param y_gpu Output vector (y = P × x)
     * @param nsample Number of samples
     */
    void symmetric_matrix_vector_multiply(
        const dp_t* P_upper_gpu,
        const dp_t* x_gpu,
        dp_t* y_gpu,
        integer_t nsample
    );
    
    /**
     * @brief Truncate factors to maintain rank (for incremental updates)
     * 
     * After incremental updates, rank may grow. This truncates to maintain target rank.
     * Uses SVD to find best rank-r approximation.
     * 
     * @param target_rank Target rank after truncation
     */
    void truncate_rank(integer_t target_rank);
    
    integer_t trees_processed_;  // Track number of trees for periodic truncation
    
    // GPU memory for low-rank factors
    // Primary storage: quantized factors (FP16/INT8/NF4) for memory efficiency
    void* A_quantized_gpu_;  // Quantized A factor (primary storage)
    void* B_quantized_gpu_;  // Quantized B factor (primary storage)
    QuantizationLevel current_quant_level_;
    
    // Temporary FP32 buffers (only allocated when needed for dequantization)
    dp_t* A_gpu_;  // FP32 A factor (temporary, only if dequantized)
    dp_t* B_gpu_;  // FP32 B factor (temporary, only if dequantized)
    
    // Scaling parameters for INT8 quantization (host values for compatibility)
    float A_scale_, A_zero_point_;
    float B_scale_, B_zero_point_;
    
    // Device pointers for scale/zero_point (primary storage to avoid host copies)
    float* d_A_scale_;
    float* d_A_zero_point_;
    float* d_B_scale_;
    float* d_B_zero_point_;
    
    // cuBLAS and cuSolver handles
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    
    // Workspace for SVD
    dp_t* workspace_gpu_;
    size_t workspace_size_;
    
    // Temporary buffer for tree contributions
    dp_t* temp_prox_gpu_;
    
    /**
     * @brief Incremental rank-1 update: A_new = [A_old, a], B_new = [B_old, b]
     * 
     * Adds new column to A and B factors.
     * More efficient than full SVD for single tree updates.
     */
    void incremental_rank1_update(const dp_t* tree_proximity, integer_t nsample);
    
    /**
     * @brief Full SVD update (called periodically or when rank adjustment needed)
     */
    void full_svd_update(const dp_t* accumulated_prox, integer_t nsample);
    
    /**
     * @brief Allocate GPU memory
     */
    void allocate_gpu_memory();
    
    /**
     * @brief Free GPU memory
     */
    void free_gpu_memory();
    
    /**
     * @brief Ensure cuBLAS/cuSolver handles are created (lazy initialization)
     * 
     * Creates handles on first use to avoid constructor timing issues.
     * This ensures handles are created when CUDA context is fully ready.
     */
    void ensure_handles();
    
    /**
     * @brief Ensure GPU memory is allocated (lazy allocation)
     * 
     * Allocates memory on first use to avoid CUDA context issues during initialization.
     * This ensures memory is allocated when CUDA context is fully ready.
     */
    void ensure_allocated();
};

/**
 * @brief Estimate optimal rank for given dataset size
 * 
 * Uses heuristic: rank = min(100, sqrt(n_samples), max_rank)
 * 
 * @param nsample Number of samples
 * @param max_rank Maximum rank to use
 * @return Optimal rank
 */
integer_t estimate_optimal_rank(integer_t nsample, integer_t max_rank = 1000);

} // namespace cuda
} // namespace rf

#endif // RF_PROXIMITY_LOWRANK_HPP

