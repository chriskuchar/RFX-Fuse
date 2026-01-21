#ifndef RF_PROXIMITY_OPTIMIZED_HPP
#define RF_PROXIMITY_OPTIMIZED_HPP

#include "rf_types.hpp"
#include "rf_config.hpp"
#include "rf_memory.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>

namespace rf {

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

namespace proximity_kernels {
    // Proximity kernel using shared memory optimization
    void proximity_shared_memory_kernel(
        const integer_t* nodexb, const integer_t* nin,
        integer_t nsample, const integer_t* ndbegin, const integer_t* npcase,
        dp_t* prox);
    
    // Batch proximity kernel for processing multiple trees
    void proximity_batch_kernel(
        const integer_t* nodexb_batch, const integer_t* nin_batch,
        integer_t nsample, integer_t batch_size,
        const integer_t* ndbegin_batch, const integer_t* npcase_batch,
        dp_t* output_prox);
}

// ============================================================================
// Advanced Proximity Matrix Optimizations
// ============================================================================

// Persistent proximity matrix manager for GPU
class PersistentProximityMatrix {
public:
    PersistentProximityMatrix();
    ~PersistentProximityMatrix();
    
    // Initialize persistent matrix
    bool initialize(integer_t nsample, integer_t n_trees_estimate);
    
    // Add tree contribution to persistent matrix
    void add_tree_contribution(const integer_t* nodexb, const integer_t* nin,
                              integer_t nsample, integer_t nterm,
                              const integer_t* ndbegin, const integer_t* npcase);
    
    // Get final proximity matrix
    void get_final_matrix(dp_t* host_prox);
    
    // Reset for new forest
    void reset();
    
    // Memory usage statistics
    size_t get_gpu_memory_usage() const { return gpu_memory_usage_; }
    size_t get_cpu_memory_usage() const { return cpu_memory_usage_; }

private:
    integer_t nsample_;
    integer_t n_trees_processed_;
    
    // GPU persistent storage
    __half* gpu_prox_fp16_;           // FP16 proximity matrix on GPU
    dp_t* gpu_prox_fp32_;             // FP32 proximity matrix on GPU
    integer_t* gpu_nodexb_;            // Persistent node indices
    integer_t* gpu_nin_;              // Persistent in-bag counts
    integer_t* gpu_ndbegin_;          // Persistent node begins
    integer_t* gpu_npcase_;           // Persistent case indices
    
    // CPU staging area
    dp_t* cpu_staging_;               // CPU staging buffer
    
    // Memory usage tracking
    size_t gpu_memory_usage_;
    size_t cpu_memory_usage_;
    
    // CUDA streams for overlapping operations
    cudaStream_t compute_stream_;
    cudaStream_t memory_stream_;
    
    // cuBLAS handle for optimized operations
    cublasHandle_t cublas_handle_;
    
    // Internal methods
    void allocate_gpu_memory();
    void allocate_cpu_memory();
    void free_memory();
    void convert_fp16_to_fp32();
};

// Batch proximity processor for multiple trees
class BatchProximityProcessor {
public:
    BatchProximityProcessor();
    ~BatchProximityProcessor();
    
    // Process multiple trees in batch
    void process_trees_batch(
        const std::vector<integer_t*>& nodexb_batch,
        const std::vector<integer_t*>& nin_batch,
        const std::vector<integer_t*>& ndbegin_batch,
        const std::vector<integer_t*>& npcase_batch,
        integer_t nsample, integer_t batch_size,
        dp_t* output_prox);
    
    // Get optimal batch size for current GPU
    integer_t get_optimal_batch_size(integer_t nsample) const;

private:
    // Batch processing helper methods
    void launch_batch_proximity_kernel(
        const integer_t* nodexb_batch,
        const integer_t* nin_batch,
        const integer_t* ndbegin_batch,
        const integer_t* npcase_batch,
        integer_t nsample, integer_t batch_size,
        dp_t* output_prox);
    
    // Memory management
    integer_t* gpu_nodexb_batch_;
    integer_t* gpu_nin_batch_;
    integer_t* gpu_ndbegin_batch_;
    integer_t* gpu_npcase_batch_;
    dp_t* gpu_output_prox_;
    
    size_t max_batch_size_;
    size_t allocated_memory_;
};

// Sparse proximity matrix for high-dimensional data
class SparseProximityMatrix {
public:
    struct SparseEntry {
        integer_t row;
        integer_t col;
        dp_t value;
    };
    
    SparseProximityMatrix(integer_t nsample, real_t sparsity_threshold = 0.01);
    ~SparseProximityMatrix();
    
    // Add proximity contribution
    void add_contribution(integer_t row, integer_t col, dp_t value);
    
    // Convert to dense matrix
    void to_dense(dp_t* dense_matrix);
    
    // Get memory usage
    size_t get_memory_usage() const;
    
    // Compression ratio
    real_t get_compression_ratio() const;

private:
    integer_t nsample_;
    real_t sparsity_threshold_;
    std::vector<SparseEntry> entries_;
    std::unordered_map<integer_t, std::unordered_map<integer_t, dp_t>> matrix_;
    
    // GPU sparse storage
    integer_t* gpu_row_indices_;
    integer_t* gpu_col_indices_;
    dp_t* gpu_values_;
    integer_t* gpu_nnz_;
    
    void allocate_gpu_sparse();
    void update_gpu_sparse();
};

// Memory-optimized proximity computation
class MemoryOptimizedProximity {
public:
    MemoryOptimizedProximity();
    ~MemoryOptimizedProximity();
    
    // Compute proximity with memory optimization
    void compute_proximity_optimized(
        const integer_t* nodestatus, const integer_t* nodextr,
        const integer_t* nin, integer_t nsample, integer_t nnode,
        dp_t* prox, integer_t* nod, integer_t* ncount,
        integer_t* ncn, integer_t* nodexb, integer_t* ndbegin,
        integer_t* npcase);
    
    // Enable/disable optimizations
    void enable_fp16_storage(bool enable = true);
    void enable_batch_processing(bool enable = true);
    void enable_sparse_storage(bool enable = true);
    void enable_stream_processing(bool enable = true);
    
    // Performance tuning
    void set_memory_limit(size_t max_memory_mb);
    void set_batch_size(integer_t batch_size);

private:
    // Optimization flags
    bool use_fp16_storage_;
    bool use_batch_processing_;
    bool use_sparse_storage_;
    bool use_stream_processing_;
    
    // Memory management
    size_t max_memory_mb_;
    integer_t batch_size_;
    
    // Component processors
    std::unique_ptr<PersistentProximityMatrix> persistent_matrix_;
    std::unique_ptr<BatchProximityProcessor> batch_processor_;
    std::unique_ptr<SparseProximityMatrix> sparse_matrix_;
    
    // CUDA streams
    cudaStream_t streams_[4];
    
    // Internal methods
    void initialize_components();
    void cleanup_components();
    bool should_use_sparse(integer_t nsample) const;
    bool should_use_batch(integer_t n_trees) const;
};

// ============================================================================
// Advanced CUDA Kernels
// ============================================================================

namespace proximity_kernels {

// Optimized proximity kernel with shared memory
__global__ void proximity_shared_memory_kernel(
    const integer_t* nodexb, const integer_t* nin,
    integer_t nsample, const integer_t* ndbegin,
    const integer_t* npcase, dp_t* prox);

// Vectorized proximity kernel using tensor cores
__global__ void proximity_tensor_core_kernel(
    const integer_t* nodexb, const integer_t* nin,
    integer_t nsample, const integer_t* ndbegin,
    const integer_t* npcase, __half* prox_fp16);

// Sparse proximity kernel for high-dimensional data
__global__ void proximity_sparse_kernel(
    const integer_t* nodexb, const integer_t* nin,
    integer_t nsample, const integer_t* ndbegin,
    const integer_t* npcase, integer_t* row_indices,
    integer_t* col_indices, dp_t* values, integer_t* nnz);

// Batch proximity kernel for multiple trees
__global__ void proximity_batch_kernel(
    const integer_t* nodexb_batch, const integer_t* nin_batch,
    integer_t nsample, integer_t batch_size,
    const integer_t* ndbegin_batch, const integer_t* npcase_batch,
    dp_t* output_prox);

// Memory-efficient proximity kernel with tiling
__global__ void proximity_tiled_kernel(
    const integer_t* nodexb, const integer_t* nin,
    integer_t nsample, const integer_t* ndbegin,
    const integer_t* npcase, dp_t* prox,
    integer_t tile_size);

} // namespace proximity_kernels

// ============================================================================
// Performance Monitoring
// ============================================================================

class ProximityPerformanceMonitor {
public:
    struct ProximityStats {
        double total_time_ms;
        double preprocessing_time_ms;
        double kernel_time_ms;
        double memory_transfer_time_ms;
        double postprocessing_time_ms;
        size_t memory_usage_mb;
        double memory_bandwidth_gbps;
        double compute_efficiency;
    };
    
    void start_timing(const std::string& phase);
    void end_timing(const std::string& phase);
    void record_memory_usage(size_t bytes);
    void record_kernel_launch(const std::string& kernel_name, 
                             integer_t blocks, integer_t threads);
    
    ProximityStats get_stats() const;
    void print_stats() const;
    void reset();

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::unordered_map<std::string, double> timings_;
    size_t peak_memory_usage_;
    std::vector<std::string> kernel_launches_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

// Get optimal proximity computation strategy
enum class ProximityStrategy {
    DENSE_FP32,      // Standard dense FP32 computation
    DENSE_FP16,      // Memory-optimized FP16 computation
    SPARSE,          // Sparse matrix for high-dimensional data
    BATCH,           // Batch processing for multiple trees
    PERSISTENT       // Persistent GPU memory
};

ProximityStrategy get_optimal_proximity_strategy(
    integer_t nsample, integer_t n_trees, size_t available_memory_mb);

// Benchmark proximity computation methods
struct ProximityBenchmarkResults {
    double dense_fp32_time_ms;
    double dense_fp16_time_ms;
    double sparse_time_ms;
    double batch_time_ms;
    double persistent_time_ms;
    
    size_t dense_fp32_memory_mb;
    size_t dense_fp16_memory_mb;
    size_t sparse_memory_mb;
    size_t batch_memory_mb;
    size_t persistent_memory_mb;
    
    double accuracy_vs_reference;
};

ProximityBenchmarkResults benchmark_proximity_methods(
    integer_t nsample, integer_t n_trees);

} // namespace rf

#endif // RF_PROXIMITY_OPTIMIZED_HPP
