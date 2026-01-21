#ifndef RF_CUDA_CONFIG_HPP
#define RF_CUDA_CONFIG_HPP

#include "rf_types.hpp"
#include <stdexcept>

// Forward declarations to avoid CUDA headers during import
struct cudaDeviceProp;

namespace rf {
namespace cuda {

// GPU capability detection and configuration
class CudaConfig {
public:
    static CudaConfig& instance() {
        static CudaConfig config;
        return config;
    }

    // Initialize GPU configuration (CUDA headers included only when this is called)
    bool initialize();
    
    // Get optimal block size for given problem size
    int get_optimal_block_size(int problem_size) const;
    
    // Get maximum number of blocks
    int get_max_blocks() const;
    
    // Get shared memory per block
    size_t get_shared_memory_per_block() const;
    
    // Get total GPU memory
    size_t get_total_memory() const;
    
    // Check if GPU supports FP16
    bool supports_fp16() const;
    
    // Check if GPU supports Tensor Cores
    bool supports_tensor_cores() const;
    
    // Get GPU name
    const char* get_gpu_name() const;
    
    // Get compute capability
    int get_compute_capability_major() const;
    int get_compute_capability_minor() const;
    
    // Memory management helpers
    size_t get_available_memory() const;
    
    // Adaptive memory allocation
    template<typename T>
    size_t get_max_elements_for_type() const {
        if (!initialized_) return 0;
        size_t available = get_available_memory();
        size_t element_size = sizeof(T);
        
        // get_available_memory() already applies 20% safety margin
        // so the full available amount can be used here
        return available / element_size;
    }
    
    // Problem size recommendations
    bool can_handle_problem_size(int nsample, int mdim, int ntree) const;
    
    // Memory safety checks to prevent WSL crashes
    bool should_stop_on_insufficient_memory(int nsample, int mdim, int ntree) const;
    bool can_handle_cpu_problem_size(int nsample, int mdim, int ntree) const;
    
    // Get recommended batch size for large problems
    int get_recommended_batch_size(int total_trees) const;
    
    // Check if initialized
    bool is_initialized() const { return initialized_; }
    
    // Get device properties (returns nullptr if not initialized)
    const cudaDeviceProp* get_device_props() const;

private:
    CudaConfig() : initialized_(false), device_props_(nullptr) {}
    
    void configure_for_gpu();
    
    bool initialized_;
    cudaDeviceProp* device_props_;  // Will be allocated when CUDA is actually initialized
    int default_block_size_;
    int max_concurrent_kernels_;
};

// Convenience functions
inline int get_optimal_block_size(int problem_size) {
    return CudaConfig::instance().get_optimal_block_size(problem_size);
}

inline int get_max_blocks() {
    return CudaConfig::instance().get_max_blocks();
}

inline bool gpu_can_handle_problem(int nsample, int mdim, int ntree) {
    return CudaConfig::instance().can_handle_problem_size(nsample, mdim, ntree);
}

inline bool should_stop_on_insufficient_memory(int nsample, int mdim, int ntree) {
    return CudaConfig::instance().should_stop_on_insufficient_memory(nsample, mdim, ntree);
}

inline bool cpu_can_handle_problem(int nsample, int mdim, int ntree) {
    return CudaConfig::instance().can_handle_cpu_problem_size(nsample, mdim, ntree);
}

// Declaration - implementation in rf_config_cuda.cu
int get_recommended_batch_size(int total_trees);

// CUDA context management helpers for Jupyter lifecycle
// These functions ensure CUDA context is valid and error-free throughout the model lifecycle
void cuda_clear_errors();           // Clear any stale CUDA errors before operations
bool cuda_validate_context();       // Validate CUDA context is ready for operations
void cuda_ensure_context_ready();   // Ensure CUDA context is ready before operations (like fit, predict, etc.)
void cuda_finalize_operations();    // Finalize CUDA operations after fit/predict/etc.

} // namespace cuda
} // namespace rf

#endif // RF_CUDA_CONFIG_HPP