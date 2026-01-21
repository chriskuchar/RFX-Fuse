#ifndef RF_CONFIG_HPP
#define RF_CONFIG_HPP

#include "rf_types.hpp"
#include <string>
#include <cmath>
#include <vector>

namespace rf {

// CUDA configuration constants
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int CUDA_MAX_BLOCKS = 65535;
constexpr int CUDA_WARP_SIZE = 32;

// Random Forest configuration
struct RFConfig {
    // Execution mode
    bool use_gpu = false;
    bool force_cpu = false;

    // Forest parameters
    integer_t nsample = 150;      // Number of samples
    integer_t ntree = 100;        // Number of trees
    integer_t mdim = 4;           // Number of features
    integer_t nclass = 3;         // Number of classes
    integer_t maxcat = 100;       // Max categories per feature

    // Tree parameters
    integer_t mtry = 2;           // Features to try at each split
    integer_t maxnode = 301;      // Max nodes (2*nsample + 1)
    integer_t minndsize = 1;      // Min node size

    // Computation flags
    integer_t imp = 1;            // Compute overall variable importance
    integer_t impn = 1;           // Compute local variable importance
    integer_t iprox = 1;          // Compute proximity matrix
    bool use_rfgap = false;       // Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity
    bool use_casewise = false;    // Use case-wise calculations (per-sample) vs non-case-wise (aggregated)
                                  // Non-case-wise follows UC Berkeley standard: simple averaging/aggregation
                                  // Case-wise follows Breiman-Cutler: per-sample tracking with bootstrap weights
    
    // Local importance method selection
    std::string importance_method = "local_imp";  // "local_imp" or "clique"
    integer_t clique_M = -1;      // CLIQUE quantile grid size (-1 = auto, positive integer = explicit M)

    // QLORA-Proximity configuration
    bool use_qlora = true;
    integer_t quant_mode = 1;     // 0=FP32, 1=FP16, 2=INT8, 3=NF4
    integer_t lowrank_rank = 100; // Low-rank proximity matrix rank (A and B factors: n_samples × rank)
                                  // Default: 100. Higher rank = better approximation but more memory.
                                  // Memory usage: 2 × n_samples × rank × bytes_per_element
                                  // For 100k samples: rank=100 → ~40MB (FP16), rank=200 → ~80MB (FP16)
    integer_t lowrank_power_iterations = 30; // Power iteration count for low-rank SVD (30=99% convergence, 100=99.9%)
                                              // Default: 30. Higher = more accurate but slower convergence.
                                              // Typically: 30 iterations converges for most cases.
    integer_t lora_rank = 0;      // LoRA rank (0=disabled, deprecated - use lowrank_rank instead)
    integer_t fusion_interval = 1; // Trees between fusion
    bool use_sparse = false;       // Use sparse storage
    bool upper_triangle = true;    // Store upper triangle only
    real_t sparsity_threshold = 1e-9f;

    // Random seed
    integer_t iseed = 12345;

    // GPU-only advanced features
    // Loss function type for GPU tree growing
    // 0=Gini (classification only)
    integer_t gpu_loss_function = 0;  // Default: 0 (Gini)
    real_t min_gain_to_split = 0.0f;  // Minimum gain threshold for early stopping (GPU only)
    bool gpu_parallel_mode0 = false;  // Automatically set based on batch_size (true if batch_size > 1, false if batch_size == 1). Not a user parameter.
    real_t l1_regularization = 0.0f;  // L1 regularization strength (lambda) - penalizes |leaf_value|
    real_t l2_regularization = 0.0f;  // L2 regularization strength (lambda) - penalizes leaf_value^2

    // CPU multi-threading
    integer_t n_threads_cpu = 0;  // Number of CPU threads (0 = auto-detect, use all available cores)
    
    // Histogram binning for GPU (set by fit_batch_gpu when histogram data is available)
    bool use_histogram = false;   // Whether to use histogram-based split finding on GPU
    integer_t n_bins = 256;       // Number of bins for histogram (max 256 for uint8_t storage)

    // Helper methods
    void init_from_params(integer_t ns, integer_t nt, integer_t md, integer_t nc, integer_t mc) {
        nsample = ns;
        ntree = nt;
        mdim = md;
        nclass = nc;
        maxcat = mc;

        // Derived parameters
        mtry = static_cast<integer_t>(std::sqrt(static_cast<real_t>(mdim)));
        maxnode = 2 * nsample + 1;
    }

    bool using_gpu() const {
        return use_gpu && !force_cpu;
    }
    
    // Convert quant_mode to QuantizationLevel enum
    template<typename QuantizationLevelEnum>
    QuantizationLevelEnum get_quantization_level() const {
        switch (quant_mode) {
            case 0: return static_cast<QuantizationLevelEnum>(0); // FP32
            case 1: return static_cast<QuantizationLevelEnum>(1); // FP16
            case 2: return static_cast<QuantizationLevelEnum>(2); // INT8
            case 3: return static_cast<QuantizationLevelEnum>(3); // NF4
            default: return static_cast<QuantizationLevelEnum>(1); // Default to FP16
        }
    }
    
    // Memory estimation
    size_t estimate_memory_usage() const {
        size_t base_memory = nsample * mdim * sizeof(real_t);
        size_t tree_memory = ntree * maxnode * sizeof(real_t) * 8; // Rough estimate
        size_t proximity_memory = iprox ? nsample * nsample * sizeof(dp_t) : 0;
        return base_memory + tree_memory + proximity_memory;
    }
};

// Global configuration instance
extern RFConfig g_config;

// CUDA status functions (in rf::cuda namespace)
namespace cuda {
    bool cuda_is_available();
    bool cuda_init_runtime(bool force_cpu);
    void cuda_cleanup();
    void cuda_reset_device();  // Forcefully reset CUDA device (for stuck contexts)
}

} // namespace rf

#endif // RF_CONFIG_HPP
