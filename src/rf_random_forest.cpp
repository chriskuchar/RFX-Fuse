
#include "rf_random_forest.hpp"
#include "rf_config.hpp"
#include "rf_arrays.hpp"
#include "rf_utils.hpp"
#include "rf_utilities.hpp"
#include "rf_bootstrap.hpp"
#include "rf_getamat.hpp"
#include "rf_testreebag.hpp"
#include "rf_varimp.hpp"
#include "rf_predict.hpp"
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <set>
#include <sstream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <random>  // For std::mt19937, std::shuffle (unsupervised synthetic data)
#include <numeric>  // For std::iota
#ifdef _OPENMP
#include <omp.h>
#endif

// Avoid including CUDA headers directly in .cpp file to prevent static initialization issues
// Use forward declarations for GPU functions - they're in separate .cu files that are compiled separately
// Include only CPU headers
#include "rf_growtree.hpp"  // CPU tree growing functions (cpu_growtree_wrapper)
#include "rf_proximity.hpp"  // CPU proximity implementation
#include "rf_finishprox.hpp"

// Forward declare GPU functions - they're in separate .cu files
#ifdef CUDA_FOUND
namespace rf {
    // GPU tree growing (declaration only - defaults in rf_growtree.hpp)
    extern void gpu_growtree_batch(integer_t num_trees, const real_t* x, const real_t* win, const integer_t* cl,
                                    integer_t task_type, const integer_t* cat, const integer_t* ties, integer_t* nin,
                                    const real_t* y_regression, integer_t mdim, integer_t nsample, integer_t nclass,
                                    integer_t maxnode, integer_t minndsize, integer_t mtry, integer_t ninbag,
                                    integer_t maxcat, const integer_t* seeds, integer_t ncsplit, integer_t ncmax,
                                    integer_t* amat_all, integer_t* jinbag_all, real_t* tnodewt_all, real_t* xbestsplit_all,
                                    integer_t* nodestatus_all, integer_t* bestvar_all, integer_t* treemap_all,
                                    integer_t* catgoleft_all, integer_t* nodeclass_all, integer_t* nnode_all,
                                    integer_t* jtr_all, integer_t* nodextr_all, integer_t* idmove_all,
                                    integer_t* igoleft_all, integer_t* igl_all, integer_t* nodestart_all,
                                    integer_t* nodestop_all, integer_t* itemp_all, integer_t* itmp_all,
                                    integer_t* icat_all, real_t* tcat_all, real_t* classpop_all, real_t* wr_all,
                                    real_t* wl_all, real_t* tclasscat_all, real_t* tclasspop_all, real_t* tmpclass_all,
                                    real_t* q_all, integer_t* nout_all, real_t* avimp_all,
                                    real_t* qimp_all, real_t* qimpm_all, dp_t* proximity_all,
                                    real_t* oob_predictions_all,
                                    real_t* nodepred_all,
                                    void** lowrank_proximity_ptr_out,
                                    real_t* prox_imp_all,
                                    int16_t* leaf_assignments_all,
                                    bool use_histogram,
                                    const uint8_t* X_binned,
                                    const real_t* bin_edges_all,
                                    const integer_t* n_bins_per_feature,
                                    integer_t max_bins);
    
    // GPU histogram data structure (forward declaration - implemented in cuda/rf_histogram_gpu.cu)
    namespace cuda {
        struct GPUHistogramData;
        // Cleanup function to avoid incomplete type issues
        void gpu_histogram_cleanup(void* gpu_hist_ptr);
    }
    
    // CLIQUE removed for v1.0 (v2.0 feature)
    // Forward declaration for GPU CLIQUE (defined in cuda/rf_clique.cu) - REMOVED
    /* extern void compute_clique_gpu(...); */
}
#endif

// Include CUDA headers only when CUDA is available (for RF-GAP to low-rank conversion)
#ifdef CUDA_FOUND
#include "rf_proximity_lowrank.hpp"
#include "rf_lowrank_helpers.hpp"
#include "rf_sparse_forest.cuh"  // GPU sparse training
#endif

namespace rf {
namespace cuda {
    // Forward declaration for CUDA config functions (defined in cuda/rf_config_cuda.cu)
    bool cuda_init_runtime(bool force_cpu);
    bool cuda_is_available();
    integer_t get_recommended_batch_size(integer_t ntree);  // Non-inline implementation in rf_config_cuda.cu
    // Forward declarations for CUDA context management helpers (defined in cuda/rf_config_cuda.cu)
    void cuda_clear_errors();           // Clear any stale CUDA errors before operations
    void cuda_ensure_context_ready();   // Ensure CUDA context is ready before operations
    void cuda_finalize_operations();    // Finalize CUDA operations after fit/predict/etc.
    // Forward declaration for low-rank helper function (defined in cuda/rf_lowrank_helpers.cu)
    bool get_lowrank_factors_host(void* lowrank_proximity_ptr, 
                                   dp_t** A_host, dp_t** B_host, 
                                   integer_t* r, integer_t nsamples);
    std::vector<double> compute_mds_3d_from_factors_host(void* lowrank_proximity_ptr);
    // Forward declaration for LowRankProximityMatrix (defined in cuda/rf_proximity_lowrank.cu)
    class LowRankProximityMatrix;
} // namespace cuda

// Forward declaration for GPU RF-GAP (defined in cuda/rf_proximity.cu)
void gpu_proximity_rfgap(integer_t ntree, integer_t nsample,
                        const integer_t* tree_nin_flat,
                        const integer_t* tree_nodextr_flat,
                        dp_t* prox);
} // namespace rf

namespace rf {

// Standalone train_test_split function (doesn't require RandomForest instance)
void train_test_split(const real_t* X, const void* y, integer_t nsamples, integer_t mdim,
                     real_t* X_train, void* y_train, real_t* X_test, void* y_test,
                     integer_t& n_train, integer_t& n_test, 
                     real_t test_size, integer_t random_state, bool stratify) {
    
    // Validate inputs
    if (!X || !y || !X_train || !y_train || !X_test || !y_test) {
        return;
    }
    
    if (nsamples <= 0 || mdim <= 0 || test_size <= 0.0 || test_size >= 1.0) {
        return;
    }
    
    // Calculate split sizes
    n_test = static_cast<integer_t>(nsamples * test_size);
    n_train = nsamples - n_test;
    
    // Validate split sizes
    if (n_train <= 0 || n_test <= 0 || n_train >= nsamples || n_test >= nsamples) {
        return;
    }
    
    // Simple random split without stratification for now
    // Use heap allocation to avoid stack overflow in Jupyter kernels
    std::unique_ptr<integer_t[]> indices(new integer_t[nsamples]);
    
    for (integer_t i = 0; i < nsamples; i++) {
        indices[i] = i;
    }
    
    // Simple shuffle
    MT19937 rng;
    rng.sgrnd(random_state);
    
    for (integer_t i = nsamples - 1; i > 0; i--) {
        integer_t j = static_cast<integer_t>(rng.grnd() * (i + 1));
        std::swap(indices[i], indices[j]);
    }
    
    // Copy data to train/test arrays
    for (integer_t i = 0; i < n_train; i++) {
        integer_t idx = indices[i];
        if (idx < 0 || idx >= nsamples) {
            return;
        }
        // Copy features
        for (integer_t j = 0; j < mdim; j++) {
            X_train[i * mdim + j] = X[idx * mdim + j];
        }
        // Copy labels - cast through real_t first to handle float input
        static_cast<integer_t*>(y_train)[i] = static_cast<integer_t>(static_cast<const real_t*>(y)[idx]);
    }
    
    for (integer_t i = 0; i < n_test; i++) {
        integer_t idx = indices[n_train + i];
        if (idx < 0 || idx >= nsamples) {
            return;
        }
        // Copy features
        for (integer_t j = 0; j < mdim; j++) {
            X_test[i * mdim + j] = X[idx * mdim + j];
        }
        // Copy labels - cast through real_t first to handle float input
        static_cast<integer_t*>(y_test)[i] = static_cast<integer_t>(static_cast<const real_t*>(y)[idx]);
    }
}

// Helper function to get memory information
void print_memory_info(integer_t nsample, integer_t mdim, integer_t ntree, integer_t nclass, integer_t maxnode = 1000, bool use_gpu = false) {
    // Get system memory info from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long total_mem_kb = 0, available_mem_kb = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> total_mem_kb;
        } else if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string key;
            iss >> key >> available_mem_kb;
        }
    }
    
    // Convert to MB
    double total_mem_mb = total_mem_kb / 1024.0;
    double available_mem_mb = available_mem_kb / 1024.0;
    double used_mem_mb = total_mem_mb - available_mem_mb;
    
    // Estimate training memory usage
    double estimated_mb = 0;
    estimated_mb += nsample * mdim * 4 / (1024.0 * 1024.0);  // x array
    estimated_mb += nsample * 4 / (1024.0 * 1024.0);  // y array
    estimated_mb += 2 * maxnode * ntree * 4 / (1024.0 * 1024.0);  // treemap
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // nodestatus
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // xbestsplit
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // bestvar
    estimated_mb += maxnode * ntree * 4 / (1024.0 * 1024.0);  // nodeclass
    estimated_mb += nclass * maxnode * ntree * 4 / (1024.0 * 1024.0);  // classpop
    estimated_mb += nsample * 4 / (1024.0 * 1024.0);  // qimp
    estimated_mb += nsample * mdim * 4 / (1024.0 * 1024.0);  // qimpm
    estimated_mb += mdim * 4 / (1024.0 * 1024.0);  // feature_importances
    estimated_mb += nsample * nsample * 4 / (1024.0 * 1024.0);  // proximity
    
    // Print memory information
    // NOTE: GPU memory info is now printed via Python bindings (print_gpu_memory_status())
    // to avoid std::cout stream conflicts with Python progress bars. This section is disabled.
    if (use_gpu) {
        // GPU memory info printing moved to Python bindings for safety
        // The Python bindings call print_gpu_memory_status() which uses Python's print()
        // This avoids std::codecvt crashes when C++ streams conflict with Python stdout/stderr
        (void)estimated_mb;  // Suppress unused variable warning
        /*
        std::cout << "\n🖥️  GPU MEMORY INFORMATION\n";
        std::cout << "==================================================\n";
        
        // Get GPU memory info safely (may fail in Jupyter, so wrap in try-catch)
        size_t free_mem = 0, total_mem = 0;
        bool memory_info_available = false;
        
        try {
            // Skip GPU memory check in CPU compilation path to avoid CUDA header dependency
            // Memory info will be available when GPU code is actually called
            // GPU memory queries require CUDA headers, which cause static init issues
            memory_info_available = false;  // Disable GPU memory check in CPU path
        } catch (...) {
            // Ignore - CUDA might not be ready or context might be invalid
        }
        
        if (memory_info_available) {
            std::cout << "📊 GPU Memory:\n";
            std::cout << "   Total: " << std::fixed << std::setprecision(1) << (total_mem / (1024.0 * 1024.0)) << " MB\n";
            std::cout << "   Available: " << (free_mem / (1024.0 * 1024.0)) << " MB\n";
            std::cout << "   Used: " << ((total_mem - free_mem) / (1024.0 * 1024.0)) << " MB\n";
            std::cout << "\n";
            std::cout << "📈 Estimated GPU Training Memory: " << estimated_mb << " MB\n";
            
            if (free_mem > 0) {
                std::cout << "📈 GPU Memory Usage: " << (estimated_mb / (free_mem / (1024.0 * 1024.0)) * 100) << "% of available\n";
                
                if (estimated_mb < (free_mem / (1024.0 * 1024.0)) * 0.8) {
                    std::cout << "📈 GPU Memory Safety: ✅ SAFE\n";
                } else {
                    std::cout << "📈 GPU Memory Safety: ⚠️  HIGH USAGE\n";
                }
            } else {
                std::cout << "📈 Estimated GPU Training Memory: " << estimated_mb << " MB\n";
                std::cout << "📈 GPU Memory Safety: ⚠️  Unable to query memory\n";
            }
        } else {
            // Fallback when memory info not available
            std::cout << "📊 GPU Memory: Info not available (CUDA context may not be fully initialized)\n";
            std::cout << "📈 Estimated GPU Training Memory: " << estimated_mb << " MB\n";
            std::cout << "📈 GPU Memory Safety: ⚠️  Unable to verify\n";
        }
        */
    }
}

// Train-test split method (like XGBoost) - now just calls the standalone function
void RandomForest::train_test_split(const real_t* X, const void* y, integer_t nsamples, integer_t mdim,
                                   real_t* X_train, void* y_train, real_t* X_test, void* y_test,
                                   integer_t& n_train, integer_t& n_test, 
                                   real_t test_size, integer_t random_state, bool stratify) {
    // Delegate to standalone function
    rf::train_test_split(X, y, nsamples, mdim, X_train, y_train, X_test, y_test, 
                        n_train, n_test, test_size, random_state, stratify);
}

RandomForest::RandomForest(const RandomForestConfig& config) :
    config_(config),
    oob_error_(0.0),
    oob_mse_(0.0),
    proximity_normalized_(false),
    inverted_leaf_index_built_(false),
    progress_callback_(nullptr),
    lowrank_proximity_ptr_(nullptr),
    gpu_histogram_data_(nullptr) {

    // Use only use_gpu flag - execution_mode is deprecated
    // config_.use_gpu is already set from config.use_gpu
    // Don't override use_gpu here - CUDA will be initialized lazily during fit()
    // The user's use_gpu setting from Python should be preserved
    initialize_arrays();
}

RandomForest::~RandomForest() {
    // Minimal destructor - let std::vector destructors handle cleanup automatically
    // Explicit clearing can cause issues during Python GC, so RAII is used
    // The vectors will be destroyed in reverse order of declaration automatically
    
    // Just clear the progress callback to break any circular references
    try {
        progress_callback_ = nullptr;
    } catch (...) {
        // Ignore any errors during destruction
    }
    
    // Clean up low-rank proximity matrix if it exists
    // Use exception-safe cleanup for Jupyter notebook compatibility
    if (lowrank_proximity_ptr_ != nullptr) {
        #ifdef CUDA_FOUND
        try {
            // Cast to LowRankProximityMatrix and delete
            // LowRankProximityMatrix destructor will handle GPU cleanup safely
            // The destructor is exception-safe and won't throw
            rf::cuda::LowRankProximityMatrix* lowrank_prox = 
                static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_proximity_ptr_);
            delete lowrank_prox;
        } catch (const std::exception& e) {
            // Log but don't rethrow - called from destructor
            // In Jupyter, CUDA context might be corrupted, which is OK
        } catch (...) {
            // Ignore all other errors during cleanup - process might be shutting down
            // CUDA runtime might already be destroyed
        }
        #endif
        lowrank_proximity_ptr_ = nullptr;
    }
    
    // Clean up GPU histogram data if it exists
    if (gpu_histogram_data_ != nullptr) {
        #ifdef CUDA_FOUND
        rf::cuda::gpu_histogram_cleanup(gpu_histogram_data_);
        #endif
        gpu_histogram_data_ = nullptr;
    }
    
    // Do NOT explicitly clear vectors in destructor
    // This can cause double-free errors if Python still holds references
    // Let the vector destructors handle cleanup automatically
    // Only clear if no references exist (cannot be guaranteed)
    // The vectors will be automatically destroyed when the object is destroyed
    
    // Don't explicitly clear other vectors - let their destructors handle it
    // This is safer during Python garbage collection
    // Don't call cuda_cleanup() in destructor - it's global and should only be called explicitly
}

void RandomForest::initialize_arrays() {
    // Allocate tree storage (ntree trees × maxnode nodes)
    // Use assign() to ensure all elements are zeroed when resizing
    // (resize() only zeros NEW elements, which causes bugs when nsample changes)
    // Cast to size_t to avoid integer overflow with large ntree * maxnode
    size_t tree_node_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.maxnode);
    size_t tree_2node_size = static_cast<size_t>(config_.ntree) * 2 * static_cast<size_t>(config_.maxnode);
    size_t tree_catnode_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
    
    tnodewt_.assign(tree_node_size, 0.0f);
    node_predictions_regression_.assign(tree_node_size, 0.0f);  // Regression predictions
    xbestsplit_.assign(tree_node_size, 0.0f);
    nodestatus_.assign(tree_node_size, 0);
    bestvar_.assign(tree_node_size, 0);
    treemap_.assign(tree_2node_size, 0);
    catgoleft_.assign(tree_catnode_size, 0);
    nodeclass_.assign(tree_node_size, 0);
    nnode_.assign(static_cast<size_t>(config_.ntree), 0);

    // Allocate OOB tracking - use assign() to ensure all elements are zeroed
    // (resize() only zeros NEW elements, not existing ones)
    // Cast to size_t to avoid integer overflow
    size_t q_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.nclass);
    q_.assign(q_size, 0.0f);
    nout_.assign(static_cast<size_t>(config_.nsample), 0);

    // Allocate variable type array (default: all quantitative)
    cat_.resize(config_.mdim, 1);

    // Allocate ties and sorted indices - use assign() to ensure all elements are zeroed
    // Cast to size_t to avoid integer overflow
    size_t ties_size = static_cast<size_t>(config_.mdim) * static_cast<size_t>(config_.nsample);
    ties_.assign(ties_size, 0);
    asave_.assign(ties_size, 0);

    // Initialize workspace arrays (to avoid repeated allocation/deallocation)
    initialize_workspace_arrays();

    // Allocate proximity if requested
    // NOTE: For low-rank/upper-triangle mode, defer allocation until after training
    // to avoid allocating full 80GB matrix for 100k samples
    if (config_.compute_proximity) {
        // Only allocate full matrix if NOT using low-rank
        // For low-rank mode, allocation will happen in GPU code
        // Check global config for flags (they're set via g_config, not config_)
        bool use_lowrank = config_.use_qlora;  // QLoRA is flag-based, not sample-size based
        bool use_upper_triangle = true;  // Default to true for memory efficiency
        
        if (!(use_lowrank && use_upper_triangle)) {
            // Traditional full matrix allocation for small datasets or when qlora disabled
            proximity_matrix_.assign(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
        }
        // For low-rank mode, proximity_matrix_ will be allocated later in GPU code
    }

    // Initialize CUDA configuration if using GPU
    // Note: CudaConfig is forward declared, so we can't call initialize() here
    // CUDA initialization will happen lazily when GPU code is actually called
    if (config_.use_gpu) {
        // CUDA initialization deferred to avoid static init issues
    }

    // Allocate feature importances if requested
    if (config_.compute_importance) {
        integer_t mimp = config_.mdim;
        feature_importances_.assign(mimp, 0.0f);
        
        // Allocate global importance arrays for accumulation across trees
        qimp_.assign(config_.nsample, 0.0f);
        // Only allocate qimpm_ if local importance is requested
        // qimpm_ is only needed for local importance, not for overall importance
        if (config_.compute_local_importance) {
            // Cast to size_t to avoid integer overflow
            size_t qimpm_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
            qimpm_.assign(qimpm_size, 0.0f);
        }
        avimp_.assign(config_.mdim, 0.0f);
        sqsd_.assign(config_.mdim, 0.0f);
    }
    
    // Allocate proximity importance if requested (works for all modes: classification, regression, unsupervised)
    // This measures the % of trees where permuting a feature changes the terminal node for each OOB sample.
    // Useful for interpreting WHY samples have similar proximity (especially for recommender systems).
    if (config_.compute_proximity_importance) {
        // Cast to size_t to avoid integer overflow
        size_t prox_imp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        proximity_importance_.assign(prox_imp_size, 0.0f);
    }
}

void RandomForest::detect_categorical_features(const real_t* X, integer_t nsample, integer_t mdim, integer_t maxcat) {
    // Auto-detect categorical features:
    // A feature is categorical if:
    // 1. All values are integers (or very close to integers)
    // 2. Number of unique values <= maxcat
    // 3. Values are in range [0, maxcat-1] or can be mapped to that range
    
    cat_.resize(mdim, 1);  // Initialize all as quantitative (1)
    
    for (integer_t m = 0; m < mdim; ++m) {
        std::set<real_t> unique_values;
        bool all_integers = true;
        real_t min_val = 1e10f;
        real_t max_val = -1e10f;
        
        // Check all samples for this feature
        for (integer_t n = 0; n < nsample; ++n) {
            real_t val = X[n * mdim + m];
            unique_values.insert(val);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            
            // Check if value is close to an integer
            real_t rounded = std::round(val);
            if (std::abs(val - rounded) > 1e-5f) {
                all_integers = false;
            }
        }
        
        // Check if feature is categorical
        integer_t num_unique = static_cast<integer_t>(unique_values.size());
        if (all_integers && num_unique > 1 && num_unique <= maxcat && 
            min_val >= 0.0f && max_val < static_cast<real_t>(maxcat)) {
            // Feature is categorical - set cat[m] to number of categories
            cat_[m] = num_unique;
        }
    }
}

void RandomForest::set_categorical_features(const integer_t* cat_array, integer_t mdim) {
    // Manually set categorical features from provided array
    cat_.resize(mdim);
    for (integer_t m = 0; m < mdim; ++m) {
        cat_[m] = cat_array[m];
    }
}

// Unified fit method that dispatches based on task type
void RandomForest::fit(const real_t* X, const void* y, const real_t* sample_weight) {
    // Set global config values that CPU/GPU code needs
    rf::g_config.use_casewise = config_.use_casewise;
    
    switch (config_.task_type) {
        case TaskType::CLASSIFICATION:
            fit_classification(X, static_cast<const integer_t*>(y), sample_weight);
            break;
        case TaskType::REGRESSION:
            fit_regression(X, static_cast<const real_t*>(y), sample_weight);
            break;
        case TaskType::UNSUPERVISED:
            fit_unsupervised(X, sample_weight);
            break;
    }
}

// Classification training
void RandomForest::fit_classification(const real_t* X, const integer_t* y, const real_t* sample_weight) {
    // Training message handled by Python wrapper
    
    // Lazy CUDA initialization (only when actually needed)
    // Initialize GPU if use_gpu is True
    // User explicitly requested GPU via use_gpu=True
    if (config_.use_gpu) {
#ifdef CUDA_FOUND
        try {
            bool cuda_success = rf::cuda::cuda_init_runtime(false);
            config_.use_gpu = cuda_success;
        } catch (const std::exception& e) {
            config_.use_gpu = false;
        } catch (...) {
            config_.use_gpu = false;
        }
#else
        // CPU-only build - throw error instead of silent fallback
        throw std::runtime_error("GPU requested (use_gpu=True) but this is a CPU-only build. "
                                 "Set use_gpu=False or install the GPU-enabled version.");
#endif
    } else {
        // CPU mode: use_gpu == false
        config_.use_gpu = false;
    }
    
    // Store training data
    // Cast to size_t to avoid integer overflow in pointer arithmetic
    size_t x_train_size = static_cast<size_t>(config_.mdim) * static_cast<size_t>(config_.nsample);
    X_train_.assign(X, X + x_train_size);
    y_train_classification_.assign(y, y + config_.nsample);

    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }

    // Setup classification-specific parameters
    setup_classification_task(y);
    
    // Prepare data (sort, create ties)
    prepare_data();
    
    // Check config values BEFORE accessing them in condition
    bool use_gpu_val = config_.use_gpu;
    integer_t ntree_val = config_.ntree;
    
    // CPU mode: Use sequential CPU for ANY number of trees when GPU is disabled
    // GPU mode: Always use GPU batch mode (auto-scaling)
    if (!use_gpu_val) {
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        // Low-rank proximity (qlora) is GPU-only - use full dense proximity in CPU mode
        if (config_.compute_proximity && config_.use_qlora) {
            config_.use_qlora = false;  // Disable qlora, but keep proximity enabled (use dense matrix)
        }
        
        // Allocate leaf_assignments_ if requested (CPU mode)
        if (config_.compute_leaf_assignments && leaf_assignments_.empty()) {
            // Cast to size_t to avoid integer overflow with large ntree * nsample
            size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
            if (leaf_size > leaf_assignments_.max_size()) {
                throw std::runtime_error(
                    "Cannot allocate leaf_assignments_: size " + std::to_string(leaf_size) + 
                    " exceeds max_size() " + std::to_string(leaf_assignments_.max_size())
                );
            }
            leaf_assignments_.resize(leaf_size, 0);
        }
        
        for (integer_t itree = 0; itree < ntree_val; ++itree) {
            integer_t seed_val = config_.iseed + itree;
            grow_tree_single(itree, seed_val);

            // Progress update for CPU training (callback)
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
        
        // Compute OOB error and predictions BEFORE finalize_training normalizes q_
        compute_classification_error();
        compute_oob_predictions();
        
        // Finalize training for CPU mode (normalizes OOB votes, computes proximity)
        finalize_training();
        
        // CPU mode done - return early to skip GPU code path
        return;
    }
    
    // GPU mode: Choose between sequential (batch_size=1) or batch (batch_size>1)
    integer_t batch_size;
    bool is_auto_scaled = false;
    
    // Determine batch size first
    if (config_.batch_size > 0) {
        // Use user-specified batch size
        batch_size = std::min(config_.batch_size, ntree_val);
    } else {
        // Auto-scale batch size using get_recommended_batch_size
        try {
            batch_size = rf::cuda::get_recommended_batch_size(ntree_val);
        } catch (...) {
            // Fallback: use default batch size if CUDA function fails
            batch_size = std::min(static_cast<integer_t>(10), ntree_val);
        }
        batch_size = std::min(batch_size, ntree_val);
        is_auto_scaled = true;
    }
    
    // Set gpu_parallel_mode0 based on batch_size
    // Parallel mode is enabled when batch_size > 1
    // Fixed: Removed expensive curand_init from kernel, now uses pre-initialized states
    // This must be set BEFORE calling fit_batch_gpu
    rf::g_config.gpu_parallel_mode0 = (batch_size > 1);
    
#ifdef CUDA_FOUND
    fit_batch_gpu(X, y, sample_weight_.data(), batch_size);
#else
    throw std::runtime_error("GPU support not available in this CPU-only build. Set use_gpu=False.");
#endif
    
    // Validate object state after GPU operations to catch corruption
    // Check that essential member variables are still valid
    if (config_.nsample <= 0 || config_.nclass <= 0) {
        throw std::runtime_error("Object state corrupted after fit_batch_gpu: invalid config");
    }
    if (nout_.size() < static_cast<size_t>(config_.nsample) || 
        q_.size() < static_cast<size_t>(config_.nsample * config_.nclass)) {
        throw std::runtime_error("Object state corrupted after fit_batch_gpu: invalid member arrays");
    }

    // Compute OOB error and predictions BEFORE normalization (using raw vote counts in q_)
    // Match repo version order: compute_classification_error, compute_oob_predictions, then finalize_training
    compute_classification_error();
    compute_oob_predictions();
    
    // Finalize training (normalizes OOB votes for probabilities, but we've already computed error/predictions)
    // NOTE: fit_batch_gpu() should NOT call finalize_training() - it is called here instead
    finalize_training();

    // After fit, finalize CUDA operations to ensure all kernels complete
    // This prevents stale operations from affecting subsequent cells and ensures clean return
    // This matches the pattern in fit_regression() and fit_unsupervised()
    // Store use_gpu in local variable to avoid accessing config_ after GPU operations
    bool use_gpu_local = config_.use_gpu;
    if (use_gpu_local) {
        rf::cuda::cuda_finalize_operations();
        // Clear any CUDA errors after finalization to ensure clean state
        #ifdef CUDA_FOUND
        cudaGetLastError();  // Clear any pending errors
        #endif
    }
    
    // Add memory barrier before return to ensure all operations complete
    // This helps prevent stack corruption during return in exec() context
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // Flush all output streams before return to ensure clean state

    // OOB error handled by Python wrapper
    // Match repo version exactly - simple return, no debug prints
}

// Regression training
void RandomForest::fit_regression(const real_t* X, const real_t* y, const real_t* sample_weight) {
    // Training message handled by Python wrapper
    // std::cout << "Training Random Forest Regressor with " << config_.ntree << " trees...\n";
    
    // Clear any stale CUDA errors before fit (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_clear_errors();
    }
    
    // Lazy CUDA initialization (only when actually needed)
    // Initialize GPU if use_gpu is True
    // User explicitly requested GPU via use_gpu=True
    if (config_.use_gpu) {
#ifdef CUDA_FOUND
        try {
            // std::cout << "DEBUG: Attempting CUDA initialization..." << std::endl;
            bool cuda_success = rf::cuda::cuda_init_runtime(false);
            config_.use_gpu = cuda_success;
            // Commented out to avoid stream conflicts with Python progress bars
            // if (config_.use_gpu) {
            //     std::cout << "CUDA: Using GPU acceleration\n";
            // } else {
            //     std::cout << "CUDA: Initialization failed, falling back to CPU\n";
            // }
        } catch (const std::exception& e) {
            // std::cerr << "CUDA: Initialization failed (" << e.what() << "), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        } catch (...) {
            // std::cerr << "CUDA: Initialization failed (unknown error), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        }
#else
        // CPU-only build - throw error instead of silent fallback
        throw std::runtime_error("GPU requested (use_gpu=True) but this is a CPU-only build. "
                                 "Set use_gpu=False or install the GPU-enabled version.");
#endif
    } else {
        // CPU mode: use_gpu == false
        config_.use_gpu = false;
    }
    
    // Before fit, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    integer_t ntree_val = config_.ntree;
    // Store training data
    // Cast to size_t to avoid integer overflow in pointer arithmetic
    size_t x_train_size = static_cast<size_t>(config_.mdim) * static_cast<size_t>(config_.nsample);
    X_train_.assign(X, X + x_train_size);
    y_train_regression_.assign(y, y + config_.nsample);

    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }

    // Setup regression-specific parameters
    setup_regression_task(y);

    // Prepare data (sort, create ties)
    prepare_data();

    // CPU mode: Use sequential CPU for ANY number of trees when GPU is disabled
    // GPU mode: Always use GPU batch mode (auto-scaling)
    if (!config_.use_gpu) {
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        // Low-rank proximity (qlora) is GPU-only - use full dense proximity in CPU mode
        if (config_.compute_proximity && config_.use_qlora) {
            config_.use_qlora = false;  // Disable qlora, but keep proximity enabled (use dense matrix)
        }
        
        // Allocate leaf_assignments_ if requested (CPU mode)
        if (config_.compute_leaf_assignments && leaf_assignments_.empty()) {
            // Cast to size_t to avoid integer overflow with large ntree * nsample
            size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
            if (leaf_size > leaf_assignments_.max_size()) {
                throw std::runtime_error(
                    "Cannot allocate leaf_assignments_: size " + std::to_string(leaf_size) + 
                    " exceeds max_size() " + std::to_string(leaf_assignments_.max_size())
                );
            }
            leaf_assignments_.resize(leaf_size, 0);
        }
        
        // CPU mode: Use OpenMP parallel for with thread-local workspaces
#ifdef _OPENMP
        if (g_config.n_threads_cpu > 1 && config_.ntree > 1) {
            // Parallel CPU mode with thread-local workspaces
            omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
            std::atomic<integer_t> trees_completed{0};
            
            #pragma omp parallel
            {
                // Create thread-local workspace copies (each thread gets its own)
                std::vector<real_t> local_win(win_workspace_.size());
                std::vector<integer_t> local_nin(nin_workspace_.size());
                std::vector<integer_t> local_jinbag(jinbag_workspace_.size());
                std::vector<integer_t> local_jtr(jtr_workspace_.size());
                std::vector<integer_t> local_nodextr(nodextr_workspace_.size());
                std::vector<integer_t> local_idmove(idmove_workspace_.size());
                std::vector<integer_t> local_igoleft(igoleft_workspace_.size());
                std::vector<integer_t> local_igl(igl_workspace_.size());
                std::vector<integer_t> local_nodestart(nodestart_workspace_.size());
                std::vector<integer_t> local_nodestop(nodestop_workspace_.size());
                std::vector<integer_t> local_itemp(itemp_workspace_.size());
                std::vector<integer_t> local_itmp(itmp_workspace_.size());
                std::vector<integer_t> local_icat(icat_workspace_.size());
                std::vector<real_t> local_tcat(tcat_workspace_.size());
                std::vector<real_t> local_classpop(classpop_workspace_.size());
                std::vector<real_t> local_wr(wr_workspace_.size());
                std::vector<real_t> local_wl(wl_workspace_.size());
                std::vector<real_t> local_tclasscat(tclasscat_workspace_.size());
                std::vector<real_t> local_tclasspop(tclasspop_workspace_.size());
                std::vector<real_t> local_tmpclass(tmpclass_workspace_.size());
                std::vector<integer_t> local_amat(amat_workspace_.size());
                
                #pragma omp for schedule(dynamic)
                for (integer_t itree = 0; itree < config_.ntree; ++itree) {
                    integer_t seed = config_.iseed + itree;
                    
                    // Get tree-specific output pointers (these are separate per tree, so safe)
                    size_t tree_offset = static_cast<size_t>(itree) * static_cast<size_t>(config_.maxnode);
                    size_t tree_treemap_offset = static_cast<size_t>(itree) * 2 * static_cast<size_t>(config_.maxnode);
                    size_t tree_catgoleft_offset = static_cast<size_t>(itree) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
                    
                    integer_t ninbag = 0;
                    real_t* tnodewt = tnodewt_.data() + tree_offset;
                    real_t* nodepred = (config_.task_type == TaskType::REGRESSION) ? 
                        (node_predictions_regression_.data() + tree_offset) : nullptr;
                    real_t* xbestsplit = xbestsplit_.data() + tree_offset;
                    integer_t* nodestatus = nodestatus_.data() + tree_offset;
                    integer_t* bestvar = bestvar_.data() + tree_offset;
                    integer_t* treemap = treemap_.data() + tree_treemap_offset;
                    integer_t* catgoleft = catgoleft_.data() + tree_catgoleft_offset;
                    integer_t* nodeclass = nodeclass_.data() + tree_offset;
                    
                    // Initialize tree arrays
                    std::fill(nodestatus, nodestatus + config_.maxnode, 0);
                    std::fill(bestvar, bestvar + config_.maxnode, 0);
                    std::fill(xbestsplit, xbestsplit + config_.maxnode, 0.0f);
                    std::fill(treemap, treemap + 2 * config_.maxnode, 0);
                    std::fill(nodeclass, nodeclass + config_.maxnode, 0);
                    std::fill(catgoleft, catgoleft + static_cast<size_t>(config_.maxcat) * config_.maxnode, 0);
                    std::fill(tnodewt, tnodewt + config_.maxnode, 0.0f);
                    
                    // Get y data
                    const integer_t* y_data;
                    if (config_.task_type == TaskType::CLASSIFICATION) {
                        y_data = y_train_classification_1based_.data();
                    } else if (config_.task_type == TaskType::REGRESSION) {
                        y_data = y_train_regression_int_.data();
                    } else {
                        y_data = y_train_unsupervised_.data();
                    }
                    
                    const real_t* X_data = (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) 
                        ? X_unsupervised_expanded_.data() : X_train_.data();
                    
                    // Grow tree with thread-local workspaces
                    cpu_growtree_wrapper(X_data, local_win.data(), y_data, cat_.data(),
                         ties_.data(), local_nin.data(), config_.mdim, config_.nsample,
                         config_.nclass, config_.maxnode, config_.minndsize, config_.mtry,
                         ninbag, config_.maxcat, seed, config_.ncsplit, config_.ncmax,
                         local_amat.data(), local_jinbag.data(), tnodewt, nodepred, xbestsplit, nodestatus,
                         bestvar, treemap, catgoleft, nodeclass, local_jtr.data(), local_nodextr.data(),
                         nnode_[itree], local_idmove.data(), local_igoleft.data(), local_igl.data(),
                         local_nodestart.data(), local_nodestop.data(), local_itemp.data(), local_itmp.data(),
                         local_icat.data(), local_tcat.data(), local_classpop.data(), local_wr.data(), local_wl.data(),
                         local_tclasscat.data(), local_tclasspop.data(), local_tmpclass.data(),
                         nullptr);  // No histogram for parallel mode
                    
                    // Test tree on OOB samples
                    cpu_testreebag(X_data, xbestsplit, local_nin.data(), treemap, bestvar,
                               nodeclass, cat_.data(), nodestatus, catgoleft,
                               config_.nsample, config_.mdim, nnode_[itree], config_.maxcat,
                               local_jtr.data(), local_nodextr.data());
                    
                    // Progress callback
                    integer_t completed = ++trees_completed;
                    if (progress_callback_ && completed % std::max(1, config_.ntree / 20) == 0) {
                        #pragma omp critical
                        {
                            progress_callback_(completed, config_.ntree);
                        }
                    }
                }
            }
            if (progress_callback_) {
                progress_callback_(config_.ntree, config_.ntree);
            }
        } else
#endif
        {
            // Sequential CPU mode (fallback or single-threaded)
            for (integer_t itree = 0; itree < config_.ntree; ++itree) {
                grow_tree_single(itree, config_.iseed + itree);
                if (progress_callback_) {
                    progress_callback_(itree + 1, config_.ntree);
                }
            }
        }
        
        // Finalize training for CPU mode (compute proximity, OOB error, etc.)
        finalize_training();
    } else {
        // GPU mode: Choose between sequential (batch_size=1) or batch (batch_size>1)
        integer_t batch_size;
        bool is_auto_scaled = false;
        
        // Determine batch size first
        if (config_.batch_size > 0) {
            // Use user-specified batch size
            batch_size = std::min(config_.batch_size, config_.ntree);
        } else {
            // Auto-scale batch size using get_recommended_batch_size
            try {
                batch_size = rf::cuda::get_recommended_batch_size(config_.ntree);
            } catch (...) {
                // Fallback: use default batch size if CUDA function fails
                batch_size = std::min(static_cast<integer_t>(10), config_.ntree);
            }
            batch_size = std::min(batch_size, config_.ntree);
            is_auto_scaled = true;
        }
        
        // Set gpu_parallel_mode0 based on batch_size
        // Parallel mode is enabled when batch_size > 1
        // This must be set BEFORE calling fit_batch_gpu
        rf::g_config.gpu_parallel_mode0 = (batch_size > 1);
        
        // Set g_config.use_casewise BEFORE calling fit_batch_gpu
        rf::g_config.use_casewise = config_.use_casewise;
        
        if (is_auto_scaled){
            // std::cout << "GPU: Auto-scaling selected batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        } else {
            // std::cout << "GPU: Using manual batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        }
        // std::cout << "[FIT_REGRESSION] About to call fit_batch_gpu with batch_size=" << batch_size 
        //           << ", use_casewise=" << config_.use_casewise << std::endl;
        
#ifdef CUDA_FOUND
        fit_batch_gpu(X, y, sample_weight_.data(), batch_size);
#else
        throw std::runtime_error("GPU support not available in this CPU-only build. Set use_gpu=False.");
#endif
        // std::cout << "[FIT_REGRESSION] fit_batch_gpu completed successfully" << std::endl;
        
        // Finalize training for GPU mode (CPU mode already finalized above)
        finalize_training();
    }

    // NOTE: finalize_training() is called above in BOTH CPU and GPU branches
    // DO NOT call it again here - compute_regression_mse() modifies oob_predictions_
    // and calling it multiple times would corrupt results!

    // After fit, finalize CUDA operations to ensure all kernels complete
    // This prevents stale operations from affecting subsequent cells
    if (config_.use_gpu) {
        rf::cuda::cuda_finalize_operations();
    }

    // OOB MSE handled by Python wrapper
    // std::cout << "Training complete. OOB MSE: " << oob_mse_ << "\n";
}

// Unsupervised training
void RandomForest::fit_unsupervised(const real_t* X, const real_t* sample_weight) {
    // Set global config values that CPU/GPU code needs
    rf::g_config.use_casewise = config_.use_casewise;
    
    // Training message handled by Python wrapper
    // std::cout << "Training Random Forest Unsupervised with " << config_.ntree << " trees...\n";
    
    // Clear any stale CUDA errors before fit (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_clear_errors();
    }
    
    // Lazy CUDA initialization (only when actually needed)
    // Initialize GPU if use_gpu is True
    // User explicitly requested GPU via use_gpu=True
    if (config_.use_gpu) {
#ifdef CUDA_FOUND
        try {
            // std::cout << "DEBUG: Attempting CUDA initialization..." << std::endl;
            bool cuda_success = rf::cuda::cuda_init_runtime(false);
            config_.use_gpu = cuda_success;
            // Commented out to avoid stream conflicts with Python progress bars
            // if (config_.use_gpu) {
            //     std::cout << "CUDA: Using GPU acceleration\n";
            // } else {
            //     std::cout << "CUDA: Initialization failed, falling back to CPU\n";
            // }
        } catch (const std::exception& e) {
            // std::cerr << "CUDA: Initialization failed (" << e.what() << "), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        } catch (...) {
            // std::cerr << "CUDA: Initialization failed (unknown error), using CPU fallback\n";  // Commented to avoid stream conflicts
            config_.use_gpu = false;
        }
#else
        // CPU-only build - throw error instead of silent fallback
        throw std::runtime_error("GPU requested (use_gpu=True) but this is a CPU-only build. "
                                 "Set use_gpu=False or install the GPU-enabled version.");
#endif
    } else {
        // CPU mode: use_gpu == false
        config_.use_gpu = false;
    }
    
    // Before fit, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    integer_t ntree_val = config_.ntree;

    // Print memory information (skip in CPU mode to reduce output)
    // Only print for GPU mode or if explicitly requested
    // if (config_.use_gpu) {
    //     print_memory_info(config_.nsample, config_.mdim, config_.ntree, config_.nclass, config_.maxnode, config_.use_gpu);
    // }

    // Store original training data
    X_train_.assign(X, X + config_.mdim * config_.nsample);

    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }

    // ========================================================================
    // UNSUPERVISED: Create synthetic data with permuted features (Breiman-Cutler)
    // ========================================================================
    // Real samples: class 1
    // Synthetic samples: class 0 (created by permuting each feature independently)
    // This destroys the correlation structure, allowing trees to distinguish real from random
    // ========================================================================
    
    integer_t n_real = config_.nsample;
    n_synthetic_ = static_cast<integer_t>(n_real * config_.synthetic_ratio);
    if (n_synthetic_ < 1) n_synthetic_ = 1;  // At least 1 synthetic sample
    n_total_unsupervised_ = n_real + n_synthetic_;
    
    // Expand feature matrix: [real_samples; synthetic_samples]
    X_unsupervised_expanded_.resize(n_total_unsupervised_ * config_.mdim);
    
    // Copy real data (first n_real samples)
    std::copy(X, X + n_real * config_.mdim, X_unsupervised_expanded_.begin());
    
    // Create synthetic data by permuting each feature independently
    // This destroys feature correlations - key to Breiman-Cutler unsupervised RF
    // OPTIMIZED: Parallelize across features (each feature is independent)
    
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (integer_t k = 0; k < config_.mdim; ++k) {
        // Thread-local RNG seeded deterministically per feature for reproducibility
        std::mt19937 rng(config_.iseed + k * 1000003);  // Different seed per feature
        
        // Extract this feature's values from real samples
        std::vector<real_t> feature_values(n_real);
        for (integer_t i = 0; i < n_real; ++i) {
            feature_values[i] = X[i * config_.mdim + k];  // Row-major: sample i, feature k
        }
        
        // Shuffle the feature values
        std::shuffle(feature_values.begin(), feature_values.end(), rng);
        
        // Sample n_synthetic values (with or without replacement based on ratio)
        for (integer_t i = 0; i < n_synthetic_; ++i) {
            integer_t sample_idx = (config_.synthetic_ratio <= 1.0f) 
                ? i % n_real  // Without replacement for ratio <= 1
                : rng() % n_real;  // With replacement for oversampling
            X_unsupervised_expanded_[(n_real + i) * config_.mdim + k] = feature_values[sample_idx];
        }
    }
    
    // Create labels: real=1, synthetic=0
    y_train_unsupervised_.resize(n_total_unsupervised_);
    std::fill(y_train_unsupervised_.begin(), y_train_unsupervised_.begin() + n_real, 1);  // Real = class 1
    std::fill(y_train_unsupervised_.begin() + n_real, y_train_unsupervised_.end(), 0);   // Synthetic = class 0
    
    // Expand sample weights
    std::vector<real_t> expanded_weights(n_total_unsupervised_, 1.0f);
    for (integer_t i = 0; i < n_real; ++i) {
        expanded_weights[i] = sample_weight_[i];
    }
    // Synthetic samples get weight 1.0 (already initialized)
    sample_weight_ = std::move(expanded_weights);
    
    // Update config to use expanded data size
    config_.nsample = n_total_unsupervised_;

    // Update maxnode for expanded sample count (trees need more nodes)
    config_.maxnode = calculate_maxnode(config_.nsample, config_.minndsize);

    // Setup unsupervised-specific parameters (MUST be called before initialize_arrays)
    // This sets nclass=2 for binary classification (real vs synthetic)
    setup_unsupervised_task();
    
    // Initialize ALL arrays with correct nsample (now includes synthetic samples)
    // This allocates nout_, ties_, asave_, proximity_matrix_, importance arrays, etc.
    initialize_arrays();

    // Prepare data (sort, create ties)
    prepare_data();

    // CPU mode: Use sequential CPU for ANY number of trees when GPU is disabled
    // GPU mode: Always use GPU batch mode (auto-scaling)
    if (!config_.use_gpu) {
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        // Low-rank proximity (qlora) is GPU-only - use full dense proximity in CPU mode
        if (config_.compute_proximity && config_.use_qlora) {
            config_.use_qlora = false;  // Disable qlora, but keep proximity enabled (use dense matrix)
        }
        
        // Allocate leaf_assignments_ if requested (CPU mode)
        if (config_.compute_leaf_assignments && leaf_assignments_.empty()) {
            // Cast to size_t to avoid integer overflow with large ntree * nsample
            size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
            if (leaf_size > leaf_assignments_.max_size()) {
                throw std::runtime_error(
                    "Cannot allocate leaf_assignments_: size " + std::to_string(leaf_size) + 
                    " exceeds max_size() " + std::to_string(leaf_assignments_.max_size())
                );
            }
            leaf_assignments_.resize(leaf_size, 0);
        }
        
        // CPU sequential mode: Use for ANY number of trees when GPU is disabled
        for (integer_t itree = 0; itree < config_.ntree; ++itree) {
            grow_tree_single(itree, config_.iseed + itree);
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
        
        // Compute OOB error for unsupervised mode (classifying real vs synthetic)
        // This must be called BEFORE finalize_training() which normalizes q_
        compute_classification_error();
        
        // Finalize training for CPU mode (this also computes outlier scores for unsupervised)
        finalize_training();

        // Compute cluster labels after proximity matrix is finalized (CPU only)
        if (config_.compute_proximity && !proximity_matrix_.empty() && proximity_normalized_) {
            compute_cluster_labels();
        }
    } else {
        // GPU mode: Choose between sequential (grow_gpu_parallel=False) or batch (grow_gpu_parallel=True)
        integer_t batch_size;
        bool is_auto_scaled = false;
        
        // Determine batch size first
        if (config_.batch_size > 0) {
            // Use user-specified batch size
            batch_size = std::min(config_.batch_size, config_.ntree);
        } else {
            // Auto-scale batch size using get_recommended_batch_size
            try {
                batch_size = rf::cuda::get_recommended_batch_size(config_.ntree);
            } catch (...) {
                // Fallback: use default batch size if CUDA function fails
                batch_size = std::min(static_cast<integer_t>(10), config_.ntree);
            }
            batch_size = std::min(batch_size, config_.ntree);
            is_auto_scaled = true;
        }
        
        // Set gpu_parallel_mode0 based on batch_size
        // Parallel mode is enabled when batch_size > 1
        // This must be set BEFORE calling fit_batch_gpu
        rf::g_config.gpu_parallel_mode0 = (batch_size > 1);
        
        // Set g_config.use_casewise BEFORE calling fit_batch_gpu
        rf::g_config.use_casewise = config_.use_casewise;
        
        if (is_auto_scaled){
            // std::cout << "GPU: Auto-scaling selected batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        } else {
            // std::cout << "GPU: Using manual batch size = " << batch_size << " trees (out of " << config_.ntree << " total)" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        }
        // std::cout << "[FIT_UNSUPERVISED] About to call fit_batch_gpu with batch_size=" << batch_size 
        //           << ", use_casewise=" << config_.use_casewise << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        
        // For unsupervised GPU, use EXPANDED data (real + synthetic)
        // This matches CPU behavior where grow_tree_single uses X_unsupervised_expanded_
        // CRITICAL: Update config_.nsample to expanded size BEFORE fit_batch_gpu
        integer_t original_nsample = config_.nsample;  // Save original for restore
        config_.nsample = n_total_unsupervised_;  // Use expanded sample count
        
        // Expand sample weights for synthetic samples (weight = 1.0)
        std::vector<real_t> expanded_weights(n_total_unsupervised_, 1.0f);
        for (integer_t i = 0; i < original_nsample; ++i) {
            expanded_weights[i] = sample_weight_[i];
        }
        
#ifdef CUDA_FOUND
        fit_batch_gpu(X_unsupervised_expanded_.data(), y_train_unsupervised_.data(), 
                      expanded_weights.data(), batch_size);
#else
        throw std::runtime_error("GPU support not available in this CPU-only build. Set use_gpu=False.");
#endif
        
        // Compute OOB error BEFORE restoring nsample (uses expanded sample count)
        // The q_ and nout_ arrays contain votes for all samples (real + synthetic)
        compute_classification_error();
        
        // Finalize training (normalizes OOB votes for probabilities)
        finalize_training();
        
        // Restore original nsample for user queries (they expect original sample count)
        config_.nsample = original_nsample;
        // std::cout << "[FIT_UNSUPERVISED] fit_batch_gpu completed successfully" << std::endl;  // Commented to avoid stream conflicts with Python progress bars
    }

    // After fit, finalize CUDA operations to ensure all kernels complete
    // This prevents stale operations from affecting subsequent cells
    if (config_.use_gpu) {
        rf::cuda::cuda_finalize_operations();
    }

    // Completion message handled by Python wrapper
    // std::cout << "Training complete. Proximity matrix computed.\n";
}

void RandomForest::prepare_data() {
    // Prepare data: sort and create ties array
    std::vector<integer_t> isort(config_.nsample);
    std::vector<real_t> v(config_.nsample);
    
    // For unsupervised, use expanded data (real + synthetic samples)
    const real_t* X_data = (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) 
        ? X_unsupervised_expanded_.data() 
        : X_train_.data();
    
    prepdata(X_data, config_.mdim, config_.nsample,
             cat_.data(), isort.data(), v.data(), asave_.data(), ties_.data());
    
    // Initialize histogram binning if enabled
    // Always use histogram for correct handling of binary/low-cardinality features
    if (config_.use_histogram) {
        histogram_data_ = std::make_unique<HistogramData>();
        histogram_data_->allocate(config_.nsample, config_.mdim);
        
        // GPU kernel supports up to 256 bins with shared memory
        // Use full 256 bins for best accuracy (XGBoost-like)
        integer_t n_bins_actual = std::min(config_.n_bins, static_cast<integer_t>(256));
        
        // Compute bin edges using quantiles
        compute_bin_edges_quantile(
            X_data,
            config_.nsample,
            config_.mdim,
            cat_.data(),
            n_bins_actual,
            histogram_data_->feature_info.data()
        );
        
        // Convert data to binned representation
        bin_data_cpu(
            X_data,
            config_.nsample,
            config_.mdim,
            histogram_data_->feature_info.data(),
            histogram_data_->X_binned.data()
        );
    }
}

void RandomForest::grow_tree_single(integer_t tree_id, integer_t seed) {
    // Use pre-allocated workspace arrays to avoid repeated allocation/deallocation
    integer_t ninbag, noobag;

    if (sample_weight_.empty()) {
        // printf("ERROR: sample_weight_ is empty!\n");  // Commented to avoid stream conflicts
        // fflush(stdout);
        return;
    }
    
    // printf("DEBUG: config_.nsample=%d\n", static_cast<int>(config_.nsample));
    // printf("DEBUG: tree_id=%d\n", static_cast<int>(tree_id));
    // fflush(stdout);
    
    // Bootstrap sampling
    // printf("DEBUG: Actually calling cpu_bootstrap now...\n");
    // fflush(stdout);
    cpu_bootstrap(sample_weight_.data(), config_.nsample, win_workspace_.data(), nin_workspace_.data(),
              nout_.data(), jinbag_workspace_.data(), joobag_workspace_.data(), ninbag, noobag, tree_id);
    // printf("DEBUG: cpu_bootstrap returned successfully\n");
    // fflush(stdout);
    
    // printf("DEBUG: About to call cpu_getamat...\n");
    // fflush(stdout);
    
    // Create sorted index matrix for this bootstrap sample
    // For sparse mode, skip cpu_getamat since sparse tree growing handles sorting internally
    if (!is_sparse_) {
        cpu_getamat(asave_.data(), nin_workspace_.data(), cat_.data(), config_.nsample,
                config_.mdim, amat_workspace_.data());
    }
    
    // printf("DEBUG: cpu_getamat returned successfully\n");
    // fflush(stdout);

    // Clear workspace arrays for this tree
    // printf("DEBUG: About to clear workspace arrays\n");
    // fflush(stdout);
    std::fill(jtr_workspace_.begin(), jtr_workspace_.end(), 0);
    std::fill(nodextr_workspace_.begin(), nodextr_workspace_.end(), 0);
    // printf("DEBUG: Workspace arrays cleared\n");
    // Get pointers to this tree's storage
    // Cast to size_t to avoid integer overflow in pointer arithmetic
    size_t tree_offset = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
    size_t tree_treemap_offset = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
    size_t tree_catgoleft_offset = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
    
    real_t* tnodewt = tnodewt_.data() + tree_offset;
    real_t* nodepred = (config_.task_type == TaskType::REGRESSION) ? 
        (node_predictions_regression_.data() + tree_offset) : nullptr;
    real_t* xbestsplit = xbestsplit_.data() + tree_offset;
    integer_t* nodestatus = nodestatus_.data() + tree_offset;
    integer_t* bestvar = bestvar_.data() + tree_offset;
    integer_t* treemap = treemap_.data() + tree_treemap_offset;
    integer_t* catgoleft = catgoleft_.data() + tree_catgoleft_offset;
    integer_t* nodeclass = nodeclass_.data() + tree_offset;
    
    // Initialize tree arrays to zero (matching Fortran initialization)
    std::fill(nodestatus, nodestatus + config_.maxnode, 0);
    std::fill(bestvar, bestvar + config_.maxnode, 0);
    std::fill(xbestsplit, xbestsplit + config_.maxnode, 0.0f);
    // Cast to size_t to avoid integer overflow in pointer arithmetic for std::fill
    size_t treemap_size = 2 * static_cast<size_t>(config_.maxnode);
    size_t catgoleft_size = static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
    std::fill(treemap, treemap + treemap_size, 0);
    std::fill(nodeclass, nodeclass + config_.maxnode, 0);
    std::fill(catgoleft, catgoleft + catgoleft_size, 0);
    std::fill(tnodewt, tnodewt + config_.maxnode, 0.0f);  // Initialize tnodewt for this tree

    // Grow tree
    const integer_t* y_data;
    if (config_.task_type == TaskType::CLASSIFICATION) {
        // Use 0-based class labels directly (matching reference setup_classification_task)
        // Note: cpu_growtree expects 0-based labels (checks class_idx >= 0 && class_idx < nclass)
        y_data = y_train_classification_1based_.data();
    } else if (config_.task_type == TaskType::REGRESSION) {
        // For regression, we need to convert real_t to integer_t
        // Scale y values to preserve precision (small values like 0.1 become 100)
        // This prevents all values from becoming 0 when cast to integers
        const real_t y_scale = 1000.0f;
        if (y_train_regression_int_.empty()) {
            y_train_regression_int_.resize(y_train_regression_.size());
            for (size_t i = 0; i < y_train_regression_.size(); ++i) {
                y_train_regression_int_[i] = static_cast<integer_t>(y_train_regression_[i] * y_scale);
            }
        }
        y_data = y_train_regression_int_.data();
    } else {
        // For unsupervised, use synthetic labels (real=1, synthetic=0)
        // These are set up in fit_unsupervised with proper Breiman-Cutler labels
        y_data = y_train_unsupervised_.data();
    }
    
    // Use the standard tree growing function for all task types
    // Check if we're in sparse mode and use the appropriate function
    if (is_sparse_ && X_train_sparse_.is_valid()) {
        // NATIVE SPARSE TREE GROWING - O(nnz) memory, no dense conversion
        cpu_growtree_sparse(X_train_sparse_, win_workspace_.data(), y_data, cat_.data(),
             ties_.data(), nin_workspace_.data(), config_.mdim, config_.nsample,
             config_.nclass, config_.maxnode, config_.minndsize, config_.mtry,
             ninbag, config_.maxcat, seed, config_.ncsplit, config_.ncmax,
             amat_workspace_.data(), jinbag_workspace_.data(), tnodewt, nodepred, xbestsplit, nodestatus,
             bestvar, treemap, catgoleft, nodeclass, jtr_workspace_.data(), nodextr_workspace_.data(),
             nnode_[tree_id], idmove_workspace_.data(), igoleft_workspace_.data(), igl_workspace_.data(),
             nodestart_workspace_.data(), nodestop_workspace_.data(), itemp_workspace_.data(), itmp_workspace_.data(),
             icat_workspace_.data(), tcat_workspace_.data(), classpop_workspace_.data(), wr_workspace_.data(), wl_workspace_.data(),
             tclasscat_workspace_.data(), tclasspop_workspace_.data(), tmpclass_workspace_.data());
    } else {
        // Dense tree growing (original path)
        // For unsupervised, use expanded data (real + synthetic samples)
        const real_t* X_data = (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) 
            ? X_unsupervised_expanded_.data() 
            : X_train_.data();
        
        cpu_growtree_wrapper(X_data, win_workspace_.data(), y_data, cat_.data(),
             ties_.data(), nin_workspace_.data(), config_.mdim, config_.nsample,
             config_.nclass, config_.maxnode, config_.minndsize, config_.mtry,
             ninbag, config_.maxcat, seed, config_.ncsplit, config_.ncmax,
             amat_workspace_.data(), jinbag_workspace_.data(), tnodewt, nodepred, xbestsplit, nodestatus,
             bestvar, treemap, catgoleft, nodeclass, jtr_workspace_.data(), nodextr_workspace_.data(),
             nnode_[tree_id], idmove_workspace_.data(), igoleft_workspace_.data(), igl_workspace_.data(),
             nodestart_workspace_.data(), nodestop_workspace_.data(), itemp_workspace_.data(), itmp_workspace_.data(),
             icat_workspace_.data(), tcat_workspace_.data(), classpop_workspace_.data(), wr_workspace_.data(), wl_workspace_.data(),
             tclasscat_workspace_.data(), tclasspop_workspace_.data(), tmpclass_workspace_.data(),
             histogram_data_.get());  // Pass histogram data for histogram-based split finding
    }
    
    // Call testreebag to get OOB predictions and terminal nodes for this tree
    if (is_sparse_ && X_train_sparse_.is_valid()) {
        // NATIVE SPARSE TESTREEBAG
        cpu_testreebag_sparse(X_train_sparse_, xbestsplit, nin_workspace_.data(), treemap, bestvar,
                   nodeclass, cat_.data(), nodestatus, catgoleft,
                   config_.nsample, config_.mdim, nnode_[tree_id], config_.maxcat,
                   jtr_workspace_.data(), nodextr_workspace_.data());
    } else {
        // For unsupervised, use expanded data (real + synthetic samples)
        const real_t* X_data = (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) 
            ? X_unsupervised_expanded_.data() 
            : X_train_.data();
        
        cpu_testreebag(X_data, xbestsplit, nin_workspace_.data(), treemap, bestvar,
                   nodeclass, cat_.data(), nodestatus, catgoleft,
                   config_.nsample, config_.mdim, nnode_[tree_id], config_.maxcat,
                   jtr_workspace_.data(), nodextr_workspace_.data());
    }
    
    // Compute tnodewt for classification casewise mode
    // For classification with casewise=True, tnodewt[node] = mean bootstrap weight of in-bag samples in that node
    // For classification with casewise=False, we don't use tnodewt (always weight=1.0)
    // For regression, tnodewt is already computed during tree growing (mean y value in node)
    if ((config_.task_type == TaskType::CLASSIFICATION || config_.task_type == TaskType::UNSUPERVISED) && config_.use_casewise) {
        // Initialize tnodewt to 0
        std::fill(tnodewt, tnodewt + config_.maxnode, 0.0f);
        
        // Count samples and sum weights for each terminal node
        std::vector<real_t> node_weight_sum(config_.maxnode, 0.0f);
        std::vector<integer_t> node_sample_count(config_.maxnode, 0);
        
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] > 0) {  // In-bag sample
                // Traverse tree from root to find terminal node
                integer_t current_node = 0;
                while (nodestatus[current_node] == 1) {  // While not terminal
                    integer_t split_var = bestvar[current_node];
                    bool goes_left = false;
                    
                    // Check if categorical
                    integer_t split_numcat = (cat_.empty() || split_var >= config_.mdim) ? 1 : cat_[split_var];
                    
                    // Get sample value (sparse or dense access)
                    // For unsupervised, use expanded data (real + synthetic samples)
                    real_t sample_value;
                    if (is_sparse_ && X_train_sparse_.is_valid()) {
                        sample_value = X_train_sparse_.get(n, split_var);
                    } else if (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) {
                        sample_value = X_unsupervised_expanded_[n * config_.mdim + split_var];
                    } else {
                        sample_value = X_train_[n * config_.mdim + split_var];
                    }
                    
                    // NOTE: Binary features (numcat <= 2) use quantitative path (threshold split)
                    // Only 3+ category features use categorical split
                    if (split_numcat > 2) {  // Categorical (3+ categories)
                        integer_t category_idx = static_cast<integer_t>(sample_value + 0.5f);
                        integer_t catgoleft_offset = current_node * config_.maxcat;
                        goes_left = (category_idx >= 0 && category_idx < config_.maxcat && 
                                   catgoleft[catgoleft_offset + category_idx] == 1);
                    } else {  // Quantitative (including binary features)
                        real_t split_point = xbestsplit[current_node];
                        goes_left = (sample_value <= split_point);
                    }
                    
                    current_node = treemap[current_node * 2 + (goes_left ? 0 : 1)];
                    if (current_node < 0 || current_node >= nnode_[tree_id]) break;
                }
                
                // Add to terminal node statistics
                if (current_node >= 0 && current_node < nnode_[tree_id] && nodestatus[current_node] == -1) {
                    node_weight_sum[current_node] += static_cast<real_t>(nin_workspace_[n]);
                    node_sample_count[current_node]++;
                }
            }
        }
        
        // Compute mean weight for each terminal node
        for (integer_t node = 0; node < nnode_[tree_id]; ++node) {
            if (node_sample_count[node] > 0) {
                tnodewt[node] = node_weight_sum[node] / static_cast<real_t>(node_sample_count[node]);
            }
        }
    }
    
    // For regression, jtr_workspace_ contains class predictions (not meaningful for regression)
    // tnodewt[nodextr_workspace_[n]] is used for regression predictions instead
    // nodextr_workspace_ is still needed to know which terminal node each OOB sample reached

    // Accumulate OOB votes/predictions
    for (integer_t n = 0; n < config_.nsample; ++n) {
        if (nin_workspace_[n] == 0) {  // Sample is out-of-bag
            // NOTE: nout_[n]++ is already done in cpu_bootstrap() - DO NOT increment again here!
            
            if (config_.task_type == TaskType::CLASSIFICATION || config_.task_type == TaskType::UNSUPERVISED) {
                // For classification/unsupervised: accumulate class votes in q_ array
            if (jtr_workspace_[n] >= 0 && jtr_workspace_[n] < config_.nclass) {
                // Use 0-based indexing directly
                // q_ array is indexed as [sample * nclass + class]
                // Casewise vs non-casewise weighting for OOB votes
                // Non-casewise: weight = 1.0 (UC Berkeley standard)
                // Casewise: weight = tnodewt[terminal_node] (bootstrap frequency weighted)
                real_t vote_weight = 1.0f;
                if (config_.use_casewise) {
                    integer_t terminal_node = nodextr_workspace_[n];
                    if (terminal_node >= 0 && terminal_node < nnode_[tree_id]) {
                        vote_weight = tnodewt[terminal_node];
                    }
                }
                // Cast to size_t to avoid integer overflow in index calculation
                size_t class_idx = static_cast<size_t>(n) * static_cast<size_t>(config_.nclass) + static_cast<size_t>(jtr_workspace_[n]);
                if (class_idx < q_.size()) {
                    q_[class_idx] += vote_weight;
                    }
                }
            } else if (config_.task_type == TaskType::REGRESSION) {
                // For regression: accumulate regression predictions
                // oob_predictions_ stores the sum of predictions across all trees
                // Final prediction = oob_predictions_[n] / nout_[n] (average across trees)
                integer_t terminal_node = nodextr_workspace_[n];
                if (terminal_node >= 0 && terminal_node < nnode_[tree_id]) {
                    // nodepred stores regression predictions (mean of y in terminal node)
                    real_t regression_prediction = nodepred[terminal_node];
                    // Always use weight=1.0 for regression OOB predictions
                    // (we divide by tree count, not weight sum - matches sklearn)
                    if (n < oob_predictions_.size()) {
                        oob_predictions_[n] += regression_prediction;
                    }
                }
            }
            
            // Debug: Print OOB vote accumulation for first few samples
            // if (tree_id < 3 && n < 10) {
            //     std::cout << "CPU Tree " << tree_id << " OOB sample " << n 
            //              << " predicted class " << jtr_workspace_[n] << std::endl;
            // }
        }
    }

    // ========================================================================
    // LEAF ASSIGNMENTS STORAGE (CPU)
    // ========================================================================
    // Store terminal node assignments for all samples (for on-demand proximity computation)
    // leaf_assignments_ is stored as [tree * nsample + sample]
    // NOTE: cpu_testreebag only computes terminal nodes for OOB samples
    // For in-bag samples, we need to compute terminal nodes by traversing the tree
    // ========================================================================
    if (config_.compute_leaf_assignments && !leaf_assignments_.empty()) {
        // Cast to size_t to avoid integer overflow in offset calculation
        size_t offset = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.nsample);
        
        // Get data pointer for tree traversal
        const real_t* X_ptr_leaf = nullptr;
        if (is_sparse_ && X_train_sparse_.is_valid()) {
            // Will use sparse access
        } else if (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) {
            X_ptr_leaf = X_unsupervised_expanded_.data();
        } else {
            X_ptr_leaf = X_train_.data();
        }
        
        // Get tree structure pointers (non-const for find_terminal_node)
        // Cast to size_t to avoid integer overflow in pointer arithmetic
        size_t tree_offset_leaf = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
        size_t tree_treemap_offset_leaf = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
        size_t tree_catgoleft_offset_leaf = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
        
        integer_t* nodestatus_ptr_leaf = &nodestatus_[tree_offset_leaf];
        integer_t* bestvar_ptr_leaf = &bestvar_[tree_offset_leaf];
        real_t* xbestsplit_ptr_leaf = &xbestsplit_[tree_offset_leaf];
        integer_t* treemap_ptr_leaf = &treemap_[tree_treemap_offset_leaf];
        integer_t* catgoleft_ptr_leaf = &catgoleft_[tree_catgoleft_offset_leaf];
        
        // Initialize oob_trees_ on first tree
        if (tree_id == 0 && oob_trees_.empty()) {
            oob_trees_.resize(config_.nsample);
        }
        
        for (integer_t n = 0; n < config_.nsample; ++n) {
            // Breiman's approach: store leaf assignments for ALL samples
            // OOB status tracked separately in oob_trees_[sample]
            integer_t terminal_node;
            
            if (nin_workspace_[n] == 0) {
                // OOB sample - terminal node already computed by cpu_testreebag
                terminal_node = nodextr_workspace_[n];
                // Track OOB trees for this sample (for fast query lookup)
                oob_trees_[n].push_back(tree_id);
            } else {
                // In-bag sample - compute terminal node by traversing tree
                if (is_sparse_ && X_train_sparse_.is_valid()) {
                    terminal_node = find_terminal_node_sparse(
                        n, nodestatus_ptr_leaf, bestvar_ptr_leaf,
                        xbestsplit_ptr_leaf, treemap_ptr_leaf, catgoleft_ptr_leaf);
                } else {
                    terminal_node = find_terminal_node(
                        const_cast<real_t*>(X_ptr_leaf) + n * config_.mdim, nodestatus_ptr_leaf, bestvar_ptr_leaf,
                        xbestsplit_ptr_leaf, treemap_ptr_leaf, catgoleft_ptr_leaf);
                }
            }
            // Store leaf for ALL samples (both OOB and in-bag)
            // Cast n to size_t to avoid integer overflow in index calculation
            size_t leaf_idx = offset + static_cast<size_t>(n);
            leaf_assignments_[leaf_idx] = static_cast<int16_t>(terminal_node);
        }
    }

    // ========================================================================
    // PROXIMITY IMPORTANCE COMPUTATION (CPU only for now)
    // ========================================================================
    // For each OOB sample WHERE PREDICTION IS CORRECT, for each feature, 
    // measure how often permuting that feature changes the terminal node.
    // This tells us which features are driving the proximity structure.
    // 
    // Matches Breiman-Cutler methodology for local variable importance:
    // - Only accumulate for samples where OOB prediction was CORRECT
    // - Normalize per-tree by (1.0 / noob) - same as local importance
    // 
    // Output: proximity_importance_[n * mdim + k] = sum of (1/noob_t) across
    //         trees where permuting feature k changed terminal node for sample n
    // ========================================================================
    if (config_.compute_proximity_importance && !proximity_importance_.empty()) {
        // Get pointers to tree structure
        // Cast to size_t to avoid integer overflow in pointer arithmetic
        size_t tree_offset_prox = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
        size_t tree_treemap_offset_prox = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
        size_t tree_catgoleft_offset_prox = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
        
        const integer_t* nodestatus_ptr = &nodestatus_[tree_offset_prox];
        const integer_t* bestvar_ptr = &bestvar_[tree_offset_prox];
        const real_t* xbestsplit_ptr = &xbestsplit_[tree_offset_prox];
        const integer_t* treemap_ptr = &treemap_[tree_treemap_offset_prox];
        const integer_t* catgoleft_ptr = &catgoleft_[tree_catgoleft_offset_prox];
        const integer_t* nodeclass_ptr = &nodeclass_[tree_offset_prox];
        integer_t actual_nnode = nnode_[tree_id];
        
        // Count OOB samples for this tree (for per-tree normalization)
        integer_t noob = 0;
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] == 0) noob++;
        }
        
        if (noob <= 0) {
            // No OOB samples for this tree, skip proximity importance
            // (can happen with very high bootstrap sampling rate)
        } else {
            real_t inv_noob = 1.0f / static_cast<real_t>(noob);
            
            // Determine if this is regression (skip correctness check)
            bool is_regression = (config_.task_type == TaskType::REGRESSION);
            
            // Get the correct label array based on task type
            const integer_t* true_labels = nullptr;
            if (config_.task_type == TaskType::UNSUPERVISED) {
                true_labels = y_train_unsupervised_.data();
            } else if (config_.task_type == TaskType::CLASSIFICATION) {
                true_labels = y_train_classification_.data();
            }
        
        // OPTIMIZED: Parallelize across OOB samples with pre-cached feature values
        // Get pointer to feature data based on mode (computed once outside loop)
        const real_t* X_ptr = nullptr;
        bool use_sparse_access = is_sparse_ && X_train_sparse_.is_valid();
        if (!use_sparse_access) {
            if (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) {
                X_ptr = X_unsupervised_expanded_.data();
            } else {
                X_ptr = X_train_.data();
            }
        }
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 32)
#endif
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] != 0) continue;  // Skip in-bag samples
            
            integer_t original_node = nodextr_workspace_[n];
            if (original_node < 0 || original_node >= actual_nnode) continue;
                
                // Only accumulate for samples with CORRECT predictions
                // (matches Breiman-Cutler methodology for local variable importance)
                if (!is_regression && true_labels != nullptr) {
                    // For classification/unsupervised: jtr = nodeclass[nodextr]
                    integer_t jtr = nodeclass_ptr[original_node];
                    if (jtr != true_labels[n]) continue;  // Skip incorrect predictions
                }
            
            // Pre-cache sample features for fast inner loop access
            const real_t* sample_features = nullptr;
            std::vector<real_t> sparse_cache;
            if (use_sparse_access) {
                sparse_cache.resize(config_.mdim);
                for (integer_t k = 0; k < config_.mdim; ++k) {
                    sparse_cache[k] = X_train_sparse_.get(n, k);
                }
                sample_features = sparse_cache.data();
            } else {
                sample_features = X_ptr + n * config_.mdim;
            }
            
            // For each feature, test if permuting it changes the terminal node
            for (integer_t k = 0; k < config_.mdim; ++k) {
                // Get donor sample for permutation (deterministic hash for reproducibility)
                integer_t donor = (n + k * 31 + tree_id * 17) % config_.nsample;
                if (donor == n) donor = (donor + 1) % config_.nsample;
                
                // Get permuted feature value
                real_t permuted_value;
                if (use_sparse_access) {
                    permuted_value = X_train_sparse_.get(donor, k);
                } else {
                    permuted_value = X_ptr[donor * config_.mdim + k];
                }
                
                // Traverse tree with permuted value for feature k
                integer_t current_node = 0;
                while (nodestatus_ptr[current_node] == 1 && current_node >= 0 && current_node < actual_nnode) {
                    integer_t split_var = bestvar_ptr[current_node];
                    if (split_var < 0 || split_var >= config_.mdim) break;
                    
                    // Use cached feature value (or permuted if this is feature k)
                    real_t feature_value = (split_var == k) ? permuted_value : sample_features[split_var];
                    
                    // Fast path for quantitative variables (most common)
                    // Binary features (cat <= 2) use quantitative path
                    integer_t split_numcat = (cat_.empty() || split_var >= static_cast<integer_t>(cat_.size())) ? 1 : cat_[split_var];
                    bool goes_left = (split_numcat <= 2) 
                        ? (feature_value <= xbestsplit_ptr[current_node])
                        : (static_cast<integer_t>(feature_value + 0.5f) >= 0 && 
                           static_cast<integer_t>(feature_value + 0.5f) < config_.maxcat &&
                           catgoleft_ptr[current_node * config_.maxcat + static_cast<integer_t>(feature_value + 0.5f)] == 1);
                    
                    current_node = treemap_ptr[current_node * 2 + (goes_left ? 0 : 1)];
                }
                
                // If terminal node changed, increment proximity importance
                    // Use per-tree normalization (1/noob) - same as local variable importance
                if (current_node != original_node) {
                    // Cast to size_t to avoid integer overflow in index calculation
                    size_t prox_imp_idx = static_cast<size_t>(n) * static_cast<size_t>(config_.mdim) + static_cast<size_t>(k);
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                        proximity_importance_[prox_imp_idx] += inv_noob;
                    }
                }
            }
        }
    }
    
    // Variable importance computation if requested
    if (config_.compute_importance) {
        // Set global config for importance computation (must be set before cpu_varimp)
        rf::g_config.use_casewise = config_.use_casewise;
        
        // printf("DEBUG: Starting variable importance computation for CPU\n");
        // printf("DEBUG: compute_importance flag is TRUE in grow_tree_single\n");
        // fflush(stdout);
        std::vector<real_t> qimp_temp(config_.nsample);
        // Cast to size_t to avoid integer overflow
        size_t qimpm_temp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        std::vector<real_t> qimpm_temp(qimpm_temp_size);
        std::vector<real_t> avimp_temp(config_.mdim);
        std::vector<real_t> sqsd_temp(config_.mdim);
        
        // Initialize temporary arrays
        std::fill(qimp_temp.begin(), qimp_temp.end(), 0.0f);
        std::fill(qimpm_temp.begin(), qimpm_temp.end(), 0.0f);
        std::fill(avimp_temp.begin(), avimp_temp.end(), 0.0f);
        std::fill(sqsd_temp.begin(), sqsd_temp.end(), 0.0f);
        
        // Compute variable importance for this tree
        // Create temporary arrays for non-const parameters
        std::vector<integer_t> temp_jvr(config_.nsample, 0);  // jvr(nsample) in Fortran - FIXED!
        std::vector<integer_t> temp_nodexvr(config_.nsample, 0);  // nodexvr(nsample) in Fortran - FIXED!
        std::vector<integer_t> temp_joob(config_.nsample, 0);  // joob(nsample) in Fortran
        std::vector<integer_t> temp_pjoob(config_.nsample, 0);  // pjoob(nsample) in Fortran
        std::vector<integer_t> temp_iv(config_.mdim, 0);  // iv(mdim) in Fortran
        
        // Set impn based on compute_local_importance flag
        // impn = 1 means compute local importance, impn = 0 means skip local importance
        integer_t impn = config_.compute_local_importance ? 1 : 0;
        
        // Dispatch to appropriate importance calculation based on task type
        if (config_.task_type == TaskType::REGRESSION) {
            // Use regression-specific importance (MSE-based, Breiman 2001)
            integer_t actual_nnode = nnode_[tree_id];
            if (actual_nnode == 0) {
                actual_nnode = 1;  // Avoid division by zero
            }
            
            // Build OOB predictions from terminal node values
            // y_pred[n] = nodepred[nodextr_workspace_[n]] for OOB samples
            // Cast to size_t to avoid integer overflow in pointer arithmetic
            size_t tree_offset_nodepred = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
            real_t* nodepred = node_predictions_regression_.data() + tree_offset_nodepred;
            std::vector<real_t> y_pred_temp(config_.nsample, 0.0f);
            for (integer_t n = 0; n < config_.nsample; ++n) {
                if (nin_workspace_[n] == 0) {  // OOB sample
                    integer_t node_idx = nodextr_workspace_[n];
                    if (node_idx >= 0 && node_idx < actual_nnode) {
                        y_pred_temp[n] = nodepred[node_idx];
                    }
                }
            }
            
            // Call MSE-based regression importance (Breiman 2001)
            if (is_sparse_ && X_train_sparse_.is_valid()) {
                // SPARSE VERSION
                cpu_varimp_regression_sparse(X_train_sparse_, config_.nsample, config_.mdim,
                    y_train_regression_.data(), nin_workspace_.data(), y_pred_temp.data(), impn,
                    qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                    treemap, nodestatus, xbestsplit, bestvar,
                    nodepred, tnodewt, actual_nnode,
                    reinterpret_cast<const integer_t*>(cat_.data()),
                    config_.maxcat, catgoleft, nodextr_workspace_.data());
            } else {
                cpu_varimp_regression(X_train_.data(), config_.nsample, config_.mdim,
                    y_train_regression_.data(), nin_workspace_.data(), y_pred_temp.data(), impn,
                    qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                    treemap, nodestatus, xbestsplit, bestvar,
                    nodepred, tnodewt, actual_nnode,
                    reinterpret_cast<const integer_t*>(cat_.data()),
                    config_.maxcat, catgoleft, nodextr_workspace_.data());
            }
        } else {
            // Use classification importance for classification and unsupervised
            // Use actual number of nodes for this tree (nnode_[tree_id]) instead of maxnode
            integer_t actual_nnode = nnode_[tree_id];
            if (actual_nnode == 0) {
                // Tree not grown yet or empty tree - skip importance computation
                actual_nnode = 1;  // Use minimum value to avoid division by zero
            }
            if (is_sparse_ && X_train_sparse_.is_valid()) {
            //     // SPARSE VERSION - use printf to avoid iostream corruption
            //     printf("[DEBUG VARIMP] tree=%d, sparse nrows=%d, nsample=%d, mdim=%d\n", 
            //            (int)tree_id, (int)X_train_sparse_.nrows, (int)config_.nsample, (int)config_.mdim);
            //     printf("[DEBUG VARIMP] indptr.size=%zu, data.size=%zu, indices.size=%zu, nnz=%d\n",
            //            X_train_sparse_.indptr.size(), X_train_sparse_.data.size(), 
            //            X_train_sparse_.indices.size(), (int)X_train_sparse_.nnz);
            //     printf("[DEBUG VARIMP] y_data=%p, nin=%p, jtr=%p\n", 
            //            (void*)y_data, (void*)nin_workspace_.data(), (void*)jtr_workspace_.data());
            //     printf("[DEBUG VARIMP] qimp_temp.size=%zu, qimpm_temp.size=%zu\n",
            //            qimp_temp.size(), qimpm_temp.size());
            //     printf("[DEBUG VARIMP] cat_.size=%zu, maxcat=%d, actual_nnode=%d\n",
            //            cat_.size(), (int)config_.maxcat, (int)actual_nnode);
            //     printf("[DEBUG VARIMP] jtr_workspace_.size=%zu, nodextr_workspace_.size=%zu\n",
            //            jtr_workspace_.size(), nodextr_workspace_.size());
            //     fflush(stdout);
                
                // // Validate sparse matrix internal consistency
                // if (X_train_sparse_.indptr.size() != static_cast<size_t>(X_train_sparse_.nrows + 1)) {
                //     printf("[ERROR] indptr.size mismatch: %zu != %d\n", 
                //            X_train_sparse_.indptr.size(), X_train_sparse_.nrows + 1);
                //     fflush(stdout);
                //     return;
                // }
                
                cpu_varimp_sparse(X_train_sparse_, config_.nsample, config_.mdim,  
                       y_data, nin_workspace_.data(), jtr_workspace_.data(), impn,
                       qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                       treemap, nodestatus, xbestsplit, bestvar, nodeclass, actual_nnode,
                       reinterpret_cast<const integer_t*>(cat_.data()),
                       temp_jvr.data(), temp_nodexvr.data(),
                       config_.maxcat, catgoleft, tnodewt, nodextr_workspace_.data());
            } else {
                // For unsupervised, use expanded data (real + synthetic samples)
                const real_t* X_data = (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) 
                    ? X_unsupervised_expanded_.data() 
                    : X_train_.data();
                
                cpu_varimp(X_data, config_.nsample, config_.mdim,  
                       y_data, nin_workspace_.data(), jtr_workspace_.data(), impn,
                       qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                       treemap, nodestatus, xbestsplit, bestvar, nodeclass, actual_nnode,
                       reinterpret_cast<const integer_t*>(cat_.data()),
                       temp_jvr.data(), temp_nodexvr.data(),
                       config_.maxcat, catgoleft, tnodewt, nodextr_workspace_.data());
            }
        }
        
        
        // Accumulate into global importance arrays
        for (integer_t i = 0; i < config_.nsample; i++) {
            qimp_[i] += qimp_temp[i];
        }
        
        // Only accumulate qimpm_ if local importance was requested
        if (config_.compute_local_importance) {
            // Cast to size_t to avoid integer overflow
            size_t qimpm_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
            for (size_t i = 0; i < qimpm_size; i++) {
                qimpm_[i] += qimpm_temp[i];
            }
        }
        
        for (integer_t i = 0; i < config_.mdim; i++) {
            avimp_[i] += avimp_temp[i];
            sqsd_[i] += sqsd_temp[i];
        }
    }


    // Store RF-GAP data if requested
    // NOTE: Skip RF-GAP for sparse mode - not yet implemented
    if (config_.compute_proximity && config_.use_rfgap && !is_sparse_) {
        // Ensure storage is allocated - resize to at least tree_id+1 to avoid out-of-bounds
        if (tree_nin_rfgap_.size() <= static_cast<size_t>(tree_id)) {
            // Resize to ntree to avoid repeated reallocations
            size_t new_size = static_cast<size_t>(config_.ntree);
            tree_nin_rfgap_.resize(new_size);
            tree_nodextr_rfgap_.resize(new_size);
        }
        
        // Store bootstrap multiplicities for this tree
        tree_nin_rfgap_[tree_id].resize(config_.nsample);
        std::copy(nin_workspace_.begin(), nin_workspace_.begin() + config_.nsample,
                  tree_nin_rfgap_[tree_id].begin());
        
        // Store terminal nodes for OOB samples (already computed by cpu_testreebag)
        tree_nodextr_rfgap_[tree_id].resize(config_.nsample, -1);
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] == 0) {
                // OOB sample - terminal node already computed
                tree_nodextr_rfgap_[tree_id][n] = nodextr_workspace_[n];
            }
        }
        
        // Compute terminal nodes for in-bag samples
        // Samples must be run through the tree to get their terminal nodes
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] > 0) {
                // In-bag sample - compute terminal node
                integer_t terminal_node = find_terminal_node(
                    X_train_.data() + n * config_.mdim,
                    nodestatus, bestvar, xbestsplit, treemap, catgoleft);
                tree_nodextr_rfgap_[tree_id][n] = terminal_node;
            }
        }
    }
    
    
    // Proximity computation if requested (standard proximity, not RF-GAP)
    // NOTE: proximity_matrix_ should be initialized ONCE before all trees, not per tree
    // This function is called per tree, so resize/reset should NOT occur here
    // Only allocate if empty (first tree), otherwise accumulate across trees
    if (config_.compute_proximity && !config_.use_rfgap) {
        // Only initialize on first tree (when empty), then accumulate across trees
        if (proximity_matrix_.empty()) {
            proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
            proximity_matrix_.reserve(static_cast<size_t>(config_.nsample) * config_.nsample);
        }
        // If not empty, proximity_matrix_ already exists and accumulation occurs into it
        
        std::vector<integer_t> nod(config_.maxnode);  // nod(nnode) in Fortran
        std::vector<integer_t> ncount(config_.nsample);  // ncount(nsample) in Fortran
        std::vector<integer_t> ncn(config_.nsample);  // ncn(nsample) in Fortran
        std::vector<integer_t> nodexb(config_.nsample);  // nodexb(nsample) in Fortran
        // Allocate ndbegin with size maxnode+1 to prevent out-of-bounds access
        // The proximity code accesses ndbegin[k+1], so at least maxnode+1 elements are required
        // nterm+1 is used where nterm <= maxnode, so maxnode+1 is safe
        std::vector<integer_t> ndbegin(config_.maxnode + 1);  // ndbegin(nterm+1) in Fortran - FIXED!
        std::vector<integer_t> npcase(config_.nsample);  // npcase(nsample) in Fortran - FIXED!

        // cpu_testreebag only sets nodextr_workspace_ for OOB samples
        // For standard proximity, terminal nodes are needed for ALL samples (both OOB and in-bag)
        // Compute terminal nodes for in-bag samples
        for (integer_t n = 0; n < config_.nsample; ++n) {
            if (nin_workspace_[n] > 0) {
                // In-bag sample - compute terminal node by running it through the tree
                integer_t terminal_node;
                if (is_sparse_ && X_train_sparse_.is_valid()) {
                    terminal_node = find_terminal_node_sparse(n, nodestatus, bestvar, 
                                                              xbestsplit, treemap, catgoleft);
                } else {
                    // For unsupervised, use expanded data (real + synthetic samples)
                    const real_t* X_data = (config_.task_type == TaskType::UNSUPERVISED && !X_unsupervised_expanded_.empty()) 
                        ? X_unsupervised_expanded_.data() + n * config_.mdim
                        : X_train_.data() + n * config_.mdim;
                    terminal_node = find_terminal_node(X_data,
                                                       nodestatus, bestvar, xbestsplit, treemap, catgoleft);
                }
                nodextr_workspace_[n] = terminal_node;
            }
        }

        cpu_proximity(nodestatus, nodextr_workspace_.data(), nin_workspace_.data(), config_.nsample,
                 nnode_[tree_id], proximity_matrix_.data(), nod.data(),
                 ncount.data(), ncn.data(), nodexb.data(), ndbegin.data(),
                 npcase.data());
    }
    
}

#ifdef CUDA_FOUND
void RandomForest::fit_batch_gpu(const real_t* X, const void* y,
                                   const real_t* sample_weight, integer_t batch_size) {
    // DEBUG: Enable verbose logging to track segfaults
    const bool DEBUG_SEGFAULT = false;  // Set to true to debug segfaults
    if (DEBUG_SEGFAULT) {
        fprintf(stderr, "[DEBUG] fit_batch_gpu ENTRY - nsample=%d, ntree=%d, mdim=%d, maxnode=%d, batch_size=%d\n",
                config_.nsample, config_.ntree, config_.mdim, config_.maxnode, batch_size);
        fprintf(stderr, "[DEBUG] Memory sizes: ntree*nsample=%zu, ntree*maxnode=%zu\n",
                static_cast<size_t>(config_.ntree) * config_.nsample,
                static_cast<size_t>(config_.ntree) * config_.maxnode);
        fflush(stderr);
    }
    
    // Before fit_batch_gpu, ensure CUDA context is ready (handles context between Jupyter cells)
    // This is especially important for GPU sequential mode (batch_size=1)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    if (DEBUG_SEGFAULT) { fprintf(stderr, "[DEBUG] CUDA context ready\n"); fflush(stderr); }
    
    // Debug: Check y data
    // if (y != nullptr) {
    //     const integer_t* y_data = static_cast<const integer_t*>(y);
    //     std::cout << "C++ fit_batch_gpu: y data first " << std::min(10, (int)config_.nsample) << ": ";
    //     for (int i = 0; i < std::min(10, (int)config_.nsample); i++) {
    //         std::cout << y_data[i] << " ";
    //     }
        // std::cout << std::endl;
    //     if (config_.nsample > 50) {
    //         std::cout << "C++ fit_batch_gpu: y data samples 50-59: ";
    //         for (int i = 50; i < std::min(60, (int)config_.nsample); i++) {
    //             std::cout << y_data[i] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // } else {
    //     std::cout << "C++ fit_batch_gpu: y data is nullptr" << std::endl;
    // }
    
    // std::cout << "Using GPU batch mode with batch size: " << batch_size << "\n";
    
    // Initialize variable importance array ONCE at the start of fit_batch_gpu
    // Do NOT reset it between batches - it needs to accumulate across all batches
    // This initialization happens once per call to fit_batch_gpu (which is called once per fit)
    if (config_.compute_importance) {
        // Only initialize if not already initialized (avimp_ should be empty or zero at first call)
        if (avimp_.empty() || std::all_of(avimp_.begin(), avimp_.end(), [](real_t v) { return v == 0.0f; })) {
            std::fill(avimp_.begin(), avimp_.end(), 0.0f);
        }
        // Otherwise, avimp_ already has values from previous batches - keep accumulating
    }

    // Allocate leaf_assignments_ if requested (GPU Dense)
    if (config_.compute_leaf_assignments && leaf_assignments_.empty()) {
        // Cast to size_t to avoid integer overflow with large ntree * nsample
        size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
        if (DEBUG_SEGFAULT) {
            fprintf(stderr, "[DEBUG] Allocating leaf_assignments_: size=%zu (%.2f MB)\n", 
                    leaf_size, leaf_size * sizeof(int16_t) / (1024.0 * 1024.0));
            fflush(stderr);
        }
        if (leaf_size > leaf_assignments_.max_size()) {
            throw std::runtime_error(
                "Cannot allocate leaf_assignments_: size " + std::to_string(leaf_size) + 
                " exceeds max_size() " + std::to_string(leaf_assignments_.max_size())
            );
        }
        leaf_assignments_.resize(leaf_size, 0);
        if (DEBUG_SEGFAULT) { fprintf(stderr, "[DEBUG] leaf_assignments_ allocated OK\n"); fflush(stderr); }
    }
    
    // Allocate and transfer GPU histogram data if histogram binning is enabled and data exists
    // This is done ONCE before all batches to avoid repeated memory transfers
    if (config_.use_histogram && histogram_data_ && histogram_data_->is_valid && 
        gpu_histogram_data_ == nullptr) {
        // GPU histogram allocation and transfer is handled via external function
        // to avoid including CUDA headers in this file
        // The actual allocation happens in the first call to gpu_growtree_batch
        // We just set a flag here to indicate histogram should be used on GPU
        rf::g_config.use_histogram = true;
    } else {
        rf::g_config.use_histogram = false;
    }
    
    // Process trees in batches
    if (DEBUG_SEGFAULT) { 
        fprintf(stderr, "[DEBUG] Starting batch loop: ntree=%d, batch_size=%d\n", config_.ntree, batch_size); 
        fflush(stderr); 
    }
    
    for (integer_t batch_start = 0; batch_start < config_.ntree; batch_start += batch_size) {
        integer_t batch_end = std::min(batch_start + batch_size, config_.ntree);
        integer_t num_trees_batch = batch_end - batch_start;

        if (DEBUG_SEGFAULT) {
            fprintf(stderr, "[DEBUG] BATCH %d-%d (num=%d)\n", batch_start, batch_end-1, num_trees_batch);
            fflush(stderr);
        }
        
        // std::cout << "DEBUG: fit_batch_gpu LOOP START - batch_start=" << batch_start 
        //           << ", batch_end=" << batch_end << ", num_trees_batch=" << num_trees_batch << std::endl;
        
        // std::cout << "  Processing trees " << batch_start << "-" << (batch_end - 1) << "...\n";
        
        // Resize workspace arrays for batch processing
        // For batch processing, we DON'T resize the main tree arrays (nodestatus_, bestvar_, etc.)
        // because they need to persist across batches. Only resize per-sample workspace arrays.
        // Cast to size_t to avoid integer overflow with large num_trees_batch * config_.nsample
        size_t num_trees_batch_size = static_cast<size_t>(num_trees_batch);
        size_t nsample_size = static_cast<size_t>(config_.nsample);
        size_t mdim_size = static_cast<size_t>(config_.mdim);
        size_t maxcat_size = static_cast<size_t>(config_.maxcat);
        size_t maxnode_size = static_cast<size_t>(config_.maxnode);
        size_t nclass_size = static_cast<size_t>(config_.nclass);
        
        amat_workspace_.resize(num_trees_batch_size * mdim_size * nsample_size);
        jinbag_workspace_.resize(num_trees_batch_size * nsample_size);
        // DON'T resize: tnodewt_, xbestsplit_, nodestatus_, bestvar_, treemap_, catgoleft_, nodeclass_, nnode_
        // These are persistent across batches and were sized in initialize_arrays()!
        jtr_workspace_.resize(num_trees_batch_size * nsample_size);
        nodextr_workspace_.resize(num_trees_batch_size * nsample_size);
        idmove_workspace_.resize(num_trees_batch_size * nsample_size);
        igoleft_workspace_.resize(num_trees_batch_size * maxcat_size);
        igl_workspace_.resize(num_trees_batch_size * maxcat_size);
        nodestart_workspace_.resize(num_trees_batch_size * maxnode_size);
        nodestop_workspace_.resize(num_trees_batch_size * maxnode_size);
        itemp_workspace_.resize(num_trees_batch_size * nsample_size);
        itmp_workspace_.resize(num_trees_batch_size * maxcat_size);
        icat_workspace_.resize(num_trees_batch_size * maxcat_size);
        tcat_workspace_.resize(num_trees_batch_size * maxcat_size);
        classpop_workspace_.resize(num_trees_batch_size * nclass_size * maxnode_size);
        wr_workspace_.resize(num_trees_batch_size * nsample_size);
        wl_workspace_.resize(num_trees_batch_size * nsample_size);
        tclasscat_workspace_.resize(num_trees_batch_size * nclass_size * maxcat_size);
        tclasspop_workspace_.resize(num_trees_batch_size * nclass_size);
        tmpclass_workspace_.resize(num_trees_batch_size * nclass_size);
        
        // Prepare seeds for this batch
        std::vector<integer_t> batch_seeds(num_trees_batch);
        for (integer_t i = 0; i < num_trees_batch; ++i) {
            batch_seeds[i] = config_.iseed + batch_start + i;
        }
        
        // Allocate batch-specific nin workspace (needed for bootstrap results from all trees)
        // Cast to size_t to avoid integer overflow with large num_trees_batch * config_.nsample
        size_t nin_batch_size = static_cast<size_t>(num_trees_batch) * static_cast<size_t>(config_.nsample);
        std::vector<integer_t> nin_batch(nin_batch_size);
        
        // std::cout << "DEBUG: fit_batch_gpu - About to set proximity_ptr. nsample=" << config_.nsample 
        //           << ", compute_proximity=" << config_.compute_proximity 
        //           << ", use_qlora=" << config_.use_qlora << std::endl;
        
        // Cast y to integer_t* for GPU function
        const integer_t* y_data = static_cast<const integer_t*>(y);
        
        // For low-rank mode, DO NOT allocate full proximity matrix (80GB for 100k samples!)
        // Keep proximity in low-rank form (~40MB) and reconstruct on-demand only if needed
        dp_t* proximity_ptr = nullptr;
        bool use_lowrank = false;
        bool use_upper_triangle = false;
        if (config_.compute_proximity && !config_.use_rfgap) {
            // Only compute standard proximity if RF-GAP is not enabled
            // MODIFIED: Always use low-rank mode on GPU, regardless of dataset size
            // This enables low-rank factors for all datasets when using GPU
            // NOTE: Low-rank mode is GPU-only (requires CUDA kernels, cuBLAS, cuSolver)
            // CPU mode will never use low-rank - it always computes full proximity matrix
            // Check if GPU is actually being used (use_gpu flag)
            bool actually_using_gpu = config_.use_gpu;
            if (actually_using_gpu && config_.use_qlora) {
                use_lowrank = true;  // Force low-rank for all GPU datasets
                use_upper_triangle = true;  // Default to true for memory efficiency
            } else {
                // CPU mode or GPU without qlora: use full matrix (low-rank not available on CPU)
                use_lowrank = false;  // CPU cannot use low-rank (CUDA-only feature)
                use_upper_triangle = true;  // Default to true for memory efficiency
            }
            
            if (use_lowrank && use_upper_triangle) {
                // For low-rank mode, DO NOT allocate full matrix - keep in low-rank form!
                // This saves ~80GB of memory. Reconstruction can be done on-demand if needed.
                proximity_ptr = nullptr;  // Pass nullptr to indicate low-rank mode
            } else {
                // Traditional mode: allocate full matrix (for smaller datasets)
                if (proximity_matrix_.empty()) {
                    proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
                }
                proximity_ptr = proximity_matrix_.data();
            }
        } else if (config_.compute_proximity && config_.use_rfgap) {
            // RF-GAP mode: don't compute standard proximity during GPU batching
            // RF-GAP proximity is computed after all batches complete
            proximity_ptr = nullptr;
            use_lowrank = false;
            use_upper_triangle = false;
        }
        
        // std::cout << "DEBUG: fit_batch_gpu - proximity_ptr=" << (proximity_ptr != nullptr ? "non-null" : "null") << std::endl;
        
        // Copy config values to global config for GPU code BEFORE calling gpu_growtree_batch
        // Set global config values that GPU code needs (before first batch)
        // Always set use_casewise, not just on first batch
        // This ensures proximity kernels have the correct flag
        rf::g_config.use_casewise = config_.use_casewise;
        
        if (batch_start == 0) {
            // compute_proximity overrides use_qlora - if proximity is disabled, disable qlora too
            // This prevents low-rank proximity initialization when proximity is not requested
            rf::g_config.use_qlora = config_.compute_proximity ? config_.use_qlora : false;
            rf::g_config.quant_mode = config_.quant_mode;
            // Only compute standard proximity in GPU if RF-GAP is not enabled
            rf::g_config.iprox = (config_.compute_proximity && !config_.use_rfgap) ? 1 : 0;
            rf::g_config.use_rfgap = config_.use_rfgap;
            rf::g_config.lowrank_rank = config_.lowrank_rank;
            // Set impn flag for GPU importance computation (1 = compute local importance, 0 = skip)
            rf::g_config.impn = config_.compute_local_importance ? 1 : 0;
        } else {
            // Ensure use_casewise is set for subsequent batches too
            rf::g_config.use_casewise = config_.use_casewise;
        }
        // std::cout << "  lowrank_ptr=" << ((use_lowrank && use_upper_triangle) ? "will-pass" : "nullptr") << std::endl;
        
        // For low-rank mode (use_qlora=True), we MUST use low-rank mode (no temp buffer)
        // If proximity_ptr is not nullptr and use_qlora is enabled, this would trigger temp buffer allocation in GPU code
        if (proximity_ptr != nullptr && config_.use_qlora) {
            // std::cerr << "ERROR: proximity_ptr is non-null for low-rank mode (use_qlora=True). This would trigger huge temp buffer allocation. Forcing nullptr." << std::endl;  // Commented to avoid stream conflicts
            proximity_ptr = nullptr;  // Force nullptr to prevent temp buffer allocation
        }
        
        // Prepare histogram data pointers (if enabled and available)
        bool use_hist_gpu = config_.use_histogram && histogram_data_ && histogram_data_->is_valid;
        const uint8_t* X_binned_ptr = use_hist_gpu ? histogram_data_->X_binned.data() : nullptr;
        const real_t* bin_edges_ptr = nullptr;
        const integer_t* n_bins_ptr = nullptr;
        std::vector<real_t> bin_edges_flat;
        std::vector<integer_t> n_bins_vec;
        
        // GPU kernel supports up to 256 bins with 10 classes in shared memory
        const integer_t GPU_HIST_MAX_BINS = 256;
        
        if (use_hist_gpu) {
            // Flatten bin_edges for GPU: [mdim × 257]
            // Cast to size_t to avoid integer overflow
            size_t bin_edges_size = static_cast<size_t>(config_.mdim) * 257;
            bin_edges_flat.resize(bin_edges_size, 0.0f);
            n_bins_vec.resize(config_.mdim);
            for (integer_t f = 0; f < config_.mdim; ++f) {
                // Cap n_bins at GPU_HIST_MAX_BINS for GPU kernel shared memory limits
                integer_t cpu_bins = histogram_data_->feature_info[f].n_bins;
                n_bins_vec[f] = std::min(cpu_bins, GPU_HIST_MAX_BINS);
                for (integer_t e = 0; e < 257; ++e) {
                    bin_edges_flat[f * 257 + e] = histogram_data_->feature_info[f].bin_edges[e];
                }
            }
            bin_edges_ptr = bin_edges_flat.data();
            n_bins_ptr = n_bins_vec.data();
        }
        
        // Pass offset pointers for persistent arrays (they contain ALL trees)
        // Cast to size_t to avoid integer overflow in pointer arithmetic
        size_t batch_offset = static_cast<size_t>(batch_start) * static_cast<size_t>(config_.maxnode);
        size_t batch_treemap_offset = static_cast<size_t>(batch_start) * 2 * static_cast<size_t>(config_.maxnode);
        size_t batch_catgoleft_offset = static_cast<size_t>(batch_start) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
        
        if (DEBUG_SEGFAULT) {
            fprintf(stderr, "[DEBUG] Calling gpu_growtree_batch: batch_offset=%zu, batch_treemap_offset=%zu\n",
                    batch_offset, batch_treemap_offset);
            fflush(stderr);
        }
        
        gpu_growtree_batch(
            num_trees_batch, X, sample_weight, y_data,
            static_cast<integer_t>(config_.task_type),
            cat_.data(), ties_.data(), nin_batch.data(),
            (config_.task_type == TaskType::REGRESSION) ? y_train_regression_.data() : nullptr,
            config_.mdim, config_.nsample, config_.nclass, config_.maxnode,
            config_.minndsize, config_.mtry, config_.nsample, config_.maxcat,
            batch_seeds.data(), config_.ncsplit, config_.ncmax,
            amat_workspace_.data(), jinbag_workspace_.data(), 
            tnodewt_.data() + batch_offset,
            xbestsplit_.data() + batch_offset,
            nodestatus_.data() + batch_offset,
            bestvar_.data() + batch_offset,
            treemap_.data() + batch_treemap_offset,
            catgoleft_.data() + batch_catgoleft_offset,
            nodeclass_.data() + batch_offset,
            nnode_.data() + static_cast<size_t>(batch_start),
            jtr_workspace_.data(), nodextr_workspace_.data(),
            idmove_workspace_.data(), igoleft_workspace_.data(), igl_workspace_.data(),
            nodestart_workspace_.data(), nodestop_workspace_.data(), itemp_workspace_.data(), itmp_workspace_.data(),
            icat_workspace_.data(), tcat_workspace_.data(), classpop_workspace_.data(), wr_workspace_.data(),
            wl_workspace_.data(), tclasscat_workspace_.data(), tclasspop_workspace_.data(), tmpclass_workspace_.data(),
            q_.data(), nout_.data(),             config_.compute_importance ? avimp_.data() : nullptr,
            (config_.compute_importance || config_.compute_local_importance) ? qimp_.data() : nullptr,
            config_.compute_local_importance ? qimpm_.data() : nullptr,  // Only pass qimpm if local importance requested
            proximity_ptr,  // Pass proximity matrix pointer (allocated above for low-rank mode)
            (config_.task_type == TaskType::REGRESSION) ? oob_predictions_.data() : nullptr,  // For regression: accumulate OOB predictions
            (config_.task_type == TaskType::REGRESSION) ? (node_predictions_regression_.data() + batch_offset) : nullptr,  // For regression: terminal node predictions
            (use_lowrank && use_upper_triangle) ? &lowrank_proximity_ptr_ : nullptr,  // Store LowRankProximityMatrix pointer if low-rank mode
            config_.compute_proximity_importance ? proximity_importance_.data() : nullptr,  // Proximity importance
            config_.compute_leaf_assignments ? (leaf_assignments_.data() + static_cast<size_t>(batch_start) * static_cast<size_t>(config_.nsample)) : nullptr,  // Leaf assignments
            // HISTOGRAM PARAMETERS
            use_hist_gpu,
            X_binned_ptr,
            bin_edges_ptr,
            n_bins_ptr,
            config_.n_bins
        );
        
        // std::cout << "[FIT_BATCH] gpu_growtree_batch RETURNED for batch " << batch_start << "-" << (batch_end-1) << std::endl;
        
        // Sync after batch to ensure GPU ops complete before next batch
        // Use stream sync for better Jupyter compatibility and performance
        #ifdef CUDA_FOUND
        cudaStreamSynchronize(0);
        #endif
        
        // Build oob_trees_ from nin_batch for Breiman's OOB-query approach
        // This tracks which trees each sample was OOB for (for fast top-K lookup)
        if (config_.compute_leaf_assignments) {
            // Initialize oob_trees_ on first batch
            if (batch_start == 0 && oob_trees_.empty()) {
                oob_trees_.resize(config_.nsample);
            }
            
            // Add trees where each sample was OOB
            for (integer_t i = 0; i < num_trees_batch; ++i) {
                integer_t tree_id = batch_start + i;
                integer_t nin_offset = i * config_.nsample;
                
                for (integer_t n = 0; n < config_.nsample; ++n) {
                    if (nin_batch[nin_offset + n] == 0) {  // OOB sample
                        oob_trees_[n].push_back(tree_id);
                    }
                }
            }
        }
        
        // Batch completed successfully - no debug print to avoid potential issues
        
        // Store RF-GAP data if requested (after batch completes, simple memory copy - doesn't slow down GPU)
        if (config_.compute_proximity && config_.use_rfgap) {
            // Ensure storage is allocated - resize to ntree to avoid repeated reallocations
            if (tree_nin_rfgap_.size() < static_cast<size_t>(config_.ntree)) {
                size_t new_size = static_cast<size_t>(config_.ntree);
                tree_nin_rfgap_.resize(new_size);
                tree_nodextr_rfgap_.resize(new_size);
            }
            
            // Copy nin and nodextr for each tree in this batch (already computed by GPU)
            for (integer_t i = 0; i < num_trees_batch; ++i) {
                integer_t tree_id = batch_start + i;
                // Cast to size_t to avoid integer overflow in offset calculation
                size_t nin_offset = static_cast<size_t>(i) * static_cast<size_t>(config_.nsample);
                size_t nodextr_offset = static_cast<size_t>(i) * static_cast<size_t>(config_.nsample);
                
                // Store bootstrap multiplicities
                tree_nin_rfgap_[tree_id].resize(config_.nsample);
                std::copy(nin_batch.data() + nin_offset,
                         nin_batch.data() + nin_offset + config_.nsample,
                         tree_nin_rfgap_[tree_id].begin());
                
                // Store terminal nodes (OOB samples already have terminal nodes, in-bag will be computed later)
                tree_nodextr_rfgap_[tree_id].resize(config_.nsample, -1);
                std::copy(nodextr_workspace_.data() + nodextr_offset,
                         nodextr_workspace_.data() + nodextr_offset + config_.nsample,
                         tree_nodextr_rfgap_[tree_id].begin());
            }
        }
        
        // Update progress for GPU batch processing (callback)
        if (progress_callback_) {
            progress_callback_(batch_end, config_.ntree);
        }
    }
    
    // After all batches complete, compute terminal nodes for in-bag samples (for RF-GAP)
    // This is done in parallel on CPU and doesn't slow down GPU batching
    // Store config values in local variables to avoid accessing potentially corrupted config_
    bool compute_proximity_local = config_.compute_proximity;
    bool use_rfgap_local = config_.use_rfgap;
    
    if (compute_proximity_local && use_rfgap_local) {
        // std::cout << "[FIT_BATCH] Computing RF-GAP terminal nodes..." << std::endl;  // Commented to avoid stream conflicts with Python progress bars
        // Compute terminal nodes for in-bag samples in parallel
#ifdef _OPENMP
        if (g_config.n_threads_cpu > 0) {
            omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
        }
#pragma omp parallel for
#endif
        for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
            // Get tree-specific data
            // Cast to size_t to avoid integer overflow in pointer arithmetic
            size_t tree_offset = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
            size_t tree_treemap_offset = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
            size_t tree_catgoleft_offset = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
            
            integer_t* nodestatus = nodestatus_.data() + tree_offset;
            integer_t* bestvar = bestvar_.data() + tree_offset;
            real_t* xbestsplit = xbestsplit_.data() + tree_offset;
            integer_t* treemap = treemap_.data() + tree_treemap_offset;
            integer_t* catgoleft = catgoleft_.data() + tree_catgoleft_offset;
            
            // Compute terminal nodes for in-bag samples
            for (integer_t n = 0; n < config_.nsample; ++n) {
                if (tree_nin_rfgap_[tree_id][n] > 0) {
                    // In-bag sample - compute terminal node
                    integer_t terminal_node = find_terminal_node(
                        X_train_.data() + n * config_.mdim,
                        nodestatus, bestvar, xbestsplit, treemap, catgoleft);
                    tree_nodextr_rfgap_[tree_id][n] = terminal_node;
                }
            }
        }
    }
    
    // Ensure all CUDA operations complete before accessing results
    // For Jupyter compatibility, skip explicit CUDA sync here
    // The GPU code already synchronizes before returning from each batch
    // Additional sync can cause crashes if CUDA context is in an invalid state
    // This matches the pattern from commit 6a9a52f which avoids aggressive syncs
    // Store use_gpu in local variable before accessing to avoid potential corruption
    bool use_gpu_local = config_.use_gpu;
    if (use_gpu_local) {
        // Just clear any pending CUDA errors - don't sync (GPU code already synced)
        // This prevents crashes when CUDA context is in an invalid state
        #ifdef CUDA_FOUND
        cudaGetLastError();  // Clear any pending errors
        #endif
    }
    
    // NOTE: GPU Dense proximity importance is computed entirely on GPU in gpu_growtree_batch
    // It accumulates across batches using d_prox_imp_all and is copied to host at the end
    // The normalization by nout_ is done in finalize_training()
    
    // ========================================================================
    // GPU DENSE REGRESSION: node_predictions_regression_ is computed on GPU
    // ========================================================================
    // NOTE: gpu_growtree_batch already computes node_predictions_regression_ correctly
    // using only in-bag samples (via gpu_compute_nodepred_batch_kernel which filters
    // by nin_all[sample_offset + n] > 0). The result is copied back to host.
    // DO NOT overwrite it here - the previous post-hoc code was buggy because it
    // used ALL samples instead of just in-bag samples.
    
    // Add memory barrier before return to ensure all writes are visible
    // This helps prevent issues with stack corruption or invalid memory access during return
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // Note: Variable importance computation is now handled within gpu_growtree_batch
    // to ensure OOB predictions are available for each tree individually
    
    // NOTE: finalize_training() is called in fit_classification() after compute_oob_predictions()
    // Match repo version - don't call it here to avoid double call
}
#endif // CUDA_FOUND

void RandomForest::finalize_training() {
    // Normalize OOB votes
    // Match repo version - no safety checks, just normalize
    for (integer_t n = 0; n < config_.nsample; ++n) {
        if (nout_[n] > 0) {
            for (integer_t j = 0; j < config_.nclass; ++j) {
                // Cast to size_t to avoid integer overflow in index calculation
                size_t q_idx = static_cast<size_t>(n) * static_cast<size_t>(config_.nclass) + static_cast<size_t>(j);
                q_[q_idx] /= static_cast<real_t>(nout_[n]);
            }
        }
    }
    
    // ========================================================================
    // PROXIMITY IMPORTANCE NORMALIZATION - SKIPPED
    // ========================================================================
    // Per-tree normalization (1/noob) is now done during tree training, matching
    // Breiman-Cutler methodology for local variable importance.
    // 
    // The final proximity_importance_[n * mdim + k] value is:
    //   sum of (1/noob_t) across trees where:
    //     - sample n was OOB
    //     - sample n was correctly predicted
    //     - permuting feature k changed the terminal node
    // 
    // No additional normalization needed here.
    // ========================================================================

    // Compute OOB predictions and error based on task type
    // Note: OOB error is computed in compute_classification_error() which is called
    // in fit_classification() after finalize_training(). No recomputation needed here.
    // The oob_error_ is already set by compute_classification_error().
    if (config_.task_type == TaskType::CLASSIFICATION) {
        // OOB error already computed in compute_classification_error() - nothing to do here
    } else if (config_.task_type == TaskType::REGRESSION) {
        // For regression, compute MSE instead of classification error
        compute_regression_mse();
    }
    
    // Finalize proximity if computed (only once at the end)
    if (config_.compute_proximity && !proximity_normalized_) {
        if (config_.use_rfgap) {
            // RF-GAP proximity mode
            #ifdef CUDA_FOUND
            // Check if RF-GAP was already computed with QLoRA during GPU batching
            if (config_.use_gpu && config_.use_qlora && lowrank_proximity_ptr_ != nullptr) {
                // RF-GAP was already computed incrementally in low-rank form during GPU batching
                // No need to compute full matrix - low-rank factors are already available
                // Set OOB counts for RF-GAP normalization in MDS
                rf::cuda::LowRankProximityMatrix* lowrank_prox = 
                    static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_proximity_ptr_);
                lowrank_prox->set_oob_counts_for_rfgap(nout_.data());
                // Just mark as normalized
                proximity_normalized_ = true;
            } else {
                // RF-GAP was not computed with QLoRA during batching, compute full matrix now
                if (tree_nin_rfgap_.size() != static_cast<size_t>(config_.ntree) ||
                    tree_nodextr_rfgap_.size() != static_cast<size_t>(config_.ntree)) {
                    throw std::runtime_error("RF-GAP proximity: per-tree data not available. "
                                            "Ensure all trees were grown with use_rfgap=True.");
                }
                
                // Ensure proximity matrix is allocated
                if (proximity_matrix_.empty()) {
                    proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
                }
                
                // Compute RF-GAP proximity (full matrix - expensive but necessary)
                // Use GPU if available for better performance
                if (config_.use_gpu && rf::cuda::cuda_is_available()) {
                    // Flatten per-tree data for GPU
                    // Cast to size_t to avoid integer overflow
                    size_t tree_flat_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
                    std::vector<integer_t> tree_nin_flat(tree_flat_size);
                    std::vector<integer_t> tree_nodextr_flat(tree_flat_size);
                    
                    for (integer_t t = 0; t < config_.ntree; ++t) {
                        for (integer_t s = 0; s < config_.nsample; ++s) {
                            // Cast to size_t to avoid integer overflow in index calculation
                            size_t flat_idx = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(s);
                            tree_nin_flat[flat_idx] = tree_nin_rfgap_[t][s];
                            tree_nodextr_flat[flat_idx] = tree_nodextr_rfgap_[t][s];
                        }
                    }
                    
                    // Use GPU RF-GAP for better performance
                    rf::gpu_proximity_rfgap(config_.ntree, config_.nsample,
                                           tree_nin_flat.data(), tree_nodextr_flat.data(),
                                           proximity_matrix_.data());
                } else {
                    // Fall back to CPU RF-GAP
                    cpu_proximity_rfgap(config_.ntree, config_.nsample,
                                       tree_nin_rfgap_, tree_nodextr_rfgap_,
                                       proximity_matrix_.data());
                }
                
                // If QLoRA is enabled, convert RF-GAP full matrix to low-rank factors for memory efficiency
                // This allows GPU MDS to work with RF-GAP proximity
                if (config_.use_gpu && config_.use_qlora) {
                    // Create low-rank proximity matrix and convert RF-GAP to factors
                    rf::cuda::LowRankProximityMatrix* lowrank_rfgap = 
                        new rf::cuda::LowRankProximityMatrix(
                            config_.nsample, 
                            config_.lowrank_rank,
                            static_cast<rf::cuda::QuantizationLevel>(config_.quant_mode),
                            config_.lowrank_rank * 2,  // max_rank
                            config_.lowrank_power_iterations  // power_iterations
                        );
                    
                    if (!lowrank_rfgap->initialize()) {
                        delete lowrank_rfgap;
                        throw std::runtime_error("Failed to initialize low-rank proximity matrix for RF-GAP");
                    }
                    
                    // Convert full RF-GAP matrix to low-rank factors
                    // This uses SVD to approximate P ≈ A × B'
                    lowrank_rfgap->add_tree_contribution(proximity_matrix_.data(), config_.nsample, false);
                    
                    // Store low-rank pointer and free full matrix to save memory
                    // Use exception-safe deletion to prevent double-free
                    if (lowrank_proximity_ptr_ != nullptr) {
                        try {
                            delete static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_proximity_ptr_);
                        } catch (...) {
                            // Ignore errors during deletion (may be called multiple times)
                        }
                        lowrank_proximity_ptr_ = nullptr;
                    }
                    lowrank_proximity_ptr_ = lowrank_rfgap;
                    
                    // Set OOB counts for RF-GAP normalization in MDS
                    lowrank_rfgap->set_oob_counts_for_rfgap(nout_.data());
                    
                    // Free full matrix to save memory (can be reconstructed from factors if needed)
                    proximity_matrix_.clear();
                    proximity_matrix_.shrink_to_fit();
                }
                
                proximity_normalized_ = true;
            }
            #else
            // CUDA not available - use CPU RF-GAP
            if (tree_nin_rfgap_.size() != static_cast<size_t>(config_.ntree) ||
                tree_nodextr_rfgap_.size() != static_cast<size_t>(config_.ntree)) {
                throw std::runtime_error("RF-GAP proximity: per-tree data not available. "
                                        "Ensure all trees were grown with use_rfgap=True.");
            }
            
            // Ensure proximity matrix is allocated
            if (proximity_matrix_.empty()) {
                proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
            }
            
            // Ensure proximity matrix is properly allocated before RF-GAP computation
            // This prevents memory corruption if the vector was moved or resized
            if (proximity_matrix_.size() != static_cast<size_t>(config_.nsample) * config_.nsample) {
                proximity_matrix_.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
            }
            
            // Verify RF-GAP data vectors are properly sized
            if (tree_nin_rfgap_.size() != static_cast<size_t>(config_.ntree) ||
                tree_nodextr_rfgap_.size() != static_cast<size_t>(config_.ntree)) {
                throw std::runtime_error("RF-GAP proximity: per-tree data size mismatch. "
                                        "Expected " + std::to_string(config_.ntree) + " trees, "
                                        "got tree_nin_rfgap_.size()=" + std::to_string(tree_nin_rfgap_.size()) + 
                                        ", tree_nodextr_rfgap_.size()=" + std::to_string(tree_nodextr_rfgap_.size()));
            }
            
            // Ensure vectors are stable before passing to cpu_proximity_rfgap
            // Reserve capacity to prevent reallocation during function call
            // This prevents memory corruption if vectors are moved
            tree_nin_rfgap_.reserve(config_.ntree);
            tree_nodextr_rfgap_.reserve(config_.ntree);
            for (size_t t = 0; t < static_cast<size_t>(config_.ntree); ++t) {
                if (t < tree_nin_rfgap_.size()) {
                    tree_nin_rfgap_[t].reserve(config_.nsample);
                    tree_nodextr_rfgap_[t].reserve(config_.nsample);
                }
            }
            
            // Ensure proximity matrix is stable - reserve capacity to prevent reallocation
            proximity_matrix_.reserve(static_cast<size_t>(config_.nsample) * config_.nsample);
            
            cpu_proximity_rfgap(config_.ntree, config_.nsample,
                               tree_nin_rfgap_, tree_nodextr_rfgap_,
                               proximity_matrix_.data());
            proximity_normalized_ = true;
            #endif
        } else {
        // For low-rank mode, proximity_matrix_ is empty - don't try to normalize
        if (lowrank_proximity_ptr_ == nullptr && !proximity_matrix_.empty()) {
            // Traditional mode: normalize proximity matrix
            // Cast to size_t to avoid integer overflow
            size_t proxsym_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.nsample);
            std::vector<dp_t> proxsym(proxsym_size);
            cpu_finishprox(config_.nsample, nout_.data(), proximity_matrix_.data(), proxsym.data());
            proximity_matrix_ = proxsym;  // Use symmetrized version
            proximity_normalized_ = true;
        } else if (lowrank_proximity_ptr_ != nullptr) {
            // Low-rank mode: proximity stored in low-rank factors, no full matrix to normalize
            proximity_normalized_ = true;  // Mark as normalized (no full matrix to normalize)
            // std::cout << "GPU Proximity: Low-rank mode - proximity stored in factors (no full matrix to normalize)" << std::endl;
            }
        }
    }
    
    // NOTE: Outlier detection is ON-DEMAND only via compute_outlier_scores()
    // It uses leaf_assignments (memory-efficient) NOT full proximity matrix
    // Users call: model.compute_outlier_scores(mode='greedy') or mode='full'
    // Requires: compute_leaf_assignments=True when creating the model

    // Finalize feature importance if computed
    if (config_.compute_importance) {
        // Copy accumulated importance values to final feature_importances_ array
        // Note: avimp_ accumulates across all trees - no normalization needed
        // (importance is already normalized per tree by noob in varimp calculation)
        // 
        // SKIP this copy if avimp_ is empty - this happens for GPU SPARSE mode where
        // feature_importances_ was already set directly in fit_sparse_gpu_* functions.
        // Also check if avimp_ has any non-zero values to avoid overwriting good data with zeros.
        bool avimp_has_data = !avimp_.empty() && avimp_.size() == static_cast<size_t>(config_.mdim);
        if (avimp_has_data) {
            // Check if avimp_ has any non-zero values
            real_t avimp_sum = 0;
            for (integer_t i = 0; i < config_.mdim; i++) {
                avimp_sum += std::abs(avimp_[i]);
            }
            // Only copy if avimp_ has actual data
            if (avimp_sum > 0) {
                for (integer_t i = 0; i < config_.mdim; i++) {
                    feature_importances_[i] = avimp_[i];
                }
            }
        }
        
        // Compute local importance if requested
        if (config_.compute_local_importance) {
            // CLIQUE removed for v1.0 (v2.0 feature)
            if (config_.importance_method == "clique") {
                throw std::runtime_error("CLIQUE importance is not available in v1.0. Use importance_method='local_imp' instead.");
            }
            // Standard local importance (original method)
            // For REGRESSION: use localimp_regression (opposite sign - increase in MSE)
            // For CLASSIFICATION/UNSUPERVISED: use localimp (decrease in accuracy)
            if (config_.task_type == TaskType::REGRESSION) {
                localimp_regression(config_.nsample, config_.mdim, config_.ntree, qimp_.data(), qimpm_.data());
            } else {
                localimp(config_.nsample, config_.mdim, config_.ntree, qimp_.data(), qimpm_.data());
            }
        }
    }
    
    // NOTE: Proximity importance normalization is now done in finalize_training()
    // to ensure it happens exactly once after all trees are grown.
}

#ifdef CUDA_FOUND
// ============================================================================
// GPU SPARSE CLASSIFICATION - Uses parallel GPU sparse tree growing
// ============================================================================
void RandomForest::fit_sparse_gpu_classification(const SparseMatrixCSR& X, 
                                                  const integer_t* y, 
                                                  const real_t* sample_weight) {
    // Ensure CUDA context is ready
    rf::cuda::cuda_ensure_context_ready();
    
    // Configure GPU sparse training
    rf::cuda::GpuSparseForestConfig gpu_config;
    gpu_config.nsample = config_.nsample;
    gpu_config.mdim = config_.mdim;
    gpu_config.nclass = config_.nclass;
    gpu_config.ntree = config_.ntree;
    gpu_config.mtry = config_.mtry;
    gpu_config.maxnode = config_.maxnode;
    gpu_config.min_node_size = config_.minndsize;
    gpu_config.rank = config_.lowrank_rank;
    gpu_config.max_depth = 100;  // Safety limit
    gpu_config.compute_importance = config_.compute_importance;
    gpu_config.compute_local_importance = config_.compute_local_importance;
    gpu_config.compute_proximity = config_.compute_proximity;
    gpu_config.compute_proximity_importance = config_.compute_proximity_importance;
    gpu_config.compute_leaf_assignments = config_.compute_leaf_assignments;
    gpu_config.use_casewise = config_.use_casewise;
    gpu_config.use_qlora = config_.use_qlora;
    gpu_config.seed = config_.iseed;
    gpu_config.task_type = 0;  // Classification
    gpu_config.use_histogram = config_.use_histogram;
    gpu_config.n_bins = config_.n_bins;
    gpu_config.batch_size = config_.batch_size;  // Batched training support
    
    // Allocate output arrays
    // Cast to size_t to avoid integer overflow with large ntree * maxnode
    size_t tree_node_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.maxnode);
    size_t tree_2node_size = static_cast<size_t>(config_.ntree) * 2 * static_cast<size_t>(config_.maxnode);
    std::vector<integer_t> h_nodestatus(tree_node_size);
    std::vector<integer_t> h_bestvar(tree_node_size);
    std::vector<real_t> h_xbestsplit(tree_node_size);
    std::vector<integer_t> h_treemap(tree_2node_size);
    std::vector<integer_t> h_nodeclass(tree_node_size);
    std::vector<integer_t> h_nnode(static_cast<size_t>(config_.ntree));
    std::vector<real_t> h_importance(config_.mdim, 0.0f);
    std::vector<real_t> h_local_importance;
    std::vector<dp_t> h_proximity;
    std::vector<real_t> h_prox_importance;
    std::vector<real_t> h_overall_prox_importance(config_.mdim, 0.0f);
    
    if (config_.compute_local_importance) {
        // Cast to size_t to avoid integer overflow
        size_t local_imp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        h_local_importance.resize(local_imp_size, 0.0f);
    }
    if (config_.compute_proximity_importance) {
        // Cast to size_t to avoid integer overflow
        size_t prox_imp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        h_prox_importance.resize(prox_imp_size, 0.0f);
    }
    
    // Allocate leaf assignments buffer (for on-demand proximity computation)
    // NOTE: GPU sparse uses leaf_assignments, not low-rank factors for proximity
    std::vector<int16_t> h_leaf_assignments;
    if (config_.compute_leaf_assignments) {
        // Cast to size_t to avoid integer overflow with large ntree * nsample
        size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
        if (leaf_size > h_leaf_assignments.max_size()) {
            throw std::runtime_error(
                "Cannot allocate leaf assignments: size " + std::to_string(leaf_size) + 
                " exceeds max_size() " + std::to_string(h_leaf_assignments.max_size())
            );
        }
        h_leaf_assignments.resize(leaf_size, 0);
    }
    
    // Allocate nin_all buffer for oob_trees_ population (Breiman's proximity)
    std::vector<integer_t> h_nin_all;
    if (config_.compute_leaf_assignments) {
        // Cast to size_t to avoid integer overflow with large ntree * nsample
        size_t nin_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
        if (nin_size > h_nin_all.max_size()) {
            throw std::runtime_error(
                "Cannot allocate nin_all: size " + std::to_string(nin_size) + 
                " exceeds max_size() " + std::to_string(h_nin_all.max_size())
            );
        }
        h_nin_all.resize(nin_size, 0);
    }
    
    // Ensure q_ and nout_ are sized correctly for OOB vote collection
    // Cast to size_t to avoid integer overflow
    size_t q_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.nclass);
    q_.assign(q_size, 0.0f);
    nout_.assign(static_cast<size_t>(config_.nsample), 0);
    
    // Allocate qimp for local importance (per-sample original correct weights)
    std::vector<real_t> h_qimp;
    if (config_.compute_local_importance) {
        h_qimp.resize(config_.nsample, 0.0f);
    }
    
    // Call GPU sparse training
    // NOTE: GPU sparse uses leaf_assignments for on-demand proximity
    integer_t err = rf::cuda::train_sparse_forest_gpu(
        X, y, nullptr,  // Classification: no regression targets
        gpu_config,
        h_nodestatus.data(),
        h_bestvar.data(),
        h_xbestsplit.data(),
        h_treemap.data(),
        h_nodeclass.data(),
        h_nnode.data(),
        q_.data(),
        nout_.data(),
        nullptr,  // Classification: no regression OOB predictions
        nullptr,  // Classification: no regression node predictions
        h_importance.data(),
        config_.compute_local_importance ? h_qimp.data() : nullptr,
        config_.compute_local_importance ? h_local_importance.data() : nullptr,
        config_.compute_proximity_importance ? h_prox_importance.data() : nullptr,
        config_.compute_proximity_importance ? h_overall_prox_importance.data() : nullptr,
        // Leaf assignments (for on-demand proximity computation)
        config_.compute_leaf_assignments ? h_leaf_assignments.data() : nullptr,
        // OOB tracking for Breiman's proximity
        config_.compute_leaf_assignments ? h_nin_all.data() : nullptr,
        // Progress callback
        progress_callback_
    );
    
    if (err != 0) {
        throw std::runtime_error("GPU sparse training failed with error code: " + std::to_string(err));
    }
    
    // Copy results back to class members
    nodestatus_ = h_nodestatus;
    bestvar_ = h_bestvar;
    xbestsplit_ = h_xbestsplit;
    treemap_ = h_treemap;
    nodeclass_ = h_nodeclass;
    nnode_.assign(h_nnode.begin(), h_nnode.end());
    
    if (config_.compute_importance) {
        feature_importances_ = h_importance;
        if (config_.compute_local_importance) {
            // Copy qimp (per-sample original correct weights) from GPU
            qimp_ = h_qimp;
            // Copy raw accumulated qimpm values from GPU
            qimpm_ = h_local_importance;
            // NOTE: localimp() will be called in finalize_training() - don't call here
            // to avoid double application of the transformation
        }
    }
    
    if (config_.compute_proximity_importance) {
        proximity_importance_ = h_prox_importance;
        // NOTE: GPU sparse already normalized by nout in gpu_finalize_proximity_importance_sparse
        // DO NOT normalize again here - that would cause double-normalization!
    }
    
    // Store leaf assignments (for on-demand proximity computation)
    if (config_.compute_leaf_assignments && !h_leaf_assignments.empty()) {
        leaf_assignments_ = h_leaf_assignments;
        
        // Build oob_trees_ from h_nin_all for Breiman's proximity
        if (!h_nin_all.empty()) {
            oob_trees_.resize(static_cast<size_t>(config_.nsample));
            for (integer_t t = 0; t < config_.ntree; ++t) {
                for (integer_t n = 0; n < config_.nsample; ++n) {
                    // Cast to size_t to avoid integer overflow in index calculation
                    size_t idx = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(n);
                    if (h_nin_all[idx] == 0) {  // OOB sample
                        oob_trees_[n].push_back(t);
                    }
                }
            }
        }
    }
    
    // Compute OOB error and predictions, then finalize training
    compute_classification_error();
    compute_oob_predictions();
    finalize_training();
}

// ============================================================================
// GPU SPARSE REGRESSION - Uses parallel GPU sparse tree growing
// ============================================================================
void RandomForest::fit_sparse_gpu_regression(const SparseMatrixCSR& X, 
                                              const real_t* y, 
                                              const real_t* sample_weight) {
    // NOTE: Regression GPU sparse uses same local importance handling as classification
    // The localimp() transformation is applied after copying qimpm_ from GPU
    
    // Ensure CUDA context is ready
    rf::cuda::cuda_ensure_context_ready();
    
    // For regression, convert y to integer labels (scaled)
    const real_t y_scale = 1000.0f;
    std::vector<integer_t> y_int(config_.nsample);
    for (integer_t i = 0; i < config_.nsample; ++i) {
        y_int[i] = static_cast<integer_t>(y[i] * y_scale);
    }
    
    // Configure GPU sparse training
    rf::cuda::GpuSparseForestConfig gpu_config;
    gpu_config.nsample = config_.nsample;
    gpu_config.mdim = config_.mdim;
    gpu_config.nclass = 1;  // Regression
    gpu_config.ntree = config_.ntree;
    gpu_config.mtry = config_.mtry;
    gpu_config.maxnode = config_.maxnode;
    gpu_config.min_node_size = config_.minndsize;
    gpu_config.rank = config_.lowrank_rank;
    gpu_config.max_depth = 100;  // Safety limit
    gpu_config.compute_importance = config_.compute_importance;
    gpu_config.compute_local_importance = config_.compute_local_importance;
    gpu_config.compute_proximity = config_.compute_proximity;
    gpu_config.compute_proximity_importance = config_.compute_proximity_importance;
    gpu_config.compute_leaf_assignments = config_.compute_leaf_assignments;
    gpu_config.use_casewise = config_.use_casewise;
    gpu_config.use_qlora = config_.use_qlora;
    gpu_config.seed = config_.iseed;
    gpu_config.task_type = 1;  // Regression
    gpu_config.use_histogram = config_.use_histogram;
    gpu_config.n_bins = config_.n_bins;
    gpu_config.batch_size = config_.batch_size;  // Batched training support
    
    // Allocate output arrays
    // Cast to size_t to avoid integer overflow with large ntree * maxnode
    size_t tree_node_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.maxnode);
    size_t tree_2node_size = static_cast<size_t>(config_.ntree) * 2 * static_cast<size_t>(config_.maxnode);
    std::vector<integer_t> h_nodestatus(tree_node_size);
    std::vector<integer_t> h_bestvar(tree_node_size);
    std::vector<real_t> h_xbestsplit(tree_node_size);
    std::vector<integer_t> h_treemap(tree_2node_size);
    std::vector<integer_t> h_nodeclass(tree_node_size);
    std::vector<integer_t> h_nnode(static_cast<size_t>(config_.ntree));
    std::vector<real_t> h_importance(config_.mdim, 0.0f);
    std::vector<real_t> h_local_importance;
    std::vector<dp_t> h_proximity;
    std::vector<real_t> h_prox_importance;
    std::vector<real_t> h_overall_prox_importance(config_.mdim, 0.0f);
    
    // Allocate qimp for local importance (per-sample original correct weights)
    std::vector<real_t> h_qimp;
    if (config_.compute_local_importance) {
        // Cast to size_t to avoid integer overflow
        size_t local_imp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        h_local_importance.resize(local_imp_size, 0.0f);
        h_qimp.resize(static_cast<size_t>(config_.nsample), 0.0f);
    }
    if (config_.compute_proximity && !config_.use_qlora) {
        h_proximity.resize(static_cast<size_t>(config_.nsample) * config_.nsample, 0.0);
    }
    if (config_.compute_proximity_importance) {
        // Cast to size_t to avoid integer overflow
        size_t prox_imp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        h_prox_importance.resize(prox_imp_size, 0.0f);
    }
    
    // Allocate leaf assignments buffer (for on-demand proximity computation)
    std::vector<int16_t> h_leaf_assignments;
    if (config_.compute_leaf_assignments) {
        // Cast to size_t to avoid integer overflow with large ntree * nsample
        size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
        if (leaf_size > h_leaf_assignments.max_size()) {
            throw std::runtime_error(
                "Cannot allocate leaf assignments: size " + std::to_string(leaf_size) + 
                " exceeds max_size() " + std::to_string(h_leaf_assignments.max_size())
            );
        }
        h_leaf_assignments.resize(leaf_size, 0);
    }
    
    // Allocate nin_all buffer for oob_trees_ population (Breiman's proximity)
    std::vector<integer_t> h_nin_all;
    if (config_.compute_leaf_assignments) {
        // Cast to size_t to avoid integer overflow with large ntree * nsample
        size_t nin_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
        if (nin_size > h_nin_all.max_size()) {
            throw std::runtime_error(
                "Cannot allocate nin_all: size " + std::to_string(nin_size) + 
                " exceeds max_size() " + std::to_string(h_nin_all.max_size())
            );
        }
        h_nin_all.resize(nin_size, 0);
    }
    
    // Call GPU sparse training - now supports regression OOB predictions
    // NOTE: GPU sparse uses leaf_assignments for proximity, not low-rank factors
    integer_t err = rf::cuda::train_sparse_forest_gpu(
        X, y_int.data(), y,  // Regression: pass actual float targets
        gpu_config,
        h_nodestatus.data(),
        h_bestvar.data(),
        h_xbestsplit.data(),
        h_treemap.data(),
        h_nodeclass.data(),
        h_nnode.data(),
        nullptr,  // h_q - not used for regression
        nout_.data(),  // h_nout - needed for regression OOB normalization
        oob_predictions_.data(),  // Regression OOB predictions (weighted sum)
        node_predictions_regression_.data(),  // Terminal node predictions for predict() on new data
        h_importance.data(),
        config_.compute_local_importance ? h_qimp.data() : nullptr,
        config_.compute_local_importance ? h_local_importance.data() : nullptr,
        config_.compute_proximity_importance ? h_prox_importance.data() : nullptr,
        config_.compute_proximity_importance ? h_overall_prox_importance.data() : nullptr,
        // Leaf assignments (for on-demand proximity computation)
        config_.compute_leaf_assignments ? h_leaf_assignments.data() : nullptr,
        // OOB tracking for Breiman's proximity
        config_.compute_leaf_assignments ? h_nin_all.data() : nullptr,
        // Progress callback
        progress_callback_
    );
    
    if (err != 0) {
        throw std::runtime_error("GPU sparse regression training failed with error code: " + std::to_string(err));
    }
    
    // Copy results back to class members
    nodestatus_ = h_nodestatus;
    bestvar_ = h_bestvar;
    xbestsplit_ = h_xbestsplit;
    treemap_ = h_treemap;
    nodeclass_ = h_nodeclass;
    nnode_.assign(h_nnode.begin(), h_nnode.end());
    
    if (config_.compute_importance) {
        feature_importances_ = h_importance;
        if (config_.compute_local_importance) {
            // Copy qimp from GPU
            qimp_ = h_qimp;
            // Copy raw accumulated values from GPU
            qimpm_ = h_local_importance;
            // NOTE: localimp_regression() will be called in finalize_training() - don't call here
            // to avoid double application of the transformation
        }
    }
    
    if (config_.compute_proximity_importance) {
        proximity_importance_ = h_prox_importance;
        // NOTE: GPU sparse already normalized by nout in gpu_finalize_proximity_importance_sparse
        // DO NOT normalize again here - that would cause double-normalization!
    }
    
    // Store leaf assignments (for on-demand proximity computation)
    if (config_.compute_leaf_assignments && !h_leaf_assignments.empty()) {
        leaf_assignments_ = h_leaf_assignments;
        
        // Build oob_trees_ from h_nin_all for Breiman's proximity
        if (!h_nin_all.empty()) {
            oob_trees_.resize(static_cast<size_t>(config_.nsample));
            for (integer_t t = 0; t < config_.ntree; ++t) {
                for (integer_t n = 0; n < config_.nsample; ++n) {
                    // Cast to size_t to avoid integer overflow in index calculation
                    size_t idx = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(n);
                    if (h_nin_all[idx] == 0) {  // OOB sample
                        oob_trees_[n].push_back(t);
                    }
                }
            }
        }
    }
    
    // Finalize training (applies localimp_regression(), normalizes q_, etc.)
    finalize_training();
}

// ============================================================================
// GPU SPARSE UNSUPERVISED - Uses parallel GPU sparse tree growing
// ============================================================================
void RandomForest::fit_sparse_gpu_unsupervised(const SparseMatrixCSR& X, 
                                                const real_t* sample_weight) {
    // Ensure CUDA context is ready
    rf::cuda::cuda_ensure_context_ready();
    
    // Use the synthetic labels created by fit_unsupervised_sparse
    // y_train_unsupervised_ should already be populated with 0s (real) and 1s (synthetic)
    
    // Configure GPU sparse training
    rf::cuda::GpuSparseForestConfig gpu_config;
    gpu_config.nsample = config_.nsample;  // Total = real + synthetic
    gpu_config.mdim = config_.mdim;
    gpu_config.nclass = 2;  // Binary: real vs synthetic
    gpu_config.ntree = config_.ntree;
    gpu_config.mtry = config_.mtry;
    gpu_config.maxnode = config_.maxnode;
    gpu_config.min_node_size = config_.minndsize;
    gpu_config.rank = config_.lowrank_rank;
    gpu_config.max_depth = 100;  // Safety limit
    gpu_config.compute_importance = config_.compute_importance;
    gpu_config.compute_local_importance = config_.compute_local_importance;
    gpu_config.compute_proximity = config_.compute_proximity;
    gpu_config.compute_proximity_importance = config_.compute_proximity_importance;
    gpu_config.compute_leaf_assignments = config_.compute_leaf_assignments;
    gpu_config.use_casewise = config_.use_casewise;
    gpu_config.use_qlora = config_.use_qlora;
    gpu_config.seed = config_.iseed;
    gpu_config.task_type = 2;  // Unsupervised (classification-based)
    gpu_config.use_histogram = config_.use_histogram;
    gpu_config.n_bins = config_.n_bins;
    gpu_config.batch_size = config_.batch_size;  // Batched training support
    
    // Allocate output arrays
    // Cast to size_t to avoid integer overflow with large ntree * maxnode
    size_t tree_node_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.maxnode);
    size_t tree_2node_size = static_cast<size_t>(config_.ntree) * 2 * static_cast<size_t>(config_.maxnode);
    std::vector<integer_t> h_nodestatus(tree_node_size);
    std::vector<integer_t> h_bestvar(tree_node_size);
    std::vector<real_t> h_xbestsplit(tree_node_size);
    std::vector<integer_t> h_treemap(tree_2node_size);
    std::vector<integer_t> h_nodeclass(tree_node_size);
    std::vector<integer_t> h_nnode(static_cast<size_t>(config_.ntree));
    std::vector<real_t> h_importance(config_.mdim, 0.0f);
    std::vector<real_t> h_local_importance;
    std::vector<dp_t> h_proximity;
    std::vector<real_t> h_prox_importance;
    std::vector<real_t> h_overall_prox_importance(config_.mdim, 0.0f);
    
    // Allocate qimp for local importance (per-sample original correct weights)
    std::vector<real_t> h_qimp;
    if (config_.compute_local_importance) {
        // Cast to size_t to avoid integer overflow
        size_t local_imp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        h_local_importance.resize(local_imp_size, 0.0f);
        h_qimp.resize(static_cast<size_t>(config_.nsample), 0.0f);
    }
    if (config_.compute_proximity_importance) {
        // Cast to size_t to avoid integer overflow
        size_t prox_imp_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
        h_prox_importance.resize(prox_imp_size, 0.0f);
    }
    
    // Allocate leaf assignments buffer (for on-demand proximity computation)
    // NOTE: GPU sparse uses leaf_assignments, not low-rank factors for proximity
    std::vector<int16_t> h_leaf_assignments;
    if (config_.compute_leaf_assignments) {
        // Cast to size_t to avoid integer overflow with large ntree * nsample
        size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
        if (leaf_size > h_leaf_assignments.max_size()) {
            throw std::runtime_error(
                "Cannot allocate leaf assignments: size " + std::to_string(leaf_size) + 
                " exceeds max_size() " + std::to_string(h_leaf_assignments.max_size())
            );
        }
        h_leaf_assignments.resize(leaf_size, 0);
    }
    
    // Allocate nin_all buffer for oob_trees_ population (Breiman's proximity)
    std::vector<integer_t> h_nin_all;
    if (config_.compute_leaf_assignments) {
        // Cast to size_t to avoid integer overflow with large ntree * nsample
        size_t nin_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
        if (nin_size > h_nin_all.max_size()) {
            throw std::runtime_error(
                "Cannot allocate nin_all: size " + std::to_string(nin_size) + 
                " exceeds max_size() " + std::to_string(h_nin_all.max_size())
            );
        }
        h_nin_all.resize(nin_size, 0);
    }
    
    // Ensure q_ and nout_ are sized correctly for OOB vote collection
    // Cast to size_t to avoid integer overflow
    size_t q_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.nclass);
    q_.assign(q_size, 0.0f);
    nout_.assign(static_cast<size_t>(config_.nsample), 0);
    
    // Call GPU sparse training with unsupervised labels
    // NOTE: GPU sparse uses leaf_assignments for on-demand proximity
    integer_t err = rf::cuda::train_sparse_forest_gpu(
        X_train_sparse_, y_train_unsupervised_.data(), nullptr,  // Unsupervised: no regression targets
        gpu_config,
        h_nodestatus.data(),
        h_bestvar.data(),
        h_xbestsplit.data(),
        h_treemap.data(),
        h_nodeclass.data(),
        h_nnode.data(),
        q_.data(),
        nout_.data(),
        nullptr,  // Unsupervised: no regression OOB predictions
        nullptr,  // Unsupervised: no regression node predictions
        h_importance.data(),
        config_.compute_local_importance ? h_qimp.data() : nullptr,
        config_.compute_local_importance ? h_local_importance.data() : nullptr,
        config_.compute_proximity_importance ? h_prox_importance.data() : nullptr,
        config_.compute_proximity_importance ? h_overall_prox_importance.data() : nullptr,
        // Leaf assignments (for on-demand proximity computation)
        config_.compute_leaf_assignments ? h_leaf_assignments.data() : nullptr,
        // OOB tracking for Breiman's proximity
        config_.compute_leaf_assignments ? h_nin_all.data() : nullptr,
        // Progress callback
        progress_callback_
    );
    
    if (err != 0) {
        throw std::runtime_error("GPU sparse unsupervised training failed with error code: " + std::to_string(err));
    }
    
    // Copy results back to class members
    nodestatus_ = h_nodestatus;
    bestvar_ = h_bestvar;
    xbestsplit_ = h_xbestsplit;
    treemap_ = h_treemap;
    nodeclass_ = h_nodeclass;
    nnode_.assign(h_nnode.begin(), h_nnode.end());
    
    if (config_.compute_importance) {
        feature_importances_ = h_importance;
        if (config_.compute_local_importance) {
            // Copy qimp from GPU
            qimp_ = h_qimp;
            // Copy raw accumulated values from GPU
            qimpm_ = h_local_importance;
            // NOTE: localimp() will be called in finalize_training() - don't call here
            // to avoid double application of the transformation
        }
    }
    
    if (config_.compute_proximity_importance) {
        proximity_importance_ = h_prox_importance;
        // NOTE: GPU sparse already normalized by nout in gpu_finalize_proximity_importance_sparse
        // DO NOT normalize again here - that would cause double-normalization!
    }
    
    // Store leaf assignments (for on-demand proximity computation)
    if (config_.compute_leaf_assignments && !h_leaf_assignments.empty()) {
        leaf_assignments_ = h_leaf_assignments;
        
        // Build oob_trees_ from h_nin_all for Breiman's proximity
        if (!h_nin_all.empty()) {
            oob_trees_.resize(static_cast<size_t>(config_.nsample));
            for (integer_t t = 0; t < config_.ntree; ++t) {
                for (integer_t n = 0; n < config_.nsample; ++n) {
                    // Cast to size_t to avoid integer overflow in index calculation
                    size_t idx = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(n);
                    if (h_nin_all[idx] == 0) {  // OOB sample
                        oob_trees_[n].push_back(t);
                    }
                }
            }
        }
    }
    
    // Compute OOB error for unsupervised mode (classifying real vs synthetic)
    compute_classification_error();
    
    // Finalize training (normalizes OOB votes, applies localimp() for local importance)
    finalize_training();
    
    // NOTE: Outlier detection is ON-DEMAND only via compute_outlier_scores()
    // It uses leaf_assignments (memory-efficient) NOT full proximity matrix
    // Users call: model.compute_outlier_scores(mode='greedy') or mode='full'
    // Requires: compute_leaf_assignments=True when creating the model
}
#endif // CUDA_FOUND

// Task-specific setup methods
void RandomForest::setup_classification_task(const integer_t* y) {
    // For classification, determine number of classes
    std::set<integer_t> unique_classes(y, y + config_.nsample);
    config_.nclass = static_cast<integer_t>(unique_classes.size());
    
    // Store class labels directly (0-based)
    y_train_classification_1based_.resize(config_.nsample);
    for (integer_t i = 0; i < config_.nsample; ++i) {
        y_train_classification_1based_[i] = y[i];  // Keep 0-based
    }
    
    // Ensure mtry is reasonable for classification
    if (config_.mtry == 0) {
        config_.mtry = std::max(1, static_cast<integer_t>(std::sqrt(static_cast<real_t>(config_.mdim))));
    }
}

void RandomForest::setup_regression_task(const real_t* y) {
    // For regression, a single "class" is used but continuous values are predicted
    config_.nclass = 1;
    
    // Use regression-specific mtry ONLY if not set by user (mtry == 0)
    // R's default for regression is mdim/3, rounded down
    if (config_.mtry == 0) {
        config_.mtry = std::max(1, static_cast<integer_t>(config_.mdim / 3));
    }
    
    // Initialize regression-specific arrays
    oob_predictions_.resize(config_.nsample, 0.0f);
    // Cast to size_t to avoid integer overflow
    size_t terminal_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.maxnode);
    terminal_values_.resize(terminal_size, 0.0f);
}

void RandomForest::setup_unsupervised_task() {
    // For unsupervised, proximity-based clustering is used
    // Use nclass=2 (binary classification style) instead of nclass=1 to avoid issues
    // with tree growing code that expects nclass >= 2 for classification-style splits
    // The synthetic labels are all zeros, which will be treated as class 0
    config_.nclass = 2;  // Changed from 1 to 2 to match classification-style unsupervised mode
    // Don't override compute_proximity - respect user's setting
    // config_.compute_proximity = true;  // REMOVED: Let user control proximity computation
    
    // Set mtry based on unsupervised mode (only if not explicitly set by user, i.e., mtry == 0)
    if (config_.mtry == 0) {
        if (config_.unsupervised_mode == UnsupervisedMode::CLASSIFICATION_STYLE) {
            // Classification-style: use sqrt(mdim)
            config_.mtry = std::max(1, static_cast<integer_t>(std::sqrt(static_cast<real_t>(config_.mdim))));
        } else {
            // Regression-style: use mdim / 3
            config_.mtry = std::max(1, static_cast<integer_t>(config_.mdim / 3));
        }
    }
    
    // Initialize unsupervised-specific arrays
    outlier_scores_.resize(config_.nsample, 0.0f);
    cluster_labels_.resize(config_.nsample, 0);
}

// Unified predict method that dispatches based on task type
void RandomForest::predict(const real_t* X, integer_t nsamples, void* predictions) {
    switch (config_.task_type) {
        case TaskType::CLASSIFICATION:
            predict_classification(X, nsamples, static_cast<integer_t*>(predictions));
            break;
        case TaskType::REGRESSION:
            predict_regression(X, nsamples, static_cast<real_t*>(predictions));
            break;
        case TaskType::UNSUPERVISED:
            predict_unsupervised(X, nsamples, static_cast<integer_t*>(predictions));
            break;
    }
}

// Classification prediction - Use OOB for training data, tree traversal for new data
void RandomForest::predict_classification(const real_t* X, integer_t nsamples, integer_t* predictions) {
    // Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check if predicting on training data (same size and potentially same data)
    bool is_training_data = (nsamples == config_.nsample);
    
    if (is_training_data) {
        // Use stored OOB predictions (calculated during fit())
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            if (sample < oob_class_predictions_.size()) {
                predictions[sample] = oob_class_predictions_[sample];
            } else {
                predictions[sample] = 0; // Default to class 0
            }
        }
    } else {
        // Use tree traversal for new/held-out data
        std::vector<integer_t> votes(config_.nclass, 0);
        
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            // Reset votes for this sample
            std::fill(votes.begin(), votes.end(), 0);
            
            // Get sample features
            const real_t* sample_features = X + sample * config_.mdim;
            
            // Vote from each tree
            for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
                // Get pointers to this tree's storage
                // Cast to size_t to avoid integer overflow in pointer arithmetic
                size_t tree_offset_pred = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
                size_t tree_treemap_offset_pred = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
                size_t tree_catgoleft_offset_pred = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
                
                integer_t* nodestatus = nodestatus_.data() + tree_offset_pred;
                integer_t* bestvar = bestvar_.data() + tree_offset_pred;
                real_t* xbestsplit = xbestsplit_.data() + tree_offset_pred;
                integer_t* treemap = treemap_.data() + tree_treemap_offset_pred;
                integer_t* nodeclass = nodeclass_.data() + tree_offset_pred;
                integer_t* catgoleft = catgoleft_.data() + tree_catgoleft_offset_pred;
                
                // Traverse tree starting from root (node 0) - EXACT match to testreebag.f90
                integer_t kt = 0; // Current node index (0-based, matching Fortran kt=1 converted to 0-based)
                
                // Loop through all possible nodes (matching original algorithm exactly)
                for (integer_t k = 0; k < config_.maxnode; ++k) {
                    if (kt >= config_.maxnode) {
                        // Safety check - node index out of bounds
                        votes[0]++; // Default to class 0
                        break;
                    }
                    
                    if (nodestatus[kt] == -1) {
                        // Terminal node - get class prediction (matching testreebag.f90 line 29)
                        integer_t predicted_class = nodeclass[kt];
                        if (predicted_class >= 0 && predicted_class < config_.nclass) {
                            votes[predicted_class]++; // Use 0-based indexing directly
                        } else {
                            votes[0]++; // Default to class 0 for invalid predictions
                        }
                        break;
                    }
                    
                    // Split node - traverse based on split (matching testreebag.f90 lines 33-48)
                    integer_t m = bestvar[kt]; // Split variable
                    
                    if (m >= 0 && m < config_.mdim) {
                        real_t feature_value = sample_features[m];
                        
                        // Check if variable is quantitative (cat == 1) or categorical
                        if (cat_[m] == 1) {
                            // Quantitative variable (matching testreebag.f90 lines 35-40)
                            if (feature_value <= xbestsplit[kt]) {
                                kt = treemap[kt * 2]; // Left child (0-based indexing)
                            } else {
                                kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                            }
                        } else {
                            // Categorical variable (matching testreebag.f90 lines 42-47)
                            integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                            if (jcat >= 0 && jcat < cat_[m]) {
                                if (catgoleft[(kt * config_.maxcat) + jcat] == 1) {
                                    kt = treemap[kt * 2]; // Left child (0-based indexing)
                                } else {
                                    kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                                }
                            } else {
                                // Invalid category - default to class 0
                                votes[0]++;
                                break;
                            }
                        }
                    } else {
                        // Invalid split variable - default to class 0
                        votes[0]++;
                        break;
                    }
                }
            }
            
            // Find class with most votes (matching predict.f lines 15-28)
            integer_t predicted_class = 0;
            integer_t max_votes = votes[0];
            for (integer_t c = 1; c < config_.nclass; ++c) {
                if (votes[c] > max_votes) {
                    max_votes = votes[c];
                    predicted_class = c;
                }
            }
            
            predictions[sample] = predicted_class;
        }
    }
}

// Regression prediction
void RandomForest::predict_regression(const real_t* X, integer_t nsamples, real_t* predictions) {
    // Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Implement regression prediction matching original Breiman-Cutler algorithm
    // For regression, prediction is the average of terminal node values from all trees
    
    for (integer_t sample = 0; sample < nsamples; ++sample) {
        real_t sum_predictions = 0.0;
        integer_t tree_count = 0;
        
        // Get sample features
        const real_t* sample_features = X + sample * config_.mdim;
        
        // Sum predictions from each tree
        for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
            // Get pointers to this tree's storage
            // Cast to size_t to avoid integer overflow in pointer arithmetic
            size_t tree_offset_reg = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
            size_t tree_treemap_offset_reg = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
            size_t tree_catgoleft_offset_reg = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
            
            integer_t* nodestatus = nodestatus_.data() + tree_offset_reg;
            integer_t* bestvar = bestvar_.data() + tree_offset_reg;
            real_t* xbestsplit = xbestsplit_.data() + tree_offset_reg;
            integer_t* treemap = treemap_.data() + tree_treemap_offset_reg;
            real_t* nodepred = node_predictions_regression_.data() + tree_offset_reg;
            integer_t* catgoleft = catgoleft_.data() + tree_catgoleft_offset_reg;
            
            // Traverse tree starting from root (node 0) - matching testreebag.f
            integer_t current_node = 0;
            
            // Loop through all possible nodes (matching original algorithm)
            for (integer_t k = 0; k < config_.maxnode; ++k) {
                if (nodestatus[current_node] == -1) {
                    // Terminal node - get prediction value (nodepred for regression)
                    sum_predictions += nodepred[current_node];
                    tree_count++;
                    break;
                }
                
                // Split node - traverse based on split (matching testreebag.f lines 31-46)
                integer_t split_var = bestvar[current_node];
                real_t split_value = xbestsplit[current_node];
                
                if (split_var >= 0 && split_var < config_.mdim) {
                    real_t feature_value = sample_features[split_var];
                    
                    // Check if variable is quantitative (cat == 1) or categorical
                    if (cat_[split_var] == 1) {
                        // Quantitative variable (matching testreebag.f lines 33-38)
                        if (feature_value <= split_value) {
                            current_node = treemap[current_node * 2]; // Left child
                        } else {
                            current_node = treemap[current_node * 2 + 1]; // Right child
                        }
                    } else {
                        // Categorical variable (matching testreebag.f lines 40-45)
                        integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                        if (jcat >= 0 && jcat < cat_[split_var]) {
                            if (catgoleft[(current_node * config_.maxcat) + jcat] == 1) {
                                current_node = treemap[current_node * 2]; // Left child
                            } else {
                                current_node = treemap[current_node * 2 + 1]; // Right child
                            }
                        } else {
                            // Invalid category - use root terminal value
                            sum_predictions += nodepred[0];
                            tree_count++;
                            break;
                        }
                    }
                } else {
                    // Invalid split variable - use root terminal value
                    sum_predictions += nodepred[0];
                    tree_count++;
                    break;
                }
            }
        }
        
        // Average prediction across all trees
        if (tree_count > 0) {
            predictions[sample] = sum_predictions / static_cast<real_t>(tree_count);
        } else {
            predictions[sample] = 0.0;
        }
    }
}

// Unsupervised prediction (clustering)
void RandomForest::predict_unsupervised(const real_t* X, integer_t nsamples, integer_t* cluster_labels) {
    // Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Implement unsupervised prediction using proximity-based clustering
    // This is a simplified implementation - in practice, you might want to use
    // more sophisticated clustering algorithms on the proximity matrix
    
    // For now, assign cluster labels based on proximity to training samples
    // This is a basic implementation that could be enhanced
    
    for (integer_t sample = 0; sample < nsamples; ++sample) {
        // Find the training sample with highest proximity
        integer_t best_match = 0;
        real_t max_proximity = 0.0;
        
        // Get sample features
        const real_t* sample_features = X + sample * config_.mdim;
        
        // Compute proximity to each training sample by traversing trees
        for (integer_t train_sample = 0; train_sample < config_.nsample; ++train_sample) {
            real_t proximity_sum = 0.0;
            
            // For each tree, check if both samples end up in same terminal node
            for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
                // Get pointers to this tree's storage
                // Cast to size_t to avoid integer overflow in pointer arithmetic
                size_t tree_offset_clust = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
                size_t tree_treemap_offset_clust = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
                size_t tree_catgoleft_offset_clust = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
                
                integer_t* nodestatus = nodestatus_.data() + tree_offset_clust;
                integer_t* bestvar = bestvar_.data() + tree_offset_clust;
                real_t* xbestsplit = xbestsplit_.data() + tree_offset_clust;
                integer_t* treemap = treemap_.data() + tree_treemap_offset_clust;
                integer_t* catgoleft = catgoleft_.data() + tree_catgoleft_offset_clust;
                
                // Find terminal node for test sample
                integer_t test_node = find_terminal_node(sample_features, nodestatus, bestvar, 
                                                       xbestsplit, treemap, catgoleft);
                
                // Find terminal node for training sample
                const real_t* train_features = X_train_.data() + train_sample * config_.mdim;
                integer_t train_node = find_terminal_node(train_features, nodestatus, bestvar,
                                                        xbestsplit, treemap, catgoleft);
                
                // If both samples end up in same terminal node, add to proximity
                if (test_node == train_node) {
                    proximity_sum += 1.0;
                }
            }
            
            real_t avg_proximity = proximity_sum / static_cast<real_t>(config_.ntree);
            if (avg_proximity > max_proximity) {
                max_proximity = avg_proximity;
                best_match = train_sample;
            }
        }
        
        // Assign cluster label based on best match
        // For simplicity, use the cluster label of the best matching training sample
        if (cluster_labels_.size() > best_match) {
            cluster_labels[sample] = cluster_labels_[best_match];
        } else {
            cluster_labels[sample] = 0; // Default cluster
        }
    }
}

// Helper function to find terminal node for a sample (SPARSE version)
integer_t RandomForest::find_terminal_node_sparse(integer_t sample_idx,
                                          integer_t* nodestatus, integer_t* bestvar,
                                          real_t* xbestsplit, integer_t* treemap,
                                          integer_t* catgoleft) {
    integer_t current_node = 0;
    
    // Loop through all possible nodes (matching original algorithm)
    for (integer_t k = 0; k < config_.maxnode; ++k) {
        if (nodestatus[current_node] == -1) {
            // Terminal node found
            return current_node;
        }
        
        // Split node - traverse based on split
        integer_t split_var = bestvar[current_node];
        real_t split_value = xbestsplit[current_node];
        
        if (split_var >= 0 && split_var < config_.mdim) {
            // Get feature value from sparse matrix
            real_t feature_value = X_train_sparse_.get(sample_idx, split_var);
            
            // Check if variable is quantitative (cat == 1) or categorical
            if (cat_[split_var] == 1) {
                // Quantitative variable
                if (feature_value <= split_value) {
                    current_node = treemap[current_node * 2]; // Left child
                } else {
                    current_node = treemap[current_node * 2 + 1]; // Right child
                }
            } else {
                // Categorical variable
                integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                if (jcat >= 0 && jcat < cat_[split_var]) {
                    if (catgoleft[(current_node * config_.maxcat) + jcat] == 1) {
                        current_node = treemap[current_node * 2]; // Left child
                    } else {
                        current_node = treemap[current_node * 2 + 1]; // Right child
                    }
                } else {
                    // Invalid category - return root node
                    return 0;
                }
            }
        } else {
            // Invalid split variable - return root node
            return 0;
        }
    }
    
    // Should never reach here if tree is properly formed
    return current_node;
}

// Helper function to find terminal node for a sample
integer_t RandomForest::find_terminal_node(const real_t* sample_features,
                                          integer_t* nodestatus, integer_t* bestvar,
                                          real_t* xbestsplit, integer_t* treemap,
                                          integer_t* catgoleft) {
    integer_t current_node = 0;
    
    // Loop through all possible nodes (matching original algorithm)
    for (integer_t k = 0; k < config_.maxnode; ++k) {
        if (nodestatus[current_node] == -1) {
            // Terminal node found
            return current_node;
        }
        
        // Split node - traverse based on split
        integer_t split_var = bestvar[current_node];
        real_t split_value = xbestsplit[current_node];
        
        if (split_var >= 0 && split_var < config_.mdim) {
            real_t feature_value = sample_features[split_var];
            
            // Check if variable is quantitative (cat == 1) or categorical
            if (cat_[split_var] == 1) {
                // Quantitative variable
                if (feature_value <= split_value) {
                    current_node = treemap[current_node * 2]; // Left child
                } else {
                    current_node = treemap[current_node * 2 + 1]; // Right child
                }
            } else {
                // Categorical variable
                integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                if (jcat >= 0 && jcat < cat_[split_var]) {
                    if (catgoleft[(current_node * config_.maxcat) + jcat] == 1) {
                        current_node = treemap[current_node * 2]; // Left child
                    } else {
                        current_node = treemap[current_node * 2 + 1]; // Right child
                    }
                } else {
                    // Invalid category - return root node
                    return 0;
                }
            }
        } else {
            // Invalid split variable - return root node
            return 0;
        }
    }
    
    // Should not reach here, but return root node as fallback
    return 0;
}

void RandomForest::compute_oob_predictions() {
    // Compute OOB class predictions and probabilities from the q_ array (raw OOB votes)
    // This must be called BEFORE finalize_training() which normalizes q_
    oob_class_predictions_.resize(config_.nsample);
    // Cast to size_t to avoid integer overflow
    size_t oob_prob_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.nclass);
    oob_probabilities_.resize(oob_prob_size);
    
    for (integer_t sample = 0; sample < config_.nsample; ++sample) {
        if (nout_[sample] > 0) {  // Sample was out-of-bag
            // Find the class with maximum raw OOB votes for this sample
            // In case of ties, prefer the class with the higher index (more conservative)
            integer_t best_class = 0;
            // Cast to size_t to avoid integer overflow in index calculation
            size_t q_base_idx = static_cast<size_t>(sample) * static_cast<size_t>(config_.nclass);
            real_t max_votes = q_[q_base_idx];
            real_t total_votes = 0.0;
            
            // Calculate total votes and find best class using raw vote counts
            for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                size_t idx = q_base_idx + static_cast<size_t>(class_idx);
                if (idx < q_.size()) {
                    real_t votes = q_[idx];  // Raw vote count
                    total_votes += votes;
                    // Use >= instead of > to break ties in favor of higher class index
                    // This ensures class 2 gets preference over class 0/1 when votes are equal
                    if (votes >= max_votes) {
                        max_votes = votes;
                        best_class = class_idx;
                    }
                }
            }
            
            oob_class_predictions_[sample] = best_class;
            
            // Calculate probabilities from raw votes
            if (total_votes > 0.0) {
                for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                    size_t idx = q_base_idx + static_cast<size_t>(class_idx);
                    size_t prob_idx = static_cast<size_t>(sample) * static_cast<size_t>(config_.nclass) + static_cast<size_t>(class_idx);
                    if (idx < q_.size()) {
                        oob_probabilities_[prob_idx] = q_[idx] / total_votes;
                    } else {
                        oob_probabilities_[prob_idx] = 1.0 / config_.nclass;
                    }
                }
            } else {
                // No OOB votes - use uniform distribution
                for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                    size_t prob_idx = static_cast<size_t>(sample) * static_cast<size_t>(config_.nclass) + static_cast<size_t>(class_idx);
                    oob_probabilities_[prob_idx] = 1.0 / config_.nclass;
                }
            }
        } else {
            // Sample was not out-of-bag - no prediction
            oob_class_predictions_[sample] = 0;
            for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                // Cast to size_t to avoid integer overflow in index calculation
                size_t prob_idx = static_cast<size_t>(sample) * static_cast<size_t>(config_.nclass) + static_cast<size_t>(class_idx);
                oob_probabilities_[prob_idx] = 1.0 / config_.nclass;
            }
        }
    }
}

void RandomForest::predict_proba(const real_t* X, integer_t nsamples, real_t* probabilities) {
    // Before predict, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check if predicting on training data (same size and potentially same data)
    bool is_training_data = (nsamples == config_.nsample);
    
    if (is_training_data) {
        // Use stored OOB probabilities (calculated during fit())
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            for (integer_t class_idx = 0; class_idx < config_.nclass; ++class_idx) {
                // Cast to size_t to avoid integer overflow in index calculation
                size_t prob_idx = static_cast<size_t>(sample) * static_cast<size_t>(config_.nclass) + static_cast<size_t>(class_idx);
                if (prob_idx < oob_probabilities_.size()) {
                    probabilities[prob_idx] = oob_probabilities_[prob_idx];
                } else {
                    probabilities[prob_idx] = 1.0 / config_.nclass;
                }
            }
        }
    } else {
        // Use tree traversal for new/held-out data
        std::vector<integer_t> votes(config_.nclass, 0);
        
        for (integer_t sample = 0; sample < nsamples; ++sample) {
            // Reset votes for this sample
            std::fill(votes.begin(), votes.end(), 0);
            
            // Get sample features
            const real_t* sample_features = X + sample * config_.mdim;
            
            // Vote from each tree
            for (integer_t tree_id = 0; tree_id < config_.ntree; ++tree_id) {
                // Get pointers to this tree's storage
                // Cast to size_t to avoid integer overflow in pointer arithmetic
                size_t tree_offset_pred = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxnode);
                size_t tree_treemap_offset_pred = static_cast<size_t>(tree_id) * 2 * static_cast<size_t>(config_.maxnode);
                size_t tree_catgoleft_offset_pred = static_cast<size_t>(tree_id) * static_cast<size_t>(config_.maxcat) * static_cast<size_t>(config_.maxnode);
                
                integer_t* nodestatus = nodestatus_.data() + tree_offset_pred;
                integer_t* bestvar = bestvar_.data() + tree_offset_pred;
                real_t* xbestsplit = xbestsplit_.data() + tree_offset_pred;
                integer_t* treemap = treemap_.data() + tree_treemap_offset_pred;
                integer_t* nodeclass = nodeclass_.data() + tree_offset_pred;
                integer_t* catgoleft = catgoleft_.data() + tree_catgoleft_offset_pred;
                
                // Traverse tree starting from root (node 0) - EXACT match to testreebag.f90
                integer_t kt = 0; // Current node index (0-based, matching Fortran kt=1 converted to 0-based)
                
                // Loop through all possible nodes (matching original algorithm exactly)
                for (integer_t k = 0; k < config_.maxnode; ++k) {
                    if (kt >= config_.maxnode) {
                        // Safety check - node index out of bounds
                        votes[0]++; // Default to class 0
                        break;
                    }
                    
                    if (nodestatus[kt] == -1) {
                        // Terminal node - get class prediction (matching testreebag.f90 line 29)
                        integer_t predicted_class = nodeclass[kt];
                        if (predicted_class >= 0 && predicted_class < config_.nclass) {
                            votes[predicted_class]++; // Use 0-based indexing directly
                        } else {
                            votes[0]++; // Default to class 0 for invalid predictions
                        }
                        break;
                    }
                    
                    // Split node - traverse based on split (matching testreebag.f90 lines 33-48)
                    integer_t m = bestvar[kt]; // Split variable
                    
                    if (m >= 0 && m < config_.mdim) {
                        real_t feature_value = sample_features[m];
                        
                        // Check if variable is quantitative (cat == 1) or categorical
                        if (cat_[m] == 1) {
                            // Quantitative variable (matching testreebag.f90 lines 35-40)
                            if (feature_value <= xbestsplit[kt]) {
                                kt = treemap[kt * 2]; // Left child (0-based indexing)
                            } else {
                                kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                            }
                        } else {
                            // Categorical variable (matching testreebag.f90 lines 42-47)
                            integer_t jcat = static_cast<integer_t>(feature_value + 0.5); // Round to nearest integer
                            if (jcat >= 0 && jcat < cat_[m]) {
                                if (catgoleft[(kt * config_.maxcat) + jcat] == 1) {
                                    kt = treemap[kt * 2]; // Left child (0-based indexing)
                                } else {
                                    kt = treemap[kt * 2 + 1]; // Right child (0-based indexing)
                                }
                            } else {
                                // Invalid category - default to class 0
                                votes[0]++;
                                break;
                            }
                        }
                    } else {
                        // Invalid split variable - default to class 0
                        votes[0]++;
                        break;
                    }
                }
            }
            
            // Convert votes to probabilities (matching predict.f logic)
            real_t total_votes = static_cast<real_t>(config_.ntree);
            for (integer_t c = 0; c < config_.nclass; ++c) {
                // Cast to size_t to avoid integer overflow in index calculation
                size_t prob_idx = static_cast<size_t>(sample) * static_cast<size_t>(config_.nclass) + static_cast<size_t>(c);
                probabilities[prob_idx] = static_cast<real_t>(votes[c]) / total_votes;
            }
        }
    }
}

void RandomForest::compute_classification_error() {
    // Compute OOB error for classification using raw vote counts in q_ array
    // This must be called BEFORE finalize_training() which normalizes q_
    // Use the same logic as compute_oob_predictions() to ensure consistency
    
    // Validate object state before accessing member variables
    if (config_.nsample <= 0 || config_.nclass <= 0) {
        throw std::runtime_error("Invalid config in compute_classification_error");
    }
    if (nout_.size() < static_cast<size_t>(config_.nsample) || 
        q_.size() < static_cast<size_t>(config_.nsample * config_.nclass)) {
        throw std::runtime_error("Invalid member arrays in compute_classification_error");
    }
    
    // Select correct label array based on task type
    // For unsupervised: use y_train_unsupervised_ (real=1, synthetic=0)
    // For classification: use y_train_classification_
    const integer_t* true_labels = nullptr;
    if (config_.task_type == TaskType::UNSUPERVISED) {
        if (y_train_unsupervised_.empty() || y_train_unsupervised_.size() < static_cast<size_t>(config_.nsample)) {
            throw std::runtime_error("Invalid y_train_unsupervised_ in compute_classification_error for UNSUPERVISED mode");
        }
        true_labels = y_train_unsupervised_.data();
    } else {
        if (y_train_classification_.empty() || y_train_classification_.size() < static_cast<size_t>(config_.nsample)) {
            throw std::runtime_error("Invalid y_train_classification_ in compute_classification_error");
        }
        true_labels = y_train_classification_.data();
    }
    
    integer_t correct = 0;
    integer_t total_oob = 0;
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        if (nout_[i] > 0) {  // Sample was out-of-bag
            total_oob++;
            
            // Find predicted class with maximum raw votes (same logic as compute_oob_predictions)
            // In case of ties, prefer the class with the higher index (more conservative)
            integer_t predicted_class = 0;
            real_t max_votes = q_[i * config_.nclass];
            for (integer_t j = 1; j < config_.nclass; ++j) {
                integer_t idx = i * config_.nclass + j;
                if (idx >= 0 && idx < q_.size()) {
                    real_t votes = q_[idx];
                    // Use >= instead of > to break ties in favor of higher class index
                    if (votes >= max_votes) {
                        max_votes = votes;
                        predicted_class = j;
                    }
                }
            }
            
            // Check if prediction is correct
            if (predicted_class == true_labels[i]) {
                correct++;
            }
        }
    }
    
    if (total_oob > 0) {
        oob_error_ = 1.0f - static_cast<real_t>(correct) / static_cast<real_t>(total_oob);
    } else {
        oob_error_ = 0.0f;  // No OOB samples
    }
}

void RandomForest::compute_regression_mse() {
    // Compute OOB MSE for regression
    // oob_predictions_ contains the SUM of predictions across all trees
    // Division by nout_[i] (number of trees that had this sample OOB) is required to get the average
    real_t sum_squared_error = 0.0f;
    integer_t total_oob = 0;
    
    // Debug: Check sums
    real_t sum_oob_preds = 0.0f;
    integer_t sum_nout = 0;
    real_t sum_y = 0.0f;
    for (integer_t i = 0; i < config_.nsample; ++i) {
        sum_oob_preds += oob_predictions_[i];
        sum_nout += nout_[i];
        sum_y += y_train_regression_[i];
    }
    // std::cout << "[MSE DEBUG] is_sparse=" << is_sparse_ 
    //           << " sum_oob_preds=" << sum_oob_preds 
    //           << " sum_nout=" << sum_nout 
    //           << " sum_y=" << sum_y << std::endl;
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        if (nout_[i] > 0) {  // Sample was out-of-bag
            total_oob++;
            // Normalize: divide sum by number of trees to get average prediction
            real_t prediction = oob_predictions_[i] / static_cast<real_t>(nout_[i]);
            real_t error = y_train_regression_[i] - prediction;
            sum_squared_error += error * error;
            
            // Debug first 3 samples
            // if (total_oob <= 3) {
            //     std::cout << "[MSE DEBUG] sample " << i << ": oob_pred_sum=" << oob_predictions_[i]
            //               << " nout=" << nout_[i] << " pred=" << prediction
            //               << " y=" << y_train_regression_[i] << " err=" << error << std::endl;
            // }
            
            // Update oob_predictions_ to store the final average (for get_oob_predictions())
            oob_predictions_[i] = prediction;
        }
    }
    
    // std::cout << "[MSE DEBUG] total_oob=" << total_oob << " sum_sq_err=" << sum_squared_error << std::endl;
    
    if (total_oob > 0) {
        oob_mse_ = sum_squared_error / static_cast<real_t>(total_oob);
        oob_error_ = oob_mse_;  // Also set oob_error_ for consistency with get_oob_error()
    }
}

void RandomForest::compute_outlier_scores_basic() {
    // Legacy: Compute outlier scores based on full proximity matrix
    // Use compute_outlier_scores() for the full Breiman-Cutler implementation
    if (proximity_matrix_.empty()) {
        return;
    }
    
    // Ensure GPU operations are complete before accessing proximity matrix
    // GPU synchronization is handled by GPU code itself
    // Avoid direct CUDA calls in CPU compilation path
    
    outlier_scores_.resize(config_.nsample, 0.0f);
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        real_t prox_sum = 0.0f;
        // Cast to size_t to avoid integer overflow in index calculation
        size_t prox_base_idx = static_cast<size_t>(i) * static_cast<size_t>(config_.nsample);
        for (integer_t j = 0; j < config_.nsample; ++j) {
            size_t prox_idx = prox_base_idx + static_cast<size_t>(j);
            real_t prox_val = proximity_matrix_[prox_idx];
            prox_sum += prox_val * prox_val;
        }
        if (prox_sum > 0) {
            outlier_scores_[i] = 1.0f / prox_sum;
        }
    }
    
    // Normalize by median
    std::vector<real_t> sorted_scores = outlier_scores_;
    std::sort(sorted_scores.begin(), sorted_scores.end());
    real_t median = sorted_scores[config_.nsample / 2];
    if (median > 0) {
        for (integer_t i = 0; i < config_.nsample; ++i) {
            outlier_scores_[i] /= median;
        }
    }
}

// ============================================================================
// BREIMAN-CUTLER OUTLIER DETECTION
// ============================================================================
// Original formula from Breiman-Cutler Random Forests documentation:
//   raw_outlier[n] = N_class / sum(prox[n,m]^2 for m in same class, m != n)
//   normalized[n] = (raw - median_class) / MAD_class
// Scores > 10 typically indicate outliers
// ============================================================================

std::vector<real_t> RandomForest::compute_outlier_scores(
    const std::string& mode,
    integer_t n_anchors) {
    
    // Require leaf assignments for efficient computation
    if (leaf_assignments_.empty()) {
        throw std::runtime_error("Outlier detection requires compute_leaf_assignments=True");
    }
    
    // Build inverted leaf index if not already built
    if (!inverted_leaf_index_built_) {
        const_cast<RandomForest*>(this)->build_inverted_leaf_index();
    }
    
    std::vector<real_t> raw_outlier(config_.nsample, 0.0f);
    
    if (mode == "greedy") {
        // GREEDY MODE: Use anchor subset for fast approximate computation
        if (outlier_anchors_.empty() || static_cast<integer_t>(outlier_anchors_.size()) != n_anchors) {
            outlier_anchors_ = select_outlier_anchors(n_anchors, "random");
        }
        
        #pragma omp parallel for schedule(dynamic)
        for (integer_t n = 0; n < config_.nsample; ++n) {
            real_t prox_sum_sq = 0.0f;
            
            for (integer_t anchor : outlier_anchors_) {
                if (anchor == n) continue;  // Skip self
                
                // Compute proximity between n and anchor using leaf assignments
                integer_t shared = 0;
                for (integer_t t = 0; t < config_.ntree; ++t) {
                    size_t idx_n = static_cast<size_t>(t) * config_.nsample + n;
                    size_t idx_a = static_cast<size_t>(t) * config_.nsample + anchor;
                    if (leaf_assignments_[idx_n] == leaf_assignments_[idx_a]) {
                        shared++;
                    }
                }
                real_t prox = static_cast<real_t>(shared) / config_.ntree;
                prox_sum_sq += prox * prox;
            }
            
            // Raw outlier measure: N / sum(prox^2)
            // Large value = isolated (low proximity to anchors)
            if (prox_sum_sq > 1e-10f) {
                raw_outlier[n] = static_cast<real_t>(n_anchors) / prox_sum_sq;
            } else {
                raw_outlier[n] = static_cast<real_t>(n_anchors) * 1e10f;  // Very isolated
            }
        }
    } else {
        // FULL MODE: Use inverted leaf index for exact computation
        #pragma omp parallel for schedule(dynamic)
        for (integer_t n = 0; n < config_.nsample; ++n) {
            // Count proximity to all other samples using inverted index
            std::unordered_map<integer_t, integer_t> prox_counts;
            
            for (integer_t t = 0; t < config_.ntree; ++t) {
                size_t idx = static_cast<size_t>(t) * config_.nsample + n;
                int16_t leaf = leaf_assignments_[idx];
                
                // Get all samples in same leaf
                const auto& samples_in_leaf = inverted_leaf_index_[t].at(leaf);
                for (integer_t other : samples_in_leaf) {
                    if (other != n) {
                        prox_counts[other]++;
                    }
                }
            }
            
            // Compute sum of squared proximities
            real_t prox_sum_sq = 0.0f;
            for (const auto& pair : prox_counts) {
                real_t prox = static_cast<real_t>(pair.second) / config_.ntree;
                prox_sum_sq += prox * prox;
            }
            
            // Raw outlier measure
            if (prox_sum_sq > 1e-10f) {
                raw_outlier[n] = static_cast<real_t>(config_.nsample - 1) / prox_sum_sq;
            } else {
                raw_outlier[n] = static_cast<real_t>(config_.nsample) * 1e10f;
            }
        }
    }
    
    // Normalize using MAD (Median Absolute Deviation)
    // For classification/unsupervised: normalize per class
    // For regression: treat all as one class
    
    outlier_scores_.resize(config_.nsample);
    
    if (config_.task_type == TaskType::CLASSIFICATION && !y_train_classification_.empty()) {
        // Per-class normalization
        std::map<integer_t, std::vector<integer_t>> class_samples;
        for (integer_t n = 0; n < config_.nsample; ++n) {
            class_samples[y_train_classification_[n]].push_back(n);
        }
        
        for (const auto& pair : class_samples) {
            const std::vector<integer_t>& samples = pair.second;
            if (samples.empty()) continue;
            
            // Get raw scores for this class
            std::vector<real_t> class_raw;
            class_raw.reserve(samples.size());
            for (integer_t idx : samples) {
                class_raw.push_back(raw_outlier[idx]);
            }
            
            // Compute median
            std::vector<real_t> sorted_raw = class_raw;
            std::sort(sorted_raw.begin(), sorted_raw.end());
            real_t median = sorted_raw[sorted_raw.size() / 2];
            
            // Compute MAD (Median Absolute Deviation)
            std::vector<real_t> abs_devs;
            abs_devs.reserve(class_raw.size());
            for (real_t val : class_raw) {
                abs_devs.push_back(std::abs(val - median));
            }
            std::sort(abs_devs.begin(), abs_devs.end());
            real_t mad = abs_devs[abs_devs.size() / 2];
            if (mad < 1e-10f) mad = 1.0f;  // Avoid division by zero
            
            // Normalize scores for this class
            for (size_t i = 0; i < samples.size(); ++i) {
                integer_t idx = samples[i];
                outlier_scores_[idx] = (raw_outlier[idx] - median) / mad;
            }
        }
    } else {
        // Single-class normalization (regression or unsupervised)
        std::vector<real_t> sorted_raw = raw_outlier;
        std::sort(sorted_raw.begin(), sorted_raw.end());
        real_t median = sorted_raw[config_.nsample / 2];
        
        // Compute MAD
        std::vector<real_t> abs_devs(config_.nsample);
        for (integer_t n = 0; n < config_.nsample; ++n) {
            abs_devs[n] = std::abs(raw_outlier[n] - median);
        }
        std::sort(abs_devs.begin(), abs_devs.end());
        real_t mad = abs_devs[config_.nsample / 2];
        if (mad < 1e-10f) mad = 1.0f;
        
        // Normalize
        for (integer_t n = 0; n < config_.nsample; ++n) {
            outlier_scores_[n] = (raw_outlier[n] - median) / mad;
        }
    }
    
    return outlier_scores_;
}

std::vector<std::pair<integer_t, real_t>> RandomForest::compute_outliers(
    integer_t k,
    const std::string& mode,
    integer_t n_anchors) {
    
    // Compute all outlier scores
    std::vector<real_t> scores = compute_outlier_scores(mode, n_anchors);
    
    // Create (index, score) pairs
    std::vector<std::pair<integer_t, real_t>> indexed_scores;
    indexed_scores.reserve(config_.nsample);
    for (integer_t i = 0; i < config_.nsample; ++i) {
        indexed_scores.push_back({i, scores[i]});
    }
    
    // Sort by score descending
    std::sort(indexed_scores.begin(), indexed_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top-k
    k = std::min(k, config_.nsample);
    indexed_scores.resize(k);
    
    return indexed_scores;
}

std::vector<integer_t> RandomForest::select_outlier_anchors(
    integer_t n_anchors,
    const std::string& method) {
    
    if (n_anchors >= config_.nsample) {
        // If requesting more anchors than samples, return all
        std::vector<integer_t> all_samples(config_.nsample);
        std::iota(all_samples.begin(), all_samples.end(), 0);
        return all_samples;
    }
    
    std::vector<integer_t> anchors;
    anchors.reserve(n_anchors);
    
    if (method == "diverse") {
        // DIVERSE: Iteratively select samples with low proximity to already-selected
        // This gives better coverage of the feature space
        
        // Start with random sample
        std::mt19937 rng(config_.iseed);
        std::uniform_int_distribution<integer_t> dist(0, config_.nsample - 1);
        anchors.push_back(dist(rng));
        
        // Track sum of proximities to anchors for each sample
        std::vector<real_t> min_prox_to_anchors(config_.nsample, 1.0f);
        
        while (static_cast<integer_t>(anchors.size()) < n_anchors) {
            integer_t last_anchor = anchors.back();
            
            // Update min proximity for each sample
            #pragma omp parallel for
            for (integer_t n = 0; n < config_.nsample; ++n) {
                // Compute proximity to last anchor
                integer_t shared = 0;
                for (integer_t t = 0; t < config_.ntree; ++t) {
                    size_t idx_n = static_cast<size_t>(t) * config_.nsample + n;
                    size_t idx_a = static_cast<size_t>(t) * config_.nsample + last_anchor;
                    if (leaf_assignments_[idx_n] == leaf_assignments_[idx_a]) {
                        shared++;
                    }
                }
                real_t prox = static_cast<real_t>(shared) / config_.ntree;
                min_prox_to_anchors[n] = std::min(min_prox_to_anchors[n], prox);
            }
            
            // Find sample with lowest max proximity to any anchor (most diverse)
            integer_t best_idx = -1;
            real_t best_min_prox = 2.0f;  // Larger than max possible
            for (integer_t n = 0; n < config_.nsample; ++n) {
                // Skip already-selected anchors
                bool already_anchor = false;
                for (integer_t a : anchors) {
                    if (a == n) { already_anchor = true; break; }
                }
                if (already_anchor) continue;
                
                if (min_prox_to_anchors[n] < best_min_prox) {
                    best_min_prox = min_prox_to_anchors[n];
                    best_idx = n;
                }
            }
            
            if (best_idx >= 0) {
                anchors.push_back(best_idx);
            } else {
                break;  // Shouldn't happen
            }
        }
    } else {
        // RANDOM: Simple random sampling (fast)
        std::vector<integer_t> all_indices(config_.nsample);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        std::mt19937 rng(config_.iseed);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        
        anchors.assign(all_indices.begin(), all_indices.begin() + n_anchors);
    }
    
    outlier_anchors_ = anchors;
    return anchors;
}

real_t RandomForest::compute_outlier_score_single(
    integer_t sample_idx,
    const std::vector<integer_t>& anchors) const {
    
    if (leaf_assignments_.empty()) {
        throw std::runtime_error("Outlier detection requires compute_leaf_assignments=True");
    }
    
    real_t prox_sum_sq = 0.0f;
    integer_t n_anchors_used = 0;
    
    for (integer_t anchor : anchors) {
        if (anchor == sample_idx) continue;
        n_anchors_used++;
        
        // Compute proximity using leaf assignments
        integer_t shared = 0;
        for (integer_t t = 0; t < config_.ntree; ++t) {
            size_t idx_s = static_cast<size_t>(t) * config_.nsample + sample_idx;
            size_t idx_a = static_cast<size_t>(t) * config_.nsample + anchor;
            if (leaf_assignments_[idx_s] == leaf_assignments_[idx_a]) {
                shared++;
            }
        }
        real_t prox = static_cast<real_t>(shared) / config_.ntree;
        prox_sum_sq += prox * prox;
    }
    
    // Raw outlier score (not normalized - would need class statistics)
    if (prox_sum_sq > 1e-10f) {
        return static_cast<real_t>(n_anchors_used) / prox_sum_sq;
    }
    return static_cast<real_t>(n_anchors_used) * 1e10f;
}

void RandomForest::compute_cluster_labels() {
    // Simple clustering based on proximity matrix
    cluster_labels_.resize(config_.nsample, 0);
    
    if (proximity_matrix_.empty()) {
        return;
    }
    
    // Use a simple threshold-based clustering
    std::vector<bool> assigned(config_.nsample, false);
    integer_t current_cluster = 0;
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        if (!assigned[i]) {
            cluster_labels_[i] = current_cluster;
            assigned[i] = true;
            
            // Assign nearby samples to the same cluster
            // Cast to size_t to avoid integer overflow in index calculation
            size_t prox_base_idx_clust = static_cast<size_t>(i) * static_cast<size_t>(config_.nsample);
            for (integer_t j = i + 1; j < config_.nsample; ++j) {
                size_t prox_idx_clust = prox_base_idx_clust + static_cast<size_t>(j);
                if (!assigned[j] && proximity_matrix_[prox_idx_clust] > 0.5f) {
                    cluster_labels_[j] = current_cluster;
                    assigned[j] = true;
                }
            }
            current_cluster++;
        }
    }
}

void RandomForest::initialize_workspace_arrays() {
    // Use assign() instead of resize() to ensure all elements are zeroed
    // This is critical when nsample changes (e.g., unsupervised with synthetic data)
    win_workspace_.assign(config_.nsample, 0.0f);
    nin_workspace_.assign(config_.nsample, 0);
    jinbag_workspace_.assign(config_.nsample, 0);
    joobag_workspace_.assign(config_.nsample, 0);
    // Cast to size_t to avoid integer overflow
    size_t amat_size = static_cast<size_t>(config_.mdim) * static_cast<size_t>(config_.nsample);
    amat_workspace_.assign(amat_size, 0);
    idmove_workspace_.assign(config_.nsample, 0);
    igoleft_workspace_.assign(config_.maxcat, 0);
    igl_workspace_.assign(config_.maxcat, 0);
    nodestart_workspace_.assign(config_.maxnode, 0);
    nodestop_workspace_.assign(config_.maxnode, 0);
    itemp_workspace_.assign(config_.nsample, 0);
    itmp_workspace_.assign(config_.maxcat, 0);
    icat_workspace_.assign(config_.maxcat, 0);
    tcat_workspace_.assign(config_.maxcat, 0.0f);
    // Cast to size_t to avoid integer overflow
    size_t classpop_size = static_cast<size_t>(config_.nclass) * static_cast<size_t>(config_.maxnode);
    classpop_workspace_.assign(classpop_size, 0.0f);
    wr_workspace_.assign(config_.nclass, 0.0f);
    wl_workspace_.assign(config_.nclass, 0.0f);
    // Cast to size_t to avoid integer overflow
    size_t tclasscat_size = static_cast<size_t>(config_.nclass) * static_cast<size_t>(config_.maxcat);
    tclasscat_workspace_.assign(tclasscat_size, 0.0f);
    tclasspop_workspace_.assign(config_.nclass, 0.0f);
    tmpclass_workspace_.assign(config_.nclass, 0.0f);
    jtr_workspace_.assign(config_.nsample, 0);
    nodextr_workspace_.assign(config_.nsample, 0);
}

// Synchronized getters for GPU memory safety
const dp_t* RandomForest::get_proximity_matrix() const {
    // Before get_proximity, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // For low-rank mode, DO NOT auto-reconstruct - return nullptr
    // Users should call get_lowrank_factors() or compute_mds_3d_from_factors() instead
    // Auto-reconstruction was causing performance issues and memory problems
    if (lowrank_proximity_ptr_ != nullptr) {
        // Low-rank mode active - do not auto-reconstruct full matrix
        // Use get_lowrank_factors() to get A, B factors directly
        // Use compute_mds_3d_from_factors() for MDS visualization
        return nullptr;
    }
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    // Return pointer only if vector is not empty and properly sized
    // This prevents returning invalid pointers if the vector was moved or resized
    if (proximity_matrix_.empty()) {
        return nullptr;
    }
    // Verify the vector size matches expected size to prevent out-of-bounds access
    size_t expected_size = static_cast<size_t>(config_.nsample) * config_.nsample;
    if (proximity_matrix_.size() != expected_size) {
        // std::cerr << "WARNING: proximity_matrix_ size mismatch. Expected " << expected_size 
        //           << ", got " << proximity_matrix_.size() << std::endl;  // Commented to avoid stream conflicts
        return nullptr;
    }
    
    // Ensure the vector's data pointer is valid
    // Check if the vector's capacity is sufficient (prevents issues if vector was moved)
    if (proximity_matrix_.capacity() < expected_size) {
        // std::cerr << "WARNING: proximity_matrix_ capacity insufficient. Expected at least "  // Commented to avoid stream conflicts 
                //   << expected_size << ", got " << proximity_matrix_.capacity() << std::endl;
        return nullptr;
    }
    
    // Verify the data pointer is actually valid
    // Try to access the first element to ensure the memory is valid
    // This helps catch cases where the vector was moved or the memory was freed
    try {
        volatile dp_t test = proximity_matrix_[0];
        (void)test;  // Suppress unused variable warning
    } catch (...) {
        // std::cerr << "WARNING: proximity_matrix_ data pointer is invalid (vector may have been moved)" << std::endl;  // Commented to avoid stream conflicts
        return nullptr;
    }
    
    // Return pointer - vector should remain stable as long as model exists
    // The Python binding will copy the data immediately, so this is safe
    // However, the vector must not be moved or cleared while Python holds the pointer
    return proximity_matrix_.data();
}

const real_t* RandomForest::get_qimpm() const {
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    return qimpm_.data();
}

const real_t* RandomForest::get_avimp() const {
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    return avimp_.data();
}

const real_t* RandomForest::get_sqsd() const {
    // GPU synchronization is handled by GPU code itself before returning
    // Avoid direct CUDA calls in CPU compilation path
    return sqsd_.data();
}

bool RandomForest::get_lowrank_factors(dp_t** A_host, dp_t** B_host, integer_t* r) const {
    // Before get_lowrank_factors, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check for sparse low-rank factors first (GPU Sparse training stores factors directly)
    if (!sparse_lowrank_A_.empty() && !sparse_lowrank_B_.empty() && sparse_lowrank_rank_ > 0) {
        // Allocate and copy sparse low-rank factors
        size_t factor_size = config_.nsample * sparse_lowrank_rank_;
        *A_host = new dp_t[factor_size];
        *B_host = new dp_t[factor_size];
        std::copy(sparse_lowrank_A_.begin(), sparse_lowrank_A_.end(), *A_host);
        std::copy(sparse_lowrank_B_.begin(), sparse_lowrank_B_.end(), *B_host);
        *r = sparse_lowrank_rank_;
        return true;
    }
    
    // Check if GPU Dense low-rank mode is active
    if (lowrank_proximity_ptr_ == nullptr) {
        return false;  // Low-rank mode not active
    }
    
    #ifdef CUDA_FOUND
    try {
        // Use helper function from CUDA file to avoid CUDA header inclusion in CPU compilation
        bool result = rf::cuda::get_lowrank_factors_host(lowrank_proximity_ptr_, A_host, B_host, r, get_n_samples());
        
        // After GPU operation, finalize to ensure all kernels complete
        if (config_.use_gpu && result) {
            rf::cuda::cuda_finalize_operations();
        }
        
        return result;
    } catch (...) {
        return false;  // Error occurred
    }
    #else
    // CPU-only compilation - low-rank mode requires CUDA
    return false;
    #endif
}

real_t RandomForest::compute_proximity_from_leaves(integer_t sample_i, integer_t sample_j) const {
    // Compute exact proximity between two samples using leaf assignments
    // Returns fraction of trees where both samples land in same terminal node
    
    if (leaf_assignments_.empty()) {
        return 0.0f;  // Leaf assignments not available
    }
    
    if (sample_i < 0 || sample_i >= config_.nsample ||
        sample_j < 0 || sample_j >= config_.nsample) {
        return 0.0f;  // Invalid sample indices
    }
    
    // Breiman's approach: only use trees where sample_i was OOB
    // Compare to sample_j in those trees (j can be in-bag or OOB)
    if (oob_trees_.empty() || sample_i >= static_cast<integer_t>(oob_trees_.size())) {
        return 0.0f;  // OOB tracking not available
    }
    
    const auto& oob_trees_for_i = oob_trees_[sample_i];
    if (oob_trees_for_i.empty()) {
        return 0.0f;  // No trees where sample_i was OOB
    }
    
    integer_t matches = 0;
    for (integer_t t : oob_trees_for_i) {
        // Cast to size_t to avoid integer overflow in index calculation
        size_t idx_ti = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(sample_i);
        size_t idx_tj = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(sample_j);
        int16_t leaf_i = leaf_assignments_[idx_ti];
        int16_t leaf_j = leaf_assignments_[idx_tj];
        if (leaf_i == leaf_j) {
            ++matches;
        }
    }
    
    return static_cast<real_t>(matches) / static_cast<real_t>(oob_trees_for_i.size());
}

void RandomForest::build_inverted_leaf_index() {
    // Build inverted leaf index for fast top-K lookups
    // Maps (tree, leaf_id) -> vector of sample indices
    // This enables O(ntree * avg_samples_per_leaf) lookups instead of O(ntree * nsample)
    
    if (leaf_assignments_.empty()) {
        throw std::runtime_error("Cannot build inverted leaf index: leaf assignments not available. "
                                "Set compute_leaf_assignments=True when creating the model.");
    }
    
    if (inverted_leaf_index_built_) {
        return;  // Already built
    }
    
    // Allocate one map per tree
    inverted_leaf_index_.resize(config_.ntree);
    
    // Build the index - include ALL samples (both OOB and in-bag)
    // Breiman's approach: query sample must be OOB, but compare to all samples
    for (integer_t t = 0; t < config_.ntree; ++t) {
        inverted_leaf_index_[t].clear();
        
        for (integer_t n = 0; n < config_.nsample; ++n) {
            // Cast to size_t to avoid integer overflow in index calculation
            size_t idx = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(n);
            int16_t leaf = leaf_assignments_[idx];
            inverted_leaf_index_[t][leaf].push_back(n);
        }
    }
    
    inverted_leaf_index_built_ = true;
}

std::vector<std::pair<integer_t, real_t>> RandomForest::get_top_k_similar(
    integer_t query_idx, 
    integer_t k,
    bool exclude_self) const {
    // Breiman's approach: only use trees where query was OOB
    // Compare to ALL samples in those trees (both OOB and in-bag)
    // Uses dual inverted indexes for O(num_oob_trees * avg_samples_per_leaf) complexity
    
    if (leaf_assignments_.empty()) {
        return {};  // No leaf assignments available
    }
    
    if (query_idx < 0 || query_idx >= config_.nsample) {
        return {};  // Invalid query index
    }
    
    // Get trees where query was OOB
    if (oob_trees_.empty() || query_idx >= static_cast<integer_t>(oob_trees_.size())) {
        return {};  // OOB tracking not available
    }
    
    const auto& oob_trees_for_query = oob_trees_[query_idx];
    if (oob_trees_for_query.empty()) {
        return {};  // No trees where query was OOB
    }
    
    // Similarity scores: count of trees where samples share the same leaf as query
    std::vector<integer_t> co_occurrence_counts(config_.nsample, 0);
    
    if (inverted_leaf_index_built_) {
        // Fast path: use inverted index - O(num_oob_trees * avg_samples_per_leaf)
        for (integer_t t : oob_trees_for_query) {
            // Cast to size_t to avoid integer overflow in index calculation
            size_t query_idx_t = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(query_idx);
            int16_t query_leaf = leaf_assignments_[query_idx_t];
            
            // Find all samples in the same leaf (includes both OOB and in-bag)
            auto it = inverted_leaf_index_[t].find(query_leaf);
            if (it != inverted_leaf_index_[t].end()) {
                for (integer_t sample_idx : it->second) {
                    if (!exclude_self || sample_idx != query_idx) {
                        co_occurrence_counts[sample_idx]++;
                    }
                }
            }
        }
    } else {
        // Fallback: O(num_oob_trees * nsample) scan
        for (integer_t t : oob_trees_for_query) {
            // Cast to size_t to avoid integer overflow in index calculation
            size_t query_idx_t = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(query_idx);
            int16_t query_leaf = leaf_assignments_[query_idx_t];
            
            for (integer_t n = 0; n < config_.nsample; ++n) {
                if (exclude_self && n == query_idx) continue;
                
                // Cast to size_t to avoid integer overflow in index calculation
                size_t idx_tn = static_cast<size_t>(t) * static_cast<size_t>(config_.nsample) + static_cast<size_t>(n);
                if (leaf_assignments_[idx_tn] == query_leaf) {
                    co_occurrence_counts[n]++;
                }
            }
        }
    }
    
    // Convert counts to similarity scores (fraction of trees)
    real_t norm = 1.0f / static_cast<real_t>(config_.ntree);
    
    // Find top-K using partial sort
    // Create index-score pairs for sorting
    std::vector<std::pair<integer_t, integer_t>> indexed_counts;
    indexed_counts.reserve(config_.nsample);
    
    for (integer_t i = 0; i < config_.nsample; ++i) {
        if (exclude_self && i == query_idx) continue;
        if (co_occurrence_counts[i] > 0) {
            indexed_counts.emplace_back(i, co_occurrence_counts[i]);
        }
    }
    
    // Partial sort to get top-K (more efficient than full sort for large datasets)
    integer_t actual_k = std::min(k, static_cast<integer_t>(indexed_counts.size()));
    
    if (actual_k > 0) {
        std::partial_sort(indexed_counts.begin(), 
                         indexed_counts.begin() + actual_k,
                         indexed_counts.end(),
                         [](const auto& a, const auto& b) { 
                             return a.second > b.second;  // Descending by count
                         });
    }
    
    // Build result
    std::vector<std::pair<integer_t, real_t>> result;
    result.reserve(actual_k);
    
    for (integer_t i = 0; i < actual_k; ++i) {
        result.emplace_back(
            indexed_counts[i].first, 
            static_cast<real_t>(indexed_counts[i].second) * norm
        );
    }
    
    return result;
}

// ============================================================================
// PREDICT LEAF ASSIGNMENTS FOR NEW DATA
// Works for all configurations: casewise/non-casewise, dense/sparse, CPU/GPU
// ============================================================================

void RandomForest::predict_leaf_assignments(const real_t* X_new, integer_t nsamples_new, int16_t* leaf_out) const {
    // Output layout: leaf_out[sample * ntree + tree] = leaf node for sample in tree
    // This is (nsamples_new, ntree) layout for easier Python access
    
    const integer_t ntree = config_.ntree;
    const integer_t maxnode = config_.maxnode;
    const integer_t mdim = config_.mdim;
    
    // Process each tree
    for (integer_t tree = 0; tree < ntree; ++tree) {
        // Get tree structure pointers for this tree
        const integer_t* nodestatus = nodestatus_.data() + tree * maxnode;
        const integer_t* bestvar = bestvar_.data() + tree * maxnode;
        const real_t* xbestsplit = xbestsplit_.data() + tree * maxnode;
        const integer_t* treemap = treemap_.data() + tree * maxnode * 2;
        const integer_t* catgoleft = catgoleft_.data() + tree * maxnode * config_.maxcat;
        
        // Process each new sample
        for (integer_t sample = 0; sample < nsamples_new; ++sample) {
            const real_t* sample_features = X_new + sample * mdim;
            
            // Traverse tree to find terminal node
            integer_t current_node = 0;
            
            for (integer_t k = 0; k < maxnode; ++k) {
                if (nodestatus[current_node] == -1) {
                    // Terminal node found
                    break;
                }
                
                integer_t split_var = bestvar[current_node];
                real_t split_value = xbestsplit[current_node];
                
                if (split_var >= 0 && split_var < mdim) {
                    real_t feature_value = sample_features[split_var];
                    
                    if (cat_[split_var] == 1) {
                        // Quantitative variable
                        if (feature_value <= split_value) {
                            current_node = treemap[current_node * 2];
                        } else {
                            current_node = treemap[current_node * 2 + 1];
                        }
                    } else {
                        // Categorical variable
                        integer_t jcat = static_cast<integer_t>(feature_value + 0.5);
                        if (jcat >= 0 && jcat < cat_[split_var]) {
                            if (catgoleft[(current_node * config_.maxcat) + jcat] == 1) {
                                current_node = treemap[current_node * 2];
                            } else {
                                current_node = treemap[current_node * 2 + 1];
                            }
                        }
                    }
                }
            }
            
            // Store leaf node ID
            // Cast to size_t to avoid integer overflow in index calculation
            size_t leaf_out_idx = static_cast<size_t>(sample) * static_cast<size_t>(ntree) + static_cast<size_t>(tree);
            leaf_out[leaf_out_idx] = static_cast<int16_t>(current_node);
        }
    }
}

void RandomForest::predict_leaf_assignments_sparse(const SparseMatrixCSR& X_new, int16_t* leaf_out) const {
    // Output layout: leaf_out[sample * ntree + tree] = leaf node for sample in tree
    
    const integer_t ntree = config_.ntree;
    const integer_t maxnode = config_.maxnode;
    const integer_t nsamples_new = X_new.nrows;
    
    // Process each tree
    for (integer_t tree = 0; tree < ntree; ++tree) {
        // Get tree structure pointers for this tree
        const integer_t* nodestatus = nodestatus_.data() + tree * maxnode;
        const integer_t* bestvar = bestvar_.data() + tree * maxnode;
        const real_t* xbestsplit = xbestsplit_.data() + tree * maxnode;
        const integer_t* treemap = treemap_.data() + tree * maxnode * 2;
        const integer_t* catgoleft = catgoleft_.data() + tree * maxnode * config_.maxcat;
        
        // Process each new sample
        for (integer_t sample = 0; sample < nsamples_new; ++sample) {
            // Get sparse row for this sample
            integer_t row_start = X_new.indptr[sample];
            integer_t row_end = X_new.indptr[sample + 1];
            
            // Traverse tree to find terminal node
            integer_t current_node = 0;
            
            for (integer_t k = 0; k < maxnode; ++k) {
                if (nodestatus[current_node] == -1) {
                    // Terminal node found
                    break;
                }
                
                integer_t split_var = bestvar[current_node];
                real_t split_value = xbestsplit[current_node];
                
                if (split_var >= 0 && split_var < X_new.ncols) {
                    // Get feature value from sparse row (binary search)
                    real_t feature_value = 0.0f;  // Default for missing features
                    
                    for (integer_t j = row_start; j < row_end; ++j) {
                        if (X_new.indices[j] == split_var) {
                            feature_value = X_new.data[j];
                            break;
                        } else if (X_new.indices[j] > split_var) {
                            break;  // Indices are sorted, feature not present
                        }
                    }
                    
                    if (cat_[split_var] == 1) {
                        // Quantitative variable
                        if (feature_value <= split_value) {
                            current_node = treemap[current_node * 2];
                        } else {
                            current_node = treemap[current_node * 2 + 1];
                        }
                    } else {
                        // Categorical variable
                        integer_t jcat = static_cast<integer_t>(feature_value + 0.5);
                        if (jcat >= 0 && jcat < cat_[split_var]) {
                            if (catgoleft[(current_node * config_.maxcat) + jcat] == 1) {
                                current_node = treemap[current_node * 2];
                            } else {
                                current_node = treemap[current_node * 2 + 1];
                            }
                        }
                    }
                }
            }
            
            // Store leaf node ID
            // Cast to size_t to avoid integer overflow in index calculation
            size_t leaf_out_idx = static_cast<size_t>(sample) * static_cast<size_t>(ntree) + static_cast<size_t>(tree);
            leaf_out[leaf_out_idx] = static_cast<int16_t>(current_node);
        }
    }
}

std::vector<double> RandomForest::compute_mds_from_factors(rf::integer_t k) const {
    // Before compute_mds, ensure CUDA context is ready (handles context between Jupyter cells)
    if (config_.use_gpu) {
        rf::cuda::cuda_ensure_context_ready();
    }
    
    // Check if low-rank mode is active
    if (lowrank_proximity_ptr_ == nullptr) {
        return std::vector<double>();  // Low-rank mode not active
    }
    
    #ifdef CUDA_FOUND
    try {
        // Use helper function from CUDA file to avoid CUDA header inclusion in CPU compilation
        std::vector<double> result = rf::cuda::compute_mds_from_factors_host(lowrank_proximity_ptr_, k);
        
        // After GPU operation, finalize to ensure all kernels complete
        if (config_.use_gpu && !result.empty()) {
            rf::cuda::cuda_finalize_operations();
        }
        
        return result;
    } catch (...) {
        return std::vector<double>();  // Error occurred
    }
    #else
    // CPU-only compilation - low-rank mode requires CUDA
    return std::vector<double>();
    #endif
}

// ============================================================================
// SPARSE MATRIX SUPPORT - Native CSR input (no dense conversion)
// ============================================================================
// For recommender systems with massive sparse user-item matrices
// Pipeline: CSR → fit → importance → proximity (full or low-rank)
// Memory: O(nnz) for data, O(n×rank) for low-rank proximity
// ============================================================================

void RandomForest::fit_sparse(const SparseMatrixCSR& X, const void* y, const real_t* sample_weight) {
    // Validate sparse matrix
    if (!X.is_valid()) {
        throw std::runtime_error("Invalid sparse matrix: CSR structure is malformed");
    }
    
    // Update config dimensions from sparse matrix
    config_.nsample = X.nrows;
    config_.mdim = X.ncols;
    
    // Store sparse data reference
    X_train_sparse_ = X;
    is_sparse_ = true;
    
    // Dispatch based on task type
    switch (config_.task_type) {
        case TaskType::CLASSIFICATION:
            fit_classification_sparse(X, static_cast<const integer_t*>(y), sample_weight);
            break;
        case TaskType::REGRESSION:
            fit_regression_sparse(X, static_cast<const real_t*>(y), sample_weight);
            break;
        case TaskType::UNSUPERVISED:
            fit_unsupervised_sparse(X, sample_weight);
            break;
    }
}

void RandomForest::fit_classification_sparse(const SparseMatrixCSR& X, const integer_t* y, const real_t* sample_weight) {
    // ============================================================================
    // NATIVE SPARSE CLASSIFICATION - O(nnz) memory, NO dense conversion
    // ============================================================================
    
    is_sparse_ = true;
    X_train_sparse_ = X;
    
    
    // Update config with sparse matrix dimensions
    config_.nsample = X.nrows;
    config_.mdim = X.ncols;
    
    // Store y data (classification labels)
    y_train_classification_.assign(y, y + config_.nsample);
    
    // Initialize sample weights
    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }
    
    // Setup classification task (creates 1-based labels, counts classes)
    setup_classification_task(y);
    
    // Initialize workspace arrays (uses config_.nsample, config_.mdim, etc.)
    initialize_workspace_arrays();
    
    // For sparse mode, we don't call prepare_data() - the sparse tree growing
    // handles sorting internally. But we need to initialize ties_ and asave_
    // to avoid null pointer issues in other functions.
    // Cast to size_t to avoid integer overflow
    size_t ties_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
    ties_.resize(ties_size, 0);
    asave_.resize(ties_size, 0);
    
    // Initialize cat_ array (all quantitative by default for sparse)
    if (cat_.empty()) {
        cat_.resize(config_.mdim, 1);  // 1 = quantitative
    }
    
    // GPU sparse is now available - keep user's use_gpu setting
    // Note: use_qlora for proximity requires GPU
    if (!config_.use_gpu && config_.compute_proximity && config_.use_qlora) {
        config_.use_qlora = false;  // Low-rank is GPU-only
    }
    
    // Grow trees using native sparse tree growing
    if (config_.use_gpu) {
#ifdef CUDA_FOUND
        // GPU SPARSE PATH - use parallel GPU sparse training
        fit_sparse_gpu_classification(X, y, sample_weight);
#else
        throw std::runtime_error("GPU support not available in this CPU-only build. Set use_gpu=False.");
#endif
    } else {
        // CPU SPARSE PATH - sequential tree growing
        // Allocate leaf_assignments_ if requested (CPU sparse mode)
        if (config_.compute_leaf_assignments && leaf_assignments_.empty()) {
            // Cast to size_t to avoid integer overflow
            size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
            if (leaf_size > leaf_assignments_.max_size()) {
                throw std::runtime_error(
                    "Cannot allocate leaf_assignments_: size " + std::to_string(leaf_size) + 
                    " exceeds max_size() " + std::to_string(leaf_assignments_.max_size())
                );
            }
            leaf_assignments_.resize(leaf_size, 0);
        }
        
        for (integer_t itree = 0; itree < config_.ntree; ++itree) {
            integer_t seed_val = config_.iseed + itree;
            grow_tree_single(itree, seed_val);  // Uses cpu_growtree_sparse
            
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
        
        // Compute OOB error and predictions BEFORE finalize_training normalizes q_
        compute_classification_error();
        compute_oob_predictions();
        
        // Finalize training (proximity, normalize OOB votes)
        finalize_training();
    }
    
}

void RandomForest::fit_regression_sparse(const SparseMatrixCSR& X, const real_t* y, const real_t* sample_weight) {
    // ============================================================================
    // NATIVE SPARSE REGRESSION - O(nnz) memory, NO dense conversion
    // ============================================================================
    is_sparse_ = true;
    X_train_sparse_ = X;
    
    // Update config with sparse matrix dimensions
    config_.nsample = X.nrows;
    config_.mdim = X.ncols;
    config_.nclass = 1;  // Regression uses nclass=1
    
    // Store y data (regression targets)
    y_train_regression_.assign(y, y + config_.nsample);
    
    // Initialize sample weights
    if (sample_weight) {
        sample_weight_.assign(sample_weight, sample_weight + config_.nsample);
    } else {
        sample_weight_.resize(config_.nsample, 1.0f);
    }
    
    // Setup regression task
    setup_regression_task(y);
    
    // Initialize workspace arrays
    initialize_workspace_arrays();
    
    // For sparse mode, initialize ties_ and asave_ to avoid null pointer issues
    // Cast to size_t to avoid integer overflow
    size_t ties_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
    ties_.resize(ties_size, 0);
    asave_.resize(ties_size, 0);
    
    // Initialize cat_ array (all quantitative by default for sparse)
    if (cat_.empty()) {
        cat_.resize(config_.mdim, 1);  // 1 = quantitative
    }
    
    // GPU sparse is now available - keep user's use_gpu setting
    if (!config_.use_gpu && config_.compute_proximity && config_.use_qlora) {
        config_.use_qlora = false;  // Low-rank is GPU-only
    }
    
    // Grow trees using native sparse tree growing
    if (config_.use_gpu) {
#ifdef CUDA_FOUND
        // GPU SPARSE PATH - use parallel GPU sparse training
        fit_sparse_gpu_regression(X, y, sample_weight);
#else
        throw std::runtime_error("GPU support not available in this CPU-only build. Set use_gpu=False.");
#endif
    } else {
        // CPU SPARSE PATH - sequential tree growing
        // Allocate leaf_assignments_ if requested (CPU sparse mode)
        if (config_.compute_leaf_assignments && leaf_assignments_.empty()) {
            // Cast to size_t to avoid integer overflow
            size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
            if (leaf_size > leaf_assignments_.max_size()) {
                throw std::runtime_error(
                    "Cannot allocate leaf_assignments_: size " + std::to_string(leaf_size) + 
                    " exceeds max_size() " + std::to_string(leaf_assignments_.max_size())
                );
            }
            leaf_assignments_.resize(leaf_size, 0);
        }
        
        for (integer_t itree = 0; itree < config_.ntree; ++itree) {
            integer_t seed_val = config_.iseed + itree;
            grow_tree_single(itree, seed_val);  // Uses cpu_growtree_sparse
            
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
        
        // Finalize training (this also computes regression MSE)
        finalize_training();
    }
}

void RandomForest::fit_unsupervised_sparse(const SparseMatrixCSR& X, const real_t* sample_weight) {
    // ============================================================================
    // NATIVE SPARSE UNSUPERVISED - O(nnz) memory, NO dense conversion
    // Breiman-Cutler: Real samples (class 1) vs Synthetic samples (class 0)
    // Synthetic: For each feature, pick random donor sample (preserves sparsity)
    // ============================================================================
    is_sparse_ = true;
    
    integer_t n_real = X.nrows;
    integer_t mdim = X.ncols;
    
    // Calculate number of synthetic samples
    n_synthetic_ = static_cast<integer_t>(n_real * config_.synthetic_ratio);
    if (n_synthetic_ < 1) n_synthetic_ = 1;
    n_total_unsupervised_ = n_real + n_synthetic_;
    
    // ========================================================================
    // NATIVE SPARSE SYNTHETIC GENERATION - O(nnz) memory
    // For each synthetic sample, each feature comes from a random real sample
    // If donor has 0 for that feature, synthetic also has 0 (no storage)
    // ========================================================================
    
    // Step 1: Build synthetic data directly in COO format (no dense allocation!)
    // Thread-local COO accumulators for parallel construction
    std::vector<std::vector<integer_t>> thread_coo_rows;
    std::vector<std::vector<integer_t>> thread_coo_cols;
    std::vector<std::vector<real_t>> thread_coo_vals;
    
    integer_t num_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif
    thread_coo_rows.resize(num_threads);
    thread_coo_cols.resize(num_threads);
    thread_coo_vals.resize(num_threads);
    
    // Pre-estimate capacity (assume synthetic has similar sparsity to real)
    double real_density = static_cast<double>(X.nnz) / (n_real * mdim);
    integer_t estimated_nnz_per_thread = static_cast<integer_t>(
        (n_synthetic_ * mdim * real_density) / num_threads + 1000);
    
    for (integer_t t = 0; t < num_threads; ++t) {
        thread_coo_rows[t].reserve(estimated_nnz_per_thread);
        thread_coo_cols[t].reserve(estimated_nnz_per_thread);
        thread_coo_vals[t].reserve(estimated_nnz_per_thread);
    }
    
    // Generate synthetic samples in parallel (each sample is independent)
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (integer_t syn_row = 0; syn_row < n_synthetic_; ++syn_row) {
        integer_t tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        // Thread-local RNG seeded deterministically per synthetic sample
        std::mt19937 rng(config_.iseed + syn_row * 1000003);
        
        // For each feature, pick a random donor from real samples
        for (integer_t k = 0; k < mdim; ++k) {
            // Pick random donor sample
            integer_t donor = rng() % n_real;
            
            // Get donor's value for feature k (sparse lookup)
            real_t val = X.get(donor, k);
            
            // Only store non-zeros (preserves sparsity!)
            if (val != 0.0f) {
                thread_coo_rows[tid].push_back(n_real + syn_row);  // Offset by n_real
                thread_coo_cols[tid].push_back(k);
                thread_coo_vals[tid].push_back(val);
            }
        }
    }
    
    // Step 2: Merge thread-local COO into combined COO
    std::vector<integer_t> coo_rows, coo_cols;
    std::vector<real_t> coo_vals;
    
    // Reserve space for real + all synthetic
    integer_t total_synthetic_nnz = 0;
    for (integer_t t = 0; t < num_threads; ++t) {
        total_synthetic_nnz += static_cast<integer_t>(thread_coo_vals[t].size());
    }
    coo_rows.reserve(X.nnz + total_synthetic_nnz);
    coo_cols.reserve(X.nnz + total_synthetic_nnz);
    coo_vals.reserve(X.nnz + total_synthetic_nnz);
    
    // Add real data entries (directly from input sparse matrix - no conversion!)
    for (integer_t row = 0; row < n_real; ++row) {
        integer_t start = X.indptr[row];
        integer_t end = X.indptr[row + 1];
        for (integer_t i = start; i < end; ++i) {
            coo_rows.push_back(row);
            coo_cols.push_back(X.indices[i]);
            coo_vals.push_back(X.data[i]);
        }
    }
    
    // Add synthetic data entries from all threads
    for (integer_t t = 0; t < num_threads; ++t) {
        coo_rows.insert(coo_rows.end(), thread_coo_rows[t].begin(), thread_coo_rows[t].end());
        coo_cols.insert(coo_cols.end(), thread_coo_cols[t].begin(), thread_coo_cols[t].end());
        coo_vals.insert(coo_vals.end(), thread_coo_vals[t].begin(), thread_coo_vals[t].end());
    }
    
    // Free thread-local storage
    thread_coo_rows.clear();
    thread_coo_cols.clear();
    thread_coo_vals.clear();
    
    // Step 3: Convert COO to CSR
    SparseMatrixCSR X_combined;
    X_combined.nrows = n_total_unsupervised_;
    X_combined.ncols = mdim;
    X_combined.nnz = static_cast<integer_t>(coo_vals.size());
    X_combined.indptr.resize(n_total_unsupervised_ + 1, 0);
    X_combined.indices.resize(X_combined.nnz);
    X_combined.data.resize(X_combined.nnz);
    
    // Count entries per row
    for (size_t i = 0; i < coo_rows.size(); ++i) {
        X_combined.indptr[coo_rows[i] + 1]++;
    }
    
    // Cumulative sum for indptr
    for (integer_t i = 1; i <= n_total_unsupervised_; ++i) {
        X_combined.indptr[i] += X_combined.indptr[i - 1];
    }
    
    // Fill indices and data (need to track current position per row)
    std::vector<integer_t> row_pos(n_total_unsupervised_, 0);
    for (size_t i = 0; i < coo_rows.size(); ++i) {
        integer_t row = coo_rows[i];
        integer_t pos = X_combined.indptr[row] + row_pos[row];
        X_combined.indices[pos] = coo_cols[i];
        X_combined.data[pos] = coo_vals[i];
        row_pos[row]++;
    }
    
    // Sort indices within each row (CSR requires sorted column indices)
    for (integer_t row = 0; row < n_total_unsupervised_; ++row) {
        integer_t start = X_combined.indptr[row];
        integer_t end = X_combined.indptr[row + 1];
        if (end > start) {
            // Create pairs of (col, val) and sort by col
            std::vector<std::pair<integer_t, real_t>> entries;
            for (integer_t i = start; i < end; ++i) {
                entries.push_back({X_combined.indices[i], X_combined.data[i]});
            }
            std::sort(entries.begin(), entries.end());
            for (integer_t i = start; i < end; ++i) {
                X_combined.indices[i] = entries[i - start].first;
                X_combined.data[i] = entries[i - start].second;
            }
        }
    }
    
    // Store combined sparse matrix
    X_train_sparse_ = std::move(X_combined);
    
    // NOTE: X_unsupervised_expanded_ is NOT allocated for sparse mode
    // This saves O(n × m) memory. GPU sparse uses X_train_sparse_ directly.
    
    // Create labels: real=1, synthetic=0
    y_train_unsupervised_.resize(n_total_unsupervised_);
    std::fill(y_train_unsupervised_.begin(), y_train_unsupervised_.begin() + n_real, 1);
    std::fill(y_train_unsupervised_.begin() + n_real, y_train_unsupervised_.end(), 0);
    
    // Update config with combined dimensions
    config_.nsample = n_total_unsupervised_;
    config_.mdim = mdim;
    config_.nclass = 2;  // Binary: real vs synthetic

    // Update maxnode for expanded sample count (trees need more nodes)
    config_.maxnode = calculate_maxnode(config_.nsample, config_.minndsize);
    
    // Initialize sample weights
    sample_weight_.resize(n_total_unsupervised_, 1.0f);
    if (sample_weight) {
        for (integer_t i = 0; i < n_real; ++i) {
            sample_weight_[i] = sample_weight[i];
        }
    }
    
    // Setup unsupervised task parameters
    setup_unsupervised_task();
    
    // Initialize ALL arrays (tree storage, OOB tracking, importance, proximity)
    // This MUST be called after config_.nsample is updated to include synthetic samples
    initialize_arrays();
    
    // For sparse mode, initialize ties_ and asave_ to avoid null pointer issues
    // (Note: these are also set in initialize_arrays, but we override with resize for sparse)
    // Cast to size_t to avoid integer overflow
    size_t ties_size = static_cast<size_t>(config_.nsample) * static_cast<size_t>(config_.mdim);
    ties_.resize(ties_size, 0);
    asave_.resize(ties_size, 0);
    
    // Initialize cat_ array (all quantitative by default for sparse)
    if (cat_.empty()) {
        cat_.resize(config_.mdim, 1);  // 1 = quantitative
    }
    
    // GPU sparse is now available - keep user's use_gpu setting
    if (!config_.use_gpu && config_.compute_proximity && config_.use_qlora) {
        config_.use_qlora = false;  // Low-rank is GPU-only
    }
    
    // Grow trees using native sparse tree growing
    if (config_.use_gpu) {
#ifdef CUDA_FOUND
        // GPU SPARSE PATH - use parallel GPU sparse training
        // IMPORTANT: Pass the combined matrix (real + synthetic), not the original X!
        fit_sparse_gpu_unsupervised(X_train_sparse_, sample_weight_.data());
#else
        throw std::runtime_error("GPU support not available in this CPU-only build. Set use_gpu=False.");
#endif
    } else {
        // CPU SPARSE PATH - sequential tree growing
        // Allocate leaf_assignments_ if requested (CPU sparse mode)
        if (config_.compute_leaf_assignments && leaf_assignments_.empty()) {
            // Cast to size_t to avoid integer overflow
            size_t leaf_size = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.nsample);
            if (leaf_size > leaf_assignments_.max_size()) {
                throw std::runtime_error(
                    "Cannot allocate leaf_assignments_: size " + std::to_string(leaf_size) + 
                    " exceeds max_size() " + std::to_string(leaf_assignments_.max_size())
                );
            }
            leaf_assignments_.resize(leaf_size, 0);
        }
        
        for (integer_t itree = 0; itree < config_.ntree; ++itree) {
            integer_t seed_val = config_.iseed + itree;
            grow_tree_single(itree, seed_val);  // Uses cpu_growtree_sparse
            
            if (progress_callback_) {
                progress_callback_(itree + 1, config_.ntree);
            }
        }
        
        // Compute OOB error for unsupervised mode (classifying real vs synthetic)
        compute_classification_error();
        
        // Finalize training
        finalize_training();
    }
}

void RandomForest::predict_sparse(const SparseMatrixCSR& X, void* predictions) {
    integer_t nsample = X.nrows;
    integer_t mdim = X.ncols;
    
    // Convert CSR to dense for prediction (temporary)
    std::vector<real_t> X_dense(nsample * mdim, 0.0f);
    for (integer_t row = 0; row < nsample; ++row) {
        integer_t start = X.indptr[row];
        integer_t end = X.indptr[row + 1];
        for (integer_t i = start; i < end; ++i) {
            integer_t col = X.indices[i];
            X_dense[row * mdim + col] = X.data[i];
        }
    }
    
    // Call dense predict
    predict(X_dense.data(), nsample, predictions);
}

// ============================================================================
// MODEL SERIALIZATION (save/load)
// ============================================================================

// Magic number and version for file format validation
static const uint32_t RFX_MAGIC = 0x52465830;  // "RFX0" in ASCII
static const uint32_t RFX_VERSION = 1;

// Helper functions for binary I/O
template<typename T>
static void write_value(std::ostream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
static void read_value(std::istream& is, T& value) {
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template<typename T>
static void write_vector(std::ostream& os, const std::vector<T>& vec) {
    size_t size = vec.size();
    write_value(os, size);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
    }
}

template<typename T>
static void read_vector(std::istream& is, std::vector<T>& vec) {
    size_t size;
    read_value(is, size);
    vec.resize(size);
    if (size > 0) {
        is.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    }
}

static void write_string(std::ostream& os, const std::string& str) {
    size_t size = str.size();
    write_value(os, size);
    if (size > 0) {
        os.write(str.data(), size);
    }
}

static void read_string(std::istream& is, std::string& str) {
    size_t size;
    read_value(is, size);
    str.resize(size);
    if (size > 0) {
        is.read(&str[0], size);
    }
}

void RandomForest::save(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    // Write header
    write_value(ofs, RFX_MAGIC);
    write_value(ofs, RFX_VERSION);
    
    // Write config (serialize individual fields for cross-platform compatibility)
    write_value(ofs, config_.nsample);
    write_value(ofs, config_.ntree);
    write_value(ofs, config_.mdim);
    write_value(ofs, config_.nclass);
    write_value(ofs, config_.maxcat);
    write_value(ofs, config_.mtry);
    write_value(ofs, config_.maxnode);
    write_value(ofs, config_.minndsize);
    write_value(ofs, config_.ncsplit);
    write_value(ofs, config_.ncmax);
    write_value(ofs, config_.iseed);
    write_value(ofs, static_cast<int>(config_.task_type));
    write_value(ofs, config_.compute_proximity);
    write_value(ofs, config_.compute_importance);
    write_value(ofs, config_.compute_local_importance);
    write_value(ofs, config_.compute_proximity_importance);
    write_value(ofs, config_.compute_leaf_assignments);
    write_value(ofs, config_.use_rfgap);
    write_value(ofs, config_.use_casewise);
    write_string(ofs, config_.proximity_type);
    write_string(ofs, config_.importance_method);
    write_value(ofs, config_.clique_M);
    write_value(ofs, config_.use_gpu);
    write_value(ofs, config_.use_qlora);
    write_value(ofs, config_.quant_mode);
    write_value(ofs, config_.lowrank_rank);
    write_value(ofs, config_.lowrank_power_iterations);
    write_value(ofs, config_.use_sparse);
    write_value(ofs, config_.sparsity_threshold);
    write_value(ofs, config_.batch_size);
    write_value(ofs, config_.use_histogram);
    write_value(ofs, config_.n_bins);
    write_value(ofs, config_.nodesize);
    write_value(ofs, config_.cutoff);
    write_value(ofs, config_.use_unsupervised);
    write_value(ofs, config_.outlier_threshold);
    write_value(ofs, config_.synthetic_ratio);
    write_value(ofs, static_cast<int>(config_.unsupervised_mode));
    write_value(ofs, config_.gpu_loss_function);
    write_value(ofs, config_.n_threads_cpu);
    
    // Write sparse flag
    write_value(ofs, is_sparse_);
    
    // Write tree structure arrays (required for prediction)
    write_vector(ofs, tnodewt_);
    write_vector(ofs, node_predictions_regression_);
    write_vector(ofs, xbestsplit_);
    write_vector(ofs, nodestatus_);
    write_vector(ofs, bestvar_);
    write_vector(ofs, treemap_);
    write_vector(ofs, catgoleft_);
    write_vector(ofs, nodeclass_);
    write_vector(ofs, nnode_);
    write_vector(ofs, cat_);
    
    // Write importance arrays
    write_vector(ofs, feature_importances_);
    write_vector(ofs, avimp_);
    write_vector(ofs, sqsd_);
    write_vector(ofs, qimp_);
    write_vector(ofs, qimpm_);
    write_vector(ofs, proximity_importance_);
    
    // Write leaf assignments (optional)
    write_vector(ofs, leaf_assignments_);
    
    // Write OOB tracking (for computing similarity on loaded model)
    size_t n_oob_trees = oob_trees_.size();
    write_value(ofs, n_oob_trees);
    for (const auto& trees : oob_trees_) {
        write_vector(ofs, trees);
    }
    
    // Write class label mappings
    size_t map_size = original_to_0based_label_map_.size();
    write_value(ofs, map_size);
    for (const auto& pair : original_to_0based_label_map_) {
        write_value(ofs, pair.first);
        write_value(ofs, pair.second);
    }
    write_vector(ofs, zero_based_to_original_label_map_);
    
    // Write OOB statistics
    write_value(ofs, oob_error_);
    write_value(ofs, oob_mse_);
    write_vector(ofs, nout_);
    
    if (!ofs) {
        throw std::runtime_error("Error writing to file: " + filepath);
    }
    
    ofs.close();
}

std::unique_ptr<RandomForest> RandomForest::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }
    
    // Read and validate header
    uint32_t magic, version;
    read_value(ifs, magic);
    read_value(ifs, version);
    
    if (magic != RFX_MAGIC) {
        throw std::runtime_error("Invalid RFX model file (bad magic number): " + filepath);
    }
    if (version != RFX_VERSION) {
        throw std::runtime_error("Unsupported RFX model version: " + std::to_string(version) + 
                                " (expected " + std::to_string(RFX_VERSION) + ")");
    }
    
    // Read config
    RandomForestConfig config;
    read_value(ifs, config.nsample);
    read_value(ifs, config.ntree);
    read_value(ifs, config.mdim);
    read_value(ifs, config.nclass);
    read_value(ifs, config.maxcat);
    read_value(ifs, config.mtry);
    read_value(ifs, config.maxnode);
    read_value(ifs, config.minndsize);
    read_value(ifs, config.ncsplit);
    read_value(ifs, config.ncmax);
    read_value(ifs, config.iseed);
    int task_type_int;
    read_value(ifs, task_type_int);
    config.task_type = static_cast<TaskType>(task_type_int);
    read_value(ifs, config.compute_proximity);
    read_value(ifs, config.compute_importance);
    read_value(ifs, config.compute_local_importance);
    read_value(ifs, config.compute_proximity_importance);
    read_value(ifs, config.compute_leaf_assignments);
    read_value(ifs, config.use_rfgap);
    read_value(ifs, config.use_casewise);
    read_string(ifs, config.proximity_type);
    read_string(ifs, config.importance_method);
    read_value(ifs, config.clique_M);
    read_value(ifs, config.use_gpu);
    read_value(ifs, config.use_qlora);
    read_value(ifs, config.quant_mode);
    read_value(ifs, config.lowrank_rank);
    read_value(ifs, config.lowrank_power_iterations);
    read_value(ifs, config.use_sparse);
    read_value(ifs, config.sparsity_threshold);
    read_value(ifs, config.batch_size);
    read_value(ifs, config.use_histogram);
    read_value(ifs, config.n_bins);
    read_value(ifs, config.nodesize);
    read_value(ifs, config.cutoff);
    read_value(ifs, config.use_unsupervised);
    read_value(ifs, config.outlier_threshold);
    read_value(ifs, config.synthetic_ratio);
    int unsupervised_mode_int;
    read_value(ifs, unsupervised_mode_int);
    config.unsupervised_mode = static_cast<UnsupervisedMode>(unsupervised_mode_int);
    read_value(ifs, config.gpu_loss_function);
    read_value(ifs, config.n_threads_cpu);
    
    // Create model with loaded config
    auto model = std::make_unique<RandomForest>(config);
    
    // Read sparse flag
    read_value(ifs, model->is_sparse_);
    
    // Read tree structure arrays
    read_vector(ifs, model->tnodewt_);
    read_vector(ifs, model->node_predictions_regression_);
    read_vector(ifs, model->xbestsplit_);
    read_vector(ifs, model->nodestatus_);
    read_vector(ifs, model->bestvar_);
    read_vector(ifs, model->treemap_);
    read_vector(ifs, model->catgoleft_);
    read_vector(ifs, model->nodeclass_);
    read_vector(ifs, model->nnode_);
    read_vector(ifs, model->cat_);
    
    // Read importance arrays
    read_vector(ifs, model->feature_importances_);
    read_vector(ifs, model->avimp_);
    read_vector(ifs, model->sqsd_);
    read_vector(ifs, model->qimp_);
    read_vector(ifs, model->qimpm_);
    read_vector(ifs, model->proximity_importance_);
    
    // Read leaf assignments
    read_vector(ifs, model->leaf_assignments_);
    
    // Read OOB tracking
    size_t n_oob_trees;
    read_value(ifs, n_oob_trees);
    model->oob_trees_.resize(n_oob_trees);
    for (size_t i = 0; i < n_oob_trees; ++i) {
        read_vector(ifs, model->oob_trees_[i]);
    }
    
    // Read class label mappings
    size_t map_size;
    read_value(ifs, map_size);
    for (size_t i = 0; i < map_size; ++i) {
        integer_t key, value;
        read_value(ifs, key);
        read_value(ifs, value);
        model->original_to_0based_label_map_[key] = value;
    }
    read_vector(ifs, model->zero_based_to_original_label_map_);
    
    // Read OOB statistics
    read_value(ifs, model->oob_error_);
    read_value(ifs, model->oob_mse_);
    read_vector(ifs, model->nout_);
    
    if (!ifs) {
        throw std::runtime_error("Error reading from file: " + filepath);
    }
    
    ifs.close();
    return model;
}

} // namespace rf
