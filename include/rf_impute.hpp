#ifndef RF_IMPUTE_HPP
#define RF_IMPUTE_HPP

#include "rf_types.hpp"
#include "rf_random_forest.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <sstream>
#include <iomanip>

namespace rf {

// ============================================================================
// IMPUTATION RESULTS - Detailed report of missing value handling
// ============================================================================
// Shows:
// - Missing values BEFORE imputation (per column)
// - How each column was filled (method: rough/rand, then RF prediction)
// - Missing values AFTER imputation (should be 0)
// ============================================================================

struct ColumnImputationInfo {
    integer_t column_idx;           // Column index
    std::string column_name;        // Optional column name
    integer_t missing_before;       // Count of missing values before
    integer_t missing_after;        // Count of missing values after (should be 0)
    bool is_categorical;            // True if categorical, false if numeric
    std::string initial_method;     // "median", "mode", or "random"
    real_t initial_fill_value;      // Value used for initial fill (median/mode) or -1 for random
    bool rf_refined;                // True if RF was used to refine imputation
    real_t mean_change;             // Mean change from RF refinement (0 if not refined)
};

struct ImputationResults {
    // Overall statistics
    integer_t n_samples;
    integer_t n_features;
    integer_t n_missing_total_before;
    integer_t n_missing_total_after;
    integer_t n_features_with_missing;
    integer_t n_iterations_used;
    
    // Method used
    std::string method;             // "rough" or "rand"
    bool use_gpu;
    bool is_sparse;
    
    // Per-column details
    std::vector<ColumnImputationInfo> column_info;
    
    // Feature classification
    std::vector<bool> is_categorical;
    
    // Timing
    double total_time_seconds;
    double initial_impute_time;
    double rf_refine_time;
    
    // Constructor
    ImputationResults() : 
        n_samples(0), n_features(0), 
        n_missing_total_before(0), n_missing_total_after(0),
        n_features_with_missing(0), n_iterations_used(0),
        method("rough"), use_gpu(false), is_sparse(false),
        total_time_seconds(0), initial_impute_time(0), rf_refine_time(0) {}
    
    // Generate formatted report string
    std::string to_string() const {
        std::ostringstream ss;
        ss << "\n";
        ss << "╔══════════════════════════════════════════════════════════════════════╗\n";
        ss << "║                       RFX IMPUTATION RESULTS                         ║\n";
        ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
        ss << "║ Dataset:    " << std::setw(8) << n_samples << " samples × " 
           << std::setw(6) << n_features << " features" << std::setw(24) << " ║\n";
        ss << "║ Method:     " << std::setw(10) << method;
        if (use_gpu) ss << " (GPU)";
        else ss << " (CPU)";
        if (is_sparse) ss << " [sparse]";
        else ss << " [dense]";
        ss << std::setw(20) << " ║\n";
        ss << "║ Iterations: " << std::setw(3) << n_iterations_used << std::setw(52) << " ║\n";
        ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
        ss << "║                         MISSING VALUES                               ║\n";
        ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
        
        double pct_before = 100.0 * n_missing_total_before / 
                           (static_cast<double>(n_samples) * n_features);
        double pct_after = 100.0 * n_missing_total_after / 
                          (static_cast<double>(n_samples) * n_features);
        
        ss << "║ BEFORE:     " << std::setw(10) << n_missing_total_before 
           << " (" << std::fixed << std::setprecision(2) << std::setw(6) << pct_before << "%)";
        ss << std::setw(32) << " ║\n";
        ss << "║ AFTER:      " << std::setw(10) << n_missing_total_after 
           << " (" << std::fixed << std::setprecision(2) << std::setw(6) << pct_after << "%)";
        ss << std::setw(32) << " ║\n";
        ss << "║ Features with missing: " << std::setw(5) << n_features_with_missing;
        ss << std::setw(40) << " ║\n";
        ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
        ss << "║  Col  │  Type   │ Missing Before │ Fill Method │ RF Refined │ After ║\n";
        ss << "╠═══════╪═════════╪════════════════╪═════════════╪════════════╪═══════╣\n";
        
        // Show only columns with missing values (up to 20 for readability)
        integer_t shown = 0;
        for (const auto& col : column_info) {
            if (col.missing_before > 0 && shown < 20) {
                ss << "║ " << std::setw(5) << col.column_idx << " │ ";
                ss << std::setw(7) << (col.is_categorical ? "cat" : "num") << " │ ";
                ss << std::setw(14) << col.missing_before << " │ ";
                ss << std::setw(11) << col.initial_method << " │ ";
                ss << std::setw(10) << (col.rf_refined ? "yes" : "no") << " │ ";
                ss << std::setw(5) << col.missing_after << " ║\n";
                shown++;
            }
        }
        
        if (n_features_with_missing > 20) {
            ss << "║  ... and " << (n_features_with_missing - 20) << " more columns with missing values";
            ss << std::setw(20) << " ║\n";
        }
        
        ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
        ss << "║                            TIMING                                    ║\n";
        ss << "╠══════════════════════════════════════════════════════════════════════╣\n";
        ss << "║ Initial imputation: " << std::fixed << std::setprecision(3) 
           << std::setw(8) << initial_impute_time << " sec";
        ss << std::setw(36) << " ║\n";
        ss << "║ RF refinement:      " << std::fixed << std::setprecision(3) 
           << std::setw(8) << rf_refine_time << " sec";
        ss << std::setw(36) << " ║\n";
        ss << "║ Total time:         " << std::fixed << std::setprecision(3) 
           << std::setw(8) << total_time_seconds << " sec";
        ss << std::setw(36) << " ║\n";
        ss << "╚══════════════════════════════════════════════════════════════════════╝\n";
        
        return ss.str();
    }
};

// ============================================================================
// IMPUTATION CONFIGURATION
// ============================================================================

struct ImputeConfig {
    // Method: "rough" (median/mode) or "rand" (random sampling)
    std::string method = "rough";
    
    // Number of RF refinement iterations (0 = no RF, just initial impute)
    integer_t n_iterations = 1;
    
    // RF parameters
    integer_t n_trees = 100;
    integer_t mtry = -1;         // -1 = auto (sqrt for classification, mdim/3 for regression)
    integer_t nodesize = 5;
    
    // Execution
    bool use_gpu = true;
    integer_t n_threads = 0;     // 0 = auto (all available)
    
    // Categorical detection
    integer_t max_categorical_unique = 10;  // Max unique values to consider categorical
    std::vector<bool> categorical_features; // Optional: explicit categorical markers
    
    // Convergence
    real_t convergence_threshold = 1e-6;
    
    // Verbosity
    bool verbose = true;
    
    // Random seed
    integer_t seed = 42;
};

// ============================================================================
// IMPUTATION FUNCTIONS - CPU IMPLEMENTATIONS (OpenMP parallelized)
// ============================================================================

// Main imputation function for dense data (CPU, OpenMP parallelized)
ImputationResults impute_missing_cpu_dense(
    real_t* X,                      // Data matrix (modified in place)
    integer_t n_samples,
    integer_t n_features,
    const ImputeConfig& config
);

// Imputation for sparse data (CPU, OpenMP parallelized)
// Note: Returns a NEW sparse matrix (original not modified)
std::pair<SparseMatrixCSR, ImputationResults> impute_missing_cpu_sparse(
    const SparseMatrixCSR& X,
    const ImputeConfig& config
);

// ============================================================================
// IMPUTATION FUNCTIONS - GPU IMPLEMENTATIONS (CUDA parallelized)
// ============================================================================

#ifdef CUDA_FOUND
// Main imputation function for dense data (GPU, CUDA parallelized)
ImputationResults impute_missing_gpu_dense(
    real_t* X,                      // Data matrix (modified in place)
    integer_t n_samples,
    integer_t n_features,
    const ImputeConfig& config
);

// Imputation for sparse data (GPU, CUDA parallelized)
std::pair<SparseMatrixCSR, ImputationResults> impute_missing_gpu_sparse(
    const SparseMatrixCSR& X,
    const ImputeConfig& config
);
#endif

// ============================================================================
// UNIFIED IMPUTATION INTERFACE - Auto-selects best implementation
// ============================================================================

// Dense imputation (auto-selects CPU/GPU based on config and availability)
ImputationResults impute_missing(
    real_t* X,
    integer_t n_samples,
    integer_t n_features,
    const ImputeConfig& config
);

// Sparse imputation (auto-selects CPU/GPU based on config and availability)
std::pair<SparseMatrixCSR, ImputationResults> impute_missing_sparse(
    const SparseMatrixCSR& X,
    const ImputeConfig& config
);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Detect categorical features based on unique value count
std::vector<bool> detect_categorical_features(
    const real_t* X,
    integer_t n_samples,
    integer_t n_features,
    integer_t max_unique = 10
);

// Detect categorical features in sparse matrix
std::vector<bool> detect_categorical_features_sparse(
    const SparseMatrixCSR& X,
    integer_t max_unique = 10
);

// Initial imputation: median (numeric) or mode (categorical)
void rough_impute_column(
    real_t* column_data,
    integer_t n_samples,
    const std::vector<bool>& missing_mask,
    bool is_categorical,
    real_t& fill_value     // Output: value used for fill
);

// Initial imputation: random sampling from observed values
void random_impute_column(
    real_t* column_data,
    integer_t n_samples,
    const std::vector<bool>& missing_mask,
    integer_t seed,
    real_t& fill_value     // Output: -1 (indicates random sampling)
);

} // namespace rf

#endif // RF_IMPUTE_HPP

