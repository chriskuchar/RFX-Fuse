/**
 * RFX Imputation: Random Forest-based Missing Value Imputation
 * =============================================================
 * 
 * CPU implementation with OpenMP parallelization for dense and sparse matrices.
 * 
 * Implements imputation methods from:
 *   Joshua Young (2017). "New Imputation Methods for Random Forests"
 *   Master's Project, Utah State University
 *   Major Professor: Dr. Adele Cutler (co-creator of Random Forest)
 * 
 * Methods:
 *   - rough: Initial median/mode imputation, then RF refinement
 *   - rand: Initial random sampling imputation, then RF refinement
 */

#include "rf_impute.hpp"
#include "rf_random_forest.hpp"
#include "rf_types.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <set>
#include <map>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rf {

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Detect categorical features based on unique value count
std::vector<bool> detect_categorical_features(
    const real_t* X,
    integer_t n_samples,
    integer_t n_features,
    integer_t max_unique
) {
    std::vector<bool> is_categorical(n_features, false);
    
    #pragma omp parallel for schedule(dynamic)
    for (integer_t j = 0; j < n_features; ++j) {
        std::set<real_t> unique_values;
        bool all_integers = true;
        
        for (integer_t i = 0; i < n_samples; ++i) {
            real_t val = X[i * n_features + j];
            if (!std::isnan(val)) {
                unique_values.insert(val);
                // Check if value is an integer
                if (std::fabs(val - std::round(val)) > 1e-6f) {
                    all_integers = false;
                }
                if (unique_values.size() > static_cast<size_t>(max_unique)) {
                    break;
                }
            }
        }
        
        // Categorical if few unique values AND all are integers
        is_categorical[j] = (unique_values.size() <= static_cast<size_t>(max_unique)) && all_integers;
    }
    
    return is_categorical;
}

// Detect categorical features in sparse matrix
std::vector<bool> detect_categorical_features_sparse(
    const SparseMatrixCSR& X,
    integer_t max_unique
) {
    std::vector<bool> is_categorical(X.ncols, false);
    
    #pragma omp parallel for schedule(dynamic)
    for (integer_t j = 0; j < X.ncols; ++j) {
        std::set<real_t> unique_values;
        unique_values.insert(0.0f);  // Zero is implicitly present in sparse
        bool all_integers = true;
        
        // Scan all rows for this column
        for (integer_t i = 0; i < X.nrows; ++i) {
            integer_t start = X.indptr[i];
            integer_t end = X.indptr[i + 1];
            
            for (integer_t k = start; k < end; ++k) {
                if (X.indices[k] == j) {
                    real_t val = X.data[k];
                    if (!std::isnan(val)) {
                        unique_values.insert(val);
                        if (std::fabs(val - std::round(val)) > 1e-6f) {
                            all_integers = false;
                        }
                    }
                    break;
                }
            }
            
            if (unique_values.size() > static_cast<size_t>(max_unique)) {
                break;
            }
        }
        
        is_categorical[j] = (unique_values.size() <= static_cast<size_t>(max_unique)) && all_integers;
    }
    
    return is_categorical;
}

// Compute median of non-NaN values
static real_t compute_median(const std::vector<real_t>& valid_values) {
    if (valid_values.empty()) return 0.0f;
    
    std::vector<real_t> sorted = valid_values;
    std::sort(sorted.begin(), sorted.end());
    
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0f;
    } else {
        return sorted[n/2];
    }
}

// Compute mode (most frequent value)
static real_t compute_mode(const std::vector<real_t>& valid_values) {
    if (valid_values.empty()) return 0.0f;
    
    std::unordered_map<int32_t, integer_t> counts;
    for (real_t val : valid_values) {
        int32_t key = static_cast<int32_t>(std::round(val));
        counts[key]++;
    }
    
    integer_t max_count = 0;
    int32_t mode_key = 0;
    for (const auto& pair : counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            mode_key = pair.first;
        }
    }
    
    return static_cast<real_t>(mode_key);
}

// Initial imputation: median (numeric) or mode (categorical)
void rough_impute_column(
    real_t* column_data,
    integer_t n_samples,
    const std::vector<bool>& missing_mask,
    bool is_categorical,
    real_t& fill_value
) {
    // Collect valid (non-missing) values
    std::vector<real_t> valid_values;
    valid_values.reserve(n_samples);
    
    for (integer_t i = 0; i < n_samples; ++i) {
        if (!missing_mask[i]) {
            valid_values.push_back(column_data[i]);
        }
    }
    
    // Compute fill value
    if (valid_values.empty()) {
        fill_value = 0.0f;
    } else if (is_categorical) {
        fill_value = compute_mode(valid_values);
    } else {
        fill_value = compute_median(valid_values);
    }
    
    // Fill missing values
    for (integer_t i = 0; i < n_samples; ++i) {
        if (missing_mask[i]) {
            column_data[i] = fill_value;
        }
    }
}

// Initial imputation: random sampling from observed values
void random_impute_column(
    real_t* column_data,
    integer_t n_samples,
    const std::vector<bool>& missing_mask,
    integer_t seed,
    real_t& fill_value
) {
    fill_value = -1.0f;  // Indicates random sampling
    
    // Collect valid (non-missing) values
    std::vector<real_t> valid_values;
    valid_values.reserve(n_samples);
    
    for (integer_t i = 0; i < n_samples; ++i) {
        if (!missing_mask[i]) {
            valid_values.push_back(column_data[i]);
        }
    }
    
    if (valid_values.empty()) {
        // All missing - fill with 0
        for (integer_t i = 0; i < n_samples; ++i) {
            if (missing_mask[i]) {
                column_data[i] = 0.0f;
            }
        }
        return;
    }
    
    // Random sampling from observed values
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, valid_values.size() - 1);
    
    for (integer_t i = 0; i < n_samples; ++i) {
        if (missing_mask[i]) {
            column_data[i] = valid_values[dist(rng)];
        }
    }
}

// ============================================================================
// CPU DENSE IMPUTATION (OpenMP parallelized)
// ============================================================================

ImputationResults impute_missing_cpu_dense(
    real_t* X,
    integer_t n_samples,
    integer_t n_features,
    const ImputeConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ImputationResults results;
    results.n_samples = n_samples;
    results.n_features = n_features;
    results.method = config.method;
    results.use_gpu = false;
    results.is_sparse = false;
    results.n_iterations_used = config.n_iterations;
    
    // Set number of threads
    #ifdef _OPENMP
    if (config.n_threads > 0) {
        omp_set_num_threads(config.n_threads);
    }
    #endif
    
    // Step 1: Find missing values and detect categorical features
    // Use uint8_t instead of bool for thread safety (vector<bool> uses bit packing)
    std::vector<uint8_t> missing_mask(n_samples * n_features, 0);
    std::vector<integer_t> missing_count_per_col(n_features, 0);
    integer_t total_missing = 0;
    
    #pragma omp parallel for reduction(+:total_missing)
    for (integer_t j = 0; j < n_features; ++j) {
        integer_t col_missing = 0;
        for (integer_t i = 0; i < n_samples; ++i) {
            if (std::isnan(X[i * n_features + j])) {
                missing_mask[i * n_features + j] = 1;
                col_missing++;
            }
        }
        missing_count_per_col[j] = col_missing;
        total_missing += col_missing;
    }
    
    results.n_missing_total_before = total_missing;
    
    if (total_missing == 0) {
        // No missing values
        if (config.verbose) {
            std::cout << "No missing values found - returning original data\n";
        }
        results.n_missing_total_after = 0;
        return results;
    }
    
    // Detect categorical features
    std::vector<bool> is_categorical;
    if (config.categorical_features.empty()) {
        is_categorical = detect_categorical_features(X, n_samples, n_features, config.max_categorical_unique);
    } else {
        is_categorical = config.categorical_features;
    }
    results.is_categorical = is_categorical;
    
    // Find features with missing values (sorted by count for stability)
    std::vector<integer_t> features_with_missing;
    for (integer_t j = 0; j < n_features; ++j) {
        if (missing_count_per_col[j] > 0) {
            features_with_missing.push_back(j);
        }
    }
    results.n_features_with_missing = features_with_missing.size();
    
    // Sort by missing count (ascending) for better stability
    std::sort(features_with_missing.begin(), features_with_missing.end(),
              [&](integer_t a, integer_t b) { return missing_count_per_col[a] < missing_count_per_col[b]; });
    
    // Initialize column info
    results.column_info.resize(n_features);
    for (integer_t j = 0; j < n_features; ++j) {
        results.column_info[j].column_idx = j;
        results.column_info[j].missing_before = missing_count_per_col[j];
        results.column_info[j].is_categorical = is_categorical[j];
        results.column_info[j].rf_refined = false;
        results.column_info[j].mean_change = 0.0f;
    }
    
    if (config.verbose) {
        std::cout << "RFX Imputation (CPU Dense, " << config.method << ")\n";
        std::cout << "  Samples: " << n_samples << ", Features: " << n_features << "\n";
        std::cout << "  Total missing: " << total_missing << " (" 
                  << (100.0 * total_missing / (n_samples * n_features)) << "%)\n";
        std::cout << "  Features with missing: " << features_with_missing.size() << "\n";
        #ifdef _OPENMP
        std::cout << "  Threads: " << omp_get_max_threads() << "\n";
        #endif
    }
    
    // Step 2: Initial imputation (OpenMP parallelized across columns)
    auto initial_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < features_with_missing.size(); ++idx) {
        integer_t j = features_with_missing[idx];
        
        // Extract column data
        std::vector<real_t> col_data(n_samples);
        std::vector<bool> col_missing_bool(n_samples);
        for (integer_t i = 0; i < n_samples; ++i) {
            col_data[i] = X[i * n_features + j];
            col_missing_bool[i] = (missing_mask[i * n_features + j] != 0);
        }
        
        real_t fill_value;
        if (config.method == "rand") {
            random_impute_column(col_data.data(), n_samples, col_missing_bool,
                                config.seed + j, fill_value);
            results.column_info[j].initial_method = "random";
        } else {
            rough_impute_column(col_data.data(), n_samples, col_missing_bool,
                               is_categorical[j], fill_value);
            results.column_info[j].initial_method = is_categorical[j] ? "mode" : "median";
        }
        results.column_info[j].initial_fill_value = fill_value;
        
        // Write back to X
        for (integer_t i = 0; i < n_samples; ++i) {
            X[i * n_features + j] = col_data[i];
        }
    }
    
    auto initial_end = std::chrono::high_resolution_clock::now();
    results.initial_impute_time = std::chrono::duration<double>(initial_end - initial_start).count();
    
    if (config.verbose) {
        std::cout << "  Initial imputation: " << config.method << " (" 
                  << results.initial_impute_time << "s)\n";
    }
    
    // Step 3: RF refinement (if n_iterations > 0)
    if (config.n_iterations > 0 && !features_with_missing.empty()) {
        auto rf_start = std::chrono::high_resolution_clock::now();
        
        for (integer_t iter = 0; iter < config.n_iterations; ++iter) {
            if (config.verbose) {
                std::cout << "\n  RF Iteration " << (iter + 1) << "/" << config.n_iterations << "\n";
            }
            
            real_t total_change = 0.0f;
            
            // Process each feature with missing values
            // Note: Cannot parallelize this loop due to data dependencies
            for (integer_t j : features_with_missing) {
                integer_t n_missing_j = missing_count_per_col[j];
                if (n_missing_j == 0) continue;
                
                // Prepare training data: samples without missing in this feature
                std::vector<integer_t> train_indices;
                std::vector<integer_t> test_indices;
                train_indices.reserve(n_samples);
                test_indices.reserve(n_missing_j);
                
                for (integer_t i = 0; i < n_samples; ++i) {
                    if (missing_mask[i * n_features + j]) {
                        test_indices.push_back(i);
                    } else {
                        train_indices.push_back(i);
                    }
                }
                
                integer_t n_train = train_indices.size();
                if (n_train < 10) {
                    if (config.verbose) {
                        std::cout << "    Feature " << j << ": Skipped (only " << n_train << " training samples)\n";
                    }
                    continue;
                }
                
                // Build feature matrix for training (exclude current feature)
                integer_t n_pred_features = n_features - 1;
                std::vector<real_t> X_train(n_train * n_pred_features);
                std::vector<real_t> y_train(n_train);
                
                #pragma omp parallel for
                for (integer_t ti = 0; ti < n_train; ++ti) {
                    integer_t i = train_indices[ti];
                    integer_t feat_idx = 0;
                    for (integer_t f = 0; f < n_features; ++f) {
                        if (f != j) {
                            X_train[ti * n_pred_features + feat_idx] = X[i * n_features + f];
                            feat_idx++;
                        }
                    }
                    y_train[ti] = X[i * n_features + j];
                }
                
                // Build feature matrix for prediction
                integer_t n_test = test_indices.size();
                std::vector<real_t> X_test(n_test * n_pred_features);
                
                #pragma omp parallel for
                for (integer_t ti = 0; ti < n_test; ++ti) {
                    integer_t i = test_indices[ti];
                    integer_t feat_idx = 0;
                    for (integer_t f = 0; f < n_features; ++f) {
                        if (f != j) {
                            X_test[ti * n_pred_features + feat_idx] = X[i * n_features + f];
                            feat_idx++;
                        }
                    }
                }
                
                // Train RF and predict
                RandomForestConfig rf_config;
                rf_config.nsample = n_train;
                rf_config.mdim = n_pred_features;
                rf_config.ntree = config.n_trees;
                rf_config.nodesize = config.nodesize;
                rf_config.use_gpu = false;  // CPU mode
                rf_config.compute_importance = false;
                rf_config.compute_proximity = false;
                
                // Set mtry
                if (config.mtry > 0) {
                    rf_config.mtry = config.mtry;
                } else {
                    rf_config.mtry = std::max(1, static_cast<integer_t>(std::sqrt(n_pred_features)));
                }
                
                std::vector<real_t> predictions(n_test);
                
                try {
                    if (is_categorical[j]) {
                        // Classification
                        rf_config.task_type = TaskType::CLASSIFICATION;

                        // Get unique classes
                        std::set<integer_t> classes_set;
                        for (real_t v : y_train) {
                            classes_set.insert(static_cast<integer_t>(v));
                        }
                        std::vector<integer_t> classes_vec(classes_set.begin(), classes_set.end());
                        rf_config.nclass = classes_vec.size();

                        if (classes_vec.size() < 2) {
                            // Only 1 unique class — fill with that value
                            for (integer_t i = 0; i < n_test; ++i) {
                                predictions[i] = static_cast<real_t>(classes_vec[0]);
                            }
                        } else {
                            // Remap raw labels to 0-based for fit_classification
                            std::map<integer_t, integer_t> label_to_idx;
                            for (integer_t idx = 0; idx < static_cast<integer_t>(classes_vec.size()); ++idx) {
                                label_to_idx[classes_vec[idx]] = idx;
                            }

                            std::vector<integer_t> y_train_int(n_train);
                            for (integer_t i = 0; i < n_train; ++i) {
                                y_train_int[i] = label_to_idx[static_cast<integer_t>(y_train[i])];
                            }

                            RandomForest rf(rf_config);
                            rf.fit_classification(X_train.data(), y_train_int.data(), nullptr);

                            std::vector<integer_t> pred_int(n_test);
                            rf.predict_classification(X_test.data(), n_test, pred_int.data());

                            // predict_classification returns original labels (reverse-mapped internally)
                            for (integer_t i = 0; i < n_test; ++i) {
                                predictions[i] = static_cast<real_t>(pred_int[i]);
                            }
                        }
                    } else {
                        // Regression
                        rf_config.task_type = TaskType::REGRESSION;
                        rf_config.nclass = 1;
                        
                        RandomForest rf(rf_config);
                        rf.fit_regression(X_train.data(), y_train.data(), nullptr);
                        rf.predict_regression(X_test.data(), n_test, predictions.data());
                    }
                    
                    // Calculate change and update values
                    real_t change = 0.0f;
                    for (integer_t ti = 0; ti < n_test; ++ti) {
                        integer_t i = test_indices[ti];
                        real_t old_val = X[i * n_features + j];
                        real_t new_val = predictions[ti];
                        change += std::fabs(new_val - old_val);
                        X[i * n_features + j] = new_val;
                    }
                    
                    real_t mean_change = change / n_test;
                    total_change += mean_change;
                    results.column_info[j].rf_refined = true;
                    results.column_info[j].mean_change = mean_change;
                    
                    if (config.verbose) {
                        const char* type_str = is_categorical[j] ? "cat" : "num";
                        std::cout << "    Feature " << j << " (" << type_str << "): imputed " 
                                  << n_test << " values, mean change=" << mean_change << "\n";
                    }
                    
                } catch (const std::exception& e) {
                    if (config.verbose) {
                        std::cout << "    Feature " << j << ": Error - " << e.what() << "\n";
                    }
                }
            }
            
            if (config.verbose) {
                std::cout << "  Total change this iteration: " << total_change << "\n";
            }
            
            // Check convergence
            if (total_change < config.convergence_threshold && iter > 0) {
                if (config.verbose) {
                    std::cout << "  Converged!\n";
                }
                results.n_iterations_used = iter + 1;
                break;
            }
        }
        
        auto rf_end = std::chrono::high_resolution_clock::now();
        results.rf_refine_time = std::chrono::duration<double>(rf_end - rf_start).count();
    }
    
    // Count remaining missing (should be 0)
    integer_t remaining_missing = 0;
    #pragma omp parallel for reduction(+:remaining_missing)
    for (integer_t j = 0; j < n_features; ++j) {
        integer_t col_missing = 0;
        for (integer_t i = 0; i < n_samples; ++i) {
            if (std::isnan(X[i * n_features + j])) {
                col_missing++;
            }
        }
        results.column_info[j].missing_after = col_missing;
        remaining_missing += col_missing;
    }
    results.n_missing_total_after = remaining_missing;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    if (config.verbose) {
        std::cout << results.to_string();
    }
    
    return results;
}

// ============================================================================
// CPU SPARSE IMPUTATION (OpenMP parallelized)
// ============================================================================

std::pair<SparseMatrixCSR, ImputationResults> impute_missing_cpu_sparse(
    const SparseMatrixCSR& X,
    const ImputeConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ImputationResults results;
    results.n_samples = X.nrows;
    results.n_features = X.ncols;
    results.method = config.method;
    results.use_gpu = false;
    results.is_sparse = true;
    results.n_iterations_used = config.n_iterations;
    
    // For sparse matrices, NaN is typically not used - we treat explicit NaN entries as missing
    // First, convert sparse to dense for imputation (sparse with NaN is rare)
    std::vector<real_t> X_dense(X.nrows * X.ncols, 0.0f);
    std::vector<bool> has_value(X.nrows * X.ncols, false);
    
    // Fill dense matrix from sparse
    #pragma omp parallel for
    for (integer_t i = 0; i < X.nrows; ++i) {
        integer_t start = X.indptr[i];
        integer_t end = X.indptr[i + 1];
        for (integer_t k = start; k < end; ++k) {
            integer_t j = X.indices[k];
            real_t val = X.data[k];
            X_dense[i * X.ncols + j] = val;
            has_value[i * X.ncols + j] = true;
        }
    }
    
    // Check for explicit NaN values in sparse entries
    integer_t n_explicit_nan = 0;
    for (integer_t k = 0; k < X.nnz; ++k) {
        if (std::isnan(X.data[k])) {
            n_explicit_nan++;
        }
    }
    
    if (n_explicit_nan == 0) {
        // No missing values in sparse matrix
        if (config.verbose) {
            std::cout << "No NaN values in sparse matrix - returning original\n";
        }
        results.n_missing_total_before = 0;
        results.n_missing_total_after = 0;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
        // Return copy of original
        SparseMatrixCSR X_out = X;
        return std::make_pair(X_out, results);
    }
    
    // Mark NaN entries as missing and impute using dense method
    for (integer_t i = 0; i < X.nrows; ++i) {
        integer_t start = X.indptr[i];
        integer_t end = X.indptr[i + 1];
        for (integer_t k = start; k < end; ++k) {
            integer_t j = X.indices[k];
            if (std::isnan(X.data[k])) {
                X_dense[i * X.ncols + j] = std::nanf("");  // Mark as NaN for imputation
            }
        }
    }
    
    // Perform dense imputation
    ImputationResults dense_results = impute_missing_cpu_dense(
        X_dense.data(), X.nrows, X.ncols, config);
    
    // Copy results
    results = dense_results;
    results.is_sparse = true;
    
    // Convert back to sparse (only store non-zero values, excluding imputed zeros)
    std::vector<real_t> new_data;
    std::vector<integer_t> new_indices;
    std::vector<integer_t> new_indptr(X.nrows + 1, 0);
    
    for (integer_t i = 0; i < X.nrows; ++i) {
        new_indptr[i] = new_data.size();
        for (integer_t j = 0; j < X.ncols; ++j) {
            real_t val = X_dense[i * X.ncols + j];
            // Keep original sparsity pattern plus imputed values
            if (has_value[i * X.ncols + j] || val != 0.0f) {
                new_data.push_back(val);
                new_indices.push_back(j);
            }
        }
    }
    new_indptr[X.nrows] = new_data.size();
    
    // Build output sparse matrix
    SparseMatrixCSR X_out;
    X_out.nrows = X.nrows;
    X_out.ncols = X.ncols;
    X_out.nnz = new_data.size();
    X_out.data = std::move(new_data);
    X_out.indices = std::move(new_indices);
    X_out.indptr = std::move(new_indptr);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    return std::make_pair(X_out, results);
}

// ============================================================================
// UNIFIED INTERFACE - Auto-selects CPU/GPU
// ============================================================================

ImputationResults impute_missing(
    real_t* X,
    integer_t n_samples,
    integer_t n_features,
    const ImputeConfig& config
) {
    #ifdef CUDA_FOUND
    if (config.use_gpu) {
        // Check if GPU is available
        extern bool cuda_is_available_check();
        if (cuda_is_available_check()) {
            return impute_missing_gpu_dense(X, n_samples, n_features, config);
        }
    }
    #endif
    
    // Fall back to CPU
    return impute_missing_cpu_dense(X, n_samples, n_features, config);
}

std::pair<SparseMatrixCSR, ImputationResults> impute_missing_sparse(
    const SparseMatrixCSR& X,
    const ImputeConfig& config
) {
    #ifdef CUDA_FOUND
    if (config.use_gpu) {
        extern bool cuda_is_available_check();
        if (cuda_is_available_check()) {
            return impute_missing_gpu_sparse(X, config);
        }
    }
    #endif
    
    // Fall back to CPU
    return impute_missing_cpu_sparse(X, config);
}

} // namespace rf

