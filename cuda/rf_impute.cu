/**
 * RFX Imputation: GPU CUDA-Parallelized Missing Value Imputation
 * ===============================================================
 * 
 * GPU implementation with CUDA parallelization for dense and sparse matrices.
 * 
 * Implements imputation methods from:
 *   Joshua Young (2017). "New Imputation Methods for Random Forests"
 *   Master's Project, Utah State University
 *   Major Professor: Dr. Adele Cutler (co-creator of Random Forest)
 */

// Avoid fp8/fp16 conflicts - must be before other CUDA includes
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_BFLOAT16_CONVERSIONS__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "rf_impute.cuh"
#include "../include/rf_impute.hpp"
#include "../include/rf_random_forest.hpp"
#include <chrono>
#include <iostream>
#include <set>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace rf {
namespace cuda {

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
        } \
    } while(0)

// ============================================================================
// KERNEL IMPLEMENTATIONS - DENSE MATRIX
// ============================================================================

// Kernel to find NaN values and compute missing mask
__global__ void find_missing_kernel(
    const float* X,
    bool* missing_mask,
    int* missing_count_per_col,
    int n_samples,
    int n_features
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    
    if (j >= n_features) return;
    
    int col_missing = 0;
    for (int i = 0; i < n_samples; ++i) {
        int idx = i * n_features + j;
        bool is_missing = isnan(X[idx]);
        missing_mask[idx] = is_missing;
        if (is_missing) col_missing++;
    }
    
    missing_count_per_col[j] = col_missing;
}

// Kernel to detect categorical features
__global__ void detect_categorical_kernel(
    const float* X,
    bool* is_categorical,
    int* unique_counts,
    int n_samples,
    int n_features,
    int max_unique
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_features) return;
    
    // Simple approach: check if all values are integers and count unique
    // This is a simplified version - production would use hash maps
    bool all_integers = true;
    int n_unique = 0;
    float seen[32];  // Max 32 unique values tracked
    
    for (int i = 0; i < n_samples && n_unique <= max_unique; ++i) {
        float val = X[i * n_features + j];
        if (!isnan(val)) {
            // Check if integer
            if (fabsf(val - roundf(val)) > 1e-6f) {
                all_integers = false;
            }
            
            // Check if already seen
            bool found = false;
            for (int k = 0; k < n_unique && k < 32; ++k) {
                if (fabsf(seen[k] - val) < 1e-6f) {
                    found = true;
                    break;
                }
            }
            
            if (!found && n_unique < 32) {
                seen[n_unique] = val;
                n_unique++;
            }
        }
    }
    
    is_categorical[j] = (n_unique <= max_unique) && all_integers;
    unique_counts[j] = n_unique;
}

// Kernel to compute fill values (median for numeric, mode for categorical)
// This kernel is launched per column with enough threads to cover all samples
__global__ void compute_column_fill_values_kernel(
    const float* sorted_valid_values,
    const int* valid_counts,
    const int* valid_offsets,
    const bool* is_categorical,
    float* fill_values,
    int n_features
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_features) return;
    
    int count = valid_counts[j];
    if (count == 0) {
        fill_values[j] = 0.0f;
        return;
    }
    
    int offset = valid_offsets[j];
    
    if (is_categorical[j]) {
        // Mode: find most frequent value (values are sorted, count consecutive)
        float mode_val = sorted_valid_values[offset];
        int mode_count = 1;
        float current_val = mode_val;
        int current_count = 1;
        
        for (int i = 1; i < count; ++i) {
            float val = sorted_valid_values[offset + i];
            if (fabsf(val - current_val) < 1e-6f) {
                current_count++;
            } else {
                if (current_count > mode_count) {
                    mode_val = current_val;
                    mode_count = current_count;
                }
                current_val = val;
                current_count = 1;
            }
        }
        if (current_count > mode_count) {
            mode_val = current_val;
        }
        
        fill_values[j] = mode_val;
    } else {
        // Median: middle value of sorted array
        if (count % 2 == 0) {
            fill_values[j] = (sorted_valid_values[offset + count/2 - 1] + 
                             sorted_valid_values[offset + count/2]) / 2.0f;
        } else {
            fill_values[j] = sorted_valid_values[offset + count/2];
        }
    }
}

// Kernel to apply initial imputation (rough: median/mode)
__global__ void rough_impute_kernel(
    float* X,
    const bool* missing_mask,
    const float* fill_values,
    int n_samples,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_samples * n_features;
    
    if (idx >= total) return;
    
    if (missing_mask[idx]) {
        int j = idx % n_features;  // Column index
        X[idx] = fill_values[j];
    }
}

// Kernel for random imputation using cuRAND
__global__ void random_impute_kernel(
    float* X,
    const bool* missing_mask,
    const float* valid_values,
    const int* valid_counts,
    const int* valid_offsets,
    unsigned int seed,
    int n_samples,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_samples * n_features;
    
    if (idx >= total) return;
    
    if (missing_mask[idx]) {
        int j = idx % n_features;
        int count = valid_counts[j];
        
        if (count == 0) {
            X[idx] = 0.0f;
        } else {
            // Initialize cuRAND state
            curandState state;
            curand_init(seed, idx, 0, &state);
            
            // Sample random index
            int random_idx = curand(&state) % count;
            X[idx] = valid_values[valid_offsets[j] + random_idx];
        }
    }
}

// ============================================================================
// SPARSE MATRIX KERNELS
// ============================================================================

__global__ void find_missing_sparse_kernel(
    const float* data,
    const int* indices,
    const int* indptr,
    bool* missing_mask,
    int* missing_count_per_col,
    int nrows,
    int ncols,
    int nnz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k >= nnz) return;
    
    bool is_missing = isnan(data[k]);
    missing_mask[k] = is_missing;
    
    if (is_missing) {
        int col = indices[k];
        ::atomicAdd(&missing_count_per_col[col], 1);
    }
}

__global__ void impute_sparse_kernel(
    float* data,
    const int* indices,
    const int* indptr,
    const bool* missing_mask,
    const float* fill_values,
    int nrows,
    int ncols,
    int nnz
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k >= nnz) return;
    
    if (missing_mask[k]) {
        int col = indices[k];
        data[k] = fill_values[col];
    }
}

} // namespace cuda

// ============================================================================
// GPU DENSE IMPUTATION - Main Implementation
// ============================================================================

ImputationResults impute_missing_gpu_dense(
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
    results.use_gpu = true;
    results.is_sparse = false;
    results.n_iterations_used = config.n_iterations;
    
    size_t total_elements = static_cast<size_t>(n_samples) * n_features;
    
    // Allocate device memory
    float* d_X;
    bool* d_missing_mask;
    int* d_missing_count;
    bool* d_is_categorical;
    float* d_fill_values;
    
    CUDA_CHECK(cudaMalloc(&d_X, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_missing_mask, total_elements * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_missing_count, n_features * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_categorical, n_features * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_fill_values, n_features * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X, X, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_missing_count, 0, n_features * sizeof(int)));
    
    // Step 1: Find missing values
    int block_size = 256;
    int grid_size = (n_features + block_size - 1) / block_size;
    
    cuda::find_missing_kernel<<<grid_size, block_size>>>(
        d_X, d_missing_mask, d_missing_count, n_samples, n_features);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy missing counts back to host
    std::vector<int> h_missing_count(n_features);
    CUDA_CHECK(cudaMemcpy(h_missing_count.data(), d_missing_count, 
                          n_features * sizeof(int), cudaMemcpyDeviceToHost));
    
    integer_t total_missing = 0;
    for (integer_t j = 0; j < n_features; ++j) {
        total_missing += h_missing_count[j];
    }
    results.n_missing_total_before = total_missing;
    
    if (total_missing == 0) {
        if (config.verbose) {
            std::cout << "No missing values found - returning original data\n";
        }
        cudaFree(d_X);
        cudaFree(d_missing_mask);
        cudaFree(d_missing_count);
        cudaFree(d_is_categorical);
        cudaFree(d_fill_values);
        results.n_missing_total_after = 0;
        return results;
    }
    
    // Step 2: Detect categorical features
    int* d_unique_counts;
    CUDA_CHECK(cudaMalloc(&d_unique_counts, n_features * sizeof(int)));
    
    cuda::detect_categorical_kernel<<<grid_size, block_size>>>(
        d_X, d_is_categorical, d_unique_counts, n_samples, n_features, config.max_categorical_unique);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy categorical info to host - use char instead of bool for .data() support
    std::vector<char> h_is_categorical_raw(n_features);
    CUDA_CHECK(cudaMemcpy(h_is_categorical_raw.data(), d_is_categorical,
                          n_features * sizeof(bool), cudaMemcpyDeviceToHost));
    std::vector<bool> h_is_categorical(n_features);
    for (integer_t j = 0; j < n_features; ++j) {
        h_is_categorical[j] = h_is_categorical_raw[j] != 0;
    }
    results.is_categorical = h_is_categorical;
    
    // Count features with missing
    std::vector<integer_t> features_with_missing;
    for (integer_t j = 0; j < n_features; ++j) {
        if (h_missing_count[j] > 0) {
            features_with_missing.push_back(j);
        }
    }
    results.n_features_with_missing = features_with_missing.size();
    
    // Initialize column info
    results.column_info.resize(n_features);
    for (integer_t j = 0; j < n_features; ++j) {
        results.column_info[j].column_idx = j;
        results.column_info[j].missing_before = h_missing_count[j];
        results.column_info[j].is_categorical = h_is_categorical[j];
        results.column_info[j].rf_refined = false;
        results.column_info[j].mean_change = 0.0f;
    }
    
    if (config.verbose) {
        std::cout << "RFX Imputation (GPU Dense, " << config.method << ")\n";
        std::cout << "  Samples: " << n_samples << ", Features: " << n_features << "\n";
        std::cout << "  Total missing: " << total_missing << " (" 
                  << (100.0 * total_missing / total_elements) << "%)\n";
        std::cout << "  Features with missing: " << features_with_missing.size() << "\n";
    }
    
    // Step 3: Initial imputation
    auto initial_start = std::chrono::high_resolution_clock::now();
    
    // Collect valid values for each column on host, then upload to GPU
    // This is more efficient than doing complex reductions on GPU for median/mode
    std::vector<float> h_X(total_elements);
    std::vector<char> h_missing_mask_raw(total_elements);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_missing_mask_raw.data(), d_missing_mask, total_elements * sizeof(bool), cudaMemcpyDeviceToHost));
    
    // Convert to vector<bool> for convenient access
    std::vector<bool> h_missing_mask(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
        h_missing_mask[i] = h_missing_mask_raw[i] != 0;
    }
    
    // Compute fill values on CPU (median/mode computation is complex on GPU)
    std::vector<float> h_fill_values(n_features);
    
    #pragma omp parallel for
    for (integer_t j = 0; j < n_features; ++j) {
        std::vector<float> valid_values;
        valid_values.reserve(n_samples);
        
        for (integer_t i = 0; i < n_samples; ++i) {
            if (!h_missing_mask[i * n_features + j]) {
                valid_values.push_back(h_X[i * n_features + j]);
            }
        }
        
        if (valid_values.empty()) {
            h_fill_values[j] = 0.0f;
        } else if (h_is_categorical[j]) {
            // Mode
            std::unordered_map<int, int> counts;
            for (float v : valid_values) {
                counts[static_cast<int>(roundf(v))]++;
            }
            int max_count = 0;
            int mode_val = 0;
            for (auto& p : counts) {
                if (p.second > max_count) {
                    max_count = p.second;
                    mode_val = p.first;
                }
            }
            h_fill_values[j] = static_cast<float>(mode_val);
        } else {
            // Median
            std::sort(valid_values.begin(), valid_values.end());
            size_t n = valid_values.size();
            if (n % 2 == 0) {
                h_fill_values[j] = (valid_values[n/2 - 1] + valid_values[n/2]) / 2.0f;
            } else {
                h_fill_values[j] = valid_values[n/2];
            }
        }
        
        results.column_info[j].initial_method = h_is_categorical[j] ? "mode" : "median";
        results.column_info[j].initial_fill_value = h_fill_values[j];
    }
    
    // Upload fill values to GPU
    CUDA_CHECK(cudaMemcpy(d_fill_values, h_fill_values.data(), 
                          n_features * sizeof(float), cudaMemcpyHostToDevice));
    
    // Apply imputation on GPU (highly parallel)
    int total_grid = (total_elements + block_size - 1) / block_size;
    
    if (config.method == "rand") {
        // For random imputation, need to collect valid values per column
        // Build flattened valid values array
        std::vector<float> all_valid_values;
        std::vector<int> valid_counts(n_features);
        std::vector<int> valid_offsets(n_features);
        
        for (integer_t j = 0; j < n_features; ++j) {
            valid_offsets[j] = all_valid_values.size();
            for (integer_t i = 0; i < n_samples; ++i) {
                if (!h_missing_mask[i * n_features + j]) {
                    all_valid_values.push_back(h_X[i * n_features + j]);
                }
            }
            valid_counts[j] = all_valid_values.size() - valid_offsets[j];
        }
        
        // Upload to GPU
        float* d_valid_values;
        int* d_valid_counts;
        int* d_valid_offsets;
        
        CUDA_CHECK(cudaMalloc(&d_valid_values, all_valid_values.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_valid_counts, n_features * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valid_offsets, n_features * sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(d_valid_values, all_valid_values.data(), 
                              all_valid_values.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valid_counts, valid_counts.data(), 
                              n_features * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_valid_offsets, valid_offsets.data(), 
                              n_features * sizeof(int), cudaMemcpyHostToDevice));
        
        cuda::random_impute_kernel<<<total_grid, block_size>>>(
            d_X, d_missing_mask, d_valid_values, d_valid_counts, d_valid_offsets,
            config.seed, n_samples, n_features);
        
        cudaFree(d_valid_values);
        cudaFree(d_valid_counts);
        cudaFree(d_valid_offsets);
        
        for (integer_t j = 0; j < n_features; ++j) {
            results.column_info[j].initial_method = "random";
            results.column_info[j].initial_fill_value = -1.0f;
        }
    } else {
        // Rough imputation
        cuda::rough_impute_kernel<<<total_grid, block_size>>>(
            d_X, d_missing_mask, d_fill_values, n_samples, n_features);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto initial_end = std::chrono::high_resolution_clock::now();
    results.initial_impute_time = std::chrono::duration<double>(initial_end - initial_start).count();
    
    if (config.verbose) {
        std::cout << "  Initial imputation: " << config.method << " (" 
                  << results.initial_impute_time << "s)\n";
    }
    
    // Copy imputed data back to host for RF refinement
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Step 4: RF refinement (if n_iterations > 0)
    if (config.n_iterations > 0 && !features_with_missing.empty()) {
        auto rf_start = std::chrono::high_resolution_clock::now();
        
        // Sort features by missing count
        std::sort(features_with_missing.begin(), features_with_missing.end(),
                  [&](integer_t a, integer_t b) { return h_missing_count[a] < h_missing_count[b]; });
        
        for (integer_t iter = 0; iter < config.n_iterations; ++iter) {
            if (config.verbose) {
                std::cout << "\n  RF Iteration " << (iter + 1) << "/" << config.n_iterations << "\n";
            }
            
            real_t total_change = 0.0f;
            
            for (integer_t j : features_with_missing) {
                integer_t n_missing_j = h_missing_count[j];
                if (n_missing_j == 0) continue;
                
                // Prepare training/test indices
                std::vector<integer_t> train_indices, test_indices;
                train_indices.reserve(n_samples);
                test_indices.reserve(n_missing_j);
                
                for (integer_t i = 0; i < n_samples; ++i) {
                    if (h_missing_mask[i * n_features + j]) {
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
                
                // Build training data (exclude current feature)
                integer_t n_pred_features = n_features - 1;
                std::vector<real_t> X_train(n_train * n_pred_features);
                std::vector<real_t> y_train(n_train);
                
                for (integer_t ti = 0; ti < n_train; ++ti) {
                    integer_t i = train_indices[ti];
                    integer_t feat_idx = 0;
                    for (integer_t f = 0; f < n_features; ++f) {
                        if (f != j) {
                            X_train[ti * n_pred_features + feat_idx] = h_X[i * n_features + f];
                            feat_idx++;
                        }
                    }
                    y_train[ti] = h_X[i * n_features + j];
                }
                
                // Build test data
                integer_t n_test = test_indices.size();
                std::vector<real_t> X_test(n_test * n_pred_features);
                
                for (integer_t ti = 0; ti < n_test; ++ti) {
                    integer_t i = test_indices[ti];
                    integer_t feat_idx = 0;
                    for (integer_t f = 0; f < n_features; ++f) {
                        if (f != j) {
                            X_test[ti * n_pred_features + feat_idx] = h_X[i * n_features + f];
                            feat_idx++;
                        }
                    }
                }
                
                // Train RF and predict (using GPU)
                RandomForestConfig rf_config;
                rf_config.nsample = n_train;
                rf_config.mdim = n_pred_features;
                rf_config.ntree = config.n_trees;
                rf_config.nodesize = config.nodesize;
                rf_config.use_gpu = true;  // Use GPU for RF training
                rf_config.compute_importance = false;
                rf_config.compute_proximity = false;
                
                if (config.mtry > 0) {
                    rf_config.mtry = config.mtry;
                } else {
                    rf_config.mtry = std::max(1, static_cast<integer_t>(std::sqrt(n_pred_features)));
                }
                
                std::vector<real_t> predictions(n_test);
                
                try {
                    if (h_is_categorical[j]) {
                        // Classification
                        rf_config.task_type = TaskType::CLASSIFICATION;
                        
                        std::set<integer_t> classes_set;
                        for (real_t v : y_train) {
                            classes_set.insert(static_cast<integer_t>(v));
                        }
                        rf_config.nclass = classes_set.size();
                        
                        std::vector<integer_t> y_train_int(n_train);
                        for (integer_t i = 0; i < n_train; ++i) {
                            y_train_int[i] = static_cast<integer_t>(y_train[i]);
                        }
                        
                        RandomForest rf(rf_config);
                        rf.fit_classification(X_train.data(), y_train_int.data(), nullptr);
                        
                        std::vector<integer_t> pred_int(n_test);
                        rf.predict_classification(X_test.data(), n_test, pred_int.data());
                        
                        for (integer_t i = 0; i < n_test; ++i) {
                            predictions[i] = static_cast<real_t>(pred_int[i]);
                        }
                    } else {
                        // Regression
                        rf_config.task_type = TaskType::REGRESSION;
                        rf_config.nclass = 1;
                        
                        RandomForest rf(rf_config);
                        rf.fit_regression(X_train.data(), y_train.data(), nullptr);
                        rf.predict_regression(X_test.data(), n_test, predictions.data());
                    }
                    
                    // Update values and calculate change
                    real_t change = 0.0f;
                    for (integer_t ti = 0; ti < n_test; ++ti) {
                        integer_t i = test_indices[ti];
                        real_t old_val = h_X[i * n_features + j];
                        real_t new_val = predictions[ti];
                        change += std::fabs(new_val - old_val);
                        h_X[i * n_features + j] = new_val;
                    }
                    
                    real_t mean_change = change / n_test;
                    total_change += mean_change;
                    results.column_info[j].rf_refined = true;
                    results.column_info[j].mean_change = mean_change;
                    
                    if (config.verbose) {
                        const char* type_str = h_is_categorical[j] ? "cat" : "num";
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
    
    // Copy final result back to X
    std::memcpy(X, h_X.data(), total_elements * sizeof(float));
    
    // Count remaining missing
    integer_t remaining_missing = 0;
    for (integer_t j = 0; j < n_features; ++j) {
        integer_t col_missing = 0;
        for (integer_t i = 0; i < n_samples; ++i) {
            if (std::isnan(h_X[i * n_features + j])) {
                col_missing++;
            }
        }
        results.column_info[j].missing_after = col_missing;
        remaining_missing += col_missing;
    }
    results.n_missing_total_after = remaining_missing;
    
    // Cleanup GPU memory
    cudaFree(d_X);
    cudaFree(d_missing_mask);
    cudaFree(d_missing_count);
    cudaFree(d_is_categorical);
    cudaFree(d_fill_values);
    cudaFree(d_unique_counts);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    if (config.verbose) {
        std::cout << results.to_string();
    }
    
    return results;
}

// ============================================================================
// GPU SPARSE IMPUTATION
// ============================================================================

std::pair<SparseMatrixCSR, ImputationResults> impute_missing_gpu_sparse(
    const SparseMatrixCSR& X,
    const ImputeConfig& config
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ImputationResults results;
    results.n_samples = X.nrows;
    results.n_features = X.ncols;
    results.method = config.method;
    results.use_gpu = true;
    results.is_sparse = true;
    results.n_iterations_used = config.n_iterations;
    
    // Check for NaN values in sparse data
    integer_t n_explicit_nan = 0;
    for (integer_t k = 0; k < X.nnz; ++k) {
        if (std::isnan(X.data[k])) {
            n_explicit_nan++;
        }
    }
    
    results.n_missing_total_before = n_explicit_nan;
    
    if (n_explicit_nan == 0) {
        if (config.verbose) {
            std::cout << "No NaN values in sparse matrix - returning original\n";
        }
        results.n_missing_total_after = 0;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
        return std::make_pair(X, results);
    }
    
    // For sparse with NaN, convert to dense, impute, convert back
    // This maintains GPU acceleration benefits
    
    // Convert sparse to dense
    size_t total_elements = static_cast<size_t>(X.nrows) * X.ncols;
    std::vector<real_t> X_dense(total_elements, 0.0f);
    std::vector<bool> has_value(total_elements, false);
    
    for (integer_t i = 0; i < X.nrows; ++i) {
        integer_t start = X.indptr[i];
        integer_t end = X.indptr[i + 1];
        for (integer_t k = start; k < end; ++k) {
            integer_t j = X.indices[k];
            X_dense[i * X.ncols + j] = X.data[k];
            has_value[i * X.ncols + j] = true;
        }
    }
    
    // Perform GPU dense imputation
    ImputationResults dense_results = impute_missing_gpu_dense(
        X_dense.data(), X.nrows, X.ncols, config);
    
    // Copy results
    results = dense_results;
    results.is_sparse = true;
    
    // Convert back to sparse (preserve original sparsity pattern + imputed values)
    std::vector<real_t> new_data;
    std::vector<integer_t> new_indices;
    std::vector<integer_t> new_indptr(X.nrows + 1, 0);
    
    for (integer_t i = 0; i < X.nrows; ++i) {
        new_indptr[i] = new_data.size();
        for (integer_t j = 0; j < X.ncols; ++j) {
            real_t val = X_dense[i * X.ncols + j];
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

// Helper function to check CUDA availability
bool cuda_is_available_check() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // namespace rf

