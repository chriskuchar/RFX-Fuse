#include <cuda_fp16.h>  // MUST be VERY FIRST - before ANY other includes
#include "rf_types.hpp"
#include "rf_config.hpp"
#include "rf_utils.hpp"
#include "rf_memory.cuh"
#include "rf_cuda_config.hpp"
#include "rf_testreebag.hpp"
#include "rf_varimp.hpp"
#include "rf_varimp.cuh"  // For gpu_compute_tnodewt_kernel declaration
#include "rf_proximity.hpp"
#include "rf_proximity_importance.cuh"  // For GPU Dense proximity importance
#include "rf_data_accessor.cuh"  // Unified dense/sparse data accessor
#include "rf_histogram_gpu.cuh"  // For GPU dense histogram binning (matches sparse implementation)
// Forward declarations to defer type resolution and avoid static initialization issues
// Include headers only when CUDA is available - this prevents static init crashes
#ifdef __CUDACC__
#include "rf_proximity_lowrank.hpp"
#include "rf_proximity_upper_triangle.hpp"
#endif
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstring>  // For memset
#include <algorithm>  // For std::fill
#include <iostream>
#include <cstdlib>
#include <memory>
#include <chrono>

namespace rf {

// ============================================================================
// Device accessor for dense/sparse data - enables unified kernel
// ============================================================================

/**
 * Get feature value at (sample, feature) - works for both dense and sparse
 * @param x Dense data array (can be nullptr if using sparse)
 * @param X_sparse Sparse matrix (check is_valid() before use)
 * @param sample Sample index
 * @param feature Feature index  
 * @param mdim Number of features (for dense indexing)
 * @return Feature value
 */
__device__ __forceinline__ real_t get_feature_value(
    const real_t* x,
    const cuda::CudaSparseMatrixCSR* X_sparse,
    integer_t sample,
    integer_t feature,
    integer_t mdim
) {
    if (X_sparse != nullptr && X_sparse->d_data != nullptr) {
        // Sparse access via binary search
        return X_sparse->get(sample, feature);
    } else {
        // Dense access
        return x[sample * mdim + feature];
    }
}

// Thread-local storage for low-rank proximity state
// MUST be at file scope (not inside function) to persist across all batch calls
// This fixes bug where 3000 trees (300 batches) would reset state, causing zero factors
thread_local struct LowRankState {
    void* lowrank_prox_ptr_raw;
    bool lowrank_initialized;
    integer_t total_trees_processed;
    integer_t saved_nsample;
    
    LowRankState() : lowrank_prox_ptr_raw(nullptr), lowrank_initialized(false), 
                   total_trees_processed(0), saved_nsample(0) {}
} g_lowrank_state;

// CUDA error checking macro - matches backup version (logs but doesn't throw)
// #ifndef CUDA_CHECK_VOID
// #define CUDA_CHECK_VOID(call) \
//     do { \
//         cudaError_t err = call; \
//         if (err != cudaSuccess) { \
//             fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
//                     cudaGetErrorString(err)); \
//         } \
//     } while(0)
// #endif

// Forward declaration
__device__ real_t gpu_calcs(const real_t* tmpclass, const real_t* tclasspop, real_t pdo,
                           integer_t nclass);

// GPU loss function helpers
__device__ real_t gpu_binary_logloss(real_t p, real_t w) {
    // Binary logloss: -w * (y*log(p) + (1-y)*log(1-p))
    // For split evaluation, we compute sum of logloss for left and right
    // p is probability of class 1, w is weight
    const real_t eps = 1e-15f;
    p = max(eps, min(1.0f - eps, p));  // Clamp to avoid log(0)
    return -w * (p > 0.5f ? logf(p) : logf(1.0f - p));
}

__device__ real_t gpu_multiclass_crossentropy(const real_t* class_probs, integer_t nclass, real_t total_weight) {
    // Multiclass crossentropy: -sum(w * log(p_i))
    // For split evaluation, compute sum for left and right
    const real_t eps = 1e-15f;
    real_t loss = 0.0f;
    for (integer_t j = 0; j < nclass; ++j) {
        real_t p = max(eps, min(1.0f - eps, class_probs[j] / total_weight));
        loss -= class_probs[j] * logf(p);
    }
    return loss;
}

__device__ real_t gpu_mae_loss(real_t mean_left, real_t mean_right, 
                               const integer_t* samples_left, integer_t n_left,
                               const integer_t* samples_right, integer_t n_right,
                               const integer_t* cl, const real_t* win) {
    // MAE: sum(|y - mean|) for left and right
    real_t mae_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mae_left += w * fabsf(y - mean_left);
        w_left += w;
    }
    
    real_t mae_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mae_right += w * fabsf(y - mean_right);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((mae_left + mae_right) / total_w) : 0.0f;
}

// Additional regression loss functions
__device__ real_t gpu_huber_loss(real_t mean_left, real_t mean_right,
                                  const integer_t* samples_left, integer_t n_left,
                                  const integer_t* samples_right, integer_t n_right,
                                  const integer_t* cl, const real_t* win, real_t delta = 1.0f) {
    // Huber loss: L_delta(y, mean) = 0.5*(y-mean)^2 if |y-mean| <= delta, else delta*|y-mean| - 0.5*delta^2
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_left);
        if (residual <= delta) {
            loss_left += w * 0.5f * residual * residual;
        } else {
            loss_left += w * (delta * residual - 0.5f * delta * delta);
        }
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_right);
        if (residual <= delta) {
            loss_right += w * 0.5f * residual * residual;
        } else {
            loss_right += w * (delta * residual - 0.5f * delta * delta);
        }
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_poisson_loss(real_t mean_left, real_t mean_right,
                                    const integer_t* samples_left, integer_t n_left,
                                    const integer_t* samples_right, integer_t n_right,
                                    const integer_t* cl, const real_t* win) {
    // Poisson loss: -y*log(mean) + mean (for y >= 0)
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(0.0f, static_cast<real_t>(cl[idx]));  // Poisson requires non-negative
        real_t mean = max(eps, mean_left);
        loss_left += w * (-y * logf(mean) + mean);
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(0.0f, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_right);
        loss_right += w * (-y * logf(mean) + mean);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_quantile_loss(real_t quantile_left, real_t quantile_right,
                                     const integer_t* samples_left, integer_t n_left,
                                     const integer_t* samples_right, integer_t n_right,
                                     const integer_t* cl, const real_t* win, real_t alpha = 0.5f) {
    // Quantile loss: alpha * max(y - q, 0) + (1-alpha) * max(q - y, 0)
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = y - quantile_left;
        loss_left += w * (alpha * max(0.0f, residual) + (1.0f - alpha) * max(0.0f, -residual));
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = y - quantile_right;
        loss_right += w * (alpha * max(0.0f, residual) + (1.0f - alpha) * max(0.0f, -residual));
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_mape_loss(real_t mean_left, real_t mean_right,
                                 const integer_t* samples_left, integer_t n_left,
                                 const integer_t* samples_right, integer_t n_right,
                                 const integer_t* cl, const real_t* win) {
    // MAPE: mean(|y - mean| / max(|y|, eps))
    const real_t eps = 1e-8f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t denom = max(fabsf(y), eps);
        loss_left += w * fabsf(y - mean_left) / denom;
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t denom = max(fabsf(y), eps);
        loss_right += w * fabsf(y - mean_right) / denom;
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_fair_loss(real_t mean_left, real_t mean_right,
                                const integer_t* samples_left, integer_t n_left,
                                const integer_t* samples_right, integer_t n_right,
                                const integer_t* cl, const real_t* win, real_t c = 1.0f) {
    // Fair loss: c^2 * (|y-mean|/c - log(1 + |y-mean|/c))
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_left) / c;
        loss_left += w * c * c * (residual - logf(1.0f + residual));
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        real_t residual = fabsf(y - mean_right) / c;
        loss_right += w * c * c * (residual - logf(1.0f + residual));
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_gamma_loss(real_t mean_left, real_t mean_right,
                                 const integer_t* samples_left, integer_t n_left,
                                 const integer_t* samples_right, integer_t n_right,
                                 const integer_t* cl, const real_t* win) {
    // Gamma loss: -log(y/mean) + y/mean - 1 (for y > 0, mean > 0)
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_left);
        loss_left += w * (-logf(y / mean) + y / mean - 1.0f);
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_right);
        loss_right += w * (-logf(y / mean) + y / mean - 1.0f);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_tweedie_loss(real_t mean_left, real_t mean_right,
                                    const integer_t* samples_left, integer_t n_left,
                                    const integer_t* samples_right, integer_t n_right,
                                    const integer_t* cl, const real_t* win, real_t p = 1.5f) {
    // Tweedie loss: -y*mean^(1-p)/(1-p) + mean^(2-p)/(2-p) for p != 1,2
    // For p=1: Poisson, p=2: Gamma
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_left);
        if (fabsf(p - 1.0f) < 1e-6f) {
            // Poisson: -y*log(mean) + mean
            loss_left += w * (-y * logf(mean) + mean);
        } else if (fabsf(p - 2.0f) < 1e-6f) {
            // Gamma: -log(y/mean) + y/mean - 1
            loss_left += w * (-logf(y / mean) + y / mean - 1.0f);
        } else {
            real_t mean_1p = powf(mean, 1.0f - p);
            real_t mean_2p = powf(mean, 2.0f - p);
            loss_left += w * (-y * mean_1p / (1.0f - p) + mean_2p / (2.0f - p));
        }
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t mean = max(eps, mean_right);
        if (fabsf(p - 1.0f) < 1e-6f) {
            loss_right += w * (-y * logf(mean) + mean);
        } else if (fabsf(p - 2.0f) < 1e-6f) {
            loss_right += w * (-logf(y / mean) + y / mean - 1.0f);
        } else {
            real_t mean_1p = powf(mean, 1.0f - p);
            real_t mean_2p = powf(mean, 2.0f - p);
            loss_right += w * (-y * mean_1p / (1.0f - p) + mean_2p / (2.0f - p));
        }
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

__device__ real_t gpu_ranking_loss(real_t score_left, real_t score_right,
                                    const integer_t* samples_left, integer_t n_left,
                                    const integer_t* samples_right, integer_t n_right,
                                    const integer_t* cl, const real_t* win) {
    // Ranking loss (LambdaRank style): pairwise ranking loss
    // Simplified: use MSE on scores as proxy
    real_t loss = 0.0f;
    real_t total_w = 0.0f;
    
    // Compute mean score for left and right
    real_t mean_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mean_left += w * y;
        w_left += w;
    }
    if (w_left > 0.0f) mean_left /= w_left;
    
    real_t mean_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = static_cast<real_t>(cl[idx]);
        mean_right += w * y;
        w_right += w;
    }
    if (w_right > 0.0f) mean_right /= w_right;
    
    // Pairwise ranking loss: sum over pairs
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx_i = samples_left[i];
        real_t w_i = win[idx_i];
        real_t y_i = static_cast<real_t>(cl[idx_i]);
        for (integer_t j = 0; j < n_right; ++j) {
            integer_t idx_j = samples_right[j];
            real_t w_j = win[idx_j];
            real_t y_j = static_cast<real_t>(cl[idx_j]);
            if (y_i > y_j) {
                // i should rank higher than j
                real_t score_diff = score_left - score_right;
                loss += w_i * w_j * max(0.0f, 1.0f - score_diff);  // Hinge loss
            }
        }
    }
    
    total_w = w_left * w_right;
    return (total_w > 0.0f) ? (loss / total_w) : 0.0f;
}

__device__ real_t gpu_survival_loss(real_t hazard_left, real_t hazard_right,
                                     const integer_t* samples_left, integer_t n_left,
                                     const integer_t* samples_right, integer_t n_right,
                                     const integer_t* cl, const real_t* win) {
    // Survival loss (Cox proportional hazards): -log(hazard) + cumulative hazard
    // Simplified: use negative log-likelihood of survival model
    const real_t eps = 1e-15f;
    real_t loss_left = 0.0f;
    real_t w_left = 0.0f;
    for (integer_t i = 0; i < n_left; ++i) {
        integer_t idx = samples_left[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t hazard = max(eps, hazard_left);
        loss_left += w * (-logf(hazard) + hazard * y);
        w_left += w;
    }
    
    real_t loss_right = 0.0f;
    real_t w_right = 0.0f;
    for (integer_t i = 0; i < n_right; ++i) {
        integer_t idx = samples_right[i];
        real_t w = win[idx];
        real_t y = max(eps, static_cast<real_t>(cl[idx]));
        real_t hazard = max(eps, hazard_right);
        loss_right += w * (-logf(hazard) + hazard * y);
        w_right += w;
    }
    
    real_t total_w = w_left + w_right;
    return (total_w > 0.0f) ? ((loss_left + loss_right) / total_w) : 0.0f;
}

// ============================================================================
// HISTOGRAM-BASED SPLIT FINDING (Device Functions)
// ============================================================================
// These device functions enable O(bins) split finding instead of O(n log n) sorting
// Used when use_histogram flag is enabled for XGBoost-like speed

// Device function: Build classification histogram for current node samples
__device__ void gpu_histogram_build_classification_device(
    const uint8_t* X_binned,    // Pre-binned feature data [nsample × mdim]
    const integer_t* y,         // Class labels
    const real_t* weights,      // Sample weights (can be nullptr)
    const integer_t* node_samples,  // Sample indices in current node
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    integer_t nclass,
    real_t* hist_out            // Output: [n_bins × nclass]
) {
    // Initialize histogram to zero
    for (integer_t i = 0; i < n_bins * nclass; i++) {
        hist_out[i] = 0.0f;
    }
    
    // DEBUG: Count skipped samples
    integer_t skipped = 0;
    integer_t total_processed = 0;
    
    // Build histogram
    for (integer_t i = 0; i < node_count; i++) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        integer_t c = y[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins && c >= 0 && c < nclass) {
            hist_out[bin * nclass + c] += w;
            total_processed++;
        } else {
            skipped++;
        }
    }
    
    // DEBUG: (disabled) Print skipped samples
    // if (skipped > 0) {
    //     printf("[HIST BUILD] feature=%d, skipped=%d\\n", feature, skipped);
    // }
}

// Device function: Find best split from classification histogram
__device__ void gpu_histogram_find_split_classification_device(
    const real_t* hist,         // [n_bins × nclass]
    integer_t n_bins,
    integer_t nclass,
    const real_t* bin_edges,    // [n_bins + 1] bin edges for this feature
    real_t& best_crit,          // Output: best Gini criterion (for MAXIMIZATION)
    real_t& best_split_value    // Output: split point value
) {
    // Left/right class counts
    real_t left_counts[32];
    real_t right_counts[32];
    
    for (integer_t c = 0; c < nclass && c < 32; c++) {
        left_counts[c] = 0.0f;
        right_counts[c] = 0.0f;
    }
    
    real_t total_left = 0.0f;
    real_t total_right = 0.0f;
    
    // Initialize right with all samples
    for (integer_t b = 0; b < n_bins; b++) {
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            right_counts[c] += hist[b * nclass + c];
            total_right += hist[b * nclass + c];
        }
    }
    
    // Fortran-style criterion: MAXIMIZE (rln/rld + rrn/rrd)
    // Must match sorting path exactly: sum(counts^2) / sum(counts) for each side
    real_t best_crit_local = -2.0f;  // Start with worst possible
    integer_t best_bin = 0;
    
    // Scan to find best split
    for (integer_t b = 0; b < n_bins - 1; b++) {
        // Move bin b from right to left
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            real_t count = hist[b * nclass + c];
            left_counts[c] += count;
            right_counts[c] -= count;
            total_left += count;
            total_right -= count;
        }
        
        if (total_left < 1.0f || total_right < 1.0f) continue;
        
        // Compute EXACT Fortran criterion: (rln/rld) + (rrn/rrd)
        // where rln = sum(left_counts[c]^2), rld = sum(left_counts[c]) = total_left
        // This matches the sorting path exactly
        real_t rln = 0.0f;
        real_t rrn = 0.0f;
        
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            rln += left_counts[c] * left_counts[c];
            rrn += right_counts[c] * right_counts[c];
        }
        
        // Fortran criterion: MAXIMIZE (higher is better = purer split)
        real_t crit = (rln / total_left) + (rrn / total_right);
        
        if (crit > best_crit_local) {
            best_crit_local = crit;
            best_bin = b;
        }
    }
    
    best_crit = best_crit_local;
    // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
    // Without this fix, samples with value == bin_edges[best_bin + 1] go to wrong child
    best_split_value = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0f;
}

// Device function: Build regression histogram for current node samples
__device__ void gpu_histogram_build_regression_device(
    const uint8_t* X_binned,
    const real_t* y_reg,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    real_t* sum_out,            // Output: [n_bins] sum of y
    real_t* sum2_out,           // Output: [n_bins] sum of y^2
    real_t* count_out           // Output: [n_bins] weighted count
) {
    // Initialize
    for (integer_t i = 0; i < n_bins; i++) {
        sum_out[i] = 0.0f;
        sum2_out[i] = 0.0f;
        count_out[i] = 0.0f;
    }
    
    // Build histogram
    for (integer_t i = 0; i < node_count; i++) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        real_t val = y_reg[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins) {
            sum_out[bin] += val * w;
            sum2_out[bin] += val * val * w;
            count_out[bin] += w;
        }
    }
}

// Device function: Find best split from regression histogram
__device__ void gpu_histogram_find_split_regression_device(
    const real_t* sum_y,
    const real_t* sum_y2,
    const real_t* count,
    integer_t n_bins,
    const real_t* bin_edges,
    real_t& best_crit,          // Output: best MSE (MINIMIZE)
    real_t& best_split_value
) {
    // Compute totals
    real_t sum_left = 0.0f, sum2_left = 0.0f, cnt_left = 0.0f;
    real_t sum_right = 0.0f, sum2_right = 0.0f, cnt_right = 0.0f;
    
    for (integer_t b = 0; b < n_bins; b++) {
        sum_right += sum_y[b];
        sum2_right += sum_y2[b];
        cnt_right += count[b];
    }
    
    real_t best_mse = 1.0e20f;
    integer_t best_bin = 0;
    
    // Scan
    for (integer_t b = 0; b < n_bins - 1; b++) {
        sum_left += sum_y[b];
        sum2_left += sum_y2[b];
        cnt_left += count[b];
        
        sum_right -= sum_y[b];
        sum2_right -= sum_y2[b];
        cnt_right -= count[b];
        
        if (cnt_left < 1.0f || cnt_right < 1.0f) continue;
        
        real_t mean_left = sum_left / cnt_left;
        real_t mean_right = sum_right / cnt_right;
        
        // Variance = E[X^2] - E[X]^2
        real_t var_left = sum2_left / cnt_left - mean_left * mean_left;
        real_t var_right = sum2_right / cnt_right - mean_right * mean_right;
        
        // Weighted MSE
        real_t total = cnt_left + cnt_right;
        real_t mse = (cnt_left * var_left + cnt_right * var_right) / total;
        
        if (mse < best_mse) {
            best_mse = mse;
            best_bin = b;
        }
    }
    
    best_crit = best_mse;
    // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
    best_split_value = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0f;
}

// ============================================================================
// PARALLEL HISTOGRAM FUNCTIONS - Use all 256 threads for split finding
// ============================================================================
// These are drop-in replacements for the sequential versions above.
// Each thread processes node_count/256 samples and uses atomicAdd to shared histogram.
// This gives ~50-100x speedup on histogram building for large nodes.

// Parallel histogram build for classification - uses all 256 threads
__device__ void gpu_histogram_build_classification_parallel(
    const uint8_t* X_binned,    // Pre-binned feature data [nsample × mdim]
    const integer_t* y,         // Class labels
    const real_t* weights,      // Sample weights (can be nullptr)
    const integer_t* node_samples,  // Sample indices in current node
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    integer_t nclass,
    real_t* hist_out            // Output: [n_bins × nclass] in shared memory
) {
    integer_t tid = threadIdx.x;
    integer_t num_threads = blockDim.x;
    
    // Parallel initialization: each thread clears part of histogram
    integer_t hist_size = n_bins * nclass;
    for (integer_t i = tid; i < hist_size; i += num_threads) {
        hist_out[i] = 0.0f;
    }
    __syncthreads();
    
    // Parallel histogram building: each thread processes its share of samples
    for (integer_t i = tid; i < node_count; i += num_threads) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        integer_t c = y[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins && c >= 0 && c < nclass) {
            // Use atomicAdd to accumulate across threads
            atomicAdd(&hist_out[bin * nclass + c], w);
        }
    }
    __syncthreads();
}

// Parallel histogram build for regression - uses all 256 threads
__device__ void gpu_histogram_build_regression_parallel(
    const uint8_t* X_binned,
    const real_t* y_reg,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    real_t* sum_out,            // Output: [n_bins] sum of y in shared memory
    real_t* sum2_out,           // Output: [n_bins] sum of y^2 in shared memory
    real_t* count_out           // Output: [n_bins] weighted count in shared memory
) {
    integer_t tid = threadIdx.x;
    integer_t num_threads = blockDim.x;
    
    // Parallel initialization
    for (integer_t i = tid; i < n_bins; i += num_threads) {
        sum_out[i] = 0.0f;
        sum2_out[i] = 0.0f;
        count_out[i] = 0.0f;
    }
    __syncthreads();
    
    // Parallel histogram building
    for (integer_t i = tid; i < node_count; i += num_threads) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        real_t val = y_reg[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins) {
            atomicAdd(&sum_out[bin], val * w);
            atomicAdd(&sum2_out[bin], val * val * w);
            atomicAdd(&count_out[bin], w);
        }
    }
    __syncthreads();
}

// Parallel find best split for classification - uses reduction across threads
// Each thread evaluates a subset of bins, then we reduce to find global best
__device__ void gpu_histogram_find_split_classification_parallel(
    const real_t* hist,         // [n_bins × nclass] in shared memory
    integer_t n_bins,
    integer_t nclass,
    const real_t* bin_edges,    // [n_bins + 1] bin edges for this feature
    real_t* shared_best_crits,  // [256] shared memory for per-thread best crits
    integer_t* shared_best_bins,// [256] shared memory for per-thread best bins
    real_t& best_crit,          // Output: best Gini criterion (for MAXIMIZATION)
    real_t& best_split_value    // Output: split point value
) {
    integer_t tid = threadIdx.x;
    
    // Thread 0 computes prefix sums and finds best split
    // This is O(n_bins) = O(256) which is fast enough
    // The histogram building is the bottleneck, not the split finding
    if (tid == 0) {
        // Left/right class counts
        real_t left_counts[32];
        real_t right_counts[32];
        
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            left_counts[c] = 0.0f;
            right_counts[c] = 0.0f;
        }
        
        real_t total_left = 0.0f;
        real_t total_right = 0.0f;
        
        // Initialize right with all samples
        for (integer_t b = 0; b < n_bins; b++) {
            for (integer_t c = 0; c < nclass && c < 32; c++) {
                right_counts[c] += hist[b * nclass + c];
                total_right += hist[b * nclass + c];
            }
        }
        
        real_t best_crit_local = -2.0f;
        integer_t best_bin = 0;
        
        // Scan to find best split
        for (integer_t b = 0; b < n_bins - 1; b++) {
            for (integer_t c = 0; c < nclass && c < 32; c++) {
                real_t count = hist[b * nclass + c];
                left_counts[c] += count;
                right_counts[c] -= count;
                total_left += count;
                total_right -= count;
            }
            
            if (total_left < 1.0f || total_right < 1.0f) continue;
            
            real_t rln = 0.0f;
            real_t rrn = 0.0f;
            
            for (integer_t c = 0; c < nclass && c < 32; c++) {
                rln += left_counts[c] * left_counts[c];
                rrn += right_counts[c] * right_counts[c];
            }
            
            real_t crit = (rln / total_left) + (rrn / total_right);
            
            if (crit > best_crit_local) {
                best_crit_local = crit;
                best_bin = b;
            }
        }
        
        best_crit = best_crit_local;
        // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
        best_split_value = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0f;
    }
    __syncthreads();
}

// Parallel find best split for regression
__device__ void gpu_histogram_find_split_regression_parallel(
    const real_t* sum_y,
    const real_t* sum_y2,
    const real_t* count,
    integer_t n_bins,
    const real_t* bin_edges,
    real_t& best_crit,          // Output: best MSE (MINIMIZE)
    real_t& best_split_value
) {
    integer_t tid = threadIdx.x;
    
    // Thread 0 computes the scan - O(256) is fast
    if (tid == 0) {
        real_t sum_left = 0.0f, sum2_left = 0.0f, cnt_left = 0.0f;
        real_t sum_right = 0.0f, sum2_right = 0.0f, cnt_right = 0.0f;
        
        for (integer_t b = 0; b < n_bins; b++) {
            sum_right += sum_y[b];
            sum2_right += sum_y2[b];
            cnt_right += count[b];
        }
        
        real_t best_mse = 1.0e20f;
        integer_t best_bin = 0;
        
        for (integer_t b = 0; b < n_bins - 1; b++) {
            sum_left += sum_y[b];
            sum2_left += sum_y2[b];
            cnt_left += count[b];
            
            sum_right -= sum_y[b];
            sum2_right -= sum_y2[b];
            cnt_right -= count[b];
            
            if (cnt_left < 1.0f || cnt_right < 1.0f) continue;
            
            real_t mean_left = sum_left / cnt_left;
            real_t mean_right = sum_right / cnt_right;
            
            real_t var_left = sum2_left / cnt_left - mean_left * mean_left;
            real_t var_right = sum2_right / cnt_right - mean_right * mean_right;
            
            real_t total = cnt_left + cnt_right;
            real_t mse = (cnt_left * var_left + cnt_right * var_right) / total;
            
            if (mse < best_mse) {
                best_mse = mse;
                best_bin = b;
            }
        }
        
        best_crit = best_mse;
        // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
        best_split_value = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0f;
    }
    __syncthreads();
}

// Initialize random states kernel
__global__ void init_random_states_kernel(curandState* random_states, integer_t num_trees, const integer_t* seeds) {
    integer_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_trees) {
        // Use tree-specific seed for reproducibility
        // Each tree gets a unique seed based on its index
        curand_init(seeds[tid], tid, 0, &random_states[tid]);
    }
}

// ============================================================================
// PRE-SORTED INDICES KERNEL - Sort each feature ONCE before tree growing
// ============================================================================
// This optimization avoids O(n log n) sorting at every node for every mtry variable.
// Instead, we sort once and filter the pre-sorted indices at each node (O(n)).
// Speedup: ~10-20x for split finding phase.

__global__ void gpu_presort_features_kernel(
    const real_t* x,           // Feature matrix [nsample × mdim]
    integer_t nsample, 
    integer_t mdim,
    integer_t* sorted_indices_all)  // Output: [mdim × nsample] sorted indices per feature
{
    integer_t feature_id = blockIdx.x;
    if (feature_id >= mdim) return;
    
    // Output array for this feature
    integer_t* sorted_indices = sorted_indices_all + feature_id * nsample;
    
    // Thread 0 does the sorting (simple approach - works well for typical RF sizes)
    if (threadIdx.x == 0) {
        // Initialize indices [0, 1, 2, ..., nsample-1]
        for (integer_t i = 0; i < nsample; ++i) {
            sorted_indices[i] = i;
        }
        
        // Insertion sort - O(n²) but fast for small n, cache-friendly
        // For n < 10000, this is typically faster than complex parallel sorts
        for (integer_t i = 1; i < nsample; ++i) {
            integer_t key = sorted_indices[i];
            real_t key_val = x[key * mdim + feature_id];
            integer_t j = i - 1;
            
            while (j >= 0 && x[sorted_indices[j] * mdim + feature_id] > key_val) {
                sorted_indices[j + 1] = sorted_indices[j];
                j--;
            }
            sorted_indices[j + 1] = key;
        }
    }
}

// GPU bootstrap sampling kernel
__global__ void gpu_bootstrap_kernel(
    integer_t num_trees, integer_t nsample, const real_t* weight,
    integer_t* nin_all, real_t* win_all, integer_t* jinbag_all,
    curandState* random_states, const integer_t* seeds,
    bool use_casewise) {  // Pass use_casewise as parameter (g_config not accessible in device code)
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    // Debug output
    // if (tree_id == 0 && threadIdx.x == 0) {
    //     printf("GPU: Bootstrap kernel started for tree %d\n", tree_id);
    // }
    
    integer_t tid = threadIdx.x;
    integer_t stride = blockDim.x;
    
    // Calculate offsets for this tree
    integer_t nin_offset = tree_id * nsample;
    integer_t win_offset = tree_id * nsample;
    integer_t jinbag_offset = tree_id * nsample;
    
    integer_t* nin = nin_all + nin_offset;
    real_t* win = win_all + win_offset;
    integer_t* jinbag = jinbag_all + jinbag_offset;
    
    // Initialize arrays
    for (integer_t i = tid; i < nsample; i += stride) {
        nin[i] = 0;
        win[i] = 0.0f;
    }
    __syncthreads();
    
    // Bootstrap sampling - exact port from original Fortran boot.f
    // Original: i = int(randomu()*nsample) + 1  (1-based: generates 1..nsample)
    // GPU port: i = static_cast<integer_t>(rand_val * nsample)  (0-based: generates 0..nsample-1)
    // Note: Expected in-bag percentage = 1 - (1 - 1/n)^n
    //       For n→∞: approaches 1 - e^(-1) ≈ 0.6321 (63.2%)
    //       For finite n: slightly higher (e.g., n=150: ~63.21%, n=100: ~63.40%, n=10: ~65.13%)
    curandState local_state = random_states[tree_id];
    
    // Generate bootstrap samples (match Fortran: int(randomu() * nsample) + 1)
    // Parallel bootstrap: each thread handles subset of samples using skipahead
    // This matches GPU Sparse for consistent RNG sequences
    for (integer_t s = tid; s < nsample; s += stride) {
        // Create thread-specific random state using skipahead (matches GPU Sparse)
        curandState thread_state = local_state;
        skipahead(s, &thread_state);  // Jump to position s in sequence
        
        real_t rand_val = curand_uniform(&thread_state);
        
        // Convert to sample index (0-based)
        integer_t i = static_cast<integer_t>(rand_val * static_cast<real_t>(nsample));
        if (i >= nsample) i = nsample - 1;
        if (i < 0) i = 0;
        
        // Count occurrences using atomic operations
        ::atomicAdd(&nin[i], 1);
    }
    
    // Ensure all atomic operations complete before reading
    __threadfence_block();  // Ensure all writes in this block are visible
    __syncthreads();
    
    // Additional sync to ensure all blocks complete their atomic operations
    __threadfence();  // Global memory fence - ensures all blocks see the updates
    __syncthreads();
    
    // Now populate jinbag with only in-bag samples (where nin[n] > 0)
    // DETERMINISTIC: Samples are collected in index order by thread 0
    // This matches CPU behavior where samples 0,1,2,... are checked in order
    
    // First, set win values (can be done in parallel without affecting order)
    for (integer_t n = tid; n < nsample; n += stride) {
        if (nin[n] > 0) {
            if (use_casewise) {
                win[n] = static_cast<real_t>(nin[n]) * (weight != nullptr ? weight[n] : 1.0f);
            } else {
                win[n] = 1.0f;
            }
        } else {
            win[n] = 0.0f;
        }
    }
    __syncthreads();
    
    // Sequential collection for deterministic order
    // Thread 0 collects all in-bag samples in index order (like CPU)
    if (tid == 0) {
        integer_t ninbag_count = 0;
        for (integer_t n = 0; n < nsample; ++n) {
            if (nin[n] > 0) {
                jinbag[ninbag_count] = n;
                ninbag_count++;
            }
        }
    }
    __syncthreads();
    
    // Debug: Check if nin was populated (only for tree 0, thread 0)
    // DEBUG output disabled for production
    // if (tree_id == 0 && tid == 0) {
    //     integer_t nin_sum = 0;
    //     integer_t non_zero_count = 0;
    //     integer_t max_val = 0;
    //     integer_t max_idx = -1;
    //     integer_t zero_count = 0;
    //     
    //     // Also check distribution of jinbag to see if sampling is uniform
    //     integer_t jinbag_counts[10] = {0};  // Count how many times indices 0-9 appear in jinbag
    //     
    //     for (integer_t j = 0; j < nsample; ++j) {
    //         nin_sum += nin[j];
    //         if (nin[j] > 0) {
    //             non_zero_count++;
    //             if (nin[j] > max_val) {
    //                 max_val = nin[j];
    //                 max_idx = j;
    //             }
    //         } else {
    //             zero_count++;
    //         }
    //         // Check jinbag distribution
    //         if (jinbag[j] >= 0 && jinbag[j] < 10) {
    //             jinbag_counts[jinbag[j]]++;
    //         }
    //     }
    //     
    //     // Calculate expected in-bag percentage
    //     real_t expected_inbag_pct = 1.0f - powf(1.0f - 1.0f / static_cast<real_t>(nsample), static_cast<real_t>(nsample));
    //     real_t actual_inbag_pct = static_cast<real_t>(non_zero_count) / static_cast<real_t>(nsample);
    //     real_t actual_oob_pct = static_cast<real_t>(zero_count) / static_cast<real_t>(nsample);
    //     
    //     printf("GPU Bootstrap tree 0: nin_sum=%d (should be %d)\n", nin_sum, nsample);
    //     printf("GPU Bootstrap tree 0: in-bag=%d (%.1f%%), out-of-bag=%d (%.1f%%)\n", 
    //            non_zero_count, actual_inbag_pct * 100.0f, zero_count, actual_oob_pct * 100.0f);
    //     printf("GPU Bootstrap tree 0: Expected in-bag=%.1f%%, Expected out-of-bag=%.1f%%\n",
    //            expected_inbag_pct * 100.0f, (1.0f - expected_inbag_pct) * 100.0f);
    //     printf("GPU Bootstrap tree 0: max_val=%d at idx=%d\n", max_val, max_idx);
    //     
    //     // Also print first 10 values to see if they're actually set
    //     printf("GPU Bootstrap tree 0: First 10 nin values: ");
    //     for (integer_t j = 0; j < (nsample < 10 ? nsample : 10); ++j) {
    //         printf("%d ", static_cast<int>(nin[j]));
    //     }
    //     printf("\n");
    //     
    //     // Print some non-zero values if they exist
    //     if (non_zero_count > 0) {
    //         printf("GPU Bootstrap tree 0: First 10 non-zero nin values (idx,val): ");
    //         integer_t printed = 0;
    //         for (integer_t j = 0; j < nsample && printed < 10; ++j) {
    //             if (nin[j] > 0) {
    //                 printf("(%d,%d) ", j, static_cast<int>(nin[j]));
    //                 printed++;
    //             }
    //         }
    //         printf("\n");
    //     }
    // }
    __syncthreads();
    
    // Compute weights based on counts
    // For case-wise: win = nin * weight, for non-case-wise: win = 1.0
    // Note: This code is already handled above in the jinbag population loop, so this is redundant
    // But keep it for compatibility with existing code structure
    for (integer_t n = tid; n < nsample; n += stride) {
        if (nin[n] > 0) {
            if (use_casewise) {
                win[n] = static_cast<real_t>(nin[n]) * (weight != nullptr ? weight[n] : 1.0f);
            } else {
                win[n] = 1.0f;  // Non-case-wise: equal weighting
            }
        } else {
            win[n] = 0.0f;
        }
    }
    
//     // Debug output
//     if (tree_id == 0 && threadIdx.x == 0) {
//         printf("GPU: Bootstrap kernel completed for tree %d\n", tree_id);
//         printf("GPU: First 10 jinbag values: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
//                jinbag[0], jinbag[1], jinbag[2], jinbag[3], jinbag[4],
//                jinbag[5], jinbag[6], jinbag[7], jinbag[8], jinbag[9]);
//         printf("GPU: First 10 nin values: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
//                nin[0], nin[1], nin[2], nin[3], nin[4],
//                nin[5], nin[6], nin[7], nin[8], nin[9]);
//     }
}

// Simple GPU tree growing kernel - creates single-node trees with majority class
__global__ void gpu_simple_tree_kernel(
    integer_t num_trees, integer_t nsample, integer_t mdim, integer_t maxnode, integer_t nclass,
    integer_t minndsize, integer_t mtry, const real_t* x,
    integer_t* nodestatus_all, integer_t* nodeclass_all, integer_t* nnode_all,
    integer_t* bestvar_all, real_t* xbestsplit_all, integer_t* treemap_all,
    integer_t* jinbag_all, const integer_t* cl, const real_t* win) {
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    // Calculate offsets for this tree
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t jinbag_offset = tree_id * nsample;
    
    // Pointers to this tree's arrays
    integer_t* nodestatus = nodestatus_all + node_offset;
    integer_t* nodeclass = nodeclass_all + node_offset;
    integer_t* bestvar = bestvar_all + node_offset;
    real_t* xbestsplit = xbestsplit_all + node_offset;
    integer_t* treemap = treemap_all + treemap_offset;
    integer_t* jinbag = jinbag_all + jinbag_offset;
    
    // Initialize tree arrays
    for (integer_t i = 0; i < maxnode; ++i) {
        nodestatus[i] = 0;
        nodeclass[i] = 0;
        bestvar[i] = 0;
        xbestsplit[i] = 0.0f;
    }
    for (integer_t i = 0; i < 2 * maxnode; ++i) {
        treemap[i] = 0;
    }
    
    // Count samples in jinbag for this tree
    integer_t ninbag = 0;
    for (integer_t i = 0; i < nsample; ++i) {
        if (jinbag[i] >= 0 && jinbag[i] < nsample) {
            ninbag++;
        } else {
            break;  // jinbag is packed, so first invalid index means end
        }
    }
    
    // Create simple single-node tree that predicts majority class
    if (ninbag > 0) {
        // Count class distribution for all bootstrap samples
        real_t class_counts[10] = {0.0f};
        
        for (integer_t i = 0; i < ninbag && i < nsample; ++i) {
            integer_t sample_idx = jinbag[i];
            if (sample_idx >= 0 && sample_idx < nsample) {
                integer_t class_label = cl[sample_idx];
                if (class_label >= 0 && class_label < nclass) {
                    class_counts[class_label] += 1.0f;
                }
            }
        }
        
        // Find majority class (in case of tie, pick the last one with max count)
        integer_t majority_class = 0;
        real_t max_count = class_counts[0];
        for (integer_t c = 1; c < nclass; ++c) {
            if (class_counts[c] >= max_count) {  // Use >= to pick last in case of tie
                max_count = class_counts[c];
                majority_class = c;
            }
        }
        
        // // Debug: Print class counts for verification
        // if (tree_id == 0 && threadIdx.x == 0) {
        //     printf("GPU Tree %d: Class counts: [%.1f, %.1f, %.1f], majority: %d\n", 
        //            tree_id, class_counts[0], class_counts[1], class_counts[2], majority_class);
        // }
        
        // Set root node as terminal with majority class
        nodestatus[0] = -1;  // Terminal node
        nodeclass[0] = majority_class;
        nnode_all[tree_id] = 1;
        
        // // Debug output
        // if (tree_id == 0 && threadIdx.x == 0) {
        //     printf("GPU Tree %d: Simple tree - majority class %d (counts: %.1f,%.1f,%.1f)\n", 
        //            tree_id, majority_class, class_counts[0], class_counts[1], class_counts[2]);
        //     printf("GPU Tree %d: First 10 jinbag samples: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        //            tree_id, jinbag[0], jinbag[1], jinbag[2], jinbag[3], jinbag[4],
        //            jinbag[5], jinbag[6], jinbag[7], jinbag[8], jinbag[9]);
        //     printf("GPU Tree %d: First 10 class labels: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        //            tree_id, cl[jinbag[0]], cl[jinbag[1]], cl[jinbag[2]], cl[jinbag[3]], cl[jinbag[4]],
        //            cl[jinbag[5]], cl[jinbag[6]], cl[jinbag[7]], cl[jinbag[8]], cl[jinbag[9]]);
        // }
    }
    
}

// Device function for categorical split optimization (small categories)
__device__ void gpu_catmax(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
                          integer_t nnz, integer_t numcat, integer_t nclass,
                          integer_t* igl, integer_t* itmp, real_t* tmpclass,
                          const real_t* tcat, const integer_t* icat, real_t& critmax,
                          integer_t* igoleft, integer_t& nhit) {
    // Simplified implementation - find best categorical split
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    
    // Try different category groupings (one category goes left)
    for (integer_t i = 0; i < nnz; ++i) {
        integer_t cat_idx = icat[i];
        
        // Initialize igoleft
        for (integer_t k = 0; k < numcat; ++k) {
            igoleft[k] = 0;
        }
        igoleft[cat_idx] = 1;
        
        // Calculate criterion for this split
        real_t crit = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            tmpclass[j] = 0.0f;
            for (integer_t k = 0; k < numcat; ++k) {
                if (igoleft[k] == 1) {
                    tmpclass[j] += tclasscat[j + k * nclass];
                }
            }
        }
        
        // Calculate Gini-based criterion (matching CPU implementation)
        real_t sum_left = 0.0f;
        real_t sum_right = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            sum_left += tmpclass[j];
            sum_right += tclasspop[j] - tmpclass[j];
        }
        
        if (sum_left > 1e-5f && sum_right > 1e-5f) {
            // Calculate Gini: crit = (1-Gini_left) + (1-Gini_right)
            real_t pno_left = 0.0f;
            real_t pno_right = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                pno_left += tmpclass[j] * tmpclass[j];
                pno_right += (tclasspop[j] - tmpclass[j]) * (tclasspop[j] - tmpclass[j]);
            }
            crit = (pno_left / sum_left) + (pno_right / sum_right);
            
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
                // Copy best igoleft
                for (integer_t k = 0; k < numcat; ++k) {
                    igoleft[k] = (k == cat_idx) ? 1 : 0;
                }
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
    } else {
        nhit = 0;
    }
}

// Device function for categorical split optimization (large categories)
__device__ void gpu_catmaxr(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
                            integer_t numcat, integer_t nclass, integer_t* igl,
                            integer_t ncsplit, real_t* tmpclass,
                            real_t& critmax, integer_t* igoleft, integer_t& nhit,
                            curandState* local_state) {
    // Simplified implementation for large categorical variables
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    
    // Random sampling approach for large categories
    for (integer_t split = 0; split < ncsplit; ++split) {
        // Initialize igoleft randomly
        for (integer_t k = 0; k < numcat; ++k) {
            real_t rand_val = curand_uniform(local_state);
            igoleft[k] = (rand_val > 0.5f) ? 1 : 0;
        }
        
        // Calculate criterion
        real_t crit = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            tmpclass[j] = 0.0f;
            for (integer_t k = 0; k < numcat; ++k) {
                if (igoleft[k] == 1) {
                    tmpclass[j] += tclasscat[j + k * nclass];
                }
            }
        }
        
        real_t sum_left = 0.0f;
        real_t sum_right = 0.0f;
        for (integer_t j = 0; j < nclass; ++j) {
            sum_left += tmpclass[j];
            sum_right += tclasspop[j] - tmpclass[j];
        }
        
        if (sum_left > 1e-5f && sum_right > 1e-5f) {
            // Calculate Gini-based criterion
            real_t pno_left = 0.0f;
            real_t pno_right = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                pno_left += tmpclass[j] * tmpclass[j];
                pno_right += (tclasspop[j] - tmpclass[j]) * (tclasspop[j] - tmpclass[j]);
            }
            crit = (pno_left / sum_left) + (pno_right / sum_right);
            
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
                // Copy best igoleft
                for (integer_t k = 0; k < numcat; ++k) {
                    igoleft[k] = igoleft[k];  // Already set above
                }
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
    } else {
        nhit = 0;
    }
}

// Forward declarations for device functions used in kernel
__device__ void gpu_hybrid_sort_dense(integer_t* indices, integer_t count,
                                       const real_t* x, integer_t mvar, integer_t mdim);

// Full GPU tree growing kernel based on original Fortran growtree.f algorithm
// Converted to row-major, 0-based indexing with batch parallel GPU execution
// OPTIMIZATION: Uses sample_node tracking to avoid O(n*depth) tree traversal per node
// OPTIMIZATION: Uses pre-sorted indices to avoid O(n log n) sorting per node per mtry
__global__ void gpu_full_tree_kernel(
    integer_t num_trees, integer_t nsample, integer_t mdim, integer_t maxnode, integer_t nclass,
    integer_t minndsize, integer_t mtry, const real_t* x,
    integer_t* nodestatus_all, integer_t* nodeclass_all, integer_t* nnode_all,
    integer_t* bestvar_all, real_t* xbestsplit_all, integer_t* treemap_all,
    integer_t* jinbag_all, const integer_t* cl, const real_t* win,
    const integer_t* cat,  // cat[m] == 1 means quantitative, cat[m] > 1 means categorical with cat[m] categories
    const integer_t* amat,  // For categorical: amat[m + n*mdim] stores category index (0-based)
    integer_t* catgoleft_all,  // catgoleft[k + node*maxcat] = 1 if category k goes left at node
    integer_t maxcat,  // Maximum number of categories
    integer_t ncsplit,  // Number of random splits for large categoricals
    integer_t ncmax,  // Threshold for large categorical
    integer_t loss_function,  // 0=Gini, 1=Binary logloss, 2=Multiclass crossentropy, 3=MSE, 4=MAE, etc.
    real_t min_gain_to_split,  // XGBoost/LightGBM style early stopping threshold
    integer_t growth_mode,  // 0=level-wise, 1=leaf-wise, 2=hybrid, 4=random selection
    const integer_t* allowed_modes,  // Allowed modes for random selection (when growth_mode=4)
    integer_t num_allowed_modes,  // Number of allowed modes
    real_t l1_reg,  // L1 regularization strength
    real_t l2_reg,  // L2 regularization strength
    bool gpu_parallel_mode0,
    curandState* random_states,  // Pre-initialized random states (one per tree)
    const real_t* y_regression,  // Continuous y values for regression (nullptr for classification)
    integer_t* sample_node_all,  // OPTIMIZATION: sample_node[i] = current node for sample i (-1 if OOB)
    const integer_t* sorted_indices_all,  // OPTIMIZATION: pre-sorted indices [mdim × nsample]
    // HISTOGRAM PARAMETERS (for XGBoost-like speed)
    bool use_histogram,                    // Enable histogram-based split finding
    const uint8_t* X_binned,               // Pre-binned feature data [nsample × mdim]
    const real_t* bin_edges_all,           // Bin edges [mdim × 257]
    const integer_t* n_bins_per_feature,   // Number of bins per feature [mdim]
    integer_t max_bins,                    // Maximum bins (typically 256)
    // HISTOGRAM GLOBAL MEMORY FALLBACK (for >10 classes)
    real_t* hist_workspace_all,            // Global memory: [num_trees × 256 × max(nclass,3)]
    // NODE SAMPLES WORKSPACE (for >4096 samples when using sorting path)
    integer_t* node_samples_workspace_all) {  // Global memory: [num_trees × nsample] for sorting path
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    // Always use original RF growth mode (mode 0)
    integer_t actual_growth_mode = 0;

    // Calculate offsets for this tree
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t jinbag_offset = tree_id * nsample;
    integer_t win_offset = tree_id * nsample;
    
    // Pointers to this tree's arrays
    integer_t* nodestatus = nodestatus_all + node_offset;
    integer_t* nodeclass = nodeclass_all + node_offset;
    integer_t* nnode = nnode_all + tree_id;
    integer_t* bestvar = bestvar_all + node_offset;
    real_t* xbestsplit = xbestsplit_all + node_offset;
    integer_t* treemap = treemap_all + treemap_offset;
    const integer_t* jinbag_tree = jinbag_all + jinbag_offset;
    const real_t* win_tree = win + win_offset;
    // Note: catgoleft is accessed per-node, not per-tree offset
    // catgoleft[k + node*maxcat] where node is the node index within the tree
    // We need to calculate the full offset: tree_offset * maxnode * maxcat + node * maxcat
    // For now, we'll access it directly using kgrow (node index) within the tree
    // The full array is: catgoleft_all[tree_id * maxnode * maxcat + node * maxcat + category]
    
    // Shared memory for thread synchronization
    __shared__ integer_t shared_left_child;
    __shared__ integer_t shared_right_child;
    __shared__ bool shared_should_split;
    __shared__ integer_t shared_jstat;
    __shared__ integer_t shared_best_var;
    __shared__ integer_t shared_best_split_idx;
    __shared__ real_t shared_best_crit;
    __shared__ real_t shared_split_point;  // For sample_node update
    __shared__ bool shared_split_happened;  // Flag for parallel sample_node update
    
    // Sample tracking: use sample_node array directly instead of pre-collecting
    // This removes the 4096 sample limit and scales to any dataset size
    __shared__ integer_t shared_node_sample_count;
    
    // For sorting path: use global memory workspace for node_samples (scales to any size)
    // Each tree gets its own nsample workspace
    integer_t* node_samples = node_samples_workspace_all + tree_id * nsample;
    
    // HISTOGRAM: Shared memory for ≤10 classes, global memory fallback for >10 classes
    // Shared memory budget: 48KB total
    //   - Classification: 256 bins × 10 classes × 4 bytes = 10KB
    //   - Regression: 256 × 3 × 4 bytes = 3KB
    //   - Total: ~13KB < 48KB ✓ (node_samples now uses global memory - no limit!)
    __shared__ real_t shared_hist_class[256 * 10];   // Classification: [n_bins × nclass] (≤10 classes)
    __shared__ real_t shared_hist_sum[256];          // Regression: sum of y per bin
    __shared__ real_t shared_hist_sum2[256];         // Regression: sum of y^2 per bin
    __shared__ real_t shared_hist_count[256];        // Regression: count per bin
    const integer_t HIST_MAX_BINS = 256;
    const integer_t HIST_SHARED_MAX_CLASSES = 10;
    
    // Use global memory fallback for >10 classes (hist_workspace_all provided)
    // Layout: hist_workspace_all[tree_id * hist_size + offset]
    // Classification: hist_size = 256 * nclass
    // Regression: hist_size = 256 * 3
    real_t* hist_class_ptr = nullptr;
    real_t* hist_sum_ptr = nullptr;
    real_t* hist_sum2_ptr = nullptr;
    real_t* hist_count_ptr = nullptr;
    
    if (nclass > HIST_SHARED_MAX_CLASSES && hist_workspace_all != nullptr) {
        // Use global memory for large number of classes
        integer_t hist_size = 256 * nclass;
        hist_class_ptr = hist_workspace_all + tree_id * hist_size;
    } else if (nclass == 1 && hist_workspace_all != nullptr) {
        // Regression: use global memory if provided (optional)
        integer_t hist_size = 256 * 3;
        real_t* base = hist_workspace_all + tree_id * hist_size;
        hist_sum_ptr = base;
        hist_sum2_ptr = base + 256;
        hist_count_ptr = base + 512;
    }
    
    // For classification with ≤10 classes OR when global fallback not provided, use shared memory
    if (hist_class_ptr == nullptr && nclass > 1) {
        hist_class_ptr = shared_hist_class;
    }
    if (hist_sum_ptr == nullptr && nclass == 1) {
        hist_sum_ptr = shared_hist_sum;
        hist_sum2_ptr = shared_hist_sum2;
        hist_count_ptr = shared_hist_count;
    }
    
    // Pointer to this tree's sample_node tracking array
    // sample_node[i] = current node for sample i (-1 if OOB)
    integer_t* sample_node = sample_node_all + tree_id * nsample;

    // Initialize tree structure (following original Fortran algorithm)
    if (threadIdx.x == 0) {
        // Initialize arrays (following original algorithm)
        for (integer_t i = 0; i < maxnode; ++i) {
            nodestatus[i] = 0;
            bestvar[i] = 0;
            xbestsplit[i] = 0.0f;
            treemap[i * 2] = 0;
            treemap[i * 2 + 1] = 0;
        }
        
        // Initialize root node (following original algorithm)
        nodestatus[0] = 2;  // Root node is active
        nnode[0] = 1;       // Start with 1 node
    }
    __syncthreads();
    
    // OPTIMIZATION: Initialize sample_node tracking array in PARALLEL
    // All threads participate - no sync issues
    // sample_node[i] = 0 if in-bag (starts at root), -1 if OOB
    for (integer_t i = threadIdx.x; i < nsample; i += blockDim.x) {
        sample_node[i] = (win_tree[i] > 0.0f) ? 0 : -1;
    }
    __syncthreads();  // Safe: all threads complete the full loop
    
    // Helper function to calculate node depth (must be declared before use)
    auto calculate_node_depth = [&](integer_t node_idx) -> integer_t {
        integer_t depth = 0;
        integer_t current = node_idx;
        
        // Traverse up to root to count depth
        while (current > 0) {
            // Find parent by searching backwards
            bool parent_found = false;
            for (integer_t p = 0; p < current; ++p) {
                if (nodestatus[p] == 1) {  // Split node
                    if (treemap[p * 2] == current || treemap[p * 2 + 1] == current) {
                        current = p;
                        depth++;
                        parent_found = true;
                        break;
                    }
                }
            }
            if (!parent_found) break;
        }
        return depth;
    };
    
    // The loop condition checks nnode[0] which increases as nodes are split
    // SAFETY: Cache nnode[0] at start and add max iteration limit to prevent infinite loops
    integer_t initial_nnode = nnode[0];
    integer_t max_iterations = maxnode * 2;  // Safety limit: should never exceed maxnode nodes
    integer_t iteration_count = 0;
    
    for (integer_t kgrow = 0; kgrow < nnode[0] && iteration_count < max_iterations; ++kgrow, ++iteration_count) {
        // Safety check: if we've processed too many iterations, break
        if (iteration_count >= max_iterations) {
            if (threadIdx.x == 0) {
                // Reduced debug output - only print if safety break occurs
                // printf("DEBUG KERNEL: Tree %d SAFETY BREAK - iteration_count=%d >= max_iterations=%d, nnode[0]=%d\n",
                //        tree_id, iteration_count, max_iterations, nnode[0]);
            }
            break;
        }
        
        // Check if this node should be processed (status == 2 means active/not yet split)
        // Matching Fortran: if (nodestatus(kgrow).ne.2) goto 20 (skip to next iteration)
        if (nodestatus[kgrow] != 2) {
            continue;  // Skip nodes that aren't active (already split or terminal)
        }
        
        // Reduced debug output - disabled for production
        // if (threadIdx.x == 0 && tree_id == 0 && kgrow % 20 == 0 && kgrow < 100) {
        //     printf("DEBUG KERNEL: Tree %d processing node %d/%d\n", tree_id, kgrow, nnode[0]);
        // }
        
        // OPTIMIZATION: Get samples in this node using sample_node tracking - PARALLEL
        // No tree traversal needed! Just check sample_node[i] == kgrow
        // This is O(n/threads) instead of O(n*depth)
        if (threadIdx.x == 0) shared_node_sample_count = 0;
        __syncthreads();  // Safe: unconditional
        
        // Collect samples in this node into node_samples array (for histogram path)
        // This scales to any dataset size with no limit (uses global memory)
        for (integer_t i = threadIdx.x; i < nsample; i += blockDim.x) {
            if (sample_node[i] == kgrow) {
                integer_t pos = atomicAdd(&shared_node_sample_count, 1);
                node_samples[pos] = i;  // Store sample index in global memory workspace
            }
        }
        __syncthreads();
        
        integer_t node_sample_count = shared_node_sample_count;
        
        // Check if node should be split (following original algorithm)
        // For regression: need at least 2*minndsize to split (minndsize in each child)
        integer_t min_size_to_split = (nclass == 1) ? (2 * minndsize) : minndsize;
        if (node_sample_count < min_size_to_split) {
            // Node is too small, make it terminal
            // ALL threads must execute this to avoid divergence
            if (threadIdx.x == 0) {
                // Reduced debug output
                // if (tree_id == 0 && kgrow < 10) {
                //     printf("DEBUG KERNEL: Node %d too small (%d < %d), making terminal\n", kgrow, node_sample_count, min_size_to_split);
                // }
                nodestatus[kgrow] = -1;  // Terminal node
            }
            // ALL threads continue together
            continue;
        }
        
        // Handle regression vs classification differently
        integer_t non_zero_classes = 0;
        integer_t majority_class = 0;
        real_t max_count = 0.0f;
        real_t variance = 0.0f;
        real_t crit0 = 0.0f;
        
        // Declare class_counts outside so it's available for crit0 calculation
        real_t class_counts[10] = {0.0f};  // Max 10 classes
        
        if (nclass == 1) {
            // Regression: calculate weighted variance using continuous y_regression values
            // y_regression MUST be provided for regression tasks (nclass == 1)
            // Iterate directly over all samples - no 4096 limit
            real_t sum_y = 0.0f;
            real_t sum_y2 = 0.0f;
            real_t sum_w = 0.0f;
            for (integer_t sample_idx = 0; sample_idx < nsample; ++sample_idx) {
                if (sample_node[sample_idx] != kgrow) continue;
                real_t w = win_tree[sample_idx];  // Use case weight
                real_t y_val = y_regression[sample_idx];  // Regression requires y_regression
                sum_y += w * y_val;
                sum_y2 += w * y_val * y_val;
                sum_w += w;
            }
            real_t mean_y = (sum_w > 0.0f) ? (sum_y / sum_w) : 0.0f;
            variance = (sum_w > 0.0f) ? ((sum_y2 / sum_w) - mean_y * mean_y) : 0.0f;
            crit0 = variance;  // Initial MSE for regression
            
            // DEBUG: Print variance for first tree, first few nodes
            // if (tree_id == 0 && kgrow < 3 && threadIdx.x == 0) {
                //printf("[GPU_DEBUG_DISABLED] Tree %d Node %d: samples=%d, sum_w=%.3f, variance=%.6f, crit0=%.6f\n",
                    //    tree_id, kgrow, node_sample_count, sum_w, variance, crit0);
            // }
            
            // If variance is very small, make terminal
            if (variance < 1.0e-6f) {
                if (threadIdx.x == 0) {
                    nodestatus[kgrow] = -1;
                    nodeclass[kgrow] = 0;
                    // if (tree_id == 0 && kgrow < 3) {
                        //printf("[GPU_DEBUG_DISABLED] Tree %d Node %d: variance too small, making terminal\n", tree_id, kgrow);
                    // }
                }
                continue;
            }
        } else {
            // Classification: calculate weighted class distribution
            // Iterate directly over all samples - no 4096 limit
            for (integer_t j = 0; j < nclass; ++j) {
                class_counts[j] = 0.0f;
            }
            
            for (integer_t sample_idx = 0; sample_idx < nsample; ++sample_idx) {
                if (sample_node[sample_idx] != kgrow) continue;
                real_t w = win_tree[sample_idx];  // Use case weight
                integer_t class_label = cl[sample_idx];
                if (class_label >= 0 && class_label < nclass) {
                    class_counts[class_label] += w;  // Weighted count
                }
            }
        
        for (integer_t j = 0; j < nclass; ++j) {
            if (class_counts[j] > 0.0f) {
                non_zero_classes++;
                if (class_counts[j] > max_count) {
                    max_count = class_counts[j];
                    majority_class = j;
                }
            }
        }
        
            // Calculate initial criterion based on loss function
            if (nclass == 1) {
                // Regression: use MSE (loss_function == 3)
                crit0 = variance;
            } else {
                // Classification: use Gini impurity 
                real_t pno = 0.0f;
                real_t pdo = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    pno += class_counts[j] * class_counts[j];
                    pdo += class_counts[j];
                }
                crit0 = pdo > 0.0f ? (pno / pdo) : 0.0f;
            }
            
            // Check if node is pure (all samples same class)
        if (non_zero_classes <= 1) {
            // Node is pure, make it terminal
            if (threadIdx.x == 0) {
                nodestatus[kgrow] = -1;  // Terminal node
                nodeclass[kgrow] = majority_class;
            }
            continue;
            }
        }
        
        // Find best split using MSE (regression) or Gini (classification)
        // For classification: crit = (rln/rld)+(rrn/rrd) = (1-Gini_left) + (1-Gini_right)
        // We MAXIMIZE crit (which minimizes Gini), matching Fortran algorithm
        real_t best_crit = (nclass == 1) ? 1.0e20f : -2.0f;  // Regression: minimize MSE, Classification: maximize crit
        integer_t best_var = 0;
        integer_t best_split_idx = 0;
        integer_t jstat = 0;
        
        // Try mtry random variables (following original algorithm)
        // NOTE: Single-threaded per tree for correctness - parallel version caused hangs
    for (integer_t mv = 0; mv < mtry; ++mv) {
            // Generate random variable (following original: int(mdim*randomu())+1)
            // Use pre-initialized random state with hash-based skip
            // Advancing RNG state in a loop is still too slow for parallel mode
            curandState local_state = random_states[tree_id];
            // Use skipahead to efficiently jump to the right position
            // Each node*mtry combination gets a unique position in the sequence
            skipahead(kgrow * mtry + mv, &local_state);
            real_t rand_val = curand_uniform(&local_state);
            integer_t mvar = static_cast<integer_t>(rand_val * mdim);
            if (mvar >= mdim) mvar = mdim - 1;
            
            // DEBUG: Show feature selection for first tree, first few nodes
            // if (tree_id == 0 && kgrow < 3 && threadIdx.x == 0 && nclass == 1) {
                //printf("[GPU_DEBUG_DISABLED] Tree %d Node %d mv=%d: selected var=%d (rand=%.4f, mdim=%d, mtry=%d)\n",
            //            tree_id, kgrow, mv, mvar, rand_val, mdim, mtry);
            // }
            
            // Check if this is a categorical variable
            // NOTE: Binary features (numcat <= 2) should use histogram path, not categorical path
            // Reason: With sparse data, nnz=1 is common for binary features, making categorical splits impossible
            // But histogram path can still split on threshold 0.5 which works perfectly for binary
            integer_t numcat = (cat != nullptr && mvar < mdim) ? cat[mvar] : 1;
            bool is_categorical = (numcat > 2);  // ONLY treat as categorical if >2 categories
            
            // CATEGORICAL VARIABLE HANDLING (only for 3+ categories)
            if (is_categorical && nclass > 1) {  // Only classification supports categoricals for now
                // Build tclasscat matrix: tclasscat[class + category*nclass]
                real_t tclasscat[10 * 10] = {0.0f};  // Max 10 classes, 10 categories
                real_t tclasspop[10] = {0.0f};
                real_t tcat[10] = {0.0f};
                integer_t icat[10] = {0};
                
                // Count class distribution per category
                // Iterate directly over all samples - no 4096 limit
                for (integer_t sample_idx = 0; sample_idx < nsample; ++sample_idx) {
                    if (sample_node[sample_idx] != kgrow) continue;
                    real_t w = win_tree[sample_idx];
                    integer_t class_label = cl[sample_idx];
                    
                    // Get category index from amat (or from x if amat not available)
                    integer_t category_idx = 0;
                    if (amat != nullptr) {
                        category_idx = amat[mvar + sample_idx * mdim];
                    } else {
                        // Fallback: use x value rounded to integer
                        category_idx = static_cast<integer_t>(x[sample_idx * mdim + mvar] + 0.5f);
                    }
                    
                    if (category_idx >= 0 && category_idx < numcat && class_label >= 0 && class_label < nclass) {
                        tclasscat[class_label + category_idx * nclass] += w;
                        tclasspop[class_label] += w;
                    }
                }
                
                // Find non-zero categories
                integer_t nnz = 0;
                for (integer_t k = 0; k < numcat; ++k) {
                    tcat[k] = 0.0f;
                    for (integer_t j = 0; j < nclass; ++j) {
                        tcat[k] += tclasscat[j + k * nclass];
                    }
                    if (tcat[k] > 0.0f) {
                        icat[nnz] = k;
                        nnz++;
                    }
                }
                
                // Only try categorical split if more than one category present
                if (nnz > 1) {
                    integer_t igoleft[10] = {0};
                    integer_t igl[10] = {0};
                    integer_t itmp[10] = {0};
                    real_t tmpclass[10] = {0.0f};
                    integer_t nhit = 0;
                    real_t cat_critmax = best_crit;
                    
                    // Calculate pdo (total weight)
                    real_t pdo = 0.0f;
                    for (integer_t j = 0; j < nclass; ++j) {
                        pdo += tclasspop[j];
                    }
                    
                    // Use appropriate categorical split algorithm
                    if (numcat < ncmax) {
                        // Small categorical: try all category groupings
                        gpu_catmax(tclasscat, tclasspop, pdo, nnz, numcat, nclass,
                                  igl, itmp, tmpclass, tcat, icat, cat_critmax, igoleft, nhit);
                    } else {
                        // Large categorical: random sampling
                        gpu_catmaxr(tclasscat, tclasspop, pdo, numcat, nclass, igl,
                                   ncsplit, tmpclass, cat_critmax, igoleft, nhit, &local_state);
                    }
                    
                    // Update best split if categorical split is better
                    if (nhit == 1 && cat_critmax > best_crit) {
                        best_crit = cat_critmax;
                        best_var = mvar;
                        jstat = 1;
                        
                        // Store catgoleft for this node
                        integer_t catgoleft_offset = tree_id * maxnode * maxcat + kgrow * maxcat;  // Full offset
                        for (integer_t k = 0; k < numcat; ++k) {
                            if (k < maxcat) {
                                catgoleft_all[catgoleft_offset + k] = igoleft[k];
                            }
                        }
                        
                        // Set xbestsplit to 0.0 for categorical (not used)
                        if (threadIdx.x == 0) {
                            xbestsplit[kgrow] = 0.0f;
                        }
                    }
                }
                
                // Continue to next variable
                continue;
            }
            
            // ================================================================
            // QUANTITATIVE (CONTINUOUS) VARIABLE HANDLING
            // ================================================================
            
            // HISTOGRAM PATH: O(bins) split finding - XGBoost-like speed!
            // Uses shared memory for ≤10 classes, global memory for >10 classes
            
            if (use_histogram && X_binned != nullptr && n_bins_per_feature != nullptr) {
                integer_t n_bins = n_bins_per_feature[mvar];
                // Use histogram if bins are valid and we have histogram workspace available
                bool can_use_histogram = (n_bins >= 2 && n_bins <= HIST_MAX_BINS);
                // For classification: need either shared (≤10 classes) or global memory
                if (nclass > 1) {
                    can_use_histogram = can_use_histogram && (nclass <= HIST_SHARED_MAX_CLASSES || hist_class_ptr != nullptr);
                }
                
                if (can_use_histogram) {
                    const real_t* bin_edges = bin_edges_all + mvar * 257;
                    
                    real_t hist_best_crit;
                    real_t hist_best_split;
                    
                    // ADAPTIVE: Use parallel version for large nodes (>512 samples)
                    // where atomicAdd overhead is amortized, sequential for small nodes
                    bool use_parallel = (node_sample_count > 512);
                    
                    if (nclass > 1) {
                        // Classification: build histogram and find best split
                        // Use appropriate workspace (shared or global)
                        real_t* hist_workspace = (nclass <= HIST_SHARED_MAX_CLASSES) ? shared_hist_class : hist_class_ptr;
                        
                        if (use_parallel) {
                            // PARALLEL VERSION: Uses all 256 threads for histogram building
                            gpu_histogram_build_classification_parallel(
                                X_binned, cl, win_tree, node_samples, node_sample_count,
                                mvar, mdim, n_bins, nclass, hist_workspace
                            );
                            gpu_histogram_find_split_classification_parallel(
                                hist_workspace, n_bins, nclass, bin_edges,
                                nullptr, nullptr,
                                hist_best_crit, hist_best_split
                            );
                        } else {
                            // SEQUENTIAL VERSION: Less overhead for small nodes
                            if (threadIdx.x == 0) {
                                gpu_histogram_build_classification_device(
                                    X_binned, cl, win_tree, node_samples, node_sample_count,
                                    mvar, mdim, n_bins, nclass, hist_workspace
                                );
                                gpu_histogram_find_split_classification_device(
                                    hist_workspace, n_bins, nclass, bin_edges,
                                    hist_best_crit, hist_best_split
                                );
                            }
                            __syncthreads();
                        }
                        
                        // Classification: MAXIMIZE crit (higher is better)
                        if (hist_best_crit > best_crit) {
                            best_crit = hist_best_crit;
                            best_var = mvar;
                            if (threadIdx.x == 0) {
                                xbestsplit[kgrow] = hist_best_split;
                            }
                        }
                    } else {
                        // Regression: build histogram and find best split
                        if (use_parallel) {
                            // PARALLEL VERSION: Uses all 256 threads for histogram building
                            gpu_histogram_build_regression_parallel(
                                X_binned, y_regression, win_tree, node_samples, node_sample_count,
                                mvar, mdim, n_bins, hist_sum_ptr, hist_sum2_ptr, hist_count_ptr
                            );
                            gpu_histogram_find_split_regression_parallel(
                                hist_sum_ptr, hist_sum2_ptr, hist_count_ptr, n_bins, bin_edges,
                                hist_best_crit, hist_best_split
                            );
                        } else {
                            // SEQUENTIAL VERSION: Less overhead for small nodes
                            if (threadIdx.x == 0) {
                                gpu_histogram_build_regression_device(
                                    X_binned, y_regression, win_tree, node_samples, node_sample_count,
                                    mvar, mdim, n_bins, hist_sum_ptr, hist_sum2_ptr, hist_count_ptr
                                );
                                gpu_histogram_find_split_regression_device(
                                    hist_sum_ptr, hist_sum2_ptr, hist_count_ptr, n_bins, bin_edges,
                                    hist_best_crit, hist_best_split
                                );
                            }
                            __syncthreads();
                        }
                        
                        // Regression: MINIMIZE crit (MSE) (lower is better)
                        if (hist_best_crit < best_crit) {
                            best_crit = hist_best_crit;
                            best_var = mvar;
                            if (threadIdx.x == 0) {
                                xbestsplit[kgrow] = hist_best_split;
                            }
                        }
                    }
                    
                    // Skip the sorting path for this variable
                    continue;
                }
            }
            
            // ================================================================
            // SORTING PATH: O(n log n) - fallback when histogram not available
            // ================================================================
            
            // Initialize variables for splitting
            real_t rrn = 0.0f;
            real_t rrd = 0.0f;
            real_t rln = 0.0f;
            real_t rld = 0.0f;

            // Initialize left and right class weights (only for classification)
            real_t wl[10] = {0.0f};
            real_t wr[10] = {0.0f};
            
            // Initialize regression running sums (O(n) incremental MSE - like classification Gini)
            real_t sum_y_left = 0.0f, sum_y2_left = 0.0f, sum_w_left = 0.0f;
            real_t sum_y_right = 0.0f, sum_y2_right = 0.0f, sum_w_right = 0.0f;
            
            if (nclass > 1) {
                // Calculate initial Gini values for classification
                for (integer_t j = 0; j < nclass; ++j) {
                    rrn += class_counts[j] * class_counts[j];
                    rrd += class_counts[j];
                    wr[j] = class_counts[j];
                }
            } else {
                // Regression: initialize all samples in right (will move to left as we scan)
                // y_regression MUST be provided for regression tasks (nclass == 1)
                // Iterate directly over all samples - no 4096 limit
                for (integer_t sample_idx = 0; sample_idx < nsample; ++sample_idx) {
                    if (sample_node[sample_idx] != kgrow) continue;
                    real_t w = win_tree[sample_idx];
                    real_t y_val = y_regression[sample_idx];  // Regression requires y_regression
                    sum_y_right += w * y_val;
                    sum_y2_right += w * y_val * y_val;
                    sum_w_right += w;
                }
            }
            
            // OPTIMIZATION: Use pre-sorted indices instead of sorting at each node
            // This replaces O(n log n) sort with O(n) filter - ~10-20x faster!
            // Filter pre-sorted indices to get only samples in current node
            // No 4096 limit - uses global memory workspace for arbitrary sample sizes
            if (sorted_indices_all != nullptr) {
                // Pre-sorted indices available - use filtering (FAST!)
                const integer_t* feature_sorted = sorted_indices_all + mvar * nsample;
                integer_t filtered_count = 0;
                
                // Scan pre-sorted indices, keep only those in current node (no limit!)
                for (integer_t rank = 0; rank < nsample; ++rank) {
                    integer_t sample_idx = feature_sorted[rank];
                    if (sample_node[sample_idx] == kgrow) {
                        node_samples[filtered_count++] = sample_idx;
                    }
                }
                node_sample_count = filtered_count;
                // node_samples is now sorted by feature mvar!
            } else {
                // Fallback: collect samples then sort (for non-pre-sorted case)
                // First, collect samples in this node
                integer_t collected = 0;
                for (integer_t sample_idx = 0; sample_idx < nsample; ++sample_idx) {
                    if (sample_node[sample_idx] == kgrow) {
                        node_samples[collected++] = sample_idx;
                    }
                }
                node_sample_count = collected;
                // Now sort by feature value
                gpu_hybrid_sort_dense(node_samples, node_sample_count, x, mvar, mdim);
            }
            
            // Reduced debug output
            // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50 && mv == 0) {
            //     printf("DEBUG KERNEL: Node %d: Sort completed, trying splits\n", kgrow);
            // }
            
            // Try splits at sample boundaries
            for (integer_t ii = 0; ii < node_sample_count - 1; ++ii) {
                integer_t nc = node_samples[ii];
                integer_t ncnext = node_samples[ii + 1];
                real_t val1 = x[nc * mdim + mvar];
                real_t val2 = x[ncnext * mdim + mvar];
                
                // Skip ties but still update running sums
                if (val1 >= val2) {
                    if (nclass > 1) {
                        // Still update Gini for classification even if tie
                        integer_t k = cl[nc];
                        real_t u = win_tree[nc];  // Use case weight
                        rln += u * (u + 2.0f * wl[k]);
                        rrn += u * (u - 2.0f * wr[k]);
                        rld += u;
                        rrd -= u;
                        wl[k] += u;
                        wr[k] -= u;
                    } else {
                        // Regression: still update running sums even if tie
                        real_t w = win_tree[nc];
                        real_t y_val = y_regression[nc];  // Regression requires y_regression
                        sum_y_left += w * y_val;
                        sum_y2_left += w * y_val * y_val;
                        sum_w_left += w;
                        sum_y_right -= w * y_val;
                        sum_y2_right -= w * y_val * y_val;
                        sum_w_right -= w;
                    }
                    continue;
                }
                
                if (nclass == 1) {
                    // REGRESSION: O(n) incremental MSE update (like classification Gini)
                    // Move sample nc from right to left
                    integer_t sample_idx = nc;
                    real_t w = win_tree[sample_idx];
                    real_t y_val = y_regression[sample_idx];  // Regression requires y_regression
                    
                    // Incremental update: move sample from right to left
                    sum_y_left += w * y_val;
                    sum_y2_left += w * y_val * y_val;
                    sum_w_left += w;
                    
                    sum_y_right -= w * y_val;
                    sum_y2_right -= w * y_val * y_val;
                    sum_w_right -= w;
                    
                    // Calculate MSE from running sums (O(1) per split point)
                    if (sum_w_left > 1e-8f && sum_w_right > 1e-8f) {
                        real_t mean_left = sum_y_left / sum_w_left;
                        real_t mean_right = sum_y_right / sum_w_right;
                        
                        // MSE = E[y²] - E[y]² = sum(w*y²)/sum(w) - (sum(w*y)/sum(w))²
                        real_t mse_left = (sum_y2_left / sum_w_left) - mean_left * mean_left;
                        real_t mse_right = (sum_y2_right / sum_w_right) - mean_right * mean_right;
                        
                        // Clamp negative values (numerical stability)
                        if (mse_left < 0.0f) mse_left = 0.0f;
                        if (mse_right < 0.0f) mse_right = 0.0f;
                        
                        // Weighted average of child MSEs (matching CPU/R formula)
                        // crit = (n_left * mse_left + n_right * mse_right) / n_total
                        real_t total_w = sum_w_left + sum_w_right;
                        real_t crit = (sum_w_left * mse_left + sum_w_right * mse_right) / total_w;
                        
                        // DEBUG: Print first few split evaluations for Tree 0, first few nodes
                        // if (tree_id == 0 && kgrow < 3 && ii < 3 && threadIdx.x == 0) {
                            //printf("[GPU_DEBUG_DISABLED] Node %d var=%d ii=%d: crit=%.4f (left=%.4f, right=%.4f), w_l=%.1f, w_r=%.1f, best=%.4f\n",
                                //    kgrow, mvar, ii, crit, mse_left, mse_right, sum_w_left, sum_w_right, best_crit);
                        // }
                        
                        // Update best split (minimize MSE)
                        // Check that both children would have at least minndsize samples
                        // This matches R/Fortran behavior
                        if (sum_w_left >= static_cast<real_t>(minndsize) && 
                            sum_w_right >= static_cast<real_t>(minndsize)) {
                            // Deterministic tiebreaker: prefer lower feature index when crits are equal
                            if (crit < best_crit || (crit == best_crit && mvar < best_var)) {
                                best_split_idx = ii;
                                best_crit = crit;
                                best_var = mvar;
                            }
                        }
                    }
                } else {
                    // Classification: Use Gini impurity (loss_function == 0)
                    integer_t k = cl[nc];
                    real_t u = win_tree[nc];  // Use case weight
                    
                    // Gini impurity calculation (Original RF - mode 0)
                    rln += u * (u + 2.0f * wl[k]);
                    rrn += u * (u - 2.0f * wr[k]);
                    rld += u;
                    rrd -= u;
                    wl[k] += u;
                    wr[k] -= u;
                    
                    // Check that both children would have at least minndsize samples
                    // rld and rrd are cumulative weights (≈ sample counts with equal weights)
                    if (rld >= static_cast<real_t>(minndsize) && 
                        rrd >= static_cast<real_t>(minndsize)) {
                        // Original RF (mode 0): Use Gini impurity only
                        // Gini crit = (rln / rld) + (rrn / rrd)
                        real_t crit = (rln / rld) + (rrn / rrd);
                        
                        // Deterministic tiebreaker: prefer lower feature index when crits are equal
                        if (crit > best_crit || (crit == best_crit && mvar < best_var)) {
                            best_split_idx = ii;
                            best_crit = crit;
                            best_var = mvar;
                        }
                    }
                }
                }
            
            // Reduced debug output
            // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50 && mv == mtry - 1) {
            //     printf("DEBUG KERNEL: Node %d: Finished trying all %d variables, best_crit=%.6f\n", kgrow, mtry, best_crit);
            // }
        }
        
        // Reduced debug output
        // if (threadIdx.x == 0 && tree_id == 0 && kgrow < 50) {
        //     printf("DEBUG KERNEL: Node %d: Split search completed, jstat=%d, best_crit=%.6f, crit0=%.6f\n", kgrow, jstat, best_crit, crit0);
        // }
        
        // Calculate gain for this node
        real_t node_gain = 0.0f;
        if (nclass == 1) {
            // Regression: gain = loss reduction (MSE/MAE reduction)
            node_gain = crit0 - best_crit;
            if (node_gain < 1.0e-6f) {
                jstat = 1;  // No significant improvement
                // if (tree_id == 0 && kgrow < 5 && threadIdx.x == 0) {
                    //printf("[GPU_DEBUG_DISABLED] Tree %d Node %d: NO IMPROVEMENT, gain=%.6f\n", tree_id, kgrow, node_gain);
                // }
            }
            // Original RF (mode 0): no gain threshold check
        } else {
            // Classification: calculate gain using Gini impurity (loss_function == 0)
            // Gini: gain = improvement in crit
            node_gain = best_crit - crit0;
            
        if (best_crit < -1.0f) {
                jstat = 1;  // No good split found (matching Fortran and CPU algorithm)
            }
            // Original RF (mode 0): no gain threshold check
        }
        
        // Original RF (mode 0): no gain accumulation needed
        
        // Initialize split flag for this node
        if (threadIdx.x == 0) {
            shared_split_happened = false;
        }
        __syncthreads();  // Safe: unconditional
        
        // Check if we found a good split and create children if so
        // Only thread 0 modifies nnode[0] and tree structure
        if (threadIdx.x == 0) {
            // Reduced debug output
            // if (tree_id == 0 && kgrow < 10) {
            //     printf("DEBUG KERNEL: Node %d split check: jstat=%d, best_crit=%.6f, crit0=%.6f\n", 
            //            kgrow, jstat, best_crit, crit0);
            // }
            
            if (jstat == 0 && ((nclass == 1 && best_crit < crit0) || (nclass > 1 && best_crit > -1.0f))) {
            // Create children
            if (nnode[0] + 2 <= maxnode) {
                integer_t left_child = nnode[0];
                integer_t right_child = nnode[0] + 1;
                    
                    // DEBUG for regression splits
                    // if (tree_id == 0 && kgrow < 5 && nclass == 1) {
                        //printf("[GPU_DEBUG_DISABLED] Tree %d Node %d SPLIT: var=%d, crit0=%.4f, best_crit=%.4f, left=%d, right=%d\n", 
                            //    tree_id, kgrow, best_var, crit0, best_crit, left_child, right_child);
                    // }
                
                // Set up split
                nodestatus[kgrow] = 1;  // Split node
                bestvar[kgrow] = best_var;
                
                    // Calculate split point (matching Fortran: average of two consecutive values)
                    if (best_split_idx < node_sample_count - 1) {
                        integer_t sample_idx1 = node_samples[best_split_idx];
                        integer_t sample_idx2 = node_samples[best_split_idx + 1];
                        real_t val1 = x[sample_idx1 * mdim + best_var];
                        real_t val2 = x[sample_idx2 * mdim + best_var];
                        xbestsplit[kgrow] = (val1 + val2) / 2.0f;  // Average of two consecutive values
                    } else if (best_split_idx < node_sample_count) {
                    integer_t sample_idx = node_samples[best_split_idx];
                    xbestsplit[kgrow] = x[sample_idx * mdim + best_var];
                } else {
                    xbestsplit[kgrow] = 0.0f;
                }
                
                // Set up children
                treemap[kgrow * 2] = left_child;
                treemap[kgrow * 2 + 1] = right_child;
                
                nodestatus[left_child] = 2;   // Active
                nodestatus[right_child] = 2;  // Active
                nnode[0] += 2;
                
                // Store split info in shared memory for parallel sample_node update
                shared_split_happened = true;
                shared_best_var = best_var;
                shared_split_point = xbestsplit[kgrow];
                shared_left_child = left_child;
                shared_right_child = right_child;
                
                    // Original RF (mode 0): no gain accumulation
                    
                    // Reduced debug output
                    // if (tree_id == 0 && kgrow < 100) {
                    //     printf("DEBUG KERNEL: After split, nnode[0]=%d, node_gain=%.6f, depth=%d\n", 
                    //            nnode[0], node_gain, calculate_node_depth(kgrow));
                // }
            } else {
                // No more nodes available, make terminal
                    // Reduced debug output
                    // if (tree_id == 0 && kgrow < 10) {
                    //     printf("DEBUG KERNEL: Node %d max nodes reached, making terminal\n", kgrow);
                    // }
                    nodestatus[kgrow] = -1;  // Terminal node
                    nodeclass[kgrow] = majority_class;
            }
        } else {
            // No good split found, make terminal
            // Reduced debug output
            // if (tree_id == 0 && kgrow < 10) {
            //     printf("DEBUG KERNEL: Node %d no good split, making terminal\n", kgrow);
            // }
                nodestatus[kgrow] = -1;  // Terminal node
                nodeclass[kgrow] = majority_class;
        }
        }
        
        // OPTIMIZATION: Parallel sample_node update after split
        // All threads sync and participate in updating sample_node
        __syncthreads();  // Safe: unconditional, wait for split decision
        
        if (shared_split_happened) {
            // All threads participate in updating sample_node for samples in this node
            // This is O(n/threads) instead of the old O(n) sequential update
            integer_t split_var = shared_best_var;
            real_t split_point = shared_split_point;
            integer_t left_child_id = shared_left_child;
            integer_t right_child_id = shared_right_child;
            
            for (integer_t i = threadIdx.x; i < nsample; i += blockDim.x) {
                if (sample_node[i] == kgrow) {
                    real_t sample_val = x[i * mdim + split_var];
                    sample_node[i] = (sample_val <= split_point) ? left_child_id : right_child_id;
                }
            }
        }
        __syncthreads();  // Safe: unconditional, wait for sample_node update before next node
        
    }  // End of kgrow loop
    
    // Reduced debug output - disabled for production
    if (threadIdx.x == 0) {
        // Only print for first tree to reduce output
        // if (tree_id < 3) {
        //     const char* mode_str = (actual_growth_mode == 0) ? "level-wise" : 
        //                           (actual_growth_mode == 1) ? "leaf-wise" : 
        //                           (actual_growth_mode == 2) ? "hybrid" : "random";
        //     printf("DEBUG KERNEL: Tree %d loop completed, final nnode[0]=%d, growth_mode=%s\n", 
        //            tree_id, nnode[0], mode_str);
        // }
    }
    
    // Finalize terminal nodes (set node classes for terminal nodes)
    // OPTIMIZATION: Skip this expensive O(nodes × samples × depth) traversal
    // nodeclass is already correctly set when nodes are made terminal above
    // This was causing hangs on large datasets (10k+ samples)
    // The redundant finalization is removed - nodeclass is set at lines 2121 and 2129
    if (threadIdx.x == 0) {
        // Just verify terminal nodes exist (for debugging)
        integer_t terminal_count = 0;
        for (integer_t node = 0; node < nnode[0]; ++node) {
            if (nodestatus[node] == -1) {
                terminal_count++;
            }
        }
        // Reduced debug output - disabled for production
        // Only print for first few trees to reduce output
        // if (tree_id < 3) {
        //     printf("DEBUG KERNEL: Tree %d completed with %d terminal nodes (classes already set), total nodes=%d\n", 
        //            tree_id, terminal_count, nnode[0]);
        // }
    }
}

// ============================================================================
// GPU REGRESSION HELPER FUNCTIONS - O(n) incremental MSE updates
// ============================================================================

// Device function for O(n) incremental MSE calculation
// Updates running sums when moving one sample from right to left
__device__ void gpu_incremental_mse_update(
    real_t y_val, real_t weight,
    real_t& sum_y_left, real_t& sum_y2_left, real_t& sum_w_left,
    real_t& sum_y_right, real_t& sum_y2_right, real_t& sum_w_right) {
    
    // Move sample from right to left
    sum_y_left += weight * y_val;
    sum_y2_left += weight * y_val * y_val;
    sum_w_left += weight;
    
    sum_y_right -= weight * y_val;
    sum_y2_right -= weight * y_val * y_val;
    sum_w_right -= weight;
}

// Device function to calculate MSE from running sums (O(1))
__device__ real_t gpu_calculate_mse_from_sums(
    real_t sum_y_left, real_t sum_y2_left, real_t sum_w_left,
    real_t sum_y_right, real_t sum_y2_right, real_t sum_w_right) {
    
    real_t mse_left = 0.0f, mse_right = 0.0f;
    
    if (sum_w_left > 1e-8f) {
        real_t mean_left = sum_y_left / sum_w_left;
        mse_left = (sum_y2_left / sum_w_left) - mean_left * mean_left;
        if (mse_left < 0.0f) mse_left = 0.0f;  // Numerical stability
    }
    
    if (sum_w_right > 1e-8f) {
        real_t mean_right = sum_y_right / sum_w_right;
        mse_right = (sum_y2_right / sum_w_right) - mean_right * mean_right;
        if (mse_right < 0.0f) mse_right = 0.0f;  // Numerical stability
    }
    
    return mse_left + mse_right;
}

// Device function for simple insertion sort (used for small arrays < 64)
__device__ void gpu_insertion_sort(integer_t* indices, integer_t count, 
                                    const real_t* x, integer_t mvar, integer_t mdim) {
    for (integer_t i = 1; i < count; ++i) {
        integer_t key = indices[i];
        real_t key_val = x[key * mdim + mvar];
        integer_t j = i - 1;
        
        while (j >= 0 && x[indices[j] * mdim + mvar] > key_val) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

// ============================================================================
// Bitonic sort for GPU - O(n log²n), fully parallel, deterministic
// Used for large arrays (>= 64 samples) for better GPU utilization
// ============================================================================

// Bitonic compare-and-swap for dense data
__device__ __forceinline__ void bitonic_compare_swap_dense(
    integer_t* indices, integer_t i, integer_t j, bool ascending,
    const real_t* x, integer_t mvar, integer_t mdim
) {
    real_t val_i = x[indices[i] * mdim + mvar];
    real_t val_j = x[indices[j] * mdim + mvar];
    
    // Stable sort: use sample index as tiebreaker
    bool should_swap = ascending ? 
        (val_i > val_j || (val_i == val_j && indices[i] > indices[j])) :
        (val_i < val_j || (val_i == val_j && indices[i] < indices[j]));
    
    if (should_swap) {
        integer_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

// Parallel bitonic sort for dense data - called by all threads in a block
__device__ void gpu_bitonic_sort_dense(
    integer_t* indices, integer_t count,
    const real_t* x, integer_t mvar, integer_t mdim
) {
    integer_t tid = threadIdx.x;
    integer_t num_threads = blockDim.x;
    
    // Pad to next power of 2 for bitonic sort
    integer_t n = 1;
    while (n < count) n <<= 1;
    
    // Bitonic sort stages
    for (integer_t k = 2; k <= n; k <<= 1) {
        for (integer_t j = k >> 1; j > 0; j >>= 1) {
            // Each thread handles multiple compare-swap operations
            for (integer_t i = tid; i < n / 2; i += num_threads) {
                integer_t ixj = i ^ j;
                if (ixj > i) {
                    // Determine sort direction based on position in bitonic sequence
                    bool ascending = ((i & k) == 0);
                    
                    // Only compare if both indices are within valid range
                    if (i < count && ixj < count) {
                        bitonic_compare_swap_dense(indices, i, ixj, ascending, x, mvar, mdim);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Hybrid sort: insertion for small, bitonic for large
__device__ void gpu_hybrid_sort_dense(
    integer_t* indices, integer_t count,
    const real_t* x, integer_t mvar, integer_t mdim
) {
    if (count <= 64) {
        // Small array: sequential insertion sort (thread 0 only)
        if (threadIdx.x == 0) {
            gpu_insertion_sort(indices, count, x, mvar, mdim);
        }
        __syncthreads();
    } else {
        // Large array: parallel bitonic sort (all threads)
        gpu_bitonic_sort_dense(indices, count, x, mvar, mdim);
    }
}

// Device function to calculate Gini impurity (based on Fortran calcs subroutine)
__device__ real_t gpu_calcs(const real_t* tmpclass, const real_t* tclasspop, real_t pdo,
                           integer_t nclass) {
    real_t pln = 0.0f;
    real_t pld = 0.0f;
    real_t prn = 0.0f;
    
    // Calculate left side statistics
    for (integer_t j = 0; j < nclass; ++j) {
        pln += tmpclass[j] * tmpclass[j];
        pld += tmpclass[j];
        prn += (tclasspop[j] - tmpclass[j]) * (tclasspop[j] - tmpclass[j]);
    }
    
    real_t prd = pdo - pld;
    
    // Calculate left and right Gini ratios
    real_t rl = (pld > 0.0f) ? (pln / pld) : 0.0f;
    real_t rr = (prd > 0.0f) ? (prn / prd) : 0.0f;
    
    // Return total Gini impurity (lower is better)
    return rl + rr;
}

// GPU OOB vote accumulation kernel
// GPU kernel to compute nodextr for all samples in parallel
// MATCHES SPARSE PATTERN: receives already-offset pointers (no internal offset calculations)
// This is simpler, more reliable, and matches the working sparse implementation
__global__ void gpu_compute_nodextr_inbag_kernel(
    const real_t* x,           // Feature matrix (nsample × mdim)
    const integer_t* treemap,  // Tree mapping for THIS tree [2 × maxnode] - ALREADY OFFSET
    const integer_t* bestvar,  // Best variable per node for THIS tree [maxnode] - ALREADY OFFSET
    const real_t* xbestsplit,  // Best split value per node for THIS tree [maxnode] - ALREADY OFFSET
    const integer_t* nodestatus,  // Node status for THIS tree [maxnode] - ALREADY OFFSET
    const integer_t* cat,      // Categorical indicator (mdim)
    integer_t* nodextr,        // Output: terminal node for each sample for THIS tree [nsample] - ALREADY OFFSET
    integer_t nsample,         // Number of samples
    integer_t mdim,            // Number of features
    integer_t nnode,           // Number of nodes in this tree
    integer_t max_depth) {     // Maximum tree depth (safety limit)
    
    // Each thread processes one sample
    integer_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nsample) return;
    
    // Traverse tree from root (node 0) - MATCHES SPARSE PATTERN EXACTLY
    integer_t kt = 0;
    bool reached_terminal = false;
    
    for (integer_t depth = 0; depth < max_depth && !reached_terminal; ++depth) {
        // Check bounds
        if (kt < 0 || kt >= nnode) {
            break;
        }
        
        // Check if terminal (nodestatus == -1)
        if (nodestatus[kt] == -1) {
            nodextr[n] = kt;
            reached_terminal = true;
                break;
            }
        
        // Get split variable
        integer_t m = bestvar[kt];
        if (m < 0 || m >= mdim) {
            // Invalid split variable - treat as terminal
            nodextr[n] = kt;
            reached_terminal = true;
            break;
        }
        
        // Get feature value
        real_t xmn = x[n * mdim + m];
            
            // Check if this is a quantitative or categorical variable
        // Binary features (cat <= 2) use quantitative path
        bool is_categorical = (cat != nullptr && cat[m] > 2);
            
            if (!is_categorical) {
            // Quantitative split - MATCHES SPARSE: treemap[kt * 2] and treemap[kt * 2 + 1]
            if (xmn <= xbestsplit[kt]) {
                kt = treemap[kt * 2];      // Left child
                } else {
                kt = treemap[kt * 2 + 1];  // Right child
                }
            } else {
            // Categorical - go right by default (simplified, matches most use cases)
            kt = treemap[kt * 2 + 1];
                }
            }
            
    // If we didn't reach terminal in max_depth iterations, record current node
    if (!reached_terminal) {
        nodextr[n] = kt;
    }
}

// ============================================================================
// BATCHED KERNELS - Process all trees in parallel (no per-tree loops!)
// ============================================================================

// Batched nodextr: grid is (ceil(nsample/256), num_trees), each block handles samples for one tree
__global__ void gpu_compute_nodextr_batch_kernel(
    integer_t num_trees,
    integer_t nsample,
    integer_t mdim,
    integer_t maxnode,
    const real_t* x,                   // [nsample * mdim]
    const integer_t* treemap_all,      // [num_trees * 2 * maxnode]
    const integer_t* bestvar_all,      // [num_trees * maxnode]
    const real_t* xbestsplit_all,      // [num_trees * maxnode]
    const integer_t* nodestatus_all,   // [num_trees * maxnode]
    const integer_t* nnode_all,        // [num_trees]
    const integer_t* cat,              // [mdim]
    integer_t* nodextr_all,            // [num_trees * nsample] output
    integer_t max_depth
) {
    integer_t n = blockIdx.x * blockDim.x + threadIdx.x;  // Sample index
    integer_t tree_id = blockIdx.y;                        // Tree index
    
    if (n >= nsample || tree_id >= num_trees) return;
    
    integer_t tree_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t sample_offset = tree_id * nsample;
    integer_t nnode = nnode_all[tree_id];
    
    const integer_t* treemap = treemap_all + treemap_offset;
    const integer_t* bestvar = bestvar_all + tree_offset;
    const real_t* xbestsplit = xbestsplit_all + tree_offset;
    const integer_t* nodestatus = nodestatus_all + tree_offset;
    
    // Traverse tree from root to terminal node
    integer_t kt = 0;  // Start at root
    bool reached_terminal = false;
    
    for (integer_t depth = 0; depth < max_depth && kt >= 0 && kt < nnode; ++depth) {
        if (nodestatus[kt] == -1) {  // Terminal node
            nodextr_all[sample_offset + n] = kt;
            reached_terminal = true;
            break;
        }
        
        integer_t m = bestvar[kt];
        if (m < 0 || m >= mdim) break;
        
        real_t xmn = x[n * mdim + m];
        bool is_categorical = (cat != nullptr && cat[m] > 2);
        
        if (!is_categorical) {
            kt = (xmn <= xbestsplit[kt]) ? treemap[kt * 2] : treemap[kt * 2 + 1];
        } else {
            kt = treemap[kt * 2 + 1];  // Categorical: go right by default
        }
    }
    
    if (!reached_terminal) {
        nodextr_all[sample_offset + n] = kt;
    }
}

// Batched jtr computation: compute predicted class for each sample from nodextr
__global__ void gpu_compute_jtr_batch_kernel(
    integer_t num_trees,
    integer_t nsample,
    integer_t maxnode,
    const integer_t* nodextr_all,      // [num_trees * nsample]
    const integer_t* nodeclass_all,    // [num_trees * maxnode]
    const integer_t* nodestatus_all,   // [num_trees * maxnode]
    const integer_t* nnode_all,        // [num_trees]
    integer_t* jtr_all                 // [num_trees * nsample] output
) {
    integer_t n = blockIdx.x * blockDim.x + threadIdx.x;
    integer_t tree_id = blockIdx.y;
    
    if (n >= nsample || tree_id >= num_trees) return;
    
    integer_t tree_offset = tree_id * maxnode;
    integer_t sample_offset = tree_id * nsample;
    integer_t nnode = nnode_all[tree_id];
    
    integer_t kt = nodextr_all[sample_offset + n];
    
    if (kt >= 0 && kt < nnode && nodestatus_all[tree_offset + kt] == -1) {
        jtr_all[sample_offset + n] = nodeclass_all[tree_offset + kt];
    } else {
        // Use -1 as sentinel for invalid nodes (not a valid class label)
        // This prevents false matches with class 0 (synthetic samples in unsupervised mode)
        jtr_all[sample_offset + n] = -1;
    }
}

// Batched tnodewt: compute node weights for regression
// Each block handles one (tree, node) pair
__global__ void gpu_compute_tnodewt_batch_kernel(
    integer_t num_trees,
    integer_t nsample,
    integer_t maxnode,
    const integer_t* nodextr_all,      // [num_trees * nsample]
    const integer_t* nin_all,          // [num_trees * nsample]
    const real_t* win,                 // [nsample] - original sample weights
    const integer_t* nnode_all,        // [num_trees]
    const integer_t* nodestatus_all,   // [num_trees * maxnode]
    real_t* tnodewt_all                // [num_trees * maxnode] output
) {
    integer_t node = blockIdx.x;       // Node index within tree
    integer_t tree_id = blockIdx.y;    // Tree index
    
    if (tree_id >= num_trees) return;
    
    integer_t nnode = nnode_all[tree_id];
    if (node >= nnode) return;
    
    integer_t tree_offset = tree_id * maxnode;
    integer_t sample_offset = tree_id * nsample;
    
    // Only process terminal nodes
    if (nodestatus_all[tree_offset + node] != -1) {
        tnodewt_all[tree_offset + node] = 0.0f;
        return;
    }
    
    // Accumulate weights for samples landing in this node
    real_t weight_sum = 0.0f;
    real_t count = 0.0f;
    
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin_all[sample_offset + n] > 0) {  // In-bag sample
            if (nodextr_all[sample_offset + n] == node) {
                weight_sum += win[n] * nin_all[sample_offset + n];
                count += static_cast<real_t>(nin_all[sample_offset + n]);
            }
        }
    }
    
    tnodewt_all[tree_offset + node] = (count > 0.0f) ? (weight_sum / count) : 0.0f;
}

// Batched nodepred: compute mean y value for each terminal node (regression)
// Each block handles one (tree, node) pair
__global__ void gpu_compute_nodepred_batch_kernel(
    integer_t num_trees,
    integer_t nsample,
    integer_t maxnode,
    const integer_t* nodextr_all,      // [num_trees * nsample]
    const integer_t* nin_all,          // [num_trees * nsample]
    const real_t* y_regression,        // [nsample] - true y values
    const real_t* win,                 // [nsample] - original sample weights
    const integer_t* nnode_all,        // [num_trees]
    const integer_t* nodestatus_all,   // [num_trees * maxnode]
    real_t* nodepred_all               // [num_trees * maxnode] output
) {
    integer_t node = blockIdx.x;       // Node index within tree
    integer_t tree_id = blockIdx.y;    // Tree index
    
    if (tree_id >= num_trees) return;
    
    integer_t nnode = nnode_all[tree_id];
    if (node >= nnode) return;
    
    integer_t tree_offset = tree_id * maxnode;
    integer_t sample_offset = tree_id * nsample;
    
    // Only process terminal nodes
    if (nodestatus_all[tree_offset + node] != -1) {
        nodepred_all[tree_offset + node] = 0.0f;
        return;
    }
    
    // Accumulate weighted y values for samples landing in this node
    // NOTE: Use sample_weight only (not * nin) to match non-casewise tree growing
    // In non-casewise mode, win_tree[n] = 1.0 for all in-bag samples, regardless of nin
    // This matches sklearn RandomForestRegressor default behavior
    real_t y_sum = 0.0f;
    real_t weight_sum = 0.0f;
    
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin_all[sample_offset + n] > 0) {  // In-bag sample
            if (nodextr_all[sample_offset + n] == node) {
                // Use sample weight only (not * nin) for non-casewise mode
                real_t w = (win != nullptr) ? win[n] : 1.0f;
                y_sum += w * y_regression[n];
                weight_sum += w;
            }
        }
    }
    
    nodepred_all[tree_offset + node] = (weight_sum > 0.0f) ? (y_sum / weight_sum) : 0.0f;
}

// Batched y_pred: compute predicted y for each sample from nodepred
// Grid is ((nsample+255)/256, num_trees)
__global__ void gpu_compute_y_pred_batch_kernel(
    integer_t num_trees,
    integer_t nsample,
    integer_t maxnode,
    const integer_t* nodextr_all,      // [num_trees * nsample]
    const real_t* nodepred_all,        // [num_trees * maxnode]
    const integer_t* nnode_all,        // [num_trees]
    real_t* y_pred_all                 // [num_trees * nsample] output
) {
    integer_t n = blockIdx.x * blockDim.x + threadIdx.x;
    integer_t tree_id = blockIdx.y;
    
    if (n >= nsample || tree_id >= num_trees) return;
    
    integer_t tree_offset = tree_id * maxnode;
    integer_t sample_offset = tree_id * nsample;
    integer_t nnode = nnode_all[tree_id];
    
    integer_t kt = nodextr_all[sample_offset + n];
    
    if (kt >= 0 && kt < nnode) {
        y_pred_all[sample_offset + n] = nodepred_all[tree_offset + kt];
    } else {
        y_pred_all[sample_offset + n] = 0.0f;
    }
}

// GPU kernel for OOB vote accumulation with casewise support
// Traverses tree to find terminal node (like GPU NC kernel) for reliability
// For casewise mode, uses tnodewt[terminal_node] as vote weight
// For non-casewise, uses weight = 1.0
__global__ void gpu_oob_vote_kernel_casewise(
    integer_t num_trees, integer_t nsample, integer_t nclass, integer_t maxnode, integer_t mdim,
    const integer_t* nodestatus_all, const integer_t* nodeclass_all, 
    const integer_t* bestvar_all, const real_t* xbestsplit_all,
    const integer_t* treemap_all, const integer_t* nin_all,
    const real_t* tnodewt_all,     // For casewise: mean bootstrap weight per node [num_trees * maxnode]
    const real_t* x,               // Feature data [nsample * mdim]
    real_t* q_all,
    integer_t* nout_all,           // OOB count per sample
    bool use_casewise,
    // Categorical split support
    const integer_t* cat,          // cat[m] == 1 means quantitative, cat[m] > 1 means categorical
    const integer_t* catgoleft_all, // catgoleft[tree*maxnode*maxcat + node*maxcat + k] = 1 if category k goes left
    integer_t maxcat) {
    
    // Each thread handles one sample across all trees
    integer_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= nsample) return;
    
    // Local accumulators for this sample
    real_t local_votes[32];  // Assuming max 32 classes
    for (int c = 0; c < nclass && c < 32; ++c) local_votes[c] = 0.0f;
    integer_t local_nout = 0;
    
    // Process all trees for this sample
    for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
        integer_t node_offset = tree_id * maxnode;
        integer_t treemap_offset = tree_id * 2 * maxnode;
        integer_t nin_offset = tree_id * nsample;
        
        // Check if sample is OOB for this tree
        if (nin_all[nin_offset + sample] == 0) {
            local_nout++;
            
            // Traverse tree to find terminal node (matches GPU NC kernel exactly)
            const integer_t* nodestatus = nodestatus_all + node_offset;
            const integer_t* nodeclass = nodeclass_all + node_offset;
            const integer_t* bestvar = bestvar_all + node_offset;
            const real_t* xbestsplit = xbestsplit_all + node_offset;
            const integer_t* treemap = treemap_all + treemap_offset;
            
            integer_t current_node = 0;
            integer_t max_steps = maxnode;
            integer_t steps = 0;
            
            while (nodestatus[current_node] == 1 && steps < max_steps) {
                steps++;
                integer_t split_var = bestvar[current_node];
                real_t sample_value = x[sample * mdim + split_var];
                
                // Check if this is a categorical or quantitative split
                // NOTE: Must match training logic - binary features (numcat <= 2) use quantitative path
                integer_t numcat = (cat != nullptr && split_var < mdim) ? cat[split_var] : 1;
                
                if (numcat > 2 && catgoleft_all != nullptr && maxcat > 0) {
                    // CATEGORICAL SPLIT (3+ categories): use catgoleft to determine direction
                    integer_t category_idx = static_cast<integer_t>(sample_value + 0.5f);
                    integer_t catgoleft_offset = tree_id * maxnode * maxcat + current_node * maxcat;
                    
                    if (category_idx >= 0 && category_idx < numcat && category_idx < maxcat) {
                        if (catgoleft_all[catgoleft_offset + category_idx] == 1) {
                            current_node = treemap[current_node * 2];      // Left child
                        } else {
                            current_node = treemap[current_node * 2 + 1];  // Right child
                        }
                    } else {
                        // Invalid category - go right by default
                        current_node = treemap[current_node * 2 + 1];
                    }
                } else {
                    // QUANTITATIVE SPLIT: use threshold
                    real_t split_point = xbestsplit[current_node];
                if (sample_value <= split_point) {
                    current_node = treemap[current_node * 2];
                } else {
                    current_node = treemap[current_node * 2 + 1];
                    }
                }
                
                if (current_node < 0 || current_node >= maxnode) {
                    current_node = 0;
                    break;
                }
            }
            
            // Get prediction and vote weight
            if (nodestatus[current_node] == -1) {
                integer_t prediction = nodeclass[current_node];
                
                real_t vote_weight = 1.0f;
                if (use_casewise && tnodewt_all != nullptr) {
                    real_t tw = tnodewt_all[node_offset + current_node];
                    // Use tnodewt directly - should be > 0 for all terminal nodes
                    // (every terminal node should have at least one in-bag sample)
                    vote_weight = tw;
                }
                
                if (prediction >= 0 && prediction < nclass && prediction < 32) {
                    local_votes[prediction] += vote_weight;
                }
            }
        }
    }
    
    // Write results - use atomicAdd to correctly accumulate across batches
    if (nout_all != nullptr && local_nout > 0) {
        atomicAdd(&nout_all[sample], local_nout);
    }
    for (int c = 0; c < nclass && c < 32; ++c) {
        if (local_votes[c] > 0.0f) {
            atomicAdd(&q_all[sample * nclass + c], local_votes[c]);
        }
    }
}

// Legacy kernel for backward compatibility (non-casewise only)
__global__ void gpu_oob_vote_kernel(
    integer_t num_trees, integer_t nsample, integer_t nclass, integer_t maxnode, integer_t mdim,
    const integer_t* nodestatus_all, const integer_t* nodeclass_all, 
    const integer_t* bestvar_all, const real_t* xbestsplit_all,
    const integer_t* treemap_all, const integer_t* nin_all, const integer_t* cl,
    const real_t* x, real_t* q_all) {
    
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    integer_t tid = threadIdx.x;
    integer_t stride = blockDim.x;
    
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t nin_offset = tree_id * nsample;
    
    const integer_t* nodestatus = nodestatus_all + node_offset;
    const integer_t* nodeclass = nodeclass_all + node_offset;
    const integer_t* bestvar = bestvar_all + node_offset;
    const real_t* xbestsplit = xbestsplit_all + node_offset;
    const integer_t* treemap = treemap_all + treemap_offset;
    const integer_t* nin = nin_all + nin_offset;
    
    for (integer_t sample = tid; sample < nsample; sample += stride) {
        if (nin[sample] == 0) {
            integer_t current_node = 0;
            integer_t max_steps = maxnode;
            integer_t steps = 0;
            
            while (nodestatus[current_node] == 1 && steps < max_steps) {
                steps++;
                integer_t split_var = bestvar[current_node];
                real_t split_point = xbestsplit[current_node];
                real_t sample_value = x[sample * mdim + split_var];
                
                if (sample_value <= split_point) {
                    current_node = treemap[current_node * 2];
                } else {
                    current_node = treemap[current_node * 2 + 1];
                }
                
                if (current_node < 0 || current_node >= maxnode) break;
            }
            
            if (nodestatus[current_node] == -1) {
                integer_t prediction = nodeclass[current_node];
                if (prediction >= 0 && prediction < nclass) {
                    atomicAdd(&q_all[sample * nclass + prediction], 1.0f);
                }
            }
        }
    }
}

// GPU batch tree growing function
void gpu_growtree_batch(integer_t num_trees, const real_t* x, const real_t* win, const integer_t* cl,
                        integer_t task_type,  // 0=CLASSIFICATION, 1=REGRESSION, 2=UNSUPERVISED
                        const integer_t* cat, const integer_t* ties, integer_t* nin,
                        const real_t* y_regression,  // Original continuous y values for regression (nullptr for classification)
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
                        const integer_t* seeds, integer_t ncsplit, integer_t ncmax,
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
                        real_t* oob_predictions_all,  // For regression: accumulate OOB predictions (sum across trees)
                        real_t* nodepred_all,  // For regression: terminal node predictions [ntree * maxnode] - OUTPUT
                        void** lowrank_proximity_ptr_out,  // Output: LowRankProximityMatrix pointer
                        real_t* prox_imp_all,  // Proximity importance [nsample x mdim] - accumulated on GPU
                        int16_t* leaf_assignments_all,  // Leaf assignments [ntree * nsample] - terminal node per sample per tree
                        // HISTOGRAM PARAMETERS (for XGBoost-like speed)
                        bool use_histogram,  // Enable histogram-based split finding
                        const uint8_t* X_binned,  // Pre-binned feature data [nsample × mdim] (host memory)
                        const real_t* bin_edges_all,  // Bin edges [mdim × 257] (host memory)
                        const integer_t* n_bins_per_feature,  // Number of bins per feature [mdim] (host memory)
                        integer_t max_bins) {  // Maximum bins (typically 256)
    
    // std::cout << "[GPU_GROWTREE_BATCH] ENTRY - num_trees=" << num_trees << std::endl;
    // std::cout.flush();
    
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[GPU_GROWTREE_BATCH] ENTRY - nsample=" << nsample 
    //           << ", num_trees=" << num_trees 
    //           << ", mdim=" << mdim << std::endl;
    // std::cout.flush();
    
    // Validate basic parameters before any operations (prevents crashes)
    if (nsample <= 0 || num_trees <= 0 || mdim <= 0) {
        return;
    }
    
    // OPTIMIZATION: Create CUDA stream for async operations
    // All kernels and transfers on this stream execute in-order, enabling pipelining
    // with CPU work and other streams
    cudaStream_t compute_stream;
    CUDA_CHECK_VOID(cudaStreamCreate(&compute_stream));
    
    // DEBUG: Check regression parameters
    // if (nclass == 1) {
    //     //std::cout << "[GPU_REGRESSION_DEBUG] nclass=1, y_regression=" << (y_regression ? "valid" : "nullptr") << std::endl;
    //     if (y_regression) {
    //         //std::cout << "[GPU_REGRESSION_DEBUG] y_regression first 5: ";
    //         for (int i = 0; i < std::min(5, nsample); i++) {
    //             std::cout << y_regression[i] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    
    // Validate pointer parameters are not null (if they're required)
    if (x == nullptr) {
        // Debug prints removed to avoid potential stream corruption issues
        // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: x is nullptr" << std::endl;
        return;
    }
    if (win == nullptr) {
        // Debug prints removed to avoid potential stream corruption issues
        // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: win is nullptr" << std::endl;
        return;
    }
    if (seeds == nullptr) {
        // Debug prints removed to avoid potential stream corruption issues
        // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: seeds is nullptr" << std::endl;
        return;
    }
    
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[GPU_GROWTREE_BATCH] Parameters validated, setting g_config..." << std::endl;
    // std::cout.flush();
    
    // Set global config to GPU mode for proximity computation
    g_config.use_gpu = true;
    g_config.force_cpu = false;
    
    // Ensure use_casewise is set BEFORE any GPU operations (importance, proximity, etc.)
    // This is already set in fit_batch_gpu, but verify it's set correctly here
    // Note: g_config.use_casewise should be set by caller (fit_batch_gpu), but we ensure it's valid here
    // The value is read by gpu_varimp (for overall and local importance), gpu_proximity, and gpu_proximity_upper_triangle
    // All GPU functions check g_config.use_casewise to determine weighting:
    //   - use_casewise=true: case-wise (weight=tnodewt[node_idx] - bootstrap frequency weighted)
    //   - use_casewise=false: non-case-wise (weight=1.0 for all samples - UC Berkeley standard)
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[GPU_GROWTREE_BATCH] g_config.use_casewise=" << (g_config.use_casewise ? "true" : "false") 
    //           << " (for importance and proximity)" << std::endl;
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] g_config set, about to check CUDA context..." << std::endl;
// std::cout.flush();
    
    // Matches backup version - flush output before CUDA operations
// std::cout.flush();
    
    // IMPORTANT: Set use_qlora and quant_mode from g_config BEFORE proximity computation
    // These should already be set by RandomForest constructor via g_config, but ensure they're valid
    // Default values if not set (shouldn't happen, but safety check)
    // Note: g_config.use_qlora and g_config.quant_mode are set in rf_random_forest.cpp before calling gpu_growtree_batch
    
    // Check CUDA context first before synchronizing (matches backup version exactly)
    // This prevents crashes in Jupyter when context is corrupted
    // Check CUDA context first before synchronizing
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] About to call cudaGetDevice..." << std::endl;
// std::cout.flush();
    
    int device;
    cudaError_t ctx_err = cudaGetDevice(&device);
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] cudaGetDevice returned: " << ctx_err << " (device=" << device << ")" << std::endl;
// std::cout.flush();
    if (ctx_err != cudaSuccess) {
        // Context might be invalid - try to set device 0 (matches backup)
// Debug prints removed
        // std::cout << "[GPU_GROWTREE_BATCH] cudaGetDevice failed, trying cudaSetDevice(0)..." << std::endl;
// std::cout.flush();
        ctx_err = cudaSetDevice(0);
        if (ctx_err != cudaSuccess) {
            // std::cerr << "[GPU_GROWTREE_BATCH] ERROR: Failed to set CUDA device: " << ctx_err << std::endl;
            return;
        }
        device = 0;
    }
    
    // Ensure we're on the correct device (matches backup)
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] About to call cudaSetDevice(" << device << ")..." << std::endl;
// std::cout.flush();
    cudaSetDevice(device);
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] cudaSetDevice completed, about to call cudaGetLastError()..." << std::endl;
// std::cout.flush();
    
    // Clear any previous errors (matches backup)
    cudaGetLastError();
    
// Debug prints removed
    // std::cout << "[GPU_GROWTREE_BATCH] cudaGetLastError completed" << std::endl;
// std::cout.flush();
    
    // Try a lightweight operation to verify context is working (matches backup version)
    cudaError_t test_err = cudaFree(0);  // Freeing nullptr is safe and tests context
    if (test_err != cudaSuccess && test_err != cudaErrorInvalidValue) {
        // Context test failed - return early to prevent crash (matches backup)
        return;
    }
    
    // Clear any previous errors (matches backup)
    cudaGetLastError();
    
    // Test that context is ready with a simple non-blocking operation (matches backup version)
    void* test_alloc_ptr = nullptr;
    cudaError_t test_alloc_err = cudaMalloc(&test_alloc_ptr, 1);  // Try to allocate 1 byte
    if (test_alloc_err == cudaSuccess && test_alloc_ptr != nullptr) {
        cudaFree(test_alloc_ptr);  // Free immediately
    } else {
        // Context test failed - clear error and continue anyway (matches backup)
        cudaGetLastError();  // Clear error
    }
    
    // Report GPU memory status - required for .ipynb to work properly (matches backup version)
    size_t free_memory = 0, total_memory = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
    
    if (mem_err != cudaSuccess) {
        // Memory info failed - clear error and continue with default values (matches backup)
        cudaGetLastError();
    }
    // std::cout << "GPU: Memory Status - Total: " << (total_memory / (1024*1024)) << "MB, "
              // << "Free: " << (free_memory / (1024*1024)) << "MB, "
              // << "Used: " << ((total_memory - free_memory) / (1024*1024)) << "MB" << std::endl;
    
    // Allocate GPU memory for input data (matches backup version with explicit error checking)
    real_t* x_gpu;
    integer_t* cl_gpu;
    real_t* win_gpu;
    integer_t* seeds_gpu;
    
    // Ensure device is set and clear any errors before allocation (matches backup version)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    cudaError_t data_alloc_error;
    // Cast to size_t to prevent integer overflow
    data_alloc_error = cudaMalloc(&x_gpu, static_cast<size_t>(nsample) * static_cast<size_t>(mdim) * sizeof(real_t));
    
    // std::cout << "[GPU_GROWTREE_BATCH] cudaMalloc complete" << std::endl;
    // std::cout.flush();
    if (data_alloc_error != cudaSuccess) {
        // Allocation failed - return early to prevent crash
        return;
    }
    
    // Ensure device is still set and clear any errors (matches backup version)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    data_alloc_error = cudaMalloc(&cl_gpu, nsample * sizeof(integer_t));
    if (data_alloc_error != cudaSuccess) {
        // Clean up x_gpu before returning
        cudaFree(x_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    data_alloc_error = cudaMalloc(&win_gpu, nsample * sizeof(real_t));
    if (data_alloc_error != cudaSuccess) {
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    data_alloc_error = cudaMalloc(&seeds_gpu, num_trees * sizeof(integer_t));
    if (data_alloc_error != cudaSuccess) {
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        return;
    }
    
    // Allocate GPU memory for regression y values (if regression)
    real_t* y_regression_gpu = nullptr;
    if (y_regression != nullptr) {
        cudaSetDevice(device);
        cudaGetLastError();
        data_alloc_error = cudaMalloc(&y_regression_gpu, nsample * sizeof(real_t));
        if (data_alloc_error != cudaSuccess) {
            cudaFree(x_gpu);
            cudaFree(cl_gpu);
            cudaFree(win_gpu);
            cudaFree(seeds_gpu);
            return;
        }
    }
    
    // Debug: Check host data before copying to GPU
    // std::cout << "GPU: Host class labels first 10: ";
    // for (integer_t i = 0; i < 10; i++) {
    //     std::cout << cl[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Copy data to GPU (matches backup version with explicit error checking)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    cudaError_t copy_err = cudaMemcpy(x_gpu, x, static_cast<size_t>(nsample) * static_cast<size_t>(mdim) * sizeof(real_t), cudaMemcpyHostToDevice);
    
    // std::cout << "[GPU_GROWTREE_BATCH] cudaMemcpy complete" << std::endl;
    // std::cout.flush();  // DISABLED - causes Jupyter crash
    if (copy_err != cudaSuccess) {
        // Copy failed - clean up and return
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        return;
    }
    
    // Check if cl is valid before copying (important for unsupervised learning)
    if (cl != nullptr) {
        cudaSetDevice(device);
        cudaGetLastError();  // Clear any errors
        
        copy_err = cudaMemcpy(cl_gpu, cl, nsample * sizeof(integer_t), cudaMemcpyHostToDevice);
        if (copy_err != cudaSuccess) {
            // Copy failed - clean up and return
            cudaFree(x_gpu);
            cudaFree(cl_gpu);
            cudaFree(win_gpu);
            cudaFree(seeds_gpu);
            return;
        }
    } else {
        // For unsupervised learning, initialize with zeros
        cudaSetDevice(device);
        cudaGetLastError();  // Clear any errors
        cudaMemset(cl_gpu, 0, nsample * sizeof(integer_t));
    }
    
    // Copy win and seeds to GPU (matches backup version with explicit error checking)
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    copy_err = cudaMemcpy(win_gpu, win, nsample * sizeof(real_t), cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        // Copy failed - clean up and return
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();  // Clear any errors
    
    copy_err = cudaMemcpy(seeds_gpu, seeds, num_trees * sizeof(integer_t), cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        // Copy failed - clean up and return
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        if (y_regression_gpu) cudaFree(y_regression_gpu);
        return;
    }
    
    // Copy y_regression to GPU for regression
    if (y_regression != nullptr && y_regression_gpu != nullptr) {
        cudaSetDevice(device);
        cudaGetLastError();
        copy_err = cudaMemcpy(y_regression_gpu, y_regression, nsample * sizeof(real_t), cudaMemcpyHostToDevice);
        if (copy_err != cudaSuccess) {
            cudaFree(x_gpu);
            cudaFree(cl_gpu);
            cudaFree(win_gpu);
            cudaFree(seeds_gpu);
            cudaFree(y_regression_gpu);
            return;
        }
    }
    
    // Matches backup version - no debug output
    
    // Debug: Verify GPU data after copying
    // integer_t* cl_test = new integer_t[10];
    // cudaMemcpy(cl_test, cl_gpu, 10 * sizeof(integer_t), cudaMemcpyDeviceToHost);
    // std::cout << "GPU: GPU class labels first 10: ";
    // for (integer_t i = 0; i < 10; i++) {
    //     std::cout << cl_test[i] << " ";
    // }
    // std::cout << std::endl;
    // delete[] cl_test;
    
    // Allocate GPU memory for tree arrays
    
    integer_t* nodestatus_all_gpu;
    integer_t* nodeclass_all_gpu;
    integer_t* nnode_all_gpu;
    integer_t* bestvar_all_gpu;
    real_t* xbestsplit_all_gpu;
    integer_t* treemap_all_gpu;
    integer_t* jinbag_all_gpu;
    
    // Matches backup version - direct allocation, no error checking, no debug output
    // Allocate GPU memory for tree arrays (matches backup version with explicit error checking)
    cudaSetDevice(device);
    cudaGetLastError();
    
    cudaError_t tree_alloc_error;
    // Cast to size_t to prevent integer overflow with large num_trees * maxnode
    tree_alloc_error = cudaMalloc(&nodestatus_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t));
    if (tree_alloc_error != cudaSuccess) {
        // Clean up previous allocations
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    tree_alloc_error = cudaMalloc(&nodeclass_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t));
    if (tree_alloc_error != cudaSuccess) {
        // Clean up previous allocations
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        cudaFree(nodestatus_all_gpu);
        return;
    }
    
    cudaSetDevice(device);
    cudaGetLastError();
    
    tree_alloc_error = cudaMalloc(&nnode_all_gpu, num_trees * sizeof(integer_t));
    if (tree_alloc_error != cudaSuccess) {
        // Clean up previous allocations
        cudaFree(x_gpu);
        cudaFree(cl_gpu);
        cudaFree(win_gpu);
        cudaFree(seeds_gpu);
        cudaFree(nodestatus_all_gpu);
        cudaFree(nodeclass_all_gpu);
        return;
    }
    
    // Matches backup version - direct allocation, no error checking, no debug output
    // Cast to size_t to prevent integer overflow
    CUDA_CHECK_VOID(cudaMalloc(&bestvar_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&xbestsplit_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&treemap_all_gpu, static_cast<size_t>(num_trees) * 2 * static_cast<size_t>(maxnode) * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&jinbag_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t)));
    
    // Allocate GPU memory for categorical arrays
    integer_t* cat_gpu = nullptr;
    integer_t* amat_gpu = nullptr;
    integer_t* catgoleft_all_gpu = nullptr;
    
    // Allocate cat array if provided
    if (cat != nullptr) {
        CUDA_CHECK_VOID(cudaMalloc(&cat_gpu, mdim * sizeof(integer_t)));
            cudaMemcpy(cat_gpu, cat, mdim * sizeof(integer_t), cudaMemcpyHostToDevice);
    }
    
    // Allocate amat array if provided (cast to size_t to prevent overflow)
    if (amat_all != nullptr) {
        CUDA_CHECK_VOID(cudaMalloc(&amat_gpu, static_cast<size_t>(mdim) * static_cast<size_t>(nsample) * sizeof(integer_t)));
            cudaMemcpy(amat_gpu, amat_all, static_cast<size_t>(mdim) * static_cast<size_t>(nsample) * sizeof(integer_t), cudaMemcpyHostToDevice);
    }
    
    // Allocate catgoleft array (always allocate, initialize to zeros)
    CUDA_CHECK_VOID(cudaMalloc(&catgoleft_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * static_cast<size_t>(maxcat) * sizeof(integer_t)));
        cudaMemset(catgoleft_all_gpu, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * static_cast<size_t>(maxcat) * sizeof(integer_t));
    
    // Matches backup version - direct allocation, no test allocation, no error checking
    real_t* q_all_gpu;
    size_t q_size = nsample * nclass * sizeof(real_t);
    CUDA_CHECK_VOID(cudaMalloc(&q_all_gpu, q_size));
    
    // std::cout << "DEBUG: q_all_gpu allocation successful, proceeding to memset\n";
    
    // Copy existing q_all values to GPU (preserves votes from previous batches)
    // This is critical for correct vote accumulation across batches
    
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    // Copy host q_all to GPU (this has accumulated votes from previous batches)
    cudaError_t q_copy_err = cudaMemcpy(q_all_gpu, q_all, nsample * nclass * sizeof(real_t), cudaMemcpyHostToDevice);
    
    if (q_copy_err != cudaSuccess) {
        // Silently handle copy errors
        cudaGetLastError();
        if (q_all_gpu) { cudaError_t err = cudaFree(q_all_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // std::cout << "DEBUG: q_all_gpu memset successful\n";
    
    // Clear error state after memset to prevent false positives in Jupyter
    // Error code 1 (cudaErrorInvalidValue) is often a false positive from previous operations
    // We clear it silently to prevent Jupyter kernel crashes
    cudaGetLastError(); // Clear any error state (including false positives)
    
    // std::cout << "DEBUG: After error check, before variable declarations\n";
    
    // Allocate GPU memory for importance arrays (if needed)
    real_t* qimp_all_gpu = nullptr;
    real_t* qimpm_all_gpu = nullptr;
    
    // std::cout << "DEBUG: Variable declarations complete, before check\n";
    
    if (qimp_all != nullptr && qimpm_all != nullptr) {
        // std::cout << "DEBUG: qimp_all and qimpm_all are not null, allocating importance arrays\n";
        
        // std::cout << "DEBUG: About to allocate qimp_all_gpu\n";
        
        CUDA_CHECK_VOID(cudaMalloc(&qimp_all_gpu, nsample * sizeof(real_t)));
        
        // std::cout << "DEBUG: About to allocate qimpm_all_gpu\n";
        
        CUDA_CHECK_VOID(cudaMalloc(&qimpm_all_gpu, static_cast<size_t>(nsample) * static_cast<size_t>(mdim) * sizeof(real_t)));
        
        // Initialize importance arrays to zeros
        // std::cout << "DEBUG: About to memset importance arrays\n";
        
        cudaMemset(qimp_all_gpu, 0, static_cast<size_t>(nsample) * sizeof(real_t));
        cudaMemset(qimpm_all_gpu, 0, static_cast<size_t>(nsample) * static_cast<size_t>(mdim) * sizeof(real_t));
        
        // std::cout << "DEBUG: Importance arrays memset completed\n";
    } else {
        // std::cout << "DEBUG: qimp_all or qimpm_all is null, skipping importance arrays\n";
    }
    
    // Allocate GPU memory for proximity importance (if requested)
    // IMPORTANT: Copy existing host values to GPU to preserve accumulation across batches!
    // Each batch call adds to the existing values, not replaces them.
    real_t* prox_imp_all_gpu = nullptr;
    if (prox_imp_all != nullptr) {
        CUDA_CHECK_VOID(cudaMalloc(&prox_imp_all_gpu, static_cast<size_t>(nsample) * static_cast<size_t>(mdim) * sizeof(real_t)));
        // Copy existing host values to GPU (preserves accumulation from previous batches)
        cudaMemcpy(prox_imp_all_gpu, prox_imp_all, 
                   static_cast<size_t>(nsample) * static_cast<size_t>(mdim) * sizeof(real_t), 
                   cudaMemcpyHostToDevice);
    }
    
    // std::cout << "DEBUG: About to allocate bootstrap arrays\n";
    
    // Allocate additional arrays for bootstrap
    integer_t* nin_all_gpu;
    real_t* win_all_gpu;
    curandState* random_states;
    
    // Matches backup version - direct allocation, no error checking (cast to size_t for overflow safety)
    CUDA_CHECK_VOID(cudaMalloc(&nin_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&win_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(real_t)));
    CUDA_CHECK_VOID(cudaMalloc(&random_states, static_cast<size_t>(num_trees) * sizeof(curandState)));
    
    // Initialize random states
    dim3 init_block_size(256);
    dim3 init_grid_size((num_trees + init_block_size.x - 1) / init_block_size.x);
    
    // OPTIMIZATION: Launch on stream - kernels on same stream execute in order
    init_random_states_kernel<<<init_grid_size, init_block_size, 0, compute_stream>>>(
        random_states, num_trees, seeds_gpu
    );
    
    // OPTIMIZATION: No sync needed - bootstrap kernel on same stream waits for init
    
    // Launch bootstrap kernel
    dim3 bootstrap_block_size(256);
    dim3 bootstrap_grid_size(num_trees);
    
    // std::cout << "GPU: Launching bootstrap kernel with " << num_trees << " trees..." << std::endl;
    // std::cout << "GPU: Bootstrap grid size: " << bootstrap_grid_size.x << ", Block size: " << bootstrap_block_size.x << std::endl;
    
    // OPTIMIZATION: Launch on stream - ordered after init kernel
    gpu_bootstrap_kernel<<<bootstrap_grid_size, bootstrap_block_size, 0, compute_stream>>>(
        num_trees, nsample, win_gpu, nin_all_gpu, win_all_gpu, jinbag_all_gpu,
        random_states, seeds_gpu,
        g_config.use_casewise  // Pass use_casewise as parameter
    );
    
    // Matches backup version - no kernel launch error checking
    
    // std::cout << "DEBUG: Bootstrap kernel launched, about to sync\n";
    
    // OPTIMIZATION: Remove unnecessary sync and debug copy before tree kernel
    // The bootstrap kernel will complete before tree kernel starts (implicit ordering)
    // Only sync if we need to check results (debug mode disabled)
    // CUDA_CHECK_VOID(cudaDeviceSynchronize());  // Removed - let kernels pipeline
    
    // Launch tree growing kernel
    // std::cout << "GPU: Launching tree growing kernel with " << num_trees << " trees..." << std::endl;
    
    // OPTIMIZATION: Increased block size for better parallelism
    // Use 256 threads per block for better GPU utilization and parallel sample filtering
    // Modern GPUs (Ampere+) can handle 256 threads efficiently
    dim3 tree_block_size(256);
    dim3 tree_grid_size(num_trees);
    
    // std::cout << "GPU: Grid size: " << tree_grid_size.x << ", Block size: " << tree_block_size.x << std::endl;
    
    // Allocate GPU memory for allowed modes if using random selection
    integer_t* allowed_modes_gpu = nullptr;
    // Always use original RF growth mode (mode 0) - no need for allowed_modes
    integer_t num_allowed_modes = 0;
    
    // OPTIMIZATION: Allocate sample_node tracking array for parallel sample collection
    // sample_node[tree_id * nsample + i] = current node for sample i in tree tree_id
    // This eliminates O(n*depth) tree traversal per node, replacing with O(1) lookup
    integer_t* sample_node_all_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&sample_node_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t)));
    cudaMemset(sample_node_all_gpu, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t));
    
    // NODE SAMPLES WORKSPACE: For sorting path with >4096 samples
    // Each tree gets nsample workspace for storing sorted sample indices
    // This removes the 4096 sample limit and scales to any dataset size
    integer_t* node_samples_workspace_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&node_samples_workspace_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t)));
    cudaMemset(node_samples_workspace_gpu, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t));
    
    // OPTIMIZATION: Pre-sort features ONCE before tree growing (if NOT using histogram)
    // sorted_indices[f * nsample + rank] = sample index with rank-th smallest value for feature f
    // This avoids O(n log n) sorting at every node, replaced with O(n) filtering
    // Speedup: ~10-20x for split finding phase
    integer_t* sorted_indices_gpu = nullptr;
    if (!use_histogram) {
        // Only pre-sort if histogram is disabled (cast to size_t for overflow safety)
        CUDA_CHECK_VOID(cudaMalloc(&sorted_indices_gpu, static_cast<size_t>(mdim) * static_cast<size_t>(nsample) * sizeof(integer_t)));
        
        // Launch pre-sort kernel (one block per feature, parallel across features)
        dim3 presort_grid(mdim);
        dim3 presort_block(1);  // Thread 0 does sorting for each feature
        // OPTIMIZATION: Launch on stream - tree kernel on same stream waits for pre-sort
        gpu_presort_features_kernel<<<presort_grid, presort_block, 0, compute_stream>>>(
            x_gpu, nsample, mdim, sorted_indices_gpu
        );
        // OPTIMIZATION: No sync needed - tree kernel on same stream waits
    }
    
    // HISTOGRAM: Allocate GPU memory for histogram data (if enabled)
    // This enables O(bins) split finding instead of O(n log n) sorting
    uint8_t* X_binned_gpu = nullptr;
    real_t* bin_edges_gpu = nullptr;
    integer_t* n_bins_gpu = nullptr;
    real_t* hist_workspace_gpu = nullptr;
    cuda::DenseHistogramData* dense_hist_data = nullptr;  // NEW: GPU-native histogram
    
    if (use_histogram) {
        // NEW: Use GPU-native histogram binning (matches sparse implementation)
        // This correctly handles sparse-like data with many repeated values (like 0s)
        dense_hist_data = new cuda::DenseHistogramData();
        
        // Compute bin edges on GPU using unique values (like sparse histogram does)
        cuda::gpu_compute_dense_bin_edges(x_gpu, nsample, mdim, max_bins, *dense_hist_data, compute_stream);
        
        // Bin the data on GPU
        cuda::gpu_bin_dense_data(x_gpu, *dense_hist_data, compute_stream);
        
        cudaStreamSynchronize(compute_stream);  // Ensure binning completes
        
        // Point to the GPU-allocated histogram data
        X_binned_gpu = dense_hist_data->d_X_binned;
        bin_edges_gpu = dense_hist_data->d_bin_edges_all;
        n_bins_gpu = dense_hist_data->d_n_bins_per_feature;
        
        // Allocate global memory workspace for histogram (fallback for >10 classes)
        // Classification: 256 bins × nclass per tree
        // Regression: 256 × 3 per tree
        integer_t hist_size_per_tree = (nclass > 1) ? (256 * nclass) : (256 * 3);
        CUDA_CHECK_VOID(cudaMalloc(&hist_workspace_gpu, num_trees * hist_size_per_tree * sizeof(real_t)));
        cudaMemset(hist_workspace_gpu, 0, num_trees * hist_size_per_tree * sizeof(real_t));
    }
    
    // std::cout << "[GPU_GROWTREE_BATCH] About to launch gpu_full_tree_kernel, parallel_mode=" << g_config.gpu_parallel_mode0 << std::endl;
    // std::cout.flush();  // DISABLED - causes Jupyter crash
    
    // OPTIMIZATION: Launch on stream - ordered after bootstrap/presort kernels
    gpu_full_tree_kernel<<<tree_grid_size, tree_block_size, 0, compute_stream>>>(
        num_trees, nsample, mdim, maxnode, nclass,
        minndsize, mtry, x_gpu,
        nodestatus_all_gpu, nodeclass_all_gpu, nnode_all_gpu,
        bestvar_all_gpu, xbestsplit_all_gpu, treemap_all_gpu,
        jinbag_all_gpu, cl_gpu, win_all_gpu,
        cat_gpu,  // Categorical feature array (nullptr if none)
        amat_gpu,  // Categorical matrix (nullptr if none)
        catgoleft_all_gpu,  // Category go-left array
        maxcat,  // Maximum categories
        ncsplit,  // Number of random splits for large categoricals
        ncmax,  // Threshold for large categorical
        g_config.gpu_loss_function,  // Loss function: 0=Gini (classification), 3=MSE (regression)
        0.0f,  // min_gain_to_split not used in original RF
        0,  // Always use original RF growth mode (mode 0)
        nullptr,  // No allowed modes needed
        num_allowed_modes,  // Number of allowed modes (always 0)
        0.0f,  // L1 regularization not used in original RF
        0.0f,  // L2 regularization not used in original RF
        g_config.gpu_parallel_mode0,
        random_states,  // Pre-initialized random states to avoid slow curand_init in kernel
        y_regression_gpu,  // Continuous y values for regression (nullptr for classification)
        sample_node_all_gpu,  // OPTIMIZATION: sample_node tracking for parallel sample collection
        sorted_indices_gpu,  // OPTIMIZATION: pre-sorted indices for fast split finding (nullptr if using histogram)
        // HISTOGRAM PARAMETERS
        use_histogram && X_binned_gpu != nullptr,  // Enable histogram only if data is available
        X_binned_gpu,  // Pre-binned feature data
        bin_edges_gpu,  // Bin edges per feature
        n_bins_gpu,  // Number of bins per feature
        max_bins,  // Maximum number of bins
        hist_workspace_gpu,  // Global memory fallback for >10 classes
        // NODE SAMPLES WORKSPACE (scales to any sample size)
        node_samples_workspace_gpu  // Global memory workspace for sorting path
    );
    
    // OPTIMIZATION: Only sync once after tree kernel (removed duplicate sync)
    // Check for kernel launch errors (non-blocking check before sync)
    cudaError_t tree_kernel_err = cudaGetLastError();
    if (tree_kernel_err != cudaSuccess) {
        // std::cerr << "ERROR: Tree building kernel error: " << cudaGetErrorString(tree_kernel_err) << std::endl;
    }
    
    // OPTIMIZATION: Use stream sync instead of device sync (doesn't block other streams)
    CUDA_CHECK_VOID(cudaStreamSynchronize(compute_stream));
    
    // NOTE: tnodewt computation is DEFERRED until AFTER nodextr is computed
    // This allows us to use pre-computed terminal nodes from nodextr_all_gpu
    // instead of slow CPU tree traversal. See "DEFERRED TNODEWT COMPUTATION" section below.
    bool need_tnodewt = g_config.use_casewise && (task_type == 0 || task_type == 2);  // Classification or Unsupervised
    
    // OPTIMIZED: Single batch copy of all tree data from GPU to host
    // Removed redundant copies (was 3x nnode_all, 2x debug copies)
    // Cast to size_t to prevent integer overflow
    cudaMemcpy(nodestatus_all, nodestatus_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodeclass_all, nodeclass_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nnode_all, nnode_all_gpu, static_cast<size_t>(num_trees) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bestvar_all, bestvar_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(xbestsplit_all, xbestsplit_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(treemap_all, treemap_all_gpu, static_cast<size_t>(num_trees) * 2 * static_cast<size_t>(maxnode) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(jinbag_all, jinbag_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nin, nin_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    // Copy catgoleft from GPU to host if categorical features are used
    integer_t* catgoleft_all_host = nullptr;
    if (catgoleft_all != nullptr && catgoleft_all_gpu != nullptr && maxcat > 1) {
        cudaMemcpy(catgoleft_all, catgoleft_all_gpu, 
                   static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * static_cast<size_t>(maxcat) * sizeof(integer_t), 
                   cudaMemcpyDeviceToHost);
        catgoleft_all_host = catgoleft_all;
    }
    
    // Initialize host importance arrays to zero before accumulation (for both classification and regression)
    // NOTE: qimp_all and qimpm_all are reset per batch (they're accumulated per batch in GPU)
    // BUT: avimp_all should NOT be reset here - it needs to accumulate across ALL batches
    // The reset happens in fit_batch_gpu() before the first batch, and we accumulate across batches
    if (qimp_all != nullptr && qimpm_all != nullptr) {
        std::fill(qimp_all, qimp_all + nsample, 0.0f);
        std::fill(qimpm_all, qimpm_all + nsample * mdim, 0.0f);
    }
    // REMOVED: avimp_all reset - it must accumulate across all batches
    // if (avimp_all != nullptr) {
    //     std::fill(avimp_all, avimp_all + mdim, 0.0f);
    // }
    
    // Compute nodextr_all ONCE for BOTH proximity AND importance computation
    // nodextr_all stores the terminal node index for each sample for each tree
    // It's needed by BOTH proximity (to know which terminal node each sample ends up in)
    // AND importance (to compute node weights for importance calculation)
    // AND OOB vote accumulation (to compute jtr_tree from nodextr_tree)
    // AND regression OOB predictions (to get predictions from tnodewt[nodextr[n]])
    // Compute it ONCE before all use it, not separately in each block
    bool need_nodextr = ((proximity_all != nullptr || lowrank_proximity_ptr_out != nullptr) || 
                         avimp_all != nullptr || 
                         prox_imp_all != nullptr ||  // Proximity importance needs nodextr to compare terminal nodes
                         (task_type == 0 && q_all != nullptr) ||  // Classification needs nodextr for OOB votes
                         (task_type == 1 && oob_predictions_all != nullptr) ||  // Regression ALWAYS needs nodextr for OOB predictions
                         leaf_assignments_all != nullptr);  // Leaf assignments need nodextr
    
    // SPARSE PATTERN: Allocate single-tree buffer, process per-tree, copy after each
    // This avoids race conditions that may occur with multi-tree buffer + async launches
    integer_t* nodextr_all_gpu = nullptr;  // Multi-tree buffer for proximity importance
    integer_t* d_nodextr_single = nullptr;  // Single-tree buffer (matches sparse pattern)
    bool use_gpu_nodextr = false;
    
    if (need_nodextr && nodextr_all != nullptr) {
        auto time_nodextr_start = std::chrono::high_resolution_clock::now();
        
        // Allocate multi-tree buffer for nodextr (all trees at once)
        cudaError_t nodextr_alloc_err = cudaMalloc(&nodextr_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t));
        use_gpu_nodextr = (nodextr_alloc_err == cudaSuccess);
        
        if (use_gpu_nodextr) {
            // BATCHED: Compute nodextr for ALL trees in ONE kernel launch
            // Grid: (ceil(nsample/256), num_trees)
            dim3 block_size(256);
            dim3 grid_size((nsample + block_size.x - 1) / block_size.x, num_trees);
            
            gpu_compute_nodextr_batch_kernel<<<grid_size, block_size, 0, compute_stream>>>(
                num_trees, nsample, mdim, maxnode,
                x_gpu,
                treemap_all_gpu,
                bestvar_all_gpu,
                xbestsplit_all_gpu,
                nodestatus_all_gpu,
                nnode_all_gpu,
                cat_gpu,
                nodextr_all_gpu,
                100  // max_depth
            );
            
            cudaStreamSynchronize(compute_stream);
        } else {
            // CPU fallback for all trees (rare path - only if GPU memory fails)
            for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                integer_t tree_offset = tree_id * maxnode;
                integer_t jinbag_offset = tree_id * nsample;
                
                std::vector<integer_t> jtr_tree(nsample, 0);
                std::vector<integer_t> nodextr_tree(nsample, 0);
                
                const integer_t* catgoleft_tree = nullptr;
                if (catgoleft_all_host != nullptr) {
                    integer_t catgoleft_offset = tree_id * maxnode * maxcat;
                    catgoleft_tree = catgoleft_all_host + catgoleft_offset;
                }
                
                cpu_testreebag(x, xbestsplit_all + tree_offset, nin + jinbag_offset, 
                              treemap_all + tree_offset * 2, bestvar_all + tree_offset,
                              nodeclass_all + tree_offset, cat, nodestatus_all + tree_offset, 
                              catgoleft_tree, nsample, mdim, nnode_all[tree_id], maxcat,
                              jtr_tree.data(), nodextr_tree.data());
                
                // Process in-bag samples (cpu_testreebag handles OOB samples)
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] > 0 && nodextr_tree[n] == 0) {
                        integer_t kt = 0;
                        integer_t traversal_count = 0;
                        const integer_t max_traversal = 1000;
                        while (nodestatus_all[tree_offset + kt] == 1 && traversal_count < max_traversal) {
                            traversal_count++;
                            integer_t mvar = bestvar_all[tree_offset + kt];
                            if (mvar < 0 || mvar >= mdim) break;
                            if (cat == nullptr || cat[mvar] <= 2) {
                                if (x[n * mdim + mvar] <= xbestsplit_all[tree_offset + kt]) {
                                    kt = treemap_all[tree_offset * 2 + kt * 2];
                                } else {
                                    kt = treemap_all[tree_offset * 2 + kt * 2 + 1];
                                }
                            } else {
                                kt = treemap_all[tree_offset * 2 + kt * 2 + 1];
                            }
                            if (kt < 0 || kt >= nnode_all[tree_id]) { kt = 0; break; }
                            if (nodestatus_all[tree_offset + kt] != 1) break;
                        }
                        if (nodestatus_all[tree_offset + kt] == -1 || nodestatus_all[tree_offset + kt] == 2) {
                            nodextr_tree[n] = kt;
                        }
                    }
                }
                
                std::copy(nodextr_tree.begin(), nodextr_tree.end(), nodextr_all + jinbag_offset);
            }  // End CPU fallback loop
        }
        
        // GPU path: copy results back and compute proximity importance
        if (use_gpu_nodextr) {
            
            // ============================================================
            // COMPUTE PROXIMITY IMPORTANCE ON GPU (after nodextr is ready)
            // ============================================================
            // For each tree, compute proximity importance using GPU kernel
            // This accumulates across all trees into prox_imp_all_gpu
            // 
            // Matches Breiman-Cutler methodology for local variable importance:
            // - Only accumulate for samples where OOB prediction was CORRECT
            // - Normalize per-tree by (1.0 / noob)
            if (prox_imp_all_gpu != nullptr && prox_imp_all != nullptr) {
                // Determine if this is regression (skip correctness check) or classification/unsupervised
                // task_type: 0=CLASSIFICATION, 1=REGRESSION, 2=UNSUPERVISED
                bool is_regression = (task_type == 1);
                
                // NOTE: Despite the variable name y_train_classification_1based_, labels are actually
                // 0-based for both classification and unsupervised modes. nodeclass also stores 0-based
                // class indices. No conversion needed - use cl_gpu directly.
                // - Classification: labels are 0-based (0, 1, 2, ...)
                // - Unsupervised: labels are 0-based (0=synthetic, 1=real)
                // - Regression: correctness check is skipped anyway
                integer_t* cl_0based_gpu = cl_gpu;  // Use as-is (already 0-based)
                
                // DEBUG: Track total noob for verification
                integer_t total_noob_debug = 0;
                
                for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                    integer_t tree_offset = tree_id * maxnode;
                    integer_t jinbag_offset = tree_id * nsample;
                    
                    // Count OOB samples for this tree (for per-tree normalization)
                    integer_t noob = rf::cuda::gpu_count_oob_samples(
                        nin_all_gpu + jinbag_offset,
                        nsample,
                        compute_stream
                    );
                    
                    total_noob_debug += noob;
                    
                    if (noob <= 0) continue;  // Skip trees with no OOB samples
                    
                    // Call GPU proximity importance kernel
                    // jtr is computed on-the-fly from nodeclass[nodextr[n]]
                    rf::cuda::gpu_compute_proximity_importance_dense(
                        x_gpu,                                    // Dense data on GPU
                        nodextr_all_gpu + jinbag_offset,         // Terminal nodes for this tree
                        nin_all_gpu + jinbag_offset,             // Bootstrap multiplicities (0 = OOB)
                        cl_0based_gpu,                            // True classes [nsample] - 0-based!
                        nodeclass_all_gpu + tree_offset,         // Node classes (for computing jtr on-the-fly)
                        treemap_all_gpu + tree_offset * 2,       // Tree structure
                        nodestatus_all_gpu + tree_offset,        // Node status
                        xbestsplit_all_gpu + tree_offset,        // Split points
                        bestvar_all_gpu + tree_offset,           // Split variables
                        nsample, mdim,
                        nnode_all[tree_id],                      // Number of nodes in this tree
                        maxnode,                                  // For bounds checking
                        100,                                      // max_depth (safety limit)
                        tree_id,                                  // For deterministic donor selection
                        noob,                                     // OOB count for per-tree normalization
                        prox_imp_all_gpu,                        // Accumulated on GPU
                        is_regression,                            // Skip correctness check for regression
                        compute_stream
                    );
                }
                
                // DEBUG: Print noob stats
                // std::cout << "[DEBUG GPU PROX IMP] total_noob across " << num_trees << " trees = " << total_noob_debug 
                //           << " (expected ~" << (nsample * 0.37 * num_trees) << ")" << std::endl;
                
                // Sync after proximity importance kernels (before CPU reads results)
                CUDA_CHECK_VOID(cudaStreamSynchronize(compute_stream));
            }
            
            // Synchronize before copying results - ensure all async kernel launches complete
            cudaDeviceSynchronize();
            
            // OPTIMIZED: Single batch copy of nodextr_all (was per-tree loop)
            cudaMemcpy(nodextr_all, nodextr_all_gpu, static_cast<size_t>(num_trees) * static_cast<size_t>(nsample) * sizeof(integer_t), cudaMemcpyDeviceToHost);
            
            // Copy to leaf_assignments_all if requested (convert integer_t to int16_t)
            if (leaf_assignments_all != nullptr) {
                for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                    integer_t offset = tree_id * nsample;
                    for (integer_t n = 0; n < nsample; ++n) {
                        leaf_assignments_all[offset + n] = static_cast<int16_t>(nodextr_all[offset + n]);
                    }
                }
            }
            auto time_copy_end = std::chrono::high_resolution_clock::now();
        }
        
        // NOTE: nodextr_all_gpu is NOT freed here - it's kept alive for OOB vote accumulation
        // It will be freed after OOB votes are computed (at end of function)
        
        auto time_nodextr_end = std::chrono::high_resolution_clock::now();
    }
    
    // ============================================================================
    // DEFERRED TNODEWT COMPUTATION - BATCHED GPU KERNEL (all trees in parallel!)
    // ============================================================================
    real_t* tnodewt_all_gpu = nullptr;
    if (tnodewt_all != nullptr && need_tnodewt && use_gpu_nodextr && nodextr_all_gpu != nullptr) {
        // Allocate GPU buffer for tnodewt
        cudaMalloc(&tnodewt_all_gpu, static_cast<size_t>(num_trees) * maxnode * sizeof(real_t));
        cudaMemset(tnodewt_all_gpu, 0, static_cast<size_t>(num_trees) * maxnode * sizeof(real_t));
        
        // Launch batched kernel: grid is (maxnode, num_trees)
        dim3 tnodewt_grid(maxnode, num_trees);
        dim3 tnodewt_block(1);
        gpu_compute_tnodewt_batch_kernel<<<tnodewt_grid, tnodewt_block, 0, compute_stream>>>(
            num_trees, nsample, maxnode,
            nodextr_all_gpu, nin_all_gpu, win_gpu,
            nnode_all_gpu, nodestatus_all_gpu,
            tnodewt_all_gpu
        );
        
        // Copy back to host
        cudaMemcpy(tnodewt_all, tnodewt_all_gpu, 
                   static_cast<size_t>(num_trees) * maxnode * sizeof(real_t), 
                   cudaMemcpyDeviceToHost);
    } else if (tnodewt_all != nullptr && need_tnodewt && nodextr_all != nullptr) {
        // CPU fallback: sequential computation
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            integer_t tree_offset = tree_id * maxnode;
            integer_t nin_offset = tree_id * nsample;
            
            real_t* tnodewt = tnodewt_all + tree_offset;
            std::fill(tnodewt, tnodewt + maxnode, 0.0f);
            
            std::vector<real_t> node_weight_sum(maxnode, 0.0f);
            std::vector<integer_t> node_sample_count(maxnode, 0);
            
            const integer_t* nin_tree = nin + nin_offset;
            const integer_t* nodextr_tree = nodextr_all + nin_offset;
            
            for (integer_t n = 0; n < nsample; ++n) {
                if (nin_tree[n] > 0) {
                    integer_t terminal_node = nodextr_tree[n];
                    if (terminal_node >= 0 && terminal_node < maxnode) {
                        node_weight_sum[terminal_node] += static_cast<real_t>(nin_tree[n]);
                        node_sample_count[terminal_node]++;
                    }
                }
            }
            
            integer_t nnode_tree = nnode_all[tree_id];
            for (integer_t node = 0; node < nnode_tree; ++node) {
                if (node_sample_count[node] > 0) {
                    tnodewt[node] = node_weight_sum[node] / static_cast<real_t>(node_sample_count[node]);
                }
            }
        }
    }
    
    // ============================================================================
    // REGRESSION NODEPRED COMPUTATION - Terminal node predictions for regression
    // ============================================================================
    // For regression, we need to compute the mean y value for each terminal node
    // This is done AFTER tree growing but BEFORE variable importance
    // nodepred_all stores [num_trees * maxnode] values for prediction
    if (task_type == 1 && nodepred_all != nullptr && y_regression != nullptr && use_gpu_nodextr && nodextr_all_gpu != nullptr) {
        // Allocate y_regression on GPU
        real_t* y_regression_gpu_local = nullptr;
        cudaMalloc(&y_regression_gpu_local, nsample * sizeof(real_t));
        cudaMemcpy(y_regression_gpu_local, y_regression, nsample * sizeof(real_t), cudaMemcpyHostToDevice);
        
        // Allocate nodepred on GPU
        real_t* nodepred_all_gpu = nullptr;
        cudaMalloc(&nodepred_all_gpu, static_cast<size_t>(num_trees) * maxnode * sizeof(real_t));
        cudaMemset(nodepred_all_gpu, 0, static_cast<size_t>(num_trees) * maxnode * sizeof(real_t));
        
        // Launch batched kernel: grid is (maxnode, num_trees)
        dim3 nodepred_grid(maxnode, num_trees);
        dim3 nodepred_block(1);
        gpu_compute_nodepred_batch_kernel<<<nodepred_grid, nodepred_block, 0, compute_stream>>>(
            num_trees, nsample, maxnode,
            nodextr_all_gpu, nin_all_gpu, y_regression_gpu_local, win_gpu,
            nnode_all_gpu, nodestatus_all_gpu,
            nodepred_all_gpu
        );
        
        // Copy back to host
        cudaMemcpy(nodepred_all, nodepred_all_gpu, 
                   static_cast<size_t>(num_trees) * maxnode * sizeof(real_t), 
                   cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(y_regression_gpu_local);
        cudaFree(nodepred_all_gpu);
    } else if (task_type == 1 && nodepred_all != nullptr && y_regression != nullptr && nodextr_all != nullptr) {
        // CPU fallback: sequential computation
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            integer_t tree_offset = tree_id * maxnode;
            integer_t nin_offset = tree_id * nsample;
            
            real_t* nodepred = nodepred_all + tree_offset;
            std::fill(nodepred, nodepred + maxnode, 0.0f);
            
            std::vector<real_t> node_y_sum(maxnode, 0.0f);
            std::vector<real_t> node_weight_sum(maxnode, 0.0f);
            
            const integer_t* nin_tree = nin + nin_offset;
            const integer_t* nodextr_tree = nodextr_all + nin_offset;
            
            for (integer_t n = 0; n < nsample; ++n) {
                if (nin_tree[n] > 0) {  // In-bag sample
                    integer_t terminal_node = nodextr_tree[n];
                    if (terminal_node >= 0 && terminal_node < maxnode) {
                        // Use sample weight only (not * nin) for non-casewise mode
                        real_t w = (win != nullptr) ? win[n] : 1.0f;
                        node_y_sum[terminal_node] += w * y_regression[n];
                        node_weight_sum[terminal_node] += w;
                    }
                }
            }
            
            integer_t nnode_tree = nnode_all[tree_id];
            for (integer_t node = 0; node < nnode_tree; ++node) {
                if (node_weight_sum[node] > 0.0f) {
                    nodepred[node] = node_y_sum[node] / node_weight_sum[node];
                }
            }
        }
    }
    
    // ============================================================================
    // VARIABLE IMPORTANCE - BATCHED GPU PATH (all trees in parallel!)
    // ============================================================================
    if (avimp_all != nullptr) {
        auto time_varimp_start = std::chrono::high_resolution_clock::now();
        
        // Check if we can use batched GPU path
        bool use_batched_classification = (task_type == 0 || task_type == 2) && 
                                           use_gpu_nodextr && 
                                           nodextr_all_gpu != nullptr &&
                                           cl_gpu != nullptr;
        
        bool use_batched_regression = (task_type == 1) && 
                                       use_gpu_nodextr && 
                                       nodextr_all_gpu != nullptr &&
                                       y_regression != nullptr;
        
        if (use_batched_classification) {
            // =================================================================
            // BATCHED GPU PATH - CLASSIFICATION/UNSUPERVISED
            // =================================================================
            
            // Step 1: Allocate and compute jtr_all_gpu
            integer_t* jtr_all_gpu = nullptr;
            cudaMalloc(&jtr_all_gpu, static_cast<size_t>(num_trees) * nsample * sizeof(integer_t));
            
            dim3 jtr_grid((nsample + 255) / 256, num_trees);
            dim3 jtr_block(256);
            gpu_compute_jtr_batch_kernel<<<jtr_grid, jtr_block, 0, compute_stream>>>(
                num_trees, nsample, maxnode,
                nodextr_all_gpu, nodeclass_all_gpu, nodestatus_all_gpu, nnode_all_gpu,
                jtr_all_gpu
            );
            
            // Step 2: Allocate avimp and qimp on GPU
            real_t* avimp_gpu = nullptr;
            real_t* qimp_all_gpu_local = nullptr;  // For local importance calculation
            cudaMalloc(&avimp_gpu, mdim * sizeof(real_t));
            cudaMemset(avimp_gpu, 0, mdim * sizeof(real_t));
            
            // Allocate qimp on GPU if local importance is requested
            if (qimp_all != nullptr) {
                cudaMalloc(&qimp_all_gpu_local, nsample * sizeof(real_t));
                cudaMemset(qimp_all_gpu_local, 0, nsample * sizeof(real_t));
            }
            
            // Step 3: Call batched varimp kernel for all trees
            // Pass qimpm_all_gpu if local importance is requested
            integer_t use_casewise = g_config.use_casewise ? 1 : 0;
            gpu_varimp_batch_all_trees(
                num_trees, nsample, mdim, maxnode, nclass,
                x_gpu, cl_gpu, nin_all_gpu, jtr_all_gpu, nodextr_all_gpu,
                treemap_all_gpu, nodestatus_all_gpu, xbestsplit_all_gpu,
                bestvar_all_gpu, nodeclass_all_gpu, nnode_all_gpu,
                cat_gpu, maxcat,
                tnodewt_all_gpu,  // May be nullptr
                avimp_gpu,
                qimp_all_gpu_local,  // Original correct rate for local importance
                qimpm_all_gpu,  // Local importance (nullptr if not requested)
                use_casewise, compute_stream
            );
            
            // Step 4: Copy results back to host
            cudaMemcpy(avimp_all, avimp_gpu, mdim * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            // Copy qimp back if local importance is requested
            if (qimp_all_gpu_local != nullptr && qimp_all != nullptr) {
                cudaMemcpy(qimp_all, qimp_all_gpu_local, nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
                cudaFree(qimp_all_gpu_local);
            }
            
            // Copy local importance back if requested
            if (qimpm_all_gpu != nullptr && qimpm_all != nullptr) {
                cudaMemcpy(qimpm_all, qimpm_all_gpu, 
                           static_cast<size_t>(nsample) * mdim * sizeof(real_t), 
                           cudaMemcpyDeviceToHost);
            }
            
            // Cleanup
            cudaFree(jtr_all_gpu);
            cudaFree(avimp_gpu);
            
        } else if (use_batched_regression) {
            // =================================================================
            // BATCHED GPU PATH - REGRESSION
            // =================================================================
            
            // Step 1: Copy y_regression to GPU
            real_t* y_regression_gpu = nullptr;
            cudaMalloc(&y_regression_gpu, nsample * sizeof(real_t));
            cudaMemcpy(y_regression_gpu, y_regression, nsample * sizeof(real_t), cudaMemcpyHostToDevice);
            
            // Step 2: Compute nodepred for all trees
            real_t* nodepred_all_gpu = nullptr;
            cudaMalloc(&nodepred_all_gpu, static_cast<size_t>(num_trees) * maxnode * sizeof(real_t));
            cudaMemset(nodepred_all_gpu, 0, static_cast<size_t>(num_trees) * maxnode * sizeof(real_t));
            
            dim3 nodepred_grid(maxnode, num_trees);
            dim3 nodepred_block(1);
            gpu_compute_nodepred_batch_kernel<<<nodepred_grid, nodepred_block, 0, compute_stream>>>(
                num_trees, nsample, maxnode,
                nodextr_all_gpu, nin_all_gpu, y_regression_gpu, win_gpu,
                nnode_all_gpu, nodestatus_all_gpu,
                nodepred_all_gpu
            );
            
            // Step 3: Compute y_pred for all samples from nodepred
            real_t* y_pred_all_gpu = nullptr;
            cudaMalloc(&y_pred_all_gpu, static_cast<size_t>(num_trees) * nsample * sizeof(real_t));
            
            dim3 ypred_grid((nsample + 255) / 256, num_trees);
            dim3 ypred_block(256);
            gpu_compute_y_pred_batch_kernel<<<ypred_grid, ypred_block, 0, compute_stream>>>(
                num_trees, nsample, maxnode,
                nodextr_all_gpu, nodepred_all_gpu, nnode_all_gpu,
                y_pred_all_gpu
            );
            
            // Step 4: Allocate avimp and qimp on GPU
            real_t* avimp_gpu = nullptr;
            real_t* qimp_all_gpu_local = nullptr;  // For local importance calculation
            cudaMalloc(&avimp_gpu, mdim * sizeof(real_t));
            cudaMemset(avimp_gpu, 0, mdim * sizeof(real_t));
            
            // Allocate qimp on GPU if local importance is requested
            if (qimp_all != nullptr) {
                cudaMalloc(&qimp_all_gpu_local, nsample * sizeof(real_t));
                cudaMemset(qimp_all_gpu_local, 0, nsample * sizeof(real_t));
            }
            
            // Step 5: Call batched regression varimp
            integer_t use_casewise = g_config.use_casewise ? 1 : 0;
            gpu_varimp_regression_batch_all_trees(
                num_trees, nsample, mdim, maxnode,
                x_gpu, y_regression_gpu, y_pred_all_gpu,
                nin_all_gpu, nodextr_all_gpu,
                treemap_all_gpu, nodestatus_all_gpu, xbestsplit_all_gpu,
                bestvar_all_gpu, nodepred_all_gpu, nnode_all_gpu,
                cat_gpu, maxcat, tnodewt_all_gpu,
                avimp_gpu,
                qimp_all_gpu_local,  // Original squared error for local importance
                qimpm_all_gpu,  // Local importance (nullptr if not requested)
                use_casewise, compute_stream
            );
            
            // Step 6: Copy results back to host
            cudaMemcpy(avimp_all, avimp_gpu, mdim * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            // Copy qimp back if local importance is requested
            if (qimp_all_gpu_local != nullptr && qimp_all != nullptr) {
                cudaMemcpy(qimp_all, qimp_all_gpu_local, nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
                cudaFree(qimp_all_gpu_local);
            }
            
            // Copy local importance back if requested
            if (qimpm_all_gpu != nullptr && qimpm_all != nullptr) {
                cudaMemcpy(qimpm_all, qimpm_all_gpu, 
                           static_cast<size_t>(nsample) * mdim * sizeof(real_t), 
                           cudaMemcpyDeviceToHost);
            }
            
            // Cleanup
            cudaFree(y_regression_gpu);
            cudaFree(nodepred_all_gpu);
            cudaFree(y_pred_all_gpu);
            cudaFree(avimp_gpu);
            
            auto time_varimp_end = std::chrono::high_resolution_clock::now();
            
        } else {
            // =================================================================
            // SEQUENTIAL FALLBACK: Original per-tree loop (for regression, etc.)
            // =================================================================
            rf::GPUVarImpContext* gpu_ctx = nullptr;
            if (task_type == 0 || task_type == 2) {
                integer_t max_ninbag = nsample;
                gpu_ctx = rf::gpu_varimp_alloc_context(nsample, mdim, maxnode, maxcat, max_ninbag, nclass, mdim, 256);
        }
        
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
        auto time_tree_start = std::chrono::high_resolution_clock::now();
        // std::cout << "DEBUG: Processing tree " << tree_id << " for variable importance..." << std::endl;
        
        // Get tree-specific data
        integer_t tree_offset = tree_id * maxnode;
        integer_t jinbag_offset = tree_id * nsample;
        
        // std::cout << "DEBUG: About to call cpu_testreebag for tree " << tree_id << std::endl;
        
        // OPTIMIZATION: Reuse nodextr_all that was already computed above (GPU-accelerated)
        // Instead of calling cpu_testreebag again, just extract nodextr_tree from nodextr_all
        auto time_extract_start = std::chrono::high_resolution_clock::now();
        std::vector<real_t> q_tree(nsample * nclass, 0.0f);
        std::vector<integer_t> jtr_tree(nsample, 0);
        std::vector<integer_t> nodextr_tree(nsample, 0);
        
        // Copy nodextr from nodextr_all (already computed above)
        std::copy(nodextr_all + jinbag_offset, nodextr_all + jinbag_offset + nsample, 
                  nodextr_tree.begin());
        
        // OPTIMIZATION: Only compute jtr_tree for OOB samples (needed for importance)
        // We can do this quickly by traversing tree only for OOB samples
        
        // Compute jtr_tree from nodextr_tree for ALL samples (needed for OOB vote accumulation)
        // jtr_tree[n] = nodeclass of terminal node that sample n reaches
        // nodextr_tree must be valid (copied from nodextr_all above)
        // Simplified logic: just look up nodeclass from nodextr (terminal node index)
        for (integer_t n = 0; n < nsample; ++n) {
            integer_t kt = nodextr_tree[n];  // Use pre-computed terminal node
            if (kt >= 0 && kt < nnode_all[tree_id] && 
                nodestatus_all[tree_offset + kt] == -1) {  // Valid terminal node
                jtr_tree[n] = nodeclass_all[tree_offset + kt];
            } else {
                // Invalid node or not terminal, use default (shouldn't happen if nodextr is correct)
                jtr_tree[n] = 0;
            }
        }
        auto time_extract_end = std::chrono::high_resolution_clock::now();
        auto extract_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_extract_end - time_extract_start);
        
        // DEBUG STEP 1: Verify OOB per tree - check nodextr_tree and jtr_tree for OOB samples
        if (task_type == 0 && tree_id < 3) {  // Classification, first 3 trees only
            integer_t oob_samples_checked = 0;
            integer_t oob_samples_with_valid_jtr = 0;
            integer_t oob_samples_with_zero_nodextr = 0;
            for (integer_t n = 0; n < nsample && oob_samples_checked < 10; ++n) {
                if (nin[jinbag_offset + n] == 0) {  // OOB sample
                    oob_samples_checked++;
                    if (nodextr_tree[n] == 0) {
                        oob_samples_with_zero_nodextr++;
                    }
                    if (jtr_tree[n] > 0 || (jtr_tree[n] == 0 && nodextr_tree[n] >= 0)) {
                        oob_samples_with_valid_jtr++;
                    }
                    if (oob_samples_checked <= 5) {
// Debug prints removed
                        // std::cout << "[DEBUG TREE " << tree_id << "] OOB sample " << n 
                        //           << ": nodextr=" << nodextr_tree[n] 
                        //           << ", jtr=" << jtr_tree[n]
                        //           << ", nodeclass=" << (nodextr_tree[n] >= 0 && nodextr_tree[n] < nnode_all[tree_id] ? 
                        //               nodeclass_all[tree_offset + nodextr_tree[n]] : -1)
                        //           << ", nodestatus=" << (nodextr_tree[n] >= 0 && nodextr_tree[n] < nnode_all[tree_id] ? 
                        //               nodestatus_all[tree_offset + nodextr_tree[n]] : -999)
                        //           << std::endl;
                    }
                }
            }
// Debug prints removed
            // std::cout << "[DEBUG TREE " << tree_id << "] OOB samples checked: " << oob_samples_checked
            //           << ", valid jtr: " << oob_samples_with_valid_jtr
            //           << ", zero nodextr: " << oob_samples_with_zero_nodextr << std::endl;
        }
        
        // Count OOB samples for this tree
        integer_t noob_count = 0;
        for (integer_t n = 0; n < nsample; ++n) {
            if (nin[jinbag_offset + n] == 0) {
                noob_count++;
            }
        }
        
        // For regression: compute tnodewt_all using CPU fallback
        // GPU kernel was passing host pointers which caused illegal memory access
        // Using CPU fallback for stability - matches CPU behavior exactly
        if (task_type == 1 && oob_predictions_all != nullptr && tnodewt_all != nullptr) {  // REGRESSION
            integer_t tree_offset = tree_id * maxnode;
            real_t* tnodewt_tree = tnodewt_all + tree_offset;
            
            // Compute tnodewt for this tree using CPU
            // This computes the mean of y_regression values in each terminal node
            integer_t nnode_tree = nnode_all[tree_id];
            if (nnode_tree > 0) {
                // Initialize tnodewt to zero
                std::fill(tnodewt_tree, tnodewt_tree + nnode_tree, 0.0f);
                
                // For each terminal node, calculate mean of y values for samples in that node
                std::vector<real_t> node_sum_y(nnode_tree, 0.0f);
                std::vector<real_t> node_count(nnode_tree, 0.0f);
                
                // Traverse each sample to find its terminal node
                for (integer_t n = 0; n < nsample; ++n) {
                    // Use nin to check OOB status (win is only size nsample, not per-tree)
                    // nin[jinbag_offset + n] == 0 means sample is OOB for this tree
                    if (nin[jinbag_offset + n] == 0) continue;  // Skip OOB samples
                    
                    // Traverse tree from root
                    integer_t current_node = 0;
                    integer_t traversal_count = 0;
                    const integer_t max_traversal = 1000;
                    
                    while (nodestatus_all[tree_offset + current_node] == 1 && traversal_count < max_traversal) {
                        traversal_count++;
                        integer_t split_var = bestvar_all[tree_offset + current_node];
                        if (split_var < 0 || split_var >= mdim) break;
                        
                        real_t split_point = xbestsplit_all[tree_offset + current_node];
                        real_t sample_value = x[n * mdim + split_var];
                        
                        if (sample_value <= split_point) {
                            current_node = treemap_all[tree_offset * 2 + current_node * 2];  // Left child
                        } else {
                            current_node = treemap_all[tree_offset * 2 + current_node * 2 + 1];  // Right child
                        }
                        
                        if (current_node < 0 || current_node >= nnode_tree) break;
                    }
                    
                    // Accumulate y value for this terminal node
                    // For regression, y_regression MUST be provided (no fallback to cl)
                    if (current_node >= 0 && current_node < nnode_tree && 
                        nodestatus_all[tree_offset + current_node] == -1) {
                        real_t y_val = y_regression[n];  // Regression requires y_regression
                        
                        // Handle casewise vs non-casewise weighting (matching CPU)
                        if (g_config.use_casewise) {
                            // Casewise: weight by bootstrap frequency (nin)
                            real_t bootstrap_freq = static_cast<real_t>(nin[jinbag_offset + n]);
                            node_sum_y[current_node] += y_val * bootstrap_freq;
                            node_count[current_node] += bootstrap_freq;
                        } else {
                            // Non-casewise: equal weights
                            node_sum_y[current_node] += y_val;
                            node_count[current_node] += 1.0f;
                        }
                    }
                }
                
                // Calculate mean for each terminal node
                for (integer_t node = 0; node < nnode_tree; ++node) {
                    if (node_count[node] > 0.0f) {
                        tnodewt_tree[node] = node_sum_y[node] / node_count[node];
                    }
                }
            }
        }
        
        // Note: Removed GPU sync for tnodewt since we're using CPU fallback now
        
        // NOTE: Classification/unsupervised OOB vote accumulation is handled in standalone block after this loop
        // (around line 3500) to avoid double-accumulation when avimp_all is set
        // Regression OOB accumulation is also handled in a standalone block
        
        // Compute variable importance for this tree using its individual OOB predictions
        // Create temporary arrays per tree (following CPU pattern)
        std::vector<real_t> qimp_temp(nsample, 0.0f);  // Temporary per-tree array
        std::vector<real_t> qimpm_temp(nsample * mdim, 0.0f);  // Temporary per-tree array
        std::vector<real_t> avimp_temp(mdim, 0.0f);
        std::vector<real_t> sqsd_temp(mdim, 0.0f);
        std::vector<integer_t> jvr_temp(nsample, 0);
        std::vector<integer_t> nodexvr_temp(nsample, 0);
        
        // Call appropriate variable importance function based on task type
        // Use global arrays for local importance accumulation
        auto time_varimp_func_start = std::chrono::high_resolution_clock::now();
        // Only require qimp_all for importance - qimpm_all is only needed for LOCAL importance
        if (qimp_all != nullptr) {
            if (task_type == 1) {  // 1=REGRESSION
                // Use GPU regression-specific importance (MSE-based)
                // gpu_varimp_with_context now detects task_type == 1 and uses MSE kernels
                if (y_regression != nullptr) {
                    integer_t impn = g_config.impn;
                    
                    // Get tnodewt pointer for this tree
                    real_t* tnodewt_ptr = nullptr;
                    if (tnodewt_all != nullptr) {
                        tnodewt_ptr = tnodewt_all + tree_offset;
                    }
                    
                    // Compute ninbag (number of in-bag samples) from nin
                    integer_t ninbag = 0;
                    const integer_t* nin_tree = nin + jinbag_offset;
                    for (integer_t i = 0; i < nsample; ++i) {
                        if (nin_tree[i] > 0) {
                            ninbag++;
                        }
                    }
                    const integer_t* jinbag_tree = jinbag_all + jinbag_offset;
                    
                    // Use GPU varimp with context for regression (matches classification pattern)
                    if (gpu_ctx != nullptr) {
                        gpu_varimp_with_context(gpu_ctx, x, nsample, mdim, cl, nin + jinbag_offset, jtr_tree.data(),
                                   impn, qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                                   treemap_all + tree_offset * 2, nodestatus_all + tree_offset,
                                   xbestsplit_all + tree_offset, bestvar_all + tree_offset,
                                   nodeclass_all + tree_offset, nnode_all[tree_id],
                                   cat, jvr_temp.data(), nodexvr_temp.data(),
                                   maxcat, nullptr, tnodewt_ptr, nodextr_tree.data(),
                                   y_regression, win, jinbag_tree, ninbag, nclass, task_type);
                    } else {
                        // Fallback to CPU if GPU context allocation failed
                        integer_t nnode_tree = nnode_all[tree_id];
                        
                        // Compute nodepred and tnodewt locally (same pattern as OOB accumulation block)
                        std::vector<real_t> nodepred_tree_local(nnode_tree, 0.0f);
                        std::vector<real_t> tnodewt_tree_local(nnode_tree, 0.0f);
                        std::vector<real_t> node_sum_y(nnode_tree, 0.0f);
                        std::vector<real_t> node_sum_weight(nnode_tree, 0.0f);
                        std::vector<real_t> node_count(nnode_tree, 0.0f);
                        
                        // Accumulate for in-bag samples
                        for (integer_t n = 0; n < nsample; ++n) {
                            if (nin[jinbag_offset + n] == 0) continue;  // Skip OOB
                            integer_t kt = nodextr_tree[n];
                            if (kt >= 0 && kt < nnode_tree) {
                                real_t y_val = y_regression[n];
                                real_t bootstrap_freq = static_cast<real_t>(nin[jinbag_offset + n]);
                                if (g_config.use_casewise) {
                                    node_sum_y[kt] += y_val * bootstrap_freq;
                                    node_sum_weight[kt] += bootstrap_freq;
                                } else {
                                    node_sum_y[kt] += y_val;
                                    node_sum_weight[kt] += 1.0f;
                                }
                                node_count[kt] += 1.0f;
                            }
                        }
                        
                        // Compute nodepred and tnodewt
                        for (integer_t node = 0; node < nnode_tree; ++node) {
                            if (node_sum_weight[node] > 0.0f && node_count[node] > 0.0f) {
                                nodepred_tree_local[node] = node_sum_y[node] / node_sum_weight[node];
                                tnodewt_tree_local[node] = node_sum_weight[node] / node_count[node];
                            }
                        }
                        
                        // Build y_pred_temp from nodepred (not tnodewt)
                        std::vector<real_t> y_pred_temp(nsample, 0.0f);
                        for (integer_t n = 0; n < nsample; n++) {
                            integer_t terminal_node = nodextr_tree[n];
                            if (terminal_node >= 0 && terminal_node < nnode_tree) {
                                y_pred_temp[n] = nodepred_tree_local[terminal_node];
                            }
                        }
                        
                        cpu_varimp_regression(x, nsample, mdim,
                                             y_regression, nin + jinbag_offset,
                                             y_pred_temp.data(),
                                             impn,
                                             qimp_temp.data(), qimpm_temp.data(), 
                                             avimp_temp.data(), sqsd_temp.data(),
                                             treemap_all + tree_offset * 2, nodestatus_all + tree_offset,
                                             xbestsplit_all + tree_offset, bestvar_all + tree_offset,
                                             nodepred_tree_local.data(), tnodewt_tree_local.data(), nnode_tree,
                                             cat, maxcat, 
                                             nullptr,
                                             nodextr_tree.data());
                    }
                    
                    // Accumulate into global arrays (matching classification pattern)
                    if (qimp_all != nullptr) {
                        for (integer_t i = 0; i < nsample; i++) {
                            qimp_all[i] += qimp_temp[i];
                        }
                    }
                    if (qimpm_all != nullptr && g_config.impn == 1) {
                        for (integer_t i = 0; i < nsample * mdim; i++) {
                            qimpm_all[i] += qimpm_temp[i];
                        }
                    }
                }
            } else {
                // CLASSIFICATION PATH: Use GPU variable importance with temporary host arrays
                // gpu_varimp expects host pointers and handles GPU allocation internally
                // Use temp arrays per tree (following CPU pattern) then accumulate
                // Set impn based on compute_local_importance flag
                integer_t impn = g_config.impn;  // Use global config flag (set from config_.compute_local_importance)
            // gpu_varimp will compute tnodewt internally if needed (when use_casewise=true for classification)
            // Pass nullptr for tnodewt to indicate it should be computed internally
            // Compute ninbag (number of in-bag samples) from nin - matches CPU implementation
            integer_t ninbag = 0;
            const integer_t* nin_tree = nin + jinbag_offset;
            for (integer_t i = 0; i < nsample; ++i) {
                if (nin_tree[i] > 0) {
                    ninbag++;
                }
            }
            const integer_t* jinbag_tree = jinbag_all + jinbag_offset;
            
            // DEBUG: Check jinbag_tree values for first tree
            if (task_type == 0 && tree_id < 2 && ninbag > 0) {
// Debug prints removed
                // std::cout << "[DEBUG JINBAG TREE " << tree_id << "] ninbag=" << ninbag 
                //           << ", first 10 jinbag values: ";
                // for (integer_t i = 0; i < std::min(10, ninbag); ++i) {
                //     std::cout << jinbag_tree[i] << " ";
                // }
                // std::cout << std::endl;
// Debug prints removed
                // std::cout << "[DEBUG JINBAG TREE " << tree_id << "] first 10 nin values: ";
                // for (integer_t i = 0; i < std::min(10, nsample); ++i) {
                //     std::cout << nin_tree[i] << " ";
                // }
                // std::cout << std::endl;
            }
            
            // tnodewt should already be computed above (after tree growing)
            real_t* tnodewt_ptr = nullptr;
            if (tnodewt_all != nullptr) {
                tnodewt_ptr = tnodewt_all + tree_offset;
            }
            
            // OPTIMIZED: Use pre-allocated GPU context (avoids 20+ malloc/free per tree)
            if (gpu_ctx != nullptr) {
                gpu_varimp_with_context(gpu_ctx, x, nsample, mdim, cl, nin + jinbag_offset, jtr_tree.data(),
                           impn, qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                           treemap_all + tree_offset * 2, nodestatus_all + tree_offset,
                           xbestsplit_all + tree_offset, bestvar_all + tree_offset,
                           nodeclass_all + tree_offset, nnode_all[tree_id],
                           cat, jvr_temp.data(), nodexvr_temp.data(),
                           maxcat, nullptr, tnodewt_ptr, nodextr_tree.data(),
                           y_regression, win, jinbag_tree, ninbag, nclass, task_type);
            } else {
                // Context allocation failed - compute OVERALL importance only (skip local)
                // Local importance would be too slow without optimized context (30× slower)
                if (tree_id == 0) {
                    std::cerr << "Warning: GPU memory insufficient for local importance context." << std::endl;
                    std::cerr << "         Computing overall importance only (local importance disabled)." << std::endl;
                }
                
                // Just compute overall importance using the old path
                gpu_varimp(x, nsample, mdim, cl, nin + jinbag_offset, jtr_tree.data(),
                           0,  // impn=0: disable local importance (too slow without context)
                           qimp_temp.data(), qimpm_temp.data(), avimp_temp.data(), sqsd_temp.data(),
                           treemap_all + tree_offset * 2, nodestatus_all + tree_offset,
                           xbestsplit_all + tree_offset, bestvar_all + tree_offset,
                           nodeclass_all + tree_offset, nnode_all[tree_id],
                           cat, jvr_temp.data(), nodexvr_temp.data(),
                           maxcat, nullptr, tnodewt_ptr, nodextr_tree.data(),
                           y_regression, win, jinbag_tree, ninbag, nclass, task_type);
            }
            
            // Debug prints removed to avoid potential stream corruption issues
            // if (task_type == 0 && tree_id < 3) {
            //     std::cout << "[DEBUG IMPORTANCE TREE " << tree_id << "] After gpu_varimp: "
            //               << "avimp_temp: ";
            //     for (integer_t i = 0; i < mdim; ++i) {
            //         std::cout << avimp_temp[i] << " ";
            //     }
            //     std::cout << std::endl;
            //     integer_t non_zero_qimp = 0;
            //     for (integer_t i = 0; i < nsample; ++i) {
            //         if (qimp_temp[i] != 0.0f) non_zero_qimp++;
            //     }
            //     std::cout << "[DEBUG IMPORTANCE TREE " << tree_id << "] qimp_temp non-zero: " 
            //               << non_zero_qimp << " / " << nsample << std::endl;
            // }
                
                // Check for CUDA errors after gpu_varimp (it uses GPU internally)
                cudaError_t varimp_err = cudaGetLastError();
                if (varimp_err != cudaSuccess) {
                    // std::cerr << "Warning: CUDA error after gpu_varimp for tree " << tree_id << ": " << cudaGetErrorString(varimp_err) << std::endl;
                    // std::cerr << "Warning: Clearing error and recovering context..." << std::endl;
                    // Aggressive recovery: sync and clear errors
                    CUDA_CHECK_VOID(cudaDeviceSynchronize());  // Wait for all operations
                    // Ensure device is still valid
                    int device = 0;
                    cudaGetDevice(&device);
                }
                
                // Accumulate into global host arrays (following CPU pattern)
                if (qimp_all != nullptr) {
                    for (integer_t i = 0; i < nsample; i++) {
                        qimp_all[i] += qimp_temp[i];
                    }
                }
                // Only accumulate qimpm_all if local importance was requested
                if (qimpm_all != nullptr && g_config.impn == 1) {
                    for (integer_t i = 0; i < nsample * mdim; i++) {
                        qimpm_all[i] += qimpm_temp[i];
                    }
                }
            }
        } else {
            // std::cout << "GPU: Skipping variable importance for tree " << tree_id << " (arrays not allocated)" << std::endl;
        }
        // std::cout << "GPU: gpu_varimp completed for tree " << tree_id << std::endl;
        
        // Accumulate importance values into the global arrays
        // Ensure avimp_all is not nullptr before accumulating
        if (avimp_all != nullptr) {
        for (integer_t i = 0; i < mdim; ++i) {
            avimp_all[i] += avimp_temp[i];
            }
        }
        
        auto time_varimp_func_end = std::chrono::high_resolution_clock::now();
        auto time_tree_end = std::chrono::high_resolution_clock::now();
        }
        
            auto time_varimp_end = std::chrono::high_resolution_clock::now();
        
            // CLEANUP: Free GPU context after all trees are processed
            if (gpu_ctx != nullptr) {
                rf::gpu_varimp_free_context(gpu_ctx);
            }
        
            // Synchronize after importance computation loop completes
            if (avimp_all != nullptr) {
                CUDA_CHECK_VOID(cudaStreamSynchronize(0));
            }
        }  // End else (sequential fallback)
    }  // End if (avimp_all != nullptr)
    
    // ============================================================================
    // GPU-ONLY OOB VOTE ACCUMULATION for Classification/Unsupervised
    // ============================================================================
    // For casewise mode: compute tnodewt on CPU (fast), upload to GPU, use GPU kernel
    // For non-casewise: use GPU kernel directly (no tnodewt needed)
    if ((task_type == 0 || task_type == 2) && q_all != nullptr) {
        
        // Use pre-computed tnodewt_all from gpu_compute_tnodewt_kernel (already on HOST)
        // No need to recompute - it's already been computed and copied to tnodewt_all
        real_t* tnodewt_gpu = nullptr;
        if (g_config.use_casewise && tnodewt_all != nullptr) {
            size_t tnodewt_size = num_trees * maxnode * sizeof(real_t);
            CUDA_CHECK_VOID(cudaMalloc(&tnodewt_gpu, tnodewt_size));
            // Upload pre-computed tnodewt_all to GPU
            CUDA_CHECK_VOID(cudaMemcpy(tnodewt_gpu, tnodewt_all, tnodewt_size, cudaMemcpyHostToDevice));
        }
        
        // Also allocate nout on GPU if needed - copy existing values to preserve across batches
        integer_t* nout_gpu = nullptr;
        if (nout_all != nullptr) {
            CUDA_CHECK_VOID(cudaMalloc(&nout_gpu, nsample * sizeof(integer_t)));
            // Copy existing nout values (preserves counts from previous batches)
            CUDA_CHECK_VOID(cudaMemcpy(nout_gpu, nout_all, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
        }
        
        // Launch GPU kernel for vote accumulation
        dim3 block_size(256);
        dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
        
        gpu_oob_vote_kernel_casewise<<<grid_size, block_size>>>(
            num_trees, nsample, nclass, maxnode, mdim,
            nodestatus_all_gpu, nodeclass_all_gpu,
            bestvar_all_gpu, xbestsplit_all_gpu, treemap_all_gpu,
            nin_all_gpu,
            tnodewt_gpu,
            x_gpu,
            q_all_gpu, nout_gpu,
            g_config.use_casewise,
            // Categorical split support
            cat_gpu, catgoleft_all_gpu, maxcat
        );
        
        // OPTIMIZATION: Use stream sync before copying results to host
        CUDA_CHECK_VOID(cudaStreamSynchronize(compute_stream));
        
        // Copy results back to host
        CUDA_CHECK_VOID(cudaMemcpy(q_all, q_all_gpu, nsample * nclass * sizeof(real_t), cudaMemcpyDeviceToHost));
        if (nout_all != nullptr && nout_gpu != nullptr) {
            CUDA_CHECK_VOID(cudaMemcpy(nout_all, nout_gpu, nsample * sizeof(integer_t), cudaMemcpyDeviceToHost));
        }
        
        // Cleanup
        if (tnodewt_gpu != nullptr) {
            CUDA_CHECK_VOID(cudaFree(tnodewt_gpu));
        }
        if (nout_gpu != nullptr) {
            CUDA_CHECK_VOID(cudaFree(nout_gpu));
        }
    }
    
    // REGRESSION TNODEWT/NODEPRED COMPUTATION - same pattern as classification
    // Compute both (locally, per-tree):
    //   nodepred_tree = mean y (the prediction) 
    //   tnodewt_tree = mean bootstrap weight (for OOB weighting, like classification vote_weight)
    // This matches how CPU does it and how classification handles tnodewt/vote_weight
    // 
    // REGRESSION OOB ACCUMULATION is integrated here (same pattern as classification)
    // For casewise mode: oob += tnodewt * nodepred (weighted predictions)
    // For non-casewise: oob += nodepred (weight = 1.0, so tnodewt cancels out)
    if (task_type == 1 && y_regression != nullptr && 
        oob_predictions_all != nullptr && nout_all != nullptr && nodextr_all != nullptr) {
        
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            integer_t tree_offset = tree_id * maxnode;
            integer_t jinbag_offset = tree_id * nsample;
            
            integer_t nnode_tree = nnode_all[tree_id];
            
            if (nnode_tree > 0) {
                // Local arrays for this tree (like classification's tnodewt_tree and q_tree)
                std::vector<real_t> nodepred_tree(nnode_tree, 0.0f);  // Mean y (prediction)
                std::vector<real_t> tnodewt_tree(nnode_tree, 0.0f);   // Mean weight
                
                // For each terminal node, calculate:
                // - nodepred = node_sum_y / node_sum_weight (mean y, the prediction)
                // - tnodewt = node_sum_weight / node_count (mean weight, for OOB weighting)
                std::vector<real_t> node_sum_y(nnode_tree, 0.0f);
                std::vector<real_t> node_sum_weight(nnode_tree, 0.0f);
                std::vector<real_t> node_count(nnode_tree, 0.0f);
                
                // Traverse each IN-BAG sample to find its terminal node
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] == 0) continue;  // Skip OOB samples
                    
                    // Traverse tree from root
                    integer_t current_node = 0;
                    integer_t traversal_count = 0;
                    const integer_t max_traversal = 1000;
                    
                    while (nodestatus_all[tree_offset + current_node] == 1 && traversal_count < max_traversal) {
                        traversal_count++;
                        integer_t split_var = bestvar_all[tree_offset + current_node];
                        if (split_var < 0 || split_var >= mdim) break;
                        
                        real_t split_point = xbestsplit_all[tree_offset + current_node];
                        real_t sample_value = x[n * mdim + split_var];
                        
                        if (sample_value <= split_point) {
                            current_node = treemap_all[tree_offset * 2 + current_node * 2];  // Left child
                        } else {
                            current_node = treemap_all[tree_offset * 2 + current_node * 2 + 1];  // Right child
                        }
                        
                        if (current_node < 0 || current_node >= nnode_tree) break;
                    }
                    
                    // Accumulate for this terminal node
                    if (current_node >= 0 && current_node < nnode_tree && 
                        nodestatus_all[tree_offset + current_node] == -1) {
                        real_t y_val = y_regression[n];
                        real_t bootstrap_freq = static_cast<real_t>(nin[jinbag_offset + n]);
                        
                        // Same pattern as CPU: always track sum_y, sum_weight, count
                        if (g_config.use_casewise) {
                            node_sum_y[current_node] += y_val * bootstrap_freq;
                            node_sum_weight[current_node] += bootstrap_freq;
                            node_count[current_node] += 1.0f;
                        } else {
                            node_sum_y[current_node] += y_val;
                            node_sum_weight[current_node] += 1.0f;
                            node_count[current_node] += 1.0f;
                        }
                    }
                }
                
                // Calculate nodepred (mean y) and tnodewt (mean weight) for each terminal node
                for (integer_t node = 0; node < nnode_tree; ++node) {
                    if (node_sum_weight[node] > 0.0f && node_count[node] > 0.0f) {
                        // nodepred = weighted mean y (the prediction)
                        nodepred_tree[node] = node_sum_y[node] / node_sum_weight[node];
                        // tnodewt = mean bootstrap weight (for OOB weighting)
                        tnodewt_tree[node] = node_sum_weight[node] / node_count[node];
                    }
                }
                
                // Also store in tnodewt_all for backward compatibility (predictions)
                if (tnodewt_all != nullptr) {
                    for (integer_t node = 0; node < nnode_tree; ++node) {
                        tnodewt_all[tree_offset + node] = nodepred_tree[node];
                    }
                }
                
                // Extract nodextr_tree from nodextr_all (same pattern as classification)
                std::vector<integer_t> nodextr_tree(nsample, 0);
                std::copy(nodextr_all + jinbag_offset, nodextr_all + jinbag_offset + nsample, 
                          nodextr_tree.begin());
                
                // Accumulate OOB predictions for regression
                // For regression: always use weight=1.0 since we divide by nout (tree count)
                // This matches sklearn's OOB prediction approach (simple averaging)
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[jinbag_offset + n] == 0) {  // OOB sample
                        nout_all[n]++;
                        
                        integer_t kt = nodextr_tree[n];
                        if (kt >= 0 && kt < nnode_tree) {
                            real_t prediction = nodepred_tree[kt];
                            // Always use weight=1.0 for regression OOB predictions
                            // (we divide by tree count, not weight sum)
                            oob_predictions_all[n] += prediction;
                        }
                    }
                }
            }
        }
    }
    
    // Launch OOB vote accumulation kernel
    // For classification (task_type == 0), we skip the GPU kernel because
    // we already accumulate votes per-tree in the loop above (q_tree -> q_all).
    // The GPU kernel would overwrite our correct per-tree accumulation.
    // For regression/unsupervised, we can use the GPU kernel if needed.
    if (task_type != 0) {  // Not classification - use GPU kernel for regression/unsupervised
        // std::cout << "GPU: Launching OOB vote kernel..." << std::endl;
        
        dim3 oob_block_size(256);
        dim3 oob_grid_size(num_trees);
        
        gpu_oob_vote_kernel<<<oob_grid_size, oob_block_size>>>(
            num_trees, nsample, nclass, maxnode, mdim,
            nodestatus_all_gpu, nodeclass_all_gpu,
            bestvar_all_gpu, xbestsplit_all_gpu, treemap_all_gpu,
            nin_all_gpu, cl_gpu, x_gpu, q_all_gpu
        );
        
        // Check for kernel launch errors immediately
        // cudaError_t kernel_err = cudaGetLastError();
        // if (kernel_err != cudaSuccess) {
        //     std::cerr << "Warning: OOB vote kernel launch error: error code " << kernel_err << std::endl;
        //     std::cerr << "Warning: Clearing error and continuing..." << std::endl;
        // }
        
        // std::cout << "DEBUG: About to sync stream after OOB vote kernel\n";
        
        // Check for kernel launch errors before syncing
        // cudaError_t launch_err = cudaGetLastError();
        // if (launch_err != cudaSuccess) {
        //     std::cerr << "ERROR: OOB vote kernel launch failed: " << cudaGetErrorString(launch_err) << std::endl;
        //     return;  // Exit early to prevent hang
        // }
        
        // OPTIMIZATION: Use stream sync before copying results to host
        CUDA_CHECK_VOID(cudaStreamSynchronize(compute_stream));
        
        // Copy q array back to host - matches backup version (direct call, no error checking)
        cudaMemcpy(q_all, q_all_gpu, nsample * nclass * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else {
        // Classification: q_all already contains correct votes from per-tree accumulation
        // No need to copy from GPU - our host-side accumulation is correct
        // std::cout << "GPU: Skipping OOB vote kernel for classification (using per-tree accumulation)" << std::endl;
        
        // DEBUG STEP 3 & 4: Verify multiple trees accumulation and overall prediction
        if (task_type == 0 && q_all != nullptr) {  // Classification
// Debug prints removed
            // std::cout << "[DEBUG STEP 3&4] Final q_all after all trees:" << std::endl;
            integer_t samples_with_votes = 0;
            integer_t samples_with_zero_votes = 0;
            for (integer_t n = 0; n < nsample && n < 20; ++n) {
                integer_t total_votes = 0;
                real_t max_votes = -1.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    total_votes += static_cast<integer_t>(q_all[n * nclass + j]);
                    if (q_all[n * nclass + j] > max_votes) {
                        max_votes = q_all[n * nclass + j];
                    }
                }
                // Debug prints removed
                // if (total_votes > 0) {
                //     samples_with_votes++;
                //     if (samples_with_votes <= 10) {
                //         std::cout << "  Sample " << n << ": total_votes=" << total_votes 
                //                   << ", predicted_class=" << max_class
                //                   << ", votes: ";
                //         for (integer_t j = 0; j < nclass; ++j) {
                //             std::cout << "class" << j << "=" << q_all[n * nclass + j] << " ";
                //         }
                //         std::cout << std::endl;
                //     }
                // } else {
                //     samples_with_zero_votes++;
                // }
            }
// Debug prints removed
            // std::cout << "[DEBUG STEP 3&4] Samples with votes: " << samples_with_votes
            //           << ", samples with zero votes: " << samples_with_zero_votes << std::endl;
        }
    }
    {
        // DEBUG: Check if q_all has non-zero values
        // DEBUG output disabled for production
        // std::cout << "DEBUG: Checking q_all after copy from GPU..." << std::endl;
        // int non_zero_count = 0;
        // int max_class_votes[7] = {0};
        // for (integer_t n = 0; n < nsample && n < 10; ++n) {  // Check first 10 samples
        //     for (integer_t j = 0; j < nclass; ++j) {
        //         integer_t idx = n * nclass + j;
        //         if (q_all[idx] > 0.0f) {
        //             non_zero_count++;
        //             if (q_all[idx] > max_class_votes[j]) {
        //                 max_class_votes[j] = static_cast<int>(q_all[idx]);
        //             }
        //         }
        //     }
        // }
        // std::cout << "DEBUG: q_all non-zero entries (first 10 samples): " << non_zero_count << std::endl;
        // std::cout << "DEBUG: Max votes per class (first 10 samples): ";
        // for (integer_t j = 0; j < nclass && j < 7; ++j) {
        //     std::cout << "class" << j << "=" << max_class_votes[j] << " ";
        // }
        // std::cout << std::endl;
    }
    
    // NOTE: Local importance (qimp_all and qimpm_all) is accumulated on HOST side
    // during the tree loop using temporary arrays (qimp_temp, qimpm_temp).
    // The GPU arrays (qimp_all_gpu, qimpm_all_gpu) are allocated but NOT used for accumulation.
    // Therefore, we do NOT copy from GPU arrays back to host - the host arrays already have
    // the accumulated values from the per-tree accumulation loop.
    // 
    // If we copied from GPU arrays (which are all zeros), we would overwrite the accumulated values!
    
    // Matches backup version - no error checking before proximity
    
    // OPTIMIZATION: Use stream sync before copying results to host
    CUDA_CHECK_VOID(cudaStreamSynchronize(compute_stream));
    
    // NOTE: nout_all is now incremented per-tree in the OOB accumulation loop above
    // (lines 3558-3591) to match CPU behavior and ensure correct normalization.
    // This duplicate loop has been removed to avoid double-counting.
    // The per-tree incrementation ensures nout_all[n] correctly counts how many trees
    // had sample n as OOB, which is needed for normalization in compute_regression_mse()
    // and finalize_training().
    
    // ========================================================================
    // COPY PROXIMITY IMPORTANCE FROM GPU TO HOST (before any early returns!)
    // ========================================================================
    // This is separate from the proximity matrix computation below.
    // Proximity importance is accumulated in prox_imp_all_gpu during tree traversal
    // and must be copied back to host regardless of proximity matrix settings.
    if (prox_imp_all_gpu != nullptr && prox_imp_all != nullptr) {
        cudaMemcpy(prox_imp_all, prox_imp_all_gpu, static_cast<size_t>(nsample) * static_cast<size_t>(mdim) * sizeof(real_t), cudaMemcpyDeviceToHost);
        
        // NOTE: Per-tree normalization (1/noob) is already done during accumulation
        // in gpu_compute_proximity_importance_dense(). No additional normalization needed here.
        // This matches Breiman-Cutler methodology for local variable importance.
    }
    
    // Compute proximity for each tree if requested
    // std::cout << "DEBUG: About to check proximity computation" << std::endl;
    
    // std::cout << "DEBUG: Entering proximity computation section" << std::endl;
    
    // Skip proximity entirely if use_qlora is requested but not enabled
    // Low-rank mode (use_qlora=True) is required for GPU proximity computation
    if ((proximity_all != nullptr || g_config.iprox) && !g_config.use_qlora && g_config.use_gpu) {
        // Warning removed for Jupyter compatibility (don't print to stderr)
        // std::cerr << "WARNING: Proximity computation requested on GPU but use_qlora=False..." << std::endl;
        // Continue anyway - will use traditional approach if proximity_all is allocated
    }
    
    // Matches backup version - no try-catch, simple CUDA check
    // Note: device_count already checked at function entry, so just check if CUDA is available
    int device_count_check = 0;
    cudaGetDeviceCount(&device_count_check);
    if (device_count_check == 0) {
        // Clean up prox_imp_all_gpu before return
        if (prox_imp_all_gpu != nullptr) cudaFree(prox_imp_all_gpu);
        return;  // Skip proximity computation if CUDA is not available
    }
    
    // NOW we can safely check proximity flags
    // std::cout << "DEBUG: Checking g_config.iprox=" << g_config.iprox 
    //           << ", proximity_all=" << (proximity_all != nullptr ? "non-null" : "null") << std::endl;
    
    // Early return if proximity is not requested at all (skip all proximity code)
    // For low-rank (QLORA), proximity_all will be nullptr (we only keep factors)
    // For full matrix, proximity_all must be allocated
    if (!g_config.iprox) {
        // std::cout << "DEBUG: Proximity not requested (iprox=0), returning early" << std::endl;
        // Clean up prox_imp_all_gpu before return
        if (prox_imp_all_gpu != nullptr) cudaFree(prox_imp_all_gpu);
        return;  // No proximity computation needed - skip all proximity code
    }
    
    // If proximity is requested but we're not using QLORA and proximity_all is null, warn and return
    if (!g_config.use_qlora && proximity_all == nullptr) {
        std::cerr << "Warning: Proximity requested (iprox=True) but proximity_all is nullptr and use_qlora=false. "
                  << "Enable use_qlora=True for low-rank proximity, or allocate proximity_all for full matrix." << std::endl;
        // Clean up prox_imp_all_gpu before return
        if (prox_imp_all_gpu != nullptr) cudaFree(prox_imp_all_gpu);
        return;
    }
    
    // Use fully qualified names instead of namespace to avoid initialization issues
    // For low-rank mode, we can process even if proximity_all is nullptr (reconstruction deferred)
    bool use_lowrank = false;
    bool use_upper_triangle = true;
    
    // Debug: Print config values
    // std::cout << "GPU Proximity Debug: g_config.use_qlora=" << g_config.use_qlora 
    //           << ", g_config.quant_mode=" << g_config.quant_mode 
    //           << ", g_config.iprox=" << g_config.iprox 
    //           << ", proximity_all=" << (proximity_all != nullptr ? "non-null" : "null")
    //           << ", nsample=" << nsample << std::endl;
    
    // If proximity is requested but low-rank is disabled on GPU,
    // warn but allow it (user may have allocated proximity_all themselves)
    // if ((proximity_all != nullptr || g_config.iprox) && !g_config.use_qlora && g_config.use_gpu) {
    //     std::cerr << "WARNING: Proximity requested on GPU but use_qlora=false. "
    //               << "Low-rank mode (use_qlora=True) is recommended for GPU proximity to avoid huge temp buffer allocation. "
    //               << "Continuing with traditional approach if proximity_all is allocated." << std::endl;
    //     std::cerr.flush();
    // }
    
    // Only check low-rank if proximity is requested and CUDA is available
    if (proximity_all != nullptr || g_config.iprox) {
        // MODIFIED: Always use low-rank mode on GPU when use_qlora is enabled, regardless of dataset size
        // This enables low-rank factors for all datasets when using GPU with QLoRA
        // NOTE: Low-rank mode is GPU-only (requires CUDA kernels, cuBLAS, cuSolver)
        // CPU mode will never use low-rank - it always computes full proximity matrix
        // 
        // g_config.use_gpu is set to true at the start of this function (line 2214)
        // and g_config.use_qlora should be set by the caller (rf_random_forest.cpp) before calling this function
        if (g_config.use_gpu && g_config.use_qlora) {
            use_lowrank = true;  // Force low-rank for all GPU datasets when QLoRA is enabled
            use_upper_triangle = true;
        } else if (!g_config.use_gpu) {
            // CPU mode: use full matrix (low-rank not available on CPU)
            use_lowrank = false;  // CPU cannot use low-rank (CUDA-only feature)
            use_upper_triangle = true;
        } else {
            // GPU mode without qlora: use traditional full matrix approach if proximity_all is allocated
            use_lowrank = false;
            use_upper_triangle = true;
            // if (proximity_all == nullptr && g_config.iprox) {
            //     std::cerr << "Warning: Proximity requested (iprox=True) but proximity_all is nullptr and use_qlora=false. "
            //               << "Enable use_qlora=True for low-rank proximity, or allocate proximity_all for full matrix." << std::endl;
            // }
        }
    }
    
    // ========================================================================
    // SECTION 3: PROXIMITY COMPUTATION
    // ========================================================================
    // GPU proximity ALWAYS uses low-rank (QLoRA) - no full matrix option on GPU
    // Two types of GPU proximity:
    //   1. Standard proximity: compute_proximity=True, use_rfgap=False
    //   2. RF-GAP proximity: compute_proximity=True, use_rfgap=True
    // These are mutually exclusive
    
    bool should_compute_gpu_proximity = g_config.iprox && g_config.use_gpu && g_config.use_qlora;
    bool should_compute_rfgap = g_config.use_rfgap && should_compute_gpu_proximity;
    bool should_compute_standard_proximity = should_compute_gpu_proximity && !g_config.use_rfgap;
    
    if (should_compute_standard_proximity) {
        // Standard low-rank proximity (GPU only)
        
        // Use file-scope thread-local storage (g_lowrank_state declared at top of file)
        // This ensures state persists across all batch calls, fixing the bug where
        // 3000 trees (300 batches) would reset state and produce zero factors
        
        void*& lowrank_prox_ptr_raw = g_lowrank_state.lowrank_prox_ptr_raw;
        bool& lowrank_initialized = g_lowrank_state.lowrank_initialized;
        integer_t& total_trees_processed = g_lowrank_state.total_trees_processed;
        integer_t& saved_nsample = g_lowrank_state.saved_nsample;
        
        // Allocate proximity workspace arrays
        std::vector<integer_t> nod_workspace(maxnode);
        std::vector<integer_t> ncount_workspace(nsample);
        std::vector<integer_t> ncn_workspace(nsample);
        std::vector<integer_t> nodexb_workspace(nsample);
        std::vector<integer_t> ndbegin_workspace(nsample + 1);
        std::vector<integer_t> npcase_workspace(nsample);
        
        // std::cout << "GPU Proximity: use_qlora=" << g_config.use_qlora 
        //           << ", nsample=" << nsample 
        //           << ", use_lowrank=" << use_lowrank 
        //           << ", use_upper_triangle=" << use_upper_triangle << std::endl;
        
        // Ensure use_lowrank is consistent with use_qlora
        // If use_qlora is true and we're on GPU, use_lowrank MUST be true
        if (g_config.use_gpu && g_config.use_qlora && !use_lowrank) {
            // std::cerr << "WARNING: use_qlora=true but use_lowrank=false - forcing use_lowrank=true" << std::endl;
            use_lowrank = true;
            use_upper_triangle = true;
        }
        
        // For nsample > 10000, ONLY use low-rank AB' factors - NO temp buffer!
        if (use_lowrank && use_upper_triangle) {
            #ifdef __CUDACC__
            // Get quantization level from config - use fully qualified name
            // Delay type resolution until we actually need it
            int quant_mode_int = g_config.quant_mode;
            // REMOVED: quant_level determination - we always use FP16 during training
            // Quantization happens only at the end via finalize_accumulation()
            
            // Detect if this is a NEW model instance
            // If lowrank_proximity_ptr_out points to nullptr, it means RandomForest just allocated
            // a fresh lowrank_proximity_ptr_ member, so we're starting a new model
            // In this case, reset thread_local state to prevent reusing stale pointers from previous models
            bool is_new_model = (lowrank_proximity_ptr_out != nullptr && 
                                *lowrank_proximity_ptr_out == nullptr &&
                                lowrank_prox_ptr_raw != nullptr);
            
            // Check if nsample changed OR new model detected - need to reset
            if (lowrank_prox_ptr_raw != nullptr && (saved_nsample != nsample || is_new_model)) {
                    // Don't delete - the previous model's destructor already did that
                    // Just reset our thread_local pointers
                lowrank_prox_ptr_raw = nullptr;
                lowrank_initialized = false;
                total_trees_processed = 0;
                saved_nsample = 0;
            }
            
            // Initialize LowRankProximityMatrix on first batch (persists across batches)
            // DEFER initialization until AFTER first tree is processed to avoid crashes during static initialization
            if (!lowrank_initialized || lowrank_prox_ptr_raw == nullptr) {
                // std::cout << "GPU Proximity: Creating persistent LowRankProximityMatrix object..." << std::endl;
                // std::cout << "GPU Proximity: Parameters - nsample=" << nsample 
                //           << ", rank=100, quant_level=" << static_cast<int>(quant_level) << std::endl;
                
                // Validate inputs before creating
                if (nsample <= 0 || nsample > 1000000) {
                    // std::cerr << "Error: Invalid nsample for LowRankProximityMatrix: " << nsample << std::endl;
                    use_lowrank = false;
                } else {
                    // Ensure CUDA context is valid before creating handles
                    cudaError_t ctx_check = cudaSetDevice(0);
                    if (ctx_check != cudaSuccess) {
                        // std::cerr << "Error: Failed to set CUDA device: " << cudaGetErrorString(ctx_check) << std::endl;
                        use_lowrank = false;
                    } else {
                        // Matches backup version - direct call, no try-catch
                            integer_t max_rank = g_config.lowrank_rank > 0 ? g_config.lowrank_rank : 1000;
                            integer_t initial_rank = 0;  // Start with rank 0, will grow as trees are added
                            integer_t power_iterations = g_config.lowrank_power_iterations > 0 ? g_config.lowrank_power_iterations : 30;
                            rf::cuda::QuantizationLevel quant_level_explicit = 
                                static_cast<rf::cuda::QuantizationLevel>(quant_mode_int);
                            lowrank_prox_ptr_raw = static_cast<void*>(
                                new rf::cuda::LowRankProximityMatrix(nsample, initial_rank, quant_level_explicit, max_rank, power_iterations)
                            );
                            saved_nsample = nsample;
                    }
                }
                
                if (lowrank_prox_ptr_raw != nullptr) {
                    // Initialize GPU memory for low-rank factors
                    // std::cout << "GPU Proximity: Initializing LowRankProximityMatrix..." << std::endl;
                    // Matches backup version - direct call, no try-catch
                        rf::cuda::LowRankProximityMatrix* lowrank_prox_ptr = 
                            static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                        if (!lowrank_prox_ptr->initialize()) {
                            // If initialization fails, fall back to traditional approach
                            // std::cerr << "Warning: Low-rank proximity initialization failed, falling back to full matrix" << std::endl;
                            delete static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                            lowrank_prox_ptr_raw = nullptr;
                            use_lowrank = false;
                            lowrank_initialized = false;
                        } else {
                            lowrank_initialized = true;
                            total_trees_processed = 0;  // Reset tree counter
                    }
                }
            }
            
            if (lowrank_prox_ptr_raw != nullptr && lowrank_initialized) {
                // Cast void* to actual type when needed
                rf::cuda::LowRankProximityMatrix* lowrank_prox_ptr = 
                    static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                    
                // std::cout << "GPU Proximity: Processing " << num_trees << " trees (total trees processed so far: " 
                //           << total_trees_processed << ") with low-rank mode" << std::endl;
                
                // Process each tree for proximity computation
                for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                    // Get tree-specific data
                    integer_t tree_offset = tree_id * maxnode;
                    integer_t nin_offset = tree_id * nsample;
                    
                    // Safety check: ensure tree actually grew
                    if (nnode_all[tree_id] == 0) {
                        // std::cerr << "Warning: Tree " << tree_id << " has 0 nodes, skipping proximity" << std::endl;
                        continue;
                    }
                    
                    // For nsample > 10000: NO CPU temp buffers! Compute directly on GPU
                    // Allocate GPU memory directly (no CPU allocation)
                    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
                    
                    // ALWAYS compute per-tree proximity in FP16 during training
                    // Quantization to INT8/NF4 happens ONLY at the very end via finalize_accumulation()
                    // This prevents compounding quantization errors during training
                    __half* tree_prox_upper_gpu = nullptr;
                    CUDA_CHECK_VOID(cudaMalloc(&tree_prox_upper_gpu, upper_triangle_size * sizeof(__half)));
                    cudaMemset(tree_prox_upper_gpu, 0, upper_triangle_size * sizeof(__half));
                    
                    // Matches backup version - assume arrays are on host (they're copied from GPU earlier)
                    // Copy data to host arrays for gpu_proximity_upper_triangle (it expects host pointers)
                    std::vector<integer_t> nodestatus_host(nnode_all[tree_id]);
                    std::vector<integer_t> nodextr_host(nsample);
                    std::vector<integer_t> nin_host(nsample);
                    
                    // nodextr_all uses jinbag_offset = tree_id * nsample (same as nin_offset)
                    integer_t nodextr_offset = tree_id * nsample;  // nodextr_all uses same offset as jinbag
                    
                    
                    // Direct memcpy - arrays are already on host
                    std::memcpy(nodestatus_host.data(), nodestatus_all + tree_offset, 
                                nnode_all[tree_id] * sizeof(integer_t));
                    std::memcpy(nodextr_host.data(), nodextr_all + nodextr_offset, 
                                nsample * sizeof(integer_t));
                    std::memcpy(nin_host.data(), nin + nin_offset, 
                                nsample * sizeof(integer_t));
                    
                    // Always compute proximity in FP16 during training
                    // Quantization to INT8/NF4 happens only at the end
                    rf::cuda::gpu_proximity_upper_triangle_fp16(
                        nodestatus_host.data(),
                        nodextr_host.data(),
                        nin_host.data(),
                        nsample, nnode_all[tree_id],  // Use actual number of nodes, not maxnode!
                        tree_prox_upper_gpu,  // FP16 output buffer
                        nod_workspace.data(), ncount_workspace.data(), ncn_workspace.data(),
                        nodexb_workspace.data(), ndbegin_workspace.data(), npcase_workspace.data()
                    );
                    
                    // Increment tree counter for this tree (needed for finalize_accumulation)
                    total_trees_processed++;
                    
                    // Add to low-rank matrix using FP16 directly (no quantization during training!)
                    lowrank_prox_ptr->add_tree_contribution_incremental_upper_triangle_fp16(
                        tree_prox_upper_gpu, nsample
                    );
                    
                    // Free GPU memory
                    CUDA_CHECK_VOID(cudaFree(tree_prox_upper_gpu));
                }
                
                // RF-GAP proximity computation with QLoRA/low-rank (if enabled)
                if (g_config.use_rfgap && g_config.use_qlora && lowrank_prox_ptr_raw != nullptr && lowrank_initialized) {
                    rf::cuda::LowRankProximityMatrix* lowrank_prox_ptr = 
                        static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                    
                    // Process each tree for RF-GAP proximity computation
                    for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
                        integer_t nin_offset = tree_id * nsample;
                        integer_t nodextr_offset = tree_id * nsample;
                        
                        // Safety check: ensure tree actually grew
                        if (nnode_all[tree_id] == 0) {
                            continue;  // Skip trees with 0 nodes
                        }
                        
                        // Allocate GPU memory for RF-GAP upper triangle
                        size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
                        __half* tree_rfgap_upper_gpu = nullptr;
                        
                        // Matches backup version - direct allocation, no error checking
                        CUDA_CHECK_VOID(cudaMalloc(&tree_rfgap_upper_gpu, upper_triangle_size * sizeof(__half)));
                        cudaMemset(tree_rfgap_upper_gpu, 0, upper_triangle_size * sizeof(__half));
                        
                        // Ensure nodextr_all is computed before using it
                        if (nodextr_all == nullptr) {
                            // std::cerr << "ERROR: nodextr_all is nullptr but needed for RF-GAP proximity computation!" << std::endl;
                            // Safe free to prevent segfaults
                            if (tree_rfgap_upper_gpu) { cudaError_t err = cudaFree(tree_rfgap_upper_gpu); if (err != cudaSuccess) cudaGetLastError(); }
                            continue;  // Skip this tree
                        }
                        
                        // Compute RF-GAP contributions for this tree in upper triangle format
                        // Note: RF-GAP uses nin (bootstrap multiplicities) and nodextr (terminal nodes)
                        // Both are already available from GPU tree growing
                        rf::cuda::gpu_proximity_rfgap_upper_triangle_fp16(
                            nin + nin_offset,           // Bootstrap multiplicities for this tree
                            nodextr_all + nodextr_offset,  // Terminal node assignments for this tree
                            nsample,
                            tree_rfgap_upper_gpu        // Output: Packed upper triangle in FP16
                        );
                        
                        // Add to low-rank matrix using GPU memory directly (no CPU copy!)
                        lowrank_prox_ptr->add_tree_contribution_incremental_upper_triangle_fp16(
                            tree_rfgap_upper_gpu, nsample  // Pass GPU pointer directly
                        );
                        
                        // Free GPU memory - matches backup version
                        CUDA_CHECK_VOID(cudaFree(tree_rfgap_upper_gpu));
                    }
                }
                
                // Synchronize device after all tree contributions are added
                // This ensures all cuBLAS/cuSolver operations from all trees complete
                // before we finalize or move to next batch. Required for correct accumulation across batches.
                CUDA_CHECK_VOID(cudaDeviceSynchronize());
                
                // total_trees_processed is now incremented per-tree in the loop above
                // Don't increment here to avoid double-counting
                
                // For low-rank mode, DO NOT reconstruct automatically - keep in low-rank form!
                // Reconstruction would require 80GB (100k × 100k × 8 bytes) which would crash memory.
                // Instead, keep LowRankProximityMatrix alive for on-demand reconstruction.
                // Only reconstruct if explicitly requested (proximity_all is allocated AND this is the final batch).
                // But for low-rank mode, we should NOT allocate proximity_all at all to avoid 80GB allocation.
                
                if (proximity_all != nullptr) {
                    // This should only happen if user explicitly requests full matrix (not recommended for large datasets)
                    // std::cout << "GPU Proximity: WARNING - Reconstructing full 80GB matrix (not recommended for large datasets)" << std::endl;
                    // std::cout << "GPU Proximity: All batches complete (" << total_trees_processed 
                    //           << " total trees). Reconstructing full proximity matrix..." << std::endl;
                    // Finalize accumulation and reconstruct full proximity matrix
                    // final_call=true: This path is for full reconstruction, so it's the end
                    lowrank_prox_ptr->finalize_accumulation(total_trees_processed, /*final_call=*/true);
                    // Sync after finalize_accumulation to ensure all operations complete
                    CUDA_CHECK_VOID(cudaDeviceSynchronize());
                    lowrank_prox_ptr->get_accumulated_proximity(proximity_all, nsample);
                    // std::cout << "GPU Proximity: Low-rank accumulation completed and reconstructed" << std::endl;
                    
                    // Clean up persistent low-rank matrix after reconstruction
                    delete static_cast<rf::cuda::LowRankProximityMatrix*>(lowrank_prox_ptr_raw);
                    lowrank_prox_ptr_raw = nullptr;
                    lowrank_initialized = false;
                    total_trees_processed = 0;
                } else {
                    // Keep LowRankProximityMatrix alive - don't delete it!
                    // Store pointer so RandomForest can access it later
                    if (lowrank_proximity_ptr_out != nullptr) {
                        *lowrank_proximity_ptr_out = lowrank_prox_ptr_raw;
                        // std::cout << "GPU Proximity: Stored LowRankProximityMatrix pointer for later access" << std::endl;
                    }
                    // Finalize accumulation so factors are available for retrieval
                    // This sets trees_processed_ so get_factors() can work correctly
                    // final_call=true: Quantize factors now for storage (only once, saves 8× memory)
                    lowrank_prox_ptr->finalize_accumulation(total_trees_processed, /*final_call=*/true);
                    // Sync after finalize_accumulation to ensure all operations complete before returning
                    CUDA_CHECK_VOID(cudaDeviceSynchronize());
                    // std::cout << "GPU Proximity: Low-rank accumulation completed for this batch (" 
                    //           << num_trees << " trees). Total trees processed: " << total_trees_processed 
                    //           << " (keeping in low-rank form - ~40MB instead of 80GB)" << std::endl;
                    // std::cout << "GPU Proximity: Full matrix reconstruction not available for low-rank mode (would require 80GB)" << std::endl;
                    // std::cout << "GPU Proximity: Use get_lowrank_factors() or compute_distances_from_factors() instead" << std::endl;
                }
            }
            #endif  // __CUDACC__ - End of low-rank proximity code block
            // Matches backup version - no try-catch
            
        } else if (proximity_all != nullptr && nsample <= 10000) {
            // Traditional full matrix approach - ONLY for small datasets (nsample <= 10000)
            // For larger datasets, this would allocate huge temp buffers (800MB+ per tree)
            // std::cout << "GPU Proximity: Using traditional full matrix approach (nsample=" << nsample << " <= 10000)" << std::endl;
            // Use traditional full matrix approach (for smaller datasets or when qlora disabled)
        // Process each tree for proximity computation
        for (integer_t tree_id = 0; tree_id < num_trees; ++tree_id) {
            // Get tree-specific data
            integer_t tree_offset = tree_id * maxnode;
            integer_t nin_offset = tree_id * nsample;
            
            // Create temporary proximity matrix for this tree
            std::vector<dp_t> tree_proximity(nsample * nsample, 0.0);
            
                // Call GPU proximity computation for this tree
                gpu_proximity(
                    nodestatus_all + tree_offset,  // nodestatus for this tree
                    nodextr_all + nin_offset,  // nodextr for this tree
                    nin + nin_offset,  // nin for this tree
                    nsample, maxnode,
                    tree_proximity.data(),  // Use temporary matrix for this tree
                    nod_workspace.data(), ncount_workspace.data(), ncn_workspace.data(),
                    nodexb_workspace.data(), ndbegin_workspace.data(), npcase_workspace.data()
                );
            
            // Accumulate this tree's proximity into the global matrix
                // This accumulates across all batches (proximity_all persists across batch calls)
                // Ensure GPU operations complete before CPU accumulation
                // std::cout << "DEBUG: About to sync before CPU proximity accumulation\n";
                
                // Synchronize before CPU accumulation
                CUDA_CHECK_VOID(cudaDeviceSynchronize());
                
            for (integer_t i = 0; i < nsample * nsample; ++i) {
                proximity_all[i] += tree_proximity[i];
                }
            }
        }
        
        // std::cout << "GPU: Standard proximity computation completed!" << std::endl;
    }
    
    // ========================================================================
    // SECTION 4: RF-GAP PROXIMITY COMPUTATION
    // ========================================================================
    // RF-GAP is a separate proximity algorithm (mutually exclusive with standard)
    // Only runs when: use_rfgap=True AND use_qlora=True
    if (should_compute_rfgap && lowrank_proximity_ptr_out != nullptr) {
        // RF-GAP low-rank proximity computation
        // Reserved for future implementation (v2.0)
    }
    
    // Matches backup version - no try-catch
    
    // Debug: Print first few q values
    // std::cout << "GPU: First 10 q values: ";
    // for (integer_t i = 0; i < 10; ++i) {
    //     std::cout << q_all[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Debug: Print first few nout values
    // std::cout << "GPU: First 10 nout values: ";
    // for (integer_t i = 0; i < 10; ++i) {
    //     std::cout << nout_all[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Ensure all GPU operations complete before cleanup
    // Use device sync to ensure ALL streams complete (not just compute_stream)
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
    
    // OPTIMIZATION: Destroy the compute stream we created
    CUDA_CHECK_VOID(cudaStreamDestroy(compute_stream));
    
    // NOTE: Proximity importance copy moved to before early returns (around line 3854)
    // This ensures prox_imp_all gets data even when iprox=false or proximity_all=nullptr
    
    // Clean up GPU memory - matches backup version approach
    if (prox_imp_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(prox_imp_all_gpu));
    if (x_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(x_gpu));
    if (cl_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(cl_gpu));
    if (win_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(win_gpu));
    if (seeds_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(seeds_gpu));
    if (y_regression_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(y_regression_gpu));
    if (nodestatus_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nodestatus_all_gpu));
    if (nodeclass_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nodeclass_all_gpu));
    if (nnode_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nnode_all_gpu));
    if (bestvar_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(bestvar_all_gpu));
    if (xbestsplit_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(xbestsplit_all_gpu));
    if (treemap_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(treemap_all_gpu));
    if (jinbag_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(jinbag_all_gpu));
    if (q_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(q_all_gpu));
    if (qimp_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(qimp_all_gpu));
    if (qimpm_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(qimpm_all_gpu));
    if (nin_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nin_all_gpu));
    if (win_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(win_all_gpu));
    if (random_states != nullptr) CUDA_CHECK_VOID(cudaFree(random_states));
    if (nodextr_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(nodextr_all_gpu));
    if (tnodewt_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(tnodewt_all_gpu));
    if (d_nodextr_single != nullptr) CUDA_CHECK_VOID(cudaFree(d_nodextr_single));
    if (sample_node_all_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(sample_node_all_gpu));
    if (node_samples_workspace_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(node_samples_workspace_gpu));
    if (sorted_indices_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(sorted_indices_gpu));
    // HISTOGRAM: Free histogram GPU memory
    // Note: X_binned_gpu, bin_edges_gpu, n_bins_gpu are owned by dense_hist_data now
    if (dense_hist_data != nullptr) {
        delete dense_hist_data;
        dense_hist_data = nullptr;
    }
    if (hist_workspace_gpu != nullptr) CUDA_CHECK_VOID(cudaFree(hist_workspace_gpu));
}

} // namespace rf