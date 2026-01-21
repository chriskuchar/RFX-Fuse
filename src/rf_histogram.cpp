#include "rf_histogram.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>

namespace rf {

// ============================================================================
// Find bin index using binary search
// ============================================================================

uint8_t find_bin(real_t value, const real_t* edges, integer_t n_bins) {
    // Binary search for the bin containing value
    // edges[i] <= value < edges[i+1] means bin i
    
    if (value <= edges[0]) return 0;
    if (value >= edges[n_bins]) return static_cast<uint8_t>(n_bins - 1);
    
    integer_t lo = 0, hi = n_bins;
    while (lo < hi) {
        integer_t mid = (lo + hi) / 2;
        if (value < edges[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return static_cast<uint8_t>(std::max(0, static_cast<int>(lo) - 1));
}

// ============================================================================
// Compute bin edges using quantiles
// ============================================================================

void compute_bin_edges_quantile(
    const real_t* X,
    integer_t nsample,
    integer_t mdim,
    const integer_t* cat,
    integer_t n_bins,
    FeatureBinInfo* out_info
) {
    std::vector<real_t> col_values(nsample);
    
    for (integer_t f = 0; f < mdim; f++) {
        // Extract values for this feature
        for (integer_t i = 0; i < nsample; i++) {
            col_values[i] = X[i * mdim + f];
        }
        
        // Check if categorical (3+ categories only)
        // Binary features (cat[f] == 2) use quantitative histogram path for better sparsity handling
        if (cat != nullptr && cat[f] > 2) {
            out_info[f].is_categorical = true;
            out_info[f].n_bins = cat[f];
            out_info[f].is_binary = false;
            continue;
        }
        
        // Sort to find quantiles
        std::sort(col_values.begin(), col_values.end());
        
        out_info[f].min_val = col_values[0];
        out_info[f].max_val = col_values[nsample - 1];
        
        // ======================================================================
        // MATCH SPARSE IMPLEMENTATION EXACTLY: Create separate unique_values vector
        // This is the key fix - use the same algorithm as rf_sparse_histogram.cu
        // ======================================================================
        std::vector<real_t> unique_values;
        unique_values.push_back(col_values[0]);
        for (integer_t i = 1; i < nsample; i++) {
            if (col_values[i] != unique_values.back()) {
                unique_values.push_back(col_values[i]);
            }
        }
        integer_t n_unique = static_cast<integer_t>(unique_values.size());
        
        // Compute bin edges - EXACTLY as in rf_sparse_histogram.cu
        real_t* edges = out_info[f].bin_edges;
        
        if (n_unique == 0) {
            // No values at all - use default edges
            out_info[f].n_bins = 1;
            out_info[f].is_binary = false;
            out_info[f].is_categorical = false;
            for (integer_t b = 0; b <= n_bins; b++) {
                edges[b] = static_cast<real_t>(b);
            }
        } else if (n_unique <= 2) {
            // BINARY OR NEAR-BINARY FEATURE: Use midpoint threshold
            out_info[f].n_bins = 2;
            out_info[f].is_binary = true;
            out_info[f].is_categorical = false;
            real_t min_val = unique_values[0];
            real_t max_val = unique_values[n_unique - 1];
            edges[0] = min_val - 1.0f;                    // Below minimum
            edges[1] = (min_val + max_val) / 2.0f;        // MIDPOINT threshold
            edges[2] = max_val + 1.0f;                    // Above maximum
            // Fill remaining edges
            for (integer_t b = 3; b <= n_bins; b++) {
                edges[b] = edges[2];
        }
        } else {
            // CONTINUOUS FEATURE: Use quantile binning over UNIQUE values
        integer_t actual_bins = std::min(n_bins, n_unique);
        out_info[f].n_bins = actual_bins;
            out_info[f].is_binary = false;
        out_info[f].is_categorical = false;
            
            edges[0] = unique_values[0] - 1.0f;  // Min edge below smallest value
            for (integer_t b = 1; b <= actual_bins; b++) {
                integer_t quantile_idx = (b * n_unique) / actual_bins;
                if (quantile_idx >= n_unique) quantile_idx = n_unique - 1;
                edges[b] = unique_values[quantile_idx];
            }
            // Fill remaining edges with max value
            for (integer_t b = actual_bins + 1; b <= n_bins; b++) {
                edges[b] = edges[actual_bins];
        }
        
        // Ensure edges are strictly increasing (handle ties)
        for (integer_t b = 1; b <= actual_bins; b++) {
                if (edges[b] <= edges[b-1]) {
                    edges[b] = edges[b-1] + 1e-7f;
                }
            }
        }
    }
}

// ============================================================================
// Convert data to binned representation
// ============================================================================

void bin_data_cpu(
    const real_t* X,
    integer_t nsample,
    integer_t mdim,
    const FeatureBinInfo* info,
    uint8_t* X_binned
) {
    for (integer_t i = 0; i < nsample; i++) {
        for (integer_t f = 0; f < mdim; f++) {
            real_t val = X[i * mdim + f];
            
            if (info[f].is_categorical) {
                // Categorical: use value directly (assume 0-indexed)
                integer_t cat_val = static_cast<integer_t>(val + 0.5f);
                cat_val = std::max(0, std::min(cat_val, info[f].n_bins - 1));
                X_binned[i * mdim + f] = static_cast<uint8_t>(cat_val);
            } else if (info[f].is_binary) {
                // Binary: simple threshold
                X_binned[i * mdim + f] = (val > info[f].bin_edges[1]) ? 1 : 0;
            } else {
                // Continuous: find bin using edges
                X_binned[i * mdim + f] = find_bin(val, info[f].bin_edges, info[f].n_bins);
            }
        }
    }
}

// ============================================================================
// Build Classification Histogram
// ============================================================================

void build_classification_histogram_cpu(
    const uint8_t* X_binned,
    const integer_t* y,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    const FeatureBinInfo& info,
    integer_t nclass,
    ClassificationHistogram& hist
) {
    hist.allocate(info.n_bins, nclass);
    hist.clear();
    
    for (integer_t i = 0; i < node_count; i++) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        integer_t c = y[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < info.n_bins && c < nclass) {
            hist.at(bin, c) += w;
        }
    }
}

// ============================================================================
// Build Regression Histogram
// ============================================================================

void build_regression_histogram_cpu(
    const uint8_t* X_binned,
    const real_t* y_reg,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    const FeatureBinInfo& info,
    RegressionHistogram& hist
) {
    hist.allocate(info.n_bins);
    hist.clear();
    
    for (integer_t i = 0; i < node_count; i++) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        real_t val = y_reg[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < info.n_bins) {
            hist.sum_y[bin] += val * w;
            hist.sum_y2[bin] += val * val * w;
            hist.count[bin] += w;
        }
    }
}

// ============================================================================
// Find Best Split from Classification Histogram (Gini)
// ============================================================================

void find_best_split_classification_histogram(
    const ClassificationHistogram& hist,
    const FeatureBinInfo& info,
    integer_t nclass,
    real_t* best_crit,
    integer_t* best_bin,
    real_t* best_split_value
) {
    // Compute total counts per class (right side initially has all)
    std::vector<real_t> left_counts(nclass, 0.0f);
    std::vector<real_t> right_counts(nclass, 0.0f);
    
    real_t total_left = 0.0f;
    real_t total_right = 0.0f;
    
    // Initialize right with all samples
    for (integer_t b = 0; b < hist.n_bins; b++) {
        for (integer_t c = 0; c < nclass; c++) {
            right_counts[c] += hist.at(b, c);
            total_right += hist.at(b, c);
        }
    }
    
    real_t best_gini = 1e20f;
    integer_t best_b = 0;
    
    // Scan left to right, moving one bin at a time from right to left
    for (integer_t b = 0; b < hist.n_bins - 1; b++) {
        // Move bin b from right to left
        for (integer_t c = 0; c < nclass; c++) {
            real_t count = hist.at(b, c);
            left_counts[c] += count;
            right_counts[c] -= count;
            total_left += count;
            total_right -= count;
        }
        
        // Skip if either side is empty
        if (total_left < 1.0f || total_right < 1.0f) continue;
        
        // Compute Gini impurity for left and right
        real_t gini_left = 1.0f;
        real_t gini_right = 1.0f;
        
        for (integer_t c = 0; c < nclass; c++) {
            real_t p_left = left_counts[c] / total_left;
            real_t p_right = right_counts[c] / total_right;
            gini_left -= p_left * p_left;
            gini_right -= p_right * p_right;
        }
        
        // Weighted average Gini
        real_t total = total_left + total_right;
        real_t gini = (total_left * gini_left + total_right * gini_right) / total;
        
        if (gini < best_gini) {
            best_gini = gini;
            best_b = b;
        }
    }
    
    *best_crit = best_gini;
    *best_bin = best_b;
    
    // Convert bin to actual split value
    if (info.is_categorical) {
        *best_split_value = static_cast<real_t>(best_b);
    } else {
        // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
        *best_split_value = (info.bin_edges[best_b] + info.bin_edges[best_b + 1]) / 2.0f;
    }
}

// ============================================================================
// Find Best Split from Regression Histogram (MSE)
// ============================================================================

void find_best_split_regression_histogram(
    const RegressionHistogram& hist,
    const FeatureBinInfo& info,
    real_t* best_crit,
    integer_t* best_bin,
    real_t* best_split_value
) {
    // Compute totals for right side
    real_t sum_left = 0.0f, sum2_left = 0.0f, cnt_left = 0.0f;
    real_t sum_right = 0.0f, sum2_right = 0.0f, cnt_right = 0.0f;
    
    for (integer_t b = 0; b < hist.n_bins; b++) {
        sum_right += hist.sum_y[b];
        sum2_right += hist.sum_y2[b];
        cnt_right += hist.count[b];
    }
    
    real_t best_mse = 1e20f;
    integer_t best_b = 0;
    
    // Scan left to right
    for (integer_t b = 0; b < hist.n_bins - 1; b++) {
        // Move bin b from right to left
        sum_left += hist.sum_y[b];
        sum2_left += hist.sum_y2[b];
        cnt_left += hist.count[b];
        
        sum_right -= hist.sum_y[b];
        sum2_right -= hist.sum_y2[b];
        cnt_right -= hist.count[b];
        
        // Skip if either side is empty
        if (cnt_left < 1.0f || cnt_right < 1.0f) continue;
        
        // Compute MSE: E[Y^2] - E[Y]^2
        real_t mean_left = sum_left / cnt_left;
        real_t mean_right = sum_right / cnt_right;
        
        real_t mse_left = sum2_left / cnt_left - mean_left * mean_left;
        real_t mse_right = sum2_right / cnt_right - mean_right * mean_right;
        
        // Weighted sum of MSE (variance reduction criterion)
        real_t total = cnt_left + cnt_right;
        real_t mse = (cnt_left * mse_left + cnt_right * mse_right) / total;
        
        if (mse < best_mse) {
            best_mse = mse;
            best_b = b;
        }
    }
    
    *best_crit = best_mse;
    *best_bin = best_b;
    
    // Convert bin to actual split value
    if (info.is_categorical) {
        *best_split_value = static_cast<real_t>(best_b);
    } else {
        // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
        *best_split_value = (info.bin_edges[best_b] + info.bin_edges[best_b + 1]) / 2.0f;
    }
}

// ============================================================================
// Sparse Histogram Functions
// ============================================================================

void build_classification_histogram_sparse_cpu(
    const real_t* sparse_values,
    const integer_t* sparse_indices,
    const integer_t* sparse_indptr,
    const integer_t* y,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    const FeatureBinInfo& info,
    integer_t nclass,
    ClassificationHistogram& hist
) {
    hist.allocate(info.n_bins, nclass);
    hist.clear();
    
    // Find bin for zero value
    uint8_t zero_bin = 0;
    if (!info.is_categorical) {
        zero_bin = find_bin(0.0f, info.bin_edges, info.n_bins);
    }
    
    // Count samples that have explicit values for this feature
    integer_t explicit_count = 0;
    
    for (integer_t i = 0; i < node_count; i++) {
        integer_t s = node_samples[i];
        integer_t c = y[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        // Search for feature in sparse row
        bool found = false;
        for (integer_t j = sparse_indptr[s]; j < sparse_indptr[s + 1]; j++) {
            if (sparse_indices[j] == feature) {
                // Found explicit value
                real_t val = sparse_values[j];
                uint8_t bin;
                if (info.is_categorical) {
                    bin = static_cast<uint8_t>(val + 0.5f);
                } else {
                    bin = find_bin(val, info.bin_edges, info.n_bins);
                }
                if (bin < info.n_bins && c < nclass) {
                    hist.at(bin, c) += w;
                }
                explicit_count++;
                found = true;
                break;
            }
        }
        
        // If not found, it's an implicit zero
        if (!found && c < nclass) {
            hist.at(zero_bin, c) += w;
        }
    }
}

void build_regression_histogram_sparse_cpu(
    const real_t* sparse_values,
    const integer_t* sparse_indices,
    const integer_t* sparse_indptr,
    const real_t* y_reg,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    const FeatureBinInfo& info,
    RegressionHistogram& hist
) {
    hist.allocate(info.n_bins);
    hist.clear();
    
    // Find bin for zero value
    uint8_t zero_bin = 0;
    if (!info.is_categorical) {
        zero_bin = find_bin(0.0f, info.bin_edges, info.n_bins);
    }
    
    for (integer_t i = 0; i < node_count; i++) {
        integer_t s = node_samples[i];
        real_t val_y = y_reg[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        // Search for feature in sparse row
        bool found = false;
        for (integer_t j = sparse_indptr[s]; j < sparse_indptr[s + 1]; j++) {
            if (sparse_indices[j] == feature) {
                real_t val = sparse_values[j];
                uint8_t bin;
                if (info.is_categorical) {
                    bin = static_cast<uint8_t>(val + 0.5f);
                } else {
                    bin = find_bin(val, info.bin_edges, info.n_bins);
                }
                if (bin < info.n_bins) {
                    hist.sum_y[bin] += val_y * w;
                    hist.sum_y2[bin] += val_y * val_y * w;
                    hist.count[bin] += w;
                }
                found = true;
                break;
            }
        }
        
        // If not found, it's an implicit zero
        if (!found) {
            hist.sum_y[zero_bin] += val_y * w;
            hist.sum_y2[zero_bin] += val_y * val_y * w;
            hist.count[zero_bin] += w;
        }
    }
}

} // namespace rf

