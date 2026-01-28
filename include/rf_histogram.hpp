#ifndef RF_HISTOGRAM_HPP
#define RF_HISTOGRAM_HPP

#include "rf_types.hpp"
#include <vector>
#include <memory>
#include <cstring>

namespace rf {

// ============================================================================
// Configuration
// ============================================================================

struct HistogramConfig {
    integer_t n_bins = 256;           // Number of bins for continuous features
    bool use_quantile_bins = true;    // Quantile (true) vs equal-width (false)
    integer_t min_samples_bin = 1;    // Minimum samples to create a bin
    bool enabled = false;             // Whether histogram binning is enabled
};

// ============================================================================
// Per-Feature Bin Information
// ============================================================================

struct FeatureBinInfo {
    integer_t n_bins;                 // Actual bins used (2 for binary, ncat for categorical, <=256 for continuous)
    real_t bin_edges[257];            // Bin edges for continuous (n_bins + 1 edges)
    bool is_categorical;              // If true, value is used directly as bin index
    bool is_binary;                   // Special case: only 2 unique values
    real_t min_val;                   // Minimum value seen
    real_t max_val;                   // Maximum value seen
    
    FeatureBinInfo() : n_bins(256), is_categorical(false), is_binary(false), 
                       min_val(0), max_val(0) {
        std::memset(bin_edges, 0, sizeof(bin_edges));
    }
};

// ============================================================================
// Histogram Data Storage
// ============================================================================

struct HistogramData {
    std::vector<uint8_t> X_binned;           // [nsample × mdim] binned data (1 byte per value)
    std::vector<FeatureBinInfo> feature_info; // [mdim] per-feature bin information
    integer_t nsample;
    integer_t mdim;
    bool is_valid;
    
    HistogramData() : nsample(0), mdim(0), is_valid(false) {}
    
    void allocate(integer_t n, integer_t p) {
        nsample = n;
        mdim = p;
        X_binned.resize(n * p);
        feature_info.resize(p);
        is_valid = true;
    }
    
    void clear() {
        X_binned.clear();
        feature_info.clear();
        nsample = 0;
        mdim = 0;
        is_valid = false;
    }
    
    // Get binned value for sample i, feature f
    uint8_t get_bin(integer_t sample, integer_t feature) const {
        return X_binned[sample * mdim + feature];
    }
};

// ============================================================================
// Node Histogram for Split Finding
// ============================================================================

// Classification histogram: count per (bin, class)
struct ClassificationHistogram {
    std::vector<real_t> counts;       // [n_bins × nclass] weighted counts
    integer_t n_bins;
    integer_t nclass;
    
    ClassificationHistogram() : n_bins(0), nclass(0) {}
    
    void allocate(integer_t bins, integer_t classes) {
        n_bins = bins;
        nclass = classes;
        counts.resize(bins * classes, 0.0f);
    }
    
    void clear() {
        std::fill(counts.begin(), counts.end(), 0.0f);
    }
    
    real_t& at(integer_t bin, integer_t cls) {
        return counts[bin * nclass + cls];
    }
    
    real_t at(integer_t bin, integer_t cls) const {
        return counts[bin * nclass + cls];
    }
    
    // Get total count for a bin across all classes
    real_t bin_total(integer_t bin) const {
        real_t total = 0;
        for (integer_t c = 0; c < nclass; c++) {
            total += counts[bin * nclass + c];
        }
        return total;
    }
};

// Regression histogram: sum, sum_sq, count per bin
struct RegressionHistogram {
    std::vector<real_t> sum_y;        // [n_bins] sum of y values
    std::vector<real_t> sum_y2;       // [n_bins] sum of y^2 values  
    std::vector<real_t> count;        // [n_bins] weighted count
    integer_t n_bins;
    
    RegressionHistogram() : n_bins(0) {}
    
    void allocate(integer_t bins) {
        n_bins = bins;
        sum_y.resize(bins, 0.0f);
        sum_y2.resize(bins, 0.0f);
        count.resize(bins, 0.0f);
    }
    
    void clear() {
        std::fill(sum_y.begin(), sum_y.end(), 0.0f);
        std::fill(sum_y2.begin(), sum_y2.end(), 0.0f);
        std::fill(count.begin(), count.end(), 0.0f);
    }
};

// ============================================================================
// CPU Functions (declared here, implemented in rf_histogram.cpp)
// ============================================================================

// Compute bin edges using quantiles (robust to unnormalized/skewed data)
void compute_bin_edges_quantile(
    const real_t* X,                  // [nsample × mdim] input data
    integer_t nsample,
    integer_t mdim,
    const integer_t* cat,             // [mdim] category count per feature (1 = continuous)
    integer_t n_bins,                 // Target number of bins for continuous
    FeatureBinInfo* out_info          // [mdim] output bin info
);

// Convert raw data to binned representation
void bin_data_cpu(
    const real_t* X,                  // [nsample × mdim] input data
    integer_t nsample,
    integer_t mdim,
    const FeatureBinInfo* info,       // [mdim] bin info
    uint8_t* X_binned                 // [nsample × mdim] output binned data
);

// Find bin index for a value using binary search
uint8_t find_bin(real_t value, const real_t* edges, integer_t n_bins);

// Build classification histogram for a feature at a node
void build_classification_histogram_cpu(
    const uint8_t* X_binned,          // [nsample × mdim] binned data
    const integer_t* y,               // [nsample] class labels
    const real_t* weights,            // [nsample] sample weights (nullptr for NC)
    const integer_t* node_samples,    // Indices of samples in current node
    integer_t node_count,             // Number of samples in node
    integer_t feature,                // Feature to build histogram for
    integer_t mdim,                   // Total features
    const FeatureBinInfo& info,       // Bin info for this feature
    integer_t nclass,                 // Number of classes
    ClassificationHistogram& hist     // Output histogram
);

// Build regression histogram for a feature at a node
void build_regression_histogram_cpu(
    const uint8_t* X_binned,          // [nsample × mdim] binned data
    const real_t* y_reg,              // [nsample] regression targets
    const real_t* weights,            // [nsample] sample weights (nullptr for NC)
    const integer_t* node_samples,    // Indices of samples in current node
    integer_t node_count,             // Number of samples in node
    integer_t feature,                // Feature to build histogram for
    integer_t mdim,                   // Total features
    const FeatureBinInfo& info,       // Bin info for this feature
    RegressionHistogram& hist         // Output histogram
);

// Find best split from classification histogram
void find_best_split_classification_histogram(
    const ClassificationHistogram& hist,
    const FeatureBinInfo& info,
    integer_t nclass,
    real_t* best_crit,                // Output: best criterion value
    integer_t* best_bin,              // Output: best bin index to split at
    real_t* best_split_value          // Output: actual split value (from bin edge)
);

// Find best split from regression histogram  
void find_best_split_regression_histogram(
    const RegressionHistogram& hist,
    const FeatureBinInfo& info,
    real_t* best_crit,                // Output: best criterion value (MSE)
    integer_t* best_bin,              // Output: best bin index to split at
    real_t* best_split_value          // Output: actual split value
);

// ============================================================================
// Sparse Data Histogram Functions
// ============================================================================

// Build histogram for sparse data (handles implicit zeros)
void build_classification_histogram_sparse_cpu(
    const real_t* sparse_values,      // CSR values (binned)
    const integer_t* sparse_indices,  // CSR column indices
    const integer_t* sparse_indptr,   // CSR row pointers
    const integer_t* y,               // [nsample] class labels
    const real_t* weights,            // [nsample] sample weights
    const integer_t* node_samples,    // Samples in current node
    integer_t node_count,
    integer_t feature,
    const FeatureBinInfo& info,
    integer_t nclass,
    ClassificationHistogram& hist
);

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
);

} // namespace rf

#endif // RF_HISTOGRAM_HPP

