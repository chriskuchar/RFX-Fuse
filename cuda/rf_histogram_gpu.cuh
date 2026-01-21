#ifndef RF_HISTOGRAM_GPU_CUH
#define RF_HISTOGRAM_GPU_CUH

#include "rf_types.hpp"
#include "rf_histogram.hpp"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

// ============================================================================
// GPU Histogram Data Structure
// Manages GPU memory for histogram binning
// ============================================================================

struct GPUHistogramData {
    // Device pointers (allocated on GPU)
    uint8_t* d_X_binned;              // [nsample × mdim] binned data
    real_t* d_bin_edges;              // [mdim × 257] all bin edges flattened
    integer_t* d_n_bins;              // [mdim] number of bins per feature
    integer_t* d_is_categorical;      // [mdim] categorical flags
    integer_t* d_is_binary;           // [mdim] binary flags
    
    // Temporary workspace for split finding
    real_t* d_hist_class;             // [max_bins × nclass] for classification histogram
    real_t* d_hist_sum;               // [max_bins] for regression sum
    real_t* d_hist_sum2;              // [max_bins] for regression sum^2
    real_t* d_hist_count;             // [max_bins] for regression count
    
    // Dimensions
    integer_t nsample;
    integer_t mdim;
    integer_t max_bins;
    integer_t nclass;
    
    // State
    bool is_allocated;
    cudaStream_t stream;
    
    GPUHistogramData() : 
        d_X_binned(nullptr), d_bin_edges(nullptr), d_n_bins(nullptr),
        d_is_categorical(nullptr), d_is_binary(nullptr),
        d_hist_class(nullptr), d_hist_sum(nullptr), d_hist_sum2(nullptr), d_hist_count(nullptr),
        nsample(0), mdim(0), max_bins(256), nclass(2),
        is_allocated(false), stream(0) {}
    
    ~GPUHistogramData() {
        free();
    }
    
    // Allocate GPU memory
    bool allocate(integer_t n, integer_t p, integer_t nc, integer_t bins = 256) {
        if (is_allocated) free();
        
        nsample = n;
        mdim = p;
        nclass = nc;
        max_bins = bins;
        
        cudaError_t err;
        
        // Allocate binned data
        err = cudaMalloc(&d_X_binned, nsample * mdim * sizeof(uint8_t));
        if (err != cudaSuccess) { free(); return false; }
        
        // Allocate bin info arrays
        err = cudaMalloc(&d_bin_edges, mdim * 257 * sizeof(real_t));
        if (err != cudaSuccess) { free(); return false; }
        
        err = cudaMalloc(&d_n_bins, mdim * sizeof(integer_t));
        if (err != cudaSuccess) { free(); return false; }
        
        err = cudaMalloc(&d_is_categorical, mdim * sizeof(integer_t));
        if (err != cudaSuccess) { free(); return false; }
        
        err = cudaMalloc(&d_is_binary, mdim * sizeof(integer_t));
        if (err != cudaSuccess) { free(); return false; }
        
        // Allocate histogram workspace (enough for max_bins)
        err = cudaMalloc(&d_hist_class, max_bins * nclass * sizeof(real_t));
        if (err != cudaSuccess) { free(); return false; }
        
        err = cudaMalloc(&d_hist_sum, max_bins * sizeof(real_t));
        if (err != cudaSuccess) { free(); return false; }
        
        err = cudaMalloc(&d_hist_sum2, max_bins * sizeof(real_t));
        if (err != cudaSuccess) { free(); return false; }
        
        err = cudaMalloc(&d_hist_count, max_bins * sizeof(real_t));
        if (err != cudaSuccess) { free(); return false; }
        
        is_allocated = true;
        return true;
    }
    
    // Transfer histogram data from CPU to GPU
    bool transfer_from_cpu(const HistogramData& cpu_data, cudaStream_t s = 0) {
        if (!is_allocated) return false;
        if (!cpu_data.is_valid) return false;
        if (cpu_data.nsample != nsample || cpu_data.mdim != mdim) return false;
        
        stream = s;
        cudaError_t err;
        
        // Transfer binned data
        err = cudaMemcpyAsync(d_X_binned, cpu_data.X_binned.data(), 
                             nsample * mdim * sizeof(uint8_t),
                             cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return false;
        
        // Prepare and transfer bin info
        std::vector<real_t> h_edges(mdim * 257, 0.0f);
        std::vector<integer_t> h_n_bins(mdim);
        std::vector<integer_t> h_is_cat(mdim);
        std::vector<integer_t> h_is_bin(mdim);
        
        for (integer_t f = 0; f < mdim; f++) {
            h_n_bins[f] = cpu_data.feature_info[f].n_bins;
            h_is_cat[f] = cpu_data.feature_info[f].is_categorical ? 1 : 0;
            h_is_bin[f] = cpu_data.feature_info[f].is_binary ? 1 : 0;
            for (integer_t e = 0; e < 257; e++) {
                h_edges[f * 257 + e] = cpu_data.feature_info[f].bin_edges[e];
            }
        }
        
        err = cudaMemcpyAsync(d_bin_edges, h_edges.data(), 
                             mdim * 257 * sizeof(real_t),
                             cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyAsync(d_n_bins, h_n_bins.data(),
                             mdim * sizeof(integer_t),
                             cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyAsync(d_is_categorical, h_is_cat.data(),
                             mdim * sizeof(integer_t),
                             cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyAsync(d_is_binary, h_is_bin.data(),
                             mdim * sizeof(integer_t),
                             cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return false;
        
        // Synchronize to ensure transfer completes
        cudaStreamSynchronize(stream);
        
        return true;
    }
    
    // Free GPU memory
    void free() {
        if (d_X_binned) { cudaFree(d_X_binned); d_X_binned = nullptr; }
        if (d_bin_edges) { cudaFree(d_bin_edges); d_bin_edges = nullptr; }
        if (d_n_bins) { cudaFree(d_n_bins); d_n_bins = nullptr; }
        if (d_is_categorical) { cudaFree(d_is_categorical); d_is_categorical = nullptr; }
        if (d_is_binary) { cudaFree(d_is_binary); d_is_binary = nullptr; }
        if (d_hist_class) { cudaFree(d_hist_class); d_hist_class = nullptr; }
        if (d_hist_sum) { cudaFree(d_hist_sum); d_hist_sum = nullptr; }
        if (d_hist_sum2) { cudaFree(d_hist_sum2); d_hist_sum2 = nullptr; }
        if (d_hist_count) { cudaFree(d_hist_count); d_hist_count = nullptr; }
        is_allocated = false;
    }
};

// ============================================================================
// GPU Histogram Split Finding Functions (declared here, implemented in .cu)
// ============================================================================

// Find best split using histogram for classification (single feature)
void gpu_histogram_find_best_split_classification(
    const GPUHistogramData& hist_data,
    const integer_t* d_node_samples,  // Samples in current node
    integer_t node_count,
    integer_t feature,
    const integer_t* d_y,             // Class labels
    const real_t* d_weights,          // Sample weights (nullptr for NC)
    integer_t nclass,
    real_t& best_crit,                // Output
    real_t& best_split_value,         // Output
    cudaStream_t stream = 0
);

// Find best split using histogram for regression (single feature)
void gpu_histogram_find_best_split_regression(
    const GPUHistogramData& hist_data,
    const integer_t* d_node_samples,
    integer_t node_count,
    integer_t feature,
    const real_t* d_y_reg,            // Regression targets
    const real_t* d_weights,
    real_t& best_crit,                // Output (MSE)
    real_t& best_split_value,         // Output
    cudaStream_t stream = 0
);

// ============================================================================
// NEW: GPU Dense Histogram Data (self-contained, computes its own bin edges)
// Mirrors the sparse histogram implementation for correctness
// ============================================================================

struct DenseHistogramData {
    // Device pointers
    uint8_t* d_X_binned;              // [nsample × mdim] binned data
    real_t* d_bin_edges_all;          // [mdim × (max_bins+1)] bin edges
    integer_t* d_n_bins_per_feature;  // [mdim] number of bins per feature
    
    // Dimensions
    integer_t nsample;
    integer_t mdim;
    integer_t max_bins;
    
    bool allocated;
    
    DenseHistogramData() : 
        d_X_binned(nullptr), d_bin_edges_all(nullptr), d_n_bins_per_feature(nullptr),
        nsample(0), mdim(0), max_bins(256), allocated(false) {}
    
    void allocate(integer_t n, integer_t p, integer_t bins = 256) {
        if (allocated) free();
        
        nsample = n;
        mdim = p;
        max_bins = bins;
        
        cudaMalloc(&d_X_binned, static_cast<size_t>(nsample) * mdim * sizeof(uint8_t));
        cudaMalloc(&d_bin_edges_all, static_cast<size_t>(mdim) * (max_bins + 1) * sizeof(real_t));
        cudaMalloc(&d_n_bins_per_feature, static_cast<size_t>(mdim) * sizeof(integer_t));
        
        cudaMemset(d_X_binned, 0, static_cast<size_t>(nsample) * mdim * sizeof(uint8_t));
        cudaMemset(d_n_bins_per_feature, 0, static_cast<size_t>(mdim) * sizeof(integer_t));
        
        allocated = true;
    }
    
    void free() {
        if (d_X_binned) { cudaFree(d_X_binned); d_X_binned = nullptr; }
        if (d_bin_edges_all) { cudaFree(d_bin_edges_all); d_bin_edges_all = nullptr; }
        if (d_n_bins_per_feature) { cudaFree(d_n_bins_per_feature); d_n_bins_per_feature = nullptr; }
        allocated = false;
    }
    
    ~DenseHistogramData() { free(); }
};

// Compute bin edges for dense data using unique values (like sparse)
// This ensures proper handling of sparse-like data with many repeated values
void gpu_compute_dense_bin_edges(
    const real_t* d_X,                // Dense feature matrix [nsample × mdim] on GPU
    integer_t nsample,
    integer_t mdim,
    integer_t max_bins,
    DenseHistogramData& hist_data,
    cudaStream_t stream = 0
);

// Bin dense data using computed bin edges
void gpu_bin_dense_data(
    const real_t* d_X,                // Dense feature matrix [nsample × mdim] on GPU
    DenseHistogramData& hist_data,
    cudaStream_t stream = 0
);

// Cleanup function for use from non-CUDA code (avoids incomplete type issues)
void gpu_histogram_cleanup(void* gpu_hist_ptr);

} // namespace cuda
} // namespace rf

#endif // RF_HISTOGRAM_GPU_CUH

