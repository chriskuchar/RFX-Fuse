#include "rf_histogram.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace rf {
namespace cuda {

// ============================================================================
// Device function: Find bin for a value
// ============================================================================

__device__ uint8_t gpu_find_bin(real_t value, const real_t* edges, integer_t n_bins) {
    if (value <= edges[0]) return 0;
    if (value >= edges[n_bins]) return static_cast<uint8_t>(n_bins - 1);
    
    // Binary search
    integer_t lo = 0, hi = n_bins;
    while (lo < hi) {
        integer_t mid = (lo + hi) / 2;
        if (value < edges[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return static_cast<uint8_t>(max(0, static_cast<int>(lo) - 1));
}

// ============================================================================
// Kernel: Bin data (convert X to X_binned)
// ============================================================================

__global__ void gpu_bin_data_kernel(
    const real_t* X,
    integer_t nsample,
    integer_t mdim,
    const integer_t* n_bins_per_feature,      // [mdim]
    const integer_t* is_categorical,          // [mdim] 
    const integer_t* is_binary,               // [mdim]
    const real_t* all_bin_edges,              // [mdim × 257] flattened
    uint8_t* X_binned
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    integer_t total = nsample * mdim;
    
    if (idx >= total) return;
    
    integer_t sample = idx / mdim;
    integer_t feature = idx % mdim;
    real_t val = X[idx];
    
    integer_t n_bins = n_bins_per_feature[feature];
    const real_t* edges = all_bin_edges + feature * 257;
    
    uint8_t bin;
    if (is_categorical[feature]) {
        // Categorical: use value directly
        integer_t cat_val = static_cast<integer_t>(val + 0.5f);
        cat_val = max(0, min(cat_val, n_bins - 1));
        bin = static_cast<uint8_t>(cat_val);
    } else if (is_binary[feature]) {
        // Binary: simple threshold at edge[1]
        bin = (val > edges[1]) ? 1 : 0;
    } else {
        // Continuous: binary search for bin
        bin = gpu_find_bin(val, edges, n_bins);
    }
    
    X_binned[idx] = bin;
}

// ============================================================================
// Kernel: Build Classification Histogram
// Uses shared memory for fast accumulation, then writes to global
// ============================================================================

__global__ void gpu_build_classification_histogram_kernel(
    const uint8_t* X_binned,
    const integer_t* y,
    const real_t* weights,            // nullptr for NC (non-casewise)
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    integer_t nclass,
    real_t* hist_out                  // [n_bins × nclass] output
) {
    // Shared memory histogram - max 256 bins × 32 classes = 8KB
    extern __shared__ real_t shared_hist[];
    
    integer_t hist_size = n_bins * nclass;
    
    // Initialize shared memory to zero (all threads participate)
    for (integer_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        shared_hist[i] = 0.0f;
    }
    __syncthreads();  // SAFE: unconditional, all threads hit this
    
    // Each thread processes multiple samples
    for (integer_t i = threadIdx.x; i < node_count; i += blockDim.x) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        integer_t c = y[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins && c < nclass) {
            ::atomicAdd(&shared_hist[bin * nclass + c], w);
        }
    }
    // No sync needed here - atomicAdd handles concurrency
    
    __syncthreads();  // SAFE: unconditional, wait for all accumulation
    
    // Write shared histogram to global memory
    for (integer_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        hist_out[i] = shared_hist[i];
    }
}

// ============================================================================
// Kernel: Build Regression Histogram
// ============================================================================

__global__ void gpu_build_regression_histogram_kernel(
    const uint8_t* X_binned,
    const real_t* y_reg,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    real_t* sum_y_out,                // [n_bins]
    real_t* sum_y2_out,               // [n_bins]
    real_t* count_out                 // [n_bins]
) {
    // Shared memory for 3 arrays
    extern __shared__ real_t shared_mem[];
    real_t* s_sum = shared_mem;
    real_t* s_sum2 = shared_mem + n_bins;
    real_t* s_count = shared_mem + 2 * n_bins;
    
    // Initialize shared memory
    for (integer_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
        s_sum[i] = 0.0f;
        s_sum2[i] = 0.0f;
        s_count[i] = 0.0f;
    }
    __syncthreads();  // SAFE: unconditional
    
    // Accumulate
    for (integer_t i = threadIdx.x; i < node_count; i += blockDim.x) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        real_t val = y_reg[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins) {
            ::atomicAdd(&s_sum[bin], val * w);
            ::atomicAdd(&s_sum2[bin], val * val * w);
            ::atomicAdd(&s_count[bin], w);
        }
    }
    
    __syncthreads();  // SAFE: unconditional
    
    // Write out
    for (integer_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
        sum_y_out[i] = s_sum[i];
        sum_y2_out[i] = s_sum2[i];
        count_out[i] = s_count[i];
    }
}

// ============================================================================
// Kernel: Find best split from classification histogram (single thread)
// ============================================================================

__global__ void gpu_find_best_split_classification_kernel(
    const real_t* hist,               // [n_bins × nclass]
    integer_t n_bins,
    integer_t nclass,
    const real_t* bin_edges,          // [257] for this feature
    bool is_categorical,
    real_t* best_crit_out,
    integer_t* best_bin_out,
    real_t* best_split_value_out
) {
    // Single thread does the scan (256 bins is small)
    if (threadIdx.x != 0) return;
    
    // Allocate local arrays (on stack, not shared - single thread)
    real_t left_counts[32];   // Max 32 classes
    real_t right_counts[32];
    
    for (integer_t c = 0; c < nclass && c < 32; c++) {
        left_counts[c] = 0.0f;
        right_counts[c] = 0.0f;
    }
    
    real_t total_left = 0.0f;
    real_t total_right = 0.0f;
    
    // Initialize right with all
    for (integer_t b = 0; b < n_bins; b++) {
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            right_counts[c] += hist[b * nclass + c];
            total_right += hist[b * nclass + c];
        }
    }
    
    real_t best_gini = 1e20f;
    integer_t best_b = 0;
    
    // Scan
    for (integer_t b = 0; b < n_bins - 1; b++) {
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            real_t count = hist[b * nclass + c];
            left_counts[c] += count;
            right_counts[c] -= count;
            total_left += count;
            total_right -= count;
        }
        
        if (total_left < 1.0f || total_right < 1.0f) continue;
        
        real_t gini_left = 1.0f;
        real_t gini_right = 1.0f;
        
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            real_t p_left = left_counts[c] / total_left;
            real_t p_right = right_counts[c] / total_right;
            gini_left -= p_left * p_left;
            gini_right -= p_right * p_right;
        }
        
        real_t total = total_left + total_right;
        real_t gini = (total_left * gini_left + total_right * gini_right) / total;
        
        if (gini < best_gini) {
            best_gini = gini;
            best_b = b;
        }
    }
    
    *best_crit_out = best_gini;
    *best_bin_out = best_b;
    
    if (is_categorical) {
        *best_split_value_out = static_cast<real_t>(best_b);
    } else {
        // FIX: Split point is at upper edge of best_bin (boundary between bin b and bin b+1)
        *best_split_value_out = bin_edges[best_b + 1];
    }
}

// ============================================================================
// Kernel: Find best split from regression histogram
// ============================================================================

__global__ void gpu_find_best_split_regression_kernel(
    const real_t* sum_y,
    const real_t* sum_y2,
    const real_t* count,
    integer_t n_bins,
    const real_t* bin_edges,
    bool is_categorical,
    real_t* best_crit_out,
    integer_t* best_bin_out,
    real_t* best_split_value_out
) {
    if (threadIdx.x != 0) return;
    
    // Compute totals
    real_t sum_left = 0.0f, sum2_left = 0.0f, cnt_left = 0.0f;
    real_t sum_right = 0.0f, sum2_right = 0.0f, cnt_right = 0.0f;
    
    for (integer_t b = 0; b < n_bins; b++) {
        sum_right += sum_y[b];
        sum2_right += sum_y2[b];
        cnt_right += count[b];
    }
    
    real_t best_mse = 1e20f;
    integer_t best_b = 0;
    
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
        
        real_t mse_left = sum2_left / cnt_left - mean_left * mean_left;
        real_t mse_right = sum2_right / cnt_right - mean_right * mean_right;
        
        real_t total = cnt_left + cnt_right;
        real_t mse = (cnt_left * mse_left + cnt_right * mse_right) / total;
        
        if (mse < best_mse) {
            best_mse = mse;
            best_b = b;
        }
    }
    
    *best_crit_out = best_mse;
    *best_bin_out = best_b;
    
    if (is_categorical) {
        *best_split_value_out = static_cast<real_t>(best_b);
    } else {
        // FIX: Split point is at upper edge of best_bin (boundary between bin b and bin b+1)
        *best_split_value_out = bin_edges[best_b + 1];
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void gpu_bin_data(
    const real_t* d_X,
    integer_t nsample,
    integer_t mdim,
    const FeatureBinInfo* h_info,     // Host-side feature info
    uint8_t* d_X_binned,
    cudaStream_t stream
) {
    // Prepare device arrays for feature info
    integer_t* d_n_bins;
    integer_t* d_is_categorical;
    integer_t* d_is_binary;
    real_t* d_all_edges;
    
    cudaMalloc(&d_n_bins, mdim * sizeof(integer_t));
    cudaMalloc(&d_is_categorical, mdim * sizeof(integer_t));
    cudaMalloc(&d_is_binary, mdim * sizeof(integer_t));
    cudaMalloc(&d_all_edges, mdim * 257 * sizeof(real_t));
    
    // Copy feature info to device
    std::vector<integer_t> h_n_bins(mdim);
    std::vector<integer_t> h_is_cat(mdim);
    std::vector<integer_t> h_is_bin(mdim);
    std::vector<real_t> h_edges(mdim * 257);
    
    for (integer_t f = 0; f < mdim; f++) {
        h_n_bins[f] = h_info[f].n_bins;
        h_is_cat[f] = h_info[f].is_categorical ? 1 : 0;
        h_is_bin[f] = h_info[f].is_binary ? 1 : 0;
        for (integer_t e = 0; e < 257; e++) {
            h_edges[f * 257 + e] = h_info[f].bin_edges[e];
        }
    }
    
    cudaMemcpyAsync(d_n_bins, h_n_bins.data(), mdim * sizeof(integer_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_is_categorical, h_is_cat.data(), mdim * sizeof(integer_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_is_binary, h_is_bin.data(), mdim * sizeof(integer_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_all_edges, h_edges.data(), mdim * 257 * sizeof(real_t), cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    integer_t total = nsample * mdim;
    integer_t block_size = 256;
    integer_t grid_size = (total + block_size - 1) / block_size;
    
    gpu_bin_data_kernel<<<grid_size, block_size, 0, stream>>>(
        d_X, nsample, mdim,
        d_n_bins, d_is_categorical, d_is_binary, d_all_edges,
        d_X_binned
    );
    
    // Cleanup
    cudaStreamSynchronize(stream);
    cudaFree(d_n_bins);
    cudaFree(d_is_categorical);
    cudaFree(d_is_binary);
    cudaFree(d_all_edges);
}

void gpu_build_classification_histogram(
    const uint8_t* d_X_binned,
    const integer_t* d_y,
    const real_t* d_weights,
    const integer_t* d_node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    integer_t nclass,
    real_t* d_hist_out,
    cudaStream_t stream
) {
    integer_t block_size = 256;
    integer_t shared_mem = n_bins * nclass * sizeof(real_t);
    
    // Limit shared memory usage
    if (shared_mem > 48 * 1024) {
        // Fall back to smaller histogram or error
        printf("Warning: Histogram too large for shared memory\n");
        return;
    }
    
    gpu_build_classification_histogram_kernel<<<1, block_size, shared_mem, stream>>>(
        d_X_binned, d_y, d_weights, d_node_samples,
        node_count, feature, mdim, n_bins, nclass, d_hist_out
    );
}

void gpu_build_regression_histogram(
    const uint8_t* d_X_binned,
    const real_t* d_y_reg,
    const real_t* d_weights,
    const integer_t* d_node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    real_t* d_sum_y_out,
    real_t* d_sum_y2_out,
    real_t* d_count_out,
    cudaStream_t stream
) {
    integer_t block_size = 256;
    integer_t shared_mem = 3 * n_bins * sizeof(real_t);
    
    gpu_build_regression_histogram_kernel<<<1, block_size, shared_mem, stream>>>(
        d_X_binned, d_y_reg, d_weights, d_node_samples,
        node_count, feature, mdim, n_bins,
        d_sum_y_out, d_sum_y2_out, d_count_out
    );
}

} // namespace cuda
} // namespace rf

