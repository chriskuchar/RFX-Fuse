// Thrust/CUB compatibility for CUDA 12.8
// Must define half types BEFORE cub/thrust includes
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Now include thrust
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "rf_histogram_gpu.cuh"
#include <device_launch_parameters.h>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace rf {
namespace cuda {

// ============================================================================
// Device helper: Get bin for sample/feature
// ============================================================================

__device__ uint8_t get_bin_device(const uint8_t* X_binned, integer_t sample, integer_t feature, integer_t mdim) {
    return X_binned[sample * mdim + feature];
}

// ============================================================================
// Kernel: Build classification histogram for a node
// ============================================================================

__global__ void gpu_histogram_build_classification_kernel(
    const uint8_t* X_binned,
    const integer_t* y,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    integer_t nclass,
    real_t* hist_out
) {
    // Shared memory for histogram
    extern __shared__ real_t s_hist[];
    
    integer_t hist_size = n_bins * nclass;
    
    // Initialize shared histogram to zero
    for (integer_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        s_hist[i] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes multiple samples
    for (integer_t i = threadIdx.x; i < node_count; i += blockDim.x) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        integer_t c = y[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins && c >= 0 && c < nclass) {
            ::atomicAdd(&s_hist[bin * nclass + c], w);
        }
    }
    __syncthreads();
    
    // Copy to global memory
    for (integer_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        hist_out[i] = s_hist[i];
    }
}

// ============================================================================
// Kernel: Find best split from classification histogram
// Single thread scans the histogram
// ============================================================================

__global__ void gpu_histogram_find_split_classification_kernel(
    const real_t* hist,
    integer_t n_bins,
    integer_t nclass,
    const real_t* bin_edges,
    bool is_categorical,
    real_t* out_best_crit,
    real_t* out_best_split
) {
    if (threadIdx.x != 0) return;
    
    // Left/right class counts (small arrays on stack)
    real_t left_counts[32];
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
    
    real_t best_gini = 1.0e20f;
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
        
        // Compute Gini impurity
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
            best_bin = b;
        }
    }
    
    *out_best_crit = best_gini;
    
    if (is_categorical) {
        *out_best_split = static_cast<real_t>(best_bin);
    } else {
        // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
        *out_best_split = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0f;
    }
}

// ============================================================================
// Kernel: Build regression histogram for a node
// ============================================================================

__global__ void gpu_histogram_build_regression_kernel(
    const uint8_t* X_binned,
    const real_t* y_reg,
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    integer_t n_bins,
    real_t* sum_out,
    real_t* sum2_out,
    real_t* count_out
) {
    // Shared memory: 3 arrays
    extern __shared__ real_t s_mem[];
    real_t* s_sum = s_mem;
    real_t* s_sum2 = s_mem + n_bins;
    real_t* s_count = s_mem + 2 * n_bins;
    
    // Initialize
    for (integer_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
        s_sum[i] = 0.0f;
        s_sum2[i] = 0.0f;
        s_count[i] = 0.0f;
    }
    __syncthreads();
    
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
    __syncthreads();
    
    // Copy to global
    for (integer_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
        sum_out[i] = s_sum[i];
        sum2_out[i] = s_sum2[i];
        count_out[i] = s_count[i];
    }
}

// ============================================================================
// Kernel: Find best split from regression histogram
// ============================================================================

__global__ void gpu_histogram_find_split_regression_kernel(
    const real_t* sum_y,
    const real_t* sum_y2,
    const real_t* count,
    integer_t n_bins,
    const real_t* bin_edges,
    bool is_categorical,
    real_t* out_best_crit,
    real_t* out_best_split
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
        
        real_t mse_left = sum2_left / cnt_left - mean_left * mean_left;
        real_t mse_right = sum2_right / cnt_right - mean_right * mean_right;
        
        // Weighted MSE
        real_t total = cnt_left + cnt_right;
        real_t mse = (cnt_left * mse_left + cnt_right * mse_right) / total;
        
        if (mse < best_mse) {
            best_mse = mse;
            best_bin = b;
        }
    }
    
    *out_best_crit = best_mse;
    
    if (is_categorical) {
        *out_best_split = static_cast<real_t>(best_bin);
    } else {
        // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
        *out_best_split = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0f;
    }
}

// ============================================================================
// Host wrapper: Find best split for classification
// ============================================================================

void gpu_histogram_find_best_split_classification(
    const GPUHistogramData& hist_data,
    const integer_t* d_node_samples,
    integer_t node_count,
    integer_t feature,
    const integer_t* d_y,
    const real_t* d_weights,
    integer_t nclass,
    real_t& best_crit,
    real_t& best_split_value,
    cudaStream_t stream
) {
    if (!hist_data.is_allocated || node_count < 2) {
        best_crit = 1.0e20f;
        best_split_value = 0.0f;
        return;
    }
    
    // Get feature info (from device memory - need to read)
    integer_t h_n_bins;
    integer_t h_is_cat;
    cudaMemcpy(&h_n_bins, hist_data.d_n_bins + feature, sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_is_cat, hist_data.d_is_categorical + feature, sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    if (h_n_bins < 2) {
        best_crit = 1.0e20f;
        best_split_value = 0.0f;
        return;
    }
    
    // Build histogram
    integer_t block_size = 256;
    integer_t shared_mem = h_n_bins * nclass * sizeof(real_t);
    
    if (shared_mem > 48 * 1024) {
        // Histogram too large for shared memory - fall back
        best_crit = 1.0e20f;
        best_split_value = 0.0f;
        return;
    }
    
    gpu_histogram_build_classification_kernel<<<1, block_size, shared_mem, stream>>>(
        hist_data.d_X_binned,
        d_y,
        d_weights,
        d_node_samples,
        node_count,
        feature,
        hist_data.mdim,
        h_n_bins,
        nclass,
        hist_data.d_hist_class
    );
    
    // Find best split
    real_t* d_crit;
    real_t* d_split;
    cudaMalloc(&d_crit, sizeof(real_t));
    cudaMalloc(&d_split, sizeof(real_t));
    
    gpu_histogram_find_split_classification_kernel<<<1, 32, 0, stream>>>(
        hist_data.d_hist_class,
        h_n_bins,
        nclass,
        hist_data.d_bin_edges + feature * 257,
        h_is_cat != 0,
        d_crit,
        d_split
    );
    
    // Copy results back
    cudaMemcpy(&best_crit, d_crit, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&best_split_value, d_split, sizeof(real_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_crit);
    cudaFree(d_split);
}

// ============================================================================
// Host wrapper: Find best split for regression
// ============================================================================

void gpu_histogram_find_best_split_regression(
    const GPUHistogramData& hist_data,
    const integer_t* d_node_samples,
    integer_t node_count,
    integer_t feature,
    const real_t* d_y_reg,
    const real_t* d_weights,
    real_t& best_crit,
    real_t& best_split_value,
    cudaStream_t stream
) {
    if (!hist_data.is_allocated || node_count < 2) {
        best_crit = 1.0e20f;
        best_split_value = 0.0f;
        return;
    }
    
    // Get feature info
    integer_t h_n_bins;
    integer_t h_is_cat;
    cudaMemcpy(&h_n_bins, hist_data.d_n_bins + feature, sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_is_cat, hist_data.d_is_categorical + feature, sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    if (h_n_bins < 2) {
        best_crit = 1.0e20f;
        best_split_value = 0.0f;
        return;
    }
    
    // Build histogram
    integer_t block_size = 256;
    integer_t shared_mem = 3 * h_n_bins * sizeof(real_t);
    
    if (shared_mem > 48 * 1024) {
        best_crit = 1.0e20f;
        best_split_value = 0.0f;
        return;
    }
    
    gpu_histogram_build_regression_kernel<<<1, block_size, shared_mem, stream>>>(
        hist_data.d_X_binned,
        d_y_reg,
        d_weights,
        d_node_samples,
        node_count,
        feature,
        hist_data.mdim,
        h_n_bins,
        hist_data.d_hist_sum,
        hist_data.d_hist_sum2,
        hist_data.d_hist_count
    );
    
    // Find best split
    real_t* d_crit;
    real_t* d_split;
    cudaMalloc(&d_crit, sizeof(real_t));
    cudaMalloc(&d_split, sizeof(real_t));
    
    gpu_histogram_find_split_regression_kernel<<<1, 32, 0, stream>>>(
        hist_data.d_hist_sum,
        hist_data.d_hist_sum2,
        hist_data.d_hist_count,
        h_n_bins,
        hist_data.d_bin_edges + feature * 257,
        h_is_cat != 0,
        d_crit,
        d_split
    );
    
    // Copy results back
    cudaMemcpy(&best_crit, d_crit, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&best_split_value, d_split, sizeof(real_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_crit);
    cudaFree(d_split);
}

// ============================================================================
// NEW: GPU Dense Histogram - Compute bin edges using unique values
// Uses thrust for GPU-accelerated sorting (no CPU roundtrip!)
// ============================================================================

// Kernel: Extract a single column from row-major matrix
__global__ void extract_column_kernel(
    const real_t* X,           // [nsample × mdim] row-major
    real_t* col_out,           // [nsample] output column
    integer_t nsample,
    integer_t mdim,
    integer_t col_idx
) {
    integer_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nsample) {
        col_out[i] = X[i * mdim + col_idx];
    }
}

// Kernel: Compute bin edges from sorted unique values (runs on GPU)
__global__ void compute_bin_edges_kernel(
    const real_t* unique_values,  // Sorted unique values
    integer_t n_unique,
    integer_t max_bins,
    real_t* edges,                // Output: [max_bins + 1] edges
    integer_t* n_bins_out         // Output: actual number of bins
) {
    // Single thread computes edges (fast for small arrays)
    if (threadIdx.x != 0) return;
    
    if (n_unique == 0) {
        *n_bins_out = 1;
        for (integer_t b = 0; b <= max_bins; b++) {
            edges[b] = static_cast<real_t>(b);
        }
    } else if (n_unique <= 2) {
        // BINARY OR NEAR-BINARY FEATURE
        *n_bins_out = 2;
        real_t min_val = unique_values[0];
        real_t max_val = unique_values[n_unique - 1];
        edges[0] = min_val - 1.0f;
        edges[1] = (min_val + max_val) / 2.0f;
        edges[2] = max_val + 1.0f;
        for (integer_t b = 3; b <= max_bins; b++) {
            edges[b] = edges[2];
        }
    } else {
        // CONTINUOUS FEATURE: Quantile binning
        integer_t actual_bins = min(max_bins, n_unique);
        *n_bins_out = actual_bins;
        
        edges[0] = unique_values[0] - 1.0f;
        for (integer_t b = 1; b <= actual_bins; b++) {
            integer_t quantile_idx = (b * n_unique) / actual_bins;
            if (quantile_idx >= n_unique) quantile_idx = n_unique - 1;
            edges[b] = unique_values[quantile_idx];
        }
        for (integer_t b = actual_bins + 1; b <= max_bins; b++) {
            edges[b] = edges[actual_bins];
        }
        
        // Ensure strictly increasing
        for (integer_t b = 1; b <= actual_bins; b++) {
            if (edges[b] <= edges[b-1]) {
                edges[b] = edges[b-1] + 1e-7f;
            }
        }
    }
}

void gpu_compute_dense_bin_edges(
    const real_t* d_X,
    integer_t nsample,
    integer_t mdim,
    integer_t max_bins,
    DenseHistogramData& hist_data,
    cudaStream_t stream
) {
    // Allocate histogram data
    hist_data.allocate(nsample, mdim, max_bins);
    
    // Allocate temporary GPU buffer for column extraction and sorting
    real_t* d_col_buffer = nullptr;
    cudaMalloc(&d_col_buffer, nsample * sizeof(real_t));
    
    // Temporary storage for unique values (worst case: all unique)
    real_t* d_unique_buffer = nullptr;
    cudaMalloc(&d_unique_buffer, nsample * sizeof(real_t));
    
    // Temporary for n_unique count
    integer_t* d_n_unique = nullptr;
    cudaMalloc(&d_n_unique, sizeof(integer_t));
    
    // Process each feature on GPU
    integer_t block_size = 256;
    integer_t grid_size = (nsample + block_size - 1) / block_size;
    
    for (integer_t col = 0; col < mdim; col++) {
        // 1. Extract column to contiguous buffer (GPU kernel)
        extract_column_kernel<<<grid_size, block_size, 0, stream>>>(
            d_X, d_col_buffer, nsample, mdim, col
        );
        
        // 2. Sort the column using thrust (GPU)
        thrust::device_ptr<real_t> d_ptr(d_col_buffer);
        thrust::sort(d_ptr, d_ptr + nsample);
        
        // 3. Find unique values using thrust (GPU)
        thrust::device_ptr<real_t> unique_end = thrust::unique(d_ptr, d_ptr + nsample);
        integer_t n_unique = static_cast<integer_t>(unique_end - d_ptr);
        
        // 4. Compute bin edges (GPU kernel)
        real_t* d_edges = hist_data.d_bin_edges_all + col * (max_bins + 1);
        integer_t* d_n_bins = hist_data.d_n_bins_per_feature + col;
        
        compute_bin_edges_kernel<<<1, 1, 0, stream>>>(
            d_col_buffer, n_unique, max_bins, d_edges, d_n_bins
        );
    }
    
    // Sync to ensure all edges are computed before binning
    cudaStreamSynchronize(stream);
    
    // Cleanup temporary buffers
    cudaFree(d_col_buffer);
    cudaFree(d_unique_buffer);
    cudaFree(d_n_unique);
}

// Kernel: Bin dense data values
__global__ void gpu_bin_dense_data_kernel(
    const real_t* X,                   // [nsample × mdim]
    integer_t nsample,
    integer_t mdim,
    const real_t* bin_edges_all,       // [mdim × (max_bins+1)]
    const integer_t* n_bins_per_feature,
    integer_t max_bins,
    uint8_t* X_binned                  // [nsample × mdim] output
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    integer_t total = nsample * mdim;
    
    if (idx >= total) return;
    
    integer_t sample = idx / mdim;
    integer_t feature = idx % mdim;
    real_t val = X[sample * mdim + feature];
    
    // Get bin edges for this feature
    const real_t* edges = bin_edges_all + feature * (max_bins + 1);
    integer_t n_bins = n_bins_per_feature[feature];
    
    // Binary search for bin - MATCHES GPU SPARSE EXACTLY
    // Find smallest bin where val <= edges[bin + 1]
    integer_t lo = 0, hi = n_bins;
    while (lo < hi) {
        integer_t mid = (lo + hi) / 2;
        if (val <= edges[mid + 1]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    // Clamp to valid range
    if (lo >= n_bins) lo = n_bins - 1;

    X_binned[idx] = static_cast<uint8_t>(lo);
}

void gpu_bin_dense_data(
    const real_t* d_X,
    DenseHistogramData& hist_data,
    cudaStream_t stream
) {
    integer_t total = hist_data.nsample * hist_data.mdim;
    integer_t block_size = 256;
    integer_t grid_size = (total + block_size - 1) / block_size;
    
    gpu_bin_dense_data_kernel<<<grid_size, block_size, 0, stream>>>(
        d_X,
        hist_data.nsample,
        hist_data.mdim,
        hist_data.d_bin_edges_all,
        hist_data.d_n_bins_per_feature,
        hist_data.max_bins,
        hist_data.d_X_binned
    );
}

// Cleanup function for use from non-CUDA code (avoids incomplete type issues)
void gpu_histogram_cleanup(void* gpu_hist_ptr) {
    if (gpu_hist_ptr != nullptr) {
        GPUHistogramData* hist = static_cast<GPUHistogramData*>(gpu_hist_ptr);
        delete hist;
    }
}

} // namespace cuda
} // namespace rf

