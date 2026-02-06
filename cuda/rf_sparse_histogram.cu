/**
 * @file rf_sparse_histogram.cu
 * @brief GPU sparse pre-sorting and histogram binning implementation
 * 
 * Provides XGBoost-like speed for sparse random forests by:
 * 1. Pre-sorting feature indices once before training
 * 2. Histogram binning for O(256) split finding
 * 
 * Memory safety:
 * - Single stream for all operations
 * - ::atomicAdd for concurrent histogram updates
 * - Bounds checking on all array accesses
 */

#include "rf_growtree_sparse.cuh"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace rf {
namespace cuda {

// ============================================================================
// SparseHistogramData implementation
// ============================================================================

void SparseHistogramData::allocate(integer_t nnz_, integer_t mdim_, integer_t max_bins_) {
    if (allocated) free();
    
    nnz = nnz_;
    mdim = mdim_;
    max_bins = max_bins_;
    
    cudaMalloc(&d_X_binned_values, nnz * sizeof(uint8_t));
    cudaMalloc(&d_bin_edges_all, mdim * (max_bins + 1) * sizeof(real_t));
    cudaMalloc(&d_n_bins_per_feature, mdim * sizeof(integer_t));
    
    cudaMemset(d_X_binned_values, 0, nnz * sizeof(uint8_t));
    cudaMemset(d_n_bins_per_feature, 0, mdim * sizeof(integer_t));
    
    allocated = true;
}

void SparseHistogramData::free() {
    if (d_X_binned_values) { cudaFree(d_X_binned_values); d_X_binned_values = nullptr; }
    if (d_bin_edges_all) { cudaFree(d_bin_edges_all); d_bin_edges_all = nullptr; }
    if (d_n_bins_per_feature) { cudaFree(d_n_bins_per_feature); d_n_bins_per_feature = nullptr; }
    allocated = false;
}

// ============================================================================
// Pre-sort kernel: Sort each column of sparse matrix
// Each block handles one feature (column)
// ============================================================================

__global__ void gpu_presort_sparse_kernel(
    CudaSparseMatrixCSR X_sparse,
    integer_t nsample,
    integer_t mdim,
    integer_t* sorted_indices  // [mdim × nsample]
) {
    integer_t feature = blockIdx.x;
    if (feature >= mdim) return;
    
    integer_t tid = threadIdx.x;
    
    // Thread 0 does the sorting for this feature
    // (Parallel sorting within a feature is complex for sparse - keep simple)
    if (tid == 0) {
        integer_t* my_indices = sorted_indices + feature * nsample;
        
        // Extract (value, index) pairs for this feature
        // Use local arrays (stack) for small datasets, or iterate for large
        const integer_t MAX_LOCAL = 4096;
        
        if (nsample <= MAX_LOCAL) {
            // Small dataset: use local arrays
            real_t local_values[MAX_LOCAL];
            integer_t local_indices[MAX_LOCAL];
            
            for (integer_t i = 0; i < nsample; i++) {
                local_values[i] = X_sparse.get(i, feature);
                local_indices[i] = i;
            }
            
            // Simple insertion sort (stable, good for partially sorted data)
            for (integer_t i = 1; i < nsample; i++) {
                real_t key_val = local_values[i];
                integer_t key_idx = local_indices[i];
                integer_t j = i - 1;
                
                while (j >= 0 && local_values[j] > key_val) {
                    local_values[j + 1] = local_values[j];
                    local_indices[j + 1] = local_indices[j];
                    j--;
                }
                local_values[j + 1] = key_val;
                local_indices[j + 1] = key_idx;
            }
            
            // Copy sorted indices to output
            for (integer_t i = 0; i < nsample; i++) {
                my_indices[i] = local_indices[i];
            }
        } else {
            // Large dataset: sort in-place using the output array
            // First initialize with identity
            for (integer_t i = 0; i < nsample; i++) {
                my_indices[i] = i;
            }
            
            // Shell sort (better than insertion for large arrays)
            integer_t gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};
            for (integer_t g = 0; g < 8; g++) {
                integer_t gap = gaps[g];
                if (gap >= nsample) continue;
                
                for (integer_t i = gap; i < nsample; i++) {
                    integer_t temp_idx = my_indices[i];
                    real_t temp_val = X_sparse.get(temp_idx, feature);
                    
                    integer_t j = i;
                    while (j >= gap) {
                        real_t cmp_val = X_sparse.get(my_indices[j - gap], feature);
                        if (cmp_val <= temp_val) break;
                        my_indices[j] = my_indices[j - gap];
                        j -= gap;
                    }
                    my_indices[j] = temp_idx;
                }
            }
        }
    }
}

void gpu_presort_sparse_features(
    const CudaSparseMatrixCSR& X_sparse,
    integer_t nsample,
    integer_t mdim,
    integer_t* sorted_indices,
    cudaStream_t stream
) {
    // One block per feature, thread 0 does sorting
    dim3 grid(mdim);
    dim3 block(1);  // Single thread per feature (sorting is sequential)
    
    gpu_presort_sparse_kernel<<<grid, block, 0, stream>>>(
        X_sparse, nsample, mdim, sorted_indices
    );
    // No sync here - caller will sync when needed
}

// ============================================================================
// Histogram binning kernels
// ============================================================================

// Kernel to bin sparse data values
__global__ void gpu_bin_sparse_data_kernel(
    CudaSparseMatrixCSR X_sparse,
    real_t* bin_edges_all,        // [mdim × (max_bins+1)]
    integer_t max_bins,
    uint8_t* X_binned_values      // [nnz] output
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= X_sparse.nnz) return;
    
    // Get the column (feature) for this non-zero
    integer_t col = X_sparse.d_indices[idx];
    real_t val = X_sparse.d_data[idx];
    
    // Get bin edges for this column
    real_t* edges = bin_edges_all + col * (max_bins + 1);
    
    // Binary search for bin
    integer_t lo = 0, hi = max_bins;
    while (lo < hi) {
        integer_t mid = (lo + hi) / 2;
        if (val <= edges[mid + 1]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    
    // Clamp to valid range
    if (lo >= max_bins) lo = max_bins - 1;
    
    // SAFE: Each thread writes to unique location
    X_binned_values[idx] = static_cast<uint8_t>(lo);
}

void gpu_compute_sparse_bin_edges(
    const CudaSparseMatrixCSR& X_sparse,
    integer_t max_bins,
    SparseHistogramData& hist_data,
    cudaStream_t stream
) {
    // Allocate histogram data
    hist_data.allocate(X_sparse.nnz, X_sparse.ncols, max_bins);
    
    // For quantile binning, we need to compute bin edges on CPU
    // (GPU quantile computation is complex, and this is one-time cost)
    
    // Copy sparse data to host
    std::vector<real_t> h_data(X_sparse.nnz);
    std::vector<integer_t> h_indices(X_sparse.nnz);
    std::vector<integer_t> h_indptr(X_sparse.nrows + 1);
    
    cudaMemcpy(h_data.data(), X_sparse.d_data, X_sparse.nnz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices.data(), X_sparse.d_indices, X_sparse.nnz * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indptr.data(), X_sparse.d_indptr, (X_sparse.nrows + 1) * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    // Compute bin edges per feature
    std::vector<real_t> h_bin_edges(X_sparse.ncols * (max_bins + 1));
    std::vector<integer_t> h_n_bins(X_sparse.ncols);
    
    for (integer_t col = 0; col < X_sparse.ncols; col++) {
        // Collect all non-zero values for this column
        std::vector<real_t> col_values;
        integer_t explicit_count = 0;
        for (integer_t row = 0; row < X_sparse.nrows; row++) {
            for (integer_t i = h_indptr[row]; i < h_indptr[row + 1]; i++) {
                if (h_indices[i] == col) {
                    col_values.push_back(h_data[i]);
                    explicit_count++;
                }
            }
        }
        
        // CRITICAL FIX: Include implicit zeros in bin edge computation
        // Sparse format doesn't store zeros, but they exist and must be considered
        integer_t n_implicit_zeros = X_sparse.nrows - explicit_count;
        bool has_zeros = (n_implicit_zeros > 0);
        
        // Add zeros to the value list if there are implicit zeros
        if (has_zeros) {
            col_values.push_back(0.0f);
        }
        
        // Sort values for quantile computation
        std::sort(col_values.begin(), col_values.end());
        
        // Count unique values
        std::vector<real_t> unique_values;
        if (!col_values.empty()) {
            unique_values.push_back(col_values[0]);
            for (size_t i = 1; i < col_values.size(); i++) {
                if (col_values[i] != unique_values.back()) {
                    unique_values.push_back(col_values[i]);
                }
            }
        }
        integer_t n_unique = static_cast<integer_t>(unique_values.size());
        
        // Compute bin edges
        real_t* edges = h_bin_edges.data() + col * (max_bins + 1);
        
        if (n_unique == 0) {
            // No values at all - use default edges
            h_n_bins[col] = 1;
            for (integer_t b = 0; b <= max_bins; b++) {
                edges[b] = static_cast<real_t>(b);
            }
        } else if (n_unique <= 2) {
            // BINARY OR NEAR-BINARY FEATURE: Use midpoint threshold
            // This is critical for proper handling of 0/1 indicator variables
            h_n_bins[col] = 2;
            real_t min_val = unique_values[0];
            real_t max_val = unique_values[n_unique - 1];
            edges[0] = min_val - 1.0f;                    // Below minimum
            edges[1] = (min_val + max_val) / 2.0f;        // MIDPOINT threshold
            edges[2] = max_val + 1.0f;                    // Above maximum
            // Fill remaining edges
            for (integer_t b = 3; b <= max_bins; b++) {
                edges[b] = edges[2];
            }
        } else {
            // CONTINUOUS FEATURE: Use quantile binning
            integer_t actual_bins = std::min(max_bins, n_unique);
            h_n_bins[col] = actual_bins;
            
            edges[0] = unique_values[0] - 1.0f;  // Min edge below smallest value
            for (integer_t b = 1; b <= actual_bins; b++) {
                integer_t quantile_idx = (b * n_unique) / actual_bins;
                if (quantile_idx >= n_unique) quantile_idx = n_unique - 1;
                edges[b] = unique_values[quantile_idx];
            }
            // Fill remaining edges with max value
            for (integer_t b = actual_bins + 1; b <= max_bins; b++) {
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
    
    // Copy bin edges to GPU
    cudaMemcpy(hist_data.d_bin_edges_all, h_bin_edges.data(), 
               X_sparse.ncols * (max_bins + 1) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(hist_data.d_n_bins_per_feature, h_n_bins.data(),
               X_sparse.ncols * sizeof(integer_t), cudaMemcpyHostToDevice);
}

void gpu_bin_sparse_data(
    const CudaSparseMatrixCSR& X_sparse,
    SparseHistogramData& hist_data,
    cudaStream_t stream
) {
    integer_t nnz_blocks = (X_sparse.nnz + 255) / 256;
    
    gpu_bin_sparse_data_kernel<<<nnz_blocks, 256, 0, stream>>>(
        X_sparse,
        hist_data.d_bin_edges_all,
        hist_data.max_bins,
        hist_data.d_X_binned_values
    );
    // No sync - caller will sync when needed
}

// ============================================================================
// Histogram-based split finding device functions
// ============================================================================

// Build histogram for classification (uses ::atomicAdd for safety)
__device__ void gpu_sparse_histogram_build_classification(
    CudaSparseMatrixCSR X_sparse,
    const uint8_t* X_binned_values,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    const integer_t* cl,
    const real_t* win,
    integer_t nclass,
    integer_t max_bins,
    real_t* histogram  // [max_bins × nclass] - shared or global memory
) {
    integer_t tid = threadIdx.x;
    integer_t num_threads = blockDim.x;
    
    // Initialize histogram to zero (parallel)
    for (integer_t i = tid; i < max_bins * nclass; i += num_threads) {
        histogram[i] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes subset of samples
    for (integer_t i = tid; i < node_count; i += num_threads) {
        integer_t sample = node_samples[i];
        real_t weight = win[sample];
        if (weight <= 0.0f) continue;
        
        // Get bin for this sample/feature
        // Default bin 0 for zeros (not in sparse structure)
        uint8_t bin = 0;
        
        // Search for feature in this sample's row
        integer_t row_start = X_sparse.d_indptr[sample];
        integer_t row_end = X_sparse.d_indptr[sample + 1];
        
        for (integer_t j = row_start; j < row_end; j++) {
            if (X_sparse.d_indices[j] == feature) {
                bin = X_binned_values[j];
                break;
            }
        }
        
        integer_t class_label = cl[sample];
        if (class_label >= 0 && class_label < nclass) {
            // ATOMIC: Multiple threads may write to same bin
            ::atomicAdd(&histogram[bin * nclass + class_label], weight);
        }
    }
    __syncthreads();
}

// Build histogram for regression
__device__ void gpu_sparse_histogram_build_regression(
    CudaSparseMatrixCSR X_sparse,
    const uint8_t* X_binned_values,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    const real_t* y,
    const real_t* win,
    integer_t max_bins,
    real_t* sum_y,      // [max_bins]
    real_t* sum_y2,     // [max_bins]
    real_t* sum_count   // [max_bins]
) {
    integer_t tid = threadIdx.x;
    integer_t num_threads = blockDim.x;
    
    // Initialize to zero (parallel)
    for (integer_t i = tid; i < max_bins; i += num_threads) {
        sum_y[i] = 0.0f;
        sum_y2[i] = 0.0f;
        sum_count[i] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes subset of samples
    for (integer_t i = tid; i < node_count; i += num_threads) {
        integer_t sample = node_samples[i];
        real_t weight = win[sample];
        if (weight <= 0.0f) continue;
        
        real_t y_val = y[sample];
        
        // Get bin for this sample/feature
        uint8_t bin = 0;
        
        integer_t row_start = X_sparse.d_indptr[sample];
        integer_t row_end = X_sparse.d_indptr[sample + 1];
        
        for (integer_t j = row_start; j < row_end; j++) {
            if (X_sparse.d_indices[j] == feature) {
                bin = X_binned_values[j];
                break;
            }
        }
        
        // ATOMIC: Multiple threads may write to same bin
        ::atomicAdd(&sum_y[bin], weight * y_val);
        ::atomicAdd(&sum_y2[bin], weight * y_val * y_val);
        ::atomicAdd(&sum_count[bin], weight);
    }
    __syncthreads();
}

// Find best split from histogram (classification - Gini)
__device__ void gpu_sparse_histogram_find_split_classification(
    real_t* histogram,  // [max_bins × nclass]
    real_t* bin_edges,  // [max_bins + 1]
    integer_t max_bins,
    integer_t n_bins,
    integer_t nclass,
    real_t total_weight,
    real_t* best_crit,
    real_t* best_threshold,
    integer_t* jstat
) {
    // Single thread does split finding (small loop)
    if (threadIdx.x != 0) return;
    
    // Compute total per-class weights
    real_t total_class[32];
    for (integer_t c = 0; c < nclass && c < 32; c++) {
        total_class[c] = 0.0f;
        for (integer_t b = 0; b < n_bins; b++) {
            total_class[c] += histogram[b * nclass + c];
        }
    }
    
    // Scan bins for best split
    real_t left_class[32];
    for (integer_t c = 0; c < nclass && c < 32; c++) left_class[c] = 0.0f;
    
    real_t best = -1.0f;
    real_t best_thresh = 0.0f;
    integer_t found = 0;
    
    real_t left_total = 0.0f;
    
    for (integer_t b = 0; b < n_bins - 1; b++) {
        // Accumulate left side
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            left_class[c] += histogram[b * nclass + c];
        }
        left_total += histogram[b * nclass];  // Assuming weight in class 0 or sum
        
        // Recompute left_total properly
        left_total = 0.0f;
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            left_total += left_class[c];
        }
        real_t right_total = total_weight - left_total;
        
        if (left_total < 1.0f || right_total < 1.0f) continue;
        
        // Compute Gini impurity reduction
        real_t gini_left = 1.0f;
        real_t gini_right = 1.0f;
        
        for (integer_t c = 0; c < nclass && c < 32; c++) {
            real_t p_left = left_class[c] / left_total;
            real_t p_right = (total_class[c] - left_class[c]) / right_total;
            gini_left -= p_left * p_left;
            gini_right -= p_right * p_right;
        }
        
        real_t weighted_gini = (left_total * gini_left + right_total * gini_right) / total_weight;
        real_t crit = 1.0f - weighted_gini;  // Higher is better
        
        if (crit > best) {
            best = crit;
            // FIX: Split point is at upper edge of bin b (boundary between bin b and bin b+1)
            best_thresh = bin_edges[b + 1];
            found = 1;
        }
    }
    
    *best_crit = best;
    *best_threshold = best_thresh;
    *jstat = found ? 0 : 1;  // 0 = good split, 1 = no split
}

// Find best split from histogram (regression - MSE)
__device__ void gpu_sparse_histogram_find_split_regression(
    real_t* sum_y,
    real_t* sum_y2,
    real_t* sum_count,
    real_t* bin_edges,
    integer_t max_bins,
    integer_t n_bins,
    real_t total_sum_y,
    real_t total_sum_y2,
    real_t total_count,
    real_t* best_crit,
    real_t* best_threshold,
    integer_t* jstat
) {
    if (threadIdx.x != 0) return;
    
    real_t best = 1.0e20f;  // MSE - lower is better
    real_t best_thresh = 0.0f;
    integer_t found = 0;
    
    real_t left_sum_y = 0.0f;
    real_t left_sum_y2 = 0.0f;
    real_t left_count = 0.0f;
    
    for (integer_t b = 0; b < n_bins - 1; b++) {
        left_sum_y += sum_y[b];
        left_sum_y2 += sum_y2[b];
        left_count += sum_count[b];
        
        real_t right_count = total_count - left_count;
        if (left_count < 1.0f || right_count < 1.0f) continue;
        
        real_t right_sum_y = total_sum_y - left_sum_y;
        real_t right_sum_y2 = total_sum_y2 - left_sum_y2;
        
        real_t mean_left = left_sum_y / left_count;
        real_t mean_right = right_sum_y / right_count;
        
        real_t mse_left = (left_sum_y2 / left_count) - mean_left * mean_left;
        real_t mse_right = (right_sum_y2 / right_count) - mean_right * mean_right;
        
        real_t weighted_mse = (left_count * mse_left + right_count * mse_right) / total_count;
        
        if (weighted_mse < best) {
            best = weighted_mse;
            // FIX: Split point is at upper edge of bin b (boundary between bin b and bin b+1)
            best_thresh = bin_edges[b + 1];
            found = 1;
        }
    }
    
    *best_crit = best;
    *best_threshold = best_thresh;
    *jstat = found ? 0 : 1;
}

// ============================================================================
// Main V2 kernel with pre-sort and histogram support
// Each block grows ONE complete tree (matches original kernel structure)
// ============================================================================

// Shared memory structure for parallel split finding
struct ThreadSplitResultV2 {
    real_t crit;
    integer_t var;
    real_t threshold;
    integer_t split_idx;
};

__global__ void gpu_sparse_tree_parallel_kernel_v2(
    CudaSparseMatrixCSR X_sparse,
    integer_t num_trees,
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    integer_t task_type,           // 0=classification, 1=regression
    const integer_t* cl,
    const real_t* y_regression,
    const integer_t* nin_all,
    const real_t* win_all,
    // Pre-sort parameters
    bool use_presort,
    const integer_t* sorted_indices,  // [mdim × nsample]
    // Histogram parameters
    bool use_histogram,
    const uint8_t* X_binned_values,   // [nnz]
    const real_t* bin_edges_all,      // [mdim × (max_bins+1)]
    const integer_t* n_bins_per_feature,
    integer_t max_bins,
    // Outputs
    integer_t* nodestatus_all,
    integer_t* bestvar_all,
    real_t* xbestsplit_all,
    integer_t* treemap_all,
    integer_t* nodeclass_all,
    real_t* tnodewt_all,
    integer_t* nnode_all,
    curandState* random_states
) {
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    integer_t tid = threadIdx.x;
    integer_t num_threads = blockDim.x;
    
    // Calculate offsets for this tree
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t sample_offset = tree_id * nsample;
    
    // Pointers to this tree's arrays
    integer_t* nodestatus = nodestatus_all + node_offset;
    integer_t* nodeclass = nodeclass_all + node_offset;
    real_t* tnodewt = tnodewt_all ? tnodewt_all + node_offset : nullptr;
    integer_t* bestvar = bestvar_all + node_offset;
    real_t* xbestsplit = xbestsplit_all + node_offset;
    integer_t* treemap = treemap_all + treemap_offset;
    integer_t* nnode = nnode_all + tree_id;
    const integer_t* nin = nin_all + sample_offset;
    const real_t* win = win_all + sample_offset;
    
    curandState local_rng = random_states[tree_id];
    
    // Shared memory for sample tracking
    __shared__ integer_t node_samples[2048];
    __shared__ integer_t node_sample_count;
    __shared__ integer_t shared_nnode;
    __shared__ integer_t shared_majority_class;
    __shared__ real_t shared_crit0;
    __shared__ real_t shared_mean_y;
    __shared__ integer_t thread_counts[256];
    __shared__ integer_t thread_offsets[256];
    
    // Shared memory for parallel split finding and histograms
    extern __shared__ char shared_mem[];
    ThreadSplitResultV2* thread_results = reinterpret_cast<ThreadSplitResultV2*>(shared_mem);
    
    // Histogram work area (after thread results)
    // Size depends on nclass for classification or just max_bins for regression
    integer_t hist_size_per_thread = use_histogram ? 
        (task_type == 1 ? max_bins * 3 : max_bins * min(nclass, 32)) : 0;
    real_t* hist_work = reinterpret_cast<real_t*>(shared_mem + num_threads * sizeof(ThreadSplitResultV2));
    
    // Per-class weights for incremental Gini (non-histogram path)
    real_t wl[32], wr[32];
    real_t sum_y_left, sum_y_right, sum_w_left, sum_w_right;
    
    // Initialize tree structure
    if (tid == 0) {
        for (integer_t i = 0; i < maxnode; ++i) {
            nodestatus[i] = 0;
            bestvar[i] = 0;
            xbestsplit[i] = 0.0f;
            treemap[i * 2] = 0;
            treemap[i * 2 + 1] = 0;
            nodeclass[i] = 0;
        }
        nodestatus[0] = 2;
        shared_nnode = 1;
        nnode[0] = 1;
    }
    __syncthreads();
    
    // Main node processing loop
    integer_t max_iterations = maxnode * 2;
    for (integer_t kgrow = 0; kgrow < shared_nnode && kgrow < max_iterations; ++kgrow) {
        __syncthreads();
        
        if (nodestatus[kgrow] != 2) continue;
        
        // =====================================================================
        // PARALLEL SAMPLE COLLECTION
        // =====================================================================
        
        if (tid == 0) node_sample_count = 0;
        thread_counts[tid] = 0;
        __syncthreads();
        
        integer_t my_count = 0;
        integer_t samples_per_thread = (nsample + num_threads - 1) / num_threads;
        integer_t my_start = tid * samples_per_thread;
        integer_t my_end = min(my_start + samples_per_thread, nsample);
        
        integer_t my_samples[32];
        
        for (integer_t i = my_start; i < my_end && my_count < 32; ++i) {
            bool reaches_node = false;
            
            if (kgrow == 0) {
                reaches_node = (nin[i] > 0);
            } else {
                if (win[i] > 0.0f) {
                    integer_t current = 0;
                    integer_t depth = 0;
                    while (depth < 100) {
                        if (current == kgrow) {
                            reaches_node = true;
                            break;
                        }
                        if (nodestatus[current] != 1) break;
                        
                        integer_t split_var = bestvar[current];
                        real_t split_val = xbestsplit[current];
                        real_t sample_val = X_sparse.get(i, split_var);
                        
                        if (sample_val <= split_val) {
                            current = treemap[current * 2];
                        } else {
                            current = treemap[current * 2 + 1];
                        }
                        if (current <= 0 || current >= maxnode) break;
                        depth++;
                    }
                }
            }
            
            if (reaches_node) {
                my_samples[my_count++] = i;
            }
        }
        
        thread_counts[tid] = my_count;
        __syncthreads();
        
        if (tid == 0) {
            thread_offsets[0] = 0;
            for (integer_t t = 1; t < num_threads; ++t) {
                thread_offsets[t] = thread_offsets[t-1] + thread_counts[t-1];
            }
            node_sample_count = thread_offsets[num_threads-1] + thread_counts[num_threads-1];
            if (node_sample_count > 2048) node_sample_count = 2048;
        }
        __syncthreads();
        
        integer_t my_offset = thread_offsets[tid];
        for (integer_t i = 0; i < my_count && my_offset + i < 2048; ++i) {
            node_samples[my_offset + i] = my_samples[i];
        }
        __syncthreads();
        
        integer_t n_samples = node_sample_count;
        
        // Check minimum node size
        if (n_samples < 2 * min_node_size) {
            if (tid == 0) {
                nodestatus[kgrow] = -1;
                if (task_type == 1 && y_regression != nullptr) {
                    real_t sum_y = 0.0f, sum_w = 0.0f;
                    for (integer_t i = 0; i < n_samples; ++i) {
                        integer_t idx = node_samples[i];
                        real_t w = win[idx];
                        sum_y += w * y_regression[idx];
                        sum_w += w;
                    }
                    if (tnodewt) tnodewt[kgrow] = (sum_w > 0.0f) ? (sum_y / sum_w) : 0.0f;
                    nodeclass[kgrow] = 0;
                } else {
                    integer_t class_counts[32] = {0};
                    for (integer_t i = 0; i < n_samples; ++i) {
                        integer_t c = cl[node_samples[i]];
                        if (c >= 0 && c < 32) class_counts[c]++;
                    }
                    integer_t best_class = 0;
                    for (integer_t c = 1; c < nclass && c < 32; c++) {
                        if (class_counts[c] > class_counts[best_class]) best_class = c;
                    }
                    nodeclass[kgrow] = best_class;
                }
            }
            continue;
        }
        
        // Compute class distribution/mean and parent criterion
        if (tid == 0) {
            if (task_type == 1 && y_regression != nullptr) {
                real_t sum_y = 0.0f, sum_w = 0.0f;
                for (integer_t i = 0; i < n_samples; ++i) {
                    integer_t idx = node_samples[i];
                    real_t w = win[idx];
                    sum_y += w * y_regression[idx];
                    sum_w += w;
                }
                shared_mean_y = (sum_w > 0.0f) ? (sum_y / sum_w) : 0.0f;
                shared_majority_class = 0;
                
                real_t ss = 0.0f;
                for (integer_t i = 0; i < n_samples; ++i) {
                    integer_t idx = node_samples[i];
                    real_t w = win[idx];
                    real_t diff = y_regression[idx] - shared_mean_y;
                    ss += w * diff * diff;
                }
                shared_crit0 = ss;
                
                bool all_same = true;
                real_t first_y = y_regression[node_samples[0]];
                for (integer_t i = 1; i < n_samples && all_same; ++i) {
                    if (y_regression[node_samples[i]] != first_y) all_same = false;
                }
                if (all_same) {
                    nodestatus[kgrow] = -1;
                    if (tnodewt) tnodewt[kgrow] = shared_mean_y;
                }
            } else {
                integer_t class_counts[32] = {0};
                for (integer_t i = 0; i < n_samples; ++i) {
                    integer_t c = cl[node_samples[i]];
                    if (c >= 0 && c < 32) class_counts[c]++;
                }
                
                integer_t max_count = 0, non_zero = 0;
                for (integer_t c = 0; c < nclass && c < 32; c++) {
                    if (class_counts[c] > max_count) {
                        max_count = class_counts[c];
                        shared_majority_class = c;
                    }
                    if (class_counts[c] > 0) non_zero++;
                }
                
                real_t pno = 0.0f, pdo = 0.0f;
                for (integer_t c = 0; c < nclass && c < 32; c++) {
                    pno += class_counts[c] * class_counts[c];
                    pdo += class_counts[c];
                }
                shared_crit0 = pdo > 0.0f ? (pno / pdo) : 0.0f;
                
                if (non_zero <= 1) {
                    nodestatus[kgrow] = -1;
                    nodeclass[kgrow] = shared_majority_class;
                }
            }
        }
        __syncthreads();
        
        if (nodestatus[kgrow] == -1) continue;
        
        // =====================================================================
        // SPLIT FINDING - Uses histogram or standard path
        // =====================================================================
        
        thread_results[tid].crit = (task_type == 1) ? -1e30f : -2.0f;
        thread_results[tid].var = -1;
        thread_results[tid].threshold = 0.0f;
        thread_results[tid].split_idx = -1;
        
        // Each thread handles subset of mtry features
        for (integer_t mv = tid; mv < mtry; mv += num_threads) {
            curandState temp_state = local_rng;
            skipahead(kgrow * mtry + mv, &temp_state);
            real_t rand_val = curand_uniform(&temp_state);
            integer_t mvar = static_cast<integer_t>(rand_val * mdim);
            if (mvar >= mdim) mvar = mdim - 1;
            
            if (use_histogram && max_bins > 0) {
                // =====================================================================
                // HISTOGRAM-BASED SPLIT FINDING (O(max_bins) instead of O(n log n))
                // =====================================================================
                
                integer_t actual_bins = max_bins;
                const real_t* edges = bin_edges_all + mvar * (max_bins + 1);
                
                if (task_type == 1 && y_regression != nullptr) {
                    // REGRESSION: Build and scan histogram
                    real_t sum_y[256], sum_count[256];
                    for (integer_t b = 0; b < actual_bins; b++) {
                        sum_y[b] = 0.0f;
                        sum_count[b] = 0.0f;
                    }
                    
                    real_t total_sum_y = 0.0f, total_count = 0.0f;
                    
                    for (integer_t i = 0; i < n_samples; ++i) {
                        integer_t sample = node_samples[i];
                        real_t weight = win[sample];
                        if (weight <= 0.0f) continue;
                        
                        real_t y_val = y_regression[sample];
                        
                        // Get bin for this sample/feature
                        uint8_t bin = 0;
                        integer_t row_start = X_sparse.d_indptr[sample];
                        integer_t row_end = X_sparse.d_indptr[sample + 1];
                        
                        for (integer_t j = row_start; j < row_end; j++) {
                            if (X_sparse.d_indices[j] == mvar) {
                                bin = X_binned_values[j];
                                break;
                            }
                        }
                        if (bin >= actual_bins) bin = actual_bins - 1;
                        
                        sum_y[bin] += weight * y_val;
                        sum_count[bin] += weight;
                        total_sum_y += weight * y_val;
                        total_count += weight;
                    }
                    
                    // Scan for best split
                    real_t left_sum_y = 0.0f, left_count = 0.0f;
                    
                    for (integer_t b = 0; b < actual_bins - 1; b++) {
                        left_sum_y += sum_y[b];
                        left_count += sum_count[b];
                        
                        real_t right_count = total_count - left_count;
                        if (left_count < min_node_size || right_count < min_node_size) continue;
                        
                        real_t right_sum_y = total_sum_y - left_sum_y;
                        
                        if (left_count > 1e-5f && right_count > 1e-5f) {
                            real_t crit = (left_sum_y * left_sum_y / left_count) +
                                         (right_sum_y * right_sum_y / right_count);
                            
                            if (crit > thread_results[tid].crit ||
                                (crit == thread_results[tid].crit && mvar < thread_results[tid].var)) {
                                thread_results[tid].crit = crit;
                                thread_results[tid].var = mvar;
                                thread_results[tid].threshold = edges[b + 1];
                                thread_results[tid].split_idx = b;
                            }
                        }
                    }
                } else {
                    // CLASSIFICATION: Build histogram per class
                    real_t histogram[32 * 256];  // Up to 32 classes, 256 bins
                    integer_t eff_nclass = min(nclass, 32);
                    integer_t eff_bins = min(actual_bins, 256);
                    
                    for (integer_t i = 0; i < eff_bins * eff_nclass; i++) {
                        histogram[i] = 0.0f;
                    }
                    
                    real_t total_class[32] = {0};
                    real_t total_weight = 0.0f;
                    
                    for (integer_t i = 0; i < n_samples; ++i) {
                        integer_t sample = node_samples[i];
                        real_t weight = win[sample];
                        if (weight <= 0.0f) continue;
                        
                        integer_t class_label = cl[sample];
                        if (class_label < 0 || class_label >= eff_nclass) continue;
                        
                        // Get bin
                        uint8_t bin = 0;
                        integer_t row_start = X_sparse.d_indptr[sample];
                        integer_t row_end = X_sparse.d_indptr[sample + 1];
                        
                        for (integer_t j = row_start; j < row_end; j++) {
                            if (X_sparse.d_indices[j] == mvar) {
                                bin = X_binned_values[j];
                                break;
                            }
                        }
                        if (bin >= eff_bins) bin = eff_bins - 1;
                        
                        histogram[bin * eff_nclass + class_label] += weight;
                        total_class[class_label] += weight;
                        total_weight += weight;
                    }
                    
                    // Scan for best split (incremental Gini)
                    real_t left_class[32] = {0};
                    real_t rln = 0.0f, rld = 0.0f;
                    real_t rrn = 0.0f, rrd = total_weight;
                    
                    for (integer_t c = 0; c < eff_nclass; c++) {
                        rrn += total_class[c] * total_class[c];
                    }
                    
                    for (integer_t b = 0; b < eff_bins - 1; b++) {
                        for (integer_t c = 0; c < eff_nclass; c++) {
                            real_t u = histogram[b * eff_nclass + c];
                            if (u > 0.0f) {
                                rln += u * (u + 2.0f * left_class[c]);
                                rrn += u * (u - 2.0f * (total_class[c] - left_class[c]));
                                rld += u;
                                rrd -= u;
                                left_class[c] += u;
                            }
                        }
                        
                        if (rld < min_node_size || rrd < min_node_size) continue;
                        
                        if (rld > 1e-5f && rrd > 1e-5f) {
                            real_t crit = (rln / rld) + (rrn / rrd);
                            
                            if (crit > thread_results[tid].crit ||
                                (crit == thread_results[tid].crit && mvar < thread_results[tid].var)) {
                                thread_results[tid].crit = crit;
                                thread_results[tid].var = mvar;
                                thread_results[tid].threshold = edges[b + 1];
                                thread_results[tid].split_idx = b;
                            }
                        }
                    }
                }
            } else {
                // =====================================================================
                // STANDARD PATH (with optional pre-sorting)
                // =====================================================================
                
                integer_t sorted_samples[512];
                real_t sorted_values[512];
                integer_t n_valid = 0;
                
                if (use_presort && sorted_indices != nullptr) {
                    // USE PRE-SORTED INDICES
                    const integer_t* presort = sorted_indices + mvar * nsample;
                    
                    // Filter to only samples in this node (already sorted)
                    // Create a quick lookup for node membership
                    for (integer_t rank = 0; rank < nsample && n_valid < 512; ++rank) {
                        integer_t sample_idx = presort[rank];
                        // Check if sample is in this node
                        for (integer_t ni = 0; ni < n_samples; ++ni) {
                            if (node_samples[ni] == sample_idx) {
                                sorted_samples[n_valid] = sample_idx;
                                sorted_values[n_valid] = X_sparse.get(sample_idx, mvar);
                                n_valid++;
                                break;
                            }
                        }
                    }
                } else {
                    // STANDARD PATH: Collect and sort
                    for (integer_t i = 0; i < n_samples && n_valid < 512; ++i) {
                        integer_t sample_idx = node_samples[i];
                        sorted_samples[n_valid] = sample_idx;
                        sorted_values[n_valid] = X_sparse.get(sample_idx, mvar);
                        n_valid++;
                    }
                    
                    // Insertion sort for small arrays
                    for (integer_t i = 1; i < n_valid; i++) {
                        integer_t key_idx = sorted_samples[i];
                        real_t key_val = sorted_values[i];
                        integer_t j = i - 1;
                        
                        while (j >= 0 && sorted_values[j] > key_val) {
                            sorted_samples[j + 1] = sorted_samples[j];
                            sorted_values[j + 1] = sorted_values[j];
                            j--;
                        }
                        sorted_samples[j + 1] = key_idx;
                        sorted_values[j + 1] = key_val;
                    }
                }
                
                // Standard incremental split finding (same as original kernel)
                if (task_type == 1 && y_regression != nullptr) {
                    sum_y_left = 0.0f; sum_w_left = 0.0f;
                    sum_y_right = 0.0f; sum_w_right = 0.0f;
                    
                    for (integer_t i = 0; i < n_valid; ++i) {
                        integer_t idx = sorted_samples[i];
                        real_t w = win[idx];
                        sum_y_right += w * y_regression[idx];
                        sum_w_right += w;
                    }
                    
                    for (integer_t ii = 0; ii < n_valid - 1; ++ii) {
                        integer_t sample_idx = sorted_samples[ii];
                        real_t val1 = sorted_values[ii];
                        real_t val2 = sorted_values[ii + 1];
                        
                        real_t w = win[sample_idx];
                        real_t y_val = y_regression[sample_idx];
                        
                        sum_y_left += w * y_val;
                        sum_w_left += w;
                        sum_y_right -= w * y_val;
                        sum_w_right -= w;
                        
                        if (val1 >= val2) continue;
                        if (sum_w_left < min_node_size || sum_w_right < min_node_size) continue;
                        
                        if (sum_w_left > 1e-5f && sum_w_right > 1e-5f) {
                            real_t crit = (sum_y_left * sum_y_left / sum_w_left) +
                                         (sum_y_right * sum_y_right / sum_w_right);
                            
                            if (crit > thread_results[tid].crit ||
                                (crit == thread_results[tid].crit && mvar < thread_results[tid].var)) {
                                thread_results[tid].crit = crit;
                                thread_results[tid].var = mvar;
                                thread_results[tid].threshold = (val1 + val2) / 2.0f;
                                thread_results[tid].split_idx = ii;
                            }
                        }
                    }
                } else {
                    for (integer_t c = 0; c < 32; c++) {
                        wl[c] = 0.0f;
                        wr[c] = 0.0f;
                    }
                    
                    for (integer_t i = 0; i < n_samples; ++i) {
                        integer_t c = cl[node_samples[i]];
                        if (c >= 0 && c < 32) {
                            wr[c] += win[node_samples[i]];
                        }
                    }
                    
                    real_t rln = 0.0f, rld = 0.0f;
                    real_t rrn = 0.0f, rrd = 0.0f;
                    
                    for (integer_t c = 0; c < nclass && c < 32; c++) {
                        rrn += wr[c] * wr[c];
                        rrd += wr[c];
                    }
                    
                    for (integer_t ii = 0; ii < n_valid - 1; ++ii) {
                        integer_t sample_idx = sorted_samples[ii];
                        real_t val1 = sorted_values[ii];
                        real_t val2 = sorted_values[ii + 1];
                        
                        integer_t k = cl[sample_idx];
                        real_t u = win[sample_idx];
                        
                        if (k >= 0 && k < 32) {
                            rln += u * (u + 2.0f * wl[k]);
                            rrn += u * (u - 2.0f * wr[k]);
                            rld += u;
                            rrd -= u;
                            wl[k] += u;
                            wr[k] -= u;
                        }
                        
                        if (val1 >= val2) continue;
                        if (rld < min_node_size || rrd < min_node_size) continue;
                        
                        if (rld > 1e-5f && rrd > 1e-5f) {
                            real_t crit = (rln / rld) + (rrn / rrd);
                            
                            if (crit > thread_results[tid].crit ||
                                (crit == thread_results[tid].crit && mvar < thread_results[tid].var)) {
                                thread_results[tid].crit = crit;
                                thread_results[tid].var = mvar;
                                thread_results[tid].threshold = (val1 + val2) / 2.0f;
                                thread_results[tid].split_idx = ii;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // =====================================================================
        // REDUCTION
        // =====================================================================
        
        real_t best_crit = (task_type == 1) ? -1e30f : -2.0f;
        integer_t best_var = -1;
        real_t best_threshold = 0.0f;
        
        if (tid == 0) {
            for (integer_t t = 0; t < num_threads; ++t) {
                if (thread_results[t].var >= 0) {
                    if (thread_results[t].crit > best_crit ||
                        (thread_results[t].crit == best_crit && thread_results[t].var < best_var)) {
                        best_crit = thread_results[t].crit;
                        best_var = thread_results[t].var;
                        best_threshold = thread_results[t].threshold;
                    }
                }
            }
        }
        __syncthreads();
        
        // Apply split or make terminal
        if (tid == 0) {
            real_t min_valid_crit = (task_type == 1) ? -1e29f : -1.0f;
            
            if (best_var >= 0 && best_crit > min_valid_crit && shared_nnode + 2 <= maxnode) {
                integer_t left_child = shared_nnode;
                integer_t right_child = shared_nnode + 1;
                
                nodestatus[kgrow] = 1;
                bestvar[kgrow] = best_var;
                xbestsplit[kgrow] = best_threshold;
                
                treemap[kgrow * 2] = left_child;
                treemap[kgrow * 2 + 1] = right_child;
                
                nodestatus[left_child] = 2;
                nodestatus[right_child] = 2;
                
                shared_nnode += 2;
                nnode[0] = shared_nnode;
            } else {
                nodestatus[kgrow] = -1;
                if (task_type == 1 && y_regression != nullptr && tnodewt != nullptr) {
                    tnodewt[kgrow] = shared_mean_y;
                }
                nodeclass[kgrow] = shared_majority_class;
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        random_states[tree_id] = local_rng;
        nnode[0] = shared_nnode;
    }
}

// ============================================================================
// Host wrapper for V2 kernel
// ============================================================================

integer_t gpu_growtree_sparse_parallel_batch_v2(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const real_t* d_y_regression,
    const integer_t* d_nin_all,
    const real_t* d_win_all,
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    integer_t num_trees,
    integer_t task_type,
    curandState* d_rng_states,
    bool use_presort,
    const integer_t* sorted_indices,
    bool use_histogram,
    const SparseHistogramData* hist_data,
    integer_t* d_nodestatus_all,
    integer_t* d_bestvar_all,
    real_t* d_xbestsplit_all,
    integer_t* d_treemap_all,
    integer_t* d_nodeclass_all,
    real_t* d_tnodewt_all,
    integer_t* d_nnode_all,
    cudaStream_t stream
) {
    // Initialize output arrays (cast to size_t to prevent integer overflow)
    cudaMemsetAsync(d_nodestatus_all, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t), stream);
    cudaMemsetAsync(d_treemap_all, 0, static_cast<size_t>(num_trees) * 2 * static_cast<size_t>(maxnode) * sizeof(integer_t), stream);
    cudaMemsetAsync(d_nodeclass_all, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t), stream);
    cudaMemsetAsync(d_nnode_all, 0, static_cast<size_t>(num_trees) * sizeof(integer_t), stream);
    if (d_tnodewt_all) {
        cudaMemsetAsync(d_tnodewt_all, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(real_t), stream);
    }
    
    // Calculate optimal threads
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    integer_t optimal_threads = mtry;
    if (optimal_threads < 32) optimal_threads = 32;
    optimal_threads = ((optimal_threads + 31) / 32) * 32;
    if (optimal_threads > 256) optimal_threads = 256;
    if (optimal_threads > props.maxThreadsPerBlock) {
        optimal_threads = (props.maxThreadsPerBlock / 32) * 32;
    }
    
    // Calculate shared memory: ThreadSplitResultV2 only (histogram uses local arrays)
    size_t shared_mem_needed = optimal_threads * sizeof(ThreadSplitResultV2);
    if (shared_mem_needed > props.sharedMemPerBlock) {
        optimal_threads = static_cast<integer_t>(props.sharedMemPerBlock / sizeof(ThreadSplitResultV2));
        optimal_threads = (optimal_threads / 32) * 32;
        if (optimal_threads < 32) optimal_threads = 32;
    }
    
    dim3 grid(num_trees);
    dim3 block(optimal_threads);
    
    // Extract histogram parameters (if available)
    uint8_t* binned_values = nullptr;
    real_t* bin_edges = nullptr;
    integer_t* n_bins = nullptr;
    integer_t max_bins = 256;
    
    if (use_histogram && hist_data != nullptr && hist_data->allocated) {
        binned_values = hist_data->d_X_binned_values;
        bin_edges = hist_data->d_bin_edges_all;
        n_bins = hist_data->d_n_bins_per_feature;
        max_bins = hist_data->max_bins;
    }
    
    gpu_sparse_tree_parallel_kernel_v2<<<grid, block, shared_mem_needed, stream>>>(
        X_sparse,
        num_trees,
        nsample,
        mdim,
        nclass,
        mtry,
        maxnode,
        min_node_size,
        task_type,
        d_cl,
        d_y_regression,
        d_nin_all,
        d_win_all,
        use_presort,
        sorted_indices,
        use_histogram,
        binned_values,
        bin_edges,
        n_bins,
        max_bins,
        d_nodestatus_all,
        d_bestvar_all,
        d_xbestsplit_all,
        d_treemap_all,
        d_nodeclass_all,
        d_tnodewt_all,
        d_nnode_all,
        d_rng_states
    );
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUDA_ERROR_MEMORY_ALLOCATION;
    }
    
    cudaStreamSynchronize(stream);
    
    return CUDA_OK;
}

} // namespace cuda
} // namespace rf

