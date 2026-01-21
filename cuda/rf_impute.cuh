#ifndef RF_IMPUTE_CUH
#define RF_IMPUTE_CUH

#include "../include/rf_types.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace rf {
namespace cuda {

// ============================================================================
// GPU IMPUTATION KERNELS - CUDA Parallelized for Dense and Sparse
// ============================================================================

// Kernel to find NaN values and compute missing mask
__global__ void find_missing_kernel(
    const float* X,
    bool* missing_mask,
    int* missing_count_per_col,
    int n_samples,
    int n_features
);

// Kernel to compute column statistics (median/mode) - reduction kernel
__global__ void compute_column_stats_kernel(
    const float* X,
    const bool* missing_mask,
    const bool* is_categorical,
    float* fill_values,
    int n_samples,
    int n_features
);

// Kernel to apply initial imputation (rough: median/mode)
__global__ void rough_impute_kernel(
    float* X,
    const bool* missing_mask,
    const float* fill_values,
    int n_samples,
    int n_features
);

// Kernel to apply random imputation
__global__ void random_impute_kernel(
    float* X,
    const bool* missing_mask,
    const float* valid_values,      // Flattened valid values per column
    const int* valid_counts,         // Count of valid values per column
    const int* valid_offsets,        // Offset into valid_values for each column
    unsigned int seed,
    int n_samples,
    int n_features
);

// Kernel to detect categorical features
__global__ void detect_categorical_kernel(
    const float* X,
    bool* is_categorical,
    int* unique_counts,
    int n_samples,
    int n_features,
    int max_unique
);

// ============================================================================
// SPARSE MATRIX KERNELS
// ============================================================================

// Kernel to find NaN in sparse data
__global__ void find_missing_sparse_kernel(
    const float* data,
    const int* indices,
    const int* indptr,
    bool* missing_mask,     // Output: per-element (nnz) mask
    int* missing_count_per_col,
    int nrows,
    int ncols,
    int nnz
);

// Kernel to impute sparse NaN values
__global__ void impute_sparse_kernel(
    float* data,
    const int* indices,
    const int* indptr,
    const bool* missing_mask,
    const float* fill_values,
    int nrows,
    int ncols,
    int nnz
);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Compute median on GPU using parallel reduction
void gpu_compute_median(
    const float* d_values,
    int n_values,
    float* d_median,
    cudaStream_t stream = 0
);

// Compute mode on GPU using parallel counting
void gpu_compute_mode(
    const float* d_values,
    int n_values,
    float* d_mode,
    cudaStream_t stream = 0
);

// Sort column values on GPU (for median computation)
void gpu_sort_column(
    float* d_values,
    int n_values,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace rf

#endif // RF_IMPUTE_CUH

