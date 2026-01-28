/**
 * @file rf_data_accessor.cuh
 * @brief Unified data accessor for GPU kernels supporting both dense and sparse matrices
 * 
 * This abstraction allows the same GPU kernel code to work with either:
 * - Dense matrices: O(1) direct array access
 * - Sparse matrices: O(log nnz_per_row) binary search access
 * 
 * Usage in kernels:
 *   real_t value = accessor.get(sample_idx, feature_idx);
 * 
 * This enables GPU Sparse to use the same single-kernel architecture as GPU Dense,
 * resulting in:
 * - Identical trees (same RNG sequence, same split decisions)
 * - Faster execution (no per-node kernel launches)
 * - Better GPU utilization
 */

#ifndef RF_DATA_ACCESSOR_CUH
#define RF_DATA_ACCESSOR_CUH

#include "rf_types.hpp"
#include "rf_sparse_cuda.cuh"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

/**
 * Unified data accessor for GPU kernels
 * 
 * Can operate in two modes:
 * 1. Dense mode: direct array access via x[sample*mdim + feature]
 * 2. Sparse mode: binary search access via CudaSparseMatrixCSR::get()
 */
struct DataAccessor {
    // Dense data pointer (nullptr if sparse mode)
    const real_t* d_x;
    
    // Sparse matrix (only valid if is_sparse=true)
    CudaSparseMatrixCSR sparse;
    
    // Mode flag
    bool is_sparse;
    
    // Dimensions (used for dense access)
    integer_t mdim;
    
    /**
     * Default constructor - dense mode with null data
     */
    __host__ __device__ DataAccessor() 
        : d_x(nullptr), is_sparse(false), mdim(0) {}
    
    /**
     * Dense constructor
     * @param x Dense data array (row-major: x[sample * mdim + feature])
     * @param num_features Number of features (mdim)
     */
    __host__ DataAccessor(const real_t* x, integer_t num_features)
        : d_x(x), is_sparse(false), mdim(num_features) {}
    
    /**
     * Sparse constructor
     * @param sparse_matrix GPU-side CSR sparse matrix
     */
    __host__ DataAccessor(const CudaSparseMatrixCSR& sparse_matrix)
        : d_x(nullptr), sparse(sparse_matrix), is_sparse(true), mdim(sparse_matrix.ncols) {}
    
    /**
     * Get element value at (sample, feature)
     * 
     * For dense: O(1) direct access
     * For sparse: O(log nnz_per_row) binary search
     * 
     * @param sample Sample index (row)
     * @param feature Feature index (column)
     * @return Feature value (0.0f for sparse zeros)
     */
    __device__ __forceinline__ real_t get(integer_t sample, integer_t feature) const {
        if (is_sparse) {
            return sparse.get(sample, feature);
        } else {
            return d_x[sample * mdim + feature];
        }
    }
    
    /**
     * Check if accessor is valid
     */
    __host__ bool is_valid() const {
        if (is_sparse) {
            return sparse.is_valid();
        } else {
            return d_x != nullptr && mdim > 0;
        }
    }
};

/**
 * Create a dense data accessor
 * @param x Dense data array
 * @param mdim Number of features
 * @return DataAccessor configured for dense access
 */
inline DataAccessor make_dense_accessor(const real_t* x, integer_t mdim) {
    return DataAccessor(x, mdim);
}

/**
 * Create a sparse data accessor
 * @param sparse_matrix GPU-side CSR sparse matrix
 * @return DataAccessor configured for sparse access
 */
inline DataAccessor make_sparse_accessor(const CudaSparseMatrixCSR& sparse_matrix) {
    return DataAccessor(sparse_matrix);
}

} // namespace cuda
} // namespace rf

#endif // RF_DATA_ACCESSOR_CUH

