/**
 * @file rf_sparse_cuda.cuh
 * @brief GPU-side CSR sparse matrix structure and operations
 * 
 * Provides efficient sparse matrix storage and access on GPU.
 * Uses binary search for element access - O(log(nnz_per_row)).
 */

#ifndef RF_SPARSE_CUDA_CUH
#define RF_SPARSE_CUDA_CUH

#include "rf_types.hpp"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

/**
 * GPU-side CSR (Compressed Sparse Row) sparse matrix
 * 
 * Storage format:
 * - d_data[nnz]: Non-zero values
 * - d_indices[nnz]: Column indices for each non-zero
 * - d_indptr[nrows+1]: Row pointers (start of each row in data/indices)
 * 
 * Element access: O(log(nnz_per_row)) via binary search
 */
struct CudaSparseMatrixCSR {
    real_t* d_data;        // Non-zero values on GPU
    integer_t* d_indices;  // Column indices on GPU
    integer_t* d_indptr;   // Row pointers on GPU
    integer_t nrows;       // Number of rows
    integer_t ncols;       // Number of columns
    integer_t nnz;         // Number of non-zeros
    
    // Default constructor
    __host__ CudaSparseMatrixCSR() 
        : d_data(nullptr), d_indices(nullptr), d_indptr(nullptr),
          nrows(0), ncols(0), nnz(0) {}
    
    /**
     * Device function: Get element at (row, col)
     * 
     * ROW-MAJOR ACCESS (matches CPU SparseMatrixCSR):
     *   - row = sample index
     *   - col = feature index
     *   - Usage: X.get(sample, feature) NOT X.get(feature, sample)
     * 
     * Uses binary search for efficient access: O(log(nnz_per_row))
     * Returns 0.0f if element not found (sparse zero)
     */
    __device__ real_t get(integer_t row, integer_t col) const {
        // Bounds check
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
            return 0.0f;
        }
        
        integer_t start = d_indptr[row];
        integer_t end = d_indptr[row + 1];
        
        // Binary search for column in this row
        while (start < end) {
            integer_t mid = (start + end) / 2;
            integer_t mid_col = d_indices[mid];
            
            if (mid_col == col) {
                return d_data[mid];
            } else if (mid_col < col) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        
        return 0.0f;  // Not found = sparse zero
    }
    
    /**
     * Device function: Get row pointer range
     * Returns start and end indices in d_data/d_indices for given row
     */
    __device__ void get_row_range(integer_t row, integer_t& start, integer_t& end) const {
        if (row < 0 || row >= nrows) {
            start = 0;
            end = 0;
            return;
        }
        start = d_indptr[row];
        end = d_indptr[row + 1];
    }
    
    /**
     * Device function: Get number of non-zeros in row
     */
    __device__ integer_t row_nnz(integer_t row) const {
        if (row < 0 || row >= nrows) return 0;
        return d_indptr[row + 1] - d_indptr[row];
    }
    
    /**
     * Host function: Check if valid
     */
    __host__ bool is_valid() const {
        return d_data != nullptr && d_indices != nullptr && 
               d_indptr != nullptr && nnz > 0;
    }
    
    /**
     * Host function: Upload from CPU SparseMatrixCSR
     * @param host_sparse CPU-side sparse matrix
     * @param stream CUDA stream for async copy
     */
    __host__ void upload(const SparseMatrixCSR& host_sparse, cudaStream_t stream = 0);
    
    /**
     * Host function: Free GPU memory
     */
    __host__ void free();
    
    /**
     * Host function: Get memory usage in bytes
     */
    __host__ size_t memory_bytes() const {
        return nnz * sizeof(real_t) +           // d_data
               nnz * sizeof(integer_t) +        // d_indices
               (nrows + 1) * sizeof(integer_t); // d_indptr
    }
};

} // namespace cuda
} // namespace rf

#endif // RF_SPARSE_CUDA_CUH

