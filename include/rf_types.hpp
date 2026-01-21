#ifndef RF_TYPES_HPP
#define RF_TYPES_HPP

#include <cstdint>
#include <vector>

namespace rf {

// Precision types
using float32 = float;
using float64 = double;
using int32 = int32_t;

// Default precision (matches Fortran REAL and INTEGER)
using real_t = float32;
using integer_t = int32;

// Double precision (matches Fortran DOUBLE PRECISION)
using dp_t = float64;

// FP16 type for CUDA (will be __half in CUDA code)
#ifdef __CUDACC__
#include <cuda_fp16.h>
using fp16_t = __half;
#else
using fp16_t = uint16_t;  // Placeholder for non-CUDA builds
#endif

// ============================================================================
// SPARSE MATRIX SUPPORT (CSR format)
// ============================================================================
// CSR (Compressed Sparse Row) format:
//   - data[nnz]: non-zero values
//   - indices[nnz]: column indices for each non-zero value
//   - indptr[nrows+1]: row pointers (indptr[i] to indptr[i+1] gives range for row i)
//
// To access row i: data[indptr[i]:indptr[i+1]], indices[indptr[i]:indptr[i+1]]
// This is efficient for row-wise operations (perfect for RF sample iteration)
// ============================================================================

struct SparseMatrixCSR {
    std::vector<real_t> data;       // Non-zero values (size = nnz)
    std::vector<integer_t> indices; // Column indices (size = nnz)
    std::vector<integer_t> indptr;  // Row pointers (size = nrows + 1)
    integer_t nrows;                // Number of rows (samples)
    integer_t ncols;                // Number of columns (features)
    integer_t nnz;                  // Number of non-zeros
    
    // Default constructor
    SparseMatrixCSR() : nrows(0), ncols(0), nnz(0) {}
    
    // Constructor from arrays
    SparseMatrixCSR(const real_t* data_ptr, const integer_t* indices_ptr, 
                    const integer_t* indptr_ptr, integer_t nrows_, integer_t ncols_, integer_t nnz_)
        : nrows(nrows_), ncols(ncols_), nnz(nnz_) {
        data.assign(data_ptr, data_ptr + nnz_);
        indices.assign(indices_ptr, indices_ptr + nnz_);
        indptr.assign(indptr_ptr, indptr_ptr + nrows_ + 1);
    }
    
    // Get value at (row, col) - O(log(nnz_per_row)) with binary search, O(nnz_per_row) linear
    inline real_t get(integer_t row, integer_t col) const {
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) return 0.0f;
        
        integer_t start = indptr[row];
        integer_t end = indptr[row + 1];
        
        // Linear search (fast for sparse rows with few elements)
        for (integer_t i = start; i < end; ++i) {
            if (indices[i] == col) return data[i];
            if (indices[i] > col) break;  // Indices are sorted, early exit
        }
        return 0.0f;  // Not found = zero
    }
    
    // Check if matrix is valid
    inline bool is_valid() const {
        return nrows > 0 && ncols > 0 && 
               indptr.size() == static_cast<size_t>(nrows + 1) &&
               data.size() == static_cast<size_t>(nnz) &&
               indices.size() == static_cast<size_t>(nnz);
    }
    
    // Get density (nnz / total_elements)
    inline double density() const {
        if (nrows == 0 || ncols == 0) return 0.0;
        return static_cast<double>(nnz) / (static_cast<double>(nrows) * ncols);
    }
};

// ============================================================================
// DATA MATRIX WRAPPER - Unified interface for dense/sparse data
// ============================================================================
// This wrapper allows tree growing code to work with both dense and sparse matrices
// using the same interface, with automatic dispatch to the appropriate method.
// ============================================================================

struct DataMatrix {
    // Dense data (row-major: X[sample * ncols + feature])
    const real_t* dense_data = nullptr;
    
    // Sparse data (CSR format)
    const SparseMatrixCSR* sparse_data = nullptr;
    
    // Dimensions
    integer_t nrows = 0;
    integer_t ncols = 0;
    
    // Is this sparse?
    bool is_sparse = false;
    
    // Constructor for dense data
    DataMatrix(const real_t* data, integer_t nrows_, integer_t ncols_)
        : dense_data(data), nrows(nrows_), ncols(ncols_), is_sparse(false) {}
    
    // Constructor for sparse data
    DataMatrix(const SparseMatrixCSR* sparse, integer_t nrows_, integer_t ncols_)
        : sparse_data(sparse), nrows(nrows_), ncols(ncols_), is_sparse(true) {}
    
    // Get value at (row, col) - works for both dense and sparse
    inline real_t get(integer_t row, integer_t col) const {
        if (is_sparse && sparse_data != nullptr) {
            return sparse_data->get(row, col);
        } else if (dense_data != nullptr) {
            return dense_data[row * ncols + col];
        }
        return 0.0f;
    }
    
    // Get pointer to row (only for dense, returns nullptr for sparse)
    inline const real_t* row_ptr(integer_t row) const {
        if (!is_sparse && dense_data != nullptr) {
            return dense_data + row * ncols;
        }
        return nullptr;
    }
};

} // namespace rf

#endif // RF_TYPES_HPP
