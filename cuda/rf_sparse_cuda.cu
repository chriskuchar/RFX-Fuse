/**
 * @file rf_sparse_cuda.cu
 * @brief Implementation of GPU sparse matrix operations
 */

#include "rf_sparse_cuda.cuh"
#include <cstdio>

namespace rf {
namespace cuda {

void CudaSparseMatrixCSR::upload(const SparseMatrixCSR& host_sparse, cudaStream_t stream) {
    // Store dimensions
    nrows = host_sparse.nrows;
    ncols = host_sparse.ncols;
    nnz = host_sparse.nnz;
    
    if (nnz == 0) {
        d_data = nullptr;
        d_indices = nullptr;
        d_indptr = nullptr;
        return;
    }
    
    // Allocate GPU memory
    cudaError_t err;
    
    err = cudaMalloc(&d_data, nnz * sizeof(real_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_indices, nnz * sizeof(integer_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        d_data = nullptr;
        return;
    }
    
    err = cudaMalloc(&d_indptr, (nrows + 1) * sizeof(integer_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_indptr: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_indices);
        d_data = nullptr;
        d_indices = nullptr;
        return;
    }
    
    // Copy data to GPU (async with stream)
    cudaMemcpyAsync(d_data, host_sparse.data.data(), 
                    nnz * sizeof(real_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_indices, host_sparse.indices.data(), 
                    nnz * sizeof(integer_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_indptr, host_sparse.indptr.data(), 
                    (nrows + 1) * sizeof(integer_t), cudaMemcpyHostToDevice, stream);
    
    // Sync to ensure upload complete
    cudaStreamSynchronize(stream);
}

void CudaSparseMatrixCSR::free() {
    if (d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    }
    if (d_indices) {
        cudaFree(d_indices);
        d_indices = nullptr;
    }
    if (d_indptr) {
        cudaFree(d_indptr);
        d_indptr = nullptr;
    }
    nrows = 0;
    ncols = 0;
    nnz = 0;
}

} // namespace cuda
} // namespace rf

