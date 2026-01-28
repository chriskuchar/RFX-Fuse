// Helper function to get low-rank factors from CPU code
// Defined in CUDA file to avoid CUDA header inclusion in CPU compilation
#include <cuda_fp16.h>  // MUST be before cublas_v2.h
#include "rf_proximity_lowrank.hpp"
#include "rf_lowrank_helpers.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace rf {
namespace cuda {

// Helper function to copy low-rank factors from GPU to host
bool get_lowrank_factors_host(void* lowrank_proximity_ptr, 
                               dp_t** A_host, dp_t** B_host, 
                               integer_t* r, integer_t nsamples) {
    if (lowrank_proximity_ptr == nullptr) {
        return false;
    }
    
    try {
        LowRankProximityMatrix* lowrank_prox = 
            static_cast<LowRankProximityMatrix*>(lowrank_proximity_ptr);
        
        // Get GPU pointers and rank
        dp_t* A_gpu = nullptr;
        dp_t* B_gpu = nullptr;
        integer_t factor_rank = 0;
        lowrank_prox->get_factors(&A_gpu, &B_gpu, &factor_rank);
        
        if (!A_gpu || !B_gpu || factor_rank == 0) {
            return false;
        }
        
        *r = factor_rank;
        size_t factor_size = static_cast<size_t>(nsamples) * factor_rank;
        
        // Allocate host memory
        *A_host = new dp_t[factor_size];
        *B_host = new dp_t[factor_size];
        
        // Copy from GPU to host
        cudaMemcpy(*A_host, A_gpu, factor_size * sizeof(dp_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(*B_host, B_gpu, factor_size * sizeof(dp_t), cudaMemcpyDeviceToHost);
        
        // Synchronize to ensure copy completed
        cudaStreamSynchronize(0);  // Use stream sync for Jupyter safety
        
        return true;
    } catch (...) {
        return false;
    }
}

// Helper function to compute MDS coordinates from low-rank factors
std::vector<double> compute_mds_from_factors_host(void* lowrank_proximity_ptr, integer_t k) {
    if (lowrank_proximity_ptr == nullptr) {
        return std::vector<double>();  // Return empty vector on failure
    }
    
    try {
        LowRankProximityMatrix* lowrank_prox = 
            static_cast<LowRankProximityMatrix*>(lowrank_proximity_ptr);
        
        // Call the MDS computation method with k dimensions
        return lowrank_prox->compute_mds_from_factors(k);
    } catch (...) {
        return std::vector<double>();  // Return empty vector on error
    }
}

// Helper function to reconstruct full proximity matrix from low-rank factors (with RF-GAP normalization if enabled)
// WARNING: This function allocates O(n²) memory and can crash the system for large datasets!
bool reconstruct_proximity_matrix_host(void* lowrank_proximity_ptr, 
                                       dp_t** output_prox, 
                                       integer_t nsamples) {
    if (lowrank_proximity_ptr == nullptr) {
        return false;
    }
    
    // WARNING: Reconstructing full proximity matrix requires O(n²) memory!
    size_t matrix_size_bytes = static_cast<size_t>(nsamples) * nsamples * sizeof(dp_t);
    size_t matrix_size_gb = matrix_size_bytes / (1024ULL * 1024ULL * 1024ULL);
    
    if (nsamples > 10000) {
        std::cerr << "WARNING: Reconstructing full proximity matrix for " << nsamples 
                  << " samples requires ~" << matrix_size_gb 
                  << " GB of memory. This may crash your system!" << std::endl;
        std::cerr << "WARNING: Consider using low-rank factors or MDS from factors instead." << std::endl;
    }
    
    if (nsamples > 50000) {
        std::cerr << "ERROR: Dataset too large (" << nsamples 
                  << " samples) for full matrix reconstruction. Requires ~" << matrix_size_gb 
                  << " GB. Aborting to prevent system crash." << std::endl;
        return false;
    }
    
    try {
        LowRankProximityMatrix* lowrank_prox = 
            static_cast<LowRankProximityMatrix*>(lowrank_proximity_ptr);
        
        // Allocate GPU memory for reconstruction
        dp_t* prox_gpu = nullptr;
        cudaError_t err = cudaMalloc(&prox_gpu, nsamples * nsamples * sizeof(dp_t));
        if (err != cudaSuccess) {
            std::cerr << "ERROR: Failed to allocate GPU memory for proximity matrix reconstruction. "
                      << "Required: ~" << matrix_size_gb << " GB. Error: " 
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Use RAII-style cleanup to ensure memory is freed even if exceptions occur
        struct CudaMemoryGuard {
            dp_t* ptr;
            CudaMemoryGuard(dp_t* p) : ptr(p) {}
            ~CudaMemoryGuard() {
                if (ptr != nullptr) {
                    cudaError_t free_err = cudaFree(ptr);
                    if (free_err != cudaSuccess) {
                        // Suppress common errors in Jupyter (corrupted context)
                        if (free_err != cudaErrorInvalidValue && 
                            free_err != cudaErrorInvalidDevicePointer) {
                            std::cerr << "Warning: Failed to free GPU memory: " 
                                      << cudaGetErrorString(free_err) << std::endl;
                        }
                        cudaGetLastError();  // Clear error
                    }
                }
            }
        } guard(prox_gpu);
        
        // Reconstruct full matrix (with RF-GAP normalization if oob_counts_rfgap_ is set)
        lowrank_prox->reconstruct_full_matrix(prox_gpu, nsamples, QuantizationLevel::FP32);
        
        // Allocate host memory
        *output_prox = new dp_t[nsamples * nsamples];
        
        // Copy from GPU to host
        cudaError_t copy_err = cudaMemcpy(*output_prox, prox_gpu, nsamples * nsamples * sizeof(dp_t), cudaMemcpyDeviceToHost);
        if (copy_err != cudaSuccess) {
            delete[] *output_prox;
            *output_prox = nullptr;
            std::cerr << "ERROR: Failed to copy proximity matrix from GPU to host: " 
                      << cudaGetErrorString(copy_err) << std::endl;
            return false;
        }
        
        cudaStreamSynchronize(0);  // Use stream sync for Jupyter safety  // Ensure copy completes
        
        // Memory will be freed automatically by guard destructor
        
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace cuda
} // namespace rf

