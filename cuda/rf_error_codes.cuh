/**
 * @file rf_error_codes.cuh
 * @brief Error codes for CUDA kernels - NO printf/cout allowed in kernels!
 * 
 * All kernels must use error codes instead of printing for debugging.
 * This prevents hangs caused by printf buffer issues in CUDA.
 */

#ifndef RF_ERROR_CODES_CUH
#define RF_ERROR_CODES_CUH

#include "rf_types.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace rf {
namespace cuda {

// Error codes for kernel debugging
// Use atomicExch to set these in kernels, check on host after sync
enum CudaErrorCode : integer_t {
    CUDA_OK = 0,
    CUDA_ERROR_INVALID_INDEX = 1,
    CUDA_ERROR_DIVIDE_BY_ZERO = 2,
    CUDA_ERROR_OUT_OF_BOUNDS = 3,
    CUDA_ERROR_INVALID_NODE = 4,
    CUDA_ERROR_NULL_POINTER = 5,
    CUDA_ERROR_INVALID_SAMPLE = 6,
    CUDA_ERROR_TRAVERSAL_LIMIT = 7,
    CUDA_ERROR_INVALID_FEATURE = 8,
    CUDA_ERROR_SPARSE_ACCESS = 9,
    CUDA_ERROR_MEMORY_ALLOCATION = 10
};

// Convert error code to string (host-side only)
inline const char* cuda_error_string(integer_t code) {
    switch (code) {
        case CUDA_OK: return "OK";
        case CUDA_ERROR_INVALID_INDEX: return "Invalid index";
        case CUDA_ERROR_DIVIDE_BY_ZERO: return "Divide by zero";
        case CUDA_ERROR_OUT_OF_BOUNDS: return "Out of bounds";
        case CUDA_ERROR_INVALID_NODE: return "Invalid node";
        case CUDA_ERROR_NULL_POINTER: return "Null pointer";
        case CUDA_ERROR_INVALID_SAMPLE: return "Invalid sample";
        case CUDA_ERROR_TRAVERSAL_LIMIT: return "Traversal limit exceeded";
        case CUDA_ERROR_INVALID_FEATURE: return "Invalid feature";
        case CUDA_ERROR_SPARSE_ACCESS: return "Sparse matrix access error";
        case CUDA_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        default: return "Unknown error";
    }
}

} // namespace cuda
} // namespace rf

// Host-side error checking macro - use after cudaStreamSynchronize
#define CHECK_KERNEL_ERROR(d_error, stream) \
    do { \
        cudaStreamSynchronize(stream); \
        rf::integer_t h_error = 0; \
        cudaMemcpy(&h_error, d_error, sizeof(rf::integer_t), cudaMemcpyDeviceToHost); \
        if (h_error != rf::cuda::CUDA_OK) { \
            fprintf(stderr, "Kernel error %d (%s) at %s:%d\n", \
                    static_cast<int>(h_error), \
                    rf::cuda::cuda_error_string(h_error), \
                    __FILE__, __LINE__); \
        } \
    } while(0)

// Macro to check CUDA API calls (not kernel errors)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA API error %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while(0)
#endif

#endif // RF_ERROR_CODES_CUH

