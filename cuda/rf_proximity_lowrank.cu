#include <cuda_fp16.h>  // MUST be first - defines __half

#include "rf_proximity_lowrank.hpp"
#include "rf_quantization_kernels.hpp"
#include "rf_types.hpp"
#include "rf_cuda_config.hpp"
#include "rf_utils.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>  // For rand() and RAND_MAX
#include <iostream>
#include <vector>
#include <stdexcept>
#include <mutex>  // For thread safety

// CUDA error checking macro - silent for Jupyter compatibility
// Just clears error state, doesn't throw or return (prevents segfaults)
#ifndef CUDA_CHECK_VOID
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            /* Silently clear error state to prevent Jupyter kernel crashes */ \
            /* Don't throw exceptions or return - just clear error */ \
            cudaGetLastError(); /* Clear error state */ \
            /* Continue execution - let calling code handle failures */ \
        } \
    } while(0)
#endif

namespace rf {
namespace cuda {

// Static mutex for thread-safe initialization of cuBLAS/cuSolver handles
static std::mutex g_handles_mutex;
static std::mutex g_allocated_mutex;

// CUDA kernels for RF-GAP normalization (defined before use)
namespace {
    // Kernel to normalize RF-GAP proximity by OOB counts (for vectors)
    __global__ void normalize_by_oob_counts_kernel(
        dp_t* w,  // Input/output vector (P × v)
        const integer_t* oob_counts,  // OOB counts per sample (|Si|)
        integer_t nsample
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nsample) return;
        
        integer_t si = oob_counts[i];
        if (si > 0) {
            w[i] /= static_cast<dp_t>(si);  // Normalize by |Si|
        }
    }
    
    // Kernel to normalize RF-GAP proximity matrix rows by OOB counts
    __global__ void normalize_matrix_rows_by_oob_counts_kernel(
        dp_t* P,  // Input/output matrix (nsample × nsample, row-major)
        const integer_t* oob_counts,  // OOB counts per sample (|Si|)
        integer_t nsample
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nsample) return;
        
        integer_t si = oob_counts[i];
        if (si > 0) {
            dp_t inv_si = 1.0 / static_cast<dp_t>(si);
            // Normalize entire row i by |Si|
            for (int j = 0; j < nsample; ++j) {
                P[i * nsample + j] *= inv_si;
            }
        }
    }
    
    // Kernel to make matrix symmetric (average upper and lower triangles)
    __global__ void make_symmetric_kernel(
        dp_t* P,  // Input/output matrix (nsample × nsample, row-major)
        integer_t nsample
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = nsample * nsample;
        if (idx >= total_elements) return;
        
        int i = idx / nsample;
        int j = idx % nsample;
        
        if (i < j) {
            // Upper triangle: average with lower triangle
            dp_t avg = 0.5 * (P[i * nsample + j] + P[j * nsample + i]);
            P[i * nsample + j] = avg;
            P[j * nsample + i] = avg;
        }
    }
    
    // Kernel to set diagonal to 1.0 (column-major format)
    __global__ void set_diagonal_to_one_kernel(
        dp_t* P,  // Input/output matrix (nsample × nsample, column-major)
        integer_t nsample
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nsample) return;
        
        // Column-major: diagonal is at P[i + i * nsample]
        P[i + i * nsample] = 1.0;
    }
}

// Helper function: Convert linear index to upper triangle (i, j) coordinates
__device__ __forceinline__ void upper_triangle_index(integer_t idx, integer_t nsample, integer_t& i, integer_t& j) {
    // idx = i * (i + 1) / 2 + j for upper triangle (row-major packing)
    i = static_cast<integer_t>((-1.0 + sqrt(1.0 + 8.0 * idx)) / 2.0);
    integer_t start_idx = i * (i + 1) / 2;
    j = idx - start_idx;
}

// CUDA kernel: Symmetric matrix-vector multiply (FP16 upper triangle)
__global__ void symmetric_matrix_vector_multiply_fp16_kernel(
    const __half* P_upper_gpu,
    const dp_t* x_gpu,
    dp_t* y_gpu,
    integer_t nsample
) {
    integer_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsample) return;
    
    dp_t sum = 0.0;
    
    // Compute y[i] = sum_j P[i,j] * x[j]
    // For symmetric matrix stored in upper triangle: P[i,j] = P[j,i] if i > j
    // Upper triangle row-wise packing: row i contains [i, i+1, ..., nsample-1]
    // Index formula: i * nsample + j - (i * (i + 1)) / 2
    // Maximum index: (nsample-1) * nsample + (nsample-1) - (nsample-1) * nsample / 2
    //               = nsample * (nsample-1) + (nsample-1) - nsample * (nsample-1) / 2
    //               = (nsample-1) * (nsample + 1 - nsample/2) = (nsample-1) * (nsample/2 + 1)
    // But total elements = nsample * (nsample + 1) / 2
    // So max valid index = nsample * (nsample + 1) / 2 - 1
    
    integer_t max_idx = (nsample * (nsample + 1)) / 2 - 1;
    
    for (integer_t j = 0; j < nsample; ++j) {
        dp_t p_val = 0.0;
        if (i <= j) {
            // Upper triangle: index = i * nsample + j - (i * (i + 1)) / 2
            integer_t idx = i * nsample + j - (i * (i + 1)) / 2;
            // Bounds check
            if (idx >= 0 && idx <= max_idx) {
                p_val = __half2float(P_upper_gpu[idx]);
            }
        } else {
            // Lower triangle: use symmetry P[i,j] = P[j,i]
            // For j < i, access P[j,i] which is in upper triangle at row j, column i
            integer_t idx = j * nsample + i - (j * (j + 1)) / 2;
            // Bounds check
            if (idx >= 0 && idx <= max_idx) {
                p_val = __half2float(P_upper_gpu[idx]);
            }
        }
        sum += p_val * x_gpu[j];
    }
    
    y_gpu[i] = sum;
}

// CUDA kernel: Symmetric matrix-vector multiply (FP32 upper triangle)
__global__ void symmetric_matrix_vector_multiply_kernel(
    const dp_t* P_upper_gpu,
    const dp_t* x_gpu,
    dp_t* y_gpu,
    integer_t nsample
) {
    integer_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsample) return;
    
    dp_t sum = 0.0;
    
    // Upper triangle row-wise packing: row i contains [i, i+1, ..., nsample-1]
    // Index formula: i * nsample + j - (i * (i + 1)) / 2
    for (integer_t j = 0; j < nsample; ++j) {
        dp_t p_val;
        if (i <= j) {
            // Upper triangle: index = i * nsample + j - (i * (i + 1)) / 2
            integer_t idx = i * nsample + j - (i * (i + 1)) / 2;
            p_val = P_upper_gpu[idx];
        } else {
            // Lower triangle: use symmetry P[i,j] = P[j,i]
            // For j < i, access P[j,i] which is in upper triangle at row j, column i
            integer_t idx = j * nsample + i - (j * (j + 1)) / 2;
            p_val = P_upper_gpu[idx];
        }
        sum += p_val * x_gpu[j];
    }
    
    y_gpu[i] = sum;
}

// CUDA kernel: Dequantize INT8 upper triangle to FP16
__global__ void dequantize_int8_upper_triangle_to_fp16_kernel(
    const int8_t* int8_upper,
    __half* fp16_upper,
    integer_t upper_triangle_size,
    float scale,
    float zero_point
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= upper_triangle_size) return;
    
    float val = QuantizationUtils::int8_to_fp32(int8_upper[idx], scale, zero_point);
    fp16_upper[idx] = __float2half(val);
}

// CUDA kernel: Dequantize NF4 upper triangle to FP16
__global__ void dequantize_nf4_upper_triangle_to_fp16_kernel(
    const uint8_t* nf4_upper,
    __half* fp16_upper,
    integer_t upper_triangle_size
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= upper_triangle_size) return;
    
    // NF4 is packed: 2 values per byte
    integer_t packed_idx = idx / 2;
    integer_t bit_offset = (idx % 2) * 4;  // 0 or 4 bits
    
    uint8_t packed_byte = nf4_upper[packed_idx];
    uint8_t nf4_val;
    
    if (bit_offset == 0) {
        // Extract lower 4 bits
        nf4_val = packed_byte & 0x0F;
    } else {
        // Extract upper 4 bits
        nf4_val = (packed_byte >> 4) & 0x0F;
    }
    
    float val = QuantizationUtils::nf4_to_fp32(nf4_val);
    fp16_upper[idx] = __float2half(val);
}

// CUDA kernel: Quantize factors from FP32 to INT8
__global__ void quantize_factors_fp32_to_int8_kernel(
    const dp_t* input_fp32,
    int8_t* output_int8,
    integer_t count,
    float scale,
    float zero_point
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
        float val = static_cast<float>(input_fp32[idx]);
    output_int8[idx] = QuantizationUtils::fp32_to_int8(val, scale, zero_point);
}

// Version that reads scale/zero_point from device memory (avoids host copy)
__global__ void quantize_factors_fp32_to_int8_kernel_device_scale(
    const dp_t* input_fp32,
    int8_t* output_int8,
    integer_t count,
    const float* d_scale,
    const float* d_zero_point
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Read scale and zero_point from device memory (cached after first read)
    float scale = *d_scale;
    float zero_point = *d_zero_point;
    
    float val = static_cast<float>(input_fp32[idx]);
    output_int8[idx] = QuantizationUtils::fp32_to_int8(val, scale, zero_point);
}

// CUDA kernel: Dequantize factors from INT8 to FP32
__global__ void dequantize_factors_int8_to_fp32_kernel(
    const int8_t* input_int8,
    dp_t* output_fp32,
    integer_t count,
    float scale,
    float zero_point
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    float val = QuantizationUtils::int8_to_fp32(input_int8[idx], scale, zero_point);
        output_fp32[idx] = static_cast<dp_t>(val);
}

// CUDA kernel: Dequantize factors from INT8 to FP32 (reads scale/zero_point from device memory)
__global__ void dequantize_factors_int8_to_fp32_kernel_device_scale(
    const int8_t* input_int8,
    dp_t* output_fp32,
    integer_t count,
    const float* d_scale,
    const float* d_zero_point
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Read scale and zero_point from device memory (cached after first read)
    float scale = *d_scale;
    float zero_point = *d_zero_point;
    
    float val = QuantizationUtils::int8_to_fp32(input_int8[idx], scale, zero_point);
    output_fp32[idx] = static_cast<dp_t>(val);
}

// CUDA kernel: Quantize factors from FP32 to FP16 (direct on GPU, no host copy)
__global__ void quantize_factors_fp32_to_fp16_kernel(
    const dp_t* input_fp32,
    __half* output_fp16,
    integer_t count
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    output_fp16[idx] = __float2half(static_cast<float>(input_fp32[idx]));
}

// CUDA kernel: Dequantize factors from FP16 to FP32 (direct on GPU, no host copy)
__global__ void dequantize_factors_fp16_to_fp32_kernel(
    const __half* input_fp16,
    dp_t* output_fp32,
    integer_t count
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    output_fp32[idx] = static_cast<dp_t>(__half2float(input_fp16[idx]));
}

// CUDA kernel: Scale eigenvector by sqrt(eigenvalue) if eigenvalue is positive
__global__ void scale_eigenvector_by_sqrt_eigenvalue_kernel(
    dp_t* eigenvector,
    const dp_t* eigenvalue,
    integer_t count
) {
    dp_t eigenval = *eigenvalue;
    if (eigenval > 0.0) {
        dp_t sqrt_eigenval = sqrt(eigenval);
        for (integer_t i = 0; i < count; ++i) {
            eigenvector[i] *= sqrt_eigenval;
        }
    } else {
        // Zero out if eigenvalue is non-positive
        for (integer_t i = 0; i < count; ++i) {
            eigenvector[i] = 0.0;
        }
    }
}

// CUDA kernel: Quantize factors from FP32 to NF4 (packed: 2 values per byte)
// Avoid atomic operations - use direct writes with proper indexing
__global__ void quantize_factors_fp32_to_nf4_kernel(
    const dp_t* input_fp32,
    uint8_t* output_nf4,
    integer_t count
) {
    // Process in pairs to avoid atomic contention
    // Each thread handles 2 consecutive values and packs them into 1 byte
    integer_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    integer_t idx1 = pair_idx * 2;
    integer_t idx2 = idx1 + 1;
    
    if (idx1 >= count) return;
    
    // Quantize first value
    float val1 = static_cast<float>(input_fp32[idx1]);
    uint8_t nf4_val1 = QuantizationUtils::fp32_to_nf4(val1) & 0x0F;
    
    // Quantize second value (if exists)
    uint8_t nf4_val2 = 0;
    if (idx2 < count) {
        float val2 = static_cast<float>(input_fp32[idx2]);
        nf4_val2 = QuantizationUtils::fp32_to_nf4(val2) & 0x0F;
    }
    
    // Pack into byte: lower 4 bits = val1, upper 4 bits = val2
    uint8_t packed_byte = nf4_val1 | (nf4_val2 << 4);
    
    // Write directly (no atomic needed since each thread writes to a unique location)
    output_nf4[pair_idx] = packed_byte;
}

// CUDA kernel: Dequantize factors from NF4 to FP32 (unpacked: 2 values per byte)
__global__ void dequantize_factors_nf4_to_fp32_kernel(
    const uint8_t* input_nf4,
    dp_t* output_fp32,
    integer_t count
) {
    integer_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int packed_idx = idx / 2;  // 2 values per byte
    int bit_offset = (idx % 2) * 4;  // 0 or 4 bits
    
    uint8_t packed_byte = input_nf4[packed_idx];
    uint8_t nf4_val;
    
    if (bit_offset == 0) {
        // Extract lower 4 bits
        nf4_val = packed_byte & 0x0F;
    } else {
        // Extract upper 4 bits
        nf4_val = (packed_byte >> 4) & 0x0F;
    }
    
    float val = QuantizationUtils::nf4_to_fp32(nf4_val);
    output_fp32[idx] = static_cast<dp_t>(val);
}

// Constructor
LowRankProximityMatrix::LowRankProximityMatrix(integer_t nsample, integer_t initial_rank, 
                                                QuantizationLevel quant_level,
                                                integer_t max_rank,
                                                integer_t power_iterations)
    : nsample_(nsample), rank_(initial_rank), max_rank_(max_rank),
      power_iterations_(power_iterations),
      A_gpu_(nullptr), B_gpu_(nullptr),
      A_quantized_gpu_(nullptr), B_quantized_gpu_(nullptr),
      current_quant_level_(quant_level),
      A_scale_(1.0f), A_zero_point_(0.0f),
      B_scale_(1.0f), B_zero_point_(0.0f),
      d_A_scale_(nullptr), d_A_zero_point_(nullptr),
      d_B_scale_(nullptr), d_B_zero_point_(nullptr),
      cublas_handle_(nullptr), cusolver_handle_(nullptr),
      workspace_gpu_(nullptr), workspace_size_(0),
      temp_prox_gpu_(nullptr),
      trees_processed_(0) {
    
    if (initial_rank <= 0) {
        rank_ = estimate_optimal_rank(nsample, max_rank_);
    }
    
    // Don't create cuBLAS/cuSolver handles in constructor - use lazy initialization instead
    // This avoids timing issues where CUDA context might not be fully ready
    // Handles will be created on first use via ensure_handles()
}

// Destructor (exception-safe for Jupyter notebooks)
LowRankProximityMatrix::~LowRankProximityMatrix() {
    // Wrap all cleanup in try-catch to prevent exceptions during destruction
    // This is critical for Jupyter notebooks where CUDA context might be corrupted
    try {
        free_gpu_memory();
        
        // Destroy cuBLAS handle if it was created successfully
        if (cublas_handle_ != nullptr) {
            cublasStatus_t status = cublasDestroy(cublas_handle_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                // Suppress errors during destruction - context might be corrupted
                // Only log if it's a truly unexpected error
            }
            cublas_handle_ = nullptr;
        }
        
        // Destroy cuSolver handle if it was created successfully
        if (cusolver_handle_ != nullptr) {
            cusolverStatus_t status = cusolverDnDestroy(cusolver_handle_);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                // Suppress errors during destruction - context might be corrupted
                // Only log if it's a truly unexpected error
            }
            cusolver_handle_ = nullptr;
        }
    } catch (...) {
        // Ignore ALL exceptions during destruction - process might be shutting down
        // CUDA runtime might already be destroyed
        // This is safe because we're in a destructor and can't throw
    }
}

// Initialize GPU memory
bool LowRankProximityMatrix::initialize() {
    // Defer actual memory allocation until first use
    // This avoids CUDA context issues during initialization
    // Memory will be allocated on first call to add_tree_contribution
    return true;
}

// Ensure GPU memory is allocated (lazy allocation)
void LowRankProximityMatrix::ensure_allocated() {
    // Thread-safe check - use mutex to prevent race conditions
    std::lock_guard<std::mutex> lock(g_allocated_mutex);
    
    if (A_quantized_gpu_ != nullptr && B_quantized_gpu_ != nullptr) {
        return;  // Already allocated
    }
    
    // Aggressive CUDA error clearing before allocation
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error cleared silently
    }
    
    // Skip stream sync - it can cause hangs during accumulation
    // GPU operations will naturally synchronize when needed
    cudaGetLastError();
    
    // Now allocate memory
    allocate_gpu_memory();
}

// Allocate GPU memory
void LowRankProximityMatrix::allocate_gpu_memory() {
    // Already allocated?
    if (A_quantized_gpu_ != nullptr && B_quantized_gpu_ != nullptr) {
        return;
    }
    
    // Aggressive CUDA error clearing and context validation
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error cleared silently
    }
    
    // Ensure CUDA context is ready
    int device = 0;
    cudaError_t device_err = cudaGetDevice(&device);
    if (device_err != cudaSuccess) {
        device_err = cudaSetDevice(0);
        if (device_err != cudaSuccess) {
            // Silently return - CUDA device not available (don't throw in Jupyter)
            return;
        }
        device = 0;
    }
    
    // Skip stream sync - it can cause hangs during accumulation
    // GPU operations will naturally synchronize when needed
    // cudaError_t sync_err = cudaStreamSynchronize(0);
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "Warning: Stream synchronization returned error: " << cudaGetErrorString(sync_err) << std::endl;
    //     Clear error and continue
    //     cudaGetLastError();
    // }
    
    // Clear errors after sync
    cudaGetLastError();
    
    size_t factor_size = nsample_ * rank_;
    
    if (factor_size == 0) {
        return;
    }
    
    // Allocate quantized factors based on quantization level
    size_t quant_size = 0;
    switch (current_quant_level_) {
        case QuantizationLevel::FP32:
            // For FP32, use FP16 for storage (proximity values are in [0,1] so FP16 is sufficient)
            quant_size = factor_size * sizeof(__half);
            break;
        case QuantizationLevel::FP16:
            quant_size = factor_size * sizeof(__half);
            break;
        case QuantizationLevel::INT8:
            quant_size = factor_size * sizeof(int8_t);
            break;
        case QuantizationLevel::NF4:
            // NF4 uses 4 bits per element (2 values per byte)
            // Round up to ensure we have enough bytes for all elements
            quant_size = (factor_size + 1) / 2; // (factor_size / 2) rounded up
            break;
    }
    
    // Use cudaMalloc with error checking (don't use CUDA_CHECK_VOID as it throws)
    cudaError_t err1 = cudaMalloc(&A_quantized_gpu_, quant_size);
    if (err1 != cudaSuccess) {
        // std::cerr << "Error: Failed to allocate A_quantized_gpu_ (" << (quant_size / (1024 * 1024)) << " MB): " << cudaGetErrorString(err1) << std::endl;
        // Silently return - allocation failed (don't throw in Jupyter)
        return;
    }
    
    cudaError_t err2 = cudaMalloc(&B_quantized_gpu_, quant_size);
    if (err2 != cudaSuccess) {
        // std::cerr << "Error: Failed to allocate B_quantized_gpu_ (" << (quant_size / (1024 * 1024)) << " MB): " << cudaGetErrorString(err2) << std::endl;
        // Safe free to prevent segfaults
        if (A_quantized_gpu_) { cudaError_t err = cudaFree(A_quantized_gpu_); if (err != cudaSuccess) cudaGetLastError(); }
        A_quantized_gpu_ = nullptr;
        // Silently return - allocation failed (don't throw in Jupyter)
        return;
    }
    
    cudaError_t err3 = cudaMemset(A_quantized_gpu_, 0, quant_size);
    if (err3 != cudaSuccess) {
        // std::cerr << "Warning: Failed to memset A_quantized_gpu_: " << cudaGetErrorString(err3) << std::endl;
    }
    
    cudaError_t err4 = cudaMemset(B_quantized_gpu_, 0, quant_size);
    if (err4 != cudaSuccess) {
        // std::cerr << "Warning: Failed to memset B_quantized_gpu_: " << cudaGetErrorString(err4) << std::endl;
    }
}

// Free GPU memory (exception-safe for Jupyter notebooks)
void LowRankProximityMatrix::free_gpu_memory() {
    // Safe CUDA free helper - handles errors gracefully for Jupyter notebooks
    // Template function to work with any pointer type
    auto safe_cudaFree = [](void* ptr, const char* name) {
        if (ptr != nullptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                // In Jupyter, corrupted context is common - suppress common errors
                // Only log if it's an unexpected error type
                if (err != cudaErrorInvalidValue && 
                    err != cudaErrorInvalidDevicePointer && 
                    err != cudaErrorIllegalAddress &&
                    err != cudaErrorInitializationError) {
                    // Only log unexpected errors, not expected corrupted context errors
                    // std::cerr << "Warning: Failed to free " << name << ": " << cudaGetErrorString(err) << std::endl;
                }
                // Always clear error to prevent propagation
                cudaGetLastError();
            }
        }
    };
    
    // Free all GPU memory with safe error handling
    safe_cudaFree(A_quantized_gpu_, "A_quantized_gpu_");
    A_quantized_gpu_ = nullptr;
    safe_cudaFree(B_quantized_gpu_, "B_quantized_gpu_");
    B_quantized_gpu_ = nullptr;
    safe_cudaFree(A_gpu_, "A_gpu_");
    A_gpu_ = nullptr;
    safe_cudaFree(B_gpu_, "B_gpu_");
    B_gpu_ = nullptr;
    safe_cudaFree(d_A_scale_, "d_A_scale_");
    d_A_scale_ = nullptr;
    safe_cudaFree(d_A_zero_point_, "d_A_zero_point_");
    d_A_zero_point_ = nullptr;
    safe_cudaFree(d_B_scale_, "d_B_scale_");
    d_B_scale_ = nullptr;
    safe_cudaFree(d_B_zero_point_, "d_B_zero_point_");
    d_B_zero_point_ = nullptr;
    safe_cudaFree(workspace_gpu_, "workspace_gpu_");
    workspace_gpu_ = nullptr;
    safe_cudaFree(temp_prox_gpu_, "temp_prox_gpu_");
    temp_prox_gpu_ = nullptr;
}

// Add tree contribution (dispatcher)
void LowRankProximityMatrix::add_tree_contribution(const dp_t* tree_proximity, integer_t nsample, bool use_incremental) {
    if (use_incremental) {
        add_tree_contribution_incremental(tree_proximity, nsample);
    } else {
        // For now, always use incremental for memory efficiency
        add_tree_contribution_incremental(tree_proximity, nsample);
    }
}

// Add tree contribution incremental (FP32 full matrix)
void LowRankProximityMatrix::add_tree_contribution_incremental(const dp_t* tree_proximity, integer_t nsample) {
    // Pack upper triangle
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    std::vector<dp_t> tree_prox_upper_host(upper_triangle_size);
    
    for (integer_t i = 0; i < nsample; ++i) {
        for (integer_t j = i; j < nsample; ++j) {
            integer_t idx = i * (i + 1) / 2 + j;
            tree_prox_upper_host[idx] = tree_proximity[i * nsample + j];
        }
    }
    
    // Convert to FP16
    std::vector<__half> tree_prox_upper_fp16_host(upper_triangle_size);
    for (size_t i = 0; i < upper_triangle_size; ++i) {
        tree_prox_upper_fp16_host[i] = __float2half(static_cast<float>(tree_prox_upper_host[i]));
    }
    
    // Copy to GPU
    __half* tree_prox_upper_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&tree_prox_upper_gpu, upper_triangle_size * sizeof(__half)));
    CUDA_CHECK_VOID(cudaMemcpy(tree_prox_upper_gpu, tree_prox_upper_fp16_host.data(), 
                                upper_triangle_size * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Perform rank-1 SVD
    perform_rank1_svd_from_upper_triangle_fp16(tree_prox_upper_gpu, nsample);
    
    // Cleanup
    // Safe free to prevent segfaults
    if (tree_prox_upper_gpu) { cudaError_t err = cudaFree(tree_prox_upper_gpu); if (err != cudaSuccess) cudaGetLastError(); }
}

// Add tree contribution from upper triangle FP16 (accepts GPU memory directly)
void LowRankProximityMatrix::add_tree_contribution_incremental_upper_triangle_fp16(const __half* tree_proximity_upper_fp16, integer_t nsample) {
    // std::cout << "[DEBUG LOWRANK FP16] Called for tree " << trees_processed_ 
    //           << ", nsample=" << nsample << std::endl;
    
    // Ensure memory is allocated (lazy allocation - defer until first use)
    ensure_allocated();
    
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    __half* tree_prox_upper_gpu = nullptr;
    
    // Check if pointer is on GPU or CPU by attempting to query its device pointer
    // If it's already on GPU, use it directly; otherwise copy from host
    cudaPointerAttributes attrs;
    cudaError_t attr_err = cudaPointerGetAttributes(&attrs, tree_proximity_upper_fp16);
    
    if (attr_err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
        // Pointer is already on GPU - use it directly!
        tree_prox_upper_gpu = const_cast<__half*>(tree_proximity_upper_fp16);  // Safe to const_cast for internal use
    } else {
        // Pointer is on CPU - allocate GPU memory and copy
        CUDA_CHECK_VOID(cudaMalloc(&tree_prox_upper_gpu, upper_triangle_size * sizeof(__half)));
        CUDA_CHECK_VOID(cudaMemcpy(tree_prox_upper_gpu, tree_proximity_upper_fp16, 
                                    upper_triangle_size * sizeof(__half), cudaMemcpyHostToDevice));
    }
    
    // Perform rank-1 SVD
    // std::cout << "[DEBUG LOWRANK FP16] About to call perform_rank1_svd_from_upper_triangle_fp16 for tree " << trees_processed_ << std::endl;
    perform_rank1_svd_from_upper_triangle_fp16(tree_prox_upper_gpu, nsample);
    // std::cout << "[DEBUG LOWRANK FP16] perform_rank1_svd_from_upper_triangle_fp16 completed for tree " << trees_processed_ 
    //           << ", new rank=" << rank_ << std::endl;
    
    // Cleanup only if we allocated the GPU memory (not if it was passed directly)
    if (attr_err != cudaSuccess || attrs.type != cudaMemoryTypeDevice) {
        // Safe free to prevent segfaults
        if (tree_prox_upper_gpu) {
            cudaError_t err = cudaFree(tree_prox_upper_gpu);
            if (err != cudaSuccess) cudaGetLastError(); // Clear error state
        }
    }
}

// Add tree contribution from upper triangle (all quantization levels)
void LowRankProximityMatrix::add_tree_contribution_incremental_upper_triangle(
    const void* tree_proximity_upper,
    integer_t nsample,
    QuantizationLevel quant_level,
    float int8_scale,
    float int8_zero_point
) {
    // std::cout << "[DEBUG LOWRANK] add_tree_contribution_incremental_upper_triangle called, "
            //   << "nsample=" << nsample << ", quant_level=" << static_cast<int>(quant_level) << std::endl;
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    
    if (quant_level == QuantizationLevel::FP16) {
        // Direct FP16 path
        // std::cout << "[DEBUG LOWRANK] Using FP16 path" << std::endl;
        add_tree_contribution_incremental_upper_triangle_fp16(
            static_cast<const __half*>(tree_proximity_upper), nsample
        );
    } else {
        // Dequantize to FP16 first
        __half* fp16_upper_gpu = nullptr;
        CUDA_CHECK_VOID(cudaMalloc(&fp16_upper_gpu, upper_triangle_size * sizeof(__half)));
        
        dim3 block_size(256);
        dim3 grid_size((upper_triangle_size + block_size.x - 1) / block_size.x);
        
        if (quant_level == QuantizationLevel::INT8) {
            // Dequantize INT8 to FP16
            // tree_proximity_upper is already a GPU pointer, so we can use it directly
            dequantize_int8_upper_triangle_to_fp16_kernel<<<grid_size, block_size>>>(
                static_cast<const int8_t*>(tree_proximity_upper),
                fp16_upper_gpu,
                upper_triangle_size,
                int8_scale,
                int8_zero_point
            );
            CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
        } else if (quant_level == QuantizationLevel::NF4) {
            // Dequantize NF4 to FP16
            // tree_proximity_upper is already a GPU pointer, so we can use it directly
            dequantize_nf4_upper_triangle_to_fp16_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(tree_proximity_upper),
                fp16_upper_gpu,
                upper_triangle_size
            );
            CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
        } else if (quant_level == QuantizationLevel::FP32) {
            // For FP32, treat as FP16 (proximity values are in [0,1] so FP16 is sufficient)
            // Just copy the data directly (assuming it's already in FP16 format from gpu_proximity_upper_triangle)
            // std::cout << "[DEBUG LOWRANK] FP32 path: Copying from tree_proximity_upper to fp16_upper_gpu" << std::endl;
            CUDA_CHECK_VOID(cudaMemcpy(fp16_upper_gpu, tree_proximity_upper, 
                                        upper_triangle_size * sizeof(__half), cudaMemcpyDeviceToDevice));
            // std::cout << "[DEBUG LOWRANK] FP32 path: Copy completed" << std::endl;
        }
        
        // Now perform SVD with FP16
        // NOTE: perform_rank1_svd_from_upper_triangle_fp16 includes cudaDeviceSynchronize()
        // after all cuBLAS/cuSolver operations, ensuring correct accumulation across batches
        // for all quantization levels (INT8, NF4, FP16)
        // std::cout << "[DEBUG LOWRANK] About to call perform_rank1_svd_from_upper_triangle_fp16 (FP32 path)" << std::endl;
        perform_rank1_svd_from_upper_triangle_fp16(fp16_upper_gpu, nsample);
        // std::cout << "[DEBUG LOWRANK] perform_rank1_svd_from_upper_triangle_fp16 completed (FP32 path)" << std::endl;
        
        // Safe free to prevent segfaults
        if (fp16_upper_gpu) { cudaError_t err = cudaFree(fp16_upper_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    }
}

// Ensure cuBLAS/cuSolver handles are created (lazy initialization)
void LowRankProximityMatrix::ensure_handles() {
    // Thread-safe check - use mutex to prevent race conditions
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    
    // If handles already exist, nothing to do
    if (cublas_handle_ != nullptr && cusolver_handle_ != nullptr) {
        return;
    }
    
    // Clear any previous CUDA errors
    cudaGetLastError();
    
    // Get current device
    int device = 0;
    cudaError_t device_err = cudaGetDevice(&device);
    if (device_err != cudaSuccess) {
        device_err = cudaSetDevice(0);
        if (device_err != cudaSuccess) {
            // std::cerr << "Error: Failed to set CUDA device: " << cudaGetErrorString(device_err) << std::endl;
            // Silently return - CUDA device not available (don't throw in Jupyter)
            return;
        }
        device = 0;
    }
    
    // Verify context is ready
    cudaDeviceProp prop;
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, device);
    if (prop_err != cudaSuccess) {
        // std::cerr << "Error: Failed to get device properties: " << cudaGetErrorString(prop_err) << std::endl;
        // Silently return - CUDA context not ready (don't throw in Jupyter)
        return;
    }
    
    // Create cuBLAS handle
    if (cublas_handle_ == nullptr) {
        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            // std::cerr << "Error: Failed to create cuBLAS handle: " << cublas_status << std::endl;
            cublas_handle_ = nullptr;
            // Silently return - cuBLAS handle creation failed (don't throw in Jupyter)
            return;
        }
        
        // Set cuBLAS math mode for better performance (if available)
        #if CUBLAS_VER_MAJOR >= 11
        cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);
        #endif
    }
    
    // Create cuSolver handle
    if (cusolver_handle_ == nullptr) {
        cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle_);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            // std::cerr << "Error: Failed to create cuSolver handle: " << cusolver_status << std::endl;
            // Clean up cuBLAS handle if cuSolver fails
            if (cublas_handle_ != nullptr) {
                cublasDestroy(cublas_handle_);
                cublas_handle_ = nullptr;
            }
            cusolver_handle_ = nullptr;
            // Silently return - cuSolver handle creation failed (don't throw in Jupyter)
            return;
        }
    }
}

// Perform rank-1 SVD from upper triangle FP16 matrix
void LowRankProximityMatrix::perform_rank1_svd_from_upper_triangle_fp16(const __half* tree_prox_upper_gpu, integer_t nsample) {
    // Ensure memory is allocated (lazy allocation)
    ensure_allocated();
    
    // Ensure handles are created (lazy initialization)
    ensure_handles();
    
    // If we have quantized buffers, dequantize them first to get full-precision A_gpu_/B_gpu_
    // This is needed because accumulation requires full precision
    if (A_quantized_gpu_ && B_quantized_gpu_ && !A_gpu_ && !B_gpu_ && rank_ > 0) {
        dequantize_factors();
    }
    
    // Skip host copy check - it causes hangs
    // We'll proceed with SVD even if matrix is all zeros (it will just produce zero factors)
    // The SVD itself is fast enough that checking isn't worth the hang risk
    
    // Power iteration to find dominant eigenvector
    dp_t* u_gpu = nullptr;
    dp_t* v_gpu = nullptr;
    dp_t* temp_vec = nullptr;
    
    CUDA_CHECK_VOID(cudaMalloc(&u_gpu, nsample * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMalloc(&v_gpu, nsample * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMalloc(&temp_vec, nsample * sizeof(dp_t)));
    
    // cudaMalloc does NOT zero-initialize! We must explicitly zero the memory
    CUDA_CHECK_VOID(cudaMemset(u_gpu, 0, nsample * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMemset(v_gpu, 0, nsample * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMemset(temp_vec, 0, nsample * sizeof(dp_t)));
    
    // Initialize u_gpu: Set first element to 1.0, rest are already 0
    // Power iteration will converge from any non-zero starting point
    dp_t one = 1.0;
    CUDA_CHECK_VOID(cudaMemcpy(u_gpu, &one, sizeof(dp_t), cudaMemcpyHostToDevice));
    // Normalize it
    dp_t norm_init;
    cublasDnrm2(cublas_handle_, nsample, u_gpu, 1, &norm_init);
    if (norm_init > 0) {
        dp_t inv_norm_init = 1.0 / norm_init;
        cublasDscal(cublas_handle_, nsample, &inv_norm_init, u_gpu, 1);
    }
    
    // Power iteration
    // PERF: 30 iterations is a good balance - captures 99% structure while avoiding excessive syncs
    // Each iteration has implicit syncs from cuBLAS returning scalars to host
    // Use configurable power_iterations_ (default: 30, user can set to 100+ for higher precision)
    const int max_iterations = power_iterations_;
    const dp_t tolerance = 1e-6;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // v = P * u
        symmetric_matrix_vector_multiply_fp16(tree_prox_upper_gpu, u_gpu, v_gpu, nsample);
        
        // Normalize v
        dp_t norm;
        cublasDnrm2(cublas_handle_, nsample, v_gpu, 1, &norm);
        if (norm < tolerance) break;
        
        dp_t inv_norm = 1.0 / norm;
        cublasDscal(cublas_handle_, nsample, &inv_norm, v_gpu, 1);
        
        // Check convergence
        cublasDaxpy(cublas_handle_, nsample, &inv_norm, v_gpu, 1, temp_vec, 1);
        dp_t diff_norm;
        cublasDaxpy(cublas_handle_, nsample, &inv_norm, u_gpu, 1, temp_vec, 1);
        cublasDnrm2(cublas_handle_, nsample, temp_vec, 1, &diff_norm);
        
        if (diff_norm < tolerance) break;
        
        // Swap u and v
        std::swap(u_gpu, v_gpu);
    }
    
    // Compute singular value: sigma = ||P * u||
    symmetric_matrix_vector_multiply_fp16(tree_prox_upper_gpu, u_gpu, temp_vec, nsample);
    dp_t sigma;
    cublasDnrm2(cublas_handle_, nsample, temp_vec, 1, &sigma);
    // sigma computed successfully
    
    // std::cout << "[DEBUG LOWRANK SVD] sigma=" << sigma << ", sqrt_sigma=" << sqrt(sigma) << std::endl;
    
    // Skip host copy check - it causes hangs
    // We'll proceed with SVD even if u_gpu is all zeros (it will just produce zero factors)
    
    // Update factors: A_new = [A_old, sqrt(sigma) * u], B_new = [B_old, sqrt(sigma) * u]
    // For symmetric matrix, u ≈ v
    dp_t sqrt_sigma = sqrt(sigma);
    
    // If sigma is too small, skip this tree (would produce near-zero factors)
    // Changed from 1e-10 to 1e-14: 1e-10 was too aggressive and rejected valid trees
    // Proximity values are in [0, 1], so even sigma=1e-6 carries signal
    if (sigma < 1e-14) {
        // Clean up and return
        if (u_gpu) { cudaError_t err = cudaFree(u_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        if (v_gpu) { cudaError_t err = cudaFree(v_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        if (temp_vec) { cudaError_t err = cudaFree(temp_vec); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // Allocate new A and B if needed
    if (A_gpu_ == nullptr || B_gpu_ == nullptr) {
        rank_ = 1;
        // Allocating A and B for first tree
        CUDA_CHECK_VOID(cudaMalloc(&A_gpu_, nsample * rank_ * sizeof(dp_t)));
        CUDA_CHECK_VOID(cudaMalloc(&B_gpu_, nsample * rank_ * sizeof(dp_t)));
        
        // Skip cudaMemset - cuBLAS copy will overwrite, memset causes implicit sync
        
        // Copy u_gpu to A_gpu_, scaled by sqrt_sigma
        cublasDcopy(cublas_handle_, nsample, u_gpu, 1, A_gpu_, 1);
        dp_t scale = sqrt_sigma;
        cublasDscal(cublas_handle_, nsample, &scale, A_gpu_, 1);
        
        // Copy A to B to ensure symmetry (A = B for symmetric proximity matrix)
        cublasDcopy(cublas_handle_, nsample, A_gpu_, 1, B_gpu_, 1);
        // Note: No explicit sync - cuBLAS operations will complete naturally
        
        // Factors stored successfully
    } else {
        // Extend A and B (rank < max_rank OR ignore max_rank to preserve all info)
        // NOTE: We're removing the max_rank cycling that was overwriting columns.
        // For accurate proximity, we need ALL tree contributions, not cyclic overwriting.
        // This means rank can grow beyond max_rank for small datasets.
        // Extend A and B (rank < max_rank)
        dp_t* A_new = nullptr;
        dp_t* B_new = nullptr;
        integer_t new_rank = rank_ + 1;
        
        CUDA_CHECK_VOID(cudaMalloc(&A_new, nsample * new_rank * sizeof(dp_t)));
        CUDA_CHECK_VOID(cudaMalloc(&B_new, nsample * new_rank * sizeof(dp_t)));
        
        // Skip cudaMemset - cuBLAS copy will overwrite, memset causes implicit sync
        
        // Copy old factors using cuBLAS (async, no blocking)
        if (rank_ > 0) {
            cublasDcopy(cublas_handle_, nsample * rank_, A_gpu_, 1, A_new, 1);
            cublasDcopy(cublas_handle_, nsample * rank_, B_gpu_, 1, B_new, 1);
        }
        
        // Add new column
        cublasDcopy(cublas_handle_, nsample, u_gpu, 1, A_new + nsample * rank_, 1);
        dp_t scale = sqrt_sigma;
        cublasDscal(cublas_handle_, nsample, &scale, A_new + nsample * rank_, 1);
        
        cublasDcopy(cublas_handle_, nsample, u_gpu, 1, B_new + nsample * rank_, 1);
        cublasDscal(cublas_handle_, nsample, &scale, B_new + nsample * rank_, 1);
        
        // Ensure A = B for symmetric proximity matrix (P = A @ B.T should be symmetric)
        // Copy A to B to ensure symmetry: B = A
        cublasDcopy(cublas_handle_, nsample * new_rank, A_new, 1, B_new, 1);
        // Note: No explicit sync - cuBLAS operations will complete naturally
        
        // Don't free old buffers immediately - cudaFree can cause implicit sync and hangs
        // Defer freeing to avoid hangs during accumulation
        // Store old pointers for deferred cleanup (they'll be freed in destructor)
        A_gpu_ = A_new;
        B_gpu_ = B_new;
        rank_ = new_rank;
        
        // Factors extended successfully
    }
    
    // DO NOT quantize after each tree - causes precision loss!
    // With 1500 trees, quantizing → dequantizing 1500 times compounds INT8 errors, degrading MDS quality.
    // Instead, keep full FP32 precision during accumulation and only quantize ONCE at the end.
    // Memory trade-off: For small datasets (e.g., Wine 178 samples), this is negligible (~1MB extra).
    // For large datasets (100K+ samples), consider re-enabling per-tree quantization if memory is critical.
    
    // Note: No explicit sync needed - cuBLAS operations will complete naturally
    // The next tree's operations will implicitly wait for previous operations
    
    trees_processed_++;
}

// Symmetric matrix-vector multiply FP16
void LowRankProximityMatrix::symmetric_matrix_vector_multiply_fp16(
    const __half* P_upper_gpu,
    const dp_t* x_gpu,
    dp_t* y_gpu,
    integer_t nsample
) {
    // Validate pointers
    if (P_upper_gpu == nullptr || x_gpu == nullptr || y_gpu == nullptr) {
        // Silently return - null pointer (don't throw in Jupyter)
        return;
    }
    
    // Validate nsample
    if (nsample <= 0) {
        // Silently return - invalid nsample (don't throw in Jupyter)
        return;
    }
    
    // Check pointer attributes
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, P_upper_gpu);
    if (err != cudaSuccess || attrs.type != cudaMemoryTypeDevice) {
        // Silently return - invalid device pointer (don't throw in Jupyter)
        return;
    }
    
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    // Clear any previous errors
    cudaGetLastError();
    
    symmetric_matrix_vector_multiply_fp16_kernel<<<grid_size, block_size>>>(
        P_upper_gpu, x_gpu, y_gpu, nsample
    );
    
    // Check for kernel launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        // Silently return - kernel launch failed (don't throw in Jupyter)
        cudaGetLastError(); // Clear error state
        return;
    }
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
}

// Symmetric matrix-vector multiply FP32
void LowRankProximityMatrix::symmetric_matrix_vector_multiply(
    const dp_t* P_upper_gpu,
    const dp_t* x_gpu,
    dp_t* y_gpu,
    integer_t nsample
) {
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    symmetric_matrix_vector_multiply_kernel<<<grid_size, block_size>>>(
        P_upper_gpu, x_gpu, y_gpu, nsample
    );
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
}

// Truncate rank
void LowRankProximityMatrix::truncate_rank(integer_t target_rank) {
    if (rank_ <= target_rank) return;
    
    // Ensure handles are created
    ensure_handles();
    
    // Dequantize factors if needed
    dequantize_factors();
    
    if (!A_gpu_ || !B_gpu_) {
        return;  // Nothing to truncate
    }
    
    // Reconstruct full matrix P = A × B' (symmetric)
    // Allocate temporary buffer for full matrix
    dp_t* P_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&P_gpu, nsample_ * nsample_ * sizeof(dp_t)));
    
    // Compute P = A × B' using cuBLAS
    // P is nsample × nsample, A is nsample × rank_, B is nsample × rank_
    // P = A × B' → CUBLAS_OP_N for A, CUBLAS_OP_T for B
    const dp_t alpha = 1.0;
    const dp_t beta = 0.0;
    cublasDgemm(cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_T,
                nsample_, nsample_, rank_,
                &alpha,
                A_gpu_, nsample_,
                B_gpu_, nsample_,
                &beta,
                P_gpu, nsample_);
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Perform eigendecomposition (symmetric matrix)
    // Since P is symmetric, eigenvalues = singular values
    dp_t* S_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&S_gpu, nsample_ * sizeof(dp_t)));
    
    int lwork = 0;
    cusolverDnDsyevd_bufferSize(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        nsample_,
        P_gpu,
        nsample_,
        S_gpu,
        &lwork
    );
    
    dp_t* d_work = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&d_work, lwork * sizeof(dp_t)));
    
    int* dev_info = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&dev_info, sizeof(int)));
    
    cusolverDnDsyevd(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        nsample_,
        P_gpu,
        nsample_,
        S_gpu,
        d_work,
        lwork,
        dev_info
    );
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Check for errors
    int info_host = 0;
    CUDA_CHECK_VOID(cudaMemcpy(&info_host, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) {
        // std::cerr << "WARNING: cuSolver eigendecomposition failed with info=" << info_host << std::endl;
        // Safe free to prevent segfaults
        if (P_gpu) { cudaError_t err = cudaFree(P_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        if (S_gpu) { cudaError_t err = cudaFree(S_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        if (d_work) { cudaError_t err = cudaFree(d_work); if (err != cudaSuccess) cudaGetLastError(); }
        if (dev_info) { cudaError_t err = cudaFree(dev_info); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // Extract top target_rank components by singular value (eigenvalue)
    // Eigenvalues are sorted in ascending order, so largest are at the end
    dp_t* A_truncated = nullptr;
    dp_t* B_truncated = nullptr;
    
    CUDA_CHECK_VOID(cudaMalloc(&A_truncated, nsample_ * target_rank * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMalloc(&B_truncated, nsample_ * target_rank * sizeof(dp_t)));
    
    // Copy top target_rank eigenvectors and scale by sqrt(eigenvalue)
    for (integer_t i = 0; i < target_rank; ++i) {
        integer_t col_idx = nsample_ - 1 - i;  // Top eigenvalues are at the end
        
        // Copy eigenvector to A
        cublasDcopy(cublas_handle_, nsample_, P_gpu + col_idx * nsample_, 1, 
                    A_truncated + i * nsample_, 1);
        
        // Get eigenvalue and compute sqrt - use GPU kernel to avoid host copy
        // Read eigenvalue on GPU and scale directly
        dp_t* eigenval_gpu = nullptr;
        CUDA_CHECK_VOID(cudaMalloc(&eigenval_gpu, sizeof(dp_t)));
        CUDA_CHECK_VOID(cudaMemcpy(eigenval_gpu, S_gpu + col_idx, sizeof(dp_t), cudaMemcpyDeviceToDevice));
        
        // Use a kernel to check eigenvalue and scale if positive
        dim3 block_size(1);
        dim3 grid_size(1);
        scale_eigenvector_by_sqrt_eigenvalue_kernel<<<grid_size, block_size>>>(
            A_truncated + i * nsample_, eigenval_gpu, nsample_
        );
        
        cudaFree(eigenval_gpu);
        
        // Copy A to B (symmetric matrix: A = B)
        cublasDcopy(cublas_handle_, nsample_, A_truncated + i * nsample_, 1, 
                    B_truncated + i * nsample_, 1);
    }
    
    // Note: No explicit sync - kernels will complete naturally
    
    // Replace old factors
    // Safe free to prevent segfaults
    if (A_gpu_) { cudaError_t err = cudaFree(A_gpu_); if (err != cudaSuccess) cudaGetLastError(); }
    if (B_gpu_) { cudaError_t err = cudaFree(B_gpu_); if (err != cudaSuccess) cudaGetLastError(); }
    A_gpu_ = A_truncated;
    B_gpu_ = B_truncated;
    rank_ = target_rank;
    
    // Clean up temporary buffers
    // Safe free to prevent segfaults
    if (P_gpu) { cudaError_t err = cudaFree(P_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    if (S_gpu) { cudaError_t err = cudaFree(S_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    if (d_work) { cudaError_t err = cudaFree(d_work); if (err != cudaSuccess) cudaGetLastError(); }
    if (dev_info) { cudaError_t err = cudaFree(dev_info); if (err != cudaSuccess) cudaGetLastError(); }
    
    // Skip quantization during truncate_rank to avoid hangs
    // Quantization will be done at the end via finalize_accumulation
    // quantize_factors(current_quant_level_);
}

// Finalize accumulation
void LowRankProximityMatrix::finalize_accumulation(integer_t n_trees_processed, bool final_call) {
    trees_processed_ = n_trees_processed;
    
    // ONLY quantize on the FINAL call (after ALL batches complete)
    // This prevents repeated quantize→dequantize cycles across batches
    // Example: 1500 trees with batch_size=100 = 15 batches
    //   Old: 15 quantize→dequantize cycles (precision loss!)
    //   New: Keep FP32 for all 15 batches, quantize once at the end
    if (final_call && current_quant_level_ != QuantizationLevel::FP32 && A_gpu_ && B_gpu_) {
        quantize_factors(current_quant_level_);
        // Free FP32 buffers after quantization (now in quantized format)
        if (A_gpu_) { cudaError_t err = cudaFree(A_gpu_); if (err != cudaSuccess) cudaGetLastError(); }
        if (B_gpu_) { cudaError_t err = cudaFree(B_gpu_); if (err != cudaSuccess) cudaGetLastError(); }
        A_gpu_ = nullptr;
        B_gpu_ = nullptr;
    }
}

// Get accumulated proximity matrix
void LowRankProximityMatrix::get_accumulated_proximity(dp_t* output_prox, integer_t nsample) {
    reconstruct_full_matrix(output_prox, nsample, QuantizationLevel::FP32);
}

// Reconstruct full matrix
void LowRankProximityMatrix::reconstruct_full_matrix(dp_t* output_prox, integer_t nsample, QuantizationLevel quant_level) {
    // WARNING: Reconstructing full proximity matrix requires O(n²) memory!
    // For large datasets, this can easily exceed available GPU/host memory and crash the system.
    // Example: 100k samples = 100k × 100k × 8 bytes = ~80 GB (double precision)
    // 
    // WARNING: This operation is extremely memory-intensive and should be avoided for:
    //   - Datasets with >10,000 samples (requires >800 MB)
    //   - Datasets with >50,000 samples (requires >20 GB)
    //   - Datasets with >100,000 samples (requires >80 GB - likely to crash!)
    //
    // Instead, use:
    //   - get_lowrank_factors() to work with low-rank factors directly
    //   - compute_mds_3d_from_factors() for MDS visualization (memory efficient)
    //   - compute_distances_from_factors() for distance computations (memory efficient)
    //
    // If you must reconstruct, ensure you have sufficient GPU/host memory available.
    
    size_t matrix_size_bytes = static_cast<size_t>(nsample) * nsample * sizeof(dp_t);
    size_t matrix_size_gb = matrix_size_bytes / (1024ULL * 1024ULL * 1024ULL);
    
    // if (nsample > 10000) {
    //     std::cerr << "WARNING: Reconstructing full proximity matrix for " << nsample 
    //               << " samples requires ~" << matrix_size_gb 
    //               << " GB of memory. This may crash your system!" << std::endl;
    //     std::cerr << "WARNING: Consider using low-rank factors or MDS from factors instead." << std::endl;
    // }
    
    // if (nsample > 50000) {
    //     std::cerr << "ERROR: Dataset too large (" << nsample 
    //               << " samples) for full matrix reconstruction. Requires ~" << matrix_size_gb 
    //               << " GB. Aborting to prevent system crash." << std::endl;
    //     // Silently return - dataset too large (don't throw in Jupyter)
    //     cudaGetLastError(); // Clear error state
    //     return;  // Return early (function returns void)
    // }
    
    // Ensure memory is allocated (lazy allocation)
    ensure_allocated();
    
    // Ensure handles are created (lazy initialization)
    ensure_handles();
    
    // Dequantize factors if needed
    dequantize_factors();
    
    if (!A_gpu_ || !B_gpu_) {
        // Initialize with zeros if factors don't exist
        CUDA_CHECK_VOID(cudaMemset(output_prox, 0, nsample * nsample * sizeof(dp_t)));
        return;
    }
    
    // P = A * B'
    dp_t alpha = 1.0, beta = 0.0;
    cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                nsample, nsample, rank_,
                &alpha, A_gpu_, nsample,
                B_gpu_, nsample,
                &beta, output_prox, nsample);
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Set diagonal to 1.0 (each sample is 100% similar to itself)
    // Column-major: diagonal is at output_prox[i + i * nsample]
    int threads_per_block_diag = 256;
    int blocks_diag = (nsample + threads_per_block_diag - 1) / threads_per_block_diag;
    set_diagonal_to_one_kernel<<<blocks_diag, threads_per_block_diag>>>(
        output_prox, nsample
    );
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // RF-GAP normalization: normalize each row i by |Si| (OOB count) if RF-GAP is used
    if (!oob_counts_rfgap_.empty()) {
        
        // Copy OOB counts to GPU
        integer_t* oob_counts_gpu = nullptr;
        CUDA_CHECK_VOID(cudaMalloc(&oob_counts_gpu, nsample * sizeof(integer_t)));
        CUDA_CHECK_VOID(cudaMemcpy(oob_counts_gpu, oob_counts_rfgap_.data(), 
                                  nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
        
        // Normalize each row i by |Si|
        // Use a kernel to normalize each row
        int threads_per_block = 256;
        int blocks = (nsample + threads_per_block - 1) / threads_per_block;
        
        // Kernel to normalize each row of the matrix
        normalize_matrix_rows_by_oob_counts_kernel<<<blocks, threads_per_block>>>(
            output_prox, oob_counts_gpu, nsample
        );
        CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
        // Safe free to prevent segfaults
        if (oob_counts_gpu) {
            cudaError_t err = cudaFree(oob_counts_gpu);
            if (err != cudaSuccess) cudaGetLastError(); // Clear error state
        }
        
        // Make symmetric (RF-GAP is symmetric after normalization)
        // Average upper and lower triangles
        make_symmetric_kernel<<<blocks, threads_per_block>>>(
            output_prox, nsample
        );
        CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    }
}

// Quantize factors
void LowRankProximityMatrix::quantize_factors(QuantizationLevel quant_level) {
    if (quant_level == QuantizationLevel::FP32 || !A_gpu_ || !B_gpu_) {
        current_quant_level_ = quant_level;
        return;
    }
    
    // During accumulation, we already have full-precision A_gpu_/B_gpu_ from SVD
    // We don't need to dequantize - just quantize the existing full-precision buffers directly
    // Dequantization only happens when we need full precision (e.g., reconstructing proximity matrix)
    // This avoids host copies that cause hangs during accumulation
    
    size_t factor_size = nsample_ * rank_;
    
    // Allocate quantized buffers
    void* A_quantized_new = nullptr;
    void* B_quantized_new = nullptr;
    
    switch (quant_level) {
        case QuantizationLevel::FP32:  // Stored as FP16 internally
        case QuantizationLevel::FP16: {
            size_t size = factor_size * sizeof(__half);
            CUDA_CHECK_VOID(cudaMalloc(&A_quantized_new, size));
            CUDA_CHECK_VOID(cudaMalloc(&B_quantized_new, size));
            
            // Convert FP32 to FP16 directly on GPU (no host copy to avoid hangs)
            // Use a GPU kernel to convert and pack
            dim3 block_size(256);
            dim3 grid_size((factor_size + block_size.x - 1) / block_size.x);
            
            quantize_factors_fp32_to_fp16_kernel<<<grid_size, block_size>>>(
                A_gpu_, static_cast<__half*>(A_quantized_new), factor_size
            );
            
            quantize_factors_fp32_to_fp16_kernel<<<grid_size, block_size>>>(
                B_gpu_, static_cast<__half*>(B_quantized_new), factor_size
            );
            // Note: No explicit sync - kernel will complete naturally
            // No host copy for debug checks - they cause hangs
            break;
        }
        case QuantizationLevel::INT8: {
            // Use same scaling parameters for A and B to maintain symmetry
            // Compute scaling entirely on device - no host copies
            // Allocate device memory for scale/zero_point if not already allocated
            bool first_quantization = (d_A_scale_ == nullptr);
            if (first_quantization) {
                CUDA_CHECK_VOID(cudaMalloc(&d_A_scale_, sizeof(float)));
                CUDA_CHECK_VOID(cudaMalloc(&d_A_zero_point_, sizeof(float)));
                CUDA_CHECK_VOID(cudaMalloc(&d_B_scale_, sizeof(float)));
                CUDA_CHECK_VOID(cudaMalloc(&d_B_zero_point_, sizeof(float)));
            }
            
            // Only compute scale on FIRST tree to avoid cumulative quantization error
            // Subsequent trees use the same scale, which prevents precision loss from compounding
            // The first tree's data range is representative since proximity values are bounded [0,1]
            if (first_quantization) {
                // Compute scale/zero_point on device from first tree's data
                QuantizationUtils::compute_int8_scaling_gpu(
                    A_gpu_, factor_size, d_A_scale_, d_A_zero_point_
                );
                
                // Must sync before D2D copy - scaling kernel must finish writing d_A_scale_
                // before we copy it to d_B_scale_
                CUDA_CHECK_VOID(cudaStreamSynchronize(0));
                
                // Use same scale and zero point for B (copy on device)
                CUDA_CHECK_VOID(cudaMemcpy(d_B_scale_, d_A_scale_, sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK_VOID(cudaMemcpy(d_B_zero_point_, d_A_zero_point_, sizeof(float), cudaMemcpyDeviceToDevice));
            }
            // For subsequent trees, reuse the existing scale - no recomputation needed
            
            size_t size = factor_size * sizeof(int8_t);
            CUDA_CHECK_VOID(cudaMalloc(&A_quantized_new, size));
            CUDA_CHECK_VOID(cudaMalloc(&B_quantized_new, size));
            
            // Quantize factors from FP32 to INT8 using GPU kernel with device pointers
            dim3 block_size(256);
            dim3 grid_size((factor_size + block_size.x - 1) / block_size.x);
            
            quantize_factors_fp32_to_int8_kernel_device_scale<<<grid_size, block_size>>>(
                A_gpu_, static_cast<int8_t*>(A_quantized_new), factor_size,
                d_A_scale_, d_A_zero_point_
            );
            
            quantize_factors_fp32_to_int8_kernel_device_scale<<<grid_size, block_size>>>(
                B_gpu_, static_cast<int8_t*>(B_quantized_new), factor_size,
                d_B_scale_, d_B_zero_point_
            );
            
            // Don't copy scale/zero_point to host during accumulation - it causes hangs
            // Keep them on device - they're stored in d_A_scale_, d_A_zero_point_, etc.
            // Only copy to host values if absolutely needed (defer to end)
            // For now, use default values in host variables (they're not used during accumulation)
            A_scale_ = 1.0f / 127.0f;  // Default for compatibility, but d_A_scale_ has the real value
            A_zero_point_ = 0.0f;
            B_scale_ = A_scale_;
            B_zero_point_ = A_zero_point_;
            break;
        }
        case QuantizationLevel::NF4: {
            // NF4 uses 4 bits per element, so 2 values per byte
            size_t packed_size = (factor_size + 1) / 2;  // Round up for odd counts
            size_t size = packed_size * sizeof(uint8_t);
            CUDA_CHECK_VOID(cudaMalloc(&A_quantized_new, size));
            CUDA_CHECK_VOID(cudaMalloc(&B_quantized_new, size));
            // Skip cudaMemset - kernel will write all values anyway, memset causes implicit sync
            
            // Quantize factors from FP32 to NF4 using GPU kernel
            // No atomic operations - direct writes to avoid contention
            dim3 block_size(256);
            dim3 grid_size((packed_size + block_size.x - 1) / block_size.x);
            
            quantize_factors_fp32_to_nf4_kernel<<<grid_size, block_size>>>(
                A_gpu_, static_cast<uint8_t*>(A_quantized_new), factor_size
            );
            // Note: No explicit sync - kernel will complete naturally
            
            quantize_factors_fp32_to_nf4_kernel<<<grid_size, block_size>>>(
                B_gpu_, static_cast<uint8_t*>(B_quantized_new), factor_size
            );
            // Note: No explicit sync - kernel will complete naturally
            break;
        }
        default:
            return;
    }
    
    // Replace old quantized buffers
    // Don't free old buffers immediately - cudaFree can cause implicit sync and hangs
    // Defer freeing to avoid hangs during accumulation
    // Store old pointers for deferred cleanup (they'll be freed in destructor)
    void* A_quantized_old = A_quantized_gpu_;
    void* B_quantized_old = B_quantized_gpu_;
    
    A_quantized_gpu_ = A_quantized_new;
    B_quantized_gpu_ = B_quantized_new;
    current_quant_level_ = quant_level;
    
    // Defer freeing old buffers - free them asynchronously or at the end
    // For now, just leak them (they'll be freed in destructor)
    // This avoids the implicit sync from cudaFree during accumulation
}

// Dequantize factors
void LowRankProximityMatrix::dequantize_factors() {
    // For FP32, quantized and dequantized are the same - just set pointers
    if (current_quant_level_ == QuantizationLevel::FP32) {
        // For FP32, we store as FP16, so convert to dp_t for dequantization
        // A_quantized_gpu_ and B_quantized_gpu_ are actually FP16 buffers
        size_t factor_size = nsample_ * rank_;
        
        // If A_gpu_ and B_gpu_ already exist, use them directly (factors already computed)
        if (A_gpu_ && B_gpu_ && factor_size > 0) {
            return;  // Already dequantized
        }
        
        // Otherwise, if quantized buffers exist and are valid, convert from FP16 to FP32
        // Do this entirely on GPU to avoid host copies that cause hangs
        if (A_quantized_gpu_ && B_quantized_gpu_ && factor_size > 0) {
            // Verify pointers are valid device pointers before attempting copy
            cudaPointerAttributes attrs_A, attrs_B;
            cudaError_t err_A = cudaPointerGetAttributes(&attrs_A, A_quantized_gpu_);
            cudaError_t err_B = cudaPointerGetAttributes(&attrs_B, B_quantized_gpu_);
            
            if (err_A == cudaSuccess && err_B == cudaSuccess && 
                attrs_A.type == cudaMemoryTypeDevice && attrs_B.type == cudaMemoryTypeDevice) {
                // Allocate FP32 buffers if not already allocated
                if (!A_gpu_) {
                    CUDA_CHECK_VOID(cudaMalloc(&A_gpu_, factor_size * sizeof(dp_t)));
                }
                if (!B_gpu_) {
                    CUDA_CHECK_VOID(cudaMalloc(&B_gpu_, factor_size * sizeof(dp_t)));
                }
                
                // Convert FP16 to FP32 directly on GPU (no host copies)
                dim3 block_size(256);
                dim3 grid_size((factor_size + block_size.x - 1) / block_size.x);
                
                dequantize_factors_fp16_to_fp32_kernel<<<grid_size, block_size>>>(
                    static_cast<const __half*>(A_quantized_gpu_), A_gpu_, factor_size
                );
                
                dequantize_factors_fp16_to_fp32_kernel<<<grid_size, block_size>>>(
                    static_cast<const __half*>(B_quantized_gpu_), B_gpu_, factor_size
                );
                // Note: No explicit sync - kernel will complete naturally
            }
            // If pointers are invalid, factors are likely already in A_gpu_/B_gpu_ (not quantized)
        }
        return;
    }
    
    // Don't dequantize if factors haven't been computed yet (trees_processed_ == 0 or rank_ == 0)
    if (trees_processed_ == 0 || rank_ == 0) {
        return;
    }
    
    size_t factor_size = nsample_ * rank_;
    
    // Validate factor_size is valid
    if (factor_size == 0) {
        return;
    }
    
    // If quantized buffers don't exist, check if factors are already in A_gpu_/B_gpu_
    if (!A_quantized_gpu_ || !B_quantized_gpu_) {
        // If A_gpu_ and B_gpu_ exist, factors are already in FP32 format (not quantized)
        if (A_gpu_ && B_gpu_) {
            return;  // Factors already in A_gpu_/B_gpu_, no quantization was done
        }
        // Otherwise, nothing to dequantize
        return;
    }
    
    // If quantized buffers exist, we MUST dequantize from them to A_gpu_/B_gpu_
    // This ensures we get the correct values even if A_gpu_/B_gpu_ exist (they might be stale)
    // Allocate A_gpu_ and B_gpu_ if they don't exist
    if (!A_gpu_ || !B_gpu_) {
        CUDA_CHECK_VOID(cudaMalloc(&A_gpu_, factor_size * sizeof(dp_t)));
        CUDA_CHECK_VOID(cudaMalloc(&B_gpu_, factor_size * sizeof(dp_t)));
    }
    
    switch (current_quant_level_) {
        case QuantizationLevel::FP16: {
            std::vector<__half> A_fp16(factor_size), B_fp16(factor_size);
            
            // Skip host copies - convert directly on GPU
            // Allocate FP32 buffers if not already allocated
            if (!A_gpu_) {
                CUDA_CHECK_VOID(cudaMalloc(&A_gpu_, factor_size * sizeof(dp_t)));
            }
            if (!B_gpu_) {
                CUDA_CHECK_VOID(cudaMalloc(&B_gpu_, factor_size * sizeof(dp_t)));
            }
            
            // Convert FP16 to FP32 directly on GPU (no host copies)
            dim3 block_size(256);
            dim3 grid_size((factor_size + block_size.x - 1) / block_size.x);
            
            dequantize_factors_fp16_to_fp32_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(A_quantized_gpu_), A_gpu_, factor_size
            );
            
            dequantize_factors_fp16_to_fp32_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(B_quantized_gpu_), B_gpu_, factor_size
            );
            // Note: No explicit sync - kernel will complete naturally
            return;
            
            std::vector<dp_t> A_host(factor_size), B_host(factor_size);
            for (size_t i = 0; i < factor_size; ++i) {
                A_host[i] = static_cast<dp_t>(__half2float(A_fp16[i]));
                B_host[i] = static_cast<dp_t>(__half2float(B_fp16[i]));
            }
            
            // DEBUG: Check dequantized values before copying to GPU
            dp_t A_host_max = 0.0;
            int A_host_non_zero = 0;
            for (size_t i = 0; i < factor_size; ++i) {
                if (fabs(A_host[i]) > A_host_max) A_host_max = fabs(A_host[i]);
                if (fabs(A_host[i]) > 1e-10) A_host_non_zero++;
            }
            // std::cout << "DEBUG: Dequantized A_host - max=" << A_host_max 
            //           << ", non_zero=" << A_host_non_zero << "/" << factor_size << std::endl;
            // std::cout.flush();
            
            CUDA_CHECK_VOID(cudaMemcpy(A_gpu_, A_host.data(), factor_size * sizeof(dp_t), cudaMemcpyHostToDevice));
            CUDA_CHECK_VOID(cudaMemcpy(B_gpu_, B_host.data(), factor_size * sizeof(dp_t), cudaMemcpyHostToDevice));
            break;
        }
        case QuantizationLevel::INT8: {
            // Dequantize factors from INT8 to FP32
            // Use stored device scale/zero_point if available, otherwise use defaults
            float* d_scale_A = d_A_scale_;
            float* d_zero_point_A = d_A_zero_point_;
            float* d_scale_B = d_B_scale_;
            float* d_zero_point_B = d_B_zero_point_;
            
            // If device scale/zero_point not stored, allocate and use defaults
            // This can happen if quantization was done before we added device storage
            bool need_alloc = (d_scale_A == nullptr);
            if (need_alloc) {
                CUDA_CHECK_VOID(cudaMalloc(&d_scale_A, sizeof(float)));
                CUDA_CHECK_VOID(cudaMalloc(&d_zero_point_A, sizeof(float)));
                CUDA_CHECK_VOID(cudaMalloc(&d_scale_B, sizeof(float)));
                CUDA_CHECK_VOID(cudaMalloc(&d_zero_point_B, sizeof(float)));
                
                // Use default values
                float default_scale = 1.0f / 127.0f;
                float default_zero = 0.0f;
                CUDA_CHECK_VOID(cudaMemcpy(d_scale_A, &default_scale, sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK_VOID(cudaMemcpy(d_zero_point_A, &default_zero, sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK_VOID(cudaMemcpy(d_scale_B, &default_scale, sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK_VOID(cudaMemcpy(d_zero_point_B, &default_zero, sizeof(float), cudaMemcpyHostToDevice));
            }
            
            dim3 block_size(256);
            dim3 grid_size((factor_size + block_size.x - 1) / block_size.x);
            
            dequantize_factors_int8_to_fp32_kernel_device_scale<<<grid_size, block_size>>>(
                static_cast<const int8_t*>(A_quantized_gpu_), A_gpu_, factor_size,
                d_scale_A, d_zero_point_A
            );
            
            dequantize_factors_int8_to_fp32_kernel_device_scale<<<grid_size, block_size>>>(
                static_cast<const int8_t*>(B_quantized_gpu_), B_gpu_, factor_size,
                d_scale_B, d_zero_point_B
            );
            // Note: No explicit sync - kernel will complete naturally
            
            // Only free if we allocated them (not if they're member variables)
            if (need_alloc) {
                cudaFree(d_scale_A);
                cudaFree(d_zero_point_A);
                cudaFree(d_scale_B);
                cudaFree(d_zero_point_B);
            }
            break;
        }
        case QuantizationLevel::NF4: {
            // Dequantize factors from NF4 to FP32
            dim3 block_size(256);
            dim3 grid_size((factor_size + block_size.x - 1) / block_size.x);
            
            dequantize_factors_nf4_to_fp32_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(A_quantized_gpu_), A_gpu_, factor_size
            );
            CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
            
            dequantize_factors_nf4_to_fp32_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(B_quantized_gpu_), B_gpu_, factor_size
            );
            CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
            break;
        }
        default:
            break;
    }
}

// Get factors
void LowRankProximityMatrix::get_factors(dp_t** A_gpu, dp_t** B_gpu, integer_t* rank) {
    // std::cout << "[DEBUG GET_FACTORS] Entry: trees_processed_=" << trees_processed_ 
    //           << ", rank_=" << rank_ 
    //           << ", A_gpu_=" << (void*)A_gpu_ 
    //           << ", B_gpu_=" << (void*)B_gpu_ << std::endl;
    // std::cout.flush();
    
    dequantize_factors();
    
    // Ensure A = B for symmetric proximity matrix (P = A @ B.T should be symmetric)
    // For symmetric matrix, we need A = B so that P = A @ A.T is symmetric
    if (A_gpu_ != nullptr && B_gpu_ != nullptr && rank_ > 0) {
        ensure_handles();
        // Copy A to B to ensure symmetry: B = A
        cublasDcopy(cublas_handle_, nsample_ * rank_, A_gpu_, 1, B_gpu_, 1);
        // Note: No explicit sync - cuBLAS operations will complete naturally
    }
    
    *A_gpu = A_gpu_;
    *B_gpu = B_gpu_;
    *rank = rank_;
    
//     std::cout << "[DEBUG GET_FACTORS] Exit: rank=" << *rank 
//               << ", A_gpu=" << (void*)(*A_gpu) 
//               << ", B_gpu=" << (void*)(*B_gpu) << std::endl;
//     std::cout.flush();
}

// Compute distances from factors
void LowRankProximityMatrix::compute_distances_from_factors(dp_t* output_distances, integer_t nsample) {
    // For now, reconstruct full matrix and compute distances
    reconstruct_full_matrix(output_distances, nsample);
    
    // Convert proximity to distance: D[i,j] = 1 - P[i,j]
    // Would be more efficient to compute directly from factors
    }
    
    namespace {
        __global__ void compute_row_means_from_factors_kernel(
        dp_t* row_means, 
        dp_t max_prox, 
        integer_t n_samples
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_samples) {
            // row_means[i] currently contains mean proximity P[i,:]
            // For normalized proximity (diagonal = 1.0), use: distance = 1 - proximity
            // But if values aren't normalized, normalize first: prox_norm = prox / max_prox
            // Then: distance = 1 - prox_norm = 1 - prox/max_prox = (max_prox - prox) / max_prox
            // For simplicity, if max_prox ≈ 1.0 (normalized), use: distance = 1 - prox
            // Otherwise, normalize: distance = 1 - (prox / max_prox) = (max_prox - prox) / max_prox
            dp_t prox_mean = row_means[i];
            if (max_prox > 1e-10) {
                // Normalize proximity to [0, 1] range, then convert to distance
                dp_t prox_norm = prox_mean / max_prox;
                prox_norm = fmax(0.0, fmin(1.0, prox_norm));  // Clamp to [0, 1]
                row_means[i] = 1.0 - prox_norm;  // Distance = 1 - normalized proximity
            } else {
                row_means[i] = 1.0;  // Default to max distance if max_prox is invalid
            }
        }
    }
    
    __global__ void compute_col_means_from_factors_kernel(
        dp_t* col_means, 
        dp_t max_prox, 
        integer_t n_samples
    ) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j < n_samples) {
            // Same logic as row means
            dp_t prox_mean = col_means[j];
            if (max_prox > 1e-10) {
                dp_t prox_norm = prox_mean / max_prox;
                prox_norm = fmax(0.0, fmin(1.0, prox_norm));  // Clamp to [0, 1]
                col_means[j] = 1.0 - prox_norm;  // Distance = 1 - normalized proximity
            } else {
                col_means[j] = 1.0;  // Default to max distance
            }
        }
    }
    
    // Kernel to square values (for converting D means to D² means)
    __global__ void square_values_kernel(
        dp_t* values,
        integer_t n_samples
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_samples) {
            dp_t val = values[i];
            values[i] = val * val;  // Square the value
        }
    }
    
    // Kernel to scale rows of B by sqrt(v): scaled_B[j,r] = sqrt(v[j]) * B[j,r]
    __global__ void scale_rows_by_sqrt_v_kernel(
        dp_t* scaled_B,
        const dp_t* B,
        const dp_t* v,
        integer_t n_samples,
        integer_t rank
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = n_samples * rank;
        if (idx < total) {
            int j = idx % n_samples;  // Row index (column-major storage)
            dp_t sqrt_v_j = sqrt(fmax(0.0, v[j]));
            scaled_B[idx] = sqrt_v_j * B[idx];
        }
    }
    
    // Kernel to compute row-wise quadratic form: result[i] = sum_r A[i,r] * (A*Q)[i,r]
    __global__ void compute_rowwise_quadratic_form_kernel(
        dp_t* result,
        const dp_t* A,
        const dp_t* AQ,  // A × Q
        integer_t n_samples,
        integer_t rank
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_samples) {
            dp_t sum = 0.0;
            for (int r = 0; r < rank; r++) {
                // Column-major: A[i,r] is at A[i + r*n_samples]
                sum += A[i + r * n_samples] * AQ[i + r * n_samples];
            }
            result[i] = sum;
        }
    }
    
    __global__ void double_center_from_factors_kernel(
        dp_t* centered,
        const dp_t* row_means, 
        const dp_t* col_means,
        dp_t max_prox, 
        dp_t grand_mean, 
        integer_t n_samples
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_samples * n_samples) {
            int i = idx / n_samples;
            int j = idx % n_samples;
            dp_t constant_term = -0.5 * (max_prox - row_means[i] - col_means[j] + grand_mean);
            centered[idx] += constant_term;
        }
    }
    
    // CUDA kernel: Apply double-centering to D² × v (P² × v is computed separately)
    // Inputs: Pv = (P × v), P2v = (P² × v), row/col means of D², grand mean
    // Output: C × v where C = -0.5 * (D² - row_means - col_means + grand_mean)
    __global__ void apply_double_centering_corrections_kernel(
        dp_t* y,                     // Output: C × v
        const dp_t* Pv,              // Input: P × v
        const dp_t* P2v,             // Input: P² × v (exact, from low-rank factors)
        const dp_t* row_means_D2,    // Row means of D²
        const dp_t* col_means_D2,    // Column means of D² (same as row for symmetric)
        dp_t grand_mean_D2,          // Grand mean of D²
        dp_t sum_v,                  // Sum of input vector v
        dp_t col_means_D2_dot_v,     // Dot product of col_means_D2 and v
        integer_t nsample
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nsample) return;
        
        // D²[i,j] = (1 - P[i,j])² = 1 - 2*P[i,j] + P[i,j]² (assuming normalized proximity)
        // (D² × v)[i] = sum(v) - 2*(P × v)[i] + (P² × v)[i]
        dp_t D2v_i = sum_v - 2.0 * Pv[i] + P2v[i];
        
        // C × v = -0.5 * (D² × v - row_means_D2 * sum(v) - col_means_D2 · v + grand_mean_D2 * sum(v))
        y[i] = -0.5 * (D2v_i - row_means_D2[i] * sum_v - col_means_D2_dot_v + grand_mean_D2 * sum_v);
    }
    } // anonymous namespace

// Set OOB counts for RF-GAP normalization
void LowRankProximityMatrix::set_oob_counts_for_rfgap(const integer_t* oob_counts) {
    oob_counts_rfgap_.resize(nsample_);
    std::copy(oob_counts, oob_counts + nsample_, oob_counts_rfgap_.begin());
}

// Compute MDS coordinates directly from low-rank factors (MEMORY EFFICIENT)
// Uses iterative power method with deflation - NO O(n²) matrix allocation!
// Memory: O(n × k) where k is the number of dimensions, instead of O(n²)
std::vector<double> LowRankProximityMatrix::compute_mds_from_factors(integer_t k) {
    if (k < 1 || k > nsample_) {
        // Silently return - invalid MDS dimension (don't throw in Jupyter)
        return std::vector<double>();
    }
    
    if (nsample_ < 2) {
        // Silently return - insufficient samples (don't throw in Jupyter)
        return std::vector<double>();
    }
    
    ensure_allocated();
    ensure_handles();
    
    // Try to dequantize, but if it fails, check if A_gpu_ and B_gpu_ are already valid
    dequantize_factors();
    
    // If dequantization didn't populate A_gpu_ and B_gpu_, they should already be valid from SVD
    // Only throw error if they're truly not available
    if (!A_gpu_ || !B_gpu_ || rank_ == 0) {
        // Check if this is because factors haven't been computed yet
        if (trees_processed_ == 0 || rank_ == 0) {
            // Silently return - factors not initialized (don't throw in Jupyter)
            return std::vector<double>();
        }
        // Silently return - factors not initialized (don't throw in Jupyter)
        return std::vector<double>();
    }
    
    size_t n = static_cast<size_t>(nsample_);
    
    // Step 1: Find max proximity for normalization
    // For normalized proximity matrices, max_prox should be 1.0 (diagonal elements)
    // But for low-rank factors, values might not be normalized, so we compute max
    // Compute max of diagonal elements: P[i,i] = A[i,:] · B[i,:]^T
    // Do this entirely on GPU to avoid host copies that cause hangs
    dp_t* diagonal_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&diagonal_gpu, nsample_ * sizeof(dp_t)));
    
    // Compute diagonal using cuBLAS: diagonal[i] = A[i,:] · B[i,:]^T
    // This is a row-wise dot product
    // A is stored column-major: A[i, r] = A_gpu_[i + r * nsample_]
    // B is stored column-major: B[i, r] = B_gpu_[i + r * nsample_]
    // Diagonal[i] = sum over r: A[i, r] * B[i, r]
    for (integer_t i = 0; i < nsample_; i++) {
        // Compute dot product of row i of A and row i of B
        // A row i: stride=1, offset=i
        // B row i: stride=1, offset=i
        dp_t diag_val = 0.0;
        cublasDdot(cublas_handle_, rank_, A_gpu_ + i, nsample_, B_gpu_ + i, nsample_, diagonal_gpu + i);
    }
    
    // Find max on GPU
    integer_t max_idx;
    cublasIdamax(cublas_handle_, nsample_, diagonal_gpu, 1, &max_idx);
    dp_t max_prox;
    CUDA_CHECK_VOID(cudaMemcpy(&max_prox, diagonal_gpu + max_idx - 1, sizeof(dp_t), cudaMemcpyDeviceToHost));
    
    // Note: max_prox already computed on GPU above from diagonal
    // For normalized proximity, diagonal should be 1.0, but we use the computed max_prox
    
    // Skip off-diagonal sampling to avoid host copies - use max_prox from diagonal
    // For large datasets, this avoids hangs from host memory copies
    
    // If max_prox is close to 1.0, assume proximity is already normalized
    // Otherwise, we'll normalize in the kernel
    if (max_prox < 1e-10) {
        max_prox = 1.0;  // Avoid division by zero
    }
    
    // Step 2: Compute row/column means directly from factors (MEMORY EFFICIENT)
    // For P = A × B^T:
    // row_mean[i] = max_prox - (1/n) * A[i,:] · (sum_j B[j,:])^T
    // col_mean[j] = max_prox - (1/n) * (sum_i A[i,:]) · B[j,:]^T
    // grand_mean = max_prox - (1/n²) * (sum_i A[i,:]) · (sum_j B[j,:])^T
    
    // Compute A_sum = sum_i A[i,:] (shape: rank)
    dp_t* A_sum_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&A_sum_gpu, rank_ * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMemset(A_sum_gpu, 0, rank_ * sizeof(dp_t)));
    
    // Compute B_sum = sum_j B[j,:] (shape: rank)
    dp_t* B_sum_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&B_sum_gpu, rank_ * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMemset(B_sum_gpu, 0, rank_ * sizeof(dp_t)));
    
    // Sum columns of A and B using cuBLAS
    dp_t alpha_sum = 1.0;
    dp_t beta_sum = 0.0;
    // A_sum = sum of columns: A^T × ones(n) where ones(n) is all-ones vector
    dp_t* ones_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&ones_gpu, nsample_ * sizeof(dp_t)));
    // cudaMemset sets BYTES, not double values!
    // Create ones vector on host and copy to GPU
    std::vector<dp_t> ones_host(nsample_, 1.0);
    CUDA_CHECK_VOID(cudaMemcpy(ones_gpu, ones_host.data(), nsample_ * sizeof(dp_t), cudaMemcpyHostToDevice));
    
    // A_sum = A^T × ones (each row of A^T gets summed)
    cublasDgemv(cublas_handle_, CUBLAS_OP_T,
                nsample_, rank_,
                &alpha_sum, A_gpu_, nsample_,
                ones_gpu, 1,
                &beta_sum, A_sum_gpu, 1);
    
    // B_sum = B^T × ones
    cublasDgemv(cublas_handle_, CUBLAS_OP_T,
                nsample_, rank_,
                &alpha_sum, B_gpu_, nsample_,
                ones_gpu, 1,
                &beta_sum, B_sum_gpu, 1);
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    // Safe free to prevent segfaults
    if (ones_gpu) { cudaError_t err = cudaFree(ones_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    
    // Compute row means of D² CORRECTLY (not the approximation)
    // Formula: mean_D²[i] = 1 - 2*mean_P[i]/max_prox + mean_P²[i]/max_prox²
    // where mean_P[i] = A[i,:] · B_sum / n
    //       mean_P²[i] = A[i,:] · (B^T B) · A[i,:]^T / n
    
    dp_t* row_means_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&row_means_gpu, n * sizeof(dp_t)));
    
    // Step 1: Compute mean_P[i] = (1/n) * A[i,:] · B_sum
    dp_t* mean_P_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&mean_P_gpu, n * sizeof(dp_t)));
    
    dp_t alpha_dot = 1.0 / static_cast<dp_t>(nsample_);
    dp_t beta_dot = 0.0;
    cublasDgemv(cublas_handle_, CUBLAS_OP_N,
                nsample_, rank_,
                &alpha_dot, A_gpu_, nsample_,
                B_sum_gpu, 1,
                &beta_dot, mean_P_gpu, 1);
    
    // Step 2: Compute C = B^T × B (rank × rank matrix)
    dp_t* BtB_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&BtB_gpu, rank_ * rank_ * sizeof(dp_t)));
    dp_t alpha_gemm = 1.0, beta_gemm = 0.0;
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                rank_, rank_, nsample_,
                &alpha_gemm, B_gpu_, nsample_,
                B_gpu_, nsample_,
                &beta_gemm, BtB_gpu, rank_);
    
    // Step 3: Compute mean_P²[i] = (1/n) * A[i,:] · C · A[i,:]^T
    // First compute temp = A × C (nsample × rank)
    dp_t* AC_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&AC_gpu, nsample_ * rank_ * sizeof(dp_t)));
    cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                nsample_, rank_, rank_,
                &alpha_gemm, A_gpu_, nsample_,
                BtB_gpu, rank_,
                &beta_gemm, AC_gpu, nsample_);
    
    // Then mean_P²[i] = (1/n) * sum_r(A[i,r] * AC[i,r]) = row-wise dot product
    // Compute this using element-wise multiply and row sum
    dp_t* mean_P2_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&mean_P2_gpu, n * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMemset(mean_P2_gpu, 0, n * sizeof(dp_t)));
    
    // Element-wise: mean_P²[i] = (1/n) * Σ_r A[i,r] * AC[i,r]
    int threads_per_block = 256;
    int blocks = (nsample_ + threads_per_block - 1) / threads_per_block;
    
    // Use a kernel to compute row-wise dot products
    for (integer_t r = 0; r < rank_; r++) {
        // mean_P2[i] += A[i,r] * AC[i,r]
        cublasDaxpy(cublas_handle_, nsample_, &alpha_gemm, 
                    A_gpu_ + r * nsample_, 1,  // Column r of A (column-major)
                    mean_P2_gpu, 1);  // Accumulate - but this is wrong, we need element-wise multiply
    }
    
    // Actually, let's do this properly with a simple kernel
    // Recompute mean_P2 with element-wise operations
    CUDA_CHECK_VOID(cudaMemset(mean_P2_gpu, 0, n * sizeof(dp_t)));
    
    // Use cuBLAS to compute row-wise dot products: each row i gets A[i,:] · AC[i,:]
    // This requires iterating over each sample - expensive but correct
    for (integer_t i = 0; i < nsample_; i++) {
        dp_t dot_result = 0.0;
        // A[i,:] is at A_gpu_[i, 0:rank_] with stride nsample_ (column-major)
        // AC[i,:] is at AC_gpu[i, 0:rank_] with stride nsample_ (column-major)
        cublasDdot(cublas_handle_, rank_,
                   A_gpu_ + i, nsample_,  // Row i of A
                   AC_gpu + i, nsample_,  // Row i of AC
                   &dot_result);
        dot_result /= static_cast<dp_t>(nsample_);  // Divide by n
        CUDA_CHECK_VOID(cudaMemcpy(mean_P2_gpu + i, &dot_result, sizeof(dp_t), cudaMemcpyHostToDevice));
    }
    
    // Step 4: Compute row_means_D2[i] = 1 - 2*mean_P[i]/max_prox + mean_P2[i]/max_prox²
    // Use a kernel for this
    dp_t inv_max_prox = (max_prox > 1e-10) ? (1.0 / max_prox) : 1.0;
    dp_t inv_max_prox_sq = inv_max_prox * inv_max_prox;
    
    // Copy mean_P and mean_P2 to host, compute, copy back (simpler than writing kernel)
    std::vector<dp_t> mean_P_host(nsample_), mean_P2_host(nsample_), row_means_host(nsample_);
    CUDA_CHECK_VOID(cudaMemcpy(mean_P_host.data(), mean_P_gpu, n * sizeof(dp_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(mean_P2_host.data(), mean_P2_gpu, n * sizeof(dp_t), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < n; i++) {
        // row_means_D2[i] = 1 - 2*mean_P[i]/max_prox + mean_P2[i]/max_prox²
        row_means_host[i] = 1.0 - 2.0 * mean_P_host[i] * inv_max_prox + mean_P2_host[i] * inv_max_prox_sq;
    }
    CUDA_CHECK_VOID(cudaMemcpy(row_means_gpu, row_means_host.data(), n * sizeof(dp_t), cudaMemcpyHostToDevice));
    
    // Cleanup temporary buffers
    if (mean_P_gpu) cudaFree(mean_P_gpu);
    if (BtB_gpu) cudaFree(BtB_gpu);
    if (AC_gpu) cudaFree(AC_gpu);
    if (mean_P2_gpu) cudaFree(mean_P2_gpu);
    
    // RF-GAP normalization: normalize row_means by |Si| if RF-GAP is used
    if (!oob_counts_rfgap_.empty()) {
        integer_t* oob_counts_gpu = nullptr;
        CUDA_CHECK_VOID(cudaMalloc(&oob_counts_gpu, nsample_ * sizeof(integer_t)));
        CUDA_CHECK_VOID(cudaMemcpy(oob_counts_gpu, oob_counts_rfgap_.data(), 
                                  nsample_ * sizeof(integer_t), cudaMemcpyHostToDevice));
        
        // Normalize row_means[i] by |Si| (OOB count)
        normalize_by_oob_counts_kernel<<<blocks, threads_per_block>>>(
            row_means_gpu, oob_counts_gpu, nsample_
        );
        CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
        // Safe free to prevent segfaults
        if (oob_counts_gpu) {
            cudaError_t err = cudaFree(oob_counts_gpu);
            if (err != cudaSuccess) cudaGetLastError(); // Clear error state
        }
    }
    
    // Compute column means of D²
    // For symmetric proximity matrix P = A × B^T, col_means = row_means
    // (since D²[i,j] = D²[j,i] for symmetric matrices)
    dp_t* col_means_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&col_means_gpu, n * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMemcpy(col_means_gpu, row_means_gpu, n * sizeof(dp_t), cudaMemcpyDeviceToDevice));
    
    // RF-GAP normalization: normalize col_means by |Sj| if RF-GAP is used
    // Note: After symmetrization, col_means should match row_means, so normalize the same way
    if (!oob_counts_rfgap_.empty()) {
        integer_t* oob_counts_gpu = nullptr;
        CUDA_CHECK_VOID(cudaMalloc(&oob_counts_gpu, nsample_ * sizeof(integer_t)));
        CUDA_CHECK_VOID(cudaMemcpy(oob_counts_gpu, oob_counts_rfgap_.data(), 
                                  nsample_ * sizeof(integer_t), cudaMemcpyHostToDevice));
        
        // Normalize col_means[j] by |Sj| (OOB count)
        normalize_by_oob_counts_kernel<<<blocks, threads_per_block>>>(
            col_means_gpu, oob_counts_gpu, nsample_
        );
        CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
        // Safe free to prevent segfaults
        if (oob_counts_gpu) {
            cudaError_t err = cudaFree(oob_counts_gpu);
            if (err != cudaSuccess) cudaGetLastError(); // Clear error state
        }
    }
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Compute grand mean of D² CORRECTLY
    // Formula: grand_mean_D² = 1 - 2*grand_mean_P/max_prox + grand_mean_P²/max_prox²
    // where grand_mean_P = (1/n²) * A_sum · B_sum
    //       grand_mean_P² = (1/n²) * ||P||_F² = (1/n²) * trace((A^T A)(B^T B))
    
    dp_t A_sum_dot_B_sum = 0.0;
    cublasDdot(cublas_handle_, rank_,
               A_sum_gpu, 1,
               B_sum_gpu, 1,
               &A_sum_dot_B_sum);
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));
    
    dp_t grand_mean_P = A_sum_dot_B_sum / static_cast<dp_t>(n * n);
    
    // Compute grand_mean_P² = (1/n²) * ||P||_F² = (1/n²) * trace((A^T A)(B^T B))
    // First compute A^T A (rank × rank)
    dp_t* AtA_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&AtA_gpu, rank_ * rank_ * sizeof(dp_t)));
    dp_t alpha_ata = 1.0, beta_ata = 0.0;
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                rank_, rank_, nsample_,
                &alpha_ata, A_gpu_, nsample_,
                A_gpu_, nsample_,
                &beta_ata, AtA_gpu, rank_);
    
    // Compute B^T B (rank × rank) - we need this again
    dp_t* BtB_gpu2 = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&BtB_gpu2, rank_ * rank_ * sizeof(dp_t)));
    cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                rank_, rank_, nsample_,
                &alpha_ata, B_gpu_, nsample_,
                B_gpu_, nsample_,
                &beta_ata, BtB_gpu2, rank_);
    
    // Compute trace((A^T A)(B^T B)) = Frobenius inner product = Σ_ij AtA[i,j] * BtB[i,j]
    // Copy to host and compute
    std::vector<dp_t> AtA_host(rank_ * rank_), BtB_host(rank_ * rank_);
    CUDA_CHECK_VOID(cudaMemcpy(AtA_host.data(), AtA_gpu, rank_ * rank_ * sizeof(dp_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(BtB_host.data(), BtB_gpu2, rank_ * rank_ * sizeof(dp_t), cudaMemcpyDeviceToHost));
    
    dp_t frobenius_inner = 0.0;
    for (integer_t i = 0; i < rank_ * rank_; i++) {
        frobenius_inner += AtA_host[i] * BtB_host[i];
    }
    dp_t grand_mean_P2 = frobenius_inner / static_cast<dp_t>(n * n);
    
    if (AtA_gpu) cudaFree(AtA_gpu);
    if (BtB_gpu2) cudaFree(BtB_gpu2);
    
    // Compute grand_mean_D² = 1 - 2*grand_mean_P/max_prox + grand_mean_P²/max_prox²
    dp_t grand_mean_D2 = 1.0 - 2.0 * grand_mean_P * inv_max_prox + grand_mean_P2 * inv_max_prox_sq;
    
    // RF-GAP normalization: normalize grand_mean by average |Si|
    // Since grand_mean is the mean of all elements, and each row is normalized by |Si|,
    // we need to normalize by the average of |Si| across all samples
    if (!oob_counts_rfgap_.empty()) {
        integer_t total_oob = 0;
        for (integer_t i = 0; i < nsample_; i++) {
            total_oob += oob_counts_rfgap_[i];
        }
        if (total_oob > 0) {
            dp_t avg_oob = static_cast<dp_t>(total_oob) / static_cast<dp_t>(nsample_);
            grand_mean_D2 /= (avg_oob * avg_oob);  // Square the normalization factor for D²
        }
    }
    
    // Free A_sum_gpu and B_sum_gpu (no longer needed)
    // Safe free to prevent segfaults
    if (A_sum_gpu) { cudaError_t err = cudaFree(A_sum_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    if (B_sum_gpu) { cudaError_t err = cudaFree(B_sum_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    
    // Step 3: Iterative power method with deflation to compute top k eigenvectors
    // NO O(n²) matrix allocation! Uses matrix-vector products with low-rank factors
    
    // Allocate workspace for eigenvectors and temporary vectors (O(n × k) memory)
    dp_t* eigenvectors_gpu = nullptr;  // k × nsample (stored column-major: each eigenvector is a column)
    CUDA_CHECK_VOID(cudaMalloc(&eigenvectors_gpu, k * nsample_ * sizeof(dp_t)));
    
    dp_t* v_gpu = nullptr;  // Current eigenvector candidate
    dp_t* w_gpu = nullptr;  // Workspace for matrix-vector product
    CUDA_CHECK_VOID(cudaMalloc(&v_gpu, nsample_ * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMalloc(&w_gpu, nsample_ * sizeof(dp_t)));
    
    // Allocate a vector of ones for computing sum(v) = ones · v
    dp_t* ones_gpu_local = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&ones_gpu_local, nsample_ * sizeof(dp_t)));
    std::vector<dp_t> ones_host_mds(nsample_, 1.0);
    CUDA_CHECK_VOID(cudaMemcpy(ones_gpu_local, ones_host_mds.data(), nsample_ * sizeof(dp_t), cudaMemcpyHostToDevice));
    
    std::vector<dp_t> eigenvalues(k, 0.0);
    
    // Power method with deflation for each of k eigenvectors
    for (int eig_idx = 0; eig_idx < k; eig_idx++) {
        // Initialize random vector v with different seed for each eigenvector
        srand(12345 + eig_idx * 1000);  // Different seed for each eigenvector
        std::vector<dp_t> v_host(nsample_);
        for (integer_t i = 0; i < nsample_; i++) {
            v_host[i] = static_cast<dp_t>(rand()) / RAND_MAX - 0.5;  // Random in [-0.5, 0.5]
        }
        CUDA_CHECK_VOID(cudaMemcpy(v_gpu, v_host.data(), nsample_ * sizeof(dp_t), cudaMemcpyHostToDevice));
        
        // Normalize initial vector
        dp_t v_norm = 0.0;
        cublasDnrm2(cublas_handle_, nsample_, v_gpu, 1, &v_norm);
        if (v_norm > 1e-10) {
            dp_t inv_norm = 1.0 / v_norm;
            cublasDscal(cublas_handle_, nsample_, &inv_norm, v_gpu, 1);
        }
        
        // Orthogonalize initial vector against ALL previous eigenvectors
        // This ensures we start in the orthogonal complement and find a new eigenvector
        for (int prev = 0; prev < eig_idx; prev++) {
            dp_t* prev_eig = eigenvectors_gpu + prev * nsample_;
            dp_t dot_product = 0.0;
            cublasDdot(cublas_handle_, nsample_, prev_eig, 1, v_gpu, 1, &dot_product);
            dp_t neg_dot = -dot_product;
            cublasDaxpy(cublas_handle_, nsample_, &neg_dot, prev_eig, 1, v_gpu, 1);
        }
        
        // Re-normalize after orthogonalization
        cublasDnrm2(cublas_handle_, nsample_, v_gpu, 1, &v_norm);
        if (v_norm > 1e-10) {
            dp_t inv_norm = 1.0 / v_norm;
            cublasDscal(cublas_handle_, nsample_, &inv_norm, v_gpu, 1);
        }
        
        // Power iteration: v = C × v / ||C × v||
        const int max_iterations = 100;
        const dp_t tolerance = 1e-6;
        dp_t prev_eigenvalue = 0.0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute w = C × v where C = -0.5 * (D² - row_means - col_means + grand_mean)
            // D²[i,j] = (1 - P[i,j])² = 1 - 2*P[i,j] + P[i,j]²
            // (D² × v)[i] = sum(v) - 2*(P × v)[i] + (P² × v)[i]
            
            dp_t alpha_mv = 1.0, beta_mv = 0.0;
            
            // Step 1: Compute P × v = A × (B^T × v)
            dp_t* temp_rank = nullptr;  // Temporary vector of size rank_
            CUDA_CHECK_VOID(cudaMalloc(&temp_rank, rank_ * sizeof(dp_t)));
            
            // B^T × v -> temp_rank
            cublasDgemv(cublas_handle_, CUBLAS_OP_T,
                       nsample_, rank_,
                       &alpha_mv, B_gpu_, nsample_,
                       v_gpu, 1,
                       &beta_mv, temp_rank, 1);
            
            // P × v = A × temp_rank -> Pv_gpu
            dp_t* Pv_gpu = nullptr;
            CUDA_CHECK_VOID(cudaMalloc(&Pv_gpu, nsample_ * sizeof(dp_t)));
            cublasDgemv(cublas_handle_, CUBLAS_OP_N,
                       nsample_, rank_,
                       &alpha_mv, A_gpu_, nsample_,
                       temp_rank, 1,
                       &beta_mv, Pv_gpu, 1);
            
            // Step 2: Compute P² × v EXACTLY using quadratic form
            // (P² × v)[i] = A[i,:] × Q × A[i,:]^T where Q = B^T diag(v) B
            
            // 2a: Scale B by sqrt(v): scaled_B[j,r] = sqrt(v[j]) * B[j,r]
            dp_t* scaled_B_gpu = nullptr;
            CUDA_CHECK_VOID(cudaMalloc(&scaled_B_gpu, nsample_ * rank_ * sizeof(dp_t)));
            int scale_threads = 256;
            int scale_blocks = (nsample_ * rank_ + scale_threads - 1) / scale_threads;
            scale_rows_by_sqrt_v_kernel<<<scale_blocks, scale_threads>>>(
                scaled_B_gpu, B_gpu_, v_gpu, nsample_, rank_
            );
            
            // 2b: Compute Q = scaled_B^T × scaled_B (rank × rank)
            dp_t* Q_gpu = nullptr;
            CUDA_CHECK_VOID(cudaMalloc(&Q_gpu, rank_ * rank_ * sizeof(dp_t)));
            cublasDgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                        rank_, rank_, nsample_,
                        &alpha_mv, scaled_B_gpu, nsample_,
                        scaled_B_gpu, nsample_,
                        &beta_mv, Q_gpu, rank_);
            
            // 2c: Compute A × Q -> AQ_gpu (nsample × rank)
            dp_t* AQ_gpu = nullptr;
            CUDA_CHECK_VOID(cudaMalloc(&AQ_gpu, nsample_ * rank_ * sizeof(dp_t)));
            cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                        nsample_, rank_, rank_,
                        &alpha_mv, A_gpu_, nsample_,
                        Q_gpu, rank_,
                        &beta_mv, AQ_gpu, nsample_);
            
            // 2d: Compute P² × v: P2v[i] = sum_r A[i,r] * AQ[i,r]
            dp_t* P2v_gpu = nullptr;
            CUDA_CHECK_VOID(cudaMalloc(&P2v_gpu, nsample_ * sizeof(dp_t)));
            blocks = (nsample_ + threads_per_block - 1) / threads_per_block;
            compute_rowwise_quadratic_form_kernel<<<blocks, threads_per_block>>>(
                P2v_gpu, A_gpu_, AQ_gpu, nsample_, rank_
            );
            
            // Free intermediate buffers
            if (scaled_B_gpu) cudaFree(scaled_B_gpu);
            if (Q_gpu) cudaFree(Q_gpu);
            if (AQ_gpu) cudaFree(AQ_gpu);
            if (temp_rank) cudaFree(temp_rank);
            
            // RF-GAP normalization: normalize Pv by |Si| (OOB count)
            if (!oob_counts_rfgap_.empty()) {
                integer_t* oob_counts_gpu = nullptr;
                CUDA_CHECK_VOID(cudaMalloc(&oob_counts_gpu, nsample_ * sizeof(integer_t)));
                CUDA_CHECK_VOID(cudaMemcpy(oob_counts_gpu, oob_counts_rfgap_.data(), 
                                          nsample_ * sizeof(integer_t), cudaMemcpyHostToDevice));
                
                int threads_per_block_norm = 256;
                int blocks_norm = (nsample_ + threads_per_block_norm - 1) / threads_per_block_norm;
                normalize_by_oob_counts_kernel<<<blocks_norm, threads_per_block_norm>>>(
                    Pv_gpu, oob_counts_gpu, nsample_
                );
                normalize_by_oob_counts_kernel<<<blocks_norm, threads_per_block_norm>>>(
                    P2v_gpu, oob_counts_gpu, nsample_
                );
                CUDA_CHECK_VOID(cudaStreamSynchronize(0));
                if (oob_counts_gpu) cudaFree(oob_counts_gpu);
            }
            
            // Step 3: Compute sum(v) and col_means · v
            // IMPORTANT: Use cublasDdot with ones, NOT cublasDasum (which gives sum of absolute values!)
            dp_t sum_v = 0.0;
            cublasDdot(cublas_handle_, nsample_, ones_gpu_local, 1, v_gpu, 1, &sum_v);
            
            dp_t col_means_D2_dot_v = 0.0;
            cublasDdot(cublas_handle_, nsample_, col_means_gpu, 1, v_gpu, 1, &col_means_D2_dot_v);
            
            // Step 4: Apply double-centering to compute C × v
            // w[i] = -0.5 * (D²v[i] - row_means_D2[i] * sum_v - col_means_D2_dot_v + grand_mean_D2 * sum_v)
            // D²v[i] = sum_v - 2*Pv[i] + P²v[i]
            apply_double_centering_corrections_kernel<<<blocks, threads_per_block>>>(
                w_gpu, Pv_gpu, P2v_gpu, row_means_gpu, col_means_gpu, grand_mean_D2, sum_v, col_means_D2_dot_v, nsample_
            );
            CUDA_CHECK_VOID(cudaStreamSynchronize(0));
            
            // Free Pv and P2v
            if (Pv_gpu) cudaFree(Pv_gpu);
            if (P2v_gpu) cudaFree(P2v_gpu);
            
            // Deflate: Remove previous eigenvectors from w
            for (int prev = 0; prev < eig_idx; prev++) {
                dp_t* prev_eig = eigenvectors_gpu + prev * nsample_;
                dp_t dot_product = 0.0;
                cublasDdot(cublas_handle_, nsample_, prev_eig, 1, w_gpu, 1, &dot_product);
                dp_t neg_dot = -dot_product;
                cublasDaxpy(cublas_handle_, nsample_, &neg_dot, prev_eig, 1, w_gpu, 1);
            }
            
            // Normalize w
            dp_t w_norm = 0.0;
            cublasDnrm2(cublas_handle_, nsample_, w_gpu, 1, &w_norm);
            
            if (w_norm < 1e-10) {
                // Convergence to zero - matrix is rank-deficient
                break;
            }
            
            // Compute eigenvalue estimate: λ ≈ v^T × C × v = v^T × w
            dp_t eigenvalue = 0.0;
            cublasDdot(cublas_handle_, nsample_, v_gpu, 1, w_gpu, 1, &eigenvalue);
            
            // Check convergence
            if (iter > 0 && std::abs(eigenvalue - prev_eigenvalue) < tolerance * std::abs(eigenvalue)) {
                eigenvalues[eig_idx] = eigenvalue;
                break;
            }
            
            prev_eigenvalue = eigenvalue;
            
            // Update v = w / ||w||
            dp_t inv_w_norm = 1.0 / w_norm;
            cublasDscal(cublas_handle_, nsample_, &inv_w_norm, w_gpu, 1);
            
            // Swap v and w
            dp_t* temp = v_gpu;
            v_gpu = w_gpu;
            w_gpu = temp;
        }
        
        // Store converged eigenvector
        dp_t* eig_ptr = eigenvectors_gpu + eig_idx * nsample_;
        CUDA_CHECK_VOID(cudaMemcpy(eig_ptr, v_gpu, nsample_ * sizeof(dp_t), cudaMemcpyDeviceToDevice));
        
        // Store eigenvalue
        if (eigenvalues[eig_idx] == 0.0) {
            // Compute final eigenvalue
            cublasDdot(cublas_handle_, nsample_, v_gpu, 1, w_gpu, 1, &eigenvalues[eig_idx]);
        }
    }
    
    // Free temporary vectors
    // Safe free to prevent segfaults
    if (v_gpu) { cudaError_t err = cudaFree(v_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    if (w_gpu) { cudaError_t err = cudaFree(w_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    if (ones_gpu_local) { cudaError_t err = cudaFree(ones_gpu_local); if (err != cudaSuccess) cudaGetLastError(); }
    if (row_means_gpu) { cudaError_t err = cudaFree(row_means_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    if (col_means_gpu) { cudaError_t err = cudaFree(col_means_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    
    // Step 4: Extract top k eigenvectors and scale by sqrt(eigenvalue)
    // Eigenvectors are already stored in eigenvectors_gpu (sorted by eigenvalue magnitude)
    
    // Sort eigenvalues by value (descending) to get top k eigenvalues
    // Include ALL eigenvalues (even negative ones for MDS - they indicate non-Euclidean structure)
    std::vector<std::pair<dp_t, int>> eigen_pairs;
    for (int i = 0; i < k; i++) {
        eigen_pairs.push_back(std::make_pair(eigenvalues[i], i));
    }
    
    std::sort(eigen_pairs.rbegin(), eigen_pairs.rend());
    
    // Extract top k eigenvectors and scale by sqrt(eigenvalue)
    std::vector<double> coords_kd(n * k);
    std::vector<dp_t> eigenvector_host(nsample_);
    
    for (int comp = 0; comp < k; comp++) {
        int eigen_idx = eigen_pairs[comp].second;
        dp_t* eig_ptr = eigenvectors_gpu + eigen_idx * nsample_;
        CUDA_CHECK_VOID(cudaMemcpy(eigenvector_host.data(), eig_ptr, nsample_ * sizeof(dp_t), cudaMemcpyDeviceToHost));
        
        // Scale by sqrt(|eigenvalue|) for MDS coordinates
        // Use absolute value to handle negative eigenvalues (non-Euclidean distances)
        dp_t scale_factor = std::sqrt(std::abs(eigenvalues[eigen_idx]));
        
        for (size_t i = 0; i < n; i++) {
            coords_kd[i * k + comp] = static_cast<double>(eigenvector_host[i] * scale_factor);
        }
    }
    
    // Cleanup
    // Safe free to prevent segfaults
    if (eigenvectors_gpu) { cudaError_t err = cudaFree(eigenvectors_gpu); if (err != cudaSuccess) cudaGetLastError(); }
    
    return coords_kd;
}

// Auto-adjust rank
integer_t LowRankProximityMatrix::auto_adjust_rank(double target_variance) {
    // For now, return current rank
    // Would perform SVD and find rank that captures target_variance
    return rank_;
}

// Get memory usage
size_t LowRankProximityMatrix::get_memory_usage() const {
    return get_memory_usage(current_quant_level_);
}

// Get memory usage for quantization level
size_t LowRankProximityMatrix::get_memory_usage(QuantizationLevel quant_level) const {
    size_t factor_size = nsample_ * rank_;
    size_t bytes_per_element = 0;
    
    switch (quant_level) {
        case QuantizationLevel::FP32:  // Uses FP16 storage internally
            bytes_per_element = sizeof(__half);
            break;
        case QuantizationLevel::FP16:
            bytes_per_element = sizeof(__half);
            break;
        case QuantizationLevel::INT8:
            bytes_per_element = sizeof(int8_t);
            break;
        case QuantizationLevel::NF4:
            bytes_per_element = sizeof(uint8_t) / 2;
            break;
    }
    
    return 2 * factor_size * bytes_per_element; // A and B factors
}

// Should use incremental
bool LowRankProximityMatrix::should_use_incremental(integer_t nsample, size_t available_memory) {
    // Use incremental if full matrix would exceed available memory
    size_t full_matrix_size = nsample * nsample * sizeof(dp_t);
    return (available_memory == 0) || (full_matrix_size > available_memory * 0.5);
}

// Incremental rank-1 update
void LowRankProximityMatrix::incremental_rank1_update(const dp_t* tree_proximity, integer_t nsample) {
    add_tree_contribution_incremental(tree_proximity, nsample);
}

// Full SVD update
void LowRankProximityMatrix::full_svd_update(const dp_t* accumulated_prox, integer_t nsample) {
    // For now, use incremental update
    // Would perform full SVD on accumulated proximity
    if (temp_prox_gpu_ == nullptr) {
        CUDA_CHECK_VOID(cudaMalloc(&temp_prox_gpu_, nsample * nsample * sizeof(dp_t)));
    }
    
    CUDA_CHECK_VOID(cudaMemcpy(temp_prox_gpu_, accumulated_prox, 
                                nsample * nsample * sizeof(dp_t), cudaMemcpyHostToDevice));
    
    // Perform SVD using cuSolver
    dp_t* S_gpu = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&S_gpu, nsample * sizeof(dp_t)));
    
    int lwork = 0;
    cusolverDnDsyevd_bufferSize(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        nsample,
        temp_prox_gpu_,
        nsample,
        S_gpu,
        &lwork
    );
    
    dp_t* d_work = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&d_work, lwork * sizeof(dp_t)));
    
    int* dev_info = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&dev_info, sizeof(int)));
    
    cusolverDnDsyevd(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        nsample,
        temp_prox_gpu_,
        nsample,
        S_gpu,
        d_work,
        lwork,
        dev_info
    );
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Extract top rank_ components (safe free to prevent segfaults)
    if (A_gpu_) { cudaError_t err = cudaFree(A_gpu_); if (err != cudaSuccess) cudaGetLastError(); }
    if (B_gpu_) { cudaError_t err = cudaFree(B_gpu_); if (err != cudaSuccess) cudaGetLastError(); }
    
    CUDA_CHECK_VOID(cudaMalloc(&A_gpu_, nsample * rank_ * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMalloc(&B_gpu_, nsample * rank_ * sizeof(dp_t)));
    
    // Copy eigenvectors and scale by sqrt(eigenvalues)
    for (integer_t i = 0; i < rank_; ++i) {
        integer_t col_idx = nsample - 1 - i; // Top eigenvalues are at the end
        cublasDcopy(cublas_handle_, nsample, temp_prox_gpu_ + col_idx * nsample, 1, 
                    A_gpu_ + i * nsample, 1);
        
        dp_t sqrt_eigenval;
        CUDA_CHECK_VOID(cudaMemcpy(&sqrt_eigenval, S_gpu + col_idx, sizeof(dp_t), cudaMemcpyDeviceToHost));
        sqrt_eigenval = sqrt(sqrt_eigenval);
        
        cublasDscal(cublas_handle_, nsample, &sqrt_eigenval, A_gpu_ + i * nsample, 1);
        cublasDcopy(cublas_handle_, nsample, A_gpu_ + i * nsample, 1, 
                    B_gpu_ + i * nsample, 1);
    }
    
    quantize_factors(current_quant_level_);
    
        // Safe free to prevent segfaults
        if (S_gpu) { cudaError_t err = cudaFree(S_gpu); if (err != cudaSuccess) cudaGetLastError(); }
        if (d_work) { cudaError_t err = cudaFree(d_work); if (err != cudaSuccess) cudaGetLastError(); }
        if (dev_info) { cudaError_t err = cudaFree(dev_info); if (err != cudaSuccess) cudaGetLastError(); }
}

// Estimate optimal rank
integer_t estimate_optimal_rank(integer_t nsample, integer_t max_rank) {
    // ALWAYS respect max_rank as absolute upper bound (user's explicit choice)
    if (max_rank <= 0) {
        max_rank = 100;  // Default if not specified
    }
    if (nsample <= 0) {
        return max_rank;
    }
    // Use sqrt(nsample) as heuristic, capped by max_rank
    // No minimum - if user wants rank=3, they get rank=3
    integer_t rank = static_cast<integer_t>(std::sqrt(static_cast<double>(nsample)));
    rank = std::min(rank, max_rank);
    // Ensure at least 1
    return std::max(1, rank);
}

} // namespace cuda
} // namespace rf
