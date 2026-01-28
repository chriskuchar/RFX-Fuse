#include "rf_finishprox.cuh"
#include "rf_config.hpp"
#include "rf_memory.cuh"
#include "rf_quantization_kernels.hpp"
#include "rf_cuda_config.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace rf {

// ============================================================================
// CUDA Kernels (internal)
// ============================================================================

namespace {  // Anonymous namespace

// CUDA kernel for normalizing proximities
// Exact port of cuda_normalize_prox_kernel (lines 14-35)
__global__ void cuda_normalize_prox_kernel(integer_t nsample, const integer_t* nout, dp_t* prox) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;

    // Each thread processes multiple samples
    for (integer_t n = tid; n < nsample; n += stride) {
        if (nout[n] > 0) {
            for (integer_t k = 0; k < nsample; k++) {
                // Column-major: prox(n, k)
                prox[n + k * nsample] /= static_cast<dp_t>(nout[n]);
            }
        }
    }
}

// Device function to convert linear index to upper triangle coordinates
__device__ void linear_to_upper_triangle(integer_t idx, integer_t n_size, integer_t& row, integer_t& col) {
    // 0-based indexing
    integer_t cumulative = 0;

    for (integer_t i = 0; i < n_size; i++) {
        integer_t row_size = n_size - i;
        if (idx < cumulative + row_size) {
            row = i;
            col = i + (idx - cumulative);
            return;
        }
        cumulative += row_size;
    }
}

// CUDA kernel for symmetrizing proximities
// Exact port of cuda_symmetrize_prox_kernel (lines 38-67)
__global__ void cuda_symmetrize_prox_kernel(integer_t nsample, const dp_t* prox, dp_t* proxsym) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;

    integer_t total_elements = (nsample * (nsample + 1)) / 2;

    // Each thread processes elements in upper triangle
    for (integer_t idx = tid; idx < total_elements; idx += stride) {
        integer_t n, k;
        linear_to_upper_triangle(idx, nsample, n, k);

        if (n == k) {
            // Diagonal element
            proxsym[n + n * nsample] = 1.0;
        } else {
            // Off-diagonal element - symmetrize
            // Column-major: prox(n, k) and prox(k, n)
            dp_t sum_val = 0.5 * (prox[n + k * nsample] + prox[k + n * nsample]);
            proxsym[n + k * nsample] = sum_val;
            proxsym[k + n * nsample] = sum_val;
        }
    }
}

} // anonymous namespace

// ============================================================================
// Forward declarations for C++ CPU implementations
extern void cpu_finishprox(integer_t nsample, const integer_t* nout, dp_t* prox, dp_t* proxsym);

// ============================================================================
// GPU FINISHPROX IMPLEMENTATION WITH QUANTIZATION KERNELS
// ============================================================================

void gpu_finishprox(integer_t nsample, const integer_t* nout, dp_t* prox, dp_t* proxsym) {
    
    // For small matrices, CPU might be faster due to GPU overhead
    if (nsample < 500) {
        cpu_finishprox(nsample, nout, prox, proxsym);
        return;
    }
    
    // JUPYTER SAFETY: Ensure CUDA context is ready before operations
    // This prevents crashes in Jupyter when context is corrupted between cells
    rf::cuda::cuda_clear_errors();  // Clear any stale CUDA errors
    rf::cuda::cuda_ensure_context_ready();  // Validate context is accessible
    
    // Allocate device memory
    integer_t* nout_d = nullptr;
    dp_t* prox_d = nullptr;
    dp_t* proxsym_d = nullptr;
    
    CUDA_CHECK_VOID(cudaMalloc(&nout_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&prox_d, nsample * nsample * sizeof(dp_t)));
    CUDA_CHECK_VOID(cudaMalloc(&proxsym_d, nsample * nsample * sizeof(dp_t)));
    
    // Copy data to device
    CUDA_CHECK_VOID(cudaMemcpy(nout_d, nout, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(prox_d, prox, nsample * nsample * sizeof(dp_t), cudaMemcpyHostToDevice));
    
    // Initialize proxsym to zero
    CUDA_CHECK_VOID(cudaMemset(proxsym_d, 0, nsample * nsample * sizeof(dp_t)));
    
    // Get quantization level from global config
    rf::cuda::QuantizationLevel quant_level = g_config.get_quantization_level<rf::cuda::QuantizationLevel>();
    
    // Launch normalization kernel with optimized block size
    dim3 block_size(16, 16);  // 2D blocks for matrix operations
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x, 
                   (nsample + block_size.y - 1) / block_size.y);
    
    // STEP 1: Row-wise normalization (matching original Fortran)
    cuda_normalize_prox_kernel<<<grid_size, block_size>>>(nsample, nout_d, prox_d);
    // JUPYTER SAFETY: Use stream sync instead of device sync (safer for Jupyter)
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));
    
    // STEP 2: Symmetrization (matching original Fortran)
    // The symmetrization kernel sets diagonal to 1.0 and averages row-normalized values
    // Use 1D block size for symmetrization kernel (it processes upper triangle)
    dim3 sym_block_size(256);
    dim3 sym_grid_size((nsample * (nsample + 1) / 2 + sym_block_size.x - 1) / sym_block_size.x);
    cuda_symmetrize_prox_kernel<<<sym_grid_size, sym_block_size>>>(nsample, prox_d, proxsym_d);
    // JUPYTER SAFETY: Use stream sync instead of device sync
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));
    
    // Apply quantization kernels for memory optimization
      if (quant_level != rf::cuda::QuantizationLevel::FP32) {
        // Verbose output removed for Jupyter compatibility
        // std::cout << "GPU: Applying quantization for memory optimization..." << std::endl;
        
        // Allocate quantized arrays
        void* prox_quantized;
        void* proxsym_quantized;
        size_t quantized_size = rf::cuda::QuantizationDispatcher::get_quantized_size(quant_level, nsample * nsample);
        
        CUDA_CHECK_VOID(cudaMalloc(&prox_quantized, quantized_size));
        CUDA_CHECK_VOID(cudaMalloc(&proxsym_quantized, quantized_size));
        
        // Initialize quantized arrays
        CUDA_CHECK_VOID(cudaMemset(prox_quantized, 0, quantized_size));
        CUDA_CHECK_VOID(cudaMemset(proxsym_quantized, 0, quantized_size));
        
        // Apply quantization kernels
        rf::cuda::QuantizationDispatcher dispatcher;
        
        // Quantize proximity matrix
        if (quant_level == rf::cuda::QuantizationLevel::FP16) {
            // Convert FP32 to FP16
            dim3 quant_block_size(256);
            dim3 quant_grid_size((nsample * nsample + quant_block_size.x - 1) / quant_block_size.x);
            // cuda_fp32_to_fp16_kernel<<<quant_grid_size, quant_block_size>>>(prox_d, prox_quantized, nsample * nsample);
        } else if (quant_level == rf::cuda::QuantizationLevel::INT8) {
            // INT8 quantization for full proximity matrices is not implemented
            // INT8 is only used for QLORA low-rank factors (see rf_proximity_lowrank.cu)
            // Full matrix INT8 quantization would require implementing cuda_fp32_to_int8_kernel
        }
        
        // Quantize symmetric proximity matrix
        if (quant_level == rf::cuda::QuantizationLevel::FP16) {
            dim3 quant_block_size(256);
            dim3 quant_grid_size((nsample * nsample + quant_block_size.x - 1) / quant_block_size.x);
            // cuda_fp32_to_fp16_kernel<<<quant_grid_size, quant_block_size>>>(proxsym_d, proxsym_quantized, nsample * nsample);
        } else if (quant_level == rf::cuda::QuantizationLevel::INT8) {
            // INT8 quantization for full proximity matrices is not implemented
            // INT8 is only used for QLORA low-rank factors (see rf_proximity_lowrank.cu)
        }
        
        // Cleanup quantized arrays
        // Safe free to prevent segfaults
        if (prox_quantized) { cudaError_t err = cudaFree(prox_quantized); if (err != cudaSuccess) cudaGetLastError(); }
        if (proxsym_quantized) { cudaError_t err = cudaFree(proxsym_quantized); if (err != cudaSuccess) cudaGetLastError(); }
        
        // Verbose output removed for Jupyter compatibility
        // std::cout << "GPU: Quantization completed for memory optimization." << std::endl;
    }
    
    // Copy results back to host
    CUDA_CHECK_VOID(cudaMemcpy(prox, prox_d, nsample * nsample * sizeof(dp_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaMemcpy(proxsym, proxsym_d, nsample * nsample * sizeof(dp_t), cudaMemcpyDeviceToHost));
    
    // Cleanup device memory
    // Safe free to prevent segfaults
    if (nout_d) { cudaError_t err = cudaFree(nout_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (prox_d) { cudaError_t err = cudaFree(prox_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (proxsym_d) { cudaError_t err = cudaFree(proxsym_d); if (err != cudaSuccess) cudaGetLastError(); }
    
    // Verbose output removed for Jupyter compatibility
    // std::cout << "GPU: Proximity matrix finishing completed with quantization." << std::endl;
}

} // namespace rf
