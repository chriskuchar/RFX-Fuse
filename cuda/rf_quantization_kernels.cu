#include "rf_quantization_kernels.hpp"
#include "rf_cuda_config.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <vector>

// Define CUDA_CHECK_VOID if not already defined
#ifndef CUDA_CHECK_VOID
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cudaGetLastError(); \
        } \
    } while(0)
#endif

namespace rf {
namespace cuda {

// CUDA kernel to compute min/max on GPU (avoids host copy)
__global__ void compute_minmax_kernel(const dp_t* data, size_t count, float* min_out, float* max_out) {
    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize thread's min/max
    float my_min = (idx < count) ? static_cast<float>(data[idx]) : 1e30f;
    float my_max = (idx < count) ? static_cast<float>(data[idx]) : -1e30f;
    
    // Load into shared memory
    smin[tid] = my_min;
    smax[tid] = my_max;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write block results
    if (tid == 0) {
        min_out[blockIdx.x] = smin[0];
        max_out[blockIdx.x] = smax[0];
    }
}

// Second-stage reduction kernel (reduces block results to single value)
__global__ void reduce_minmax_final_kernel(const float* min_blocks, const float* max_blocks, 
                                           int num_blocks, float* min_out, float* max_out) {
    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;
    
    int tid = threadIdx.x;
    
    // Load block results
    float my_min = (tid < num_blocks) ? min_blocks[tid] : 1e30f;
    float my_max = (tid < num_blocks) ? max_blocks[tid] : -1e30f;
    
    smin[tid] = my_min;
    smax[tid] = my_max;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write final results
    if (tid == 0) {
        *min_out = smin[0];
        *max_out = smax[0];
    }
}

// Kernel to compute scale and zero_point from min/max (all on device)
__global__ void compute_scale_from_minmax_kernel(const float* d_min, const float* d_max,
                                                  float* d_scale, float* d_zero_point) {
    float min_val = *d_min;
    float max_val = *d_max;
    
    float scale = (max_val - min_val) / 255.0f;
    if (scale < 1e-10f) scale = 1.0f / 127.0f;  // Avoid division by zero
    float zero_point = -min_val / scale;
    
    *d_scale = scale;
    *d_zero_point = zero_point;
}

// QuantizationUtils implementation - device functions are now inline in header
// Only keeping host function implementation here

void QuantizationUtils::compute_int8_scaling(const float* data, size_t count, 
                                           float& scale, float& zero_point) {
    float min_val = data[0];
    float max_val = data[0];
    
    for (size_t i = 1; i < count; i++) {
        min_val = min(min_val, data[i]);
        max_val = max(max_val, data[i]);
    }
    
    scale = (max_val - min_val) / 255.0f;
    zero_point = -min_val / scale;
}

// GPU version that computes min/max on GPU and stores results in device memory
// This function avoids ALL host copies - everything stays on device
// OPTIMIZATION: Uses persistent static buffers to avoid allocation/sync overhead per call
void QuantizationUtils::compute_int8_scaling_gpu(const dp_t* data_gpu, size_t count, 
                                                 float* d_scale, float* d_zero_point) {
    if (count == 0) {
        // Set default values on device
        float default_scale = 1.0f / 127.0f;
        float default_zero = 127.0f;
        CUDA_CHECK_VOID(cudaMemcpy(d_scale, &default_scale, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_VOID(cudaMemcpy(d_zero_point, &default_zero, sizeof(float), cudaMemcpyHostToDevice));
        return;
    }
    
    // Use simple reduction for small arrays, or multi-block reduction for large arrays
    const int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;
    
    // OPTIMIZATION: Use persistent static buffers to avoid allocation/free overhead
    // These buffers persist across calls, eliminating need for sync before free
    thread_local struct ScalingBuffers {
        float* d_min_blocks = nullptr;
        float* d_max_blocks = nullptr;
        float* d_min_final = nullptr;
        float* d_max_final = nullptr;
        int allocated_blocks = 0;
        
        void ensure_allocated(int num_blocks) {
            if (num_blocks > allocated_blocks) {
                // Need to reallocate - free old buffers first
                if (d_min_blocks) cudaFree(d_min_blocks);
                if (d_max_blocks) cudaFree(d_max_blocks);
                if (d_min_final) cudaFree(d_min_final);
                if (d_max_final) cudaFree(d_max_final);
                
                // Allocate with some headroom to avoid frequent reallocation
                int alloc_blocks = std::max(num_blocks, 256);
                cudaMalloc(&d_min_blocks, alloc_blocks * sizeof(float));
                cudaMalloc(&d_max_blocks, alloc_blocks * sizeof(float));
                cudaMalloc(&d_min_final, sizeof(float));
                cudaMalloc(&d_max_final, sizeof(float));
                allocated_blocks = alloc_blocks;
            }
        }
    } buffers;
    
    buffers.ensure_allocated(num_blocks);
    
    // Launch first-stage kernel with shared memory
    size_t shared_mem = 2 * block_size * sizeof(float);
    compute_minmax_kernel<<<num_blocks, block_size, shared_mem>>>(
        data_gpu, count, buffers.d_min_blocks, buffers.d_max_blocks
    );
    
    // Launch second-stage reduction kernel (single block)
    int final_block_size = 256;
    size_t final_shared_mem = 2 * final_block_size * sizeof(float);
    reduce_minmax_final_kernel<<<1, final_block_size, final_shared_mem>>>(
        buffers.d_min_blocks, buffers.d_max_blocks, num_blocks, buffers.d_min_final, buffers.d_max_final
    );
    
    // Compute scale and zero_point on device using a kernel
    // This avoids any host copy
    dim3 scale_block(1);
    dim3 scale_grid(1);
    compute_scale_from_minmax_kernel<<<scale_grid, scale_block>>>(
        buffers.d_min_final, buffers.d_max_final, d_scale, d_zero_point
    );
    
    // NO SYNC NEEDED: Buffers are persistent and not freed
    // They will be reused on next call, so kernel can complete asynchronously
}

// FP16 Proximity Kernel (existing implementation)
__global__ void cuda_proximity_fp16_kernel(const integer_t* nodexb,
                                          const integer_t* nin,
                                          integer_t nsample,
                                          const integer_t* ndbegin,
                                          const integer_t* npcase,
                                          __half* prox_fp16) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        integer_t k = nodexb[sample_n];
        integer_t nodesize = 0;
        
        // First pass - calculate nodesize
        for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
            integer_t kk = npcase[j];
            if (nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities with FP16 precision
        if (nodesize > 0) {
            for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
                integer_t kk = npcase[j];
                if (nin[kk] > 0) {
                    __half weight_contrib = __float2half(static_cast<float>(nin[kk]) / static_cast<float>(nodesize));
                    // UPPER TRIANGLE OPTIMIZATION: Only update upper triangle + diagonal
                    if (sample_n <= kk) {
                        ::atomicAdd(reinterpret_cast<float*>(&prox_fp16[sample_n + kk * nsample]), __half2float(weight_contrib));
                    }
                }
            }
        }
    }
}

// Standard FP32 Proximity Kernel (existing implementation)
__global__ void cuda_proximity_kernel(const integer_t* nodexb,
                                     const integer_t* nin,
                                     integer_t nsample,
                                     const integer_t* ndbegin,
                                     const integer_t* npcase,
                                     dp_t* prox) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        integer_t k = nodexb[sample_n];
        integer_t nodesize = 0;
        
        // First pass - calculate nodesize
        for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
            integer_t kk = npcase[j];
            if (nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities
        if (nodesize > 0) {
            for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
                integer_t kk = npcase[j];
                if (nin[kk] > 0) {
                    dp_t weight_contrib = static_cast<dp_t>(nin[kk]) / static_cast<dp_t>(nodesize);
                    // UPPER TRIANGLE OPTIMIZATION: Only update upper triangle + diagonal
                    if (sample_n <= kk) {
                        ::atomicAdd(reinterpret_cast<float*>(&prox[sample_n + kk * nsample]), static_cast<float>(weight_contrib));
                    }
                }
            }
        }
    }
}

// INT8 Proximity Kernel
__global__ void cuda_proximity_int8_kernel(const integer_t* nodexb,
                                          const integer_t* nin,
                                          integer_t nsample,
                                          const integer_t* ndbegin,
                                          const integer_t* npcase,
                                          int8_t* prox_int8,
                                          float scale,
                                          float zero_point) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        integer_t k = nodexb[sample_n];
        integer_t nodesize = 0;
        
        // First pass - calculate nodesize
        for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
            integer_t kk = npcase[j];
            if (nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities with INT8 quantization
        if (nodesize > 0) {
            for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
                integer_t kk = npcase[j];
                if (nin[kk] > 0) {
                    float weight_contrib = static_cast<float>(nin[kk]) / static_cast<float>(nodesize);
                    int8_t quantized_weight = QuantizationUtils::fp32_to_int8(weight_contrib, scale, zero_point);
                    // UPPER TRIANGLE OPTIMIZATION: Only update upper triangle + diagonal
                    if (sample_n <= kk) {
                        ::atomicAdd(reinterpret_cast<int*>(&prox_int8[sample_n + kk * nsample]), static_cast<int>(quantized_weight));
                    }
                }
            }
        }
    }
}

// NF4 Proximity Kernel  
__global__ void cuda_proximity_nf4_kernel(const integer_t* nodexb,
                                         const integer_t* nin,
                                         integer_t nsample,
                                         const integer_t* ndbegin,
                                         const integer_t* npcase,
                                         uint8_t* prox_nf4) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        integer_t k = nodexb[sample_n];
        integer_t nodesize = 0;
        
        // First pass - calculate nodesize
        for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
            integer_t kk = npcase[j];
            if (nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities with NF4 quantization
        if (nodesize > 0) {
            for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
                integer_t kk = npcase[j];
                if (nin[kk] > 0) {
                    float weight_contrib = static_cast<float>(nin[kk]) / static_cast<float>(nodesize);
                    uint8_t quantized_weight = QuantizationUtils::fp32_to_nf4(weight_contrib);
                    // UPPER TRIANGLE OPTIMIZATION: Only update upper triangle + diagonal
                    if (sample_n <= kk) {
                        ::atomicAdd(reinterpret_cast<unsigned int*>(&prox_nf4[sample_n + kk * nsample]), static_cast<unsigned int>(quantized_weight));
                    }
                }
            }
        }
    }
}

} // namespace cuda
} // namespace rf

// QuantizationDispatcher implementation
namespace rf {
namespace cuda {

void QuantizationDispatcher::launch_proximity_kernel(QuantizationLevel level,
                                                   const integer_t* nodexb,
                                                   const integer_t* nin,
                                                   integer_t nsample,
                                                   const integer_t* ndbegin,
                                                   const integer_t* npcase,
                                                   void* prox_data,
                                                   int blocks, int threads) {
    switch (level) {
        case QuantizationLevel::FP32: {
            dp_t* prox = static_cast<dp_t*>(prox_data);
            cuda_proximity_kernel<<<blocks, threads>>>(nodexb, nin, nsample, ndbegin, npcase, prox);
            break;
        }
        case QuantizationLevel::FP16: {
            __half* prox_fp16 = static_cast<__half*>(prox_data);
            cuda_proximity_fp16_kernel<<<blocks, threads>>>(nodexb, nin, nsample, ndbegin, npcase, prox_fp16);
            break;
        }
        case QuantizationLevel::INT8: {
            // For INT8, we need to compute scaling parameters
            // This is a simplified version - in practice, you'd compute these from data
            float scale = 1.0f / 127.0f;
            float zero_point = 0.0f;
            int8_t* prox_int8 = static_cast<int8_t*>(prox_data);
            cuda_proximity_int8_kernel<<<blocks, threads>>>(nodexb, nin, nsample, ndbegin, npcase, prox_int8, scale, zero_point);
            break;
        }
        case QuantizationLevel::NF4: {
            uint8_t* prox_nf4 = static_cast<uint8_t*>(prox_data);
            cuda_proximity_nf4_kernel<<<blocks, threads>>>(nodexb, nin, nsample, ndbegin, npcase, prox_nf4);
            break;
        }
    }
}

size_t QuantizationDispatcher::get_quantized_size(QuantizationLevel level, size_t base_size) {
    switch (level) {
        case QuantizationLevel::FP32:
            return base_size * sizeof(dp_t);
        case QuantizationLevel::FP16:
            return base_size * sizeof(__half);
        case QuantizationLevel::INT8:
            return base_size * sizeof(int8_t);
        case QuantizationLevel::NF4:
            return base_size * sizeof(uint8_t);
        default:
            return base_size * sizeof(dp_t);
    }
}

} // namespace cuda
} // namespace rf
