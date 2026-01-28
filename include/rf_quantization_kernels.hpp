#ifndef RF_QUANTIZATION_KERNELS_HPP
#define RF_QUANTIZATION_KERNELS_HPP

#include <cuda_fp16.h>  // MUST be first
#include "rf_types.hpp"
#include "rf_config.hpp"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

// Quantization level enumeration
enum class QuantizationLevel {
    FP32 = 0,    // 32-bit single precision
    FP16 = 1,    // 16-bit half precision  
    INT8 = 2,    // 8-bit integer
    NF4 = 3      // 4-bit NormalFloat4
};

// Quantization utilities
class QuantizationUtils {
public:
    // Convert FP32 to INT8 with dynamic scaling
    static __device__ __forceinline__ int8_t fp32_to_int8(float value, float scale, float zero_point) {
        float result = value / scale + zero_point;
        // Clamp to int8 range manually (since __saturatef may not be available in all contexts)
        result = fmaxf(-128.0f, fminf(127.0f, result));
        return static_cast<int8_t>(result);
    }
    
    // Convert INT8 back to FP32
    static __device__ __forceinline__ float int8_to_fp32(int8_t value, float scale, float zero_point) {
        return (static_cast<float>(value) - zero_point) * scale;
    }
    
    // NF4 quantization (simplified)
    static __device__ __forceinline__ uint8_t fp32_to_nf4(float value) {
        // NF4: 4-bit quantization, 16 levels [0-15]
        // Map [0, 1] to [0, 7] and [-1, 0) to [8, 15]
        // This gives symmetric range around 0
        if (value >= 0.0f) {
            // Positive values: [0.0, 1.0] → [0, 7]
            // Note: 7/7 = 1.0, so mapping is to [0, 7] inclusive
            return static_cast<uint8_t>(fminf(7.0f, roundf(value * 7.0f)));
        } else {
            // Negative values: [-1.0, 0.0) → [15, 8]
            // Note: Mapping is -1.0 → 15, approaching 0 → 8
            return static_cast<uint8_t>(fmaxf(8.0f, roundf(8.0f - value * 7.0f)));
        }
    }
    
    // Convert NF4 back to FP32
    static __device__ __forceinline__ float nf4_to_fp32(uint8_t value) {
        // NF4: 4-bit quantization, 16 levels [0-15]
        // [0, 7] → [0.0, 1.0]
        // [8, 15] → [-1.0, 0.0)
        if (value <= 7) {
            // Positive values: [0, 7] → [0.0, 1.0]
            return static_cast<float>(value) / 7.0f;
        } else {
            // Negative values: [8, 15] → [0.0, -1.0]
            // value=8 → 0.0, value=15 → -1.0
            return -(static_cast<float>(value - 8)) / 7.0f;
        }
    }
    
    // Compute dynamic scaling for INT8 (host pointer version)
    static void compute_int8_scaling(const float* data, size_t count, 
                                   float& scale, float& zero_point);
    
    // Compute dynamic scaling for INT8 (GPU pointer version - stores results in device memory)
    // Returns device pointers to min/max values - avoids all host copies
    static void compute_int8_scaling_gpu(const dp_t* data_gpu, size_t count, 
                                        float* d_scale, float* d_zero_point);
};

// FP16 Proximity Kernel declaration
__global__ void cuda_proximity_fp16_kernel(const integer_t* nodexb,
                                          const integer_t* nin,
                                          integer_t nsample,
                                          const integer_t* ndbegin,
                                          const integer_t* npcase,
                                          __half* prox_fp16);

// Standard FP32 Proximity Kernel declaration
__global__ void cuda_proximity_kernel(const integer_t* nodexb,
                                     const integer_t* nin,
                                     integer_t nsample,
                                     const integer_t* ndbegin,
                                     const integer_t* npcase,
                                     dp_t* prox);

// INT8 Proximity Kernel declaration
__global__ void cuda_proximity_int8_kernel(const integer_t* nodexb,
                                          const integer_t* nin,
                                          integer_t nsample,
                                          const integer_t* ndbegin,
                                          const integer_t* npcase,
                                          int8_t* prox_int8,
                                          float scale,
                                          float zero_point);

// NF4 Proximity Kernel declaration
__global__ void cuda_proximity_nf4_kernel(const integer_t* nodexb,
                                         const integer_t* nin,
                                         integer_t nsample,
                                         const integer_t* ndbegin,
                                         const integer_t* npcase,
                                         uint8_t* prox_nf4);

// Dynamic quantization dispatcher
class QuantizationDispatcher {
public:
    static void launch_proximity_kernel(QuantizationLevel level,
                                      const integer_t* nodexb,
                                      const integer_t* nin,
                                      integer_t nsample,
                                      const integer_t* ndbegin,
                                      const integer_t* npcase,
                                      void* prox_data,
                                      int blocks, int threads);
    
    static size_t get_quantized_size(QuantizationLevel level, size_t base_size);
};

} // namespace cuda
} // namespace rf

#endif // RF_QUANTIZATION_KERNELS_HPP