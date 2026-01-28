#include "rf_proximity_upper_triangle.hpp"
#include "rf_types.hpp"
#include "rf_quantization_kernels.hpp"
#include "rf_cuda_config.hpp"
#include "rf_config.hpp"  // For g_config.use_casewise
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <stdexcept>

// CUDA error checking macro
#ifndef CUDA_CHECK_VOID
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            /* Don't throw exceptions in Jupyter - just log and return */ \
            /* Exceptions can cause kernel crashes */ \
            cudaGetLastError(); /* Clear error state */ \
            return; /* Silently return on CUDA errors to prevent kernel crashes */ \
        } \
    } while(0)
#endif

namespace rf {
namespace cuda {

// ============================================================================
// CUDA Kernels: Upper Triangle Packed Format (for incremental low-rank)
// ============================================================================

// Helper function: Convert (i, j) to packed upper triangle index
__device__ __forceinline__ integer_t upper_triangle_index(integer_t i, integer_t j, integer_t nsample) {
    // Packed upper triangle: row-wise storage
    // Row i: elements [i, i+1, ..., nsample-1]
    // Index: i * nsample + j - i*(i+1)/2
    return i * nsample + j - (i * (i + 1)) / 2;
}

// Kernel to convert full matrix to upper triangle packed format
__global__ void convert_full_to_upper_triangle_kernel(
    const __half* prox_full,  // Full matrix (nsample * nsample), column-major
    __half* prox_upper,       // Output: Packed upper triangle (n(n+1)/2)
    integer_t nsample
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    
    for (size_t idx = tid; idx < upper_triangle_size; idx += stride) {
        // Convert linear index to (i, j) coordinates
        // idx = i * nsample + j - i*(i+1)/2
        // Solve for i: i * nsample + j - i*(i+1)/2 = idx, where j >= i
        integer_t i = 0;
        integer_t j = 0;
        
        // Find i such that i * (nsample - (i+1)/2) <= idx < (i+1) * (nsample - (i+2)/2)
        // Simplified: find i where the cumulative elements up to row i-1 <= idx < cumulative up to row i
        integer_t cumul = 0;
        for (i = 0; i < nsample; ++i) {
            integer_t row_size = nsample - i;  // Number of elements in row i
            if (idx < cumul + row_size) {
                j = i + (idx - cumul);
                break;
            }
            cumul += row_size;
        }
        
        if (i < nsample && j < nsample && i <= j) {
            // Copy from full matrix (column-major: i + j * nsample) to upper triangle
            prox_upper[idx] = prox_full[i + j * nsample];
        }
    }
}

// FP16 Full Matrix Kernel (matches backup C++ implementation exactly)
// Supports both case-wise (bootstrap frequency weighted) and non-case-wise (simple co-occurrence)
// Uses atomicAdd on float* like the backup (works because full matrix has proper alignment)
__global__ void cuda_proximity_fp16_full_kernel(
    const integer_t* nodexb,
    const integer_t* nin,
    integer_t nsample,
    integer_t nterm,  // Number of terminal nodes (for bounds checking)
    const integer_t* ndbegin,
    const integer_t* npcase,
    __half* prox_fp16,  // Full matrix (nsample * nsample elements), column-major
    bool use_casewise  // Case-wise: nin[kk]/nodesize, Non-case-wise: 1.0
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        // Match backup C++ implementation exactly
        integer_t k = nodexb[sample_n];
        
        // Bounds check: k must be a valid terminal node index [0, nterm-1]
        if (k < 0 || k >= nterm) {
            continue;  // Skip invalid nodexb entries (OOB samples or invalid nodes)
        }
        
        integer_t nodesize = 0;
        integer_t start_idx = ndbegin[k];
        integer_t end_idx = ndbegin[k+1];
        
        // Bounds check for ndbegin range
        if (start_idx < 0 || end_idx < start_idx) {
            continue;  // Skip invalid ndbegin ranges
        }
        
        // First pass - calculate nodesize (always calculate, like backup)
        for (integer_t j = start_idx; j < end_idx; j++) {
            if (j < 0 || j >= nsample) continue;  // Bounds check for npcase access
            integer_t kk = npcase[j];
            if (kk >= 0 && kk < nsample && nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities with FP16 precision (full matrix, column-major)
        if (nodesize > 0) {
            for (integer_t j = start_idx; j < end_idx; j++) {
                if (j < 0 || j >= nsample) continue;  // Bounds check for npcase access
                integer_t kk = npcase[j];
                if (kk >= 0 && kk < nsample && nin[kk] > 0) {
                    // Calculate weight contribution
                    float weight_contrib_float = use_casewise ?
                        static_cast<float>(nin[kk]) / static_cast<float>(nodesize) :
                        1.0f;
                    
                    // UPPER TRIANGLE OPTIMIZATION: Only update upper triangle + diagonal
                    // This matches the backup implementation exactly
                    if (sample_n <= kk) {
                        // Column-major indexing: sample_n + kk * nsample
                        // Use atomicAdd on float* (same approach as backup - works because full matrix has proper alignment)
                        ::atomicAdd(reinterpret_cast<float*>(&prox_fp16[sample_n + kk * nsample]), weight_contrib_float);
                    }
                }
            }
        }
    }
}

// FP16 Upper Triangle Packed Kernel (memory efficient for large datasets)
// Supports both case-wise (bootstrap frequency weighted) and non-case-wise (simple co-occurrence)
// Uses compare-and-swap loop for safe atomic operations on __half in packed format
// This avoids needing full matrix (20GB for 100k rows) while maintaining correctness
__global__ void cuda_proximity_upper_triangle_fp16_kernel(
    const integer_t* nodexb,
    const integer_t* nin,
    integer_t nsample,
    integer_t nterm,  // Number of terminal nodes (for bounds checking)
    const integer_t* ndbegin,
    const integer_t* npcase,
    __half* prox_upper_fp16,  // Packed upper triangle (n(n+1)/2 elements)
    bool use_casewise  // Case-wise: nin[kk]/nodesize, Non-case-wise: 1.0
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // DEBUG: Thread 0 prints kernel entry info
    // if (tid == 0) {
    //     integer_t valid_nodexb = 0;
    //     integer_t invalid_nodexb = 0;
    //     integer_t oob_count = 0;
    //     integer_t inbag_count = 0;
    //     for (int i = 0; i < nsample; ++i) {
    //         integer_t k = nodexb[i];
    //         if (k >= 0 && k < nterm) {
    //             valid_nodexb++;
    //         } else {
    //             invalid_nodexb++;
    //         }
    //         if (nin[i] == 0) {
    //             oob_count++;
    //         } else {
    //             inbag_count++;
    //         }
    //     }
        // printf("[DEBUG PROXIMITY KERNEL] Entry: nsample=%d, nterm=%d, valid_nodexb=%d, invalid_nodexb=%d, oob=%d, inbag=%d\n",
        //        nsample, nterm, valid_nodexb, invalid_nodexb, oob_count, inbag_count);
    // }
    __syncthreads();
    
    integer_t threads_processed = 0;
    integer_t threads_wrote = 0;
    integer_t threads_skipped_oob = 0;
    integer_t threads_skipped_invalid_nodexb = 0;
    integer_t threads_skipped_invalid_ndbegin = 0;
    integer_t threads_skipped_zero_nodesize = 0;
    integer_t total_updates = 0;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        threads_processed++;
        // Match CPU implementation - only process OOB samples (nin[sample_n] == 0)
        // CPU code (rf_proximity.cpp line 180): if (nin[n] == 0) { // OOB sample
        // The proximity matrix measures how often OOB samples co-occur with in-bag samples
        if (nin[sample_n] > 0) {
            continue;  // Skip in-bag samples - only process OOB samples
        }
        
        integer_t k = nodexb[sample_n];
        
        // Bounds check: k must be a valid terminal node index [0, nterm-1]
        if (k < 0 || k >= nterm) {
            threads_skipped_invalid_nodexb++;
            continue;  // Skip invalid nodexb entries
        }
        
        integer_t nodesize = 0;
        // Match CPU exactly: access ndbegin[k+1] with bounds check
        // ndbegin has size nterm+1, so k+1 <= nterm is valid
        integer_t start_idx = ndbegin[k];
        integer_t end_idx = (k + 1 <= nterm) ? ndbegin[k + 1] : nsample;
        if (end_idx > nsample) end_idx = nsample;  // Additional safety check
        
        // Minimal bounds check
        if (start_idx < 0 || end_idx < start_idx) {
            threads_skipped_invalid_ndbegin++;
            continue;  // Skip invalid ndbegin ranges
        }
        
        // First pass - calculate nodesize (sum of nin[kk] for all in-bag cases in this node)
        // This matches CPU implementation (rf_proximity.cpp line 196-203)
        for (integer_t j = start_idx; j < end_idx; j++) {
            if (j < 0 || j >= nsample) continue;  // Bounds check for npcase access
            integer_t kk = npcase[j];
            if (kk >= 0 && kk < nsample && nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities in packed upper triangle format
        // Case-wise: weight = nin[kk]/nodesize (if nodesize > 0)
        // Non-case-wise: weight = 1.0
        // Match CPU: if (nodesize > 0 || !use_casewise) (rf_proximity.cpp line 207)
        if (nodesize > 0 || !use_casewise) {
            integer_t updates_this_sample = 0;
            for (integer_t j = start_idx; j < end_idx; j++) {
                if (j < 0 || j >= nsample) continue;  // Bounds check for npcase access
                integer_t kk = npcase[j];
                if (kk >= 0 && kk < nsample && nin[kk] > 0) {  // In-bag samples only
                    // For upper triangle, we need i <= j
                    // If sample_n <= kk, use (sample_n, kk)
                    // If sample_n > kk, use (kk, sample_n) - swap to get into upper triangle
                    integer_t i = (sample_n <= kk) ? sample_n : kk;
                    integer_t j_idx = (sample_n <= kk) ? kk : sample_n;
                    
                    // Calculate weight contribution
                    // Case-wise: use bootstrap frequency weighting
                    // Non-case-wise: use simple co-occurrence (1.0)
                    float weight_contrib_float = use_casewise ?
                        static_cast<float>(nin[kk]) / static_cast<float>(nodesize) :
                        1.0f;
                    
                    integer_t upper_idx = upper_triangle_index(i, j_idx, nsample);
                    // Bounds check for upper_idx
                    integer_t max_upper_idx = (nsample * (nsample + 1)) / 2 - 1;
                    if (upper_idx >= 0 && upper_idx <= max_upper_idx) {
                        // Use atomicAdd on unsigned short* (__half is 2 bytes = unsigned short)
                        // Convert weight to __half, then use atomicAdd on the bits
                        __half weight_half = __float2half(weight_contrib_float);
                        unsigned short weight_bits = *reinterpret_cast<unsigned short*>(&weight_half);
                        unsigned short* prox_ushort_ptr = reinterpret_cast<unsigned short*>(&prox_upper_fp16[upper_idx]);
                        
                        // Atomic add using compare-and-swap loop (required for __half)
                        unsigned short old_bits, new_bits;
                        float old_val, new_val;
                        int attempts = 0;
                        const int max_attempts = 100;
                        do {
                            old_bits = *prox_ushort_ptr;
                            old_val = __half2float(*reinterpret_cast<__half*>(&old_bits));
                            new_val = old_val + weight_contrib_float;
                            __half new_half = __float2half(new_val);
                            new_bits = *reinterpret_cast<unsigned short*>(&new_half);
                            attempts++;
                            if (attempts > max_attempts) {
                                // Fallback: just write (not atomic, but prevents infinite loop)
                                *prox_ushort_ptr = new_bits;
                                break;
                            }
                        } while (atomicCAS(prox_ushort_ptr, old_bits, new_bits) != old_bits);
                        updates_this_sample++;
                        total_updates++;
                    }
                }
            }
            if (updates_this_sample > 0) {
                threads_wrote++;
            }
        } else {
            threads_skipped_zero_nodesize++;
        }
    }
    
    // DEBUG: Thread 0 prints summary at end
    // if (tid == 0) {
    //     printf("[DEBUG PROXIMITY KERNEL] Exit: threads_processed=%d, threads_wrote=%d, "
    //            "skipped_invalid_nodexb=%d, skipped_invalid_ndbegin=%d, skipped_zero_nodesize=%d, "
    //            "total_updates=%d\n",
    //            threads_processed, threads_wrote, threads_skipped_invalid_nodexb,
    //            threads_skipped_invalid_ndbegin, threads_skipped_zero_nodesize, total_updates);
    // }
}

// INT8 Upper Triangle Packed Kernel
__global__ void cuda_proximity_upper_triangle_int8_kernel(
    const integer_t* nodexb,
    const integer_t* nin,
    integer_t nsample,
    integer_t nterm,
    const integer_t* ndbegin,
    const integer_t* npcase,
    int8_t* prox_upper_int8,  // Packed upper triangle (n(n+1)/2 elements)
    float scale,
    float zero_point
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        integer_t k = nodexb[sample_n];
        
        // Validate k before using it in ndbegin[k]
        // Invalid nodexb values (-1) would cause out-of-bounds access
        if (k < 0 || k >= nterm) continue;
        
        integer_t nodesize = 0;
        
        // First pass - calculate nodesize
        for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
            integer_t kk = npcase[j];
            if (kk >= 0 && kk < nsample && nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities in packed upper triangle format
        if (nodesize > 0) {
            for (integer_t j = ndbegin[k]; j < ndbegin[k+1]; j++) {
                integer_t kk = npcase[j];
                if (kk >= 0 && kk < nsample && nin[kk] > 0) {  // In-bag samples only
                    // For upper triangle, we need i <= j
                    // If sample_n <= kk, use (sample_n, kk)
                    // If sample_n > kk, use (kk, sample_n) - swap to get into upper triangle
                    integer_t i = (sample_n <= kk) ? sample_n : kk;
                    integer_t j_idx = (sample_n <= kk) ? kk : sample_n;
                    
                    float weight_contrib = static_cast<float>(nin[kk]) / static_cast<float>(nodesize);
                    int8_t quantized_weight = QuantizationUtils::fp32_to_int8(weight_contrib, scale, zero_point);
                    integer_t upper_idx = upper_triangle_index(i, j_idx, nsample);
                    
                    // atomicAdd on int* requires 4-byte alignment
                    // Align to 4-byte boundary and update the specific byte within the word
                    int32_t* word_ptr = reinterpret_cast<int32_t*>(&prox_upper_int8[upper_idx & ~3]);  // Align to 4-byte boundary
                    int byte_offset = upper_idx & 3;  // Byte offset within word (0-3)
                    
                    // Use atomicCAS to update the specific byte within the word
                    int32_t old_word, new_word;
                    int attempts = 0;
                    const int max_attempts = 100;
                    do {
                        old_word = *word_ptr;
                        int8_t old_byte = static_cast<int8_t>((old_word >> (byte_offset * 8)) & 0xFF);
                        int8_t new_byte = old_byte + quantized_weight;
                        new_word = old_word;
                        new_word &= ~(0xFFUL << (byte_offset * 8));  // Clear the byte
                        new_word |= (static_cast<int32_t>(static_cast<uint8_t>(new_byte)) << (byte_offset * 8));  // Set the byte
                        attempts++;
                        if (attempts > max_attempts) {
                            // Fallback: just write (not atomic, but prevents infinite loop)
                            *word_ptr = new_word;
                            break;
                        }
                    } while (atomicCAS(reinterpret_cast<unsigned int*>(word_ptr), 
                                     static_cast<unsigned int>(old_word), 
                                     static_cast<unsigned int>(new_word)) != static_cast<unsigned int>(old_word));
                }
            }
        }
    }
}

// NF4 Upper Triangle Packed Kernel
__global__ void cuda_proximity_upper_triangle_nf4_kernel(
    const integer_t* nodexb,
    const integer_t* nin,
    integer_t nsample,
    integer_t nterm,
    const integer_t* ndbegin,
    const integer_t* npcase,
    uint8_t* prox_upper_nf4,  // Packed upper triangle (n(n+1)/2 elements, 2 values per byte)
    bool use_casewise
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // DEBUG: Thread 0 prints kernel entry info
    // if (tid == 0) {
    //     integer_t valid_nodexb = 0;
    //     integer_t invalid_nodexb = 0;
    //     integer_t oob_count = 0;
    //     integer_t inbag_count = 0;
    //     for (int i = 0; i < nsample; ++i) {
    //         integer_t k = nodexb[i];
    //         if (k >= 0 && k < nterm) {
    //             valid_nodexb++;
    //         } else {
    //             invalid_nodexb++;
    //         }
    //         if (nin[i] == 0) {
    //             oob_count++;
    //         } else {
    //             inbag_count++;
    //         }
    //     }
        // printf("[DEBUG PROXIMITY NF4 KERNEL] Entry: nsample=%d, nterm=%d, valid_nodexb=%d, invalid_nodexb=%d, oob=%d, inbag=%d\n",
        //        nsample, nterm, valid_nodexb, invalid_nodexb, oob_count, inbag_count);
    // }
    __syncthreads();
    
    // Use shared memory for debug counters (aggregated across all threads)
    __shared__ integer_t shared_threads_processed;
    __shared__ integer_t shared_threads_wrote;
    __shared__ integer_t shared_threads_skipped_invalid_nodexb;
    __shared__ integer_t shared_threads_skipped_invalid_ndbegin;
    __shared__ integer_t shared_threads_skipped_zero_nodesize;
    __shared__ integer_t shared_total_updates;
    
    // Initialize shared memory (thread 0 only)
    if (tid == 0) {
        shared_threads_processed = 0;
        shared_threads_wrote = 0;
        shared_threads_skipped_invalid_nodexb = 0;
        shared_threads_skipped_invalid_ndbegin = 0;
        shared_threads_skipped_zero_nodesize = 0;
        shared_total_updates = 0;
    }
    __syncthreads();
    
    // Thread-local counters
    integer_t threads_processed = 0;
    integer_t threads_wrote = 0;
    integer_t threads_skipped_invalid_nodexb = 0;
    integer_t threads_skipped_invalid_ndbegin = 0;
    integer_t threads_skipped_zero_nodesize = 0;
    integer_t total_updates = 0;
    
    for (int sample_n = tid; sample_n < nsample; sample_n += stride) {
        threads_processed++;
        // Match CPU implementation - only process OOB samples (nin[sample_n] == 0)
        if (nin[sample_n] > 0) {
            continue;  // Skip in-bag samples - only process OOB samples
        }
        
        integer_t k = nodexb[sample_n];
        
        // Bounds check: k must be a valid terminal node index [0, nterm-1]
        if (k < 0 || k >= nterm) {
            threads_skipped_invalid_nodexb++;
            continue;  // Skip invalid nodexb entries
        }
        
        integer_t nodesize = 0;
        // Match CPU exactly: access ndbegin[k+1] with bounds check
        integer_t start_idx = ndbegin[k];
        integer_t end_idx = (k + 1 <= nterm) ? ndbegin[k + 1] : nsample;
        if (end_idx > nsample) end_idx = nsample;  // Additional safety check
        
        // Minimal bounds check
        if (start_idx < 0 || end_idx < start_idx) {
            threads_skipped_invalid_ndbegin++;
            continue;  // Skip invalid ndbegin ranges
        }
        
        // First pass - calculate nodesize (sum of nin[kk] for all in-bag cases in this node)
        for (integer_t j = start_idx; j < end_idx; j++) {
            if (j < 0 || j >= nsample) continue;  // Bounds check for npcase access
            integer_t kk = npcase[j];
            if (kk >= 0 && kk < nsample && nin[kk] > 0) {
                nodesize += nin[kk];
            }
        }
        
        // Second pass - update proximities in packed upper triangle format
        // Case-wise: weight = nin[kk]/nodesize (if nodesize > 0)
        // Non-case-wise: weight = 1.0
        if (nodesize > 0 || !use_casewise) {
            integer_t updates_this_sample = 0;
            for (integer_t j = start_idx; j < end_idx; j++) {
                if (j < 0 || j >= nsample) continue;  // Bounds check for npcase access
                integer_t kk = npcase[j];
                // For OOB sample_n, update proximity with in-bag sample kk
                if (kk >= 0 && kk < nsample && nin[kk] > 0) {
                    // For upper triangle, we need i <= j
                    // If sample_n <= kk, use (sample_n, kk)
                    // If sample_n > kk, use (kk, sample_n) - swap to get into upper triangle
                    integer_t i = (sample_n <= kk) ? sample_n : kk;
                    integer_t j_idx = (sample_n <= kk) ? kk : sample_n;
                    
                    // Calculate weight contribution
                    float weight_contrib_float = use_casewise ?
                        static_cast<float>(nin[kk]) / static_cast<float>(nodesize) :
                        1.0f;
                    
                    integer_t upper_idx = upper_triangle_index(i, j_idx, nsample);
                    
                    // Bounds check for upper_idx
                    integer_t max_upper_idx = (nsample * (nsample + 1)) / 2 - 1;
                    if (upper_idx >= 0 && upper_idx <= max_upper_idx) {
                        // NF4 is packed: 2 values per byte
                        integer_t packed_idx = upper_idx / 2;
                        integer_t bit_offset = (upper_idx % 2) * 4;  // 0 or 4 bits
                        
                        // Bounds check for packed_idx
                        integer_t max_packed_idx = ((nsample * (nsample + 1)) / 2 + 1) / 2 - 1;
                        if (packed_idx >= 0 && packed_idx <= max_packed_idx) {
                            // atomicCAS requires 4-byte alignment
                            // Align to 4-byte boundary and update the specific byte within the word
                            uint32_t* word_ptr = reinterpret_cast<uint32_t*>(&prox_upper_nf4[packed_idx & ~3]);  // Align to 4-byte boundary
                            int byte_offset = packed_idx & 3;  // Byte offset within word (0-3)
                            
                            uint32_t old_word, new_word;
                            int attempts = 0;
                            const int max_attempts = 100;
                            
                            do {
                                old_word = *word_ptr;
                                uint8_t old_byte = (old_word >> (byte_offset * 8)) & 0xFF;
                                uint8_t new_byte = old_byte;
                                
                                if (bit_offset == 0) {
                                    // Update lower 4 bits
                                    uint8_t old_lower = old_byte & 0x0F;
                                    float old_val = QuantizationUtils::nf4_to_fp32(old_lower);
                                    float new_val = old_val + weight_contrib_float;
                                    uint8_t new_lower = QuantizationUtils::fp32_to_nf4(new_val) & 0x0F;
                                    new_byte = (old_byte & 0xF0) | new_lower;
                                } else {
                                    // Update upper 4 bits
                                    uint8_t old_upper = (old_byte >> 4) & 0x0F;
                                    float old_val = QuantizationUtils::nf4_to_fp32(old_upper);
                                    float new_val = old_val + weight_contrib_float;
                                    uint8_t new_upper = QuantizationUtils::fp32_to_nf4(new_val) & 0x0F;
                                    new_byte = (old_byte & 0x0F) | (new_upper << 4);
                                }
                                
                                // Update the word with the new byte value
                                new_word = old_word;
                                new_word &= ~(0xFFUL << (byte_offset * 8));  // Clear the byte
                                new_word |= (static_cast<uint32_t>(new_byte) << (byte_offset * 8));  // Set the byte
                                
                                attempts++;
                                if (attempts > max_attempts) {
                                    // Fallback: just write (not atomic, but prevents infinite loop)
                                    *word_ptr = new_word;
                                    break;
                                }
                            } while (atomicCAS(word_ptr, old_word, new_word) != old_word);
                            updates_this_sample++;
                            total_updates++;
                        }
                    }
                }  // End if (kk valid and in-bag)
            }
            if (updates_this_sample > 0) {
                threads_wrote++;
            }
        } else {
            threads_skipped_zero_nodesize++;
        }
    }
    
    // Aggregate thread-local counters into shared memory using atomic operations
    atomicAdd(&shared_threads_processed, threads_processed);
    atomicAdd(&shared_threads_wrote, threads_wrote);
    atomicAdd(&shared_threads_skipped_invalid_nodexb, threads_skipped_invalid_nodexb);
    atomicAdd(&shared_threads_skipped_invalid_ndbegin, threads_skipped_invalid_ndbegin);
    atomicAdd(&shared_threads_skipped_zero_nodesize, threads_skipped_zero_nodesize);
    atomicAdd(&shared_total_updates, total_updates);
    __syncthreads();
    
    // DEBUG: Thread 0 prints summary at end (aggregated across all threads)
    // if (tid == 0) {
    //     printf("[DEBUG PROXIMITY NF4 KERNEL] Exit: threads_processed=%d, threads_wrote=%d, "
    //            "skipped_invalid_nodexb=%d, skipped_invalid_ndbegin=%d, skipped_zero_nodesize=%d, "
    //            "total_updates=%d\n",
    //            shared_threads_processed, shared_threads_wrote, shared_threads_skipped_invalid_nodexb,
    //            shared_threads_skipped_invalid_ndbegin, shared_threads_skipped_zero_nodesize, shared_total_updates);
    // }
}

// ============================================================================
// GPU Proximity Functions: Upper Triangle Packed Format
// ============================================================================

void gpu_proximity_upper_triangle_fp16(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    __half* prox_upper_fp16,  // __half from global namespace
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
) {
    // std::cout << "[DEBUG PROXIMITY FP16] Entry: nsample=" << nsample 
    //           << ", nnode=" << nnode << std::endl;
    // Check GPU availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        return;
    }
    // std::cout << "gpu_proximity_upper_triangle_fp16: GPU available, device_count=" << device_count << std::endl;
    // std::cout.flush();
    
    // Compute workspace arrays from nodestatus and nodextr (if not provided)
    // This matches the pattern used in cpu_proximity and gpu_proximity
    // std::cout << "gpu_proximity_upper_triangle_fp16: Allocating workspace arrays..." << std::endl;
    // std::cout.flush();
    std::vector<integer_t> nod_local(nnode);
    // ncount_local must be size nterm (number of terminal nodes), not nsample!
    // We'll resize it after computing nterm
    std::vector<integer_t> ncount_local;
    std::vector<integer_t> ncn_local(nsample);
    std::vector<integer_t> nodexb_local(nsample);
    // Will resize after computing nterm
    std::vector<integer_t> ndbegin_local;
    std::vector<integer_t> npcase_local(nsample);
    // std::cout << "gpu_proximity_upper_triangle_fp16: Workspace arrays allocated" << std::endl;
    // std::cout.flush();
    
    // Compute nod (terminal node mapping) first to get nterm
    integer_t nterm = 0;
    
    // Validate nnode before using it
    if (nnode <= 0) {
        // Silently return - invalid nnode (don't print errors in Jupyter)
        return;
    }
    
    for (integer_t k = 0; k < nnode; ++k) {
        if (nodestatus[k] == -1) {
            nod_local[k] = nterm;
            nterm++;
        } else {
            nod_local[k] = -1;
        }
    }
    
    // Now that we know nterm, resize ncount_local to the correct size
    // ncount[k] counts in-bag samples in terminal node k, so we need nterm elements
    ncount_local.resize(nterm, 0);
    // Validate inputs before proceeding
    if (nodextr == nullptr || nodestatus == nullptr) {
        // Silently return - null pointers (don't print errors in Jupyter)
        return;
    }
    
    if (nterm == 0) {
        // Silently return - no terminal nodes means no proximity to compute
        // Verbose error messages can cause kernel crashes in Jupyter
        return;
    }
    
    // Check if we have any in-bag samples (silently skip if not)
    integer_t inbag_count = 0;
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin[n] > 0) inbag_count++;
    }
    // No warning output - just return if no in-bag samples
    if (inbag_count == 0) {
        return;
    }
    
    // Compute nodexb and ncount
    // Compute nodexb for ALL samples (both in-bag and OOB)
    // The kernel will filter to only process OOB samples, but nodexb must be valid for all
    // This matches the backup implementations (Fortran and C++ backup)
    integer_t valid_nodexb_count = 0;
    integer_t invalid_nodexb_count = 0;
    integer_t oob_samples_count = 0;
    
    // DEBUG: Check nterm and nnode before computing nodexb
    // std::cout << "[DEBUG PROXIMITY PRE-WORKSPACE] nterm=" << nterm 
    //           << ", nnode=" << nnode << ", nsample=" << nsample << std::endl;
    for (integer_t n = 0; n < nsample; ++n) {
        // Count OOB samples (for debugging)
        if (nin[n] == 0) {
            oob_samples_count++;
        }
        
        // Check if nodextr[n] is valid and points to a terminal node
        if (nodextr[n] < 0 || nodextr[n] >= nnode) {
            nodexb_local[n] = -1;
            ncn_local[n] = 0;
            invalid_nodexb_count++;
            continue;
        }
        
        integer_t k = nod_local[nodextr[n]];
        if (k >= 0 && k < nterm) {
            // Only count in-bag samples for ncount (used to compute ndbegin)
            // OOB samples still get valid nodexb, but don't contribute to ncount
            if (nin[n] > 0) {
            ncount_local[k]++;
            ncn_local[n] = ncount_local[k];
            } else {
                // OOB sample: set ncn to 0 (not used for ndbegin computation)
                ncn_local[n] = 0;
            }
            nodexb_local[n] = k;
            valid_nodexb_count++;
        } else {
            nodexb_local[n] = -1;
            ncn_local[n] = 0;
            invalid_nodexb_count++;
        }
    }
    // Resize ndbegin_local to nterm + 1 (only need space for terminal nodes)
    ndbegin_local.resize(nterm + 1, 0);
    
    // Compute ndbegin
    // ndbegin[k] is the starting index in npcase for terminal node k
    // ndbegin[k+1] - ndbegin[k] = ncount[k] (number of in-bag samples in node k)
    // ndbegin[0] = 0, ndbegin[nterm] = total number of in-bag samples (should be <= nsample)
    if (nterm > 0) {
        ndbegin_local[0] = 0;
        for (integer_t k = 1; k <= nterm; ++k) {
            // Ensure ncount[k-1] is non-negative (it should be, but validate)
            integer_t count = (k-1 < nterm) ? ncount_local[k-1] : 0;
            if (count < 0) count = 0;  // Safety: ensure non-negative
            ndbegin_local[k] = ndbegin_local[k-1] + count;
        }
        
        // Validate that ndbegin[nterm] <= nsample
        // This ensures we don't exceed array bounds when accessing npcase
        // ndbegin[nterm] is the total number of in-bag samples across all terminal nodes
        // It should never exceed nsample (total number of samples)
        // 
        // Additionally, ndbegin[nterm] should equal the total number of in-bag samples
        // If it doesn't, there's a mismatch in ncount computation
        integer_t total_inbag_from_ndbegin = ndbegin_local[nterm];
        integer_t total_inbag_from_nin = 0;
        for (integer_t n = 0; n < nsample; ++n) {
            if (nin[n] > 0) total_inbag_from_nin++;
        }
        
        // Validate: ndbegin[nterm] should equal total in-bag samples and be <= nsample
        if (total_inbag_from_ndbegin > nsample || 
            (total_inbag_from_ndbegin != total_inbag_from_nin && total_inbag_from_nin > 0)) {
            // Mismatch detected: ndbegin[nterm] doesn't match actual in-bag count
            // This indicates ncount was computed incorrectly
            // DEBUG: Print mismatch for debugging
            // std::cout << "[DEBUG PROXIMITY VALIDATION] Mismatch: ndbegin[nterm]=" << total_inbag_from_ndbegin 
            //           << ", total_inbag_from_nin=" << total_inbag_from_nin 
            //           << ", nterm=" << nterm << std::endl;
            // Don't return early - try to continue anyway (might still work)
            // But fix ndbegin[nterm] to match total_inbag_from_nin to avoid bounds issues
            if (total_inbag_from_ndbegin != total_inbag_from_nin && total_inbag_from_nin > 0) {
                ndbegin_local[nterm] = total_inbag_from_nin;  // Fix the mismatch
            }
        }
    }
    
    // DEBUG output removed for Jupyter stability - debug statements can cause kernel crashes
    
    // Compute npcase
    // Build npcase for ALL samples (both in-bag and OOB) with valid nodexb
    // This matches the CPU implementation (rf_proximity.cpp line 161-171)
    // CPU code adds ALL samples where k >= 0 && k < nterm, regardless of ncn[n]
    // However, ncn[n] is 0 for OOB samples, so kn = ndbegin[k] + ncn[n] - 1 = ndbegin[k] - 1
    // This would be out of bounds, so CPU effectively only adds in-bag samples (where ncn[n] > 0)
    // But the condition doesn't explicitly check ncn[n] > 0, it just uses the calculation
    // For GPU, we match CPU exactly: add all samples where k >= 0 && k < nterm
    // But only if the calculated index is valid (which will only be true for in-bag samples)
    for (integer_t n = 0; n < nsample; ++n) {
        integer_t k = nodexb_local[n];
        if (k >= 0 && k < nterm) {  // Match CPU: check k, not ncn
            // Position in terminal node structure: ndbegin[k] + (ncn[n] - 1)
            // ncn[n] is 1-based position, so subtract 1 for 0-based indexing
            // For OOB samples, ncn[n] = 0, so idx = ndbegin[k] - 1, which is invalid
            // So effectively only in-bag samples (ncn[n] > 0) get added
            integer_t idx = ndbegin_local[k] + ncn_local[n] - 1;
            if (idx >= 0 && idx < nsample) {
                npcase_local[idx] = n;
            }
        }
    }
    
    // DEBUG: Check workspace arrays (always print, not conditional)
    // std::cout << "[DEBUG PROXIMITY WORKSPACE] nterm=" << nterm 
    //           << ", valid_nodexb_count=" << valid_nodexb_count 
    //           << ", invalid_nodexb_count=" << invalid_nodexb_count << std::endl;
    
    // DEBUG: Check workspace arrays (always print for debugging)
    // std::cout << "[DEBUG PROXIMITY WORKSPACE CHECK] nterm=" << nterm 
    //           << ", valid_nodexb_count=" << valid_nodexb_count << std::endl;
    
    if (nterm > 0 && valid_nodexb_count > 0) {
        integer_t nodexb_valid = 0;
        integer_t nodexb_invalid = 0;
        integer_t npcase_count = 0;
        integer_t npcase_valid = 0;
        integer_t npcase_invalid = 0;
        integer_t oob_count = 0;
        integer_t inbag_count = 0;
        for (integer_t i = 0; i < nsample; ++i) {
            if (npcase[i] >= 0 && npcase[i] < nsample) {
                npcase_valid++;
            } else if (npcase[i] != -1) {
                npcase_invalid++;
            }
            if (nin[i] == 0) {
                oob_count++;
            } else {
                inbag_count++;
            }
        }
        // std::cout << "[DEBUG PROXIMITY WORKSPACE] nterm=" << nterm 
        //           << ", nodexb_valid=" << nodexb_valid 
        //           << ", nodexb_invalid=" << nodexb_invalid
        //           << ", npcase_valid=" << npcase_valid 
        //           << ", npcase_invalid=" << npcase_invalid 
        //           << ", npcase_count=" << npcase_count
        //           << ", oob=" << oob_count << ", inbag=" << inbag_count << std::endl;
        integer_t total_inbag_debug = 0;
        for (integer_t n = 0; n < nsample; ++n) {
            if (nodexb_local[n] >= 0 && nodexb_local[n] < nterm) {
                nodexb_valid++;
            } else if (nodexb_local[n] == -1) {
                nodexb_invalid++;
            }
            if (nin[n] > 0) total_inbag_debug++;
        }
        for (integer_t k = 0; k < nterm; ++k) {
            if (ncount_local[k] > 0) {
                npcase_count += ncount_local[k];
            }
        }
        // std::cout << "[DEBUG PROXIMITY WORKSPACE] nterm=" << nterm 
        //           << ", nodexb_valid=" << nodexb_valid 
        //           << ", nodexb_invalid=" << nodexb_invalid
        //           << ", total_inbag=" << total_inbag_debug
        //           << ", ndbegin[nterm]=" << (nterm > 0 ? ndbegin_local[nterm] : 0)
        //           << ", npcase_count=" << npcase_count << std::endl;
    }
    
    // Use computed workspace arrays
    const integer_t* nodexb_ptr = nodexb_local.data();
    const integer_t* ndbegin_ptr = ndbegin_local.data();
    const integer_t* npcase_ptr = npcase_local.data();
    
    // Compute sizes
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    
    // For low-rank (QLoRA), we always use upper triangle directly - no need for full matrix
    // This is more memory efficient and avoids unnecessary conversions
    bool use_full_matrix_temp = false;  // Always use upper triangle for low-rank
    
    // DEBUG: Print which path we're using
    // std::cerr << "DEBUG: gpu_proximity_upper_triangle_fp16 - nsample=" << nsample 
    //           << ", use_full_matrix_temp=" << use_full_matrix_temp << std::endl;
    // std::cerr.flush();
    
    // For GPU mode, we always receive GPU pointers - no need to check
    // Allocate GPU memory (or use provided GPU pointer)
    integer_t* nodexb_d, *nin_d, *ndbegin_d, *npcase_d;
    __half* prox_upper_fp16_d;  // Output upper triangle (provided pointer)
    bool nin_is_gpu = false;  // Will be set below
    
    // Validate output pointer before using it
    cudaPointerAttributes prox_attrs;
    cudaError_t prox_attr_err = cudaPointerGetAttributes(&prox_attrs, prox_upper_fp16);
    if (prox_attr_err != cudaSuccess || prox_attrs.type != cudaMemoryTypeDevice) {
        cudaGetLastError(); // Clear error state
        return;
    }
    
    // Use provided GPU pointer directly for output - no allocation needed!
    prox_upper_fp16_d = prox_upper_fp16;
    
    cudaError_t err;
    err = cudaMalloc(&nodexb_d, nsample * sizeof(integer_t));
    if (err != cudaSuccess) {
        // Silently return on CUDA errors (don't throw exceptions in Jupyter)
        cudaGetLastError(); // Clear error state
        return;
    }
    
    // Check if nin is already on GPU before allocating
    cudaPointerAttributes nin_attrs;
    cudaError_t nin_attr_err = cudaPointerGetAttributes(&nin_attrs, const_cast<integer_t*>(nin));
    nin_is_gpu = (nin_attr_err == cudaSuccess && nin_attrs.type == cudaMemoryTypeDevice);
    
    if (!nin_is_gpu) {
        err = cudaMalloc(&nin_d, nsample * sizeof(integer_t));
        if (err != cudaSuccess) {
            // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
            // Safe free to prevent segfaults
            if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
            cudaGetLastError(); // Clear error state
            return;
        }
    } else {
        // nin is already on GPU - set nin_d to point to it (don't allocate)
        nin_d = const_cast<integer_t*>(nin);  // Safe cast since we're only reading
    }
    
    err = cudaMalloc(&ndbegin_d, (nterm + 1) * sizeof(integer_t));
    if (err != cudaSuccess) {
        // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        cudaGetLastError(); // Clear error state
        return;
    }
    
    err = cudaMalloc(&npcase_d, nsample * sizeof(integer_t));
    if (err != cudaSuccess) {
        // std::cerr << "ERROR: cudaMalloc failed for npcase_d (size=" << (nsample * sizeof(integer_t)) 
        //           << "): " << cudaGetErrorString(err) << std::endl;
        // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        // Safe free to prevent segfaults
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        cudaGetLastError(); // Clear error state
        return; // Return silently instead of throwing
    }
    
    // Initialize upper triangle to zero
    err = cudaMemset(prox_upper_fp16_d, 0, upper_triangle_size * sizeof(__half));
    if (err != cudaSuccess) {
        // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        // Safe free to prevent segfaults
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        cudaGetLastError(); // Clear error state
        return;
    }
    
    // Copy data to GPU with error checking
    err = cudaMemcpy(nodexb_d, nodexb_ptr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        // Safe free to prevent segfaults
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        cudaGetLastError(); // Clear error state
        return;
    }
    
    // Use nin based on whether it's on GPU or CPU (nin_is_gpu was set above)
    if (!nin_is_gpu) {
        // nin is on CPU - copy to GPU
        err = cudaMemcpy(nin_d, nin, nsample * sizeof(integer_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
            // Safe free to prevent segfaults
            if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
            // Safe free to prevent segfaults
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
            // Safe free to prevent segfaults
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
            cudaGetLastError(); // Clear error state
            return;
        }
    }
    // If nin_is_gpu, nin_d was already set above to point to the GPU memory
    
    err = cudaMemcpy(ndbegin_d, ndbegin_ptr, (nterm + 1) * sizeof(integer_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        // Safe free to prevent segfaults
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        cudaGetLastError(); // Clear error state
        return;
    }
    
    err = cudaMemcpy(npcase_d, npcase_ptr, nsample * sizeof(integer_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        // Silently free and return on CUDA errors (don't throw exceptions in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        // Safe free to prevent segfaults
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        cudaGetLastError(); // Clear error state
        return;
    }
    
    // Get use_casewise from global config
    bool use_casewise = rf::g_config.use_casewise;
    
    // Validate all parameters before kernel launch to prevent "invalid argument" errors
    // This is especially important in Jupyter where errors can crash the kernel
    if (nsample <= 0) {
        // Silently return - invalid nsample (don't print errors in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    if (nterm <= 0 || nterm > nsample) {
        // Silently return - invalid nterm (don't print errors in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // Validate device pointers are not null
    if (nodexb_d == nullptr || nin_d == nullptr || ndbegin_d == nullptr || 
        npcase_d == nullptr || prox_upper_fp16_d == nullptr) {
        // Silently return - null pointers (don't print errors in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // Validate ndbegin array bounds (ndbegin[k+1] must be >= ndbegin[k] and <= nsample)
    bool ndbegin_valid = true;
    for (integer_t k = 0; k < nterm; ++k) {
        if (ndbegin_local[k] < 0 || ndbegin_local[k] > nsample ||
            ndbegin_local[k+1] < ndbegin_local[k] || ndbegin_local[k+1] > nsample) {
            ndbegin_valid = false;
            break;
        }
    }
    if (!ndbegin_valid) {
        // Silently return - invalid ndbegin array (don't print errors in Jupyter)
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    // Ensure grid_size is at least 1 (CUDA requires at least 1 block)
    if (grid_size.x == 0) {
        grid_size.x = 1;
    }
    
    // Get use_casewise from global config (for low-rank mode, we always use casewise from config)
    bool use_casewise_val = g_config.use_casewise;
    
    // DEBUG: Check workspace arrays before kernel launch
    // std::cout << "[DEBUG PROXIMITY KERNEL LAUNCH] grid_size=" << grid_size.x 
    //           << ", block_size=" << block_size.x << ", nsample=" << nsample 
    //           << ", nterm=" << nterm << ", use_casewise=" << use_casewise_val << std::endl;
    
    // DEBUG: Print workspace info before kernel launch
    // std::cout << "[DEBUG PROXIMITY KERNEL] About to launch: nsample=" << nsample 
    //           << ", nterm=" << nterm << ", use_casewise=" << use_casewise_val << std::endl;
    
    // Always use upper triangle kernel directly for low-rank (no full matrix needed)
    cuda_proximity_upper_triangle_fp16_kernel<<<grid_size, block_size>>>(
        nodexb_d, nin_d, nsample, nterm, ndbegin_d, npcase_d, prox_upper_fp16_d, use_casewise_val
    );
    
    // Check kernel execution
    cudaError_t kernel_err = cudaGetLastError();
    // if (kernel_err != cudaSuccess) {
    //     std::cerr << "[ERROR] Kernel launch failed: " << cudaGetErrorString(kernel_err) << std::endl;
    // }
    
    // Synchronize to ensure kernel completes before checking results
    cudaError_t sync_err = cudaDeviceSynchronize();
    // if (sync_err != cudaSuccess) {
    //     std::cerr << "[ERROR] Kernel sync failed: " << cudaGetErrorString(sync_err) << std::endl;
    // }
    
    // Check for kernel launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        // Silently handle launch errors (don't print in Jupyter to avoid crashes)
        // The error is likely due to invalid parameters, which we've now validated above
        cudaGetLastError(); // Clear error state
        // Cleanup and return
        // Safe free to prevent segfaults
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            // Safe free to prevent segfaults
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        // Safe free to prevent segfaults
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        return;
    }
    
    // std::cout << "[PROXIMITY] About to sync after proximity kernel (use_casewise=" << use_casewise << ")..." << std::endl;
    // std::cout.flush();
    
    // Use stream sync for Jupyter safety with explicit error checking
    cudaError_t sync_err_prox = cudaStreamSynchronize(0);
    if (sync_err_prox != cudaSuccess) {
        // std::cerr << "ERROR: Stream sync failed after proximity kernel: " << cudaGetErrorString(sync_err_prox) << std::endl;
        // std::cerr << "ERROR: This may indicate a kernel hang. use_casewise=" << use_casewise << std::endl;
        cudaGetLastError();  // Clear error state
        // Cleanup and return
        if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (!nin_is_gpu) {
            if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
        }
        if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
        if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
        return;  // Exit early to prevent hang
    }
    
    // Debug print removed to avoid potential stream corruption issues
    // std::cout << "[PROXIMITY] Proximity kernel sync completed successfully" << std::endl;
    // std::cout.flush();
    
    // Check for any remaining errors (for both paths)
    
    // DEBUG: Check if proximity matrix has non-zero values
    std::vector<__half> prox_check(upper_triangle_size);
    cudaError_t check_err = cudaMemcpy(prox_check.data(), prox_upper_fp16_d, 
                                        upper_triangle_size * sizeof(__half), cudaMemcpyDeviceToHost);
    if (check_err == cudaSuccess) {
        float max_prox = 0.0f;
        float sum_prox = 0.0f;
        int non_zero_prox = 0;
        for (size_t i = 0; i < upper_triangle_size; ++i) {
            float val = __half2float(prox_check[i]);
            if (val > max_prox) max_prox = val;
            sum_prox += val;
            if (val > 1e-6f) non_zero_prox++;
        }
        // DEBUG output removed for Jupyter stability
        if (max_prox < 1e-6f) {
            // Silently skip - proximity is all zeros (don't print warnings in Jupyter)
        }
    } else {
        // Silently skip - failed to check proximity matrix (don't print warnings in Jupyter)
    }
    
    // Cleanup: Free only the arrays we allocated (not the ones passed in)
    // Safe free to prevent segfaults
    if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (!nin_is_gpu && nin_d) {
        cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); // Only free if we allocated it
    }
    if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
    
    // No need to free prox_upper_fp16_d - it was passed in and will be freed by caller
}

void gpu_proximity_upper_triangle_int8(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    int8_t* prox_upper_int8,  // Output: Packed upper triangle INT8
    float scale,
    float zero_point,
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
) {
    // Check GPU availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        return;
    }
    
    // Validate inputs
    if (nnode <= 0 || nodestatus == nullptr || nodextr == nullptr) {
        return;
    }
    
    // =========================================================================
    // COMPUTE WORKSPACE ARRAYS INTERNALLY (matches FP16 implementation)
    // =========================================================================
    std::vector<integer_t> nod_local(nnode);
    std::vector<integer_t> ncount_local;
    std::vector<integer_t> ncn_local(nsample);
    std::vector<integer_t> nodexb_local(nsample);
    std::vector<integer_t> ndbegin_local;
    std::vector<integer_t> npcase_local(nsample);
    
    // Compute nod (terminal node mapping) and nterm
    integer_t nterm = 0;
    for (integer_t k = 0; k < nnode; ++k) {
        if (nodestatus[k] == -1) {
            nod_local[k] = nterm;
            nterm++;
        } else {
            nod_local[k] = -1;
        }
    }
    
    if (nterm == 0) {
        return;  // No terminal nodes
    }
    
    ncount_local.resize(nterm, 0);
    
    // Check for any in-bag samples
    integer_t inbag_count = 0;
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin[n] > 0) inbag_count++;
    }
    if (inbag_count == 0) {
        return;  // No in-bag samples
    }
    
    // Compute nodexb and ncount
    for (integer_t n = 0; n < nsample; ++n) {
        if (nodextr[n] < 0 || nodextr[n] >= nnode) {
            nodexb_local[n] = -1;
            ncn_local[n] = 0;
            continue;
        }
        
        integer_t k = nod_local[nodextr[n]];
        if (k >= 0 && k < nterm) {
            if (nin[n] > 0) {
                ncount_local[k]++;
                ncn_local[n] = ncount_local[k];
            } else {
                ncn_local[n] = 0;
            }
            nodexb_local[n] = k;
        } else {
            nodexb_local[n] = -1;
            ncn_local[n] = 0;
        }
    }
    
    // Compute ndbegin
    ndbegin_local.resize(nterm + 1, 0);
    ndbegin_local[0] = 0;
    for (integer_t k = 1; k <= nterm; ++k) {
        integer_t count = (k-1 < nterm) ? ncount_local[k-1] : 0;
        if (count < 0) count = 0;
        ndbegin_local[k] = ndbegin_local[k-1] + count;
    }
    
    // Compute npcase
    for (integer_t n = 0; n < nsample; ++n) {
        integer_t k = nodexb_local[n];
        if (k >= 0 && k < nterm) {
            integer_t idx = ndbegin_local[k] + ncn_local[n] - 1;
            if (idx >= 0 && idx < nsample) {
                npcase_local[idx] = n;
            }
        }
    }
    
    // =========================================================================
    // GPU COMPUTATION
    // =========================================================================
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    
    // Allocate GPU memory
    integer_t* nodexb_d = nullptr;
    integer_t* nin_d = nullptr;
    integer_t* ndbegin_d = nullptr;
    integer_t* npcase_d = nullptr;
    int8_t* prox_upper_int8_d = nullptr;
    
    CUDA_CHECK_VOID(cudaMalloc(&prox_upper_int8_d, upper_triangle_size * sizeof(int8_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nodexb_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nin_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&ndbegin_d, (nterm + 1) * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&npcase_d, nsample * sizeof(integer_t)));
    
    // Initialize proximity matrix to zero
    CUDA_CHECK_VOID(cudaMemset(prox_upper_int8_d, 0, upper_triangle_size * sizeof(int8_t)));
    
    // Copy computed workspace arrays to GPU
    CUDA_CHECK_VOID(cudaMemcpy(nodexb_d, nodexb_local.data(), nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nin_d, nin, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ndbegin_d, ndbegin_local.data(), (nterm + 1) * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(npcase_d, npcase_local.data(), nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    cuda_proximity_upper_triangle_int8_kernel<<<grid_size, block_size>>>(
        nodexb_d, nin_d, nsample, nterm, ndbegin_d, npcase_d, prox_upper_int8_d, scale, zero_point
    );
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));
    
    // Check if output pointer is on GPU or CPU
    cudaPointerAttributes out_attrs;
    cudaError_t out_check = cudaPointerGetAttributes(&out_attrs, prox_upper_int8);
    bool out_is_gpu = (out_check == cudaSuccess && out_attrs.type == cudaMemoryTypeDevice);
    
    if (out_is_gpu) {
        CUDA_CHECK_VOID(cudaMemcpy(prox_upper_int8, prox_upper_int8_d, upper_triangle_size * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    } else {
        CUDA_CHECK_VOID(cudaMemcpy(prox_upper_int8, prox_upper_int8_d, upper_triangle_size * sizeof(int8_t), cudaMemcpyDeviceToHost));
    }
    
    // Cleanup
    if (prox_upper_int8_d) { cudaError_t err = cudaFree(prox_upper_int8_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
}

void gpu_proximity_upper_triangle_nf4(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    uint8_t* prox_upper_nf4,  // Output: Packed upper triangle NF4
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
) {
    // std::cout << "[DEBUG PROXIMITY NF4] Entry: nsample=" << nsample 
    //           << ", nnode=" << nnode << std::endl;
    // Check GPU availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        // std::cout << "[DEBUG PROXIMITY NF4] No GPU available, returning" << std::endl;
        return;
    }
    
    // Compute workspace arrays first (same as FP16 version)
    // These arrays are NOT pre-computed - we need to compute them here
    std::vector<integer_t> nod_local(nnode);
    std::vector<integer_t> ncount_local(nsample);
    std::vector<integer_t> ncn_local(nsample);
    std::vector<integer_t> nodexb_local(nsample);
    std::vector<integer_t> ndbegin_local(nsample + 1);
    std::vector<integer_t> npcase_local(nsample);
    
    // Copy nodestatus to local array
    std::copy(nodestatus, nodestatus + nnode, nod_local.data());
    
    // Compute nterm (number of terminal nodes)
    integer_t nterm = 0;
    for (integer_t k = 0; k < nnode; ++k) {
        if (nodestatus[k] == -1) {
            nterm++;
        }
    }
    
    // Compute nodexb (terminal node index for each sample)
    for (integer_t n = 0; n < nsample; ++n) {
        integer_t node_idx = nodextr[n];
        if (node_idx >= 0 && node_idx < nnode && nodestatus[node_idx] == -1) {
            // Find which terminal node this is (0-based index among terminal nodes)
            integer_t term_idx = 0;
            for (integer_t k = 0; k < node_idx; ++k) {
                if (nodestatus[k] == -1) {
                    term_idx++;
                }
            }
            nodexb_local[n] = term_idx;
        } else {
            nodexb_local[n] = -1;
        }
    }
    
    // Compute ncount (number of in-bag samples in each terminal node)
    std::fill(ncount_local.begin(), ncount_local.end(), 0);
    for (integer_t n = 0; n < nsample; ++n) {
        integer_t k = nodexb_local[n];
        if (k >= 0 && k < nterm && nin[n] > 0) {
            ncount_local[k]++;
        }
    }
    
    // Compute ncn (position of sample within its terminal node, 1-based)
    std::fill(ncn_local.begin(), ncn_local.end(), 0);
    std::vector<integer_t> node_counters(nterm, 0);
    for (integer_t n = 0; n < nsample; ++n) {
        integer_t k = nodexb_local[n];
        if (k >= 0 && k < nterm && nin[n] > 0) {
            node_counters[k]++;
            ncn_local[n] = node_counters[k];
        }
    }
    
    // Compute ndbegin (starting index in npcase for each terminal node)
    ndbegin_local.resize(nterm + 1, 0);
    if (nterm > 0) {
        ndbegin_local[0] = 0;
        for (integer_t k = 1; k <= nterm; ++k) {
            integer_t count = (k-1 < nterm) ? ncount_local[k-1] : 0;
            if (count < 0) count = 0;
            ndbegin_local[k] = ndbegin_local[k-1] + count;
        }
    }
    
    // Compute npcase (maps position to case index)
    std::fill(npcase_local.begin(), npcase_local.end(), -1);
    for (integer_t n = 0; n < nsample; ++n) {
        integer_t k = nodexb_local[n];
        if (k >= 0 && k < nterm) {
            integer_t idx = ndbegin_local[k] + ncn_local[n] - 1;
            if (idx >= 0 && idx < nsample) {
                npcase_local[idx] = n;
            }
        }
    }
    
    // Compute upper triangle size
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    
    // Check if output pointer is on GPU or CPU
    cudaPointerAttributes out_attrs;
    cudaError_t out_check = cudaPointerGetAttributes(&out_attrs, prox_upper_nf4);
    bool out_is_gpu = (out_check == cudaSuccess && out_attrs.type == cudaMemoryTypeDevice);
    
    // Allocate GPU memory
    integer_t* nodexb_d, *nin_d, *ndbegin_d, *npcase_d;
    uint8_t* prox_upper_nf4_d;
    
    // NF4: 4 bits per element, so we need (upper_triangle_size + 1) / 2 bytes
    size_t nf4_buffer_size = (upper_triangle_size + 1) / 2;
    
    if (out_is_gpu) {
        // Output is already on GPU - use it directly, no temp buffer needed!
        prox_upper_nf4_d = prox_upper_nf4;
    } else {
        // Output is on CPU - need temp GPU buffer
        CUDA_CHECK_VOID(cudaMalloc(&prox_upper_nf4_d, nf4_buffer_size * sizeof(uint8_t)));
    }
    
    CUDA_CHECK_VOID(cudaMalloc(&nodexb_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&nin_d, nsample * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&ndbegin_d, (nterm + 1) * sizeof(integer_t)));
    CUDA_CHECK_VOID(cudaMalloc(&npcase_d, nsample * sizeof(integer_t)));
    
    // Initialize proximity matrix to zero
    CUDA_CHECK_VOID(cudaMemset(prox_upper_nf4_d, 0, nf4_buffer_size * sizeof(uint8_t)));
    
    // Copy computed workspace arrays to GPU
    CUDA_CHECK_VOID(cudaMemcpy(nodexb_d, nodexb_local.data(), nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nin_d, nin, nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(ndbegin_d, ndbegin_local.data(), (nterm + 1) * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(npcase_d, npcase_local.data(), nsample * sizeof(integer_t), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    // Ensure grid_size is at least 1 (CUDA requires at least 1 block)
    if (grid_size.x == 0) {
        grid_size.x = 1;
    }
    
    // Get use_casewise from global config
    bool use_casewise_val = g_config.use_casewise;
    
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout << "[DEBUG PROXIMITY NF4] Launching kernel: grid_size=" << grid_size.x 
    //           << ", block_size=" << block_size.x << ", nterm=" << nterm 
    //           << ", use_casewise=" << use_casewise_val << std::endl;
    // std::cout.flush();
    
    cuda_proximity_upper_triangle_nf4_kernel<<<grid_size, block_size>>>(
        nodexb_d, nin_d, nsample, nterm, ndbegin_d, npcase_d, prox_upper_nf4_d, use_casewise_val
    );
    
    // Check kernel launch immediately
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        // std::cerr << "[ERROR] NF4 kernel launch failed: " << cudaGetErrorString(launch_err) << std::endl;
        // std::cerr.flush();
        cudaGetLastError(); // Clear error state
        return;  // Early return on launch failure
    }
    
    // Synchronize to ensure kernel completes and printf output appears
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        // std::cerr << "[ERROR] NF4 kernel sync failed (kernel may have crashed): " 
                //   << cudaGetErrorString(sync_err) << std::endl;
        // std::cerr.flush();
        // Don't return - continue to check output
    }
    
    // Debug prints removed to avoid potential stream corruption issues
    // std::cout.flush();
    // fflush(stdout);
    // std::cout << "[DEBUG PROXIMITY NF4] Kernel completed, checking output..." << std::endl;
    // std::cout.flush();
    
    if (!out_is_gpu) {
        // Output is on CPU - copy from GPU to host
        CUDA_CHECK_VOID(cudaMemcpy(prox_upper_nf4, prox_upper_nf4_d, nf4_buffer_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        // Free temp GPU buffer (only if we allocated it)
        if (prox_upper_nf4_d) { cudaError_t err = cudaFree(prox_upper_nf4_d); if (err != cudaSuccess) cudaGetLastError(); }
    }
    // else: output is already on GPU, no copy needed, and prox_upper_nf4_d points to prox_upper_nf4 so don't free it
    
    // Cleanup other GPU buffers
    if (nodexb_d) { cudaError_t err = cudaFree(nodexb_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nin_d) { cudaError_t err = cudaFree(nin_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (ndbegin_d) { cudaError_t err = cudaFree(ndbegin_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (npcase_d) { cudaError_t err = cudaFree(npcase_d); if (err != cudaSuccess) cudaGetLastError(); }
}

// Unified function: Choose quantization level automatically
void gpu_proximity_upper_triangle(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    QuantizationLevel quant_level,
    void* prox_upper_output,  // Output: Packed upper triangle (type depends on quant_level)
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
) {
    // std::cout << "[DEBUG PROXIMITY WRAPPER] nsample=" << nsample << ", nnode=" << nnode 
    //           << ", quant_level=" << static_cast<int>(quant_level) << std::endl;
    switch (quant_level) {
        case QuantizationLevel::FP32: {
            // For FP32, just use FP16 directly - proximity values are in [0,1] range
            // so FP16 precision is sufficient and avoids unnecessary conversions
            __half* prox_fp16 = static_cast<__half*>(prox_upper_output);
            // std::cout << "[DEBUG PROXIMITY WRAPPER] Calling FP16 with FP32" << std::endl;
            gpu_proximity_upper_triangle_fp16(nodestatus, nodextr, nin, nsample, nnode,
                                            prox_fp16, nod, ncount, ncn, nodexb, ndbegin, npcase);
            break;
        }
        case QuantizationLevel::FP16: {
            __half* prox_fp16 = static_cast<__half*>(prox_upper_output);
            // std::cout << "[DEBUG PROXIMITY WRAPPER] Calling FP16 with FP16" << std::endl;
            gpu_proximity_upper_triangle_fp16(nodestatus, nodextr, nin, nsample, nnode,
                                            prox_fp16, nod, ncount, ncn, nodexb, ndbegin, npcase);
            break;
        }
        case QuantizationLevel::INT8: {
            int8_t* prox_int8 = static_cast<int8_t*>(prox_upper_output);
            // Compute scaling parameters (simplified - in practice compute from data)
            float scale = 1.0f / 127.0f;
            float zero_point = 0.0f;
            gpu_proximity_upper_triangle_int8(nodestatus, nodextr, nin, nsample, nnode,
                                            prox_int8, scale, zero_point,
                                            nod, ncount, ncn, nodexb, ndbegin, npcase);
            break;
        }
        case QuantizationLevel::NF4: {
            uint8_t* prox_nf4 = static_cast<uint8_t*>(prox_upper_output);
            gpu_proximity_upper_triangle_nf4(nodestatus, nodextr, nin, nsample, nnode,
                                            prox_nf4, nod, ncount, ncn, nodexb, ndbegin, npcase);
            break;
        }
        default:
            // Silently return - unsupported quantization level (don't throw exceptions in Jupyter)
            return;
    }
}

// ============================================================================
// GPU RF-GAP PROXIMITY: Upper Triangle Packed Format (for incremental low-rank)
// ============================================================================

// GPU kernel for RF-GAP proximity in upper triangle format
// Each thread processes one OOB sample i
__global__ void cuda_proximity_rfgap_upper_triangle_fp16_kernel(
    const integer_t* nin_tree,      // Bootstrap multiplicities [sample] (0=OOB, >0=in-bag)
    const integer_t* nodextr_tree,  // Terminal node assignments [sample]
    integer_t nsample,
    __half* prox_upper_fp16  // Packed upper triangle output
) {
    // Each thread processes one OOB sample i
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsample) return;
    
    // Skip if sample i is not OOB
    if (nin_tree[i] != 0) return;
    
    // Get terminal node for OOB sample i
    integer_t terminal_node_i = nodextr_tree[i];
    if (terminal_node_i < 0) return;  // Invalid terminal node
    
    // First pass: compute |Mi(t)| = sum of multiplicities of in-bag samples in this node
    integer_t total_inbag_multiplicity = 0;
    for (integer_t j = 0; j < nsample; ++j) {
        integer_t cj_t = nin_tree[j];
        if (cj_t > 0 && nodextr_tree[j] == terminal_node_i) {
            // j is in-bag and in the same terminal node as i
            total_inbag_multiplicity += cj_t;
        }
    }
    
    if (total_inbag_multiplicity == 0) return;  // No in-bag samples in this terminal node
    
    // Second pass: accumulate contributions to upper triangle
    // Only store upper triangle (i <= j) for memory efficiency
    float inv_total = 1.0f / static_cast<float>(total_inbag_multiplicity);
    for (integer_t j = 0; j < nsample; ++j) {
        integer_t cj_t = nin_tree[j];
        if (cj_t > 0 && nodextr_tree[j] == terminal_node_i && i <= j) {
            // j is in-bag, in the same terminal node as i, and in upper triangle
            // Add cj(t) / |Mi(t)| to pGAP(i, j)
            float contribution = static_cast<float>(cj_t) * inv_total;
            integer_t upper_idx = upper_triangle_index(i, j, nsample);
            integer_t max_upper_idx = (nsample * (nsample + 1)) / 2 - 1;
            if (upper_idx >= 0 && upper_idx <= max_upper_idx) {
                // Use atomic add for thread safety (though each (i,j) should be unique)
                float current_val = __half2float(prox_upper_fp16[upper_idx]);
                float new_val = current_val + contribution;
                prox_upper_fp16[upper_idx] = __float2half(new_val);
            }
        }
    }
}

// Host function for GPU RF-GAP upper triangle computation
void gpu_proximity_rfgap_upper_triangle_fp16(
    const integer_t* nin_tree,      // Bootstrap multiplicities for this tree [sample] (0=OOB, >0=in-bag)
    const integer_t* nodextr_tree,  // Terminal node assignments for this tree [sample]
    integer_t nsample,
    __half* prox_upper_fp16  // Output: Packed upper triangle (n(n+1)/2 elements) in FP16
) {
    // Check CUDA availability
    if (!cuda_is_available()) {
        // Silently return - CUDA not available (don't throw exceptions in Jupyter)
        return;
    }
    
    // Allocate device memory
    integer_t* nin_tree_d;
    integer_t* nodextr_tree_d;
    __half* prox_upper_d;
    
    size_t sample_data_size = static_cast<size_t>(nsample) * sizeof(integer_t);
    size_t upper_triangle_size = (static_cast<size_t>(nsample) * (nsample + 1)) / 2;
    size_t prox_upper_size = upper_triangle_size * sizeof(__half);
    
    CUDA_CHECK_VOID(cudaMalloc(&nin_tree_d, sample_data_size));
    CUDA_CHECK_VOID(cudaMalloc(&nodextr_tree_d, sample_data_size));
    CUDA_CHECK_VOID(cudaMalloc(&prox_upper_d, prox_upper_size));
    
    // Copy data to device
    CUDA_CHECK_VOID(cudaMemcpy(nin_tree_d, nin_tree, sample_data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(nodextr_tree_d, nodextr_tree, sample_data_size, cudaMemcpyHostToDevice));
    
    // Initialize upper triangle to zero on device
    CUDA_CHECK_VOID(cudaMemset(prox_upper_d, 0, prox_upper_size));
    
    // Launch kernel with optimal block size
    dim3 block_size(256);
    dim3 grid_size((nsample + block_size.x - 1) / block_size.x);
    
    cuda_proximity_rfgap_upper_triangle_fp16_kernel<<<grid_size, block_size>>>(
        nin_tree_d, nodextr_tree_d, nsample, prox_upper_d
    );
    
    CUDA_CHECK_VOID(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
    
    // Copy results back to host
    CUDA_CHECK_VOID(cudaMemcpy(prox_upper_fp16, prox_upper_d, prox_upper_size, cudaMemcpyDeviceToHost));
    
    // Cleanup
    // Safe free to prevent segfaults
    if (nin_tree_d) { cudaError_t err = cudaFree(nin_tree_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (nodextr_tree_d) { cudaError_t err = cudaFree(nodextr_tree_d); if (err != cudaSuccess) cudaGetLastError(); }
    if (prox_upper_d) { cudaError_t err = cudaFree(prox_upper_d); if (err != cudaSuccess) cudaGetLastError(); }
}

} // namespace cuda
} // namespace rf


