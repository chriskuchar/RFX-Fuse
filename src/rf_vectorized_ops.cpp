#include "rf_vectorized_ops.hpp"
#include <immintrin.h>
#include <cpuid.h>
#include <cstring>
#include <random>
#include <cmath>
#include <algorithm>

namespace rf {

// ============================================================================
// CPU Feature Detection
// ============================================================================

bool has_avx_support() {
    int cpu_info[4];
    __cpuid(1, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    return (cpu_info[2] & (1 << 28)) != 0;
}

bool has_avx2_support() {
    int cpu_info[4];
    __cpuid(7, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    return (cpu_info[1] & (1 << 5)) != 0;
}

bool has_avx512_support() {
    int cpu_info[4];
    __cpuid(7, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    return (cpu_info[1] & (1 << 16)) != 0;
}

integer_t get_optimal_simd_width() {
    if (has_avx512_support()) return 16; // AVX-512: 16 floats
    if (has_avx2_support()) return 8;   // AVX2: 8 floats
    if (has_avx_support()) return 8;     // AVX: 8 floats
    return 4; // SSE: 4 floats
}

// ============================================================================
// VectorizedGiniCalculator Implementation
// ============================================================================

std::pair<real_t, real_t> VectorizedGiniCalculator::find_best_split(
    const real_t* values,
    const integer_t* labels,
    const integer_t* indices,
    integer_t n_samples,
    integer_t n_classes,
    integer_t min_samples_split
) {
    if (n_samples < min_samples_split * 2) {
        return {0.0f, 0.0f}; // No valid split possible
    }

    real_t best_split_value = 0.0f;
    real_t best_impurity_reduction = 0.0f;

    // Use FastArray for temporary storage
    FastIntArray class_counts_left(n_classes);
    FastIntArray class_counts_right(n_classes);
    FastIntArray total_class_counts(n_classes);

    // Initialize total class counts
    std::fill(total_class_counts.data(), total_class_counts.data() + n_classes, 0);
    for (integer_t i = 0; i < n_samples; ++i) {
        total_class_counts[indices[i]]++;
    }

    // Initialize right node counts (all samples start in right node)
    std::copy(total_class_counts.data(), 
              total_class_counts.data() + n_classes, 
              class_counts_right.data());

    // Try each possible split point
    for (integer_t split_idx = min_samples_split; 
         split_idx < n_samples - min_samples_split; 
         ++split_idx) {
        
        // Move sample from right to left node
        integer_t sample_idx = indices[split_idx];
        class_counts_left[sample_idx]++;
        class_counts_right[sample_idx]--;

        // Calculate impurity for this split
        real_t left_impurity = calculate_impurity_vectorized(
            class_counts_left.data(), split_idx + 1, n_classes);
        real_t right_impurity = calculate_impurity_vectorized(
            class_counts_right.data(), n_samples - split_idx - 1, n_classes);

        // Calculate weighted impurity
        real_t weighted_impurity = 
            (split_idx + 1) * left_impurity + 
            (n_samples - split_idx - 1) * right_impurity;
        weighted_impurity /= n_samples;

        // Calculate impurity reduction
        real_t total_impurity = calculate_impurity_vectorized(
            total_class_counts.data(), n_samples, n_classes);
        real_t impurity_reduction = total_impurity - weighted_impurity;

        // Update best split if this is better
        if (impurity_reduction > best_impurity_reduction) {
            best_impurity_reduction = impurity_reduction;
            best_split_value = values[split_idx];
        }
    }

    return {best_split_value, best_impurity_reduction};
}

real_t VectorizedGiniCalculator::calculate_gini(
    const integer_t* class_counts_left,
    const integer_t* class_counts_right,
    integer_t n_left,
    integer_t n_right,
    integer_t n_classes
) {
    real_t left_gini = calculate_impurity_vectorized(class_counts_left, n_left, n_classes);
    real_t right_gini = calculate_impurity_vectorized(class_counts_right, n_right, n_classes);
    
    real_t weighted_gini = (n_left * left_gini + n_right * right_gini) / (n_left + n_right);
    return weighted_gini;
}

void VectorizedGiniCalculator::accumulate_class_counts_avx(
    const integer_t* labels,
    const integer_t* indices,
    integer_t start_idx,
    integer_t end_idx,
    integer_t* class_counts,
    integer_t n_classes
) {
    // Use AVX for vectorized accumulation when possible
    if (has_avx_support() && n_classes >= 8) {
        // Vectorized accumulation for large class counts
        __m256i counts_vec = _mm256_setzero_si256();
        
        for (integer_t i = start_idx; i < end_idx; i += 8) {
            if (i + 8 <= end_idx) {
                // Load 8 indices
                __m256i idx_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&indices[i]));
                
                // For each index, increment corresponding class count
                // This is a simplified version - full implementation would be more complex
                for (integer_t j = 0; j < 8; ++j) {
                    integer_t idx = indices[i + j];
                    if (idx < n_classes) {
                        class_counts[idx]++;
                    }
                }
            } else {
                // Handle remaining elements
                for (integer_t j = i; j < end_idx; ++j) {
                    integer_t idx = indices[j];
                    if (idx < n_classes) {
                        class_counts[idx]++;
                    }
                }
                break;
            }
        }
    } else {
        // Fallback to scalar accumulation
        for (integer_t i = start_idx; i < end_idx; ++i) {
            integer_t idx = indices[i];
            if (idx < n_classes) {
                class_counts[idx]++;
            }
        }
    }
}

real_t VectorizedGiniCalculator::calculate_impurity_vectorized(
    const integer_t* class_counts,
    integer_t total_samples,
    integer_t n_classes
) {
    if (total_samples == 0) return 0.0f;

    real_t gini = 1.0f;
    
    // Use vectorized operations for large class counts
    if (has_avx_support() && n_classes >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (integer_t i = 0; i < n_classes; i += 8) {
            if (i + 8 <= n_classes) {
                // Load 8 class counts
                __m256 counts_vec = _mm256_cvtepi32_ps(
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(&class_counts[i])));
                
                // Calculate squared proportions
                __m256 proportions = _mm256_div_ps(counts_vec, _mm256_set1_ps(total_samples));
                __m256 squared_proportions = _mm256_mul_ps(proportions, proportions);
                
                // Accumulate
                sum_vec = _mm256_add_ps(sum_vec, squared_proportions);
            } else {
                // Handle remaining classes
                for (integer_t j = i; j < n_classes; ++j) {
                    real_t proportion = static_cast<real_t>(class_counts[j]) / total_samples;
                    gini -= proportion * proportion;
                }
                break;
            }
        }
        
        // Sum the vector elements
        real_t sum_array[8];
        _mm256_store_ps(sum_array, sum_vec);
        for (integer_t i = 0; i < 8; ++i) {
            gini -= sum_array[i];
        }
    } else {
        // Fallback to scalar calculation
        for (integer_t i = 0; i < n_classes; ++i) {
            real_t proportion = static_cast<real_t>(class_counts[i]) / total_samples;
            gini -= proportion * proportion;
        }
    }

    return gini;
}

// ============================================================================
// VectorizedMSECalculator Implementation
// ============================================================================

std::pair<real_t, real_t> VectorizedMSECalculator::find_best_split(
    const real_t* values,
    const real_t* targets,
    const integer_t* indices,
    integer_t n_samples,
    integer_t min_samples_split
) {
    if (n_samples < min_samples_split * 2) {
        return {0.0f, 0.0f}; // No valid split possible
    }

    real_t best_split_value = 0.0f;
    real_t best_mse_reduction = 0.0f;

    // Calculate total variance (baseline)
    real_t total_mean = calculate_mean_vectorized(targets, n_samples);
    real_t total_variance = calculate_variance_vectorized(targets, total_mean, n_samples);

    // Use FastArray for temporary storage
    FastRealArray left_targets(n_samples);
    FastRealArray right_targets(n_samples);

    // Try each possible split point
    for (integer_t split_idx = min_samples_split; 
         split_idx < n_samples - min_samples_split; 
         ++split_idx) {
        
        // Split targets into left and right
        integer_t n_left = split_idx + 1;
        integer_t n_right = n_samples - split_idx - 1;
        
        for (integer_t i = 0; i < n_left; ++i) {
            left_targets[i] = targets[indices[i]];
        }
        for (integer_t i = 0; i < n_right; ++i) {
            right_targets[i] = targets[indices[split_idx + 1 + i]];
        }

        // Calculate MSE for this split
        real_t left_mean = calculate_mean_vectorized(left_targets.data(), n_left);
        real_t right_mean = calculate_mean_vectorized(right_targets.data(), n_right);
        
        real_t left_variance = calculate_variance_vectorized(left_targets.data(), left_mean, n_left);
        real_t right_variance = calculate_variance_vectorized(right_targets.data(), right_mean, n_right);
        
        real_t weighted_variance = (n_left * left_variance + n_right * right_variance) / n_samples;
        real_t mse_reduction = total_variance - weighted_variance;

        // Update best split if this is better
        if (mse_reduction > best_mse_reduction) {
            best_mse_reduction = mse_reduction;
            best_split_value = values[split_idx];
        }
    }

    return {best_split_value, best_mse_reduction};
}

real_t VectorizedMSECalculator::calculate_mse(
    const real_t* targets_left,
    const real_t* targets_right,
    integer_t n_left,
    integer_t n_right
) {
    real_t left_mean = calculate_mean_vectorized(targets_left, n_left);
    real_t right_mean = calculate_mean_vectorized(targets_right, n_right);
    
    real_t left_variance = calculate_variance_vectorized(targets_left, left_mean, n_left);
    real_t right_variance = calculate_variance_vectorized(targets_right, right_mean, n_right);
    
    return (n_left * left_variance + n_right * right_variance) / (n_left + n_right);
}

real_t VectorizedMSECalculator::calculate_mean_vectorized(
    const real_t* values,
    integer_t n_samples
) {
    if (n_samples == 0) return 0.0f;

    real_t sum = 0.0f;
    
    // Use vectorized sum for large arrays
    if (has_avx_support() && n_samples >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (integer_t i = 0; i < n_samples; i += 8) {
            if (i + 8 <= n_samples) {
                __m256 values_vec = _mm256_load_ps(&values[i]);
                sum_vec = _mm256_add_ps(sum_vec, values_vec);
            } else {
                // Handle remaining elements
                for (integer_t j = i; j < n_samples; ++j) {
                    sum += values[j];
                }
                break;
            }
        }
        
        // Sum the vector elements
        real_t sum_array[8];
        _mm256_store_ps(sum_array, sum_vec);
        for (integer_t i = 0; i < 8; ++i) {
            sum += sum_array[i];
        }
    } else {
        // Fallback to scalar sum
        for (integer_t i = 0; i < n_samples; ++i) {
            sum += values[i];
        }
    }

    return sum / n_samples;
}

real_t VectorizedMSECalculator::calculate_variance_vectorized(
    const real_t* values,
    real_t mean,
    integer_t n_samples
) {
    if (n_samples == 0) return 0.0f;

    real_t sum_squared_diff = 0.0f;
    
    // Use vectorized operations for large arrays
    if (has_avx_support() && n_samples >= 8) {
        __m256 mean_vec = _mm256_set1_ps(mean);
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (integer_t i = 0; i < n_samples; i += 8) {
            if (i + 8 <= n_samples) {
                __m256 values_vec = _mm256_load_ps(&values[i]);
                __m256 diff_vec = _mm256_sub_ps(values_vec, mean_vec);
                __m256 squared_diff_vec = _mm256_mul_ps(diff_vec, diff_vec);
                sum_vec = _mm256_add_ps(sum_vec, squared_diff_vec);
            } else {
                // Handle remaining elements
                for (integer_t j = i; j < n_samples; ++j) {
                    real_t diff = values[j] - mean;
                    sum_squared_diff += diff * diff;
                }
                break;
            }
        }
        
        // Sum the vector elements
        real_t sum_array[8];
        _mm256_store_ps(sum_array, sum_vec);
        for (integer_t i = 0; i < 8; ++i) {
            sum_squared_diff += sum_array[i];
        }
    } else {
        // Fallback to scalar calculation
        for (integer_t i = 0; i < n_samples; ++i) {
            real_t diff = values[i] - mean;
            sum_squared_diff += diff * diff;
        }
    }

    return sum_squared_diff / n_samples;
}

// ============================================================================
// VectorizedDataMovement Implementation
// ============================================================================

void VectorizedDataMovement::copy_aligned(
    void* dest,
    const void* src,
    size_t size_bytes
) {
    if (has_avx_support() && size_bytes >= 32) {
        // Use AVX for large aligned copies
        size_t avx_size = size_bytes & ~31; // Round down to 32-byte boundary
        
        for (size_t i = 0; i < avx_size; i += 32) {
            __m256 data = _mm256_load_ps(reinterpret_cast<const float*>(
                static_cast<const char*>(src) + i));
            _mm256_store_ps(reinterpret_cast<float*>(
                static_cast<char*>(dest) + i), data);
        }
        
        // Handle remaining bytes
        for (size_t i = avx_size; i < size_bytes; ++i) {
            static_cast<char*>(dest)[i] = static_cast<const char*>(src)[i];
        }
    } else {
        // Fallback to memcpy
        std::memcpy(dest, src, size_bytes);
    }
}

void VectorizedDataMovement::initialize_vectorized(
    real_t* array,
    real_t value,
    integer_t n_elements
) {
    if (has_avx_support() && n_elements >= 8) {
        __m256 value_vec = _mm256_set1_ps(value);
        
        for (integer_t i = 0; i < n_elements; i += 8) {
            if (i + 8 <= n_elements) {
                _mm256_store_ps(&array[i], value_vec);
            } else {
                // Handle remaining elements
                for (integer_t j = i; j < n_elements; ++j) {
                    array[j] = value;
                }
                break;
            }
        }
    } else {
        // Fallback to scalar initialization
        std::fill(array, array + n_elements, value);
    }
}

real_t VectorizedDataMovement::sum_vectorized(
    const real_t* array,
    integer_t n_elements
) {
    if (n_elements == 0) return 0.0f;
    
    // Use a simple loop for now - the private method access needs to be fixed
    real_t sum = 0.0f;
    for (integer_t i = 0; i < n_elements; ++i) {
        sum += array[i];
    }
    return sum;
}

real_t VectorizedDataMovement::dot_product_vectorized(
    const real_t* a,
    const real_t* b,
    integer_t n_elements
) {
    real_t sum = 0.0f;
    
    if (has_avx_support() && n_elements >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (integer_t i = 0; i < n_elements; i += 8) {
            if (i + 8 <= n_elements) {
                __m256 a_vec = _mm256_load_ps(&a[i]);
                __m256 b_vec = _mm256_load_ps(&b[i]);
                __m256 product_vec = _mm256_mul_ps(a_vec, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, product_vec);
            } else {
                // Handle remaining elements
                for (integer_t j = i; j < n_elements; ++j) {
                    sum += a[j] * b[j];
                }
                break;
            }
        }
        
        // Sum the vector elements
        real_t sum_array[8];
        _mm256_store_ps(sum_array, sum_vec);
        for (integer_t i = 0; i < 8; ++i) {
            sum += sum_array[i];
        }
    } else {
        // Fallback to scalar calculation
        for (integer_t i = 0; i < n_elements; ++i) {
            sum += a[i] * b[i];
        }
    }

    return sum;
}

real_t VectorizedDataMovement::max_vectorized(
    const real_t* array,
    integer_t n_elements
) {
    if (n_elements == 0) return 0.0f;
    
    real_t max_val = array[0];
    
    if (has_avx_support() && n_elements >= 8) {
        __m256 max_vec = _mm256_set1_ps(max_val);
        
        for (integer_t i = 0; i < n_elements; i += 8) {
            if (i + 8 <= n_elements) {
                __m256 values_vec = _mm256_load_ps(&array[i]);
                max_vec = _mm256_max_ps(max_vec, values_vec);
            } else {
                // Handle remaining elements
                for (integer_t j = i; j < n_elements; ++j) {
                    max_val = std::max(max_val, array[j]);
                }
                break;
            }
        }
        
        // Find maximum in vector
        real_t max_array[8];
        _mm256_store_ps(max_array, max_vec);
        for (integer_t i = 0; i < 8; ++i) {
            max_val = std::max(max_val, max_array[i]);
        }
    } else {
        // Fallback to scalar calculation
        for (integer_t i = 1; i < n_elements; ++i) {
            max_val = std::max(max_val, array[i]);
        }
    }

    return max_val;
}

real_t VectorizedDataMovement::min_vectorized(
    const real_t* array,
    integer_t n_elements
) {
    if (n_elements == 0) return 0.0f;
    
    real_t min_val = array[0];
    
    if (has_avx_support() && n_elements >= 8) {
        __m256 min_vec = _mm256_set1_ps(min_val);
        
        for (integer_t i = 0; i < n_elements; i += 8) {
            if (i + 8 <= n_elements) {
                __m256 values_vec = _mm256_load_ps(&array[i]);
                min_vec = _mm256_min_ps(min_vec, values_vec);
            } else {
                // Handle remaining elements
                for (integer_t j = i; j < n_elements; ++j) {
                    min_val = std::min(min_val, array[j]);
                }
                break;
            }
        }
        
        // Find minimum in vector
        real_t min_array[8];
        _mm256_store_ps(min_array, min_vec);
        for (integer_t i = 0; i < 8; ++i) {
            min_val = std::min(min_val, min_array[i]);
        }
    } else {
        // Fallback to scalar calculation
        for (integer_t i = 1; i < n_elements; ++i) {
            min_val = std::min(min_val, array[i]);
        }
    }

    return min_val;
}

} // namespace rf
