#ifndef RF_ADVANCED_OPTIMIZATIONS_HPP
#define RF_ADVANCED_OPTIMIZATIONS_HPP

#include "rf_types.hpp"
#include "rf_config.hpp"
#include "rf_memory_pool.hpp"
#include "rf_vectorized_ops.hpp"
#include <immintrin.h>
#include <memory>
#include <array>
#include <bitset>
#include <type_traits>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <stdexcept>

namespace rf {

// ============================================================================
// Cache-Optimized Data Structures
// ============================================================================

// Cache-friendly tree node structure (64 bytes = 1 cache line)
struct alignas(64) CacheOptimizedTreeNode {
    // Hot data (frequently accessed) - 32 bytes
    integer_t nodestatus;      // 4 bytes
    integer_t bestvar;         // 4 bytes
    integer_t left_child;      // 4 bytes
    integer_t right_child;     // 4 bytes
    real_t xbestsplit;         // 4 bytes
    real_t nodewt;             // 4 bytes
    integer_t nodestart;       // 4 bytes
    integer_t nodestop;        // 4 bytes
    
    // Cold data (less frequently accessed) - 32 bytes
    integer_t nodeclass;       // 4 bytes
    integer_t depth;           // 4 bytes
    real_t impurity;           // 4 bytes
    integer_t n_samples;       // 4 bytes
    integer_t n_classes;       // 4 bytes
    real_t split_gain;         // 4 bytes
    integer_t parent;          // 4 bytes
    integer_t sibling;         // 4 bytes
    
    // Padding to ensure 64-byte alignment
    char padding[0];
};

// Cache-friendly feature matrix with prefetching
template<typename T>
class CacheOptimizedMatrix {
public:
    CacheOptimizedMatrix(integer_t rows, integer_t cols) 
        : rows_(rows), cols_(cols), data_(nullptr) {
        // Align to cache line boundary
        int result = posix_memalign(reinterpret_cast<void**>(&data_), 64, 
                                   rows * cols * sizeof(T));
        if (result != 0) {
            throw std::runtime_error("Failed to allocate aligned memory");
        }
    }
    
    ~CacheOptimizedMatrix() {
        if (data_) free(data_);
    }
    
    // Prefetch next cache line
    void prefetch_next(integer_t row, integer_t col) const {
        integer_t next_idx = (row * cols_ + col + 16) & ~15; // Next cache line
        if (next_idx < rows_ * cols_) {
            __builtin_prefetch(&data_[next_idx], 1, 3); // Read, high temporal locality
        }
    }
    
    // Cache-friendly access with prefetching
    T& operator()(integer_t row, integer_t col) {
        prefetch_next(row, col);
        return data_[row * cols_ + col];
    }
    
    const T& operator()(integer_t row, integer_t col) const {
        prefetch_next(row, col);
        return data_[row * cols_ + col];
    }

private:
    integer_t rows_, cols_;
    T* data_;
};

// ============================================================================
// Template Metaprogramming Optimizations
// ============================================================================

// Compile-time feature selection optimization
template<integer_t N_FEATURES>
struct FeatureSelectionOptimizer {
    static constexpr integer_t optimal_mtry() {
        if constexpr (N_FEATURES <= 10) return N_FEATURES / 2;
        else if constexpr (N_FEATURES <= 100) return static_cast<integer_t>(std::sqrt(N_FEATURES));
        else if constexpr (N_FEATURES <= 1000) return N_FEATURES / 3;
        else return N_FEATURES / 4;
    }
    
    static constexpr integer_t optimal_batch_size() {
        if constexpr (N_FEATURES <= 50) return 32;
        else if constexpr (N_FEATURES <= 200) return 16;
        else return 8;
    }
};

// Compile-time SIMD width optimization
template<typename T>
struct SIMDOptimizer {
    static constexpr integer_t vector_width() {
        if constexpr (std::is_same_v<T, float>) {
            return 8; // AVX2: 8 floats
        } else if constexpr (std::is_same_v<T, double>) {
            return 4; // AVX2: 4 doubles
        } else if constexpr (std::is_same_v<T, integer_t>) {
            return 8; // AVX2: 8 ints
        } else {
            return 1; // Scalar fallback
        }
    }
};

// Compile-time loop unrolling
template<integer_t N>
struct LoopUnroller {
    template<typename Func>
    static void unroll(Func&& func) {
        if constexpr (N > 0) {
            func(N - 1);
            LoopUnroller<N - 1>::unroll(std::forward<Func>(func));
        }
    }
};

// ============================================================================
// Branch Prediction Optimizations
// ============================================================================

// Branch prediction hints
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

// Optimized split finding with branch prediction
class BranchOptimizedSplitFinder {
public:
    template<typename T>
    static std::pair<real_t, real_t> find_best_split_optimized(
        const T* values, const integer_t* labels, const integer_t* indices,
        integer_t n_samples, integer_t n_classes, integer_t min_samples_split) {
        
        real_t best_split_value = 0.0f;
        real_t best_impurity_reduction = 0.0f;
        
        // Pre-compute total impurity (likely to be used)
        real_t total_impurity = compute_total_impurity(labels, indices, n_samples, n_classes);
        
        // Main loop with branch prediction
        for (integer_t split_idx = min_samples_split; 
             LIKELY(split_idx < n_samples - min_samples_split); 
             ++split_idx) {
            
            // Early termination for unlikely splits
            if (UNLIKELY(values[split_idx] == values[split_idx + 1])) {
                continue; // Skip tied values
            }
            
            real_t impurity_reduction = compute_split_impurity_reduction(
                values, labels, indices, split_idx, n_samples, n_classes, total_impurity);
            
            if (LIKELY(impurity_reduction > best_impurity_reduction)) {
                best_impurity_reduction = impurity_reduction;
                best_split_value = values[split_idx];
            }
        }
        
        return {best_split_value, best_impurity_reduction};
    }

private:
    static real_t compute_total_impurity(const integer_t* labels, const integer_t* indices,
                                       integer_t n_samples, integer_t n_classes);
    static real_t compute_split_impurity_reduction(const real_t* values, const integer_t* labels,
                                                 const integer_t* indices, integer_t split_idx,
                                                 integer_t n_samples, integer_t n_classes,
                                                 real_t total_impurity);
};

// ============================================================================
// Memory Prefetching Optimizations
// ============================================================================

// Advanced memory prefetching for tree traversal
class MemoryPrefetcher {
public:
    static void prefetch_tree_data(const CacheOptimizedTreeNode* nodes, 
                                  integer_t node_id, integer_t depth) {
        // Prefetch current node
        __builtin_prefetch(&nodes[node_id], 0, 3);
        
        // Prefetch likely next nodes (left child)
        if (LIKELY(nodes[node_id].left_child >= 0)) {
            __builtin_prefetch(&nodes[nodes[node_id].left_child], 0, 2);
        }
        
        // Prefetch likely next nodes (right child)
        if (LIKELY(nodes[node_id].right_child >= 0)) {
            __builtin_prefetch(&nodes[nodes[node_id].right_child], 0, 2);
        }
        
        // Prefetch sibling node for parallel processing
        if (LIKELY(nodes[node_id].sibling >= 0)) {
            __builtin_prefetch(&nodes[nodes[node_id].sibling], 0, 1);
        }
    }
    
    static void prefetch_feature_data(const real_t* features, integer_t feature_idx,
                                    integer_t sample_start, integer_t sample_end) {
        // Prefetch feature values for current batch
        for (integer_t i = sample_start; i < sample_end; i += 16) {
            __builtin_prefetch(&features[feature_idx * sample_end + i], 0, 3);
        }
    }
};

// ============================================================================
// Lock-Free Data Structures
// ============================================================================

// Lock-free atomic operations for parallel tree growing
template<typename T>
class LockFreeAccumulator {
public:
    LockFreeAccumulator() : value_(0) {}
    
    void add(T increment) {
        if constexpr (std::is_integral_v<T>) {
            __atomic_add_fetch(&value_, increment, __ATOMIC_RELAXED);
        } else {
            // For floating point, use compare-and-swap loop
            T current = value_.load(std::memory_order_relaxed);
            T desired;
            do {
                desired = current + increment;
            } while (!value_.compare_exchange_weak(current, desired, 
                                                   std::memory_order_relaxed));
        }
    }
    
    T get() const { return value_.load(std::memory_order_relaxed); }
    void reset() { value_.store(0, std::memory_order_relaxed); }

private:
    std::atomic<T> value_;
};

// Lock-free circular buffer for tree processing
template<typename T, size_t N>
class LockFreeCircularBuffer {
public:
    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % N;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }
        
        item = buffer_[current_head];
        head_.store((current_head + 1) % N, std::memory_order_release);
        return true;
    }

private:
    std::array<T, N> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
};

// ============================================================================
// Custom Memory Allocators
// ============================================================================

// High-performance custom allocator with memory pools
template<typename T>
class HighPerformanceAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    
    HighPerformanceAllocator() : pool_(nullptr) {
        pool_ = std::aligned_alloc(64, POOL_SIZE);
        current_ptr_ = static_cast<char*>(pool_);
        end_ptr_ = static_cast<char*>(pool_) + POOL_SIZE;
    }
    
    ~HighPerformanceAllocator() {
        if (pool_) std::free(pool_);
    }
    
    pointer allocate(size_type n) {
        size_type bytes = n * sizeof(T);
        size_type aligned_bytes = (bytes + 63) & ~63; // 64-byte alignment
        
        if (current_ptr_ + aligned_bytes <= end_ptr_) {
            pointer result = reinterpret_cast<pointer>(current_ptr_);
            current_ptr_ += aligned_bytes;
            return result;
        }
        
        // Fallback to system allocation
        return static_cast<pointer>(std::aligned_alloc(64, bytes));
    }
    
    void deallocate(pointer p, size_type n) {
        // Custom allocator doesn't free individual blocks
        // Memory is freed when allocator is destroyed
    }

private:
    static constexpr size_t POOL_SIZE = 1024 * 1024; // 1MB pool
    void* pool_;
    char* current_ptr_;
    char* end_ptr_;
};

// ============================================================================
// SIMD-Enhanced Algorithms
// ============================================================================

// Advanced SIMD-optimized Gini calculation
class SIMDEnhancedGiniCalculator {
public:
    // AVX-512 optimized Gini calculation
    static real_t calculate_gini_avx512(const integer_t* class_counts, 
                                       integer_t total_samples, integer_t n_classes) {
        if (n_classes < 8) {
            return calculate_gini_avx2(class_counts, total_samples, n_classes);
        }
        
        __m512 sum_vec = _mm512_setzero_ps();
        __m512 total_vec = _mm512_set1_ps(total_samples);
        
        for (integer_t i = 0; i < n_classes; i += 16) {
            if (i + 16 <= n_classes) {
                // Load 16 class counts
                __m512i counts_vec = _mm512_load_si512(&class_counts[i]);
                __m512 counts_fp = _mm512_cvtepi32_ps(counts_vec);
                
                // Calculate squared proportions
                __m512 proportions = _mm512_div_ps(counts_fp, total_vec);
                __m512 squared_proportions = _mm512_mul_ps(proportions, proportions);
                
                // Accumulate
                sum_vec = _mm512_add_ps(sum_vec, squared_proportions);
            } else {
                // Handle remaining classes
                for (integer_t j = i; j < n_classes; ++j) {
                    real_t proportion = static_cast<real_t>(class_counts[j]) / total_samples;
                    sum_vec = _mm512_add_ps(sum_vec, _mm512_set1_ps(proportion * proportion));
                }
                break;
            }
        }
        
        // Sum the vector elements
        real_t sum = _mm512_reduce_add_ps(sum_vec);
        return 1.0f - sum;
    }
    
    // AVX2 optimized Gini calculation
    static real_t calculate_gini_avx2(const integer_t* class_counts, 
                                     integer_t total_samples, integer_t n_classes) {
        __m256 sum_vec = _mm256_setzero_ps();
        __m256 total_vec = _mm256_set1_ps(total_samples);
        
        for (integer_t i = 0; i < n_classes; i += 8) {
            if (i + 8 <= n_classes) {
                // Load 8 class counts
                __m256i counts_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&class_counts[i]));
                __m256 counts_fp = _mm256_cvtepi32_ps(counts_vec);
                
                // Calculate squared proportions
                __m256 proportions = _mm256_div_ps(counts_fp, total_vec);
                __m256 squared_proportions = _mm256_mul_ps(proportions, proportions);
                
                // Accumulate
                sum_vec = _mm256_add_ps(sum_vec, squared_proportions);
            } else {
                // Handle remaining classes
                for (integer_t j = i; j < n_classes; ++j) {
                    real_t proportion = static_cast<real_t>(class_counts[j]) / total_samples;
                    sum_vec = _mm256_add_ps(sum_vec, _mm256_set1_ps(proportion * proportion));
                }
                break;
            }
        }
        
        // Sum the vector elements
        real_t sum_array[8];
        _mm256_store_ps(sum_array, sum_vec);
        real_t sum = 0.0f;
        for (integer_t i = 0; i < 8; ++i) {
            sum += sum_array[i];
        }
        
        return 1.0f - sum;
    }
};

// ============================================================================
// Profile-Guided Optimization
// ============================================================================

// Runtime profiling and optimization
class ProfileGuidedOptimizer {
public:
    struct ProfileData {
        std::unordered_map<std::string, double> function_times;
        std::unordered_map<std::string, size_t> function_calls;
        std::unordered_map<std::string, size_t> cache_misses;
        std::unordered_map<std::string, size_t> branch_mispredictions;
    };
    
    void start_profiling();
    void stop_profiling();
    void optimize_based_on_profile();
    
    ProfileData get_profile_data() const { return profile_data_; }

private:
    ProfileData profile_data_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    void collect_hardware_counters();
    void optimize_hot_paths();
    void optimize_cold_paths();
};

// ============================================================================
// Convenience Functions
// ============================================================================

// Get optimal configuration based on compile-time parameters
template<integer_t N_SAMPLES, integer_t N_FEATURES, integer_t N_CLASSES>
struct OptimalConfiguration {
    static constexpr integer_t optimal_mtry = FeatureSelectionOptimizer<N_FEATURES>::optimal_mtry();
    static constexpr integer_t optimal_batch_size = FeatureSelectionOptimizer<N_FEATURES>::optimal_batch_size();
    static constexpr integer_t optimal_threads = std::min(static_cast<integer_t>(std::thread::hardware_concurrency()), 
                                                         static_cast<integer_t>(N_SAMPLES / 1000));
    static constexpr bool use_gpu = N_SAMPLES > 10000 && N_FEATURES > 100;
    static constexpr bool use_sparse = N_FEATURES > 1000;
};

// Compile-time optimization selection
template<typename Config>
void optimize_for_config() {
    if constexpr (Config::use_gpu) {
        // Enable GPU optimizations
    }
    if constexpr (Config::use_sparse) {
        // Enable sparse optimizations
    }
    // ... other optimizations
}

} // namespace rf

#endif // RF_ADVANCED_OPTIMIZATIONS_HPP
