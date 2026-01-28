#ifndef RF_VECTORIZED_OPS_HPP
#define RF_VECTORIZED_OPS_HPP

#include "rf_types.hpp"
#include "rf_memory_pool.hpp"
#include <immintrin.h>  // AVX/SSE intrinsics
#include <algorithm>
#include <numeric>

namespace rf {

// Vectorized operations for Random Forest computations
// Uses SIMD instructions for maximum performance

// ============================================================================
// Vectorized Split Finding
// ============================================================================

// Vectorized Gini impurity calculation for classification
class VectorizedGiniCalculator {
public:
    // Calculate Gini impurity for all possible splits
    // Returns best split value and impurity reduction
    static std::pair<real_t, real_t> find_best_split(
        const real_t* values,           // Feature values (sorted)
        const integer_t* labels,       // Class labels
        const integer_t* indices,      // Sample indices
        integer_t n_samples,           // Number of samples
        integer_t n_classes,           // Number of classes
        integer_t min_samples_split   // Minimum samples for split
    );

    // Calculate Gini impurity for a specific split
    static real_t calculate_gini(
        const integer_t* class_counts_left,   // Left node class counts
        const integer_t* class_counts_right,  // Right node class counts
        integer_t n_left,                     // Left node sample count
        integer_t n_right,                    // Right node sample count
        integer_t n_classes                   // Number of classes
    );

private:
    // Vectorized class count accumulation
    static void accumulate_class_counts_avx(
        const integer_t* labels,
        const integer_t* indices,
        integer_t start_idx,
        integer_t end_idx,
        integer_t* class_counts,
        integer_t n_classes
    );

    // Vectorized impurity calculation
    static real_t calculate_impurity_vectorized(
        const integer_t* class_counts,
        integer_t total_samples,
        integer_t n_classes
    );
};

// Vectorized MSE calculation for regression
class VectorizedMSECalculator {
public:
    // Calculate MSE for all possible splits
    static std::pair<real_t, real_t> find_best_split(
        const real_t* values,           // Feature values (sorted)
        const real_t* targets,          // Target values
        const integer_t* indices,       // Sample indices
        integer_t n_samples,            // Number of samples
        integer_t min_samples_split    // Minimum samples for split
    );

    // Calculate MSE for a specific split
    static real_t calculate_mse(
        const real_t* targets_left,     // Left node targets
        const real_t* targets_right,   // Right node targets
        integer_t n_left,               // Left node sample count
        integer_t n_right               // Right node sample count
    );

private:
    // Vectorized mean calculation
    static real_t calculate_mean_vectorized(
        const real_t* values,
        integer_t n_samples
    );

    // Vectorized variance calculation
    static real_t calculate_variance_vectorized(
        const real_t* values,
        real_t mean,
        integer_t n_samples
    );
};

// ============================================================================
// Vectorized Sorting and Searching
// ============================================================================

class VectorizedSorting {
public:
    // Vectorized quicksort for small arrays
    static void quicksort_vectorized(
        real_t* values,
        integer_t* indices,
        integer_t n_samples
    );

    // Vectorized binary search
    static integer_t binary_search_vectorized(
        const real_t* values,
        real_t target,
        integer_t n_samples
    );

    // Vectorized merge for merge sort
    static void merge_vectorized(
        real_t* values,
        integer_t* indices,
        integer_t left,
        integer_t mid,
        integer_t right
    );

private:
    // Vectorized partition for quicksort
    static integer_t partition_vectorized(
        real_t* values,
        integer_t* indices,
        integer_t low,
        integer_t high
    );
};

// ============================================================================
// Vectorized Data Movement
// ============================================================================

class VectorizedDataMovement {
public:
    // Vectorized array copy with alignment
    static void copy_aligned(
        void* dest,
        const void* src,
        size_t size_bytes
    );

    // Vectorized array initialization
    static void initialize_vectorized(
        real_t* array,
        real_t value,
        integer_t n_elements
    );

    // Vectorized array sum
    static real_t sum_vectorized(
        const real_t* array,
        integer_t n_elements
    );

    // Vectorized array dot product
    static real_t dot_product_vectorized(
        const real_t* a,
        const real_t* b,
        integer_t n_elements
    );

    // Vectorized array maximum
    static real_t max_vectorized(
        const real_t* array,
        integer_t n_elements
    );

    // Vectorized array minimum
    static real_t min_vectorized(
        const real_t* array,
        integer_t n_elements
    );
};

// ============================================================================
// Vectorized Bootstrap Sampling
// ============================================================================

class VectorizedBootstrap {
public:
    // Vectorized bootstrap sampling
    static void sample_with_replacement(
        const integer_t* indices,
        integer_t* sampled_indices,
        integer_t n_samples,
        integer_t n_bootstrap,
        integer_t seed
    );

    // Vectorized in-bag/out-of-bag determination
    static void determine_inbag_oob(
        const integer_t* sampled_indices,
        integer_t n_bootstrap,
        integer_t n_samples,
        integer_t* inbag_counts,
        integer_t* oob_indices
    );

private:
    // Vectorized random number generation
    static void generate_random_vectorized(
        real_t* random_values,
        integer_t n_values,
        integer_t seed
    );
};

// ============================================================================
// Vectorized Proximity Computation
// ============================================================================

class VectorizedProximity {
public:
    // Vectorized proximity matrix update
    static void update_proximity_vectorized(
        dp_t* proximity_matrix,
        const integer_t* terminal_nodes,
        const integer_t* node_sizes,
        integer_t n_samples,
        integer_t tree_id
    );

    // Vectorized proximity matrix normalization
    static void normalize_proximity_vectorized(
        dp_t* proximity_matrix,
        integer_t n_samples,
        integer_t n_trees
    );

    // Vectorized proximity matrix symmetrization
    static void symmetrize_proximity_vectorized(
        dp_t* proximity_matrix,
        integer_t n_samples
    );

private:
    // Vectorized matrix operations
    static void matrix_add_vectorized(
        dp_t* matrix_a,
        const dp_t* matrix_b,
        integer_t n_elements
    );

    static void matrix_multiply_scalar_vectorized(
        dp_t* matrix,
        dp_t scalar,
        integer_t n_elements
    );
};

// ============================================================================
// Vectorized Variable Importance
// ============================================================================

class VectorizedImportance {
public:
    // Vectorized variable importance calculation
    static void calculate_importance_vectorized(
        const real_t* oob_errors,
        const real_t* permuted_errors,
        real_t* importance_scores,
        integer_t n_variables,
        integer_t n_samples
    );

    // Vectorized importance accumulation
    static void accumulate_importance_vectorized(
        real_t* total_importance,
        const real_t* tree_importance,
        integer_t n_variables
    );

private:
    // Vectorized error calculation
    static real_t calculate_error_vectorized(
        const real_t* predictions,
        const real_t* targets,
        integer_t n_samples
    );
};

// ============================================================================
// Utility Functions
// ============================================================================

// Check if CPU supports AVX instructions
bool has_avx_support();

// Check if CPU supports AVX2 instructions
bool has_avx2_support();

// Check if CPU supports AVX-512 instructions
bool has_avx512_support();

// Get optimal SIMD width for current CPU
integer_t get_optimal_simd_width();

// Align pointer to SIMD boundary
template<typename T>
T* align_simd(T* ptr) {
    constexpr size_t alignment = 32; // AVX alignment
    return reinterpret_cast<T*>(
        (reinterpret_cast<uintptr_t>(ptr) + alignment - 1) & ~(alignment - 1)
    );
}

// Check if pointer is SIMD aligned
template<typename T>
bool is_simd_aligned(const T* ptr) {
    constexpr size_t alignment = 32; // AVX alignment
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

} // namespace rf

#endif // RF_VECTORIZED_OPS_HPP
