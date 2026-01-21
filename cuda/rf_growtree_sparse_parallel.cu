/**
 * @file rf_growtree_sparse_parallel.cu
 * @brief Parallel sparse tree growing - matches GPU Dense single-kernel architecture
 * 
 * Each CUDA block handles one tree (same as GPU Dense).
 * Uses sparse matrix access via CudaSparseMatrixCSR::get().
 * Parallel feature evaluation, sorting, and split finding.
 */

#include "rf_growtree_sparse.cuh"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

// Shared memory structure for parallel split finding
struct ThreadSplitResult {
    real_t crit;
    integer_t var;
    real_t threshold;
    integer_t split_idx;
};

// ============================================================================
// Thread-local bitonic sort for sparse data
// Each thread sorts its own local array (no synchronization needed)
// ============================================================================

__device__ __forceinline__ void bitonic_compare_swap_local(
    integer_t* indices, real_t* values, integer_t i, integer_t j, bool ascending
) {
    // Stable sort: use sample index as tiebreaker for equal values
    bool should_swap = ascending ? 
        (values[i] > values[j] || (values[i] == values[j] && indices[i] > indices[j])) :
        (values[i] < values[j] || (values[i] == values[j] && indices[i] < indices[j]));
    
    if (should_swap) {
        integer_t tmp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_idx;
        
        real_t tmp_val = values[i];
        values[i] = values[j];
        values[j] = tmp_val;
    }
}

// Thread-local bitonic sort (no __syncthreads - each thread works independently)
__device__ void gpu_bitonic_sort_local(
    integer_t* indices, real_t* values, integer_t count
) {
    // Pad to next power of 2
    integer_t n = 1;
    while (n < count) n <<= 1;
    
    // Bitonic sort stages
    for (integer_t k = 2; k <= n; k <<= 1) {
        for (integer_t j = k >> 1; j > 0; j >>= 1) {
            for (integer_t i = 0; i < n / 2; ++i) {
                integer_t ixj = i ^ j;
                if (ixj > i && i < count && ixj < count) {
                    bool ascending = ((i & k) == 0);
                    bitonic_compare_swap_local(indices, values, i, ixj, ascending);
                }
            }
        }
    }
}

// Hybrid sort: insertion for small, bitonic for large (thread-local)
__device__ void gpu_hybrid_sort_local(
    integer_t* indices, real_t* values, integer_t count
) {
    if (count <= 64) {
        // Insertion sort for small arrays
        for (integer_t i = 1; i < count; i++) {
            integer_t key_idx = indices[i];
            real_t key_val = values[i];
            integer_t j = i - 1;
            
            while (j >= 0 && values[j] > key_val) {
                indices[j + 1] = indices[j];
                values[j + 1] = values[j];
                j--;
            }
            indices[j + 1] = key_idx;
            values[j + 1] = key_val;
        }
    } else {
        // Bitonic sort for large arrays
        gpu_bitonic_sort_local(indices, values, count);
    }
}

// ============================================================================
// Main parallel sparse tree growing kernel - matches GPU Dense structure
// Each block grows ONE complete tree
// Uses parallel sorting and incremental Gini/MSE (matching GPU Dense)
// Supports both Classification (Gini) and Regression (MSE)
// ============================================================================

__global__ void gpu_sparse_tree_parallel_kernel(
    CudaSparseMatrixCSR X_sparse,  // Sparse data matrix
    integer_t num_trees,
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    integer_t task_type,           // 0=classification, 1=regression
    const integer_t* cl,           // Class labels [nsample] (classification)
    const real_t* y_regression,    // Regression targets [nsample] (regression)
    const integer_t* nin_all,      // Bootstrap freq [num_trees * nsample]
    const real_t* win_all,         // Bootstrap weights [num_trees * nsample]
    integer_t* nodestatus_all,     // Output: [num_trees * maxnode]
    integer_t* bestvar_all,        // Output: [num_trees * maxnode]
    real_t* xbestsplit_all,        // Output: [num_trees * maxnode]
    integer_t* treemap_all,        // Output: [num_trees * 2 * maxnode]
    integer_t* nodeclass_all,      // Output: [num_trees * maxnode] (classification)
    real_t* tnodewt_all,           // Output: [num_trees * maxnode] (regression predictions)
    integer_t* nnode_all,          // Output: [num_trees]
    curandState* random_states     // RNG states [num_trees]
) {
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    integer_t tid = threadIdx.x;
    integer_t num_threads = blockDim.x;
    
    // Calculate offsets for this tree
    integer_t node_offset = tree_id * maxnode;
    integer_t treemap_offset = tree_id * 2 * maxnode;
    integer_t sample_offset = tree_id * nsample;
    
    // Pointers to this tree's arrays
    integer_t* nodestatus = nodestatus_all + node_offset;
    integer_t* nodeclass = nodeclass_all + node_offset;
    real_t* tnodewt = tnodewt_all ? tnodewt_all + node_offset : nullptr;
    integer_t* bestvar = bestvar_all + node_offset;
    real_t* xbestsplit = xbestsplit_all + node_offset;
    integer_t* treemap = treemap_all + treemap_offset;
    integer_t* nnode = nnode_all + tree_id;
    const integer_t* nin = nin_all + sample_offset;
    const real_t* win = win_all + sample_offset;
    
    // Local RNG state
    curandState local_rng = random_states[tree_id];
    
    // Shared memory for sample tracking (matching GPU Dense size)
    __shared__ integer_t node_samples[2048];
    __shared__ integer_t node_sample_count;
    __shared__ integer_t shared_nnode;
    __shared__ integer_t shared_majority_class;
    __shared__ real_t shared_crit0;  // Parent criterion
    __shared__ real_t shared_mean_y;  // For regression: mean y in node
    
    // Shared memory for parallel split finding
    extern __shared__ char shared_mem[];
    ThreadSplitResult* thread_results = reinterpret_cast<ThreadSplitResult*>(shared_mem);
    
    // Per-class weights for incremental Gini (matching GPU Dense)
    real_t wl[32], wr[32];
    
    // For regression: incremental MSE tracking
    real_t sum_y_left, sum_y_right, sum_w_left, sum_w_right;
    
    // Initialize tree structure (thread 0 only)
    if (tid == 0) {
        for (integer_t i = 0; i < maxnode; ++i) {
            nodestatus[i] = 0;
            bestvar[i] = 0;
            xbestsplit[i] = 0.0f;
            treemap[i * 2] = 0;
            treemap[i * 2 + 1] = 0;
            nodeclass[i] = 0;
        }
        
        nodestatus[0] = 2;  // Root active
        shared_nnode = 1;
        nnode[0] = 1;
    }
    __syncthreads();
    
    // Main node processing loop
    integer_t max_iterations = maxnode * 2;
    for (integer_t kgrow = 0; kgrow < shared_nnode && kgrow < max_iterations; ++kgrow) {
        __syncthreads();
        
        if (nodestatus[kgrow] != 2) continue;
        
        // =====================================================================
        // PARALLEL SAMPLE COLLECTION - All threads participate
        // Uses parallel scan for deterministic ordering (matches CPU)
        // =====================================================================
        
        // Shared memory for parallel compaction
        __shared__ integer_t thread_counts[256];  // Count per thread
        __shared__ integer_t thread_offsets[256]; // Prefix sum offsets
        
        if (tid == 0) node_sample_count = 0;
        thread_counts[tid] = 0;
        __syncthreads();
        
        // Pass 1: Each thread counts its qualifying samples
        integer_t my_count = 0;
        integer_t samples_per_thread = (nsample + num_threads - 1) / num_threads;
        integer_t my_start = tid * samples_per_thread;
        integer_t my_end = min(my_start + samples_per_thread, nsample);
        
        // Temporary local storage for this thread's samples (max 32 per thread)
        integer_t my_samples[32];
        
        for (integer_t i = my_start; i < my_end && my_count < 32; ++i) {
            bool reaches_node = false;
            
            if (kgrow == 0) {
                // Root node: check if in-bag
                reaches_node = (nin[i] > 0);
            } else {
                // Non-root: traverse tree (read-only, safe to parallelize)
                if (win[i] > 0.0f) {
                    integer_t current = 0;
                    integer_t depth = 0;
                    while (depth < 100) {
                        if (current == kgrow) {
                            reaches_node = true;
                            break;
                        }
                        if (nodestatus[current] != 1) break;
                        
                        integer_t split_var = bestvar[current];
                        real_t split_val = xbestsplit[current];
                        real_t sample_val = X_sparse.get(i, split_var);
                        
                        if (sample_val <= split_val) {
                            current = treemap[current * 2];
                        } else {
                            current = treemap[current * 2 + 1];
                        }
                        if (current <= 0 || current >= maxnode) break;
                        depth++;
                    }
                }
            }
            
            if (reaches_node) {
                my_samples[my_count++] = i;
            }
        }
        
        thread_counts[tid] = my_count;
        __syncthreads();
        
        // Pass 2: Compute prefix sum (sequential by thread 0 for determinism)
        if (tid == 0) {
            thread_offsets[0] = 0;
            for (integer_t t = 1; t < num_threads; ++t) {
                thread_offsets[t] = thread_offsets[t-1] + thread_counts[t-1];
            }
            node_sample_count = thread_offsets[num_threads-1] + thread_counts[num_threads-1];
            if (node_sample_count > 2048) node_sample_count = 2048;
        }
        __syncthreads();
        
        // Pass 3: Each thread writes its samples to correct position
        integer_t my_offset = thread_offsets[tid];
        for (integer_t i = 0; i < my_count && my_offset + i < 2048; ++i) {
            node_samples[my_offset + i] = my_samples[i];
        }
        __syncthreads();
        
        integer_t n_samples = node_sample_count;
        
        // Check minimum node size
        if (n_samples < 2 * min_node_size) {
            if (tid == 0) {
                nodestatus[kgrow] = -1;
                if (task_type == 1 && y_regression != nullptr) {
                    // REGRESSION: Compute mean y for terminal node
                    real_t sum_y = 0.0f, sum_w = 0.0f;
                    for (integer_t i = 0; i < n_samples; ++i) {
                        integer_t idx = node_samples[i];
                        real_t w = win[idx];
                        sum_y += w * y_regression[idx];
                        sum_w += w;
                    }
                    if (tnodewt) tnodewt[kgrow] = (sum_w > 0.0f) ? (sum_y / sum_w) : 0.0f;
                    nodeclass[kgrow] = 0;  // Not used for regression
                } else {
                    // CLASSIFICATION: Find majority class
                    integer_t class_counts[32] = {0};
                    for (integer_t i = 0; i < n_samples; ++i) {
                        integer_t c = cl[node_samples[i]];
                        if (c >= 0 && c < 32) class_counts[c]++;
                    }
                    integer_t best_class = 0;
                    for (integer_t c = 1; c < nclass && c < 32; c++) {
                        if (class_counts[c] > class_counts[best_class]) best_class = c;
                    }
                    nodeclass[kgrow] = best_class;
                }
            }
            continue;
        }
        
        // Compute class distribution/mean and parent criterion (thread 0)
        if (tid == 0) {
            if (task_type == 1 && y_regression != nullptr) {
                // REGRESSION: Compute mean and variance
                real_t sum_y = 0.0f, sum_w = 0.0f;
                for (integer_t i = 0; i < n_samples; ++i) {
                    integer_t idx = node_samples[i];
                    real_t w = win[idx];
                    sum_y += w * y_regression[idx];
                    sum_w += w;
                }
                shared_mean_y = (sum_w > 0.0f) ? (sum_y / sum_w) : 0.0f;
                shared_majority_class = 0;  // Not used for regression
                
                // Parent MSE criterion (variance * n)
                // For regression, we use weighted variance as the criterion
                // Lower variance = purer node (opposite of Gini where higher = better)
                real_t ss = 0.0f;
                for (integer_t i = 0; i < n_samples; ++i) {
                    integer_t idx = node_samples[i];
                    real_t w = win[idx];
                    real_t diff = y_regression[idx] - shared_mean_y;
                    ss += w * diff * diff;
                }
                shared_crit0 = ss;  // Parent MSE (to be minimized)
                
                // Check if all y values are the same (pure node)
                bool all_same = true;
                real_t first_y = y_regression[node_samples[0]];
                for (integer_t i = 1; i < n_samples && all_same; ++i) {
                    if (y_regression[node_samples[i]] != first_y) all_same = false;
                }
                if (all_same) {
                    nodestatus[kgrow] = -1;
                    if (tnodewt) tnodewt[kgrow] = shared_mean_y;
                }
            } else {
                // CLASSIFICATION: Compute class distribution and Gini
                integer_t class_counts[32] = {0};
                for (integer_t i = 0; i < n_samples; ++i) {
                    integer_t c = cl[node_samples[i]];
                    if (c >= 0 && c < 32) class_counts[c]++;
                }
                
                integer_t max_count = 0, non_zero = 0;
                for (integer_t c = 0; c < nclass && c < 32; c++) {
                    if (class_counts[c] > max_count) {
                        max_count = class_counts[c];
                        shared_majority_class = c;
                    }
                    if (class_counts[c] > 0) non_zero++;
                }
                
                // Parent Gini criterion (matching GPU Dense: pno/pdo)
                real_t pno = 0.0f, pdo = 0.0f;
                for (integer_t c = 0; c < nclass && c < 32; c++) {
                    pno += class_counts[c] * class_counts[c];
                    pdo += class_counts[c];
                }
                shared_crit0 = pdo > 0.0f ? (pno / pdo) : 0.0f;
                
                if (non_zero <= 1) {
                    nodestatus[kgrow] = -1;
                    nodeclass[kgrow] = shared_majority_class;
                }
            }
        }
        __syncthreads();
        
        if (nodestatus[kgrow] == -1) continue;
        
        // =====================================================================
        // PARALLEL SPLIT FINDING - Each thread evaluates different features
        // Uses incremental Gini (classification) or MSE (regression) like GPU Dense
        // =====================================================================
        
        // For regression: MSE criterion is NEGATIVE (we want to MAXIMIZE reduction in MSE)
        // So best_crit starts at -inf and we look for higher values (more negative MSE = better)
        thread_results[tid].crit = (task_type == 1) ? -1e30f : -2.0f;
        thread_results[tid].var = -1;
        thread_results[tid].threshold = 0.0f;
        thread_results[tid].split_idx = -1;
        
        // Each thread handles a subset of mtry features
        for (integer_t mv = tid; mv < mtry; mv += num_threads) {
            // Deterministic random feature selection - use skipahead (matches GPU Dense)
            // This is faster than loop and produces identical RNG sequence
            curandState temp_state = local_rng;
            skipahead(kgrow * mtry + mv, &temp_state);  // Efficient jump to position
            real_t rand_val = curand_uniform(&temp_state);
            integer_t mvar = static_cast<integer_t>(rand_val * mdim);
            if (mvar >= mdim) mvar = mdim - 1;
            
            // Collect (sample_idx, value) pairs and sort by value
            // Use local arrays since each thread works independently
            integer_t sorted_samples[512];
            real_t sorted_values[512];
            integer_t n_valid = 0;
            
            for (integer_t i = 0; i < n_samples && n_valid < 512; ++i) {
                integer_t sample_idx = node_samples[i];
                real_t v = X_sparse.get(sample_idx, mvar);
                sorted_samples[n_valid] = sample_idx;
                sorted_values[n_valid] = v;
                n_valid++;
            }
            
            // Hybrid sort: insertion for small (<= 64), bitonic for large
            // Both are deterministic and produce identical results across runs
            gpu_hybrid_sort_local(sorted_samples, sorted_values, n_valid);
            
            if (task_type == 1 && y_regression != nullptr) {
                // =====================================================================
                // REGRESSION: Incremental MSE computation (matching GPU Dense)
                // =====================================================================
                
                // Initialize: all samples on right
                sum_y_left = 0.0f;
                sum_w_left = 0.0f;
                sum_y_right = 0.0f;
                sum_w_right = 0.0f;
                
                for (integer_t i = 0; i < n_valid; ++i) {
                    integer_t idx = sorted_samples[i];
                    real_t w = win[idx];
                    sum_y_right += w * y_regression[idx];
                    sum_w_right += w;
                }
                
                // Try each split point
                for (integer_t ii = 0; ii < n_valid - 1; ++ii) {
                    integer_t sample_idx = sorted_samples[ii];
                    real_t val1 = sorted_values[ii];
                    real_t val2 = sorted_values[ii + 1];
                    
                    real_t w = win[sample_idx];
                    real_t y_val = y_regression[sample_idx];
                    
                    // Move sample from right to left
                    sum_y_left += w * y_val;
                    sum_w_left += w;
                    sum_y_right -= w * y_val;
                    sum_w_right -= w;
                    
                    // Skip ties
                    if (val1 >= val2) continue;
                    
                    // Check minimum node size (by weight)
                    if (sum_w_left < min_node_size || sum_w_right < min_node_size) continue;
                    
                    // Compute weighted MSE for each child
                    // MSE_left = sum(w * (y - mean_left)^2) / sum_w_left
                    // = sum(w * y^2) / sum_w - mean_left^2
                    // But for split quality, we use: sum_w_left * var_left + sum_w_right * var_right
                    // Which simplifies to: total_ss - (sum_y_left^2/sum_w_left + sum_y_right^2/sum_w_right)
                    
                    if (sum_w_left > 1.0e-5f && sum_w_right > 1.0e-5f) {
                        // Criterion: (sum_y_left^2/sum_w_left + sum_y_right^2/sum_w_right)
                        // Higher = better (more variance explained)
                        real_t crit = (sum_y_left * sum_y_left / sum_w_left) + 
                                      (sum_y_right * sum_y_right / sum_w_right);
                        
                        // Deterministic tiebreaker: prefer lower var
                        if (crit > thread_results[tid].crit || 
                            (crit == thread_results[tid].crit && mvar < thread_results[tid].var)) {
                            thread_results[tid].crit = crit;
                            thread_results[tid].var = mvar;
                            thread_results[tid].threshold = (val1 + val2) / 2.0f;
                            thread_results[tid].split_idx = ii;
                        }
                    }
                }
            } else {
                // =====================================================================
                // CLASSIFICATION: Incremental Gini computation (matching GPU Dense)
                // =====================================================================
                
                // Initialize per-class weights for incremental Gini
                for (integer_t c = 0; c < 32; c++) {
                    wl[c] = 0.0f;
                    wr[c] = 0.0f;
                }
                
                // Count total samples per class for this node
                for (integer_t i = 0; i < n_samples; ++i) {
                    integer_t c = cl[node_samples[i]];
                    if (c >= 0 && c < 32) {
                        wr[c] += win[node_samples[i]];
                    }
                }
                
                // Incremental Gini computation (matching GPU Dense exactly)
                real_t rln = 0.0f, rld = 0.0f;
                real_t rrn = 0.0f, rrd = 0.0f;
                
                // Initialize rrn and rrd from right side
                for (integer_t c = 0; c < nclass && c < 32; c++) {
                    rrn += wr[c] * wr[c];
                    rrd += wr[c];
                }
                
                // Try each split point with O(1) incremental updates
                for (integer_t ii = 0; ii < n_valid - 1; ++ii) {
                    integer_t sample_idx = sorted_samples[ii];
                    real_t val1 = sorted_values[ii];
                    real_t val2 = sorted_values[ii + 1];
                    
                    integer_t k = cl[sample_idx];
                    real_t u = win[sample_idx];
                    
                    if (k >= 0 && k < 32) {
                        // Incremental Gini update (matching GPU Dense)
                        rln += u * (u + 2.0f * wl[k]);
                        rrn += u * (u - 2.0f * wr[k]);
                        rld += u;
                        rrd -= u;
                        wl[k] += u;
                        wr[k] -= u;
                    }
                    
                    // Skip ties
                    if (val1 >= val2) continue;
                    
                    // Check minimum node size
                    if (rld < min_node_size || rrd < min_node_size) continue;
                    
                    // Compute criterion (matching GPU Dense)
                    if (rld > 1.0e-5f && rrd > 1.0e-5f) {
                        real_t crit = (rln / rld) + (rrn / rrd);
                        
                        // Deterministic tiebreaker: prefer lower var
                        if (crit > thread_results[tid].crit || 
                            (crit == thread_results[tid].crit && mvar < thread_results[tid].var)) {
                            thread_results[tid].crit = crit;
                            thread_results[tid].var = mvar;
                            thread_results[tid].threshold = (val1 + val2) / 2.0f;
                            thread_results[tid].split_idx = ii;
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // =====================================================================
        // REDUCTION - Thread 0 finds global best with deterministic tie-breaking
        // =====================================================================
        
        // For regression, initial crit should match what was set for thread_results
        real_t best_crit = (task_type == 1) ? -1e30f : -2.0f;
        integer_t best_var = -1;
        real_t best_threshold = 0.0f;
        
        if (tid == 0) {
            for (integer_t t = 0; t < num_threads; ++t) {
                if (thread_results[t].var >= 0) {
                    if (thread_results[t].crit > best_crit || 
                        (thread_results[t].crit == best_crit && thread_results[t].var < best_var)) {
                        best_crit = thread_results[t].crit;
                        best_var = thread_results[t].var;
                        best_threshold = thread_results[t].threshold;
                    }
                }
            }
            
            // No need to advance RNG - skipahead uses absolute positioning
            // from the base state, so local_rng stays at its initial value
        }
        __syncthreads();
        
        // Apply split or make terminal
        if (tid == 0) {
            // Check for valid split (threshold depends on task type)
            real_t min_valid_crit = (task_type == 1) ? -1e29f : -1.0f;
            
            if (best_var >= 0 && best_crit > min_valid_crit && shared_nnode + 2 <= maxnode) {
                integer_t left_child = shared_nnode;
                integer_t right_child = shared_nnode + 1;
                
                nodestatus[kgrow] = 1;
                bestvar[kgrow] = best_var;
                xbestsplit[kgrow] = best_threshold;
                
                treemap[kgrow * 2] = left_child;
                treemap[kgrow * 2 + 1] = right_child;
                
                nodestatus[left_child] = 2;
                nodestatus[right_child] = 2;
                
                shared_nnode += 2;
                nnode[0] = shared_nnode;
            } else {
                nodestatus[kgrow] = -1;
                if (task_type == 1 && y_regression != nullptr && tnodewt != nullptr) {
                    // REGRESSION: Store mean y for terminal node
                    tnodewt[kgrow] = shared_mean_y;
                }
                nodeclass[kgrow] = shared_majority_class;
            }
        }
        __syncthreads();
    }
    
    // Save RNG state
    if (tid == 0) {
        random_states[tree_id] = local_rng;
        nnode[0] = shared_nnode;
    }
}

// ============================================================================
// Host wrapper function
// ============================================================================

integer_t gpu_growtree_sparse_parallel_batch(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const real_t* d_y_regression,
    const integer_t* d_nin_all,
    const real_t* d_win_all,
    integer_t nsample,
    integer_t mdim,
    integer_t nclass,
    integer_t mtry,
    integer_t maxnode,
    integer_t min_node_size,
    integer_t num_trees,
    integer_t task_type,
    curandState* d_rng_states,
    integer_t* d_nodestatus_all,
    integer_t* d_bestvar_all,
    real_t* d_xbestsplit_all,
    integer_t* d_treemap_all,
    integer_t* d_nodeclass_all,
    real_t* d_tnodewt_all,
    integer_t* d_nnode_all,
    cudaStream_t stream
) {
    // Initialize output arrays (cast to size_t to prevent integer overflow)
    cudaMemsetAsync(d_nodestatus_all, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t), stream);
    cudaMemsetAsync(d_treemap_all, 0, static_cast<size_t>(num_trees) * 2 * static_cast<size_t>(maxnode) * sizeof(integer_t), stream);
    cudaMemsetAsync(d_nodeclass_all, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(integer_t), stream);
    cudaMemsetAsync(d_nnode_all, 0, static_cast<size_t>(num_trees) * sizeof(integer_t), stream);
    if (d_tnodewt_all) {
        cudaMemsetAsync(d_tnodewt_all, 0, static_cast<size_t>(num_trees) * static_cast<size_t>(maxnode) * sizeof(real_t), stream);
    }
    
    // Dynamic threads per block based on GPU and mtry
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Calculate optimal threads
    integer_t optimal_threads = mtry;
    if (optimal_threads < 32) optimal_threads = 32;
    optimal_threads = ((optimal_threads + 31) / 32) * 32;
    if (optimal_threads > 256) optimal_threads = 256;
    if (optimal_threads > props.maxThreadsPerBlock) {
        optimal_threads = (props.maxThreadsPerBlock / 32) * 32;
    }
    
    // Check shared memory constraint
    size_t shared_mem_needed = optimal_threads * sizeof(ThreadSplitResult);
    if (shared_mem_needed > props.sharedMemPerBlock) {
        optimal_threads = static_cast<integer_t>(props.sharedMemPerBlock / sizeof(ThreadSplitResult));
        optimal_threads = (optimal_threads / 32) * 32;
        if (optimal_threads < 32) optimal_threads = 32;
    }
    
    dim3 grid(num_trees);
    dim3 block(optimal_threads);
    size_t shared_mem_size = optimal_threads * sizeof(ThreadSplitResult);
    
    gpu_sparse_tree_parallel_kernel<<<grid, block, shared_mem_size, stream>>>(
        X_sparse,
        num_trees,
        nsample,
        mdim,
        nclass,
        mtry,
        maxnode,
        min_node_size,
        task_type,
        d_cl,
        d_y_regression,
        d_nin_all,
        d_win_all,
        d_nodestatus_all,
        d_bestvar_all,
        d_xbestsplit_all,
        d_treemap_all,
        d_nodeclass_all,
        d_tnodewt_all,
        d_nnode_all,
        d_rng_states
    );
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        return CUDA_ERROR_MEMORY_ALLOCATION;
    }
    
    cudaStreamSynchronize(stream);
    
    return CUDA_OK;
}

} // namespace cuda
} // namespace rf
