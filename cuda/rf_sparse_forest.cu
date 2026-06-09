/**
 * @file rf_sparse_forest.cu
 * @brief GPU sparse random forest orchestration implementation
 * 
 * Main training loop that coordinates all sparse GPU operations.
 * Matches CPU training flow exactly for reproducibility.
 */

#include "rf_sparse_forest.cuh"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include "rf_config.hpp"
#include "rf_growtree_sparse.cuh"
#include "rf_testreebag_sparse.cuh"
#include "rf_varimp_sparse.cuh"
#include "rf_proximity_importance_sparse.cuh"
#include "rf_proximity_importance.cuh"  // For gpu_count_oob_samples
// NOTE: rf_proximity_sparse.cuh removed - GPU sparse now uses leaf_assignments for proximity
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <iostream>
#include <cstring>  // For std::memcpy

// Forward declaration for histogram functions
namespace rf {
namespace cuda {
    // From rf_sparse_histogram.cu
    void gpu_compute_sparse_bin_edges(
        const CudaSparseMatrixCSR& X_sparse,
        integer_t max_bins,
        SparseHistogramData& hist_data,
        cudaStream_t stream
    );
    
    void gpu_bin_sparse_data(
        const CudaSparseMatrixCSR& X_sparse,
        SparseHistogramData& hist_data,
        cudaStream_t stream
    );
}
}

namespace rf {
namespace cuda {

// ============================================================================
// Compute tnodewt (node weights) for sparse data - needed for casewise mode
// Each thread handles one node, computes sum of weights for samples in that node
// ============================================================================

__global__ void gpu_compute_tnodewt_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    const integer_t* cl,
    const real_t* win,
    const integer_t* nin,
    integer_t nclass,
    integer_t task_type,  // 0=classification, 1=regression
    real_t* tnodewt
) {
    integer_t node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= nnode) return;
    
    // Only compute for terminal nodes
    if (nodestatus[node_id] != -1) {
        tnodewt[node_id] = 0.0f;
        return;
    }
    
    real_t sum_nin = 0.0f;
    integer_t sample_count = 0;
    
    // For each in-bag sample, check if it reaches this terminal node
    for (integer_t n = 0; n < nsample; n++) {
        if (nin[n] <= 0) continue;  // Skip OOB samples
        
        // Traverse tree from root to find terminal node for this sample
        integer_t current_node = 0;
        integer_t max_depth = 100;
        
        for (integer_t d = 0; d < max_depth; d++) {
            if (nodestatus[current_node] == -1) break;  // Terminal
            if (nodestatus[current_node] != 1) break;   // Not splittable
            
            integer_t split_var = bestvar[current_node];
            if (split_var < 0 || split_var >= mdim) break;
            
            real_t split_point = xbestsplit[current_node];
            real_t sample_value = X_sparse.get(n, split_var);
            
            if (sample_value <= split_point) {
                current_node = treemap[current_node * 2];      // Left child
            } else {
                current_node = treemap[current_node * 2 + 1];  // Right child
            }
            
            if (current_node < 0 || current_node >= nnode) break;
        }
        
        // If this sample reaches our target node, accumulate weights
        if (current_node == node_id) {
            sum_nin += static_cast<real_t>(nin[n]);
            sample_count++;
        }
    }
    
    // For classification casewise: tnodewt = sum(nin) / count = mean bootstrap frequency
    // This matches CPU and GPU Dense (CPU fallback)
    if (sample_count > 0) {
        tnodewt[node_id] = sum_nin / static_cast<real_t>(sample_count);
    } else {
        tnodewt[node_id] = 0.0f;
    }
}

// ============================================================================
// Bootstrap kernel
// ============================================================================

// Sequential bootstrap kernel (for single tree)
// sample_weights_cum: cumulative weights for weighted sampling (nullptr = uniform)
__global__ void gpu_bootstrap_kernel(
    integer_t nsample,
    integer_t tree_id,
    curandState* rng_states,
    integer_t* nin,  // [nsample] - bootstrap frequency
    real_t* win,     // [nsample] - bootstrap weights
    const real_t* sample_weights_cum,
    real_t total_sample_weight
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;
    
    for (integer_t i = 0; i < nsample; i++) {
        nin[i] = 0;
        win[i] = 0.0f;
    }
    
    curandState local_state = rng_states[tree_id];
    for (integer_t i = 0; i < nsample; i++) {
        real_t rand_val = curand_uniform(&local_state);
        integer_t idx;
        
        if (sample_weights_cum != nullptr) {
            real_t target = rand_val * total_sample_weight;
            integer_t lo = 0, hi = nsample - 1;
            while (lo < hi) {
                integer_t mid = lo + (hi - lo) / 2;
                if (sample_weights_cum[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            idx = lo;
        } else {
            idx = static_cast<integer_t>(rand_val * nsample);
            if (idx >= nsample) idx = nsample - 1;
        }
        
        nin[idx]++;
        win[idx] += 1.0f;
    }
    rng_states[tree_id] = local_state;
}

// Parallel bootstrap kernel - one block per tree, all threads participate
// sample_weights_cum: cumulative weights for weighted sampling (nullptr = uniform)
__global__ void gpu_bootstrap_batch_kernel(
    integer_t nsample,
    integer_t num_trees,
    curandState* rng_states,
    integer_t* nin_all,  // [num_trees * nsample] - bootstrap frequency
    real_t* win_all,     // [num_trees * nsample] - bootstrap weights
    const real_t* sample_weights_cum,
    real_t total_sample_weight
) {
    integer_t tree_id = blockIdx.x;
    if (tree_id >= num_trees) return;
    
    integer_t tid = threadIdx.x;
    integer_t stride = blockDim.x;
    
    integer_t offset = tree_id * nsample;
    integer_t* nin = nin_all + offset;
    real_t* win = win_all + offset;
    
    for (integer_t i = tid; i < nsample; i += stride) {
        nin[i] = 0;
        win[i] = 0.0f;
    }
    __syncthreads();
    
    curandState local_state = rng_states[tree_id];
    
    for (integer_t i = tid; i < nsample; i += stride) {
        curandState thread_state = local_state;
        skipahead(i, &thread_state);
        
        real_t rand_val = curand_uniform(&thread_state);
        integer_t idx;
        
        if (sample_weights_cum != nullptr) {
            real_t target = rand_val * total_sample_weight;
            integer_t lo = 0, hi = nsample - 1;
            while (lo < hi) {
                integer_t mid = lo + (hi - lo) / 2;
                if (sample_weights_cum[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            idx = lo;
        } else {
            idx = static_cast<integer_t>(rand_val * nsample);
            if (idx >= nsample) idx = nsample - 1;
        }
        
        ::atomicAdd(&nin[idx], 1);
        ::atomicAdd(&win[idx], 1.0f);
    }
    __syncthreads();
}

// ============================================================================
// RNG initialization kernel
// ============================================================================

__global__ void gpu_init_rng_kernel(
    curandState* states,
    const integer_t* seeds,  // Per-tree seeds to match GPU Dense
    integer_t n_states
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_states) {
        // Use per-tree seed (matches GPU Dense: curand_init(seeds[tid], tid, 0, ...))
        curand_init(seeds[tid], tid, 0, &states[tid]);
    }
}

// ============================================================================
// Set jtr predictions from terminal nodes (for variable importance)
// ============================================================================

__global__ void gpu_set_jtr_from_nodextr_kernel(
    const integer_t* nodextr,      // Terminal node for each sample
    const integer_t* nodeclass,    // Class at each node
    const integer_t* nin,          // Bootstrap frequency (0 = OOB)
    integer_t nsample,
    integer_t nnode,
    integer_t* jtr                 // Output predictions
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nsample) return;
    
    // Set prediction from terminal node
    integer_t terminal_node = nodextr[tid];
    if (terminal_node >= 0 && terminal_node < nnode) {
        jtr[tid] = nodeclass[terminal_node];
    } else {
        // Use -1 as sentinel for invalid nodes (not a valid class label)
        // This prevents false matches with class 0 (synthetic samples in unsupervised mode)
        jtr[tid] = -1;
    }
}

// ============================================================================
// Compute OOB predictions from accumulated votes
// ============================================================================

__global__ void gpu_compute_oob_predictions_kernel(
    const real_t* oob_votes,      // [nsample * nclass] vote counts (real_t for fractional casewise weights)
    const integer_t* oob_counts,  // [nsample] number of votes
    const integer_t* y_true,      // True labels
    integer_t nsample,
    integer_t nclass,
    integer_t* oob_predictions,   // Output: predicted class
    integer_t* n_correct,         // Output: count of correct predictions (atomic)
    integer_t* n_total            // Output: count of total OOB predictions (atomic)
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nsample) return;

    // Only process samples that have OOB predictions
    if (oob_counts[tid] == 0) {
        oob_predictions[tid] = -1;  // No prediction
        return;
    }

    // Find majority vote (using real_t for fractional casewise weights)
    integer_t best_class = 0;
    real_t max_votes = oob_votes[tid * nclass];
    for (integer_t c = 1; c < nclass; c++) {
        if (oob_votes[tid * nclass + c] > max_votes) {
            max_votes = oob_votes[tid * nclass + c];
            best_class = c;
        }
    }
    
    oob_predictions[tid] = best_class;
    
    // Count correct/total
    ::atomicAdd(n_total, 1);
    if (best_class == y_true[tid]) {
        ::atomicAdd(n_correct, 1);
    }
}

// ============================================================================
// Prediction kernel for sparse data
// ============================================================================

__global__ void gpu_predict_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    const integer_t* nodestatus,
    const integer_t* bestvar,
    const real_t* xbestsplit,
    const integer_t* treemap,
    const integer_t* nodeclass,
    integer_t nsample,
    integer_t mdim,
    integer_t ntree,
    integer_t maxnode,
    integer_t max_depth,
    integer_t* predictions,
    integer_t* error_code
) {
    integer_t sample = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample >= nsample) return;
    
    // Vote counts per class (max 32 classes)
    integer_t votes[32];
    for (integer_t c = 0; c < 32; c++) votes[c] = 0;
    
    // Traverse each tree
    for (integer_t t = 0; t < ntree; t++) {
        integer_t tree_offset = t * maxnode;
        
        integer_t kt = 0;  // Start at root
        
        for (integer_t depth = 0; depth < max_depth; depth++) {
            if (kt < 0 || kt >= maxnode) break;
            
            // Terminal node?
            if (nodestatus[tree_offset + kt] == -1) {
                integer_t pred_class = nodeclass[tree_offset + kt];
                if (pred_class >= 0 && pred_class < 32) {
                    votes[pred_class]++;
                }
                break;
            }
            
            // Get split info
            integer_t m = bestvar[tree_offset + kt];
            real_t threshold = xbestsplit[tree_offset + kt];
            
            if (m < 0 || m >= mdim) break;
            
            // Get feature value (sparse access)
            real_t val = X_sparse.get(sample, m);
            
            // Traverse
            integer_t treemap_offset = t * 2 * maxnode;
            if (val <= threshold) {
                kt = treemap[treemap_offset + kt * 2];      // Left
            } else {
                kt = treemap[treemap_offset + kt * 2 + 1];  // Right
            }
        }
    }
    
    // Find majority vote
    integer_t best_class = 0;
    integer_t max_votes = votes[0];
    for (integer_t c = 1; c < 32; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            best_class = c;
        }
    }
    
    predictions[sample] = best_class;
}

// ============================================================================
// Main training function
// ============================================================================

integer_t train_sparse_forest_gpu(
    const SparseMatrixCSR& h_X_sparse,
    const integer_t* h_y,
    const real_t* h_y_regression,   // Regression targets (nullptr for classification)
    const GpuSparseForestConfig& config,
    integer_t* h_nodestatus,
    integer_t* h_bestvar,
    real_t* h_xbestsplit,
    integer_t* h_treemap,
    integer_t* h_nodeclass,
    integer_t* h_nnode,
    real_t* h_q,           // OOB vote counts [nsample * nclass]
    integer_t* h_nout,     // OOB count per sample [nsample]
    real_t* h_oob_predictions_regression,  // [nsample] - Regression OOB predictions (sum, divide by nout)
    real_t* h_nodepred,    // [ntree * maxnode] - Terminal node predictions (mean y) for regression
    real_t* h_importance,
    real_t* h_qimp,        // Per-sample original correct weights (for localimp) [nsample]
    real_t* h_local_importance,
    real_t* h_prox_importance,
    real_t* h_overall_prox_importance,
    // Leaf assignments (for on-demand proximity computation)
    // This replaces full proximity matrix - compute proximity on-demand from leaves
    int16_t* h_leaf_assignments,  // [ntree * nsample] - terminal node per sample per tree
    // OOB tracking for Breiman's proximity (optional, nullptr to skip)
    integer_t* h_nin_all,  // [ntree * nsample] - bootstrap membership per tree (0 = OOB)
    // Progress callback (optional, nullptr to skip)
    std::function<void(integer_t, integer_t)> progress_callback,
    // Bootstrap weights (optional, nullptr = uniform bootstrap)
    const real_t* bootstrap_weights
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Upload sparse matrix
    CudaSparseMatrixCSR d_X_sparse;
    d_X_sparse.upload(h_X_sparse, stream);
    
    integer_t* d_error;
    cudaMalloc(&d_error, sizeof(integer_t));
    
    // ========================================================================
    // BATCH SIZE DETERMINATION - For memory-efficient training
    // ========================================================================
    integer_t batch_size = config.batch_size;
    if (batch_size <= 0) {
        // Auto-determine batch size based on available GPU memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        // Estimate memory per tree: ~(nsample + maxnode) * 20 bytes
        size_t mem_per_tree = static_cast<size_t>(config.nsample + config.maxnode) * 20;
        
        // Use 60% of free memory for batching, leave rest for other operations
        size_t available_for_batch = static_cast<size_t>(free_mem * 0.6);
        batch_size = static_cast<integer_t>(available_for_batch / mem_per_tree);
        
        // Clamp to reasonable range
        if (batch_size < 1) batch_size = 1;
        if (batch_size > config.ntree) batch_size = config.ntree;
        if (batch_size > 100) batch_size = 100;  // Cap at 100 trees per batch for progress updates
    }
    if (batch_size > config.ntree) batch_size = config.ntree;
    
    // Allocate device arrays - BATCH-SIZED (not ntree-sized)
    integer_t* d_y;
    integer_t* d_nin_batch;       // BATCH: [batch_size * nsample]
    real_t* d_win_batch;          // BATCH: [batch_size * nsample]
    integer_t* d_nodestatus_batch;   // BATCH: [batch_size * maxnode]
    integer_t* d_bestvar_batch;      // BATCH: [batch_size * maxnode]
    real_t* d_xbestsplit_batch;      // BATCH: [batch_size * maxnode]
    integer_t* d_treemap_batch;      // BATCH: [batch_size * 2 * maxnode]
    integer_t* d_nodeclass_batch;    // BATCH: [batch_size * maxnode]
    real_t* d_tnodewt_batch;         // BATCH: [batch_size * maxnode]
    integer_t* d_nnode_batch;        // BATCH: [batch_size]
    integer_t* d_jtr;
    integer_t* d_nodextr;
    
    // Cast to size_t to prevent integer overflow
    cudaMalloc(&d_y, static_cast<size_t>(config.nsample) * sizeof(integer_t));
    cudaMalloc(&d_nin_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.nsample) * sizeof(integer_t));
    cudaMalloc(&d_win_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.nsample) * sizeof(real_t));
    cudaMalloc(&d_nodestatus_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.maxnode) * sizeof(integer_t));
    cudaMalloc(&d_bestvar_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.maxnode) * sizeof(integer_t));
    cudaMalloc(&d_xbestsplit_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.maxnode) * sizeof(real_t));
    cudaMalloc(&d_treemap_batch, static_cast<size_t>(batch_size) * 2 * static_cast<size_t>(config.maxnode) * sizeof(integer_t));
    cudaMalloc(&d_nodeclass_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.maxnode) * sizeof(integer_t));
    cudaMalloc(&d_tnodewt_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.maxnode) * sizeof(real_t));
    cudaMalloc(&d_nnode_batch, static_cast<size_t>(batch_size) * sizeof(integer_t));
    cudaMalloc(&d_jtr, static_cast<size_t>(config.nsample) * sizeof(integer_t));
    cudaMalloc(&d_nodextr, static_cast<size_t>(config.nsample) * sizeof(integer_t));
    
    // Host buffer for batch nnode (to access tree sizes for post-processing)
    std::vector<integer_t> h_nnode_batch(batch_size);
    
    // Category handling for sparse data (default: all quantitative features)
    // For sparse data, we typically don't have categorical features
    const integer_t maxcat = 1;   // No categorical features
    const integer_t ncmax = 0;    // Not used for quantitative
    const integer_t ncsplit = 0;  // Not used for quantitative
    
    // Allocate category counts (all 1 = quantitative)
    integer_t* d_cat;
    cudaMalloc(&d_cat, config.mdim * sizeof(integer_t));
    std::vector<integer_t> h_cat(config.mdim, 1);  // All features are quantitative
    cudaMemcpy(d_cat, h_cat.data(), config.mdim * sizeof(integer_t), cudaMemcpyHostToDevice);
    
    // Allocate catgoleft (dummy - not used for quantitative features) - BATCH-SIZED
    integer_t* d_catgoleft_batch;
    cudaMalloc(&d_catgoleft_batch, static_cast<size_t>(batch_size) * static_cast<size_t>(config.maxnode) * static_cast<size_t>(maxcat) * sizeof(integer_t));
    cudaMemset(d_catgoleft_batch, 0, static_cast<size_t>(batch_size) * static_cast<size_t>(config.maxnode) * static_cast<size_t>(maxcat) * sizeof(integer_t));
    
    cudaMemcpy(d_y, h_y, config.nsample * sizeof(integer_t), cudaMemcpyHostToDevice);
    
    // Allocate regression targets if needed
    real_t* d_y_regression = nullptr;
    if (config.task_type == 1 && h_y_regression != nullptr) {
        cudaMalloc(&d_y_regression, config.nsample * sizeof(real_t));
        cudaMemcpy(d_y_regression, h_y_regression, config.nsample * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    
    // Allocate importance arrays
    real_t* d_avimp = nullptr;
    real_t* d_qimp = nullptr;   // Per-sample original correct weights (for localimp)
    real_t* d_qimpm = nullptr;
    if (config.compute_importance) {
        cudaMalloc(&d_avimp, config.mdim * sizeof(real_t));
        cudaMemset(d_avimp, 0, config.mdim * sizeof(real_t));
        if (config.compute_local_importance) {
            cudaMalloc(&d_qimp, config.nsample * sizeof(real_t));
            cudaMemset(d_qimp, 0, config.nsample * sizeof(real_t));
            cudaMalloc(&d_qimpm, static_cast<size_t>(config.nsample) * static_cast<size_t>(config.mdim) * sizeof(real_t));
            cudaMemset(d_qimpm, 0, static_cast<size_t>(config.nsample) * static_cast<size_t>(config.mdim) * sizeof(real_t));
        }
    }
    
    // NOTE: GPU sparse proximity matrix removed - use leaf_assignments for on-demand proximity
    // This is more memory efficient for Netflix-scale datasets
    
    // Allocate proximity importance
    real_t* d_prox_imp = nullptr;
    if (config.compute_proximity_importance) {
        cudaMalloc(&d_prox_imp, static_cast<size_t>(config.nsample) * static_cast<size_t>(config.mdim) * sizeof(real_t));
        cudaMemset(d_prox_imp, 0, static_cast<size_t>(config.nsample) * static_cast<size_t>(config.mdim) * sizeof(real_t));
    }
    
    // Allocate leaf assignments (for on-demand proximity computation)
    // Stores terminal node ID per sample per tree: shape [ntree, nsample]
    // Memory: ntree * nsample * 2 bytes = 2MB for 100 trees × 10K samples
    int16_t* d_leaf_assignments = nullptr;
    if (config.compute_leaf_assignments && h_leaf_assignments) {
        cudaMalloc(&d_leaf_assignments, static_cast<size_t>(config.ntree) * static_cast<size_t>(config.nsample) * sizeof(int16_t));
        cudaMemset(d_leaf_assignments, 0, static_cast<size_t>(config.ntree) * static_cast<size_t>(config.nsample) * sizeof(int16_t));
    }
    
    // ========================================================================
    // OOB VOTE ACCUMULATION - Critical for OOB error computation
    // ========================================================================
    real_t* d_oob_votes;      // [nsample * nclass] - vote counts per class per sample (real_t for fractional casewise weights)
    integer_t* d_oob_counts;  // [nsample] - number of OOB predictions per sample
    cudaMalloc(&d_oob_votes, static_cast<size_t>(config.nsample) * static_cast<size_t>(config.nclass) * sizeof(real_t));
    cudaMalloc(&d_oob_counts, static_cast<size_t>(config.nsample) * sizeof(integer_t));
    cudaMemset(d_oob_votes, 0, static_cast<size_t>(config.nsample) * static_cast<size_t>(config.nclass) * sizeof(real_t));
    cudaMemset(d_oob_counts, 0, static_cast<size_t>(config.nsample) * sizeof(integer_t));

    // Breiman class prior weights
    real_t* d_classwt = nullptr;
    if (!g_config.classwt.empty() && config.task_type == 0) {
        cudaMalloc(&d_classwt, g_config.classwt.size() * sizeof(real_t));
        cudaMemcpy(d_classwt, g_config.classwt.data(), g_config.classwt.size() * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    
    // Regression OOB predictions (accumulated weighted predictions)
    real_t* d_oob_predictions_regression = nullptr;
    real_t* d_oob_weight_sums_regression = nullptr;  // For casewise: track sum of tnodewt weights
    if (config.task_type == 1 && h_oob_predictions_regression != nullptr) {
        cudaMalloc(&d_oob_predictions_regression, config.nsample * sizeof(real_t));
        cudaMemset(d_oob_predictions_regression, 0, config.nsample * sizeof(real_t));
        if (config.use_casewise) {
            cudaMalloc(&d_oob_weight_sums_regression, config.nsample * sizeof(real_t));
            cudaMemset(d_oob_weight_sums_regression, 0, config.nsample * sizeof(real_t));
        }
    }
    
    // Initialize RNG - one state per tree to match GPU Dense architecture
    // This ensures identical random sequences for bootstrap and split selection
    curandState* d_rng_states;
    integer_t n_rng_states = config.ntree;  // Match GPU Dense: one per tree
    cudaMalloc(&d_rng_states, n_rng_states * sizeof(curandState));
    
    // Create tree-specific seeds on host (matches GPU Dense pattern)
    std::vector<integer_t> h_seeds(config.ntree);
    for (integer_t i = 0; i < config.ntree; ++i) {
        h_seeds[i] = config.seed + i;
    }
    integer_t* d_seeds;
    cudaMalloc(&d_seeds, config.ntree * sizeof(integer_t));
    cudaMemcpy(d_seeds, h_seeds.data(), config.ntree * sizeof(integer_t), cudaMemcpyHostToDevice);
    
    dim3 rng_block(256);
    dim3 rng_grid((n_rng_states + 255) / 256);
    gpu_init_rng_kernel<<<rng_grid, rng_block, 0, stream>>>(
        d_rng_states, d_seeds, n_rng_states  // Use per-tree seeds
    );
    cudaStreamSynchronize(stream);
    
    // ========================================================================
    // HISTOGRAM BINNING (if enabled) - O(256) split finding
    // Compute once before batch loop
    // ========================================================================
    SparseHistogramData hist_data;
    if (config.use_histogram) {
        gpu_compute_sparse_bin_edges(d_X_sparse, config.n_bins, hist_data, stream);
        gpu_bin_sparse_data(d_X_sparse, hist_data, stream);
        cudaStreamSynchronize(stream);
    }
    
    // Weight sums for casewise regression (track across all trees)
    std::vector<real_t> h_weight_sums;
    if (config.use_casewise && config.task_type == 1) {
        h_weight_sums.resize(config.nsample, 0.0f);
    }
    
    // Compute cumulative bootstrap weights on GPU (if provided)
    real_t* d_sample_weights_cum = nullptr;
    real_t total_sample_weight = 0.0f;
    if (bootstrap_weights != nullptr) {
        std::vector<real_t> cum(config.nsample);
        cum[0] = bootstrap_weights[0];
        for (integer_t i = 1; i < config.nsample; ++i) {
            cum[i] = cum[i - 1] + bootstrap_weights[i];
        }
        total_sample_weight = cum[config.nsample - 1];
        cudaMalloc(&d_sample_weights_cum, config.nsample * sizeof(real_t));
        cudaMemcpyAsync(d_sample_weights_cum, cum.data(), config.nsample * sizeof(real_t),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    
    // ========================================================================
    // BATCHED TRAINING LOOP - Process batch_size trees at a time
    // ========================================================================
    integer_t trees_completed = 0;
    
    for (integer_t batch_start = 0; batch_start < config.ntree; batch_start += batch_size) {
        integer_t batch_end = std::min(batch_start + batch_size, config.ntree);
        integer_t batch_trees = batch_end - batch_start;
        
        // Step 1: Bootstrap batch trees in parallel
        gpu_bootstrap_batch_kernel<<<batch_trees, 256, 0, stream>>>(
            config.nsample, batch_trees, d_rng_states + batch_start, d_nin_batch, d_win_batch,
            d_sample_weights_cum, total_sample_weight
        );
        cudaStreamSynchronize(stream);
        
        // Step 2: Grow batch trees in parallel
    integer_t grow_err;
    
    if (config.use_histogram) {
        grow_err = gpu_growtree_sparse_parallel_batch_v2(
            d_X_sparse,
            d_y,
            d_y_regression,
                d_nin_batch,
                d_win_batch,
            config.nsample,
            config.mdim,
            config.nclass,
            config.mtry,
            config.maxnode,
            config.min_node_size,
                batch_trees,  // Use batch_trees, not config.ntree
            config.task_type,
                d_rng_states + batch_start,  // RNG states for this batch
                false,
                nullptr,
                true,
                &hist_data,
                d_nodestatus_batch,
                d_bestvar_batch,
                d_xbestsplit_batch,
                d_treemap_batch,
                d_nodeclass_batch,
                d_tnodewt_batch,
                d_nnode_batch,
            stream
        );
    } else {
        grow_err = gpu_growtree_sparse_parallel_batch(
            d_X_sparse,
            d_y,
            d_y_regression,
                d_nin_batch,
                d_win_batch,
            config.nsample,
            config.mdim,
            config.nclass,
            config.mtry,
            config.maxnode,
            config.min_node_size,
                batch_trees,  // Use batch_trees, not config.ntree
            config.task_type,
                d_rng_states + batch_start,  // RNG states for this batch
                d_nodestatus_batch,
                d_bestvar_batch,
                d_xbestsplit_batch,
                d_treemap_batch,
                d_nodeclass_batch,
                d_tnodewt_batch,
                d_nnode_batch,
            stream
        );
    }
    
    if (grow_err != CUDA_OK) {
        // Cleanup and return on error
        d_X_sparse.free();
        cudaFree(d_y);
            cudaFree(d_nin_batch);
            cudaFree(d_win_batch);
            cudaFree(d_nodestatus_batch);
            cudaFree(d_bestvar_batch);
            cudaFree(d_xbestsplit_batch);
            cudaFree(d_treemap_batch);
            cudaFree(d_nodeclass_batch);
            cudaFree(d_tnodewt_batch);
        cudaFree(d_cat);
            cudaFree(d_catgoleft_batch);
            cudaFree(d_nnode_batch);
        cudaFree(d_jtr);
        cudaFree(d_nodextr);
        cudaFree(d_rng_states);
        cudaFree(d_seeds);
        if (d_sample_weights_cum) cudaFree(d_sample_weights_cum);
        cudaFree(d_oob_votes);
        cudaFree(d_oob_counts);
        if (d_classwt) cudaFree(d_classwt);
        if (d_oob_predictions_regression) cudaFree(d_oob_predictions_regression);
        cudaFree(d_error);
        if (d_avimp) cudaFree(d_avimp);
        if (d_qimpm) cudaFree(d_qimpm);
        if (d_prox_imp) cudaFree(d_prox_imp);
        if (d_leaf_assignments) cudaFree(d_leaf_assignments);
        if (hist_data.allocated) hist_data.free();
        cudaStreamDestroy(stream);
        return grow_err;
    }
    
        // Copy batch nnode to host for tree sizes
        cudaMemcpy(h_nnode_batch.data(), d_nnode_batch, 
                   batch_trees * sizeof(integer_t), cudaMemcpyDeviceToHost);
        
        // OPTIMIZED: Copy entire batch at once, then scatter to correct host offsets
        // This reduces batch_size * 5 memcpy calls to just 5 memcpy calls per batch
        size_t batch_node_size = static_cast<size_t>(batch_trees) * static_cast<size_t>(config.maxnode);
        size_t batch_treemap_size = static_cast<size_t>(batch_trees) * 2 * static_cast<size_t>(config.maxnode);
        
        // Temporary host buffers for batch data
        std::vector<integer_t> h_nodestatus_batch_temp(batch_node_size);
        std::vector<integer_t> h_bestvar_batch_temp(batch_node_size);
        std::vector<real_t> h_xbestsplit_batch_temp(batch_node_size);
        std::vector<integer_t> h_nodeclass_batch_temp(batch_node_size);
        std::vector<integer_t> h_treemap_batch_temp(batch_treemap_size);
        
        // Single batch copies from GPU to host
        cudaMemcpy(h_nodestatus_batch_temp.data(), d_nodestatus_batch, batch_node_size * sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bestvar_batch_temp.data(), d_bestvar_batch, batch_node_size * sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_xbestsplit_batch_temp.data(), d_xbestsplit_batch, batch_node_size * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nodeclass_batch_temp.data(), d_nodeclass_batch, batch_node_size * sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_treemap_batch_temp.data(), d_treemap_batch, batch_treemap_size * sizeof(integer_t), cudaMemcpyDeviceToHost);
        
        // Scatter batch data to correct host offsets (CPU-side memcpy is fast)
        for (integer_t b = 0; b < batch_trees; b++) {
            integer_t global_tree = batch_start + b;
            size_t host_tree_offset = static_cast<size_t>(global_tree) * static_cast<size_t>(config.maxnode);
            size_t batch_tree_offset = static_cast<size_t>(b) * static_cast<size_t>(config.maxnode);
            
            h_nnode[global_tree] = h_nnode_batch[b];
            
            std::memcpy(h_nodestatus + host_tree_offset, h_nodestatus_batch_temp.data() + batch_tree_offset,
                       config.maxnode * sizeof(integer_t));
            std::memcpy(h_bestvar + host_tree_offset, h_bestvar_batch_temp.data() + batch_tree_offset,
                       config.maxnode * sizeof(integer_t));
            std::memcpy(h_xbestsplit + host_tree_offset, h_xbestsplit_batch_temp.data() + batch_tree_offset,
                       config.maxnode * sizeof(real_t));
            std::memcpy(h_nodeclass + host_tree_offset, h_nodeclass_batch_temp.data() + batch_tree_offset,
                       config.maxnode * sizeof(integer_t));
            
            size_t host_treemap_offset = static_cast<size_t>(global_tree) * 2 * static_cast<size_t>(config.maxnode);
            size_t batch_treemap_offset = static_cast<size_t>(b) * 2 * static_cast<size_t>(config.maxnode);
            std::memcpy(h_treemap + host_treemap_offset, h_treemap_batch_temp.data() + batch_treemap_offset,
                       2 * config.maxnode * sizeof(integer_t));
    }
    
        // Step 3: Post-processing loop for OOB, importance, proximity (for each tree in batch)
        for (integer_t b = 0; b < batch_trees; b++) {
            integer_t t = batch_start + b;  // Global tree index
            integer_t nnode = h_nnode_batch[b];
            
            // Point to this tree's data in batch arrays
            size_t batch_tree_offset = static_cast<size_t>(b) * static_cast<size_t>(config.maxnode);
            size_t batch_sample_offset = static_cast<size_t>(b) * static_cast<size_t>(config.nsample);
        
            integer_t* d_nodestatus = d_nodestatus_batch + batch_tree_offset;
            integer_t* d_bestvar = d_bestvar_batch + batch_tree_offset;
            real_t* d_xbestsplit = d_xbestsplit_batch + batch_tree_offset;
            integer_t* d_treemap = d_treemap_batch + static_cast<size_t>(b) * 2 * static_cast<size_t>(config.maxnode);
            integer_t* d_nodeclass = d_nodeclass_batch + batch_tree_offset;
            real_t* d_tnodewt = d_tnodewt_batch + batch_tree_offset;
            integer_t* d_nin = d_nin_batch + batch_sample_offset;
            real_t* d_win = d_win_batch + batch_sample_offset;
        
        // Test OOB samples - find terminal nodes and accumulate votes
        // For non-casewise: unweighted vote accumulation (weight=1) - done here
        // For casewise: we need to re-accumulate with tnodewt weights afterwards
        gpu_testreebag_sparse(
            d_X_sparse,
            d_treemap,
            d_nodestatus,
            d_xbestsplit,
            d_bestvar,
            d_nodeclass,
            d_nin,
            nnode,
            config.nsample,
            config.nclass,
            config.max_depth,
            d_nodextr,
            d_oob_votes,
            d_oob_counts,
            stream,
            d_classwt
        );
        
        // REGRESSION OOB ACCUMULATION - same pattern as classification vote accumulation
        // For regression: accumulate weighted predictions (weight * nodepred)
        if (config.task_type == 1 && d_oob_predictions_regression != nullptr && d_y_regression != nullptr) {
            // Copy data to host for accumulation (same pattern as classification)
            std::vector<integer_t> h_nodextr_temp(config.nsample);
            std::vector<integer_t> h_nin_temp(config.nsample);
            std::vector<real_t> h_tnodewt_temp(nnode);
            cudaMemcpy(h_nodextr_temp.data(), d_nodextr, config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_nin_temp.data(), d_nin, config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_tnodewt_temp.data(), d_tnodewt, nnode * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            // Compute nodepred (mean y) and tnodewt (mean weight) for each terminal node
            // Following the same pattern as GPU Dense and CPU
            std::vector<real_t> nodepred_tree(nnode, 0.0f);
            std::vector<real_t> tnodewt_tree(nnode, 0.0f);
            std::vector<real_t> node_sum_y(nnode, 0.0f);
            std::vector<real_t> node_sum_weight(nnode, 0.0f);
            std::vector<real_t> node_count(nnode, 0.0f);
            
            // Copy y_regression to host for this computation
            std::vector<real_t> h_y_regression_temp(config.nsample);
            cudaMemcpy(h_y_regression_temp.data(), d_y_regression, config.nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            // First pass: accumulate for in-bag samples
            for (integer_t n = 0; n < config.nsample; ++n) {
                if (h_nin_temp[n] > 0) {  // In-bag sample
                    integer_t kt = h_nodextr_temp[n];
                    if (kt >= 0 && kt < nnode) {
                        real_t y_val = h_y_regression_temp[n];
                        real_t bootstrap_freq = static_cast<real_t>(h_nin_temp[n]);
                        if (config.use_casewise) {
                            node_sum_y[kt] += y_val * bootstrap_freq;
                            node_sum_weight[kt] += bootstrap_freq;
                            node_count[kt] += 1.0f;
                        } else {
                            node_sum_y[kt] += y_val;
                            node_sum_weight[kt] += 1.0f;
                            node_count[kt] += 1.0f;
                        }
                    }
                }
            }
            
            // Compute nodepred and tnodewt
            for (integer_t node = 0; node < nnode; ++node) {
                if (node_sum_weight[node] > 0.0f && node_count[node] > 0.0f) {
                    nodepred_tree[node] = node_sum_y[node] / node_sum_weight[node];
                    tnodewt_tree[node] = node_sum_weight[node] / node_count[node];
                }
            }
            
            // Store nodepred for this tree (needed for predict() on new data)
            if (h_nodepred != nullptr) {
                size_t tree_offset_pred = static_cast<size_t>(t) * static_cast<size_t>(config.maxnode);
                for (integer_t node = 0; node < nnode && node < config.maxnode; ++node) {
                    h_nodepred[tree_offset_pred + node] = nodepred_tree[node];
                }
            }
            
            // Copy current OOB predictions to host, accumulate, copy back
            std::vector<real_t> h_oob_pred_temp(config.nsample);
            cudaMemcpy(h_oob_pred_temp.data(), d_oob_predictions_regression, 
                       config.nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            // Also need to copy and update OOB counts (nout_)
            std::vector<integer_t> h_oob_counts_temp(config.nsample);
            cudaMemcpy(h_oob_counts_temp.data(), d_oob_counts, 
                       config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
            
            // Second pass: accumulate OOB predictions
            // For casewise: weight each tree's vote by tnodewt (Breiman's approach)
            // For non-casewise: simple averaging (weight = 1)
            
            for (integer_t n = 0; n < config.nsample; ++n) {
                if (h_nin_temp[n] == 0) {  // OOB sample
                    integer_t kt = h_nodextr_temp[n];
                    if (kt >= 0 && kt < nnode) {
                        real_t prediction = nodepred_tree[kt];
                        
                        if (config.use_casewise) {
                            // Casewise: weight by tnodewt
                            // tnodewt = mean bootstrap freq in terminal node (higher = more reliable)
                            real_t weight = tnodewt_tree[kt];
                            if (weight <= 0.0f) weight = 1.0f;  // Fallback to 1.0 if invalid
                            h_oob_pred_temp[n] += prediction * weight;
                            h_weight_sums[n] += weight;
                        } else {
                            // Non-casewise: simple averaging (weight = 1)
                            h_oob_pred_temp[n] += prediction;
                        }
                        // NOTE: Don't increment h_oob_counts_temp here!
                        // gpu_testreebag_sparse already increments d_oob_counts for OOB samples
                    }
                }
            }
            
            // For casewise: copy weight sums to device after last tree
            if (config.use_casewise && t == config.ntree - 1 && d_oob_weight_sums_regression != nullptr) {
                cudaMemcpy(d_oob_weight_sums_regression, h_weight_sums.data(),
                           config.nsample * sizeof(real_t), cudaMemcpyHostToDevice);
            }
            
            cudaMemcpy(d_oob_predictions_regression, h_oob_pred_temp.data(),
                       config.nsample * sizeof(real_t), cudaMemcpyHostToDevice);
            // NOTE: Don't copy h_oob_counts_temp back - gpu_testreebag_sparse manages d_oob_counts
        }
        
        // Compute predictions for this tree
        dim3 jtr_block(256);
        dim3 jtr_grid((config.nsample + 255) / 256);
        gpu_set_jtr_from_nodextr_kernel<<<jtr_grid, jtr_block, 0, stream>>>(
            d_nodextr, d_nodeclass, d_nin,
            config.nsample, nnode, d_jtr
        );
        cudaStreamSynchronize(stream);
        
        // Store leaf assignments for this tree (for on-demand proximity computation)
        // d_nodextr[i] = terminal node for sample i in tree t
        // Copy to d_leaf_assignments[t * nsample : (t+1) * nsample] as int16_t
        if (d_leaf_assignments) {
            // Simple kernel to convert integer_t to int16_t and copy
            // For now, use thrust or a copy kernel. Use simple device-to-device copy with conversion.
            std::vector<integer_t> h_nodextr_temp(config.nsample);
            cudaMemcpy(h_nodextr_temp.data(), d_nodextr, config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
            std::vector<int16_t> h_leaf_temp(config.nsample);
            for (integer_t i = 0; i < config.nsample; ++i) {
                h_leaf_temp[i] = static_cast<int16_t>(h_nodextr_temp[i]);
            }
            cudaMemcpy(d_leaf_assignments + static_cast<size_t>(t) * static_cast<size_t>(config.nsample), h_leaf_temp.data(), 
                       static_cast<size_t>(config.nsample) * sizeof(int16_t), cudaMemcpyHostToDevice);
        }
        
        // Compute tnodewt for casewise mode
        if (config.use_casewise && nnode > 0) {
            dim3 tnodewt_block(256);
            dim3 tnodewt_grid((nnode + 255) / 256);
            gpu_compute_tnodewt_sparse_kernel<<<tnodewt_grid, tnodewt_block, 0, stream>>>(
                d_X_sparse,
                config.nsample,
                config.mdim,
                nnode,
                d_treemap,
                d_nodestatus,
                d_xbestsplit,
                d_bestvar,
                d_y,
                d_win,
                d_nin,
                config.nclass,
                0,
                d_tnodewt
            );
            cudaStreamSynchronize(stream);
            
            // CASEWISE CLASSIFICATION/UNSUPERVISED: Adjust OOB votes with tnodewt weights
            // gpu_testreebag_sparse already added +1 vote per OOB sample
            // For casewise, we need to adjust: subtract the +1 and add +tnodewt instead
            // OPTIMIZED: Now uses GPU kernel instead of CPU round-trip (6 memcpy -> 0 memcpy)
            if (config.task_type == 0 || config.task_type == 2) {  // Classification or Unsupervised
                gpu_adjust_casewise_votes(
                    d_nodextr,
                    d_nodeclass,
                    d_tnodewt,
                    d_nin,
                    config.nsample,
                    config.nclass,
                    nnode,
                    d_oob_votes,
                    d_oob_counts,
                    stream,
                    d_classwt
                );
            }
        }
        
        // Variable importance
        if (config.compute_importance) {
            real_t* d_avimp_temp;
            cudaMalloc(&d_avimp_temp, config.mdim * sizeof(real_t));
            cudaMemsetAsync(d_avimp_temp, 0, config.mdim * sizeof(real_t), stream);
            
            if (config.task_type == 1 && d_y_regression != nullptr) {
                // REGRESSION: Use MSE-based importance
                // Compute nodepred (mean y per terminal node) - same pattern as OOB accumulation
                std::vector<integer_t> h_nodextr_temp(config.nsample);
                std::vector<integer_t> h_nin_temp(config.nsample);
                std::vector<real_t> h_y_regression_temp(config.nsample);
                cudaMemcpy(h_nodextr_temp.data(), d_nodextr, config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_nin_temp.data(), d_nin, config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_y_regression_temp.data(), d_y_regression, config.nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
                
                // Compute nodepred and tnodewt (same pattern as OOB accumulation)
                std::vector<real_t> nodepred_tree(nnode, 0.0f);
                std::vector<real_t> tnodewt_tree(nnode, 0.0f);
                std::vector<real_t> node_sum_y(nnode, 0.0f);
                std::vector<real_t> node_sum_weight(nnode, 0.0f);
                std::vector<real_t> node_count(nnode, 0.0f);
                
                for (integer_t n = 0; n < config.nsample; ++n) {
                    if (h_nin_temp[n] == 0) continue;  // Skip OOB
                    integer_t kt = h_nodextr_temp[n];
                    if (kt >= 0 && kt < nnode) {
                        real_t y_val = h_y_regression_temp[n];
                        real_t bootstrap_freq = static_cast<real_t>(h_nin_temp[n]);
                        if (config.use_casewise) {
                            node_sum_y[kt] += y_val * bootstrap_freq;
                            node_sum_weight[kt] += bootstrap_freq;
                        } else {
                            node_sum_y[kt] += y_val;
                            node_sum_weight[kt] += 1.0f;
                        }
                        node_count[kt] += 1.0f;
                    }
                }
                
                for (integer_t node = 0; node < nnode; ++node) {
                    if (node_sum_weight[node] > 0.0f && node_count[node] > 0.0f) {
                        nodepred_tree[node] = node_sum_y[node] / node_sum_weight[node];
                        tnodewt_tree[node] = node_sum_weight[node] / node_count[node];
                    }
                }
                
                // Build y_pred from nodepred (not tnodewt)
                std::vector<real_t> h_y_pred(config.nsample, 0.0f);
                for (integer_t n = 0; n < config.nsample; ++n) {
                    integer_t node_idx = h_nodextr_temp[n];
                    if (node_idx >= 0 && node_idx < nnode) {
                        h_y_pred[n] = nodepred_tree[node_idx];
                    }
                }
                
                real_t* d_y_pred;
                real_t* d_nodepred;
                real_t* d_tnodewt_tree;
                cudaMalloc(&d_y_pred, config.nsample * sizeof(real_t));
                cudaMalloc(&d_nodepred, nnode * sizeof(real_t));
                cudaMalloc(&d_tnodewt_tree, nnode * sizeof(real_t));
                cudaMemcpy(d_y_pred, h_y_pred.data(), config.nsample * sizeof(real_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_nodepred, nodepred_tree.data(), nnode * sizeof(real_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_tnodewt_tree, tnodewt_tree.data(), nnode * sizeof(real_t), cudaMemcpyHostToDevice);
                
                gpu_varimp_sparse_regression(
                    d_X_sparse, d_y_pred, d_y_regression, d_nin, d_nodextr,
                    d_treemap, d_nodestatus, d_xbestsplit,
                    d_bestvar, d_nodepred, d_tnodewt_tree,
                    config.nsample, config.mdim, nnode, config.max_depth,
                    config.use_casewise,
                    config.compute_local_importance ? 1 : 0,
                    d_avimp_temp, d_qimp, d_qimpm,
                    d_rng_states, stream
                );
                
                cudaFree(d_y_pred);
                cudaFree(d_nodepred);
                cudaFree(d_tnodewt_tree);
            } else {
                // CLASSIFICATION: Use accuracy-based importance
                gpu_varimp_sparse(
                    d_X_sparse, d_y, d_jtr, d_nin, d_nodextr,
                    d_treemap, d_nodestatus, d_xbestsplit,
                    d_bestvar, d_nodeclass, d_tnodewt,
                    config.nsample, config.mdim, nnode, config.max_depth,
                    config.use_casewise,
                    config.compute_local_importance ? 1 : 0,
                    d_avimp_temp, d_qimp, d_qimpm,
                    d_rng_states, stream
                );
            }
            cudaStreamSynchronize(stream);
            
            // Accumulate temp into main array
            std::vector<real_t> h_temp(config.mdim);
            cudaMemcpy(h_temp.data(), d_avimp_temp, config.mdim * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            std::vector<real_t> h_avimp(config.mdim);
            cudaMemcpy(h_avimp.data(), d_avimp, config.mdim * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            for (integer_t i = 0; i < config.mdim; ++i) {
                h_avimp[i] += h_temp[i];
            }
            
            cudaMemcpy(d_avimp, h_avimp.data(), config.mdim * sizeof(real_t), cudaMemcpyHostToDevice);
            
            cudaFree(d_avimp_temp);
        }
        
        // NOTE: GPU sparse proximity matrix computation removed.
        // Use leaf_assignments for on-demand proximity computation instead.
        // This saves memory and is more scalable for Netflix-sized datasets.
        
        // Proximity importance - process this tree (accumulates into d_prox_imp)
        // Matches Breiman-Cutler methodology for local variable importance:
        // - Only accumulate for samples where OOB prediction was CORRECT
        // - Normalize per-tree by (1.0 / noob)
        if (config.compute_proximity_importance) {
            // Count OOB samples for this tree (for per-tree normalization)
            integer_t noob = gpu_count_oob_samples(d_nin, config.nsample, stream);
            
            // Determine if this is regression (skip correctness check)
            bool is_regression = (config.task_type == 1);  // 1 = regression
            
            if (noob > 0) {
                gpu_compute_proximity_importance_sparse(
                    d_X_sparse, d_nodextr, d_nin,
                    d_y,          // True classes [nsample]
                    d_nodeclass,  // Node classes [maxnode] - for computing jtr on-the-fly
                    d_treemap, d_nodestatus, d_xbestsplit, d_bestvar,
                    config.nsample, config.mdim, nnode, config.max_depth,
                    t,            // For deterministic donor selection (matches GPU Dense)
                    noob,         // OOB count for per-tree normalization
                    d_prox_imp,
                    is_regression,
                    stream
                );
            }
        }
        
            // Copy nin to host array at correct offset (needed for oob_trees_ population)
            if (h_nin_all) {
                size_t host_sample_offset = static_cast<size_t>(t) * static_cast<size_t>(config.nsample);
                cudaMemcpy(h_nin_all + host_sample_offset, d_nin, 
                           config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
            }
            
            trees_completed++;
        }  // end for b (batch tree loop)
        
        // Progress callback after each batch
        if (progress_callback) {
            progress_callback(trees_completed, config.ntree);
        }
    }  // end for batch_start (batch loop)
    
    // ========================================================================
    // Compute OOB Error from accumulated votes
    // ========================================================================
    {
        integer_t* d_oob_predictions;
        integer_t* d_n_correct;
        integer_t* d_n_total;
        cudaMalloc(&d_oob_predictions, config.nsample * sizeof(integer_t));
        cudaMalloc(&d_n_correct, sizeof(integer_t));
        cudaMalloc(&d_n_total, sizeof(integer_t));
        cudaMemset(d_n_correct, 0, sizeof(integer_t));
        cudaMemset(d_n_total, 0, sizeof(integer_t));
        
        dim3 oob_block(256);
        dim3 oob_grid((config.nsample + 255) / 256);
        gpu_compute_oob_predictions_kernel<<<oob_grid, oob_block, 0, stream>>>(
            d_oob_votes, d_oob_counts, d_y,
            config.nsample, config.nclass,
            d_oob_predictions, d_n_correct, d_n_total
        );
        cudaStreamSynchronize(stream);
        
        // Copy results to host for OOB error calculation
        integer_t h_n_correct, h_n_total;
        cudaMemcpy(&h_n_correct, d_n_correct, sizeof(integer_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_n_total, d_n_total, sizeof(integer_t), cudaMemcpyDeviceToHost);
        
        // Store OOB error rate in the importance array if available
        // (The caller can compute it from h_n_correct / h_n_total)
        // For now, we print diagnostic info
        if (h_n_total > 0) {
            real_t oob_error = 1.0f - static_cast<real_t>(h_n_correct) / static_cast<real_t>(h_n_total);
            // OOB error is returned via the importance mechanism or stored elsewhere
            // For debugging:
            // std::cout << "[GPU SPARSE] OOB: " << h_n_correct << "/" << h_n_total 
            //           << " correct, error=" << oob_error << std::endl;
        }
        
        cudaFree(d_oob_predictions);
        cudaFree(d_n_correct);
        cudaFree(d_n_total);
    }
    
    // ========================================================================
    // Finalize
    // ========================================================================
    
    // Normalize importance
    if (config.compute_importance && d_avimp) {
        cudaMemcpy(h_importance, d_avimp, config.mdim * sizeof(real_t), cudaMemcpyDeviceToHost);
        // Note: No normalization by ntree - Breiman formula accumulates (right-rightimp)/noob across trees
        
        if (config.compute_local_importance && d_qimpm) {
            cudaMemcpy(h_local_importance, d_qimpm, 
                       static_cast<size_t>(config.nsample) * static_cast<size_t>(config.mdim) * sizeof(real_t), cudaMemcpyDeviceToHost);
            // Note: No normalization by ntree for local importance either
            
            // Also copy qimp (per-sample original correct weights) - needed by localimp()
            if (d_qimp && h_qimp) {
                cudaMemcpy(h_qimp, d_qimp, config.nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
            }
        }
    }
    
    // NOTE: GPU sparse proximity matrix removed - use leaf_assignments instead
    // Proximity is computed on-demand from leaf_assignments
    
    // Finalize proximity importance
    // Per-tree normalization (1/noob) is already done during accumulation
    // Just compute overall (mean across samples) for summary statistics
    if (config.compute_proximity_importance && d_prox_imp) {
        real_t* d_overall;
        cudaMalloc(&d_overall, config.mdim * sizeof(real_t));
        
        gpu_finalize_proximity_importance_sparse(
            d_prox_imp, config.nsample, config.mdim,
            d_overall, stream
        );
        
        cudaMemcpy(h_prox_importance, d_prox_imp,
                   static_cast<size_t>(config.nsample) * static_cast<size_t>(config.mdim) * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_overall_prox_importance, d_overall,
                   config.mdim * sizeof(real_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_overall);
    }
    
    // ========================================================================
    // Copy OOB votes back to host (needed for compute_oob_predictions)
    // ========================================================================
    if (h_q != nullptr && h_nout != nullptr) {
        cudaMemcpy(h_q, d_oob_votes,
                   config.nsample * config.nclass * sizeof(real_t), cudaMemcpyDeviceToHost);

        cudaMemcpy(h_nout, d_oob_counts,
                   config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
    }
    
    // ========================================================================
    // Copy regression OOB predictions back to host (NOT normalized here)
    // NOTE: Normalization by nout happens in compute_regression_oob_error()
    // We store raw accumulated sums here
    // ========================================================================
    if (d_oob_predictions_regression != nullptr && h_oob_predictions_regression != nullptr) {
        if (config.use_casewise && d_oob_weight_sums_regression != nullptr) {
            // Casewise: normalize by weight sums NOW (since compute_regression_oob_error
            // will divide by nout, we need to adjust)
            // Store: sum(pred * weight) / sum(weight) * nout
            // This way when compute_regression_oob_error divides by nout, we get the correct weighted mean
            std::vector<real_t> h_oob_pred_raw(config.nsample);
            cudaMemcpy(h_oob_pred_raw.data(), d_oob_predictions_regression,
                       config.nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            std::vector<real_t> h_weight_sums(config.nsample);
            cudaMemcpy(h_weight_sums.data(), d_oob_weight_sums_regression,
                       config.nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
            
            std::vector<integer_t> h_counts(config.nsample);
            cudaMemcpy(h_counts.data(), d_oob_counts,
                       config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
            
            for (integer_t n = 0; n < config.nsample; ++n) {
                if (h_weight_sums[n] > 0.0f && h_counts[n] > 0) {
                    // Adjust so that division by nout in compute_regression_oob_error gives weighted mean
                    // weighted_mean = sum(pred*w) / sum(w)
                    // We want: result / nout = weighted_mean
                    // So: result = weighted_mean * nout = (sum(pred*w) / sum(w)) * nout
                    real_t weighted_mean = h_oob_pred_raw[n] / h_weight_sums[n];
                    h_oob_predictions_regression[n] = weighted_mean * static_cast<real_t>(h_counts[n]);
                } else {
                    h_oob_predictions_regression[n] = 0.0f;
                }
            }
        } else {
            // Non-casewise: just copy raw sums - compute_regression_oob_error will normalize by nout
            cudaMemcpy(h_oob_predictions_regression, d_oob_predictions_regression,
                       config.nsample * sizeof(real_t), cudaMemcpyDeviceToHost);
        }
        
        // Copy nout for external use
        if (h_nout != nullptr) {
            cudaMemcpy(h_nout, d_oob_counts,
                       config.nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
        }
    }
    
    // ========================================================================
    // Copy leaf assignments back to host (for on-demand proximity)
    // ========================================================================
    if (d_leaf_assignments && h_leaf_assignments) {
        cudaMemcpy(h_leaf_assignments, d_leaf_assignments,
                   static_cast<size_t>(config.ntree) * static_cast<size_t>(config.nsample) * sizeof(int16_t), cudaMemcpyDeviceToHost);
    }
    
    // NOTE: h_nin_all is now copied incrementally during batched training loop
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    d_X_sparse.free();
    
    cudaFree(d_y);
    // Batch arrays
    cudaFree(d_nin_batch);
    cudaFree(d_win_batch);
    cudaFree(d_nodestatus_batch);
    cudaFree(d_bestvar_batch);
    cudaFree(d_xbestsplit_batch);
    cudaFree(d_treemap_batch);
    cudaFree(d_nodeclass_batch);
    cudaFree(d_tnodewt_batch);
    cudaFree(d_cat);
    cudaFree(d_catgoleft_batch);
    cudaFree(d_nnode_batch);
    cudaFree(d_oob_votes);
    cudaFree(d_oob_counts);
    if (d_classwt) cudaFree(d_classwt);
    if (d_oob_predictions_regression) cudaFree(d_oob_predictions_regression);
    if (d_oob_weight_sums_regression) cudaFree(d_oob_weight_sums_regression);
    cudaFree(d_jtr);
    cudaFree(d_nodextr);
    cudaFree(d_rng_states);
    cudaFree(d_seeds);
    if (d_sample_weights_cum) cudaFree(d_sample_weights_cum);
    cudaFree(d_error);
    
    if (d_avimp) cudaFree(d_avimp);
    if (d_qimp) cudaFree(d_qimp);
    if (d_qimpm) cudaFree(d_qimpm);
    if (d_prox_imp) cudaFree(d_prox_imp);
    if (d_leaf_assignments) cudaFree(d_leaf_assignments);
    if (d_y_regression) cudaFree(d_y_regression);
    if (hist_data.allocated) hist_data.free();
    
    cudaStreamDestroy(stream);
    
    return CUDA_OK;
}

// ============================================================================
// Prediction function
// ============================================================================

integer_t predict_sparse_gpu(
    const CudaSparseMatrixCSR& d_X_sparse,
    const integer_t* d_nodestatus,
    const integer_t* d_bestvar,
    const real_t* d_xbestsplit,
    const integer_t* d_treemap,
    const integer_t* d_nodeclass,
    integer_t nsample_new,
    integer_t ntree,
    integer_t maxnode,
    integer_t max_depth,
    integer_t* d_predictions,
    cudaStream_t stream
) {
    integer_t* d_error;
    cudaMalloc(&d_error, sizeof(integer_t));
    cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
    
    dim3 block(256);
    dim3 grid((nsample_new + 255) / 256);
    
    gpu_predict_sparse_kernel<<<grid, block, 0, stream>>>(
        d_X_sparse,
        d_nodestatus,
        d_bestvar,
        d_xbestsplit,
        d_treemap,
        d_nodeclass,
        nsample_new,
        d_X_sparse.ncols,  // mdim
        ntree,
        maxnode,
        max_depth,
        d_predictions,
        d_error
    );
    
    cudaStreamSynchronize(stream);
    
    integer_t h_error;
    cudaMemcpy(&h_error, d_error, sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_error);
    return h_error;
}

} // namespace cuda
} // namespace rf

