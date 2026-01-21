/**
 * @file rf_varimp_sparse.cu
 * @brief GPU sparse variable importance implementation
 * 
 * Matches CPU cpu_varimp_sparse exactly.
 * Key difference: uses CudaSparseMatrixCSR.get(sample, feature) for data access.
 * 
 * IMPORTANT: Row-major access pattern (matches CPU):
 *   X_sparse.get(sample, feature) NOT X_sparse.get(feature, sample)
 */

#include "rf_varimp_sparse.cuh"
#include "rf_sparse_cuda.cuh"
#include "rf_error_codes.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <set>

namespace rf {
namespace cuda {

// ============================================================================
// Permutation kernel
// ============================================================================

/**
 * GPU kernel: Fisher-Yates shuffle for OOB indices
 * Creates permuted copy of joob in pjoob
 */
__global__ void gpu_permute_oob_kernel(
    const integer_t* joob,
    integer_t* pjoob,
    integer_t noob,
    curandState* rng_states,
    integer_t* error_code
) {
    // Single thread performs permutation (sequential Fisher-Yates)
    // This matches CPU exactly - parallelizing shuffle is complex
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;
    
    // Copy joob to pjoob
    for (integer_t i = 0; i < noob; i++) {
        pjoob[i] = joob[i];
    }
    
    // Fisher-Yates shuffle
    for (integer_t j = noob - 1; j > 0; j--) {
        float rnd = curand_uniform(&rng_states[0]);
        integer_t k = static_cast<integer_t>((j + 1) * rnd);
        if (k > j) k = j;
        
        // Swap
        integer_t tmp = pjoob[j];
        pjoob[j] = pjoob[k];
        pjoob[k] = tmp;
    }
}

// ============================================================================
// Tree traversal with permuted feature (sparse)
// ============================================================================

__global__ void gpu_testreeimp_sparse_kernel(
    const CudaSparseMatrixCSR X_sparse,
    integer_t nsample,
    integer_t mdim,
    const integer_t* joob,
    const integer_t* pjoob,
    integer_t noob,
    integer_t mr,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    const integer_t* nodeclass,
    integer_t nnode,
    integer_t max_depth,
    integer_t* jvr,
    integer_t* nodexvr,
    integer_t* error_code
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple OOB samples
    for (integer_t n = tid; n < noob; n += stride) {
        integer_t sample_orig = joob[n];     // Original sample index
        integer_t sample_perm = pjoob[n];    // Permuted sample index (for feature mr)
        
        // Bounds check
        if (sample_orig < 0 || sample_orig >= nsample ||
            sample_perm < 0 || sample_perm >= nsample) {
            atomicExch(error_code, CUDA_ERROR_INVALID_SAMPLE);
            continue;
        }
        
        // Traverse tree from root
        integer_t kt = 0;
        bool reached_terminal = false;
        
        // Fixed iteration count to avoid warp divergence from while loops
        for (integer_t depth = 0; depth < max_depth && !reached_terminal; ++depth) {
            // Bounds check node
            if (kt < 0 || kt >= nnode) {
                atomicExch(error_code, CUDA_ERROR_INVALID_NODE);
                break;
            }
            
            // Check if terminal
            if (nodestatus[kt] == -1) {
                jvr[n] = nodeclass[kt];
                nodexvr[n] = kt;
                reached_terminal = true;
                break;
            }
            
            // Get split variable
            integer_t m = bestvar[kt];
            if (m < 0 || m >= mdim) {
                // Invalid - treat as terminal
                jvr[n] = nodeclass[kt];
                nodexvr[n] = kt;
                reached_terminal = true;
                break;
            }
            
            // Get feature value - KEY DIFFERENCE: permuted for feature mr
            // ROW-MAJOR: X_sparse.get(sample, feature)
            real_t xmn;
            if (m == mr) {
                // Use permuted sample for feature mr
                xmn = X_sparse.get(sample_perm, m);
            } else {
                // Use original sample for other features
                xmn = X_sparse.get(sample_orig, m);
            }
            
            // Determine child node (quantitative split - simplified)
            // treemap layout: [left0, right0, left1, right1, ...]
            if (xmn <= xbestsplit[kt]) {
                kt = treemap[kt * 2];      // Left child
            } else {
                kt = treemap[kt * 2 + 1];  // Right child
            }
        }
        
        // If didn't reach terminal, record current node
        if (!reached_terminal) {
            jvr[n] = (kt >= 0 && kt < nnode) ? nodeclass[kt] : 0;
            nodexvr[n] = kt;
        }
    }
}

// ============================================================================
// Importance computation kernel
// ============================================================================

// Kernel to compute "qimp" (per-sample original correct prediction weights) for local importance
// This is accumulated across trees and used by localimp() to compute final local importance
__global__ void gpu_compute_qimp_kernel(
    const integer_t* joob,
    const integer_t* cl,
    const integer_t* jtr,
    const integer_t* nodextr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t nnode,
    bool use_casewise,
    real_t* qimp,
    integer_t impn
) {
    if (impn != 1) return;  // Only compute if local importance requested
    
    // Each thread processes multiple OOB samples
    for (integer_t n = threadIdx.x + blockIdx.x * blockDim.x; n < noob; n += blockDim.x * gridDim.x) {
        integer_t nn = joob[n];
        
        // If original prediction was correct, add weight to qimp
        if (jtr[nn] == cl[nn]) {
            real_t weight = 1.0f;
            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                weight = tnodewt[nodextr[nn]];
            }
            ::atomicAdd(&qimp[nn], weight / static_cast<real_t>(noob));
        }
    }
}

// Kernel to compute "right" - total correct OOB predictions (before any permutation)
__global__ void gpu_compute_right_kernel(
    const integer_t* joob,
    const integer_t* cl,
    const integer_t* jtr,
    const integer_t* nodextr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t nnode,
    bool use_casewise,
    real_t* right_ptr  // Output: single value
) {
    __shared__ real_t s_right[256];
    
    integer_t tid = threadIdx.x;
    real_t local_right = 0.0f;
    
    // Each thread processes multiple OOB samples
    for (integer_t n = tid; n < noob; n += blockDim.x) {
        integer_t nn = joob[n];
        if (jtr[nn] == cl[nn]) {
            real_t weight = 1.0f;
            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                weight = tnodewt[nodextr[nn]];
            }
            local_right += weight;
        }
    }
    
    s_right[tid] = local_right;
    __syncthreads();
    
    // Reduction
    for (integer_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_right[tid] += s_right[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *right_ptr = s_right[0];
    }
}

// Kernel to handle UNUSED variables - matches CPU: original prediction weights to qimpm
// For variables not used in splits, permuting has no effect, so we just copy original weights
__global__ void gpu_update_qimpm_unused_kernel(
    const integer_t* joob,
    const integer_t* cl,
    const integer_t* jtr,
    const integer_t* nodextr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t mr,          // Feature index (unused variable)
    integer_t mdim,
    integer_t nnode,
    bool use_casewise,
    real_t* qimpm,
    integer_t impn
) {
    if (impn != 1) return;  // Only update qimpm if local importance requested
    
    // Each thread processes multiple OOB samples
    for (integer_t n = threadIdx.x + blockIdx.x * blockDim.x; n < noob; n += blockDim.x * gridDim.x) {
        integer_t nn = joob[n];
        
        // If original prediction was correct, add weight to qimpm
        if (jtr[nn] == cl[nn]) {
            real_t weight = 1.0f;
            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                weight = tnodewt[nodextr[nn]];
            }
            ::atomicAdd(&qimpm[nn * mdim + mr], weight / static_cast<real_t>(noob));
        }
    }
}

// Main importance kernel - ONE BLOCK per feature (matches GPU Dense architecture)
__global__ void gpu_compute_importance_kernel(
    const integer_t* joob,
    const integer_t* cl,
    const integer_t* jtr,
    const integer_t* jvr,
    const integer_t* nodextr,
    const integer_t* nodexvr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t mr,
    integer_t mdim,
    integer_t nnode,
    bool use_casewise,
    real_t right,      // Pre-computed total correct (passed as value)
    real_t* avimp,
    real_t* qimpm,
    integer_t impn,
    integer_t* error_code
) {
    // Shared memory for reduction
    __shared__ real_t s_rightimp[256];
    
    integer_t tid = threadIdx.x;
    real_t local_rightimp = 0.0f;
    
    // Each thread processes multiple OOB samples (ONE block processes ALL samples)
    for (integer_t n = tid; n < noob; n += blockDim.x) {
        integer_t nn = joob[n];
        integer_t true_class = cl[nn];
        integer_t perm_pred = jvr[n];
        
        // Permuted accuracy contribution
        if (perm_pred == true_class) {
            real_t weight = 1.0f;
            if (use_casewise && nodexvr[n] >= 0 && nodexvr[n] < nnode) {
                weight = tnodewt[nodexvr[n]];
            }
            local_rightimp += weight;
            
            // Local importance (per-sample, per-feature)
            if (impn == 1) {
                ::atomicAdd(&qimpm[nn * mdim + mr], weight / static_cast<real_t>(noob));
            }
        }
    }
    
    s_rightimp[tid] = local_rightimp;
    __syncthreads();
    
    // Reduction
    for (integer_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rightimp[tid] += s_rightimp[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes final importance and adds to global array
    // Matches GPU Dense: avimp[k] += (right - rightimp) / noob
    // Use atomicAdd to ensure proper accumulation across tree calls
    if (tid == 0 && noob > 0) {
        real_t rightimp = s_rightimp[0];
        real_t rr = (right - rightimp) / static_cast<real_t>(noob);
        ::atomicAdd(&avimp[mr], rr);
    }
}

// ============================================================================
// Host function: Full sparse variable importance
// ============================================================================

integer_t gpu_varimp_sparse(
    const CudaSparseMatrixCSR& X_sparse,
    const integer_t* d_cl,
    const integer_t* d_jtr,
    const integer_t* d_nin,
    const integer_t* d_nodextr,
    const integer_t* d_treemap,
    const integer_t* d_nodestatus,
    const real_t* d_xbestsplit,
    const integer_t* d_bestvar,
    const integer_t* d_nodeclass,
    const real_t* d_tnodewt,
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t max_depth,
    bool use_casewise,
    integer_t impn,
    real_t* d_avimp,
    real_t* d_qimp,      // NEW: per-sample original correct weights (for localimp)
    real_t* d_qimpm,
    curandState* d_rng_states,
    cudaStream_t stream
) {
    // Allocate error code
    integer_t* d_error;
    cudaMalloc(&d_error, sizeof(integer_t));
    cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
    
    // Step 1: Find OOB samples (nin == 0)
    // Count OOB samples
    std::vector<integer_t> h_nin(nsample);
    cudaMemcpy(h_nin.data(), d_nin, nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    std::vector<integer_t> h_joob;
    for (integer_t n = 0; n < nsample; n++) {
        if (h_nin[n] == 0) {
            h_joob.push_back(n);
        }
    }
    integer_t noob = static_cast<integer_t>(h_joob.size());
    
    if (noob == 0) {
        cudaFree(d_error);
        return CUDA_OK;  // No OOB samples
    }
    
    // Upload OOB indices
    integer_t* d_joob;
    integer_t* d_pjoob;
    integer_t* d_jvr;
    integer_t* d_nodexvr;
    
    cudaMalloc(&d_joob, noob * sizeof(integer_t));
    cudaMalloc(&d_pjoob, noob * sizeof(integer_t));
    cudaMalloc(&d_jvr, noob * sizeof(integer_t));
    cudaMalloc(&d_nodexvr, noob * sizeof(integer_t));
    
    cudaMemcpyAsync(d_joob, h_joob.data(), noob * sizeof(integer_t), 
                    cudaMemcpyHostToDevice, stream);
    
    // Step 2: Mark which variables are used in splits
    std::vector<integer_t> h_bestvar(nnode);
    std::vector<integer_t> h_nodestatus(nnode);
    cudaMemcpy(h_bestvar.data(), d_bestvar, nnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nodestatus.data(), d_nodestatus, nnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    std::vector<integer_t> used_vars;
    for (integer_t jj = 0; jj < nnode; jj++) {
        // Check for split nodes (status == 1), not just non-terminal (status != -1)
        // Status 0 = unprocessed, 1 = split, -1 = terminal
        if (h_nodestatus[jj] == 1) {
            integer_t var = h_bestvar[jj];
            if (var >= 0 && var < mdim) {
                // Check if not already in list
                bool found = false;
                for (auto v : used_vars) {
                    if (v == var) { found = true; break; }
                }
                if (!found) used_vars.push_back(var);
            }
        }
    }
    
    // Step 3: Compute "right" once for this tree (total correct OOB predictions)
    real_t* d_right;
    cudaMalloc(&d_right, sizeof(real_t));
    cudaMemsetAsync(d_right, 0, sizeof(real_t), stream);
    
    dim3 block_size(256);
    
    // Compute right with single block (processes all OOB samples)
    gpu_compute_right_kernel<<<1, block_size, 0, stream>>>(
        d_joob, d_cl, d_jtr, d_nodextr, d_tnodewt,
        noob, nnode, use_casewise, d_right
    );
    
    // Copy right to host (needed to pass as kernel argument)
    real_t h_right = 0.0f;
    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_right, d_right, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(d_right);
    
    // DEBUG: Print h_right value
    // Step 3b: Compute qimp (per-sample original correct weights) for local importance
    // This is needed by localimp() to transform raw qimpm into final local importance
    if (impn == 1 && d_qimp != nullptr) {
        dim3 qimp_grid((noob + block_size.x - 1) / block_size.x);
        gpu_compute_qimp_kernel<<<qimp_grid, block_size, 0, stream>>>(
            d_joob, d_cl, d_jtr, d_nodextr, d_tnodewt,
            noob, nnode, use_casewise, d_qimp, impn
        );
    }
    
    // Step 4: For each used variable, compute importance
    // Launch ONE block per variable (matches GPU Dense architecture)
    for (integer_t mr : used_vars) {
        // Reset error code
        cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
        
        // Permute OOB indices
        gpu_permute_oob_kernel<<<1, 1, 0, stream>>>(
            d_joob, d_pjoob, noob, d_rng_states, d_error
        );
        
        // Clear outputs
        cudaMemsetAsync(d_jvr, 0, noob * sizeof(integer_t), stream);
        cudaMemsetAsync(d_nodexvr, 0, noob * sizeof(integer_t), stream);
        
        // Test tree with permuted variable (can use multiple blocks for large noob)
        dim3 test_grid((noob + block_size.x - 1) / block_size.x);
        gpu_testreeimp_sparse_kernel<<<test_grid, block_size, 0, stream>>>(
            X_sparse,
            nsample,
            mdim,
            d_joob,
            d_pjoob,
            noob,
            mr,
            d_treemap,
            d_nodestatus,
            d_xbestsplit,
            d_bestvar,
            d_nodeclass,
            nnode,
            max_depth,
            d_jvr,
            d_nodexvr,
            d_error
        );
        
        // Compute importance with SINGLE block (matches GPU Dense)
        // One block processes all OOB samples, computes rightimp, adds (right-rightimp)/noob
        gpu_compute_importance_kernel<<<1, block_size, 0, stream>>>(
            d_joob,
            d_cl,
            d_jtr,
            d_jvr,
            d_nodextr,
            d_nodexvr,
            d_tnodewt,
            noob,
            mr,
            mdim,
            nnode,
            use_casewise,
            h_right,  // Pass pre-computed right as value
            d_avimp,
            d_qimpm,
            impn,
            d_error
        );
    }
    
    // Step 5: Handle UNUSED features - update qimpm with original prediction weights
    // For variables not used in splits, permuting has no effect (importance = 0)
    // But we need to update qimpm so that localimp() formula works correctly
    // This matches CPU dense/sparse behavior exactly
    if (impn == 1) {
        // Create set of used variables for O(1) lookup
        std::set<integer_t> used_set(used_vars.begin(), used_vars.end());
        
        for (integer_t mr = 0; mr < mdim; ++mr) {
            if (used_set.find(mr) == used_set.end()) {
                // Variable mr is NOT used in splits - update qimpm with original weights
                dim3 unused_grid((noob + block_size.x - 1) / block_size.x);
                gpu_update_qimpm_unused_kernel<<<unused_grid, block_size, 0, stream>>>(
                    d_joob,
                    d_cl,
                    d_jtr,
                    d_nodextr,
                    d_tnodewt,
                    noob,
                    mr,
                    mdim,
                    nnode,
                    use_casewise,
                    d_qimpm,
                    impn
                );
            }
        }
    }
    
    // Final sync and error check
    cudaStreamSynchronize(stream);
    integer_t h_error = 0;
    cudaMemcpy(&h_error, d_error, sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_error);
    cudaFree(d_joob);
    cudaFree(d_pjoob);
    cudaFree(d_jvr);
    cudaFree(d_nodexvr);
    
    return h_error;
}

// ============================================================================
// REGRESSION: Tree traversal with permuted feature (returns prediction as tnodewt)
// ============================================================================

__global__ void gpu_testreeimp_sparse_regression_kernel(
    const CudaSparseMatrixCSR X_sparse,
    integer_t nsample,
    integer_t mdim,
    const integer_t* joob,
    const integer_t* pjoob,
    integer_t noob,
    integer_t mr,
    const integer_t* treemap,
    const integer_t* nodestatus,
    const real_t* xbestsplit,
    const integer_t* bestvar,
    const real_t* nodepred,  // Terminal node predictions (mean y)
    integer_t nnode,
    integer_t max_depth,
    real_t* pred_perm,  // Output: permuted predictions
    integer_t* nodexvr   // Output: terminal nodes
) {
    integer_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    integer_t stride = blockDim.x * gridDim.x;
    
    for (integer_t n = tid; n < noob; n += stride) {
        integer_t sample_orig = joob[n];
        integer_t sample_perm = pjoob[n];
        
        if (sample_orig < 0 || sample_orig >= nsample ||
            sample_perm < 0 || sample_perm >= nsample) {
            continue;
        }
        
        // Traverse tree from root
        integer_t kt = 0;
        bool reached_terminal = false;
        
        for (integer_t depth = 0; depth < max_depth && !reached_terminal; ++depth) {
            if (kt < 0 || kt >= nnode) break;
            
            if (nodestatus[kt] == -1) {
                // Terminal node - get prediction from nodepred (mean y)
                pred_perm[n] = nodepred[kt];
                nodexvr[n] = kt;
                reached_terminal = true;
                break;
            }
            
            integer_t m = bestvar[kt];
            if (m < 0 || m >= mdim) {
                pred_perm[n] = nodepred[kt];
                nodexvr[n] = kt;
                reached_terminal = true;
                break;
            }
            
            // Get feature value - permuted for feature mr
            real_t xmn;
            if (m == mr) {
                xmn = X_sparse.get(sample_perm, m);
            } else {
                xmn = X_sparse.get(sample_orig, m);
            }
            
            // Traverse
            if (xmn <= xbestsplit[kt]) {
                kt = treemap[kt * 2];
            } else {
                kt = treemap[kt * 2 + 1];
            }
        }
        
        if (!reached_terminal && kt >= 0 && kt < nnode) {
            pred_perm[n] = nodepred[kt];  // Use nodepred (mean y) for prediction
            nodexvr[n] = kt;
        }
    }
}

// REGRESSION: Compute original MSE for OOB samples
__global__ void gpu_compute_mse_orig_regression_kernel(
    const integer_t* joob,
    const real_t* y_pred,
    const real_t* y_true,
    const integer_t* nodextr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t nnode,
    bool use_casewise,
    real_t* mse_orig_ptr,  // Output: single value
    real_t* qimp,          // Output: per-sample orig squared error [nsample]
    integer_t impn
) {
    __shared__ real_t s_mse[256];
    
    integer_t tid = threadIdx.x;
    real_t local_mse = 0.0f;
    
    for (integer_t n = tid; n < noob; n += blockDim.x) {
        integer_t nn = joob[n];
        real_t pred = y_pred[nn];
        real_t diff = pred - y_true[nn];
        
        real_t weight = 1.0f;
        if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
            weight = tnodewt[nodextr[nn]];
        }
        
        local_mse += weight * diff * diff;
        
        // Update qimp: original squared error / noob
        if (impn == 1 && qimp != nullptr) {
            ::atomicAdd(&qimp[nn], weight * diff * diff / static_cast<real_t>(noob));
        }
    }
    
    s_mse[tid] = local_mse;
    __syncthreads();
    
    // Reduction
    for (integer_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mse[tid] += s_mse[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0 && noob > 0) {
        *mse_orig_ptr = s_mse[0] / static_cast<real_t>(noob);
    }
}

// REGRESSION: Compute importance from permuted predictions
// avimp[mr] += (mse_perm - mse_orig)
__global__ void gpu_compute_importance_regression_kernel(
    const integer_t* joob,
    const real_t* y_true,
    const real_t* pred_perm,
    const integer_t* nodextr,
    const integer_t* nodexvr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t mr,
    integer_t mdim,
    integer_t nnode,
    bool use_casewise,
    real_t mse_orig,
    real_t* avimp,
    real_t* qimpm,
    integer_t impn
) {
    __shared__ real_t s_mse_perm[256];
    
    integer_t tid = threadIdx.x;
    real_t local_mse_perm = 0.0f;
    
    for (integer_t n = tid; n < noob; n += blockDim.x) {
        integer_t nn = joob[n];
        real_t diff = pred_perm[n] - y_true[nn];
        
        real_t weight = 1.0f;
        if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
            weight = tnodewt[nodextr[nn]];
        }
        
        local_mse_perm += weight * diff * diff;
        
        // Update qimpm: permuted squared error / noob (NOT delta!)
        // Breiman Cutler: qimpm stores perm², localimp_regression computes (qimpm - qimp)
        if (impn == 1 && qimpm != nullptr) {
            ::atomicAdd(&qimpm[nn * mdim + mr], weight * diff * diff / static_cast<real_t>(noob));
        }
    }
    
    s_mse_perm[tid] = local_mse_perm;
    __syncthreads();
    
    // Reduction
    for (integer_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mse_perm[tid] += s_mse_perm[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0 && noob > 0) {
        real_t mse_perm = s_mse_perm[0] / static_cast<real_t>(noob);
        // Variable importance = INCREASE in MSE (Breiman 2001)
        real_t rr = mse_perm - mse_orig;
        ::atomicAdd(&avimp[mr], rr);
    }
}

// REGRESSION: Handle unused variables
__global__ void gpu_update_qimpm_unused_regression_kernel(
    const integer_t* joob,
    const real_t* y_pred,
    const real_t* y_true,
    const integer_t* nodextr,
    const real_t* tnodewt,
    integer_t noob,
    integer_t mr,
    integer_t mdim,
    integer_t nnode,
    bool use_casewise,
    real_t* qimpm,
    integer_t impn
) {
    if (impn != 1) return;
    
    for (integer_t n = threadIdx.x + blockIdx.x * blockDim.x; n < noob; n += blockDim.x * gridDim.x) {
        integer_t nn = joob[n];
        real_t pred = y_pred[nn];
        real_t diff = pred - y_true[nn];
        
        real_t weight = 1.0f;
        if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
            weight = tnodewt[nodextr[nn]];
        }
        
        // For unused variables, perm² = orig² (no change)
        ::atomicAdd(&qimpm[nn * mdim + mr], weight * diff * diff / static_cast<real_t>(noob));
    }
}

// ============================================================================
// Host function: Sparse REGRESSION variable importance
// ============================================================================

integer_t gpu_varimp_sparse_regression(
    const CudaSparseMatrixCSR& X_sparse,
    const real_t* d_y_pred,
    const real_t* d_y_true,
    const integer_t* d_nin,
    const integer_t* d_nodextr,
    const integer_t* d_treemap,
    const integer_t* d_nodestatus,
    const real_t* d_xbestsplit,
    const integer_t* d_bestvar,
    const real_t* d_nodepred,   // Terminal node predictions (mean y)
    const real_t* d_tnodewt,    // Terminal node weights (mean bootstrap weight)
    integer_t nsample,
    integer_t mdim,
    integer_t nnode,
    integer_t max_depth,
    bool use_casewise,
    integer_t impn,
    real_t* d_avimp,
    real_t* d_qimp,
    real_t* d_qimpm,
    curandState* d_rng_states,
    cudaStream_t stream
) {
    // Step 1: Find OOB samples
    std::vector<integer_t> h_nin(nsample);
    cudaMemcpy(h_nin.data(), d_nin, nsample * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    std::vector<integer_t> h_joob;
    for (integer_t n = 0; n < nsample; n++) {
        if (h_nin[n] == 0) {
            h_joob.push_back(n);
        }
    }
    integer_t noob = static_cast<integer_t>(h_joob.size());
    
    if (noob == 0) return CUDA_OK;
    
    // Allocate device arrays
    integer_t* d_joob;
    integer_t* d_pjoob;
    real_t* d_pred_perm;
    integer_t* d_nodexvr;
    integer_t* d_error;
    
    cudaMalloc(&d_joob, noob * sizeof(integer_t));
    cudaMalloc(&d_pjoob, noob * sizeof(integer_t));
    cudaMalloc(&d_pred_perm, noob * sizeof(real_t));
    cudaMalloc(&d_nodexvr, noob * sizeof(integer_t));
    cudaMalloc(&d_error, sizeof(integer_t));
    
    cudaMemcpyAsync(d_joob, h_joob.data(), noob * sizeof(integer_t), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_error, 0, sizeof(integer_t), stream);
    
    // Step 2: Find used variables
    std::vector<integer_t> h_bestvar(nnode);
    std::vector<integer_t> h_nodestatus(nnode);
    cudaMemcpy(h_bestvar.data(), d_bestvar, nnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nodestatus.data(), d_nodestatus, nnode * sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    std::vector<integer_t> used_vars;
    for (integer_t jj = 0; jj < nnode; jj++) {
        if (h_nodestatus[jj] != -1) {
            integer_t var = h_bestvar[jj];
            if (var >= 0 && var < mdim) {
                bool found = false;
                for (auto v : used_vars) {
                    if (v == var) { found = true; break; }
                }
                if (!found) used_vars.push_back(var);
            }
        }
    }
    
    dim3 block_size(256);
    
    // Step 3: Compute original MSE
    real_t* d_mse_orig;
    cudaMalloc(&d_mse_orig, sizeof(real_t));
    cudaMemsetAsync(d_mse_orig, 0, sizeof(real_t), stream);
    
    gpu_compute_mse_orig_regression_kernel<<<1, block_size, 0, stream>>>(
        d_joob, d_y_pred, d_y_true, d_nodextr, d_tnodewt,
        noob, nnode, use_casewise, d_mse_orig, d_qimp, impn
    );
    
    real_t h_mse_orig = 0.0f;
    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_mse_orig, d_mse_orig, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(d_mse_orig);
    
    // Step 4: For each used variable, compute importance
    for (integer_t mr : used_vars) {
        // Permute OOB indices
        gpu_permute_oob_kernel<<<1, 1, 0, stream>>>(
            d_joob, d_pjoob, noob, d_rng_states, d_error
        );
        
        // Clear outputs
        cudaMemsetAsync(d_pred_perm, 0, noob * sizeof(real_t), stream);
        cudaMemsetAsync(d_nodexvr, 0, noob * sizeof(integer_t), stream);
        
        // Test tree with permuted variable
        dim3 test_grid((noob + block_size.x - 1) / block_size.x);
        gpu_testreeimp_sparse_regression_kernel<<<test_grid, block_size, 0, stream>>>(
            X_sparse,
            nsample,
            mdim,
            d_joob,
            d_pjoob,
            noob,
            mr,
            d_treemap,
            d_nodestatus,
            d_xbestsplit,
            d_bestvar,
            d_nodepred,  // Use nodepred for predictions (not tnodewt)
            nnode,
            max_depth,
            d_pred_perm,
            d_nodexvr
        );
        
        // Compute importance
        gpu_compute_importance_regression_kernel<<<1, block_size, 0, stream>>>(
            d_joob,
            d_y_true,
            d_pred_perm,
            d_nodextr,
            d_nodexvr,
            d_tnodewt,
            noob,
            mr,
            mdim,
            nnode,
            use_casewise,
            h_mse_orig,
            d_avimp,
            d_qimpm,
            impn
        );
    }
    
    // Step 5: Handle unused variables
    if (impn == 1) {
        std::set<integer_t> used_set(used_vars.begin(), used_vars.end());
        
        for (integer_t mr = 0; mr < mdim; ++mr) {
            if (used_set.find(mr) == used_set.end()) {
                dim3 unused_grid((noob + block_size.x - 1) / block_size.x);
                gpu_update_qimpm_unused_regression_kernel<<<unused_grid, block_size, 0, stream>>>(
                    d_joob,
                    d_y_pred,
                    d_y_true,
                    d_nodextr,
                    d_tnodewt,
                    noob,
                    mr,
                    mdim,
                    nnode,
                    use_casewise,
                    d_qimpm,
                    impn
                );
            }
        }
    }
    
    // Final sync
    cudaStreamSynchronize(stream);
    integer_t h_error = 0;
    cudaMemcpy(&h_error, d_error, sizeof(integer_t), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_error);
    cudaFree(d_joob);
    cudaFree(d_pjoob);
    cudaFree(d_pred_perm);
    cudaFree(d_nodexvr);
    
    return h_error;
}

} // namespace cuda
} // namespace rf

