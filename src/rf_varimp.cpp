#include "rf_varimp.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_config.hpp"
#include <cstdio>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rf {

// ============================================================================
// CPU VARIABLE IMPORTANCE IMPLEMENTATION
// ============================================================================

void cpu_permobmr(const integer_t* joob, integer_t* pjoob, integer_t noob) {
    // Create a copy of joob for permutation
    for (integer_t i = 0; i < noob; ++i) {
        pjoob[i] = joob[i];
    }
    
    // Fisher-Yates shuffle (exact match of original Breiman Fortran permobmr)
    // Iterates backwards from noob-1 down to 1, picking swap target from [0, j]
    MT19937 rng;
    rng.sgrnd(42);  // Fixed seed for reproducibility
    
    for (integer_t j = noob - 1; j > 0; --j) {
        // Pick random k in [0, j] inclusive (0-indexed C++ version of Fortran's [1, j])
        integer_t k = static_cast<integer_t>(rng.randomu() * (j + 1));
        if (k > j) k = j;  // Clamp in case randomu() returns exactly 1.0
        std::swap(pjoob[j], pjoob[k]);
    }
}

// ============================================================================
// PRE-SHUFFLED PERMUTATION CACHE IMPLEMENTATION
// ============================================================================
// Optimized approach: Pre-compute Fisher-Yates shuffle ONCE at fit() start,
// then reuse across all trees and all features. Eliminates RNG overhead.
// NOTE: Using regular global (not thread_local) to share across OpenMP threads

PreShuffledPermutation g_preshuffle_cache;

void PreShuffledPermutation::initialize(integer_t n, unsigned int seed) {
    nsample = n;
    shuffled_indices.resize(n);
    
    // Initialize with identity [0, 1, 2, ..., n-1]
    for (integer_t i = 0; i < n; ++i) {
        shuffled_indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    MT19937 rng;
    rng.sgrnd(seed);
    
    for (integer_t j = n - 1; j > 0; --j) {
        integer_t k = static_cast<integer_t>(rng.randomu() * (j + 1));
        if (k > j) k = j;
        std::swap(shuffled_indices[j], shuffled_indices[k]);
    }
}

void init_preshuffle_cache(integer_t nsample, unsigned int seed) {
    g_preshuffle_cache.initialize(nsample, seed);
}

// ============================================================================
// BATCHED TREE TRAVERSAL (processes all OOB samples at once)
// ============================================================================
// Uses pre-shuffled permutation for O(1) lookup instead of per-call RNG

void cpu_testreeimp_batched(
    const real_t* x, integer_t nsample, integer_t mdim,
    const integer_t* joob, integer_t noob, integer_t mr,
    const PreShuffledPermutation& preshuffle,
    const integer_t* treemap, const integer_t* nodestatus,
    const real_t* xbestsplit, const integer_t* bestvar,
    const integer_t* nodeclass, integer_t nnode,
    const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
    integer_t* jvr, integer_t* nodexvr) {
    
    // Parallel tree traversal for all OOB samples
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (integer_t n = 0; n < noob; ++n) {
        integer_t kt = 0;  // Start at root node
        integer_t orig_sample = joob[n];
        integer_t perm_sample = preshuffle.get_permuted(orig_sample);
        
        // Traverse tree
        for (integer_t k = 0; k < nnode; ++k) {
            if (nodestatus[kt] == -1) {  // Terminal node
                jvr[n] = nodeclass[kt];
                nodexvr[n] = kt;
                break;
            }
            
            integer_t m = bestvar[kt];
            if (m < 0 || m >= mdim) break;
            
            real_t xmn;
            if (m == mr) {
                // Use PERMUTED sample for variable mr (pre-shuffled lookup)
                integer_t idx = perm_sample * mdim + m;
                xmn = x[idx];
            } else {
                // Use ORIGINAL sample for other variables
                integer_t idx = orig_sample * mdim + m;
                xmn = x[idx];
            }
            
            // Determine if case goes left or right
            if (cat == nullptr || cat[m] <= 2) {
                if (xmn <= xbestsplit[kt]) {
                    kt = treemap[0 + kt * 2];
                } else {
                    kt = treemap[1 + kt * 2];
                }
            } else {
                integer_t jcat = static_cast<integer_t>(xmn + 0.5f);
                if (catgoleft != nullptr && jcat >= 0 && jcat < maxcat) {
                    if (catgoleft[jcat + kt * maxcat] == 1) {
                        kt = treemap[0 + kt * 2];
                    } else {
                        kt = treemap[1 + kt * 2];
                    }
                } else {
                    kt = treemap[0 + kt * 2];
                }
            }
            
            if (kt <= 0) {
                jvr[n] = nodeclass[0];
                nodexvr[n] = 0;
                break;
            }
        }
    }
}

void cpu_testreeimp(const real_t* x, integer_t nsample, integer_t mdim,
                   const integer_t* joob, integer_t noob, integer_t mr,
                   const integer_t* treemap, const integer_t* nodestatus,
                   const real_t* xbestsplit, const integer_t* bestvar,
                   const integer_t* nodeclass, integer_t nnode,
                   const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
                   integer_t* jvr, integer_t* nodexvr) {
    
    // Create permuted OOB samples
    std::vector<integer_t> pjoob(noob);
    cpu_permobmr(joob, pjoob.data(), noob);
    
    // For each OOB sample
    for (integer_t n = 0; n < noob; ++n) {
        integer_t kt = 0;  // Start at root node
        
        // Traverse tree
        for (integer_t k = 0; k < nnode; ++k) {
            if (nodestatus[kt] == -1) {  // Terminal node
                jvr[n] = nodeclass[kt];
                nodexvr[n] = kt;
                break;
            }
            
            integer_t m = bestvar[kt];
            if (m < 0 || m >= mdim) break;  // Invalid variable index
            real_t xmn;
            
            if (m == mr) {
                // Use permuted value for variable mr
                // Row-major indexing: x[row * ncols + col] where row=pjoob[n], col=m
                if (pjoob[n] < 0 || pjoob[n] >= nsample) break;
                integer_t idx = pjoob[n] * mdim + m;
                if (idx >= 0 && idx < nsample * mdim) {
                    xmn = x[idx];
                } else {
                    break;
                }
            } else {
                // Use original value
                // Row-major indexing: x[row * ncols + col] where row=joob[n], col=m
                if (joob[n] < 0 || joob[n] >= nsample) break;
                integer_t idx = joob[n] * mdim + m;
                if (idx >= 0 && idx < nsample * mdim) {
                    xmn = x[idx];
                } else {
                    break;
                }
            }
            
            // Determine if case goes left or right
            if (cat == nullptr || cat[m] <= 2) {  // Quantitative variable (including binary)
                if (xmn <= xbestsplit[kt]) {
                    kt = treemap[0 + kt * 2];  // Left child
                } else {
                    kt = treemap[1 + kt * 2];  // Right child
                }
            } else {  // Categorical variable
                integer_t jcat = static_cast<integer_t>(xmn + 0.5f);  // Round to nearest integer
                if (catgoleft != nullptr && jcat >= 0 && jcat < maxcat) {
                    if (catgoleft[jcat + kt * maxcat] == 1) {
                        kt = treemap[0 + kt * 2];  // Left child
                    } else {
                        kt = treemap[1 + kt * 2];  // Right child
                    }
                } else {
                    // Invalid categorical value, cannot traverse further
                    break;
                }
            }
        }
    }
}

void cpu_varimp(const real_t* x, integer_t nsample, integer_t mdim,
               const integer_t* cl, const integer_t* nin, const integer_t* jtr,
               integer_t impn, real_t* qimp, real_t* qimpm,
               real_t* avimp, real_t* sqsd,
               const integer_t* treemap, const integer_t* nodestatus,
               const real_t* xbestsplit, const integer_t* bestvar,
               const integer_t* nodeclass, integer_t nnode,
               const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
               integer_t maxcat, const integer_t* catgoleft,
               const real_t* tnodewt, const integer_t* nodextr) {
    
    // Set OpenMP thread count from config
#ifdef _OPENMP
    if (g_config.n_threads_cpu > 0) {
        omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
    }
#endif
    
    // Initialize arrays
    zervr(qimp, nsample);
    zervr(qimpm, nsample * mdim);
    zervr(avimp, mdim);
    zervr(sqsd, mdim);
    
    // Step 1: Find OOB samples and calculate original accuracy
    integer_t noob = 0;
    real_t right = 0.0f;
    std::vector<integer_t> joob(nsample);
    
    // Check if case-wise (bootstrap frequency weighted) or non-case-wise (simple averaging) should be used
    bool use_casewise = g_config.use_casewise;
    
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin[n] == 0) {  // OOB sample
            // Update count of correct OOB classifications
            if (jtr[n] == cl[n]) {
                // Case-wise: use bootstrap frequency weighted (tnodewt)
                // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                if (use_casewise && nodextr[n] >= 0 && nodextr[n] < nnode) {
                    right += tnodewt[nodextr[n]];
                } else {
                    right += 1.0f;
                }
            }
            joob[noob] = n;  // Store OOB sample index
            noob++;
        }
    }
    
    // Step 2: Update qimp for local importance (if impn == 1)
    // Case-wise: Match Fortran varimp.f exactly: qimp(nn) = qimp(nn) + tnodewt(nodextr(nn))/noob
    // Non-case-wise: Simple averaging: qimp(nn) = qimp(nn) + 1.0/noob
    if (impn == 1) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (integer_t n = 0; n < noob; ++n) {
            integer_t nn = joob[n];
            if (jtr[nn] == cl[nn]) {
                real_t weight = 1.0f;
                if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                    weight = tnodewt[nodextr[nn]];
                }
                qimp[nn] += weight / static_cast<real_t>(noob);
            }
        }
    }
    
    // Step 3: Mark which variables were used in splits
    std::vector<integer_t> iv(mdim, 0);
    for (integer_t jj = 0; jj < nnode; ++jj) {
        if (nodestatus[jj] != -1) {  // Non-terminal node
            iv[bestvar[jj]] = 1;
        }
    }
    
    // OPTIMIZATION: Initialize pre-shuffled permutation cache ONCE
    // This eliminates per-feature RNG overhead (Fisher-Yates done once, reused for all features)
    init_preshuffle_cache(nsample, 42);
    
    // Step 4: Calculate importance for each variable (parallelize across variables)
    // Use thread-local arrays to avoid race conditions
#ifdef _OPENMP
#pragma omp parallel
    {
        // Thread-local arrays for each thread
        std::vector<integer_t> jvr_local(noob, 0);
        std::vector<integer_t> nodexvr_local(noob, 0);
        
#pragma omp for nowait
        for (integer_t k = 0; k < mdim; ++k) {
            if (iv[k] == 1) {  // Variable was used in splits
                // OPTIMIZED: Use batched tree traversal with pre-shuffled permutation
                cpu_testreeimp_batched(x, nsample, mdim, joob.data(), noob, k,
                              g_preshuffle_cache,
                              treemap, nodestatus, xbestsplit, bestvar, nodeclass,
                              nnode, cat, maxcat, catgoleft, jvr_local.data(), nodexvr_local.data());
                
                // Calculate permuted accuracy
                real_t rightimp = 0.0f;
                for (integer_t n = 0; n < noob; ++n) {
                    integer_t nn = joob[n];  // Original case index
                    if (impn == 1) {
                        if (jvr_local[n] == cl[nn]) {
                            // Case-wise: use bootstrap frequency weighted (tnodewt)
                            // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                            real_t weight = 1.0f;
                            if (use_casewise && nodexvr_local[n] >= 0 && nodexvr_local[n] < nnode) {
                                weight = tnodewt[nodexvr_local[n]];
                            }
                            qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                        }
                    }
                    if (jvr_local[n] == cl[nn]) {
                        real_t weight = 1.0f;
                        if (use_casewise && nodexvr_local[n] >= 0 && nodexvr_local[n] < nnode) {
                            weight = tnodewt[nodexvr_local[n]];
                        }
                        rightimp += weight;
                    }
                }
                
                // Calculate importance (use atomic operations for thread safety)
                real_t rr = (right - rightimp) / static_cast<real_t>(noob);
                avimp[k] += rr;
                sqsd[k] += rr * rr;
            } else {
                // Variable not used in splits - use original predictions (permuting has no effect)
                for (integer_t n = 0; n < noob; ++n) {
                    integer_t nn = joob[n];
                    if (impn == 1) {
                        if (jtr[nn] == cl[nn]) {
                            // Case-wise: use bootstrap frequency weighted (tnodewt)
                            // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                            real_t weight = 1.0f;
                            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                                weight = tnodewt[nodextr[nn]];
                            }
                            qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                        }
                    }
                }
            }
        }
    }
#else
    // Sequential version (no OpenMP) - still use optimized batched traversal
    for (integer_t k = 0; k < mdim; ++k) {
        if (iv[k] == 1) {  // Variable was used in splits
            // OPTIMIZED: Use batched tree traversal with pre-shuffled permutation
            cpu_testreeimp_batched(x, nsample, mdim, joob.data(), noob, k,
                          g_preshuffle_cache,
                          treemap, nodestatus, xbestsplit, bestvar, nodeclass,
                          nnode, cat, maxcat, catgoleft, jvr, nodexvr);
            
            // Calculate permuted accuracy
            real_t rightimp = 0.0f;
            for (integer_t n = 0; n < noob; ++n) {
                integer_t nn = joob[n];  // Original case index
                if (impn == 1) {
                    if (jvr[n] == cl[nn]) {
                        // Case-wise: use bootstrap frequency weighted (tnodewt)
                        // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                        real_t weight = 1.0f;
                        if (use_casewise && nodexvr[n] >= 0 && nodexvr[n] < nnode) {
                            weight = tnodewt[nodexvr[n]];
                        }
                        qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                    }
                }
                if (jvr[n] == cl[nn]) {
                    real_t weight = 1.0f;
                    if (use_casewise && nodexvr[n] >= 0 && nodexvr[n] < nnode) {
                        weight = tnodewt[nodexvr[n]];
                    }
                    rightimp += weight;
                }
            }
            
            // Calculate importance
            real_t rr = (right - rightimp) / static_cast<real_t>(noob);
            avimp[k] += rr;
            sqsd[k] += rr * rr;
        } else {
            // Variable not used in splits - use original predictions (permuting has no effect)
            for (integer_t n = 0; n < noob; ++n) {
                integer_t nn = joob[n];
                    if (impn == 1) {
                        if (jtr[nn] == cl[nn]) {
                            // Case-wise: use bootstrap frequency weighted (tnodewt)
                            // Non-case-wise: use simple 1.0 (UC Berkeley standard)
                            real_t weight = 1.0f;
                            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                                weight = tnodewt[nodextr[nn]];
                            }
                            qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                        }
                    }
            }
        }
    }
#endif
}

// ============================================================================
// CPU VARIABLE IMPORTANCE FOR REGRESSION (MSE-based, Breiman 2001)
// ============================================================================

// Helper function: Traverse tree and return predicted value (nodepred) for regression
// NOTE: nodepred stores the prediction (mean y), tnodewt stores the mean weight
real_t cpu_predict_regression(const real_t* x, integer_t sample_idx, integer_t mdim,
                               const integer_t* treemap, const integer_t* nodestatus,
                               const real_t* xbestsplit, const integer_t* bestvar,
                               const real_t* nodepred, integer_t nnode,
                               const integer_t* cat, integer_t maxcat, 
                               const integer_t* catgoleft,
                               integer_t permute_var, integer_t permuted_sample_idx) {
    
    integer_t kt = 0;  // Start at root node
    
    for (integer_t k = 0; k < nnode; ++k) {
        if (nodestatus[kt] == -1) {  // Terminal node
            return nodepred[kt];  // Return terminal node prediction (mean y)
        }
        
        integer_t m = bestvar[kt];
        if (m < 0 || m >= mdim) break;
        
        // Get feature value (use permuted sample for permute_var)
        integer_t row = (m == permute_var) ? permuted_sample_idx : sample_idx;
        integer_t idx = m + row * mdim;  // Column-major indexing
        real_t xmn = x[idx];
        
        // Navigate tree
        if (cat == nullptr || cat[m] <= 2) {  // Quantitative variable (including binary)
            if (xmn <= xbestsplit[kt]) {
                kt = treemap[0 + kt * 2];  // Left
            } else {
                kt = treemap[1 + kt * 2];  // Right
            }
        } else {  // Categorical variable
            integer_t jcat = static_cast<integer_t>(xmn + 0.5f);
            if (catgoleft != nullptr && jcat >= 0 && jcat < maxcat) {
                if (catgoleft[jcat + kt * maxcat] == 1) {
                    kt = treemap[0 + kt * 2];
                } else {
                    kt = treemap[1 + kt * 2];
                }
            } else {
                break;
            }
        }
        
        if (kt < 0 || kt >= nnode) break;
    }
    
    return 0.0f;  // Fallback
}

void cpu_varimp_regression(const real_t* x, integer_t nsample, integer_t mdim,
                           const real_t* y, const integer_t* nin, 
                           const real_t* y_pred,  // OOB predictions (nodepred values)
                           integer_t impn, real_t* qimp, real_t* qimpm,
                           real_t* avimp, real_t* sqsd,
                           const integer_t* treemap, const integer_t* nodestatus,
                           const real_t* xbestsplit, const integer_t* bestvar,
                           const real_t* nodepred,  // Terminal node predictions (mean y)
                           const real_t* tnodewt,   // Terminal node weights (mean bootstrap weight)
                           integer_t nnode,
                           const integer_t* cat, integer_t maxcat, 
                           const integer_t* catgoleft,
                           const integer_t* nodextr) {
    
    // Set OpenMP thread count from config
#ifdef _OPENMP
    if (g_config.n_threads_cpu > 0) {
        omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
    }
#endif
    
    // Initialize output arrays
    zervr(qimp, nsample);
    zervr(qimpm, nsample * mdim);
    zervr(avimp, mdim);
    zervr(sqsd, mdim);
    
    bool use_casewise = g_config.use_casewise;
    
    // Step 1: Find OOB samples
    integer_t noob = 0;
    std::vector<integer_t> joob(nsample);
    
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin[n] == 0) {  // OOB sample
            joob[noob] = n;
            noob++;
        }
    }
    
    if (noob == 0) return;  // No OOB samples
    
    // Step 2: Calculate original MSE for OOB samples
    // MSE = (1/n) * Σ(y_pred - y_true)²
    real_t mse_orig = 0.0f;
    for (integer_t n = 0; n < noob; ++n) {
        integer_t nn = joob[n];
        real_t pred = y_pred[nn];  // OOB prediction for this sample
        real_t diff = pred - y[nn];
        
        // Casewise: weight by tnodewt, Non-casewise: equal weight
        real_t weight = 1.0f;
        if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
            weight = tnodewt[nodextr[nn]];
        }
        mse_orig += weight * diff * diff;
        
        // Update local importance qimp: original contribution
        if (impn == 1) {
            qimp[nn] += weight * diff * diff / static_cast<real_t>(noob);
        }
    }
    mse_orig /= static_cast<real_t>(noob);
    
    // Step 3: Use pre-shuffled permutation (OPTIMIZED - no per-tree RNG overhead)
    init_preshuffle_cache(nsample, 42);
    // Create pjoob using pre-shuffle cache
    std::vector<integer_t> pjoob(noob);
    for (integer_t n = 0; n < noob; ++n) {
        pjoob[n] = g_preshuffle_cache.get_permuted(joob[n]);
    }
    
    // Step 4: For each variable, calculate permuted MSE
    // Breiman (2001): "Randomly permute the values of variable m in the oob cases
    // and put these cases down the tree. The decrease in number of votes for the
    // correct class (for classification) or increase in MSE (for regression) is the
    // raw importance score for variable m."
    
    for (integer_t k = 0; k < mdim; ++k) {
        real_t mse_perm = 0.0f;
        
        for (integer_t n = 0; n < noob; ++n) {
            integer_t nn = joob[n];  // Original sample index
            integer_t pn = pjoob[n];  // Permuted sample index (for variable k)
            
            // Predict with permuted variable k
            real_t pred_perm = cpu_predict_regression(x, nn, mdim,
                                                      treemap, nodestatus,
                                                      xbestsplit, bestvar,
                                                      nodepred, nnode,
                                                      cat, maxcat, catgoleft,
                                                      k, pn);
            
            real_t diff = pred_perm - y[nn];
            
            real_t weight = 1.0f;
            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                weight = tnodewt[nodextr[nn]];
            }
            mse_perm += weight * diff * diff;
            
            // Update local importance qimpm: permuted squared error
            // Breiman Cutler: qimpm stores permuted squared error (NOT the delta)
            // Then localimp_regression computes: 100 * (qimpm - qimp) / ntree
            if (impn == 1) {
                qimpm[nn * mdim + k] += weight * diff * diff / static_cast<real_t>(noob);
            }
        }
        mse_perm /= static_cast<real_t>(noob);
        
        // Variable importance = increase in MSE (Breiman 2001)
        real_t rr = mse_perm - mse_orig;
        avimp[k] += rr;
        sqsd[k] += rr * rr;
    }
}

// ============================================================================
// SPARSE MATRIX SUPPORT - Variable Importance with CSR format
// ============================================================================

void cpu_testreeimp_sparse(const SparseMatrixCSR& x_sparse, integer_t nsample, integer_t mdim,
                           const integer_t* joob, integer_t noob, integer_t mr,
                           const integer_t* treemap, const integer_t* nodestatus,
                           const real_t* xbestsplit, const integer_t* bestvar,
                           const integer_t* nodeclass, integer_t nnode,
                           const integer_t* cat, integer_t maxcat, const integer_t* catgoleft,
                           integer_t* jvr, integer_t* nodexvr) {
    
    // Create permuted OOB samples
    std::vector<integer_t> pjoob(noob);
    cpu_permobmr(joob, pjoob.data(), noob);
    
    // For each OOB sample
    for (integer_t n = 0; n < noob; ++n) {
        integer_t kt = 0;  // Start at root node
        
        // Traverse tree
        for (integer_t k = 0; k < nnode; ++k) {
            if (nodestatus[kt] == -1) {  // Terminal node
                jvr[n] = nodeclass[kt];
                nodexvr[n] = kt;
                break;
            }
            
            integer_t m = bestvar[kt];
            if (m < 0 || m >= mdim) break;
            
            real_t xmn;
            if (m == mr) {
                // Use permuted value for variable mr - SPARSE ACCESS
                if (pjoob[n] >= 0 && pjoob[n] < nsample) {
                    xmn = x_sparse.get(pjoob[n], m);
                } else {
                    break;
                }
            } else {
                // Use original value - SPARSE ACCESS
                if (joob[n] >= 0 && joob[n] < nsample) {
                    xmn = x_sparse.get(joob[n], m);
                } else {
                    break;
                }
            }
            
            // Determine if case goes left or right
            if (cat == nullptr || cat[m] <= 2) {
                if (xmn <= xbestsplit[kt]) {
                    kt = treemap[0 + kt * 2];
                } else {
                    kt = treemap[1 + kt * 2];
                }
            } else {
                integer_t jcat = static_cast<integer_t>(xmn + 0.5f);
                if (catgoleft != nullptr && jcat >= 0 && jcat < maxcat) {
                    if (catgoleft[jcat + kt * maxcat] == 1) {
                        kt = treemap[0 + kt * 2];
                    } else {
                        kt = treemap[1 + kt * 2];
                    }
                } else {
                    break;
                }
            }
            
            if (kt < 0 || kt >= nnode) {
                jvr[n] = nodeclass[0];
                nodexvr[n] = 0;
                break;
            }
        }
    }
}

void cpu_varimp_sparse(const SparseMatrixCSR& x_sparse, integer_t nsample, integer_t mdim,
                       const integer_t* cl, const integer_t* nin, const integer_t* jtr,
                       integer_t impn, real_t* qimp, real_t* qimpm,
                       real_t* avimp, real_t* sqsd,
                       const integer_t* treemap, const integer_t* nodestatus,
                       const real_t* xbestsplit, const integer_t* bestvar,
                       const integer_t* nodeclass, integer_t nnode,
                       const integer_t* cat, integer_t* jvr, integer_t* nodexvr,
                       integer_t maxcat, const integer_t* catgoleft,
                       const real_t* tnodewt, const integer_t* nodextr) {
    
    // printf("[VARIMP SPARSE ENTRY] nsample=%d, mdim=%d, nnode=%d\n", (int)nsample, (int)mdim, (int)nnode);
    // printf("[VARIMP SPARSE ENTRY] x_sparse valid=%d\n", x_sparse.is_valid() ? 1 : 0);
    // fflush(stdout);
    
    // Initialize arrays
    zervr(qimp, nsample);
    zervr(qimpm, nsample * mdim);
    zervr(avimp, mdim);
    zervr(sqsd, mdim);
    
    // std::cout << "[VARIMP SPARSE] Arrays initialized" << std::endl;
    // std::cout.flush();
    
    // Step 1: Find OOB samples and calculate original accuracy
    integer_t noob = 0;
    real_t right = 0.0f;
    std::vector<integer_t> joob(nsample);
    
    // std::cout << "[VARIMP SPARSE] joob allocated, checking nin pointer..." << std::endl;
    // std::cout.flush();
    
    // if (nin == nullptr) {
    //     std::cout << "[VARIMP SPARSE] ERROR: nin is null!" << std::endl;
    //     std::cout.flush();
    //     return;
    // }
    // if (jtr == nullptr) {
    //     std::cout << "[VARIMP SPARSE] ERROR: jtr is null!" << std::endl;
    //     std::cout.flush();
    //     return;
    // }
    // if (cl == nullptr) {
    //     std::cout << "[VARIMP SPARSE] ERROR: cl is null!" << std::endl;
    //     std::cout.flush();
    //     return;
    // }
    
    // std::cout << "[VARIMP SPARSE] Pointers OK, starting OOB loop..." << std::endl;
    // std::cout.flush();
    
    bool use_casewise = g_config.use_casewise;
    
    // // Debug: print first few values to verify arrays are valid
    // printf("[VARIMP SPARSE DEBUG] First 5 nin values: ");
    // for (int i = 0; i < 5 && i < nsample; i++) printf("%d ", nin[i]);
    // printf("\n");
    // printf("[VARIMP SPARSE DEBUG] First 5 jtr values: ");
    // for (int i = 0; i < 5 && i < nsample; i++) printf("%d ", jtr[i]);
    // printf("\n");
    // printf("[VARIMP SPARSE DEBUG] First 5 cl values: ");
    // for (int i = 0; i < 5 && i < nsample; i++) printf("%d ", cl[i]);
    // printf("\n");
    // fflush(stdout);
    
    for (integer_t n = 0; n < nsample; ++n) {
        // Print every 50th iteration, or every iteration near the end
        // if (n % 50 == 0 || n >= 350) {
        //     printf("[VARIMP SPARSE] Processing n=%d/%d, noob=%d\n", (int)n, (int)nsample, (int)noob);
        //     fflush(stdout);
        // }
        
        // Bounds check before accessing arrays
        // if (n >= nsample) {
        //     printf("[VARIMP SPARSE] ERROR: n=%d >= nsample=%d\n", (int)n, (int)nsample);
        //     fflush(stdout);
        //     break;
        // }
        
        integer_t nin_val = nin[n];
        if (nin_val == 0) {
            integer_t jtr_val = jtr[n];
            integer_t cl_val = cl[n];
            
            // if (noob < 5) {
            //     printf("[VARIMP SPARSE] OOB sample n=%d: jtr=%d, cl=%d\n", 
            //            (int)n, (int)jtr_val, (int)cl_val);
            //     fflush(stdout);
            // }
            
            if (jtr_val == cl_val) {
                if (use_casewise && nodextr[n] >= 0 && nodextr[n] < nnode) {
                    right += tnodewt[nodextr[n]];
                } else {
                    right += 1.0f;
                }
            }
            joob[noob] = n;
            noob++;
        }
    }
    
    // printf("[VARIMP SPARSE] OOB loop completed! noob=%d\n", (int)noob);
    // fflush(stdout);
    
    // printf("[VARIMP SPARSE] Found %d OOB samples\n", (int)noob);
    // fflush(stdout);
    
    // // Step 2: Update qimp for local importance
    // printf("[VARIMP SPARSE] Step 2: local importance (impn=%d)\n", (int)impn);
    // fflush(stdout);
    
    if (impn == 1) {
        for (integer_t n = 0; n < noob; ++n) {
            integer_t nn = joob[n];
            if (jtr[nn] == cl[nn]) {
                real_t weight = 1.0f;
                if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                    weight = tnodewt[nodextr[nn]];
                }
                qimp[nn] += weight / static_cast<real_t>(noob);
            }
        }
    }
    
    // printf("[VARIMP SPARSE] Step 3: mark used variables\n");
    // fflush(stdout);
    
    // Step 3: Mark which variables were used in splits
    std::vector<integer_t> iv(mdim, 0);
    
    // if (nodestatus == nullptr) {
    //     std::cout << "[VARIMP SPARSE] ERROR: nodestatus is null!" << std::endl;
    //     return;
    // }
    // if (bestvar == nullptr) {
    //     std::cout << "[VARIMP SPARSE] ERROR: bestvar is null!" << std::endl;
    //     return;
    // }
    
    for (integer_t jj = 0; jj < nnode; ++jj) {
        if (nodestatus[jj] != -1) {
            if (bestvar[jj] >= 0 && bestvar[jj] < mdim) {
                iv[bestvar[jj]] = 1;
            }
        }
    }
    
    integer_t used_vars = 0;
    for (integer_t k = 0; k < mdim; ++k) {
        if (iv[k] == 1) used_vars++;
    }
    // std::cout << "[VARIMP SPARSE] " << used_vars << " variables used in splits" << std::endl;
    // std::cout.flush();
    
    // // Step 4: Calculate importance for each variable
    // std::cout << "[VARIMP SPARSE] Step 4: calculate importance" << std::endl;
    // std::cout.flush();
    
    std::vector<integer_t> jvr_local(noob, 0);
    std::vector<integer_t> nodexvr_local(noob, 0);
    
    for (integer_t k = 0; k < mdim; ++k) {
        if (iv[k] == 1) {
            // Variable was used in splits - test tree with permuted variable k
            cpu_testreeimp_sparse(x_sparse, nsample, mdim, joob.data(), noob, k,
                                  treemap, nodestatus, xbestsplit, bestvar, nodeclass,
                                  nnode, cat, maxcat, catgoleft, jvr_local.data(), nodexvr_local.data());
            
            // Calculate permuted accuracy
            real_t rightimp = 0.0f;
            for (integer_t n = 0; n < noob; ++n) {
                integer_t nn = joob[n];
                if (impn == 1) {
                    if (jvr_local[n] == cl[nn]) {
                        real_t weight = 1.0f;
                        if (use_casewise && nodexvr_local[n] >= 0 && nodexvr_local[n] < nnode) {
                            weight = tnodewt[nodexvr_local[n]];
                        }
                        qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                    }
                }
                if (jvr_local[n] == cl[nn]) {
                    real_t weight = 1.0f;
                    if (use_casewise && nodexvr_local[n] >= 0 && nodexvr_local[n] < nnode) {
                        weight = tnodewt[nodexvr_local[n]];
                    }
                    rightimp += weight;
                }
            }
            
            // Variable importance = decrease in accuracy
            real_t rr = (right - rightimp) / static_cast<real_t>(noob);
            avimp[k] += rr;
            sqsd[k] += rr * rr;
        } else {
            // Variable not used in splits - use original predictions (permuting has no effect)
            // This matches dense cpu_varimp exactly
            for (integer_t n = 0; n < noob; ++n) {
                integer_t nn = joob[n];
                if (impn == 1) {
                    if (jtr[nn] == cl[nn]) {
                        real_t weight = 1.0f;
                        if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                            weight = tnodewt[nodextr[nn]];
                        }
                        qimpm[nn * mdim + k] += weight / static_cast<real_t>(noob);
                    }
                }
            }
        }
    }
    
    // std::cout << "[VARIMP SPARSE] Finished successfully, returning" << std::endl;
    // std::cout.flush();
}

void cpu_varimp_regression_sparse(const SparseMatrixCSR& x_sparse, integer_t nsample, integer_t mdim,
                                  const real_t* y, const integer_t* nin, 
                                  const real_t* y_pred,
                                  integer_t impn, real_t* qimp, real_t* qimpm,
                                  real_t* avimp, real_t* sqsd,
                                  const integer_t* treemap, const integer_t* nodestatus,
                                  const real_t* xbestsplit, const integer_t* bestvar,
                                  const real_t* nodepred,  // Terminal node predictions (mean y)
                                  const real_t* tnodewt,   // Terminal node weights (mean bootstrap weight)
                                  integer_t nnode,
                                  const integer_t* cat, integer_t maxcat, 
                                  const integer_t* catgoleft,
                                  const integer_t* nodextr) {
    
    // Initialize
    zervr(qimp, nsample);
    zervr(qimpm, nsample * mdim);
    zervr(avimp, mdim);
    zervr(sqsd, mdim);
    
    // Find OOB samples
    integer_t noob = 0;
    std::vector<integer_t> joob(nsample);
    
    for (integer_t n = 0; n < nsample; ++n) {
        if (nin[n] == 0) {
            joob[noob] = n;
            noob++;
        }
    }
    
    if (noob == 0) return;
    
    bool use_casewise = g_config.use_casewise;
    
    // Calculate original MSE (matches dense version exactly)
    real_t mse_orig = 0.0f;
    for (integer_t n = 0; n < noob; ++n) {
        integer_t nn = joob[n];
        real_t pred = y_pred[nn];
        real_t diff = pred - y[nn];
        
        real_t weight = 1.0f;
        if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
            weight = tnodewt[nodextr[nn]];
        }
        mse_orig += weight * diff * diff;
        
        if (impn == 1) {
            qimp[nn] += weight * diff * diff / static_cast<real_t>(noob);
        }
    }
    mse_orig /= static_cast<real_t>(noob);
    
    // Create ONE permutation for all variables (matches original Fortran)
    std::vector<integer_t> pjoob(noob);
    cpu_permobmr(joob.data(), pjoob.data(), noob);
    
    // Mark used variables
    std::vector<integer_t> iv(mdim, 0);
    for (integer_t jj = 0; jj < nnode; ++jj) {
        if (nodestatus[jj] != -1) {
            if (bestvar[jj] >= 0 && bestvar[jj] < mdim) {
                iv[bestvar[jj]] = 1;
            }
        }
    }
    
    // Permutation importance for each variable (matches dense version exactly)
    for (integer_t k = 0; k < mdim; ++k) {
        real_t mse_perm = 0.0f;
        
        for (integer_t n = 0; n < noob; ++n) {
            integer_t nn = joob[n];   // Original sample index
            integer_t pn = pjoob[n];  // Permuted sample index
            
            // Traverse tree with permuted variable k using SPARSE ACCESS
            integer_t kt = 0;
            while (nodestatus[kt] != -1 && kt >= 0 && kt < nnode) {
                integer_t m = bestvar[kt];
                if (m < 0 || m >= mdim) break;
                
                real_t xmn;
                if (m == k) {
                    // Use permuted sample's value for variable k
                    xmn = x_sparse.get(pn, m);
                } else {
                    // Use original sample's value for other variables
                    xmn = x_sparse.get(nn, m);
                }
                
                if (cat == nullptr || cat[m] <= 2) {
                    kt = (xmn <= xbestsplit[kt]) ? treemap[kt * 2] : treemap[kt * 2 + 1];
                } else {
                    integer_t jcat = static_cast<integer_t>(xmn + 0.5f);
                    if (jcat >= 0 && jcat < maxcat && catgoleft != nullptr) {
                        kt = (catgoleft[jcat + kt * maxcat] == 1) ? treemap[kt * 2] : treemap[kt * 2 + 1];
                    } else {
                        break;
                    }
                }
                
                if (kt < 0 || kt >= nnode) break;
            }
            
            // Get prediction from terminal node (nodepred stores mean y)
            real_t pred_perm = (kt >= 0 && kt < nnode) ? nodepred[kt] : y_pred[nn];
            real_t diff = pred_perm - y[nn];
            
            real_t weight = 1.0f;
            if (use_casewise && nodextr[nn] >= 0 && nodextr[nn] < nnode) {
                weight = tnodewt[nodextr[nn]];
            }
            mse_perm += weight * diff * diff;
            
            // Update local importance qimpm: permuted squared error
            // Breiman Cutler: qimpm stores permuted squared error (NOT the delta)
            // Then localimp_regression computes: 100 * (qimpm - qimp) / ntree
            if (impn == 1) {
                qimpm[nn * mdim + k] += weight * diff * diff / static_cast<real_t>(noob);
            }
        }
        mse_perm /= static_cast<real_t>(noob);
        
        // Variable importance = increase in MSE (Breiman 2001)
        real_t rr = mse_perm - mse_orig;
        avimp[k] += rr;
        sqsd[k] += rr * rr;
    }
}

} // namespace rf
