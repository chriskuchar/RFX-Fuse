#include "rf_growtree.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_config.hpp"
#include "rf_vectorized_ops.hpp"
#include "rf_histogram.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rf {

// ============================================================================
// HISTOGRAM-BASED SPLIT FINDING (O(n + bins) vs O(n log n) sorting)
// ============================================================================

// Find best split using histogram for a single feature (classification)
static void histogram_find_best_split_classification(
    const uint8_t* X_binned,           // [nsample × mdim] binned data
    const integer_t* y,                 // [nsample] class labels  
    const real_t* weights,              // [nsample] sample weights (or nullptr)
    const integer_t* node_samples,      // Samples in this node
    integer_t node_count,               // Number of samples in node
    integer_t feature,                  // Feature to evaluate
    integer_t mdim,                     // Number of features
    const FeatureBinInfo& info,         // Bin info for this feature
    integer_t nclass,                   // Number of classes
    real_t& best_crit,                  // Output: best criterion (Gini)
    real_t& best_split_value            // Output: split value
) {
    integer_t n_bins = info.n_bins;
    
    // Build histogram: counts per (bin, class)
    std::vector<real_t> hist(n_bins * nclass, 0.0f);
    
    for (integer_t i = 0; i < node_count; ++i) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        integer_t c = y[s];
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins && c < nclass) {
            hist[bin * nclass + c] += w;
        }
    }
    
    // Scan to find best split
    std::vector<real_t> left_counts(nclass, 0.0f);
    std::vector<real_t> right_counts(nclass, 0.0f);
    real_t total_left = 0.0f, total_right = 0.0f;
    
    // Initialize right with all
    for (integer_t b = 0; b < n_bins; ++b) {
        for (integer_t c = 0; c < nclass; ++c) {
            right_counts[c] += hist[b * nclass + c];
            total_right += hist[b * nclass + c];
        }
    }
    
    best_crit = 1e20f;  // Want to minimize Gini
    best_split_value = info.bin_edges[0];
    
    // Scan left to right
    for (integer_t b = 0; b < n_bins - 1; ++b) {
        // Move bin b from right to left
        for (integer_t c = 0; c < nclass; ++c) {
            real_t count = hist[b * nclass + c];
            left_counts[c] += count;
            right_counts[c] -= count;
            total_left += count;
            total_right -= count;
        }
        
        if (total_left < 1.0f || total_right < 1.0f) continue;
        
        // Compute weighted Gini
        real_t gini_left = 1.0f, gini_right = 1.0f;
        for (integer_t c = 0; c < nclass; ++c) {
            real_t p_left = left_counts[c] / total_left;
            real_t p_right = right_counts[c] / total_right;
            gini_left -= p_left * p_left;
            gini_right -= p_right * p_right;
        }
        
        real_t total = total_left + total_right;
        real_t gini = (total_left * gini_left + total_right * gini_right) / total;
        
        if (gini < best_crit) {
            best_crit = gini;
            // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
            // Without this fix, samples with value == bin_edges[b + 1] go to wrong child
            if (info.is_categorical) {
                best_split_value = static_cast<real_t>(b);
            } else {
                best_split_value = (info.bin_edges[b] + info.bin_edges[b + 1]) / 2.0f;
            }
        }
    }
}

// Find best split using histogram for a single feature (regression)
static void histogram_find_best_split_regression(
    const uint8_t* X_binned,
    const integer_t* y_scaled,          // Scaled regression targets (as integers)
    const real_t* weights,
    const integer_t* node_samples,
    integer_t node_count,
    integer_t feature,
    integer_t mdim,
    const FeatureBinInfo& info,
    real_t y_scale,                     // Scale factor for y values
    real_t& best_crit,                  // Output: best MSE
    real_t& best_split_value
) {
    integer_t n_bins = info.n_bins;
    
    // Build histogram: sum_y, sum_y2, count per bin
    std::vector<real_t> sum_y(n_bins, 0.0f);
    std::vector<real_t> sum_y2(n_bins, 0.0f);
    std::vector<real_t> count(n_bins, 0.0f);
    
    for (integer_t i = 0; i < node_count; ++i) {
        integer_t s = node_samples[i];
        uint8_t bin = X_binned[s * mdim + feature];
        real_t val = static_cast<real_t>(y_scaled[s]) / y_scale;
        real_t w = (weights != nullptr) ? weights[s] : 1.0f;
        
        if (bin < n_bins) {
            sum_y[bin] += val * w;
            sum_y2[bin] += val * val * w;
            count[bin] += w;
        }
    }
    
    // Scan to find best split
    real_t sum_left = 0.0f, sum2_left = 0.0f, cnt_left = 0.0f;
    real_t sum_right = 0.0f, sum2_right = 0.0f, cnt_right = 0.0f;
    
    // Initialize right with all
    for (integer_t b = 0; b < n_bins; ++b) {
        sum_right += sum_y[b];
        sum2_right += sum_y2[b];
        cnt_right += count[b];
    }
    
    best_crit = 1e20f;
    best_split_value = info.bin_edges[0];
    
    // Scan left to right
    for (integer_t b = 0; b < n_bins - 1; ++b) {
        sum_left += sum_y[b];
        sum2_left += sum_y2[b];
        cnt_left += count[b];
        
        sum_right -= sum_y[b];
        sum2_right -= sum_y2[b];
        cnt_right -= count[b];
        
        if (cnt_left < 1.0f || cnt_right < 1.0f) continue;
        
        // Compute MSE
        real_t mean_left = sum_left / cnt_left;
        real_t mean_right = sum_right / cnt_right;
        real_t mse_left = sum2_left / cnt_left - mean_left * mean_left;
        real_t mse_right = sum2_right / cnt_right - mean_right * mean_right;
        
        real_t total = cnt_left + cnt_right;
        real_t mse = (cnt_left * mse_left + cnt_right * mse_right) / total;
        
        if (mse < best_crit) {
            best_crit = mse;
            // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
            // Without this fix, samples with value == bin_edges[b + 1] go to wrong child
            if (info.is_categorical) {
                best_split_value = static_cast<real_t>(b);
            } else {
                best_split_value = (info.bin_edges[b] + info.bin_edges[b + 1]) / 2.0f;
            }
        }
    }
}

// ============================================================================
// CPU HELPER FUNCTIONS
// ============================================================================

// CPU implementation of findbestsplit
void cpu_findbestsplit(const real_t* tclasspop, const real_t* win, const integer_t* cl,
                       const integer_t* jinbag, const integer_t* cat, const integer_t* amat,
                       const integer_t* ties, integer_t mdim, integer_t nsample,
                       integer_t nclass, integer_t maxcat, integer_t ndstart, integer_t ndstop,
                       integer_t mtry, integer_t ncmax, integer_t ncsplit, integer_t iseed,
                       integer_t* igoleft, integer_t& msplit, integer_t& nbest,
                       integer_t& jstat, real_t* tclasscat, real_t* tmpclass,
                       real_t* tcat, integer_t* icat, real_t* wr, real_t* wl,
                       integer_t* igl, integer_t* itmp, MT19937& rng) {

    // Compute initial Gini using vectorized operations
    VectorizedGiniCalculator gini_calc;
    real_t pno = 0.0f;
    real_t pdo = 0.0f;
    
    // Use vectorized operations for initial Gini calculation
    // Convert tclasspop to integer counts for vectorized operations
    std::vector<integer_t> class_counts(nclass);
    for (integer_t j = 0; j < nclass; ++j) {
        class_counts[j] = static_cast<integer_t>(tclasspop[j]);
        pno += tclasspop[j] * tclasspop[j];
        pdo += tclasspop[j];
    }

    real_t critmax = -2.0f;
    jstat = 0;
    msplit = 0;
    nbest = 0;

    // Select mtry variables randomly without replacement
    std::vector<integer_t> mv_idx(mtry);
    std::vector<integer_t> used(mdim, 0);
    integer_t mtry_count = 0;

    while (mtry_count < mtry) {
        real_t rand_val = rng.randomu();
        integer_t idx = static_cast<integer_t>(mdim * rand_val);
        if (idx >= mdim) idx = mdim - 1;

        if (used[idx] == 0) {
            mv_idx[mtry_count] = idx;
            used[idx] = 1;
            ++mtry_count;
        }
    }

    // Evaluate each selected variable
    for (integer_t mv = 0; mv < mtry; ++mv) {
        integer_t mvar = mv_idx[mv];

        // Binary features (cat == 2) use quantitative path for better sparsity handling
        if (cat[mvar] <= 2) {
            // Quantitative variable (including binary) - using proven algorithm
            real_t rrn = pno;
            real_t rrd = pdo;
            real_t rln = 0.0f;
            real_t rld = 0.0f;

            for (integer_t j = 0; j < nclass; ++j) {
                wl[j] = 0.0f;
                wr[j] = tclasspop[j];
            }

            // Use vectorized operations for the main loop
            for (integer_t ii = ndstart; ii < ndstop; ++ii) {
                integer_t nc = amat[mvar + ii * mdim];
                integer_t ncnext = (ii + 1 < ndstop) ? amat[mvar + (ii + 1) * mdim] : nc;
                real_t u = win[nc];
                integer_t k = cl[nc];

                // Vectorized-friendly Gini calculation
                rln += u * (u + 2.0f * wl[k]);
                rrn += u * (u - 2.0f * wr[k]);
                rld += u;
                rrd -= u;
                wl[k] += u;
                wr[k] -= u;

                if (ii + 1 < ndstop && ties[mvar + nc * mdim] < ties[mvar + ncnext * mdim]) {
                    if (std::min(rrd, rld) > 1.0e-5f) {
                        // Use vectorized criterion calculation
                        std::vector<integer_t> left_counts(nclass);
                        std::vector<integer_t> right_counts(nclass);
                        for (integer_t j = 0; j < nclass; ++j) {
                            left_counts[j] = static_cast<integer_t>(wl[j]);
                            right_counts[j] = static_cast<integer_t>(wr[j]);
                        }
                        real_t crit = VectorizedGiniCalculator::calculate_gini(left_counts.data(), right_counts.data(), 
                                                                              static_cast<integer_t>(rld), static_cast<integer_t>(rrd), nclass);
                        
                        if (crit > critmax) {
                            nbest = ii;
                            critmax = crit;
                            msplit = mvar;
                        }
                    }
                }
            }

        } else {
            // Categorical variable
            integer_t numcat = cat[mvar];

            for (integer_t j = 0; j < nclass; ++j) {
                for (integer_t k = 0; k < numcat; ++k) {
                    tclasscat[j + k * nclass] = 0.0f;
                }
            }

            for (integer_t ii = ndstart; ii < ndstop; ++ii) {
                integer_t nc = jinbag[ii];
                integer_t k = amat[mvar + nc * mdim];
                if (k >= 0 && k < numcat) {
                    tclasscat[cl[nc] + k * nclass] += win[nc];
                }
            }

            integer_t nnz = 0;
            for (integer_t k = 0; k < numcat; ++k) {
                tcat[k] = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    tcat[k] += tclasscat[j + k * nclass];
                }
                if (tcat[k] > 0.0f) {
                    icat[nnz] = k;
                    ++nnz;
                }
            }

            if (nnz > 1) {
                integer_t nhit = 0;
                if (numcat < ncmax) {
                    cpu_catmax(tclasscat, tclasspop, pdo, nnz, numcat, nclass,
                          igl, itmp, tmpclass, tcat, icat, critmax, igoleft, nhit);
                } else {
                    cpu_catmaxr(tclasscat, tclasspop, pdo, numcat, nclass, igl,
                           ncsplit, tmpclass, critmax, igoleft, nhit);
                }

                if (nhit == 1) {
                    msplit = mvar;
                }
            }
        }
    }

    if (critmax < -1.0f) {
        jstat = 1;
    }
}

// CPU implementation of movedata
void cpu_movedata(integer_t mdim, integer_t nsample, integer_t msplit,
                  integer_t nbest, integer_t ndstart, integer_t ndstop,
                  integer_t maxcat, const integer_t* cat, const integer_t* igoleft,
                  integer_t* amat, integer_t* jinbag, integer_t& ndstopl,
                  integer_t* idmove, integer_t* itemp) {
    // Column-major: amat(mdim, nsample)

    // Compute idmove = indicator of case nos. going left
    if (cat[msplit] <= 2) {
        // Quantitative variable (including binary features)
        ndstopl = nbest;
        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            integer_t nc = amat[msplit + ii * mdim];
            if (ii <= ndstopl) {
                idmove[nc] = 1;
            } else {
                idmove[nc] = 0;
            }
            jinbag[ii] = nc;
        }
    } else {
        // Categorical variable
        ndstopl = ndstart - 1;
        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            integer_t nc = jinbag[ii];
            if (igoleft[amat[msplit + nc * mdim]] == 1) {
                idmove[nc] = 1;
                ++ndstopl;
            } else {
                idmove[nc] = 0;
            }
        }

        integer_t il = ndstart;
        integer_t ir = ndstopl + 1;
        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            integer_t nc = jinbag[ii];
            if (idmove[nc] == 1) {
                itemp[il] = nc;
                ++il;
            } else {
                itemp[ir] = nc;
                ++ir;
            }
        }

        for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
            jinbag[ii] = itemp[ii];
        }
    }

    // Shift case nos. right and left for numerical variables using vectorized operations
    for (integer_t ms = 0; ms < mdim; ++ms) {
        if (cat[ms] <= 2) {
            // Use vectorized data movement for better performance (includes binary features)
            integer_t il = ndstart;
            integer_t ir = ndstopl + 1;
            
            // Vectorized partitioning
            std::vector<integer_t> left_samples, right_samples;
            left_samples.reserve(ndstop - ndstart + 1);
            right_samples.reserve(ndstop - ndstart + 1);
            
            for (integer_t ii = ndstart; ii <= ndstop; ++ii) {
                integer_t nc = amat[ms + ii * mdim];
                if (idmove[nc] == 1) {
                    left_samples.push_back(nc);
                } else {
                    right_samples.push_back(nc);
                }
            }
            
            // Copy back to amat using vectorized operations
            std::copy(left_samples.begin(), left_samples.end(), &amat[ms + ndstart * mdim]);
            std::copy(right_samples.begin(), right_samples.end(), &amat[ms + (ndstopl + 1) * mdim]);
        }
    }
}

// CPU implementation of catmax (categorical variable optimization)
// Breiman-Cutler: Exhaustive search over all 2^(nnz-1) - 1 binary partitions
// For small number of categories (nnz <= ncmax), try all possible subsets
void cpu_catmax(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
            integer_t nnz, integer_t numcat, integer_t nclass,
            integer_t* igl, integer_t* itmp, real_t* tmpclass,
            const real_t* tcat, const integer_t* icat, real_t& critmax,
            integer_t* igoleft, integer_t& nhit) {
    
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    std::vector<integer_t> best_igoleft(numcat, 0);
    
    // For binary categorical (nnz=2), there's only one meaningful split
    // For larger nnz, try all 2^(nnz-1) - 1 possible partitions
    // (excluding empty set and full set which are equivalent)
    integer_t max_subsets = (1 << (nnz - 1));  // 2^(nnz-1) subsets (half, due to symmetry)
    
    for (integer_t subset = 1; subset < max_subsets; ++subset) {
        // Set igoleft based on subset bits
        for (integer_t k = 0; k < numcat; ++k) {
            igoleft[k] = 0;
        }
        
        for (integer_t i = 0; i < nnz; ++i) {
            if ((subset >> i) & 1) {
                igoleft[icat[i]] = 1;
            }
        }
        
        // Calculate class counts for left and right children
        real_t sum_left = 0.0f;
        real_t sum_right = 0.0f;
        real_t sum_sq_left = 0.0f;
        real_t sum_sq_right = 0.0f;
        
        for (integer_t j = 0; j < nclass; ++j) {
            tmpclass[j] = 0.0f;  // Left class count
            real_t right_count = 0.0f;
            
            for (integer_t k = 0; k < numcat; ++k) {
                if (igoleft[k] == 1) {
                    tmpclass[j] += tclasscat[j + k * nclass];
                } else {
                    right_count += tclasscat[j + k * nclass];
                }
            }
            
            sum_left += tmpclass[j];
            sum_right += right_count;
            sum_sq_left += tmpclass[j] * tmpclass[j];
            sum_sq_right += right_count * right_count;
        }
        
        // Gini criterion: sum of squared class proportions
        // Higher is better (more pure nodes)
        if (sum_left > 1.0f && sum_right > 1.0f) {
            real_t crit = sum_sq_left / sum_left + sum_sq_right / sum_right;
            
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
                // Save best igoleft
                for (integer_t k = 0; k < numcat; ++k) {
                    best_igoleft[k] = igoleft[k];
                }
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
        // Copy best partition back to igoleft
        for (integer_t k = 0; k < numcat; ++k) {
            igoleft[k] = best_igoleft[k];
        }
    }
}

// CPU implementation of catmaxr (categorical variable optimization for large categories)
// Breiman-Cutler: Random sampling approach when exhaustive search is too expensive
void cpu_catmaxr(const real_t* tclasscat, const real_t* tclasspop, real_t pdo,
             integer_t numcat, integer_t nclass, integer_t* igl,
             integer_t ncsplit, real_t* tmpclass,
             real_t& critmax, integer_t* igoleft, integer_t& nhit) {
    
    real_t best_crit = critmax;
    integer_t best_nhit = 0;
    std::vector<integer_t> best_igoleft(numcat, 0);
    
    // Random sampling approach for large categories
    // Try ncsplit random partitions
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> coin(0, 1);
    
    for (integer_t split = 0; split < ncsplit; ++split) {
        // Generate random partition
        integer_t left_count = 0;
        for (integer_t k = 0; k < numcat; ++k) {
            igoleft[k] = coin(rng);
            left_count += igoleft[k];
        }
        
        // Ensure non-trivial split (at least one on each side)
        if (left_count == 0 || left_count == numcat) {
            igoleft[0] = 1;  // Force at least one left
            igoleft[numcat > 1 ? numcat - 1 : 0] = 0;  // Force at least one right if possible
        }
        
        // Calculate class counts for left and right children using Gini criterion
        real_t sum_left = 0.0f;
        real_t sum_right = 0.0f;
        real_t sum_sq_left = 0.0f;
        real_t sum_sq_right = 0.0f;
        
        for (integer_t j = 0; j < nclass; ++j) {
            tmpclass[j] = 0.0f;  // Left class count
            real_t right_count = 0.0f;
            
            for (integer_t k = 0; k < numcat; ++k) {
                if (igoleft[k] == 1) {
                    tmpclass[j] += tclasscat[j + k * nclass];
                } else {
                    right_count += tclasscat[j + k * nclass];
                }
            }
            
            sum_left += tmpclass[j];
            sum_right += right_count;
            sum_sq_left += tmpclass[j] * tmpclass[j];
            sum_sq_right += right_count * right_count;
        }
        
        // Gini criterion: sum of squared class proportions (higher = more pure)
        if (sum_left > 1.0f && sum_right > 1.0f) {
            real_t crit = sum_sq_left / sum_left + sum_sq_right / sum_right;
            
            if (crit > best_crit) {
                best_crit = crit;
                best_nhit = 1;
                // Save best igoleft
                for (integer_t k = 0; k < numcat; ++k) {
                    best_igoleft[k] = igoleft[k];
                }
            }
        }
    }
    
    if (best_nhit == 1) {
        critmax = best_crit;
        nhit = 1;
        // Copy best partition back to igoleft
        for (integer_t k = 0; k < numcat; ++k) {
            igoleft[k] = best_igoleft[k];
        }
    }
}

// CPU implementation of createnode
void createnode(const real_t* win, integer_t inode, integer_t nsample, integer_t nclass,
                integer_t minndsize, integer_t* nodestart, integer_t* nodestop,
                const integer_t* jinbag, const integer_t* cl, real_t* classpop,
                integer_t* nodestatus) {
    
    // Calculate class population for this node
    for (integer_t j = 0; j < nclass; ++j) {
        classpop[j + inode * nclass] = 0.0f;
    }
    
    for (integer_t n = nodestart[inode]; n <= nodestop[inode]; ++n) {
        integer_t sample_idx = jinbag[n];
        integer_t class_idx = cl[sample_idx];
        if (class_idx >= 0 && class_idx < nclass) {
            classpop[class_idx + inode * nclass] += win[sample_idx];
        }
    }
    
    // Check if node should be terminal
    integer_t node_size = nodestop[inode] - nodestart[inode] + 1;
    if (node_size < minndsize) {
        nodestatus[inode] = -1;  // Terminal
    } else {
        nodestatus[inode] = 2;   // Non-terminal
    }
}

// ============================================================================
// CPU TREE GROWING IMPLEMENTATION
// ============================================================================

// CPU fallback implementation of tree growing
// This is a pure CPU implementation that doesn't depend on CUDA
void cpu_growtree(
    const real_t* x, const real_t* win, const integer_t* cl,
    const integer_t* cat, const integer_t* ties, const integer_t* nin,
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
    integer_t iseed, integer_t ncsplit, integer_t ncmax,
    integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* nodepred, real_t* xbestsplit,
    integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
    integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
    integer_t* nodextr, integer_t& nnode, integer_t* idmove,
    integer_t* igoleft, integer_t* igl, integer_t* nodestart,
    integer_t* nodestop, integer_t* itemp, integer_t* itmp,
    integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
    real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass,
    const HistogramData* histogram_data) {
    
    // Check if histogram-based split finding should be used
    const bool use_histogram = (histogram_data != nullptr && histogram_data->is_valid);

    // Initialize RNG
    MT19937 rng;
    rng.sgrnd(iseed);

    // Initialize arrays to zero
    zerv(nodestatus, maxnode);
    zerv(nodestart, maxnode);
    zerv(nodestop, maxnode);
    zerv(nodeclass, maxnode);
    zerv(bestvar, maxnode);
    zerm(treemap, 2, maxnode);
    zerm(catgoleft, maxcat, maxnode);

    // Initialize root node
    nodestart[0] = 0;  // 0-based
    nodestop[0] = ninbag - 1;

    for (integer_t j = 0; j < nclass; ++j) {
        classpop[j] = 0.0f;  // Column-major: classpop(nclass, maxnode)
    }

    for (integer_t n = 0; n < nsample; ++n) {
        // Use 0-based class labels directly (already converted in setup_classification_task)
        integer_t class_idx = cl[n];
        if (class_idx >= 0 && class_idx < nclass) {
            classpop[class_idx] += win[n];
        }
    }

    nodestatus[0] = 2;  // Non-terminal
    nnode = 1;

    // ========================================================================
    // O(n) SAMPLE TRACKING - Track which node each sample is in (like sklearn!)
    // This eliminates O(depth) traversal per sample per node = HUGE speedup!
    // ========================================================================
    std::vector<integer_t> sample_node(ninbag, 0);  // All samples start at root (node 0)
    std::vector<integer_t> sample_idx_map(ninbag);  // Map from local index to sample_idx
    integer_t valid_samples = 0;
    for (integer_t i = 0; i < ninbag; ++i) {
        if (jinbag[i] >= 0 && jinbag[i] < nsample) {
            sample_idx_map[valid_samples++] = jinbag[i];
        }
    }
    sample_idx_map.resize(valid_samples);
    sample_node.resize(valid_samples);
    std::fill(sample_node.begin(), sample_node.end(), 0);  // All at root

    // Pre-allocate node_samples vector (reused for each node)
    std::vector<integer_t> node_samples;
    node_samples.reserve(valid_samples);

    // Main loop - grow tree
    for (integer_t kgrow = 0; kgrow < nnode; ++kgrow) {
        if (nodestatus[kgrow] != 2) continue;

        // ====================================================================
        // O(n) NODE SAMPLE COLLECTION - Just filter by sample_node[i] == kgrow
        // ====================================================================
        node_samples.clear();
        for (integer_t i = 0; i < valid_samples; ++i) {
            if (sample_node[i] == kgrow) {
                node_samples.push_back(sample_idx_map[i]);
            }
        }
        
        integer_t node_sample_count = node_samples.size();
        
        // Check if node should be split (following original algorithm)
        // For regression: need at least 2*minndsize to split (minndsize in each child)
        // For classification: need at least minndsize
        integer_t min_size_to_split = (nclass == 1) ? 2 * minndsize : minndsize;
        if (node_sample_count < min_size_to_split) {
            // Node is too small, make it terminal
            nodestatus[kgrow] = -1;
            continue;
        }
        
        // Calculate class distribution for this node (using casewise weights)
        std::vector<real_t> class_counts(nclass, 0.0f);
        for (integer_t i = 0; i < node_sample_count; ++i) {
            integer_t sample_idx = node_samples[i];
            integer_t class_label = cl[sample_idx];
            if (class_label >= 0 && class_label < nclass) {
                class_counts[class_label] += win[sample_idx];  // Use casewise weight
            }
        }
        
        // Check if node is pure (all samples same class)
        integer_t non_zero_classes = 0;
        integer_t majority_class = 0;
        real_t max_count = 0.0f;
        
        for (integer_t j = 0; j < nclass; ++j) {
            if (class_counts[j] > 0.0f) {
                non_zero_classes++;
                if (class_counts[j] > max_count) {
                    max_count = class_counts[j];
                    majority_class = j;
                }
            }
        }
        
        // ====================================================================
        // REGRESSION: Pre-compute weighted sums in ONE pass (no extra vector allocation!)
        // ====================================================================
        const real_t y_scale = 1000.0f;
        real_t total_sum_y = 0.0f;
        real_t total_sum_y2 = 0.0f;
        real_t total_weight = 0.0f;
        
        if (nclass == 1) {
            // Compute weighted sums - using win for casewise weighting
            for (integer_t i = 0; i < node_sample_count; ++i) {
                integer_t idx = node_samples[i];
                real_t y = static_cast<real_t>(cl[idx]) / y_scale;
                real_t w = win[idx];  // Casewise weight
                total_sum_y += y * w;
                total_sum_y2 += y * y * w;
                total_weight += w;
            }
            
            // Variance check (using weighted sums)
            if (total_weight > 1.0e-5f) {
                real_t mean_y = total_sum_y / total_weight;
                real_t variance = (total_sum_y2 / total_weight) - mean_y * mean_y;
                if (variance < 1.0e-6f) {
                    nodestatus[kgrow] = -1;
                    nodeclass[kgrow] = 0;
                    continue;
                }
            } else {
                nodestatus[kgrow] = -1;
                nodeclass[kgrow] = 0;
                continue;
            }
        } else if (non_zero_classes <= 1) {
            nodestatus[kgrow] = -1;
            nodeclass[kgrow] = majority_class;
            continue;
        }
        
        // Find best split
        real_t best_crit = (nclass == 1) ? 1.0e20f : -2.0f;
        integer_t best_var = 0;
        integer_t jstat = 0;
        real_t best_split_value = 0.0f;
        
        // Initial criterion
        real_t crit0;
        if (nclass == 1) {
            // For casewise: use total_weight (sum of bootstrap frequencies)
            // For non-casewise: total_weight == node_sample_count (all weights are 1.0)
            real_t mean_y = total_sum_y / total_weight;
            crit0 = (total_sum_y2 / total_weight) - mean_y * mean_y;
        } else {
            real_t pno = 0.0f, pdo = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                pno += class_counts[j] * class_counts[j];
                pdo += class_counts[j];
            }
            crit0 = pno / pdo;
        }
        
        // ====================================================================
        // REGRESSION: Histogram-based OR sorting-based split finding
        // ====================================================================
        if (nclass == 1) {
            if (use_histogram) {
                // ============================================================
                // HISTOGRAM-BASED SPLIT FINDING FOR REGRESSION
                // ============================================================
                for (integer_t mv = 0; mv < mtry; ++mv) {
                    integer_t mvar = static_cast<integer_t>(rng.randomu() * mdim);
                    if (mvar >= mdim) mvar = mdim - 1;
                    
                    real_t feat_crit = 1e20f;
                    real_t feat_split = 0.0f;
                    
                    histogram_find_best_split_regression(
                        histogram_data->X_binned.data(),
                        cl,  // y_scaled (regression targets stored in cl)
                        win,  // weights (casewise weighting)
                        node_samples.data(),
                        node_sample_count,
                        mvar,
                        mdim,
                        histogram_data->feature_info[mvar],
                        y_scale,
                        feat_crit,
                        feat_split
                    );
                    
                    if (feat_crit < best_crit) {
                        best_crit = feat_crit;
                        best_var = mvar;
                        best_split_value = feat_split;
                        jstat = 0;
                    }
                }
            } else {
                // ============================================================
                // SORTING-BASED SPLIT FINDING FOR REGRESSION
                // ============================================================
                // Select mtry random variables first (avoid RNG in parallel loop)
                std::vector<integer_t> mvar_list(mtry);
                for (integer_t mv = 0; mv < mtry; ++mv) {
                    mvar_list[mv] = static_cast<integer_t>(rng.randomu() * mdim);
                    if (mvar_list[mv] >= mdim) mvar_list[mv] = mdim - 1;
                }
                
                // Thread-local best results
                real_t local_best_crit = best_crit;
                integer_t local_best_var = best_var;
                real_t local_best_split = best_split_value;
                integer_t local_jstat = jstat;
                
                // NOTE: Don't use reduction(min:local_best_crit) - it conflicts with critical section
                // The critical section properly synchronizes local_best_crit with local_best_var/split
                #ifdef _OPENMP
                #pragma omp parallel for if(node_sample_count > 100)
                #endif
                for (integer_t mv = 0; mv < mtry; ++mv) {
                    integer_t mvar = mvar_list[mv];
                    
                    // Thread-local sorted samples
                    std::vector<std::pair<real_t, integer_t>> sorted_samples;
                    sorted_samples.reserve(node_sample_count);
                    for (integer_t i = 0; i < node_sample_count; ++i) {
                        integer_t idx = node_samples[i];
                        sorted_samples.push_back({x[idx * mdim + mvar], idx});
                    }
                    std::sort(sorted_samples.begin(), sorted_samples.end());
                    
                    real_t sum_y_left = 0.0f, sum_y2_left = 0.0f, cnt_left = 0.0f;
                    real_t sum_y_right = total_sum_y, sum_y2_right = total_sum_y2;
                    real_t cnt_right = total_weight;  // Use total weight instead of count
                    
                    real_t thread_best_crit = 1.0e20f;
                    real_t thread_best_split = 0.0f;
                    
                    for (integer_t ii = 0; ii < node_sample_count - 1; ++ii) {
                        if (sorted_samples[ii].first >= sorted_samples[ii + 1].first) continue;
                        
                        integer_t sample_idx = sorted_samples[ii].second;
                        real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;
                        real_t w = win[sample_idx];  // Use casewise weight
                        real_t y_w = y_val * w;
                        real_t y2_w = y_val * y_val * w;
                        
                        sum_y_left += y_w; sum_y2_left += y2_w; cnt_left += w;
                        sum_y_right -= y_w; sum_y2_right -= y2_w; cnt_right -= w;
                        
                        if (cnt_left < 1.0e-5f || cnt_right < 1.0e-5f) continue;
                        
                        real_t inv_l = 1.0f / cnt_left, inv_r = 1.0f / cnt_right;
                        real_t mean_l = sum_y_left * inv_l, mean_r = sum_y_right * inv_r;
                        real_t mse = (cnt_left * (sum_y2_left * inv_l - mean_l * mean_l) +
                                      cnt_right * (sum_y2_right * inv_r - mean_r * mean_r)) / total_weight;
                        
                        if (mse < thread_best_crit) {
                            thread_best_crit = mse;
                            thread_best_split = sorted_samples[ii].first;
                        }
                    }
                    
                    // Update global best (critical section)
                    #ifdef _OPENMP
                    #pragma omp critical
                    #endif
                    {
                        if (thread_best_crit < local_best_crit) {
                            local_best_crit = thread_best_crit;
                            local_best_var = mvar;
                            local_best_split = thread_best_split;
                            local_jstat = 0;
                        }
                    }
                }
                
                best_crit = local_best_crit;
                best_var = local_best_var;
                best_split_value = local_best_split;
                jstat = local_jstat;
            }
        }
        // ====================================================================
        // CLASSIFICATION: Histogram-based OR sorting-based split finding
        // ====================================================================
        else {
            if (use_histogram) {
                // ============================================================
                // HISTOGRAM-BASED SPLIT FINDING (O(n + bins) per feature)
                // ============================================================
                for (integer_t mv = 0; mv < mtry; ++mv) {
                    integer_t mvar = static_cast<integer_t>(rng.randomu() * mdim);
                    if (mvar >= mdim) mvar = mdim - 1;
                    
                    real_t feat_crit = 1e20f;
                    real_t feat_split = 0.0f;
                    
                    histogram_find_best_split_classification(
                        histogram_data->X_binned.data(),
                        cl,
                        win,  // weights (casewise weighting)
                        node_samples.data(),
                        node_sample_count,
                        mvar,
                        mdim,
                        histogram_data->feature_info[mvar],
                        nclass,
                        feat_crit,
                        feat_split
                    );
                    
                    // Convert Gini (lower is better) to criterion (higher is better)
                    // Note: histogram returns Gini directly, need to invert for comparison
                    real_t converted_crit = 2.0f - feat_crit;  // Approximate conversion
                    
                    if (converted_crit > best_crit) {
                        best_crit = converted_crit;
                        best_var = mvar;
                        best_split_value = feat_split;
                        jstat = 0;
                    }
                }
            } else {
                // ============================================================
                // SORTING-BASED SPLIT FINDING (O(n log n) per feature)
                // ============================================================
                std::vector<real_t> wl(nclass, 0.0f), wr(nclass, 0.0f);
                std::vector<std::pair<real_t, integer_t>> sorted_samples;
                sorted_samples.reserve(node_sample_count);
                
                real_t base_pno = 0.0f, base_pdo = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    base_pno += class_counts[j] * class_counts[j];
                    base_pdo += class_counts[j];
                }
                
                for (integer_t mv = 0; mv < mtry; ++mv) {
                    integer_t mvar = static_cast<integer_t>(rng.randomu() * mdim);
                    if (mvar >= mdim) mvar = mdim - 1;
                    
                    real_t rrn = base_pno, rrd = base_pdo, rln = 0.0f, rld = 0.0f;
                    std::fill(wl.begin(), wl.end(), 0.0f);
                    for (integer_t j = 0; j < nclass; ++j) wr[j] = class_counts[j];
                    
                    sorted_samples.clear();
                    for (integer_t i = 0; i < node_sample_count; ++i) {
                        integer_t sample_idx = node_samples[i];
                        sorted_samples.push_back({x[sample_idx * mdim + mvar], sample_idx});
                    }
                    std::sort(sorted_samples.begin(), sorted_samples.end());
                    
                    for (integer_t ii = 0; ii < node_sample_count - 1; ++ii) {
                        if (sorted_samples[ii].first >= sorted_samples[ii + 1].first) continue;
                        
                        integer_t nc = sorted_samples[ii].second;
                        integer_t k = cl[nc];
                        real_t u = win[nc];  // Use casewise weight instead of 1.0f
                        rln += u * (u + 2.0f * wl[k]);
                        rrn += u * (u - 2.0f * wr[k]);
                        rld += u; rrd -= u;
                        wl[k] += u; wr[k] -= u;
                        
                        if (std::min(rrd, rld) > 1.0e-5f) {
                            real_t crit = (rln / rld) + (rrn / rrd);
                            if (crit > best_crit) {
                                best_crit = crit;
                                best_var = mvar;
                                best_split_value = sorted_samples[ii].first;
                                jstat = 0;
                            }
                        }
                    }
                }
            }
        }
        
        // Check if a good split was found
        if (nclass == 1) {
            // Regression: check if MSE reduction is significant
            real_t mse_reduction = crit0 - best_crit;
            if (mse_reduction < 1.0e-6f) {
                jstat = 1;  // No significant improvement
            }
        } else {
            // Classification: check if Gini improved
            if (best_crit < -1.0f) {
                jstat = 1;  // No good split found
            }
        }
        
        // Create split if improvement found
        if (jstat == 0 && ((nclass == 1 && best_crit < crit0) || (nclass > 1 && best_crit > -1.0f))) {
            // Create children
            if (nnode + 2 <= maxnode) {
                integer_t left_child = nnode;
                integer_t right_child = nnode + 1;
                
                // Set up split
                nodestatus[kgrow] = 1;  // Split node
                bestvar[kgrow] = best_var;
                
                // Use the stored split value directly (no vector lookup needed!)
                xbestsplit[kgrow] = best_split_value;
                
                // Set up children
                treemap[kgrow * 2] = left_child;
                treemap[kgrow * 2 + 1] = right_child;
                
                nodestatus[left_child] = 2;   // Active
                nodestatus[right_child] = 2;  // Active
                
                nnode += 2;
                
                // ============================================================
                // O(n) UPDATE: Move samples to left or right child
                // ============================================================
                integer_t split_var_upd = bestvar[kgrow];
                real_t split_point_upd = xbestsplit[kgrow];
                for (integer_t i = 0; i < valid_samples; ++i) {
                    if (sample_node[i] == kgrow) {
                        real_t val = x[sample_idx_map[i] * mdim + split_var_upd];
                        sample_node[i] = (val <= split_point_upd) ? left_child : right_child;
                    }
                }
            } else {
                // No more nodes available, make terminal
                nodestatus[kgrow] = -1;  // Terminal node
                nodeclass[kgrow] = majority_class;
            }
        } else {
            // No good split found, make terminal
            nodestatus[kgrow] = -1;  // Terminal node
            nodeclass[kgrow] = majority_class;
        }
    }

    // ========================================================================
    // CLASSIFICATION ONLY: Finalize terminal nodes with majority class voting
    // Uses O(n) sample_node lookup instead of O(n × depth) tree traversal!
    // ========================================================================
    if (nclass > 1) {
        for (integer_t node = 0; node < nnode; ++node) {
            if (nodestatus[node] == -1) {
                // Terminal node - count classes using sample_node (O(n) not O(n×depth)!)
                std::vector<real_t> class_counts(nclass, 0.0f);
                
                for (integer_t i = 0; i < valid_samples; ++i) {
                    if (sample_node[i] == node) {
                        integer_t sample_idx = sample_idx_map[i];
                        integer_t class_label = cl[sample_idx];
                        if (class_label >= 0 && class_label < nclass) {
                            class_counts[class_label] += win[sample_idx];  // Use casewise weight
                        }
                    }
                }
                
                // Find majority class
                integer_t maj_class = 0;
                real_t max_count = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    if (class_counts[j] > max_count) {
                        max_count = class_counts[j];
                        maj_class = j;
                    }
                }
                nodeclass[node] = maj_class;
            }
        }
    }

    // ========================================================================
    // REGRESSION: O(n) tnodewt computation using sample_node tracking!
    // ========================================================================
    // NEW (replace entire block):
    if (nclass == 1 && nodepred != nullptr) {
        const real_t y_scale = 1000.0f;
        std::vector<real_t> node_sum_y(nnode, 0.0f);
        std::vector<real_t> node_sum_weight(nnode, 0.0f);  // For nodepred
        std::vector<real_t> node_count(nnode, 0.0f);       // For tnodewt
        
        for (integer_t i = 0; i < valid_samples; ++i) {
            integer_t terminal_node = sample_node[i];
            integer_t sample_idx = sample_idx_map[i];
            real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;
            real_t bootstrap_freq = static_cast<real_t>(nin[sample_idx]);
            
            if (g_config.use_casewise) {
                node_sum_y[terminal_node] += y_val * bootstrap_freq;
                node_sum_weight[terminal_node] += bootstrap_freq;
                node_count[terminal_node] += 1.0f;
            } else {
                node_sum_y[terminal_node] += y_val;
                node_sum_weight[terminal_node] += 1.0f;
                node_count[terminal_node] += 1.0f;
            }
        }
        
        for (integer_t node = 0; node < nnode; ++node) {
            if (nodestatus[node] == -1) {
                // nodepred: weighted mean y (prediction)
                nodepred[node] = (node_sum_weight[node] > 0.0f) ? 
                    (node_sum_y[node] / node_sum_weight[node]) : 0.0f;
                // tnodewt: mean bootstrap weight (for OOB weighting)
                tnodewt[node] = (node_count[node] > 0.0f) ? 
                    (node_sum_weight[node] / node_count[node]) : 0.0f;
            }
        }
    }

    // Set jtr and nodextr (not used in current algorithm, but initialize)
    for (integer_t n = 0; n < nsample; ++n) {
        jtr[n] = 0;
        nodextr[n] = 0;
    }
}

// ============================================================================
// NATIVE SPARSE CSR TREE GROWING
// ============================================================================
// This version uses SparseMatrixCSR directly - NO DENSE CONVERSION
// Memory: O(nnz) instead of O(n × m)
// For recommender systems with sparse user-item matrices
// ============================================================================

void cpu_growtree_sparse(
    const SparseMatrixCSR& x_sparse, const real_t* win, const integer_t* cl,
    const integer_t* cat, const integer_t* ties, const integer_t* nin,
    integer_t mdim, integer_t nsample, integer_t nclass, integer_t maxnode,
    integer_t minndsize, integer_t mtry, integer_t ninbag, integer_t maxcat,
    integer_t iseed, integer_t ncsplit, integer_t ncmax,
    integer_t* amat, integer_t* jinbag, real_t* tnodewt, real_t* nodepred, real_t* xbestsplit,
    integer_t* nodestatus, integer_t* bestvar, integer_t* treemap,
    integer_t* catgoleft, integer_t* nodeclass, integer_t* jtr,
    integer_t* nodextr, integer_t& nnode, integer_t* idmove,
    integer_t* igoleft, integer_t* igl, integer_t* nodestart,
    integer_t* nodestop, integer_t* itemp, integer_t* itmp,
    integer_t* icat, real_t* tcat, real_t* classpop, real_t* wr,
    real_t* wl, real_t* tclasscat, real_t* tclasspop, real_t* tmpclass) {

    // Initialize RNG
    MT19937 rng;
    rng.sgrnd(iseed);

    // Initialize arrays to zero
    zerv(nodestatus, maxnode);
    zerv(nodestart, maxnode);
    zerv(nodestop, maxnode);
    zerv(nodeclass, maxnode);
    zerv(bestvar, maxnode);
    zerm(treemap, 2, maxnode);
    zerm(catgoleft, maxcat, maxnode);

    // Initialize root node
    nodestart[0] = 0;
    nodestop[0] = ninbag - 1;

    for (integer_t j = 0; j < nclass; ++j) {
        classpop[j] = 0.0f;
    }

    for (integer_t n = 0; n < nsample; ++n) {
        integer_t class_idx = cl[n];
        if (class_idx >= 0 && class_idx < nclass) {
            classpop[class_idx] += win[n];
        }
    }

    nodestatus[0] = 2;  // Non-terminal
    nnode = 1;

    // Sample tracking (O(n) algorithm like sklearn)
    std::vector<integer_t> sample_node(ninbag, 0);
    std::vector<integer_t> sample_idx_map(ninbag);
    integer_t valid_samples = 0;
    for (integer_t i = 0; i < ninbag; ++i) {
        if (jinbag[i] >= 0 && jinbag[i] < nsample) {
            sample_idx_map[valid_samples++] = jinbag[i];
        }
    }
    sample_idx_map.resize(valid_samples);
    sample_node.resize(valid_samples);
    std::fill(sample_node.begin(), sample_node.end(), 0);

    std::vector<integer_t> node_samples;
    node_samples.reserve(valid_samples);

    // Main loop - grow tree
    for (integer_t kgrow = 0; kgrow < nnode; ++kgrow) {
        if (nodestatus[kgrow] != 2) continue;

        // Collect samples for this node
        node_samples.clear();
        for (integer_t i = 0; i < valid_samples; ++i) {
            if (sample_node[i] == kgrow) {
                node_samples.push_back(sample_idx_map[i]);
            }
        }
        
        integer_t node_sample_count = node_samples.size();
        integer_t min_size_to_split = (nclass == 1) ? 2 * minndsize : minndsize;
        if (node_sample_count < min_size_to_split) {
            nodestatus[kgrow] = -1;
            continue;
        }
        
        // Calculate class distribution (using casewise weights)
        std::vector<real_t> class_counts(nclass, 0.0f);
        for (integer_t i = 0; i < node_sample_count; ++i) {
            integer_t sample_idx = node_samples[i];
            integer_t class_label = cl[sample_idx];
            if (class_label >= 0 && class_label < nclass) {
                class_counts[class_label] += win[sample_idx];  // Use casewise weight
            }
        }
        
        // Check purity
        integer_t non_zero_classes = 0;
        integer_t majority_class = 0;
        real_t max_count = 0.0f;
        
        for (integer_t j = 0; j < nclass; ++j) {
            if (class_counts[j] > 0.0f) {
                non_zero_classes++;
                if (class_counts[j] > max_count) {
                    max_count = class_counts[j];
                    majority_class = j;
                }
            }
        }
        
        // Regression pre-computation
        const real_t y_scale = 1000.0f;
        real_t total_sum_y = 0.0f;
        real_t total_sum_y2 = 0.0f;
        
        if (nclass == 1) {
            for (integer_t i = 0; i < node_sample_count; ++i) {
                real_t y = static_cast<real_t>(cl[node_samples[i]]) / y_scale;
                total_sum_y += y;
                total_sum_y2 += y * y;
            }
            
            if (node_sample_count > 0) {
                real_t mean_y = total_sum_y / static_cast<real_t>(node_sample_count);
                real_t variance = (total_sum_y2 / static_cast<real_t>(node_sample_count)) - mean_y * mean_y;
                if (variance < 1.0e-6f) {
                    nodestatus[kgrow] = -1;
                    nodeclass[kgrow] = 0;
                    continue;
                }
            } else {
                nodestatus[kgrow] = -1;
                nodeclass[kgrow] = 0;
                continue;
            }
        } else if (non_zero_classes <= 1) {
            nodestatus[kgrow] = -1;
            nodeclass[kgrow] = majority_class;
            continue;
        }
        
        // Find best split
        real_t best_crit = (nclass == 1) ? 1.0e20f : -2.0f;
        integer_t best_var = 0;
        integer_t jstat = 0;
        real_t best_split_value = 0.0f;
        
        // ====================================================================
        // SPARSE SPLIT FINDING - Uses x_sparse.get(row, col) instead of x[]
        // ====================================================================
        
        if (nclass == 1) {
            // REGRESSION with SPARSE access
            std::vector<integer_t> mvar_list(mtry);
            for (integer_t mv = 0; mv < mtry; ++mv) {
                mvar_list[mv] = static_cast<integer_t>(rng.randomu() * mdim);
                if (mvar_list[mv] >= mdim) mvar_list[mv] = mdim - 1;
            }
            
            real_t local_best_crit = best_crit;
            integer_t local_best_var = best_var;
            real_t local_best_split = best_split_value;
            integer_t local_jstat = jstat;
            
            // NOTE: Don't use reduction(min:local_best_crit) - it conflicts with critical section
            // The critical section properly synchronizes local_best_crit with local_best_var/split
            #ifdef _OPENMP
            #pragma omp parallel for if(node_sample_count > 100)
            #endif
            for (integer_t mv = 0; mv < mtry; ++mv) {
                integer_t mvar = mvar_list[mv];
                
                // SPARSE ACCESS: Use x_sparse.get() instead of x[]
                std::vector<std::pair<real_t, integer_t>> sorted_samples;
                sorted_samples.reserve(node_sample_count);
                for (integer_t i = 0; i < node_sample_count; ++i) {
                    integer_t idx = node_samples[i];
                    // NATIVE SPARSE ACCESS - O(nnz_per_row) per lookup
                    real_t val = x_sparse.get(idx, mvar);
                    sorted_samples.push_back({val, idx});
                }
                std::sort(sorted_samples.begin(), sorted_samples.end());
                
                real_t sum_y_left = 0.0f, sum_y2_left = 0.0f;
                real_t sum_y_right = total_sum_y, sum_y2_right = total_sum_y2;
                integer_t n_left = 0, n_right = node_sample_count;
                
                real_t thread_best_crit = 1.0e20f;
                real_t thread_best_split = 0.0f;
                
                for (integer_t ii = 0; ii < node_sample_count - 1; ++ii) {
                    if (sorted_samples[ii].first >= sorted_samples[ii + 1].first) continue;
                    
                    integer_t sample_idx = sorted_samples[ii].second;
                    real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;
                    real_t y_sq = y_val * y_val;
                    
                    sum_y_left += y_val; sum_y2_left += y_sq;
                    sum_y_right -= y_val; sum_y2_right -= y_sq;
                    n_left++; n_right--;
                    
                    if (n_left < 1 || n_right < 1) continue;
                    
                    real_t inv_l = 1.0f / n_left, inv_r = 1.0f / n_right;
                    real_t mean_l = sum_y_left * inv_l, mean_r = sum_y_right * inv_r;
                    real_t mse = (n_left * (sum_y2_left * inv_l - mean_l * mean_l) +
                                  n_right * (sum_y2_right * inv_r - mean_r * mean_r)) / node_sample_count;
                    
                    if (mse < thread_best_crit) {
                        thread_best_crit = mse;
                        thread_best_split = sorted_samples[ii].first;
                    }
                }
                
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    if (thread_best_crit < local_best_crit) {
                        local_best_crit = thread_best_crit;
                        local_best_var = mvar;
                        local_best_split = thread_best_split;
                        local_jstat = 0;
                    }
                }
            }
            
            best_crit = local_best_crit;
            best_var = local_best_var;
            best_split_value = local_best_split;
            jstat = local_jstat;
        }
        else {
            // CLASSIFICATION with SPARSE access
            std::vector<real_t> wl_local(nclass, 0.0f), wr_local(nclass, 0.0f);
            std::vector<std::pair<real_t, integer_t>> sorted_samples;
            sorted_samples.reserve(node_sample_count);
            
            real_t base_pno = 0.0f, base_pdo = 0.0f;
            for (integer_t j = 0; j < nclass; ++j) {
                base_pno += class_counts[j] * class_counts[j];
                base_pdo += class_counts[j];
            }
            
            for (integer_t mv = 0; mv < mtry; ++mv) {
                integer_t mvar = static_cast<integer_t>(rng.randomu() * mdim);
                if (mvar >= mdim) mvar = mdim - 1;
                
                real_t rrn = base_pno, rrd = base_pdo, rln = 0.0f, rld = 0.0f;
                std::fill(wl_local.begin(), wl_local.end(), 0.0f);
                for (integer_t j = 0; j < nclass; ++j) wr_local[j] = class_counts[j];
                
                sorted_samples.clear();
                for (integer_t i = 0; i < node_sample_count; ++i) {
                    integer_t sample_idx = node_samples[i];
                    // NATIVE SPARSE ACCESS
                    real_t val = x_sparse.get(sample_idx, mvar);
                    sorted_samples.push_back({val, sample_idx});
                }
                std::sort(sorted_samples.begin(), sorted_samples.end());
                
                for (integer_t ii = 0; ii < node_sample_count - 1; ++ii) {
                    if (sorted_samples[ii].first >= sorted_samples[ii + 1].first) continue;
                    
                    integer_t nc = sorted_samples[ii].second;
                    integer_t k = cl[nc];
                    real_t u = 1.0f;
                    rln += u * (u + 2.0f * wl_local[k]);
                    rrn += u * (u - 2.0f * wr_local[k]);
                    rld += u; rrd -= u;
                    wl_local[k] += u; wr_local[k] -= u;
                    
                    if (std::min(rrd, rld) > 1.0e-5f) {
                        real_t crit = (rln / rld) + (rrn / rrd);
                        if (crit > best_crit) {
                            best_crit = crit;
                            best_var = mvar;
                            best_split_value = sorted_samples[ii].first;
                            jstat = 0;
                        }
                    }
                }
            }
        }
        
        // Compute base criterion for comparison (must be computed BEFORE use)
        real_t base_pno_local = 0.0f, base_pdo_local = 0.0f;
        if (nclass > 1) {
            for (integer_t j = 0; j < nclass; ++j) {
                base_pno_local += class_counts[j] * class_counts[j];
                base_pdo_local += class_counts[j];
            }
        }
        
        // Create split if improvement found
        real_t crit0 = (nclass == 1) ? 
            ((total_sum_y2 / node_sample_count) - (total_sum_y / node_sample_count) * (total_sum_y / node_sample_count)) :
            (base_pdo_local > 0.0f ? (base_pno_local / base_pdo_local) : 0.0f);
        
        if (jstat == 0 && ((nclass == 1 && best_crit < crit0) || (nclass > 1 && best_crit > -1.0f))) {
            if (nnode + 2 <= maxnode) {
                integer_t left_child = nnode;
                integer_t right_child = nnode + 1;
                
                nodestatus[kgrow] = 1;
                bestvar[kgrow] = best_var;
                xbestsplit[kgrow] = best_split_value;
                
                treemap[kgrow * 2] = left_child;
                treemap[kgrow * 2 + 1] = right_child;
                
                nodestatus[left_child] = 2;
                nodestatus[right_child] = 2;
                
                nnode += 2;
                
                // SPARSE ACCESS: Move samples to children
                integer_t split_var_upd = bestvar[kgrow];
                real_t split_point_upd = xbestsplit[kgrow];
                for (integer_t i = 0; i < valid_samples; ++i) {
                    if (sample_node[i] == kgrow) {
                        // NATIVE SPARSE ACCESS
                        real_t val = x_sparse.get(sample_idx_map[i], split_var_upd);
                        sample_node[i] = (val <= split_point_upd) ? left_child : right_child;
                    }
                }
            } else {
                nodestatus[kgrow] = -1;
                nodeclass[kgrow] = majority_class;
            }
        } else {
            nodestatus[kgrow] = -1;
            nodeclass[kgrow] = majority_class;
        }
    }

    // Finalize terminal nodes
    if (nclass > 1) {
        for (integer_t node = 0; node < nnode; ++node) {
            if (nodestatus[node] == -1) {
                std::vector<real_t> class_counts_final(nclass, 0.0f);
                for (integer_t i = 0; i < valid_samples; ++i) {
                    if (sample_node[i] == node) {
                        integer_t class_label = cl[sample_idx_map[i]];
                        if (class_label >= 0 && class_label < nclass) {
                            class_counts_final[class_label] += 1.0f;
                        }
                    }
                }
                integer_t maj_class = 0;
                real_t max_cnt = 0.0f;
                for (integer_t j = 0; j < nclass; ++j) {
                    if (class_counts_final[j] > max_cnt) {
                        max_cnt = class_counts_final[j];
                        maj_class = j;
                    }
                }
                nodeclass[node] = maj_class;
            }
        }
    }

    // Regression: compute tnodewt (matches dense version - handles casewise mode)
    if (nclass == 1 && nodepred != nullptr) {
        const real_t y_scale = 1000.0f;
        std::vector<real_t> node_sum_y(nnode, 0.0f);
        std::vector<real_t> node_sum_weight(nnode, 0.0f);  // For nodepred
        std::vector<real_t> node_count(nnode, 0.0f);       // For tnodewt
        
        for (integer_t i = 0; i < valid_samples; ++i) {
            integer_t terminal_node = sample_node[i];
            integer_t sample_idx = sample_idx_map[i];
            real_t y_val = static_cast<real_t>(cl[sample_idx]) / y_scale;
            real_t bootstrap_freq = static_cast<real_t>(nin[sample_idx]);
            
            if (g_config.use_casewise) {
                node_sum_y[terminal_node] += y_val * bootstrap_freq;
                node_sum_weight[terminal_node] += bootstrap_freq;
                node_count[terminal_node] += 1.0f;
            } else {
                node_sum_y[terminal_node] += y_val;
                node_sum_weight[terminal_node] += 1.0f;
                node_count[terminal_node] += 1.0f;
            }
        }
        
        for (integer_t node = 0; node < nnode; ++node) {
            if (nodestatus[node] == -1) {
                // nodepred: weighted mean y (prediction)
                nodepred[node] = (node_sum_weight[node] > 0.0f) ? 
                    (node_sum_y[node] / node_sum_weight[node]) : 0.0f;
                // tnodewt: mean bootstrap weight (for OOB weighting)
                tnodewt[node] = (node_count[node] > 0.0f) ? 
                    (node_sum_weight[node] / node_count[node]) : 0.0f;
            }
        }
    }

    // Set jtr and nodextr
    for (integer_t n = 0; n < nsample; ++n) {
        jtr[n] = 0;
        nodextr[n] = 0;
    }
}

} // namespace rf

