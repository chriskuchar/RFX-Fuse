#include "rf_proximity.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_advanced_optimizations.hpp"
#include "rf_block_sparse_proximity.hpp"
#include "rf_config.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rf {

// ============================================================================
// CPU PROXIMITY IMPLEMENTATION
// ============================================================================

void cpu_proximity(const integer_t* nodestatus, const integer_t* nodextr,
                  const integer_t* nin, integer_t nsample, integer_t nnode,
                  dp_t* prox, integer_t* nod, integer_t* ncount,
                  integer_t* ncn, integer_t* nodexb, integer_t* ndbegin,
                  integer_t* npcase) {
    
    // Set OpenMP thread count from config
#ifdef _OPENMP
    if (g_config.n_threads_cpu > 0) {
        omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
    }
#endif
    
    // Check if case-wise (bootstrap frequency weighted) or non-case-wise (simple co-occurrence) should be used
    // Case-wise: prox(n, kk) = prox(n, kk) + nin(kk)/nodesize (Fortran proximity.f line 75)
    // Non-case-wise: prox(n, kk) = prox(n, kk) + 1.0 (UC Berkeley standard: simple co-occurrence counting)
    bool use_casewise = g_config.use_casewise;
    
    // Choose optimization strategy based on config or problem size
    bool use_sparse = g_config.use_sparse || (nsample > 5000);  // Use sparse if enabled or for large matrices
    bool use_simd = nsample > 100;     // Use SIMD for medium+ matrices
    
    if (use_sparse) {
        // Use block sparse proximity matrix for memory efficiency
        BlockSparseProximityMatrix sparse_prox(nsample, BlockSparseConfig::CONSERVATIVE_THRESHOLD);
        
        // Process trees with sparse accumulation
        for (integer_t tree = 0; tree < nnode; ++tree) {
            if (nodestatus[tree] == -1) {  // Terminal node
                // Find all samples that end up in this terminal node
                std::vector<integer_t> samples_in_node;
                
                for (integer_t n = 0; n < nsample; ++n) {
                    if (nin[n] == 0) {  // Out-of-bag samples only
                        if (nodextr[n] == tree) {
                            samples_in_node.push_back(n);
                        }
                    }
                }
                
                // For sparse mode, we still need to match Fortran algorithm
                // Build terminal node structure for this tree
                integer_t k = 0;  // This tree is terminal node k
                integer_t nodesize = 0;
                
                // Calculate nodesize = sum of nin(kk) for all in-bag cases in this node
                for (integer_t kk = 0; kk < nsample; ++kk) {
                    if (nodextr[kk] == tree && nin[kk] > 0) {
                        nodesize += nin[kk];
                    }
                }
                
                // Update sparse proximity matrix for OOB samples to in-bag samples
                for (size_t i = 0; i < samples_in_node.size(); ++i) {
                    integer_t sample_i = samples_in_node[i];  // OOB sample
                    
                    // Find all in-bag samples in this node
                    for (integer_t kk = 0; kk < nsample; ++kk) {
                        if (nodextr[kk] == tree && nin[kk] > 0) {  // In-bag case in same node
                            // Case-wise: weight = nin(kk)/nodesize
                            // Non-case-wise: weight = 1.0
                            real_t weight = use_casewise && nodesize > 0 ?
                                static_cast<real_t>(nin[kk]) / static_cast<real_t>(nodesize) :
                                1.0f;
                            sparse_prox.add_proximity(sample_i, kk, weight);
                        }
                    }
                }
            }
        }
        
        // Convert sparse matrix to dense output
        // Create temporary float array for conversion
        std::vector<real_t> temp_prox(nsample * nsample);
        sparse_prox.to_dense_matrix(temp_prox.data());
        
        // Convert from real_t (float) to dp_t (double)
        for (integer_t i = 0; i < nsample * nsample; ++i) {
            prox[i] = static_cast<dp_t>(temp_prox[i]);
        }
        
    } else {
        // Match Fortran proximity.f exactly:
        // 1. For each OOB sample n (nin(n) == 0)
        // 2. Find all cases kk in the same terminal node (including in-bag cases with nin(kk) > 0)
        // 3. Calculate nodesize = sum of nin(kk) for all in-bag cases in that node
        // 4. Update prox(n, kk) += weight, where:
        //    - Case-wise: weight = nin(kk)/nodesize (bootstrap frequency weighted)
        //    - Non-case-wise: weight = 1.0 (simple co-occurrence)
        
        // First, build terminal node structure (0-based indexing throughout)
        // Find terminal nodes and assign 0-based indices
        integer_t nterm = 0;
        // Initialize nod array to -1 (invalid) for all nodes
        std::fill(nod, nod + nnode, -1);
        
        for (integer_t k = 0; k < nnode; ++k) {
            if (nodestatus[k] == -1) {  // Terminal node
                nod[k] = nterm;  // 0-based terminal node index
                nterm++;
            }
        }
        
        // Count cases in each terminal node and assign cases (0-based indexing)
        std::fill(ncount, ncount + nsample, 0);
        std::fill(ncn, ncn + nsample, 0);
        std::fill(nodexb, nodexb + nsample, -1);  // Initialize to -1 (invalid)
        
        for (integer_t n = 0; n < nsample; ++n) {
            integer_t node_idx = nodextr[n];  // Node index for sample n
            if (node_idx >= 0 && node_idx < nnode) {
                integer_t k = nod[node_idx];  // Terminal node index (0-based)
                if (k >= 0 && k < nterm) {  // Valid terminal node
                    ncount[k]++;  // Count cases in this terminal node (0-based)
                    ncn[n] = ncount[k];  // Position of case n in terminal node k
                    nodexb[n] = k;  // Terminal node index for case n (0-based)
                }
            }
        }
        
        // Build ndbegin array (points to first case in each terminal node)
        // Allocate ndbegin with size nterm+1 to prevent out-of-bounds access
        // ndbegin[k+1] is accessed in the loop below, so at least nterm+1 elements are required
        // 0-based indexing: ndbegin[k] = start index for terminal node k
        ndbegin[0] = 0;
        for (integer_t k = 1; k <= nterm; ++k) {
            ndbegin[k] = ndbegin[k - 1] + ncount[k - 1];
        }
        // Ensure ndbegin[nterm] is set to total number of cases
        // This is the end index for the last terminal node (nterm-1)
        // When k == nterm-1, ndbegin[k+1] = ndbegin[nterm] is accessed, which should be nsample
        if (nterm > 0) {
            ndbegin[nterm] = ndbegin[nterm - 1] + ncount[nterm - 1];
        }
        
        // Build npcase array (maps position to case index)
        // 0-based indexing: npcase[j] = case index at position j in terminal node structure
        // Initialize to -1 (invalid) - use loop to avoid compiler warning about large sizes
        if (nsample > 0 && nsample < 100000000) {  // Sanity check: reasonable size
            for (integer_t j = 0; j < nsample; ++j) {
                npcase[j] = -1;
            }
        }
        for (integer_t n = 0; n < nsample; ++n) {
            integer_t k = nodexb[n];  // Terminal node index (0-based)
            if (k >= 0 && k < nterm) {  // Valid terminal node
                // Position in terminal node structure: ndbegin[k] + (ncn[n] - 1)
                // ncn[n] is 1-based position, so subtract 1 for 0-based indexing
                integer_t kn = ndbegin[k] + ncn[n] - 1;
                if (kn >= 0 && kn < nsample) {
                    npcase[kn] = n;
                }
            }
        }
        
        // Process OOB samples and update proximity (matching Fortran lines 56-79)
#ifdef _OPENMP
#pragma omp parallel for
#endif
        // Process OOB samples and update proximity (0-based indexing, column-major storage)
        // Column-major: prox[i + j * nsample] = element at row i, column j
        for (integer_t n = 0; n < nsample; ++n) {
            if (nin[n] == 0) {  // OOB sample
                integer_t k = nodexb[n];  // Terminal node index (0-based)
                
                // Bounds check: ensure k is valid and ndbegin[k+1] is within bounds
                if (k < 0 || k >= nterm) {
                    continue;  // Invalid terminal node index
                }
                
                // Calculate nodesize = sum of nin(kk) for all in-bag cases in this node
                integer_t nodesize = 0;
                // Safe bounds check for ndbegin[k+1]
                // k is 0-based, so k+1 can be at most nterm (which is valid since ndbegin has size nterm+1)
                integer_t end_idx = (k + 1 <= nterm) ? ndbegin[k + 1] : nsample;
                if (end_idx > nsample) end_idx = nsample;  // Additional safety check
                
                // First pass: compute nodesize (sum of bootstrap multiplicities for in-bag cases)
                for (integer_t j = ndbegin[k]; j < end_idx; ++j) {
                    if (j >= 0 && j < nsample) {  // Additional bounds check
                        integer_t kk = npcase[j];  // Case index at position j
                        if (kk >= 0 && kk < nsample && nin[kk] > 0) {  // In-bag case with bounds check
                            nodesize += nin[kk];
                        }
                    }
                }
                
                // Update proximity for all in-bag cases in this node
                // Column-major: prox[n + kk * nsample] = proximity from OOB sample n to in-bag sample kk
                if (nodesize > 0 || !use_casewise) {
                    for (integer_t j = ndbegin[k]; j < end_idx; ++j) {
                        if (j >= 0 && j < nsample) {  // Additional bounds check
                            integer_t kk = npcase[j];  // Case index at position j
                            if (kk >= 0 && kk < nsample && nin[kk] > 0) {  // In-bag case with bounds check
                                // Case-wise: prox(n, kk) = prox(n, kk) + nin(kk)/nodesize (Fortran line 75)
                                // Non-case-wise: prox(n, kk) = prox(n, kk) + 1.0 (UC Berkeley standard)
                                dp_t weight = use_casewise && nodesize > 0 ? 
                                    static_cast<dp_t>(nin[kk]) / static_cast<dp_t>(nodesize) : 
                                    1.0;
                                // Column-major, 0-based: prox[n + kk * nsample] = row n, column kk
                                integer_t prox_idx = n + kk * nsample;
                                if (prox_idx >= 0 && prox_idx < nsample * nsample) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                                    prox[prox_idx] += weight;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// RF-GAP PROXIMITY IMPLEMENTATION
// ============================================================================

void cpu_proximity_rfgap(
    integer_t ntree, integer_t nsample,
    const std::vector<std::vector<integer_t>>& tree_nin,  // Per-tree bootstrap multiplicities [tree][sample] (0=OOB, >0=in-bag)
    const std::vector<std::vector<integer_t>>& tree_nodextr,  // Per-tree terminal nodes [tree][sample]
    dp_t* prox) {
    
    // Set OpenMP thread count from config
#ifdef _OPENMP
    if (g_config.n_threads_cpu > 0) {
        omp_set_num_threads(static_cast<int>(g_config.n_threads_cpu));
    }
#endif
    
    // Initialize proximity matrix to zero
    std::fill(prox, prox + nsample * nsample, 0.0);
    
    // For each sample i, compute pGAP(i, j) for all j
    // pGAP(i, j) = (1/|Si|) * Σ(t∈Si) [cj(t) * I(j ∈ Ji(t)) / |Mi(t)|]
    // Where:
    //   Si = {t | i ∈ O(t)} - trees where observation i is OOB
    //   Ji(t) = vi(t) ∩ B(t) - in-bag observations in the same terminal node as i
    //   Mi(t) = multiset of in-bag indices in the terminal node shared with i
    //   cj(t) = in-bag multiplicity of observation j in tree t
    
    // Bounds checking: ensure tree_nin and tree_nodextr are properly sized
    if (static_cast<size_t>(ntree) != tree_nin.size() || 
        static_cast<size_t>(ntree) != tree_nodextr.size()) {
        // std::cerr << "ERROR: cpu_proximity_rfgap: tree_nin.size()=" << tree_nin.size() 
        //           << " or tree_nodextr.size()=" << tree_nodextr.size() 
        //           << " != ntree=" << ntree << std::endl;
        return;
    }
    
    for (integer_t t = 0; t < ntree; ++t) {
        if (static_cast<size_t>(nsample) != tree_nin[t].size() || 
            static_cast<size_t>(nsample) != tree_nodextr[t].size()) {
            // std::cerr << "ERROR: cpu_proximity_rfgap: tree_nin[" << t << "].size()=" 
            //           << tree_nin[t].size() << " or tree_nodextr[" << t << "].size()=" 
            //           << tree_nodextr[t].size() << " != nsample=" << nsample << std::endl;
            return;
        }
    }
    
    // Create flattened arrays to avoid issues with nested vector references
    // This prevents memory corruption if the outer vectors are reallocated during access
    // Flatten tree_nin: [tree][sample] -> [tree * nsample + sample]
    // Cast to size_t to prevent integer overflow
    std::vector<integer_t> tree_nin_flat(static_cast<size_t>(ntree) * static_cast<size_t>(nsample));
    std::vector<integer_t> tree_nodextr_flat(static_cast<size_t>(ntree) * static_cast<size_t>(nsample));
    
    for (integer_t t = 0; t < ntree; ++t) {
        for (integer_t s = 0; s < nsample; ++s) {
            size_t flat_idx = static_cast<size_t>(t) * static_cast<size_t>(nsample) + s;
            tree_nin_flat[flat_idx] = tree_nin[t][s];
            tree_nodextr_flat[flat_idx] = tree_nodextr[t][s];
        }
    }
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (integer_t i = 0; i < nsample; ++i) {
        // Find all trees where sample i is OOB (Si)
        std::vector<integer_t> trees_oob_i;
        trees_oob_i.reserve(ntree);  // Reserve capacity to avoid reallocations
        for (integer_t t = 0; t < ntree; ++t) {
            // Check if i is OOB in tree t (tree_nin_flat[t * nsample + i] == 0)
            // Use flattened array to avoid nested vector access issues
            // Cast to size_t to prevent integer overflow
            size_t flat_idx = static_cast<size_t>(t) * static_cast<size_t>(nsample) + i;
            if (tree_nin_flat[flat_idx] == 0) {
                trees_oob_i.push_back(t);
            }
        }
        
        integer_t num_trees_oob_i = static_cast<integer_t>(trees_oob_i.size());
        if (num_trees_oob_i == 0) {
            // Sample i is never OOB, set diagonal to 1.0 (self-proximity)
            prox[i + i * nsample] = 1.0;
            continue;
        }
        
        // For each tree t where i is OOB
        for (integer_t t : trees_oob_i) {
            // Get terminal node for OOB sample i in tree t
            // Use flattened array to avoid nested vector access issues
            // Cast to size_t to prevent integer overflow
            size_t base_idx_t = static_cast<size_t>(t) * static_cast<size_t>(nsample);
            integer_t terminal_node_i = tree_nodextr_flat[base_idx_t + i];
            
            if (terminal_node_i < 0) {
                continue;  // Invalid terminal node
            }
            
            // Find all in-bag samples in the same terminal node (Ji(t))
            // Also compute |Mi(t)| = sum of multiplicities of in-bag samples in this node
            integer_t total_inbag_multiplicity = 0;
            
            // First pass: compute |Mi(t)|
            for (integer_t j = 0; j < nsample; ++j) {
                // Use flattened arrays to avoid nested vector access issues
                integer_t cj_t = tree_nin_flat[base_idx_t + j];  // Multiplicity of j in bootstrap for tree t
                if (cj_t > 0 && tree_nodextr_flat[base_idx_t + j] == terminal_node_i) {
                    // j is in-bag and in the same terminal node as i
                    total_inbag_multiplicity += cj_t;
                }
            }
            
            if (total_inbag_multiplicity == 0) {
                continue;  // No in-bag samples in this terminal node
            }
            
            // Second pass: accumulate contributions to pGAP(i, j)
            dp_t inv_total_multiplicity = 1.0 / static_cast<dp_t>(total_inbag_multiplicity);
            for (integer_t j = 0; j < nsample; ++j) {
                // Use flattened arrays to avoid nested vector access issues
                integer_t cj_t = tree_nin_flat[base_idx_t + j];  // Multiplicity of j in bootstrap for tree t
                if (cj_t > 0 && tree_nodextr_flat[base_idx_t + j] == terminal_node_i) {
                    // j is in-bag and in the same terminal node as i
                    // Add cj(t) / |Mi(t)| to pGAP(i, j)
                    dp_t contribution = static_cast<dp_t>(cj_t) * inv_total_multiplicity;
                    // Use atomic operations for thread safety in parallel region
#ifdef _OPENMP
#pragma omp atomic
#endif
                    prox[i + j * nsample] += contribution;
                }
            }
        }
        
        // Normalize by |Si|
        // Note: Each thread processes a different i, so no race condition here
        if (num_trees_oob_i > 0) {
            dp_t norm_factor = 1.0 / static_cast<dp_t>(num_trees_oob_i);
            for (integer_t j = 0; j < nsample; ++j) {
                prox[i + j * nsample] *= norm_factor;
            }
        }
    }
    
    // Make symmetric (pGAP is symmetric)
    // Use OpenMP for parallelization if available
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (integer_t i = 0; i < nsample; ++i) {
        for (integer_t j = i + 1; j < nsample; ++j) {
            dp_t avg = (prox[i + j * nsample] + prox[j + i * nsample]) / 2.0;
            // Use atomic operations for thread safety
#ifdef _OPENMP
#pragma omp atomic write
#endif
            prox[i + j * nsample] = avg;
#ifdef _OPENMP
#pragma omp atomic write
#endif
            prox[j + i * nsample] = avg;
        }
    }
}

} // namespace rf


