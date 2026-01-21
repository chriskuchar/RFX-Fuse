#ifndef RF_PROXIMITY_HPP
#define RF_PROXIMITY_HPP

#include "rf_types.hpp"
#include <vector>

namespace rf {

// CPU proximity implementation
void cpu_proximity(const integer_t* nodestatus, const integer_t* nodextr,
                  const integer_t* nin, integer_t nsample, integer_t nnode,
                  dp_t* prox, integer_t* nod, integer_t* ncount,
                  integer_t* ncn, integer_t* nodexb, integer_t* ndbegin,
                  integer_t* npcase);

// GPU proximity implementation with block sparse upper triangle updates
void gpu_proximity(const integer_t* nodestatus, const integer_t* nodextr,
                  const integer_t* nin, integer_t nsample, integer_t nnode,
                  dp_t* prox, integer_t* nod, integer_t* ncount,
                  integer_t* ncn, integer_t* nodexb, integer_t* ndbegin,
                  integer_t* npcase);

// RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity computation
// pGAP(i, j) = (1/|Si|) * Σ(t∈Si) [cj(t) * I(j ∈ Ji(t)) / |Mi(t)|]
// Where:
//   Si = {t | i ∈ O(t)} - trees where observation i is OOB
//   Ji(t) = vi(t) ∩ B(t) - in-bag observations in the same terminal node as i
//   Mi(t) = multiset of in-bag indices in the terminal node shared with i
//   cj(t) = in-bag multiplicity of observation j in tree t
// 
// Data structure:
//   tree_nin[tree][sample] = bootstrap multiplicity of sample in tree (0 if OOB, >0 if in-bag)
//   tree_nodextr[tree][sample] = terminal node index for sample in tree (valid for both OOB and in-bag)
void cpu_proximity_rfgap(
    integer_t ntree, integer_t nsample,
    const std::vector<std::vector<integer_t>>& tree_nin,  // Per-tree bootstrap multiplicities [tree][sample] (0=OOB, >0=in-bag)
    const std::vector<std::vector<integer_t>>& tree_nodextr,  // Per-tree terminal nodes [tree][sample]
    dp_t* prox);

// GPU-efficient RF-GAP proximity computation
// Uses flattened arrays: tree_nin_flat[tree * nsample + sample], tree_nodextr_flat[tree * nsample + sample]
void gpu_proximity_rfgap(
    integer_t ntree, integer_t nsample,
    const integer_t* tree_nin_flat,  // Flattened: [tree * nsample + sample] = bootstrap multiplicity
    const integer_t* tree_nodextr_flat,  // Flattened: [tree * nsample + sample] = terminal node index
    dp_t* prox);

} // namespace rf

#endif // RF_PROXIMITY_HPP
