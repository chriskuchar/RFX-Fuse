#ifndef RF_PROXIMITY_UPPER_TRIANGLE_HPP
#define RF_PROXIMITY_UPPER_TRIANGLE_HPP

#include <cuda_fp16.h>  // MUST be FIRST - before any includes that might put it in a namespace
#include "rf_types.hpp"
#include "rf_quantization_kernels.hpp"

namespace rf {
namespace cuda {

/**
 * @brief Compute proximity matrix in packed upper triangle format (memory efficient!)
 * 
 * These functions compute proximity matrices directly in packed upper triangle format,
 * avoiding the need for full matrix storage. This reduces memory by 50% (upper triangle)
 * plus quantization savings (FP16: 4x, INT8: 8x, NF4: 16x).
 * 
 * Memory savings for 100k samples:
 * - FP32 full: 80 GB
 * - FP32 upper: 40 GB
 * - FP16 upper: 10 GB
 * - INT8 upper: 5 GB
 * - NF4 upper: 2.5 GB
 */

/**
 * @brief Compute proximity in packed upper triangle FP16 format
 * 
 * @param nodestatus Node status array
 * @param nodextr Node index array
 * @param nin In-bag indicator array
 * @param nsample Number of samples
 * @param nnode Number of nodes
 * @param prox_upper_fp16 Output: Packed upper triangle (n(n+1)/2 elements, FP16)
 * @param nod Node array
 * @param ncount Node count array
 * @param ncn Node case array
 * @param nodexb Node index array
 * @param ndbegin Node begin array
 * @param npcase Node case array
 */
void gpu_proximity_upper_triangle_fp16(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    __half* prox_upper_fp16,  // __half from global namespace (cuda_fp16.h included first)
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
);

/**
 * @brief Compute proximity in packed upper triangle INT8 format
 * 
 * @param nodestatus Node status array
 * @param nodextr Node index array
 * @param nin In-bag indicator array
 * @param nsample Number of samples
 * @param nnode Number of nodes
 * @param prox_upper_int8 Output: Packed upper triangle (n(n+1)/2 elements, INT8)
 * @param scale Quantization scale
 * @param zero_point Quantization zero point
 * @param nod Node array
 * @param ncount Node count array
 * @param ncn Node case array
 * @param nodexb Node index array
 * @param ndbegin Node begin array
 * @param npcase Node case array
 */
void gpu_proximity_upper_triangle_int8(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    int8_t* prox_upper_int8,
    float scale,
    float zero_point,
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
);

/**
 * @brief Compute proximity in packed upper triangle NF4 format
 * 
 * @param nodestatus Node status array
 * @param nodextr Node index array
 * @param nin In-bag indicator array
 * @param nsample Number of samples
 * @param nnode Number of nodes
 * @param prox_upper_nf4 Output: Packed upper triangle (n(n+1)/2 elements, NF4)
 * @param nod Node array
 * @param ncount Node count array
 * @param ncn Node case array
 * @param nodexb Node index array
 * @param ndbegin Node begin array
 * @param npcase Node case array
 */
void gpu_proximity_upper_triangle_nf4(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    uint8_t* prox_upper_nf4,
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
);

/**
 * @brief Unified function: Compute proximity in packed upper triangle format
 * 
 * Automatically selects the appropriate quantization level.
 * 
 * @param nodestatus Node status array
 * @param nodextr Node index array
 * @param nin In-bag indicator array
 * @param nsample Number of samples
 * @param nnode Number of nodes
 * @param quant_level Quantization level (FP16, INT8, NF4)
 * @param prox_upper_output Output: Packed upper triangle (type depends on quant_level)
 * @param nod Node array
 * @param ncount Node count array
 * @param ncn Node case array
 * @param nodexb Node index array
 * @param ndbegin Node begin array
 * @param npcase Node case array
 */
void gpu_proximity_upper_triangle(
    const integer_t* nodestatus,
    const integer_t* nodextr,
    const integer_t* nin,
    integer_t nsample,
    integer_t nnode,
    QuantizationLevel quant_level,
    void* prox_upper_output,
    integer_t* nod,
    integer_t* ncount,
    integer_t* ncn,
    integer_t* nodexb,
    integer_t* ndbegin,
    integer_t* npcase
);

/**
 * @brief GPU RF-GAP proximity computation in packed upper triangle format (for incremental low-rank)
 * 
 * Computes RF-GAP contributions per tree: pGAP_tree(i, j) = cj(t) * I(j ∈ Ji(t)) / |Mi(t)|
 * Where:
 *   - i is OOB (nin[i] == 0)
 *   - j is in-bag (nin[j] > 0) 
 *   - Both i and j are in the same terminal node (nodextr[i] == nodextr[j])
 *   - cj(t) = nin[j] (bootstrap multiplicity)
 *   - |Mi(t)| = sum of nin[k] for all in-bag samples k in the same terminal node as i
 * 
 * This computes per-tree contributions that can be added incrementally to low-rank factors,
 * avoiding the need to store the full O(n²) RF-GAP matrix.
 * 
 * @param nin_tree Bootstrap multiplicities for this tree [sample] (0=OOB, >0=in-bag)
 * @param nodextr_tree Terminal node assignments for this tree [sample]
 * @param nsample Number of samples
 * @param prox_upper_fp16 Output: Packed upper triangle (n(n+1)/2 elements) in FP16
 */
void gpu_proximity_rfgap_upper_triangle_fp16(
    const integer_t* nin_tree,      // Bootstrap multiplicities for this tree [sample] (0=OOB, >0=in-bag)
    const integer_t* nodextr_tree,  // Terminal node assignments for this tree [sample]
    integer_t nsample,
    __half* prox_upper_fp16  // Output: Packed upper triangle (n(n+1)/2 elements) in FP16
);

} // namespace cuda
} // namespace rf

#endif // RF_PROXIMITY_UPPER_TRIANGLE_HPP

