#include "rf_finishprox.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include <iostream>

namespace rf {

// ============================================================================
// CPU FINISHPROX IMPLEMENTATION
// ============================================================================

void cpu_finishprox(integer_t nsample, const integer_t* nout, dp_t* prox, dp_t* proxsym) {
    
    // Proximity matrix is stored in COLUMN-MAJOR format
    // prox[i + j * nsample] = element at row i, column j
    // This matches how cpu_proximity writes: prox[n + kk * nsample]
    
    // Copy prox to proxsym (both are column-major)
    for (integer_t i = 0; i < nsample * nsample; ++i) {
        proxsym[i] = prox[i];
    }
    
    // Fix diagonal elements - set them to nout[i] before normalization
    // (The raw diagonal values are half of what they should be due to how proximity is accumulated)
    // Column-major: diagonal is at prox[i + i * nsample]
    for (integer_t i = 0; i < nsample; ++i) {
        proxsym[i + i * nsample] = static_cast<dp_t>(nout[i]);
    }
    
    // Symmetrize the proximity matrix (skip diagonal - it's already correct)
    // Column-major: prox[i + j * nsample] = row i, column j
    //               prox[j + i * nsample] = row j, column i
    for (integer_t i = 0; i < nsample; ++i) {
        for (integer_t j = i + 1; j < nsample; ++j) {
            // Get both (i,j) and (j,i) elements (column-major indexing)
            dp_t prox_ij = prox[i + j * nsample];  // Row i, column j
            dp_t prox_ji = prox[j + i * nsample];  // Row j, column i
            dp_t avg_prox = (prox_ij + prox_ji) / 2.0;
            proxsym[i + j * nsample] = avg_prox;
            proxsym[j + i * nsample] = avg_prox;
        }
    }
    
    // STEP 1: Row-wise normalization (matching original Fortran finishprox.f)
    // Normalize each row i by nout[i] (number of trees where sample i was OOB)
    // Column-major: prox[i + j * nsample] = row i, column j
    for (integer_t i = 0; i < nsample; ++i) {
        if (nout[i] > 0) {
            dp_t inv_nout_i = 1.0 / static_cast<dp_t>(nout[i]);
            for (integer_t j = 0; j < nsample; ++j) {
                integer_t idx = i + j * nsample;  // Column-major indexing
                proxsym[idx] *= inv_nout_i;  // Normalize row i by nout[i]
            }
        }
    }
    
    // STEP 2: Symmetrization (matching original Fortran finishprox.f)
    // After row-wise normalization, symmetrize by averaging
    // Column-major: prox[i + j * nsample] = row i, column j
    //               prox[j + i * nsample] = row j, column i
    for (integer_t i = 0; i < nsample; ++i) {
        // Diagonal element: set to 1.0 (matching original Fortran)
        proxsym[i + i * nsample] = 1.0;
        
        // Off-diagonal elements: symmetrize by averaging
        for (integer_t j = i + 1; j < nsample; ++j) {
            dp_t prox_ij = proxsym[i + j * nsample];  // Row i, column j (already row-normalized)
            dp_t prox_ji = proxsym[j + i * nsample];  // Row j, column i (already row-normalized)
            dp_t avg_prox = 0.5 * (prox_ij + prox_ji);
            proxsym[i + j * nsample] = avg_prox;
            proxsym[j + i * nsample] = avg_prox;
        }
    }
}

} // namespace rf
