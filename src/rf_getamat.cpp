#include "rf_getamat.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include <algorithm>
#include <vector>

namespace rf {

// ============================================================================
// CPU GETAMAT IMPLEMENTATION
// ============================================================================

void cpu_getamat(const integer_t* asave, const integer_t* nin, const integer_t* cat,
                 integer_t nsample, integer_t mdim, integer_t* amat) {
    
    // Initialize amat to zero
    zerm(amat, mdim, nsample);
    
    // Exact copy of Fortran getamat.f algorithm
    // asave is (mdim, nsample) column-major: asave[mvar + n * mdim]
    // amat is (mdim, nsample) column-major: amat[mvar + n * mdim]
    
    // For each variable
    for (integer_t mvar = 0; mvar < mdim; ++mvar) {
        integer_t nt = 0;  // Output index for amat (0-based)
        
        if (cat[mvar] <= 2) {
            // Quantitative variable (including binary) - copy sorted indices from asave, filtering by nin
            // Fortran: do n = 1, nsample; if (nin(asave(m, n)).ge.1) then amat(m, nt) = asave(m, n); nt = nt+1
            for (integer_t n = 0; n < nsample; ++n) {
                // asave is column-major: asave[mvar + n * mdim] gives asave(mvar, n) in Fortran
                integer_t sample_idx = asave[mvar + n * mdim];
                
                // Bounds check
                if (sample_idx < 0 || sample_idx >= nsample) {
                    continue;  // Skip invalid indices
                }
                
                // Check if this sample is in-bag: nin[sample_idx] >= 1
                if (nin[sample_idx] >= 1) {
                    // Copy the index to amat (column-major)
                    amat[mvar + nt * mdim] = sample_idx;
                    nt++;
                }
            }
        } else {
            // Categorical variable - copy all indices from asave (no filtering)
            // Fortran: do n = 1, nsample; amat(m, n) = asave(m, n)
            for (integer_t n = 0; n < nsample; ++n) {
                integer_t sample_idx = asave[mvar + n * mdim];
                
                // Bounds check
                if (sample_idx >= 0 && sample_idx < nsample) {
                    amat[mvar + n * mdim] = sample_idx;
                }
            }
        }
    }
}

} // namespace rf
