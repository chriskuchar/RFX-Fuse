#include "rf_bootstrap.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_config.hpp"
#include <iostream>
#include <cstdio>

namespace rf {

// ============================================================================
// CPU BOOTSTRAP IMPLEMENTATION
// ============================================================================

void cpu_bootstrap(const real_t* weight, integer_t nsample,
                   real_t* win, integer_t* nin, integer_t* nout,
                   integer_t* jinbag, integer_t* joobag,
                   integer_t& ninbag, integer_t& noobag,
                   integer_t tree_id) {
    // Exact copy of original boot algorithm
    zerv(nin, nsample);
    
    // Create local RNG instance for this tree to avoid global state issues
    MT19937 local_rng;
    local_rng.sgrnd(g_config.iseed + tree_id);
    
    // Check bounds before loop
    // if (nsample <= 0 || nsample > 1000000) {
    //     printf("ERROR: Invalid nsample=%d in cpu_bootstrap\n", static_cast<int>(nsample));
    //     fflush(stdout);
    //     return;
    // }

    for (integer_t n = 0; n < nsample; n++) {
        // Exact port from original Fortran boot.f (lines 26-32)
        // Original: i = int(randomu()*nsample) + 1  (1-based: generates 1..nsample)
        // C++ port: i = static_cast<integer_t>(rand_val * nsample)  (0-based: generates 0..nsample-1)
        // Note: Expected in-bag percentage = 1 - (1 - 1/n)^n
        //       For n→∞: approaches 1 - e^(-1) ≈ 0.6321 (63.2%)
        //       For finite n: slightly higher (e.g., n=150: ~63.21%, n=100: ~63.40%, n=10: ~65.13%)
        real_t rand_val = local_rng.randomu();
        integer_t i = static_cast<integer_t>(rand_val * nsample);  // 0-based indexing
        if (i >= nsample) i = nsample - 1;  // Clamp to valid range (matches Fortran: if (i.gt.nsample) i = nsample)
        if (i < 0) i = 0;  // Safety check
        
        // Bounds check before array access
        // if (i < 0 || i >= nsample) {
        //     printf("ERROR: Index out of bounds: i=%d, nsample=%d, n=%d\n", static_cast<int>(i), static_cast<int>(nsample), static_cast<int>(n));
        //     fflush(stdout);
        //     return;
        // }
        
        nin[i] = nin[i] + 1;
    }

    ninbag = 0;
    noobag = 0;
    zervr(win, nsample);

    for (integer_t n = 0; n < nsample; n++) {
        if (nin[n] == 0) {
            nout[n] = nout[n] + 1;
            joobag[noobag] = n;
            noobag++;
        } else {
            // For case-wise: win = nin * weight (bootstrap frequency weighting)
            // For non-case-wise: win = 1.0 (equal weighting, matches R's randomForest)
            if (g_config.use_casewise) {
                win[n] = static_cast<real_t>(nin[n]) * weight[n];
            } else {
                win[n] = 1.0f;  // Non-case-wise: equal weighting
            }
            // DEBUG: Print win values for first tree, first 10 in-bag samples
            if (tree_id == 0 && ninbag < 10) {
                // Debug print removed to avoid potential stream corruption issues
                // fprintf(stderr, "[BOOTSTRAP] tree=%d, sample=%d, nin=%d, win=%.6f, use_casewise=%d\n",
                //        tree_id, n, nin[n], win[n], g_config.use_casewise ? 1 : 0);
            }
            jinbag[ninbag] = n;
            ninbag++;
        }
    }
}

} // namespace rf
