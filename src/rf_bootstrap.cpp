#include "rf_bootstrap.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include "rf_config.hpp"
#include <vector>
#include <algorithm>
#include <cstdio>

namespace rf {

// ============================================================================
// CPU BOOTSTRAP IMPLEMENTATION
// ============================================================================

void cpu_bootstrap(const real_t* weight, integer_t nsample,
                   real_t* win, integer_t* nin, integer_t* nout,
                   integer_t* jinbag, integer_t* joobag,
                   integer_t& ninbag, integer_t& noobag,
                   integer_t tree_id,
                   const real_t* sample_weights) {
    zerv(nin, nsample);

    MT19937 local_rng;
    local_rng.sgrnd(g_config.iseed + tree_id);

    // Build cumulative distribution if sample_weights provided
    // When sample_weights is null, use original uniform sampling
    std::vector<real_t> cum_weights;
    real_t total_weight = 0.0f;

    if (sample_weights) {
        cum_weights.resize(nsample);
        for (integer_t i = 0; i < nsample; i++) {
            total_weight += sample_weights[i];
            cum_weights[i] = total_weight;
        }
    }

    for (integer_t n = 0; n < nsample; n++) {
        integer_t i;
        real_t rand_val = local_rng.randomu();

        if (sample_weights) {
            // Weighted sampling: binary search into cumulative weights
            real_t target = rand_val * total_weight;
            i = static_cast<integer_t>(
                std::lower_bound(cum_weights.data(), cum_weights.data() + nsample, target)
                - cum_weights.data()
            );
            if (i >= nsample) i = nsample - 1;
        } else {
            // Original uniform sampling (Breiman-Cutler boot.f)
            i = static_cast<integer_t>(rand_val * nsample);
            if (i >= nsample) i = nsample - 1;
            if (i < 0) i = 0;
        }

        nin[i] = nin[i] + 1;
    }

    // Post-bootstrap: compute win for downstream use
    // This casewise weighting logic is completely unchanged
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
                win[n] = 1.0f;
            }
            jinbag[ninbag] = n;
            ninbag++;
        }
    }
}

} // namespace rf
