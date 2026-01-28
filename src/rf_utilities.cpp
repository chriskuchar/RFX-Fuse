#include "rf_utilities.hpp"
#include "rf_utils.hpp"
#include "rf_arrays.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rf {

// ----------------------------------------------------------------------
// Compute error rate (exact port of comperr.f90)
// ----------------------------------------------------------------------
void comperr(const integer_t* jest, const integer_t* cl, integer_t nsample, dp_t& errtr) {
    errtr = 0.0;
    for (integer_t n = 0; n < nsample; ++n) {
        if (jest[n] != cl[n]) {
            errtr += 1.0;
        }
    }
    errtr = errtr / static_cast<dp_t>(nsample);
}

// ----------------------------------------------------------------------
// Local importance calculation (exact port of localimp.f90)
// ----------------------------------------------------------------------
void localimp(integer_t nsample, integer_t mdim, integer_t ntree,
              const real_t* qimp, real_t* qimpm) {
    // Array layout: row-major format (C-style, matches CPU/GPU accumulation)
    // qimpm indexing: qimpm[sample * mdim + feature] = qimpm[n * mdim + m]
    // CPU varimp.cpp and GPU varimp.cu both accumulate with: qimpm[nn * mdim + k]
    for (integer_t n = 0; n < nsample; ++n) {
        for (integer_t m = 0; m < mdim; ++m) {
            // Fortran: qimpm(n, m) = 100.0 * (qimp(n) - qimpm(n, m)) / real(ntree)
            // Row-major: qimpm[n * mdim + m] (sample-major, matches CPU/GPU)
            integer_t idx = n * mdim + m;  // Row-major: sample * mdim + feature
            qimpm[idx] = 100.0f * (qimp[n] - qimpm[idx]) / static_cast<real_t>(ntree);
        }
    }
}

// ----------------------------------------------------------------------
// Local importance calculation for REGRESSION (Breiman Cutler adaptation)
// For regression: importance = INCREASE in MSE when feature is permuted
// So we compute: 100 * (qimpm - qimp) / ntree (opposite sign from classification)
// ----------------------------------------------------------------------
void localimp_regression(integer_t nsample, integer_t mdim, integer_t ntree,
                         const real_t* qimp, real_t* qimpm) {
    // Array layout: row-major for CPU regression (sample * mdim + feature)
    // qimpm[n * mdim + m] = local importance for sample n, feature m
    for (integer_t n = 0; n < nsample; ++n) {
        for (integer_t m = 0; m < mdim; ++m) {
            // Regression: higher permuted error = more important
            // qimpm = permuted squared error, qimp = original squared error
            // Local importance = 100 * (perm - orig) / ntree = increase in MSE %
            integer_t idx = n * mdim + m;  // Row-major: sample * mdim + feature
            qimpm[idx] = 100.0f * (qimpm[idx] - qimp[n]) / static_cast<real_t>(ntree);
        }
    }
}

// ----------------------------------------------------------------------
// Prepare data - sort and handle ties (exact port of prepdata.f90)
// OPTIMIZED: Parallelized across features (each feature is independent)
// ----------------------------------------------------------------------
void prepdata(const real_t* x, integer_t mdim, integer_t nsample,
              const integer_t* cat, integer_t* isort, real_t* v,
              integer_t* asave, integer_t* ties) {

    // x is (nsample, mdim) row-major
    // asave is (mdim, nsample) column-major (keep for compatibility)
    // ties is (mdim, nsample) column-major (keep for compatibility)
    // NOTE: isort and v are legacy parameters, now unused (thread-local buffers used instead)

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (integer_t mvar = 0; mvar < mdim; ++mvar) {
        if (cat[mvar] <= 2) {
            // Quantitative variable (including binary) - need to sort and handle ties
            
            // Thread-local buffers for sorting
            std::vector<std::pair<real_t, integer_t>> pairs(nsample);

            // Copy column to pairs for sorting
            for (integer_t n = 0; n < nsample; ++n) {
                pairs[n] = {x[n * mdim + mvar], n};  // Row-major access: x[sample, feature]
            }

            // Sort by value
            std::sort(pairs.begin(), pairs.end());

            // Copy sorted indices to asave and initialize ties
            for (integer_t n = 0; n < nsample; ++n) {
                integer_t idx = mvar + n * mdim;
                asave[idx] = pairs[n].second;
                ties[idx] = 0;
            }

            // Handle ties - mark tied values
            integer_t n = 0;
            while (n < nsample - 1) {
                // Check if current and next values are tied
                if (std::abs(pairs[n].first - pairs[n + 1].first) < 1e-6f) {
                    // Found a tie - mark all tied values
                    while (n < nsample - 1 && std::abs(pairs[n].first - pairs[n + 1].first) < 1e-6f) {
                        integer_t idx = mvar + n * mdim;
                        ties[idx] = 1;
                        ++n;
                    }
                    integer_t idx = mvar + n * mdim;
                    ties[idx] = 1;  // Mark last in tie group
                }
                ++n;
            }

        } else {
            // Categorical variable - just copy rounded values
            for (integer_t n = 0; n < nsample; ++n) {
                asave[mvar + n * mdim] = static_cast<integer_t>(std::round(x[mvar + n * mdim]));
                ties[mvar + n * mdim] = 0;  // No ties for categorical
            }
        }
    }
}

} // namespace rf
