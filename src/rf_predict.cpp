#include "rf_predict.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"

namespace rf {

// ============================================================================
// CPU PREDICT IMPLEMENTATION
// ============================================================================

void cpu_predict(const real_t* q, const integer_t* out, integer_t nsample,
                integer_t nclass, integer_t* predictions) {
    
    // For each sample
    for (integer_t n = 0; n < nsample; ++n) {
        if (out[n] > 0) {  // Sample was out-of-bag
            // Find class with maximum votes
            integer_t best_class = 0;
            real_t max_votes = q[n * nclass + 0];  // Row-major indexing
            
            for (integer_t c = 1; c < nclass; ++c) {
                real_t votes = q[n * nclass + c];  // Row-major indexing
                if (votes > max_votes) {
                    max_votes = votes;
                    best_class = c;
                }
            }
            
            predictions[n] = best_class;
        } else {
            // Sample was not out-of-bag, use default prediction
            predictions[n] = 0;
        }
    }
}

} // namespace rf
