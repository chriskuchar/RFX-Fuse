#ifndef RF_PREDICT_HPP
#define RF_PREDICT_HPP

#include "rf_types.hpp"

namespace rf {

// CPU predict implementation
void cpu_predict(const real_t* q, const integer_t* out, integer_t nsample,
                integer_t nclass, integer_t* predictions);

// GPU predict implementation with parallel tree traversal
void gpu_predict(const real_t* q, const integer_t* out, integer_t nsample,
                integer_t nclass, integer_t* predictions);

} // namespace rf

#endif // RF_PREDICT_HPP
