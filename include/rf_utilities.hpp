#ifndef RF_UTILITIES_HPP
#define RF_UTILITIES_HPP

#include "rf_types.hpp"

namespace rf {

// Compute error rate (comperr.f90)
void comperr(const integer_t* jest, const integer_t* cl, integer_t nsample, dp_t& errtr);

// Local importance calculation (localimp.f90) - CLASSIFICATION
void localimp(integer_t nsample, integer_t mdim, integer_t ntree,
              const real_t* qimp, real_t* qimpm);

// Local importance calculation for REGRESSION (Breiman Cutler adaptation)
// For regression: importance = INCREASE in MSE when feature is permuted
void localimp_regression(integer_t nsample, integer_t mdim, integer_t ntree,
                         const real_t* qimp, real_t* qimpm);

// Prepare data - sort and handle ties (prepdata.f90)
void prepdata(const real_t* x, integer_t mdim, integer_t nsample,
              const integer_t* cat, integer_t* isort, real_t* v,
              integer_t* asave, integer_t* ties);

} // namespace rf

#endif // RF_UTILITIES_HPP
