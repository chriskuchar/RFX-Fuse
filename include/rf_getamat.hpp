#ifndef RF_GETAMAT_HPP
#define RF_GETAMAT_HPP

#include "rf_types.hpp"

namespace rf {

// CPU getamat implementation
void cpu_getamat(const integer_t* asave, const integer_t* nin, const integer_t* cat,
                 integer_t nsample, integer_t mdim, integer_t* amat);

// GPU getamat implementation with parallel matrix operations
void gpu_getamat(const integer_t* asave, const integer_t* nin, const integer_t* cat,
                 integer_t nsample, integer_t mdim, integer_t* amat);

} // namespace rf

#endif // RF_GETAMAT_HPP
