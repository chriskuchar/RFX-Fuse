#ifndef RF_BOOTSTRAP_HPP
#define RF_BOOTSTRAP_HPP

#include "rf_types.hpp"

namespace rf {

// CPU bootstrap implementation
void cpu_bootstrap(const real_t* weight, integer_t nsample,
                   real_t* win, integer_t* nin, integer_t* nout,
                   integer_t* jinbag, integer_t* joobag,
                   integer_t& ninbag, integer_t& noobag,
                   integer_t tree_id = 0);

} // namespace rf

#endif // RF_BOOTSTRAP_HPP
