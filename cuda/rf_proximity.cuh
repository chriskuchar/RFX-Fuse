#ifndef RF_PROXIMITY_CUH
#define RF_PROXIMITY_CUH

#include "rf_types.hpp"

namespace rf {

// Main wrapper - automatically selects GPU or CPU (matches proximity_cuda in Fortran)
void proximity(const integer_t* nodestatus, const integer_t* nodextr,
               const integer_t* nin, integer_t nsample, integer_t nnode,
               dp_t* prox, integer_t* nod, integer_t* ncount,
               integer_t* ncn, integer_t* nodexb, integer_t* ndbegin,
               integer_t* npcase);

// CPU fallback - exact copy of original algorithm (matches proximity_cpu_compute_only in Fortran)
void proximity_cpu_fallback(const integer_t* nodestatus, const integer_t* nodextr,
                            const integer_t* nin, integer_t nsample, integer_t nnode,
                            dp_t* prox, integer_t* nod, integer_t* ncount,
                            integer_t* ncn, integer_t* nodexb, integer_t* ndbegin,
                            integer_t* npcase);

// Cleanup function for global proximity states
void cleanup_proximity_states();

} // namespace rf

#endif // RF_PROXIMITY_CUH
