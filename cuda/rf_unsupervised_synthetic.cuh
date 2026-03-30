#ifndef RF_UNSUPERVISED_SYNTHETIC_CUH
#define RF_UNSUPERVISED_SYNTHETIC_CUH

#include "rf_types.hpp"
#include <cuda_runtime.h>

namespace rf {
namespace cuda {

/**
 * Generate synthetic (Breiman-Cutler) data directly on GPU.
 *
 * For unsupervised random forests, synthetic samples are created by
 * independently permuting each feature column — destroying inter-feature
 * correlations while preserving marginal distributions.
 *
 * x_gpu must already contain the real data in rows [0, n_real).
 * This function fills rows [n_real, n_real + n_synthetic) with
 * per-feature shuffled copies of the real data.
 *
 * @param x_gpu      Device pointer to feature matrix [n_total x mdim], row-major.
 *                   Rows 0..n_real-1 must be populated with real data on entry.
 * @param n_real     Number of real samples (already in x_gpu).
 * @param n_synthetic Number of synthetic samples to generate.
 * @param mdim       Number of features (columns).
 * @param seed       Base random seed (each feature gets seed + feature_id).
 * @param stream     CUDA stream for async execution.
 */
void gpu_generate_synthetic_unsupervised(
    real_t* x_gpu,
    integer_t n_real,
    integer_t n_synthetic,
    integer_t mdim,
    integer_t seed,
    cudaStream_t stream = 0
);

}  // namespace cuda
}  // namespace rf

#endif  // RF_UNSUPERVISED_SYNTHETIC_CUH
