#ifndef RF_MEMORY_CUH
#define RF_MEMORY_CUH

#include "rf_types.hpp"
#include "rf_config.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

// CUDA error checking macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)
#endif

#ifndef CUDA_CHECK_VOID
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)
#endif

// Device data structure for tree nodes (matches cuda_rf_types.cuf)
struct TreeNode {
    integer_t nodestatus;    // 1=split, 2=not split, -1=terminal
    integer_t bestvar;       // variable used for splitting
    integer_t nodeclass;     // majority class in node
    integer_t left_child;    // left child node index
    integer_t right_child;   // right child node index
    real_t xbestsplit;       // split value for quantitative variables
    real_t nodewt;           // node weight
    integer_t nodestart;     // start index in data
    integer_t nodestop;      // stop index in data
};

// Device data structure for proximity computation
struct ProximityData {
    integer_t* terminal_nodes;
    integer_t* node_sizes;
    integer_t* node_starts;
    dp_t* proximities;
};

// Device data structure for bootstrap sampling
struct BootstrapData {
    integer_t* sample_indices;
    integer_t* sample_counts;
    real_t* sample_weights;
    bool* in_bag;
};

// Device random state for CURAND
struct RandomState {
    curandState* states;
    integer_t num_states;
};

// Allocate device memory for proximity matrices (matches cuda_allocate_proximity)
bool allocate_proximity(integer_t nsample, dp_t*& prox_dev, dp_t*& proxsym_dev);

// Deallocate device proximity matrices
void deallocate_proximity(dp_t* prox_dev, dp_t* proxsym_dev);

// Transfer host data to device (real)
bool host_to_device_real(const real_t* host_data, real_t* device_data, integer_t n);

// Transfer host data to device (integer)
bool host_to_device_int(const integer_t* host_data, integer_t* device_data, integer_t n);

// Transfer device data to host (real)
bool device_to_host_real(const real_t* device_data, real_t* host_data, integer_t n);

// Transfer device data to host (double precision)
bool device_to_host_double(const dp_t* device_data, dp_t* host_data, integer_t n, integer_t m);

// Allocate device memory for tree data
bool allocate_tree_data(integer_t nsample, integer_t nnode, integer_t maxcat,
                        integer_t*& nodestatus_dev,
                        integer_t*& bestvar_dev,
                        integer_t*& nodeclass_dev,
                        integer_t*& treemap_dev,
                        real_t*& xbestsplit_dev,
                        integer_t*& catgoleft_dev);

// Deallocate device tree data
void deallocate_tree_data(integer_t* nodestatus_dev,
                          integer_t* bestvar_dev,
                          integer_t* nodeclass_dev,
                          integer_t* treemap_dev,
                          real_t* xbestsplit_dev,
                          integer_t* catgoleft_dev);

// Initialize CURAND states for random number generation
bool init_random_states(RandomState& rand_state, integer_t num_states, integer_t seed);

// CUDA kernel to initialize CURAND states
__global__ void init_curand_kernel(curandState* states, integer_t num_states, integer_t seed);

// Cleanup CURAND states
void cleanup_random_states(RandomState& rand_state);

} // namespace cuda
} // namespace rf

#endif // RF_MEMORY_CUH
