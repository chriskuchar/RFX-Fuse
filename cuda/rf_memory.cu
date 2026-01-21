#include "rf_memory.cuh"
#include "rf_cuda_config.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rf {
namespace cuda {

// Allocate device memory for proximity matrices
bool allocate_proximity(integer_t nsample, dp_t*& prox_dev, dp_t*& proxsym_dev) {
    size_t size = nsample * nsample * sizeof(dp_t);

    CUDA_CHECK(cudaMalloc(&prox_dev, size));
    CUDA_CHECK(cudaMalloc(&proxsym_dev, size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(prox_dev, 0, size));
    CUDA_CHECK(cudaMemset(proxsym_dev, 0, size));

    return true;
}

// Deallocate device proximity matrices
void deallocate_proximity(dp_t* prox_dev, dp_t* proxsym_dev) {
    if (prox_dev) cudaFree(prox_dev);
    if (proxsym_dev) cudaFree(proxsym_dev);
}

// Transfer host data to device (real)
bool host_to_device_real(const real_t* host_data, real_t* device_data, integer_t n) {
    CUDA_CHECK(cudaMemcpy(device_data, host_data, n * sizeof(real_t), cudaMemcpyHostToDevice));
    return true;
}

// Transfer host data to device (integer)
bool host_to_device_int(const integer_t* host_data, integer_t* device_data, integer_t n) {
    CUDA_CHECK(cudaMemcpy(device_data, host_data, n * sizeof(integer_t), cudaMemcpyHostToDevice));
    return true;
}

// Transfer device data to host (real)
bool device_to_host_real(const real_t* device_data, real_t* host_data, integer_t n) {
    CUDA_CHECK(cudaMemcpy(host_data, device_data, n * sizeof(real_t), cudaMemcpyDeviceToHost));
    return true;
}

// Transfer device data to host (double precision)
bool device_to_host_double(const dp_t* device_data, dp_t* host_data, integer_t n, integer_t m) {
    CUDA_CHECK(cudaMemcpy(host_data, device_data, n * m * sizeof(dp_t), cudaMemcpyDeviceToHost));
    return true;
}

// Allocate device memory for tree data
bool allocate_tree_data(integer_t nsample, integer_t nnode, integer_t maxcat,
                        integer_t*& nodestatus_dev,
                        integer_t*& bestvar_dev,
                        integer_t*& nodeclass_dev,
                        integer_t*& treemap_dev,
                        real_t*& xbestsplit_dev,
                        integer_t*& catgoleft_dev) {
    CUDA_CHECK(cudaMalloc(&nodestatus_dev, nnode * sizeof(integer_t)));
    CUDA_CHECK(cudaMalloc(&bestvar_dev, nnode * sizeof(integer_t)));
    CUDA_CHECK(cudaMalloc(&nodeclass_dev, nnode * sizeof(integer_t)));
    CUDA_CHECK(cudaMalloc(&treemap_dev, 2 * nnode * sizeof(integer_t)));
    CUDA_CHECK(cudaMalloc(&xbestsplit_dev, nnode * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&catgoleft_dev, maxcat * nnode * sizeof(integer_t)));

    return true;
}

// Deallocate device tree data
void deallocate_tree_data(integer_t* nodestatus_dev,
                          integer_t* bestvar_dev,
                          integer_t* nodeclass_dev,
                          integer_t* treemap_dev,
                          real_t* xbestsplit_dev,
                          integer_t* catgoleft_dev) {
    if (nodestatus_dev) cudaFree(nodestatus_dev);
    if (bestvar_dev) cudaFree(bestvar_dev);
    if (nodeclass_dev) cudaFree(nodeclass_dev);
    if (treemap_dev) cudaFree(treemap_dev);
    if (xbestsplit_dev) cudaFree(xbestsplit_dev);
    if (catgoleft_dev) cudaFree(catgoleft_dev);
}

// CUDA kernel to initialize CURAND states
__global__ void init_curand_kernel(curandState* states, integer_t num_states, integer_t seed) {
    integer_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_states) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Initialize CURAND states for random number generation
bool init_random_states(RandomState& rand_state, integer_t num_states, integer_t seed) {
    rand_state.num_states = num_states;

    CUDA_CHECK(cudaMalloc(&rand_state.states, num_states * sizeof(curandState)));

    // Launch kernel to initialize states
    integer_t block_size = cuda::get_optimal_block_size(1024);
    integer_t num_blocks = (num_states + block_size - 1) / block_size;
    init_curand_kernel<<<num_blocks, block_size>>>(rand_state.states, num_states, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety

    return true;
}

// Cleanup CURAND states
void cleanup_random_states(RandomState& rand_state) {
    if (rand_state.states) {
        try {
            cudaError_t err = cudaFree(rand_state.states);
            // Ignore errors - CUDA context might be destroyed during process shutdown
            rand_state.states = nullptr;
            rand_state.num_states = 0;
        } catch (...) {
            // Ignore all exceptions during cleanup
            rand_state.states = nullptr;
            rand_state.num_states = 0;
        }
    }
}

} // namespace cuda
} // namespace rf
