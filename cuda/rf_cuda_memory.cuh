#ifndef RF_CUDA_MEMORY_CUH
#define RF_CUDA_MEMORY_CUH

#include "rf_types.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <stdexcept>

namespace rf {
namespace cuda {

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Custom deleter for CUDA device memory
template<typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

// RAII wrapper for CUDA device memory
template<typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

// Helper to allocate CUDA device memory
template<typename T>
CudaPtr<T> cuda_allocate(size_t n) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    return CudaPtr<T>(ptr);
}

// Helper to allocate and zero CUDA device memory
template<typename T>
CudaPtr<T> cuda_allocate_zero(size_t n) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    return CudaPtr<T>(ptr);
}

// RAII wrapper for CURAND states
class RandomStates {
public:
    RandomStates(integer_t num_states, integer_t seed);
    ~RandomStates();

    // No copy
    RandomStates(const RandomStates&) = delete;
    RandomStates& operator=(const RandomStates&) = delete;

    // Move allowed
    RandomStates(RandomStates&& other) noexcept;
    RandomStates& operator=(RandomStates&& other) noexcept;

    curandState* get() { return states_; }
    integer_t size() const { return num_states_; }

private:
    curandState* states_;
    integer_t num_states_;
};

// RAII wrapper for device proximity matrices
class ProximityMatrices {
public:
    ProximityMatrices(integer_t nsample);
    ~ProximityMatrices() = default;

    // No copy
    ProximityMatrices(const ProximityMatrices&) = delete;
    ProximityMatrices& operator=(const ProximityMatrices&) = delete;

    // Move allowed
    ProximityMatrices(ProximityMatrices&&) = default;
    ProximityMatrices& operator=(ProximityMatrices&&) = default;

    dp_t* prox() { return prox_.get(); }
    dp_t* proxsym() { return proxsym_.get(); }
    integer_t size() const { return nsample_; }

    // Copy to host
    void to_host(dp_t* prox_host, dp_t* proxsym_host);

private:
    integer_t nsample_;
    CudaPtr<dp_t> prox_;
    CudaPtr<dp_t> proxsym_;
};

// RAII wrapper for device tree data
class TreeData {
public:
    TreeData(integer_t nnode, integer_t maxcat);
    ~TreeData() = default;

    // No copy
    TreeData(const TreeData&) = delete;
    TreeData& operator=(const TreeData&) = delete;

    // Move allowed
    TreeData(TreeData&&) = default;
    TreeData& operator=(TreeData&&) = default;

    integer_t* nodestatus() { return nodestatus_.get(); }
    integer_t* bestvar() { return bestvar_.get(); }
    integer_t* nodeclass() { return nodeclass_.get(); }
    integer_t* treemap() { return treemap_.get(); }
    real_t* xbestsplit() { return xbestsplit_.get(); }
    integer_t* catgoleft() { return catgoleft_.get(); }

    // Copy to device from host
    void from_host(const integer_t* nodestatus_host,
                   const integer_t* bestvar_host,
                   const integer_t* nodeclass_host,
                   const integer_t* treemap_host,
                   const real_t* xbestsplit_host,
                   const integer_t* catgoleft_host,
                   integer_t nnode);

private:
    integer_t nnode_;
    integer_t maxcat_;
    CudaPtr<integer_t> nodestatus_;
    CudaPtr<integer_t> bestvar_;
    CudaPtr<integer_t> nodeclass_;
    CudaPtr<integer_t> treemap_;
    CudaPtr<real_t> xbestsplit_;
    CudaPtr<integer_t> catgoleft_;
};

// RAII wrapper for device variable importance arrays
class VarimpArrays {
public:
    VarimpArrays(integer_t nsample, integer_t mdim);
    ~VarimpArrays() = default;

    // No copy
    VarimpArrays(const VarimpArrays&) = delete;
    VarimpArrays& operator=(const VarimpArrays&) = delete;

    // Move allowed
    VarimpArrays(VarimpArrays&&) = default;
    VarimpArrays& operator=(VarimpArrays&&) = default;

    real_t* qimp() { return qimp_.get(); }
    real_t* qimpm() { return qimpm_.get(); }
    real_t* avimp() { return avimp_.get(); }
    real_t* sqsd() { return sqsd_.get(); }

    // Zero all arrays
    void zero();

    // Copy to host
    void to_host(real_t* qimp_host, real_t* qimpm_host,
                 real_t* avimp_host, real_t* sqsd_host);

private:
    integer_t nsample_;
    integer_t mdim_;
    CudaPtr<real_t> qimp_;
    CudaPtr<real_t> qimpm_;
    CudaPtr<real_t> avimp_;
    CudaPtr<real_t> sqsd_;
};

} // namespace cuda
} // namespace rf

#endif // RF_CUDA_MEMORY_CUH
