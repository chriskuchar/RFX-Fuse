#include "rf_cuda_memory.cuh"

namespace rf {
namespace cuda {

// Forward declaration of init_curand_kernel from rf_memory.cu
extern __global__ void init_curand_kernel(curandState* states, integer_t num_states, integer_t seed);

// ============================================================================
// RandomStates Implementation
// ============================================================================

RandomStates::RandomStates(integer_t num_states, integer_t seed)
    : states_(nullptr), num_states_(num_states) {
    CUDA_CHECK(cudaMalloc(&states_, num_states * sizeof(curandState)));

    // Launch kernel to initialize states
    integer_t block_size = 256;
    integer_t num_blocks = (num_states + block_size - 1) / block_size;
    init_curand_kernel<<<num_blocks, block_size>>>(states_, num_states, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));  // Use stream sync for Jupyter safety
}

RandomStates::~RandomStates() {
    if (states_) cudaFree(states_);
}

RandomStates::RandomStates(RandomStates&& other) noexcept
    : states_(other.states_), num_states_(other.num_states_) {
    other.states_ = nullptr;
    other.num_states_ = 0;
}

RandomStates& RandomStates::operator=(RandomStates&& other) noexcept {
    if (this != &other) {
        if (states_) cudaFree(states_);
        states_ = other.states_;
        num_states_ = other.num_states_;
        other.states_ = nullptr;
        other.num_states_ = 0;
    }
    return *this;
}

// ============================================================================
// ProximityMatrices Implementation
// ============================================================================

ProximityMatrices::ProximityMatrices(integer_t nsample)
    : nsample_(nsample),
      prox_(cuda_allocate_zero<dp_t>(nsample * nsample)),
      proxsym_(cuda_allocate_zero<dp_t>(nsample * nsample)) {
}

void ProximityMatrices::to_host(dp_t* prox_host, dp_t* proxsym_host) {
    size_t size = nsample_ * nsample_ * sizeof(dp_t);
    CUDA_CHECK(cudaMemcpy(prox_host, prox_.get(), size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(proxsym_host, proxsym_.get(), size, cudaMemcpyDeviceToHost));
}

// ============================================================================
// TreeData Implementation
// ============================================================================

TreeData::TreeData(integer_t nnode, integer_t maxcat)
    : nnode_(nnode),
      maxcat_(maxcat),
      nodestatus_(cuda_allocate<integer_t>(nnode)),
      bestvar_(cuda_allocate<integer_t>(nnode)),
      nodeclass_(cuda_allocate<integer_t>(nnode)),
      treemap_(cuda_allocate<integer_t>(2 * nnode)),
      xbestsplit_(cuda_allocate<real_t>(nnode)),
      catgoleft_(cuda_allocate<integer_t>(maxcat * nnode)) {
}

void TreeData::from_host(const integer_t* nodestatus_host,
                         const integer_t* bestvar_host,
                         const integer_t* nodeclass_host,
                         const integer_t* treemap_host,
                         const real_t* xbestsplit_host,
                         const integer_t* catgoleft_host,
                         integer_t nnode) {
    CUDA_CHECK(cudaMemcpy(nodestatus_.get(), nodestatus_host, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bestvar_.get(), bestvar_host, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nodeclass_.get(), nodeclass_host, nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(treemap_.get(), treemap_host, 2 * nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(xbestsplit_.get(), xbestsplit_host, nnode * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(catgoleft_.get(), catgoleft_host, maxcat_ * nnode * sizeof(integer_t), cudaMemcpyHostToDevice));
}

// ============================================================================
// VarimpArrays Implementation
// ============================================================================

VarimpArrays::VarimpArrays(integer_t nsample, integer_t mdim)
    : nsample_(nsample),
      mdim_(mdim),
      qimp_(cuda_allocate_zero<real_t>(nsample)),
      qimpm_(cuda_allocate_zero<real_t>(nsample * mdim)),
      avimp_(cuda_allocate_zero<real_t>(mdim)),
      sqsd_(cuda_allocate_zero<real_t>(mdim)) {
}

void VarimpArrays::zero() {
    CUDA_CHECK(cudaMemset(qimp_.get(), 0, nsample_ * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(qimpm_.get(), 0, nsample_ * mdim_ * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(avimp_.get(), 0, mdim_ * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(sqsd_.get(), 0, mdim_ * sizeof(real_t)));
}

void VarimpArrays::to_host(real_t* qimp_host, real_t* qimpm_host,
                            real_t* avimp_host, real_t* sqsd_host) {
    CUDA_CHECK(cudaMemcpy(qimp_host, qimp_.get(), nsample_ * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qimpm_host, qimpm_.get(), nsample_ * mdim_ * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(avimp_host, avimp_.get(), mdim_ * sizeof(real_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sqsd_host, sqsd_.get(), mdim_ * sizeof(real_t), cudaMemcpyDeviceToHost));
}

} // namespace cuda
} // namespace rf
