#include <cuda_fp16.h>  // MUST be first - defines __half before cublas
#include "rf_mds_gpu.cuh"
#include "rf_cuda_config.hpp"
#include "rf_cuda_memory.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace rf {
namespace cuda {

namespace {
    // CUDA kernel: Convert proximity to distance matrix
    __global__ void proximity_to_distance_kernel(
        const dp_t* proximity,
        dp_t* distance,
        dp_t max_proximity,
        integer_t n_samples
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_samples * n_samples) {
            int i = idx / n_samples;
            int j = idx % n_samples;
            
            if (i == j) {
                distance[idx] = 0.0;
            } else {
                distance[idx] = max_proximity - proximity[idx];
            }
        }
    }
    
    // CUDA kernel: Compute row means
    __global__ void compute_row_means_kernel(
        const dp_t* distance,
        dp_t* row_means,
        integer_t n_samples
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n_samples) {
            dp_t sum = 0.0;
            for (int j = 0; j < n_samples; j++) {
                sum += distance[i * n_samples + j];
            }
            row_means[i] = sum / static_cast<dp_t>(n_samples);
        }
    }
    
    // CUDA kernel: Double-centering
    __global__ void double_center_kernel(
        const dp_t* distance,
        dp_t* centered,
        const dp_t* row_means,
        const dp_t* col_means,
        dp_t grand_mean,
        integer_t n_samples
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_samples * n_samples) {
            int i = idx / n_samples;
            int j = idx % n_samples;
            dp_t val = distance[idx] - row_means[i] - col_means[j] + grand_mean;
            centered[idx] = -0.5 * val;
        }
    }
}

size_t estimate_mds_gpu_memory(integer_t n_samples) {
    size_t n = static_cast<size_t>(n_samples);
    
    // Memory requirements on GPU:
    // - Distance matrix: n² × sizeof(double)
    // - Centered matrix: n² × sizeof(double)
    // - Row/column means: 2n × sizeof(double)
    // - Eigenvectors (from cuSolver): n² × sizeof(double)
    // - Eigenvalues: n × sizeof(double)
    // - cuSolver workspace: ~n² × sizeof(double)
    
    size_t matrix_size = n * n * sizeof(dp_t);
    size_t means_size = 2 * n * sizeof(dp_t);
    size_t eigenvalues_size = n * sizeof(dp_t);
    size_t workspace_size = (1 + 6 * n + 2 * n * n) * sizeof(dp_t); // cuSolver workspace estimate
    
    return 2 * matrix_size + means_size + eigenvalues_size + workspace_size;
}

std::pair<bool, std::string> check_gpu_mds_memory(integer_t n_samples) {
    size_t required = estimate_mds_gpu_memory(n_samples);
    
    // Use existing GPU memory checking infrastructure
    size_t available = rf::cuda::CudaConfig::instance().get_available_memory();
    
    // Require at least 1.5x the estimated memory for safety
    size_t required_with_safety = required * 1.5;
    
    if (required_with_safety > available) {
        double required_gb = required_with_safety / (1024.0 * 1024.0 * 1024.0);
        double available_gb = available / (1024.0 * 1024.0 * 1024.0);
        
        std::string error_msg = 
            "GPU MDS would fail due to insufficient GPU memory.\n"
            "  Required (with safety margin): " + std::to_string(required_gb) + " GB\n"
            "  Available GPU memory: " + std::to_string(available_gb) + " GB\n"
            "  Dataset size: " + std::to_string(n_samples) + " samples\n"
            "\n"
            "SOLUTION: Use CPU C++ MDS or reduce dataset size";
        
        return std::make_pair(false, error_msg);
    }
    
    return std::make_pair(true, "");
}

std::vector<double> compute_mds_3d_gpu(
    const dp_t* proximity_matrix_gpu,
    integer_t n_samples,
    bool memory_check
) {
    const int k = 3; // Fixed to 3 for 3D visualization
    
    if (n_samples < 2) {
        throw std::runtime_error("MDS requires at least 2 samples");
    }
    
    if (!proximity_matrix_gpu) {
        throw std::runtime_error("Proximity matrix is null");
    }
    
    // Memory check
    if (memory_check) {
        auto [can_compute, error_msg] = check_gpu_mds_memory(n_samples);
        if (!can_compute) {
            throw std::runtime_error(error_msg);
        }
    }
    
    size_t n = static_cast<size_t>(n_samples);
    size_t n2 = n * n;
    
    // Allocate GPU memory
    dp_t* d_distance = nullptr;
    dp_t* d_centered = nullptr;
    dp_t* d_row_means = nullptr;
    dp_t* d_col_means = nullptr;
    dp_t* d_eigenvalues = nullptr;
    dp_t* d_eigenvectors = nullptr;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_distance, n2 * sizeof(dp_t));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for distance matrix");
    }
    
    err = cudaMalloc(&d_centered, n2 * sizeof(dp_t));
    if (err != cudaSuccess) {
        // Safe free to prevent segfaults
        if (d_distance) { cudaError_t err = cudaFree(d_distance); if (err != cudaSuccess) cudaGetLastError(); }
        throw std::runtime_error("Failed to allocate GPU memory for centered matrix");
    }
    
    err = cudaMalloc(&d_row_means, n * sizeof(dp_t));
    if (err != cudaSuccess) {
        // Safe free to prevent segfaults
        if (d_distance) { cudaError_t err = cudaFree(d_distance); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_centered) { cudaError_t err = cudaFree(d_centered); if (err != cudaSuccess) cudaGetLastError(); }
        throw std::runtime_error("Failed to allocate GPU memory for row means");
    }
    
    err = cudaMalloc(&d_col_means, n * sizeof(dp_t));
    if (err != cudaSuccess) {
        // Safe free to prevent segfaults
        if (d_distance) { cudaError_t err = cudaFree(d_distance); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_centered) { cudaError_t err = cudaFree(d_centered); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_row_means) { cudaError_t err = cudaFree(d_row_means); if (err != cudaSuccess) cudaGetLastError(); }
        throw std::runtime_error("Failed to allocate GPU memory for column means");
    }
    
    err = cudaMalloc(&d_eigenvalues, n * sizeof(dp_t));
    if (err != cudaSuccess) {
        // Safe free to prevent segfaults
        if (d_distance) { cudaError_t err = cudaFree(d_distance); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_centered) { cudaError_t err = cudaFree(d_centered); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_row_means) { cudaError_t err = cudaFree(d_row_means); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_col_means) { cudaError_t err = cudaFree(d_col_means); if (err != cudaSuccess) cudaGetLastError(); }
        throw std::runtime_error("Failed to allocate GPU memory for eigenvalues");
    }
    
    err = cudaMalloc(&d_eigenvectors, n2 * sizeof(dp_t));
    if (err != cudaSuccess) {
        // Safe free to prevent segfaults
        if (d_distance) { cudaError_t err = cudaFree(d_distance); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_centered) { cudaError_t err = cudaFree(d_centered); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_row_means) { cudaError_t err = cudaFree(d_row_means); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_col_means) { cudaError_t err = cudaFree(d_col_means); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_eigenvalues) { cudaError_t err = cudaFree(d_eigenvalues); if (err != cudaSuccess) cudaGetLastError(); }
        throw std::runtime_error("Failed to allocate GPU memory for eigenvectors");
    }
    
    try {
        // Step 1: Convert proximity to distance matrix
        // First find max proximity on GPU
        dp_t max_prox = 0.0;
        // Use thrust or simple reduction kernel - for now, do on CPU
        std::vector<dp_t> prox_host(n2);
        cudaMemcpy(prox_host.data(), proximity_matrix_gpu, n2 * sizeof(dp_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n2; i++) {
            if (prox_host[i] > max_prox) max_prox = prox_host[i];
        }
        
        int threads_per_block = 256;
        int blocks = (n2 + threads_per_block - 1) / threads_per_block;
        proximity_to_distance_kernel<<<blocks, threads_per_block>>>(
            proximity_matrix_gpu, d_distance, max_prox, n_samples
        );
        cudaStreamSynchronize(0);  // Use stream sync for Jupyter safety
        
        // Step 2: Double-centering
        // Compute row means
        blocks = (n_samples + threads_per_block - 1) / threads_per_block;
        compute_row_means_kernel<<<blocks, threads_per_block>>>(
            d_distance, d_row_means, n_samples
        );
        
        // Compute column means (transpose)
        // For simplicity, do this on CPU for now
        std::vector<dp_t> distance_host(n2);
        cudaMemcpy(distance_host.data(), d_distance, n2 * sizeof(dp_t), cudaMemcpyDeviceToHost);
        std::vector<dp_t> col_means_host(n);
        for (size_t j = 0; j < n; j++) {
            dp_t sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                sum += distance_host[i * n + j];
            }
            col_means_host[j] = sum / static_cast<dp_t>(n);
        }
        cudaMemcpy(d_col_means, col_means_host.data(), n * sizeof(dp_t), cudaMemcpyHostToDevice);
        
        // Compute grand mean
        std::vector<dp_t> row_means_host(n);
        cudaMemcpy(row_means_host.data(), d_row_means, n * sizeof(dp_t), cudaMemcpyDeviceToHost);
        dp_t grand_mean = 0.0;
        for (size_t i = 0; i < n; i++) {
            grand_mean += row_means_host[i];
        }
        grand_mean /= static_cast<dp_t>(n);
        
        // Apply double-centering
        blocks = (n2 + threads_per_block - 1) / threads_per_block;
        double_center_kernel<<<blocks, threads_per_block>>>(
            d_distance, d_centered, d_row_means, d_col_means, grand_mean, n_samples
        );
        cudaStreamSynchronize(0);  // Use stream sync for Jupyter safety
        
        // Step 3: Eigendecomposition using cuSolver
        cusolverDnHandle_t cusolver_handle;
        cusolverDnCreate(&cusolver_handle);
        
        // Query workspace size
        int lwork = 0;
        cusolverDnDsyevd_bufferSize(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,  // Compute eigenvectors
            CUBLAS_FILL_MODE_UPPER,     // Upper triangular
            n_samples,
            d_centered,
            n_samples,
            d_eigenvalues,
            &lwork
        );
        
        dp_t* d_work = nullptr;
        cudaMalloc(&d_work, lwork * sizeof(dp_t));
        
        int* dev_info = nullptr;
        cudaMalloc(&dev_info, sizeof(int));
        
        // Copy centered matrix to eigenvectors (will be overwritten)
        cudaMemcpy(d_eigenvectors, d_centered, n2 * sizeof(dp_t), cudaMemcpyDeviceToDevice);
        
        // Compute eigendecomposition
        cusolverDnDsyevd(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_UPPER,
            n_samples,
            d_eigenvectors,
            n_samples,
            d_eigenvalues,
            d_work,
            lwork,
            dev_info
        );
        
        cudaStreamSynchronize(0);  // Use stream sync for Jupyter safety
        
        int info = 0;
        cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (info != 0) {
            // Safe free to prevent segfaults
            if (d_work) { cudaError_t err = cudaFree(d_work); if (err != cudaSuccess) cudaGetLastError(); }
            if (dev_info) { cudaError_t err = cudaFree(dev_info); if (err != cudaSuccess) cudaGetLastError(); }
            cusolverDnDestroy(cusolver_handle);
            throw std::runtime_error("cuSolver eigendecomposition failed with error code: " + std::to_string(info));
        }
        
        // Step 4: Extract top 3 eigenvectors
        std::vector<dp_t> eigenvalues_host(n);
        std::vector<dp_t> eigenvectors_host(n2);
        cudaMemcpy(eigenvalues_host.data(), d_eigenvalues, n * sizeof(dp_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(eigenvectors_host.data(), d_eigenvectors, n2 * sizeof(dp_t), cudaMemcpyDeviceToHost);
        
        // Sort eigenvalues (cuSolver returns in ascending order)
        std::vector<std::pair<dp_t, int>> eigen_pairs(n);
        for (size_t i = 0; i < n; i++) {
            eigen_pairs[i] = std::make_pair(eigenvalues_host[i], static_cast<int>(i));
        }
        
        std::sort(eigen_pairs.rbegin(), eigen_pairs.rend(),
                  [](const std::pair<dp_t, int>& a, const std::pair<dp_t, int>& b) {
                      return std::abs(a.first) < std::abs(b.first);
                  });
        
        // Extract top k=3 eigenvectors
        std::vector<double> coords_3d(n * k);
        for (int comp = 0; comp < k; comp++) {
            int eigen_idx = eigen_pairs[comp].second;
            dp_t eigenval = eigen_pairs[comp].first;
            dp_t scale = (eigenval > 0) ? std::sqrt(eigenval) : 0.0;
            
            for (size_t i = 0; i < n; i++) {
                // cuSolver returns column-major, so eigenvector is column eigen_idx
                coords_3d[i * k + comp] = eigenvectors_host[eigen_idx * n + i] * scale;
            }
        }
        
        // Cleanup (safe free to prevent segfaults)
        if (d_work) { cudaError_t err = cudaFree(d_work); if (err != cudaSuccess) cudaGetLastError(); }
        if (dev_info) { cudaError_t err = cudaFree(dev_info); if (err != cudaSuccess) cudaGetLastError(); }
        cusolverDnDestroy(cusolver_handle);
        
        // Free GPU memory
        // Safe free to prevent segfaults
        if (d_distance) { cudaError_t err = cudaFree(d_distance); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_centered) { cudaError_t err = cudaFree(d_centered); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_row_means) { cudaError_t err = cudaFree(d_row_means); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_col_means) { cudaError_t err = cudaFree(d_col_means); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_eigenvalues) { cudaError_t err = cudaFree(d_eigenvalues); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_eigenvectors) { cudaError_t err = cudaFree(d_eigenvectors); if (err != cudaSuccess) cudaGetLastError(); }
        
        return coords_3d;
        
    } catch (...) {
        // Cleanup on error
        // Safe free to prevent segfaults
        if (d_distance) { cudaError_t err = cudaFree(d_distance); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_centered) { cudaError_t err = cudaFree(d_centered); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_row_means) { cudaError_t err = cudaFree(d_row_means); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_col_means) { cudaError_t err = cudaFree(d_col_means); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_eigenvalues) { cudaError_t err = cudaFree(d_eigenvalues); if (err != cudaSuccess) cudaGetLastError(); }
        // Safe free to prevent segfaults
        if (d_eigenvectors) { cudaError_t err = cudaFree(d_eigenvectors); if (err != cudaSuccess) cudaGetLastError(); }
        throw;
    }
}

} // namespace cuda
} // namespace rf

