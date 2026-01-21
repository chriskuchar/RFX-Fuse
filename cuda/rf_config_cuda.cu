#include "rf_config.hpp"
#include "rf_types.hpp"
#include "rf_memory.cuh"
#include "rf_bootstrap.cuh"
#include "rf_varimp.cuh"
#include "rf_proximity.cuh"
#include "rf_utils.hpp"
#include "rf_cuda_config.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <thread>
#include <chrono>
#include <unistd.h>  // For getpid() - fork detection

namespace rf {
namespace cuda {

// Global CUDA context management
static bool g_cuda_initialized = false;
static bool g_cuda_available = false;
static pid_t g_cuda_init_pid = 0;  // Track which process initialized CUDA

// Forward declaration
void cuda_reset_device();

bool cuda_is_available() {
    // WSL workaround: cudaGetDeviceCount can fail with "out of memory" even when GPU works
    // Try cudaGetDeviceCount first, but if it fails, try direct device access
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err == cudaSuccess && device_count > 0) {
        return true;
    }
    
    // If cudaGetDeviceCount failed, try direct device access (WSL workaround)
    if (err == cudaErrorMemoryAllocation || err == cudaErrorInitializationError) {
        cudaGetLastError();  // Clear error
        err = cudaSetDevice(0);
        if (err == cudaSuccess) {
            // Try a test allocation to verify device works
            void* test_ptr = nullptr;
            cudaError_t test_err = cudaMalloc(&test_ptr, 1);
            if (test_err == cudaSuccess && test_ptr != nullptr) {
                cudaFree(test_ptr);
                return true;  // Device is accessible
            }
            cudaGetLastError();  // Clear error
        }
    }
    
    return false;
}

bool cuda_init_runtime(bool force_cpu) {
    // XGBoost-style fork detection using PID tracking
    // CUDA contexts cannot survive fork() - we must detect and reinitialize
    // This is essential for Jupyter kernels which fork from a parent process
    pid_t current_pid = getpid();
    
    if (g_cuda_initialized) {
        // Check if we're in a different process (fork detected)
        if (g_cuda_init_pid != 0 && g_cuda_init_pid != current_pid) {
            // Fork detected! CUDA context from parent is invalid
            // DON'T call cudaDeviceReset() - it can crash/hang in forked child
            // Just reset flags and reinitialize fresh
            
            // Reset all flags to force reinitialization
            g_cuda_initialized = false;
            g_cuda_available = false;
            g_cuda_init_pid = 0;
            // Fall through to reinitialize
        } else {
            // Same process, return cached result
            return g_cuda_available;
        }
    }
    
    g_cuda_initialized = true;
    g_cuda_init_pid = current_pid;  // Track which PID initialized CUDA
    
    if (force_cpu) {
        g_cuda_available = false;
        return false;
    }

    cudaGetLastError();  // Clear any stale errors
    
    // WSL/CUDA workaround: Clear any existing CUDA errors first
    // In WSL, cudaGetDeviceCount can fail with "out of memory" even when GPU is available.
    // We'll try cudaGetDeviceCount first, but if it fails, we'll attempt direct device access.
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    bool device_count_ok = (err == cudaSuccess && device_count > 0);
    
    // WSL workaround: If cudaGetDeviceCount fails, try direct device access anyway
    // This is a known WSL issue where cudaGetDeviceCount can fail even when device works
    if (!device_count_ok) {
        // std::cout << "CUDA: cudaGetDeviceCount failed (" << cudaGetErrorString(err) 
                //   << "), attempting direct device access (WSL workaround)..." << std::endl;
        // Clear the error and try to set device directly
        cudaGetLastError();
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            // std::cerr << "CUDA: cudaSetDevice(0) failed: " << err << " (" << cudaGetErrorString(err) << ")" << std::endl;
        } else {
            // If we can set the device, try a test allocation
            void* test_ptr = nullptr;
            cudaError_t test_err = cudaMalloc(&test_ptr, 1);
            if (test_err != cudaSuccess) {
                // std::cerr << "CUDA: Test allocation failed: " << test_err << " (" << cudaGetErrorString(test_err) << ")" << std::endl;
                cudaGetLastError();  // Clear error
            } else if (test_ptr != nullptr) {
                cudaFree(test_ptr);
                // std::cout << "CUDA: Direct device access successful (WSL workaround)" << std::endl;
                device_count = 1;  // Assume 1 device since we can access it
                device_count_ok = true;
            }
        }
    }
    
    if (!device_count_ok) {
        // std::cerr << "CUDA: Cannot access GPU devices" << std::endl;
        g_cuda_available = false;
        return false;
    }
    
    err = cudaSetDevice(0);
    if (err == cudaErrorDeviceAlreadyInUse || err == cudaErrorDevicesUnavailable) {
        // Device is busy - reset it to clear any stuck operations
        // std::cout << "CUDA: Device busy, resetting..." << std::endl;
        cuda_reset_device();
        // Try again after reset
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            // std::cerr << "CUDA: Device reset failed: " << err << " (" << cudaGetErrorString(err) << ")" << std::endl;
            g_cuda_available = false;
            return false;
        }
    } else if (err != cudaSuccess) {
        // std::cerr << "CUDA: Failed to set device: " << err << " (" << cudaGetErrorString(err) << ")" << std::endl;
        g_cuda_available = false;
        return false;
    }

    // Get device properties (this also validates the device is accessible)
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        // std::cerr << "CUDA: Failed to get device properties: " << err << " (" << cudaGetErrorString(err) << ")" << std::endl;
        g_cuda_available = false;
        return false;
    }
    
    // std::cout << "CUDA: Device properties retrieved - " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;

    // During initialization, we use cudaDeviceSynchronize() to ensure context is ready.
    // This matches the backup version's approach. However, if there are stuck operations from
    // a previous run, this can hang. To prevent hangs, we first try a non-blocking check:
    // clear errors and test with a simple allocation. Only if that succeeds do we synchronize.
    cudaGetLastError();  // Clear any previous errors
    
    // Test context with a simple non-blocking allocation first
    void* test_alloc = nullptr;
    cudaError_t test_err = cudaMalloc(&test_alloc, 1);
    
    if (test_err != cudaSuccess) {
        // Context might be corrupted - try reset
        // std::cout << "CUDA: Context test failed (" << cudaGetErrorString(test_err) << "), resetting device..." << std::endl;
        cuda_reset_device();
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            // std::cerr << "CUDA: Failed to set device after reset: " << err << " (" << cudaGetErrorString(err) << ")" << std::endl;
            g_cuda_available = false;
            return false;
        }
        // Retry allocation after reset
        test_err = cudaMalloc(&test_alloc, 1);
        if (test_err != cudaSuccess) {
            // std::cerr << "CUDA: Context test failed after reset: " << test_err << " (" << cudaGetErrorString(test_err) << ")" << std::endl;
            g_cuda_available = false;
            return false;
        }
    }
    // Free test allocation
    if (test_alloc) { cudaError_t free_err = cudaFree(test_alloc); if (free_err != cudaSuccess) cudaGetLastError(); }
    
    // JUPYTER SAFETY: Use stream sync instead of device sync for Jupyter compatibility
    // cudaStreamSynchronize(0) is safer after fork (Jupyter kernels) than cudaDeviceSynchronize()
    // Device sync can hang if there's stale state from parent process after fork
    err = cudaStreamSynchronize(0);
    
    if (err != cudaSuccess) {
        g_cuda_available = false;
        return false;
    }

    // Test with a simple memory allocation to verify context is fully ready
    void* test_ptr = nullptr;
    err = cudaMalloc(&test_ptr, 1024);
    if (err != cudaSuccess || test_ptr == nullptr) {
        // std::cerr << "CUDA: Failed memory allocation test: " << err << " (" << cudaGetErrorString(err) << ")" << std::endl;
        g_cuda_available = false;
        return false;
    }
    // Safe free to prevent segfaults
    if (test_ptr) { cudaError_t err = cudaFree(test_ptr); if (err != cudaSuccess) cudaGetLastError(); }

    g_cuda_available = true;
    // std::cout << "CUDA: Successfully initialized GPU " << prop.name 
    //           << " (Compute " << prop.major << "." << prop.minor << ")"
    //           << " with " << (prop.totalGlobalMem / (1024*1024*1024)) << "GB memory" << std::endl;
    
    return true;
}

void cuda_cleanup() {
    // Only cleanup if initialized - makes function idempotent (safe to call multiple times)
    if (!g_cuda_initialized) {
        return;  // Already cleaned up, safe to skip
    }
    
    // Add guard to prevent cleanup during process shutdown (when CUDA runtime might be destroyed)
    static bool cleaning_up = false;
    static bool cleanup_disabled = false;
    if (cleaning_up || cleanup_disabled) return;  // Already in cleanup or disabled
    cleaning_up = true;
    
    // Check if CUDA context is still valid before trying to cleanup
    cudaError_t context_check = cudaGetDevice(nullptr);
    if (context_check != cudaSuccess) {
        // CUDA context already destroyed - don't try to cleanup
        cleanup_disabled = true;
        g_cuda_initialized = false;
        g_cuda_available = false;
        cleaning_up = false;
        return;
    }
    
    try {
        // Clean up all global CUDA states to prevent memory corruption
        cleanup_bootstrap_states();
        // cleanup_varimp_states();  // Commented out due to linking issues
        cleanup_proximity_states();
        
        // Clean up global RNG state
        cleanup_global_rng();
        
        // JUPYTER SAFETY: Use stream sync instead of device sync
        // Stream sync is safer in Jupyter where context might be partially corrupted
        cudaStreamSynchronize(0);
        cudaGetLastError();  // Clear any errors from sync
        
        // Clear any remaining CUDA errors (if context still exists)
        cudaGetLastError();  // Ignore errors
    } catch (...) {
        // Ignore ALL errors during cleanup - process might be shutting down
        // CUDA runtime might already be destroyed
    }
    
    // Reset CUDA context state for next initialization
    g_cuda_initialized = false;
    g_cuda_available = false;
    cleaning_up = false;
    
    // Don't call cudaDeviceReset() as it's too aggressive and causes memory corruption
    // The CUDA context will be cleaned up automatically when the process exits
    // cudaDeviceReset();
}

void cuda_reset_device() {
    // Forcefully reset CUDA device to free stuck contexts
    // This is more aggressive than cuda_cleanup() and should be used when
    // CUDA devices are stuck or unavailable
    try {
        // Check if CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            // std::cout << "CUDA: No devices available for reset" << std::endl;
            return;
        }
        
        // Reset all CUDA contexts
        for (int device = 0; device < device_count; device++) {
            err = cudaSetDevice(device);
            if (err == cudaSuccess) {
                cudaDeviceReset();
                // std::cout << "CUDA: Reset device " << device << std::endl;
            }
        }
        
        // Reset global state
        g_cuda_initialized = false;
        g_cuda_available = false;
        
        // Wait a bit for device to be ready
        std::this_thread::sleep_for(std::chrono::milliseconds(200));  // Increased wait time
        
        // Reinitialize CUDA runtime after reset
        // cudaDeviceReset() destroys the context, so we need to recreate it
        err = cudaSetDevice(0);
        if (err == cudaSuccess) {
            // Force context creation with a simple operation
            void* test_ptr = nullptr;
            err = cudaMalloc(&test_ptr, 1024);
            if (err == cudaSuccess && test_ptr != nullptr) {
                // Safe free to prevent segfaults
    if (test_ptr) { cudaError_t err = cudaFree(test_ptr); if (err != cudaSuccess) cudaGetLastError(); }
                // std::cout << "CUDA: Context reinitialized after reset" << std::endl;
            } else {
                // std::cout << "CUDA: Warning - Context reinitialization test failed" << std::endl;
            }
        }
        
    } catch (...) {
        // Ignore errors during reset
        // std::cout << "CUDA: Error during device reset (ignored)" << std::endl;
    }
}


// CUDA context management helpers for Jupyter lifecycle
// These functions ensure CUDA context is valid and error-free throughout the model lifecycle

// Clear any stale CUDA errors before operations
// This prevents false positives from previous operations
void cuda_clear_errors() {
    if (!g_cuda_available) return;
    cudaGetLastError();  // Clear any pending errors
}

// Validate CUDA context is ready for operations
// Returns true if context is valid, false otherwise
bool cuda_validate_context() {
    if (!g_cuda_available) return false;
    
    // Check if device is accessible
    cudaError_t err = cudaGetDevice(nullptr);
    if (err != cudaSuccess) {
        cudaGetLastError();  // Clear error
        return false;
    }
    
    // Clear any stale errors
    cudaGetLastError();
    return true;
}

// Ensure CUDA context is ready before operations (like fit, predict, etc.)
// This is called before any CUDA operation to ensure context is valid
// XGBoost-style: Actually test if CUDA works, don't just trust flags
void cuda_ensure_context_ready() {
    if (!g_cuda_available) return;
    
    // Fork detection (Jupyter kernel restart)
    // CUDA contexts cannot survive fork - we must reinitialize
    // Use a lightweight check that won't crash: cudaGetDevice
    cudaGetLastError();  // Clear any stale errors
    
    int current_device = -1;
    cudaError_t err = cudaGetDevice(&current_device);
    
    if (err != cudaSuccess || current_device < 0) {
        // CUDA context is broken (likely after fork) - reinitialize
        cudaGetLastError();  // Clear error
        
        // Reset the global flags to force reinitialization
        g_cuda_initialized = false;
        g_cuda_available = false;
        
        // Try to reinitialize without resetting (reset can hang in bad contexts)
        try {
            // Just reinitialize - let cuda_init_runtime handle device setup
            cuda_init_runtime(false);
        } catch (...) {
            // If reinitialization fails, disable GPU
            g_cuda_available = false;
        }
        return;
    }
    
    // Device check succeeded - clear any remaining errors
    cudaGetLastError();
}

// Finalize CUDA operations after fit/predict/etc.
// This ensures all operations complete and errors are cleared
// Matches pattern from commit 6a9a52f: use stream sync (safer for Jupyter)
void cuda_finalize_operations() {
    if (!g_cuda_available) return;
    
    // For Jupyter compatibility, skip synchronization entirely
    // The GPU code already synchronizes before returning, so this is redundant
    // Skipping this prevents crashes when CUDA context is in an invalid state
    // This matches the pattern from commit 6a9a52f which avoids aggressive syncs
    // Just clear any pending errors and return
    cudaGetLastError();  // Clear any pending errors
    return;
    
    // OLD CODE (commented out to prevent crashes):
    // The GPU code already synchronizes before returning, so this sync is redundant
    // and can cause crashes if the CUDA context is in an invalid state
    /*
    try {
        cudaGetLastError();  // Clear any previous errors before sync
        cudaError_t sync_err = cudaStreamSynchronize(0);
        if (sync_err != cudaSuccess) {
            cudaGetLastError();  // Clear error
            return;
        }
        cudaGetLastError();  // Clear any errors from sync
    } catch (...) {
        // Ignore ALL errors during finalization
    }
    */
}

} // namespace cuda
} // namespace rf

namespace rf {
namespace cuda {

// Non-inline implementation for external linkage (needed for forward declarations)
int get_recommended_batch_size(int total_trees) {
    // Ensure CudaConfig is initialized before using it
    CudaConfig::instance().initialize();
    
    // Call the CudaConfig method which has SM-aware batch sizing
    return CudaConfig::instance().get_recommended_batch_size(total_trees);
}

} // namespace cuda
} // namespace rf

