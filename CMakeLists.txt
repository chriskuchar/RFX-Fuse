cmake_minimum_required(VERSION 3.18)

# CPU-only build option
option(RFX_CPU_ONLY "Build without CUDA support" OFF)
# Static CUDA runtime linking (for pre-built wheel distribution)
option(RFX_CUDA_STATIC "Statically link CUDA runtime into the wheel" OFF)

# Must declare project with at least CXX before find_package calls
project(RFXFuse LANGUAGES CXX)

# Conditionally enable CUDA language
if(RFX_CPU_ONLY)
    message(STATUS "Building CPU-only version (GPU disabled)")
    set(CUDA_AVAILABLE OFF)
else()
    # Try to find CUDA
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CUDA_AVAILABLE ON)
        add_definitions(-DCUDA_FOUND)
        if(RFX_CUDA_STATIC)
            set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
            message(STATUS "Building with CUDA support (static runtime)")
        else()
            message(STATUS "Building with CUDA support")
        endif()
    else()
        set(CUDA_AVAILABLE OFF)
        message(WARNING "CUDA not found - building CPU-only version")
    endif()
endif()

# Set CMake policies to suppress warnings
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

# Fix libstdc++ version issues with conda/anaconda
# Prefer system libstdc++ over conda's older version
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# Set RPATH to prefer system libraries (fixes GLIBCXX_3.4.32 not found error)
set(CMAKE_BUILD_RPATH "/usr/lib/x86_64-linux-gnu;/usr/lib;/lib/x86_64-linux-gnu")
set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu;/usr/lib;/lib/x86_64-linux-gnu")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Only set CUDA standard if CUDA is available
if(CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Find required packages
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
# Try to find NUMA library
find_library(NUMA_LIB NAMES numa libnuma.so.2 libnuma.so)
# Try to find LAPACK (optional - we have fallback below)
find_package(LAPACK)

# Portable wheel builds must not use -march=native (causes SIGILL on other CPUs)
option(RFX_PORTABLE "Build portable binaries for wheel distribution" OFF)

# Compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /DNDEBUG")
else()
    if(RFX_PORTABLE OR RFX_CPU_ONLY)
        # -mavx2: required because rf_vectorized_ops.cpp uses AVX intrinsics
        # (runtime dispatch falls back to scalar on non-AVX CPUs)
        # AVX2 has been on all x86-64 CPUs since Haswell (2013)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -DNDEBUG")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")
    endif()
endif()

if(CUDA_AVAILABLE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O3 --use_fast_math -diag-suppress=177,20014,550")
    # Set CUDA architectures (CUDA 12.8 supported architectures)
    # Supported: 75 (Turing), 80 (Ampere), 86 (Ampere), 87, 89 (Ada), 90 (Hopper)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

if(CUDA_AVAILABLE)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/rf_config.cpp
    src/rf_arrays.cpp
    src/rf_utils.cpp
    src/rf_utilities.cpp
    src/rf_random_forest.cpp
    src/rf_cuda_config.cpp
    src/rf_parallel_tree_growing.cpp
    src/rf_vectorized_ops.cpp
    src/rf_memory_pool.cpp
    src/rf_growtree_wrapper.cpp
    src/rf_growtree.cpp
    src/rf_bootstrap.cpp
    src/rf_getamat.cpp
    src/rf_testreebag.cpp
    src/rf_varimp.cpp
    src/rf_proximity.cpp
    src/rf_finishprox.cpp
    src/rf_predict.cpp
    src/rf_mds_cpu.cpp
    src/rf_histogram.cpp
    src/rf_impute.cpp
    python/randomforest_py.cpp
)

# GPU-only source files (excluded from CPU-only builds)
if(CUDA_AVAILABLE)
    list(APPEND SOURCES src/rf_proximity_optimized.cpp)
endif()

# CUDA source files
if(CUDA_AVAILABLE)
    set(CUDA_SOURCES
        cuda/rf_config_cuda.cu
        cuda/rf_memory.cu
        cuda/rf_cuda_memory.cu
        cuda/rf_bootstrap.cu
        cuda/rf_varimp.cu
        cuda/rf_proximity.cu
        cuda/rf_testreebag.cu
        cuda/rf_finishprox.cu
        cuda/rf_getamat.cu
        cuda/rf_predict.cu
        cuda/rf_growtree.cu
        cuda/rf_quantization_kernels.cu
        cuda/rf_proximity_lowrank.cu
        cuda/rf_proximity_upper_triangle.cu
        cuda/rf_lowrank_helpers.cu
        cuda/rf_mds_gpu.cu
        # GPU sparse implementation files
        cuda/rf_sparse_cuda.cu
        cuda/rf_testreebag_sparse.cu
        cuda/rf_varimp_sparse.cu
        cuda/rf_proximity_importance_sparse.cu
        cuda/rf_proximity_importance.cu
        cuda/rf_growtree_sparse.cu
        cuda/rf_growtree_sparse_parallel.cu
        cuda/rf_sparse_forest.cu
        cuda/rf_histogram.cu
        cuda/rf_histogram_gpu.cu
        cuda/rf_sparse_histogram.cu
        cuda/rf_impute.cu
        cuda/rf_unsupervised_synthetic.cu
    )
else()
    set(CUDA_SOURCES "")
endif()

# Create Python module with pybind11
pybind11_add_module(RFXFuse ${SOURCES} ${CUDA_SOURCES})

# Link CUDA libraries and fix libstdc++ linking
target_link_libraries(RFXFuse PRIVATE
    pybind11::module
    OpenMP::OpenMP_CXX
)

# Link CUDA libraries only if available
if(CUDA_AVAILABLE)
    target_link_libraries(RFXFuse PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cusolver
        CUDA::cublas
    )
endif()

# Link NUMA if found
if(NUMA_LIB)
    target_link_libraries(RFXFuse PRIVATE ${NUMA_LIB})
    message(STATUS "Found NUMA: ${NUMA_LIB}")
else()
    message(WARNING "NUMA library not found - may cause linking issues")
endif()

# Link LAPACK and BLAS explicitly (for CPU MDS computation)
# Use find_library to locate LAPACK and BLAS
# Try multiple library name variations and paths
find_library(LAPACK_LIB 
    NAMES lapack liblapack.so.3 liblapack.so libopenblas openblas
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /lib/x86_64-linux-gnu /lib
    NO_DEFAULT_PATH
)
find_library(BLAS_LIB 
    NAMES blas libblas.so.3 libblas.so libopenblas openblas
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /lib/x86_64-linux-gnu /lib
    NO_DEFAULT_PATH
)

# If not found with NO_DEFAULT_PATH, try without it (searches more paths)
if(NOT LAPACK_LIB)
    find_library(LAPACK_LIB NAMES lapack liblapack.so.3 liblapack.so)
endif()
if(NOT BLAS_LIB)
    find_library(BLAS_LIB NAMES blas libblas.so.3 libblas.so)
endif()

if(LAPACK_LIB AND BLAS_LIB)
    target_link_libraries(RFXFuse PRIVATE ${LAPACK_LIB} ${BLAS_LIB})
    message(STATUS "Found LAPACK: ${LAPACK_LIB}")
    message(STATUS "Found BLAS: ${BLAS_LIB}")
else()
    message(WARNING "LAPACK or BLAS not found - CPU MDS may not work")
    if(NOT LAPACK_LIB)
        message(WARNING "  LAPACK_LIB not found")
    endif()
    if(NOT BLAS_LIB)
        message(WARNING "  BLAS_LIB not found")
    endif()
endif()

# Fix libstdc++ linking to use conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    target_link_directories(RFXFuse PRIVATE ${CONDA_PREFIX}/lib)
    target_link_libraries(RFXFuse PRIVATE -static-libgcc -static-libstdc++)
endif()

# Set library properties
if(CUDA_AVAILABLE)
    set_target_properties(RFXFuse PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared
    )
    
    # Add CUDA device runtime library if available
    find_library(CUDA_DEVICE_RUNTIME_LIBRARY
        NAMES cudadevrt
        PATHS ${CUDAToolkit_LIBRARY_DIR}
        NO_DEFAULT_PATH
    )
    
    if(CUDA_DEVICE_RUNTIME_LIBRARY)
        target_link_libraries(RFXFuse PRIVATE ${CUDA_DEVICE_RUNTIME_LIBRARY})
    endif()
    
    # Force device code linking
    set_property(TARGET RFXFuse PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_property(TARGET RFXFuse PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    
    # Add explicit device linking for CUDA files only
    set_property(TARGET RFXFuse PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    set_property(TARGET RFXFuse PROPERTY CUDA_ARCHITECTURES 75 80 86 87 89 90)
else()
    # CPU-only properties
    set_target_properties(RFXFuse PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Install targets
install(TARGETS RFXFuse
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/randomforest)
install(DIRECTORY cuda/ DESTINATION include/randomforest/cuda
    FILES_MATCHING PATTERN "*.cuh"
)

# Post-build: Copy .so file to python/ directory for development
# This allows Python to load the module without manual copying
add_custom_command(TARGET RFXFuse POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/python
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:RFXFuse>
        ${CMAKE_SOURCE_DIR}/python/
    COMMENT "Copying RFXFuse module to python/ directory"
)

