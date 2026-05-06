#!/bin/bash
# Build manylinux GPU wheels for rfx-fuse (CUDA 12.4)
#
# Run from project root via Docker:
#   docker run --rm --gpus all -v $(pwd):/io nvidia/cuda:12.4.1-devel-rockylinux8 /io/scripts/build_wheels_gpu.sh
set -e

cd /io

# Install build tools in the CUDA container
yum install -y gcc gcc-c++ make wget openblas-devel 2>/dev/null || true

# Install CMake (manylinux version may not be present in CUDA image)
if ! command -v cmake &>/dev/null; then
    pip3 install cmake
fi

# Install Python versions via pyenv or use available system python
# The CUDA devel image may only have one Python; build for that version
# For full matrix, use manylinux-based CUDA images or install Pythons manually

# Clean previous builds
rm -rf /io/wheelhouse_gpu /io/dist_gpu
mkdir -p /io/wheelhouse_gpu /io/dist_gpu

# Try each Python version available
for PYBIN in /opt/python/cp3{9,10,11,12,13}*/bin /usr/bin; do
    [ -d "$PYBIN" ] || continue
    PYTHON="${PYBIN}/python3"
    [ -x "$PYTHON" ] || PYTHON="${PYBIN}/python"
    [ -x "$PYTHON" ] || continue

    PYVER=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
    echo "=== Building GPU wheel for Python ${PYVER} ==="

    "${PYTHON}" -m pip install --upgrade pip setuptools wheel pybind11 numpy 2>/dev/null || \
        "${PYBIN}/pip" install --upgrade pip setuptools wheel pybind11 numpy

    RFX_CUDA_STATIC=1 "${PYTHON}" -m pip wheel /io --no-deps -w /io/wheelhouse_gpu/
done

# Repair wheels (bundle libcudart_static etc.)
for whl in /io/wheelhouse_gpu/*.whl; do
    auditwheel repair "$whl" -w /io/dist_gpu/ || cp "$whl" /io/dist_gpu/
done

echo ""
echo "=== GPU wheels built ==="
ls -lh /io/dist_gpu/
