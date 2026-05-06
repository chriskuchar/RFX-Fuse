#!/bin/bash
# Build manylinux CPU-only wheels for rfx-fuse-cpu
#
# Run from project root via Docker:
#   docker run --rm -v $(pwd):/io quay.io/pypa/manylinux_2_28_x86_64 /io/scripts/build_wheels_cpu.sh
set -e

cd /io

# Install build dependencies available in manylinux
yum install -y cmake3 openblas-devel 2>/dev/null || true
ln -sf /usr/bin/cmake3 /usr/local/bin/cmake 2>/dev/null || true

# Clean previous builds
rm -rf /io/wheelhouse_cpu /io/dist_cpu
mkdir -p /io/wheelhouse_cpu /io/dist_cpu

for PYBIN in /opt/python/cp3{9,10,11,12,13}*/bin; do
    [ -d "$PYBIN" ] || continue
    PYVER=$("${PYBIN}/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "=== Building for Python ${PYVER} ==="

    "${PYBIN}/pip" install --upgrade pip setuptools wheel pybind11 numpy
    RFX_CPU_ONLY=1 "${PYBIN}/pip" wheel /io --no-deps -w /io/wheelhouse_cpu/
done

# Repair wheels to be manylinux compliant (bundles any linked .so)
for whl in /io/wheelhouse_cpu/*.whl; do
    auditwheel repair "$whl" -w /io/dist_cpu/
done

echo ""
echo "=== CPU wheels built ==="
ls -lh /io/dist_cpu/
