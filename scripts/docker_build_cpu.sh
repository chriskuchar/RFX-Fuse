#!/bin/bash
set -e

yum install -y numactl-devel lapack-devel blas-devel 2>&1 | tail -3

cd /io
rm -rf dist/ build/ .eggs/
rm -f python/*.so python/*.egg-info -rf

rm -rf /io/dist_cpu
mkdir -p /io/dist_cpu

rm -rf /io/dist_all
mkdir -p /io/dist_all
for PYVER in cp311-cp311 cp312-cp312 cp313-cp313; do
    PYBIN=/opt/python/$PYVER/bin
    echo "=== Building with $PYBIN ==="
    $PYBIN/pip install setuptools wheel pybind11 numpy
    rm -rf dist/ build/ python/*.so python/rfx_fuse_cpu.egg-info
    RFX_CPU_ONLY=1 $PYBIN/python setup.py bdist_wheel
    cp dist/*.whl /io/dist_all/
    rm -rf build/temp.* build/lib.*
done

echo "=== Repairing wheels ==="
for whl in /io/dist_all/*.whl; do
    auditwheel repair "$whl" -w /io/dist_cpu/
done

echo "=== Done ==="
ls -la /io/dist_cpu/
