#!/bin/bash
# Upload all built wheels to PyPI
#
# Prerequisites: pip install twine
# Set TWINE_USERNAME and TWINE_PASSWORD or use ~/.pypirc
set -e

echo "=== Uploading wheels to PyPI ==="

# Upload GPU wheels (Linux + Windows)
if ls dist_gpu/*.whl 1>/dev/null 2>&1; then
    echo "Uploading GPU Linux wheels..."
    twine upload dist_gpu/*.whl
fi

if ls dist_win/rfx_fuse-*.whl 1>/dev/null 2>&1; then
    echo "Uploading GPU Windows wheels..."
    twine upload dist_win/rfx_fuse-*.whl
fi

# Upload CPU wheels (Linux + Windows)
if ls dist_cpu/*.whl 1>/dev/null 2>&1; then
    echo "Uploading CPU Linux wheels..."
    twine upload dist_cpu/*.whl
fi

if ls dist_win/rfx_fuse_cpu-*.whl 1>/dev/null 2>&1; then
    echo "Uploading CPU Windows wheels..."
    twine upload dist_win/rfx_fuse_cpu-*.whl
fi

# Upload source distribution as fallback
if ls dist/*.tar.gz 1>/dev/null 2>&1; then
    echo "Uploading source distribution..."
    twine upload dist/*.tar.gz
fi

echo "=== Upload complete ==="
