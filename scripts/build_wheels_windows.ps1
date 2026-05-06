# Build Windows wheels for rfx-fuse and rfx-fuse-cpu
#
# Prerequisites: Visual Studio 2019/2022, CMake, CUDA 12.4 toolkit
# Run from project root: .\scripts\build_wheels_windows.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$PythonVersions = @("3.9", "3.10", "3.11", "3.12", "3.13")

New-Item -ItemType Directory -Force -Path "dist_win" | Out-Null
New-Item -ItemType Directory -Force -Path "wheelhouse_win" | Out-Null

# Install delvewheel for bundling DLL dependencies
& py -3 -m pip install --upgrade delvewheel

foreach ($PyVer in $PythonVersions) {
    try {
        & py -$PyVer -c "import sys; print(sys.version)" 2>$null
    } catch {
        Write-Host "Python $PyVer not found, skipping"
        continue
    }

    & py -$PyVer -m pip install --upgrade pip setuptools wheel pybind11 numpy

    Write-Host "=== Building GPU wheel for Python $PyVer ==="
    $env:RFX_CPU_ONLY = "0"
    $env:RFX_CUDA_STATIC = "1"
    & py -$PyVer -m pip wheel . --no-deps -w wheelhouse_win/

    Write-Host "=== Building CPU wheel for Python $PyVer ==="
    $env:RFX_CPU_ONLY = "1"
    $env:RFX_CUDA_STATIC = "0"
    & py -$PyVer -m pip wheel . --no-deps -w wheelhouse_win/
}

Remove-Item Env:\RFX_CPU_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:\RFX_CUDA_STATIC -ErrorAction SilentlyContinue

# Repair wheels with delvewheel (bundles DLL dependencies like libomp)
Write-Host "=== Repairing wheels with delvewheel ==="
foreach ($whl in Get-ChildItem wheelhouse_win/*.whl) {
    try {
        & py -3 -m delvewheel repair $whl.FullName -w dist_win/
    } catch {
        Write-Host "delvewheel repair failed for $($whl.Name), copying as-is"
        Copy-Item $whl.FullName dist_win/
    }
}

Write-Host ""
Write-Host "=== Windows wheels built ==="
Get-ChildItem dist_win/*.whl | Format-Table Name, Length
