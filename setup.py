#!/usr/bin/env python3
"""
setup.py for RFX-Fuse - Breiman and Cutler's Unified ML & Similarity Engine

Original Random Forest algorithm by Leo Breiman and Adele Cutler
C++ implementation, GPU acceleration, and Python bindings by Chris Kuchar
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DPython_EXECUTABLE={sys.executable}',
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DCMAKE_CUDA_SEPARABLE_COMPILATION=ON',
            '-DRFX_PORTABLE=ON',
        ]

        # Environment-driven build flags (for wheel builds)
        # Always pass both flags explicitly to avoid CMake cache stale values
        cpu_only = os.environ.get('RFX_CPU_ONLY', '0') == '1'
        cuda_static = os.environ.get('RFX_CUDA_STATIC', '0') == '1'
        cmake_args.append(f'-DRFX_CPU_ONLY={"ON" if cpu_only else "OFF"}')
        cmake_args.append(f'-DRFX_CUDA_STATIC={"ON" if cuda_static else "OFF"}')

        if not cpu_only:
            # For wheels: target broad GPU range (Ampere 8.0+, Ada 8.9, Hopper 9.0)
            # Also include older Pascal 6.0, Volta 7.0, Turing 7.5 for compatibility
            cuda_archs = os.environ.get('CMAKE_CUDA_ARCHITECTURES', '60;70;75;80;86;89;90')
            cmake_args.append(f'-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}')

        # Add pybind11 path if available
        try:
            import pybind11
            pybind11_path = pybind11.get_cmake_dir()
            cmake_args.append(f'-DCMAKE_PREFIX_PATH={pybind11_path}')
            print(f"Using pybind11 from: {pybind11_path}")
        except ImportError:
            print("Warning: pybind11 not found, CMake will try to find it")

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        if sys.platform == 'win32':
            # Find an installed Visual Studio version via vswhere
            vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
            vs_found = False
            if os.path.isfile(vswhere):
                try:
                    vs_version = subprocess.check_output(
                        [vswhere, '-latest', '-property', 'catalog_productLineVersion'],
                        text=True
                    ).strip()
                    vs_generators = {'2022': 'Visual Studio 17 2022', '2019': 'Visual Studio 16 2019'}
                    if vs_version in vs_generators:
                        cmake_args += ['-G', vs_generators[vs_version], '-A', 'x64']
                        vs_found = True
                except Exception:
                    pass
            if not vs_found:
                cmake_args += ['-G', 'Ninja']
            build_args += ['--', f'/m:{num_cores}']
        else:
            build_args += ['--', f'-j{num_cores}']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        # Ensure CUDA bin is on PATH (nvcc may not be on default PATH)
        if not cpu_only:
            cuda_paths = ['/usr/local/cuda/bin', '/usr/local/cuda-12/bin']
            for cp in cuda_paths:
                if os.path.isdir(cp) and cp not in env.get('PATH', ''):
                    env['PATH'] = cp + os.pathsep + env.get('PATH', '')

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Always do a fresh configure to prevent stale CMake cache
        # (e.g., switching between CPU-only and GPU builds)
        cmake_cache = os.path.join(self.build_temp, 'CMakeCache.txt')
        if os.path.exists(cmake_cache):
            os.remove(cmake_cache)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)

        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Determine package variant from environment
is_cpu_only = os.environ.get('RFX_CPU_ONLY', '0') == '1'
pkg_name = 'rfx-fuse-cpu' if is_cpu_only else 'rfx-fuse'
pkg_description = (
    "RFX-Fuse: Breiman and Cutler's Unified ML Engine (CPU-only)" if is_cpu_only
    else "RFX-Fuse: Breiman and Cutler's Unified ML Engine with GPU Acceleration"
)

# Read README for long description
readme_file = 'README_PYPI_CPU.md' if is_cpu_only else 'README_PYPI.md'
long_description = ''
if os.path.exists(readme_file):
    with open(readme_file, encoding='utf-8') as f:
        long_description = f.read()
elif os.path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name=pkg_name,
    version='1.1.0',
    author='Chris Kuchar',
    author_email='chrisjkuchar@gmail.com',
    description=pkg_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chriskuchar/RFX-Fuse',
    project_urls={
        'Bug Reports': 'https://github.com/chriskuchar/RFX-Fuse/issues',
        'Source': 'https://github.com/chriskuchar/RFX-Fuse',
        'Documentation': 'https://github.com/chriskuchar/RFX-Fuse/blob/main/docs/API.md',
    },
    ext_modules=[CMakeExtension('RFXFuse', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.9',
    setup_requires=[
        'pybind11>=2.6.0',
    ],
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='random forest, machine learning, gpu, cuda, classification, visualization, proximity',
    packages=find_packages(where='python', include=['*']),
    py_modules=['rfx_impute', 'rfx_fuse_impute', 'rfviz', 'categorical_helper'],
    package_dir={'': 'python'},
    include_package_data=True,
    package_data={
        '': [
            'examples/**/*.py',
            'examples/**/*.md',
        ],
    },
)
