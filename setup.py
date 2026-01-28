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
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DCMAKE_CUDA_SEPARABLE_COMPILATION=ON',  # Fix CUDA device code linking
        ]

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
        # Use all available cores for faster builds
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        build_args += ['--', f'-j{num_cores}']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Only run cmake configure if CMakeCache.txt doesn't exist or source changed
        cmake_cache = os.path.join(self.build_temp, 'CMakeCache.txt')
        if not os.path.exists(cmake_cache):
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        else:
            # Just reconfigure with new args if cache exists (faster)
            subprocess.check_call(['cmake', '.'] + cmake_args, cwd=self.build_temp, env=env)

        # Build with incremental compilation (cmake handles caching automatically)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Read README for long description
# Use README_PYPI.md for PyPI (simplified), fallback to README.md
long_description = ''
if os.path.exists('README_PYPI.md'):
    with open('README_PYPI.md', encoding='utf-8') as f:
        long_description = f.read()
elif os.path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='rfx-fuse',
    version='1.0.0',
    author='Chris Kuchar',
    author_email='chrisjkuchar@gmail.com',
    description="RFX-Fuse: Breiman and Cutler's Unified ML Engine with GPU Acceleration",
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
    install_requires=[
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
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
    package_dir={'': 'python'},
    include_package_data=True,
    package_data={
        '': [
            'examples/**/*.py',
            'examples/**/*.md',
        ],
    },
)
