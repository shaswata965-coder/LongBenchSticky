"""
build.py — Build script for the Sticky KV CUDA extension.

Usage (on HPC node with CUDA):
    cd csrc
    python build.py install

Or with pip:
    cd csrc
    pip install -e .

Modeled after DefensiveKV's build.py.
Target: NVIDIA A6000 (Ampere, sm_86)
"""

import subprocess
import os
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# Package name for pip (can be removed via `pip uninstall sticky_kv_cuda_pkg`)
PACKAGE_NAME = "sticky_kv_cuda_pkg"

ext_modules = []
generator_flag = []
cc_flag = []

# A6000 is Ampere sm_86; also include sm_80 for A100 portability
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_86,code=sm_86")

# Uncomment for Hopper (H100, sm_90) if needed:
# cc_flag.append("-gencode")
# cc_flag.append("arch=compute_90,code=sm_90")

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# CUDA extension module
ext_modules.append(
    CUDAExtension(
        # Package name for import: `import sticky_kv_cuda`
        name="sticky_kv_cuda",
        sources=[
            "csrc/sticky_kv_kernels.cu",
        ],
        extra_compile_args={
            # C++ compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # NVCC compile flags
            "nvcc": [
                "-O3",
                "-std=c++17",
                # Enable half/bfloat16 operators
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                # Performance flags
                "--use_fast_math",
                "--ptxas-options=-v",
                "--ptxas-options=-O2",
                # Needed for lambdas in CUDA
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                # Debug info (useful for cuda-memcheck, can remove for release)
                "-lineinfo",
            ]
            + generator_flag
            + cc_flag,
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "include",
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "*.egg-info",
        )
    ),
    description="CUDA kernels for Sticky KV Cache acceleration.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
    ],
)
