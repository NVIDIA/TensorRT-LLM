#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script installs visual_gen.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

compile_svd = False
compile_fused_qk_norm_rope = False
for i in range(torch.cuda.device_count()):
    capability = torch.cuda.get_device_capability(i)
    sm = f"{capability[0]}{capability[1]}"
    if sm == "120":
        compile_svd = True
        print(f"svdquant only supports SM 120, will compile SVD for SM {sm}")
    if sm in ["80", "86", "89", "90", "100", "103", "120"]:
        compile_fused_qk_norm_rope = True

class CustomInstallCommand(install):
    def run(self):
        # First install visual_gen
        install.run(self)

class CustomDevelopCommand(develop):
    def run(self):
        # First install visual_gen in development mode
        develop.run(self)

# Read version from visual_gen/__version__.py
def get_version():
    version_file = Path(__file__).parent / "visual_gen" / "__version__.py"
    if version_file.exists():
        with open(version_file, "r") as f:
            exec(f.read())
            return locals()["__version__"]
    return "0.2.0"


# Read README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            if "cxx" not in ext.extra_compile_args:
                ext.extra_compile_args["cxx"] = []
            if "nvcc" not in ext.extra_compile_args:
                ext.extra_compile_args["nvcc"] = []
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args["cxx"] += ext.extra_compile_args["msvc"]
                ext.extra_compile_args["nvcc"] += ext.extra_compile_args["nvcc_msvc"]
            else:
                if "gcc" in ext.extra_compile_args:
                    ext.extra_compile_args["cxx"] += ext.extra_compile_args["gcc"]
        super().build_extensions()


def download_cutlass():
    """Download cutlass."""
    ROOT_DIR = os.path.dirname(__file__)
    cutlass_path = os.path.join(ROOT_DIR, "3rdparty", "cutlass")
    if not os.path.exists(cutlass_path):
        os.makedirs(cutlass_path)
    subprocess.check_call(["git", "clone", "https://github.com/NVIDIA/cutlass.git", cutlass_path])
    subprocess.check_call(["git", "checkout", "v4.3.0.dev0"], cwd=cutlass_path)

def svd_extension():
    """SVD extension. Cutlass is required."""

    ROOT_DIR = os.path.dirname(__file__)
    download_cutlass()

    INCLUDE_DIRS = [
        "visual_gen/csrc/svd_kernels/",
        "visual_gen/csrc/torch_op_bindings/",
        "3rdparty/cutlass/include",
    ]

    INCLUDE_DIRS = [os.path.join(ROOT_DIR, dir) for dir in INCLUDE_DIRS]

    DEBUG = False

    def cond(s) -> list:
        if DEBUG:
            return [s]
        else:
            return []

    sm_targets = ["120a"]

    GCC_FLAGS = ["-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-fvisibility=hidden", "-g", "-std=c++20", "-UNDEBUG", "-Og"]
    MSVC_FLAGS = ["/DENABLE_BF16=1", "/DBUILD_NUNCHAKU=1", "/std:c++20", "/UNDEBUG", "/Zc:__cplusplus", "/FS"]
    NVCC_FLAGS = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-g",
        "-std=c++20",
        "-UNDEBUG",
        "-Xcudafe",
        "--diag_suppress=20208",  # spdlog: 'long double' is treated as 'double' in device code
        *cond("-G"),
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        f"--threads={len(sm_targets)}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=--allow-expensive-optimizations=true",
    ]

    if DEBUG:
        NVCC_FLAGS.append("--generate-line-info")

    for target in sm_targets:
        NVCC_FLAGS += ["-gencode", f"arch=compute_{target},code=sm_{target}"]

    NVCC_MSVC_FLAGS = ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/FS", "-Xcompiler", "/bigobj"]

    svd_extension = CUDAExtension(
        name="visual_gen.csrc._C",
        sources=[
            "visual_gen/csrc/torch_op_bindings/pybind.cpp",
            "visual_gen/csrc/svd_kernels/torch.cpp",
            "visual_gen/csrc/svd_kernels/gemm_w4a4.cu",
            "visual_gen/csrc/svd_kernels/gemm_w4a4_launch_fp16_int4.cu",
            "visual_gen/csrc/svd_kernels/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
            "visual_gen/csrc/svd_kernels/gemm_w4a4_launch_fp16_fp4.cu",
            "visual_gen/csrc/svd_kernels/gemm_w4a4_launch_bf16_int4.cu",
            "visual_gen/csrc/svd_kernels/gemm_w4a4_launch_bf16_fp4.cu",
        ],
        extra_compile_args={"gcc": GCC_FLAGS, "msvc": MSVC_FLAGS, "nvcc": NVCC_FLAGS, "nvcc_msvc": NVCC_MSVC_FLAGS},
        include_dirs=INCLUDE_DIRS,
    )

    return svd_extension

def load_requirements(fname="requirements.txt"):
    with open(fname, encoding="utf-8") as f:
        lines = []
        for line in f.read().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines

def fused_qk_norm_rope_extension():
    """Fused QK Norm Rope extension."""

    os.path.dirname(__file__)

    fused_qk_norm_rope_extension = CUDAExtension(
        name="visual_gen.csrc.fused_qk_norm_rope",
        sources=["visual_gen/csrc/DiTRMSNormRope/fused_qk_norm_rope_kernel.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_90,code=sm_90",
                "-gencode=arch=compute_100,code=sm_100",
                "-gencode=arch=compute_103,code=sm_103",
                "-gencode=arch=compute_120,code=sm_120",
            ],
        },
    )
    return fused_qk_norm_rope_extension

ext_modules = []
if compile_svd:
    ext_modules.append(svd_extension())
if compile_fused_qk_norm_rope:
    ext_modules.append(fused_qk_norm_rope_extension())
requirements = load_requirements()
setup(
    name="visual_gen",
    version=get_version(),
    description="visual_gen",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["visual_gen*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "build_ext": CustomBuildExtension,
    },
    zip_safe=False,
)
