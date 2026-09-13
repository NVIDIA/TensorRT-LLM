# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging as _logger
import os as _os
import pathlib as _pl
import shutil
import sys as _sys

# Declared above the cpp_common import: this covers every file in this
# directory, but only for imports that run after it, and cpp_common reaches
# build_wheel while this conftest is still executing.
__extra_import_path__ = ["~/scripts"]

import defs.cpp.cpp_common as _cpp
import pytest
from build_wheel import main as build_trt_llm


@pytest.fixture(scope="session")
def build_type():
    """CMake build type for C++ builds."""
    # For debugging purposes, we can use the RelWithDebInfo build type.
    return _os.environ.get("TLLM_BUILD_TYPE", "Release")


@pytest.fixture(scope="session")
def build_dir(build_type):
    """Resolved build directory for the current build_type."""
    return _cpp.find_build_dir(build_type)


@pytest.fixture(scope="session")
def cpp_resources_dir():
    return _pl.Path("cpp") / "tests" / "resources"


@pytest.fixture(scope="session")
def python_exe():
    return _sys.executable


@pytest.fixture(scope="session")
def root_dir():
    return _cpp.find_root_dir()


@pytest.fixture(scope="session")
def lora_setup(root_dir, cpp_resources_dir, python_exe):

    cpp_script_dir = cpp_resources_dir / "scripts"
    cpp_data_dir = cpp_resources_dir / "data"

    generate_lora_data_args_tp1 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora-test-weights-tp1",
        "--tp-size=1",
    ]

    generate_lora_data_args_tp2 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora-test-weights-tp2",
        "--tp-size=2",
    ]

    generate_multi_lora_tp2_args = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/multi_lora",
        "--tp-size=2",
        "--num-loras=128",
    ]

    generate_lora_data_args_prefetch_task_3 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora_prefetch/3",
        "--target-file-name=model.lora_weights.npy",
        "--config-file-name=model.lora_config.npy",
    ]

    generate_lora_data_args_prefetch_task_5 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora_prefetch/5",
        "--target-file-name=model.lora_weights.npy",
        "--config-file-name=model.lora_config.npy",
    ]

    _cpp.run_command(generate_lora_data_args_tp1, cwd=root_dir, timeout=100)
    _cpp.run_command(generate_lora_data_args_tp2, cwd=root_dir, timeout=100)
    _cpp.run_command(generate_multi_lora_tp2_args, cwd=root_dir, timeout=100)
    _cpp.run_command(generate_lora_data_args_prefetch_task_3,
                     cwd=root_dir,
                     timeout=100)
    _cpp.run_command(generate_lora_data_args_prefetch_task_5,
                     cwd=root_dir,
                     timeout=100)


@pytest.fixture(scope="session")
def build_google_tests(request, build_type):

    cuda_arch = f"{request.param}-real"

    _logger.info(f"Using CUDA arch: {cuda_arch}")

    build_trt_llm(
        build_type=build_type,
        cuda_architectures=cuda_arch,
        job_count=12,
        use_ccache=True,
        clean=True,
        generator="Ninja",
        nixl_root="/opt/nvidia/nvda_nixl",
        skip_building_wheel=True,
        extra_make_targets=["google-tests"],
        # The DLFW base image ships no libnvrtc_static.a, so the default
        # CUDA::nvrtc_static target does not exist. The wheel builds get
        # NVRTC_DYNAMIC_LINKING=ON implicitly (via ENABLE_BOLT_COMPATIBLE);
        # this test-time build has no BOLT, so ask for it explicitly.
        nvrtc_dynamic_linking=True,
    )


@pytest.fixture(scope="session")
def build_kv_cache_compression_tests(request, build_type):
    """Build only the standalone NVFP4 cold-page kernel gtest.

    The binary uses NO_TLLM_LINKAGE, so this skips the full-library
    build that build_google_tests pays for.
    """
    cuda_arch = f"{request.param}-real"

    _logger.info(f"Using CUDA arch: {cuda_arch}")

    build_trt_llm(
        build_type=build_type,
        cuda_architectures=cuda_arch,
        job_count=12,
        use_ccache=True,
        generator="Ninja",
        nixl_root="/opt/nvidia/nvda_nixl",
        skip_building_wheel=True,
        configure_only=True,
        # Same reason as build_google_tests: no libnvrtc_static.a in the DLFW
        # base image, and this build has no BOLT to turn dynamic linking on.
        nvrtc_dynamic_linking=True,
    )

    build_dir = _cpp.find_build_dir(build_type)
    _cpp.run_command(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            build_type,
            "--parallel",
            "12",
            "--target",
            "nvfp4ColdPageKernelsTest",
        ],
        cwd=build_dir,
        env=_os.environ,
        timeout=1800,
    )


@pytest.fixture(scope="function", autouse=True)
def keep_log_files(build_dir):
    """Backup previous cpp test results when run multiple ctest invocations."""
    results_dir = build_dir

    yield

    build_parent_dir = build_dir.parent
    backup_dir_name = build_dir.name + "_backup"
    backup_dir = build_parent_dir / backup_dir_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Copy XML files from all subdirectories to backup directory
    xml_files = list(results_dir.rglob("*.xml"))
    if xml_files:
        for xml_file in xml_files:
            try:
                shutil.copy(xml_file, backup_dir)
                _logger.info(f"Copied {xml_file} to {backup_dir}")
            except Exception as e:
                _logger.error(f"Error copying {xml_file}: {str(e)}")
    else:
        _logger.info("No XML files found in the build directory.")
