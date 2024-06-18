#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import platform
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copy, rmtree
from subprocess import CalledProcessError, check_output, run
from textwrap import dedent
from typing import List


@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_project_dir():
    return Path(__file__).parent.resolve().parent


def get_source_dir():
    return get_project_dir() / "cpp"


def get_build_dir(build_dir, build_type):
    if build_dir is None:
        build_dir = get_source_dir() / ("build" if build_type == "Release" else
                                        f"build_{build_type}")
    else:
        build_dir = Path(build_dir)
    return build_dir


def main(*,
         build_type: str = "Release",
         build_dir: Path = None,
         dist_dir: Path = None,
         cuda_architectures: str = None,
         job_count: int = None,
         extra_cmake_vars: List[str] = list(),
         extra_make_targets: str = "",
         trt_root: str = None,
         nccl_root: str = None,
         clean: bool = False,
         use_ccache: bool = False,
         fast_build: bool = False,
         cpp_only: bool = False,
         install: bool = False,
         skip_building_wheel: bool = False,
         python_bindings: bool = True,
         benchmarks: bool = False,
         micro_benchmarks: bool = False,
         nvtx: bool = False):
    project_dir = get_project_dir()
    os.chdir(project_dir)
    build_run = partial(run, shell=True, check=True)

    if not (project_dir / "3rdparty/cutlass/.git").exists():
        build_run('git submodule update --init --recursive')
    on_windows = platform.system() == "Windows"
    requirements_filename = "requirements-dev-windows.txt" if on_windows else "requirements-dev.txt"
    build_run(f"\"{sys.executable}\" -m pip install -r {requirements_filename}")
    # Ensure TRT is installed on windows to prevent surprises.
    reqs = check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]
    if "tensorrt" not in installed_packages:
        error_msg = "TensorRT was not installed properly."
        if on_windows:
            error_msg += (
                " Please download the TensorRT zip file manually,"
                " install it and relaunch build_wheel.py."
                " See https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip for more details."
            )
        else:
            error_msg += " Please run `pip install tensorrt` manually and relaunch build_wheel.py"
        raise RuntimeError(error_msg)

    cmake_cuda_architectures = (
        f'"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}"'
        if cuda_architectures is not None else "")

    cmake_def_args = []
    cmake_generator = ""

    hardware_arch = platform.machine()

    if on_windows:
        # Windows does not support multi-device currently.
        extra_cmake_vars.extend(["ENABLE_MULTI_DEVICE=0"])

        # The Ninja CMake generator is used for our Windows build
        # (Easier than MSBuild to make compatible with our Docker image)
        cmake_generator = "-GNinja"

    if job_count is None:
        job_count = cpu_count()

    if len(extra_cmake_vars):
        # Backwards compatibility, we also support semicolon expansion for each value.
        # However, it is best to use flag multiple-times due to issues with spaces in CLI.
        expanded_args = []
        for var in extra_cmake_vars:
            expanded_args += var.split(";")

        extra_cmake_vars = ["\"-D{}\"".format(var) for var in expanded_args]
        # Don't include duplicate conditions
        cmake_def_args.extend(set(extra_cmake_vars))

    if trt_root is not None:
        trt_root = trt_root.replace("\\", "/")
        trt_lib_dir_candidates = (
            f"{trt_root}/targets/{hardware_arch}-linux-gnu/lib",
            f"{trt_root}/lib")
        try:
            trt_lib_dir = next(
                filter(lambda x: Path(x).exists(), trt_lib_dir_candidates))
        except StopIteration:
            trt_lib_dir = trt_lib_dir_candidates[0]
        cmake_def_args.append(f"-DTRT_LIB_DIR={trt_lib_dir}")
        cmake_def_args.append(f"-DTRT_INCLUDE_DIR={trt_root}/include")

    if nccl_root is not None:
        cmake_def_args.append(f"-DNCCL_LIB_DIR={nccl_root}/lib")
        cmake_def_args.append(f"-DNCCL_INCLUDE_DIR={nccl_root}/include")

    build_dir = get_build_dir(build_dir, build_type)
    first_build = not build_dir.exists()

    if clean and build_dir.exists():
        rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if use_ccache:
        cmake_def_args.append(
            f"-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
        )

    if fast_build:
        cmake_def_args.append(f"-DFAST_BUILD=ON")

    build_pyt = "OFF" if cpp_only else "ON"
    th_common_lib = "" if cpp_only else "th_common"
    build_pybind = "OFF" if cpp_only else "ON"
    bindings_lib = "" if cpp_only else "bindings"
    benchmarks_lib = "benchmarks" if benchmarks else ""
    build_micro_benchmarks = "ON" if micro_benchmarks else "OFF"
    micro_benchmarks_lib = "micro_benchmarks" if micro_benchmarks else ""
    disable_nvtx = "OFF" if nvtx else "ON"
    executor_worker = "" if on_windows else "executorWorker "

    source_dir = get_source_dir()
    with working_directory(build_dir):
        cmake_def_args = " ".join(cmake_def_args)
        if clean or first_build:
            build_run(
                f'cmake -DCMAKE_BUILD_TYPE="{build_type}" -DBUILD_PYT="{build_pyt}" -DBUILD_PYBIND="{build_pybind}"'
                f' -DNVTX_DISABLE="{disable_nvtx}" -DBUILD_MICRO_BENCHMARKS={build_micro_benchmarks}'
                f' {cmake_cuda_architectures} {cmake_def_args} {cmake_generator} -S "{source_dir}"'
            )
        build_run(
            f'cmake --build . --config {build_type} --parallel {job_count} '
            f'--target tensorrt_llm nvinfer_plugin_tensorrt_llm {th_common_lib} {bindings_lib} {benchmarks_lib} '
            f'{micro_benchmarks_lib} {executor_worker} {" ".join(extra_make_targets)}'
        )

    if cpp_only:
        assert not install, "Installing is not supported for cpp_only builds"
        return

    pkg_dir = project_dir / "tensorrt_llm"
    assert pkg_dir.is_dir(), f"{pkg_dir} is not a directory"
    lib_dir = pkg_dir / "libs"
    if lib_dir.exists():
        rmtree(lib_dir)
    lib_dir.mkdir(parents=True)
    if on_windows:
        copy(build_dir / "tensorrt_llm/tensorrt_llm.dll",
             lib_dir / "tensorrt_llm.dll")
        copy(build_dir / f"tensorrt_llm/thop/th_common.dll",
             lib_dir / "th_common.dll")
        copy(
            build_dir / f"tensorrt_llm/plugins/nvinfer_plugin_tensorrt_llm.dll",
            lib_dir / "nvinfer_plugin_tensorrt_llm.dll")
        copy(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/tensorrt_llm_nvrtc_wrapper.dll",
            lib_dir / "tensorrt_llm_nvrtc_wrapper.dll")
    else:
        copy(build_dir / "tensorrt_llm/libtensorrt_llm.so",
             lib_dir / "libtensorrt_llm.so")
        copy(build_dir / "tensorrt_llm/thop/libth_common.so",
             lib_dir / "libth_common.so")
        copy(
            build_dir /
            "tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so",
            lib_dir / "libnvinfer_plugin_tensorrt_llm.so")
        copy(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/libtensorrt_llm_nvrtc_wrapper.so",
            lib_dir / "libtensorrt_llm_nvrtc_wrapper.so")

    bin_dir = pkg_dir / "bin"
    if bin_dir.exists():
        rmtree(bin_dir)
    bin_dir.mkdir(parents=True)

    if not on_windows:
        copy(build_dir / "tensorrt_llm/executor_worker/executorWorker",
             bin_dir / "executorWorker")

    if not cpp_only:

        def get_pybind_lib():
            pybind_build_dir = (build_dir / "tensorrt_llm" / "pybind")
            if on_windows:
                pybind_lib = list(pybind_build_dir.glob("bindings.*.pyd"))
            else:
                pybind_lib = list(pybind_build_dir.glob("bindings.*.so"))

            assert len(
                pybind_lib
            ) == 1, f"Exactly one pybind library should be present: {pybind_lib}"
            return pybind_lib[0]

        copy(get_pybind_lib(), pkg_dir)

        with working_directory(project_dir):
            build_run(f"\"{sys.executable}\" -m pip install pybind11-stubgen")
        with working_directory(pkg_dir):
            if on_windows:
                stubgen = "stubgen.py"
                stubgen_contents = """
                # Loading torch, trt before bindings is required to avoid import errors on windows.
                # isort: off
                import torch
                import tensorrt as trt
                # isort: on
                import os
                import platform

                from pybind11_stubgen import main

                if __name__ == "__main__":
                    # Load dlls from `libs` directory before launching bindings.
                    if platform.system() == "Windows":
                        os.add_dll_directory(r\"{lib_dir}\")
                    main()
                """.format(lib_dir=lib_dir)
                (pkg_dir / stubgen).write_text(dedent(stubgen_contents))
                build_run(f"\"{sys.executable}\" {stubgen} -o . bindings")
                (pkg_dir / stubgen).unlink()
            else:
                env_ld = os.environ.copy()

                new_library_path = "/usr/local/cuda/compat/lib.real"
                if 'LD_LIBRARY_PATH' in env_ld:
                    new_library_path += f":{env_ld['LD_LIBRARY_PATH']}"
                env_ld["LD_LIBRARY_PATH"] = new_library_path
                try:
                    build_run(
                        f"\"{sys.executable}\" -m pybind11_stubgen -o . bindings",
                        env=env_ld)
                except CalledProcessError as ex:
                    print(f"Failed to build pybind11 stubgen: {ex}",
                          file=sys.stderr)

    if dist_dir is None:
        dist_dir = project_dir / "build"
    else:
        dist_dir = Path(dist_dir)

    if not dist_dir.exists():
        dist_dir.mkdir(parents=True)
    if not skip_building_wheel:
        build_run(
            f'\"{sys.executable}\" -m build {project_dir} --skip-dependency-check --no-isolation --wheel --outdir "{dist_dir}"'
        )

    if install:
        build_run(f"\"{sys.executable}\" -m pip install -e .[devel]")


def add_arguments(parser: ArgumentParser):
    parser.add_argument("--build_type",
                        "-b",
                        default="Release",
                        choices=["Release", "RelWithDebInfo", "Debug"])
    parser.add_argument("--cuda_architectures", "-a")
    parser.add_argument("--install", "-i", action="store_true")
    parser.add_argument("--clean", "-c", action="store_true")
    parser.add_argument("--use_ccache",
                        "-ccache",
                        default=False,
                        action="store_true",
                        help="Use ccache compiler driver")
    parser.add_argument(
        "--fast_build",
        "-f",
        default=False,
        action="store_true",
        help=
        "Skip compiling some kernels to accelerate compilation -- for development only"
    )
    parser.add_argument("--job_count",
                        "-j",
                        const=cpu_count(),
                        nargs="?",
                        help="Parallel job count")
    parser.add_argument(
        "--cpp_only",
        "-l",
        action="store_true",
        help="Only build the C++ library without Python dependencies")
    parser.add_argument(
        "--extra-cmake-vars",
        "-D",
        action="append",
        help=
        "Extra cmake variable definition which can be specified multiple times, example: -D \"key1=value1\" -D \"key2=value2\"",
        default=[])
    parser.add_argument(
        "--extra-make-targets",
        help="A list of additional make targets, example: \"target_1 target_2\"",
        nargs="+",
        default=[])
    parser.add_argument("--trt_root",
                        help="Directory to find TensorRT headers/libs")
    parser.add_argument("--nccl_root",
                        help="Directory to find NCCL headers/libs")
    parser.add_argument("--build_dir",
                        type=Path,
                        help="Directory where cpp sources are built")
    parser.add_argument("--dist_dir",
                        type=Path,
                        help="Directory where python wheels are built")
    parser.add_argument(
        "--skip_building_wheel",
        "-s",
        action="store_true",
        help=
        "Do not build the *.whl files (they are only needed for distribution).")
    parser.add_argument(
        "--python_bindings",
        "-p",
        action="store_true",
        help="(deprecated) Build the python bindings for the C++ runtime.")
    parser.add_argument("--benchmarks",
                        action="store_true",
                        help="Build the benchmarks for the C++ runtime.")
    parser.add_argument("--micro_benchmarks",
                        action="store_true",
                        help="Build the micro benchmarks for C++ components.")
    parser.add_argument("--nvtx",
                        action="store_true",
                        help="Enable NVTX features.")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(**vars(args))
