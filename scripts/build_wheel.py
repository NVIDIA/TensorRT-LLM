#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copy, rmtree
from subprocess import run


@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def main(build_type: str = "Release",
         build_dir: Path = None,
         dist_dir: Path = None,
         cuda_architectures: str = None,
         job_count: int = None,
         extra_cmake_vars: str = "",
         extra_make_targets: str = "",
         trt_root: str = None,
         nccl_root: str = None,
         clean: bool = False,
         use_ccache: bool = False,
         cpp_only: bool = False,
         install: bool = False,
         skip_building_wheel: bool = False):
    project_dir = Path(__file__).parent.resolve().parent
    os.chdir(project_dir)
    build_run = partial(run, shell=True, check=True)

    if not (project_dir / "3rdparty/cutlass/.git").exists():
        build_run('git submodule update --init --recursive')

    build_run(
        'pip install -r requirements.txt --extra-index-url https://pypi.ngc.nvidia.com'
    )

    cmake_cuda_architectures = (
        f'-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}'
        if cuda_architectures is not None else "")

    cmake_def_args = []

    if job_count is None:
        job_count = cpu_count()

    if len(extra_cmake_vars):
        extra_cmake_vars = extra_cmake_vars.split(";")
        extra_cmake_vars = ["-D" + var for var in extra_cmake_vars]
        cmake_def_args.extend(extra_cmake_vars)

    if trt_root is not None:
        cmake_def_args.append(
            f"-DTRT_LIB_DIR={trt_root}/targets/x86_64-linux-gnu/lib")
        cmake_def_args.append(f"-DTRT_INCLUDE_DIR={trt_root}/include")

    if nccl_root is not None:
        cmake_def_args.append(f"-DNCCL_LIB_DIR={nccl_root}/lib")
        cmake_def_args.append(f"-DNCCL_INCLUDE_DIR={nccl_root}/include")

    source_dir = project_dir / "cpp"

    if build_dir is None:
        build_dir = source_dir / ("build" if build_type == "Release" else
                                  f"build_{build_type}")
    else:
        build_dir = Path(build_dir)
    first_build = not build_dir.exists()

    if clean and build_dir.exists():
        rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if use_ccache:
        cmake_def_args.append(
            f"-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
        )

    build_pyt = "OFF" if cpp_only else "ON"
    th_common_lib = "" if cpp_only else "th_common"

    with working_directory(build_dir):
        cmake_def_args = " ".join(cmake_def_args)
        if clean or first_build:
            build_run(
                f'cmake -DCMAKE_BUILD_TYPE="{build_type}" -DBUILD_PYT="{build_pyt}" "{cmake_cuda_architectures}"'
                f' {cmake_def_args} -S "{source_dir}"')
        build_run(
            f'make -j{job_count} tensorrt_llm tensorrt_llm_static nvinfer_plugin_tensorrt_llm {th_common_lib} '
            f'{" ".join(extra_make_targets)}')

    if cpp_only:
        assert not install, "Installing is not supported for cpp_only builds"
        return

    lib_dir = project_dir / "tensorrt_llm/libs"
    if lib_dir.exists():
        rmtree(lib_dir)
    lib_dir.mkdir(parents=True)
    copy(build_dir / "tensorrt_llm/thop/libth_common.so",
         lib_dir / "libth_common.so")
    copy(build_dir / "tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so",
         lib_dir / "libnvinfer_plugin_tensorrt_llm.so")

    if dist_dir is None:
        dist_dir = project_dir / "build"
    else:
        dist_dir = Path(dist_dir)

    if not dist_dir.exists():
        dist_dir.mkdir(parents=True)
    if not skip_building_wheel:
        build_run(
            f'python3 -m build {project_dir} --skip-dependency-check --no-isolation --wheel --outdir "{dist_dir}"'
        )

    if install:
        build_run('pip install -e .')


if __name__ == "__main__":
    parser = ArgumentParser()
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
        help=
        "A list of cmake variable definition, example: \"key1=value1;key2=value2\"",
        default="")
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
    args = parser.parse_args()
    main(**vars(args))
