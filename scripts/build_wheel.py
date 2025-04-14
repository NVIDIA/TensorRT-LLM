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
import re
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copy, copytree, rmtree
from subprocess import DEVNULL, CalledProcessError, check_output, run
from tempfile import TemporaryDirectory
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


def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            rmtree(item_path)
        else:
            os.remove(item_path)


def main(*,
         build_type: str = "Release",
         generator: str = "",
         build_dir: Path = None,
         dist_dir: Path = None,
         cuda_architectures: str = None,
         job_count: int = None,
         extra_cmake_vars: List[str] = list(),
         extra_make_targets: str = "",
         trt_root: str = '/usr/local/tensorrt',
         nccl_root: str = None,
         nvrtc_wrapper_root: str = None,
         clean: bool = False,
         clean_wheel: bool = False,
         configure_cmake: bool = False,
         use_ccache: bool = False,
         fast_build: bool = False,
         cpp_only: bool = False,
         install: bool = False,
         skip_building_wheel: bool = False,
         linking_install_binary: bool = False,
         python_bindings: bool = True,
         benchmarks: bool = False,
         micro_benchmarks: bool = False,
         nvtx: bool = False,
         skip_stubs: bool = False):

    if clean:
        clean_wheel = True

    project_dir = get_project_dir()
    os.chdir(project_dir)
    build_run = partial(run, shell=True, check=True)

    # Get all submodules and check their folder exists. If not,
    # invoke git submodule update
    with open(project_dir / ".gitmodules", "r") as submodules_f:
        submodules = [
            l.split("=")[1].strip() for l in submodules_f.readlines()
            if "path = " in l
        ]
    if any(not (project_dir / submodule / ".git").exists()
           for submodule in submodules):
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

    if cuda_architectures is not None:
        if "70-real" in cuda_architectures:
            raise RuntimeError("Volta architecture is deprecated support.")

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

    if generator:
        cmake_generator = "-G" + generator

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
    first_build = not Path(build_dir, "CMakeFiles").exists()

    if clean and build_dir.exists():
        clear_folder(build_dir)  # Keep the folder in case it is mounted.
    build_dir.mkdir(parents=True, exist_ok=True)

    if use_ccache:
        cmake_def_args.append(
            f"-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
        )

    if fast_build:
        cmake_def_args.append(f"-DFAST_BUILD=ON")

    targets = ["tensorrt_llm", "nvinfer_plugin_tensorrt_llm"]

    if cpp_only:
        build_pyt = "OFF"
        build_pybind = "OFF"
    else:
        targets.extend(["bindings", "th_common"])
        build_pyt = "ON"
        build_pybind = "ON"

    if benchmarks:
        targets.append("benchmarks")

    if micro_benchmarks:
        targets.append("micro_benchmarks")
        build_micro_benchmarks = "ON"
    else:
        build_micro_benchmarks = "OFF"

    disable_nvtx = "OFF" if nvtx else "ON"

    if not on_windows:
        targets.append("executorWorker")

    source_dir = get_source_dir()

    def install_conan():
        # Determine the system ID
        with Path("/etc/os-release").open("r") as f:
            for line in f:
                if line.startswith("ID="):
                    system_id = line.split("=")[1].strip()
                    break
            else:
                system_id = "unknown"
        # Install Conan if it's not already installed
        # TODO move this install to the container image
        conan_path = "conan"
        if "rocky" not in system_id:
            build_run(f"\"{sys.executable}\" -m pip install conan==2.14.0")
        else:
            conan_dir = Path(build_dir, "tool/conan")
            conan_dir.mkdir(parents=True, exist_ok=True)
            conan_path = conan_dir / "bin/conan"
            if not conan_path.exists():
                with TemporaryDirectory() as tmpdir:
                    tmpdir_p = Path(tmpdir)
                    archive_p = tmpdir_p / "conan.tgz"
                    build_run(
                        f"wget --retry-connrefused -O {archive_p} https://github.com/conan-io/conan/releases/download/2.14.0/conan-2.14.0-linux-x86_64.tgz"
                    )
                    build_run(f"tar -C {conan_dir} -xf {archive_p}")
        # Install dependencies with Conan
        build_run(
            f"{conan_path} remote add -verror --force tensorrt-llm https://edge.urm.nvidia.com/artifactory/api/conan/sw-tensorrt-llm-conan"
        )
        build_run(f"{conan_path} profile detect -f")
        return conan_path

    conan_path = install_conan()

    # Build the NVRTC wrapper if the source directory exists
    if nvrtc_wrapper_root is not None and Path(nvrtc_wrapper_root).exists():
        print(f"Building the NVRTC wrapper from source in {nvrtc_wrapper_root}")
        conan_data = Path(source_dir, "conandata.yml").read_text()
        nvrtc_wrapper_version = re.search(
            r'tensorrt_llm_nvrtc_wrapper:\s*(\S+)', conan_data).group(1)
        build_run(
            f"{conan_path} editable add {nvrtc_wrapper_root}/conan/nvrtc_wrapper --version {nvrtc_wrapper_version}"
        )
        nvrtc_wrapper_args = ""
        if clean:
            nvrtc_wrapper_args += " -c"
        if configure_cmake:
            nvrtc_wrapper_args += " --configure_cmake"
        if use_ccache:
            nvrtc_wrapper_args += " --use_ccache"
        build_run(
            f'"{sys.executable}" {nvrtc_wrapper_root}/scripts/build_wheel.py {nvrtc_wrapper_args} -a "{cuda_architectures}" -D "USE_CXX11_ABI=1;BUILD_NVRTC_WRAPPER=1" -l'
        )
    else:
        # If the NVRTC wrapper source directory is not present, remove the editable NVRTC wrapper from the conan cache
        build_run(
            f"{conan_path} editable remove -r 'tensorrt_llm_nvrtc_wrapper/*'",
            stdout=DEVNULL,
            stderr=DEVNULL)

    with working_directory(build_dir):
        if clean or first_build or configure_cmake:
            build_run(
                f"{conan_path} install --remote=tensorrt-llm --output-folder={build_dir}/conan -s 'build_type={build_type}' {source_dir}"
            )
            cmake_def_args.append(
                f"-DCMAKE_TOOLCHAIN_FILE={build_dir}/conan/conan_toolchain.cmake"
            )
            cmake_def_args = " ".join(cmake_def_args)
            cmake_configure_command = (
                f'cmake -DCMAKE_BUILD_TYPE="{build_type}" -DBUILD_PYT="{build_pyt}" -DBUILD_PYBIND="{build_pybind}"'
                f' -DNVTX_DISABLE="{disable_nvtx}" -DBUILD_MICRO_BENCHMARKS={build_micro_benchmarks}'
                f' -DBUILD_WHEEL_TARGETS="{";".join(targets)}"'
                f' {cmake_cuda_architectures} {cmake_def_args} {cmake_generator} -S "{source_dir}"'
            )
            print("CMake Configure command: ")
            print(cmake_configure_command)
            build_run(cmake_configure_command)
        cmake_build_command = (
            f'cmake --build . --config {build_type} --parallel {job_count} '
            f'--target build_wheel_targets {" ".join(extra_make_targets)}')
        print("CMake Build command: ")
        print(cmake_build_command)
        build_run(cmake_build_command)

    if cpp_only:
        assert not install, "Installing is not supported for cpp_only builds"
        return

    pkg_dir = project_dir / "tensorrt_llm"
    assert pkg_dir.is_dir(), f"{pkg_dir} is not a directory"
    lib_dir = pkg_dir / "libs"
    include_dir = pkg_dir / "include"
    if lib_dir.exists():
        clear_folder(lib_dir)
    if include_dir.exists():
        clear_folder(include_dir)

    cache_dir = os.getenv("TRTLLM_DG_CACHE_DIR")
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
    elif on_windows:
        if os.getenv("APPDATA") is not None:
            cache_dir = Path(os.getenv("APPDATA")) / "tensorrt_llm"
        else:
            cache_dir = Path(os.getenv("TEMP")) / "tensorrt_llm"
    else:
        if os.getenv("HOME") is not None:
            cache_dir = Path(os.getenv("HOME")) / ".tensorrt_llm"
        else:
            cache_dir = Path(os.getenv("TEMP"), "/tmp") / "tensorrt_llm"
    if cache_dir.exists():
        clear_folder(cache_dir)

    install_file = copy
    install_tree = copytree
    if skip_building_wheel and linking_install_binary:

        def symlink_remove_dst(src, dst):
            src = os.path.abspath(src)
            dst = os.path.abspath(dst)
            if os.path.isdir(dst):
                dst = os.path.join(dst, os.path.basename(src))
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        install_file = symlink_remove_dst

        def symlink_remove_dst_tree(src, dst, dirs_exist_ok=True):
            src = os.path.abspath(src)
            dst = os.path.abspath(dst)
            if dirs_exist_ok and os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        install_tree = symlink_remove_dst_tree

    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    install_tree(get_source_dir() / "include" / "tensorrt_llm" / "deep_gemm",
                 include_dir / "deep_gemm",
                 dirs_exist_ok=True)
    required_cuda_headers = [
        "cuda_fp16.h", "cuda_fp16.hpp", "cuda_bf16.h", "cuda_bf16.hpp",
        "cuda_fp8.h", "cuda_fp8.hpp"
    ]
    if os.getenv("CUDA_HOME") is not None:
        cuda_include_dir = Path(os.getenv("CUDA_HOME")) / "include"
    elif os.getenv("CUDA_PATH") is not None:
        cuda_include_dir = Path(os.getenv("CUDA_PATH")) / "include"
    elif not on_windows:
        cuda_include_dir = Path("/usr/local/cuda/include")
    else:
        cuda_include_dir = None

    if cuda_include_dir is None or not cuda_include_dir.exists():
        print(
            "CUDA_HOME or CUDA_PATH should be set to enable DeepGEMM JIT compilation"
        )
    else:
        cuda_include_target_dir = include_dir / "cuda" / "include"
        cuda_include_target_dir.mkdir(parents=True, exist_ok=True)
        for header in required_cuda_headers:
            install_file(cuda_include_dir / header, include_dir / header)

    if on_windows:
        install_file(build_dir / "tensorrt_llm/tensorrt_llm.dll",
                     lib_dir / "tensorrt_llm.dll")
        install_file(build_dir / f"tensorrt_llm/thop/th_common.dll",
                     lib_dir / "th_common.dll")
        install_file(
            build_dir / f"tensorrt_llm/plugins/nvinfer_plugin_tensorrt_llm.dll",
            lib_dir / "nvinfer_plugin_tensorrt_llm.dll")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/tensorrt_llm_nvrtc_wrapper.dll",
            lib_dir / "tensorrt_llm_nvrtc_wrapper.dll")
    else:
        install_file(build_dir / "tensorrt_llm/libtensorrt_llm.so",
                     lib_dir / "libtensorrt_llm.so")
        install_file(build_dir / "tensorrt_llm/thop/libth_common.so",
                     lib_dir / "libth_common.so")
        install_file(
            build_dir /
            "tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so",
            lib_dir / "libnvinfer_plugin_tensorrt_llm.so")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/libtensorrt_llm_nvrtc_wrapper.so",
            lib_dir / "libtensorrt_llm_nvrtc_wrapper.so")
        if os.path.exists(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/ucx_utils/libtensorrt_llm_ucx_wrapper.so"
        ):
            install_file(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/ucx_utils/libtensorrt_llm_ucx_wrapper.so",
                lib_dir / "libtensorrt_llm_ucx_wrapper.so")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so",
            lib_dir / "libdecoder_attention_0.so")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so",
            lib_dir / "libdecoder_attention_1.so")

    bin_dir = pkg_dir / "bin"
    if bin_dir.exists():
        clear_folder(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)

    if not on_windows:
        install_file(build_dir / "tensorrt_llm/executor_worker/executorWorker",
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

        install_file(get_pybind_lib(), pkg_dir)
        if not skip_stubs:
            with working_directory(project_dir):
                build_run(
                    f"\"{sys.executable}\" -m pip install pybind11-stubgen")
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
                            f"\"{sys.executable}\" -m pybind11_stubgen -o . bindings --exit-code",
                            env=env_ld)
                    except CalledProcessError as ex:
                        print(f"Failed to build pybind11 stubgen: {ex}",
                              file=sys.stderr)
                        exit(1)

    if not skip_building_wheel:
        if dist_dir is None:
            dist_dir = project_dir / "build"
        else:
            dist_dir = Path(dist_dir)

        if not dist_dir.exists():
            dist_dir.mkdir(parents=True)

        if clean_wheel:
            # For incremental build, the python build module adds
            # the new files but does not remove the deleted files.
            #
            # This breaks the Windows CI/CD pipeline when building
            # and validating python changes in the whl.
            clear_folder(dist_dir)

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
    parser.add_argument("--generator", "-G", default="")
    parser.add_argument("--cuda_architectures", "-a")
    parser.add_argument("--install", "-i", action="store_true")
    parser.add_argument("--clean", "-c", action="store_true")
    parser.add_argument("--clean_wheel",
                        action="store_true",
                        help="Clear dist_dir folder creating wheel")
    parser.add_argument("--configure_cmake",
                        action="store_true",
                        help="Always configure cmake before building")
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
                        default="/usr/local/tensorrt",
                        help="Directory to find TensorRT headers/libs")
    parser.add_argument("--nccl_root",
                        help="Directory to find NCCL headers/libs")
    parser.add_argument(
        "--nvrtc_wrapper_root",
        default="/mnt/src/tensorrt_llm_nvrtc_wrapper",
        help=
        "Directory to find internal NVRTC wrapper source code. If the directory exists, the NVRTC wrapper will be built from source."
    )
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
        "--linking_install_binary",
        action="store_true",
        help="Install the built binary by symbolic linking instead of copying.")
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
    parser.add_argument("--skip-stubs",
                        action="store_true",
                        help="Skip building python stubs")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(**vars(args))
