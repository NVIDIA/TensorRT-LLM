# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from typing import List

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


class BuildPyWithProtoCompile(build_py):
    """Custom build_py command that compiles protobuf files before building."""

    def run(self):
        self.compile_grpc_protos()
        super().run()

    def compile_grpc_protos(self):
        """Compile gRPC protobuf files if the proto file exists."""
        grpc_dir = Path(__file__).parent / "tensorrt_llm" / "grpc"
        proto_file = grpc_dir / "trtllm_service.proto"
        compile_script = grpc_dir / "compile_protos.py"

        if not proto_file.exists():
            return

        # Check if pb2 files need to be generated
        pb2_file = grpc_dir / "trtllm_service_pb2.py"
        pb2_grpc_file = grpc_dir / "trtllm_service_pb2_grpc.py"

        # Regenerate if pb2 files don't exist or are older than proto file
        needs_compile = (not pb2_file.exists() or not pb2_grpc_file.exists() or
                         pb2_file.stat().st_mtime < proto_file.stat().st_mtime)

        if needs_compile and compile_script.exists():
            import subprocess
            import sys

            print("Compiling gRPC protobuf files...")
            try:
                subprocess.run(
                    [sys.executable, str(compile_script)],
                    check=True,
                    cwd=str(grpc_dir.parent.parent),
                )
                print("gRPC protobuf compilation successful")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to compile gRPC protos: {e}")
            except Exception as e:
                print(f"Warning: gRPC proto compilation skipped: {e}")


def parse_requirements(filename: os.PathLike):
    with open(filename) as f:
        requirements = f.read().splitlines()

        def extract_url(line):
            return next(filter(lambda x: x[0] != '-', line.split()))

        extra_URLs = []
        deps = []
        for line in requirements:
            if line.startswith("#") or line.startswith("-r") or line.startswith(
                    "-c"):
                continue

            # handle -i and --extra-index-url options
            if "-i " in line or "--extra-index-url" in line:
                extra_URLs.append(extract_url(line))
            # handle URLs such as git+https://github.com/flashinfer-ai/flashinfer.git@e3853dd#egg=flashinfer-python
            elif line.startswith("git+https"):
                idx = line.find("egg=")
                dep = line[idx + 4:]
                deps.append(dep)
            else:
                deps.append(line)
    return deps, extra_URLs


def sanity_check():
    tensorrt_llm_path = Path(__file__).resolve().parent / "tensorrt_llm"
    if not (tensorrt_llm_path / "bindings").exists():
        raise ImportError(
            'The `bindings` module does not exist. Please check the package integrity. '
            'If you are attempting to use the pip development mode (editable installation), '
            'please execute `scripts/build_wheel.py` first, and then run `pip install -e .`.'
        )


def get_version():
    version_file = Path(
        __file__).resolve().parent / "tensorrt_llm" / "version.py"
    version = None
    with open(version_file) as f:
        for line in f:
            if not line.startswith("__version__"):
                continue
            version = line.split('"')[1]

    if version is None:
        raise RuntimeError(f"Could not set version from {version_file}")

    return version


def get_license():
    import sysconfig
    platform_tag = sysconfig.get_platform()
    if "x86_64" in platform_tag:
        return ["LICENSE", "ATTRIBUTIONS-CPP-x86_64.md"]
    elif "arm64" in platform_tag or "aarch64" in platform_tag:
        return ["LICENSE", "ATTRIBUTIONS-CPP-aarch64.md"]
    else:
        raise RuntimeError(f"Unrecognized CPU architecture: {platform_tag}")


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


on_windows = platform.system() == "Windows"
required_deps, extra_URLs = parse_requirements(
    Path("requirements-windows.txt" if on_windows else "requirements.txt"))
devel_deps, _ = parse_requirements(
    Path("requirements-dev-windows.txt"
         if on_windows else "requirements-dev.txt"))
constraints_file = Path("constraints.txt")
if constraints_file.exists():
    constraints, _ = parse_requirements(constraints_file)
    required_deps.extend(constraints)

if on_windows:
    package_data = [
        'libs/th_common.dll', 'libs/tensorrt_llm.dll',
        'libs/nvinfer_plugin_tensorrt_llm.dll', 'bindings.*.pyd', "include/**/*"
    ]
else:
    package_data = [
        'bin/executorWorker',
        'libs/libtensorrt_llm.so',
        'libs/libth_common.so',
        'libs/libnvinfer_plugin_tensorrt_llm.so',
        'libs/libtensorrt_llm_ucx_wrapper.so',
        'libs/libdecoder_attention_0.so',
        'libs/libtensorrt_llm_nixl_wrapper.so',
        'libs/nixl/**/*',
        'tensorrt_llm_transfer_agent_binding*.so',
        'tensorrt_llm_transfer_agent_binding.pyi',
        'libs/libtensorrt_llm_mooncake_wrapper.so',
        'libs/ucx/**/*',
        'libs/libpg_utils.so',
        'libs/libdecoder_attention_1.so',
        'libs/nvshmem/License.txt',
        'libs/nvshmem/nvshmem_bootstrap_uid.so.3',
        'libs/nvshmem/nvshmem_transport_ibgda.so.103',
        'bindings.*.so',
        'deep_ep/LICENSE',
        'deep_ep/*.py',
        'deep_ep_cpp_tllm.*.so',
        "include/**/*",
        'deep_gemm/LICENSE',
        'deep_gemm/include/**/*',
        'deep_gemm/*.py',
        'deep_gemm_cpp_tllm.*.so',
        'scripts/install_tensorrt.sh',
        'flash_mla/LICENSE',
        'flash_mla/*.py',
        'flash_mla_cpp_tllm.*.so',
        'runtime/kv_cache_manager_v2/*.so',
        'runtime/kv_cache_manager_v2/**/*.so',
        'runtime/kv_cache_manager_v2/*.pyi',
        'runtime/kv_cache_manager_v2/**/*.pyi',
        'runtime/kv_cache_manager_v2/rawref/*.py',
        'runtime/kv_cache_manager_v2/rawref/*.pyi',
        'runtime/*__mypyc*.so',
    ]

package_data += [
    'bindings/*.pyi',
    'tools/plugin_gen/templates/*',
    'bench/build/benchmark_config.yml',
    'evaluate/lm_eval_tasks/**/*',
    "_torch/auto_deploy/config/*.yaml",
    # Include CUDA source for fused MoE align extension so runtime JIT can find it in wheels
    '_torch/auto_deploy/custom_ops/fused_moe/moe_align_kernel.cu',
    '_torch/auto_deploy/custom_ops/fused_moe/triton_fused_moe_configs/*'
]


def download_precompiled(workspace: str, version: str) -> str:
    import glob
    import subprocess

    from setuptools.errors import SetupError

    cmd = [
        "python3", "-m", "pip", "download", f"tensorrt_llm=={version}",
        f"--dest={workspace}", "--no-deps",
        "--extra-index-url=https://pypi.nvidia.com"
    ]
    try:
        subprocess.check_call(cmd)
        wheel_path = glob.glob(f"{workspace}/tensorrt_llm-*.whl")[0]
    except Exception as e:
        raise SetupError(
            "Failed to download the automatically resolved wheel, please try specifying TRTLLM_USE_PRECOMPILED with a link or local path to a valid wheel."
        ) from e
    else:
        return wheel_path


def extract_from_precompiled(precompiled_location: str, package_data: List[str],
                             workspace: str) -> None:
    """Extract package data (binaries and other materials) from a precompiled wheel or local directory to the working directory.
    This allows skipping the compilation, and repackaging the binaries and Python files in the working directory to a new wheel.

    Supports three source types:
    - Local directory (git clone structure): e.g., /home/dev/TensorRT-LLM
    - Local wheel file: e.g., /path/to/tensorrt_llm-*.whl
    - Remote URL: Downloads and extracts from URL (wheel or tar.gz)
    """
    import fnmatch
    import shutil
    import tarfile
    import zipfile
    from urllib.request import urlretrieve

    from setuptools.errors import SetupError

    # Handle local directory (assuming repo structure)
    if os.path.isdir(precompiled_location):
        precompiled_location = os.path.abspath(precompiled_location)
        print(
            f"Using local directory as precompiled source: {precompiled_location}"
        )
        source_tensorrt_llm = os.path.join(precompiled_location, "tensorrt_llm")
        if not os.path.isdir(source_tensorrt_llm):
            raise SetupError(
                f"Directory {precompiled_location} does not contain a tensorrt_llm folder."
            )

        # Walk through all files and match using fnmatch (consistent with wheel extraction)
        for root, dirs, files in os.walk(source_tensorrt_llm):
            for filename in files:
                src_file = os.path.join(root, filename)
                # Get path relative to precompiled_location (e.g., "tensorrt_llm/libs/libtensorrt_llm.so")
                rel_path = os.path.relpath(src_file, precompiled_location)
                dst_file = rel_path

                # Skip yaml files
                if dst_file.endswith(".yaml"):
                    continue

                # Skip .py files EXCEPT for generated C++ extension wrappers
                # (deep_gemm, deep_ep, flash_mla Python files are generated during build)
                if dst_file.endswith(".py"):
                    allowed_dirs = ("tensorrt_llm/deep_gemm/",
                                    "tensorrt_llm/deep_ep/",
                                    "tensorrt_llm/flash_mla/")
                    if not any(dst_file.startswith(d) for d in allowed_dirs):
                        continue

                # Check if file matches any pattern using fnmatch (same as wheel extraction)
                for filename_pattern in package_data:
                    if fnmatch.fnmatchcase(rel_path,
                                           f"tensorrt_llm/{filename_pattern}"):
                        break
                else:
                    continue

                dst_dir = os.path.dirname(dst_file)
                if dst_dir:
                    os.makedirs(dst_dir, exist_ok=True)
                print(f"Copying {rel_path} from local directory.")
                shutil.copy2(src_file, dst_file)
        return

    # Handle local file or remote URL
    if os.path.isfile(precompiled_location):
        precompiled_path = precompiled_location
        print(f"Using local precompiled file: {precompiled_path}.")
    else:
        precompiled_filename = precompiled_location.split("/")[-1]
        precompiled_path = os.path.join(workspace, precompiled_filename)
        print(
            f"Downloading precompiled file from {precompiled_location} to {precompiled_path}."
        )
        try:
            urlretrieve(precompiled_location, filename=precompiled_path)
        except Exception as e:
            raise SetupError(
                f"Failed to get precompiled file from {precompiled_location}."
            ) from e

    if precompiled_path.endswith("tar.gz"):
        with tarfile.open(precompiled_path, "r:gz") as tar:
            for member in tar.getmembers():
                if fnmatch.fnmatchcase(member.name,
                                       "TensorRT-LLM/tensorrt_llm-*.whl"):
                    break
            else:
                raise SetupError(
                    f"Failed to get wheel file from {precompiled_path}.") from e

            wheel_path = os.path.join(workspace, member.name)
            tar.extract(member, path=workspace, filter=tarfile.data_filter)
    else:
        wheel_path = precompiled_path

    with zipfile.ZipFile(wheel_path) as wheel:
        for file in wheel.filelist:
            # Skip yaml files
            if file.filename.endswith(".yaml"):
                continue

            # Skip .py files EXCEPT for generated C++ extension wrappers
            # (deep_gemm, deep_ep, flash_mla Python files are generated during build)
            if file.filename.endswith(".py"):
                allowed_dirs = (
                    "tensorrt_llm/deep_gemm/", "tensorrt_llm/deep_ep/",
                    "tensorrt_llm/flash_mla/",
                    "tensorrt_llm/runtime/kv_cache_manager_v2/rawref/__init__.py"
                )
                if not any(file.filename.startswith(d) for d in allowed_dirs):
                    # Exclude all .py files in kv_cache_manager_v2 except rawref/__init__.py
                    if file.filename.startswith("tensorrt_llm/runtime/kv_cache_manager_v2/") and \
                       not file.filename.endswith("rawref/__init__.py"):
                        continue
                    continue

            for filename_pattern in package_data:
                if fnmatch.fnmatchcase(file.filename,
                                       f"tensorrt_llm/{filename_pattern}"):
                    break
            else:
                continue
            print(
                f"Extracting and including {file.filename} from precompiled wheel."
            )
            wheel.extract(file)


precompiled: str | None = os.getenv("TRTLLM_USE_PRECOMPILED")
precompiled_location: str | None = os.getenv("TRTLLM_PRECOMPILED_LOCATION")
use_precompiled: bool = (precompiled is not None
                         and precompiled != "0") or (precompiled_location
                                                     is not None)

if use_precompiled:
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tempdir:
        if not precompiled_location:
            version = precompiled if precompiled != "1" else get_version()
            precompiled_location = download_precompiled(tempdir, version)
        extract_from_precompiled(precompiled_location, package_data, tempdir)

sanity_check()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    # We use find_packages with a custom exclude filter to handle the mypyc compiled modules.
    # We want to exclude the .py source files for modules that are compiled to .so.
    # We exclude the kv_cache_manager_v2 package entirely from the source list,
    # but explicitly add back the rawref subpackage (which is not compiled by mypyc).
    # The .so and .pyi files for kv_cache_manager_v2 are added via package_data.
enable_mypyc = os.getenv("TRTLLM_ENABLE_MYPYC", "0") == "1"
if enable_mypyc:
    packages = find_packages(exclude=[
        "tensorrt_llm.runtime.kv_cache_manager_v2",
        "tensorrt_llm.runtime.kv_cache_manager_v2.*",
    ]) + ["tensorrt_llm.runtime.kv_cache_manager_v2.rawref"]
    exclude_package_data = {
        "tensorrt_llm": [
            "runtime/kv_cache_manager_v2/*.py",
            "runtime/kv_cache_manager_v2/**/*.py"
        ],
        "tensorrt_llm.runtime.kv_cache_manager_v2": ["*.py", "**/*.py"],
    }
else:
    packages = find_packages()
    exclude_package_data = {}

    # Remove mypyc shared objects from package_data to avoid packaging stale files
    package_data = [
        p for p in package_data if p not in [
            'runtime/kv_cache_manager_v2/*.so',
            'runtime/kv_cache_manager_v2/**/*.so', 'runtime/*__mypyc*.so'
        ]
    ]
    # Ensure rawref is included
    package_data.append('runtime/kv_cache_manager_v2/rawref/*.so')

# Add vendored triton_kernels as an explicit top-level package.
# This is vendored from the Triton project and kept at repo root so its
# internal absolute imports (e.g., "from triton_kernels.foo import bar") work.
packages += find_packages(include=["triton_kernels", "triton_kernels.*"])

# https://setuptools.pypa.io/en/latest/references/keywords.html
setup(
    name='tensorrt_llm',
    version=get_version(),
    cmdclass={'build_py': BuildPyWithProtoCompile},
    description=
    ('TensorRT LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and supports '
     'state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.'
     ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    url="https://github.com/NVIDIA/TensorRT-LLM",
    download_url="https://github.com/NVIDIA/TensorRT-LLM/tags",
    packages=packages,
    exclude_package_data=exclude_package_data,
    # TODO Add windows support for python bindings.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.12",
    ],
    distclass=BinaryDistribution,
    license="Apache License 2.0",
    keywords="nvidia tensorrt deeplearning inference",
    package_data={
        'tensorrt_llm': package_data,
        'triton_kernels': ['LICENSE', 'VERSION', 'README.md'],
    },
    license_files=get_license(),
    entry_points={
        'console_scripts': [
            'trtllm-build=tensorrt_llm.commands.build:main',
            'trtllm-prune=tensorrt_llm.commands.prune:main',
            'trtllm-refit=tensorrt_llm.commands.refit:main',
            'trtllm-bench=tensorrt_llm.commands.bench:main',
            'trtllm-serve=tensorrt_llm.commands.serve:main',
            'trtllm-eval=tensorrt_llm.commands.eval:main'
        ],
    },
    scripts=['tensorrt_llm/llmapi/trtllm-llmapi-launch'],
    extras_require={
        "devel": devel_deps,
    },
    zip_safe=True,
    install_requires=required_deps,
    dependency_links=
    extra_URLs,  # Warning: Dependency links support has been dropped by pip 19.0
    python_requires=">=3.10, <4")
