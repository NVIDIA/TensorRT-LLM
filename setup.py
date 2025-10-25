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
from setuptools.dist import Distribution


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
        'bin/executorWorker', 'libs/libtensorrt_llm.so', 'libs/libth_common.so',
        'libs/libnvinfer_plugin_tensorrt_llm.so',
        'libs/libtensorrt_llm_ucx_wrapper.so', 'libs/libdecoder_attention_0.so',
        'libs/libtensorrt_llm_nixl_wrapper.so', 'libs/nixl/**/*',
        'libs/ucx/**/*', 'libs/libpg_utils.so',
        'libs/libdecoder_attention_1.so', 'libs/nvshmem/License.txt',
        'libs/nvshmem/nvshmem_bootstrap_uid.so.3',
        'libs/nvshmem/nvshmem_transport_ibgda.so.103', 'bindings.*.so',
        'deep_ep/LICENSE', 'deep_ep_cpp_tllm.*.so', "include/**/*",
        'deep_gemm/LICENSE', 'deep_gemm/include/**/*',
        'deep_gemm_cpp_tllm.*.so', 'scripts/install_tensorrt.sh',
        'flash_mla/LICENSE', 'flash_mla_cpp_tllm.*.so'
    ]

package_data += [
    'bindings.pyi',
    'bindings/*.pyi',
    'tools/plugin_gen/templates/*',
    'bench/build/benchmark_config.yml',
    'evaluate/lm_eval_tasks/**/*',
    "_torch/auto_deploy/config/*.yaml",
    # Include CUDA source for fused MoE align extension so runtime JIT can find it in wheels
    '_torch/auto_deploy/custom_ops/fused_moe/moe_align_kernel.cu',
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
    """Extract package data (binaries and other materials) from a precompiled wheel to the working directory.
    This allows skipping the compilation, and repackaging the binaries and Python files in the working directory to a new wheel.
    """
    import fnmatch
    import tarfile
    import zipfile
    from urllib.request import urlretrieve

    from setuptools.errors import SetupError

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
            if file.filename.endswith((".py", ".yaml")):
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

# https://setuptools.pypa.io/en/latest/references/keywords.html
setup(
    name='tensorrt_llm',
    version=get_version(),
    description='TensorRT-LLM: A TensorRT Toolbox for Large Language Models',
    long_description=
    'TensorRT-LLM: A TensorRT Toolbox for Large Language Models',
    author="NVIDIA Corporation",
    url="https://github.com/NVIDIA/TensorRT-LLM",
    download_url="https://github.com/NVIDIA/TensorRT-LLM/tags",
    packages=find_packages(),
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
