# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_only, pytest.mark.skipif(sys.platform != "linux", reason="ELF test")]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NVRTC_SOURCE = (
    _REPO_ROOT
    / "cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/src"
)
_NVRTC_SYMBOLS = (
    "nvrtcCreateProgram",
    "nvrtcCompileProgram",
    "nvrtcGetProgramLogSize",
    "nvrtcGetProgramLog",
    "nvrtcGetCUBINSize",
    "nvrtcGetCUBIN",
    "nvrtcDestroyProgram",
)


@pytest.fixture(scope="module")
def compiler() -> str:
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("A C++ compiler is required")
    return executable


def _compile(compiler: str, output: Path, *args: str) -> None:
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            "-Werror",
            *args,
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def linkage_libraries(compiler: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("nvrtc-linkage")
    # Only addresses are inspected: no CUDA toolkit or GPU is needed.
    (directory / "nvrtc.h").write_text(
        "\n".join(f'extern "C" int {symbol}();' for symbol in _NVRTC_SYMBOLS)
    )
    stubs = directory / "stubs.cpp"
    stubs.write_text(
        "\n".join(f'extern "C" int {symbol}() {{ return 0; }}' for symbol in _NVRTC_SYMBOLS)
    )
    bridge = directory / "bridge.cpp"
    bridge.write_text(
        '#include "nvrtcLinkage.h"\n'
        'extern "C" char const* linkage_error() {\n'
        "    static std::string error;\n"
        "    error = tensorrt_llm::kernels::getNvrtcLinkageError();\n"
        "    return error.c_str();\n"
        "}\n"
    )
    shared = directory / "libexternal_compiler.so"
    _compile(compiler, shared, str(stubs), "-Wl,-soname,libexternal_compiler.so")
    common = [
        str(_NVRTC_SOURCE / "nvrtcLinkage.cpp"),
        str(bridge),
        "-I",
        str(directory),
        "-I",
        str(_NVRTC_SOURCE),
        "-I",
        str(_REPO_ROOT / "cpp/include"),
        "-ldl",
    ]
    _compile(
        compiler,
        directory / "libstatic.so",
        *common,
        "-DTRTLLM_NVRTC_DYNAMIC_LINKING=0",
        str(stubs),
    )
    for name, dynamic in (("libdynamic.so", 1), ("libunexpected_dynamic.so", 0)):
        _compile(
            compiler,
            directory / name,
            *common,
            f"-DTRTLLM_NVRTC_DYNAMIC_LINKING={dynamic}",
            str(shared),
            "-Wl,-rpath,$ORIGIN",
        )
    return directory


def _diagnose(library: Path, preload: Path | None = None, local_library: Path | None = None) -> str:
    env = os.environ.copy()
    # Test each loader state in a fresh process without changing pytest's state.
    env.pop("LD_PRELOAD", None)
    if preload is not None:
        env["LD_PRELOAD"] = str(preload)
    script = (
        "import ctypes, sys\n"
        "if sys.argv[2]:\n"
        "    other = ctypes.CDLL(sys.argv[2], mode=ctypes.RTLD_LOCAL)\n"
        "library = ctypes.CDLL(sys.argv[1])\n"
        "library.linkage_error.restype = ctypes.c_char_p\n"
        "print(library.linkage_error().decode())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(library), str(local_library or "")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def test_static_nvrtc_does_not_report_a_conflict(linkage_libraries: Path) -> None:
    assert _diagnose(linkage_libraries / "libstatic.so") == ""


def test_private_dynamic_nvrtc_can_coexist_with_static_nvrtc(linkage_libraries: Path) -> None:
    assert (
        _diagnose(
            linkage_libraries / "libstatic.so",
            local_library=linkage_libraries / "libexternal_compiler.so",
        )
        == ""
    )


@pytest.mark.parametrize("symbol", _NVRTC_SYMBOLS)
def test_static_nvrtc_reports_partial_interposition(
    compiler: str, linkage_libraries: Path, tmp_path: Path, symbol: str
) -> None:
    source = tmp_path / "interpose.cpp"
    source.write_text(f'extern "C" int {symbol}() {{ return 0; }}\n')
    interposer = tmp_path / "libinterposer.so"
    _compile(compiler, interposer, str(source))

    error = _diagnose(linkage_libraries / "libstatic.so", preload=interposer)

    assert f"NVRTC_DYNAMIC_LINKING=OFF, but {symbol} resolves to {interposer}" in error
    assert str(linkage_libraries / "libstatic.so") in error
    assert "NVRTC_DYNAMIC_LINKING=ON" in error


def test_static_nvrtc_reports_an_unexpected_dynamic_dependency(linkage_libraries: Path) -> None:
    error = _diagnose(linkage_libraries / "libunexpected_dynamic.so")

    assert "NVRTC_DYNAMIC_LINKING=OFF" in error
    assert "libexternal_compiler.so" in error


def test_dynamic_nvrtc_allows_a_dynamic_provider(linkage_libraries: Path) -> None:
    assert _diagnose(linkage_libraries / "libdynamic.so") == ""
    assert (
        _diagnose(
            linkage_libraries / "libdynamic.so",
            preload=linkage_libraries / "libexternal_compiler.so",
        )
        == ""
    )


def test_pg_utils_does_not_link_torch_cuda_dependencies(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    torch_lib = tmp_path / "torch/lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "libtorch_python.so").touch()
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(pg_utils_linkage LANGUAGES CXX)\n"
        "add_library(torch_cpu INTERFACE)\n"
        "add_library(c10 INTERFACE)\n"
        "add_library(dynamic_nvrtc INTERFACE)\n"
        "set(TORCH_LIBRARIES torch_cpu c10 dynamic_nvrtc)\n"
        f'set(TORCH_INSTALL_PREFIX "{torch_lib.parent.as_posix()}")\n'
        f'add_subdirectory("{_REPO_ROOT.as_posix()}/cpp/tensorrt_llm/runtime/utils" pg_utils)\n'
        "get_target_property(link_libraries pg_utils LINK_LIBRARIES)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/link_libraries.txt" "${link_libraries}")\n'
    )
    build = tmp_path / "build"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (build / "link_libraries.txt").read_text().split(";") == ["torch_cpu", "c10"]
