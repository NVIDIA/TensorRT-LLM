# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only regression coverage for CMake components and test helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
REPO_ROOT = Path(__file__).resolve().parents[3]


def configure_kernel_ownership_fixture(tmp_path: Path, extra: str = "") -> Path:
    if not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip("CMake and Ninja are required for ownership integration coverage")
    module = REPO_ROOT / "cpp/tensorrt_llm/kernels/kernelSourceOwnership.cmake"
    (tmp_path / "owned.cpp").write_text("int owned() { return 1; }\n")
    (tmp_path / "CMakeLists.txt").write_text(
        f'''cmake_minimum_required(VERSION 3.27)
project(kernel_ownership LANGUAGES CXX)
include("{module.as_posix()}")
add_library(owner STATIC owned.cpp)
{extra}
tllm_check_kernel_source_ownership(TARGETS owner ${{other_targets}}
  ${{delegation}})
'''
    )
    return tmp_path / "build"


def test_new_kernel_file_fails_reconfiguration_before_compilation(tmp_path: Path) -> None:
    build_dir = configure_kernel_ownership_fixture(tmp_path)
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    (tmp_path / "unowned.cpp").write_text("#error This file must not be compiled\n")
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "owner"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode
    assert "Kernel source has no explicit owner: unowned.cpp" in result.stdout + result.stderr
    assert not list(build_dir.rglob("*.o"))


def test_duplicate_kernel_owners_fail_configuration(tmp_path: Path) -> None:
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "add_library(other STATIC owned.cpp)\nset(other_targets other)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode
    assert "Kernel source has multiple owners: owned.cpp: owner, other" in result.stderr


@pytest.mark.parametrize("source", ["backend/optional.cpp", "backend_extra/unowned.cpp"])
def test_backend_delegation_has_directory_boundaries(tmp_path: Path, source: str) -> None:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend/CMakeLists.txt").write_text("# Optional backend owns its selection.\n")
    path = tmp_path / source
    path.parent.mkdir(exist_ok=True)
    path.write_text("int optional() { return 0; }\n")
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "set(delegation DELEGATED_DIRECTORIES backend)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if source.startswith("backend/"):
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode
        assert f"Kernel source has no explicit owner: {source}" in result.stderr


def test_missing_backend_definition_is_rejected(tmp_path: Path) -> None:
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "set(delegation DELEGATED_DIRECTORIES missing_backend)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode
    assert "Missing delegated kernel build: missing_backend" in result.stderr


def test_kernel_helper_keeps_implementation_usage_private(tmp_path: Path) -> None:
    if not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip("CMake and Ninja are required for kernel helper integration coverage")
    module = (REPO_ROOT / "cpp/tensorrt_llm/kernels/kernelComponents.cmake").read_text()
    start = module.index("function(add_tllm_kernel_library target)")
    end = module.index("endfunction()", start) + len("endfunction()")
    (tmp_path / "helper.cmake").write_text(module[start:end] + "\n")
    (tmp_path / "public.h").write_text("int component();\n")
    (tmp_path / "private.h").write_text("int implementation();\n")
    (tmp_path / "implementation.cpp").write_text("int implementation() { return 7; }\n")
    (tmp_path / "component.cpp").write_text(
        '#include "private.h"\n#ifndef PRIVATE_USAGE\n#error Missing private usage\n#endif\n'
        "int component() { return implementation(); }\n"
    )
    (tmp_path / "consumer.cpp").write_text(
        '#include "public.h"\n#ifdef PRIVATE_USAGE\n#error Private usage leaked\n#endif\n'
        "#ifndef PUBLIC_USAGE\n#error Missing public usage\n#endif\n"
        "int main() { return component() == 7 ? 0 : 1; }\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.27)
project(kernel_helper LANGUAGES CXX)
function(add_cuda_architectures)
endfunction()
add_library(common INTERFACE)
add_library(tllm::common_cuda ALIAS common)
add_library(public_headers INTERFACE)
target_compile_definitions(public_headers INTERFACE PUBLIC_USAGE)
add_library(implementation STATIC implementation.cpp)
target_compile_definitions(implementation INTERFACE PRIVATE_USAGE)
include(helper.cmake)
add_tllm_kernel_library(tllm_kernel_example SOURCES component.cpp
  LINK_LIBRARIES implementation PUBLIC_LINK_LIBRARIES public_headers)
add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE tllm::kernel_example)
"""
    )
    build_dir = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "consumer"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    executable = "consumer.exe" if sys.platform == "win32" else "consumer"
    subprocess.run([str(build_dir / executable)], check=True, timeout=30)


def test_delegation_names_build_definition_outside_source_directory(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend/optional.cpp").write_text("int optional() { return 0; }\n")
    (tmp_path / "backend_build").mkdir()
    (tmp_path / "backend_build/CMakeLists.txt").write_text("# Backend source selection.\n")
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "set(delegation DELEGATED_DIRECTORIES backend=backend_build)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
