# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_CPP = Path(__file__).resolve().parents[4] / "cpp"
_BINDING = """\
#include <cstdlib>
#include <fstream>
#include "tensorrt_llm/visual_gen/common/namespace.h"
#define STRING_IMPL(x) #x
#define STRING(x) STRING_IMPL(x)
TRTLLM_VG_NAMESPACE_BEGIN
struct Binding {
    Binding() {
        std::ofstream(std::getenv("VG_TEST_REGISTRATIONS"), std::ios::app)
            << STRING(TRTLLM_VG_TORCH_NS) << '\\n';
    }
};
Binding registration;
TRTLLM_VG_NAMESPACE_END
"""


def _project(root):
    cpp = root / "cpp"
    shared = Path("tensorrt_llm/modeling_v2/common/familyLibrary.version")
    (cpp / shared).parent.mkdir(parents=True)
    shutil.copyfile(_CPP / shared, cpp / shared)
    (cpp / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(native_scope_test LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 17)\n"
        f'list(APPEND CMAKE_MODULE_PATH "{_CPP.as_posix()}/cmake/modules")\n'
        f'include_directories("{_CPP.as_posix()}" "{_CPP.as_posix()}/include")\n'
        f'add_subdirectory("{_CPP.as_posix()}/tensorrt_llm/visual_gen" visual_gen)\n'
    )
    return ["cmake", "-S", str(cpp), "-B", str(root / "build")]


def _declare(root, scope, arch="100f"):
    source = root / "tensorrt_llm/_torch/visual_gen/modeling_v2" / scope / "csrc"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        f"add_visual_gen_native(ARCHS {arch} SOURCES binding.cpp)\n"
    )
    (source / "binding.cpp").write_text(_BINDING)


def test_native_objects_load_with_their_owner(tmp_path, monkeypatch):
    configure = _project(tmp_path)
    scopes = {
        "catalog/activation": ("vg_catalog", "catalog"),
        "catalog/second": ("vg_catalog", "catalog"),
        "models/alpha": ("vg_alpha", "alpha"),
        "models/alpha/fast__path": ("vg_alpha_fast_path", "alpha"),
        "models/alpha/default": ("vg_alpha_default", "alpha"),
        "models/beta/only": ("vg_beta_only", "beta"),
    }
    for scope in scopes:
        _declare(tmp_path, scope)
    _declare(tmp_path, "models/beta/other_arch", arch="90a")
    legacy = tmp_path / "tensorrt_llm/_torch/visual_gen/models/alpha/csrc"
    legacy.mkdir(parents=True)
    (legacy / "CMakeLists.txt").write_text('message(FATAL_ERROR "Discovered V1 model")\n')
    subprocess.run([*configure, "-DCMAKE_CUDA_ARCHITECTURES_ORIG=100"], check=True)
    subprocess.run(
        ["cmake", "--build", str(tmp_path / "build"), "--target", "visual_gen_libs"], check=True
    )

    output = tmp_path / "build/visual_gen"
    manifest = output / "visual_gen_native.json"
    entries = json.loads(manifest.read_text())
    assert set(entries) == set(scopes)
    expected_registrations = {}
    for scope, (namespace, owner) in scopes.items():
        library = entries[scope]["library"]
        assert entries[scope]["namespace"] == namespace
        assert Path(library).stem == f"libth_visual_gen_{owner}"
        expected_registrations.setdefault(library, []).append(namespace)

    registrations = tmp_path / "registrations"
    monkeypatch.setenv("VG_TEST_REGISTRATIONS", str(registrations))
    for library, expected in expected_registrations.items():
        registrations.write_text("")
        ctypes.CDLL(str(output / library))
        assert sorted(registrations.read_text().splitlines()) == sorted(expected)

    subprocess.run([*configure, "-DCMAKE_CUDA_ARCHITECTURES_ORIG=90"], check=True)
    assert set(json.loads(manifest.read_text())) == {"models/beta/other_arch"}


def test_colliding_native_namespaces_fail_at_configure(tmp_path):
    configure = _project(tmp_path)
    _declare(tmp_path, "models/alpha/fast__path")
    _declare(tmp_path, "models/alpha/fast_path")
    result = subprocess.run(
        [*configure, "-DCMAKE_CUDA_ARCHITECTURES_ORIG=100"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "vg_alpha_fast_path" in result.stderr
