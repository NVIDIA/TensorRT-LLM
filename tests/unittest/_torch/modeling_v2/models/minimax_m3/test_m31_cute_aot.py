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
"""The family's CuTe DSL object cache (``cute_aot``): when ``compile_or_load`` loads an exported object, and when it
compiles (and exports). Objects come only from the directory ``TRTLLM_M31_CUTE_AOT_DIR`` names; none ship with the
package."""

import os

import pytest

from tensorrt_llm._torch.modeling_v2.models.minimax_m3 import cute_aot

pytestmark = pytest.mark.cpu_only

TOOL = "cutedsl-4.8.0.dev0-sm107"


class _Compiled:
    def __init__(self, tag):
        self.tag = tag
        self.exports = []

    def export_to_c(self, path, name=None, **kwargs):
        # Writes what CuTe DSL writes: <dir>/<name>.o + .h (plain), or the given object file (TVM-FFI).
        self.exports.append(kwargs)
        if path.endswith(".o"):
            open(path, "wb").write(b"obj")
        else:
            for ext in (".o", ".h"):
                open(os.path.join(path, name + ext), "wb").write(b"obj")


@pytest.fixture
def aot(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(cute_aot, "toolchain", lambda: TOOL)
    monkeypatch.setattr(cute_aot, "_loaded", {})
    monkeypatch.setattr(cute_aot, "_counts", {"loaded": 0, "compiled": 0})

    def load(path, symbol, tvm_ffi):
        loads.append((path, symbol, tvm_ffi))
        if path.read_bytes() == b"stale":
            raise RuntimeError("version mismatch")
        return _Compiled("loaded")

    monkeypatch.setattr(cute_aot, "_load", load)
    monkeypatch.setattr(cute_aot, "_armed", True)
    monkeypatch.setenv(cute_aot.DIR_ENV, str(tmp_path / "data"))
    for env in (cute_aot.EXPORT_DIR_ENV, cute_aot.DISABLE_ENV):
        monkeypatch.delenv(env, raising=False)
    return loads


def _ship(tmp_path, key, content=b"obj", sources=()):
    path = tmp_path / "data" / TOOL / f"{cute_aot.digest(cute_aot.full_key(key, sources))}.o"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_other_key_or_toolchain_is_not_served(aot, tmp_path, monkeypatch):
    _ship(tmp_path, "gemv m=8")
    assert cute_aot.compile_or_load("gemv m=32", lambda: _Compiled("jit")).tag == "jit"
    monkeypatch.setattr(cute_aot, "toolchain", lambda: "cutedsl-4.9.0-sm107")
    assert cute_aot.compile_or_load("gemv m=8", lambda: _Compiled("jit")).tag == "jit"
    assert aot == []


def test_stale_object_falls_back_to_compiling(aot, tmp_path):
    _ship(tmp_path, "gemv m=8", b"stale")
    assert cute_aot.compile_or_load("gemv m=8", lambda: _Compiled("jit")).tag == "jit"
    assert cute_aot.counts() == {"loaded": 0, "compiled": 1}


def test_edited_source_is_not_served(aot, tmp_path):
    src = tmp_path / "kernel.py"
    src.write_text("v1")
    _ship(tmp_path, "gemv m=8", sources=[str(src)])
    assert (
        cute_aot.compile_or_load("gemv m=8", lambda: _Compiled("jit"), sources=[str(src)]).tag
        == "loaded"
    )
    edited = tmp_path / "edited" / "kernel.py"
    edited.parent.mkdir()
    edited.write_text("v2")
    assert (
        cute_aot.compile_or_load("gemv m=8", lambda: _Compiled("jit"), sources=[str(edited)]).tag
        == "jit"
    )
