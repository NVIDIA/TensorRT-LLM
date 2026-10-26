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
"""Shipped CuTe DSL objects: when ``compile_or_load`` loads one, and when it compiles (and exports)."""

import os

import pytest

from tensorrt_llm._torch import cute_aot

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


def test_shipped_object_is_loaded_not_compiled(aot, tmp_path):
    path = _ship(tmp_path, "gemv m=8")
    got = cute_aot.compile_or_load("gemv m=8", lambda: pytest.fail("compiled"))
    assert got.tag == "loaded"
    assert aot == [(path, cute_aot.digest(cute_aot.full_key("gemv m=8")), False)]
    assert cute_aot.counts() == {"loaded": 1, "compiled": 0}


def test_unarmed_process_only_compiles(aot, tmp_path, monkeypatch):
    # A process whose model does not ship objects never looks one up or exports, even with the export directory set.
    monkeypatch.setattr(cute_aot, "_armed", False)
    monkeypatch.setenv(cute_aot.EXPORT_DIR_ENV, str(tmp_path / "export"))
    _ship(tmp_path, "gemv m=8")
    calls = []
    for _ in range(2):
        got = cute_aot.compile_or_load("gemv m=8", lambda: calls.append(1) or _Compiled("jit"))
    assert got.tag == "jit" and calls == [1]
    assert aot == [] and not (tmp_path / "export").exists()
    assert cute_aot.counts() == {"loaded": 0, "compiled": 0}


def test_missing_object_compiles_once_per_key(aot):
    calls = []
    for _ in range(2):
        got = cute_aot.compile_or_load("gemv m=16", lambda: calls.append(1) or _Compiled("jit"))
    assert got.tag == "jit" and calls == [1]
    assert cute_aot.counts() == {"loaded": 0, "compiled": 1}


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


def test_disable_always_compiles(aot, tmp_path, monkeypatch):
    _ship(tmp_path, "gemv m=8")
    monkeypatch.setenv(cute_aot.DISABLE_ENV, "1")
    assert cute_aot.compile_or_load("gemv m=8", lambda: _Compiled("jit")).tag == "jit"
    assert aot == []


@pytest.mark.parametrize("tvm_ffi", [False, True])
def test_export_writes_the_shipped_layout(aot, tmp_path, monkeypatch, tvm_ffi):
    monkeypatch.setenv(cute_aot.EXPORT_DIR_ENV, str(tmp_path / "export"))
    compiled = cute_aot.compile_or_load("moe rows=4", lambda: _Compiled("jit"), tvm_ffi=tvm_ffi)
    symbol = cute_aot.digest(cute_aot.full_key("moe rows=4"))
    out = tmp_path / "export" / TOOL
    assert (out / f"{symbol}.key").read_text() == cute_aot.full_key("moe rows=4") + "\n"
    expected = {f"{symbol}.o", f"{symbol}.key"} | (set() if tvm_ffi else {f"{symbol}.h"})
    assert set(os.listdir(out)) == expected  # the private export directory is gone
    if tvm_ffi:
        assert compiled.exports == [{"function_name": symbol, "export_only_tvm_ffi_symbols": True}]
    else:
        assert compiled.exports == [{"function_prefix": symbol}]


def test_digest_is_a_c_identifier():
    d = cute_aot.digest("any key: with spaces / slashes")
    assert d.isidentifier() and d == cute_aot.digest("any key: with spaces / slashes")


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


def test_source_digest_ignores_install_location(tmp_path):
    a, b = tmp_path / "a" / "k.py", tmp_path / "b" / "k.py"
    for f in (a, b):
        f.parent.mkdir()
        f.write_text("same")
    assert cute_aot.full_key("k", [str(a)]) == cute_aot.full_key("k", [str(b)])
    assert cute_aot.py_files(str(tmp_path / "a")) == (str(a),)


def test_model_switch_or_dsl_setting_changes_the_key(aot, tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith(cute_aot._ENV_PREFIXES)]:
        monkeypatch.delenv(k)
    _ship(tmp_path, "moe rows=4")
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", "/tmp/x")  # a directory: not a codegen setting
    assert cute_aot.compile_or_load("moe rows=4", lambda: _Compiled("jit")).tag == "loaded"
    monkeypatch.setattr(cute_aot, "_loaded", {})
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MOE_SH20", "1")
    assert cute_aot.compile_or_load("moe rows=4", lambda: _Compiled("jit")).tag == "jit"


def test_cached_compile_keys_on_bound_arguments(aot, tmp_path):
    calls = []

    @cute_aot.cached_compile("gemm", lambda: ())
    def build(dtype, tile, bias=False):
        calls.append((dtype, tile, bias))
        return _Compiled("jit")

    _ship(tmp_path, "gemm dtype='bf16' tile=(128, 64) bias=True")
    assert build("bf16", (128, 64), bias=True).tag == "loaded"
    assert build("bf16", (128, 64)).tag == "jit"
    assert build("bf16", tile=(128, 64)).tag == "jit"
    assert calls == [("bf16", (128, 64), False)]
