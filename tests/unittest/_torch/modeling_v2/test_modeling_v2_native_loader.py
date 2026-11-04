# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``_native`` decides, and that the installed family libraries load.

The first tests drive the loader's decisions without opening anything: which
outcomes are ``NoTarget`` (this installation does not carry the target, so
``auto`` keeps the built-in model) and which are ``NativeLoadError`` (an
installed library is broken, which is never a fallback).

The last is the load smoke test of the installed wheel: every family library in
``tensorrt_llm/libs`` opens in a fresh process, names the targets it carries, and
defines no op the target's fake_ops.py does not fake. A library aborts its
process from a static initializer if two libraries claim one torch namespace, so
each one is opened in a child process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _native
from tensorrt_llm._torch._experimental.modeling_v2._native import (
    NativeLoadError,
    load_family_ops,
    load_target_ops,
    namespace_of,
)
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss"
_TARGET = f"{_FAMILY}.gpt_oss_120b__sm_10x__default"
_TARGET_NS = "mv2_gpt_oss_gpt_oss_120b_sm_10x_default"


@pytest.fixture
def libs(monkeypatch, tmp_path):
    """An empty libs directory, a cold library cache, and ``libs.install(opens)``,
    which puts a gpt_oss library there; ``opens`` stands in for dlopen, and no
    fake_ops module is imported, since the stand-in defines no op."""
    monkeypatch.setattr(_native, "_LIBS_DIR", tmp_path)
    monkeypatch.setattr(_native, "_built_targets", {})
    monkeypatch.setattr(_native, "_family_ops", {})

    def install(opens):
        (tmp_path / "libth_modeling_v2_gpt_oss.so").write_bytes(b"")
        monkeypatch.setattr(_native, "_open_library", lambda path, family: opens())
        monkeypatch.setattr(_native, "_import_fakes", lambda module, family: None)

    return types.SimpleNamespace(install=install)


def test_names():
    assert namespace_of(_TARGET) == _TARGET_NS
    assert namespace_of(f"{_TARGET}.ops.moe.entry") == _TARGET_NS
    assert namespace_of(f"{_FAMILY}.fake_ops") == "mv2_gpt_oss"
    with pytest.raises(ValueError):
        namespace_of("tensorrt_llm._torch.models")
    with pytest.raises(ValueError, match="models.<family>.<target>"):
        load_target_ops(f"{_TARGET}.weights")


def test_what_is_not_built_and_what_is_broken(libs):
    with pytest.raises(NoTarget, match="has no libth_modeling_v2_gpt_oss.so"):
        load_target_ops(_TARGET)

    def fail():
        raise OSError("undefined symbol: _ZN12tensorrt_llm6common")

    libs.install(fail)
    with pytest.raises(NativeLoadError, match="undefined symbol"):
        load_target_ops(_TARGET)

    libs.install(lambda: frozenset({"other__sm_90__default"}))
    with pytest.raises(NoTarget, match="does not carry gpt_oss_120b__sm_10x__default"):
        load_target_ops(_TARGET)


def test_a_library_opens_once_and_registers_the_family_fakes_first(libs, monkeypatch):
    ops = types.SimpleNamespace(mv2_gpt_oss=object(), **{_TARGET_NS: object()})
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(ops=ops))
    opened, imported = [], []

    def opens():
        opened.append(1)
        return frozenset({"gpt_oss_120b__sm_10x__default"})

    libs.install(opens)
    monkeypatch.setattr(_native, "_import_fakes", lambda module, family: imported.append(module))
    assert load_target_ops(_TARGET) is getattr(ops, _TARGET_NS)
    assert load_family_ops(f"{_TARGET}.weights") is ops.mv2_gpt_oss
    assert load_family_ops(f"{_TARGET}.weights") is ops.mv2_gpt_oss
    assert len(opened) == 1
    assert imported == [f"{_FAMILY}.fake_ops", f"{_TARGET}.fake_ops"]


_LOAD_ONE_FAMILY = textwrap.dedent(
    """
    import json, sys
    import torch
    import tensorrt_llm  # noqa: F401 -- loads libth_common.so
    from tensorrt_llm._torch._experimental.modeling_v2 import _native

    built = sorted(_native._load_family_library(sys.argv[1]))
    schemas = sorted(s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith("mv2_"))
    print(json.dumps({"built": built, "ops": schemas}))
    """
)


def test_every_installed_family_library_loads():
    libraries = sorted(_native._LIBS_DIR.glob("libth_modeling_v2_*.so"))
    if not libraries:
        pytest.skip("this installation has no modeling_v2 family library")
    models = Path(_native.__file__).resolve().parent / "models"
    for library in libraries:
        family = library.name[len("libth_modeling_v2_") : -len(".so")]
        result = subprocess.run(
            [sys.executable, "-c", _LOAD_ONE_FAMILY, family],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{library.name} failed to load:\n{result.stdout}\n{result.stderr}"
        )
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert report["built"], f"{library.name} carries no target"
        for target in report["built"]:
            assert (models / family / target / "__init__.py").is_file(), (
                f"{library.name} carries {target}, which has no Python target package"
            )
            prefix = f"{_native._namespace(family, target)}::"
            target_ops = [op[len(prefix) :] for op in report["ops"] if op.startswith(prefix)]
            fake_ops = models / family / target / "fake_ops.py"
            text = fake_ops.read_text() if fake_ops.is_file() else ""
            unfaked = [op for op in target_ops if f'::{op}"' not in text]
            assert not unfaked, f"{family}/{target} defines {unfaked} with no fake in fake_ops.py"
