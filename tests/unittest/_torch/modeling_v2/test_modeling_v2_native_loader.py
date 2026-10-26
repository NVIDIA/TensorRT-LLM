# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``_native`` decides, and that the installed family libraries load.

The first half drives the loader's decisions without opening anything: which
outcomes are ``NoTarget`` (this installation does not carry the target, so
``auto`` keeps the built-in model) and which are ``NativeLoadError`` (an
installed library is broken, which is never a fallback).

The second half is the load smoke test of the installed wheel: every family
library in ``tensorrt_llm/libs`` opens in a fresh process, names the targets it
carries, and defines no op the target's fake_ops.py does not fake. A library
aborts its process from a static initializer if two libraries claim one torch
namespace, so each one is opened in a child process.
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
    family_of,
    identifier,
    library_name,
    load_family_library,
    load_family_ops,
    load_target_ops,
    namespace_of,
    split_target_package,
    torch_namespace,
)
from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

pytestmark = pytest.mark.cpu_only

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.gpt_oss_120b__sm_10x__default"
)


@pytest.fixture
def libs(monkeypatch, tmp_path):
    """An empty libs directory and a cold library cache."""
    monkeypatch.setattr(_native, "_LIBS_DIR", tmp_path)
    monkeypatch.setattr(_native, "_built_targets", {})
    monkeypatch.setattr(_native, "_family_ops", {})
    return tmp_path


def _install(libs: Path, monkeypatch, opens) -> None:
    """Put a gpt_oss library in `libs`; `opens` stands in for dlopen, and no
    fake_ops module is imported, since the stand-in defines no op."""
    (libs / library_name("gpt_oss")).write_bytes(b"")
    monkeypatch.setattr(_native, "_open_library", lambda path, family: opens())
    monkeypatch.setattr(_native, "_import_fakes", lambda module, family: None)


# --- Naming ----------------------------------------------------------------------


def test_names_of_libraries_and_namespaces():
    assert library_name("minimax_m3") == "libth_modeling_v2_minimax_m3.so"
    assert identifier("gpt_oss_120b__sm_10x__default") == "gpt_oss_120b_sm_10x_default"
    assert torch_namespace("gpt_oss") == "mv2_gpt_oss"
    assert namespace_of(_TARGET) == "mv2_gpt_oss_gpt_oss_120b_sm_10x_default"
    assert split_target_package(_TARGET) == ("gpt_oss", "gpt_oss_120b__sm_10x__default")
    with pytest.raises(ValueError):
        split_target_package("tensorrt_llm._torch.models")
    assert family_of(f"{_TARGET}.weights") == "gpt_oss"
    assert family_of(_TARGET.rpartition(".")[0] + ".fake_ops") == "gpt_oss"
    with pytest.raises(ValueError):
        family_of("tensorrt_llm._torch._experimental.modeling_v2.models")


# --- Decisions -----------------------------------------------------------------


def test_a_library_this_installation_lacks_is_no_target(libs):
    with pytest.raises(NoTarget, match="has no libth_modeling_v2_gpt_oss.so"):
        load_target_ops(_TARGET)


def test_a_library_that_fails_to_open_is_a_broken_installation(libs, monkeypatch):
    def fail():
        raise OSError("undefined symbol: _ZN12tensorrt_llm6common")

    _install(libs, monkeypatch, fail)
    with pytest.raises(NativeLoadError, match="undefined symbol"):
        load_target_ops(_TARGET)


def test_a_target_the_library_does_not_carry_is_no_target(libs, monkeypatch):
    _install(libs, monkeypatch, lambda: frozenset({"other__sm_90__default"}))
    with pytest.raises(NoTarget, match="does not carry gpt_oss_120b__sm_10x__default"):
        load_target_ops(_TARGET)


def test_a_library_opens_once(libs, monkeypatch):
    opened = []

    def opens():
        opened.append(1)
        return frozenset({"gpt_oss_120b__sm_10x__default"})

    _install(libs, monkeypatch, opens)
    assert load_family_library("gpt_oss") == {"gpt_oss_120b__sm_10x__default"}
    assert load_family_library("gpt_oss") == {"gpt_oss_120b__sm_10x__default"}
    assert len(opened) == 1


@pytest.fixture
def torch_ops(monkeypatch):
    """A stand-in torch.ops holding the gpt_oss namespaces."""
    ops = types.SimpleNamespace(
        mv2_gpt_oss=object(), mv2_gpt_oss_gpt_oss_120b_sm_10x_default=object()
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(ops=ops))
    return ops


def test_opening_a_library_registers_the_family_fakes_first(libs, monkeypatch, torch_ops):
    _install(libs, monkeypatch, lambda: frozenset({"gpt_oss_120b__sm_10x__default"}))
    imported = []
    monkeypatch.setattr(_native, "_import_fakes", lambda module, family: imported.append(module))
    assert load_target_ops(_TARGET) is torch_ops.mv2_gpt_oss_gpt_oss_120b_sm_10x_default
    family_package = _TARGET.rpartition(".")[0]
    assert imported == [f"{family_package}.fake_ops", f"{_TARGET}.fake_ops"]


def test_family_ops_are_the_family_namespace_and_load_once(libs, monkeypatch, torch_ops):
    opened = []

    def opens():
        opened.append(1)
        return frozenset({"gpt_oss_120b__sm_10x__default"})

    _install(libs, monkeypatch, opens)
    module = f"{_TARGET}.weights"
    assert load_family_ops(module) is torch_ops.mv2_gpt_oss
    assert load_family_ops(module) is torch_ops.mv2_gpt_oss
    assert len(opened) == 1


def test_family_ops_of_a_family_this_installation_lacks_are_no_target(libs):
    with pytest.raises(NoTarget, match="has no libth_modeling_v2_gpt_oss.so"):
        load_family_ops(f"{_TARGET}.weights")


def test_windows_has_no_native_targets(libs, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(NoTarget, match="Windows"):
        load_family_library("gpt_oss")


# --- The installed wheel ---------------------------------------------------------

_LOAD_ONE_FAMILY = textwrap.dedent(
    """
    import json, sys
    import torch
    import tensorrt_llm  # noqa: F401 -- loads libth_common.so
    from tensorrt_llm._torch._experimental.modeling_v2 import _native

    built = sorted(_native.load_family_library(sys.argv[1]))
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
            prefix = f"{torch_namespace(family, target)}::"
            target_ops = [op[len(prefix) :] for op in report["ops"] if op.startswith(prefix)]
            fake_ops = models / family / target / "fake_ops.py"
            text = fake_ops.read_text() if fake_ops.is_file() else ""
            unfaked = [op for op in target_ops if f'::{op}"' not in text]
            assert not unfaked, f"{family}/{target} defines {unfaked} with no fake in fake_ops.py"
