# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer NVTX ranges on the MuseSparkOpen decoder.

A flat kernel timeline cannot say which layer or which op a launch belongs to,
and this model's decode step is mostly small elementwise kernels spread across
the router, the MoE finalize and the two all-reduces. The ranges close that
gap for an eager profile.

Two properties are worth pinning, because both fail silently.

The ranges must cost nothing when profiling is off. The enable flag is
resolved once per module at construction, so `forward` enters one shared
no-op context and does no environment lookup, string formatting or allocation
per step -- 62 layers times a few ranges times every decode iteration is not a
place to do any of those.

And every op name used in a `forward` must exist in the tuple its module
declared. `LayerNvtx.__call__` indexes a dict, so a name that was renamed on
one side only raises `KeyError` -- but only when profiling is enabled, which
is exactly when someone is midway through a 30-minute capture. Nothing else in
the suite would catch it: with profiling off the lookup never happens.
"""

import ast
import inspect
import pathlib
from contextlib import nullcontext

import pytest

from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import LayerNvtx, nvtx_layer_ranges_enabled

pytestmark = pytest.mark.cpu_only

NVTX_ENV = ("TLLM_LLMAPI_ENABLE_NVTX", "TLLM_NVTX_DEBUG")


@pytest.fixture(autouse=True)
def _clear_nvtx_env(monkeypatch):
    for name in NVTX_ENV:
        monkeypatch.delenv(name, raising=False)


def test_the_ranges_are_off_by_default():
    assert nvtx_layer_ranges_enabled() is False


@pytest.mark.parametrize("var", NVTX_ENV)
def test_either_variable_turns_them_on(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    assert nvtx_layer_ranges_enabled() is True


def test_off_costs_one_shared_no_op_context():
    """No names built, and the same context object every call."""
    layer = LayerNvtx("moe", 7, ("router", "finalize"))
    assert layer.enabled is False
    assert layer.names == {}
    assert isinstance(layer("router"), nullcontext)
    # The same object for every op and every call: nothing is allocated per step.
    assert layer("router") is layer("finalize")


def test_on_builds_zero_padded_per_layer_names(monkeypatch):
    monkeypatch.setenv("TLLM_NVTX_DEBUG", "1")
    layer = LayerNvtx("moe", 7, ("router", "finalize"))
    assert layer.enabled is True
    assert layer.names == {"router": "moe.L007.router", "finalize": "moe.L007.finalize"}


def _nvtx_names_by_class() -> dict:
    """Per class: the op names declared in its `LayerNvtx`, and those used.

    Read from the file rather than by constructing the modules, which would
    need a full model config and a mapping with a real TP group.
    """
    path = pathlib.Path(inspect.getsourcefile(modeling_musespark_open))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        declared, used = set(), set()
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "LayerNvtx"
            ):
                declared |= {
                    e.value for e in node.value.args[2].elts if isinstance(e, ast.Constant)
                }
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "nvtx"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                used.add(node.args[0].value)
        if declared or used:
            out[cls.name] = (declared, used)
    return out


def test_both_annotated_modules_are_found():
    """If this drops to one, a module lost its ranges and the checks below go vacuous."""
    assert set(_nvtx_names_by_class()) == {"MuseSparkOpenMoE", "MuseSparkOpenDecoderLayer"}


@pytest.mark.parametrize("cls_name", ["MuseSparkOpenMoE", "MuseSparkOpenDecoderLayer"])
def test_every_op_used_in_forward_was_declared(cls_name):
    declared, used = _nvtx_names_by_class()[cls_name]
    assert used, f"{cls_name} declares ranges but never enters one"
    assert not used - declared, (
        f"{cls_name} enters NVTX range(s) {sorted(used - declared)} that its LayerNvtx "
        f"tuple does not declare; this raises KeyError only when profiling is enabled"
    )


@pytest.mark.parametrize("cls_name", ["MuseSparkOpenMoE", "MuseSparkOpenDecoderLayer"])
def test_no_declared_op_goes_unused(cls_name):
    """A leftover name is a range someone expects in the trace and will not find."""
    declared, used = _nvtx_names_by_class()[cls_name]
    assert not declared - used, (
        f"{cls_name} declares {sorted(declared - used)} but never enters them"
    )
