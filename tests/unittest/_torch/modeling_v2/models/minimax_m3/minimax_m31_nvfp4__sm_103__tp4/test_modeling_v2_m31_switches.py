# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The value contract of this target's kill switches (m31_switches): 1 / true / on / yes set TRTLLM_M31_DISABLE_<ID> /
TRTLLM_M31_ENABLE_<ID>, empty / 0 / false / off / no leave it unset (case-insensitive), anything else raises at import
naming the variable; the environment is read once, at import.

Each case loads a private copy of the module from its file, so the in-tree module (read by every other module of the
target at import) is never reloaded. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m31_nvfp4__sm_103__tp4"
_SWITCHES = Path(_family.__file__).resolve().parent / TARGET / "m31_switches.py"

pytestmark = pytest.mark.cpu_only

KINDS = [("TRTLLM_M31_DISABLE_", "m31_disabled"), ("TRTLLM_M31_ENABLE_", "m31_enabled")]
ITEM = "UTCONTRACT"  # an id no item uses, so the real switches of the test process cannot interfere


def _load(monkeypatch, env):
    """A private copy of m31_switches, imported under ``env`` (every other TRTLLM_M31_* variable cleared)."""
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_contract_{id(env)}", _SWITCHES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("prefix,query", KINDS)
@pytest.mark.parametrize(
    "value", ["", "  ", "0", " 0 ", "false", "FALSE", "off", " Off ", "no", "No"]
)
def test_unset_spellings_leave_the_switch_unset(monkeypatch, prefix, query, value):
    assert not getattr(_load(monkeypatch, {}), query)(ITEM)
    assert not getattr(_load(monkeypatch, {prefix + ITEM: value}), query)(ITEM)


@pytest.mark.parametrize("prefix,query", KINDS)
@pytest.mark.parametrize("value", ["1", " 1 ", "true", "TRUE", "on", "On", "yes", "YES"])
def test_set_spellings_set_the_switch(monkeypatch, prefix, query, value):
    assert getattr(_load(monkeypatch, {prefix + ITEM: value}), query)(ITEM)


@pytest.mark.parametrize("prefix", [k for k, _ in KINDS])
@pytest.mark.parametrize("value", ["2", "-1", "1.0", "y", "n", "enable", "disabled", "truee", "of"])
def test_any_other_value_fails_at_import_naming_the_variable(monkeypatch, prefix, value):
    """Typos must fail loudly: invalid switch values must not silently disable a feature."""
    with pytest.raises(ValueError, match=prefix + ITEM):
        _load(monkeypatch, {prefix + ITEM: value})


def test_the_id_is_matched_exactly(monkeypatch):
    switches = _load(monkeypatch, {"TRTLLM_M31_DISABLE_" + ITEM: "1"})
    assert switches.m31_disabled(ITEM)
    for other in (ITEM.lower(), ITEM + "_X", ITEM[:-1], ""):
        assert not switches.m31_disabled(other), other


def test_disable_and_enable_are_independent(monkeypatch):
    """Both read: an item's own code applies "DISABLE wins" (e.g. B1, L-DS1); the module reports each separately."""
    switches = _load(
        monkeypatch, {"TRTLLM_M31_DISABLE_" + ITEM: "1", "TRTLLM_M31_ENABLE_" + ITEM: "1"}
    )
    assert switches.m31_disabled(ITEM) and switches.m31_enabled(ITEM)


def test_the_environment_is_read_once_at_import(monkeypatch):
    switches = _load(monkeypatch, {})
    monkeypatch.setenv("TRTLLM_M31_DISABLE_" + ITEM, "1")
    monkeypatch.setenv("TRTLLM_M31_ENABLE_" + ITEM, "1")
    assert not switches.m31_disabled(ITEM) and not switches.m31_enabled(ITEM)


# The step-input pack items (VR2GB-01 / VR2GB-03) and the order each one needs: a killed prerequisite turns off
# every item that needs it.
_PACK_FLAGS = ("PACK_PSCATTER", "PACK_IN_GRAPH", "PACK_EARLY_COPY", "H2D_CHANGED_SPAN")
_PACK_HOOKS = {
    "pack_parallel_scatter": "PACK_PSCATTER",
    "pack_in_graph": "PACK_IN_GRAPH",
    "pack_early_copy": "PACK_EARLY_COPY",
    "h2d_changed_span": "H2D_CHANGED_SPAN",
}


def _modeling_tree():
    return ast.parse((_SWITCHES.parent / "modeling.py").read_text())


def _pack_flags(disabled):
    """The target's pack flags as its module computes them at import, with ``disabled`` the switched-off ids (the
    flags' own statements, evaluated with a stand-in m31_disabled: no import of the target)."""
    statements = [
        node
        for node in _modeling_tree().body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in _PACK_FLAGS
    ]
    assert sorted(node.targets[0].id for node in statements) == sorted(_PACK_FLAGS)
    namespace = {"m31_disabled": lambda item_id: item_id in disabled}
    exec(compile(ast.Module(body=statements, type_ignores=[]), "<pack flags>", "exec"), namespace)
    return {name: namespace[name] for name in _PACK_FLAGS}


@pytest.mark.parametrize(
    "disabled,expect_on",
    [
        ((), _PACK_FLAGS),
        (("H2D_CHANGED_SPAN",), ("PACK_PSCATTER", "PACK_IN_GRAPH", "PACK_EARLY_COPY")),
        (("PACK_EARLY_COPY",), ("PACK_PSCATTER", "PACK_IN_GRAPH", "H2D_CHANGED_SPAN")),
        (("PACK_IN_GRAPH",), ("PACK_PSCATTER", "H2D_CHANGED_SPAN")),
        (("PACK_PSCATTER",), ("H2D_CHANGED_SPAN",)),
        (("PACK_PSCATTER", "H2D_CHANGED_SPAN"), ()),
    ],
)
def test_pack_items_default_on_and_a_killed_prerequisite_turns_its_dependents_off(
    disabled, expect_on
):
    flags = _pack_flags(set(disabled))
    assert {name for name, on in flags.items() if on} == set(expect_on)


def test_runtime_hooks_take_the_pack_flags():
    """_runtime_hooks passes each pack flag to its RuntimeHooks field (not a literal True)."""
    (function,) = [
        node
        for node in _modeling_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "_runtime_hooks"
    ]
    (call,) = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RuntimeHooks"
    ]
    passed = {kw.arg: kw.value for kw in call.keywords}
    for field, flag in _PACK_HOOKS.items():
        assert isinstance(passed.get(field), ast.Name) and passed[field].id == flag, field
    assert (
        isinstance(passed["skip_unchanged_h2d"], ast.Constant)
        and passed["skip_unchanged_h2d"].value
    )
    assert isinstance(passed["pack_step_inputs"], ast.Constant) and passed["pack_step_inputs"].value
