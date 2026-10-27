# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BLOCKOFF_GUARD on this target: ``_runtime_hooks`` turns on RuntimeHooks.skip_unchanged_draft_block_offsets by default
and ``TRTLLM_M31_DISABLE_BLOCKOFF_GUARD`` turns it off. The constant is evaluated from modeling.py's source against a
private copy of m31_switches loaded under each environment, so the heavy modeling module is not reloaded. CPU only,
except the last test, which imports the in-tree module.
"""

import ast
import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m31_nvfp4__sm_107__tp2"
CONSTANT = "STEP_BLOCKOFF_GUARD"
FIELD = "skip_unchanged_draft_block_offsets"
_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _DIR / "m31_switches.py"
_MODELING = _DIR / "modeling.py"


def _module_tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _value(monkeypatch, env) -> bool:
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_blockoff_{id(env)}", _SWITCHES)
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    statements = [
        node
        for node in _module_tree().body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == CONSTANT
    ]
    assert len(statements) == 1
    namespace = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(compile(ast.Module(statements, []), str(_MODELING), "exec"), namespace)
    return namespace[CONSTANT]


@pytest.mark.cpu_only
def test_on_by_default(monkeypatch):
    assert _value(monkeypatch, {}) is True


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["1", "true", "ON", " yes "])
def test_the_kill_switch_turns_it_off(monkeypatch, value):
    assert _value(monkeypatch, {"TRTLLM_M31_DISABLE_BLOCKOFF_GUARD": value}) is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_an_unset_kill_switch_keeps_it(monkeypatch, value):
    assert _value(monkeypatch, {"TRTLLM_M31_DISABLE_BLOCKOFF_GUARD": value}) is True


@pytest.mark.cpu_only
def test_other_switches_leave_it_on(monkeypatch):
    assert _value(monkeypatch, {"TRTLLM_M31_DISABLE_HOSTTAIL": "1"}) is True


@pytest.mark.cpu_only
def test_runtime_hooks_passes_the_constant():
    fn = next(
        node
        for node in _module_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "_runtime_hooks"
    )
    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeHooks"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(keywords.get(FIELD), ast.Name) and keywords[FIELD].id == CONSTANT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="imports the target's modeling module")
def test_the_in_tree_hook_follows_the_switch():
    modeling = importlib.import_module(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{TARGET}.modeling"
    )
    hooks = modeling._runtime_hooks(None)
    disabled = os.environ.get("TRTLLM_M31_DISABLE_BLOCKOFF_GUARD", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )
    assert getattr(hooks, FIELD) is getattr(modeling, CONSTANT) is (not disabled)
    assert hooks.skips_unchanged_metadata_uploads


def _patch_value(monkeypatch, env) -> bool:
    """STEP_BLOCKOFF_PATCH as modeling.py computes it under ``env`` (it needs BLOCKOFF_GUARD)."""
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        f"_m31_switches_blockoff_patch_{id(env)}", _SWITCHES
    )
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    statements = [
        node
        for node in _module_tree().body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in (CONSTANT, "STEP_BLOCKOFF_PATCH")
    ]
    assert [node.targets[0].id for node in statements] == [CONSTANT, "STEP_BLOCKOFF_PATCH"]
    namespace = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(compile(ast.Module(statements, []), str(_MODELING), "exec"), namespace)
    return namespace["STEP_BLOCKOFF_PATCH"]


@pytest.mark.cpu_only
def test_the_patch_is_on_by_default(monkeypatch):
    assert _patch_value(monkeypatch, {}) is True


@pytest.mark.cpu_only
@pytest.mark.parametrize("item", ["BLOCKOFF_PATCH", "BLOCKOFF_GUARD"])
def test_the_patch_kill_switch_and_the_guard_kill_switch_turn_it_off(monkeypatch, item):
    assert _patch_value(monkeypatch, {"TRTLLM_M31_DISABLE_" + item: "1"}) is False


@pytest.mark.cpu_only
def test_runtime_hooks_passes_the_patch_constant():
    fn = next(
        node
        for node in _module_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "_runtime_hooks"
    )
    call = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeHooks"
    )
    keywords = {kw.arg: kw.value for kw in call.keywords}
    field = keywords.get("patch_unchanged_draft_block_offsets")
    assert isinstance(field, ast.Name) and field.id == "STEP_BLOCKOFF_PATCH"
