# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SLOT_IDS_CHANGED on this target: off by default, ``TRTLLM_M31_ENABLE_SLOT_IDS_CHANGED`` turns it on and
``TRTLLM_M31_DISABLE_SLOT_IDS_CHANGED`` still wins; ``_runtime_hooks`` passes the constant as
RuntimeHooks.spec_slot_ids_changed_only. The constant is evaluated from modeling.py's source against a private copy of
m31_switches loaded under each environment, so the heavy modeling module is not reloaded. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m31_nvfp4__sm_107__tp8"
CONSTANT = "SLOT_IDS_CHANGED"
FIELD = "spec_slot_ids_changed_only"
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
    spec = importlib.util.spec_from_file_location(f"_m31_switches_slot_ids_{id(env)}", _SWITCHES)
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
def test_off_by_default(monkeypatch):
    assert _value(monkeypatch, {}) is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["1", "true", "ON", " yes "])
def test_the_opt_in_turns_it_on(monkeypatch, value):
    assert _value(monkeypatch, {"TRTLLM_M31_ENABLE_SLOT_IDS_CHANGED": value}) is True


@pytest.mark.cpu_only
def test_the_kill_switch_wins(monkeypatch):
    env = {"TRTLLM_M31_ENABLE_SLOT_IDS_CHANGED": "1", "TRTLLM_M31_DISABLE_SLOT_IDS_CHANGED": "1"}
    assert _value(monkeypatch, env) is False


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
