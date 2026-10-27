# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S4-logits on the sm_107 TP2 target: on by default (S4STREAM, the TP8 recipe with MCAG off);
``TRTLLM_M31_DISABLE_S4_LOGITS=1`` turns it off (also over ``TRTLLM_M31_ENABLE_S4_LOGITS=1``). The assignment is
evaluated from dspark_s4.py's source against a private copy of m31_switches. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

_DIR = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2"
_SWITCHES = _DIR / "m31_switches.py"
_S4 = _DIR / "dspark_s4.py"


class _Worker:
    pass


def _value(monkeypatch, env) -> bool:
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_s4_logits_{id(env)}", _SWITCHES)
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    tree = ast.parse(_S4.read_text(), filename=str(_S4))
    statements = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_s4_logits_on"
    ]
    assert len(statements) == 1
    worker = _Worker()
    namespace = {
        "m31_disabled": switches.m31_disabled,
        "m31_enabled": switches.m31_enabled,
        "self": worker,
    }
    exec(compile(ast.Module(statements, []), str(_S4), "exec"), namespace)
    return worker._s4_logits_on


@pytest.mark.cpu_only
def test_on_by_default(monkeypatch):
    assert _value(monkeypatch, {}) is True


@pytest.mark.cpu_only
def test_the_kill_switch_turns_it_off(monkeypatch):
    assert _value(monkeypatch, {"TRTLLM_M31_DISABLE_S4_LOGITS": "1"}) is False


@pytest.mark.cpu_only
def test_the_kill_switch_wins(monkeypatch):
    env = {"TRTLLM_M31_ENABLE_S4_LOGITS": "1", "TRTLLM_M31_DISABLE_S4_LOGITS": "1"}
    assert _value(monkeypatch, env) is False
