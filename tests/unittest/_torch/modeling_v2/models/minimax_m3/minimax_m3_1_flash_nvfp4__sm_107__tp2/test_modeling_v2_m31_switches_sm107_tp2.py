# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's kill-switch and opt-in registry (m31_switches): every id the target's modules read is registered in
``ITEMS`` / ``OPT_INS``, the two sets are disjoint, and every opt-in is off by default. The parser's spelling contract
is covered by the sm_107 TP8 target's copy of this test.

Each case loads a private copy of the module from its file, so the in-tree module (read by every other module of the
target at import) is never reloaded. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp2"
_TARGET_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _TARGET_DIR / "m31_switches.py"
PREFIX = "TRTLLM_M31_DISABLE_"
ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"

pytestmark = pytest.mark.cpu_only


def _load(monkeypatch, env):
    """A private copy of m31_switches, imported under ``env`` (every other TRTLLM_M31_DISABLE_* / TRTLLM_M31_ENABLE_*
    variable cleared)."""
    for name in list(os.environ):
        if name.startswith((PREFIX, ENABLE_PREFIX)):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_contract_{id(env)}", _SWITCHES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_ids():
    """Every switch id the target's modules pass as a literal: to m31_disabled, and to gen_glue's glue_disabled (its
    part names map to ids through gen_glue._PART_SWITCH)."""
    ids = set()
    for path in _TARGET_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if name == "m31_disabled":
                ids.update(args)
            elif name == "glue_disabled":
                ids.update(args[1:])
    return ids


def test_every_id_the_target_reads_is_registered(monkeypatch):
    switches = _load(monkeypatch, {})
    ids = _read_ids()
    assert ids, "no switch read found"
    assert not ids - set(switches.ITEMS), sorted(ids - set(switches.ITEMS))


def _enabled_ids():
    """Every opt-in id the target's modules pass as a literal to m31_enabled."""
    ids = set()
    for path in _TARGET_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(
                node.func, "id", getattr(node.func, "attr", None)
            ) == ("m31_enabled"):
                ids.update(a.value for a in node.args if isinstance(a, ast.Constant))
    return ids


def test_every_opt_in_the_target_reads_is_registered(monkeypatch):
    switches = _load(monkeypatch, {})
    ids = _enabled_ids()
    assert ids, "no opt-in read found"
    assert not ids - set(switches.OPT_INS), sorted(ids - set(switches.OPT_INS))
    assert not set(switches.OPT_INS) - ids, sorted(set(switches.OPT_INS) - ids)


def test_opt_ins_and_kill_switches_are_disjoint(monkeypatch):
    switches = _load(monkeypatch, {})
    assert not set(switches.OPT_INS) & set(switches.ITEMS)


def test_every_opt_in_is_off_by_default(monkeypatch):
    switches = _load(monkeypatch, {})
    assert switches._ENABLED == frozenset()
    assert not any(switches.m31_enabled(item_id) for item_id in switches.OPT_INS)
