# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The contract of this target's kill switches and opt-ins (m31_switches).

1 / true / on / yes set TRTLLM_M31_DISABLE_<ID>, empty / 0 / false / off / no leave it unset (case-insensitive), and
anything else raises at import naming the variable; the environment is read once, at import. ``ITEMS`` registers every
id the target reads: querying another id raises, and a TRTLLM_M31_DISABLE_* variable naming an unregistered id is
ignored. The default-off items (``OPT_INS``, TRTLLM_M31_ENABLE_<ID>, ``m31_enabled``) follow the same contract.

Each case loads a private copy of the module from its file, so the in-tree module (read by every other module of the
target at import) is never reloaded. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp2"
_TARGET_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _TARGET_DIR / "m31_switches.py"
PREFIX = "TRTLLM_M31_DISABLE_"
ITEM = "TP2WIDE_LC2"  # a registered id
ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"
OPT_IN = "R6_DENSE_PREQUANT"  # a registered opt-in

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


@pytest.mark.parametrize(
    "value", ["", "  ", "0", " 0 ", "false", "FALSE", "off", " Off ", "no", "No"]
)
def test_unset_spellings_leave_the_switch_unset(monkeypatch, value):
    assert not _load(monkeypatch, {}).m31_disabled(ITEM)
    assert not _load(monkeypatch, {PREFIX + ITEM: value}).m31_disabled(ITEM)


@pytest.mark.parametrize("value", ["1", " 1 ", "true", "TRUE", "on", "On", "yes", "YES"])
def test_set_spellings_set_the_switch(monkeypatch, value):
    assert _load(monkeypatch, {PREFIX + ITEM: value}).m31_disabled(ITEM)


@pytest.mark.parametrize("value", ["2", "-1", "1.0", "y", "n", "enable", "disabled", "truee", "of"])
def test_any_other_value_fails_at_import_naming_the_variable(monkeypatch, value):
    """Typos must fail loudly: a loose parse would count any value but "" / "0" as set (DISABLE_X=false disabling X)."""
    with pytest.raises(ValueError, match=PREFIX + ITEM):
        _load(monkeypatch, {PREFIX + ITEM: value})


def test_an_unregistered_id_raises(monkeypatch):
    switches = _load(monkeypatch, {PREFIX + ITEM: "1"})
    assert switches.m31_disabled(ITEM)
    for other in (ITEM.lower(), ITEM + "_X", ITEM[:-1], ""):
        with pytest.raises(KeyError):
            switches.m31_disabled(other)


def test_an_unregistered_variable_is_ignored(monkeypatch):
    """Another target's switch in a shared environment (or an old name) is not read, whatever its value."""
    switches = _load(monkeypatch, {PREFIX + "NOT_A_SWITCH_OF_THIS_TARGET": "maybe"})
    assert switches._DISABLED == frozenset()


def test_the_environment_is_read_once_at_import(monkeypatch):
    switches = _load(monkeypatch, {})
    monkeypatch.setenv(PREFIX + ITEM, "1")
    assert not switches.m31_disabled(ITEM)


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


@pytest.mark.parametrize("value", ["1", " 1 ", "true", "TRUE", "on", "On", "yes", "YES"])
def test_set_spellings_turn_an_opt_in_on(monkeypatch, value):
    switches = _load(monkeypatch, {ENABLE_PREFIX + OPT_IN: value})
    assert switches.m31_enabled(OPT_IN)
    assert switches._ENABLED == frozenset({OPT_IN})


@pytest.mark.parametrize("value", ["", "  ", "0", "false", "off", "no"])
def test_unset_spellings_leave_an_opt_in_off(monkeypatch, value):
    assert not _load(monkeypatch, {ENABLE_PREFIX + OPT_IN: value}).m31_enabled(OPT_IN)


@pytest.mark.parametrize("value", ["2", "-1", "y", "enable", "truee"])
def test_an_opt_in_fails_at_import_on_any_other_value(monkeypatch, value):
    with pytest.raises(ValueError, match=ENABLE_PREFIX + OPT_IN):
        _load(monkeypatch, {ENABLE_PREFIX + OPT_IN: value})


def test_an_unregistered_opt_in_raises(monkeypatch):
    switches = _load(monkeypatch, {ENABLE_PREFIX + OPT_IN: "1"})
    for other in (OPT_IN.lower(), OPT_IN + "_X", ITEM, ""):
        with pytest.raises(KeyError):
            switches.m31_enabled(other)


def test_an_unregistered_opt_in_variable_is_ignored(monkeypatch):
    switches = _load(monkeypatch, {ENABLE_PREFIX + "NOT_AN_OPT_IN_OF_THIS_TARGET": "maybe"})
    assert switches._ENABLED == frozenset()


def test_an_opt_in_is_read_once_at_import(monkeypatch):
    switches = _load(monkeypatch, {})
    monkeypatch.setenv(ENABLE_PREFIX + OPT_IN, "1")
    assert not switches.m31_enabled(OPT_IN)
