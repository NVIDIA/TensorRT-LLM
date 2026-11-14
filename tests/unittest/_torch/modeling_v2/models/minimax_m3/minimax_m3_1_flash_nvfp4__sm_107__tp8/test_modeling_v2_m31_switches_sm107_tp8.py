# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The value contract of this target's kill switches (m31_switches): 1 / true / on / yes set TRTLLM_M31_DISABLE_<ID>,
empty / 0 / false / off / no leave it unset (case-insensitive), anything else raises at import naming the variable;
the environment is read once, at import. ``ITEMS`` registers every id: querying another id raises, a variable naming
another id is ignored, and every id the target's sources read is registered.

Each case loads a private copy of the module from its file, so the in-tree module (read by every other module of the
target at import) is never reloaded. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp8"
_SWITCHES = Path(_family.__file__).resolve().parent / TARGET / "m31_switches.py"

pytestmark = pytest.mark.cpu_only

PREFIX = "TRTLLM_M31_DISABLE_"
ITEM = "CTXNORMQ"  # a registered id; every other switch variable is cleared for each case


def _load(monkeypatch, env):
    """A private copy of m31_switches, imported under ``env`` (every other TRTLLM_M31_DISABLE_* variable cleared)."""
    for name in list(os.environ):
        if name.startswith(PREFIX):
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


def test_an_unregistered_id_raises_when_queried(monkeypatch):
    switches = _load(monkeypatch, {PREFIX + ITEM: "1"})
    assert switches.m31_disabled(ITEM)
    for other in (ITEM.lower(), ITEM + "_X", ITEM[:-1], ""):
        with pytest.raises(KeyError):
            switches.m31_disabled(other)


def test_a_variable_naming_another_id_is_ignored(monkeypatch):
    """Another target's switch (or a renamed id) in a shared environment neither raises nor turns anything off."""
    switches = _load(monkeypatch, {PREFIX + "NOT_A_SWITCH_OF_THIS_TARGET": "not-a-value"})
    assert not any(switches.m31_disabled(item) for item in switches.ITEMS)


def test_the_environment_is_read_once_at_import(monkeypatch):
    switches = _load(monkeypatch, {})
    monkeypatch.setenv(PREFIX + ITEM, "1")
    assert not switches.m31_disabled(ITEM)


def test_every_id_says_what_its_switch_restores(monkeypatch):
    switches = _load(monkeypatch, {})
    assert switches.ITEMS and all(
        isinstance(why, str) and why.strip() for why in switches.ITEMS.values()
    )


def _literal_reads():
    """The ids the target's sources pass to ``m31_disabled`` as string literals, with where."""
    reads = []
    for path in sorted(_SWITCHES.parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "m31_disabled"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                reads.append(
                    (node.args[0].value, f"{path.relative_to(_SWITCHES.parent)}:{node.lineno}")
                )
    return reads


def test_every_id_the_target_reads_is_registered(monkeypatch):
    """A read of an unregistered id raises at runtime; catch it here instead."""
    items = _load(monkeypatch, {}).ITEMS
    reads = _literal_reads()
    assert reads
    missing = sorted(f"{item} ({where})" for item, where in reads if item not in items)
    assert not missing, missing
