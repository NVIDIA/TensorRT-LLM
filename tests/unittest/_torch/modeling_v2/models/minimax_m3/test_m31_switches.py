# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1-Flash targets' kill-switch and opt-in registries (each target's ``m31_switches``).

- The value contract: 1 / true / on / yes set ``TRTLLM_M31_DISABLE_<ID>``, empty / 0 / false / off / no leave it unset
  (case-insensitive, surrounding blanks ignored), anything else raises at import naming the variable.
- Every id a target's modules read (``m31_disabled`` and the helpers built on it, ``m31_enabled``) is registered, and
  every registered opt-in is read.
- Every opt-in is off by default. On the targets whose registry keeps the two sets apart, no id is both a kill switch
  and an opt-in (the TP4 and TP8 registries let an opt-in carry a kill switch that wins over it).

Each case loads a private copy of the module from its file, so the in-tree module (read by every other module of the
target at import) is never reloaded. CPU only.
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch.modeling_v2.models.minimax_m3 as _family

pytestmark = pytest.mark.cpu_only

_FAMILY_DIR = Path(_family.__file__).resolve().parent
TARGETS = tuple(
    sorted(p.parent.name for p in _FAMILY_DIR.glob("minimax_m3_1_flash_nvfp4__*/m31_switches.py"))
)
# Registries that document an opt-in id may also be a kill switch (TRTLLM_M31_DISABLE_<ID>=1 still wins).
_OVERLAP_ALLOWED = frozenset(
    ("minimax_m3_1_flash_nvfp4__sm_107__tp4", "minimax_m3_1_flash_nvfp4__sm_107__tp8")
)
PREFIX = "TRTLLM_M31_DISABLE_"
ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"

# Reader -> index of its first switch-id argument: m31_disabled / m31_enabled take the id first; tp2's gen_glue
# glue_disabled(part, *legacy_ids) takes a part name first; dep2ctx's large_prefill_graphs_disabled takes the id.
_DISABLE_READERS = {"m31_disabled": 0, "glue_disabled": 1, "large_prefill_graphs_disabled": 0}


def test_every_target_with_a_registry_is_covered():
    assert len(TARGETS) >= 6, TARGETS


def _load(monkeypatch, target, env):
    """A private copy of ``target``'s m31_switches, imported under ``env`` (every other TRTLLM_M31_DISABLE_* /
    TRTLLM_M31_ENABLE_* variable cleared)."""
    for name in list(os.environ):
        if name.startswith((PREFIX, ENABLE_PREFIX)):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    path = _FAMILY_DIR / target / "m31_switches.py"
    spec = importlib.util.spec_from_file_location(
        f"_m31_switches_contract_{target}_{len(env)}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kill_switches(module):
    """The kill-switch registry (``ITEMS`` on some targets, ``SWITCHES`` on others)."""
    registry = getattr(module, "ITEMS", None)
    return registry if registry is not None else module.SWITCHES


def _literal_ids(target, readers):
    """The string-literal switch ids ``target``'s modules pass to the ``readers`` (name -> first id argument)."""
    ids = set()
    for path in sorted((_FAMILY_DIR / target).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in readers:
                continue
            ids.update(
                a.value
                for a in node.args[readers[name] :]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            )
    return ids


@pytest.mark.parametrize("target", TARGETS)
def test_spelling_contract(monkeypatch, target):
    """Typos must fail loudly: a loose parse would count any value but "" / "0" as set (DISABLE_X=false disabling X)."""
    item = sorted(_kill_switches(_load(monkeypatch, target, {})))[0]
    variable = PREFIX + item
    assert not _load(monkeypatch, target, {}).m31_disabled(item)
    for value in ("", "0", " Off ", "no"):
        assert not _load(monkeypatch, target, {variable: value}).m31_disabled(item), value
    for value in ("1", " On ", "TRUE", "yes"):
        assert _load(monkeypatch, target, {variable: value}).m31_disabled(item), value
    for value in ("2", "truee"):
        with pytest.raises(ValueError, match=variable):
            _load(monkeypatch, target, {variable: value})


@pytest.mark.parametrize("target", TARGETS)
def test_every_id_the_target_reads_is_registered(monkeypatch, target):
    """A read of an unregistered id raises at runtime; catch it here instead."""
    switches = _load(monkeypatch, target, {})
    disabled = _literal_ids(target, _DISABLE_READERS)
    assert disabled, "no switch read found"
    assert not disabled - set(_kill_switches(switches)), sorted(
        disabled - set(_kill_switches(switches))
    )
    enabled = _literal_ids(target, {"m31_enabled": 0})
    opt_ins = set(switches.OPT_INS)
    assert not enabled - opt_ins, sorted(enabled - opt_ins)
    assert not opt_ins - enabled, f"registered opt-ins nothing reads: {sorted(opt_ins - enabled)}"


@pytest.mark.parametrize("target", TARGETS)
def test_every_opt_in_is_off_by_default(monkeypatch, target):
    switches = _load(monkeypatch, target, {})
    assert switches._ENABLED == frozenset()
    assert not any(switches.m31_enabled(item_id) for item_id in switches.OPT_INS)
    if target not in _OVERLAP_ALLOWED:
        overlap = set(switches.OPT_INS) & set(_kill_switches(switches))
        assert not overlap, sorted(overlap)
