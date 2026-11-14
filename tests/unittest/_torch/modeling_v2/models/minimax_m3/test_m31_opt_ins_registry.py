# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default-off items of the MiniMax-M3.1-Flash targets (each target's m31_switches.OPT_INS, read through
``m31_enabled`` from ``TRTLLM_M31_ENABLE_<ID>``): every id a target's sources opt into is registered, every item is off
by default, the strict values hold, and an unregistered id raises when queried and is ignored when set. CPU only: each
registry is loaded from its file, the targets are not imported."""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

pytestmark = pytest.mark.cpu_only

_FAMILY = Path(_family.__file__).resolve().parent
TARGETS = ("tp2", "tp8", "dep", "tp2ctx", "dep2ctx")
ENABLE = "TRTLLM_M31_ENABLE_"


def _dir(target: str) -> Path:
    return _FAMILY / f"minimax_m3_1_flash_nvfp4__sm_107__{target}"


def _load(monkeypatch, target: str, env):
    for name in list(os.environ):
        if name.startswith(ENABLE):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        f"_m31_opt_ins_{target}_{id(env)}", _dir(target) / "m31_switches.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _opted_ids(target: str):
    ids = set()
    for path in _dir(target).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "m31_enabled"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                ids.add(node.args[0].value)
    return ids


@pytest.mark.parametrize("target", TARGETS)
def test_every_opted_id_is_registered(monkeypatch, target: str) -> None:
    ids = _opted_ids(target)
    if not ids:
        pytest.skip(f"the {target} target has no opt-in")
    opt_ins = _load(monkeypatch, target, {}).OPT_INS
    assert ids <= set(opt_ins), sorted(ids - set(opt_ins))


@pytest.mark.parametrize("target", TARGETS)
def test_every_opt_in_is_off_by_default_and_says_what_it_does(monkeypatch, target: str) -> None:
    switches = _load(monkeypatch, target, {})
    if not hasattr(switches, "OPT_INS"):
        pytest.skip(f"the {target} target has no opt-in")
    assert switches.OPT_INS
    assert not any(switches.m31_enabled(item) for item in switches.OPT_INS)
    assert all(isinstance(why, str) and why.strip() for why in switches.OPT_INS.values())


@pytest.mark.parametrize("target", TARGETS)
def test_values_and_unregistered_ids(monkeypatch, target: str) -> None:
    probe = _load(monkeypatch, target, {})
    if not hasattr(probe, "OPT_INS"):
        pytest.skip(f"the {target} target has no opt-in")
    item = sorted(probe.OPT_INS)[0]
    for value, on in (
        ("1", True),
        (" On ", True),
        ("yes", True),
        ("0", False),
        ("", False),
        ("no", False),
    ):
        assert _load(monkeypatch, target, {ENABLE + item: value}).m31_enabled(item) is on, value
    with pytest.raises(ValueError, match=ENABLE + item):
        _load(monkeypatch, target, {ENABLE + item: "maybe"})
    switches = _load(monkeypatch, target, {ENABLE + "NOT_AN_OPT_IN_OF_ANY_TARGET": "not-a-value"})
    assert not any(switches.m31_enabled(i) for i in switches.OPT_INS)
    with pytest.raises(KeyError):
        switches.m31_enabled("NOT_AN_OPT_IN_OF_ANY_TARGET")


@pytest.mark.parametrize("target", TARGETS)
def test_the_environment_is_read_once_at_import(monkeypatch, target: str) -> None:
    switches = _load(monkeypatch, target, {})
    if not hasattr(switches, "OPT_INS"):
        pytest.skip(f"the {target} target has no opt-in")
    item = sorted(switches.OPT_INS)[0]
    monkeypatch.setenv(ENABLE + item, "1")
    assert switches.m31_enabled(item) is False
