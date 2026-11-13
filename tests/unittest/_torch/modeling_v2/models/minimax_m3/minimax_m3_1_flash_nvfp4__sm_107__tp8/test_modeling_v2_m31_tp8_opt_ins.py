# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 tp8 target's default-off items (m31_switches.OPT_INS, ``TRTLLM_M31_ENABLE_<ID>=1``): the registry
contract, each item off by default and on under its variable, and the CPU-visible behaviour each item switches.
CPU only: the multicast ops and gathers are stand-ins."""

import ast
import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark,
        m31_switches,
        modeling,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
        m3_attn_lc8,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

ENABLE = "TRTLLM_M31_ENABLE_"
DISABLE = "TRTLLM_M31_DISABLE_"
_TARGET_DIR = Path(m31_switches.__file__).resolve().parent


def _load_switches(monkeypatch, env):
    """A fresh copy of the target's m31_switches read under ``env`` (the installed module is left alone)."""
    for name in list(os.environ):
        if name.startswith((ENABLE, DISABLE)):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        f"_m31_switches_opt_ins_{len(env)}_{id(env)}", m31_switches.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reload_under(monkeypatch, env, *modules):
    """Reload ``modules`` (after m31_switches) under ``env``; the caller restores them with ``_restore``."""
    for name in list(os.environ):
        if name.startswith((ENABLE, DISABLE)):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    importlib.reload(m31_switches)
    return [importlib.reload(m) for m in modules]


def _restore(monkeypatch, *modules):
    for name in list(os.environ):
        if name.startswith((ENABLE, DISABLE)):
            monkeypatch.delenv(name)
    importlib.reload(m31_switches)
    for m in modules:
        importlib.reload(m)


# ------------------------------------------------------------------------------------------------- registry contract


def test_every_opt_in_is_off_by_default(monkeypatch) -> None:
    switches = _load_switches(monkeypatch, {})
    assert switches.OPT_INS
    assert not any(switches.m31_enabled(item) for item in switches.OPT_INS)
    assert all(isinstance(why, str) and why.strip() for why in switches.OPT_INS.values())


@pytest.mark.parametrize(
    "value, on", [("1", True), ("on", True), (" YES ", True), ("0", False), ("", False)]
)
def test_the_variable_turns_an_opt_in_on(monkeypatch, value: str, on: bool) -> None:
    switches = _load_switches(monkeypatch, {ENABLE + "LC8_ONEDIV": value})
    assert switches.m31_enabled("LC8_ONEDIV") is on


def test_a_bad_value_raises_at_import(monkeypatch) -> None:
    with pytest.raises(ValueError, match=ENABLE + "LC8_ONEDIV"):
        _load_switches(monkeypatch, {ENABLE + "LC8_ONEDIV": "maybe"})


def test_an_unregistered_opt_in_raises_when_queried_and_is_ignored_when_set(monkeypatch) -> None:
    switches = _load_switches(monkeypatch, {ENABLE + "NOT_AN_OPT_IN_OF_THIS_TARGET": "not-a-value"})
    assert not any(switches.m31_enabled(item) for item in switches.OPT_INS)
    with pytest.raises(KeyError):
        switches.m31_enabled("NOT_AN_OPT_IN_OF_THIS_TARGET")


def test_opt_ins_and_kill_switches_are_separate_registries(monkeypatch) -> None:
    switches = _load_switches(monkeypatch, {DISABLE + "ONEDIV_CAP": "1"})
    assert switches.m31_disabled("ONEDIV_CAP")
    with pytest.raises(KeyError):
        switches.m31_enabled("ONEDIV_CAP")


def test_every_id_the_target_opts_into_is_registered() -> None:
    ids = set()
    for path in _TARGET_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None)) == "m31_enabled"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                ids.add(node.args[0].value)
    assert ids, "no opt-in read found"
    assert ids <= set(m31_switches.OPT_INS), sorted(ids - set(m31_switches.OPT_INS))


# ----------------------------------------------------------------------------------------------------- the items


def test_module_defaults_are_the_stock_paths() -> None:
    assert modeling.R6_DENSE_PREQUANT is False
    assert modeling.AR_AUTOTUNE_ENABLED is False
    assert modeling.SLOT_IDS_CHANGED is False
    assert modeling.TP8_LONG_FAMILY_MIN_TOKENS is None
    assert dspark.S9B_MC is False and dspark.S9B_MC_MAX_ROWS == 8
    assert dspark.MCAG is False
    assert dspark.TP8_ANCESTRAL is False
    assert m3_attn_lc8.ONE_DIV_MERGE is False and m3_attn_lc8.ONE_DIV_MAX_ROWS == 8
    assert not any(m3_attn_lc8.one_div_applies(rows) for rows in (1, 4, 8, 9, 32))


@pytest.mark.parametrize("k, expected", [(1, 0), (3, 0), (5, 0), (7, 2), (15, 2)])
def test_window_capped_decode_batch_by_draft_length(k: int, expected: int) -> None:
    assert dspark.S5_MAX_BATCH_BY_K == {3: 0, 7: 2}
    assert dspark.s5_max_batch_for(k) == expected


@pytest.mark.parametrize(
    "env, on, at_32",
    [
        ({}, False, False),
        ({ENABLE + "LC8_ONEDIV": "1"}, True, False),
        ({ENABLE + "LC8_ONEDIV": "1", DISABLE + "ONEDIV_CAP": "1"}, True, True),
        ({DISABLE + "ONEDIV_CAP": "1"}, False, False),
    ],
)
def test_lc8_onediv_and_its_row_cap(monkeypatch, env, on: bool, at_32: bool) -> None:
    try:
        (lc8,) = _reload_under(monkeypatch, env, m3_attn_lc8)
        assert lc8.ONE_DIV_MERGE is on
        assert lc8.one_div_applies(8) is on
        assert lc8.one_div_applies(9) is (on and at_32)
        assert lc8.one_div_applies(32) is at_32
    finally:
        _restore(monkeypatch, m3_attn_lc8)


@pytest.mark.parametrize(
    "item, attr, on",
    [
        ("DRAFT_AR_NORM_CLUSTER", "S9B_MC", True),
        ("DRAFT_LOGITS_ALLGATHER_MC", "MCAG", True),
        ("TP8_ANCESTRAL_DRAFT", "TP8_ANCESTRAL", True),
    ],
)
def test_dspark_opt_ins(monkeypatch, item: str, attr: str, on: bool) -> None:
    try:
        (mod,) = _reload_under(monkeypatch, {ENABLE + item: "1"}, dspark)
        assert getattr(mod, attr) is on
    finally:
        _restore(monkeypatch, dspark)
    assert getattr(dspark, attr) is False


@pytest.mark.parametrize(
    "item, check",
    [
        ("R6_DENSE_PREQUANT", lambda m: m.R6_DENSE_PREQUANT is True),
        ("AR_AUTOTUNE", lambda m: m.AR_AUTOTUNE_ENABLED is True),
        ("SLOT_IDS_CHANGED", lambda m: m.SLOT_IDS_CHANGED is True),
        ("TP8_SHORT_FAMILY", lambda m: m.TP8_LONG_FAMILY_MIN_TOKENS == 540000),
    ],
)
def test_modeling_opt_ins(monkeypatch, item: str, check) -> None:
    try:
        (mod,) = _reload_under(monkeypatch, {ENABLE + item: "1"}, modeling)
        assert check(mod)
    finally:
        _restore(monkeypatch, modeling)


def test_ar_autotune_opt_in_keeps_the_autotuner(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(modeling, "AR_AUTOTUNE_ENABLED", True)
    assert modeling._pin_allreduce_heuristic(0) is False
    assert "TLLM_DISABLE_ALLREDUCE_AUTOTUNE" not in os.environ


@pytest.mark.parametrize("rows", [9, 16, 64])
def test_ar_norm_cluster_op_declines_past_8_rows_without_a_workspace(
    monkeypatch, rows: int
) -> None:
    def no_workspace(*args, **kwargs):
        raise AssertionError("the op must decline before asking for a workspace")

    monkeypatch.setattr(dspark.m3_mc_allreduce, "get_workspace", no_workspace)
    op = dspark._s9b_mc_op(SimpleNamespace(tp_size=8))
    x = torch.zeros(rows, 16, dtype=torch.bfloat16)
    assert op(x, x, torch.ones(16, dtype=torch.bfloat16), 1e-6) is None


def test_ar_norm_cluster_op_calls_the_drafter_norm_form(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dspark.m3_mc_allreduce, "get_workspace", lambda mapping, hidden: "ws")

    def mc(x, residual, w, eps, workspace, use_pdl=False, drafter_norm=False, **kwargs):
        calls.append((tuple(x.shape), workspace, drafter_norm, kwargs))
        return "op"

    monkeypatch.setattr(dspark.m3_mc_allreduce, "m3_mc_allreduce_norm", mc)
    op = dspark._s9b_mc_op(SimpleNamespace(tp_size=8))
    x = torch.zeros(8, 16, dtype=torch.bfloat16)
    assert op(x, x, torch.ones(16, dtype=torch.bfloat16), 1e-6) == "op"
    assert calls == [((8, 16), "ws", True, {})]
    monkeypatch.setattr(dspark.m3_mc_allreduce, "get_workspace", lambda mapping, hidden: None)
    assert op(x, x, torch.ones(16, dtype=torch.bfloat16), 1e-6) is None


@pytest.fixture
def mcag_target(monkeypatch):
    calls = []
    compiled = {(7, 25008, 8)}
    monkeypatch.setattr(
        dspark._allgather_cast,
        "is_compiled",
        lambda rows, cols, world, grid=0, device=None, fp32=False: fp32
        and (rows, cols, world) in compiled,
    )
    monkeypatch.setattr(
        dspark._allgather_cast,
        "gather_last_dim_f32",
        lambda logits, **ws: calls.append((tuple(logits.shape), ws["world"])) or "mc-gathered",
    )
    target = SimpleNamespace(
        _mcag_cols=25008,
        _a1={"world": 8, "ws_uc": None, "ws_mc": None, "ws_flags": None, "rank": 0},
    )
    return target, calls


def test_draft_logits_gather_takes_compiled_fp32_rows(mcag_target) -> None:
    target, calls = mcag_target
    assert dspark.M3DSparkWorker._mcag_gather(target, torch.zeros(1, 7, 25008)) == "mc-gathered"
    assert calls == [((1, 7, 25008), 8)]


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(1, 7, 25008, dtype=torch.bfloat16),  # not fp32
        torch.zeros(1, 7, 25000),  # other columns
        torch.zeros(2, 7, 25008),  # rows not compiled
    ],
)
def test_draft_logits_gather_declines(mcag_target, logits) -> None:
    target, calls = mcag_target
    assert dspark.M3DSparkWorker._mcag_gather(target, logits) is None
    assert calls == []


def test_ancestral_draft_stays_on_the_markov_route_by_default(monkeypatch) -> None:
    monkeypatch.setattr(dspark, "TP8_ANCESTRAL", False)
    worker = SimpleNamespace()
    assert dspark.M3DSparkWorker._samples_block_ancestrally(worker, None, None) is False
