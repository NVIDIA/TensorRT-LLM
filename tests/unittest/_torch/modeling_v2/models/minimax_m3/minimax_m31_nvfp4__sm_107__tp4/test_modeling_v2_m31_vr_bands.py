# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VR-BANDS: the sm_107 TP4 decode step bound is 64 rows while every op keeps its own row range.

Steps of 49..64 rows (K7 c7..c8) take the "wide" plan (the built-in all-reduce at both boundaries); each op arms
exactly one path at every row count 1..64 (its fast path inside its range, its generic form past it), and no op's
range moves. TRTLLM_M31_DISABLE_VR_BANDS=1 restores the 48-row step bound.
"""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4 import (  # noqa: E501
    m31_switches,
    modeling,
)

OP_BOUND = 48
STEP_ROWS = range(1, 64 + 1)
ROW_TABLES = ("MX_GEMV_ROWS", "GEMV48_ROWS", "LINEAR_CUBLASLT_ROWS")


def _ranges(m) -> dict:
    return {r.path: r.keys for r in m.FAST_PATHS}


def _projections(m) -> set:
    tables = ROW_TABLES + ("MX_GEMV_BF16_MAX_TOKENS",)
    return {nk for name in tables for nk in getattr(m, name, {})}


def _linear_paths(m) -> dict:
    return {
        nk: [
            m.linear_path(*nk, rows, prequant, False)
            for rows in STEP_ROWS
            for prequant in (True, False)
        ]
        for nk in _projections(m)
    }


# The default (VR-BANDS on) module's view, recorded at collection before any reload.
_ON_RANGES = _ranges(modeling)
_ON_LINEAR_PATHS = _linear_paths(modeling)


@pytest.fixture
def vr_bands_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_VR_BANDS", "1")
    importlib.reload(m31_switches)
    try:
        yield importlib.reload(modeling)
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_VR_BANDS", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)


def test_the_step_bound_is_64_and_the_op_bound_stays() -> None:
    m = modeling
    assert m.VR_BANDS and m.DECODE_STEP_MAX == 64 and m.DECODE_MAX_TOKENS == OP_BOUND
    assert [m.is_decode_step(n) for n in range(1, 66)] == [True] * 64 + [False]


@pytest.mark.parametrize("rows", STEP_ROWS)
def test_every_step_row_count_takes_exactly_one_plan(rows: int) -> None:
    m = modeling
    plan = m.plan_for(rows)
    assert plan is [p for p in m.DECODE_PLANS if rows <= p.max_tokens][0]
    if rows > OP_BOUND:
        assert plan.name == "wide" and plan.max_tokens == 64
        assert (plan.attn_mc_ar, plan.moe_mc_ar, plan.qkv_prequant) == (False, False, False)
    else:
        assert plan.name != "wide"


@pytest.mark.parametrize("rows", STEP_ROWS)
def test_every_op_arms_exactly_one_path(rows: int) -> None:
    """Each projection's GEMM row tables never overlap and it selects one GEMM per activation form; at most one MoE
    path (the engine, the 9..13-row engine where built, the unfinalized MoE) covers the row count, and the MoE takes it
    there and the generic MoE elsewhere."""
    m = modeling
    for nk in _projections(m):
        armed = [name for name in ROW_TABLES if rows in getattr(m, name, {}).get(nk, ())]
        assert len(armed) <= 1, (nk, rows, armed)
        for prequant in (True, False):
            assert isinstance(m.linear_path(*nk, rows, prequant, False), str)
    moe_ranges = [
        getattr(m, n) for n in ("MOE_ENGINE", "MOE16_ENGINE", "MC_FINALIZE") if hasattr(m, n)
    ]
    covering = [r.path for r in moe_ranges if r.covers(rows)]
    assert len(covering) <= 1, (rows, covering)
    assert m.moe_path(rows) == (covering[0] if covering else None)


def test_no_op_range_moves(vr_bands_off) -> None:
    """The bound is the step's alone: every fast path's range and every projection's GEMM per row count is the same
    with VR-BANDS off."""
    assert _ranges(vr_bands_off) == _ON_RANGES
    assert _linear_paths(vr_bands_off) == _ON_LINEAR_PATHS


def test_kill_switch(vr_bands_off) -> None:
    m = vr_bands_off
    assert not m.VR_BANDS and m.DECODE_STEP_MAX == m.DECODE_MAX_TOKENS == OP_BOUND
    assert "wide" not in [p.name for p in m.DECODE_PLANS]
    assert m.is_decode_step(OP_BOUND) and not m.is_decode_step(OP_BOUND + 1)
