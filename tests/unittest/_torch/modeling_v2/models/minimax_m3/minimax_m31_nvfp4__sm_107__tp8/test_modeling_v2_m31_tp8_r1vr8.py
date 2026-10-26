# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R1VR8: the sm_107 TP8 decode step bound is 32 rows (K3 c8) and every fast path keeps its own row range.

The dispatch of a step of up to 8 rows is exactly the TP2-lineage one (the same plan, GEMV rows, pushes, embedding and
multicast ranges, linear paths); 9..32 rows take the 16- / 32-row plans, the GEMV on the QKV and gate_up projections up
to 32 rows and on the o_proj up to 16, the generic MoE above the engine's 8 rows; TRTLLM_M31_DISABLE_R1VR8=1 restores
the 8-row bound.
"""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.gemm.m3_mx_gemv import (  # noqa: E501
    DENSE_DOWN,
    DENSE_GATE_UP,
    O_PROJ,
    QKV,
    QKV_SPARSE,
    SHARED_DOWN,
    SHARED_GATE_UP,
)

ALL = (QKV_SPARSE, QKV, O_PROJ, SHARED_GATE_UP, SHARED_DOWN, DENSE_GATE_UP, DENSE_DOWN)


def test_bounds() -> None:
    assert modeling.TP2_ROWS == 8 and modeling.R1VR8 is True and modeling.DECODE_MAX_TOKENS == 32
    assert [p.name for p in modeling.DECODE_PLANS][:3] == ["tiny", "small", "verify"]
    assert [p.max_tokens for p in modeling.DECODE_PLANS][:3] == [8, 16, 32]


@pytest.mark.parametrize("rows", range(1, 9))
def test_up_to_8_rows_is_the_tp2_dispatch(rows: int) -> None:
    m = modeling
    assert m.is_decode_step(rows) and m.plan_for(rows).name == "tiny"
    for rr in (m.EMBED_FIRSTNORM, m.ATTN_PUSH, m.DENSE_PUSH, m.MC_ALLREDUCE, m.MOE_ENGINE):
        assert rr.covers(rows) == (rr.keys != frozenset()), rr.path
    assert m.moe_path(rows) == (m.MOE_ENGINE.path if m.MOE_ENGINE.covers(rows) else None)
    for nk in ALL:
        assert rows in m.MX_GEMV_ROWS.get(nk, ()) or not m.MX_GEMV_TP2
        assert rows <= m.MX_GEMV_BF16_MAX_TOKENS.get(nk, 0) or not m.MX_GEMV_TP2
        assert m.linear_path(*nk, rows, True, False) == m.MX_GEMV_PATH
        assert m.linear_path(*nk, rows, False, False) == m.MX_GEMV_BF16_PATH


@pytest.mark.parametrize("rows", range(9, 33))
def test_9_to_32_rows(rows: int) -> None:
    m = modeling
    assert m.is_decode_step(rows)
    assert m.plan_for(rows).name == ("small" if rows <= 16 else "verify")
    assert m.MC_ALLREDUCE.covers(rows)
    for rr in (m.EMBED_FIRSTNORM, m.ATTN_PUSH, m.DENSE_PUSH, m.MOE_ENGINE, m.MOE_PUSH):
        assert not rr.covers(rows), rr.path  # TP2-lineage paths stay at their 8 rows
    assert m.moe_path(rows) is None
    for nk in (QKV_SPARSE, QKV, SHARED_GATE_UP, DENSE_GATE_UP):
        assert m.linear_path(*nk, rows, True, False) == m.MX_GEMV_PATH
    assert m.linear_path(*O_PROJ, rows, True, False) == (
        m.MX_GEMV_PATH if rows <= 16 else m.PREQUANT_GEMM_PATH
    )
    for nk in (SHARED_DOWN, DENSE_DOWN):  # bf16 input: the pinned quantize + CuTe DSL GEMM
        assert m.linear_path(*nk, rows, False, False) == m.GENERIC_GEMM_PATH
    assert m.is_decode_step(33) is m.VR_BANDS and not m.is_decode_step(m.DECODE_STEP_MAX + 1)


def test_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_M31_DISABLE_R1VR8", "1")
    try:
        m = importlib.reload(modeling)
        assert m.DECODE_MAX_TOKENS == 8 and [p.name for p in m.DECODE_PLANS] == ["tiny"]
        assert not m.is_decode_step(9)
        assert all(m.MX_GEMV_ROWS[nk] == frozenset(range(1, 9)) for nk in ALL)
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_R1VR8", raising=False)
        importlib.reload(modeling)
