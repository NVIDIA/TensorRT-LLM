# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R1VR8 on the sm_107 TP2 target: the decode layer takes pure generation steps up to the TP4 base's 48 rows on the
M3 tp2 row buckets (tiny 8 / b16 16 / b48 48), and up to 64 with VR-BANDS; TRTLLM_M31_DISABLE_R1VR8=1 restores the
TP2-lineage 8-row bound (one "tiny" plan, every GEMV row table at 1..8, SWQ (17..32 rows) and VR-BANDS off), as the
sm_107 TP8 target's kill does. The switch follows the m31_switches value contract (read at import), so the module is
reloaded under each environment. CPU only.
"""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.gemm.m3_mx_gemv import (  # noqa: E501
    DENSE_DOWN,
    DENSE_GATE_UP,
    O_PROJ,
    QKV,
    QKV_SPARSE,
    SHARED_DOWN,
    SHARED_GATE_UP,
)

ALL = (QKV_SPARSE, QKV, O_PROJ, SHARED_GATE_UP, SHARED_DOWN, DENSE_GATE_UP, DENSE_DOWN)


@pytest.fixture
def r1vr8_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_R1VR8", "1")
    importlib.reload(m31_switches)
    try:
        yield importlib.reload(modeling)
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_R1VR8", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)


def test_default_bounds() -> None:
    m = modeling
    assert m.TP2_ROWS == 8 and m.R1VR8 is True and m.DECODE_MAX_TOKENS == 48
    assert m.TP2_BUCKETS == (8, 16, 48)
    assert [p.name for p in m.DECODE_PLANS][:3] == ["tiny", "b16", "b48"]
    assert [p.max_tokens for p in m.DECODE_PLANS][:3] == [8, 16, 48]
    assert m.is_decode_step(48) and m.is_decode_step(33)


def test_swq_takes_the_shared_down_rows_past_its_gemv() -> None:
    """SWQ: the shared down projection (6144, 1536) at 17..32 rows, the rows its decode GEMV does not serve."""
    m = modeling
    assert SHARED_DOWN == (6144, 1536) == m.SWQ_SHARED_DOWN
    assert m.SWQ is True
    assert m.SWQ_ROWS == {SHARED_DOWN: frozenset(range(17, 33))}
    assert not (m.SWQ_ROWS[SHARED_DOWN] & m.MX_GEMV_ROWS[SHARED_DOWN])


def test_kill_switch_restores_the_8_row_bound(r1vr8_off) -> None:
    m = r1vr8_off
    assert not m.R1VR8 and m.DECODE_MAX_TOKENS == 8
    assert not m.VR_BANDS and m.DECODE_STEP_MAX == 8
    assert m.TP2_BUCKETS == (8,) and [p.name for p in m.DECODE_PLANS] == ["tiny"]
    assert m.is_decode_step(8) and not m.is_decode_step(9)
    assert all(m.MX_GEMV_ROWS[nk] == frozenset(range(1, 9)) for nk in ALL)
    assert not any(m.SWQ_ROWS.values())


def test_vr_bands_kill_alone_keeps_r1vr8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_M31_DISABLE_VR_BANDS", "1")
    importlib.reload(m31_switches)
    try:
        m = importlib.reload(modeling)
        assert m.R1VR8 and not m.VR_BANDS and m.DECODE_STEP_MAX == m.DECODE_MAX_TOKENS == 48
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_VR_BANDS", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)
