# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-plan dispatch coverage. Numerical and per-rank model gates are separate."""

import importlib
import itertools
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    m31_switches,
    modeling,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("r1,bands,wide", itertools.product((False, True), repeat=3))
def test_decode_plan_switches_and_all_rows(monkeypatch, r1, bands, wide):
    try:
        with monkeypatch.context() as patch:
            for name, enabled in (("R1VR8", r1), ("VR_BANDS", bands), ("TP2WIDE_DECODE", wide)):
                patch.setenv("TRTLLM_M31_DISABLE_" + name, "0" if enabled else "1")
            importlib.reload(m31_switches)
            m = importlib.reload(modeling)
            bound = 8 if not r1 else 48 if not bands else 64 if not wide else 448
            assert m.DECODE_STEP_MAX == bound
            for rows in range(1, 450):
                assert m.is_decode_step(rows) == (rows <= bound)
                if rows > bound:
                    with pytest.raises(AssertionError):
                        m.plan_for(rows)
                    continue
                plan = m.plan_for(rows)
                if rows <= 64:
                    expected = (
                        ("tiny", 8)
                        if rows <= 8
                        else ("b16", 16)
                        if rows <= 16
                        else ("b48", 48)
                        if rows <= 48
                        else ("wide", 64)
                    )
                    assert (plan.name, plan.max_tokens) == expected
                else:
                    assert (plan.name, plan.max_tokens) == ("wide448", 448)
                flags = (plan.attn_mc_ar, plan.moe_mc_ar, plan.qkv_prequant)
                assert flags == ((True, True, True) if rows <= 48 else (False, False, False))
    finally:
        importlib.reload(m31_switches)
        importlib.reload(modeling)


@pytest.mark.parametrize("rows", [64, 65, 128, 184, 224, 256, 448, 449])
@pytest.mark.parametrize("enabled", [False, True])
def test_wide_input_keeps_original_operands_and_narrow_route(monkeypatch, rows, enabled):
    ids, table, weight, raw, normed = (object() for _ in range(5))
    calls = []

    def embedding(actual_ids, actual_table):
        assert actual_ids is ids and actual_table is table
        calls.append("embedding")
        return raw

    def rmsnorm(actual_raw, actual_weight, eps):
        assert actual_raw is raw and actual_weight is weight and eps == 1e-6
        calls.append("norm")
        return normed

    monkeypatch.setattr(modeling, "TP2WIDE_DECODE", enabled)
    monkeypatch.setattr(modeling, "DECODE_STEP_MAX", 448 if enabled else 64)
    monkeypatch.setattr(modeling, "m3_embedding", embedding)
    monkeypatch.setattr(modeling, "flashinfer_rmsnorm", rmsnorm)
    monkeypatch.setattr(modeling, "taken", lambda *args: None)
    monkeypatch.setattr(modeling, "uncovered", lambda *args: None)
    target = modeling.DecodeTarget.__new__(modeling.DecodeTarget)
    target.core = SimpleNamespace(
        w={"embed": table, "l0_in_norm": weight},
        rms_eps=1e-6,
        mapping=SimpleNamespace(tp_rank=1),
    )
    out, residual = target._embed(ids, rows)
    if enabled and 64 < rows <= 448:
        assert out is normed and residual is raw
        assert calls == ["embedding", "norm"]
    else:
        assert out is raw and residual is None
        assert calls == ["embedding"]
