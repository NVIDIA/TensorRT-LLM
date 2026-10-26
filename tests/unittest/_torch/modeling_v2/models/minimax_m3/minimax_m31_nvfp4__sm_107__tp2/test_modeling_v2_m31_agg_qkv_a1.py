# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2AGG_QKV_A1 dispatch: which steps take the A1 record, that it never touches the CTXNORMQ slot when it declines
before the take, and that its placeholder carries the packed shape without storage. The kernel's numerics are the
tp2ctx A1 UTs and the agg MB (rule 1b)."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    modeling as m,
)


def _core():
    return types.SimpleNamespace(
        heads=32,
        kv_heads=2,
        index_heads=2,
        mxfp8=None,
        w={"l3_qkv_w": torch.zeros(4992, 64), "l3_qkv_s": torch.zeros(1)},
    )


@pytest.fixture
def a1_on(monkeypatch):
    monkeypatch.setattr(m, "QKV_A1", True)
    monkeypatch.setattr(m, "CTXNORMQ", True)
    mode = types.SimpleNamespace(decode_graph=False, tuned_quantize_gemm=False)
    monkeypatch.setattr(m, "_mx_mode", lambda _: mode)
    monkeypatch.setattr(m, "is_torch_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return mode


def test_off_by_default_without_opt_in() -> None:
    """Rule 1b until its gates pass: the process takes A1 only with TRTLLM_M31_ENABLE_TP2AGG_QKV_A1=1."""
    assert m.QKV_A1 == (m.m31_enabled("TP2AGG_QKV_A1") and not m.m31_disabled("TP2AGG_QKV_A1"))


@pytest.mark.parametrize("rows", [1, 1024])
def test_small_steps_keep_the_slot(a1_on, rows) -> None:
    hidden = torch.zeros(rows, 64, dtype=torch.bfloat16)
    m._CTXNORMQ_MX.publish(hidden, "act", "sf")
    assert m._a1_defer(_core(), 3, hidden) is None
    assert m._CTXNORMQ_MX.pending is not None
    m._CTXNORMQ_MX.clear()


def test_takes_the_published_activation(a1_on) -> None:
    hidden = torch.zeros(1025, 64, dtype=torch.bfloat16)
    act = torch.zeros(1025, 64, dtype=torch.uint8)
    m._CTXNORMQ_MX.publish(hidden, act, "sf")
    rec = m._a1_defer(_core(), 3, hidden)
    assert isinstance(rec, m._DeferredA1)
    assert rec.act is act and rec.act_scale == "sf"
    assert m._CTXNORMQ_MX.pending is None
    assert rec.placeholder.shape == (1025, 4992) and rec.placeholder.dtype == torch.bfloat16
    assert rec.placeholder.untyped_storage().nbytes() == 2


def test_unpublished_or_other_tensor_declines(a1_on) -> None:
    hidden = torch.zeros(2048, 64, dtype=torch.bfloat16)
    assert m._a1_defer(_core(), 3, hidden) is None
    m._CTXNORMQ_MX.publish(hidden.clone(), "act", "sf")
    assert m._a1_defer(_core(), 3, hidden) is None


@pytest.mark.parametrize("field", ["decode_graph", "tuned_quantize_gemm"])
def test_modes_without_the_ctxnormq_gemm_decline(a1_on, field) -> None:
    setattr(a1_on, field, True)
    hidden = torch.zeros(2048, 64, dtype=torch.bfloat16)
    m._CTXNORMQ_MX.publish(hidden, "act", "sf")
    assert m._a1_defer(_core(), 3, hidden) is None
    m._CTXNORMQ_MX.clear()
