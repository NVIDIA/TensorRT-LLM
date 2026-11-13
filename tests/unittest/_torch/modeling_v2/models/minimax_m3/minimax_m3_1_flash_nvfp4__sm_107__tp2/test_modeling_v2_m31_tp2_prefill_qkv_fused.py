# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2_PREFILL_QKV_FUSED dispatch: which prefill steps take the fused QKV record, that it never touches the CTXNORMQ
slot when it declines before the take, and that its placeholder carries the packed shape without storage. The
kernel's numerics are the tp2ctx fused-QKV tests'. Off by default (``TRTLLM_M31_ENABLE_TP2_PREFILL_QKV_FUSED=1`` opts
in). CPU only."""

import os
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        modeling as m,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def _core():
    return types.SimpleNamespace(
        heads=32,
        kv_heads=2,
        index_heads=2,
        mxfp8=None,
        w={"l3_qkv_w": torch.zeros(4992, 64), "l3_qkv_s": torch.zeros(1)},
    )


@pytest.fixture
def fused_on(monkeypatch):
    monkeypatch.setattr(m, "QKV_A1", True)
    monkeypatch.setattr(m, "CTXNORMQ", True)
    monkeypatch.setattr(m, "PRODUCER_HEADS", (32, 2, 2))
    mode = types.SimpleNamespace(decode_graph=False, tuned_quantize_gemm=False)
    monkeypatch.setattr(m, "_mx_mode", lambda _: mode)
    monkeypatch.setattr(m, "is_torch_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield mode
    m._CTXNORMQ_MX.clear()
    m._SP_STATE["ovl"] = None


def test_off_by_default() -> None:
    assert m.QKV_A1 is m.m31_enabled("TP2_PREFILL_QKV_FUSED")
    if os.environ.get("TRTLLM_M31_ENABLE_TP2_PREFILL_QKV_FUSED") is None:
        assert m.QKV_A1 is False
        hidden = torch.zeros(2048, 64, dtype=torch.bfloat16)
        assert m._a1_defer(_core(), 3, hidden) is None


@pytest.mark.parametrize("rows", [1, 1024])
def test_small_steps_keep_the_slot(fused_on, rows) -> None:
    hidden = torch.zeros(rows, 64, dtype=torch.bfloat16)
    m._CTXNORMQ_MX.publish(hidden, "act", "sf")
    assert m._a1_defer(_core(), 3, hidden) is None
    assert m._CTXNORMQ_MX.pending is not None


def test_takes_the_published_activation(fused_on) -> None:
    hidden = torch.zeros(1025, 64, dtype=torch.bfloat16)
    act = torch.zeros(1025, 64, dtype=torch.uint8)
    m._CTXNORMQ_MX.publish(hidden, act, "sf")
    rec = m._a1_defer(_core(), 3, hidden)
    assert isinstance(rec, m._DeferredA1)
    assert rec.act is act and rec.act_scale == "sf" and rec.ovl is None
    assert m._CTXNORMQ_MX.pending is None
    assert rec.placeholder.shape == (1025, 4992) and rec.placeholder.dtype == torch.bfloat16
    assert rec.placeholder.untyped_storage().nbytes() == 2


def test_unpublished_or_other_tensor_declines(fused_on) -> None:
    hidden = torch.zeros(2048, 64, dtype=torch.bfloat16)
    assert m._a1_defer(_core(), 3, hidden) is None
    m._CTXNORMQ_MX.publish(hidden.clone(), "act", "sf")
    assert m._a1_defer(_core(), 3, hidden) is None


@pytest.mark.parametrize("field", ["decode_graph", "tuned_quantize_gemm"])
def test_modes_without_the_ctxnormq_gemm_decline(fused_on, field) -> None:
    setattr(fused_on, field, True)
    hidden = torch.zeros(2048, 64, dtype=torch.bfloat16)
    m._CTXNORMQ_MX.publish(hidden, "act", "sf")
    assert m._a1_defer(_core(), 3, hidden) is None


def test_other_head_geometries_decline(fused_on, monkeypatch) -> None:
    monkeypatch.setattr(m, "PRODUCER_HEADS", (64, 4, 4))
    hidden = torch.zeros(2048, 64, dtype=torch.bfloat16)
    m._CTXNORMQ_MX.publish(hidden, "act", "sf")
    assert m._a1_defer(_core(), 3, hidden) is None
