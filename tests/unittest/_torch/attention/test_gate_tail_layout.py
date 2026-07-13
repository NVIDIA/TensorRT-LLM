# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.attention_backend.trtllm as trtllm_backend
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.modules.attention import Attention


@pytest.mark.parametrize(
    "sm, expected", [(90, False), (100, True), (103, True), (120, False), (121, False)]
)
def test_strided_fused_qkv_backend_capability(monkeypatch, sm, expected):
    monkeypatch.setattr(trtllm_backend, "get_sm_version", lambda: sm)
    assert TrtllmAttention.support_strided_fused_qkv() is expected


def _make_gate_tail_attention(
    support_strided_fused_qkv: bool,
    *,
    quant_kind: str = "none",
) -> Attention:
    attention = object.__new__(Attention)
    nn.Module.__init__(attention)
    attention._gate_tail_layout_enabled = True
    attention._gate_tail_layout_active = False
    attention.attn_backend = "TRTLLM"
    attention.support_fused_qkv = True
    attention.fuse_qk_norm_rope = True
    attention.skip_rope = False
    attention.quant_config = None
    attention.mapping = SimpleNamespace(cp_size=1)
    attention.num_heads = 2
    attention.head_dim = 2
    attention.q_size = 4
    attention.kv_size = 2
    attention.attn = SimpleNamespace(support_strided_fused_qkv=lambda: support_strided_fused_qkv)

    attention.qkv_proj = nn.Linear(1, 12, bias=False)
    with torch.no_grad():
        attention.qkv_proj.weight.copy_(torch.arange(12).reshape(12, 1))
    attention.qkv_proj.has_any_quant = quant_kind != "none"
    attention.qkv_proj.has_fp8_qdq = quant_kind == "fp8_qdq"
    if quant_kind == "fp8_qdq":
        attention.qkv_proj.weight = nn.Parameter(
            attention.qkv_proj.weight.to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        attention.qkv_proj.weight_scale = nn.Parameter(torch.tensor(0.125), requires_grad=False)
    return attention


@pytest.mark.parametrize("supported", [False, True])
def test_gate_tail_layout_respects_backend_capability(supported):
    attention = _make_gate_tail_attention(supported)
    attention._maybe_permute_gate_tail_layout()

    expected_rows = [0, 1, 4, 5, 8, 9, 10, 11, 2, 3, 6, 7] if supported else list(range(12))
    torch.testing.assert_close(
        attention.qkv_proj.weight[:, 0], torch.tensor(expected_rows, dtype=torch.float32)
    )
    assert attention._gate_tail_layout_active is supported


def test_gate_tail_layout_supports_fp8_qdq():
    attention = _make_gate_tail_attention(True, quant_kind="fp8_qdq")
    original_weight_scale = attention.qkv_proj.weight_scale.clone()
    attention._maybe_permute_gate_tail_layout()

    expected_rows = torch.tensor([0, 1, 4, 5, 8, 9, 10, 11, 2, 3, 6, 7], dtype=torch.float32)
    torch.testing.assert_close(attention.qkv_proj.weight[:, 0].float(), expected_rows)
    torch.testing.assert_close(attention.qkv_proj.weight_scale, original_weight_scale)
    assert attention._gate_tail_layout_active


def test_gate_tail_layout_rejects_other_quantization():
    attention = _make_gate_tail_attention(True, quant_kind="unsupported")
    attention._maybe_permute_gate_tail_layout()

    torch.testing.assert_close(
        attention.qkv_proj.weight[:, 0], torch.arange(12, dtype=torch.float32)
    )
    assert not attention._gate_tail_layout_active
