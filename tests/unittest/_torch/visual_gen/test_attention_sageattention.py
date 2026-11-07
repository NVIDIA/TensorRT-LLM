# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical parity of the SAGEATTN backend (SageAttention INT8 QK / FP8 PV, SM120)
against torch SDPA. Requires an SM120 GPU and the ``sageattention`` package."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention.backends import get_visual_gen_attention_backend
from tensorrt_llm._torch.visual_gen.attention.backends.sageattention import (
    SageAttention,
    _sageattn_sm120,
)

_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12

pytestmark = [
    pytest.mark.skipif(not _IS_SM120, reason="SAGEATTN requires an SM120 GPU"),
    pytest.mark.skipif(_sageattn_sm120 is None, reason="sageattention package not installed"),
]


def test_factory_resolves_backend():
    assert get_visual_gen_attention_backend("sageattn") is SageAttention


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("s_q,s_kv", [(1024, 1024), (512, 77)])
def test_matches_sdpa(head_dim, s_q, s_kv):
    torch.manual_seed(0)
    B, H = 2, 8
    q = torch.randn(B, s_q, H, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, s_kv, H, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, s_kv, H, head_dim, dtype=torch.bfloat16, device="cuda")

    out = SageAttention(num_heads=H, head_dim=head_dim).forward(q, k, v)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    assert out.shape == ref.shape and out.dtype == q.dtype
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99


def test_rejects_key_padding_mask():
    x = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
    mask = torch.ones(1, 128, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="key_padding_mask"):
        SageAttention(num_heads=8, head_dim=64).forward(x, x, x, key_padding_mask=mask)
