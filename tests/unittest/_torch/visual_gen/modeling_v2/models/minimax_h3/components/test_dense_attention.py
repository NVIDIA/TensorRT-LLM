# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention import (
    deco_attention as deco_catalog,
)
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components.dense_attention import (
    H3DenseAttention,
)
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig


@pytest.mark.parametrize("backend", ["VANILLA", "FA4", "FLASHINFER", "CUTEDSL", "CUDNN"])
def test_h3_dense_attention_padding(backend):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Requires SM100")
    heads, seq, lengths = (56, 12, [8]) if backend == "FLASHINFER" else (2, 32, [17, 29])
    config = DiffusionModelConfig(attention_metadata_state={})
    attention = H3DenseAttention(
        config, backend, num_heads=heads, head_dim=128, layer_idx=0, sparse_params=None
    )
    generator = torch.Generator(device="cuda").manual_seed(10)
    q, k, v = [
        torch.randn(
            (len(lengths), seq, heads, 128),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    ]
    mask = torch.arange(seq, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]
    v.masked_fill_(~mask[:, :, None, None], 100)
    if backend in ("CUTEDSL", "CUDNN"):
        with pytest.raises(NotImplementedError, match="unpadded"):
            attention.forward(q, k, v, key_padding_mask=mask)
        mask = None
    q_hnd, k_hnd, v_hnd = (x.transpose(1, 2) for x in (q, k, v))
    expected = F.scaled_dot_product_attention(
        q_hnd.double(),
        k_hnd.double(),
        v_hnd.double(),
        attn_mask=mask[:, None, None, :] if mask is not None else None,
    )
    if backend == "VANILLA":
        actual = attention.forward(q_hnd, k_hnd, v_hnd, key_padding_mask=mask)
    else:
        actual = attention.forward(q, k, v, key_padding_mask=mask).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)


def test_h3_deco_attention_runs_short_sequences_unquantized():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Requires SM100")
    recipe = QuantAttentionConfig(qk_dtype="nvfp4", v_dtype="fp8", deco_attention=True)
    config = DiffusionModelConfig(
        attention=AttentionConfig(backend="CUTEDSL", quant_attention_config=recipe)
    )
    attention = H3DenseAttention(
        config, "CUTEDSL", num_heads=2, head_dim=128, layer_idx=0, sparse_params=None
    )
    generator = torch.Generator(device="cuda").manual_seed(11)
    q, k, v = [
        torch.randn((2, 32, 2, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    mask = torch.arange(32, device="cuda")[None, :] < torch.tensor([17, 32], device="cuda")[:, None]
    padded = ~mask[:, :, None, None]
    # Padded rows have no influence, whatever they hold.
    k.masked_fill_(padded, float("nan"))
    v.masked_fill_(padded, float("nan"))
    tags = torch.zeros(32, dtype=torch.long, device="cuda")
    with patch.object(deco_catalog, "deco_attention", side_effect=AssertionError("quantized")):
        actual = attention.forward(q, k, v, key_padding_mask=mask, token_tags=tags)
    q_hnd, k_hnd, v_hnd = (x.masked_fill(padded, 0).transpose(1, 2).double() for x in (q, k, v))
    expected = F.scaled_dot_product_attention(
        q_hnd, k_hnd, v_hnd, attn_mask=mask[:, None, None, :]
    ).transpose(1, 2)
    assert (actual[~mask] == 0).all()
    torch.testing.assert_close(actual[mask].float(), expected[mask].float(), rtol=0.02, atol=0.02)
