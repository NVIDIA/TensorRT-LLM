# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.dense_attention import (
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


@pytest.mark.parametrize("dtype", ["fp8", "mxfp8"])
def test_cudnn_quantized_self_attention(dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Requires SM100")
    config = DiffusionModelConfig(
        attention=AttentionConfig(
            backend="CUDNN",
            quant_attention_config=QuantAttentionConfig(qk_dtype=dtype, v_dtype=dtype),
        ),
        attention_metadata_state={},
    )
    attention = H3DenseAttention(
        config, "CUDNN", num_heads=4, head_dim=128, layer_idx=0, sparse_params=None
    )
    generator = torch.Generator(device="cuda").manual_seed(19)
    q, k, v = [
        torch.randn((1, 256, 4, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    output = attention.forward(q, k, v)
    expected = F.scaled_dot_product_attention(
        *[x.transpose(1, 2).float() for x in (q, k, v)]
    ).transpose(1, 2)
    assert output.shape == expected.shape and output.dtype == q.dtype
    assert F.cosine_similarity(output.float().flatten(), expected.flatten(), dim=0) > 0.995
