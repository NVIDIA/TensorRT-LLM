# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.dense_attention import (
    H3DenseAttention,
)


@pytest.mark.parametrize("backend", ["VANILLA", "FA4", "FLASHINFER", "CUTEDSL"])
def test_h3_dense_attention_padding(backend):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Requires SM100")
    config = DiffusionModelConfig(attention_metadata_state={})
    attention = H3DenseAttention(
        config, backend, num_heads=2, head_dim=128, layer_idx=0, sparse_params=None
    )
    generator = torch.Generator(device="cuda").manual_seed(10)
    q, k, v = [
        torch.randn((2, 32, 2, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    lengths = torch.tensor([17, 29], device="cuda")
    mask = torch.arange(32, device="cuda")[None, :] < lengths[:, None]
    v.masked_fill_(~mask[:, :, None, None], 100)
    if backend == "CUTEDSL":
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
