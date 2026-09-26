# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Pixtral-Large's packed-QKV dispatcher path and workspace bound."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._utils import get_sm_version


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_pixtral_sm100_attention(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare ragged head-104 attention to FP32 SDPA without quadratic workspace."""
    if get_sm_version() not in (100, 103):
        pytest.skip("Exercises the SM100-family Pixtral FMHA dispatcher")
    # Exercise thop's dispatcher regardless of optional library configuration.
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    generator = torch.Generator(device="cuda").manual_seed(6665906)
    lengths = [2048, 32]
    num_heads, head_dim = 4, 104
    metadata = TrtllmAttentionMetadata(
        max_num_requests=4096, max_num_tokens=16384, kv_cache_manager=None
    )
    metadata.num_contexts = len(lengths)
    metadata.request_ids = list(range(len(lengths)))
    metadata.prompt_lens = lengths
    metadata.seq_lens = torch.tensor(lengths, dtype=torch.int)
    metadata.max_seq_len = max(lengths)
    metadata.prepare()
    attention = TrtllmAttention(
        layer_idx=0, num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim
    )
    q, k, v = [
        torch.randn(
            sum(lengths), num_heads * head_dim, device="cuda", dtype=dtype, generator=generator
        )
        for _ in range(3)
    ]
    output = attention.forward(
        torch.cat((q, k, v), dim=-1),
        None,
        None,
        metadata,
        forward_args=AttentionForwardArgs(attention_mask=PredefinedAttentionMask.FULL),
    )
    # The unfused path reserves hundreds of GiB for this metadata capacity.
    assert metadata.effective_workspace.numel() < 64 * 1024 * 1024
    reference = []
    offset = 0
    for length in lengths:
        qi, ki, vi = [
            tensor[offset : offset + length].view(length, num_heads, head_dim).transpose(0, 1)
            for tensor in (q, k, v)
        ]
        result = F.scaled_dot_product_attention(qi.float(), ki.float(), vi.float())
        reference.append(result.transpose(0, 1).reshape(length, num_heads * head_dim))
        offset += length
    torch.testing.assert_close(output.float(), torch.cat(reference), atol=1e-2, rtol=1e-2)
