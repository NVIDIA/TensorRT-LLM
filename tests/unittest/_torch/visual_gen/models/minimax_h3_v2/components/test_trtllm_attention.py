# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.trtllm_attention import (
    H3TrtllmAttention,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.uncached_attention import (
    ContextBuffers,
    uncached_attention,
)
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="H3 TRTLLM attention uses SM100 or SM103",
)


def test_h3_uneven_padded_prefix():
    torch.manual_seed(42)
    seq, valid, heads = 65, 37, 4
    q, k, v = [
        torch.randn(2, seq, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    v[:, valid:].fill_(100)
    attn = H3TrtllmAttention(num_heads=heads, head_dim=128, attention_metadata_state={})
    actual = attn.forward(
        q, k, v, batch_size=2, seq_len=seq, discard_padded_queries=True, valid_seq_len=valid
    )
    expected = (
        F.scaled_dot_product_attention(*[x[:, :valid].transpose(1, 2).float() for x in (q, k, v)])
        .transpose(1, 2)
        .flatten(2)
    )
    torch.testing.assert_close(actual[:, :valid].float(), expected, rtol=0.02, atol=0.02)
    assert torch.count_nonzero(actual[:, valid:]) == 0


@pytest.mark.parametrize("cell", uncached_attention.CELLS)
def test_h3_uncached_context_attention(cell):
    batch, length = cell.spec["batch"], cell.spec["length"]
    heads, dim = 56, 128
    generator = torch.Generator(device="cuda").manual_seed(42)
    q, k, v = [
        torch.randn(
            batch, length, heads, dim, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(3)
    ]
    packed_qkv = torch.cat((q, k, v), dim=2).reshape(batch * length, -1)
    buffers = ContextBuffers.for_shape(batch, length, q.device)
    args = (packed_qkv, None, None, buffers, q.new_zeros((2,), dtype=torch.uint32))
    kwargs = dict(
        length=length,
        num_heads=heads,
        num_kv_heads=heads,
        head_dim=dim,
        layer_idx=0,
        quant_mode=0,
        prefill_threshold=None,
        decode_threshold=None,
        recipe=None,
    )
    uncached_attention.is_valid(*args, **kwargs)
    actual = uncached_attention(*args, **kwargs)
    expected = uncached_attention.reference(*args, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def test_h3_sage_key_smoothing():
    torch.manual_seed(42)
    batch, seq, heads, dim = 1, 257, 4, 128
    q, k, v = [
        torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    # A common offset changes quantization error while preserving the ideal attention weights.
    k += 32 * torch.randn((1, 1, heads, dim), device=k.device, dtype=k.dtype)
    cfg = QuantAttentionConfig(
        qk_dtype="fp8",
        v_dtype="fp8",
        q_block_size=1,
        k_block_size=16,
        v_block_size=1,
        smooth_k=True,
    )
    args = dict(num_heads=heads, num_kv_heads=heads, head_dim=dim, dtype=q.dtype)
    inputs = dict(batch_size=batch, seq_len=seq)
    candidate = H3TrtllmAttention(
        **args,
        attention_metadata_state={},
        attention_config=AttentionConfig(backend="TRTLLM", quant_attention_config=cfg),
    )
    reference = TrtllmAttention(**args, attention_metadata_state={}, quant_attention_config=cfg)
    expected = reference.forward(q, k, v, **inputs)
    actual = candidate.forward(q, k, v, **inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference.quant_attention_config = cfg.model_copy(update={"smooth_k": False})
    assert not torch.equal(expected, reference.forward(q, k, v, **inputs))
