# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.trtllm_attention import (
    H3TrtllmAttention,
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


def test_sol_trtllm_predictor_and_dense_prefix():
    from tensorrt_llm._torch.visual_gen.catalog.attention.sol_predictor import sol_predictor
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.dense_attention import (
        create_h3_attention,
    )
    from tensorrt_llm.visual_gen.sparse_attention import SolAttentionConfig

    sparse = SolAttentionConfig(tau=0.5, thresh_type="exact", disabled_until_timestep=0.6)
    config = DiffusionModelConfig(
        attention=AttentionConfig(backend="TRTLLM", sparse_attention_config=sparse),
        attention_metadata_state={},
    )
    attention = create_h3_attention(
        config, num_heads=2, head_dim=128, layer_idx=1, module_name="attn"
    )
    rng = torch.Generator(device="cuda").manual_seed(39)
    q, k, v = [
        (torch.randint(-16, 17, (1, 321, 2, 128), device="cuda", generator=rng) / 16).bfloat16()
        for _ in range(3)
    ]
    assert attention._sol_routes(q, k, v, 0.6) is None
    routes = attention._sol_routes(q, k, v, 0.5)
    expected = sol_predictor.reference(q, k, v, tau=0.5, sm_scale=128**-0.5, thresh_type="exact")
    for actual, reference in zip(
        (routes.exact_block_bits, routes.k_summary, routes.v_summary), expected, strict=True
    ):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
