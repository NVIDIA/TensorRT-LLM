# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV-cache compression forces the unfused-RoPE path.

Compression physically evicts cached tokens, so the KV length stops matching
the logical sequence length. The fused path derives each new token's rotary
position from the KV length inside the attention kernel; the unfused path
consumes the engine's logical ``position_ids``. With compression enabled the
attention module must therefore keep RoPE unfused so rotary positions stay
logical (original absolute positions, matching the official TriAttention
implementations) while the shortened KV length only bounds attention extent.
"""

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig


def _make_attention(model_config: ModelConfig) -> Attention:
    return Attention(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=1024,
        bias=False,
        pos_embd_params=None,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=model_config,
    )


def test_plain_attention_defaults_to_fused_rope() -> None:
    attn = _make_attention(ModelConfig())

    assert attn.rope_fusion is True


def test_kv_cache_compression_forces_unfused_rope() -> None:
    model_config = ModelConfig(kv_cache_compression_config=TriAttentionKvCacheCompressionConfig(
        model_path="/models/test", calibration_path="/calib/test.pt"))
    attn = _make_attention(model_config)

    assert attn.rope_fusion is False


def test_unfused_yarn_rope_is_applied_exactly_once() -> None:
    """With rope_fusion=False the Python-side rotary module owns RoPE, so the
    backend must receive no position-embedding params. yarn is not listed in
    PositionEmbeddingType.is_rope(), which used to leak the params through and
    made the C++ QKV preprocess rotate a second time (double RoPE)."""
    from tensorrt_llm._torch.attention_backend.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

    yarn_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams(
            dim=32,
            theta=150000,
            scale_type=RotaryScalingType.yarn,
            scale=32.0,
            max_positions=1024,
            original_max_positions=256,
            beta_fast=32,
            beta_slow=1,
            duplicate_data=False,
        ),
        is_neox=True,
    )
    model_config = ModelConfig(kv_cache_compression_config=TriAttentionKvCacheCompressionConfig(
        model_path="/models/test", calibration_path="/calib/test.pt"))
    attn = Attention(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=1024,
        bias=False,
        pos_embd_params=yarn_params,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=model_config,
    )

    assert attn.rope_fusion is False
    assert attn.rotary_emb is not None
    # The TRTLLM backend keeps the type as an int; 0 means no position
    # embedding was handed to the kernel side.
    assert attn.attn.position_embedding_type == 0
