# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical KV-length changes force the unfused-RoPE path.

When physical and logical KV lengths diverge, the fused path can no longer
derive rotary positions from physical KV length. The unfused path consumes the
engine's logical ``position_ids`` instead.
"""

import torch

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    KvCacheCompressionConfig,
    TriAttentionKvCacheCompressionConfig,
)


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


def test_physical_length_preserving_compression_keeps_fused_rope() -> None:
    model_config = ModelConfig(
        kv_cache_compression_config=KvCacheCompressionConfig(algorithm="test")
    )
    attn = _make_attention(model_config)

    assert attn.rope_fusion is True


def test_physical_kv_length_change_forces_unfused_rope() -> None:
    model_config = ModelConfig(
        kv_cache_compression_config=TriAttentionKvCacheCompressionConfig(
            calibration_path="/calib/test.pt"
        )
    )
    attn = _make_attention(model_config)

    assert attn.rope_fusion is False
