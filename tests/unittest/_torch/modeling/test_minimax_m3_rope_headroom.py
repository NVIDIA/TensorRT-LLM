# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 sizes its RoPE table past max_position_embeddings under speculative decoding.

A speculative step near max_seq_len (== max_position_embeddings for MiniMax M3) computes
RoPE for draft positions past the end of the model's range; the table must cover them.
"""

from types import SimpleNamespace

from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    _SPEC_DECODE_ROPE_HEADROOM,
    _rope_params_with_spec_headroom,
)


def _config():
    return PretrainedConfig(
        hidden_size=3072,
        num_attention_heads=48,
        head_dim=128,
        max_position_embeddings=1048576,
        rope_theta=5000000.0,
        rotary_dim=64,
    )


def test_table_matches_the_model_without_speculative_decoding():
    rope = _rope_params_with_spec_headroom(_config(), SimpleNamespace(spec_config=None))
    assert rope.max_positions == RopeParams.from_config(_config()).max_positions == 1048576


def test_table_has_headroom_with_speculative_decoding():
    rope = _rope_params_with_spec_headroom(_config(), SimpleNamespace(spec_config=object()))
    assert rope.max_positions == 1048576 + _SPEC_DECODE_ROPE_HEADROOM
    # Covers the furthest position a step can reach: max_seq_len - 1 plus a draft block.
    assert _SPEC_DECODE_ROPE_HEADROOM >= 2 * (7 + 1)
