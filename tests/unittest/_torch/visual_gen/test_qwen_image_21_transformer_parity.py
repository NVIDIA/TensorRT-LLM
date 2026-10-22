# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image_21.transformer_qwen_image_21 import (
    QwenImage21Attention,
    QwenImage21TransformerBlock,
)


def test_qwen_image_21_attention_forward_is_finite_with_nondefault_qk_norm_weights():
    """Focused M7 parity guard for the attention norm scale path.

    A small deterministic attention module with non-default Q/K norm weights
    must produce finite, shape-preserving output. This test is intentionally
    independent of checkpoints so it catches wiring regressions in the native
    TRTLLM implementation before E2E generation.
    """

    torch.manual_seed(7)
    attention = QwenImage21Attention(dim=16, heads=2, dim_head=8, eps=1e-6)
    with torch.no_grad():
        attention.norm_q.weight.fill_(0.25)
        attention.norm_k.weight.fill_(0.5)

    hidden_states = torch.randn(2, 5, 16)
    output = attention(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_qwen_image_21_transformer_block_forward_is_finite_without_rope():
    """Synthetic one-block coverage for modulation, attention, and MLP paths."""

    torch.manual_seed(11)
    block = QwenImage21TransformerBlock(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        mlp_ratio=2,
        eps=1e-6,
    )
    hidden_states = torch.randn(2, 5, 16)
    modulation = torch.randn(2, 64)

    output = block(hidden_states=hidden_states, modulation=modulation)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
