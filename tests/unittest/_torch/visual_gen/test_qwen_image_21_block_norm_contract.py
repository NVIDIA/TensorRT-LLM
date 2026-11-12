# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
"""Qwen-Image 2.1 block-level attention normalization contracts."""

import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image_21.transformer_qwen_image_21 import (
    QwenImage21RMSNorm,
    QwenImage21TransformerBlock,
    QwenImage21ZeroCenterRMSNorm,
)


def test_transformer_block_attention_norms_are_regular_rmsnorm() -> None:
    block = QwenImage21TransformerBlock(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=4,
        mlp_ratio=2,
        eps=1e-6,
    )

    assert isinstance(block.attn.norm_q, QwenImage21RMSNorm)
    assert isinstance(block.attn.norm_k, QwenImage21RMSNorm)
    assert not isinstance(block.attn.norm_q, QwenImage21ZeroCenterRMSNorm)
    assert not isinstance(block.attn.norm_k, QwenImage21ZeroCenterRMSNorm)

    state_dict = block.state_dict()
    assert state_dict["attn.norm_q.weight"].shape == (4,)
    assert state_dict["attn.norm_k.weight"].shape == (4,)
    torch.testing.assert_close(state_dict["attn.norm_q.weight"], torch.ones(4))
    torch.testing.assert_close(state_dict["attn.norm_k.weight"], torch.ones(4))


def test_transformer_block_loads_attention_norm_weights_verbatim() -> None:
    block = QwenImage21TransformerBlock(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=4,
        mlp_ratio=2,
        eps=1e-6,
    )
    state_dict = block.state_dict()
    state_dict["attn.norm_q.weight"] = torch.tensor([0.125, 0.5, 1.0, 1.5])
    state_dict["attn.norm_k.weight"] = torch.tensor([1.5, 1.0, 0.5, 0.125])

    reloaded = QwenImage21TransformerBlock(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=4,
        mlp_ratio=2,
        eps=1e-6,
    )
    reloaded.load_state_dict(state_dict)

    torch.testing.assert_close(reloaded.attn.norm_q.weight, state_dict["attn.norm_q.weight"])
    torch.testing.assert_close(reloaded.attn.norm_k.weight, state_dict["attn.norm_k.weight"])
