# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image_21.transformer_qwen_image_21 import (
    QwenImage21Attention,
    QwenImage21TextProjection,
)


def _regular_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + eps)
    return (hidden_states * rrms * weight.float()).to(input_dtype)


def _zero_center_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor,
                          eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + eps)
    return (hidden_states * rrms * (weight.float() + 1)).to(input_dtype)


def test_qwen_image_21_attention_qk_norm_uses_regular_rms_scale():
    """Qwen-Image 2.1 attention Q/K norms store the effective RMS scale.

    The text projection uses zero-centered RMSNorm, but the official Qwen-Image
    2.1 attention module uses regular RMSNorm for norm_q/norm_k.  A non-default
    weight catches accidental zero-centered behavior, where the effective scale
    would be ``weight + 1`` and can severely alter attention logits.
    """

    eps = 1e-6
    attention = QwenImage21Attention(dim=16, heads=2, dim_head=8, eps=eps)
    hidden_states = torch.linspace(-2.0, 2.0, steps=2 * 3 * 2 * 8).reshape(
        2, 3, 2, 8)

    with torch.no_grad():
        attention.norm_q.weight.fill_(0.25)
        attention.norm_k.weight.fill_(0.5)

    torch.testing.assert_close(
        attention.norm_q(hidden_states),
        _regular_rms_norm(hidden_states, attention.norm_q.weight, eps),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        attention.norm_k(hidden_states),
        _regular_rms_norm(hidden_states, attention.norm_k.weight, eps),
        rtol=1e-6,
        atol=1e-6,
    )


def test_qwen_image_21_text_projection_keeps_zero_center_rms_scale():
    """The VLM text projection still uses zero-centered RMSNorm semantics."""

    eps = 1e-6
    projection = QwenImage21TextProjection(context_in_dim=8,
                                           hidden_size=8,
                                           eps=eps)
    hidden_states = torch.linspace(-1.5, 1.5, steps=2 * 4 * 8).reshape(2, 4, 8)

    with torch.no_grad():
        projection.text_norm.weight.fill_(0.25)

    torch.testing.assert_close(
        projection.text_norm(hidden_states),
        _zero_center_rms_norm(hidden_states, projection.text_norm.weight, eps),
        rtol=1e-6,
        atol=1e-6,
    )
