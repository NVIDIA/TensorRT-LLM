# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
"""Focused Qwen-Image 2.1 normalization state-dict contract tests."""

import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image_21.transformer_qwen_image_21 import (
    QwenImage21Attention,
    QwenImage21RMSNorm,
    QwenImage21TextProjection,
    QwenImage21ZeroCenterRMSNorm,
)


def _regular_rmsnorm_reference(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + eps) * weight


def _zero_center_rmsnorm_reference(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + eps) * (weight + 1)


def test_attention_qk_norm_uses_regular_rmsnorm_checkpoint_scale() -> None:
    torch.manual_seed(0)
    attention = QwenImage21Attention(dim=16, heads=2, dim_head=4, eps=1e-6)

    assert isinstance(attention.norm_q, QwenImage21RMSNorm)
    assert isinstance(attention.norm_k, QwenImage21RMSNorm)
    assert not isinstance(attention.norm_q, QwenImage21ZeroCenterRMSNorm)
    assert not isinstance(attention.norm_k, QwenImage21ZeroCenterRMSNorm)

    weight = torch.tensor([0.25, 0.75, 1.25, 1.75], dtype=torch.float32)
    attention.norm_q.weight.data.copy_(weight)
    attention.norm_k.weight.data.copy_(weight.flip(0))
    hidden_states = torch.randn(2, 3, 2, 4, dtype=torch.float32)

    q_actual = attention.norm_q(hidden_states)
    q_expected = _regular_rmsnorm_reference(hidden_states, weight, eps=attention.norm_q.eps)
    k_actual = attention.norm_k(hidden_states)
    k_expected = _regular_rmsnorm_reference(hidden_states, weight.flip(0), eps=attention.norm_k.eps)

    torch.testing.assert_close(q_actual, q_expected)
    torch.testing.assert_close(k_actual, k_expected)

    zero_center_wrong = _zero_center_rmsnorm_reference(hidden_states, weight, eps=attention.norm_q.eps)
    assert not torch.allclose(q_actual, zero_center_wrong)


def test_attention_qk_norm_state_dict_keys_load_without_rewrite() -> None:
    attention = QwenImage21Attention(dim=16, heads=2, dim_head=4, eps=1e-6)
    state_dict = attention.state_dict()

    assert "norm_q.weight" in state_dict
    assert "norm_k.weight" in state_dict

    state_dict["norm_q.weight"] = torch.tensor([0.5, 1.0, 1.5, 2.0])
    state_dict["norm_k.weight"] = torch.tensor([2.0, 1.5, 1.0, 0.5])
    reloaded = QwenImage21Attention(dim=16, heads=2, dim_head=4, eps=1e-6)
    reloaded.load_state_dict(state_dict)

    torch.testing.assert_close(reloaded.norm_q.weight, state_dict["norm_q.weight"])
    torch.testing.assert_close(reloaded.norm_k.weight, state_dict["norm_k.weight"])


def test_text_projection_keeps_zero_center_rmsnorm_contract() -> None:
    projection = QwenImage21TextProjection(context_in_dim=4, hidden_size=8, eps=1e-6)
    assert isinstance(projection.text_norm, QwenImage21ZeroCenterRMSNorm)

    hidden_states = torch.tensor(
        [[[1.0, 2.0, -3.0, 4.0], [0.5, -0.25, 0.75, -1.0]]],
        dtype=torch.float32,
    )
    projection.text_norm.weight.data.copy_(torch.tensor([0.0, 0.25, -0.5, 1.0]))

    actual = projection.text_norm(hidden_states)
    expected = _zero_center_rmsnorm_reference(
        hidden_states,
        projection.text_norm.weight.detach(),
        eps=projection.text_norm.eps,
    )
    torch.testing.assert_close(actual, expected)
