# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exactness and input-contract tests for the H3 fused RoPE kernel."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_rope import (
    apply_minimax_h3_qk_norm_rope_bf16,
    apply_minimax_h3_rope_bf16,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _reference_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary_states = hidden_states[..., :rotary_dim]
    first_half, second_half = rotary_states.chunk(2, dim=-1)
    rotated_states = torch.cat((-second_half, first_half), dim=-1)
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    return torch.cat(
        (
            rotary_states * cos + rotated_states * sin,
            hidden_states[..., rotary_dim:],
        ),
        dim=-1,
    )


@requires_cuda
@pytest.mark.parametrize(
    "shape,rotary_dim",
    [((1, 257, 4, 128), 96), ((2, 13, 3, 8), 6), ((1, 17, 2, 8), 8), ((1, 0, 2, 8), 6)],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
def test_fused_rope_matches_eager_bf16_exactly(shape, rotary_dim, strided, table_dtype):
    torch.manual_seed(123)
    storage_shape = (*shape[:-1], shape[-1] * (2 if strided else 1))
    hidden_states = torch.randn(storage_shape, device="cuda", dtype=torch.bfloat16)
    if strided:
        hidden_states = hidden_states[..., ::2]
    angles = torch.randn((shape[1], rotary_dim * 2), device="cuda", dtype=table_dtype)
    cos, sin = angles.cos()[:, ::2], angles.sin()[:, ::2]
    expected = _reference_rotary_emb(hidden_states, cos, sin)
    actual = apply_minimax_h3_rope_bf16(hidden_states, cos, sin)
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()
    assert torch.equal(actual[..., rotary_dim:], hidden_states[..., rotary_dim:])


@requires_cuda
def test_fused_rope_compile_matches_eager_bf16_exactly():
    hidden_states = torch.randn((1, 17, 2, 128), device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((17, 96), device="cuda")
    cos, sin = angles.cos(), angles.sin()
    expected = _reference_rotary_emb(hidden_states, cos, sin)
    compiled = torch.compile(apply_minimax_h3_rope_bf16, fullgraph=True)
    actual = compiled(hidden_states, cos, sin)
    assert torch.equal(actual, expected)


@requires_cuda
@pytest.mark.parametrize(
    "invalid",
    ["rank", "dtype", "device", "shape", "odd", "wide", "table_dtype", "table_device", "grad"],
)
def test_fused_rope_rejects_invalid_inputs(invalid):
    hidden_states = torch.randn((1, 7, 2, 8), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((7, 6), device="cuda")
    sin = torch.zeros_like(cos)
    if invalid == "rank":
        hidden_states = hidden_states[0]
    elif invalid == "dtype":
        hidden_states = hidden_states.float()
    elif invalid == "device":
        hidden_states = hidden_states.cpu()
    elif invalid == "shape":
        sin = sin[:1]
    elif invalid == "odd":
        cos, sin = cos[:, :5], sin[:, :5]
    elif invalid == "wide":
        cos = torch.ones((7, 10), device="cuda")
        sin = torch.zeros_like(cos)
    elif invalid == "table_dtype":
        cos = cos.double()
    elif invalid == "table_device":
        cos = cos.cpu()
    elif invalid == "grad":
        hidden_states.requires_grad_(True)
    with pytest.raises(ValueError):
        apply_minimax_h3_rope_bf16(hidden_states, cos, sin)


def _reference_qk_norm_rope(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Eager ``RMSNormTPAware`` numerics followed by the eager split-half RoPE."""
    normalized = hidden_states.to(torch.float32)
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = weight * normalized.to(hidden_states.dtype)
    return _reference_rotary_emb(normalized, cos, sin)


@requires_cuda
@pytest.mark.parametrize(
    "shape,rotary_dim", [((1, 257, 4, 128), 96), ((2, 13, 3, 128), 128), ((1, 0, 2, 128), 96)]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("torch_reduction_order", [True, False])
def test_fused_qk_norm_rope_matches_eager_bf16(
    shape, rotary_dim, strided, table_dtype, torch_reduction_order
):
    torch.manual_seed(7)
    storage_shape = (*shape[:-1], shape[-1] * (2 if strided else 1))
    hidden_states = torch.randn(storage_shape, device="cuda", dtype=torch.bfloat16)
    if strided:
        hidden_states = hidden_states[..., ::2]
    weight = (1 + 0.1 * torch.randn(shape[-1], device="cuda")).to(torch.bfloat16)
    angles = torch.randn((shape[1], rotary_dim * 2), device="cuda", dtype=table_dtype)
    cos, sin = angles.cos()[:, ::2], angles.sin()[:, ::2]
    expected = _reference_qk_norm_rope(hidden_states, weight, cos, sin, 1e-5)
    actual = apply_minimax_h3_qk_norm_rope_bf16(
        hidden_states, weight, cos, sin, 1e-5, torch_reduction_order=torch_reduction_order
    )
    assert actual.is_contiguous()
    assert actual.shape == expected.shape
    if torch_reduction_order:
        # Torch's CUDA reduction order is reproduced, so the result is bit-exact.
        assert torch.equal(actual, expected)
    else:
        # A different FP32 summation order may move the variance by one ulp,
        # which flips a BF16 rounding for a small fraction of elements.
        assert torch.allclose(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)


@requires_cuda
def test_fused_qk_norm_rope_compile_matches_eager_bf16_exactly():
    hidden_states = torch.randn((1, 17, 2, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.rand(128, device="cuda").to(torch.bfloat16) + 0.5
    angles = torch.randn((17, 96), device="cuda")
    cos, sin = angles.cos(), angles.sin()
    expected = _reference_qk_norm_rope(hidden_states, weight, cos, sin, 1e-5)
    compiled = torch.compile(apply_minimax_h3_qk_norm_rope_bf16, fullgraph=True)
    assert torch.equal(compiled(hidden_states, weight, cos, sin, 1e-5), expected)


@requires_cuda
@pytest.mark.parametrize(
    "invalid",
    ["rank", "dtype", "weight_shape", "weight_dtype", "shape", "odd", "wide", "head_dim", "grad"],
)
def test_fused_qk_norm_rope_rejects_invalid_inputs(invalid):
    hidden_states = torch.randn((1, 7, 2, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((7, 96), device="cuda")
    sin = torch.zeros_like(cos)
    if invalid == "rank":
        hidden_states = hidden_states[0]
    elif invalid == "dtype":
        hidden_states = hidden_states.float()
    elif invalid == "weight_shape":
        weight = weight[:64]
    elif invalid == "weight_dtype":
        weight = weight.float()
    elif invalid == "shape":
        sin = sin[:1]
    elif invalid == "odd":
        cos, sin = cos[:, :95], sin[:, :95]
    elif invalid == "wide":
        cos = torch.ones((7, 130), device="cuda")
        sin = torch.zeros_like(cos)
    elif invalid == "head_dim":
        hidden_states = torch.randn((1, 7, 2, 64), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
        cos, sin = cos[:, :48], sin[:, :48]
    elif invalid == "grad":
        hidden_states.requires_grad_(True)
    with pytest.raises(ValueError):
        apply_minimax_h3_qk_norm_rope_bf16(hidden_states, weight, cos, sin, 1e-5)
