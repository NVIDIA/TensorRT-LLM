# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exactness and input-contract tests for the H3 fused RoPE kernel."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_rope import apply_minimax_h3_rope_bf16

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
