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


def _packed_inputs(batch, seq, heads, rotary_dim, table_dtype, seed=7):
    torch.manual_seed(seed)
    qkv = torch.randn((batch, seq, 3 * heads * 128), device="cuda", dtype=torch.bfloat16)
    weight_q = (1 + 0.1 * torch.randn(128, device="cuda")).to(torch.bfloat16)
    weight_k = (1 + 0.1 * torch.randn(128, device="cuda")).to(torch.bfloat16)
    angles = torch.randn((seq, rotary_dim * 2), device="cuda", dtype=table_dtype)
    cos, sin = angles.cos()[:, ::2], angles.sin()[:, ::2]
    return qkv, weight_q, weight_k, cos, sin


def _reference_packed(qkv, weight_q, weight_k, cos, sin, heads, eps):
    hd = heads * 128
    q = qkv[..., :hd].view(*qkv.shape[:2], heads, 128)
    k = qkv[..., hd : 2 * hd].view(*qkv.shape[:2], heads, 128)
    return (
        _reference_qk_norm_rope(q, weight_q, cos, sin, eps).flatten(2),
        _reference_qk_norm_rope(k, weight_k, cos, sin, eps).flatten(2),
    )


@requires_cuda
@pytest.mark.parametrize(
    "batch,seq,heads,rotary_dim", [(1, 257, 8, 96), (2, 13, 16, 128), (1, 5, 8, 32), (1, 0, 8, 96)]
)
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
def test_fused_qk_norm_rope_matches_eager_bf16_exactly(batch, seq, heads, rotary_dim, table_dtype):
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(batch, seq, heads, rotary_dim, table_dtype)
    expected_q, expected_k = _reference_packed(qkv, weight_q, weight_k, cos, sin, heads, 1e-5)
    q, k = apply_minimax_h3_qk_norm_rope_bf16(qkv, weight_q, weight_k, cos, sin, 1e-5, heads, 128)
    assert q.is_contiguous() and k.is_contiguous()
    assert torch.equal(q, expected_q)
    assert torch.equal(k, expected_k)
    # V columns are never touched.
    assert torch.equal(qkv[..., 2 * heads * 128 :], qkv[..., 2 * heads * 128 :].clone())


@requires_cuda
def test_fused_qk_norm_rope_single_rounding_is_close_not_exact():
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 1029, 8, 96, torch.float32)
    expected_q, expected_k = _reference_packed(qkv, weight_q, weight_k, cos, sin, 8, 1e-5)
    q, k = apply_minimax_h3_qk_norm_rope_bf16(
        qkv, weight_q, weight_k, cos, sin, 1e-5, 8, 128, exact_rounding=False
    )
    # FP32 math with one final rounding: the eager path rounds four times, so the two differ in
    # many elements by rounding noise, but the error stays at the BF16 scale of the activations.
    for actual, expected in ((q, expected_q), (k, expected_k)):
        diff = (actual.float() - expected.float()).abs()
        assert diff.max() <= expected.float().abs().max() * 2**-6
        assert diff.norm() / expected.float().norm() < 1e-2
    assert not torch.equal(q, expected_q)


@requires_cuda
def test_fused_qk_norm_rope_compile_matches_eager_bf16_exactly():
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 17, 8, 96, torch.float32)
    expected_q, expected_k = _reference_packed(qkv, weight_q, weight_k, cos, sin, 8, 1e-5)
    compiled = torch.compile(apply_minimax_h3_qk_norm_rope_bf16, fullgraph=True)
    q, k = compiled(qkv, weight_q, weight_k, cos, sin, 1e-5, 8, 128)
    assert torch.equal(q, expected_q) and torch.equal(k, expected_k)


@requires_cuda
@pytest.mark.parametrize(
    "invalid",
    [
        "rank",
        "dtype",
        "weight_shape",
        "weight_dtype",
        "shape",
        "rot_multiple",
        "wide",
        "head_dim",
        "columns",
        "grad",
    ],
)
def test_fused_qk_norm_rope_rejects_invalid_inputs(invalid):
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 7, 8, 96, torch.float32)
    heads, head_dim = 8, 128
    if invalid == "rank":
        qkv = qkv[0]
    elif invalid == "dtype":
        qkv = qkv.float()
    elif invalid == "weight_shape":
        weight_q = weight_q[:64]
    elif invalid == "weight_dtype":
        weight_k = weight_k.float()
    elif invalid == "shape":
        sin = sin[:1]
    elif invalid == "rot_multiple":
        cos, sin = cos[:, :80], sin[:, :80]
    elif invalid == "wide":
        cos = torch.ones((7, 160), device="cuda")
        sin = torch.zeros_like(cos)
    elif invalid == "head_dim":
        head_dim = 64
    elif invalid == "columns":
        qkv = qkv[..., : heads * 128]
    elif invalid == "grad":
        qkv.requires_grad_(True)
    with pytest.raises(ValueError):
        apply_minimax_h3_qk_norm_rope_bf16(qkv, weight_q, weight_k, cos, sin, 1e-5, heads, head_dim)
