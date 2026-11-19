# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy and input-contract tests for the H3 fused QK-norm + RoPE kernel."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.fused_rope import (
    apply_minimax_h3_qk_norm_rope_bf16,
    launch_minimax_h3_qk_norm_rope,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

HEAD_DIM = 128


def _rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE over the leading ``rotary_dim`` dims in the input dtype."""
    rotary_dim = cos.shape[-1]
    rotary_states = hidden_states[..., :rotary_dim]
    first_half, second_half = rotary_states.chunk(2, dim=-1)
    rotated_states = torch.cat((-second_half, first_half), dim=-1)
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    return torch.cat(
        (rotary_states * cos + rotated_states * sin, hidden_states[..., rotary_dim:]), dim=-1
    )


def _qk_norm_rope(hidden_states, weight, cos, sin, eps, dtype):
    """RMSNorm + RoPE with all math in ``dtype`` (the oracle for the single-rounding kernel)."""
    x = hidden_states.to(dtype)
    normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return _rotary_emb(weight.to(dtype) * normalized, cos.to(dtype), sin.to(dtype))


def _eager_qk_norm_rope(hidden_states, weight, cos, sin, eps):
    """The eager module path: FP32 variance, BF16 rounding after each step."""
    normalized = hidden_states.to(torch.float32)
    normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + eps)
    normalized = weight * normalized.to(hidden_states.dtype)
    return _rotary_emb(normalized, cos, sin)


def _packed_inputs(batch, seq, heads, rotary_dim, table_dtype, seed=7):
    torch.manual_seed(seed)
    qkv = torch.randn((batch, seq, 3 * heads * HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    weight_q = (1 + 0.1 * torch.randn(HEAD_DIM, device="cuda")).to(torch.bfloat16)
    weight_k = (1 + 0.1 * torch.randn(HEAD_DIM, device="cuda")).to(torch.bfloat16)
    angles = torch.randn((seq, rotary_dim * 2), device="cuda", dtype=table_dtype)
    cos, sin = angles.cos()[:, ::2], angles.sin()[:, ::2]
    return qkv, weight_q, weight_k, cos, sin


def _split_heads(qkv, heads):
    hd = heads * HEAD_DIM
    q = qkv[..., :hd].view(*qkv.shape[:2], heads, HEAD_DIM)
    k = qkv[..., hd : 2 * hd].view(*qkv.shape[:2], heads, HEAD_DIM)
    return q, k


def _assert_single_rounding_accuracy(actual, source, weight, cos, sin, eps):
    """The kernel rounds once from FP32: within one BF16 ulp of the output scale from an FP32
    oracle, and never farther from the FP64 oracle than the eager path's own rounding is."""
    if actual.numel() == 0:
        return
    fp32 = _qk_norm_rope(source, weight, cos, sin, eps, torch.float32).flatten(2)
    fp64 = _qk_norm_rope(source, weight, cos, sin, eps, torch.float64).flatten(2)
    eager = _eager_qk_norm_rope(source, weight, cos, sin, eps).flatten(2)
    ulp = fp32.abs().max() * 2**-7
    assert (actual.float() - fp32).abs().max() <= ulp
    fused_err = (actual.double() - fp64).abs().max()
    eager_err = (eager.double() - fp64).abs().max()
    assert fused_err <= eager_err + ulp


@requires_cuda
@pytest.mark.parametrize(
    "batch,seq,heads,rotary_dim", [(1, 257, 8, 96), (2, 13, 16, 128), (1, 5, 8, 32), (1, 0, 8, 96)]
)
@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
def test_fused_qk_norm_rope_matches_fp32_oracle(batch, seq, heads, rotary_dim, table_dtype):
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(batch, seq, heads, rotary_dim, table_dtype)
    v_before = qkv[..., 2 * heads * HEAD_DIM :].clone()
    q, k = apply_minimax_h3_qk_norm_rope_bf16(
        qkv, weight_q, weight_k, cos, sin, 1e-5, heads, HEAD_DIM
    )
    assert q.is_contiguous() and k.is_contiguous()
    assert q.shape == k.shape == (batch, seq, heads * HEAD_DIM)
    src_q, src_k = _split_heads(qkv, heads)
    _assert_single_rounding_accuracy(q, src_q, weight_q, cos, sin, 1e-5)
    _assert_single_rounding_accuracy(k, src_k, weight_k, cos, sin, 1e-5)
    # V columns are never touched: compare against the snapshot taken before the call.
    assert torch.equal(qkv[..., 2 * heads * HEAD_DIM :], v_before)


@requires_cuda
def test_fused_qk_norm_rope_passes_tail_dims_through_normalized():
    """Dims beyond the rotary width are normalized and weighted but not rotated."""
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 33, 8, 96, torch.float32)
    q, _ = apply_minimax_h3_qk_norm_rope_bf16(qkv, weight_q, weight_k, cos, sin, 1e-5, 8, HEAD_DIM)
    src_q, _ = _split_heads(qkv, 8)
    x = src_q.float()
    tail = (weight_q.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)))[..., 96:]
    actual_tail = q.view(1, 33, 8, HEAD_DIM)[..., 96:].float()
    assert (actual_tail - tail).abs().max() <= tail.abs().max() * 2**-7


@requires_cuda
@pytest.mark.parametrize("tokens_per_program,heads_per_program", [(4, 2), (8, 1), (2, 4)])
def test_fused_qk_norm_rope_tail_tokens_are_masked(tokens_per_program, heads_per_program):
    """Launch shapes that do not divide the token count exercise the tail-token mask."""
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 13, 8, 96, torch.float32)
    expected_q, expected_k = apply_minimax_h3_qk_norm_rope_bf16(
        qkv, weight_q, weight_k, cos, sin, 1e-5, 8, HEAD_DIM
    )
    q, k = torch.empty_like(expected_q), torch.empty_like(expected_k)
    launch_minimax_h3_qk_norm_rope(
        qkv,
        q,
        k,
        weight_q,
        weight_k,
        cos,
        sin,
        1e-5,
        8,
        HEAD_DIM,
        tokens_per_program=tokens_per_program,
        heads_per_program=heads_per_program,
    )
    # Same FP32 math in a different launch shape: results are identical.
    assert torch.equal(q, expected_q) and torch.equal(k, expected_k)


@requires_cuda
@pytest.mark.parametrize("heads", [12, 3, 6, 7, 28])
def test_default_head_grouping_handles_counts_eight_does_not_divide(heads):
    """The default group is the largest of 8/4/2/1 dividing the head count, never a non-power of two."""
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 9, heads, 96, torch.float32)
    q, k = apply_minimax_h3_qk_norm_rope_bf16(
        qkv, weight_q, weight_k, cos, sin, 1e-5, heads, HEAD_DIM
    )
    src_q, src_k = _split_heads(qkv, heads)
    _assert_single_rounding_accuracy(q, src_q, weight_q, cos, sin, 1e-5)
    _assert_single_rounding_accuracy(k, src_k, weight_k, cos, sin, 1e-5)


@requires_cuda
@pytest.mark.parametrize("tokens_per_program,heads_per_program", [(3, 1), (1, 0), (1, 3), (2, 16)])
def test_launch_rejects_invalid_launch_shapes(tokens_per_program, heads_per_program):
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 7, 8, 96, torch.float32)
    q = torch.empty((1, 7, 8 * HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        launch_minimax_h3_qk_norm_rope(
            qkv,
            q,
            q.clone(),
            weight_q,
            weight_k,
            cos,
            sin,
            1e-5,
            8,
            HEAD_DIM,
            tokens_per_program=tokens_per_program,
            heads_per_program=heads_per_program,
        )


@requires_cuda
def test_fused_qk_norm_rope_compiles_fullgraph_as_one_op():
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 17, 8, 96, torch.float32)
    expected_q, expected_k = apply_minimax_h3_qk_norm_rope_bf16(
        qkv, weight_q, weight_k, cos, sin, 1e-5, 8, HEAD_DIM
    )
    compiled = torch.compile(apply_minimax_h3_qk_norm_rope_bf16, fullgraph=True)
    q, k = compiled(qkv, weight_q, weight_k, cos, sin, 1e-5, 8, HEAD_DIM)
    # The custom op is opaque to Inductor, so compiled and eager launches are the same kernel.
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
        "strided",
        "zero_heads",
        "table_dtype",
        "strided_weight",
    ],
)
def test_fused_qk_norm_rope_rejects_invalid_inputs(invalid):
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 7, 8, 96, torch.float32)
    heads, head_dim = 8, HEAD_DIM
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
        qkv = qkv[..., : heads * HEAD_DIM]
    elif invalid == "grad":
        qkv.requires_grad_(True)
    elif invalid == "strided":
        qkv = qkv.expand(2, -1, -1)
    elif invalid == "zero_heads":
        heads = 0
    elif invalid == "table_dtype":
        cos = cos.double()
    elif invalid == "strided_weight":
        # A [128] view with stride 2 passes the shape check but the kernel reads contiguous storage.
        weight_q = torch.cat([weight_q, weight_q])[::2]
        assert weight_q.shape == (HEAD_DIM,) and not weight_q.is_contiguous()
    with pytest.raises(ValueError):
        apply_minimax_h3_qk_norm_rope_bf16(qkv, weight_q, weight_k, cos, sin, 1e-5, heads, head_dim)


@requires_cuda
@pytest.mark.parametrize("invalid", ["weight_shape", "columns", "strided"])
def test_custom_op_validates_inputs_without_the_wrapper(invalid):
    """Direct ``torch.ops.trtllm`` callers get the same ``ValueError`` as the wrapper."""
    qkv, weight_q, weight_k, cos, sin = _packed_inputs(1, 7, 8, 96, torch.float32)
    if invalid == "weight_shape":
        weight_q = weight_q[:64]
    elif invalid == "columns":
        qkv = qkv[..., : 8 * HEAD_DIM]
    elif invalid == "strided":
        qkv = qkv.expand(2, -1, -1)
    with pytest.raises(ValueError):
        torch.ops.trtllm.minimax_h3_qk_norm_rope(
            qkv, weight_q, weight_k, cos, sin, 1e-5, 8, HEAD_DIM
        )
    # The fake kernel validates too, so torch.compile reports the contract at trace time
    # (Dynamo wraps the fake kernel's ValueError in its own error type).
    with pytest.raises((ValueError, torch._dynamo.exc.TorchRuntimeError), match="H3|qkv"):
        torch.compile(torch.ops.trtllm.minimax_h3_qk_norm_rope, fullgraph=True)(
            qkv, weight_q, weight_k, cos, sin, 1e-5, 8, HEAD_DIM
        )
