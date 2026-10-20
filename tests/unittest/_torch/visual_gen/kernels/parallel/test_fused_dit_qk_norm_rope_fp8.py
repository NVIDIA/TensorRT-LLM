# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm::fused_dit_qk_norm_rope_fp8 and trtllm::dit_v_amax against the bf16 fused op."""

import math

import pytest
import torch
from utils.util import skip_pre_hopper

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)

pytestmark = skip_pre_hopper

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
HEAD_DIM = 128
EPS = 1e-6
QK_MUL = math.sqrt(HEAD_DIM**-0.5 / math.log(2.0))


def _inputs(tokens, num_heads, cos_rows, per_head_cos=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = num_heads * HEAD_DIM
    qkv = torch.randn(tokens, 3 * n, device="cuda", generator=g).bfloat16()
    q_w, k_w = (
        (1.0 + 0.1 * torch.randn(n, device="cuda", generator=g)).bfloat16() for _ in range(2)
    )
    cos_cols = n if per_head_cos else HEAD_DIM
    angle = torch.rand(cos_rows, cos_cols, device="cuda", generator=g) * 2 * math.pi
    cos, sin = angle.cos(), angle.sin()
    if per_head_cos:
        cos, sin = (t.view(cos_rows, num_heads, HEAD_DIM) for t in (cos, sin))
    return qkv, q_w, k_w, cos, sin


def _bf16_reference(qkv, num_heads, q_w, k_w, cos, sin, interleave):
    """Normalized and rotated bf16 Q, K and untouched V from the in-place bf16 op."""
    out = qkv.clone()
    torch.ops.trtllm.fused_dit_qk_norm_rope(
        out, num_heads, num_heads, num_heads, HEAD_DIM, EPS, q_w, k_w, None, None,
        cos, sin, -1, interleave, 0,
    )  # fmt: skip
    return out.view(qkv.shape[0], 3, num_heads, HEAD_DIM).unbind(1)


def _quant(x, mul):
    return (x.float() * mul).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def _assert_fp8_close(actual, expected):
    """Equal up to one e4m3 rounding step (different fp32 accumulation order)."""
    assert actual.dtype == E4M3 and actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.125, atol=2**-6)
    assert (actual.view(torch.uint8) == expected.view(torch.uint8)).float().mean() > 0.999


def _run(qkv, num_heads, q_w, k_w, cos, sin, interleave, v_amax=None):
    return torch.ops.trtllm.fused_dit_qk_norm_rope_fp8(
        qkv, num_heads, HEAD_DIM, EPS, q_w, k_w, cos, sin, interleave, QK_MUL, v_amax
    )


def _v_amax(qkv, num_heads):
    return qkv[:, 2 * num_heads * HEAD_DIM :].float().abs().amax()


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("num_heads", [10, 24, 40])
@pytest.mark.parametrize("batch", [1, 2])
def test_matches_bf16_op_shared_cos(interleave, num_heads, batch):
    """Shared [S, D] cos table, broadcast over the batch (pipelined kernel)."""
    seq = 1560
    qkv, q_w, k_w, cos, sin = _inputs(batch * seq, num_heads, seq)
    q8, k8, v8, v_descale = _run(qkv, num_heads, q_w, k_w, cos, sin, interleave)

    q_ref, k_ref, v_ref = _bf16_reference(qkv, num_heads, q_w, k_w, cos, sin, interleave)
    scale_v = _v_amax(qkv, num_heads) / E4M3_MAX
    torch.testing.assert_close(v_descale, scale_v.reshape(1), rtol=1e-6, atol=0)
    _assert_fp8_close(q8, _quant(q_ref, QK_MUL))
    _assert_fp8_close(k8, _quant(k_ref, QK_MUL))
    _assert_fp8_close(v8, _quant(v_ref, 1.0 / scale_v))


@pytest.mark.parametrize("interleave", [False, True])
def test_matches_bf16_op_per_head_cos(interleave):
    """Per-head [T, H, D] cos table (general kernel)."""
    tokens, num_heads = 777, 40
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens, per_head_cos=True, seed=1)
    q8, k8, v8, v_descale = _run(qkv, num_heads, q_w, k_w, cos, sin, interleave)

    q_ref, k_ref, v_ref = _bf16_reference(qkv, num_heads, q_w, k_w, cos, sin, interleave)
    _assert_fp8_close(q8, _quant(q_ref, QK_MUL))
    _assert_fp8_close(k8, _quant(k_ref, QK_MUL))
    _assert_fp8_close(v8, _quant(v_ref, 1.0 / v_descale))


def test_given_v_amax_sets_v_scale():
    """A V amax reduced elsewhere (e.g. across sequence-parallel ranks) sets the V scale."""
    tokens, num_heads = 1560, 40
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens, seed=2)
    group_amax = (_v_amax(qkv, num_heads) * 3).reshape(1)
    _, _, v8, v_descale = _run(qkv, num_heads, q_w, k_w, cos, sin, False, group_amax)

    scale_v = group_amax / E4M3_MAX
    torch.testing.assert_close(v_descale, scale_v, rtol=1e-6, atol=0)
    v_ref = qkv.view(tokens, 3, num_heads, HEAD_DIM)[:, 2]
    _assert_fp8_close(v8, _quant(v_ref, 1.0 / scale_v))


@pytest.mark.parametrize("tokens", [1, 33, 1560 * 2])
def test_dit_v_amax(tokens):
    num_heads = 40
    qkv, *_ = _inputs(tokens, num_heads, tokens, seed=3)
    amax = torch.ops.trtllm.dit_v_amax(qkv, num_heads, HEAD_DIM)
    assert amax.dtype == torch.float32 and amax.shape == (1,)
    assert amax.item() == _v_amax(qkv, num_heads).item()


def test_rejects_unsupported_layouts():
    tokens, num_heads = 64, 40
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens)
    with pytest.raises(RuntimeError, match="full-dim"):
        _run(qkv, num_heads, q_w[:HEAD_DIM], k_w[:HEAD_DIM], cos, sin, False)
    misaligned = torch.empty(tokens * qkv.shape[1] + 1, device="cuda", dtype=torch.bfloat16)
    misaligned = misaligned[1:].view_as(qkv).copy_(qkv)
    with pytest.raises(RuntimeError, match="aligned"):
        _run(misaligned, num_heads, q_w, k_w, cos, sin, False)


def test_fake_under_compile():
    tokens, num_heads = 256, 10
    qkv, q_w, k_w, cos, sin = _inputs(tokens, num_heads, tokens, seed=4)

    def fn(x):
        amax = torch.ops.trtllm.dit_v_amax(x, num_heads, HEAD_DIM)
        return _run(x, num_heads, q_w, k_w, cos, sin, False, amax)

    eager = fn(qkv)
    compiled = torch.compile(fn, fullgraph=True)(qkv)
    for a, b in zip(compiled, eager):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)
