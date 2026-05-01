# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 split Q/K fused RMSNorm+RoPE kernel.
# Phase 1: full_dim_norm=true, do_norm=true (LTX-2 self-attention).

import pytest
import torch

import tensorrt_llm  # noqa: F401  — triggers libth_common.so load (registers trtllm ops)

# ============================================================================
# Reference implementation (PyTorch fp32)
# ============================================================================


@torch.inference_mode()
def torch_ref(x_2d, weight, cos, sin, num_heads, head_dim, eps, full_dim_norm, do_norm, interleave):
    """Reference: do norm (optional) then RoPE on a single Q-or-K tensor.

    x_2d: [T, num_heads * head_dim], bf16
    weight: full_dim → [num_heads * head_dim] bf16, per_head → [head_dim]
    cos, sin: [T, head_dim], float32
    """
    T = x_2d.shape[0]
    # Match kernel: do reduce/scale/RoPE in fp32, cast to bf16 only at the end.
    out = x_2d.float()

    if do_norm:
        if full_dim_norm:
            var = out.pow(2).mean(-1, keepdim=True)
            out = out * torch.rsqrt(var + eps) * weight.float()
        else:
            x_4d = out.view(T, num_heads, head_dim)
            var = x_4d.pow(2).mean(-1, keepdim=True)
            x_4d = x_4d * torch.rsqrt(var + eps) * weight.float()
            out = x_4d.reshape(T, -1)

    # RoPE: cos/sin are [T, head_dim], broadcast over all heads.
    out_4d = out.view(T, num_heads, head_dim)
    cos_3d = cos.unsqueeze(1)
    sin_3d = sin.unsqueeze(1)
    if interleave:
        # pair (2i, 2i+1) — LTX-2 INTERLEAVED
        rot = torch.empty_like(out_4d)
        rot[..., 0::2] = -out_4d[..., 1::2]
        rot[..., 1::2] = out_4d[..., 0::2]
        out_4d = out_4d * cos_3d + rot * sin_3d
    else:
        # rotate_half: pair (i, i+D/2)
        half = head_dim // 2
        x1 = out_4d[..., :half]
        x2 = out_4d[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        out_4d = out_4d * cos_3d + rot * sin_3d

    return out_4d.reshape(T, -1).to(x_2d.dtype)


# ============================================================================
# Helper
# ============================================================================


def _call_split_op(
    tensor,
    weight,
    cos,
    sin,
    num_heads,
    head_dim,
    eps,
    full_dim_norm=True,
    do_norm=True,
    interleave=True,
):
    torch.ops.trtllm.fused_dit_split_norm_rope(
        tensor, num_heads, head_dim, eps, weight, cos, sin, full_dim_norm, do_norm, interleave
    )


def _generate_cos_sin(num_tokens, head_dim, device):
    """Generate paired cos/sin (matches original DiT test format: freqs.cos/sin in [-1,1])."""
    half_dim = head_dim // 2
    freqs = torch.randn(num_tokens, half_dim, device=device, dtype=torch.float32)
    freqs = freqs.repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


# ============================================================================
# Phase 1 tests: full_dim_norm=true, do_norm=true
# ============================================================================


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("num_tokens", [1, 64, 1024])
@pytest.mark.parametrize("interleave", [True, False])
def test_full_dim_norm_self_attn(head_dim, num_heads, num_tokens, interleave):
    """Full-dim RMSNorm + RoPE on contiguous 2D tensor (LTX-2 self-attn shape)."""
    device = "cuda"
    torch.random.manual_seed(42)

    hidden = num_heads * head_dim
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()

    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    cos, sin = _generate_cos_sin(num_tokens, head_dim, device)
    eps = 1e-6

    _call_split_op(
        x,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        eps,
        full_dim_norm=True,
        do_norm=True,
        interleave=interleave,
    )
    ref = torch_ref(
        x_copy,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        eps,
        full_dim_norm=True,
        do_norm=True,
        interleave=interleave,
    )
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


def test_full_dim_norm_ltx2_video_shape():
    """LTX-2 video self-attn shape: H=32, D=128, S=12288 (121 frames @ 768x1024).
    Single shape sanity check at production size."""
    device = "cuda"
    torch.random.manual_seed(0)

    num_tokens = 12288
    num_heads = 32
    head_dim = 128
    hidden = num_heads * head_dim

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    cos, sin = _generate_cos_sin(num_tokens, head_dim, device)

    _call_split_op(
        x,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        1e-6,
        full_dim_norm=True,
        do_norm=True,
        interleave=True,
    )
    ref = torch_ref(
        x_copy,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        1e-6,
        full_dim_norm=True,
        do_norm=True,
        interleave=True,
    )
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


def test_full_dim_norm_ltx2_audio_shape():
    """LTX-2 audio self-attn shape: H=32, D=64, S=504."""
    device = "cuda"
    torch.random.manual_seed(0)

    num_tokens = 504
    num_heads = 32
    head_dim = 64
    hidden = num_heads * head_dim

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    cos, sin = _generate_cos_sin(num_tokens, head_dim, device)

    _call_split_op(
        x,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        1e-6,
        full_dim_norm=True,
        do_norm=True,
        interleave=True,
    )
    ref = torch_ref(
        x_copy,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        1e-6,
        full_dim_norm=True,
        do_norm=True,
        interleave=True,
    )
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("num_tokens", [1, 64, 1024])
@pytest.mark.parametrize("interleave", [True, False])
def test_per_head_norm_self_attn(head_dim, num_heads, num_tokens, interleave):
    """Per-head RMSNorm + RoPE (FLUX-style: weight=[head_dim], reduce within each head)."""
    device = "cuda"
    torch.random.manual_seed(42)

    hidden = num_heads * head_dim
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()

    # per-head weight shape = [head_dim]
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    cos, sin = _generate_cos_sin(num_tokens, head_dim, device)
    eps = 1e-6

    _call_split_op(
        x,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        eps,
        full_dim_norm=False,
        do_norm=True,
        interleave=interleave,
    )
    ref = torch_ref(
        x_copy,
        weight,
        cos,
        sin,
        num_heads,
        head_dim,
        eps,
        full_dim_norm=False,
        do_norm=True,
        interleave=interleave,
    )
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


def test_full_dim_norm_rejects_non_contiguous():
    """Split kernel only supports contiguous SEPARATE_QKV input.
    For FUSE_QKV packed-view, callers must use fused_dit_qk_norm_rope instead."""
    device = "cuda"
    num_heads, head_dim, num_tokens = 32, 128, 64
    qkv = torch.randn(num_tokens, 3 * num_heads * head_dim, dtype=torch.bfloat16, device=device)
    q_view = qkv[:, : num_heads * head_dim]
    assert not q_view.is_contiguous()

    weight = torch.randn(num_heads * head_dim, dtype=torch.bfloat16, device=device) * 5.0
    cos, sin = _generate_cos_sin(num_tokens, head_dim, device)

    with pytest.raises(RuntimeError, match=r"contiguous"):
        _call_split_op(
            q_view,
            weight,
            cos,
            sin,
            num_heads,
            head_dim,
            1e-6,
            full_dim_norm=True,
            do_norm=True,
            interleave=True,
        )
