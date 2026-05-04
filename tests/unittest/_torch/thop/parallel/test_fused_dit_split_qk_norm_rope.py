# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 split Q/K fused full-dim RMSNorm + RoPE kernel.

import pytest
import torch

import tensorrt_llm  # noqa: F401  — triggers libth_common.so load (registers trtllm ops)

# ============================================================================
# Reference implementation (PyTorch fp32)
# ============================================================================


@torch.inference_mode()
def torch_ref(x_2d, weight, cos, sin, num_heads, head_dim, eps, interleave):
    """Reference: full-dim RMSNorm then RoPE on a single Q-or-K tensor.

    x_2d: [T, num_heads * head_dim], bf16
    weight: [num_heads * head_dim] bf16 (full-dim)
    cos, sin: [T, head_dim], float32
    """
    T = x_2d.shape[0]
    # Match kernel: do reduce/scale/RoPE in fp32, cast to bf16 only at the end.
    out = x_2d.float()
    var = out.pow(2).mean(-1, keepdim=True)
    out = out * torch.rsqrt(var + eps) * weight.float()

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


def _call_split_op(tensor, weight, cos, sin, num_heads, head_dim, eps, interleave=True):
    torch.ops.trtllm.fused_dit_split_norm_rope(
        tensor, num_heads, head_dim, eps, weight, cos, sin, interleave
    )


def _generate_cos_sin(num_tokens, head_dim, device, dtype=torch.float32):
    """Generate paired cos/sin (matches original DiT test format: freqs.cos/sin in [-1,1]).

    dtype: cos/sin output dtype — fp32 (default) or bf16 (B-2: kernel upcasts).
    """
    half_dim = head_dim // 2
    freqs = torch.randn(num_tokens, half_dim, device=device, dtype=torch.float32)
    freqs = freqs.repeat_interleave(2, dim=-1)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


# ============================================================================
# Full-dim norm tests (the only mode the kernel supports)
# ============================================================================


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("num_tokens", [1, 64, 1024])
@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16], ids=["fp32cos", "bf16cos"])
def test_full_dim_norm_self_attn(head_dim, num_heads, num_tokens, interleave, cos_dtype):
    """Full-dim RMSNorm + RoPE on contiguous 2D tensor (LTX-2 self-attn shape)."""
    device = "cuda"
    torch.random.manual_seed(42)

    hidden = num_heads * head_dim
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()

    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    cos, sin = _generate_cos_sin(num_tokens, head_dim, device, dtype=cos_dtype)
    eps = 1e-6

    _call_split_op(x, weight, cos, sin, num_heads, head_dim, eps, interleave=interleave)
    ref = torch_ref(x_copy, weight, cos, sin, num_heads, head_dim, eps, interleave=interleave)
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


def test_full_dim_norm_ltx2_video_shape():
    """LTX-2 video self-attn shape: H=32, D=128, S=12288."""
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

    _call_split_op(x, weight, cos, sin, num_heads, head_dim, 1e-6, interleave=True)
    ref = torch_ref(x_copy, weight, cos, sin, num_heads, head_dim, 1e-6, interleave=True)
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

    _call_split_op(x, weight, cos, sin, num_heads, head_dim, 1e-6, interleave=True)
    ref = torch_ref(x_copy, weight, cos, sin, num_heads, head_dim, 1e-6, interleave=True)
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
        _call_split_op(q_view, weight, cos, sin, num_heads, head_dim, 1e-6, interleave=True)


# ============================================================================
# LTX-2 production scenarios: rotate-half (SPLIT) + per-head cos
# Tolerance rtol=2e-2 atol=5e-3 (consistent with existing fuse-kernel checks).
# ============================================================================


def _torch_ref_per_head_cos(x_2d, weight, cos_2d, sin_2d, num_heads, head_dim, eps, interleave):
    """Reference for per-head cos: cos_2d shape [T, num_heads*head_dim]."""
    T = x_2d.shape[0]
    out = x_2d.float()
    var = out.pow(2).mean(-1, keepdim=True)
    out = out * torch.rsqrt(var + eps) * weight.float()
    out_4d = out.view(T, num_heads, head_dim)
    cos_3d = cos_2d.float().view(T, num_heads, head_dim)
    sin_3d = sin_2d.float().view(T, num_heads, head_dim)
    if interleave:
        rot = torch.empty_like(out_4d)
        rot[..., 0::2] = -out_4d[..., 1::2]
        rot[..., 1::2] = out_4d[..., 0::2]
        out_4d = out_4d * cos_3d + rot * sin_3d
    else:
        half = head_dim // 2
        x1 = out_4d[..., :half]
        x2 = out_4d[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        out_4d = out_4d * cos_3d + rot * sin_3d
    return out_4d.reshape(T, -1).to(x_2d.dtype)


def _make_per_head_cos(B, T, num_heads, head_dim, device, dtype=torch.float32):
    """LTX-2-style per-head cos: per (B, head, T) freqs, block-duplicated to head_dim.
    Returns cos_2d, sin_2d of shape (B*T, num_heads*head_dim)."""
    half = head_dim // 2
    freqs = torch.randn(B, num_heads, T, half, dtype=torch.float32, device=device)
    cos_h = freqs.cos()
    sin_h = freqs.sin()
    cos = torch.cat([cos_h, cos_h], dim=-1)  # (B, H, T, D)
    sin = torch.cat([sin_h, sin_h], dim=-1)
    cos_2d = (
        cos.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(B * T, num_heads * head_dim)
        .contiguous()
        .to(dtype)
    )
    sin_2d = (
        sin.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(B * T, num_heads * head_dim)
        .contiguous()
        .to(dtype)
    )
    return cos_2d, sin_2d


@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16], ids=["fp32cos", "bf16cos"])
@pytest.mark.parametrize(
    "label,B,T,num_heads,head_dim",
    [
        # video self-attn: 121 frames @ 768x1024 → 12288 tokens, 32 heads × 128 dim
        ("ltx2_video_self_attn", 2, 12288, 32, 128),
        # audio self-attn: 504 tokens, 32 heads × 64 dim
        ("ltx2_audio_self_attn", 2, 504, 32, 64),
        # text→video cross-attn (Q on video shape, K on text but using video cos for Q)
        ("ltx2_text_cross_video", 2, 12288, 32, 128),
        # AV a2v Q-only fused: video Q with video cos
        ("ltx2_av_a2v_q", 2, 12288, 32, 128),
    ],
)
def test_ltx2_split_rotate_half_per_head_cos(label, B, T, num_heads, head_dim, cos_dtype):
    """LTX-2 production scenario: rotate-half (SPLIT) RoPE + per-head cos + full-dim norm."""
    device = "cuda"
    torch.random.manual_seed(0)

    hidden = num_heads * head_dim
    x = torch.randn(B * T, hidden, dtype=torch.bfloat16, device=device) * 0.5
    x_copy = x.clone()
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    cos_2d, sin_2d = _make_per_head_cos(B, T, num_heads, head_dim, device, dtype=cos_dtype)
    eps = 1e-6

    torch.ops.trtllm.fused_dit_split_norm_rope(
        x,
        num_heads,
        head_dim,
        eps,
        weight,
        cos_2d,
        sin_2d,
        False,  # interleave=False → rotate-half (SPLIT)
    )
    ref = _torch_ref_per_head_cos(
        x_copy, weight, cos_2d, sin_2d, num_heads, head_dim, eps, interleave=False
    )
    torch.testing.assert_close(x, ref, rtol=2e-2, atol=5e-3)
