# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the LTX-2 split full-dim RMSNorm-only kernel (no RoPE).
# Mirror of test_fused_dit_split_qk_norm_rope.py but for the norm-only path.

import pytest
import torch

import tensorrt_llm  # noqa: F401  -- triggers libth_common.so load (registers trtllm ops)

# ============================================================================
# Reference implementation (PyTorch fp32)
# ============================================================================


@torch.inference_mode()
def torch_ref(x_2d, weight, eps):
    """Reference: full-dim RMSNorm only on a single Q-or-K tensor.

    x_2d: [T, num_heads * head_dim], bf16
    weight: [num_heads * head_dim] bf16 (full-dim)
    """
    out = x_2d.float()
    var = out.pow(2).mean(-1, keepdim=True)
    out = out * torch.rsqrt(var + eps) * weight.float()
    return out.to(x_2d.dtype)


# ============================================================================
# Helper
# ============================================================================


def _call_norm_op(tensor, weight, num_heads, head_dim, eps):
    torch.ops.trtllm.fused_dit_split_norm(tensor, num_heads, head_dim, eps, weight)


# ============================================================================
# Full-dim norm tests
# ============================================================================


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("num_tokens", [1, 64, 1024])
def test_full_dim_norm_only(head_dim, num_heads, num_tokens):
    """Full-dim RMSNorm only (no RoPE) on contiguous 2D tensor."""
    device = "cuda"
    torch.random.manual_seed(42)

    hidden = num_heads * head_dim
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()

    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    eps = 1e-6

    _call_norm_op(x, weight, num_heads, head_dim, eps)
    ref = torch_ref(x_copy, weight, eps)
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


def test_full_dim_norm_ltx2_video_shape():
    """LTX-2 video self/cross-attn shape: H=32, D=128, T=12288."""
    device = "cuda"
    torch.random.manual_seed(0)

    num_tokens = 12288
    num_heads = 32
    head_dim = 128
    hidden = num_heads * head_dim

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0

    _call_norm_op(x, weight, num_heads, head_dim, 1e-6)
    ref = torch_ref(x_copy, weight, 1e-6)
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


def test_full_dim_norm_ltx2_audio_shape():
    """LTX-2 audio self/cross-attn shape: H=32, D=64, T=504."""
    device = "cuda"
    torch.random.manual_seed(0)

    num_tokens = 504
    num_heads = 32
    head_dim = 64
    hidden = num_heads * head_dim

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    x_copy = x.clone()
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0

    _call_norm_op(x, weight, num_heads, head_dim, 1e-6)
    ref = torch_ref(x_copy, weight, 1e-6)
    torch.testing.assert_close(x, ref, rtol=1e-2, atol=5e-3)


def test_full_dim_norm_rejects_non_contiguous():
    """fused_dit_split_norm requires contiguous SEPARATE_QKV input."""
    device = "cuda"
    num_heads, head_dim, num_tokens = 32, 128, 64
    qkv = torch.randn(num_tokens, 3 * num_heads * head_dim, dtype=torch.bfloat16, device=device)
    q_view = qkv[:, : num_heads * head_dim]
    assert not q_view.is_contiguous()

    weight = torch.randn(num_heads * head_dim, dtype=torch.bfloat16, device=device) * 5.0

    with pytest.raises(RuntimeError, match=r"contiguous"):
        _call_norm_op(q_view, weight, num_heads, head_dim, 1e-6)
