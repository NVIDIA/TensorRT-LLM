# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F

# ============================================================================
# Reference implementations
# ============================================================================


def _apply_interleaved_rope(x, cos, sin):
    """Interleaved RoPE: pair (x[2i], x[2i+1])."""
    x_rot = torch.empty_like(x, dtype=torch.float32)
    x_rot[:, 0::2] = -x[:, 1::2].float()
    x_rot[:, 1::2] = x[:, 0::2].float()
    return (x.float() * cos + x_rot * sin).to(x.dtype)


def _apply_rotate_half_rope(x, cos, sin, head_dim, num_heads):
    """rotate_half RoPE: pair (x[i], x[i+D/2]) within each head."""
    T = x.shape[0]
    x_4d = x.view(T, num_heads, head_dim).float()
    half = head_dim // 2
    x1 = x_4d[..., :half]
    x2 = x_4d[..., half:]
    # cos/sin are [T, H*D] → reshape to [T, H, D]
    cos_4d = cos.view(T, num_heads, head_dim)
    sin_4d = sin.view(T, num_heads, head_dim)
    out = torch.empty_like(x_4d)
    out[..., :half] = x1 * cos_4d[..., :half] - x2 * sin_4d[..., :half]
    out[..., half:] = x2 * cos_4d[..., half:] + x1 * sin_4d[..., half:]
    return out.reshape(T, -1).to(x.dtype)


@torch.inference_mode()
def torch_ref_per_head(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    q_add_weight,
    k_add_weight,
    cos_emb,
    sin_emb,
    num_txt_tokens,
    k_cos_emb,
    k_sin_emb,
    interleave,
):
    """Reference: per-head RMSNorm + RoPE (FLUX, Cosmos3)."""
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim

    q = qkv[:, :q_size].clone()
    k = qkv[:, q_size : q_size + k_size].clone()
    v = qkv[:, q_size + k_size :].clone()

    # Per-head RMSNorm
    q_4d = q.view(num_tokens, num_heads_q, head_dim)
    k_4d = k.view(num_tokens, num_heads_k, head_dim)

    # Stay in float32 through RoPE (matches kernel which does norm+RoPE in f32
    # before the final bf16 store).
    if num_txt_tokens > 0 and q_add_weight is not None:
        txt_q = F.rms_norm(q_4d[:num_txt_tokens].float(), (head_dim,), q_add_weight.float(), eps)
        img_q = F.rms_norm(q_4d[num_txt_tokens:].float(), (head_dim,), q_weight.float(), eps)
        q_4d = torch.cat([txt_q, img_q], dim=0)
        txt_k = F.rms_norm(k_4d[:num_txt_tokens].float(), (head_dim,), k_add_weight.float(), eps)
        img_k = F.rms_norm(k_4d[num_txt_tokens:].float(), (head_dim,), k_weight.float(), eps)
        k_4d = torch.cat([txt_k, img_k], dim=0)
    else:
        q_4d = F.rms_norm(q_4d.float(), (head_dim,), q_weight.float(), eps)
        k_4d = F.rms_norm(k_4d.float(), (head_dim,), k_weight.float(), eps)

    q = q_4d.reshape(num_tokens, q_size)
    k = k_4d.reshape(num_tokens, k_size)

    # Expand cos/sin per head
    cos_q = cos_emb.unsqueeze(1).expand(-1, num_heads_q, -1).reshape(num_tokens, q_size)
    sin_q = sin_emb.unsqueeze(1).expand(-1, num_heads_q, -1).reshape(num_tokens, q_size)

    actual_k_cos = k_cos_emb if k_cos_emb is not None else cos_emb
    actual_k_sin = k_sin_emb if k_sin_emb is not None else sin_emb
    cos_k = actual_k_cos.unsqueeze(1).expand(-1, num_heads_k, -1).reshape(num_tokens, k_size)
    sin_k = actual_k_sin.unsqueeze(1).expand(-1, num_heads_k, -1).reshape(num_tokens, k_size)

    if interleave:
        q = _apply_interleaved_rope(q, cos_q, sin_q)
        k = _apply_interleaved_rope(k, cos_k, sin_k)
    else:
        q = _apply_rotate_half_rope(q, cos_q, sin_q, head_dim, num_heads_q)
        k = _apply_rotate_half_rope(k, cos_k, sin_k, head_dim, num_heads_k)

    # Convert to bf16 at the end (matches kernel's final bf16 store)
    return torch.cat([q.to(qkv.dtype), k.to(qkv.dtype), v], dim=1)


# ============================================================================
# Helpers
# ============================================================================


def _generate_cos_sin(num_tokens, head_dim, device):
    """Generate paired cos/sin embeddings matching FLUX's repeat_interleave format."""
    half_dim = head_dim // 2
    freqs = torch.randn(num_tokens, half_dim, device=device, dtype=torch.float32)
    freqs = freqs.repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


def _call_fused_kernel(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    q_add_weight,
    k_add_weight,
    cos_emb,
    sin_emb,
    num_txt_tokens,
    interleave=True,
    tokens_per_batch=0,
):
    """Call the fused DiT QK Norm + RoPE kernel (per-head only)."""
    torch.ops.trtllm.fused_dit_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        cos_emb,
        sin_emb,
        num_txt_tokens,
        interleave,
        tokens_per_batch,
    )


# ============================================================================
# Per-head norm tests (FLUX, Cosmos3)
# ============================================================================


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("num_heads", [24, 48])
@pytest.mark.parametrize("num_tokens", [1, 64, 512])
@pytest.mark.parametrize("dual_stream_config", [(-1, False), (128, True), (256, True)])
def test_per_head_interleaved(head_dim, num_heads, num_tokens, dual_stream_config):
    """Per-head norm + interleaved RoPE (FLUX pattern)."""
    device = "cuda"
    num_txt_tokens, has_add_weights = dual_stream_config
    if num_txt_tokens >= num_tokens:
        pytest.skip("num_txt_tokens >= num_tokens")

    torch.random.manual_seed(42)
    hidden_size = 3 * num_heads * head_dim
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_copy = qkv.clone()

    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    q_add = (
        torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
        if has_add_weights
        else None
    )
    k_add = (
        torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
        if has_add_weights
        else None
    )
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        q_add,
        k_add,
        cos_emb,
        sin_emb,
        num_txt_tokens,
    )

    ref = torch_ref_per_head(
        qkv_copy,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        q_add,
        k_add,
        cos_emb,
        sin_emb,
        num_txt_tokens,
        None,
        None,
        True,
    )
    torch.testing.assert_close(qkv, ref, rtol=1e-2, atol=5e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [24])
@pytest.mark.parametrize("num_tokens", [1, 64, 512])
def test_per_head_rotate_half(head_dim, num_heads, num_tokens):
    """Per-head norm + rotate_half RoPE (Cosmos3 pattern)."""
    device = "cuda"
    torch.random.manual_seed(42)
    hidden_size = 3 * num_heads * head_dim
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_copy = qkv.clone()

    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_emb,
        sin_emb,
        -1,
        interleave=False,
    )

    ref = torch_ref_per_head(
        qkv_copy,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_emb,
        sin_emb,
        -1,
        None,
        None,
        False,
    )
    torch.testing.assert_close(qkv, ref, rtol=1e-2, atol=5e-3)


# ============================================================================
# Batch size > 1 tests
# ============================================================================


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [64, 256])
def test_per_head_batched(batch_size, num_tokens):
    """Per-head norm + interleaved RoPE with batch_size > 1.

    Simulates the batch-flattening + cos/sin tiling done by
    Attention.apply_qk_norm_rope: [B, S, D] → [B*S, D].
    """
    device = "cuda"
    num_heads = 24
    head_dim = 128

    torch.random.manual_seed(42)
    total_tokens = batch_size * num_tokens
    hidden_size = 3 * num_heads * head_dim
    qkv = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_copy = qkv.clone()

    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    # Generate cos/sin for one batch element, then tile
    cos_single, sin_single = _generate_cos_sin(num_tokens, head_dim, device)
    cos_emb = cos_single.repeat(batch_size, 1)
    sin_emb = sin_single.repeat(batch_size, 1)

    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_emb,
        sin_emb,
        -1,
    )

    ref = torch_ref_per_head(
        qkv_copy,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_emb,
        sin_emb,
        -1,
        None,
        None,
        True,
    )
    torch.testing.assert_close(qkv, ref, rtol=1e-2, atol=5e-3)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_per_head_batched_dual_stream(batch_size):
    """Batch > 1 with dual-stream (FLUX pattern): tokens_per_batch selects
    text vs image norm weights using modulo on the flattened token index."""
    device = "cuda"
    num_heads = 24
    head_dim = 128
    num_tokens = 256
    num_txt_tokens = 64

    torch.random.manual_seed(42)
    total_tokens = batch_size * num_tokens
    hidden_size = 3 * num_heads * head_dim
    qkv = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_copy = qkv.clone()

    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    q_add = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_add = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 5.0

    cos_single, sin_single = _generate_cos_sin(num_tokens, head_dim, device)
    cos_emb = cos_single.repeat(batch_size, 1)
    sin_emb = sin_single.repeat(batch_size, 1)

    # tokens_per_batch = num_tokens so kernel uses modulo for local index
    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        q_add,
        k_add,
        cos_emb,
        sin_emb,
        num_txt_tokens,
        tokens_per_batch=num_tokens,
    )

    # Reference: process each batch element separately
    ref = qkv_copy.clone()
    for b in range(batch_size):
        start = b * num_tokens
        end = start + num_tokens
        batch_ref = torch_ref_per_head(
            qkv_copy[start:end],
            num_heads,
            num_heads,
            num_heads,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            q_add,
            k_add,
            cos_single,
            sin_single,
            num_txt_tokens,
            None,
            None,
            True,
        )
        ref[start:end] = batch_ref

    torch.testing.assert_close(qkv, ref, rtol=1e-2, atol=5e-3)


# ============================================================================
# V unchanged tests
# ============================================================================


@pytest.mark.parametrize("head_dim", [64, 128])
def test_per_head_v_unchanged(head_dim):
    """Verify V is not modified by per-head kernel."""
    device = "cuda"
    num_heads = 24
    num_tokens = 64
    hidden_size = 3 * num_heads * head_dim

    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    v_size = num_heads * head_dim
    v_original = qkv[:, -v_size:].clone()

    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_emb,
        sin_emb,
        -1,
    )
    torch.testing.assert_close(qkv[:, -v_size:], v_original, rtol=0, atol=0)


# ============================================================================
# Full-dim norm tests (LTX-2 / WAN packed FUSE_QKV)
# weight is [num_heads*head_dim] (full-dim per-token norm), no dual-stream,
# cos/sin can be [num_tokens, num_heads*head_dim] (per-head) or
# broadcast [num_tokens/B, num_heads*head_dim] (kernel modulo over B).
# ============================================================================


@torch.inference_mode()
def torch_ref_full_dim(
    qkv,
    num_heads,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_emb,
    sin_emb,
    interleave,
):
    """Reference: full-dim RMSNorm (weight shape [num_heads*head_dim]) on Q and K
    of a packed qkv tensor [N, 3*num_heads*head_dim], then rope; V untouched.

    cos/sin shape: [num_tokens, num_heads*head_dim] (per-head, full-dim layout).
    """
    num_tokens = qkv.shape[0]
    hidden = num_heads * head_dim
    q = qkv[:, :hidden].float()
    k = qkv[:, hidden : 2 * hidden].float()
    v = qkv[:, 2 * hidden :]

    # full-dim norm: stat over all num_heads*head_dim elements per token.
    # Keep everything in fp32 through norm + rope to match kernel internal
    # precision (single bf16 cast at the end).
    var_q = q.pow(2).mean(-1, keepdim=True)
    q = q * torch.rsqrt(var_q + eps) * q_weight.float()
    var_k = k.pow(2).mean(-1, keepdim=True)
    k = k * torch.rsqrt(var_k + eps) * k_weight.float()

    cos_3d = cos_emb.float().view(num_tokens, num_heads, head_dim)
    sin_3d = sin_emb.float().view(num_tokens, num_heads, head_dim)
    q_4d = q.view(num_tokens, num_heads, head_dim)
    k_4d = k.view(num_tokens, num_heads, head_dim)
    if interleave:
        # pair (2i, 2i+1) — INTERLEAVED rope
        def _rope_interleaved(x_4d):
            rot = torch.empty_like(x_4d)
            rot[..., 0::2] = -x_4d[..., 1::2]
            rot[..., 1::2] = x_4d[..., 0::2]
            return x_4d * cos_3d + rot * sin_3d

        q_4d = _rope_interleaved(q_4d)
        k_4d = _rope_interleaved(k_4d)
    else:
        # rotate-half: pair (i, i+D/2) within head — LTX-2 SPLIT
        half = head_dim // 2

        def _rope_rotate_half(x_4d):
            x1 = x_4d[..., :half]
            x2 = x_4d[..., half:]
            rot = torch.cat([-x2, x1], dim=-1)
            return x_4d * cos_3d + rot * sin_3d

        q_4d = _rope_rotate_half(q_4d)
        k_4d = _rope_rotate_half(k_4d)

    q_out = q_4d.reshape(num_tokens, -1).to(qkv.dtype)
    k_out = k_4d.reshape(num_tokens, -1).to(qkv.dtype)
    return torch.cat([q_out, k_out, v], dim=1)


def _make_per_head_cos_full_dim(num_tokens, num_heads, head_dim, device, dtype=torch.float32):
    """Per-head cos: [num_tokens, num_heads*head_dim] with rotate-half block-duplicate pattern.

    cos/sin generated as (num_heads, num_tokens, head_dim/2) and block-duplicated
    along the last dim (cos = cat([cos_half, cos_half], -1)) so the kernel's
    rotate-half read pattern works on the flat 2D layout.
    """
    half = head_dim // 2
    freqs = torch.randn(num_tokens, num_heads, half, device=device, dtype=torch.float32)
    cos_h = freqs.cos()
    sin_h = freqs.sin()
    cos = torch.cat([cos_h, cos_h], dim=-1)  # [N, H, D]
    sin = torch.cat([sin_h, sin_h], dim=-1)
    cos_2d = cos.reshape(num_tokens, num_heads * head_dim).contiguous().to(dtype)
    sin_2d = sin.reshape(num_tokens, num_heads * head_dim).contiguous().to(dtype)
    return cos_2d, sin_2d


@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16], ids=["fp32cos", "bf16cos"])
@pytest.mark.parametrize(
    "label,B,T,num_heads,head_dim",
    [
        # LTX-2 video self-attn (FUSE_QKV packed, full-dim norm, rotate-half)
        ("ltx2_video_self_attn", 2, 12288, 32, 128),
        # LTX-2 audio self-attn
        ("ltx2_audio_self_attn", 2, 504, 32, 64),
        # B=1 sanity (no broadcast)
        ("ltx2_video_b1", 1, 12288, 32, 128),
    ],
)
def test_full_dim_norm_packed_rotate_half(label, B, T, num_heads, head_dim, cos_dtype):
    """LTX-2 packed FUSE_QKV full-dim norm + rotate-half rope (no broadcast)."""
    device = "cuda"
    torch.random.manual_seed(0)

    hidden = num_heads * head_dim
    num_tokens = B * T
    qkv = torch.randn(num_tokens, 3 * hidden, dtype=torch.bfloat16, device=device) * 0.5
    qkv_copy = qkv.clone()
    q_weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    # Tile cos/sin to num_tokens (no kernel-side broadcast in this test)
    cos_2d_T, sin_2d_T = _make_per_head_cos_full_dim(
        T, num_heads, head_dim, device, dtype=cos_dtype
    )
    if B > 1:
        cos_2d = cos_2d_T.repeat(B, 1)
        sin_2d = sin_2d_T.repeat(B, 1)
    else:
        cos_2d, sin_2d = cos_2d_T, sin_2d_T

    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_2d,
        sin_2d,
        -1,
        interleave=False,
    )
    ref = torch_ref_full_dim(
        qkv_copy,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_2d,
        sin_2d,
        interleave=False,
    )
    torch.testing.assert_close(qkv, ref, rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16], ids=["fp32cos", "bf16cos"])
@pytest.mark.parametrize(
    "label,B,T,num_heads,head_dim",
    [
        # C7 broadcast: cos has T rows, qkv has B*T tokens; kernel does tokenIdx % T
        ("ltx2_video_bcast", 2, 12288, 32, 128),
        ("ltx2_audio_bcast", 2, 504, 32, 64),
    ],
)
def test_full_dim_norm_packed_rotate_half_broadcast(label, B, T, num_heads, head_dim, cos_dtype):
    """C7: kernel-side cos broadcast over B (cos.size(0) == T, not B*T)."""
    device = "cuda"
    torch.random.manual_seed(0)

    hidden = num_heads * head_dim
    num_tokens = B * T
    qkv = torch.randn(num_tokens, 3 * hidden, dtype=torch.bfloat16, device=device) * 0.5
    qkv_copy = qkv.clone()
    q_weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    cos_2d_T, sin_2d_T = _make_per_head_cos_full_dim(
        T, num_heads, head_dim, device, dtype=cos_dtype
    )

    # Reference: tile cos to B*T tokens (kernel does this in-place via modulo)
    cos_2d_full = cos_2d_T.repeat(B, 1)
    sin_2d_full = sin_2d_T.repeat(B, 1)

    # Call kernel with the unbroadcast T-row cos (kernel detects via cos.size(0) != num_tokens)
    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_2d_T,
        sin_2d_T,
        -1,
        interleave=False,
    )
    ref = torch_ref_full_dim(
        qkv_copy,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_2d_full,
        sin_2d_full,
        interleave=False,
    )
    torch.testing.assert_close(qkv, ref, rtol=2e-2, atol=5e-3)


def test_full_dim_norm_packed_v_unchanged():
    """Full-dim path leaves V slice untouched (sanity)."""
    device = "cuda"
    torch.random.manual_seed(0)
    num_heads, head_dim, num_tokens = 32, 128, 256
    hidden = num_heads * head_dim
    qkv = torch.randn(num_tokens, 3 * hidden, dtype=torch.bfloat16, device=device) * 0.5
    v_original = qkv[:, 2 * hidden :].clone()
    q_weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 5.0
    cos_2d, sin_2d = _make_per_head_cos_full_dim(num_tokens, num_heads, head_dim, device)

    _call_fused_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_2d,
        sin_2d,
        -1,
        interleave=False,
    )
    torch.testing.assert_close(qkv[:, 2 * hidden :], v_original, rtol=0, atol=0)
