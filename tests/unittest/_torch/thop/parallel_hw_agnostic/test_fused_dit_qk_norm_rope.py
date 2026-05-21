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
# Cross-head norm reference implementation
# ============================================================================


@torch.inference_mode()
def torch_ref_cross_head(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_emb,
    sin_emb,
    interleave,
):
    """Reference: cross-head (full-dim) RMSNorm + RoPE (WAN pattern)."""
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim

    q = qkv[:, :q_size].clone()
    k = qkv[:, q_size : q_size + k_size].clone()
    v = qkv[:, q_size + k_size :].clone()

    # Cross-head RMSNorm: norm over the full q_dim / k_dim dimension
    q = F.rms_norm(q.float(), (q_size,), q_weight.float(), eps).to(qkv.dtype)
    k = F.rms_norm(k.float(), (k_size,), k_weight.float(), eps).to(qkv.dtype)

    # Expand cos/sin per head for RoPE
    cos_q = cos_emb.unsqueeze(1).expand(-1, num_heads_q, -1).reshape(num_tokens, q_size)
    sin_q = sin_emb.unsqueeze(1).expand(-1, num_heads_q, -1).reshape(num_tokens, q_size)
    cos_k = cos_emb.unsqueeze(1).expand(-1, num_heads_k, -1).reshape(num_tokens, k_size)
    sin_k = sin_emb.unsqueeze(1).expand(-1, num_heads_k, -1).reshape(num_tokens, k_size)

    if interleave:
        q = _apply_interleaved_rope(q, cos_q, sin_q)
        k = _apply_interleaved_rope(k, cos_k, sin_k)
    else:
        q = _apply_rotate_half_rope(q, cos_q, sin_q, head_dim, num_heads_q)
        k = _apply_rotate_half_rope(k, cos_k, sin_k, head_dim, num_heads_k)

    return torch.cat([q.to(qkv.dtype), k.to(qkv.dtype), v], dim=1)


def _call_cross_head_kernel(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_emb,
    sin_emb,
    interleave=True,
):
    """Call the fused DiT cross-head QK Norm + RoPE kernel."""
    torch.ops.trtllm.fused_dit_cross_head_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
        interleave,
    )


# ============================================================================
# Cross-head norm tests (WAN pattern)
# ============================================================================


@pytest.mark.parametrize(
    "num_heads,head_dim",
    [
        (12, 128),  # WAN 1.3B: hidden_size=1536
        (40, 128),  # WAN 14B: hidden_size=5120
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 64, 512])
def test_cross_head_interleaved(num_heads, head_dim, num_tokens):
    """Cross-head norm + interleaved RoPE (WAN pattern)."""
    device = "cuda"
    torch.random.manual_seed(42)

    hidden_size = 3 * num_heads * head_dim
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_copy = qkv.clone()

    q_dim = num_heads * head_dim
    q_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device) * 5.0
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    _call_cross_head_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
        interleave=True,
    )

    ref = torch_ref_cross_head(
        qkv_copy,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
        True,
    )
    torch.testing.assert_close(qkv, ref, rtol=5e-2, atol=1e-1)


@pytest.mark.parametrize("num_heads", [12, 40])
@pytest.mark.parametrize("num_tokens", [1, 64, 512])
def test_cross_head_rotate_half(num_heads, num_tokens):
    """Cross-head norm + rotate_half RoPE."""
    device = "cuda"
    head_dim = 128
    torch.random.manual_seed(42)

    hidden_size = 3 * num_heads * head_dim
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_copy = qkv.clone()

    q_dim = num_heads * head_dim
    q_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device) * 5.0
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    _call_cross_head_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
        interleave=False,
    )

    ref = torch_ref_cross_head(
        qkv_copy,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
        False,
    )
    torch.testing.assert_close(qkv, ref, rtol=5e-2, atol=1e-1)


@pytest.mark.parametrize("num_heads", [12, 40])
def test_cross_head_v_unchanged(num_heads):
    """Verify V is not modified by cross-head kernel."""
    device = "cuda"
    head_dim = 128
    num_tokens = 64
    hidden_size = 3 * num_heads * head_dim

    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    v_size = num_heads * head_dim
    v_original = qkv[:, -v_size:].clone()

    q_dim = num_heads * head_dim
    q_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device)
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    _call_cross_head_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
    )
    torch.testing.assert_close(qkv[:, -v_size:], v_original, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_cross_head_batched(batch_size):
    """Cross-head norm + interleaved RoPE with batch_size > 1.

    Simulates the batch-flattening done by Attention.apply_qk_norm_rope.
    """
    device = "cuda"
    num_heads = 12
    head_dim = 128
    num_tokens = 256

    torch.random.manual_seed(42)
    total_tokens = batch_size * num_tokens
    hidden_size = 3 * num_heads * head_dim
    qkv = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_copy = qkv.clone()

    q_dim = num_heads * head_dim
    q_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device) * 5.0
    k_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device) * 5.0
    cos_single, sin_single = _generate_cos_sin(num_tokens, head_dim, device)
    cos_emb = cos_single.repeat(batch_size, 1)
    sin_emb = sin_single.repeat(batch_size, 1)

    _call_cross_head_kernel(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
    )

    ref = torch_ref_cross_head(
        qkv_copy,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
        True,
    )
    torch.testing.assert_close(qkv, ref, rtol=5e-2, atol=1e-1)
