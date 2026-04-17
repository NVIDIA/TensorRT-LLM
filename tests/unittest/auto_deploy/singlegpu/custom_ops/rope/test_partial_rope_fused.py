# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for auto_deploy::partial_rope_fused.

Verifies the fused Triton kernel against the reference `apply_rotary_pos_emb`
implementation in tensorrt_llm._torch.auto_deploy.models.custom.modeling_minimax_m2
across typical MiniMax-M2.7 shapes and dtypes.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401


def _ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
):
    """Reference partial RoPE — matches modeling_minimax_m2.apply_rotary_pos_emb."""
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos_u) + (_ref_rotate_half(q_rot) * sin_u)
    k_embed = (k_rot * cos_u) + (_ref_rotate_half(k_rot) * sin_u)
    return (
        torch.cat([q_embed, q_pass], dim=-1),
        torch.cat([k_embed, k_pass], dim=-1),
    )


def _make_neox_cos_sin(n_tokens: int, rotary_dim: int, dtype, device):
    """Produce cos/sin in NeoX duplicated format (emb = cat(freqs, freqs, -1))."""
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(n_tokens, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).to(device)
    sin = emb.sin().to(dtype).to(device)
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B,S,H_q,H_kv,Dh,Dr",
    [
        (1, 1, 24, 4, 128, 64),  # MiniMax-M2.7 TP=2 decode batch=1
        (1, 32, 24, 4, 128, 64),  # decode batch=32
        (1, 1024, 24, 4, 128, 64),  # prefill
        (2, 64, 48, 8, 128, 64),  # untensor-sharded M2.7
        (1, 16, 8, 8, 64, 64),  # full RoPE (Dr == Dh, no pass-through)
    ],
)
def test_partial_rope_fused_matches_reference(B, S, H_q, H_kv, Dh, Dr, dtype):
    device = "cuda"
    torch.manual_seed(42 + B * 7 + S)

    q = torch.randn(B, S, H_q, Dh, dtype=dtype, device=device)
    k = torch.randn(B, S, H_kv, Dh, dtype=dtype, device=device)
    cos_all, sin_all = _make_neox_cos_sin(B * S, Dr, dtype, device)
    cos = cos_all.view(B, S, Dr)
    sin = sin_all.view(B, S, Dr)

    ref_q, ref_k = _ref_partial_rope(q, k, cos, sin, unsqueeze_dim=2)
    out_q, out_k = torch.ops.auto_deploy.partial_rope_fused(q, k, cos, sin)

    assert out_q.shape == ref_q.shape
    assert out_k.shape == ref_k.shape
    assert out_q.dtype == dtype
    assert out_k.dtype == dtype

    # Tolerances: bf16 has ~3e-3 unit in the last place for products + sums
    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out_q, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_k, ref_k, atol=atol, rtol=rtol)


def test_partial_rope_fused_3d_flat():
    """Leading dims may be already flattened [M, H, Dh] instead of [B, S, H, Dh]."""
    device = "cuda"
    dtype = torch.bfloat16
    M, H_q, H_kv, Dh, Dr = 64, 24, 4, 128, 64
    q = torch.randn(M, H_q, Dh, dtype=dtype, device=device)
    k = torch.randn(M, H_kv, Dh, dtype=dtype, device=device)
    cos, sin = _make_neox_cos_sin(M, Dr, dtype, device)

    ref_q, ref_k = _ref_partial_rope(q, k, cos, sin, unsqueeze_dim=1)
    out_q, out_k = torch.ops.auto_deploy.partial_rope_fused(q, k, cos, sin)

    torch.testing.assert_close(out_q, ref_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_k, ref_k, atol=1e-2, rtol=1e-2)
