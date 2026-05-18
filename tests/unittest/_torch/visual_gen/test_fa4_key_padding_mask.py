# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical parity for FlashAttn4Attention's key_padding_mask.

FA4 is used for LTX-2 audio self-attn and a2v cross-attn when the user picks
``attention.backend: FA4``. With ``audio_pad_for_ulysses=True``, padded K
columns must produce zero attention contribution. FA4's cute interface accepts
``seqused_k`` per-batch valid lengths; this test verifies that translating a
True-prefix bool mask to ``seqused_k`` yields output identical to running FA4
on the unpadded K/V (within bf16 tolerance).

Requires CUDA (FA4 is GPU-only).
"""

import pytest
import torch

try:
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
        FlashAttn4Attention,
        _flash_attn_fwd,
    )

    FA4_AVAILABLE = _flash_attn_fwd is not None
except ImportError:
    FA4_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 requires CUDA"),
    pytest.mark.skipif(not FA4_AVAILABLE, reason="FA4 kernel not available"),
]


def _run_self_attn(B, S_real, S_pad, H, d_h, dtype=torch.bfloat16):
    """Helper: self-attention case (Q=K=V, padded only on the seq dim)."""
    device = "cuda"
    torch.manual_seed(0)
    S_full = S_real + S_pad

    # Build a padded sequence with random valid prefix + arbitrary junk suffix.
    x_valid = torch.randn(B, S_real, H, d_h, dtype=dtype, device=device)
    x_pad = torch.randn(B, S_pad, H, d_h, dtype=dtype, device=device)
    x_full = torch.cat([x_valid, x_pad], dim=1)
    mask = torch.zeros(B, S_full, dtype=torch.bool, device=device)
    mask[:, :S_real] = True

    fa4 = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

    # FA4 path with seqused_k (q_full = k_full = v_full = x_full padded; only valid prefix attends).
    out_padded = fa4.forward(q=x_full, k=x_full, v=x_full, key_padding_mask=mask)

    # Reference: FA4 on unpadded inputs only (Q seq = K seq = S_real).
    out_ref = fa4.forward(q=x_valid, k=x_valid, v=x_valid)

    # Only the valid Q rows must match; pad Q rows are stripped downstream by
    # the caller and are not part of the contract.
    return out_padded[:, :S_real], out_ref


def _run_cross_attn(B, S_q, S_real_kv, S_pad_kv, H, d_h, dtype=torch.bfloat16):
    """Helper: cross-attention case (Q full, K/V padded on seq dim)."""
    device = "cuda"
    torch.manual_seed(1)
    S_full_kv = S_real_kv + S_pad_kv

    q = torch.randn(B, S_q, H, d_h, dtype=dtype, device=device)
    k_valid = torch.randn(B, S_real_kv, H, d_h, dtype=dtype, device=device)
    v_valid = torch.randn(B, S_real_kv, H, d_h, dtype=dtype, device=device)
    k_pad = torch.randn(B, S_pad_kv, H, d_h, dtype=dtype, device=device)
    v_pad = torch.randn(B, S_pad_kv, H, d_h, dtype=dtype, device=device)
    k_full = torch.cat([k_valid, k_pad], dim=1)
    v_full = torch.cat([v_valid, v_pad], dim=1)
    mask = torch.zeros(B, S_full_kv, dtype=torch.bool, device=device)
    mask[:, :S_real_kv] = True

    fa4 = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

    out_padded = fa4.forward(q=q, k=k_full, v=v_full, key_padding_mask=mask)
    out_ref = fa4.forward(q=q, k=k_valid, v=v_valid)
    return out_padded, out_ref


def test_self_attn_padded_kv_with_mask_matches_unpadded():
    """FA4 self-attn: masked padded K/V matches unpadded K/V on valid Q rows."""
    out, ref = _run_self_attn(B=2, S_real=126, S_pad=2, H=8, d_h=64)
    torch.testing.assert_close(
        out,
        ref,
        rtol=2e-3,
        atol=2e-3,
        msg="FA4 self-attn key_padding_mask diverges from unpadded SDPA",
    )


def test_cross_attn_padded_kv_with_mask_matches_unpadded():
    """FA4 cross-attn: padded K/V + mask matches unpadded K/V (matches a2v use)."""
    out, ref = _run_cross_attn(B=2, S_q=320, S_real_kv=126, S_pad_kv=2, H=8, d_h=64)
    torch.testing.assert_close(
        out,
        ref,
        rtol=2e-3,
        atol=2e-3,
        msg="FA4 cross-attn key_padding_mask diverges from unpadded SDPA",
    )


def test_self_attn_pad_junk_values_dont_affect_valid_output():
    """Two different junk pad fills with same mask produce same valid-row output."""
    B, S_real, S_pad, H, d_h = 1, 64, 4, 4, 64
    device = "cuda"
    S_full = S_real + S_pad
    dtype = torch.bfloat16

    torch.manual_seed(7)
    x_valid = torch.randn(B, S_real, H, d_h, dtype=dtype, device=device)
    mask = torch.zeros(B, S_full, dtype=torch.bool, device=device)
    mask[:, :S_real] = True
    fa4 = FlashAttn4Attention(num_heads=H, head_dim=d_h, num_kv_heads=H)

    # Pad fill A: zeros. Pad fill B: random.
    pad_a = torch.zeros(B, S_pad, H, d_h, dtype=dtype, device=device)
    pad_b = torch.randn(B, S_pad, H, d_h, dtype=dtype, device=device)
    x_a = torch.cat([x_valid, pad_a], dim=1)
    x_b = torch.cat([x_valid, pad_b], dim=1)

    out_a = fa4.forward(q=x_a, k=x_a, v=x_a, key_padding_mask=mask)
    out_b = fa4.forward(q=x_b, k=x_b, v=x_b, key_padding_mask=mask)

    # Q[:S_real] sees the same K/V[:S_real]; mask zeros K/V[S_real:] contribution.
    # Only the valid Q rows are part of the contract.
    torch.testing.assert_close(
        out_a[:, :S_real],
        out_b[:, :S_real],
        rtol=1e-5,
        atol=1e-5,
        msg="FA4 pad fill leaked into valid-row output — mask is not fully suppressing pads",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
