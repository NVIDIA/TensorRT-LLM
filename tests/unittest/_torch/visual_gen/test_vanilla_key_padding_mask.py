# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical parity for VanillaAttention's key_padding_mask.

v2a Ulysses pads audio K/V so T_a is divisible by U. The mathematical
guarantee the padding relies on: masking the padded K/V columns via
key_padding_mask must produce identical output to running SDPA on the
unpadded K/V. Single-rank, CPU.
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend import VanillaAttention


def test_padded_kv_with_mask_matches_unpadded():
    torch.manual_seed(42)
    B, S_q, S_kv_valid, S_pad, H, H_kv, d_h = 2, 12, 7, 5, 8, 8, 64
    S_kv = S_kv_valid + S_pad

    attn = VanillaAttention(num_heads=H, head_dim=d_h, num_kv_heads=H_kv)

    q = torch.randn(B, H, S_q, d_h)
    k_valid = torch.randn(B, H_kv, S_kv_valid, d_h)
    v_valid = torch.randn(B, H_kv, S_kv_valid, d_h)

    ref = attn.forward(q=q, k=k_valid, v=v_valid)

    # Pad K/V with realistic magnitudes (O(1)). Real audio pad goes through
    # K/V Linear from a zero-latent, so it is bounded by the layer's bias norm.
    k_padded = torch.cat([k_valid, torch.randn(B, H_kv, S_pad, d_h)], dim=2)
    v_padded = torch.cat([v_valid, torch.randn(B, H_kv, S_pad, d_h)], dim=2)
    mask = torch.zeros(B, S_kv, dtype=torch.bool)
    mask[:, :S_kv_valid] = True

    out = attn.forward(q=q, k=k_padded, v=v_padded, key_padding_mask=mask)

    torch.testing.assert_close(
        out,
        ref,
        rtol=1e-4,
        atol=1e-4,
        msg="key_padding_mask did not produce numerical parity with unpadded SDPA",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
