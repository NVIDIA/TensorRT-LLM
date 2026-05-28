# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch


def torch_ref(q_5d, k_5d, v_5d, is_hnd):
    """Eager reference: the permute+reshape+contiguous (+transpose+contig for HND)
    chain that the kernel replaces."""

    def post(t):
        P, B, Sp, H, D = t.shape
        out = t.permute(1, 0, 2, 3, 4).reshape(B, P * Sp, H, D).contiguous()  # NHD
        if is_hnd:
            return out.transpose(1, 2).contiguous()  # [B, H, P*Sp, D]
        return out  # [B, P*Sp, H, D]

    return post(q_5d), post(k_5d), post(v_5d)


@pytest.mark.parametrize("layout", [0, 1], ids=["HND", "NHD"])
@pytest.mark.parametrize(
    "P,B,Sp,H,D",
    [
        # LTX-2 self-attn (ulysses=4): per-rank H = 32/4 = 8, D = 128
        (4, 2, 1024, 8, 128),
        # LTX-2 audio attn (ulysses=4): per-rank H = 32/4 = 8, D = 64
        (4, 2, 1024, 8, 64),
        # WAN-like (alternate H, D)
        (4, 2, 512, 16, 128),
        (8, 1, 256, 16, 128),
        # H * (D/8) edge: 64 threads/block (smallest interesting tile)
        (2, 1, 128, 4, 128),
        # Larger H with D=64
        (4, 2, 256, 32, 64),
    ],
)
@torch.inference_mode()
def test_ulysses_post_unscatter_exact_match(P, B, Sp, H, D, layout):
    """The op is a pure data movement, so output must match the eager
    permute+reshape+contiguous (+transpose+contig for HND) chain exactly
    (max_diff == 0). Both HND (layout=0) and NHD (layout=1) outputs are
    exercised."""
    is_hnd = layout == 0
    torch.manual_seed(0)
    q = torch.randn(P, B, Sp, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    k = torch.randn(P, B, Sp, H, D, device="cuda", dtype=torch.bfloat16).contiguous()
    v = torch.randn(P, B, Sp, H, D, device="cuda", dtype=torch.bfloat16).contiguous()

    q_ref, k_ref, v_ref = torch_ref(q, k, v, is_hnd=is_hnd)
    q_out, k_out, v_out = torch.ops.trtllm.ulysses_post_unscatter_qkv(q, k, v, layout)

    expected_shape = (B, H, P * Sp, D) if is_hnd else (B, P * Sp, H, D)
    assert q_out.shape == expected_shape
    assert q_out.is_contiguous() and k_out.is_contiguous() and v_out.is_contiguous()
    assert q_out.dtype == torch.bfloat16
    for name, ref, got in [("Q", q_ref, q_out), ("K", k_ref, k_out), ("V", v_ref, v_out)]:
        max_diff = (ref - got).abs().max().item()
        assert max_diff == 0, f"{name}: max_diff={max_diff} (expected exact match)"


@torch.inference_mode()
def test_ulysses_post_unscatter_rejects_invalid_layout():
    """layout must be 0 (HND) or 1 (NHD)."""
    q = torch.randn(2, 1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.ulysses_post_unscatter_qkv(q, q, q, 2)


@torch.inference_mode()
def test_ulysses_post_unscatter_rejects_d_not_multiple_of_8():
    """D must be a multiple of 8 (uint4 vec load constraint)."""
    q = torch.randn(2, 1, 128, 8, 60, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.ulysses_post_unscatter_qkv(q, q, q)


@torch.inference_mode()
def test_ulysses_post_unscatter_rejects_oversized_block():
    """Threads/block = H * (D/8) must be <= 1024 (CUDA hw limit)."""
    # H=128, D=128 -> 128 * 16 = 2048 threads, exceeds the 1024 hw cap.
    q = torch.randn(2, 1, 64, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.ulysses_post_unscatter_qkv(q, q, q)
