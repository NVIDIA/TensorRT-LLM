# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Fused partial RoPE (NeoX format) — rotate first ``rotary_dim`` channels and
pass the remaining ``head_dim - rotary_dim`` channels through in one kernel.

Replaces the 4-kernel graph pattern that MiniMax-M2 / DeepSeek-V3 style models
emit after ``apply_rotary_pos_emb``:

    slice(q, -1, 0, Dr)   \\
    slice(q, -1, Dr, ...)  >-- flashinfer_rope(q_rot, k_rot, ...) + 2x cat
    slice(k, -1, 0, Dr)   /

    (4 glue kernels ~7 µs/layer: flashinfer_rope + 3 CatArrayBatchedCopy + elemwise)

The fused Triton kernel does the full thing in one launch per tensor (Q and K).
For NeoX-format caches (``emb = cat(freqs, freqs, -1)``) the second half of
``cos`` / ``sin`` duplicates the first half, so we only read the first half.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _partial_rope_one_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    x_out_ptr,
    x_stride_m,
    x_stride_h,
    x_out_stride_m,
    x_out_stride_h,
    cos_stride_m,
    DH: tl.constexpr,  # head_dim
    DR: tl.constexpr,  # rotary_dim, must be even and <= DH
    HALF: tl.constexpr,  # DR // 2
    DH_PASS: tl.constexpr,  # DH - DR (pow-of-2 or 0)
):
    """One program per (token, head). Rotates x[:DR] NeoX-style, copies x[DR:DH].

    NOTE: merging Q and K into a single kernel (grid sized to max(H_Q, H_KV)
    with runtime predication on H_KV) was tested and produced a net
    regression — active GPU time increased by ~150 µs/iter on the MiniMax-M2.7
    tp2 reduced config. Likely cause: runtime-branched work per program hurts
    occupancy and per-program latency more than it saves in launch overhead.
    Keeping two launches (one for Q, one for K) for now.
    """
    m = tl.program_id(0)
    h = tl.program_id(1)

    x_base = x_ptr + m * x_stride_m + h * x_stride_h
    out_base = x_out_ptr + m * x_out_stride_m + h * x_out_stride_h

    # Rotary lanes
    offs_half = tl.arange(0, HALF)
    x_lo = tl.load(x_base + offs_half).to(tl.float32)
    x_hi = tl.load(x_base + HALF + offs_half).to(tl.float32)
    cos = tl.load(cos_ptr + m * cos_stride_m + offs_half).to(tl.float32)
    sin = tl.load(sin_ptr + m * cos_stride_m + offs_half).to(tl.float32)

    # NeoX rotate_half: cat(-x_hi, x_lo); rotation = x*cos + rot_half(x)*sin
    out_lo = x_lo * cos - x_hi * sin
    out_hi = x_hi * cos + x_lo * sin

    dtype = x_out_ptr.dtype.element_ty
    tl.store(out_base + offs_half, out_lo.to(dtype))
    tl.store(out_base + HALF + offs_half, out_hi.to(dtype))

    # Pass-through lanes: [DR:DH]. When DH == DR, skipped at compile time.
    if DH_PASS > 0:
        offs_pass = tl.arange(0, DH_PASS)
        x_pass = tl.load(x_base + DR + offs_pass)
        tl.store(out_base + DR + offs_pass, x_pass)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _partial_rope_fused_impl(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """Partial RoPE over leading-dim-flattened q, k.

    Args:
        q:   [..., H_q, Dh]    — bf16 or fp16.
        k:   [..., H_kv, Dh]   — bf16 or fp16.
        cos: [..., Dr]         — same dtype as q (typically bf16).
        sin: [..., Dr]         — same dtype as q.

    Returns (q_out, k_out) with the same shapes and dtypes as the inputs.

    Assumptions:
        - Dh and Dr are even; Dr <= Dh; (Dh - Dr) is a power of 2 or 0.
        - cos/sin are in NeoX duplicated format (second half == first half);
          only the first Dr/2 lanes are read.
    """
    assert q.shape[-1] == k.shape[-1], "Q and K must share head_dim"
    assert q.dtype == k.dtype, "Q and K must share dtype"
    q_shape, k_shape = q.shape, k.shape
    Dh = q_shape[-1]
    Dr = cos.shape[-1]
    Hq = q_shape[-2]
    Hkv = k_shape[-2]

    assert Dr % 2 == 0 and Dr <= Dh, f"Dr={Dr} must be even and <= Dh={Dh}"
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "last dim must be contiguous"
    assert cos.stride(-1) == 1 and sin.stride(-1) == 1, "last dim must be contiguous"

    # Flatten leading dims into a single M dim.
    M = q.numel() // (Hq * Dh)
    q_flat = q.reshape(M, Hq, Dh)
    k_flat = k.reshape(M, Hkv, Dh)
    cos_flat = cos.reshape(M, Dr)
    sin_flat = sin.reshape(M, Dr)

    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    DH_PASS = Dh - Dr
    # DH_PASS must be a power of 2 (or 0) for tl.arange.
    assert DH_PASS == 0 or DH_PASS == _next_pow2(DH_PASS), (
        f"(Dh - Dr)={DH_PASS} must be a power of two"
    )
    HALF = Dr // 2

    grid_q = (M, Hq)
    _partial_rope_one_kernel[grid_q](
        q_flat,
        cos_flat,
        sin_flat,
        q_out,
        q_flat.stride(0),
        q_flat.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        cos_flat.stride(0),
        DH=Dh,
        DR=Dr,
        HALF=HALF,
        DH_PASS=DH_PASS,
    )
    grid_k = (M, Hkv)
    _partial_rope_one_kernel[grid_k](
        k_flat,
        cos_flat,
        sin_flat,
        k_out,
        k_flat.stride(0),
        k_flat.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        cos_flat.stride(0),
        DH=Dh,
        DR=Dr,
        HALF=HALF,
        DH_PASS=DH_PASS,
    )

    return q_out.reshape(q_shape), k_out.reshape(k_shape)


@torch.library.custom_op("auto_deploy::partial_rope_fused", mutates_args=())
def partial_rope_fused(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Fused partial RoPE (NeoX format) on the last head_dim of q/k.

    Rotates the first ``cos.shape[-1]`` (== rotary_dim) channels and copies
    the remaining ``head_dim - rotary_dim`` channels unchanged. Two Triton
    kernel launches total (one for Q, one for K), replacing the
    slice+flashinfer_rope+cat pattern.
    """
    return _partial_rope_fused_impl(q, k, cos, sin)


@partial_rope_fused.register_fake
def _partial_rope_fused_fake(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k)
