# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Deferred-NVFP4 scale finalize (K2) - the shared second half of the
# deferred-SFC quantization scheme.
#
# The producer (K1: e.g. this package's fused_norm_producer with
# store="nvfp4_deferred") emits scale-invariant e2m1 payloads plus RAW fp32
# block scales a/6 in [M, K/16] row-major unswizzled. This module computes
# the global scale s = 448 / max(raw) (0-d device tensor - no .item(), no
# host sync) and re-encodes every raw block scale into the swizzled e4m3 SF
# tensor at quantization.cuh's 128x4 offsets (L703-741), zero-filling pad
# rows/cols (L831-859). The payload never needs a second pass. The consumer
# GEMM's alpha is weight_scale_2 * (max_raw / 448) == weight_scale_2 / s.
#
# This module is deliberately kernel-family-neutral so any deferred-store
# producer (norm-site, activation+quant, GEMM epilogue) can share it: keep
# it the single in-tree K2 - do not fork per producer.

from typing import Tuple

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _e4m3_encode(x):
    """Non-negative finite f32 -> e4m3fn byte code, RNE + satfinite(448).
    Bit-exact RNE via the classic mantissa-add trick on the f32 pattern;
    subnormal path (x < 2^-6) rounds x/2^-9 to int with ties-to-even.
    (CPU-verified: 0 mismatches vs the torch float8_e4m3fn cast on 2.5M
    values; GPU-confirmed: SF bitwise-identical to fp4_quantize.)"""
    xc = tl.minimum(x, 448.0)
    bits = xc.to(tl.int32, bitcast=True)
    # RNE to 3 mantissa bits: add 0x7FFFF + lsb-of-kept-mantissa.
    rb = bits + 0x7FFFF + ((bits >> 20) & 1)
    e4 = (rb >> 23) - 120  # e4m3 biased exponent (bias 7)
    m3 = (rb >> 20) & 7
    norm = (e4 << 3) | m3
    # Subnormal target: code = RNE(x * 512), ties to even; k==8 lands
    # exactly on code 8 == 2^-6 normal.
    t = xc * 512.0
    k0 = tl.floor(t)
    frac = t - k0
    k0i = k0.to(tl.int32)
    sub = tl.where(frac > 0.5, k0i + 1, tl.where(frac < 0.5, k0i, k0i + (k0i & 1)))
    code = tl.where(xc < 0.015625, sub, norm)
    return tl.minimum(code, 0x7E)  # 0x7E == 448.0


@triton.jit
def _k2_sfc_kernel(raw_ptr, s_ptr, sf_ptr, M, KB_TOT, NUM_K_TILES, KT_BLOCK: tl.constexpr):
    """One program covers one 128-row m-tile x KT_BLOCK k-tiles (of 4 SF
    cols each). Swizzled offset per quantization.cuh:703-741:
      off = mTile*(numKTiles*512) + kTile*512 + (m%32)*16 + ((m%128)//32)*4
            + (k%4)
    Pad rows/cols (raw load masked to 0 -> e4m3(0)=0) are zero-filled,
    matching quantization.cuh:831-859."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    s = tl.load(s_ptr)
    ml = tl.arange(0, 128)
    kt = pid_k * KT_BLOCK + tl.arange(0, KT_BLOCK)
    kl = tl.arange(0, 4)

    m_idx = pid_m * 128 + ml
    k_idx = kt[None, :, None] * 4 + kl[None, None, :]  # [1, KT, 4]
    valid = (m_idx[:, None, None] < M) & (k_idx < KB_TOT)
    raw = tl.load(
        raw_ptr + m_idx.to(tl.int64)[:, None, None] * KB_TOT + k_idx, mask=valid, other=0.0
    )
    byte = _e4m3_encode(raw * s).to(tl.uint8)

    pos = (ml % 32)[:, None, None] * 16 + (ml // 32)[:, None, None] * 4 + kl[None, None, :]
    off = pid_m.to(tl.int64) * NUM_K_TILES * 512 + kt.to(tl.int64)[None, :, None] * 512 + pos
    smask = (kt[None, :, None] < NUM_K_TILES) & (ml[:, None, None] >= 0) & (kl[None, None, :] >= 0)
    tl.store(sf_ptr + off, byte, mask=smask)


def sfc_finalize(
    raw: torch.Tensor, kt_block: int = 8, num_warps: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K2: raw f32 [M, K/16] -> (sf uint8 [swizzled_sf_size], s 0-d f32,
    max_raw 0-d f32). s = 448/max(raw); caller computes
    alpha = weight_scale_2 * (max_raw / 448) == weight_scale_2 / s.
    No .item() anywhere - s stays on device."""
    m, kb_tot = raw.shape
    max_raw = torch.amax(raw)
    s = FP8_MAX / max_raw
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (kb_tot + 3) // 4
    sf = torch.empty(num_m_tiles * 128 * num_k_tiles * 4, dtype=torch.uint8, device=raw.device)
    grid = (num_m_tiles, triton.cdiv(num_k_tiles, kt_block))
    _k2_sfc_kernel[grid](raw, s, sf, m, kb_tot, num_k_tiles, KT_BLOCK=kt_block, num_warps=num_warps)
    return sf, s, max_raw


@torch.library.custom_op(
    "trtllm::nvfp4_sfc_finalize",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def nvfp4_sfc_finalize(raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finalize deferred-NVFP4 raw block scales: returns (sf, s, max_raw).

    raw: fp32 [M, K/16] row-major contiguous (a/6 block scales from a
    deferred-store producer). sf: uint8 [pad128(M) * pad4(K/16)] swizzled
    e4m3 SF bytes, bitwise-identical to fp4_quantize's SF at the same s,
    pad rows/cols zero-filled. s = 448/max(raw) and max_raw are 0-d fp32
    device tensors (no host sync)."""
    if raw.ndim != 2:
        raise ValueError(f"raw: expected [M, K/16] 2D fp32, got {raw.ndim}D")
    if raw.dtype != torch.float32:
        raise ValueError(f"raw: expected fp32, got {raw.dtype}")
    if not raw.is_cuda:
        raise ValueError("raw: expected a CUDA tensor")
    if raw.stride(-1) != 1 or raw.stride(0) != raw.shape[1]:
        raise ValueError("raw: must be row-major contiguous")
    return sfc_finalize(raw)


@nvfp4_sfc_finalize.register_fake
def _nvfp4_sfc_finalize_fake(raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, kb_tot = raw.shape
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (kb_tot + 3) // 4
    sf = raw.new_empty((num_m_tiles * 128 * num_k_tiles * 4,), dtype=torch.uint8)
    return sf, raw.new_empty(()), raw.new_empty(())
