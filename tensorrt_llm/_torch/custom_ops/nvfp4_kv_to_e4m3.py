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
"""Paged NVFP4 KV gather + transcode to per-tensor-scaled E4M3: ``trtllm::nvfp4_kv_to_e4m3``.

The source is a paged pool of NVFP4 rows without a global scale (DeepSeek-V4.1's
288-byte compressed-KV row): ``kv_cache`` holds ``W/2`` bytes of packed E2M1
codes per row (even element in the low nibble) and ``kv_cache_sf`` one E4M3
scale byte per 16 elements. For every request ``b`` and every position
``t < min(seq_lens[b], L)``::

    page          = block_tables[b, t // page_size]
    v_i           = E2M1[code_i] * e4m3(sf_{i // 16})            (exact in fp32)
    out[b, t, 0, i] = e4m3_rne(clamp(v_i * inv_scale, -448, 448))   if 0 <= page < pages

Positions at or past ``seq_lens[b]`` (or ``L``) and rows whose page id is out of
range are left unwritten. ``inv_scale`` is one fp32 factor for the whole
output (a per-tensor FP8 scale ``1 / inv_scale``). The transcode is lossy in
general: ``E2M1 x E4M3`` has up to 6 significant bits and E4M3 keeps 4.

``kv_cache`` / ``kv_cache_sf`` are ``[pages, page_size, 1, C]`` with a unit
last-dim stride and free leading strides, so the two halves of one 288-byte
row are passed as two views. ``out`` is a contiguous float8_e4m3fn
``[B, L, 1, W]``. Grid ``(ceil(L / 8), B)``, eight rows per program; no
autotuning.
"""

import torch
import triton
import triton.language as tl

NVFP4_BLOCK = 16
_ROWS = 8

_CODE_DTYPES = (torch.uint8,)
_SF_DTYPES = (torch.uint8, torch.float8_e4m3fn)


@triton.jit
def _e2m1_value(n):
    """4-bit E2M1 code (int32, sign in bit 3) -> fp32 value; -0.0 for code 8."""
    m = n & 7
    mag = tl.where(
        m < 4,
        m.to(tl.float32) * 0.5,
        tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0))),
    )
    return tl.where((n & 8) != 0, -mag, mag)


@triton.jit
def _nvfp4_kv_to_e4m3_kernel(
    codes_ptr,
    sf_ptr,
    bt_ptr,
    seq_ptr,
    out_ptr,
    inv_scale,
    codes_page_stride,
    codes_slot_stride,
    sf_page_stride,
    sf_slot_stride,
    bt_stride,
    num_pages,
    page_size,
    L,
    W: tl.constexpr,
    ROWS: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    t0 = tl.program_id(0).to(tl.int64) * ROWS
    n = tl.minimum(tl.load(seq_ptr + b).to(tl.int64), L)
    if t0 < n:
        blk = tl.arange(0, W // 16)
        j = tl.arange(0, 16)
        col = blk[:, None] * 16 + j[None, :]  # element index
        byte = col // 2
        hi = (col % 2) == 1
        for r in tl.static_range(ROWS):
            t = t0 + r
            if t < n:
                page = tl.load(bt_ptr + b * bt_stride + t // page_size).to(tl.int64)
                if (page >= 0) & (page < num_pages):
                    slot = t % page_size
                    c = tl.load(
                        codes_ptr + page * codes_page_stride + slot * codes_slot_stride + byte
                    ).to(tl.int32)
                    nib = tl.where(hi, c >> 4, c) & 15
                    s = tl.load(sf_ptr + page * sf_page_stride + slot * sf_slot_stride + blk)
                    s = s.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                    v = _e2m1_value(nib) * s[:, None] * inv_scale
                    v = tl.where(
                        v != v, v, tl.minimum(tl.maximum(v, -448.0), 448.0)
                    )  # NaN stays NaN
                    y = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
                    tl.store(out_ptr + (b * L + t) * W + col, y)


@torch.library.custom_op("trtllm::nvfp4_kv_to_e4m3", mutates_args=("out",))
def nvfp4_kv_to_e4m3(
    kv_cache: torch.Tensor,
    kv_cache_sf: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    inv_scale: float,
) -> None:
    """Gather paged NVFP4 rows (packed E2M1 x E4M3 per 16, no global scale) into ``out``
    ([B, L, 1, W] float8_e4m3fn, contiguous) as ``e4m3_rne(clamp(value * inv_scale, +-448))``
    for every ``t < min(seq_lens[b], L)`` whose page id is in range; everything else of
    ``out`` is left unwritten."""
    if not kv_cache.is_cuda:
        raise ValueError(f"kv_cache must be a CUDA tensor, got device {kv_cache.device}")
    device = kv_cache.device
    if kv_cache.dtype not in _CODE_DTYPES:
        raise ValueError(f"kv_cache must be uint8 (packed E2M1), got {kv_cache.dtype}")
    if kv_cache_sf.dtype not in _SF_DTYPES:
        raise ValueError(f"kv_cache_sf must be uint8 or float8_e4m3fn, got {kv_cache_sf.dtype}")
    if (
        kv_cache.dim() != 4
        or kv_cache_sf.dim() != 4
        or kv_cache.shape[2] != 1
        or kv_cache_sf.shape[2] != 1
    ):
        raise ValueError(
            "kv_cache / kv_cache_sf must be [pages, page_size, 1, C], got "
            f"{tuple(kv_cache.shape)} / {tuple(kv_cache_sf.shape)}"
        )
    if tuple(kv_cache.shape[:2]) != tuple(kv_cache_sf.shape[:2]):
        raise ValueError(
            f"page geometry differs: {tuple(kv_cache.shape)} vs {tuple(kv_cache_sf.shape)}"
        )
    if kv_cache.stride(-1) != 1 or kv_cache_sf.stride(-1) != 1:
        raise ValueError("kv_cache and kv_cache_sf need a unit last-dim stride")
    width = kv_cache.shape[-1] * 2
    if width % NVFP4_BLOCK != 0 or kv_cache_sf.shape[-1] * NVFP4_BLOCK != width:
        raise ValueError(
            f"expected {width // NVFP4_BLOCK} scale bytes per row, got {kv_cache_sf.shape[-1]}"
        )
    if width & (width - 1):
        raise ValueError(f"row width must be a power of two, got {width}")
    if out.device != device or out.dtype != torch.float8_e4m3fn:
        raise ValueError(f"out must be float8_e4m3fn on {device}, got {out.dtype} on {out.device}")
    if out.dim() != 4 or out.shape[2] != 1 or out.shape[3] != width or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [B, L, 1, {width}], got {tuple(out.shape)}")
    if block_tables.device != device or block_tables.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"block_tables must be int32/int64 on {device}")
    if block_tables.dim() != 2 or block_tables.stride(-1) != 1:
        raise ValueError(
            f"block_tables must be [B, max_pages] with a unit last-dim stride, got {block_tables.shape}"
        )
    if seq_lens.device != device or seq_lens.dtype != torch.int32 or seq_lens.dim() != 1:
        raise ValueError("seq_lens must be a 1-D int32 CUDA tensor")
    B, L = out.shape[0], out.shape[1]
    if seq_lens.shape[0] != B or block_tables.shape[0] != B:
        raise ValueError(
            f"seq_lens {tuple(seq_lens.shape)} / block_tables {tuple(block_tables.shape)} vs B={B}"
        )
    if not (inv_scale > 0.0):
        raise ValueError(f"inv_scale must be a positive float, got {inv_scale}")
    if B == 0 or L == 0:
        return
    sf_arg = kv_cache_sf.view(torch.uint8) if kv_cache_sf.dtype != torch.uint8 else kv_cache_sf
    _nvfp4_kv_to_e4m3_kernel[(triton.cdiv(L, _ROWS), B)](
        kv_cache,
        sf_arg,
        block_tables,
        seq_lens,
        out.view(torch.uint8),
        float(inv_scale),
        kv_cache.stride(0),
        kv_cache.stride(1),
        sf_arg.stride(0),
        sf_arg.stride(1),
        block_tables.stride(0),
        kv_cache.shape[0],
        kv_cache.shape[1],
        L,
        W=width,
        ROWS=_ROWS,
        num_warps=4,
    )


@nvfp4_kv_to_e4m3.register_fake
def _nvfp4_kv_to_e4m3_fake(
    kv_cache: torch.Tensor,
    kv_cache_sf: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    inv_scale: float,
) -> None:
    return None
