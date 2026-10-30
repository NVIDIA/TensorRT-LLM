# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-E, the FP4 index-K cache format of the MiniMax-M3 MSA indexer.

Each 128-dim index-K row is stored as E2M1 values with one E4M3 scale per 16
dims. The scales are restricted to {1, 1.25, 1.5} x 2^e with e in [-6, 5], so
every dequantized value is exactly an E4M3 (and F16) value: the FP8 scorers
read dequantized keys without a second rounding.

Format contract, bit for bit the same in the torch reference, the Triton
kernels here and any other writer. Per 16-dim group of a finite BF16 row x, in
FP32 with IEEE division:

1. a = max |x_i|. If a == 0, the group stores scale byte 0 and zero codes.
2. e0 = floor(log2(a / 6)), read from the exponent field.
3. The nine candidates are s = m * 2^(e0 + d), each clamped to [2^-6, 48], in
   d-major order: d in (-1, 0, 1) outer, m in (1, 1.25, 1.5) inner.
4. For each, q_i = sign(x_i) * E2M1(|x_i| / s), rounded to nearest even and
   saturated at 6, and err = sum_i (q_i * s - x_i)^2 summed as a binary tree
   of adjacent pairs: (r0 + r1), (r2 + r3), ..., then pairs of those.
5. The smallest err wins; a tie keeps the earlier candidate.

A 4-bit code is sign << 3 | the index of |q| in {0, .5, 1, 1.5, 2, 3, 4, 6};
zero is always +0. Data byte j of a row is code(2j) | code(2j + 1) << 4 and a
scale byte is the E4M3 bit pattern of s.

A 128-token page takes 9216 B: token row r's 64 data bytes at 64 * r and its
8 scale bytes at 8192 + 8 * r. A zeroed page dequantizes to zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

INDEX_K_NVFP4_PAGE_BYTES = 9216
INDEX_K_NVFP4_SCALE_OFFSET = 8192

_HEAD_DIM = 128
_TOKENS_PER_PAGE = 128
_MIN_SCALE = 2.0**-6
_MAX_SCALE = 48.0
# Flooring a / 6 at 2^-120 keeps every 2^(e0 + d) a normal FP32 and changes no
# result: once e0 <= -8, every candidate clamps to 2^-6.
_TINY = 2.0**-120
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_DEQUANT_BLOCK_R = 32


def _e2m1_index_reference(a: torch.Tensor) -> torch.Tensor:
    # Counts the midpoints between neighbouring magnitudes that a passes; the
    # strict / non-strict comparisons send a tie to the even code.
    return (
        (a > 0.25).long()
        + (a >= 0.75).long()
        + (a > 1.25).long()
        + (a >= 1.75).long()
        + (a > 2.5).long()
        + (a >= 3.5).long()
        + (a > 5.0).long()
    )


def quantize_index_k_nvfp4_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize index-K rows [..., 128] to NVFP4-E in plain torch, on any device.

    `x` is first rounded to BF16, as the format contract requires. Returns the
    data bytes uint8 [..., 64] and the scale bytes uint8 [..., 8].
    """
    if x.shape[-1] != _HEAD_DIM:
        raise ValueError(f"index-K rows must have {_HEAD_DIM} dims, got {tuple(x.shape)}")
    groups = x.to(torch.bfloat16).float().reshape(-1, 8, 16)
    # The squared residual is sign-symmetric, so the search runs on magnitudes.
    mag = groups.abs()
    amax = mag.amax(dim=-1)
    e0 = (torch.clamp(amax / 6.0, min=_TINY).view(torch.int32) >> 23) - 127
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=x.device)
    for i in range(9):
        pow2 = ((e0 + (i // 3 - 1) + 127) << 23).view(torch.float32)
        scale = torch.clamp((1.0 + 0.25 * (i % 3)) * pow2, _MIN_SCALE, _MAX_SCALE)
        index = _e2m1_index_reference(mag / scale[..., None])
        residual = magnitudes[index] * scale[..., None] - mag
        err = residual * residual
        for _ in range(4):
            err = err[..., 0::2] + err[..., 1::2]
        err = err[..., 0]
        if i == 0:
            best_index, best_scale, best_err = index, scale, err
        else:
            take = err < best_err
            best_index = torch.where(take[..., None], index, best_index)
            best_scale = torch.where(take, scale, best_scale)
            best_err = torch.where(take, err, best_err)
    codes = torch.where((groups < 0) & (best_index > 0), best_index | 8, best_index)
    codes = codes.reshape(-1, 64, 2)
    data = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    scales = torch.where(amax > 0, best_scale, 0.0).to(torch.float8_e4m3fn).view(torch.uint8)
    return data.reshape(*x.shape[:-1], 64), scales.reshape(*x.shape[:-1], 8)


def dequantize_index_k_nvfp4_reference(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize NVFP4-E rows to FP32 [..., 128], exactly.

    `data` is uint8 [..., 64] and `scales` uint8 [..., 8], as returned by
    quantize_index_k_nvfp4_reference or read from a cache page.
    """
    if data.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(f"expected uint8 data and scales, got {data.dtype} and {scales.dtype}")
    if data.shape[-1] != 64 or scales.shape[-1] != 8:
        raise ValueError(
            f"expected [..., 64] data and [..., 8] scales, got "
            f"{tuple(data.shape)} and {tuple(scales.shape)}"
        )
    codes = torch.stack((data & 0xF, data >> 4), dim=-1).reshape(*data.shape[:-1], 8, 16).long()
    magnitude = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=data.device)[codes & 7]
    scale = scales.view(torch.float8_e4m3fn).float()[..., None]
    values = torch.where(codes >= 8, -magnitude, magnitude) * scale
    return values.reshape(*data.shape[:-1], _HEAD_DIM)


@triton.jit
def _e2m1_index(a):
    """Index of the E2M1 magnitude nearest to a >= 0, ties to even, saturating at 6."""
    return (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )


@triton.jit
def _e2m1_value(code):
    """Value of a 4-bit code (sign << 3 | magnitude index), assembled from FP32 bits."""
    index = (code & 7).to(tl.int32)
    bits = tl.where(
        index < 2, index * 0x3F000000, (((index >> 1) + 126) << 23) | ((index & 1) << 22)
    )
    return (bits | ((code & 8).to(tl.int32) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _tree_sum16(r):
    """Sum [N, 16] over the last dim as a binary tree of adjacent pairs, in index order."""
    a, b = tl.split(tl.reshape(r, (r.shape[0], 8, 2)))
    a, b = tl.split(tl.reshape(a + b, (r.shape[0], 4, 2)))
    a, b = tl.split(tl.reshape(a + b, (r.shape[0], 2, 2)))
    a, b = tl.split(a + b)
    return a + b


@triton.jit
def _quant_scatter_kernel(
    x_ptr,
    cache_ptr,
    slot_ptr,
    x_row_stride,
    page_stride,
    num_pages,
    TOKENS_PER_PAGE: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    MAX_SCALE: tl.constexpr,
    TINY: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    page = slot // TOKENS_PER_PAGE
    valid = (slot >= 0) & (page < num_pages)
    x = tl.load(x_ptr + token * x_row_stride + tl.arange(0, 128), mask=valid, other=0.0)
    x = tl.reshape(x.to(tl.float32), (8, 16))
    mag = tl.abs(x)
    amax = tl.max(mag, axis=1)
    e0 = (tl.maximum(tl.math.div_rn(amax, 6.0), TINY).to(tl.int32, bitcast=True) >> 23) - 127
    for i in tl.static_range(9):
        pow2 = ((e0 + (i // 3 - 1) + 127) << 23).to(tl.float32, bitcast=True)
        scale = tl.minimum(tl.maximum((1.0 + 0.25 * (i % 3)) * pow2, MIN_SCALE), MAX_SCALE)
        index = _e2m1_index(tl.math.div_rn(mag, scale[:, None]))
        residual = _e2m1_value(index) * scale[:, None] - mag
        err = _tree_sum16(residual * residual)
        if i == 0:
            best_index = index
            best_scale = scale
            best_err = err
        else:
            take = err < best_err
            best_index = tl.where(take[:, None], index, best_index)
            best_scale = tl.where(take, scale, best_scale)
            best_err = tl.where(take, err, best_err)
    codes = tl.where((x < 0) & (best_index > 0), best_index | 8, best_index)
    lo, hi = tl.split(tl.reshape(codes, (64, 2)))
    scale_bytes = tl.where(amax > 0, best_scale, 0.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    row_data = cache_ptr + page * page_stride + (slot % TOKENS_PER_PAGE) * 64
    row_scales = cache_ptr + page * page_stride + SCALE_OFFSET + (slot % TOKENS_PER_PAGE) * 8
    tl.store(row_data + tl.arange(0, 64), (lo | (hi << 4)).to(tl.uint8), mask=valid)
    tl.store(row_scales + tl.arange(0, 8), scale_bytes, mask=valid)


@triton.jit
def _dequant_pages_kernel(
    cache_ptr,
    page_ids_ptr,
    out_ptr,
    page_stride,
    num_pages,
    out_page_stride,
    SCALE_OFFSET: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    out_page = tl.program_id(0).to(tl.int64)
    rows = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    page = tl.load(page_ids_ptr + out_page).to(tl.int64)
    valid = (page >= 0) & (page < num_pages)
    page_base = cache_ptr + page * page_stride
    data = tl.load(page_base + rows[:, None] * 64 + tl.arange(0, 64)[None, :], mask=valid, other=0)
    scale_bytes = tl.load(
        page_base + SCALE_OFFSET + rows[:, None] * 8 + tl.arange(0, 8)[None, :],
        mask=valid,
        other=0,
    )
    codes = tl.reshape(tl.join(data & 15, data >> 4), (BLOCK_R, 8, 16))
    scale = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    values = _e2m1_value(codes) * scale[:, :, None]
    tl.store(
        out_ptr + out_page * out_page_stride + rows[:, None] * 128 + tl.arange(0, 128)[None, :],
        tl.reshape(values, (BLOCK_R, 128)).to(tl.float8e4nv),
    )


def _check_cache(cache: torch.Tensor) -> None:
    if (
        cache.dtype != torch.uint8
        or cache.dim() != 2
        or cache.stride(1) != 1
        or cache.shape[1] < INDEX_K_NVFP4_PAGE_BYTES
        or (cache.shape[0] > 1 and cache.stride(0) < INDEX_K_NVFP4_PAGE_BYTES)
    ):
        raise ValueError(
            f"NVFP4 index-K cache must be uint8 [num_pages, >= {INDEX_K_NVFP4_PAGE_BYTES}] with "
            f"contiguous, non-overlapping pages, got {cache.dtype} {tuple(cache.shape)} "
            f"strides {cache.stride()}"
        )


def _check_index(name: str, index: torch.Tensor) -> None:
    if index.dtype not in (torch.int32, torch.int64) or index.dim() != 1:
        raise ValueError(
            f"{name} must be a 1-D int32/int64 tensor, got {index.dtype} {index.dim()}-D"
        )
    if not index.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def index_k_nvfp4_quant_scatter(
    idx_k: torch.Tensor, cache: torch.Tensor, out_cache_loc: torch.Tensor
) -> None:
    """Quantize index-K rows to NVFP4-E and store each at its cache slot.

    `idx_k` is BF16 [T, 128] (rows may be strided), `cache` is uint8
    [num_pages, page_stride >= 9216] with one page per row, and `out_cache_loc`
    is int32/int64 [T], one slot (page * 128 + token row) per row. Slots < 0
    and slots on a page >= num_pages are skipped. One launch and no host
    synchronization, so the call can be captured in a CUDA graph.
    """
    _check_cache(cache)
    _check_index("out_cache_loc", out_cache_loc)
    if idx_k.dtype != torch.bfloat16 or idx_k.dim() != 2 or idx_k.shape[1] != _HEAD_DIM:
        raise ValueError(
            f"idx_k must be BF16 [T, {_HEAD_DIM}], got {idx_k.dtype} {tuple(idx_k.shape)}"
        )
    if idx_k.stride(1) != 1 or idx_k.shape[0] != out_cache_loc.shape[0]:
        raise ValueError(
            f"idx_k {tuple(idx_k.shape)} strides {idx_k.stride()} needs contiguous rows and one "
            f"slot per row, got {out_cache_loc.shape[0]} slots"
        )
    num_tokens = idx_k.shape[0]
    if num_tokens == 0:
        return
    # One warp per token, four dims per lane. The error sums decide the scale, so they
    # must round as the contract's do: no FMA contraction.
    _quant_scatter_kernel[(num_tokens,)](
        idx_k,
        cache,
        out_cache_loc,
        idx_k.stride(0),
        cache.stride(0),
        cache.shape[0],
        TOKENS_PER_PAGE=_TOKENS_PER_PAGE,
        SCALE_OFFSET=INDEX_K_NVFP4_SCALE_OFFSET,
        MIN_SCALE=_MIN_SCALE,
        MAX_SCALE=_MAX_SCALE,
        TINY=_TINY,
        num_warps=1,
        enable_fp_fusion=False,
    )


def index_k_nvfp4_dequant_pages(
    cache: torch.Tensor, page_ids: torch.Tensor, out: torch.Tensor
) -> None:
    """Dequantize whole NVFP4-E pages into an E4M3 workspace.

    `cache` is uint8 [num_pages, page_stride >= 9216] with one page per row,
    `page_ids` is int32/int64 [P] and `out` is float8_e4m3fn [P, 1, 128, 128]
    with contiguous pages; `out[i, 0]` receives page `page_ids[i]`, or zeros
    for an id < 0 or >= num_pages. The values are exact, since every NVFP4-E
    value is an E4M3 value. One launch and no host synchronization.
    """
    _check_cache(cache)
    _check_index("page_ids", page_ids)
    num_out = page_ids.shape[0]
    if (
        out.dtype != torch.float8_e4m3fn
        or tuple(out.shape) != (num_out, 1, _TOKENS_PER_PAGE, _HEAD_DIM)
        or out.stride(2) != _HEAD_DIM
        or out.stride(3) != 1
        or (num_out > 1 and out.stride(0) < _TOKENS_PER_PAGE * _HEAD_DIM)
    ):
        raise ValueError(
            f"out must be float8_e4m3fn [{num_out}, 1, {_TOKENS_PER_PAGE}, {_HEAD_DIM}] with "
            f"contiguous, non-overlapping pages, got {out.dtype} {tuple(out.shape)} "
            f"strides {out.stride()}"
        )
    if num_out == 0:
        return
    _dequant_pages_kernel[(num_out, _TOKENS_PER_PAGE // _DEQUANT_BLOCK_R)](
        cache,
        page_ids,
        out,
        cache.stride(0),
        cache.shape[0],
        out.stride(0),
        SCALE_OFFSET=INDEX_K_NVFP4_SCALE_OFFSET,
        BLOCK_R=_DEQUANT_BLOCK_R,
    )


__all__ = [
    "INDEX_K_NVFP4_PAGE_BYTES",
    "INDEX_K_NVFP4_SCALE_OFFSET",
    "dequantize_index_k_nvfp4_reference",
    "index_k_nvfp4_dequant_pages",
    "index_k_nvfp4_quant_scatter",
    "quantize_index_k_nvfp4_reference",
]
