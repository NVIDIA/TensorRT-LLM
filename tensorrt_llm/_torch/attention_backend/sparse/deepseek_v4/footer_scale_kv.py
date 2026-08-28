# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 footer-scale KV packing for DeepSeek-V4 sparse MLA.

Each token occupies 584 bytes: 448 FP8 values, 64 BF16 RoPE values, and
seven UE8M0 scales plus one padding byte in the page footer.
"""

import torch
import triton
import triton.language as tl

DIM_NOPE = 448
DIM_ROPE = 64
QUANT_TILE = 64
NUM_NOPE_TILES = DIM_NOPE // QUANT_TILE
DATA_ROW_BYTES = DIM_NOPE + DIM_ROPE * 2
FOOTER_ROW_BYTES = NUM_NOPE_TILES + 1
TOKEN_BYTES = DATA_ROW_BYTES + FOOTER_ROW_BYTES
PAGE_SIZE = 64  # the main-pool page size the SM120 kernels are built for
PAGE_BYTES = PAGE_SIZE * TOKEN_BYTES

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = torch.finfo(_FP8_DTYPE).min


@triton.jit
def _quant_scatter_kernel(
    rows_ptr,  # bf16 [num_tokens, 512]
    loc_ptr,  # int32/int64 [num_tokens] global slot ids, < 0 skips
    pool_fp8_ptr,  # fp8 view of the pool buffer
    pool_bf16_ptr,  # bf16 view of the pool buffer
    pool_u8_ptr,  # uint8 view of the pool buffer
    rows_stride0,
    PAGE_SIZE_C: tl.constexpr,
    PAGE_BYTES_C: tl.constexpr,
    DATA_ROW_BYTES_C: tl.constexpr,
    FOOTER_OFFSET: tl.constexpr,
    FOOTER_ROW_BYTES_C: tl.constexpr,
    DIM_NOPE_C: tl.constexpr,
    DIM_ROPE_C: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    loc = tl.load(loc_ptr + token_id)
    if loc >= 0:
        # Pool byte offsets can exceed int32 even when slot ids do not.
        loc64 = loc.to(tl.int64)
        loc_page = loc64 // PAGE_SIZE_C
        loc_off = loc64 % PAGE_SIZE_C

        if tile_id == NUM_TILES:
            rope_range = tl.arange(0, DIM_ROPE_C)
            in_offsets = token_id * rows_stride0 + DIM_NOPE_C + rope_range
            rope = tl.load(rows_ptr + in_offsets)
            out_offsets = (
                loc_page * (PAGE_BYTES_C // 2)
                + (loc_off * DATA_ROW_BYTES_C + DIM_NOPE_C) // 2
                + rope_range
            )
            tl.store(pool_bf16_ptr + out_offsets, rope)
        else:
            tile_range = tl.arange(0, TILE)
            in_offsets = token_id * rows_stride0 + tile_id * TILE + tile_range
            x = tl.load(rows_ptr + in_offsets).to(tl.float32)

            max_abs = tl.maximum(tl.max(tl.abs(x)), EPS)
            ceil_log2 = tl.math.ceil(tl.log2(max_abs / FP8_MAX))
            scale_inv = tl.exp2(-ceil_log2)
            x_fp8 = tl.clamp(x * scale_inv, FP8_MIN, FP8_MAX).to(pool_fp8_ptr.dtype.element_ty)

            out_offsets = (
                loc_page * PAGE_BYTES_C + loc_off * DATA_ROW_BYTES_C + tile_id * TILE + tile_range
            )
            tl.store(pool_fp8_ptr + out_offsets, x_fp8)

            scale_u8 = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)
            scale_offset = (
                loc_page * PAGE_BYTES_C + FOOTER_OFFSET + loc_off * FOOTER_ROW_BYTES_C + tile_id
            )
            tl.store(pool_u8_ptr + scale_offset, scale_u8)


def quant_scatter(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    rows_bf16: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> None:
    """Pack and scatter BF16 latent rows into footer-scale pages."""
    page_bytes = page_size * TOKEN_BYTES
    assert pool_u8.dtype == torch.uint8 and pool_u8.is_contiguous()
    assert pool_u8.shape[-1] == page_bytes, (
        f"pool page bytes {pool_u8.shape[-1]} != {page_bytes} (page_size {page_size})"
    )
    assert rows_bf16.dtype == torch.bfloat16
    assert rows_bf16.shape[-1] == DIM_NOPE + DIM_ROPE
    assert loc.dtype in (torch.int32, torch.int64) and loc.is_contiguous()
    num_tokens = rows_bf16.shape[0]
    assert loc.shape[0] == num_tokens
    if num_tokens == 0:
        return

    rows = rows_bf16.contiguous()
    pool_fp8 = pool_u8.view(_FP8_DTYPE)
    pool_bf16 = pool_u8.view(torch.bfloat16)

    _quant_scatter_kernel[(num_tokens, NUM_NOPE_TILES + 1)](
        rows,
        loc,
        pool_fp8,
        pool_bf16,
        pool_u8,
        rows.stride(0),
        PAGE_SIZE_C=page_size,
        PAGE_BYTES_C=page_bytes,
        DATA_ROW_BYTES_C=DATA_ROW_BYTES,
        FOOTER_OFFSET=page_size * DATA_ROW_BYTES,
        FOOTER_ROW_BYTES_C=FOOTER_ROW_BYTES,
        DIM_NOPE_C=DIM_NOPE,
        DIM_ROPE_C=DIM_ROPE,
        TILE=QUANT_TILE,
        NUM_TILES=NUM_NOPE_TILES,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        EPS=1e-8,
    )


def dequant_gather(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Torch reference: gather footer-scale slots back to bf16 rows (tests only)."""
    assert pool_u8.dtype == torch.uint8
    page_bytes = page_size * TOKEN_BYTES
    footer_offset = page_size * DATA_ROW_BYTES
    pages = pool_u8.reshape(-1, page_bytes)
    page, off = loc // page_size, loc % page_size
    rows = torch.empty(
        loc.shape[0], DIM_NOPE + DIM_ROPE, dtype=torch.bfloat16, device=pool_u8.device
    )
    for i in range(loc.shape[0]):
        p, o = int(page[i]), int(off[i])
        row = pages[p, o * DATA_ROW_BYTES : (o + 1) * DATA_ROW_BYTES]
        scales = pages[
            p,
            footer_offset + o * FOOTER_ROW_BYTES : footer_offset
            + o * FOOTER_ROW_BYTES
            + NUM_NOPE_TILES,
        ]
        nope = row[:DIM_NOPE].view(_FP8_DTYPE).to(torch.float32)
        tile_scales = torch.pow(2.0, scales.to(torch.float32) - 127.0).repeat_interleave(QUANT_TILE)
        rows[i, :DIM_NOPE] = (nope * tile_scales).to(torch.bfloat16)
        rows[i, DIM_NOPE:] = row[DIM_NOPE:].view(torch.bfloat16)
    return rows
