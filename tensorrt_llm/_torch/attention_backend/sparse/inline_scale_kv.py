# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline-scale KV pages for the SM120 FlashInfer sparse MLA path (DSA family).

"Inline-scale" is flashinfer's name for the V32-family cache (its DSv3.2/GLM
SM120 kernels read it; DeepSeek-V4 uses the footer-scale variant):
quantization scales live inline with each token row, FlashMLA ABI. Per-token
row layout:

  bytes [0, 512):    FP8 E4M3 nope (4 tiles x 128)
  bytes [512, 528):  4 x FP32 tile scales
  bytes [528, 656):  BF16 rope (64 elements x 2 B)

which makes ``TOKEN_BYTES = 512 + 16 + 128 = 656`` bytes per token.

Quantization: each 128-element tile of the 512-element nope half is scaled by
``max|x| / FP8_MAX`` — an arbitrary FP32 scale, not a power of two — and
stored as FP8 E4M3 with the FP32 scale inline (flashinfer's
``kv_scale_format="arbitrary_fp32"`` / GLM mode; DSv3.2 pow2 scales are a
special case this writer does not produce). The 64-element rope half stays
bf16. Semantics mirror SGLang's ``quantize_k_cache`` packer byte for byte,
including the deliberate absence of an epsilon guard on the tile max.

Sparse indices address tokens as global pool slot ids (``page = slot //
page_size``, ``offset = slot % page_size``) — the currency
``convert_req_index_to_global`` emits; negative slots are skipped so masked
writes need no host-side filtering.
"""

import torch
import triton
import triton.language as tl

DIM_NOPE = 512
DIM_ROPE = 64
QUANT_TILE = 128
NUM_NOPE_TILES = DIM_NOPE // QUANT_TILE  # 4
SCALE_BYTES = NUM_NOPE_TILES * 4  # 4 x fp32
ROPE_OFFSET = DIM_NOPE + SCALE_BYTES  # 528
TOKEN_BYTES = DIM_NOPE + SCALE_BYTES + DIM_ROPE * 2  # 656
PAGE_SIZE = 64  # the page size the SM120 DSv3.2/GLM kernels are built for
PAGE_BYTES = PAGE_SIZE * TOKEN_BYTES

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = torch.finfo(_FP8_DTYPE).min


@triton.jit
def _quant_scatter_kernel(
    rows_ptr,  # bf16 [num_tokens, 576]
    loc_ptr,  # int32/int64 [num_tokens] global slot ids, < 0 skips
    pool_fp8_ptr,  # fp8 view of the pool buffer
    pool_bf16_ptr,  # bf16 view of the pool buffer
    pool_f32_ptr,  # fp32 view of the pool buffer
    rows_stride0,
    PAGE_SIZE_C: tl.constexpr,
    PAGE_BYTES_C: tl.constexpr,
    TOKEN_BYTES_C: tl.constexpr,
    ROPE_OFFSET_C: tl.constexpr,
    DIM_NOPE_C: tl.constexpr,
    DIM_ROPE_C: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    loc = tl.load(loc_ptr + token_id)
    if loc >= 0:
        # Byte offsets below multiply the page ordinal by the page byte size;
        # keep that arithmetic 64-bit — an int32 slot id is fine in the token
        # domain, but its byte products overflow past a 2 GiB pool.
        loc64 = loc.to(tl.int64)
        loc_page = loc64 // PAGE_SIZE_C
        loc_off = loc64 % PAGE_SIZE_C
        row_base = loc_page * PAGE_BYTES_C + loc_off * TOKEN_BYTES_C

        if tile_id == NUM_TILES:
            # bf16 rope half: bytes [ROPE_OFFSET, TOKEN_BYTES) of the token
            # row. Branch-local names: Triton unifies same-named variables
            # across if/else arms, and the rope range (64) and quant tile
            # (128) have different shapes.
            rope_range = tl.arange(0, DIM_ROPE_C)
            rope_in_offsets = token_id * rows_stride0 + DIM_NOPE_C + rope_range
            rope = tl.load(rows_ptr + rope_in_offsets)
            rope_out_offsets = (row_base + ROPE_OFFSET_C) // 2 + rope_range
            tl.store(pool_bf16_ptr + rope_out_offsets, rope)
        else:
            # one 128-element nope tile: arbitrary fp32 scale + fp8 values.
            # Mirrors SGLang's packer exactly: scale = max|x| / FP8_MAX with
            # no epsilon guard, quantize as x * (1 / scale).
            tile_range = tl.arange(0, TILE)
            tile_in_offsets = token_id * rows_stride0 + tile_id * TILE + tile_range
            x = tl.load(rows_ptr + tile_in_offsets).to(tl.float32)

            y_s = tl.max(tl.abs(x)) / FP8_MAX
            y_s_inv = 1.0 / y_s
            x_fp8 = tl.clamp(x * y_s_inv, FP8_MIN, FP8_MAX).to(
                pool_fp8_ptr.dtype.element_ty
            )

            tile_out_offsets = row_base + tile_id * TILE + tile_range
            tl.store(pool_fp8_ptr + tile_out_offsets, x_fp8)

            scale_offset = (row_base + DIM_NOPE_C) // 4 + tile_id
            tl.store(pool_f32_ptr + scale_offset, y_s)


def quant_scatter(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    rows_bf16: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> None:
    """Quantize bf16 latent rows and scatter them into an inline-scale pool.

    Args:
        pool_u8: uint8 pool buffer, shape [num_pages, page_size * TOKEN_BYTES],
            contiguous, anchored where the pool's slot ids are zero.
        loc: [num_tokens] int32/int64 global slot ids; entries < 0 are skipped.
        rows_bf16: [num_tokens, 576] bf16 rows ([512 nope | 64 rope], rope
            already rotated).
        page_size: tokens per page — 64 for the SM120 DSv3.2/GLM kernels.
    """
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
    pool_f32 = pool_u8.view(torch.float32)

    _quant_scatter_kernel[(num_tokens, NUM_NOPE_TILES + 1)](
        rows,
        loc,
        pool_fp8,
        pool_bf16,
        pool_f32,
        rows.stride(0),
        PAGE_SIZE_C=page_size,
        PAGE_BYTES_C=page_bytes,
        TOKEN_BYTES_C=TOKEN_BYTES,
        ROPE_OFFSET_C=ROPE_OFFSET,
        DIM_NOPE_C=DIM_NOPE,
        DIM_ROPE_C=DIM_ROPE,
        TILE=QUANT_TILE,
        NUM_TILES=NUM_NOPE_TILES,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
    )


def dequant_gather(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Torch reference: gather inline-scale slots back to bf16 rows (tests only)."""
    assert pool_u8.dtype == torch.uint8
    page_bytes = page_size * TOKEN_BYTES
    pages = pool_u8.reshape(-1, page_bytes)
    page, off = loc // page_size, loc % page_size
    rows = torch.empty(
        loc.shape[0], DIM_NOPE + DIM_ROPE, dtype=torch.bfloat16, device=pool_u8.device
    )
    for i in range(loc.shape[0]):
        p, o = int(page[i]), int(off[i])
        row = pages[p, o * TOKEN_BYTES : (o + 1) * TOKEN_BYTES]
        nope = row[:DIM_NOPE].view(_FP8_DTYPE).to(torch.float32)
        scales = row[DIM_NOPE:ROPE_OFFSET].view(torch.float32)
        tile_scales = scales.repeat_interleave(QUANT_TILE)
        rows[i, :DIM_NOPE] = (nope * tile_scales).to(torch.bfloat16)
        rows[i, DIM_NOPE:] = row[ROPE_OFFSET:].view(torch.bfloat16)
    return rows
