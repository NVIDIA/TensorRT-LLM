# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kept-entry remap + NVFP4 -> E4M3 gather of the Helix DSA generation rows (K3.1 Rubin, the family DSA backend).

The other generation path remaps each whole ``top_k``-wide row (``convert_req_index_to_global``) and gathers every
``rows x top_k`` pair (``nvfp4_mla_kv_cache_gather``). ``helix_kept_gather`` does both in one launch over the kept
entries only (a helix row holds this rank's share of the Top-K first, then -1), and what the sparse MLA reads is
bitwise the same. The conversions are those of ``dequantizeRow<false>`` in ``nvfp4MlaKvCacheGather.cu``, run as
inline PTX on 32-bit words; the two-level RoPE (residual 64) takes the FP32 FMA order nvcc emitted for the C++ kernel,
calibrated on SM107 against the C++ op over every input.

CUDA-graph safe: a static grid over ``rows x (512 / BLOCK_E)`` programs; row lengths, indices and the global scale are
read on the device; no allocation and no host synchronization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

HEAD_DIM = 576
ROPE_DIM = 64
# The NVFP4 latent layouts served, by mla_kv_cache_residual_dim: 64 = two-level RoPE, 320 data bytes (256 NoPE + 4 x
# (8 main + 8 residual) RoPE) + 40 scale bytes (32 NoPE + 4 x (main, residual) RoPE) per token (DSACacheManagerV2);
# 0 = one level for all 576 values, 288 data + 36 scale bytes per token (the hybrid KDA / DSA cache manager).
RESIDUAL_DIMS = (ROPE_DIM, 0)

# Launch shape: a program serves BLOCK_E consecutive columns of one row in each 512-column strip.
BLOCK_E = 32
NUM_WARPS = 4
STRIP_COLS = 512


def nvfp4_row_bytes(residual_dim: int) -> Tuple[int, int]:
    """(data, scale) bytes per token of an NVFP4 MLA latent row of a ``RESIDUAL_DIMS`` layout."""
    return (HEAD_DIM + residual_dim) // 2, (HEAD_DIM + residual_dim) // 16


# Eight E2M1 values (one 32-bit word) times an FP16 block scale (both halves of $3) -> eight E4M3 bytes, the
# conversions of convertGloballyScaledE2m1x4ToE4m3: exact FP16 values, an FP16 product rounded to nearest, then one
# saturating round-to-nearest-even E4M3 conversion. Value order: low nibble first, low byte first.
_NOPE_ASM = """
{
    .reg .b8 b0, b1, b2, b3;
    .reg .b32 h0, h1, h2, h3;
    .reg .b16 e0, e1, e2, e3;
    mov.b32 {b0, b1, b2, b3}, $2;
    cvt.rn.f16x2.e2m1x2 h0, b0;
    cvt.rn.f16x2.e2m1x2 h1, b1;
    cvt.rn.f16x2.e2m1x2 h2, b2;
    cvt.rn.f16x2.e2m1x2 h3, b3;
    mul.rn.f16x2 h0, h0, $3;
    mul.rn.f16x2 h1, h1, $3;
    mul.rn.f16x2 h2, h2, $3;
    mul.rn.f16x2 h3, h3, $3;
    cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
    cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
    cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;
    cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;
    mov.b32 $0, {e0, e1};
    mov.b32 $1, {e2, e3};
}
"""

# Eight two-level RoPE values: main word $2 and residual word $3 (E2M1), FP32 scales A = $4 and B = $5 (block scale x
# G) -> e4m3(fma(main, A, residual * B)), the FP32 sequence nvcc emitted for convertGloballyScaledResidualE2m1x4ToE4m3.
_ROPE_ASM = """
{
    .reg .b8 m0, m1, m2, m3, r0, r1, r2, r3;
    .reg .b32 hm0, hm1, hm2, hm3, hr0, hr1, hr2, hr3;
    .reg .b16 a0, a1, a2, a3, a4, a5, a6, a7, c0, c1, c2, c3, c4, c5, c6, c7;
    .reg .f32 x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7;
    .reg .b16 e0, e1, e2, e3;
    mov.b32 {m0, m1, m2, m3}, $2;
    mov.b32 {r0, r1, r2, r3}, $3;
    cvt.rn.f16x2.e2m1x2 hm0, m0;
    cvt.rn.f16x2.e2m1x2 hm1, m1;
    cvt.rn.f16x2.e2m1x2 hm2, m2;
    cvt.rn.f16x2.e2m1x2 hm3, m3;
    cvt.rn.f16x2.e2m1x2 hr0, r0;
    cvt.rn.f16x2.e2m1x2 hr1, r1;
    cvt.rn.f16x2.e2m1x2 hr2, r2;
    cvt.rn.f16x2.e2m1x2 hr3, r3;
    mov.b32 {a0, a1}, hm0;
    mov.b32 {a2, a3}, hm1;
    mov.b32 {a4, a5}, hm2;
    mov.b32 {a6, a7}, hm3;
    mov.b32 {c0, c1}, hr0;
    mov.b32 {c2, c3}, hr1;
    mov.b32 {c4, c5}, hr2;
    mov.b32 {c6, c7}, hr3;
    cvt.f32.f16 x0, a0;
    cvt.f32.f16 x1, a1;
    cvt.f32.f16 x2, a2;
    cvt.f32.f16 x3, a3;
    cvt.f32.f16 x4, a4;
    cvt.f32.f16 x5, a5;
    cvt.f32.f16 x6, a6;
    cvt.f32.f16 x7, a7;
    cvt.f32.f16 y0, c0;
    cvt.f32.f16 y1, c1;
    cvt.f32.f16 y2, c2;
    cvt.f32.f16 y3, c3;
    cvt.f32.f16 y4, c4;
    cvt.f32.f16 y5, c5;
    cvt.f32.f16 y6, c6;
    cvt.f32.f16 y7, c7;
    mul.rn.f32 y0, y0, $5;
    mul.rn.f32 y1, y1, $5;
    mul.rn.f32 y2, y2, $5;
    mul.rn.f32 y3, y3, $5;
    mul.rn.f32 y4, y4, $5;
    mul.rn.f32 y5, y5, $5;
    mul.rn.f32 y6, y6, $5;
    mul.rn.f32 y7, y7, $5;
    fma.rn.f32 x0, x0, $4, y0;
    fma.rn.f32 x1, x1, $4, y1;
    fma.rn.f32 x2, x2, $4, y2;
    fma.rn.f32 x3, x3, $4, y3;
    fma.rn.f32 x4, x4, $4, y4;
    fma.rn.f32 x5, x5, $4, y5;
    fma.rn.f32 x6, x6, $4, y6;
    fma.rn.f32 x7, x7, $4, y7;
    cvt.rn.satfinite.e4m3x2.f32 e0, x1, x0;
    cvt.rn.satfinite.e4m3x2.f32 e1, x3, x2;
    cvt.rn.satfinite.e4m3x2.f32 e2, x5, x4;
    cvt.rn.satfinite.e4m3x2.f32 e3, x7, x6;
    mov.b32 $0, {e0, e1};
    mov.b32 $1, {e2, e3};
}
"""


@triton.jit
def _e4m3_to_f32(byte):
    """E4M3 byte (int32 0..255, finite encodings) -> FP32, exact."""
    e = (byte >> 3) & 15
    m = byte & 7
    normal = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125  # m * 2^-9
    mag = tl.where(e == 0, sub, normal)
    return tl.where((byte & 0x80) != 0, -mag, mag)


@triton.jit
def _kept_remap(
    topk_ptr,
    stride_topk_row,
    req,
    bt_ptr,
    stride_bt_row,
    stride_bt_col,
    max_blocks,
    num_pool_tokens,
    block_size,
    stride_factor,
    layer_term,
    row,
    cols,
    row_len,
    TOPK: tl.constexpr,
):
    """convert_req_index_to_global + the C++ gather's validity test for the kept columns of one row."""
    kept = cols < row_len
    tok = tl.load(topk_ptr + row.to(tl.int64) * stride_topk_row + cols, mask=kept, other=-1)
    has_tok = kept & (tok >= 0)
    blk = tl.where(has_tok, tok // block_size, 0)
    in_table = has_tok & (blk < max_blocks)
    base = tl.load(
        bt_ptr + req.to(tl.int64) * stride_bt_row + blk.to(tl.int64) * stride_bt_col,
        mask=in_table,
        other=-1,
    )
    gidx = base * stride_factor + (tok % block_size + layer_term)
    valid = in_table & (base >= 0) & (gidx >= 0) & (gidx.to(tl.int64) < num_pool_tokens)
    pair = row * TOPK + cols
    return valid, tl.where(valid, gidx, 0).to(tl.int64), pair


@triton.jit
def _helix_kept_gather_ptx_kernel(
    topk_ptr,
    stride_topk_row,
    lens_ptr,
    req_ptr,
    bt_ptr,
    stride_bt_row,
    stride_bt_col,
    max_blocks,
    data_ptr,  # int32 view of the data pool (80 words per token at residual 64, 72 at residual 0)
    scale_ptr,  # uint8 scale pool (40 bytes per token at residual 64, 36 at residual 0)
    dequant_ptr,
    num_pool_tokens,
    out_ptr,  # int32 view of the scratch (144 words per row)
    compact_ptr,
    block_size,
    stride_factor,
    layer_term,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NCHUNK: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    row_len = tl.minimum(tl.maximum(tl.load(lens_ptr + row), 0), TOPK)
    req = tl.load(req_ptr + row)
    gscale = tl.load(dequant_ptr)
    offs_e = tl.arange(0, BLOCK_E)
    words = tl.arange(0, 64)
    groups = tl.arange(0, 32)
    out_words = tl.arange(0, 128)
    q = tl.arange(0, 8)  # RoPE word pair: group q // 2, half q % 2
    main_word = 64 + 4 * (q // 2) + q % 2
    main_scale = 32 + 2 * (q // 2)
    rope_out = 128 + tl.arange(0, 16)
    compact_row = compact_ptr + row.to(tl.int64) * TOPK
    for s in range(TOPK // (NCHUNK * BLOCK_E)):
        col0 = s * (NCHUNK * BLOCK_E) + chunk * BLOCK_E
        cols = col0 + offs_e
        if col0 < row_len:
            valid, g64, pair = _kept_remap(
                topk_ptr,
                stride_topk_row,
                req,
                bt_ptr,
                stride_bt_row,
                stride_bt_col,
                max_blocks,
                num_pool_tokens,
                block_size,
                stride_factor,
                layer_term,
                row,
                cols,
                row_len,
                TOPK,
            )
            tl.store(compact_row + cols, tl.where(valid, pair, -1))
            v2 = valid[:, None]
            src = data_ptr + g64 * ((576 + RESIDUAL) // 8)
            ssrc = scale_ptr + g64 * ((576 + RESIDUAL) // 16)
            dst = out_ptr + pair.to(tl.int64) * 144
            # NoPE: 64 words, block scale g covers words 2g and 2g + 1.
            packed = tl.load(src[:, None] + words[None, :], mask=v2, other=0)
            sbyte = tl.load(ssrc[:, None] + groups[None, :], mask=v2, other=0).to(tl.int32)
            s16 = (_e4m3_to_f32(sbyte) * gscale).to(tl.float16).to(tl.int16, bitcast=True)
            s16 = s16.to(tl.int32) & 0xFFFF
            s2 = s16 | (s16 << 16)
            s2w = tl.join(s2, s2).reshape(BLOCK_E, 64)
            lo, hi = tl.inline_asm_elementwise(
                _NOPE_ASM_C,
                "=r,=r,r,r",
                [packed, s2w],
                dtype=(tl.int32, tl.int32),
                is_pure=True,
                pack=1,
            )
            tl.store(
                dst[:, None] + out_words[None, :], tl.join(lo, hi).reshape(BLOCK_E, 128), mask=v2
            )
            if RESIDUAL == 0:
                # One-level RoPE: groups 32-35 in words 64-71 (scale byte 32 + q // 2 per word), as NoPE.
                tw = tl.load(src[:, None] + (64 + q)[None, :], mask=v2, other=0)
                tsb = tl.load(ssrc[:, None] + (32 + q // 2)[None, :], mask=v2, other=0).to(tl.int32)
                t16 = (_e4m3_to_f32(tsb) * gscale).to(tl.float16).to(tl.int16, bitcast=True)
                t16 = t16.to(tl.int32) & 0xFFFF
                rope_lo, rope_hi = tl.inline_asm_elementwise(
                    _NOPE_ASM_C,
                    "=r,=r,r,r",
                    [tw, t16 | (t16 << 16)],
                    dtype=(tl.int32, tl.int32),
                    is_pure=True,
                    pack=1,
                )
                tl.store(
                    dst[:, None] + rope_out[None, :],
                    tl.join(rope_lo, rope_hi).reshape(BLOCK_E, 16),
                    mask=v2,
                )
            else:
                # RoPE: per group 2 main words then 2 residual words; A / B are the group's scales x G (FP32).
                mw = tl.load(src[:, None] + main_word[None, :], mask=v2, other=0)
                rw = tl.load(src[:, None] + (main_word + 2)[None, :], mask=v2, other=0)
                smb = tl.load(ssrc[:, None] + main_scale[None, :], mask=v2, other=0).to(tl.int32)
                srb = tl.load(ssrc[:, None] + (main_scale + 1)[None, :], mask=v2, other=0).to(
                    tl.int32
                )
                sa = _e4m3_to_f32(smb) * gscale
                sb = _e4m3_to_f32(srb) * gscale
                rlo, rhi = tl.inline_asm_elementwise(
                    _ROPE_ASM_C,
                    "=r,=r,r,r,f,f",
                    [mw, rw, sa, sb],
                    dtype=(tl.int32, tl.int32),
                    is_pure=True,
                    pack=1,
                )
                tl.store(
                    dst[:, None] + rope_out[None, :],
                    tl.join(rlo, rhi).reshape(BLOCK_E, 16),
                    mask=v2,
                )
        else:
            tl.store(compact_row + cols, tl.full([BLOCK_E], -1, tl.int32))


_NOPE_ASM_C = tl.constexpr(_NOPE_ASM)
_ROPE_ASM_C = tl.constexpr(_ROPE_ASM)

_POOL_VIEWS: Dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def nvfp4_pool_views(
    data_address: int, scale_address: int, num_pool_tokens: int, residual_dim: int = ROPE_DIM
) -> Tuple[torch.Tensor, torch.Tensor]:
    """uint8 views of an NVFP4 MLA data pool and scale pool from their addresses (``nvfp4_row_bytes(residual_dim)``
    per token).

    The views only carry the base pointers into the kernel (the pools are allocated once and never move); they are
    cached per (data, scale, tokens, residual)."""
    key = (int(data_address), int(scale_address), int(num_pool_tokens), int(residual_dim))
    views = _POOL_VIEWS.get(key)
    if views is None:
        data_bytes, scale_bytes = nvfp4_row_bytes(key[3])
        data = convert_to_torch_tensor(TensorWrapper(key[0], torch.uint8, [key[2] * data_bytes]))
        scale = convert_to_torch_tensor(TensorWrapper(key[1], torch.uint8, [key[2] * scale_bytes]))
        views = (data, scale)
        _POOL_VIEWS[key] = views
    return views


def helix_kept_gather(
    topk_indices: torch.Tensor,
    row_lens: torch.Tensor,
    req_idx: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    stride_factor: int,
    layer_offset: int,
    data_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    dequant_scale: torch.Tensor,
    num_pool_tokens: int,
    scratch: torch.Tensor,
    compact_indices: torch.Tensor,
    *,
    residual_dim: int = ROPE_DIM,
) -> None:
    """Remap the kept local Top-K entries of every row and gather their NVFP4 rows into the E4M3 scratch.

    Args:
        topk_indices: int32 ``[rows, top_k]`` local token indices (kept entries first, -1 padded).
        row_lens: int32 ``[rows]`` row lengths of the token-sparse MLA (the kept counts).
        req_idx: int32 ``[rows]`` request of each row (``_cached_req_idx_gen``).
        block_table: int32 ``[requests, max_blocks]`` (``_cached_block_table_gen``).
        tokens_per_block, stride_factor, layer_offset: the remap geometry of ``convert_req_index_to_global``
            (``stride_factor`` = page index scale x tokens per block).
        data_pool, scale_pool: uint8 views of the layer's NVFP4 pools (``nvfp4_pool_views``).
        dequant_scale: FP32 ``[1]`` global dequantization scale (``!= 1``; the unit-scale branch is not replicated).
        num_pool_tokens: pool rows (indices at or above it are invalid, as in the C++ gather).
        scratch: E4M3 ``[rows, top_k, 576]`` output (``nvfp4_mla_fp8_scratch[:rows]``).
        compact_indices: int32 ``[rows, top_k]`` output (``nvfp4_mla_compact_indices[:rows]``).
        residual_dim: the pools' layout (``mla_kv_cache_residual_dim``, one of ``RESIDUAL_DIMS``).
    """
    rows, top_k = topk_indices.shape
    if rows == 0:
        return
    if top_k % STRIP_COLS:
        raise ValueError(f"helix kept gather needs top_k % {STRIP_COLS} == 0")
    if residual_dim not in RESIDUAL_DIMS:
        raise ValueError(
            f"helix kept gather serves residual_dim {RESIDUAL_DIMS}, got {residual_dim}"
        )
    data_bytes, scale_bytes = nvfp4_row_bytes(residual_dim)
    if (
        data_pool.numel() != num_pool_tokens * data_bytes
        or scale_pool.numel() != num_pool_tokens * scale_bytes
    ):
        raise ValueError(
            f"NVFP4 pools of {num_pool_tokens} tokens at residual {residual_dim} hold {data_bytes} + "
            f"{scale_bytes} B per token, got {data_pool.numel()} + {scale_pool.numel()} B"
        )
    if (
        topk_indices.stride(1) != 1
        or not compact_indices.is_contiguous()
        or not scratch.is_contiguous()
    ):
        raise ValueError("helix kept gather needs unit-stride index rows and contiguous outputs")
    if scratch.shape[0] < rows or scratch.shape[1] != top_k or scratch.shape[2] != HEAD_DIM:
        raise ValueError(
            f"scratch must be [>= {rows}, {top_k}, {HEAD_DIM}], got {tuple(scratch.shape)}"
        )
    nchunk = STRIP_COLS // BLOCK_E
    common = (
        topk_indices,
        topk_indices.stride(0),
        row_lens,
        req_idx,
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        block_table.shape[1],
    )
    geometry = (tokens_per_block, stride_factor, layer_offset * tokens_per_block)
    _helix_kept_gather_ptx_kernel[(rows, nchunk)](
        *common,
        data_pool.view(torch.int32),
        scale_pool,
        dequant_scale,
        num_pool_tokens,
        scratch.view(torch.int32),
        compact_indices,
        *geometry,
        TOPK=top_k,
        BLOCK_E=BLOCK_E,
        NCHUNK=nchunk,
        RESIDUAL=int(residual_dim),
        num_warps=NUM_WARPS,
        enable_fp_fusion=False,
    )


__all__ = [
    "RESIDUAL_DIMS",
    "helix_kept_gather",
    "nvfp4_pool_views",
    "nvfp4_row_bytes",
]
