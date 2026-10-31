# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sparse decode attention layer's work between its input projection and its index
scores, one PDL launch.

The composition it replaces ran four launches over two streams: the QK norm + rope of
both planes (`qwen4_exp_qk_norm_rope`), the paged K/V append (`qsa_paged_kv_store`), and on
the side stream the index plane's raw-key and coordinate store (`qsa_paged_index_store`)
and the group compression (`qsa_prefill_compress`), then a cross-stream join. Here the
programs of `qwen4_exp_qk_norm_rope` run as they do there; the ones holding k heads also
write their rotated rows into the K pool; one program per row and v head stores v; one
more program per row stores the index key and the coordinates, and -- at a group's last
token -- pools the group with this token's key taken from the projection row (the cached
copy is the one being written), normalizes, rotates and stores it at the anchor slot, as
the compression kernel does.

Over an NVFP4 K/V pool the K/V append is `qsa_paged_kv_store_nvfp4`'s quantizing store:
the programs holding k heads quantize their rotated, bf16-rounded rows from registers and
the v programs quantize v from the projection row, with the store's arithmetic, so the
pool's e2m1 bytes and e4m3 block scales are bitwise the composition's.

A verify step (`rows_per_request` > 1, each request's rows consecutive at consecutive
positions) writes several raw keys of one group in the launch: at a group's last token
the earlier members held by the same request's preceding rows are taken from their
projection rows, the others from the index plane.

At up to `SCORE_ROWS` decode rows the index scores can ride the launch as well
(`index_scores`): further programs run `qsa_paged_index_scores`' decode tiles, each
rotating the row's index query itself and, at a group's last token, compressing the last
visible block's key itself -- the same operations on the same values as the programs that
store them, so the scores are bitwise the separate launch's.

With PDL every operand that is static or host-filled -- the page and slot, the rotation
factors at the row's positions, the norm weights, the K/V second-level scales, the
compressed keys a score tile reads, and at a group end the group's earlier keys from the
index plane with their rotation -- is read before `griddepcontrol.wait`; after it a
program reads only projection rows (a score tile of a row without a page re-reads its
last visible key, which the launch does not write).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_index_scores import (
    qsa_paged_index_scores as _index_scores,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_index_store import (
    qsa_paged_index_store as _index_store,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from .qsa_paged_kv_store import qsa_paged_kv_store as _kv_store
from .qsa_paged_kv_store_nvfp4 import _e2m1x2, _quantize_row, _rcp_approx
from .qsa_paged_kv_store_nvfp4 import qsa_paged_kv_store_nvfp4 as _kv_store_nvfp4
from .qsa_prefill_compress import qsa_prefill_compress as _compress
from .qwen4_exp_qk_norm_rope import _launch as _qk_launch
from .qwen4_exp_qk_norm_rope import qwen4_exp_qk_norm_rope as _qk_norm_rope

#: Rows up to which the index scores can ride the launch, as `qsa_paged_index_scores`' decode
#: tiles (SCORE_BN blocks a program, two warps; the norm + rope plan is two warps there too).
#: GB300, 12 chained layers at ~8K tokens, us a layer: 1 row 15.0 -> 13.7, 2 rows 15.8 -> 14.3;
#: at 4 rows the 324 extra programs a row make it slower than the separate launch (17.1 -> 21.5).
SCORE_ROWS = 2
SCORE_BN = 8


@triton.jit
def _rope_factors(positions, num_tokens, token, cos_sin, stride_cs, D: tl.constexpr, ROT: tl.constexpr,
                  S1: tl.constexpr, S2: tl.constexpr):  # fmt: skip
    """A D-wide head's rotation factors at the row's positions (interleaved mRoPE: pair j
    reads coordinate 1 when j % 3 == 1 and j < 3 * S1, coordinate 2 when j % 3 == 2 and
    j < 3 * S2, else 0): cos, and sin signed for the NeoX halves; identity past ROT."""
    HALF: tl.constexpr = ROT // 2
    d = tl.arange(0, D)
    rot = d < ROT
    j = d % HALF
    p0 = tl.load(positions + token)
    p1 = tl.load(positions + num_tokens + token)
    p2 = tl.load(positions + 2 * num_tokens + token)
    pos = tl.where((j % 3 == 1) & (j < 3 * S1), p1, tl.where((j % 3 == 2) & (j < 3 * S2), p2, p0))
    row = cos_sin + pos.to(tl.int64) * stride_cs
    c = tl.load(row + j, mask=rot, other=1.0)
    s = tl.load(row + HALF + j, mask=rot, other=0.0)
    s = tl.where(d < HALF, -s, s)
    return c, s


@triton.jit
def _norm_weights(
    heads, weight, weight_k, D: tl.constexpr, ROT: tl.constexpr, WEIGHT_SPLIT: tl.constexpr
):
    """`1 + w` for a tile's elements and for their rotation partners; heads at or past
    WEIGHT_SPLIT (when > 0) take the second weight (the k heads)."""
    HALF: tl.constexpr = ROT // 2
    d = tl.arange(0, D)
    rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, tl.where(rot, d - HALF, d))
    w = tl.load(weight + d).to(tl.float32)
    wp = tl.load(weight + partner).to(tl.float32)
    if WEIGHT_SPLIT > 0:
        is_k = (heads >= WEIGHT_SPLIT)[:, None]
        wk = tl.load(weight_k + d).to(tl.float32)
        wkp = tl.load(weight_k + partner).to(tl.float32)
        w2 = tl.where(is_k, wk[None, :], w[None, :]) + 1.0
        wp2 = tl.where(is_k, wkp[None, :], wp[None, :]) + 1.0
    else:
        w2 = w[None, :] + 1.0
        wp2 = wp[None, :] + 1.0
    return w2, wp2


@triton.jit
def _norm_rope_tile(src, heads, head_mask, w2, wp2, c, s, eps, D: tl.constexpr, ROT: tl.constexpr,
                    out_dtype: tl.constexpr):  # fmt: skip
    """`qwen4_exp_qk_norm_rope`'s tile with its static factors given: rms over each head,
    `* (1 + w)`, rotate; fp32 throughout, one rounding to `out_dtype`."""
    HALF: tl.constexpr = ROT // 2
    d = tl.arange(0, D)
    rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, tl.where(rot, d - HALF, d))
    hm = head_mask[:, None]
    x = tl.load(src + heads[:, None] * D + d[None, :], mask=hm, other=0.0).to(tl.float32)
    xp = tl.load(src + heads[:, None] * D + partner[None, :], mask=hm & rot[None, :], other=0.0).to(
        tl.float32
    )
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = x * rstd[:, None] * w2
    yp = xp * rstd[:, None] * wp2
    return (y * c[None, :] + yp * s[None, :]).to(out_dtype)


@triton.jit
def _norm_rope_apply(src, dst, heads, head_mask, w2, wp2, c, s, eps, D: tl.constexpr, ROT: tl.constexpr,
                     dst_head_stride):  # fmt: skip
    """`_norm_rope_tile` stored at `dst`. Returns the stored (rounded) tile."""
    out = _norm_rope_tile(src, heads, head_mask, w2, wp2, c, s, eps, D, ROT, dst.dtype.element_ty)
    d = tl.arange(0, D)
    tl.store(dst + heads[:, None] * dst_head_stride + d[None, :], out, mask=head_mask[:, None])
    return out


@triton.jit
def _quantize_tile(x, global_scale, ROWS: tl.constexpr, D: tl.constexpr):
    """`qsa_paged_kv_store_nvfp4`'s `_quantize_row` over a `[ROWS, D]` register tile:
    (packed e2m1 bytes `[ROWS, D/2]`, e4m3 scale bytes `[ROWS, D/16]`), the same operations
    on the same values."""
    lo, hi = tl.split(tl.reshape(x.to(tl.float32), [ROWS, D // 2, 2]))
    pair_max = tl.maximum(tl.abs(lo), tl.abs(hi))
    block_max = tl.max(tl.reshape(pair_max, [ROWS, D // 16, 8]), axis=2)
    sf_value = global_scale * (block_max * _rcp_approx(tl.full([ROWS, D // 16], 6.0, tl.float32)))
    sf_code = sf_value.to(tl.float8e4nv)
    sf_rounded = sf_code.to(tl.float32)
    out_scale = tl.where(
        block_max != 0.0,
        _rcp_approx(sf_rounded * _rcp_approx(tl.zeros_like(sf_rounded) + global_scale)),
        0.0,
    )
    lo_scaled = tl.reshape(
        tl.reshape(lo, [ROWS, D // 16, 8]) * out_scale[:, :, None], [ROWS, D // 2]
    )
    hi_scaled = tl.reshape(
        tl.reshape(hi, [ROWS, D // 16, 8]) * out_scale[:, :, None], [ROWS, D // 2]
    )
    packed = (_e2m1x2(lo_scaled, hi_scaled) & 0xFF).to(tl.uint8)
    return packed, sf_code.to(tl.uint8, bitcast=True)


@triton.jit
def _compress_factors(pos_t, pos_h, pos_w, cos_sin, cos_sin_stride, ROTARY_PAIRS: tl.constexpr,
                      IS_MROPE: tl.constexpr, MROPE_H: tl.constexpr, MROPE_W: tl.constexpr, live):  # fmt: skip
    """The compressed key's rotation factors (QSA's `_qsa_gemma_norm_rope` reads them at
    the group's first-token coordinates)."""
    pairs = tl.arange(0, ROTARY_PAIRS)
    if IS_MROPE:
        height_pair = (pairs % 3 == 1) & (pairs < 3 * MROPE_H)
        width_pair = (pairs % 3 == 2) & (pairs < 3 * MROPE_W)
        position = tl.where(height_pair, pos_h, tl.where(width_pair, pos_w, pos_t))
    else:
        position = pos_t
    cosine = tl.load(cos_sin + position * cos_sin_stride + pairs, mask=live, other=0.0)
    sine = tl.load(cos_sin + position * cos_sin_stride + ROTARY_PAIRS + pairs, mask=live, other=0.0)
    return cosine, sine


@triton.jit
def _compress_norm_rope(
    x, weight, cosine, sine, eps, HEAD_DIM: tl.constexpr, ROTARY_PAIRS: tl.constexpr
):
    """QSA's `_qsa_gemma_norm_rope` for one row with its static factors given (`weight` is
    `1 + w`; the fp32 table makes its intermediate rounding to the table's dtype a no-op)."""
    x_f32 = x.to(tl.float32)
    reciprocal_rms = tl.rsqrt(tl.sum(x_f32 * x_f32, axis=1) / HEAD_DIM + eps)
    normalized = x_f32 * reciprocal_rms[:, None] * weight[None, :]
    rotated, passthrough = tl.split(
        tl.permute(tl.reshape(normalized, (1, 2, HEAD_DIM // 2)), (0, 2, 1))
    )
    first, second = tl.split(tl.permute(tl.reshape(rotated, (1, 2, ROTARY_PAIRS)), (0, 2, 1)))
    output_first = first * cosine[None, :] - second * sine[None, :]
    output_second = second * cosine[None, :] + first * sine[None, :]
    rotated = tl.reshape(
        tl.permute(tl.join(output_first, output_second), (0, 2, 1)), (1, HEAD_DIM // 2)
    )
    return tl.reshape(
        tl.permute(tl.join(rotated, passthrough), (0, 2, 1)),
        (1, HEAD_DIM),
    ).to(tl.bfloat16)


@triton.jit(do_not_specialize=["num_tokens", "num_requests", "table_width", "num_pages"])
def _attn_decode_prologue_kernel(
    proj,
    stride_proj,
    index_q,
    stride_iq_token,
    stride_iq_head,
    positions,
    num_tokens,
    cos_sin,
    stride_cs,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    eps,
    k_cache,
    v_cache,
    s_kc_page,
    s_kc_head,
    s_kc_tok,
    s_vc_page,
    s_vc_head,
    s_vc_tok,
    request_indices,
    logical_positions,
    block_table,
    s_bt_req,
    s_bt_col,
    num_requests,
    table_width,
    num_pages,
    index_cache,
    s_ic_page,
    s_ic_slot,
    position_cache,
    s_pc_page,
    s_pc_slot,
    s_pc_axis,
    coords,
    s_c_row,
    s_c_axis,
    k_scale_cache,
    v_scale_cache,
    s_ks_page,
    s_ks_head,
    s_vs_page,
    s_vs_head,
    kv_scale_orig_quant,
    scores,
    s_sc_row,
    score_scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    IH: tl.constexpr,
    ID: tl.constexpr,
    IQ_COL: tl.constexpr,
    ROT: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    HPB: tl.constexpr,
    MAIN_PROGRAMS: tl.constexpr,
    TPB: tl.constexpr,
    CR: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    NVFP4: tl.constexpr,
    IN_LAUNCH_GROUPS: tl.constexpr,
    SCORE_TILES: tl.constexpr,
    SCORE_BLOCKS: tl.constexpr,
    SBN: tl.constexpr,
    SBH: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    program = tl.program_id(1)
    if USE_PDL:
        # The consumer (the index scores, or with the scores here the top-k) reads
        # nothing of this launch before its own wait.
        tl.extra.cuda.gdc_launch_dependents()
    row = proj + token * stride_proj

    # The row's page and slot (the K/V pool, the index plane and the coordinates share
    # the table); an unmapped row writes no cache. Host-filled: read before the wait.
    request = tl.load(request_indices + token).to(tl.int64)
    logical = tl.load(logical_positions + token).to(tl.int64)
    column = logical // TPB
    slot = logical % TPB
    live = (request >= 0) & (request < num_requests) & (logical >= 0) & (column < table_width)
    page = tl.load(block_table + request * s_bt_req + column * s_bt_col, mask=live, other=-1).to(
        tl.int64
    )
    live = live & (page >= 0) & (page < num_pages)

    if program < MAIN_PROGRAMS:
        main_heads = program * HPB + tl.arange(0, HPB)
        c, s = _rope_factors(positions, num_tokens, token, cos_sin, stride_cs, D, ROT, S1, S2)
        w2, wp2 = _norm_weights(main_heads, q_weight, k_weight, D, ROT, HQ)
        if NVFP4:
            k_global = tl.load(kv_scale_orig_quant + 1)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        out = _norm_rope_apply(
            row, row, main_heads, main_heads < HQ + HK, w2, wp2, c, s, eps, D, ROT, D
        )
        if program * HPB + HPB > HQ:
            is_k = (main_heads >= HQ) & (main_heads < HQ + HK) & live
            k_heads = (main_heads - HQ).to(tl.int64)[:, None]
            if NVFP4:
                # The rounded rows the composition's store reads back, quantized here.
                k_data, k_sf = _quantize_tile(out, k_global, HPB, D)
                pairs = tl.arange(0, D // 2)
                blocks = tl.arange(0, D // 16)
                tl.store(
                    k_cache + page * s_kc_page + k_heads * s_kc_head + slot * s_kc_tok
                    + pairs[None, :],
                    k_data,
                    mask=is_k[:, None],
                )  # fmt: skip
                # K scales token-major within the head page.
                tl.store(
                    k_scale_cache + page * s_ks_page + k_heads * s_ks_head
                    + slot * (D // 16) + blocks[None, :],
                    k_sf,
                    mask=is_k[:, None],
                )  # fmt: skip
            else:
                d = tl.arange(0, D)
                tl.store(
                    k_cache + page * s_kc_page + k_heads * s_kc_head + slot * s_kc_tok
                    + d[None, :],
                    out.to(k_cache.dtype.element_ty),
                    mask=is_k[:, None],
                )  # fmt: skip
    elif program == MAIN_PROGRAMS:
        index_heads = tl.arange(0, IH)
        c, s = _rope_factors(positions, num_tokens, token, cos_sin, stride_cs, ID, ROT, IS1, IS2)
        w2, wp2 = _norm_weights(index_heads, index_q_weight, index_q_weight, ID, ROT, 0)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        _norm_rope_apply(
            row + IQ_COL, index_q + token * stride_iq_token, index_heads, index_heads < IH,
            w2, wp2, c, s, eps, ID, ROT, stride_iq_head,
        )  # fmt: skip
    elif program >= MAIN_PROGRAMS + 2 + HK:
        if SCORE_TILES > 0:
            # `qsa_paged_index_scores`' decode tile (its prefetching kernel at up to four
            # rows: SBN blocks, the heads padded to SBH, two warps) with the row's index
            # query rotated here as the index program does, and at a group's last token the
            # last visible block's key compressed here as the index-key program does.
            tile = program - (MAIN_PROGRAMS + 2 + HK)
            visible_blocks = (logical + 1) // CR
            if tile * SBN >= visible_blocks:
                if USE_PDL:
                    tl.extra.cuda.gdc_wait()
                return
            dims = tl.arange(0, ID)
            columns = tile * SBN + tl.arange(0, SBN)
            valid = (columns < SCORE_BLOCKS) & (columns < visible_blocks)
            anchors = columns * CR + CR - 1
            key_pages = tl.load(
                block_table + request * s_bt_req + (anchors // TPB) * s_bt_col, mask=valid, other=0
            ).to(tl.int64)
            key_ptrs = (
                index_cache
                + key_pages[:, None] * s_ic_page
                + (anchors % TPB)[:, None] * s_ic_slot
                + dims[None, :]
            )
            key_mask = valid[:, None] & (dims[None, :] < ID)
            keys = tl.load(key_ptrs, mask=key_mask, other=0.0, eviction_policy="evict_first")
            index_heads = tl.arange(0, IH)
            c, s = _rope_factors(
                positions, num_tokens, token, cos_sin, stride_cs, ID, ROT, IS1, IS2
            )
            w2, wp2 = _norm_weights(index_heads, index_q_weight, index_q_weight, ID, ROT, 0)
            # The block this launch compresses (the last visible one, at a group's last
            # token): the group's earlier raw keys, the first token's coordinates, their
            # rotation and the key norm weight, as the index-key program reads them.
            fresh_tile = ((logical + 1) % CR == 0) & ((visible_blocks - 1) // SBN == tile)
            fresh = (columns == visible_blocks - 1) & fresh_tile
            compress_here = fresh_tile & live
            offsets = tl.arange(0, CR)
            members = logical - (CR - 1) + offsets
            member_pages = tl.load(
                block_table + request * s_bt_req + (tl.maximum(members, 0) // TPB) * s_bt_col,
                mask=compress_here,
                other=0,
            ).to(tl.int64)
            group = tl.load(
                index_cache
                + member_pages[:, None] * s_ic_page
                + (tl.maximum(members, 0) % TPB)[:, None] * s_ic_slot
                + dims[None, :],
                mask=(compress_here & (offsets < CR - 1))[:, None],
                other=0.0,
            )
            first = tl.maximum(logical - (CR - 1), 0)
            first_page = tl.load(
                block_table + request * s_bt_req + (first // TPB) * s_bt_col,
                mask=compress_here,
                other=0,
            ).to(tl.int64)
            first_base = position_cache + first_page * s_pc_page + (first % TPB) * s_pc_slot
            pos_t = tl.load(first_base, mask=compress_here, other=0)
            pos_h = tl.load(first_base + s_pc_axis, mask=compress_here, other=0)
            pos_w = tl.load(first_base + 2 * s_pc_axis, mask=compress_here, other=0)
            cosine, sine = _compress_factors(
                pos_t,
                pos_h,
                pos_w,
                cos_sin,
                stride_cs,
                ROTARY_PAIRS,
                IS_MROPE,
                MROPE_H,
                MROPE_W,
                compress_here,
            )
            key_weight = tl.load(index_k_weight + dims).to(tl.float32) + 1.0
            if USE_PDL:
                tl.extra.cuda.gdc_wait()
            query = _norm_rope_tile(
                row + IQ_COL, index_heads, index_heads < IH, w2, wp2, c, s, eps, ID, ROT,
                index_q.dtype.element_ty,
            )  # fmt: skip
            if fresh_tile:
                if compress_here:
                    current_k = tl.load(row + IQ_COL + IH * ID + dims)
                    group = tl.where(
                        offsets[:, None] == CR - 1,
                        current_k.to(index_cache.dtype.element_ty)[None, :],
                        group,
                    )
                    pooled = (tl.sum(group.to(tl.float32), axis=0) / CR).to(tl.bfloat16)
                    fresh_key = tl.reshape(
                        _compress_norm_rope(
                            pooled[None, :], key_weight, cosine, sine, eps, ID, ROTARY_PAIRS
                        ),
                        (ID,),
                    )
                    keys = tl.where(fresh[:, None], fresh_key[None, :].to(keys.dtype), keys)
                else:
                    # No page: nothing is written; the key is the plane's.
                    keys = tl.where(
                        fresh[:, None],
                        tl.load(key_ptrs, mask=key_mask & fresh[:, None], other=0.0),
                        keys,
                    )
            # The query as `[ID, SBH]`, heads past IH zero (the scores kernel's masked load).
            padded = tl.reshape(tl.broadcast_to(query[None, :, :], (SBH // IH, IH, ID)), (SBH, ID))
            head_rows = tl.arange(0, SBH)
            padded = tl.where((head_rows < IH)[:, None], padded, tl.zeros_like(padded))
            per_head = tl.dot(keys, tl.trans(padded), out_dtype=tl.float32)
            block_scores = tl.sum(
                tl.where(head_rows[None, :] < IH, tl.maximum(per_head, 0.0), 0.0), axis=1
            )
            tl.store(scores + token * s_sc_row + columns, block_scores * score_scale, mask=valid)
    elif program >= MAIN_PROGRAMS + 2:
        # One v head a program, as the K/V append stores it.
        h = program - MAIN_PROGRAMS - 2
        v_row = row + (HQ + HK + h) * D
        if NVFP4:
            v_global = tl.load(kv_scale_orig_quant + 2)
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        if live:
            if NVFP4:
                v_data, v_sf = _quantize_row(v_row, 1, v_global, D)
                pairs = tl.arange(0, D // 2)
                blocks = tl.arange(0, D // 16)
                tl.store(
                    v_cache + page * s_vc_page + h * s_vc_head + slot * s_vc_tok + pairs, v_data
                )
                # V scales in four-token groups within the head page.
                tl.store(
                    v_scale_cache + page * s_vs_page + h * s_vs_head
                    + (slot // 4) * (4 * (D // 16)) + blocks * 4 + slot % 4,
                    v_sf,
                )  # fmt: skip
            else:
                d = tl.arange(0, D)
                tl.store(
                    v_cache + page * s_vc_page + h * s_vc_head + slot * s_vc_tok + d,
                    tl.load(v_row + d).to(v_cache.dtype.element_ty),
                )
    else:
        # Before the wait: the coordinates; at a group's last token, the group's earlier
        # raw keys (written by earlier steps), the first token's coordinates, their
        # rotation and the key norm weight.
        axes = tl.arange(0, 4)
        coordinate = tl.load(coords + token * s_c_row + axes * s_c_axis, mask=axes < 3, other=0)
        dims = tl.arange(0, ID)
        group_end = live & ((logical + 1) % CR == 0)
        offsets = tl.arange(0, CR)
        members = logical - (CR - 1) + offsets
        member_pages = tl.load(
            block_table + request * s_bt_req + (tl.maximum(members, 0) // TPB) * s_bt_col,
            mask=group_end,
            other=0,
        ).to(tl.int64)
        earlier = group_end & (offsets < CR - 1)
        if IN_LAUNCH_GROUPS:
            # A request's rows come consecutive at consecutive positions: the earlier
            # member at row `token - (CR - 1) + o`, when that row is this request's at
            # that position, is written by this launch -- its raw key is taken from its
            # projection row after the wait, not from the index plane, and as the first
            # member its coordinates from its row of `coords`.
            member_rows = token - (CR - 1) + offsets
            maybe = earlier & (member_rows >= 0)
            member_requests = tl.load(request_indices + member_rows, mask=maybe, other=-1)
            member_logical = tl.load(logical_positions + member_rows, mask=maybe, other=-1)
            in_launch = (
                maybe
                & (member_requests.to(tl.int64) == request)
                & (member_logical.to(tl.int64) == members)
            )
        else:
            in_launch = offsets < 0
        group = tl.load(
            index_cache
            + member_pages[:, None] * s_ic_page
            + (tl.maximum(members, 0) % TPB)[:, None] * s_ic_slot
            + dims[None, :],
            mask=(earlier & ~in_launch)[:, None],
            other=0.0,
        )
        first = tl.maximum(logical - (CR - 1), 0)
        first_page = tl.load(
            block_table + request * s_bt_req + (first // TPB) * s_bt_col, mask=group_end, other=0
        ).to(tl.int64)
        first_base = position_cache + first_page * s_pc_page + (first % TPB) * s_pc_slot
        if IN_LAUNCH_GROUPS:
            # A first member written by this launch has its coordinates in its row.
            first_in_launch = tl.sum(tl.where(offsets == 0, in_launch, False).to(tl.int32)) > 0
            cached = group_end & ~first_in_launch
            first_row = coords + (token - (CR - 1)) * s_c_row
            pos_t = tl.load(first_row, mask=first_in_launch, other=0)
            pos_h = tl.load(first_row + s_c_axis, mask=first_in_launch, other=0)
            pos_w = tl.load(first_row + 2 * s_c_axis, mask=first_in_launch, other=0)
            pos_t = tl.where(cached, tl.load(first_base, mask=cached, other=0), pos_t)
            pos_h = tl.where(cached, tl.load(first_base + s_pc_axis, mask=cached, other=0), pos_h)
            pos_w = tl.where(
                cached, tl.load(first_base + 2 * s_pc_axis, mask=cached, other=0), pos_w
            )
        else:
            pos_t = tl.load(first_base, mask=group_end, other=0)
            pos_h = tl.load(first_base + s_pc_axis, mask=group_end, other=0)
            pos_w = tl.load(first_base + 2 * s_pc_axis, mask=group_end, other=0)
        cosine, sine = _compress_factors(
            pos_t,
            pos_h,
            pos_w,
            cos_sin,
            stride_cs,
            ROTARY_PAIRS,
            IS_MROPE,
            MROPE_H,
            MROPE_W,
            group_end,
        )
        key_weight = tl.load(index_k_weight + dims).to(tl.float32) + 1.0
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        if live:
            # The coordinates, as the index store writes them.
            tl.store(
                position_cache + page * s_pc_page + slot * s_pc_slot + axes * s_pc_axis,
                coordinate.to(position_cache.dtype.element_ty),
                mask=axes < 3,
            )
            # The raw index key -- or, at a group's last token, the group compressed in
            # its place (the compression kernel overwrites the same slot).
            current_k = tl.load(row + IQ_COL + IH * ID + dims)
            stored_k = current_k.to(index_cache.dtype.element_ty)
            if (logical + 1) % CR == 0:
                if IN_LAUNCH_GROUPS:
                    launch_k = tl.load(
                        proj
                        + member_rows[:, None] * stride_proj
                        + IQ_COL
                        + IH * ID
                        + dims[None, :],
                        mask=in_launch[:, None],
                        other=0.0,
                    )
                    group = tl.where(
                        in_launch[:, None], launch_k.to(index_cache.dtype.element_ty), group
                    )
                group = tl.where(offsets[:, None] == CR - 1, stored_k[None, :], group)
                pooled = (tl.sum(group.to(tl.float32), axis=0) / CR).to(tl.bfloat16)
                stored_k = tl.reshape(
                    _compress_norm_rope(
                        pooled[None, :], key_weight, cosine, sine, eps, ID, ROTARY_PAIRS
                    ),
                    (ID,),
                )
            tl.store(index_cache + page * s_ic_page + slot * s_ic_slot + dims, stored_k)


class Qwen4ExpAttnDecodePrologue(OpWrapper):
    """`qwen4_exp_qk_norm_rope`, `qsa_paged_kv_store`, `qsa_paged_index_store` and
    `qsa_prefill_compress` of one sparse decode attention layer, in one launch.

    `proj` `[T, W]` is the layer's fused input projection: q heads `[0, Hq)`, k heads
    `[Hq, Hq + Hk)` and v heads `[Hq + Hk, Hq + 2 Hk)` of `head_dim`, the index query
    heads at `index_q_col` and the raw index key right after them. Writes, as the four
    entries do: q and k normalized and rotated in place on `proj`; the index query into
    `index_q`; k and v into the paged K/V pool and the raw index key (or, at a group's
    last token, the compressed group) and the coordinates into the index plane and the
    position cache, at the row's page and slot. Returns `(proj, index_q)`.

    Over an NVFP4 pool (`k_scale_cache`, `v_scale_cache` and `kv_scale_orig_quant` given,
    `k_cache` / `v_cache` its uint8 e2m1 planes) the K/V append is
    `qsa_paged_kv_store_nvfp4`'s: e2m1 bytes and e4m3 block scales, K's token-major and
    V's in four-token groups, at the second-level scales `kv_scale_orig_quant[1]` / `[2]`.
    `rows_per_request` > 1 declares a verify step's layout: each request's rows
    consecutive, at consecutive positions.

    `index_scores` (fp32 `[T, table width * tpb / compress ratio]`, up to `SCORE_ROWS`
    decode rows) takes `qsa_paged_index_scores`' output over the index plane and query
    this launch leaves (`only_visible_blocks`: each row's visible columns written, the
    rest untouched): more programs score `SCORE_BN` blocks each, rotating the row's index
    query themselves and, at a group's last token, compressing the last visible block's
    key themselves, as the programs writing them do.

    Fusion boundary: as the four entries (with `index_scores`, and the index scores); the
    projection before and the selection after are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, 1 row at position 8699 (a group's last token)", spec=dict(pos=(8699,))),
        Cell(
            why="decode, 4 rows: three group ends (two on a page's last slot), one with no compression",
            spec=dict(pos=(8699, 8700, 4095, 31)),
        ),
        Cell(
            why="decode, 2 rows, one unmapped (request out of range)",
            spec=dict(pos=(8699, 1201), unmapped=1),
        ),
        Cell(
            why="NVFP4 pool, 1 row at position 8699 (a group's last token), K/V the role "
            "planes of the engine's [pages, 2, Hkv, tpb, D/2] pool",
            spec=dict(pos=(8699,), nvfp4=True),
        ),
        Cell(
            why="NVFP4 pool, 4 rows: three group ends (two on a page's last slot), one with "
            "no compression, with zero, -0.0 and outlier blocks in k and v",
            spec=dict(pos=(8699, 8700, 4095, 31), nvfp4=True, edge=True),
        ),
        Cell(
            why="NVFP4 pool, 2 rows, one unmapped (request out of range)",
            spec=dict(pos=(8699, 1201), unmapped=1, nvfp4=True),
        ),
        Cell(
            why="NVFP4 pool, 40 rows (two heads a program): positions across groups and pages",
            spec=dict(pos=tuple(2000 + 151 * r for r in range(40)), nvfp4=True),
        ),
        Cell(
            why="NVFP4 pool, 72 rows (eight heads a program, k heads beside q heads)",
            spec=dict(pos=tuple(1000 + 113 * r for r in range(72)), nvfp4=True),
        ),
        Cell(
            why="NVFP4 pool, 1 row at a group's last token, the index scores in the launch",
            spec=dict(pos=(8699,), nvfp4=True, scores=True),
        ),
        Cell(
            why="NVFP4 pool, 2 rows (a group end on a page's last slot, one row without), the "
            "index scores in the launch",
            spec=dict(pos=(4095, 8700), nvfp4=True, scores=True, edge=True),
        ),
        Cell(
            why="bf16 pool, 2 rows (a group end, and one at 8 tokens), the index scores in the "
            "launch",
            spec=dict(pos=(6143, 7), scores=True),
        ),
        Cell(
            why="verify, NVFP4 pool, 3 requests x 4 rows: group ends at a request's third "
            "row (two earlier members in the launch), fourth (three) and first (none)",
            spec=dict(pos=(8697, 8696, 8699), rows_per_request=4, nvfp4=True, edge=True),
        ),
        Cell(
            why="verify, bf16 pool, 2 requests x 4 rows, one group end crossing a page",
            spec=dict(pos=(4093, 8698), rows_per_request=4),
        ),
        Cell(
            why="verify, NVFP4 pool, 32 requests x 4 rows (eight heads a program)",
            spec=dict(pos=tuple(3000 + 211 * r for r in range(32)), rows_per_request=4, nvfp4=True),
        ),
    )

    note = """
    Metadata consumed: the QSA request indices, logical positions and block table; the
    position ids as `[3, T]` (the rope) and as `[T, 3]` (the coordinates stored).
    Preconditions: those of the four entries; the index plane one KV head of
    `index_head_dim` (bf16), the position cache int32 `[pages, tpb, >=3]`; generation
    rows: one per request (`rows_per_request` 1: a group's earlier tokens are already in
    the index plane), or a verify step's `rows_per_request` consecutive rows a request at
    consecutive positions (a group's earlier tokens in the index plane or in the
    request's preceding rows). A row whose page lookup fails writes no cache, as the
    stores and compression skip it.
    Over an NVFP4 pool: the preconditions of `qsa_paged_kv_store_nvfp4` (uint8
    `[pages, Hkv, tpb, D/2]` data and `[pages, Hkv, tpb, D/16]` scale planes, each scale
    head page contiguous, a contiguous fp32 `[3]` `kv_scale_orig_quant`); the page
    bound is the data plane's page count, as there.

    Numerics: the same arithmetic as the four entries; outputs bitwise theirs (the
    NVFP4 cells compare every byte of the four K/V planes).

    Compiled variants: one per row-count band of the norm + rope launch plan (up to 16,
    up to 64, more rows), pool kind (bf16 / NVFP4) and step kind (one row a request /
    verify); the row count, table height and width and the page count are not
    specialized on.
    """

    def __call__(
        self,
        proj: torch.Tensor,
        index_q: torch.Tensor,
        position_ids: torch.Tensor,
        coords: torch.Tensor,
        cos_sin: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        index_k_weight: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        compress_ratio: int,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        index_heads: int,
        index_head_dim: int,
        index_q_col: int,
        rotary_dim: int,
        eps: float,
        mrope_section1: int,
        mrope_section2: int,
        index_mrope_section1: int = 0,
        index_mrope_section2: int = 0,
        compress_mrope_section: Optional[Tuple[int, int, int]] = None,
        k_scale_cache: Optional[torch.Tensor] = None,
        v_scale_cache: Optional[torch.Tensor] = None,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        rows_per_request: int = 1,
        index_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = proj.shape[0]
        if num_tokens == 0:
            return proj, index_q
        hpb, warps = _qk_launch(num_tokens)
        main_programs = triton.cdiv(num_heads_q + num_heads_k, hpb)
        section = compress_mrope_section or (0, 0, 0)
        nvfp4 = k_scale_cache is not None
        if not nvfp4:
            # Placeholders: the bf16 kernel does not read them.
            k_scale_cache = v_scale_cache = kv_scale_orig_quant = k_cache
        score_tiles = score_blocks = 0
        if index_scores is not None:
            assert num_tokens <= SCORE_ROWS and rows_per_request == 1 and warps == 2, (
                "the index scores ride decode launches of up to SCORE_ROWS rows"
            )
            score_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
            score_tiles = triton.cdiv(score_blocks, SCORE_BN)
        pdl = get_env_enable_pdl()
        _attn_decode_prologue_kernel[(num_tokens, main_programs + 2 + num_heads_k + score_tiles)](
            proj,
            proj.stride(0),
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            position_ids,
            num_tokens,
            cos_sin,
            cos_sin.stride(0),
            q_weight,
            k_weight,
            index_q_weight,
            index_k_weight,
            eps,
            k_cache,
            v_cache,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            request_indices,
            logical_positions,
            block_table,
            block_table.stride(0),
            block_table.stride(1),
            block_table.shape[0],
            block_table.shape[1],
            k_cache.shape[0],
            index_cache,
            index_cache.stride(0),
            index_cache.stride(1),
            position_cache,
            position_cache.stride(0),
            position_cache.stride(1),
            position_cache.stride(2),
            coords,
            coords.stride(0),
            coords.stride(1),
            k_scale_cache,
            v_scale_cache,
            k_scale_cache.stride(0),
            k_scale_cache.stride(1),
            v_scale_cache.stride(0),
            v_scale_cache.stride(1),
            kv_scale_orig_quant,
            proj if index_scores is None else index_scores,
            0 if index_scores is None else index_scores.stride(0),
            index_head_dim**-0.5,
            HQ=num_heads_q,
            HK=num_heads_k,
            D=head_dim,
            IH=index_heads,
            ID=index_head_dim,
            IQ_COL=index_q_col,
            ROT=rotary_dim,
            S1=mrope_section1,
            S2=mrope_section2,
            IS1=index_mrope_section1,
            IS2=index_mrope_section2,
            HPB=hpb,
            MAIN_PROGRAMS=main_programs,
            TPB=tokens_per_block,
            CR=compress_ratio,
            ROTARY_PAIRS=cos_sin.shape[1] // 2,
            IS_MROPE=compress_mrope_section is not None,
            MROPE_H=section[1],
            MROPE_W=section[2],
            NVFP4=nvfp4,
            IN_LAUNCH_GROUPS=rows_per_request > 1,
            SCORE_TILES=score_tiles,
            SCORE_BLOCKS=score_blocks,
            SBN=SCORE_BN,
            SBH=max(16, triton.next_power_of_2(index_heads)),
            USE_PDL=pdl,
            num_warps=warps,
            launch_pdl=pdl,
        )
        return proj, index_q

    def reference(
        self,
        proj: torch.Tensor,
        index_q: torch.Tensor,
        position_ids: torch.Tensor,
        coords: torch.Tensor,
        cos_sin: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        index_k_weight: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        compress_ratio: int,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        index_heads: int,
        index_head_dim: int,
        index_q_col: int,
        rotary_dim: int,
        eps: float,
        mrope_section1: int,
        mrope_section2: int,
        index_mrope_section1: int = 0,
        index_mrope_section2: int = 0,
        compress_mrope_section: Optional[Tuple[int, int, int]] = None,
        k_scale_cache: Optional[torch.Tensor] = None,
        v_scale_cache: Optional[torch.Tensor] = None,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        rows_per_request: int = 1,
        index_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The four entries in their composition order, on the given buffers; with
        `index_scores`, then `qsa_paged_index_scores`' visible columns written into it."""
        rows = proj.shape[0]
        _qk_norm_rope(
            proj, index_q, position_ids, cos_sin, q_weight, k_weight, index_q_weight,
            num_heads_q=num_heads_q, num_heads_k=num_heads_k, head_dim=head_dim,
            index_heads=index_heads, index_head_dim=index_head_dim, index_q_col=index_q_col,
            rotary_dim=rotary_dim, eps=eps, mrope_section1=mrope_section1,
            mrope_section2=mrope_section2, index_mrope_section1=index_mrope_section1,
            index_mrope_section2=index_mrope_section2,
        )  # fmt: skip
        heads = proj.view(rows, -1, head_dim)
        k = heads[:, num_heads_q : num_heads_q + num_heads_k]
        v = heads[:, num_heads_q + num_heads_k : num_heads_q + 2 * num_heads_k]
        if k_scale_cache is None:
            _kv_store(
                k,
                v,
                k_cache,
                v_cache,
                request_indices,
                logical_positions,
                block_table,
                tokens_per_block,
            )
        else:
            _kv_store_nvfp4(
                k,
                v,
                k_cache,
                v_cache,
                k_scale_cache,
                v_scale_cache,
                request_indices,
                logical_positions,
                block_table,
                tokens_per_block,
                kv_scale_orig_quant,
            )
        token_k = proj[:, index_q_col + index_heads * index_head_dim :][:, :index_head_dim].reshape(
            rows, 1, index_head_dim
        )
        _index_store(
            token_k, coords, index_cache, position_cache, request_indices, logical_positions,
            block_table, tokens_per_block,
        )  # fmt: skip
        _compress(
            logical_positions=logical_positions, request_indices=request_indices,
            block_table=block_table, index_cache=index_cache, position_cache=position_cache,
            k_norm_weight=index_k_weight, cos_sin=cos_sin, eps=eps,
            tokens_per_block=tokens_per_block, compress_ratio=compress_ratio,
            mrope_section=compress_mrope_section,
        )  # fmt: skip
        if index_scores is not None:
            scores = _index_scores(
                q=index_q, index_cache=index_cache, block_table=block_table,
                query_positions=logical_positions, request_indices=request_indices,
                tokens_per_block=tokens_per_block, compress_ratio=compress_ratio,
                only_visible_blocks=True, context_rows=False,
            )  # fmt: skip
            for r in range(rows):
                visible = (int(logical_positions[r]) + 1) // compress_ratio
                index_scores[r, :visible] = scores[r, :visible]
        return proj, index_q

    def is_valid(self, proj, index_q, position_ids, coords, cos_sin, *args, **kwargs) -> None:
        assert proj.stride(1) == 1 and proj.dtype == torch.bfloat16, "bf16 proj, unit column stride"
        assert position_ids.is_contiguous() and position_ids.dtype == torch.int32, (
            "int32 [3, T] positions"
        )
        assert coords.shape == (proj.shape[0], 3), "coords [T, 3]"
        assert cos_sin.dtype == torch.float32, (
            "an fp32 cos|sin table (the compressed key's rotation)"
        )
        k_scale_cache = kwargs.get("k_scale_cache")
        v_scale_cache = kwargs.get("v_scale_cache")
        global_scales = kwargs.get("kv_scale_orig_quant")
        nvfp4 = (k_scale_cache, v_scale_cache, global_scales)
        assert all(t is None for t in nvfp4) or all(t is not None for t in nvfp4), (
            "an NVFP4 pool takes both scale planes and kv_scale_orig_quant"
        )
        scores = kwargs.get("index_scores")
        if scores is not None:
            assert scores.dtype == torch.float32 and scores.stride(1) == 1, "fp32 index scores"
            assert proj.shape[0] <= SCORE_ROWS and kwargs.get("rows_per_request", 1) == 1, (
                "the index scores ride decode launches of up to SCORE_ROWS rows"
            )
        if k_scale_cache is not None:
            for scales in (k_scale_cache, v_scale_cache):
                assert scales.dtype == torch.uint8 and scales.stride(3) == 1, "uint8 scale planes"
                assert scales.stride(2) == scales.shape[3], "each scale head page contiguous"
            assert (
                global_scales.dtype == torch.float32
                and global_scales.shape == (3,)
                and global_scales.is_contiguous()
            ), "a contiguous fp32 [3] kv_scale_orig_quant"


qwen4_exp_attn_decode_prologue = Qwen4ExpAttnDecodePrologue()
