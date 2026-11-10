# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sparse decode attention layer's work between its input projection and its index
scores, one PDL launch, at one rank's heads over a bf16 or e4m3 K/V pool.

This target's copy of the NVFP4 default target's entry of the same name, plus an optional row
map for the index query. It fuses both planes' `fused_qk_norm_rope`, `qsa_paged_index_store`,
`qsa_prefill_compress`, the packed-QKV copy and `qsa_paged_kv_store`. Per row: `HPB`-head main
programs (q, k, K pool), one index-query program, one index-key / coordinate / compression
program, one program per v head.

With PDL, every operand not written since the sublayer's hyper-connection up-gate-mean (page and
slot, rotation factors, norm weights, a group's earlier keys from the index plane) is read
before `griddepcontrol.wait`; after it a program reads only projection rows. The row mapping
and positions are written ahead of that up-gate-mean, which never releases its dependents early.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber


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
def _norm_rope_apply(src, dst, heads, head_mask, w2, wp2, c, s, eps, D: tl.constexpr, ROT: tl.constexpr,
                     dst_head_stride):  # fmt: skip
    """rms over each head, `* (1 + w)`, rotate; fp32 throughout, one rounding to `dst`'s
    dtype, stored at `dst`. Returns the stored (rounded) tile."""
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
    out = (y * c[None, :] + yp * s[None, :]).to(dst.dtype.element_ty)
    tl.store(dst + heads[:, None] * dst_head_stride + d[None, :], out, mask=hm)
    return out


@triton.jit
def _compress_norm_rope(x, weight, cosine, sine, eps, HEAD_DIM: tl.constexpr, ROTARY_PAIRS: tl.constexpr):  # fmt: skip
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
    out,
    stride_out,
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
    index_q_rows,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    IH: tl.constexpr,
    ID: tl.constexpr,
    IQ_COL: tl.constexpr,
    ROT: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    HPB: tl.constexpr,
    MAIN_PROGRAMS: tl.constexpr,
    TPB: tl.constexpr,
    CR: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    IN_LAUNCH_GROUPS: tl.constexpr,
    IQ_ROW_MAP: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    program = tl.program_id(1)
    if USE_PDL:
        # The consumer (the index scores) reads nothing of this launch before its own wait.
        tl.extra.cuda.gdc_launch_dependents()
    row = proj + token * stride_proj
    out_row = out + token * stride_out

    # The row's page and slot (shared by the K/V pool, index plane and coordinates); an unmapped
    # row writes no cache. Written before the up-gate-mean, so read before the wait.
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
        if USE_PDL:
            gdc_wait_with_memory_clobber()
        rotated = _norm_rope_apply(
            row, out_row, main_heads, main_heads < HQ + HK, w2, wp2, c, s, eps, D, ROT, D
        )
        if program * HPB + HPB > HQ:
            # The rounded rows the composition's store read back from the packed buffer.
            is_k = (main_heads >= HQ) & (main_heads < HQ + HK) & live
            k_heads = (main_heads - HQ).to(tl.int64)[:, None]
            d = tl.arange(0, D)
            tl.store(
                k_cache + page * s_kc_page + k_heads * s_kc_head + slot * s_kc_tok + d[None, :],
                rotated.to(k_cache.dtype.element_ty),
                mask=is_k[:, None],
            )
    elif program == MAIN_PROGRAMS:
        index_heads = tl.arange(0, IH)
        c, s = _rope_factors(positions, num_tokens, token, cos_sin, stride_cs, ID, ROT, 0, 0)
        w2, wp2 = _norm_weights(index_heads, index_q_weight, index_q_weight, ID, ROT, 0)
        if USE_PDL:
            gdc_wait_with_memory_clobber()
        iq_token = token
        if IQ_ROW_MAP:
            # The row's slot in the split index selection's permuted order.
            iq_token = tl.load(index_q_rows + token).to(tl.int64)
        _norm_rope_apply(
            row + IQ_COL, index_q + iq_token * stride_iq_token, index_heads, index_heads < IH,
            w2, wp2, c, s, eps, ID, ROT, stride_iq_head,
        )  # fmt: skip
    elif program >= MAIN_PROGRAMS + 2:
        # One v head a program: the V pool, as the K/V append stores it, and the packed buffer.
        h = program - MAIN_PROGRAMS - 2
        d = tl.arange(0, D)
        if USE_PDL:
            gdc_wait_with_memory_clobber()
        v = tl.load(row + (HQ + HK + h) * D + d)
        tl.store(out_row + (HQ + HK + h) * D + d, v)
        if live:
            tl.store(
                v_cache + page * s_vc_page + h * s_vc_head + slot * s_vc_tok + d,
                v.to(v_cache.dtype.element_ty),
            )
    else:
        # Before the wait: the coordinates and, at a group's last token, the group's earlier raw
        # keys, the first token's coordinate 0, its rotation and the key norm weight.
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
            # A request's rows are consecutive at consecutive positions: an earlier member at
            # row `token - (CR - 1) + o` of the same request is written by this launch, so its
            # raw key (and, as the first member, its coordinates) come from its own row.
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
            pos_t = tl.load(coords + (token - (CR - 1)) * s_c_row, mask=first_in_launch, other=0)
            pos_t = tl.where(cached, tl.load(first_base, mask=cached, other=0), pos_t)
        else:
            pos_t = tl.load(first_base, mask=group_end, other=0)
        # The compressed key rotates on the first token's coordinate 0 (QSA's plain rope).
        pairs = tl.arange(0, ROTARY_PAIRS)
        cs_row = cos_sin + pos_t.to(tl.int64) * stride_cs
        cosine = tl.load(cs_row + pairs, mask=group_end, other=0.0)
        sine = tl.load(cs_row + ROTARY_PAIRS + pairs, mask=group_end, other=0.0)
        key_weight = tl.load(index_k_weight + dims).to(tl.float32) + 1.0
        if USE_PDL:
            gdc_wait_with_memory_clobber()
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


def _launch(num_tokens: int) -> Tuple[int, int]:
    """Main-plane heads per program and warps per program, by row count."""
    if num_tokens <= 16:
        return 1, 2
    if num_tokens <= 64:
        return 2, 2
    return 8, 8


#: Rank geometries: tp4 / tep4 6 q heads and 1 KV head; tp2 / tep2 12 and 1; dep4 / dep2 / tp1
#: 24 and 2 (heads of 256, a 4 x 128 index plane).
_TP4 = dict(geom="tp4")
_TP2 = dict(geom="tp2")
_DEP4 = dict(geom="dep4")


class Qwen4ExpAttnDecodePrologue(OpWrapper):
    """The main and index planes' Gemma norm + rope, the packed-QKV copy, the paged K/V append,
    the index key and coordinate store and the group compression of one sparse decode
    attention layer, in one launch. Returns `(out, index_q)`.

    `proj` `[T, W]` (read only): q heads `[0, Hq)`, k heads `[Hq, Hq + Hk)` and v heads
    `[Hq + Hk, Hq + 2 Hk)` of `head_dim`, the index query heads at `index_q_col` and the raw
    index key right after them. Writes `out` `[T, (Hq + 2 Hk) * head_dim]` (q and k normalized
    and rotated, v copied); `index_q` `[T, index_heads, index_head_dim]`; the rotated k and raw
    v into `k_cache` / `v_cache` `[pages, Hk, tpb, head_dim]` (bf16 or e4m3); the raw index key
    (or, at a group's last token, the compressed group) and the coordinates into the index
    plane and the position cache.

    The rotation is NeoX over the first `rotary_dim` dims, factors from `cos_sin` (fp32
    `[P, rotary_dim]`, cos block then sin block). The main plane uses interleaved mRoPE over
    `position_ids` `[3, T]` (pair `j` reads coordinate 1 when `j % 3 == 1 and j < 3 *
    mrope_section1`, 2 when `j % 3 == 2 and j < 3 * mrope_section2`, else 0); the index plane
    and the compressed key (`qsa_prefill_compress`'s, at the group's first token) use
    coordinate 0. `rows_per_request` > 1 declares a verify step: each request's rows
    consecutive, at consecutive positions. With `index_q_rows` (`[T]` int), row t's index query
    goes to `index_q[index_q_rows[t]]`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tep4 rank, verify 2 requests x 4 rows: a group end on a page's last slot "
            "with one member in the plane, one with all three in the launch; zero, -0.0 and "
            "outlier blocks in k and v",
            spec=dict(_TP4, pos=(4093, 8696), rows_per_request=4, edge=True),
        ),
        Cell(
            why="tep4 rank, MTP-7 verify 2 requests x 8 rows, one up to position 262143 "
            "(the table's last row)",
            spec=dict(_TP4, pos=(1001, 262136), rows_per_request=8),
        ),
        Cell(
            why="tp4 rank, decode 3 rows: a group end, a mid-group row, an unmapped row",
            spec=dict(_TP4, pos=(8699, 1201, 31), unmapped=2),
        ),
        Cell(
            why="tp4 rank, decode 72 rows (eight-head programs: the one main program masks "
            "its eighth head)",
            spec=dict(_TP4, pos=tuple(1000 + 113 * r for r in range(72))),
        ),
        Cell(
            why="tep4 rank, engine-laid pool: K/V the role planes of layer slot 11 of a "
            "24-role page, the index plane strided by 12 layers' pages, rows on pages past "
            "2^31 elements of the pool",
            spec=dict(_TP4, pos=(900, 1203), rows_per_request=4, engine=True, large=True),
        ),
        Cell(
            why="tep2 rank, 32 requests x 4 MTP-3 verify rows (a generation server's 128-row "
            "graph), e4m3 pool",
            spec=dict(
                _TP2, pos=tuple(3000 + 6007 * r for r in range(32)), rows_per_request=4, fp8=True
            ),
        ),
        Cell(
            why="dep4 rank, 128 requests x 4 MTP-3 verify rows (ht_dep4's largest graph "
            "batch, eight-head programs), e4m3 pool",
            spec=dict(
                _DEP4,
                pos=tuple(3000 + 1531 * r for r in range(128)),
                rows_per_request=4,
                fp8=True,
            ),  # fmt: skip
        ),
        Cell(
            why="dep2 / tp1 rank, decode 32 requests over the core's bf16 pool",
            spec=dict(_DEP4, pos=tuple(2000 + 151 * r for r in range(32))),
        ),
    )

    note = """
    Source: the Triton kernel above, grid `(T, cdiv(Hq + Hk, HPB) + 2 + Hk)`; PDL when the
    engine enables it. Pages, rows and request slots index in int64 (pools past 2^31 elements).

    Preconditions: the index plane one KV head of `index_head_dim` (bf16); the position cache
    int32 `[pages, tpb, >=3]`; `compress_ratio` divides `tokens_per_block`; one row per request,
    or with `rows_per_request` > 1 each request's rows consecutive at consecutive positions. A
    row whose page lookup fails writes no cache; its packed row and index query are still written.

    Numerics: fp32, one rounding at each store; v, raw keys and coordinates bitwise. The rotation
    reads the fp32 cos/sin table, not `fused_qk_norm_rope`'s `__sincosf`.

    Not certified: NVFP4 pools (the forward keeps the composition there); non-zero index-plane
    mRoPE sections; mRoPE on the compressed key; `T == 0` (returns at once); head geometries other
    than 6/1, 12/1 and 24/2 x 256 with a 4 x 128 index plane.
    """

    def __call__(
        self,
        proj: torch.Tensor,
        out: torch.Tensor,
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
        rows_per_request: int = 1,
        index_q_rows: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = proj.shape[0]
        if num_tokens == 0:
            return out, index_q
        hpb, warps = _launch(num_tokens)
        main_programs = triton.cdiv(num_heads_q + num_heads_k, hpb)
        pdl = get_env_enable_pdl()
        _attn_decode_prologue_kernel[(num_tokens, main_programs + 2 + num_heads_k)](
            proj,
            proj.stride(0),
            out,
            out.stride(0),
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
            proj if index_q_rows is None else index_q_rows,
            HQ=num_heads_q,
            HK=num_heads_k,
            D=head_dim,
            IH=index_heads,
            ID=index_head_dim,
            IQ_COL=index_q_col,
            ROT=rotary_dim,
            S1=mrope_section1,
            S2=mrope_section2,
            HPB=hpb,
            MAIN_PROGRAMS=main_programs,
            TPB=tokens_per_block,
            CR=compress_ratio,
            ROTARY_PAIRS=cos_sin.shape[1] // 2,
            IN_LAUNCH_GROUPS=rows_per_request > 1,
            IQ_ROW_MAP=index_q_rows is not None,
            USE_PDL=pdl,
            num_warps=warps,
            launch_pdl=pdl,
        )
        return out, index_q

    def reference(
        self,
        proj: torch.Tensor,
        out: torch.Tensor,
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
        rows_per_request: int = 1,
        index_q_rows: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plain torch, fp32, writing the given buffers: both planes' norm + rope, then every row's
        stores, then the compression of every group a row ends (reading the plane those left)."""
        rows, dev = proj.shape[0], proj.device
        half = rotary_dim // 2
        pos = position_ids.long()
        table = cos_sin.float()
        j = torch.arange(half, device=dev)

        def factors(s1: int, s2: int):
            coord = torch.zeros(half, dtype=torch.long, device=dev)
            coord[(j % 3 == 1) & (j < 3 * s1)] = 1
            coord[(j % 3 == 2) & (j < 3 * s2)] = 2
            p = pos[coord, :].t()  # [T, half]
            return table[p, j][:, None, :], table[p, half + j][:, None, :]

        def plane(x: torch.Tensor, w: torch.Tensor, cos, sin) -> torch.Tensor:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)
            x1, x2 = x[..., :half], x[..., half:rotary_dim]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., rotary_dim:]], -1)

        main_heads = num_heads_q + num_heads_k
        width = main_heads * head_dim
        v_width = num_heads_k * head_dim
        weights = torch.cat(
            [
                q_weight.float().expand(num_heads_q, head_dim),
                k_weight.float().expand(num_heads_k, head_dim),
            ]
        )
        main = plane(
            proj[:, :width].float().reshape(rows, main_heads, head_dim),
            weights,
            *factors(mrope_section1, mrope_section2),
        ).to(out.dtype)
        out[:, :width] = main.reshape(rows, width)
        v = proj[:, width : width + v_width]
        out[:, width : width + v_width] = v
        iq = proj[:, index_q_col : index_q_col + index_heads * index_head_dim].float()
        iq = plane(
            iq.reshape(rows, index_heads, index_head_dim), index_q_weight.float(), *factors(0, 0)
        )
        iq = iq.to(index_q.dtype)
        if index_q_rows is None:
            index_q.copy_(iq)
        else:
            index_q[index_q_rows.long()] = iq
        key_col = index_q_col + index_heads * index_head_dim
        raw = proj[:, key_col : key_col + index_head_dim]

        requests = request_indices.long().tolist()
        logical = logical_positions.long().tolist()
        num_requests, table_width = block_table.shape

        def page_of(request: int, position: int):
            column = position // tokens_per_block
            if not (0 <= request < num_requests and position >= 0 and column < table_width):
                return None
            page = int(block_table[request, column])
            return page if 0 <= page < k_cache.shape[0] else None

        k_rows = main[:, num_heads_q:]
        v_rows = v.reshape(rows, num_heads_k, head_dim)
        slots = [page_of(r, p) for r, p in zip(requests, logical)]
        for t, page in enumerate(slots):
            if page is None:
                continue
            slot = logical[t] % tokens_per_block
            k_cache[page, :, slot] = k_rows[t].to(k_cache.dtype)
            v_cache[page, :, slot] = v_rows[t].to(v_cache.dtype)
            index_cache[page, slot, 0] = raw[t].to(index_cache.dtype)
            position_cache[page, slot, :3] = coords[t].to(position_cache.dtype)

        key_weight = 1.0 + index_k_weight.float()
        rot = 2 * (cos_sin.shape[1] // 2)
        for t, page in enumerate(slots):
            if page is None or (logical[t] + 1) % compress_ratio:
                continue
            members = [logical[t] - compress_ratio + 1 + o for o in range(compress_ratio)]
            keys = torch.stack(
                [
                    index_cache[page_of(requests[t], m), m % tokens_per_block, 0].float()
                    for m in members
                ]
            )
            pooled = (keys.sum(0) / compress_ratio).to(torch.bfloat16).float()
            first_page = page_of(requests[t], members[0])
            first = int(position_cache[first_page, members[0] % tokens_per_block, 0])
            x = pooled * torch.rsqrt(pooled.pow(2).mean() + eps) * key_weight
            cos, sin = table[first, : rot // 2], table[first, rot // 2 : rot]
            x1, x2 = x[: rot // 2], x[rot // 2 : rot]
            key = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[rot:]])
            index_cache[page, logical[t] % tokens_per_block, 0] = key.to(index_cache.dtype)
        return out, index_q

    def is_valid(self, proj, out, index_q, position_ids, coords, cos_sin, *args, **kwargs) -> None:
        assert proj.stride(1) == 1 and proj.dtype == torch.bfloat16, "bf16 proj, unit column stride"
        assert out.stride(1) == 1 and out.dtype == torch.bfloat16, "bf16 out, unit column stride"
        assert index_q.stride(2) == 1, "index_q must have a unit dim stride"
        assert position_ids.is_contiguous() and position_ids.shape == (3, proj.shape[0]), (
            "contiguous [3, T] positions (the kernel reads coordinate a of row t at a * T + t)"
        )
        assert position_ids.dtype == torch.int32, "int32 positions"
        assert coords.shape == (proj.shape[0], 3), "coords [T, 3]"
        assert cos_sin.dtype == torch.float32 and cos_sin.stride(1) == 1, (
            "an fp32 cos|sin table with a unit column stride"
        )


qwen4_exp_attn_decode_prologue = Qwen4ExpAttnDecodePrologue()
