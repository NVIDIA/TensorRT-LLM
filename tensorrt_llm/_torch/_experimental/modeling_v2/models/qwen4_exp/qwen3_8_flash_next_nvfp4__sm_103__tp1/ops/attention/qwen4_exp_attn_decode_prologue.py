# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sparse decode attention layer's work between its input projection and its index
scores, one PDL launch.

The composition it replaces ran four launches over two streams: the QK norm + rope of
both planes (`qwen4_exp_qk_norm_rope`), the paged K/V append (`qsa_paged_kv_store`), and on
the side stream the index plane's raw-key and coordinate store (`qsa_paged_index_store`)
and the group compression (`qsa_prefill_compress`), then a cross-stream join. Here the
programs of `qwen4_exp_qk_norm_rope` run as they do there; the ones holding k heads also
write their rotated rows into the K pool; one more program per row stores v, the index
key and the coordinates, and -- at a group's last token -- pools the group with this
token's key taken from the projection row (the cached copy is the one being written),
normalizes, rotates and stores it at the anchor slot, as the compression kernel does.

With PDL every operand that is static or host-filled -- the page and slot, the rotation
factors at the row's positions, the norm weights, and at a group end the group's earlier
keys from the index plane with their rotation -- is read before `griddepcontrol.wait`;
after it a program reads only the projection row.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from .qsa_paged_index_store import qsa_paged_index_store as _index_store
from .qsa_paged_kv_store import qsa_paged_kv_store as _kv_store
from .qsa_prefill_compress import qsa_prefill_compress as _compress
from .qwen4_exp_qk_norm_rope import _launch as _qk_launch
from .qwen4_exp_qk_norm_rope import qwen4_exp_qk_norm_rope as _qk_norm_rope


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
    """`qwen4_exp_qk_norm_rope`'s tile with its static factors given: rms over each head,
    `* (1 + w)`, rotate; fp32 throughout, one rounding at the store. Returns the stored
    (rounded) tile."""
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
    USE_PDL: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    program = tl.program_id(1)
    if USE_PDL:
        # The index scores read nothing of this launch before their own wait.
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
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        out = _norm_rope_apply(
            row, row, main_heads, main_heads < HQ + HK, w2, wp2, c, s, eps, D, ROT, D
        )
        if program * HPB + HPB > HQ:
            d = tl.arange(0, D)
            is_k = (main_heads >= HQ) & (main_heads < HQ + HK) & live
            tl.store(
                k_cache
                + page * s_kc_page
                + (main_heads - HQ).to(tl.int64)[:, None] * s_kc_head
                + slot * s_kc_tok
                + d[None, :],
                out.to(k_cache.dtype.element_ty),
                mask=is_k[:, None],
            )
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
        group = tl.load(
            index_cache
            + member_pages[:, None] * s_ic_page
            + (tl.maximum(members, 0) % TPB)[:, None] * s_ic_slot
            + dims[None, :],
            mask=group_end & (offsets < CR - 1)[:, None],
            other=0.0,
        )
        first = tl.maximum(logical - (CR - 1), 0)
        first_page = tl.load(
            block_table + request * s_bt_req + (first // TPB) * s_bt_col, mask=group_end, other=0
        ).to(tl.int64)
        first_base = position_cache + first_page * s_pc_page + (first % TPB) * s_pc_slot
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
            # v, as the K/V append stores it.
            d = tl.arange(0, D)
            for h in tl.static_range(HK):
                v = tl.load(row + (HQ + HK + h) * D + d)
                tl.store(
                    v_cache + page * s_vc_page + h * s_vc_head + slot * s_vc_tok + d,
                    v.to(v_cache.dtype.element_ty),
                )
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

    Fusion boundary: as the four entries; the projection before and the index scores
    after are the caller's.
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
    )

    note = """
    Metadata consumed: the QSA request indices, logical positions and block table; the
    position ids as `[3, T]` (the rope) and as `[T, 3]` (the coordinates stored).
    Preconditions: those of the four entries; the index plane one KV head of
    `index_head_dim` (bf16), the position cache int32 `[pages, tpb, >=3]`; decode rows,
    one per request (a group's earlier tokens are already in the index plane). A row
    whose page lookup fails writes no cache, as the stores and compression skip it.

    Numerics: the same arithmetic as the four entries; outputs bitwise theirs.
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = proj.shape[0]
        if num_tokens == 0:
            return proj, index_q
        hpb, warps = _qk_launch(num_tokens)
        main_programs = triton.cdiv(num_heads_q + num_heads_k, hpb)
        section = compress_mrope_section or (0, 0, 0)
        pdl = get_env_enable_pdl()
        _attn_decode_prologue_kernel[(num_tokens, main_programs + 2)](
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The four entries in their composition order, on the given buffers."""
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


qwen4_exp_attn_decode_prologue = Qwen4ExpAttnDecodePrologue()
