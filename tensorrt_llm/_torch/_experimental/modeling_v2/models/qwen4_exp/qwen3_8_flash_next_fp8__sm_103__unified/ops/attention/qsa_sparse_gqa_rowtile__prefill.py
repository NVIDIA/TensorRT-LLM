# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA for context batches: several query rows per program, K/V read once per tile.

A plan launch lists, per group of consecutive rows of one request, the union of the rows'
resident selected tokens as cache slots and re-indexes each row's selection onto it; a persistent
row-tile pass walks each list in 64-slot tiles, loading each tile's K/V once for every row and head
of the group. Groups whose rows span requests, or whose union costs more than the rows' own
columns, go to a lone-row pass over each row's own columns. A group is sized to whole 8-row
swizzle atoms (10 rows of 12 heads, 20 of 6).
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa as _per_row_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (
    BLOCK_M,
    BLOCK_N,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (
    eligible as _cute_eligible,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (
    num_ctas as _rowtile_ctas,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (
    rowtile_attention_cute as _rowtile_attention_cute,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

_INTEGER_DTYPES = (torch.int32, torch.int64)
# The per-row kernel's scan tile: the prefill bound is rounded up to it, as that kernel does.
_BOUND_TILE = 64
# Programs of the lone-row pass per KV head; each walks the lone list with this stride.
_LONE_PROGRAMS = 256  # capacity: fixed: a launch width; the lone list may be any length
# A one-request group takes the row-tile pass when its list's 64-slot tiles, each costing this
# many quarters of a per-row tile (measured: about half), cost no more than its rows' own
# columns; at 2 every one-request group qualifies, as a union never holds more tiles.
_SHARED_TILE_COST = 2
#: The O staging's swizzle atom in MMA rows: a group's rows x heads come in whole atoms.
_ROW_ATOM = 8
_LONE_WARPS = 2
_LONE_STAGES = 2


@dsl_user_op
def _red_or_shared(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip), cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
        "{ red.shared.or.b64 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# --------------------------------------------------------------------------------------
# The token plan (CuTe DSL): one CTA a group ORs its rows' readable selected tokens into a
# shared-memory bitmap over the table's tokens, keeping each row's highest token. A one-request
# group masks each word to resident pages, takes list positions as a block-wide prefix of the
# words' popcounts and decides shared vs lone (`_SHARED_TILE_COST`); a shared group writes each
# listed token's cache slot and each row's re-indexed bits (rank = prefix + popcount below it).
# The list is the union's resident tokens in token order.
# --------------------------------------------------------------------------------------

_PLAN_THREADS = 512
#: CTAs an SM the plan's registers are held to, so three groups' latency chains overlap.
_PLAN_MIN_BLOCKS = 3
#: Shared scratch words of the plan CTA: rows' highest tokens at 0, rows' columns read at
#: `_MISC_COLS`, the lone list base at `_MISC_LONE`, the scan's warp totals at `_MISC_SCAN`.
_MISC_COLS = 32
_MISC_LONE = 63
_MISC_SCAN = 64


@dsl_user_op
def _cu_popc64(x, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(x).ir_value(loc=loc, ip=ip)],
            "popc.b64 $0, $1;",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cu_ctz64(x, *, loc=None, ip=None):
    """Trailing zeros of a nonzero 64-bit word."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .b64 t; brev.b64 t, $1; clz.b64 $0, t; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_shared(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "{ red.shared.max.s32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b64(addr, *, loc=None, ip=None):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b64 $0, [$1];",
            "=l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_b64(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip), cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
        "{ st.shared.b64 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_b32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "{ st.shared.b32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _GroupPlanKernel:
    """The token plan pass, one CTA a group (see the section comment)."""

    def __init__(self, topk, block_topk, compress_ratio, tokens_per_block, table_width,
                 num_words, rows_per_tile, packed_words, req64, qp64):  # fmt: skip
        self.topk, self.block_topk, self.cr = topk, block_topk, compress_ratio
        self.tpb, self.width, self.num_words = tokens_per_block, table_width, num_words
        self.rpt, self.pw = rows_per_tile, packed_words
        self.req64, self.qp64 = req64, qp64
        # shared memory: union words | word prefixes | rows' packed words | row scratch
        self.o_pref = num_words * 8
        self.o_pack = self.o_pref + ((num_words * 4 + 15) // 16) * 16
        self.o_misc = self.o_pack + rows_per_tile * packed_words * 8
        self.smem = self.o_misc + 96 * 4 + 64

    @cute.jit
    def __call__(
        self,
        sel_t: cute.Tensor,
        qp_ptr: cute.Pointer,
        req_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        info_ptr: cute.Pointer,
        slots_ptr: cute.Pointer,
        packed_ptr: cute.Pointer,
        plan_ptr: cute.Pointer,
        lone_ptr: cute.Pointer,
        lone_count_ptr: cute.Pointer,
        rows: cutlass.Int32,
        num_groups: cutlass.Int32,
        sel_stride: cutlass.Int32,
        bt_stride: cutlass.Int32,
        slots_stride: cutlass.Int32,
        num_requests: cutlass.Int32,
        num_cache_pages: cutlass.Int32,
        stream,
    ):
        self.kernel(
            sel_t.iterator.toint(),
            qp_ptr.toint(),
            req_ptr.toint(),
            bt_ptr.toint(),
            info_ptr.toint(),
            slots_ptr.toint(),
            packed_ptr.toint(),
            plan_ptr.toint(),
            lone_ptr.toint(),
            lone_count_ptr.toint(),
            rows,
            sel_stride,
            bt_stride,
            slots_stride,
            num_requests,
            num_cache_pages,
        ).launch(
            grid=[num_groups, 1, 1],
            block=[_PLAN_THREADS, 1, 1],
            smem=self.smem,
            stream=stream,
            min_blocks_per_mp=_PLAN_MIN_BLOCKS,
        )

    @cute.kernel
    def kernel(self, sel_base: cutlass.Int64, qp_base: cutlass.Int64, req_base: cutlass.Int64,
               bt_base: cutlass.Int64, info_base: cutlass.Int64, slots_base: cutlass.Int64,
               packed_base: cutlass.Int64, plan_base: cutlass.Int64, lone_base: cutlass.Int64,
               lone_count_base: cutlass.Int64, rows: cutlass.Int32, sel_stride: cutlass.Int32,
               bt_stride: cutlass.Int32, slots_stride: cutlass.Int32,
               num_requests: cutlass.Int32, num_cache_pages: cutlass.Int32):  # fmt: skip
        tidx, _, _ = cute.arch.thread_idx()
        group, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tidx // 32
        nthreads: cutlass.Constexpr = _PLAN_THREADS
        nwarps: cutlass.Constexpr = _PLAN_THREADS // 32
        rpt: cutlass.Constexpr = self.rpt
        nw: cutlass.Constexpr = self.num_words
        tpb: cutlass.Constexpr = self.tpb
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem,), stride=(1,)), byte_alignment=128
        )
        s_u = cutlass.Int32(buf.iterator.toint())
        s_pref = s_u + self.o_pref
        s_pack = s_u + self.o_pack
        s_misc = s_u + self.o_misc
        r0 = group * rpt
        for i in cutlass.range_constexpr((nw + nthreads - 1) // nthreads):
            w = tidx + i * nthreads
            if w < nw:
                _st_shared_b64(s_u + w * 8, cutlass.Int64(0))
        if tidx < rpt:
            _st_shared_b32(s_misc + tidx * 4, cutlass.Int32(-1))
        # each row's request and columns read (every thread, rpt rows)
        reqs = []
        cols = []
        own_tiles = cutlass.Int32(0)
        n_live = cutlass.Int32(0)
        lowest = cutlass.Int32(2147483647)
        highest_req = cutlass.Int32(-1)
        for t in cutlass.range_constexpr(rpt):
            row = r0 + t
            live = row < rows
            srow = cutlass.min(row, rows - 1)
            if const_expr(self.req64):
                rq = cutlass.Int32(_ldg_i64(req_base + cutlass.Int64(srow) * 8))
            else:
                rq = _ldg_i32(req_base + cutlass.Int64(srow) * 4)
            if const_expr(self.qp64):
                qp = cutlass.Int32(_ldg_i64(qp_base + cutlass.Int64(srow) * 8))
            else:
                qp = _ldg_i32(qp_base + cutlass.Int64(srow) * 4)
            visible = qp + 1
            sb = cutlass.min(visible // self.cr, cutlass.Int32(self.block_topk))
            valid = sb * self.cr + visible % self.cr
            bounded = cutlass.min(valid, cutlass.Int32(self.topk))
            cr_ = cutlass.min((bounded + _BOUND_TILE - 1) // _BOUND_TILE * _BOUND_TILE,
                              cutlass.Int32(self.topk))  # fmt: skip
            cr_ = cutlass.max(cr_, cutlass.Int32(0))
            cr_ = cutlass.Int32(cutlass.select_(live, cr_, cutlass.Int32(0)))
            reqs.append(rq)
            cols.append(cr_)
            if tidx == t:
                _st_shared_b32(s_misc + (_MISC_COLS + t) * 4, cr_)
            own_tiles = own_tiles + (cr_ + BLOCK_N - 1) // BLOCK_N
            n_live = n_live + cutlass.Int32(
                cutlass.select_(live, cutlass.Int32(1), cutlass.Int32(0))
            )
            lowest = cutlass.Int32(cutlass.select_(live, cutlass.min(lowest, rq), lowest))
            highest_req = cutlass.Int32(
                cutlass.select_(live, cutlass.max(highest_req, rq), highest_req)
            )
        one_request = (lowest == highest_req) & (lowest >= 0) & (lowest < num_requests)
        cute.arch.sync_threads()
        # walk 1: the rows' readable tokens into the union bitmap, each row's highest token.
        # Items are (row, four-column quad) pairs, item = i * nthreads + tidx, kept in registers
        # for walk 2.
        nq: cutlass.Constexpr = -(-self.topk // 4)
        n_items: cutlass.Constexpr = rpt * nq
        n_rounds: cutlass.Constexpr = -(-n_items // nthreads)
        toks = []
        for i in cutlass.range_constexpr(n_rounds):
            item = i * nthreads + tidx
            t = cutlass.min(item // nq, rpt - 1)
            q = item % nq
            ncol = _ld_shared_b32(s_misc + (_MISC_COLS + t) * 4)
            ncol = cutlass.Int32(cutlass.select_(item < n_items, ncol, cutlass.Int32(0)))
            row_sel = sel_base + cutlass.Int64(cutlass.min(r0 + t, rows - 1)) * sel_stride * 4
            for k in cutlass.range_constexpr(4):
                c = q * 4 + k
                tok = cutlass.Int32(-1)
                if c < ncol:
                    tok = _ldg_i32(row_sel + cutlass.Int64(c) * 4)
                ok = (tok >= 0) & (tok // tpb < self.width)
                toks.append(cutlass.Int32(cutlass.select_(ok, tok, cutlass.Int32(-1))))
        for i in cutlass.range_constexpr(n_rounds):
            item = i * nthreads + tidx
            t = cutlass.min(item // nq, rpt - 1)
            cur_w = cutlass.Int32(-1)
            cur_m = cutlass.Int64(0)
            hi_q = cutlass.Int32(-1)
            for k in cutlass.range_constexpr(4):
                tok = toks[i * 4 + k]
                if tok >= 0:
                    w = tok >> 6
                    if w != cur_w:
                        if cur_w >= 0:
                            _red_or_shared(s_u + cur_w * 8, cur_m)
                        cur_w = w
                        cur_m = cutlass.Int64(0)
                    cur_m = cur_m | (cutlass.Int64(1) << cutlass.Int64(tok & 63))
                    hi_q = cutlass.max(hi_q, tok)
            if cur_w >= 0:
                _red_or_shared(s_u + cur_w * 8, cur_m)
            # the row's highest token: a warp's 32 items span at most two rows (nq > 32)
            t_first = cute.arch.shuffle_sync(t, 0)
            same = t == t_first
            hi_a = cutlass.Int32(cutlass.select_(same, hi_q, cutlass.Int32(-1)))
            hi_b = cutlass.Int32(cutlass.select_(same, cutlass.Int32(-1), hi_q))
            for o in (1, 2, 4, 8, 16):
                hi_a = cutlass.max(hi_a, cute.arch.shuffle_sync_bfly(hi_a, offset=o))
                hi_b = cutlass.max(hi_b, cute.arch.shuffle_sync_bfly(hi_b, offset=o))
            if lane == 0:
                if hi_a >= 0:
                    _max_shared(s_misc + t_first * 4, hi_a)
            t_last = cute.arch.shuffle_sync(t, 31)
            if lane == 31:
                if (t_last != t_first) & (hi_b >= 0):
                    _max_shared(s_misc + t_last * 4, hi_b)
        cute.arch.sync_threads()
        # row_info, the group's highest token
        grp_hi = cutlass.Int32(-1)
        for t in cutlass.range_constexpr(rpt):
            h = _ld_shared_b32(s_misc + t * 4)
            grp_hi = cutlass.max(grp_hi, h)
            if tidx == t:
                if r0 + t < rows:
                    _stg_i32(info_base + cutlass.Int64(r0 + t) * 8, h)
                    _stg_i32(info_base + cutlass.Int64(r0 + t) * 8 + 4, cols[t])
        words_needed = cutlass.Int32(
            cutlass.select_(one_request, (grp_hi + 64) // 64, cutlass.Int32(0))
        )
        words_needed = cutlass.min(words_needed, cutlass.Int32(nw))
        # resident tokens: a word's bits on a page outside the table or the pool are dropped
        bt_row = bt_base + cutlass.Int64(cutlass.max(lowest, 0)) * bt_stride * 4
        cnts = []
        for i in cutlass.range_constexpr((nw + nthreads - 1) // nthreads):
            w = tidx + i * nthreads
            cnt = cutlass.Int32(0)
            if w < words_needed:
                u = _ld_shared_b64(s_u + w * 8)
                if u != 0:
                    first = w * 64
                    for k in cutlass.range_constexpr(_word_pages(tpb)):
                        lp = first // tpb + k
                        lo_b = cutlass.max(lp * tpb - first, 0)
                        hi_b = cutlass.min((lp + 1) * tpb - first, 64)
                        if lo_b < hi_b:
                            ok = lp < self.width
                            pg = cutlass.Int32(-1)
                            if ok:
                                pg = _ldg_i32(bt_row + cutlass.Int64(lp) * 4)
                            ok = ok & (pg >= 0) & (pg < num_cache_pages)
                            span = _bits_range64(lo_b, hi_b)
                            u = cutlass.Int64(cutlass.select_(ok, u, u & ~span))
                    _st_shared_b64(s_u + w * 8, u)
                    cnt = _cu_popc64(u)
            cnts.append(cnt)
        # list positions: block-wide exclusive prefix of the words' counts in word order; thread
        # tidx owns words tidx + i * nthreads, so each round is scanned and the rounds chained.
        base = cutlass.Int32(0)
        for i in cutlass.range_constexpr((nw + nthreads - 1) // nthreads):
            incl = cnts[i]
            for o in (1, 2, 4, 8, 16):
                up = cute.arch.shuffle_sync_up(incl, offset=o)
                if lane >= o:
                    incl = incl + up
            if lane == 31:
                _st_shared_b32(s_misc + (_MISC_SCAN + warp) * 4, incl)
            cute.arch.sync_threads()
            wv = cutlass.Int32(0)
            tot = cutlass.Int32(0)
            if lane < nwarps:
                wv = _ld_shared_b32(s_misc + (_MISC_SCAN + lane) * 4)
                tot = wv
            before = cute.arch.warp_reduction_sum(
                cutlass.Int32(cutlass.select_(lane < warp, wv, cutlass.Int32(0)))
            )
            total = cute.arch.warp_reduction_sum(tot)
            w = tidx + i * nthreads
            if w < nw:
                _st_shared_b32(s_pref + w * 4, base + before + incl - cnts[i])
            base = base + total
            cute.arch.sync_threads()
        listed = base
        tiles = (listed + BLOCK_N - 1) // BLOCK_N
        shared = one_request & (tiles * _SHARED_TILE_COST <= own_tiles * 4)
        shared_i = cutlass.Int32(cutlass.select_(shared, cutlass.Int32(1), cutlass.Int32(0)))
        if tidx == 0:
            _stg_i32(
                plan_base + cutlass.Int64(group) * 8,
                cutlass.Int32(cutlass.select_(shared, tiles, cutlass.Int32(-1))),
            )
            _stg_i32(plan_base + cutlass.Int64(group) * 8 + 4, listed)
            if shared_i == 0:
                lb = _atom_add_g32(lone_count_base, n_live)
                _st_shared_b32(s_misc + _MISC_LONE * 4, lb)
        if shared_i == 0:
            cute.arch.sync_threads()
            if tidx < rpt:
                if r0 + tidx < rows:
                    lb = _ld_shared_b32(s_misc + _MISC_LONE * 4)
                    _stg_i32(lone_base + cutlass.Int64(lb + tidx) * 4, r0 + tidx)
        if (shared_i != 0) & (listed > 0):
            pw_used = (listed + 63) // 64
            # the rows' packed words, zeroed
            for i in cutlass.range(tidx, rpt * pw_used, nthreads, unroll=1):
                _st_shared_b64(s_pack + i * 8, cutlass.Int64(0))
            # slots: one thread a word, its set bits in token order
            slot_row = slots_base + cutlass.Int64(group) * slots_stride * 4
            for i in cutlass.range_constexpr((nw + nthreads - 1) // nthreads):
                w = tidx + i * nthreads
                if w < words_needed:
                    u = _ld_shared_b64(s_u + w * 8)
                    j = _ld_shared_b32(s_pref + w * 4)
                    pg_lp = cutlass.Int32(-1)
                    pg = cutlass.Int32(0)
                    while u != 0:
                        b = _cu_ctz64(u)
                        tok = w * 64 + b
                        lp = tok // tpb
                        if lp != pg_lp:
                            pg = _ldg_i32(bt_row + cutlass.Int64(lp) * 4)
                            pg_lp = lp
                        _stg_i32(slot_row + cutlass.Int64(j) * 4, pg * tpb + tok % tpb)
                        j = j + 1
                        u = u & (u - 1)
            cute.arch.sync_threads()
            # walk 2 (walk 1's registers): each row's readable resident tokens -> its bit at the
            # token's list position; a quad's ranks follow by popcount from its word's prefix.
            for i in cutlass.range_constexpr(n_rounds):
                item = i * nthreads + tidx
                t = cutlass.min(item // nq, rpt - 1)
                tok0 = cutlass.max(
                    cutlass.max(toks[i * 4], toks[i * 4 + 1]),
                    cutlass.max(toks[i * 4 + 2], toks[i * 4 + 3]),
                )
                if tok0 >= 0:
                    w = tok0 >> 6
                    u = _ld_shared_b64(s_u + w * 8)
                    pre = _ld_shared_b32(s_pref + w * 4)
                    cur_w = cutlass.Int32(-1)
                    cur_m = cutlass.Int64(0)
                    for k in cutlass.range_constexpr(4):
                        tok = toks[i * 4 + k]
                        if tok >= 0:
                            uw = u
                            pw_pre = pre
                            if (tok >> 6) != w:
                                uw = _ld_shared_b64(s_u + (tok >> 6) * 8)
                                pw_pre = _ld_shared_b32(s_pref + (tok >> 6) * 4)
                            bit = cutlass.Int64(1) << cutlass.Int64(tok & 63)
                            if (uw & bit) != 0:
                                jj = pw_pre + _cu_popc64(uw & (bit - 1))
                                pw_i = jj >> 6
                                if pw_i != cur_w:
                                    if cur_w >= 0:
                                        _red_or_shared(s_pack + (t * pw_used + cur_w) * 8, cur_m)
                                    cur_w = pw_i
                                    cur_m = cutlass.Int64(0)
                                cur_m = cur_m | (cutlass.Int64(1) << cutlass.Int64(jj & 63))
                    if cur_w >= 0:
                        _red_or_shared(s_pack + (t * pw_used + cur_w) * 8, cur_m)
            cute.arch.sync_threads()
            for i in cutlass.range(tidx, rpt * pw_used, nthreads, unroll=1):
                t = i // pw_used
                k = i % pw_used
                if r0 + t < rows:
                    _stg_i64(packed_base + (cutlass.Int64(r0 + t) * self.pw + k) * 8,
                             _ld_shared_b64(s_pack + i * 8))  # fmt: skip


def _word_pages(tpb: int) -> int:
    """Pages an aligned 64-token word spans at most."""
    if 64 % tpb == 0:
        return 64 // tpb
    if tpb % 64 == 0:
        return 1
    return -(-64 // tpb) + 1


def _bits_range64(lo, hi):
    """int64 with bits lo .. hi - 1 set (0 <= lo < hi <= 64)."""
    ones = cutlass.Int64(-1)
    below_hi = cutlass.Int64(
        cutlass.select_(hi >= 64, ones, (cutlass.Int64(1) << cutlass.Int64(hi)) - 1)
    )
    below_lo = (cutlass.Int64(1) << cutlass.Int64(lo)) - 1
    return below_hi & ~below_lo


@dsl_user_op
def _ldg_i32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.b32 $0, [$1];",
            "=r,l", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _ldg_i64(addr, *, loc=None, ip=None):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.b64 $0, [$1];",
            "=l,l", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _stg_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "{ st.global.b32 [$1], $2; mov.u32 $0, 0; }", "=r,l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _stg_i64(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
        "{ st.global.b64 [$1], $2; mov.u32 $0, 0; }", "=r,l,l", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _atom_add_g32(addr, val, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


_GROUP_PLAN: dict = {}


def _group_plan(selected_tokens, query_positions, request_indices, block_table, compress_ratio,
                tokens_per_block, table_width, num_words, rows_per_tile, num_groups, packed_words,
                row_info, slots, packed_bits, group_plan, lone_rows, lone_count, num_cache_pages):  # fmt: skip
    """Launch the token plan pass (one CTA a group)."""
    rows, final_topk = selected_tokens.shape
    block_topk = (final_topk - compress_ratio + 1) // compress_ratio
    req64 = request_indices.dtype == torch.int64
    qp64 = query_positions.dtype == torch.int64
    query_positions = query_positions.contiguous()
    key = (final_topk, compress_ratio, tokens_per_block, table_width, rows_per_tile,
           packed_words, req64, qp64)  # fmt: skip
    gm = cute.AddressSpace.gmem
    ptrs = (
        query_positions.data_ptr(), request_indices.data_ptr(), block_table.data_ptr(),
        row_info.data_ptr(), slots.data_ptr(), packed_bits.data_ptr(), group_plan.data_ptr(),
        lone_rows.data_ptr(), lone_count.data_ptr(),
    )  # fmt: skip
    scalars = (rows, num_groups, selected_tokens.stride(0), block_table.stride(0), slots.stride(0),
               block_table.shape[0], num_cache_pages)  # fmt: skip
    fn = _GROUP_PLAN.get(key)
    if fn is None:
        i32, i64 = cutlass.Int32, cutlass.Int64
        fn = cute.compile(
            _GroupPlanKernel(
                final_topk,
                block_topk,
                compress_ratio,
                tokens_per_block,
                table_width,
                num_words,
                rows_per_tile,
                packed_words,
                req64,
                qp64,
            ),  # fmt: skip
            from_dlpack(selected_tokens, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            make_ptr(i64 if qp64 else i32, ptrs[0], gm, assumed_align=4),
            make_ptr(i64 if req64 else i32, ptrs[1], gm, assumed_align=4),
            make_ptr(i32, ptrs[2], gm, assumed_align=4),
            make_ptr(i32, ptrs[3], gm, assumed_align=8),
            make_ptr(i32, ptrs[4], gm, assumed_align=4),
            make_ptr(i64, ptrs[5], gm, assumed_align=8),
            make_ptr(i32, ptrs[6], gm, assumed_align=8),
            make_ptr(i32, ptrs[7], gm, assumed_align=4),
            make_ptr(i32, ptrs[8], gm, assumed_align=4),
            *(cutlass.Int32(x) for x in scalars),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _GROUP_PLAN[key] = fn
    fn(selected_tokens, *ptrs, *scalars)


# --------------------------------------------------------------------------------------
# The block plan (CuTe DSL): the token plan's outputs, bit for bit, from the selection's 4-token
# blocks. Every token a row selects below `T0 = 4 * ((p + 1) // 4)` (`p` the group's lowest
# position) lies in a complete block: an aligned column quad below `T0` sets one bit of a block
# bitmap; tokens at or past `T0` (at most `rows_per_tile + 3`) go to one 64-token local word. A
# group naming a token below `T0` outside such a quad goes to the lone pass. Its shared-memory ORs
# are 32-bit (`red.shared.or.b64` is a compare-and-swap loop on SM100).
# --------------------------------------------------------------------------------------

_BP_THREADS = 512
#: CTAs an SM the block plan's registers are held to.
_BP_MIN_BLOCKS = 4
#: Shared scratch words of the block plan CTA (int32 slots of its misc area).
_BP_HI = 0  # rows' highest tokens (rows_per_tile)
_BP_COLS = 32  # rows' columns read
_BP_SCAN = 64  # the scan's warp totals (16)
_BP_LONE = 80  # the lone list base
_BP_FLAG = 81  # a token below T0 outside an aligned block quad
_BP_NSEG = 82  # the group's segments (runs of one request's rows), 0 for a lone group
_BP_SPLIT = 83  # the second segment's first row
_BP_NLIVE = 84
_BP_OWN = 85  # the rows' own 64-column tiles
_BP_REQ0 = 86  # each segment's request
_BP_REQ1 = 87
_BP_T00 = 88  # each segment's T0
_BP_T01 = 89
_BP_LOCN = 90  # the segment's resident local tokens
_BP_LOC = 92  # the segment's local word (int64 at words 92-93)
_BP_MISC_WORDS = 96


def _span_pages(tokens: int, tpb: int, aligned: bool) -> int:
    """Pages a run of `tokens` tokens spans at most (`aligned`: it starts on a multiple of
    `tokens`)."""
    if aligned and tokens % tpb == 0:
        return tokens // tpb
    if aligned and tpb % tokens == 0:
        return 1
    return -(-tokens // tpb) + 1


@dsl_user_op
def _stg_v4_i32(addr, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)]
        + [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }", "=r,l,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _ldg_nc_i32(addr, *, loc=None, ip=None):
    """A global int32 load the compiler may schedule freely (no side effects)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.b32 $0, [$1];",
            "=r,l", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _word_page_ids(bt_row, lp0, tpb, width):
    """The physical pages of a block word's logical pages `lp0 ..` (clamped into the table; the
    caller drops pages past it), loaded together."""
    out = []
    for k in range(_span_pages(256, tpb, True)):  # trace-time unrolled
        lp = cutlass.min(lp0 + k, width - 1)
        out.append(_ldg_nc_i32(bt_row + cutlass.Int64(lp) * 4))
    return out


@dsl_user_op
def _red_or_shared32(addr, val, *, loc=None, ip=None):
    """A native shared-memory OR (`red.shared.or.b64` is a compare-and-swap loop on SM100)."""
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "{ red.shared.or.b32 [$1], $2; mov.u32 $0, 0; }", "=r,r,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


class _BlockPlanKernel:
    """The block plan pass, one CTA a group (see the section comment)."""

    def __init__(self, topk, block_topk, tokens_per_block, table_width, block_words,
                 rows_per_tile, packed_words, req64, qp64):  # fmt: skip
        self.topk, self.block_topk = topk, block_topk
        self.tpb, self.width, self.nbw = tokens_per_block, table_width, block_words
        self.rpt, self.pw = rows_per_tile, packed_words
        self.req64, self.qp64 = req64, qp64
        # shared memory: union block words | word prefixes | rows' packed words | misc
        self.o_pref = block_words * 8
        self.o_pack = self.o_pref + ((block_words * 4 + 15) // 16) * 16
        self.o_misc = self.o_pack + rows_per_tile * packed_words * 8
        self.smem = self.o_misc + _BP_MISC_WORDS * 4

    @cute.jit
    def __call__(
        self,
        sel_t: cute.Tensor,
        qp_ptr: cute.Pointer,
        req_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        info_ptr: cute.Pointer,
        slots_ptr: cute.Pointer,
        packed_ptr: cute.Pointer,
        plan_ptr: cute.Pointer,
        lone_ptr: cute.Pointer,
        lone_count_ptr: cute.Pointer,
        rows: cutlass.Int32,
        num_groups: cutlass.Int32,
        sel_stride: cutlass.Int32,
        bt_stride: cutlass.Int32,
        slots_stride: cutlass.Int32,
        num_requests: cutlass.Int32,
        num_cache_pages: cutlass.Int32,
        stream,
    ):
        self.kernel(
            sel_t.iterator.toint(), qp_ptr.toint(), req_ptr.toint(), bt_ptr.toint(),
            info_ptr.toint(), slots_ptr.toint(), packed_ptr.toint(), plan_ptr.toint(),
            lone_ptr.toint(), lone_count_ptr.toint(), rows, sel_stride, bt_stride, slots_stride,
            num_requests, num_cache_pages,
        ).launch(
            grid=[num_groups, 1, 1],
            block=[_BP_THREADS, 1, 1],
            smem=self.smem,
            stream=stream,
            min_blocks_per_mp=_BP_MIN_BLOCKS,
        )  # fmt: skip

    def _quad(self, sel_base, r0, rows, sel_stride, s_misc, item, ra, rb):
        """Walk item `item`'s four columns (a clamped column, masked after): each readable
        token, -1 elsewhere (also for an item past the rows or outside rows [ra, rb))."""
        nq = -(-self.topk // 4)
        t = cutlass.min(item // nq, self.rpt - 1)
        q = item % nq
        ncol = _ld_shared_b32(s_misc + (_BP_COLS + t) * 4)
        in_seg = (item < self.rpt * nq) & (t >= ra) & (t < rb)
        ncol = cutlass.Int32(cutlass.select_(in_seg, ncol, cutlass.Int32(0)))
        row_sel = sel_base + cutlass.Int64(cutlass.min(r0 + t, rows - 1)) * sel_stride * 4
        tk = []
        for k in range(4):  # trace-time unrolled
            c = q * 4 + k
            cc = cutlass.Int32(cutlass.select_(c < ncol, c, cutlass.Int32(0)))
            tok = _ldg_nc_i32(row_sel + cutlass.Int64(cc) * 4)
            ok = (c < ncol) & (tok >= 0) & (tok // self.tpb < self.width)
            tk.append(cutlass.Int32(cutlass.select_(ok, tok, cutlass.Int32(-1))))
        return tk

    @cute.kernel
    def kernel(self, sel_base: cutlass.Int64, qp_base: cutlass.Int64, req_base: cutlass.Int64,
               bt_base: cutlass.Int64, info_base: cutlass.Int64, slots_base: cutlass.Int64,
               packed_base: cutlass.Int64, plan_base: cutlass.Int64, lone_base: cutlass.Int64,
               lone_count_base: cutlass.Int64, rows: cutlass.Int32, sel_stride: cutlass.Int32,
               bt_stride: cutlass.Int32, slots_stride: cutlass.Int32,
               num_requests: cutlass.Int32, num_cache_pages: cutlass.Int32):  # fmt: skip
        tidx, _, _ = cute.arch.thread_idx()
        group, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tidx // 32
        nthreads: cutlass.Constexpr = _BP_THREADS
        nwarps: cutlass.Constexpr = _BP_THREADS // 32
        rpt: cutlass.Constexpr = self.rpt
        nbw: cutlass.Constexpr = self.nbw
        tpb: cutlass.Constexpr = self.tpb
        pw: cutlass.Constexpr = self.pw
        big = cutlass.Int32(2147483647)
        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem,), stride=(1,)), byte_alignment=128
        )
        s_u = cutlass.Int32(buf.iterator.toint())
        s_pref = s_u + self.o_pref
        s_pack = s_u + self.o_pack
        s_misc = s_u + self.o_misc
        r0 = group * rpt
        # the rows (warp 0, lane t holding row t): columns read, request and position; the
        # group's segment (one request's rows; any other group is lone), live rows, own tiles, T0
        if warp == 0:
            row = r0 + lane
            live = (lane < rpt) & (row < rows)
            srow = cutlass.min(row, rows - 1)
            if const_expr(self.req64):
                rq = cutlass.Int32(_ldg_i64(req_base + cutlass.Int64(srow) * 8))
            else:
                rq = _ldg_i32(req_base + cutlass.Int64(srow) * 4)
            if const_expr(self.qp64):
                qp = cutlass.Int32(_ldg_i64(qp_base + cutlass.Int64(srow) * 8))
            else:
                qp = _ldg_i32(qp_base + cutlass.Int64(srow) * 4)
            visible = qp + 1
            sb = cutlass.min(visible // 4, cutlass.Int32(self.block_topk))
            valid = sb * 4 + visible % 4
            bounded = cutlass.min(valid, cutlass.Int32(self.topk))
            cr_ = cutlass.min((bounded + _BOUND_TILE - 1) // _BOUND_TILE * _BOUND_TILE,
                              cutlass.Int32(self.topk))  # fmt: skip
            cr_ = cutlass.max(cr_, cutlass.Int32(0))
            cr_ = cutlass.Int32(cutlass.select_(live, cr_, cutlass.Int32(0)))
            if lane < rpt:
                _st_shared_b32(s_misc + (_BP_COLS + lane) * 4, cr_)
                _st_shared_b32(s_misc + (_BP_HI + lane) * 4, cutlass.Int32(-1))
            q0 = cute.arch.shuffle_sync(rq, 0)  # row r0 is always live
            other = live & (rq != q0)
            split = cutlass.Int32(cutlass.select_(other, lane, big))
            q1_lo = cutlass.Int32(cutlass.select_(other, rq, big))
            q1_hi = cutlass.Int32(cutlass.select_(other, rq, cutlass.Int32(-1)))
            own = (cr_ + BLOCK_N - 1) // BLOCK_N
            nl = cutlass.Int32(cutlass.select_(live, cutlass.Int32(1), cutlass.Int32(0)))
            for o in (1, 2, 4, 8, 16):
                split = cutlass.min(split, cute.arch.shuffle_sync_bfly(split, offset=o))
                q1_lo = cutlass.min(q1_lo, cute.arch.shuffle_sync_bfly(q1_lo, offset=o))
                q1_hi = cutlass.max(q1_hi, cute.arch.shuffle_sync_bfly(q1_hi, offset=o))
                own = own + cute.arch.shuffle_sync_bfly(own, offset=o)
                nl = nl + cute.arch.shuffle_sync_bfly(nl, offset=o)
            # rows of request q0 past the split break the run (three runs): lone
            late0 = live & (rq == q0) & (lane > split)
            seg1 = lane >= split
            p0 = cutlass.Int32(cutlass.select_(live & (seg1 == False), qp, big))  # noqa: E712
            p1 = cutlass.Int32(cutlass.select_(live & seg1, qp, big))
            late = cutlass.Int32(cutlass.select_(late0, cutlass.Int32(1), cutlass.Int32(0)))
            for o in (1, 2, 4, 8, 16):
                p0 = cutlass.min(p0, cute.arch.shuffle_sync_bfly(p0, offset=o))
                p1 = cutlass.min(p1, cute.arch.shuffle_sync_bfly(p1, offset=o))
                late = late | cute.arch.shuffle_sync_bfly(late, offset=o)
            if lane == 0:
                ok0 = (q0 >= 0) & (q0 < num_requests) & (p0 >= 0)
                ok1 = (q1_lo == q1_hi) & (q1_lo >= 0) & (q1_lo < num_requests) & (p1 >= 0)
                ok1 = ok1 & (late == 0)
                nseg = cutlass.Int32(0)
                if split == big:
                    nseg = cutlass.Int32(cutlass.select_(ok0, cutlass.Int32(1), cutlass.Int32(0)))
                _st_shared_b32(s_misc + _BP_NSEG * 4, nseg)
                _st_shared_b32(s_misc + _BP_SPLIT * 4, cutlass.min(split, cutlass.Int32(rpt)))
                _st_shared_b32(s_misc + _BP_REQ0 * 4, cutlass.max(q0, 0))
                _st_shared_b32(s_misc + _BP_REQ1 * 4, cutlass.max(cutlass.min(q1_lo, big - 1), 0))
                _st_shared_b32(s_misc + _BP_NLIVE * 4, nl)
                _st_shared_b32(s_misc + _BP_OWN * 4, own)
                _st_shared_b32(s_misc + _BP_T00 * 4,
                               cutlass.Int32(cutlass.select_(ok0, (p0 + 1) // 4 * 4, cutlass.Int32(0))))  # fmt: skip
                _st_shared_b32(s_misc + _BP_T01 * 4,
                               cutlass.Int32(cutlass.select_(ok1, (p1 + 1) // 4 * 4, cutlass.Int32(0))))  # fmt: skip
                _st_shared_b32(s_misc + _BP_FLAG * 4, cutlass.Int32(0))
        cute.arch.sync_threads()
        nseg = _ld_shared_b32(s_misc + _BP_NSEG * 4)
        split = _ld_shared_b32(s_misc + _BP_SPLIT * 4)
        n_live = _ld_shared_b32(s_misc + _BP_NLIVE * 4)
        own_tiles = _ld_shared_b32(s_misc + _BP_OWN * 4)
        # the rows' packed words (a fixed `packed_words` stride a row), zeroed
        for i in cutlass.range_constexpr((rpt * pw + nthreads - 1) // nthreads):
            w = tidx + i * nthreads
            if w < rpt * pw:
                _st_shared_b64(s_pack + w * 8, cutlass.Int64(0))
        nq: cutlass.Constexpr = -(-self.topk // 4)
        n_items: cutlass.Constexpr = rpt * nq
        n_rounds: cutlass.Constexpr = -(-n_items // nthreads)
        listed = cutlass.Int32(0)
        for sg in cutlass.range_constexpr(1):
            # a lone group (no segment) still walks its rows once: row_info's highest tokens
            if sg < cutlass.max(nseg, 1):
                # this segment: rows [ra, rb) of request `req`, list positions from `seg_base`
                ra = cutlass.Int32(0) if sg == 0 else split
                rb = cutlass.Int32(cutlass.select_(nseg == 2, split, cutlass.Int32(rpt)))
                if sg == 1:
                    rb = cutlass.Int32(rpt)
                req = _ld_shared_b32(s_misc + (_BP_REQ0 + sg) * 4)
                t_lo = _ld_shared_b32(s_misc + (_BP_T00 + sg) * 4)
                seg_base = listed
                bt_row = bt_base + cutlass.Int64(req) * bt_stride * 4
                for i in cutlass.range_constexpr((nbw + nthreads - 1) // nthreads):
                    w = tidx + i * nthreads
                    if w < nbw:
                        _st_shared_b64(s_u + w * 8, cutlass.Int64(0))
                if tidx == 0:
                    _st_shared_b64(s_misc + _BP_LOC * 4, cutlass.Int64(0))
                cute.arch.sync_threads()
                # walk 1: items are (row, four-column quad) pairs, item = i * nthreads + tidx; a
                # block quad below T0 sets its block bit, any other quad its tokens' local bits
                # (rolled: unrolled it spills).
                for i in cutlass.range(0, n_rounds, unroll=1):
                    item = i * nthreads + tidx
                    t = cutlass.min(item // nq, rpt - 1)
                    tk = self._quad(sel_base, r0, rows, sel_stride, s_misc, item, ra, rb)
                    b0 = tk[0]
                    is_blk = (b0 >= 0) & (b0 < t_lo) & ((b0 & 3) == 0)
                    is_blk = is_blk & (tk[1] == b0 + 1) & (tk[2] == b0 + 2) & (tk[3] == b0 + 3)
                    hi_q = cutlass.max(cutlass.max(tk[0], tk[1]), cutlass.max(tk[2], tk[3]))
                    if is_blk:
                        blk = b0 >> 2
                        _red_or_shared32(s_u + (blk >> 5) * 4, cutlass.Int32(1) << (blk & 31))
                    else:
                        if hi_q >= 0:
                            lm = cutlass.Int64(0)
                            bad = cutlass.Int32(0)
                            for k in cutlass.range_constexpr(4):
                                d = tk[k] - t_lo
                                inr = (d >= 0) & (d < 64)
                                live_k = tk[k] >= 0
                                lm = lm | cutlass.Int64(
                                    cutlass.select_(
                                        live_k & inr,
                                        cutlass.Int64(1) << cutlass.Int64(d & 63),
                                        cutlass.Int64(0),
                                    )
                                )
                                bad = bad | cutlass.Int32(
                                    cutlass.select_(
                                        live_k & ((d < 0) | (d >= 64)),
                                        cutlass.Int32(1),
                                        cutlass.Int32(0),
                                    )
                                )
                            lm_lo = cutlass.Int32(lm & cutlass.Int64(0xFFFFFFFF))
                            lm_hi = cutlass.Int32(lm >> cutlass.Int64(32))
                            if lm_lo != 0:
                                _red_or_shared32(s_misc + _BP_LOC * 4, lm_lo)
                            if lm_hi != 0:
                                _red_or_shared32(s_misc + _BP_LOC * 4 + 4, lm_hi)
                            if bad != 0:
                                _st_shared_b32(s_misc + _BP_FLAG * 4, cutlass.Int32(1))
                    # the row's highest token: a warp's 32 items span at most two rows (nq > 32)
                    t_first = cute.arch.shuffle_sync(t, 0)
                    same = t == t_first
                    hi_a = cutlass.Int32(cutlass.select_(same, hi_q, cutlass.Int32(-1)))
                    hi_b = cutlass.Int32(cutlass.select_(same, cutlass.Int32(-1), hi_q))
                    for o in (1, 2, 4, 8, 16):
                        hi_a = cutlass.max(hi_a, cute.arch.shuffle_sync_bfly(hi_a, offset=o))
                        hi_b = cutlass.max(hi_b, cute.arch.shuffle_sync_bfly(hi_b, offset=o))
                    if lane == 0:
                        if hi_a >= 0:
                            _max_shared(s_misc + (_BP_HI + t_first) * 4, hi_a)
                    t_last = cute.arch.shuffle_sync(t, 31)
                    if lane == 31:
                        if (t_last != t_first) & (hi_b >= 0):
                            _max_shared(s_misc + (_BP_HI + t_last) * 4, hi_b)
                cute.arch.sync_threads()
                use = (_ld_shared_b32(s_misc + _BP_FLAG * 4) == 0) & (nseg > 0)
                words_needed = cutlass.Int32(
                    cutlass.select_(use, (t_lo // 4 + 63) // 64, cutlass.Int32(0))
                )
                words_needed = cutlass.min(words_needed, cutlass.Int32(nbw))
                # resident blocks: a word's bits on a page outside the table or the pool dropped
                cnts = []
                for i in cutlass.range_constexpr((nbw + nthreads - 1) // nthreads):
                    w = tidx + i * nthreads
                    cnt = cutlass.Int32(0)
                    if w < words_needed:
                        u = _ld_shared_b64(s_u + w * 8)
                        if u != 0:
                            first = w * 256
                            pgs = _word_page_ids(bt_row, first // tpb, tpb, self.width)
                            for k in cutlass.range_constexpr(_span_pages(256, tpb, True)):
                                lp = first // tpb + k
                                lo_b = cutlass.max(lp * tpb - first, 0) // 4
                                hi_b = cutlass.min((lp + 1) * tpb - first, 256) // 4
                                if lo_b < hi_b:
                                    span = _bits_range64(lo_b, hi_b)
                                    pg = pgs[k]
                                    ok = (lp < self.width) & (pg >= 0) & (pg < num_cache_pages)
                                    u = cutlass.Int64(cutlass.select_(ok, u, u & ~span))
                            _st_shared_b64(s_u + w * 8, u)
                            cnt = _cu_popc64(u)
                    cnts.append(cnt)
                # the local word's resident tokens
                if tidx == nthreads - 1:
                    lw = _ld_shared_b64(s_misc + _BP_LOC * 4)
                    if use & (lw != 0):
                        for k in cutlass.range_constexpr(_span_pages(64, tpb, False)):
                            lp = t_lo // tpb + k
                            lo_b = cutlass.max(lp * tpb - t_lo, 0)
                            hi_b = cutlass.min((lp + 1) * tpb - t_lo, 64)
                            if lo_b < hi_b:
                                span = _bits_range64(lo_b, hi_b)
                                if (lw & span) != 0:
                                    ok = lp < self.width
                                    pg = cutlass.Int32(-1)
                                    if ok:
                                        pg = _ldg_i32(bt_row + cutlass.Int64(lp) * 4)
                                    ok = ok & (pg >= 0) & (pg < num_cache_pages)
                                    lw = cutlass.Int64(cutlass.select_(ok, lw, lw & ~span))
                    lw = cutlass.Int64(cutlass.select_(use, lw, cutlass.Int64(0)))
                    _st_shared_b64(s_misc + _BP_LOC * 4, lw)
                    _st_shared_b32(s_misc + _BP_LOCN * 4, _cu_popc64(lw))
                cute.arch.sync_threads()
                # list positions of the blocks: exclusive prefix of the words' counts, round by round
                base = cutlass.Int32(0)
                for i in cutlass.range_constexpr((nbw + nthreads - 1) // nthreads):
                    if i * nthreads < words_needed:
                        incl = cnts[i]
                        for o in (1, 2, 4, 8, 16):
                            up = cute.arch.shuffle_sync_up(incl, offset=o)
                            if lane >= o:
                                incl = incl + up
                        if lane == 31:
                            _st_shared_b32(s_misc + (_BP_SCAN + warp) * 4, incl)
                        cute.arch.sync_threads()
                        wv = cutlass.Int32(0)
                        tot = cutlass.Int32(0)
                        if lane < nwarps:
                            wv = _ld_shared_b32(s_misc + (_BP_SCAN + lane) * 4)
                            tot = wv
                        before = cute.arch.warp_reduction_sum(
                            cutlass.Int32(cutlass.select_(lane < warp, wv, cutlass.Int32(0)))
                        )
                        total = cute.arch.warp_reduction_sum(tot)
                        w = tidx + i * nthreads
                        if w < nbw:
                            _st_shared_b32(s_pref + w * 4, base + before + incl - cnts[i])
                        base = base + total
                        cute.arch.sync_threads()
                local_base = seg_base + base * 4
                listed = local_base + _ld_shared_b32(s_misc + _BP_LOCN * 4)
                if use & (listed > seg_base):
                    # slots: a thread a block word, four (one 16-byte store) a listed block
                    slot_row = slots_base + cutlass.Int64(group) * slots_stride * 4
                    for i in cutlass.range_constexpr((nbw + nthreads - 1) // nthreads):
                        w = tidx + i * nthreads
                        if w < words_needed:
                            u = _ld_shared_b64(s_u + w * 8)
                            j = seg_base + _ld_shared_b32(s_pref + w * 4) * 4
                            lp0 = (w * 256) // tpb
                            pgs = _word_page_ids(bt_row, lp0, tpb, self.width)
                            while u != 0:
                                b = _cu_ctz64(u)
                                tok = (w * 64 + b) * 4
                                which = tok // tpb - lp0
                                pg = pgs[0]
                                for k in cutlass.range_constexpr(1, len(pgs)):
                                    pg = cutlass.Int32(cutlass.select_(which == k, pgs[k], pg))
                                s0 = pg * tpb + tok % tpb
                                if (seg_base & 3) == 0:
                                    _stg_v4_i32(slot_row + cutlass.Int64(j) * 4,
                                                s0, s0 + 1, s0 + 2, s0 + 3)  # fmt: skip
                                else:
                                    # a second segment's blocks may sit off a 16-byte boundary
                                    _stg_i32(slot_row + cutlass.Int64(j) * 4, s0)
                                    _stg_i32(slot_row + cutlass.Int64(j + 1) * 4, s0 + 1)
                                    _stg_i32(slot_row + cutlass.Int64(j + 2) * 4, s0 + 2)
                                    _stg_i32(slot_row + cutlass.Int64(j + 3) * 4, s0 + 3)
                                j = j + 4
                                u = u & (u - 1)
                    if tidx == nthreads - 1:
                        lw = _ld_shared_b64(s_misc + _BP_LOC * 4)
                        j = local_base
                        while lw != 0:
                            b = _cu_ctz64(lw)
                            tok = t_lo + b
                            pg = _ldg_i32(bt_row + cutlass.Int64(tok // tpb) * 4)
                            _stg_i32(slot_row + cutlass.Int64(j) * 4, pg * tpb + tok % tpb)
                            j = j + 1
                            lw = lw & (lw - 1)
                    # walk 2 (quads re-read): a listed block's four bits, a local token's one bit
                    lw = _ld_shared_b64(s_misc + _BP_LOC * 4)
                    for i in cutlass.range(0, n_rounds, unroll=1):
                        item = i * nthreads + tidx
                        t = cutlass.min(item // nq, rpt - 1)
                        tk = self._quad(sel_base, r0, rows, sel_stride, s_misc, item, ra, rb)
                        b0 = tk[0]
                        is_blk = (b0 >= 0) & (b0 < t_lo) & ((b0 & 3) == 0)
                        is_blk = is_blk & (tk[1] == b0 + 1) & (tk[2] == b0 + 2) & (tk[3] == b0 + 3)
                        row_pack = s_pack + t * (pw * 8)
                        if is_blk:
                            blk = b0 >> 2
                            w = blk >> 6
                            bit = cutlass.Int64(1) << cutlass.Int64(blk & 63)
                            uw = _ld_shared_b64(s_u + w * 8)
                            if (uw & bit) != 0:
                                k = _ld_shared_b32(s_pref + w * 4) + _cu_popc64(uw & (bit - 1))
                                pos = seg_base + k * 4
                                _red_or_shared32(row_pack + (pos >> 5) * 4,
                                                 cutlass.Int32(15) << (pos & 31))  # fmt: skip
                        else:
                            for k in cutlass.range_constexpr(4):
                                d = tk[k] - t_lo
                                if (tk[k] >= 0) & (d >= 0) & (d < 64):
                                    bit = cutlass.Int64(1) << cutlass.Int64(d)
                                    if (lw & bit) != 0:
                                        jj = local_base + _cu_popc64(lw & (bit - 1))
                                        _red_or_shared32(row_pack + (jj >> 5) * 4,
                                                         cutlass.Int32(1) << (jj & 31))  # fmt: skip
                cute.arch.sync_threads()
        use = (nseg > 0) & (_ld_shared_b32(s_misc + _BP_FLAG * 4) == 0)
        listed = cutlass.Int32(cutlass.select_(use, listed, cutlass.Int32(0)))
        if tidx < rpt:
            if r0 + tidx < rows:
                _stg_i32(info_base + cutlass.Int64(r0 + tidx) * 8,
                         _ld_shared_b32(s_misc + (_BP_HI + tidx) * 4))  # fmt: skip
                _stg_i32(info_base + cutlass.Int64(r0 + tidx) * 8 + 4,
                         _ld_shared_b32(s_misc + (_BP_COLS + tidx) * 4))  # fmt: skip
        tiles = (listed + BLOCK_N - 1) // BLOCK_N
        shared = use & (tiles * _SHARED_TILE_COST <= own_tiles * 4)
        shared_i = cutlass.Int32(cutlass.select_(shared, cutlass.Int32(1), cutlass.Int32(0)))
        if tidx == 0:
            _stg_i32(
                plan_base + cutlass.Int64(group) * 8,
                cutlass.Int32(cutlass.select_(shared, tiles, cutlass.Int32(-1))),
            )
            _stg_i32(plan_base + cutlass.Int64(group) * 8 + 4, listed)
            if shared_i == 0:
                lb = _atom_add_g32(lone_count_base, n_live)
                _st_shared_b32(s_misc + _BP_LONE * 4, lb)
        if shared_i == 0:
            cute.arch.sync_threads()
            if tidx < rpt:
                if r0 + tidx < rows:
                    lb = _ld_shared_b32(s_misc + _BP_LONE * 4)
                    _stg_i32(lone_base + cutlass.Int64(lb + tidx) * 4, r0 + tidx)
        if (shared_i != 0) & (listed > 0):
            pw_used = (listed + 63) // 64
            for i in cutlass.range(tidx, rpt * pw_used, nthreads, unroll=1):
                t = i // pw_used
                k = i % pw_used
                if r0 + t < rows:
                    _stg_i64(packed_base + (cutlass.Int64(r0 + t) * pw + k) * 8,
                             _ld_shared_b64(s_pack + (t * pw + k) * 8))  # fmt: skip


_BLOCK_PLAN: dict = {}


def block_plan_serves(compress_ratio: int, tokens_per_block: int, rows_per_tile: int) -> bool:
    """Whether `_block_plan` serves a launch: blocks of 4 tokens, pages of whole blocks, a
    group's rows in one warp."""
    return compress_ratio == 4 and tokens_per_block % 4 == 0 and rows_per_tile <= 32


def _block_plan(selected_tokens, query_positions, request_indices, block_table, compress_ratio,
                tokens_per_block, table_width, rows_per_tile, num_groups, packed_words,
                row_info, slots, packed_bits, group_plan, lone_rows, lone_count, num_cache_pages):  # fmt: skip
    """Launch the block plan pass (one CTA a group); the fused plan's outputs."""
    rows, final_topk = selected_tokens.shape
    block_topk = (final_topk - compress_ratio + 1) // compress_ratio
    block_words = triton.cdiv(triton.cdiv(table_width * tokens_per_block, 4), 64)
    req64 = request_indices.dtype == torch.int64
    qp64 = query_positions.dtype == torch.int64
    query_positions = query_positions.contiguous()
    key = (final_topk, tokens_per_block, table_width, rows_per_tile, packed_words, req64, qp64)
    gm = cute.AddressSpace.gmem
    ptrs = (
        query_positions.data_ptr(), request_indices.data_ptr(), block_table.data_ptr(),
        row_info.data_ptr(), slots.data_ptr(), packed_bits.data_ptr(), group_plan.data_ptr(),
        lone_rows.data_ptr(), lone_count.data_ptr(),
    )  # fmt: skip
    scalars = (rows, num_groups, selected_tokens.stride(0), block_table.stride(0), slots.stride(0),
               block_table.shape[0], num_cache_pages)  # fmt: skip
    fn = _BLOCK_PLAN.get(key)
    if fn is None:
        i32, i64 = cutlass.Int32, cutlass.Int64
        fn = cute.compile(
            _BlockPlanKernel(
                final_topk,
                block_topk,
                tokens_per_block,
                table_width,
                block_words,
                rows_per_tile,
                packed_words,
                req64,
                qp64,
            ),  # fmt: skip
            from_dlpack(selected_tokens, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            make_ptr(i64 if qp64 else i32, ptrs[0], gm, assumed_align=4),
            make_ptr(i64 if req64 else i32, ptrs[1], gm, assumed_align=4),
            make_ptr(i32, ptrs[2], gm, assumed_align=4),
            make_ptr(i32, ptrs[3], gm, assumed_align=8),
            make_ptr(i32, ptrs[4], gm, assumed_align=16),
            make_ptr(i64, ptrs[5], gm, assumed_align=8),
            make_ptr(i32, ptrs[6], gm, assumed_align=8),
            make_ptr(i32, ptrs[7], gm, assumed_align=4),
            make_ptr(i32, ptrs[8], gm, assumed_align=4),
            *(cutlass.Int32(x) for x in scalars),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _BLOCK_PLAN[key] = fn
    fn(selected_tokens, *ptrs, *scalars)


@triton.jit(do_not_specialize=["rows", "NUM_CACHE_PAGES", "NUM_REQUESTS"])
def _rowtile_lone_kernel(
    q, k_cache, v_cache, block_table, selected_tokens, request_indices, row_info,
    lone_rows, lone_count, output, softmax_scale, rows,
    q_stride_row, q_stride_head: tl.constexpr, q_stride_dim: tl.constexpr,
    k_stride_page: tl.constexpr, k_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr, k_stride_dim: tl.constexpr,
    v_stride_page: tl.constexpr, v_stride_head: tl.constexpr,
    v_stride_token: tl.constexpr, v_stride_dim: tl.constexpr,
    block_table_stride_request: tl.constexpr, block_table_stride_page: tl.constexpr,
    selected_stride_row: tl.constexpr, selected_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr, output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    NUM_CACHE_PAGES, NUM_REQUESTS, LONE_PROGRAMS,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr, TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, USE_PDL: tl.constexpr,
    LONE_STAGES: tl.constexpr,
):  # fmt: skip
    """Rows of groups the row-tile pass does not serve: each over its own columns. q's row stride
    is a runtime argument, so q in place on the projection row and a compact row share a compile."""
    if USE_PDL:
        # Independent of the row-tile pass (disjoint rows): let it start now.
        tl.extra.cuda.gdc_launch_dependents()
    kv_head = tl.program_id(1)
    for slot in range(tl.program_id(0), tl.load(lone_count), LONE_PROGRAMS):
        _rowtile_lone_row(
            q, k_cache, v_cache, block_table, selected_tokens, request_indices, row_info, output,
            softmax_scale, tl.load(lone_rows + slot).to(tl.int64), kv_head,
            q_stride_row, q_stride_head, q_stride_dim,
            k_stride_page, k_stride_head, k_stride_token, k_stride_dim,
            v_stride_page, v_stride_head, v_stride_token, v_stride_dim,
            block_table_stride_request, block_table_stride_page,
            selected_stride_row, selected_stride_token,
            output_stride_row, output_stride_head, output_stride_dim,
            NUM_CACHE_PAGES, NUM_REQUESTS, PAGE_TABLE_WIDTH, GROUP_SIZE, TOKENS_PER_BLOCK,
            BLOCK_H, BLOCK_N, HEAD_DIM, LONE_STAGES,
        )  # fmt: skip


@triton.jit
def _rowtile_lone_row(
    q, k_cache, v_cache, block_table, selected_tokens, request_indices, row_info, output,
    softmax_scale, row, kv_head,
    q_stride_row, q_stride_head, q_stride_dim,
    k_stride_page, k_stride_head, k_stride_token, k_stride_dim,
    v_stride_page, v_stride_head, v_stride_token, v_stride_dim,
    block_table_stride_request, block_table_stride_page,
    selected_stride_row, selected_stride_token,
    output_stride_row, output_stride_head, output_stride_dim,
    NUM_CACHE_PAGES, NUM_REQUESTS, PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr, TOKENS_PER_BLOCK: tl.constexpr, BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, LONE_STAGES: tl.constexpr,
):  # fmt: skip
    request = tl.load(request_indices + row).to(tl.int64)
    valid_request = (request >= 0) & (request < NUM_REQUESTS)
    safe_request = tl.minimum(tl.maximum(request, 0), NUM_REQUESTS - 1)
    page_base = block_table + safe_request * block_table_stride_request
    columns_read = tl.load(row_info + row * 2 + 1)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, HEAD_DIM)
    query_heads = kv_head * GROUP_SIZE + head_offsets
    head_live = head_offsets < GROUP_SIZE
    q_tile = tl.load(
        q + row * q_stride_row + query_heads[:, None] * q_stride_head
        + dim_offsets[None, :] * q_stride_dim,
        mask=head_live[:, None],
        other=0.0,
    )  # fmt: skip
    accumulator = tl.zeros([BLOCK_H, HEAD_DIM], tl.float32)
    running_max = tl.full([BLOCK_H], -1.0e30, tl.float32)
    running_sum = tl.zeros([BLOCK_H], tl.float32)
    token_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(0, tl.cdiv(columns_read, BLOCK_N), num_stages=LONE_STAGES):
        columns = tile * BLOCK_N + token_offsets
        tokens = tl.load(
            selected_tokens + row * selected_stride_row + columns * selected_stride_token,
            mask=columns < columns_read,
            other=-1,
        ).to(tl.int32)
        safe_tokens = tl.maximum(tokens, 0)
        logical_pages = safe_tokens // TOKENS_PER_BLOCK
        token_in_page = safe_tokens % TOKENS_PER_BLOCK
        usable = valid_request & (tokens >= 0) & (logical_pages < PAGE_TABLE_WIDTH)
        physical_pages = tl.load(
            page_base + tl.minimum(logical_pages, PAGE_TABLE_WIDTH - 1) * block_table_stride_page,
            mask=usable,
            other=-1,
        ).to(tl.int64)
        usable = usable & (physical_pages >= 0) & (physical_pages < NUM_CACHE_PAGES)
        safe_pages = tl.maximum(physical_pages, 0)
        keys = tl.load(
            k_cache + safe_pages[None, :] * k_stride_page + kv_head * k_stride_head
            + token_in_page[None, :] * k_stride_token + dim_offsets[:, None] * k_stride_dim,
            mask=usable[None, :],
            other=0.0,
        ).to(q_tile.dtype)  # fmt: skip
        values = tl.load(
            v_cache + safe_pages[:, None] * v_stride_page + kv_head * v_stride_head
            + token_in_page[:, None] * v_stride_token + dim_offsets[None, :] * v_stride_dim,
            mask=usable[:, None],
            other=0.0,
        ).to(q_tile.dtype)  # fmt: skip
        scores = tl.dot(q_tile, keys) * (softmax_scale * 1.4426950408889634)
        scores = tl.where(usable[None, :], scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        correction = tl.math.exp2(running_max - next_max)
        probabilities = tl.where(usable[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(
            probabilities.to(q_tile.dtype), values, accumulator * correction[:, None]
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = next_max
    normalized = tl.where(
        running_sum[:, None] > 0, accumulator / tl.maximum(running_sum[:, None], 1.0e-20), 0.0
    )
    tl.store(
        output + row * output_stride_row + query_heads[:, None] * output_stride_head
        + dim_offsets[None, :] * output_stride_dim,
        normalized.to(output.dtype.element_ty),
        mask=head_live[:, None],
    )  # fmt: skip


def _check_inputs(q, k_cache, v_cache, block_table, selected_tokens, request_indices,
                  tokens_per_block, softmax_scale, query_positions, compress_ratio):  # fmt: skip
    if q.ndim != 3:
        raise ValueError("QSA queries must have shape [rows, query_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA K/V caches must have matching HND paged shapes")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA sparse GQA requires a two-dimensional int32 block table")
    if selected_tokens.ndim != 2 or selected_tokens.dtype != torch.int32:
        raise ValueError("QSA selected tokens must be a two-dimensional int32 tensor")
    rows, num_q_heads, head_dim = q.shape
    if selected_tokens.shape[0] != rows or selected_tokens.shape[1] <= 0:
        raise ValueError("QSA selected-token rows must match non-empty query rows")
    if request_indices.shape != (rows,) or request_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("QSA request indices must have one integer entry per row")
    if not request_indices.is_contiguous():
        raise ValueError("QSA request indices must be contiguous")
    if tokens_per_block <= 0 or k_cache.shape[2] != tokens_per_block:
        raise ValueError("QSA cache token dimension must match positive tokens_per_block")
    if k_cache.shape[3] != head_dim or k_cache.dtype != v_cache.dtype:
        raise ValueError("QSA query and K/V cache head geometry is incompatible")
    if q.dtype != torch.bfloat16:
        raise ValueError("QSA queries must use BF16")
    if k_cache.dtype != torch.bfloat16:
        raise ValueError("QSA K/V cache must use BF16")
    if selected_tokens.stride(1) != 1:
        raise ValueError("QSA selected tokens must have unit column stride")
    num_kv_heads = k_cache.shape[1]
    if num_kv_heads <= 0 or num_q_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("QSA query heads must be divisible by non-zero KV heads")
    if k_cache.shape[0] <= 0 or block_table.shape[0] <= 0 or block_table.shape[1] <= 0:
        raise ValueError("QSA sparse GQA requires non-empty cache and block-table geometry")
    if k_cache.shape[0] * tokens_per_block >= 2**31:
        raise ValueError("QSA row-tile cache slots must fit int32")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("QSA sparse GQA softmax scale must be finite and positive")
    if query_positions is None or compress_ratio is None:
        raise ValueError("the row-tile kernel needs the prefill pair (positions, compress ratio)")
    if query_positions.shape != (rows,) or query_positions.dtype not in _INTEGER_DTYPES:
        raise ValueError("QSA query positions must be one integer entry per row")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    token_topk = selected_tokens.shape[1] - compress_ratio + 1
    if token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA selected-token width is incompatible with its compression ratio")
    devices = {t.device for t in (q, k_cache, v_cache, block_table, selected_tokens,
                                  request_indices, query_positions)}  # fmt: skip
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("QSA row-tile sparse GQA needs every tensor on one CUDA device")
    if num_q_heads // num_kv_heads > BLOCK_M:
        raise ValueError("a row's query heads per KV head must fit the 128-row tile")
    if not _cute_eligible(q, k_cache, v_cache):
        raise ValueError(
            "the row-tile pass needs SM103, head_dim 256, unit-stride heads (q's heads packed) "
            "and 16-byte-aligned rows"
        )


def rows_per_group(group_size: int) -> int:
    """Rows of one request a row-tile item serves: the most whose `rows x group_size` MMA
    rows fill whole 8-row swizzle atoms of the 128-row tile (10 at 12 heads, 20 at 6)."""
    rows = BLOCK_M // group_size
    while rows > 1 and rows * group_size % _ROW_ATOM:
        rows -= 1
    return rows


def plan_launch(block_table, selected_tokens, request_indices, query_positions,
                tokens_per_block, compress_ratio, rows_per_tile, num_cache_pages):  # fmt: skip
    """The row-tile plan of a context launch: `(row_info, slots, packed_bits, group_plan,
    lone_rows, counters)` (`counters[0]` the lone list's length, `counters[1]` the row-tile
    pass's ticket, zeroed), from one launch: on the selection's blocks wherever
    `block_plan_serves`, else on its tokens. Both give the same values."""
    rows = selected_tokens.shape[0]
    device = selected_tokens.device
    table_width = block_table.shape[1]
    num_words = triton.cdiv(table_width * tokens_per_block, 64)
    num_groups = triton.cdiv(rows, rows_per_tile)
    # A group lists at most its rows' columns: the bound on its slots and re-indexed bits.
    listed_cap = min(num_words * 64, rows_per_tile * selected_tokens.shape[1])
    packed_words = triton.cdiv(listed_cap, 64)
    slots = torch.empty((num_groups, packed_words * 64), dtype=torch.int32, device=device)
    packed_bits = torch.zeros((rows, packed_words), dtype=torch.int64, device=device)
    group_plan = torch.empty((num_groups, 2), dtype=torch.int32, device=device)
    lone_rows = torch.empty((rows,), dtype=torch.int32, device=device)
    counters = torch.zeros((2,), dtype=torch.int32, device=device)
    lone_count = counters[:1]
    row_info = torch.empty((rows, 2), dtype=torch.int32, device=device)
    if block_plan_serves(compress_ratio, tokens_per_block, rows_per_tile):
        _block_plan(
            selected_tokens, query_positions, request_indices, block_table, compress_ratio,
            tokens_per_block, table_width, rows_per_tile, num_groups, packed_words,
            row_info, slots, packed_bits, group_plan, lone_rows, lone_count, num_cache_pages,
        )  # fmt: skip
    else:
        _group_plan(
            selected_tokens, query_positions, request_indices, block_table, compress_ratio,
            tokens_per_block, table_width, num_words, rows_per_tile, num_groups, packed_words,
            row_info, slots, packed_bits, group_plan, lone_rows, lone_count, num_cache_pages,
        )  # fmt: skip
    return row_info, slots, packed_bits, group_plan, lone_rows, counters


def qsa_rowtile_sparse_gqa(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    selected_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    softmax_scale: float,
    query_positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Plan pass (`plan_launch`), lone-row pass, row-tile pass (`qsa_rowtile_attention_cute`);
    returns `[rows, Hq, D]` contiguous."""
    _check_inputs(q, k_cache, v_cache, block_table, selected_tokens, request_indices,
                  tokens_per_block, softmax_scale, query_positions, compress_ratio)  # fmt: skip
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    output = torch.empty((rows, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    if rows == 0:
        return output
    table_width = block_table.shape[1]
    rows_per_tile = rows_per_group(group_size)
    kv_strides = (
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
    )  # fmt: skip
    q_strides = (q.stride(0), q.stride(1), q.stride(2))
    common = (*q_strides, *kv_strides, block_table.stride(0), block_table.stride(1))
    geometry = dict(
        NUM_CACHE_PAGES=k_cache.shape[0], NUM_REQUESTS=block_table.shape[0],
        PAGE_TABLE_WIDTH=table_width, GROUP_SIZE=group_size,
        TOKENS_PER_BLOCK=tokens_per_block, HEAD_DIM=head_dim,
    )  # fmt: skip
    out_strides = (output.stride(0), output.stride(1), output.stride(2))
    num_groups = triton.cdiv(rows, rows_per_tile)
    row_info, slots, packed_bits, group_plan, lone_rows, counters = plan_launch(
        block_table, selected_tokens, request_indices, query_positions, tokens_per_block,
        compress_ratio, rows_per_tile, k_cache.shape[0],
    )  # fmt: skip
    lone_count = counters[:1]
    use_pdl = get_env_enable_pdl()
    _rowtile_lone_kernel[(min(rows, _LONE_PROGRAMS), num_kv_heads)](
        q, k_cache, v_cache, block_table, selected_tokens, request_indices, row_info,
        lone_rows, lone_count, output, softmax_scale, rows,
        *common, selected_tokens.stride(0), selected_tokens.stride(1), *out_strides,
        BLOCK_H=max(16, triton.next_power_of_2(group_size)), BLOCK_N=64,
        LONE_PROGRAMS=min(rows, _LONE_PROGRAMS), USE_PDL=use_pdl, LONE_STAGES=_LONE_STAGES,
        num_warps=_LONE_WARPS, **geometry,
    )  # fmt: skip
    _rowtile_attention_cute[(_rowtile_ctas(num_groups * num_kv_heads, q.device),)](
        q, k_cache, v_cache, slots, packed_bits, group_plan, output, counters[1:], softmax_scale,
        rows,
        group_size=group_size, rows_per_tile=rows_per_tile, tokens_per_block=tokens_per_block,
        use_pdl=use_pdl,
    )  # fmt: skip
    return output


_CHECKPOINT = dict(
    q_heads=24,
    kv_heads=2,
    head_dim=256,
    tokens_per_block=32,
    compress_ratio=4,
    token_topk=2048,
    cache_dtype=torch.bfloat16,
    q_dtype=torch.bfloat16,
    # q is the first heads of the packed [T, Hq + 2 Hkv, 256] QKV buffer.
    q_layout="qkv_slice",
    request_dtype=torch.int32,
    # The engine's table width at max_seq_len 262144, -1 past each sequence.
    pages=8196,
)

#: The trunk: 24 role pages a slot, layer 11's K and V read, pages straddling 2^31 elements.
_TRUNK = dict(cache_layout="trunk", slot_pages=24, layer=11, offset_edge=2**31)
#: Attention DP with one-model MTP: the draft layer's planes (layer 12) join the trunk's slot.
_TRUNK_MTP_DP = dict(cache_layout="trunk", slot_pages=26, layer=12, offset_edge=2**31)
#: tp4 / tep4 rank: 6 q heads and one KV head.
_TP4 = dict(q_heads=6, kv_heads=1, slots=128, **_TRUNK)
#: tp2 / tep2 rank: 12 q heads and one KV head.
_TP2 = dict(q_heads=12, kv_heads=1, slots=128, **_TRUNK)
#: dep4 rank over the bf16 trunk, or over the kv_fp8 pages widened into a bf16 scratch.
_DEP4 = dict(q_heads=24, kv_heads=2, slots=32, **_TRUNK)
_DEP4_SCRATCH = dict(q_heads=24, kv_heads=2, slots=32, cache_layout="scratch")


def _spec(segments, member, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CHECKPOINT, **member, "segments": tuple(segments), **overrides}


class QsaSparseGqaRowtile(OpWrapper):
    """Sparse GQA over the selected tokens of an HND paged K/V pool, for a
    launch that carries the prefill pair (`query_positions`, `compress_ratio`).

    For row `r`, query head `h`, KV head `h // (Hq // Hkv)`, over the columns
    `j < bound(r)` of `selected_tokens[r]` that name a resident token::

        t     = selected_tokens[r, j]                      # logical position
        page  = block_table[request_indices[r], t // tokens_per_block]
        s_j   = (q[r, h] . k_cache[page, kv, t % tpb]) * softmax_scale
        out[r, h] = sum_j softmax(s)_j * v_cache[page, kv, t % tpb]

    `bound(r)` is the per-row kernel's prefill bound -- `min(visible // cr,
    block_topk) * cr + visible % cr` columns with `visible = query_positions[r]
    + 1`, rounded up to 64 -- so a row reads exactly the columns
    `qsa_paged_sparse_gqa` reads on a context launch. Negative indices, pages
    past the table, unmapped or out-of-pool pages and foreign requests
    contribute nothing; a row left with nothing returns zeros. Returns a
    fresh, contiguous `[rows, Hq, D]` tensor of `q.dtype`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp4/tep4 2049 rows from 9000: a last group of 9 rows, 6 q / 1 kv heads",
            spec=_spec([(9000, 2049)], _TP4),
        ),
        Cell(
            why="tp4/tep4 deep chunk: 8192 rows from 188416 ranking one score vector "
            "(a sparse union), 6 q / 1 kv heads, trunk pool straddling 2^31",
            spec=_spec([(188416, 8192)], _TP4, shared_scores=True),
        ),
        Cell(
            why="dep4 kv_fp8 deep chunk: 8160 rows from 188416 over the widened bf16 "
            "scratch, 24 q / 2 kv heads (10-row groups), [32, 8196] table",
            spec=_spec([(188416, 8160)], _DEP4_SCRATCH, shared_scores=True),
        ),
        Cell(
            why="dep4 core: three context requests (0+3000, 40000+4000, 120000+1160) in one "
            "launch, 24 q / 2 kv heads, trunk pool straddling 2^31",
            spec=_spec([(0, 3000), (40000, 4000), (120000, 1160)], _DEP4),
        ),
        Cell(
            why="dep4 MTP draft layer: 2100 rows from 50000 over layer 12 of a 26-role-page "
            "trunk (the draft layer in the trunk's manager), 24 q / 2 kv heads",
            spec=_spec([(50000, 2100)], {**_DEP4, **_TRUNK_MTP_DP}, shared_scores=True),
        ),
        Cell(
            why="tp2/tep2 deep chunk: 8192 rows from 120000, 12 q / 1 kv heads (10-row "
            "groups), [128, 8196] table, trunk pool straddling 2^31",
            spec=_spec([(120000, 8192)], _TP2, shared_scores=True),
        ),
        Cell(
            why="tp4/tep4 engine position dtype: int64 query positions with int32 request "
            "indices, 2049 rows from 9000, 6 q / 1 kv heads",
            spec=_spec([(9000, 2049)], _TP4, position_dtype=torch.int64),
        ),
    )

    note = """
    Source: this target's plan (CuTe DSL `_block_plan` / `_group_plan`), lone-row (Triton) and
    row-tile (`qsa_rowtile_attention_cute`, tcgen05) passes: SM103, bf16 q and K/V, D 256. It
    takes `qsa_paged_sparse_gqa`'s arguments with the prefill pair required.

    Preconditions (not checkable without reading the contents):
      - within the bound, a row names each token at most once: the row-tile pass attends to
        the set of tokens a row names, the per-row kernel to every column. QSA's top-k and
        expansion never repeat one. Rows served by the lone-row pass keep per-column semantics.
      - as for the per-row kernel: each row is prefix-packed with exactly the bound's count of
        valid entries; entries past the 64-rounded bound are ignored.

    Work grows with the union of a group's selections; below a few hundred rows the per-row
    kernel is faster. Compiled once per (group size, rows per tile, tokens per block, PDL).
    Calls allocate their plan scratch, so CUDA-graph capture is not certified (the prefill path
    runs eager). Deterministic.

    Not certified: int64 request indices; rows whose selection resolves no token
    (all padding, a foreign request, unmapped pages); `rows == 0`.
    """

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        query_positions: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        return qsa_rowtile_sparse_gqa(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=query_positions,
            compress_ratio=compress_ratio,
        )

    def reference(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        selected_tokens: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        softmax_scale: float,
        query_positions: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        """`qsa_paged_sparse_gqa.reference`: fp32 sparse GQA over every resident selected token,
        rounded once (the prefill pair is ignored: the bound reads exactly the valid entries)."""
        return _per_row_entry.reference(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=softmax_scale,
            query_positions=query_positions,
            compress_ratio=compress_ratio,
        )


qsa_sparse_gqa_rowtile = QsaSparseGqaRowtile()
