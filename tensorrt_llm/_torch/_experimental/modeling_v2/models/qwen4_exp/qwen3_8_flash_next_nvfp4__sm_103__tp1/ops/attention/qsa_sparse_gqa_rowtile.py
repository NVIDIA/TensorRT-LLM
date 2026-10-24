# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA for context batches: several query rows per program, K/V read once per tile.

The first launch turns each row's selection (the columns the prefill bound
lets it read) into a bitmap over logical token positions. A plan launch forms,
per group of consecutive rows of one request, the union of its rows' resident
selected tokens as a packed list of physical cache slots, and each row's
bitmap re-indexed onto that list. The row-tile launch is persistent: one
program per SM takes (group, KV head) items longest first and walks each packed
list in 64-slot tiles, each tile's K/V loaded once for every row and head of the
group, each row masked to its own bits. A group whose rows span requests
(generation rows of a mixed batch) walks each row's own selected columns
instead, as the per-row kernel does.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .qsa_paged_sparse_gqa import qsa_paged_sparse_gqa as _per_row_entry
from .qsa_rowtile_attention_cute import BLOCK_M, BLOCK_N
from .qsa_rowtile_attention_cute import eligible as _cute_eligible
from .qsa_rowtile_attention_cute import num_ctas as _rowtile_ctas
from .qsa_rowtile_attention_cute import rowtile_attention_cute as _rowtile_attention_cute

_INTEGER_DTYPES = (torch.int32, torch.int64)
# The per-row kernel's scan tile: the prefill bound is rounded up to it, so the
# bitmap reads exactly the columns that kernel reads.
_BOUND_TILE = 64
# Programs of the lone-row pass per KV head; each walks the lone list with this stride.
_LONE_PROGRAMS = 256  # capacity: fixed: a launch width; the lone list may be any length
# Bitmap words the plan pass unions and packs per loop step.
_PLAN_WORDS = 16  # capacity: fixed: a loop step; the plan loops over every word
# A one-request group takes the row-tile pass when its list's 64-slot tiles, each costing
# this many quarters of a per-row tile (2.5x), cost no more than its rows' own columns.
_SHARED_TILE_COST = 10
_PLAN_WARPS = 4
_LONE_WARPS = 2
_LONE_STAGES = 2


# --------------------------------------------------------------------------------------
# The bitmap pass (CuTe DSL, Tier 6). Row `r`: one bit per readable selected token (a
# column below the per-row kernel's prefill bound naming a token >= 0 whose page is inside
# the table) set in `bits[r]`, and the highest such token and the number of columns read
# in `row_info[r]`. A CTA owns one row: the row's selection is staged in shared memory with
# coalesced loads, each thread walks `_BITS_RUN` consecutive columns, ORs the bits of one
# 64-token word in a register and flushes the word into a shared-memory bitmap with one
# `red.shared.or.b64` when the run leaves it; then the CTA writes every word of the row and
# `row_info`. Any column order gives the same words (OR is idempotent and commutative); the
# order only decides how few flushes a run makes, and the selection arrives in token order.
# No memset of the bitmap is needed.
# --------------------------------------------------------------------------------------

_BITS_THREADS = 128
#: Consecutive selected columns a thread walks.
_BITS_RUN = 16


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


class _RowBitsKernel:
    """The bitmap pass's `bits` and `row_info`, one CTA a row."""

    def __init__(self, topk, block_topk, compress_ratio, tokens_per_block, table_width, num_words):
        self.topk, self.block_topk, self.cr = topk, block_topk, compress_ratio
        self.tpb, self.width, self.num_words = tokens_per_block, table_width, num_words
        self.staged = topk + topk // _BITS_RUN + 1

    @cute.jit
    def __call__(
        self,
        sel_t: cute.Tensor,
        qp_ptr: cute.Pointer,
        bits_ptr: cute.Pointer,
        info_ptr: cute.Pointer,
        rows: cutlass.Int32,
        stream,
    ):
        qp = cute.make_tensor(qp_ptr, cute.make_layout((rows,), stride=(1,)))
        bits = cute.make_tensor(
            bits_ptr, cute.make_layout((rows, self.num_words), stride=(self.num_words, 1))
        )
        info = cute.make_tensor(info_ptr, cute.make_layout((rows, 2), stride=(2, 1)))
        # `sel_t` arrives as a tensor: the TVM-FFI entry takes its stream from a GPU tensor.
        self.kernel(sel_t, qp, bits, info).launch(
            grid=[rows, 1, 1],
            block=[_BITS_THREADS, 1, 1],
            smem=self.num_words * 8 + self.staged * 4 + 128,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mSel: cute.Tensor, mQp: cute.Tensor, mBits: cute.Tensor, mInfo: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        words = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((self.num_words,), stride=(1,)), byte_alignment=16
        )
        red = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((4,), stride=(1,)), byte_alignment=16
        )
        # The row's selection, one padding slot every `_BITS_RUN` columns (conflict-free runs).
        sel_s = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.staged,), stride=(1,)), byte_alignment=16
        )
        words_base = cutlass.Int32(words.iterator.toint())
        for i in cutlass.range_constexpr((self.num_words + _BITS_THREADS - 1) // _BITS_THREADS):
            w = tidx + i * _BITS_THREADS
            if w < self.num_words:
                words[w] = cutlass.Int64(0)

        # The columns the per-row kernel reads: the prefill bound rounded up to its tile.
        visible = cutlass.Int32(mQp[row]) + 1
        selected_blocks = cutlass.min(visible // self.cr, cutlass.Int32(self.block_topk))
        valid = selected_blocks * self.cr + visible % self.cr
        bounded = cutlass.min(valid, cutlass.Int32(self.topk))
        columns_read = cutlass.min(
            (bounded + _BOUND_TILE - 1) // _BOUND_TILE * _BOUND_TILE, cutlass.Int32(self.topk)
        )
        columns_read = cutlass.max(columns_read, cutlass.Int32(0))
        for i in cutlass.range_constexpr((self.topk + _BITS_THREADS - 1) // _BITS_THREADS):
            c = tidx + i * _BITS_THREADS
            if c < columns_read:
                sel_s[c + c // _BITS_RUN] = cutlass.Int32(mSel[row, c])
        cute.arch.sync_threads()

        highest = cutlass.Int32(-1)
        runs = (self.topk + _BITS_THREADS * _BITS_RUN - 1) // (_BITS_THREADS * _BITS_RUN)
        for run in cutlass.range_constexpr(runs):
            c0 = (run * _BITS_THREADS + tidx) * _BITS_RUN
            if c0 < columns_read:
                cur_w = cutlass.Int32(-1)
                cur_m = cutlass.Int64(0)
                for k in cutlass.range_constexpr(_BITS_RUN):
                    c = c0 + k
                    t = cutlass.Int32(-1)
                    if c < columns_read:
                        t = sel_s[c + c // _BITS_RUN]
                    if t >= 0:
                        if t // self.tpb < self.width:
                            w = t >> 6
                            if w != cur_w:
                                if cur_w >= 0:
                                    _red_or_shared(words_base + cur_w * 8, cur_m)
                                cur_w = w
                                cur_m = cutlass.Int64(0)
                            cur_m = cur_m | (cutlass.Int64(1) << cutlass.Int64(t & 63))
                            highest = cutlass.max(highest, t)
                if cur_w >= 0:
                    _red_or_shared(words_base + cur_w * 8, cur_m)
        for off in cutlass.range_constexpr(5):
            highest = cutlass.max(highest, cute.arch.shuffle_sync_bfly(highest, offset=1 << off))
        if tidx % 32 == 0:
            red[tidx // 32] = highest
        cute.arch.sync_threads()
        if tidx == 0:
            mInfo[row, 0] = cutlass.max(cutlass.max(red[0], red[1]), cutlass.max(red[2], red[3]))
            mInfo[row, 1] = columns_read
        for i in cutlass.range_constexpr((self.num_words + _BITS_THREADS - 1) // _BITS_THREADS):
            w = tidx + i * _BITS_THREADS
            if w < self.num_words:
                mBits[row, w] = words[w]


_ROW_BITS: dict = {}


def _row_bits(
    selected_tokens, query_positions, compress_ratio, tokens_per_block, table_width, num_words
):
    """`(bits [rows, num_words] int64, row_info [rows, 2] int32)` of the bitmap pass over a
    selection with unit column stride."""
    rows, final_topk = selected_tokens.shape
    block_topk = (final_topk - compress_ratio + 1) // compress_ratio
    device = selected_tokens.device
    row_info = torch.empty((rows, 2), dtype=torch.int32, device=device)
    query_positions = query_positions.contiguous()
    bits = torch.empty((rows, num_words), dtype=torch.int64, device=device)
    key = (
        final_topk, compress_ratio, tokens_per_block, table_width,
        query_positions.dtype, selected_tokens.dtype,
    )  # fmt: skip
    args = (selected_tokens, query_positions.data_ptr(), bits.data_ptr(), row_info.data_ptr())
    fn = _ROW_BITS.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        qty = cutlass.Int64 if query_positions.dtype == torch.int64 else cutlass.Int32
        fn = cute.compile(
            _RowBitsKernel(
                final_topk, block_topk, compress_ratio, tokens_per_block, table_width, num_words
            ),
            from_dlpack(selected_tokens, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            make_ptr(qty, args[1], gm, assumed_align=4),
            make_ptr(cutlass.Int64, args[2], gm, assumed_align=16),
            make_ptr(cutlass.Int32, args[3], gm, assumed_align=8),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _ROW_BITS[key] = fn
    fn(*args, rows)
    return bits, row_info


# --------------------------------------------------------------------------------------
# The plan pass (CuTe DSL). One warp per group of `rows_per_tile` rows, a lane per bitmap
# word: the group's union word (OR of its rows' words), its resident part (tokens on pages
# the table maps inside the pool), a warp scan of the words' token counts for their list
# offsets, the list's slots staged in shared memory and stored by the whole warp, and each
# row's word compressed onto the resident bits (Hacker's Delight `compress`, the masks computed
# once a word for every row) and OR-ed into the row's packed words in shared memory
# (`red.shared.or.b64`), stored once the list is complete. Same plan, slot list, packed words
# and lone list as the Triton kernel below, which serves int64 request indices (and tables
# too long for a CTA's shared memory).
# --------------------------------------------------------------------------------------

#: Groups (one a warp) a plan CTA takes.
_PLAN_GROUPS_PER_CTA = 4
#: Shared memory the CuTe plan may take a CTA; past it (long tables, small groups) the Triton
#: kernel plans.
_PLAN_SMEM_LIMIT = 200 * 1024


def _plan_smem_bytes(rows_per_tile: int, num_words: int) -> int:
    """A plan CTA's shared memory: per warp, its rows' packed words and a step's staged slots."""
    return _PLAN_GROUPS_PER_CTA * (rows_per_tile * num_words * 8 + 32 * 64 * 4) + 256


class _RowtilePlanKernel:
    """The plan pass's slots, packed bitmaps, group plan and lone list, a warp a group."""

    def __init__(self, rows_per_tile, tokens_per_block, table_width, num_words, shared_tile_cost):
        assert 64 % tokens_per_block == 0 or tokens_per_block % 64 == 0
        self.R = rows_per_tile
        self.tpb = tokens_per_block
        self.width = table_width
        self.nw = num_words
        self.cost = shared_tile_cost
        self.warps = _PLAN_GROUPS_PER_CTA
        # Pages a 64-token word spans, and the tokens of the word on each.
        self.ppw = max(1, 64 // tokens_per_block)
        self.seg = min(64, tokens_per_block)

    @cute.jit
    def __call__(
        self,
        req_t: cute.Tensor,
        info_ptr: cute.Pointer,
        bits_ptr: cute.Pointer,
        table_ptr: cute.Pointer,
        slots_ptr: cute.Pointer,
        packed_ptr: cute.Pointer,
        plan_ptr: cute.Pointer,
        lone_rows_ptr: cute.Pointer,
        lone_count_ptr: cute.Pointer,
        rows: cutlass.Int32,
        num_groups: cutlass.Int32,
        num_requests: cutlass.Int32,
        num_cache_pages: cutlass.Int32,
        table_stride: cutlass.Int32,
        slots_stride: cutlass.Int32,
        bits_stride: cutlass.Int32,
        stream,
    ):
        ctas = (num_groups + self.warps - 1) // self.warps
        # `req_t` arrives as a tensor: the TVM-FFI entry takes its stream from a GPU tensor.
        self.kernel(
            req_t,
            info_ptr,
            bits_ptr,
            table_ptr,
            slots_ptr,
            packed_ptr,
            plan_ptr,
            lone_rows_ptr,
            lone_count_ptr,
            rows,
            num_groups,
            num_requests,
            num_cache_pages,
            table_stride,
            slots_stride,
            bits_stride,
        ).launch(
            grid=[ctas, 1, 1],
            block=[32 * self.warps, 1, 1],
            smem=_plan_smem_bytes(self.R, self.nw),
            stream=stream,
        )

    @cute.jit
    def compress_masks(self, m):
        """The six move masks of `compress` for mask `m` (as `_compress_bits`)."""
        one = cutlass.Uint64(1)
        mk = (~m) << one
        mvs = []
        for i in cutlass.range_constexpr(6):
            mp = mk ^ (mk << one)
            mp = mp ^ (mp << cutlass.Uint64(2))
            mp = mp ^ (mp << cutlass.Uint64(4))
            mp = mp ^ (mp << cutlass.Uint64(8))
            mp = mp ^ (mp << cutlass.Uint64(16))
            mp = mp ^ (mp << cutlass.Uint64(32))
            mv = mp & m
            m = (m ^ mv) | (mv >> cutlass.Uint64(1 << i))
            mk = mk & (~mp)
            mvs.append(mv)
        return mvs

    @cute.kernel
    def kernel(
        self,
        mReq: cute.Tensor,
        info_ptr: cute.Pointer,
        bits_ptr: cute.Pointer,
        table_ptr: cute.Pointer,
        slots_ptr: cute.Pointer,
        packed_ptr: cute.Pointer,
        plan_ptr: cute.Pointer,
        lone_rows_ptr: cute.Pointer,
        lone_count_ptr: cute.Pointer,
        rows: cutlass.Int32,
        num_groups: cutlass.Int32,
        num_requests: cutlass.Int32,
        num_cache_pages: cutlass.Int32,
        table_stride: cutlass.Int32,
        slots_stride: cutlass.Int32,
        bits_stride: cutlass.Int32,
    ):
        R = self.R
        NW = self.nw
        I32 = cutlass.Int32
        U64 = cutlass.Uint64
        big = cute.make_layout((1 << 30,))
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        group = bid * self.warps + warp
        mInfo = cute.make_tensor(info_ptr, big)
        mBits = cute.make_tensor(bits_ptr, big)
        mTable = cute.make_tensor(table_ptr, big)
        mSlots = cute.make_tensor(slots_ptr, big)
        mPacked = cute.make_tensor(packed_ptr, big)
        mPlan = cute.make_tensor(plan_ptr, big)
        mLone = cute.make_tensor(lone_rows_ptr, big)
        mLoneCount = cute.make_tensor(lone_count_ptr, cute.make_layout((1,)))

        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(
            U64, cute.make_layout((self.warps * R * NW,), stride=(1,)), byte_alignment=16
        )
        buf_base = I32(buf.iterator.toint())
        # A step's slots (at most 32 words x 64 tokens), staged for coalesced stores.
        sstage = smem.allocate_tensor(
            I32, cute.make_layout((self.warps * 32 * 64,), stride=(1,)), byte_alignment=16
        )
        sbase = warp * (32 * 64)
        wbase = warp * (R * NW)
        for i in cutlass.range(lane, R * NW, 32, unroll=1):
            buf[wbase + i] = U64(0)
        cute.arch.sync_warp()

        if group < num_groups:
            row = group * R + lane
            live = lane < R and row < rows
            req = I32(-1)
            hi_tok = I32(-1)
            own = I32(0)
            if live:
                req = I32(mReq[row])
                hi_tok = mInfo[row * 2]
                own = (mInfo[row * 2 + 1] + 63) // 64
            lowest = req if live else num_requests
            highest = req
            for o in cutlass.range_constexpr(5):
                lowest = cutlass.min(lowest, cute.arch.shuffle_sync_bfly(lowest, offset=1 << o))
                highest = cutlass.max(highest, cute.arch.shuffle_sync_bfly(highest, offset=1 << o))
                hi_tok = cutlass.max(hi_tok, cute.arch.shuffle_sync_bfly(hi_tok, offset=1 << o))
                own = own + cute.arch.shuffle_sync_bfly(own, offset=1 << o)
            one_request = lowest == highest and lowest >= 0 and lowest < num_requests
            words_needed = I32(0)
            if one_request:
                words_needed = (hi_tok + 64) // 64
            req_c = cutlass.max(cutlass.min(lowest, num_requests - 1), I32(0))
            listed = I32(0)
            for base in cutlass.range(0, words_needed, 32, unroll=1):
                w = base + lane
                inr = w < words_needed
                rw = []
                union = U64(0)
                for r in cutlass.range_constexpr(R):
                    x = U64(0)
                    if inr and group * R + r < rows:
                        x = U64(mBits[(group * R + r) * bits_stride + w])
                    rw.append(x)
                    union = union | x
                # Resident tokens: the word's pages mapped inside the pool.
                resident = U64(0)
                pages = []
                for pp in cutlass.range_constexpr(self.ppw):
                    lp = (w * 64 + pp * self.seg) // self.tpb
                    pid = I32(-1)
                    if inr and lp < self.width:
                        pid = mTable[req_c * table_stride + lp]
                    pages.append(pid)
                    seg_bits = (
                        U64(0xFFFFFFFFFFFFFFFF)
                        if self.seg == 64
                        else U64(((1 << self.seg) - 1) << (pp * self.seg))
                    )
                    if pid >= 0 and pid < num_cache_pages:
                        resident = resident | seg_bits
                resident = resident & union
                count = I32(cute.arch.popc(resident))
                incl = count
                for o in cutlass.range_constexpr(5):
                    up = cute.arch.shuffle_sync_up(incl, offset=1 << o)
                    if lane >= (1 << o):
                        incl = incl + up
                total = cute.arch.shuffle_sync(incl, 31)
                first = listed + incl - count
                # The list: resident tokens in logical order, as physical cache slots, staged in
                # shared memory and stored by the whole warp.
                pos = incl - count
                for b in cutlass.range_constexpr(64):
                    if (resident >> U64(b)) & U64(1) != U64(0):
                        if cutlass.const_expr(self.tpb <= 64):
                            tok = b % self.tpb
                        else:
                            tok = (w * 64) % self.tpb + b
                        sstage[sbase + pos] = pages[b // self.seg] * self.tpb + tok
                        pos = pos + 1
                cute.arch.sync_warp()
                slot_row = group * slots_stride + listed
                for k in cutlass.range(lane, total, 32, unroll=1):
                    mSlots[slot_row + k] = sstage[sbase + k]
                cute.arch.sync_warp()
                # Each row's word compressed onto the resident bits, at the list offset.
                mvs = self.compress_masks(resident)
                d0 = first // 64
                sh = U64(first % 64)
                spills = (first % 64) + count > 64
                if count > 0:
                    for r in cutlass.range_constexpr(R):
                        x = rw[r] & resident
                        for i in cutlass.range_constexpr(6):
                            moved = x & mvs[i]
                            x = (x ^ moved) | (moved >> U64(1 << i))
                        addr = buf_base + (wbase + r * NW + d0) * 8
                        _red_or_shared(addr, cutlass.Int64(x << sh))
                        if spills:
                            _red_or_shared(
                                addr + 8,
                                cutlass.Int64((x >> U64(1)) >> (U64(63) - sh)),
                            )
                listed = listed + total
            cute.arch.sync_warp()
            tiles = (listed + 63) // 64
            shared = one_request and tiles * self.cost <= own * 4
            if lane == 0:
                mPlan[group * 2] = tiles if shared else I32(-1)
                mPlan[group * 2 + 1] = listed
            for r in cutlass.range_constexpr(R):
                rr = group * R + r
                if rr < rows:
                    for k in cutlass.range(lane, tiles, 32, unroll=1):
                        mPacked[rr * bits_stride + k] = buf[wbase + r * NW + k]
            if not shared:
                n_live = I32(1) if live else I32(0)
                for o in cutlass.range_constexpr(5):
                    n_live = n_live + cute.arch.shuffle_sync_bfly(n_live, offset=1 << o)
                start = I32(0)
                if lane == 0:
                    start = cute.arch.atomic_add(mLoneCount.iterator, n_live)
                start = cute.arch.shuffle_sync(start, 0)
                if live:
                    mLone[start + lane] = row


_ROW_PLAN: dict = {}


def _cute_plan_supported(request_indices, tokens_per_block, rows_per_tile, num_words) -> bool:
    """Whether `_RowtilePlanKernel` plans this launch: int32 request indices, pages that tile a
    64-token word (or whole words), and a CTA's packed words within `_PLAN_SMEM_LIMIT`."""
    return (
        request_indices.dtype == torch.int32
        and (64 % tokens_per_block == 0 or tokens_per_block % 64 == 0)
        and _plan_smem_bytes(rows_per_tile, num_words) <= _PLAN_SMEM_LIMIT
    )


def _rowtile_plan_cute(
    request_indices, row_info, bits, block_table, slots, packed_bits, group_plan, lone_rows,
    lone_count, rows, num_groups, *, rows_per_tile, tokens_per_block, num_cache_pages,
):  # fmt: skip
    """The plan pass on `_RowtilePlanKernel` (int32 request indices). Writes each row's packed
    words below its group's tile count only."""
    table_width = block_table.shape[1]
    num_words = bits.shape[1]
    key = (rows_per_tile, tokens_per_block, table_width, num_words)
    fn = _ROW_PLAN.get(key)
    ptrs = (row_info, bits, block_table, slots, packed_bits, group_plan, lone_rows, lone_count)
    args = (request_indices, *(t.data_ptr() for t in ptrs))
    if fn is None:
        gm = cute.AddressSpace.gmem
        dtypes = (cutlass.Int32, cutlass.Uint64, cutlass.Int32, cutlass.Int32, cutlass.Uint64,
                  cutlass.Int32, cutlass.Int32, cutlass.Int32)  # fmt: skip
        aligns = (8, 16, 4, 4, 16, 4, 4, 4)
        fn = cute.compile(
            _RowtilePlanKernel(
                rows_per_tile, tokens_per_block, table_width, num_words, _SHARED_TILE_COST
            ),
            from_dlpack(request_indices, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            *(make_ptr(dt, p, gm, assumed_align=al) for dt, p, al in zip(dtypes, args[1:], aligns)),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _ROW_PLAN[key] = fn
    fn(
        *args, rows, num_groups, block_table.shape[0], num_cache_pages, block_table.stride(0),
        slots.stride(0), bits.stride(0),
    )  # fmt: skip


@triton.jit
def _or_combine(a, b):
    return a | b


@triton.jit
def _compress_bits(x, m):
    """Per word, the bits of `x` at the set positions of `m`, packed to the low
    end in order (a parallel bit extract; Hacker's Delight `compress`, 64-bit).
    `x` and `m` are uint64 and broadcast together."""
    x = x & m
    mk = ~m << 1
    for i in tl.static_range(6):
        mp = mk ^ (mk << 1)
        mp = mp ^ (mp << 2)
        mp = mp ^ (mp << 4)
        mp = mp ^ (mp << 8)
        mp = mp ^ (mp << 16)
        mp = mp ^ (mp << 32)
        mv = mp & m
        m = (m ^ mv) | (mv >> (1 << i))
        t = x & mv
        x = (x ^ t) | (t >> (1 << i))
        mk = mk & ~mp
    return x


@triton.jit
def _rowtile_plan_kernel_triton(
    request_indices, row_info, bits, block_table, slots, packed_bits, group_plan,
    lone_rows, lone_count, rows,
    bits_stride_row, block_table_stride_request, block_table_stride_page, slots_stride_group,
    NUM_REQUESTS, NUM_CACHE_PAGES,
    ROWS_PER_TILE: tl.constexpr, ROWS_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr, SHARED_TILE_COST: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr, TOKENS_PER_BLOCK: tl.constexpr, PLAN_WORDS: tl.constexpr,
):  # fmt: skip
    """Per group of rows of one request: list the union of its rows' resident
    selected tokens, in logical order, as physical cache slots
    (`page * tokens_per_block + offset`) in `slots[group]`, and write each
    row's bitmap re-indexed onto that list into `packed_bits[row]` (bit `j`
    set when the row selects the list's `j`-th token). `group_plan[group]` is
    (tiles of the list for the row-tile pass, or -1; list length). A group the
    row-tile pass does not take -- rows of several requests, or a list whose
    tiles cost more than the rows' own columns -- appends its rows to the lone
    list instead."""
    group = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, ROWS_POW2)
    members = group * ROWS_PER_TILE + offsets
    live = (offsets < ROWS_PER_TILE) & (members < rows)
    requests = tl.load(request_indices + members, mask=live, other=-1).to(tl.int64)
    lowest = tl.min(tl.where(live, requests, NUM_REQUESTS), axis=0)
    highest_request = tl.max(tl.where(live, requests, -1), axis=0)
    highest_token = tl.max(tl.load(row_info + members * 2, mask=live, other=-1), axis=0)
    columns = tl.load(row_info + members * 2 + 1, mask=live, other=0)
    own_tiles = tl.sum(tl.cdiv(columns, BLOCK_N), axis=0)
    one_request = (lowest == highest_request) & (lowest >= 0) & (lowest < NUM_REQUESTS)
    words_needed = tl.where(one_request, (highest_token + 64) // 64, 0)
    page_base = (
        block_table
        + tl.minimum(tl.maximum(lowest, 0), NUM_REQUESTS - 1) * block_table_stride_request
    )
    word_offsets = tl.arange(0, PLAN_WORDS)
    lanes = tl.arange(0, 64)
    lane_bits = tl.full([64], 1, tl.uint64) << lanes.to(tl.uint64)
    listed = tl.sum(tl.zeros([ROWS_POW2], tl.int32), axis=0)
    for start in range(0, words_needed, PLAN_WORDS):
        words = start + word_offsets
        in_range = words < words_needed
        row_words = tl.load(
            bits + members[:, None] * bits_stride_row + words[None, :],
            mask=live[:, None] & in_range[None, :],
            other=0,
        ).to(tl.uint64, bitcast=True)  # fmt: skip
        selected = tl.reduce(row_words, 0, _or_combine)
        chosen = ((selected[:, None] >> lanes[None, :].to(tl.uint64)) & 1) != 0
        tokens = words[:, None].to(tl.int32) * 64 + lanes[None, :]
        logical_pages = tokens // TOKENS_PER_BLOCK
        pages = tl.load(
            page_base + tl.minimum(logical_pages, PAGE_TABLE_WIDTH - 1) * block_table_stride_page,
            mask=chosen & (logical_pages < PAGE_TABLE_WIDTH),
            other=-1,
        )  # fmt: skip
        kept = chosen & (pages >= 0) & (pages < NUM_CACHE_PAGES)
        kept_count = kept.to(tl.int32)
        count = tl.sum(kept_count, axis=1)
        first = listed + tl.cumsum(count, axis=0) - count
        rank = first[:, None] + tl.cumsum(kept_count, axis=1) - kept_count
        tl.store(
            slots + group * slots_stride_group + rank,
            pages * TOKENS_PER_BLOCK + tokens % TOKENS_PER_BLOCK,
            mask=kept,
        )  # fmt: skip
        resident = tl.reduce(tl.where(kept, lane_bits[None, :], 0), 1, _or_combine)
        packed = _compress_bits(row_words, resident[None, :])
        shift = (first % 64).to(tl.uint64)
        destination = packed_bits + members[:, None] * bits_stride_row + (first // 64)[None, :]
        writes = live[:, None] & in_range[None, :] & (packed != 0)
        tl.atomic_or(destination, (packed << shift[None, :]).to(tl.int64, bitcast=True),
                     mask=writes, sem="relaxed")  # fmt: skip
        # Bits past the destination word spill into the next one (never for shift 0).
        spill = (packed >> 1) >> (63 - shift[None, :])
        tl.atomic_or(destination + 1, spill.to(tl.int64, bitcast=True),
                     mask=writes & ((first % 64) + count > 64)[None, :], sem="relaxed")  # fmt: skip
        listed += tl.sum(count, axis=0)
    tiles = tl.cdiv(listed, BLOCK_N)
    shared = one_request & (tiles * SHARED_TILE_COST <= own_tiles * 4)
    tl.store(group_plan + group * 2, tl.where(shared, tiles, -1))
    tl.store(group_plan + group * 2 + 1, listed)
    if not shared:
        base = tl.atomic_add(lone_count, tl.sum(live.to(tl.int32), axis=0))
        tl.store(lone_rows + base + offsets, members.to(tl.int32), mask=live)


def _rowtile_plan(
    grid, request_indices, row_info, bits, block_table, slots, packed_bits, group_plan,
    lone_rows, lone_count, rows, *, rows_per_tile, tokens_per_block, num_cache_pages,
):  # fmt: skip
    """The plan pass over `grid = (groups,)`: `_RowtilePlanKernel` for int32 request indices
    (packed words written below each group's tile count), else the Triton kernel (packed words
    OR-ed into a zeroed `packed_bits`)."""
    (num_groups,) = grid
    if _cute_plan_supported(request_indices, tokens_per_block, rows_per_tile, bits.shape[1]):
        _rowtile_plan_cute(
            request_indices, row_info, bits, block_table, slots, packed_bits, group_plan,
            lone_rows, lone_count, rows, num_groups, rows_per_tile=rows_per_tile,
            tokens_per_block=tokens_per_block, num_cache_pages=num_cache_pages,
        )  # fmt: skip
        return
    table_width = block_table.shape[1]
    _rowtile_plan_kernel_triton[(num_groups,)](
        request_indices, row_info, bits, block_table, slots, packed_bits, group_plan,
        lone_rows, lone_count, rows,
        bits.stride(0), block_table.stride(0), block_table.stride(1), slots.stride(0),
        ROWS_PER_TILE=rows_per_tile, NUM_REQUESTS=block_table.shape[0],
        ROWS_POW2=triton.next_power_of_2(rows_per_tile), BLOCK_N=BLOCK_N,
        SHARED_TILE_COST=_SHARED_TILE_COST, NUM_CACHE_PAGES=num_cache_pages,
        PAGE_TABLE_WIDTH=table_width, TOKENS_PER_BLOCK=tokens_per_block,
        PLAN_WORDS=_PLAN_WORDS, num_warps=_PLAN_WARPS,
    )  # fmt: skip


class _PlanLauncher:
    """`_rowtile_plan_kernel[(groups,)](...)`: grid-indexed like the entry's other launches."""

    def __getitem__(self, grid):
        return functools.partial(_rowtile_plan, tuple(grid))


_rowtile_plan_kernel = _PlanLauncher()


@triton.jit
def _rowtile_lone_kernel(
    q, k_cache, v_cache, block_table, selected_tokens, request_indices, row_info,
    lone_rows, lone_count, output, softmax_scale, rows,
    q_stride_row: tl.constexpr, q_stride_head: tl.constexpr, q_stride_dim: tl.constexpr,
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
    GROUP_SIZE: tl.constexpr, TOKENS_PER_BLOCK: tl.constexpr, ROWS_PER_TILE: tl.constexpr,
    ROWS_POW2: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, USE_PDL: tl.constexpr,
    LONE_STAGES: tl.constexpr,
):  # fmt: skip
    """Rows of groups the row-tile pass does not serve: each its own columns, gathered."""
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
    """Bitmap pass, plan pass, lone-row pass, row-tile pass (`qsa_rowtile_attention_cute`);
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
    num_words = triton.cdiv(table_width * tokens_per_block, 64)
    bits, row_info = _row_bits(
        selected_tokens, query_positions, compress_ratio, tokens_per_block, table_width, num_words
    )
    rows_per_tile = BLOCK_M // group_size
    rows_pow2 = triton.next_power_of_2(rows_per_tile)
    kv_strides = (
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
    )  # fmt: skip
    q_strides = (q.stride(0), q.stride(1), q.stride(2))
    common = (*q_strides, *kv_strides, block_table.stride(0), block_table.stride(1))
    geometry = dict(
        NUM_CACHE_PAGES=k_cache.shape[0], NUM_REQUESTS=block_table.shape[0],
        PAGE_TABLE_WIDTH=table_width, GROUP_SIZE=group_size,
        TOKENS_PER_BLOCK=tokens_per_block, ROWS_PER_TILE=rows_per_tile, ROWS_POW2=rows_pow2,
        HEAD_DIM=head_dim,
    )  # fmt: skip
    out_strides = (output.stride(0), output.stride(1), output.stride(2))
    num_groups = triton.cdiv(rows, rows_per_tile)
    slots = torch.empty((num_groups, num_words * 64), dtype=torch.int32, device=q.device)
    # The CuTe plan writes every packed word the row-tile pass reads; the Triton one ORs.
    on_cute = _cute_plan_supported(request_indices, tokens_per_block, rows_per_tile, num_words)
    alloc = torch.empty if on_cute else torch.zeros
    packed_bits = alloc((rows, num_words), dtype=torch.int64, device=q.device)
    group_plan = torch.empty((num_groups, 2), dtype=torch.int32, device=q.device)
    lone_rows = torch.empty((rows,), dtype=torch.int32, device=q.device)
    # [0]: the lone list's length; [1]: the row-tile pass's work-ticket counter.
    counters = torch.zeros((2,), dtype=torch.int32, device=q.device)
    lone_count = counters[:1]
    _rowtile_plan_kernel[(num_groups,)](
        request_indices, row_info, bits, block_table, slots, packed_bits, group_plan,
        lone_rows, lone_count, rows, rows_per_tile=rows_per_tile,
        tokens_per_block=tokens_per_block, num_cache_pages=k_cache.shape[0],
    )  # fmt: skip
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
    # K and V are the two role planes of one [pages, 2, kv, tpb, D] pool.
    cache_layout="pool_planes",
    # q is the first 24 heads of the packed [T, 24 + 2 + 2, 256] QKV buffer.
    q_layout="qkv_slice",
    request_dtype=torch.int32,
    # The engine's table at max_seq_len 10240: 320 pages per slot, -1 past each sequence.
    pages=320,
)


def _spec(segments, slots=32, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CHECKPOINT, "segments": tuple(segments), "slots": slots, **overrides}


def _decodes(count, rows_each=1, first=8200):
    return [(first + (37 * g) % 1000, rows_each) for g in range(count)]


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

    Four launches. A bitmap pass: one bit per selected logical token per row.
    A plan pass per group of 10 consecutive rows of one request: the union of
    the rows' selected tokens whose page is mapped, listed in logical order
    as physical cache slots, and each row's bitmap re-indexed onto that list
    (a per-word parallel bit extract). The group goes to the row-tile pass
    when walking its list in 64-slot tiles costs at most 2.5x per-row tiles
    less than the rows' own columns would; otherwise -- or when its rows span
    requests -- its rows join the lone list. A lone-row pass (256 programs
    per KV head striding the list) serves listed rows over their own columns,
    as the per-row kernel does. The row-tile pass is persistent (one program
    per SM drawing (group, KV head) items from a ticket counter, the plan's
    last groups first); per item it loads each listed tile's K/V once for the
    120 (row, head) pairs and masks each row to its own re-indexed bits, and
    one item's epilogue overlaps the next item's first MMAs. It is launched
    programmatically dependent on the lone pass, whose rows are disjoint, so
    the two overlap. That pass is `qsa_rowtile_attention_cute` (tcgen05 MMAs,
    S / P / O in tensor memory, a cp.async gather of the K/V rows, a TMA store
    of O), so the entry serves SM103, bf16 q and K/V and D = 256.

    Fusion boundary: as `qsa_paged_sparse_gqa` -- the block-table gather,
    scale, online softmax and P.V are inside; deciding what to attend to,
    appending K/V and q's projection, norm, RoPE and gate are not.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="con 1 prefill: 8192 rows of one request from 0, across the budget",
            spec=_spec([(0, 8192)]),
        ),
        Cell(
            why="sparse union: 4096 rows from 4096 ranking one score vector",
            spec=_spec([(4096, 4096)], shared_scores=True),
        ),
        Cell(
            why="con 128 mixed: 3 chunks (16240 rows) + 52 decodes on the lone pass",
            spec=_spec([(0, 8192), (2192, 6000), (6144, 2048)] + _decodes(52), slots=128),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument, the same arguments
    `qsa_paged_sparse_gqa` takes, with the prefill pair required. What the
    target passes: `qsa_block_table` (physical page ids), the expansion's
    `-1`-padded logical selections, the layer pool's K/V role planes, and the
    logical positions with the checkpoint's compress ratio.

    Preconditions (neither is checkable without reading the contents):
      - Within the bound, a row names each token at most once. The row-tile
        pass attends to the *set* of tokens a row names; the per-row kernel
        attends to every column, so a repeated index would weigh twice there
        and once here. QSA's top-k + expansion never repeats one (distinct
        blocks, and a tail outside every complete block): measured zero
        repeats on four layers of the target's own 8192-token prefill.
        Rows served by the lone-row pass keep per-column semantics.
      - As for the per-row kernel: each row is prefix-packed with exactly the
        bound's count of valid entries; entries past the 64-rounded bound
        are ignored.

    Work: a row-tile group walks its packed list, so its MMA work grows
    with the union of its rows' selections rather than with the group's
    highest position. On the target's own selections (8192-token prefill,
    layers 0/3/6/11) that union is 13-32% shorter than every 64-token tile up
    to the highest token, and the walk loads ~5x fewer K/V tiles than the
    per-row kernel. Measured on GB300 (eager, whole call) at 8192 context
    rows of real selections: 1544-1565 us on layers 0-6 and 1274 us on layer
    11, against 1711-1715 us for a walk of every tile up to the highest token
    and 2953 us for `qsa_paged_sparse_gqa`. Uniformly random selections
    leave almost no gap in the union, and the call costs what the full walk
    does (16292-row mixed step: 3859 vs 3836 us). Below a few hundred rows
    it is slower than the per-row kernel (generation-only launches 1.4-2x);
    the target's served context launches are thousands of rows. The tensor-core
    row-tile pass alone, on six dumped 16384-row context launches of the
    target (GB300, layers 0-5 of one step): 581-775 us, against 694-886 us for
    the one-program-per-item pass it replaced (the same output, bit for bit).

    Compiled once per (group size, rows per tile, tokens per block, PDL):
    nothing in the row-tile pass's compile key depends on the row count or
    the plan.

    Scratch per call: the bitmap and the re-indexed bitmap, each `rows x
    ceil(table_width * tpb / 64)` int64 words (21 MB at 16384 rows and 320
    pages), the slot lists, `ceil(rows / 10) x` that many words' tokens as
    int32 (67 MB at 16384 rows), the plan and the lone list. Calls allocate,
    so capture in a CUDA graph is not certified; this entry serves the
    prefill path, which runs eager. Results are deterministic: the bitmap
    OR, the re-indexed bitmap OR and the lone list order do not change any
    row's arithmetic.

    Not certified: int64 request indices or positions; rows whose selection
    resolves no token (all padding, a foreign request, unmapped pages);
    `rows == 0`.
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
        """The reference `qsa_paged_sparse_gqa` certifies against: fp32 sparse
        GQA over every resident selected token, rounded once. The prefill pair
        is ignored, as there: on a prefix-packed selection the bound reads
        exactly the valid entries."""
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
