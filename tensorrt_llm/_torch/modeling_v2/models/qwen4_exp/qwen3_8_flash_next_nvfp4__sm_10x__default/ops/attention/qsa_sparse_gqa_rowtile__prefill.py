# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse GQA for context batches: several query rows per program, K/V read once per tile.

A plan launch forms, per group of consecutive rows of one request, the union of
its rows' resident selected tokens (the columns the prefill bound lets each row
read) as a packed list of physical cache slots, and each row's selection
re-indexed onto that list as bits (the group plan, one CTA a group holding the
union in shared memory; where it does not fit, a bitmap pass over the table's
logical tokens and a plan pass over those bitmaps). The row-tile launch is persistent: one
program per SM takes (group, KV head) items longest first and walks each packed
list in 64-slot tiles, each tile's K/V loaded once for every row and head of the
group, each row masked to its own bits. A group whose rows span requests
(generation rows of a mixed batch) walks each row's own selected columns
instead, as the per-row kernel does.

A row group is sized to whole 8-row swizzle atoms (10 rows of 12 heads at a dep4 rank, 20 of 6
at a tp4 / tep4 rank); scratch is bounded by the group's selections rather than the table's
width.
"""

from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr
from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.archs import SERVED_ARCHS
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa as _per_row_entry,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (  # noqa: E501
    BLOCK_M,
    BLOCK_N,
    ROW_ATOM,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (  # noqa: E501
    eligible as _cute_eligible,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (  # noqa: E501
    num_ctas as _rowtile_ctas,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_rowtile_attention_cute import (  # noqa: E501
    rowtile_attention_cute as _rowtile_attention_cute,
)

_INTEGER_DTYPES = (torch.int32, torch.int64)
# The per-row kernel's scan tile: the prefill bound is rounded up to it, so the
# bitmap reads exactly the columns that kernel reads.
_BOUND_TILE = 64
# Programs of the lone-row pass per KV head; each walks the lone list with this stride.
_LONE_PROGRAMS = 256  # capacity: fixed: a launch width; the lone list may be any length
# Bitmap words the plan pass unions and packs per loop step: wider for the 10-row groups of a
# dep4 rank, narrower for the 20-row groups of a tp4 / tep4 rank.
_PLAN_WORDS = {16: 64, 32: 16}  # capacity: fixed: a loop step; the plan loops over every word
# A one-request group takes the row-tile pass when its list's 64-slot tiles, each costing
# this many quarters of a per-row tile (1.5x), cost no more than its rows' own columns.
_SHARED_TILE_COST = 6
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
# The group plan (CuTe DSL): the bitmap pass and the plan pass in one launch, one CTA a row
# group, 128 threads. The entry takes it wherever it fits (`_group_plan_fits`); a launch it does
# not fit takes the two passes below it (`_row_bits`, `_rowtile_plan_kernel`).
#
# The two passes it replaces write, per row, a bitmap over every logical token of the block
# table (4034 words a row at an 8068-page table: 0.53 GB a 16,384-row launch) and then, per
# group, walk every 64-token word up to the group's highest token (64 words x 64 lanes a loop
# step): work that follows the request's history, 1.1 ms a layer at 90k tokens of it (GB300,
# a 16384-row launch: 13.3 ms a 12-layer step per 100k tokens of one request's history, against
# 0.04 for the group plan; 14.7 -> 3.4 ms a step at 16384 rows over 4 x 87.5k tokens). Here a
# CTA holds its group's union bitmap in shared memory (8 B a word of the table) and
#   stage   a row's selection is staged in shared memory with coalesced loads, as the bitmap
#           pass stages it (a warp's load is 128 contiguous bytes; a thread issues its loads
#           before it stores any), a row ahead of the row being walked (two buffers);
#   pass 1  walks the group's selected columns (16 consecutive columns a thread a row, as the
#           bitmap pass does), ORs each readable token into the union (one
#           `red.shared.or.b64` a run of columns in one word) and reduces each row's highest
#           readable token (`row_info`, exactly the bitmap pass's);
#   prefix  clears the tokens of pages that are not resident (a word of the union spans
#           64 / tokens_per_block table pages: their entries are loaded a word, not a token)
#           and counts the union's bits before each word (a shuffle scan a warp);
#   list    writes each set bit's physical slot `page * tokens_per_block + offset` at its rank
#           (the page entries are still in the thread's registers);
#   pass 2  walks the columns again and sets, per row, the bit at each selected token's rank
#           (`packed_bits`, staged in shared memory a row, written out while the next row
#           is walked);
#   plan    `group_plan[group] = (tiles or -1, listed)` by the plan pass's rule, and the rows of
#           a group the row-tile pass does not take appended to the lone list.
# What it writes is what the two passes write: `slots[group, :listed]`, `packed_bits` (words
# past a group's list stay zero), `group_plan`, `row_info`, the lone count and the lone rows
# (their order in the list is the launch's, as the plan pass's is). A group whose rows span
# requests lists nothing and joins the lone list, as the plan pass does. Both walks take four
# columns at a time where they hold four consecutive tokens of one 4-aligned block (the
# expansion's shape) and a column at a time elsewhere, so nothing about the selection's shape
# is assumed: any int32 selection the two passes take gives the same outputs.
# --------------------------------------------------------------------------------------

_GP_THREADS = 128
#: Consecutive selected columns a thread walks.
_GP_RUN = 16
#: The CTA's shared memory: the union bitmap, the prefix counts and one row's re-indexed bits.
_GP_MAX_SMEM = 200 << 10  # capacity: fixed: a launch above it takes the two passes
_GP_MAX_ROWS = 32  # capacity: fixed: rows a group (10 at a dep4 rank, 20 at a tp4 / tep4 one)


@dsl_user_op
def _red_max_shared(addr, val, *, loc=None, ip=None):
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


class _GroupPlanKernel:
    """`slots`, `packed_bits`, `group_plan`, `row_info` and the lone list, one CTA a group."""

    def __init__(
        self, topk, compress_ratio, tokens_per_block, table_width, rows_per_tile, packed_words
    ):  # fmt: skip
        self.topk, self.cr, self.tpb = topk, compress_ratio, tokens_per_block
        self.width = table_width
        self.block_topk = (topk - compress_ratio + 1) // compress_ratio
        self.num_words = (table_width * tokens_per_block + 63) // 64
        self.rpt, self.packed_words = rows_per_tile, packed_words
        self.chunk = (self.num_words + _GP_THREADS - 1) // _GP_THREADS
        self.pk_chunk = (packed_words + _GP_THREADS - 1) // _GP_THREADS
        self.runs = (topk + _GP_THREADS * _GP_RUN - 1) // (_GP_THREADS * _GP_RUN)
        # A 64-token word of the union spans `ppw` table pages; page p of a word holds the
        # bits `page_masks[p]` (as a signed 64-bit constant).
        assert 64 % tokens_per_block == 0
        self.ppw = 64 // tokens_per_block
        masks = [((1 << tokens_per_block) - 1) << (p * tokens_per_block) for p in range(self.ppw)]
        self.page_masks = [m - (1 << 64) if m >= (1 << 63) else m for m in masks]
        #: A staged row: its columns with one padding slot every `_GP_RUN` (conflict-free runs).
        self.staged = topk + topk // _GP_RUN + 1
        self.stage_loads = (topk + _GP_THREADS - 1) // _GP_THREADS
        self.smem_bytes = self.smem(
            topk, table_width, tokens_per_block, rows_per_tile, packed_words
        )

    @staticmethod
    def smem(topk, table_width, tokens_per_block, rows_per_tile, packed_words) -> int:
        """Shared memory of a CTA: the union (8 B a word) and its prefix counts (4 B), two
        staged rows, two rows of re-indexed bits, the per-row scalars."""
        num_words = (table_width * tokens_per_block + 63) // 64
        staged = topk + topk // _GP_RUN + 1
        return (
            num_words * 12 + 2 * staged * 4 + 2 * packed_words * 8
            + (3 * rows_per_tile + _GP_THREADS // 32) * 4 + 256
        )  # fmt: skip

    @cute.jit
    def __call__(
        self,
        sel: cute.Pointer, qp: cute.Pointer, req: cute.Pointer, table: cute.Pointer,
        slots: cute.Pointer, packed: cute.Pointer, plan: cute.Pointer, info: cute.Pointer,
        lone: cute.Pointer, lone_count: cute.Pointer,
        rows: cutlass.Int32, num_groups: cutlass.Int32, num_requests: cutlass.Int32,
        num_pages: cutlass.Int32, sel_stride: cutlass.Int32,
        table_stride_request: cutlass.Int32, table_stride_page: cutlass.Int32,
        slots_stride: cutlass.Int32, packed_stride: cutlass.Int32,
        stream,
    ):  # fmt: skip
        self.kernel(
            sel, qp, req, table, slots, packed, plan, info, lone, lone_count,
            rows, num_requests, num_pages, sel_stride, table_stride_request, table_stride_page,
            slots_stride, packed_stride,
        ).launch(
            grid=[num_groups, 1, 1],
            block=[_GP_THREADS, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
        )  # fmt: skip

    @cute.jit
    def columns_read(self, position: cutlass.Int32):
        """The columns the per-row kernel reads: the prefill bound rounded up to its tile (the
        bitmap pass's `columns_read`)."""
        visible = position + 1
        selected_blocks = cutlass.min(visible // self.cr, cutlass.Int32(self.block_topk))
        valid = selected_blocks * self.cr + visible % self.cr
        bounded = cutlass.min(valid, cutlass.Int32(self.topk))
        columns = cutlass.min(
            (bounded + _BOUND_TILE - 1) // _BOUND_TILE * _BOUND_TILE, cutlass.Int32(self.topk)
        )
        return cutlass.max(columns, cutlass.Int32(0))

    @cute.jit
    def stage(
        self, sel: cute.Pointer, sel_stride: cutlass.Int32, row: cutlass.Int32,
        tidx: cutlass.Int32, sel_s, at: cutlass.Int32,
    ):  # fmt: skip
        """A row's selection into shared memory at `at`, one padding slot every `_GP_RUN`
        columns: thread t loads columns t, t + 128, .. (a warp's load is 128 contiguous
        bytes), all of them before it stores any (the loads are independent)."""
        mSel = cute.make_tensor(
            sel + cutlass.Int64(row) * cutlass.Int64(sel_stride), cute.make_layout((1 << 30,))
        )
        v = cute.make_rmem_tensor((self.stage_loads,), cutlass.Int32)
        for i in cutlass.range_constexpr(self.stage_loads):
            v[i] = mSel[cutlass.min(tidx + i * _GP_THREADS, cutlass.Int32(self.topk - 1))]
        for i in cutlass.range_constexpr(self.stage_loads):
            c = tidx + i * _GP_THREADS
            if c < self.topk:
                sel_s[at + c + c // _GP_RUN] = v[i]

    @cute.kernel
    def kernel(
        self,
        sel: cute.Pointer, qp: cute.Pointer, req: cute.Pointer, table: cute.Pointer,
        slots: cute.Pointer, packed: cute.Pointer, plan: cute.Pointer, info: cute.Pointer,
        lone: cute.Pointer, lone_count: cute.Pointer,
        rows: cutlass.Int32, num_requests: cutlass.Int32, num_pages: cutlass.Int32,
        sel_stride: cutlass.Int32, table_stride_request: cutlass.Int32,
        table_stride_page: cutlass.Int32, slots_stride: cutlass.Int32,
        packed_stride: cutlass.Int32,
    ):  # fmt: skip
        Int32, Int64 = cutlass.Int32, cutlass.Int64
        NT = _GP_THREADS
        R = self.rpt
        tidx, _, _ = cute.arch.thread_idx()
        group, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        words = smem.allocate_tensor(
            Int64, cute.make_layout((self.num_words,), stride=(1,)), byte_alignment=16
        )
        # Two rows' re-indexed bits and two staged rows: row j uses buffer j % 2, so the next
        # row is staged (and the previous row's bits written out) while this one is walked.
        pk = smem.allocate_tensor(
            Int64, cute.make_layout((2 * self.packed_words,), stride=(1,)), byte_alignment=16
        )
        pref = smem.allocate_tensor(
            Int32, cute.make_layout((self.num_words,), stride=(1,)), byte_alignment=16
        )
        sel_s = smem.allocate_tensor(
            Int32, cute.make_layout((2 * self.staged,), stride=(1,)), byte_alignment=16
        )
        hi_s = smem.allocate_tensor(Int32, cute.make_layout((R,), stride=(1,)), byte_alignment=16)
        # Per row of the group: its request index and the columns it reads (0 past the launch).
        rq_s = smem.allocate_tensor(Int32, cute.make_layout((R,), stride=(1,)), byte_alignment=16)
        cols_s = smem.allocate_tensor(Int32, cute.make_layout((R,), stride=(1,)), byte_alignment=16)
        wt = smem.allocate_tensor(
            Int32, cute.make_layout((NT // 32,), stride=(1,)), byte_alignment=16
        )
        words_base = Int32(words.iterator.toint())
        pk_base = Int32(pk.iterator.toint())
        hi_base = Int32(hi_s.iterator.toint())
        mQp = cute.make_tensor(qp, cute.make_layout((1 << 30,)))
        mReq = cute.make_tensor(req, cute.make_layout((1 << 30,)))
        first = group * R
        last_row = rows - 1

        # The union starts empty; the rows' requests and columns; the first row staged.
        for i in cutlass.range_constexpr(self.chunk):
            w = tidx + i * NT
            if w < self.num_words:
                words[w] = Int64(0)
        if tidx < R:
            hi_s[tidx] = Int32(-1)
            rq_s[tidx] = Int32(0)
            cols_s[tidx] = Int32(0)
            if first + tidx < rows:
                rq_s[tidx] = Int32(mReq[first + tidx])
                cols_s[tidx] = self.columns_read(Int32(mQp[first + tidx]))
        self.stage(sel, sel_stride, cutlass.min(first, last_row), tidx, sel_s, Int32(0))
        cute.arch.sync_threads()
        # The group's request, as the plan pass decides it: one request, inside the table.
        lowest = Int32(0x7FFFFFFF)
        highest_request = Int32(-1)
        live_rows = Int32(0)
        own_tiles = Int32(0)
        for j in cutlass.range_constexpr(R):
            if first + j < rows:
                rq = rq_s[j]
                lowest = cutlass.min(lowest, rq)
                highest_request = cutlass.max(highest_request, rq)
                live_rows = live_rows + 1
                own_tiles = own_tiles + (cols_s[j] + BLOCK_N - 1) // BLOCK_N
        one = Int32(0)
        request = Int32(0)
        if lowest == highest_request and lowest >= 0 and lowest < num_requests:
            one = Int32(1)
            request = lowest
        mT = cute.make_tensor(
            table + Int64(request) * Int64(table_stride_request), cute.make_layout((1 << 30,))
        )

        # Pass 1: each row's highest readable token (a column below the bound naming a token
        # >= 0 whose page is inside the table), and the union of the rows' readable tokens.
        # While row j is walked from its buffer, row j + 1 is staged into the other one.
        tq = cute.make_rmem_tensor((4,), Int32)
        for j in cutlass.range(0, R, 1, unroll=1):
            row = first + j
            at = (j % 2) * self.staged
            if j + 1 < R:
                self.stage(
                    sel, sel_stride, cutlass.min(row + 1, last_row), tidx, sel_s,
                    Int32(self.staged) - at,
                )  # fmt: skip
            if row < rows:
                columns = cols_s[j]
                highest = Int32(-1)
                for run in cutlass.range_constexpr(self.runs):
                    c0 = (run * NT + tidx) * _GP_RUN
                    if c0 < columns:
                        cur_w = Int32(-1)
                        cur_m = Int64(0)
                        # Four columns at a time: four consecutive readable tokens of one
                        # 4-aligned block (one word, one page) are one OR of four bits;
                        # any other four columns are walked singly, as below.
                        for q in cutlass.range_constexpr(_GP_RUN // 4):
                            qc = c0 + 4 * q
                            for k in cutlass.range_constexpr(4):
                                cc = cutlass.min(qc + k, Int32(self.topk - 1))
                                tq[k] = sel_s[at + cc + cc // _GP_RUN]
                            t0 = tq[0]
                            if (
                                qc + 3 < columns
                                and t0 >= 0
                                and (t0 & 3) == 0
                                and tq[1] == t0 + 1
                                and tq[2] == t0 + 2
                                and tq[3] == t0 + 3
                                and t0 // self.tpb < self.width
                            ):
                                highest = cutlass.max(highest, t0 + 3)
                                w = t0 >> 6
                                if w != cur_w:
                                    if cur_w >= 0:
                                        _red_or_shared(words_base + cur_w * 8, cur_m)
                                    cur_w = w
                                    cur_m = Int64(0)
                                cur_m = cur_m | (Int64(15) << Int64(t0 & 63))
                            else:
                                for k in cutlass.range_constexpr(4):
                                    t = Int32(-1)
                                    if qc + k < columns:
                                        t = tq[k]
                                    if t >= 0:
                                        if t // self.tpb < self.width:
                                            highest = cutlass.max(highest, t)
                                            w = t >> 6
                                            if w != cur_w:
                                                if cur_w >= 0:
                                                    _red_or_shared(words_base + cur_w * 8, cur_m)
                                                cur_w = w
                                                cur_m = Int64(0)
                                            cur_m = cur_m | (Int64(1) << Int64(t & 63))
                        if cur_w >= 0:
                            _red_or_shared(words_base + cur_w * 8, cur_m)
                if highest >= 0:
                    _red_max_shared(hi_base + j * 4, highest)
            cute.arch.sync_threads()

        if tidx < R:
            if first + tidx < rows:
                mInfo = cute.make_tensor(info, cute.make_layout((1 << 30,)))
                mInfo[(first + tidx) * 2] = hi_s[tidx]
                mInfo[(first + tidx) * 2 + 1] = cols_s[tidx]

        # Residency. Thread t owns words t, t + 128, ..: it loads the table entries of their
        # pages first (unconditional loads, clamped inside the row, issued together; a warp's
        # load spans 256 bytes of the row), then keeps, per word, the bits of the pages whose
        # entry is a pool page (a group of several requests keeps nothing) and leaves the
        # word's count in `pref`.
        pgs = cute.make_rmem_tensor((self.chunk * self.ppw,), Int32)
        for i in cutlass.range_constexpr(self.chunk):
            for p in cutlass.range_constexpr(self.ppw):
                pgs[i * self.ppw + p] = mT[
                    cutlass.min((tidx + i * NT) * self.ppw + p, Int32(self.width - 1))
                    * table_stride_page
                ]
        for i in cutlass.range_constexpr(self.chunk):
            w = tidx + i * NT
            if w < self.num_words:
                keep = Int64(0)
                for p in cutlass.range_constexpr(self.ppw):
                    page = pgs[i * self.ppw + p]
                    if w * self.ppw + p < self.width and page >= 0 and page < num_pages:
                        keep = keep | Int64(self.page_masks[p])
                m = Int64(0)
                if one != 0:
                    m = words[w] & keep
                words[w] = m
                pref[w] = Int32(cute.arch.popc(m))
        cute.arch.sync_threads()
        # The listed tokens before each word: thread t sums the counts of words [t * chunk,
        # (t + 1) * chunk), a shuffle scan gives the warp's exclusive prefix, the warps' totals
        # the rest.
        local = Int32(0)
        for i in cutlass.range_constexpr(self.chunk):
            w = tidx * self.chunk + i
            if w < self.num_words:
                count = pref[w]
                pref[w] = local
                local = local + count
        lane = tidx % 32
        inclusive = local
        for off in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(inclusive, offset=1 << off)
            if lane >= (1 << off):
                inclusive = inclusive + up
        if lane == 31:
            wt[tidx // 32] = inclusive
        cute.arch.sync_threads()
        base = inclusive - local
        listed = Int32(0)
        for wi in cutlass.range_constexpr(NT // 32):
            if wi < tidx // 32:
                base = base + wt[wi]
            listed = listed + wt[wi]
        for i in cutlass.range_constexpr(self.chunk):
            w = tidx * self.chunk + i
            if w < self.num_words:
                pref[w] = pref[w] + base
        used = (listed + 63) // 64
        if one != 0:
            # Ahead of pass 2: both buffers of re-indexed bits zeroed, the first row staged.
            for i in cutlass.range_constexpr(self.pk_chunk):
                pw = tidx + i * NT
                if pw < used:
                    pk[pw] = Int64(0)
                    pk[self.packed_words + pw] = Int64(0)
            self.stage(sel, sel_stride, cutlass.min(first, last_row), tidx, sel_s, Int32(0))
        cute.arch.sync_threads()

        # The list: each set bit's physical slot at its rank, in logical order (a word's bits
        # by its pages' entries, still in the thread's registers; the threads of a warp own
        # consecutive words, so their stores land side by side).
        mSlots = cute.make_tensor(
            slots + Int64(group) * Int64(slots_stride), cute.make_layout((1 << 30,))
        )
        for i in cutlass.range_constexpr(self.chunk):
            w = tidx + i * NT
            if w < self.num_words:
                m = words[w]
                rank = pref[w]
                while m != Int64(0):
                    low = m & (Int64(0) - m)
                    b = Int32(cute.arch.popc(low - Int64(1)))
                    page = pgs[i * self.ppw]
                    for p in cutlass.range_constexpr(1, self.ppw):
                        if b >= p * self.tpb:
                            page = pgs[i * self.ppw + p]
                    mSlots[rank] = page * self.tpb + b % self.tpb
                    rank = rank + 1
                    m = m ^ low

        if one != 0:
            # Pass 2: each row's bits re-indexed onto the list. While row j is walked into its
            # buffer of `pk` (ORs), the owner of a word writes row j - 1's out of the other
            # buffer and zeroes it, and row j + 1 is staged; one barrier a row.
            for j in cutlass.range(0, R + 1, 1, unroll=1):
                row = first + j
                at = (j % 2) * self.staged
                pk_at = (j % 2) * self.packed_words
                if j >= 1:
                    if row - 1 < rows:
                        mPacked = cute.make_tensor(
                            packed + Int64(row - 1) * Int64(packed_stride),
                            cute.make_layout((1 << 30,)),
                        )
                        for i in cutlass.range_constexpr(self.pk_chunk):
                            pw = tidx + i * NT
                            if pw < used:
                                mPacked[pw] = pk[self.packed_words - pk_at + pw]
                                pk[self.packed_words - pk_at + pw] = Int64(0)
                if j + 1 < R:
                    self.stage(
                        sel, sel_stride, cutlass.min(row + 1, last_row), tidx, sel_s,
                        Int32(self.staged) - at,
                    )  # fmt: skip
                if j < R and row < rows:
                    columns = cols_s[j]
                    for run in cutlass.range_constexpr(self.runs):
                        c0 = (run * NT + tidx) * _GP_RUN
                        if c0 < columns:
                            cur_w = Int32(-1)
                            cur_m = Int64(0)
                            uw_w = Int32(-1)
                            uw = Int64(0)
                            pf = Int32(0)
                            # Four columns at a time: four consecutive tokens of one
                            # 4-aligned block that are all listed take four consecutive
                            # ranks; inside one word of the row's bits they are one OR.
                            for q in cutlass.range_constexpr(_GP_RUN // 4):
                                qc = c0 + 4 * q
                                for k in cutlass.range_constexpr(4):
                                    cc = cutlass.min(qc + k, Int32(self.topk - 1))
                                    tq[k] = sel_s[at + cc + cc // _GP_RUN]
                                t0 = tq[0]
                                fast = Int32(0)
                                if (
                                    qc + 3 < columns
                                    and t0 >= 0
                                    and (t0 & 3) == 0
                                    and tq[1] == t0 + 1
                                    and tq[2] == t0 + 2
                                    and tq[3] == t0 + 3
                                    and t0 // self.tpb < self.width
                                ):
                                    w = t0 >> 6
                                    if w != uw_w:
                                        uw_w = w
                                        uw = words[w]
                                        pf = pref[w]
                                    first_bit = Int64(1) << Int64(t0 & 63)
                                    four = Int64(15) << Int64(t0 & 63)
                                    if (uw & four) == four:
                                        rank = pf + Int32(
                                            cute.arch.popc(uw & (first_bit - Int64(1)))
                                        )
                                        if (rank & 63) <= 60:
                                            fast = Int32(1)
                                            pw = pk_at + (rank >> 6)
                                            if pw != cur_w:
                                                if cur_w >= 0:
                                                    _red_or_shared(pk_base + cur_w * 8, cur_m)
                                                cur_w = pw
                                                cur_m = Int64(0)
                                            cur_m = cur_m | (Int64(15) << Int64(rank & 63))
                                if fast == 0:
                                    for k in cutlass.range_constexpr(4):
                                        t = Int32(-1)
                                        if qc + k < columns:
                                            t = tq[k]
                                        if t >= 0:
                                            if t // self.tpb < self.width:
                                                w = t >> 6
                                                if w != uw_w:
                                                    uw_w = w
                                                    uw = words[w]
                                                    pf = pref[w]
                                                bit = Int64(1) << Int64(t & 63)
                                                if (uw & bit) != Int64(0):
                                                    rank = pf + Int32(
                                                        cute.arch.popc(uw & (bit - Int64(1)))
                                                    )
                                                    pw = pk_at + (rank >> 6)
                                                    if pw != cur_w:
                                                        if cur_w >= 0:
                                                            _red_or_shared(
                                                                pk_base + cur_w * 8, cur_m
                                                            )
                                                        cur_w = pw
                                                        cur_m = Int64(0)
                                                    cur_m = cur_m | (Int64(1) << Int64(rank & 63))
                            if cur_w >= 0:
                                _red_or_shared(pk_base + cur_w * 8, cur_m)
                cute.arch.sync_threads()

        # The plan pass's rule: the row-tile pass takes a one-request group whose list's tiles
        # cost no more than its rows' own columns; any other group's rows join the lone list.
        if tidx == 0:
            mPlan = cute.make_tensor(plan, cute.make_layout((1 << 30,)))
            tiles = (listed + BLOCK_N - 1) // BLOCK_N
            mPlan[group * 2 + 1] = listed
            if one != 0 and tiles * _SHARED_TILE_COST <= own_tiles * 4:
                mPlan[group * 2] = tiles
            else:
                mPlan[group * 2] = Int32(-1)
                mLone = cute.make_tensor(lone, cute.make_layout((1 << 30,)))
                at0 = cute.arch.atomic_add(lone_count, live_rows)
                for j in cutlass.range_constexpr(R):
                    if first + j < rows:
                        mLone[at0 + j] = first + j


@functools.lru_cache(maxsize=None)
def _group_plan_compiled(
    topk, compress_ratio, tokens_per_block, table_width, rows_per_tile, packed_words, qp64, req64
):  # fmt: skip
    """The group plan's launch, compiled once a geometry (2-3 s; the table's width is fixed by
    `max_seq_len`, so one compile a process in serving). Pointer and scalar arguments are
    runtime values; the stream is the caller's."""
    kernel = _GroupPlanKernel(
        topk, compress_ratio, tokens_per_block, table_width, rows_per_tile, packed_words
    )
    gm = cute.AddressSpace.gmem

    def fake(dtype, align):
        return make_ptr(dtype, 16 * 1024, gm, assumed_align=align)

    i32, i64 = cutlass.Int32, cutlass.Int64
    args = (
        fake(i32, 4), fake(i64 if qp64 else i32, 4), fake(i64 if req64 else i32, 4), fake(i32, 4),
        fake(i32, 4), fake(i64, 8), fake(i32, 4), fake(i32, 4), fake(i32, 4), fake(i32, 4),
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        cuda_drv.CUstream(0),
    )  # fmt: skip
    return cute.compile(kernel, *args)


def _group_plan_fits(
    selected_tokens, query_positions, request_indices, block_table, tokens_per_block,
    rows_per_tile, packed_words,
) -> bool:  # fmt: skip
    """Whether a launch fits the group plan; one that does not takes the two passes."""
    return (
        rows_per_tile <= _GP_MAX_ROWS
        and tokens_per_block in (8, 16, 32, 64)
        and _GroupPlanKernel.smem(
            selected_tokens.shape[1],
            block_table.shape[1],
            tokens_per_block,
            rows_per_tile,
            packed_words,
        )
        <= _GP_MAX_SMEM
        and selected_tokens.dtype == torch.int32
        and block_table.dtype == torch.int32
        and query_positions.dtype in _INTEGER_DTYPES
        and request_indices.dtype in _INTEGER_DTYPES
    )


def _group_plan(
    selected_tokens, query_positions, request_indices, block_table, compress_ratio,
    tokens_per_block, num_cache_pages, rows_per_tile, slots, packed_bits, group_plan, lone_rows,
    lone_count,
):  # fmt: skip
    """The bitmap pass and the plan pass in one launch. `slots` `[groups, n]` int32,
    `packed_bits` `[rows, words]` int64 zeros, `group_plan` `[groups, 2]` int32, `lone_rows`
    `[rows]` int32 and `lone_count` `[1]` int32 zero are the plan pass's operands; returns
    `row_info` `[rows, 2]` int32."""
    rows, final_topk = selected_tokens.shape
    device = selected_tokens.device
    table_width = block_table.shape[1]
    row_info = torch.empty((rows, 2), dtype=torch.int32, device=device)
    query_positions = query_positions.contiguous()
    key = (
        final_topk, compress_ratio, tokens_per_block, table_width, rows_per_tile,
        packed_bits.shape[1], query_positions.dtype == torch.int64,
        request_indices.dtype == torch.int64,
    )  # fmt: skip
    fn = _group_plan_compiled(*key)
    gm = cute.AddressSpace.gmem
    i32, i64 = cutlass.Int32, cutlass.Int64

    def ptr(t, dtype, align):
        return make_ptr(dtype, t.data_ptr(), gm, assumed_align=align)

    fn(
        ptr(selected_tokens, i32, 4),
        ptr(query_positions, i64 if key[6] else i32, 4),
        ptr(request_indices, i64 if key[7] else i32, 4),
        ptr(block_table, i32, 4),
        ptr(slots, i32, 4),
        ptr(packed_bits, i64, 8),
        ptr(group_plan, i32, 4),
        ptr(row_info, i32, 4),
        ptr(lone_rows, i32, 4),
        ptr(lone_count, i32, 4),
        rows,
        group_plan.shape[0],
        block_table.shape[0],
        num_cache_pages,
        selected_tokens.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        slots.stride(0),
        packed_bits.stride(0),
        cuda_drv.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    return row_info


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
def _rowtile_plan_kernel(
    request_indices, row_info, bits, block_table, slots, packed_bits, group_plan,
    lone_rows, lone_count, rows,
    bits_stride_row, packed_stride_row, block_table_stride_request, block_table_stride_page,
    slots_stride_group, NUM_REQUESTS, NUM_CACHE_PAGES,
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
        destination = packed_bits + members[:, None] * packed_stride_row + (first // 64)[None, :]
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
            "the row-tile pass needs SM103/SM107, head_dim 256, unit-stride heads (q's heads packed) "
            "and 16-byte-aligned rows"
        )


def rows_per_group(group_size: int) -> int:
    """Rows of one request a row-tile item serves: the most whose `rows x group_size` MMA
    rows fill whole 8-row swizzle atoms of the 128-row tile (10 at 12 heads, 20 at 6)."""
    rows = BLOCK_M // group_size
    while rows > 1 and rows * group_size % ROW_ATOM:
        rows -= 1
    return rows


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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Group plan (or bitmap pass and plan pass), lone-row pass, row-tile pass
    (`qsa_rowtile_attention_cute`); returns `[rows, Hq, D]` contiguous: `out` when given (q's
    dtype and device, contiguous, 16-byte aligned), written in place, else a fresh tensor."""
    _check_inputs(q, k_cache, v_cache, block_table, selected_tokens, request_indices,
                  tokens_per_block, softmax_scale, query_positions, compress_ratio)  # fmt: skip
    rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    if out is None:
        output = torch.empty((rows, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    elif (
        out.shape != (rows, num_q_heads, head_dim)
        or out.dtype != q.dtype
        or out.device != q.device
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError(
            "QSA row-tile `out` must be a contiguous, 16-byte-aligned [rows, Hq, D] of q's dtype"
        )
    else:
        output = out
    if rows == 0:
        return output
    table_width = block_table.shape[1]
    num_words = triton.cdiv(table_width * tokens_per_block, 64)
    rows_per_tile = rows_per_group(group_size)
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
    # A group lists at most its rows' columns: the bound on its slots and re-indexed bits,
    # far below the table's tokens at a deep position.
    listed_cap = min(num_words * 64, rows_per_tile * selected_tokens.shape[1])
    packed_words = triton.cdiv(listed_cap, 64)
    group_planned = _group_plan_fits(
        selected_tokens, query_positions, request_indices, block_table, tokens_per_block,
        rows_per_tile, packed_words,
    )  # fmt: skip
    slots = torch.empty((num_groups, packed_words * 64), dtype=torch.int32, device=q.device)
    packed_bits = torch.zeros((rows, packed_words), dtype=torch.int64, device=q.device)
    group_plan = torch.empty((num_groups, 2), dtype=torch.int32, device=q.device)
    lone_rows = torch.empty((rows,), dtype=torch.int32, device=q.device)
    # [0]: the lone list's length; [1]: the row-tile pass's work-ticket counter.
    counters = torch.zeros((2,), dtype=torch.int32, device=q.device)
    lone_count = counters[:1]
    if group_planned:
        row_info = _group_plan(
            selected_tokens, query_positions, request_indices, block_table, compress_ratio,
            tokens_per_block, k_cache.shape[0], rows_per_tile, slots, packed_bits, group_plan,
            lone_rows, lone_count,
        )  # fmt: skip
    else:
        bits, row_info = _row_bits(
            selected_tokens, query_positions, compress_ratio, tokens_per_block, table_width,
            num_words,
        )  # fmt: skip
        _rowtile_plan_kernel[(num_groups,)](
            request_indices, row_info, bits, block_table, slots, packed_bits, group_plan,
            lone_rows, lone_count, rows,
            bits.stride(0), packed_bits.stride(0), block_table.stride(0), block_table.stride(1),
            slots.stride(0), block_table.shape[0], k_cache.shape[0],
            ROWS_PER_TILE=rows_per_tile, ROWS_POW2=rows_pow2, BLOCK_N=BLOCK_N,
            SHARED_TILE_COST=_SHARED_TILE_COST, PAGE_TABLE_WIDTH=table_width,
            TOKENS_PER_BLOCK=tokens_per_block, PLAN_WORDS=_PLAN_WORDS.get(rows_pow2, 16),
            num_warps=_PLAN_WARPS,
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
    # q is the first heads of the packed [T, Hq + 2 Hkv, 256] QKV buffer.
    q_layout="qkv_slice",
    request_dtype=torch.int32,
    # The engine's table at max_seq_len 262144: `ceil(262145 / 32)` columns padded
    # to a multiple of 4, -1 past each sequence.
    pages=8196,
)

#: The engine's trunk (`Qwen4ExpHybridCacheManager.get_buffers`): a lifecycle slot holds the 24
#: role pages of the 12 attention layers, K then V each; the call's K and V are layer 11's two
#: planes, and the requests' pages straddle 2^31 elements of them.
_TRUNK = dict(cache_layout="trunk", slot_pages=24, layer=11, offset_edge=2**31)
#: A tp4 / tep4 rank (attention TP 4): 6 q heads and one of the 2 KV heads, a 128-row table.
_TP4 = dict(q_heads=6, kv_heads=1, slots=128, **_TRUNK)
#: A dep4 rank (attention DP): all 24 / 2 heads, a 32-row table; core's bf16 trunk, or the
#: kv_fp8 variant's pages widened into a contiguous `[pages, 2, 32, 256]` bf16 scratch
#: (`qsa_fp8_pages_to_bf16`) addressed through its own table.
_DEP4 = dict(q_heads=24, kv_heads=2, slots=32, **_TRUNK)
_DEP4_SCRATCH = dict(q_heads=24, kv_heads=2, slots=32, cache_layout="scratch")
#: A tp4 / tep4 rank's quantized pool: an fp8 pool widened, or an NVFP4 pool dequantized
#: (`qsa_nvfp4_pages_to_bf16`), into a contiguous `[pages, 1, 32, 256]` bf16 scratch.
_TP4_SCRATCH = dict(q_heads=6, kv_heads=1, slots=128, cache_layout="scratch")


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

    Four launches. A bitmap pass: one bit per selected logical token per row.
    A plan pass per group of `rows_per_group` consecutive rows of one request
    (10 at a dep4 rank's 12 heads a KV head, 20 at a tp4 / tep4 rank's 6): the union of
    the rows' selected tokens whose page is mapped, listed in logical order
    as physical cache slots, and each row's bitmap re-indexed onto that list
    (a per-word parallel bit extract). The group goes to the row-tile pass
    when walking its list in 64-slot tiles costs at most 1.5x per-row tiles
    less than the rows' own columns would; otherwise -- or when its rows span
    requests -- its rows join the lone list. A lone-row pass (256 programs
    per KV head striding the list) serves listed rows over their own columns,
    as the per-row kernel does. The row-tile pass is persistent (one program
    per SM drawing (group, KV head) items from a ticket counter, the plan's
    last groups first); per item it loads each listed tile's K/V once for the
    group's 120 (row, head) pairs and masks each row to its own re-indexed bits, and
    one item's epilogue overlaps the next item's first MMAs. It is launched
    programmatically dependent on the lone pass, whose rows are disjoint, so
    the two overlap. That pass is `qsa_rowtile_attention_cute` (tcgen05 MMAs,
    S / P / O in tensor memory, a cp.async gather of the K/V rows, a TMA store
    of O), so the entry serves SM103/SM107, bf16 q and K/V and D = 256.

    Fusion boundary: as `qsa_paged_sparse_gqa` -- the block-table gather,
    scale, online softmax and P.V are inside; deciding what to attend to,
    appending K/V and q's projection, norm, RoPE and gate are not.
    """

    ARCHS = SERVED_ARCHS

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp4/tep4 cold prefill: 8192 rows of one request from 0, 6 q / 1 kv heads "
            "(20-row groups), trunk pool straddling 2^31, [128, 8196] table",
            spec=_spec([(0, 8192)], _TP4),
        ),
        Cell(
            why="tp4/tep4 deep chunk: 8192 rows from 188416 ranking one score vector "
            "(a sparse union), 6 q / 1 kv heads, trunk pool straddling 2^31",
            spec=_spec([(188416, 8192)], _TP4, shared_scores=True),
        ),
        Cell(
            why="tp4/tep4 2049 rows from 9000: a last group of 9 rows, 6 q / 1 kv heads",
            spec=_spec([(9000, 2049)], _TP4),
        ),
        Cell(
            why="tp4/tep4 kv_fp8 / kv_nvfp4 chunk: 8192 rows from 16384 over the bf16 scratch, "
            "6 q / 1 kv heads (20-row groups), [128, 8196] table",
            spec=_spec([(16384, 8192)], _TP4_SCRATCH),
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
        blocks, and a tail outside every complete block).
        Rows served by the lone-row pass keep per-column semantics.
      - As for the per-row kernel: each row is prefix-packed with exactly the
        bound's count of valid entries; entries past the 64-rounded bound
        are ignored.

    Work: a row-tile group walks its packed list, so its MMA work grows
    with the union of its rows' selections rather than with the group's
    highest position. Below a few hundred rows it is slower than the per-row
    kernel; the target's served context launches are thousands of rows.

    Compiled once per (group size, rows per tile, tokens per block, PDL):
    nothing in the row-tile pass's compile key depends on the row count or
    the plan.

    Scratch per call: the bitmap, `rows x ceil(table_width * tpb / 64)` int64
    words (267 MB at 8160 rows and an 8196-page table), the re-indexed
    bitmap and the slot lists, bounded by a group's selections (`rows x
    ceil(rows_per_group * 2051 / 64)` words, 42 MB, and `ceil(rows /
    rows_per_group) x` that many words' tokens as int32, 134 MB at 8160 rows),
    the plan and the lone list. Calls allocate,
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
        out: torch.Tensor | None = None,
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
            out=out,
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
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The reference `qsa_paged_sparse_gqa` certifies against: fp32 sparse
        GQA over every resident selected token, rounded once. The prefill pair
        is ignored, as there: on a prefix-packed selection the bound reads
        exactly the valid entries. `out` only names where the call writes; the
        reference returns a fresh tensor."""
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
