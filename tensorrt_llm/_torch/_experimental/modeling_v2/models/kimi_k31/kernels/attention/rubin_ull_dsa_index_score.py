# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Adapted from vllm-project/vllm@e3ddbdbd60
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/dsa/indexer.py (IndexerScore), with the
# helpers it uses from ../common/{host,primitives}.py. Changed: an FP32 epilogue that reproduces
# FP8MQALogitsKernel's FP32 logits bit for bit (instead of BF16 relu / weights / sums and 16-bit
# scores); FP32 key scales (instead of UE8M0 bytes); FP8 keys and the pitched (DCP) logits layout
# only (no MXFP4); the optional 8-key maxima as 32-bit order-preserving images of the FP32 logits
# (instead of 16-bit scores); griddepcontrol.wait before the row ends are read; a request's tiles
# from the largest end of its rows.
"""Key-balanced persistent tcgen05 scorer of the key-shard decode indexer (FP32 logits).

``score_logits_pdl`` writes the same FP32 logits as ``logits.paged_logits_pdl`` (the tree's
``FP8MQALogitsKernel``) for every key a row sees, from one launch whose work is spread evenly over
the SMs: the 128-key tiles of all requests are cut into equal ranges, one per CTA (grid = the
device's SM count), and one ``[128 keys x q * 32]`` FP8 MMA per tile scores every row of the
request, so each key page is read once per step.

Row ``t`` (request ``t // q``) is written at ``out[t, :row_lens[t]]``; columns at or past the
row's length are not written (their content is whatever the buffer held). A request is scored
over ``ceil(max(row_lens of its rows) / 128)`` tiles of its block table.

Epilogue (per key and row; ``x_h`` the FP32 MMA accumulator of head ``h``), as
``FP8MQALogitsKernel._epi_t_body`` with FP32 accumulator / epilogue / output, one epilogue
sub-tile and the relu trick::

    a_h = x_h + |x_h|                                     (round to nearest)
    s0x = fma(a_h,     w_h,     s0x)  s0y = fma(a_h+1, w_h+1, s0y)
    s1x = fma(a_h+2,   w_h+2,   s1x)  s1y = fma(a_h+3, w_h+3, s1y)    h = 0, 4, ..., 28
    logit = (((s0x + s0y) + s1x) + s1y) * (scale * 0.5)

with the FP32 head weights of the row (already scaled) and the key's FP32 scale word.

Roles per CTA (``32 * (epilogue warps + 2)`` threads): the epilogue warpgroups (each owns one TMEM
accumulator and a share of a tile's rows: at most four rows, whose relu'd accumulators stay in 128
registers after the accumulator is released), one producer warp (bulk copies of the key scales and
head weights, TMA of the key pages and the query) and one MMA warp.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op

from .rubin_ull_dsa_index_ptx import _asm

PAGE = 64  # keys per indexer cache block
DIM = 128
HEADS = 32
TOKEN_BYTES = DIM + 4  # FP8 key codes then one FP32 scale word per key, page-major
TILE = 128  # keys per MMA tile (two pages)
ITEM_TILES = 16  # tiles per work item (the producer caches 32 page ids, one per lane)
MAX_REQUESTS = 512  # tile prefix sum in shared memory
STAGES = 6
Q8_STAGES = 4  # FP8 key tiles in flight at q = 8
ROWS_PER_WARPGROUP = 4  # relu'd FP32 accumulator rows a thread holds (32 registers each)
TMEM_COLUMNS = 512
_RN = "rn"


# --------------------------------------------------------------------------------------------
# PTX helpers
# --------------------------------------------------------------------------------------------
def _address(value, kind, loc, ip):
    if hasattr(value, "toint"):
        value = value.toint(loc=loc, ip=ip)
    return Int32(value) if kind == "r" else Int64(value)


@dsl_user_op
def bulk_g2s(dst, src, barrier, size: int, *, loc=None, ip=None):
    """``size`` bytes global -> shared, completing on ``barrier``'s transaction count."""
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];"
    operands = (
        _address(dst, "r", loc, ip),
        _address(src, "l", loc, ip),
        _address(barrier, "r", loc, ip),
        Int32(size),
    )
    _asm(code, "r,l,r,r,~{memory}", operands, loc=loc, ip=ip)


@dsl_user_op
def store_f32(address, value, *, loc=None, ip=None):
    _asm(
        "st.global.f32 [$0], $1;",
        "l,f,~{memory}",
        (Int64(address), Float32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_vector_top(logit, key, end, address, *, loc=None, ip=None):
    """The largest order-preserving image (``ptx.ordered_bits``) of the logits of 8 consecutive
    keys (lanes ``8i .. 8i + 7`` of the warp, each holding key ``key``'s ``logit``; keys at or past
    ``end`` count as 0) stored at ``address`` by the vector's first lane when its key is below
    ``end``. Every lane of the warp calls it."""
    _asm(
        "{ .reg .b32 b, m, o; .reg .pred p, s; "
        "mov.b32 b, $0; shr.s32 m, b, 31; or.b32 m, m, 0x80000000; xor.b32 b, b, m; "
        "setp.lt.s32 p, $1, $2; selp.b32 b, b, 0, p; "
        "shfl.sync.bfly.b32 o, b, 1, 0x1f, 0xffffffff; max.u32 b, b, o; "
        "shfl.sync.bfly.b32 o, b, 2, 0x1f, 0xffffffff; max.u32 b, b, o; "
        "shfl.sync.bfly.b32 o, b, 4, 0x1f, 0xffffffff; max.u32 b, b, o; "
        "and.b32 m, $1, 7; setp.eq.and.u32 s, m, 0, p; @s st.global.u32 [$3], b; }",
        "f,r,r,l,~{memory}",
        (Float32(logit), Int32(key), Int32(end), Int64(address)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32x4(pointer, *, loc=None, ip=None):
    """Four FP32 words at a 16-byte aligned shared address."""
    return _asm(
        "ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        (_address(pointer, "r", loc, ip),),
        (Float32, Float32, Float32, Float32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32(pointer, *, loc=None, ip=None):
    """One FP32 word of shared memory, as a side-effecting asm (kept after the barrier wait that
    guards it)."""
    return _asm(
        "ld.shared.f32 $0, [$1];",
        "=f,r,~{memory}",
        (_address(pointer, "r", loc, ip),),
        Float32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_i32(pointer, *, loc=None, ip=None):
    """One Int32 word of shared memory (side-effecting, as ``load_shared_f32``)."""
    return _asm(
        "ld.shared.b32 $0, [$1];",
        "=r,r,~{memory}",
        (_address(pointer, "r", loc, ip),),
        Int32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_i32(address, *, loc=None, ip=None):
    return _asm(
        "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),), Int32, loc=loc, ip=ip
    )


@dsl_user_op
def fence_tmem(before: bool = False, *, loc=None, ip=None):
    kind = "before_thread_sync" if before else "after_thread_sync"
    _asm(f"tcgen05.fence::{kind};", "~{memory}", (), loc=loc, ip=ip)


@cute.jit
def table_entry(table_address: Int64, table_pitch: Int32, r: Int32, column: Int32) -> Int32:
    """Entry (r, column) of an int32 block table with row pitch ``table_pitch``."""
    return load_i32(table_address + (Int64(r) * Int64(table_pitch) + Int64(column)) * 4)


@cute.jit
def warp_inclusive(value: Int32, lane: Int32) -> Int32:
    for i in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(value, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            value += up
    return value


@cute.jit
def request_tiles(row_ends: cute.Tensor, r: Int32, q: cutlass.Constexpr[int]) -> Int32:
    """128-key tiles of request ``r``: those of the longest of its ``q`` rows (0 if all empty)."""
    end = row_ends[r * q]
    for t in cutlass.range_constexpr(1, q):
        end = cutlass.max(end, row_ends[r * q + t])
    tiles = Int32(0)
    if end > 0:
        tiles = (end + Int32(TILE - 1)) // Int32(TILE)
    return tiles


@cute.jit
def _first_request(start: cute.Tensor, requests: Int32, g: Int32) -> Int32:
    """The request whose tiles contain global tile ``g`` (binary search)."""
    lo = Int32(0)
    hi = requests
    while lo < hi:
        mid = (lo + hi) // 2
        if start[mid + 1] > g:
            hi = mid
        else:
            lo = mid + 1
    return lo


@cute.jit
def _next_item(start: cute.Tensor, requests: Int32, g: Int32, r: Int32, hi: Int32):
    """The work item at global tile ``g`` of request ``r`` (the CTA range ends at ``hi``):
    (request, first key, tiles, next g, next r)."""
    r_end = start[r + 1]
    tiles = cutlass.min(cutlass.min(hi, r_end) - g, Int32(ITEM_TILES))
    k0 = (g - start[r]) * Int32(TILE)
    req = r
    g += tiles
    while r < requests and start[r + 1] <= g:
        r += 1
    return req, k0, tiles, g, r


def _mn_layout(t):
    """A tcgen05 accumulator as ((M), (N), stages)."""
    sh, sd = t.layout.shape, t.layout.stride
    new = cute.make_layout(
        ((sh[0][0], sh[1]), (sh[0][1], sh[2]), *sh[3:]),
        stride=((sd[0][0], sd[1]), (sd[0][1], sd[2]), *sd[3:]),
    )
    return cute.make_tensor(t.iterator, new)


def score_epilogue(q: int) -> tuple[int, int]:
    """(TMEM accumulators, epilogue warpgroups per accumulator) of a q-row scorer: a warpgroup
    holds at most ROWS_PER_WARPGROUP rows' relu'd accumulators in registers (the accumulator is
    released before the head sums); with one warpgroup per accumulator, two accumulators (the
    next tile's MMA overlaps the head sums), else one split over warpgroups."""
    split = -(-q // ROWS_PER_WARPGROUP)
    return (2 if split == 1 else 1), split


class ScoreKernel:
    """FP32 logits of ``rows`` rows (``q`` per request) at ``out + (t * pitch + key) * 4``.

    The epilogue layout is ``score_epilogue(q)``.

    ``early_schedule``: the row ends, the tile schedule and each role's first work item are read
    before ``griddepcontrol.wait``, which only the producer warp executes (see ``kernel``).

    ``preload`` (with ``early_schedule``): the producer also reads the first item's block table
    entries and issues its key-page copies (up to ``stages`` tiles) before the wait, all but the
    request's last tile. Only a request's last tile can hold keys written in this step (the
    step's new tokens are the last ``next_n`` positions: on each rank, within its last local
    page); every other page is complete before the step starts, and the block table is step
    metadata uploaded before the graph. The query, weights and the last tile follow the wait."""

    def __init__(
        self,
        rows: int,
        q: int,
        ctas: int,
        pitch: int,
        early_trigger: bool = True,
        tops_from: int = 0,
        early_schedule: bool = False,
        preload: bool = False,
    ):
        assert not preload or early_schedule
        assert rows % q == 0 and 1 <= q <= 8 and rows // q <= MAX_REQUESTS and pitch > 0
        self.rows, self.q, self.ctas, self.pitch = rows, q, ctas, pitch
        self.early_trigger = early_trigger
        # tops_from > 0: each 8 keys' largest logit image (ptx.ordered_bits) at
        # tops + (t * pitch + key) / 8 * 4, for the rows of at least tops_from keys.
        assert tops_from >= 0 and (not tops_from or pitch % 8 == 0)
        self.tops_from = tops_from
        self.early_schedule = early_schedule
        self.preload = preload
        self.accs, self.split = score_epilogue(q)
        assert -(-q // self.split) <= ROWS_PER_WARPGROUP
        self.epi_warps = 4 * self.accs * self.split
        self.producer = self.epi_warps
        self.mma_warp = self.producer + 1
        self.threads = 32 * (self.mma_warp + 1)
        self.stages = Q8_STAGES if q == 8 else STAGES
        self.n = q * HEADS
        self.tiler = (TILE, self.n, DIM)
        used = self.accs * self.n
        assert used <= TMEM_COLUMNS
        self.columns = max(32, 1 << (used - 1).bit_length())

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # Uint8 [blocks, 64 * 132]: per page 64 x 128 FP8 codes, then 64 FP32
        query: cute.Tensor,  # Uint8 (FP8) [rows * 32, 128, 1]
        weights: cute.Tensor,  # FP32 [rows * 32]
        row_ends: cute.Tensor,  # Int32 [rows]
        table_address: Int64,  # Int32 [requests, pages]: the requests' local page ids
        table_pitch: Int32,
        table_width: Int32,
        kv_address: Int64,
        page_bytes: Int64,
        logits_address: Int64,  # FP32 rows of `pitch` elements
        tops_address: Int64,  # Uint32 rows of `pitch / 8` vector maxima (``tops``)
        stream,
    ):
        ab = cutlass.Float8E4M3FN
        kv8 = cute.recast_tensor(kv, ab)
        blocks = kv.shape[0]
        keys = cute.make_tensor(
            kv8.iterator,
            cute.make_layout((PAGE, DIM, blocks), stride=(DIM, 1, kv.layout.stride[0])),
        )
        query = cute.recast_tensor(query, ab)
        mma = sm100_utils.make_trivial_tiled_mma(
            ab,
            ab,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.tiler[:2],
        )
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, ab, self.stages)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, ab, 2)
        atom = sm100_utils.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, ab)
        page_layout = cute.tile_to_shape(atom, (PAGE, DIM, 2 * self.stages), (0, 1, 2))
        tma_k, tma_k_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            keys,
            cute.slice_(page_layout, (None, None, 0)),
            (PAGE, DIM),
        )
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_q, tma_q_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma.thr_id),
            query,
            b_one,
            self.tiler,
            mma,
            vmnk.shape,
        )
        self.q_bytes = cute.size_in_bytes(ab, b_one)
        self.kernel(
            mma,
            tma_k,
            tma_k_tensor,
            tma_q,
            tma_q_tensor,
            a_layout,
            b_layout,
            page_layout,
            weights,
            row_ends,
            table_address,
            table_pitch,
            table_width,
            kv_address,
            page_bytes,
            Int32(blocks),
            logits_address,
            tops_address,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def item_tiles(self, lo: cutlass.Constexpr[int], hi: cutlass.Constexpr[int], item, bars, views):
        """One epilogue warpgroup's tiles of an item: rows [lo, hi) of every tile whose
        accumulator it owns (nothing but the barriers for an empty range)."""
        k0, ntile, it0, acc, lt, qs, row0, out, tops = item
        kv_full, sc_empty, acc_full, acc_empty = bars
        sSc, sW, sEnd, t2r, tAll, cD = views
        ACCS = self.accs
        for t in cutlass.range(ntile, unroll=1):
            it = it0 + t
            if it % ACCS == acc:
                stage = it % self.stages
                key = k0 + t * TILE + lt
                phase = (it // self.stages) & 1
                cute.arch.mbarrier_wait(kv_full + stage, phase)
                # The stage's scales are rewritten once every epilogue thread has arrived on
                # sc_empty: read them strictly between the kv_full wait and that arrival.
                kscale = load_shared_f32(sSc.iterator + (lt + stage * TILE))
                cute.arch.mbarrier_arrive(sc_empty + stage)
                cute.arch.mbarrier_wait(acc_full + acc, (it // ACCS) & 1)
                # tcgen05.ld is .sync.aligned: reconverge the warp after the per-thread wait.
                cute.arch.sync_warp()
                fence_tmem()
                if cutlass.const_expr(lo < hi):
                    rows = row0, key, kscale, qs, acc, out, tops
                    self.rows_out(lo, hi, t2r, tAll, cD, sW, sEnd, rows, acc_empty + acc)
                else:
                    cute.arch.mbarrier_arrive(acc_empty + acc)

    @cute.jit
    def rows_out(
        self,
        lo: cutlass.Constexpr[int],
        hi: cutlass.Constexpr[int],
        t2r,
        tAll,
        cD,
        sW,
        sEnd,
        rows,
        release,
    ):
        """Rows [lo, hi) of one tile: each row's accumulator is read from TMEM and relu'd
        (x + |x|), then summed over the heads with the row's weights, scaled and stored
        (FP8MQALogitsKernel's order). Every row is relu'd into registers and the accumulator
        released (``release``) before the sums, so the next MMA overlaps them."""
        row0, key, kscale, qs, acc, out, tops = rows
        scale_mul = kscale * 0.5
        w_base = sW.iterator + qs * self.n
        out_row = (row0, key, sEnd, qs, out, tops, w_base, scale_mul)
        # tcgen05.ld's registers are valid only after tcgen05.wait::ld: each row is used after
        # the wait that covers its load, while the next row's load is in flight.
        held = []
        frags = [cute.make_fragment_like(cD, Float32) for _ in range(lo, hi)]
        cute.copy(t2r, tAll[(None, None, None, 0, lo, acc)], frags[0])
        cute.arch.fence_view_async_tmem_load()
        for tok in cutlass.range_constexpr(lo, hi):
            if cutlass.const_expr(tok + 1 < hi):
                cute.copy(t2r, tAll[(None, None, None, 0, tok + 1, acc)], frags[tok + 1 - lo])
            frag = frags[tok - lo]
            values = []
            for h in cutlass.range_constexpr(0, HEADS, 2):
                x0 = frag[h]
                x1 = frag[h + 1]
                a0, a1 = cute.arch.add_packed_f32x2((x0, x1), (abs(x0), abs(x1)), rnd=_RN)
                values.append(a0)
                values.append(a1)
            held.append(values)
            if cutlass.const_expr(tok + 1 < hi):
                cute.arch.fence_view_async_tmem_load()
        # The loads are complete (the waits above) and ordered before the release that lets the
        # next MMA overwrite the accumulator.
        fence_tmem(True)
        cute.arch.mbarrier_arrive(release)
        for tok in cutlass.range_constexpr(lo, hi):
            self.row_out(tok, held[tok - lo], out_row)

    @cute.jit
    def row_out(self, tok: cutlass.Constexpr[int], a, out_row):
        """Row ``tok``'s logit of this thread's key from its relu'd head values ``a``."""
        row0, key, sEnd, qs, out, tops, w_base, scale_mul = out_row
        s0x = Float32(0.0)
        s0y = Float32(0.0)
        s1x = Float32(0.0)
        s1y = Float32(0.0)
        for h in cutlass.range_constexpr(0, HEADS, 4):
            w0, w1, w2, w3 = load_shared_f32x4(w_base + (tok * HEADS + h))
            s0x, s0y = cute.arch.fma_packed_f32x2((a[h], a[h + 1]), (w0, w1), (s0x, s0y), rnd=_RN)
            s1x, s1y = cute.arch.fma_packed_f32x2(
                (a[h + 2], a[h + 3]), (w2, w3), (s1x, s1y), rnd=_RN
            )
        result = s0x + s0y + s1x + s1y
        logit = result * scale_mul
        end = load_shared_i32(sEnd.iterator + (tok + 8 * qs))
        element = row0 + Int64(tok * self.pitch) + Int64(key)
        if cutlass.const_expr(self.tops_from > 1):
            if end >= Int32(self.tops_from):
                store_vector_top(logit, key, end, tops + (element >> 3) * 4)
        elif cutlass.const_expr(self.tops_from == 1):
            store_vector_top(logit, key, end, tops + (element >> 3) * 4)
        if key < end:
            store_f32(out + element * 4, logit)

    @cute.jit
    def issue_tile(
        self,
        t: Int32,
        it: Int32,
        my_page: Int32,
        blocks: Int32,
        kv_full,
        kv_empty,
        sc_empty,
        sSc: cute.Tensor,
        scale_base: Int64,
        page_bytes: Int64,
        tma_k: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
    ):
        """The producer's copies of tile ``t`` of the current item (``my_page``: lane ``i`` holds
        the item's page ``i``) into stage ``it % stages``: two key pages by TMA, their scale words
        by bulk copy, all on ``kv_full``."""
        stage = it % self.stages
        p0 = cute.arch.shuffle_sync(my_page, 2 * t)
        p1 = cute.arch.shuffle_sync(my_page, 2 * t + 1)
        # A partial last tile's second page may lie past the request's pages:
        # keep its (unused) scale read in bounds (the key TMA is bounds-checked).
        p1 = cutlass.min(cutlass.max(p1, Int32(0)), blocks - 1)
        ph = ((it // self.stages) & 1) ^ 1
        cute.arch.mbarrier_wait(kv_empty + stage, ph)
        cute.arch.mbarrier_wait(sc_empty + stage, ph)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(kv_full + stage, TILE * DIM + TILE * 4)
            # 64-bit page offsets: the page ids index the whole pool.
            bulk_g2s(
                sSc.iterator + stage * TILE,
                scale_base + Int64(p0) * page_bytes,
                kv_full + stage,
                PAGE * 4,
            )
            bulk_g2s(
                sSc.iterator + stage * TILE + PAGE,
                scale_base + Int64(p1) * page_bytes,
                kv_full + stage,
                PAGE * 4,
            )
        cute.copy(tma_k, tKgK[(None, 0, 0, p0)], tKsK[(None, 2 * stage)], tma_bar_ptr=kv_full + stage)
        cute.copy(
            tma_k, tKgK[(None, 0, 0, p1)], tKsK[(None, 2 * stage + 1)], tma_bar_ptr=kv_full + stage
        )

    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        page_layout: cute.ComposedLayout,
        weights: cute.Tensor,
        row_ends: cute.Tensor,
        table_address: Int64,
        table_pitch: Int32,
        table_width: Int32,
        kv_address: Int64,
        page_bytes: Int64,
        blocks: Int32,
        logits_address: Int64,
        tops_address: Int64,
    ):
        Q = self.q
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()

        @cute.struct
        class Storage:
            kv_full: cute.struct.MemRange[cutlass.Int64, self.stages]
            kv_empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            sc_empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            q_full: cute.struct.MemRange[cutlass.Int64, 2]
            q_empty: cute.struct.MemRange[cutlass.Int64, 2]
            acc_full: cute.struct.MemRange[cutlass.Int64, self.accs]
            acc_empty: cute.struct.MemRange[cutlass.Int64, self.accs]
            tmem_hold: Int32
            row_end: cute.struct.MemRange[Int32, 8 * 2]
            start: cute.struct.MemRange[Int32, MAX_REQUESTS + 1]
            w: cute.struct.Align[cute.struct.MemRange[Float32, 2 * self.n], 16]
            scale: cute.struct.Align[cute.struct.MemRange[Float32, self.stages * TILE], 16]

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        sA = smem.allocate_tensor(
            cutlass.Float8E4M3FN, a_layout.outer, 1024, swizzle=a_layout.inner
        )
        sB = smem.allocate_tensor(
            cutlass.Float8E4M3FN, b_layout.outer, 1024, swizzle=b_layout.inner
        )
        sPage = cute.make_tensor(sA.iterator, page_layout.outer)
        sW = st.w.get_tensor(cute.make_layout((self.n, 2)))
        sEnd = st.row_end.get_tensor(cute.make_layout((8, 2)))
        sStart = st.start.get_tensor(cute.make_layout((MAX_REQUESTS + 1,)))
        sSc = st.scale.get_tensor(cute.make_layout((TILE, self.stages)))

        kv_full = st.kv_full.data_ptr()
        kv_empty = st.kv_empty.data_ptr()
        sc_empty = st.sc_empty.data_ptr()
        q_full = st.q_full.data_ptr()
        q_empty = st.q_empty.data_ptr()
        acc_full = st.acc_full.data_ptr()
        acc_empty = st.acc_empty.data_ptr()
        if tidx == 0:
            for i in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(kv_full + i, 1)
                cute.arch.mbarrier_init(kv_empty + i, 1)
                cute.arch.mbarrier_init(sc_empty + i, 128 * self.split)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(q_full + i, 1)
                cute.arch.mbarrier_init(q_empty + i, 1 + self.epi_warps)
            for i in cutlass.range_constexpr(self.accs):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, 128 * self.split)
            cute.arch.mbarrier_init_fence()
        if warp == self.producer:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_q)

        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.threads
            ),
            allocator_warp_id=self.mma_warp,
        )
        tmem.allocate(self.columns)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        acc_fake = mma.make_fragment_C(cute.append(acc_shape, self.accs))
        tAcc = cute.make_tensor(tmem_ptr, acc_fake.layout)

        if cutlass.const_expr(not self.early_schedule):
            # Everything below reads the step's data (row ends, block table, query, weights,
            # keys), which earlier kernels of the graph may produce: wait for them first.
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.early_trigger):
                cute.arch.griddepcontrol_launch_dependents()
        # early_schedule: the row ends (and the schedule built from them) are read before the
        # wait, as FP8MQALogitsKernel reads its schedule metadata and context lengths. That needs
        # their last writer complete when this grid starts. The row ends are step metadata: the
        # step head's kernels (Triton, no PDL trigger) write them before layer 0, so the kernel
        # after the writer starts only once the writer is complete. A grid starts only after its
        # predecessors have started (a PDL edge: they triggered; a full edge or event join: they
        # completed), so this grid starting implies that kernel started, hence the writer is
        # complete, whatever the PDL triggers of the kernels in between. Writing the row ends
        # later in the step, or from a kernel that triggers its dependents before that write,
        # breaks this. Only the producer warp waits: it alone reads data of the predecessors
        # (block table, query, weights, keys). The MMA warp reads shared memory filled by the
        # producer's copies, and the epilogue's logits stores follow those copies through the
        # barriers.
        # The schedule: sStart[r] = first global tile of request r.
        requests = self.rows // Q
        for r in cutlass.range(tidx, requests, self.threads, unroll=1):
            sStart[r + 1] = request_tiles(row_ends, r, Q)
        cute.arch.sync_threads()
        if warp == 0:
            carry = Int32(0)
            for base in cutlass.range(0, requests, 32, unroll=1):
                r = base + lane
                tiles = Int32(0)
                if r < requests:
                    tiles = sStart[r + 1]
                tiles = warp_inclusive(tiles, lane)
                if r < requests:
                    sStart[r + 1] = carry + tiles
                carry += cute.arch.shuffle_sync(tiles, 31)
            if lane == 0:
                sStart[0] = Int32(0)
        cute.arch.sync_threads()
        total = sStart[requests]
        lo = Int32((Int64(total) * Int64(bid)) // Int64(self.ctas))
        hi = Int32((Int64(total) * Int64(bid + 1)) // Int64(self.ctas))
        r_lo = _first_request(sStart, requests, lo)

        gQ = cute.local_tile(mQ, cute.slice_(self.tiler, (0, None, None)), (None, None, None))
        thr_mma = mma.get_slice(0)
        tCgQ = thr_mma.partition_B(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )
        gK = cute.local_tile(mK, (PAGE, DIM), (None, None, None))
        tKsK, tKgK = cpasync.tma_partition(
            tma_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sPage, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        it = Int32(0)  # tiles issued / consumed by this role
        j = Int32(0)  # items issued / consumed by this role
        g = lo
        r = r_lo
        max_page = table_width - 1
        scale_base = kv_address + Int64(PAGE * DIM)
        if warp == self.producer:
            # One item of metadata ahead: the next item's dependent loads overlap the current
            # item's tiles.
            n_req = Int32(0)
            n_k0 = Int32(0)
            n_nt = Int32(0)
            n_page = Int32(0)
            n_end = Int32(0)
            # Tiles of the first item already issued before the wait (``preload``).
            pre = Int32(0)
            if cutlass.const_expr(self.early_schedule):
                # The first item's schedule and row ends before the wait, its block table
                # entries after it (``preload``: before it, with its complete tiles' copies).
                if g < hi:
                    n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                    n_end = row_ends[n_req * Q + cutlass.min(lane, Q - 1)]
                    if cutlass.const_expr(self.preload):
                        n_page = table_entry(
                            table_address,
                            table_pitch,
                            n_req,
                            cutlass.min(n_k0 // PAGE + lane, max_page),
                        )
                        # Tiles before the request's last one, at most one round of stages.
                        first = sStart[n_req] + n_k0 // TILE
                        pre = cutlass.max(
                            cutlass.min(
                                cutlass.min(n_nt, sStart[n_req + 1] - 1 - first),
                                Int32(self.stages),
                            ),
                            Int32(0),
                        )
                        for t in cutlass.range(pre, unroll=1):
                            self.issue_tile(
                                t, it, n_page, blocks, kv_full, kv_empty, sc_empty, sSc,
                                scale_base, page_bytes, tma_k, tKgK, tKsK,
                            )
                            it += 1
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.early_trigger):
                    # One warp's trigger releases the CTA's share of the dependents.
                    cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(not self.preload):
                    if n_nt > 0:
                        n_page = table_entry(
                            table_address,
                            table_pitch,
                            n_req,
                            cutlass.min(n_k0 // PAGE + lane, max_page),
                        )
            elif g < hi:
                n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                n_page = table_entry(
                    table_address,
                    table_pitch,
                    n_req,
                    cutlass.min(n_k0 // PAGE + lane, max_page),
                )
                n_end = row_ends[n_req * Q + cutlass.min(lane, Q - 1)]
            has_item = n_nt > 0
            while has_item:
                req = n_req
                ntile = n_nt
                my_page = n_page
                my_end = n_end
                n_nt = Int32(0)
                if g < hi:
                    n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                    n_page = table_entry(
                        table_address,
                        table_pitch,
                        n_req,
                        cutlass.min(n_k0 // PAGE + lane, max_page),
                    )
                    n_end = row_ends[n_req * Q + cutlass.min(lane, Q - 1)]
                qs = j % 2
                cute.arch.mbarrier_wait(q_empty + qs, ((j // 2) & 1) ^ 1)
                if lane < Q:
                    sEnd[lane, qs] = my_end
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_full + qs, self.q_bytes + self.n * 4)
                    bulk_g2s(
                        sW.iterator + qs * self.n,
                        weights.iterator + req * Q * HEADS,
                        q_full + qs,
                        self.n * 4,
                    )
                cute.copy(
                    tma_q,
                    tQgQ[(None, req, 0, 0)],
                    tQsQ[(None, qs)],
                    tma_bar_ptr=q_full + qs,
                )
                j += 1
                # The first item's preloaded tiles were issued before the wait.
                for t in cutlass.range(pre, ntile, unroll=1):
                    self.issue_tile(
                        t, it, my_page, blocks, kv_full, kv_empty, sc_empty, sSc,
                        scale_base, page_bytes, tma_k, tKgK, tKsK,
                    )
                    it += 1
                pre = Int32(0)
                has_item = n_nt > 0
        elif warp == self.mma_warp:
            tCrA = mma.make_fragment_A(sA)
            tCrB = mma.make_fragment_B(sB)
            while g < hi:
                _req, _k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                for _t in cutlass.range(ntile, unroll=1):
                    stage = it % self.stages
                    a = it % self.accs
                    cute.arch.mbarrier_wait(acc_empty + a, ((it // self.accs) & 1) ^ 1)
                    cute.arch.mbarrier_wait(kv_full + stage, (it // self.stages) & 1)
                    fence_tmem()  # after the epilogue's release
                    # Reconverge after the per-thread waits: cute.gemm and the commits below
                    # each elect one lane, and tcgen05.commit tracks only its own lane's MMAs.
                    cute.arch.sync_warp()
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(
                            mma,
                            tAcc[(None, None, None, a)],
                            tCrA[(None, None, kb, stage)],
                            tCrB[(None, None, kb, qs)],
                            tAcc[(None, None, None, a)],
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_empty + stage)
                        tcgen05.commit(acc_full + a)
                    it += 1
                with cute.arch.elect_one():
                    tcgen05.commit(q_empty + qs)
                j += 1
        else:
            # Epilogue: warpgroup w owns accumulator w // split and the w % split-th share of a
            # tile's rows.
            acc = warp // 4 // self.split
            part = warp // 4 % self.split
            lt = tidx % 128
            ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(HEADS)), Float32)
            acc_cols = cute.flat_divide(_mn_layout(tAcc), (TILE, HEADS))
            t2r = tcgen05.make_tmem_copy(ld_atom, acc_cols[(None, None, 0, 0, 0)])
            thr = t2r.get_slice(lt)
            tAll = thr.partition_S(acc_cols)
            cD = thr.partition_D(cute.make_identity_tensor((TILE, HEADS)))
            mid = -(-Q // self.split)
            while g < hi:
                req, k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                row0 = Int64(req * Q) * Int64(self.pitch)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                item = (k0, ntile, it, acc, lt, qs, row0, logits_address, tops_address)
                epi_bars = (kv_full, sc_empty, acc_full, acc_empty)
                views = (sSc, sW, sEnd, t2r, tAll, cD)
                if cutlass.const_expr(self.split == 1):
                    self.item_tiles(0, Q, item, epi_bars, views)
                elif part == 0:
                    self.item_tiles(0, mid, item, epi_bars, views)
                else:
                    self.item_tiles(mid, Q, item, epi_bars, views)
                it += ntile
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(q_empty + qs)
                j += 1

        cute.arch.sync_threads()
        if warp == self.mma_warp:
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# --------------------------------------------------------------------------------------------
# Host
# --------------------------------------------------------------------------------------------
def _architecture(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


def num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


#: The scorer's launch constants: its persistent grid is one CTA per SM (``num_sms``); it triggers
#: its PDL dependents right after its own wait (the select/pack waits for the whole grid before it
#: reads the logits). The served launch schedules and preloads complete key tiles before the wait.
EARLY_TRIGGER = True


@functools.lru_cache(maxsize=None)
def compile_score(
    rows: int,
    q: int,
    pitch: int,
    ctas: int,
    arch: str,
    early: bool = True,
    extra_options: str = "",
    tops_from: int = 0,
    early_schedule: bool = False,
    preload: bool = False,
):
    """The scorer of (rows, q) writing rows of ``pitch`` FP32 elements, on ``ctas`` CTAs
    (``extra_options``: more DSL compile options, e.g. to keep the cubin; ``tops_from`` and
    ``early_schedule``, ``preload``: see ``ScoreKernel``)."""
    kernel = ScoreKernel(rows, q, ctas, pitch, early, tops_from, early_schedule, preload)
    kv = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (cute.sym_int(), PAGE * TOKEN_BYTES),
        stride=(cute.sym_int64(), 1),
        assumed_align=16,
    )
    query = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (rows * HEADS, DIM, 1),
        stride=(DIM, 1, rows * HEADS * DIM),
        assumed_align=16,
    )
    weights = cute.runtime.make_fake_tensor(
        cutlass.Float32, (rows * HEADS,), stride=(1,), assumed_align=16
    )
    ends = cute.runtime.make_fake_tensor(cutlass.Int32, (rows,), stride=(1,), assumed_align=4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        kv,
        query,
        weights,
        ends,
        Int64(0),
        Int32(0),
        Int32(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        stream,
        options=f"--gpu-arch={arch} --enable-tvm-ffi {extra_options}".strip(),
    )


def unsupported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    row_lens: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
) -> str | None:
    """Why ``score_logits_pdl`` cannot run these arguments (None: it can)."""
    if q.dim() != 4 or q.dtype != torch.float8_e4m3fn or tuple(q.shape[2:]) != (HEADS, DIM):
        return f"query {tuple(q.shape)} {q.dtype}"
    requests, next_n = q.shape[0], q.shape[1]
    rows = requests * next_n
    if not 1 <= next_n <= 8 or requests > MAX_REQUESTS or not q.is_contiguous():
        return f"requests {requests} next_n {next_n}"
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.dim() != 4
        or tuple(kv_cache.shape[1:]) != (PAGE, 1, TOKEN_BYTES)
        or not kv_cache[0].is_contiguous()
        or kv_cache.data_ptr() % 16
        or kv_cache.stride(0) % 16
    ):
        return f"indexer cache {tuple(kv_cache.shape)} {tuple(kv_cache.stride())}"
    if (
        weights.dtype != torch.float32
        or not weights.is_contiguous()
        or weights.numel() != rows * HEADS
    ):
        return "weights"
    if weights.data_ptr() % 16:
        return "weights alignment"
    if row_lens.dtype != torch.int32 or row_lens.dim() != 1 or row_lens.shape[0] < rows:
        return "row lengths"
    if not row_lens.is_contiguous():
        return "row lengths layout"
    if block_table.dtype != torch.int32 or block_table.dim() != 2 or block_table.stride(1) != 1:
        return "block table"
    if block_table.shape[0] < requests:
        return "block table rows"
    if out.dtype != torch.float32 or out.dim() != 2 or out.stride(1) != 1 or out.shape[0] < rows:
        return "logits"
    return None


def score_logits_pdl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    row_lens: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    tops: torch.Tensor | None = None,
    tops_from: int = 1,
) -> None:
    """FP32 logits of every row's keys into ``out[t, :row_lens[t]]``, launched with PDL.

    q [B, next_n, 32, 128] FP8 (row t = b * next_n + i); kv_cache [blocks, 64, 1, 132] uint8 (the
    indexer K pool); weights [B * next_n, 32] FP32 (scaled); row_lens int32 [>= rows] (row t's
    local key count; request b scores the pages ``block_table[b, :ceil(max_i row_lens / 64)]``);
    block_table int32 [>= B, pages] with unit column stride; out FP32 [>= rows, pitch] with unit
    column stride. Callers check ``unsupported`` first. ``tops`` (uint32 [>= rows, pitch / 8], unit
    column stride): also each 8 keys' largest logit image (``ptx.ordered_bits``) at
    ``tops[t, key // 8]`` for every vector whose first key is below ``row_lens[t]``, on the rows of
    at least ``tops_from`` keys (the others' entries are not written).
    """
    requests, next_n = q.shape[0], q.shape[1]
    rows = requests * next_n
    device = q.device
    compiled = compile_score(
        rows,
        next_n,
        out.stride(0),
        num_sms(device),
        _architecture(device),
        EARLY_TRIGGER,
        tops_from=0 if tops is None else max(tops_from, 1),
        early_schedule=True,
        preload=True,
    )
    if tops is not None:
        assert out.stride(0) % 8 == 0 and tops.stride(0) * 8 == out.stride(0)
        assert tops.stride(1) == 1 and tops.shape[0] >= rows and tops.element_size() == 4
    query = q.view(torch.uint8).as_strided((rows * HEADS, DIM, 1), (DIM, 1, rows * HEADS * DIM))
    kv_flat = kv_cache.view(kv_cache.shape[0], PAGE * TOKEN_BYTES)
    compiled(
        kv_flat,
        query,
        weights.view(-1),
        row_lens[:rows],
        block_table.data_ptr(),
        block_table.stride(0),
        block_table.shape[1],
        kv_flat.data_ptr(),
        kv_flat.stride(0),
        out.data_ptr(),
        0 if tops is None else tops.data_ptr(),
    )
