# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe DSL kernel behind `cute_dsl_radix_topk_decode`: one CTA selects one decode row.

The selection is the trtllm `indexer_topk_decode` op's on its split path, stated over that
op's own order and padding:

* Scores are ranked by their fp32 bit pattern through the order-preserving map
  `key(x) = bits if the sign bit is set else ~bits & 0x7FFFFFFF`, ascending key first: a strict
  total order on bit patterns, +NaN above +inf, +0.0 above -0.0, -inf at the bottom but for
  -NaN. Two scores tie only when their bits are identical.
* The op cuts a row whose causal end is `end` into `blocks_per_row` blocks: the first
  `blocks_per_row - 1` hold `end // blocks_per_row` columns and the last the rest. A block of
  at most `top_k` columns passes all of them on, padded to `top_k` with entries scored
  -FLT_MAX that stand for column -1; a longer one passes its own top `top_k`. A merge takes
  the `top_k` best of what the blocks pass, which is the `top_k` best of the row's columns and
  `pad = sum(max(0, top_k - len))` pads.

Here the pads are virtual: `pad` entries at `key(-FLT_MAX)`, ranked below any column with
that key. A row with `end <= top_k` keeps, in column order, each column whose key is at most
`key(-FLT_MAX)`. A longer row is resolved by an exact radix select over the full 32-bit key,
digit by digit: bits 31..20 are counted over the row (read in 256-bit vectors) into a
4096-bin shared-memory histogram; the columns of the better bins are emitted and the
threshold bin's columns go to a shared-memory list, over which bits 19..10 and 9..0 are
resolved. A threshold bin too large for the list is narrowed digit by digit over the row
instead. Of the columns whose key equals the final threshold, the number still needed is
taken in no particular order, columns before pads. The selected columns are written first
and every selected pad as a -1 after them.

The op orders one value apart from its key: its first pass bins fp16(x), and the conversion
turns a negative NaN into the canonical positive NaN, the best score, while its later passes
rank it last by key. So a row holding a negative NaN takes `_emulate`, which runs the op's
stages as the op does: each block of more than `top_k` columns selects through the fp16 pass
when that pass settles it (its threshold bin holds at most 2048 entries) and by key
otherwise, and the merge does the same over the blocks' winners and the pads.

One launch, nothing outside shared memory, no host read: the row count is symbolic and
`blocks_per_row` is a launch argument the caller computes from the call's shape.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.utils import SmemAllocator

TOP_K = 2048
NUM_THREADS = 1024
NUM_WARPS = NUM_THREADS // 32
#: fp32 scores per 256-bit load in the first pass over the row.
VEC = 8
#: The row's select: the key's digits 31..20 (4096 bins), 19..10 and 9..0 (1024 bins each).
D1_SHIFT, D1_MASK, D1_BINS = 20, 0xFFF, 4096
D2_SHIFT, D2_MASK, D2_BINS = 10, 0x3FF, 1024
D3_MASK, D3_BINS = 0x3FF, 1024
#: Bins of the emulation path's digits (the op's): fp16, then 31..21, 20..10 and 9..0.
NUM_BINS = 2048
#: (key, column) slots for the columns that share the threshold prefix.
LIST_CAPACITY = 4096
#: key(-FLT_MAX): -FLT_MAX is 0xFF7FFFFF and negative, so its key is its bits.
PAD_KEY = 0xFF7FFFFF
#: The widest row the emulation path's column mask covers.
MAX_COLUMNS = 65536
#: The op's first-pass threshold bin is resolved by sorting it when it holds at most this many.
OP_FINAL_ITEMS = 2048
_FLT_MAX = 3.4028234663852886e38
#: key(-inf); a larger key is a negative NaN.
NEG_INF_KEY = 0xFF800000

# Scalar slots in shared memory.
_N_OUT = 0  # columns in the row's output buffer
_N_LIST = 1  # entries in the list
_N_TIE = 2  # tie-group columns claimed
_THR_BIN = 3  # threshold bin of the last scan
_THR_BEFORE = 4  # elements in the bins ranked above it
_THR_COUNT = 5  # elements in it
_N_NEG_NAN = 6  # columns holding a negative NaN
_NUM_SCALARS = 8


@cute.jit
def _key(x):
    """The order-preserving key of an fp32 score: ascending key is descending score."""
    bits = cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, x.ir_value()))
    key = bits
    if (bits & cutlass.Uint32(0x80000000)) == cutlass.Uint32(0):
        key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
    return key


@cute.jit
def _digit(key, shift: cutlass.Constexpr[int], mask: cutlass.Constexpr[int]):
    return cutlass.Int32((key >> shift) & cutlass.Uint32(mask))


@cute.jit
def _fp16_bin(x):
    """The op's first-pass bin of a score: its fp16 image's 11-bit ordered key.

    Only the emulation path reads it, to reproduce when the op's fp16 pass decides a
    selection. The conversion is the hardware's round-to-nearest one, which turns any NaN
    into the canonical positive NaN, so a -NaN lands in the top bin here."""
    bits = cutlass.Uint16(llvm.bitcast(cutlass.Uint16.mlir_type, x.to(cutlass.Float16).ir_value()))
    key = bits
    if (bits & cutlass.Uint16(0x8000)) == cutlass.Uint16(0):
        key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
    return cutlass.Int32((key >> 5) & cutlass.Uint16(0x7FF))


class RadixTopkDecodeKernel:
    """One CTA of NUM_THREADS threads per row; `next_n` is a compile-time constant."""

    def __init__(self, next_n: int):
        self.next_n = next_n
        self.T = NUM_THREADS
        self.W = NUM_WARPS

    # ------------------------------------------------------------------ helpers
    @cute.jit
    def _clear(self, s_hist, nbins: cutlass.Constexpr[int], tid):
        i = tid
        while i < cutlass.Int32(nbins):
            s_hist[i] = cutlass.Int32(0)
            i = i + cutlass.Int32(self.T)
        cute.arch.barrier()

    @cute.jit
    def _row_tiles(self, score, size, tid):
        """The row's first `size` scores (a multiple of VEC, 32-byte aligned) as per-thread
        256-bit vectors: the copy atom, the thread's source and coordinate partitions, and
        the tile count (`self.T * VEC` scores per tile)."""
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32 * VEC
        )
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((1, self.T), order=(1, 0)),
            cute.make_layout((1, VEC)),
        )
        row_ptr = cute.make_ptr(
            cutlass.Float32, score.iterator.toint(), cute.AddressSpace.gmem, assumed_align=32
        )
        row = cute.make_tensor(row_ptr, cute.make_layout((1, size), stride=(size, 1)))
        tiler = (1, self.T * VEC)
        g_row = cute.local_tile(row, tiler, (0, None))
        c_row = cute.local_tile(cute.make_identity_tensor((1, size)), tiler, (0, None))
        thr = tiled_copy.get_slice(tid)
        return copy_atom, thr.partition_S(g_row), thr.partition_S(c_row), g_row.shape[2]

    @cute.jit
    def _predicate(self, coords, limit):
        """Which of the thread's vectors in a tile start below `limit`."""
        pred = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    cute.size(coords, mode=[0, 1]),
                    cute.size(coords, mode=[1]),
                    cute.size(coords, mode=[2]),
                ),
                stride=(cute.size(coords, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(pred.shape[0]):
            for rest_k in range(pred.shape[2]):
                pred[rest_v, 0, rest_k] = cute.elem_less(coords[(0, rest_v), 0, rest_k][1], limit)
        return pred

    @cute.jit
    def _count(self, s_hist, b, n):
        cute.arch.atomic_add(ptr=(s_hist.iterator + b).llvm_ptr, val=n, sem="relaxed", scope="cta")

    @cute.jit
    def _warp_scan(self, x):
        """Inclusive prefix sum of `x` over the warp's lanes."""
        lane = cute.arch.lane_idx()
        for i in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(x, 1 << i, mask=0xFFFFFFFF)
            if lane >= cutlass.Int32(1 << i):
                x = x + up
        return x

    @cute.jit
    def _warp_offsets(self, s_warp, warp):
        """(sum of the warp totals in `s_warp` before `warp`, sum of all of them)."""
        lane = cute.arch.lane_idx()
        total = cutlass.Int32(0)
        if lane < cutlass.Int32(self.W):
            total = s_warp[lane]
        inclusive = self._warp_scan(total)
        before = cute.arch.shuffle_sync(inclusive, (warp + cutlass.Int32(31)) & cutlass.Int32(31))
        if warp == cutlass.Int32(0):
            before = cutlass.Int32(0)
        return before, cute.arch.shuffle_sync(inclusive, 31)

    @cute.jit
    def _claim(self, s_scal, slot: cutlass.Constexpr[int], contrib):
        """One position of the counter `s_scal[slot]` per lane with `contrib`, one atomic per
        warp. Every lane of the warp calls it; the result is meaningful where `contrib`."""
        ballot = cute.arch.vote_ballot_sync(contrib)
        pos = cutlass.Int32(0)
        if ballot != 0:
            lane = cute.arch.lane_idx()
            base = cutlass.Int32(0)
            if lane == cutlass.Int32(0):
                base = cute.arch.atomic_add(
                    ptr=(s_scal.iterator + slot).llvm_ptr,
                    val=cute.arch.popc(ballot),
                    sem="relaxed",
                    scope="cta",
                )
            below = (cutlass.Int32(1) << lane) - cutlass.Int32(1)
            pos = cute.arch.shuffle_sync(base, 0) + cute.arch.popc(ballot & below)
        return pos

    @cute.jit
    def _threshold(self, s_hist, s_warp, s_scal, nbins: cutlass.Constexpr[int], need, tid):
        """`(t, before, count)` of the bin with `before < need <= before + count`.

        The histogram's total is at least `need`, so exactly one bin qualifies. Each thread
        owns `nbins // self.T` consecutive bins; a warp shuffle scan and the warp totals
        give it its exclusive prefix. Two barriers."""
        per = cutlass.const_expr(nbins // self.T)
        lane = cute.arch.lane_idx()
        warp = tid // cutlass.Int32(32)
        base = tid * cutlass.Int32(per)
        local = cutlass.Int32(0)
        for j in cutlass.range_constexpr(per):
            local = local + s_hist[base + cutlass.Int32(j)]
        inclusive = self._warp_scan(local)
        if lane == cutlass.Int32(31):
            s_warp[warp] = inclusive
        cute.arch.barrier()
        prefix, _ = self._warp_offsets(s_warp, warp)
        prefix = prefix + inclusive - local
        for j in cutlass.range_constexpr(per):
            count = s_hist[base + cutlass.Int32(j)]
            if (prefix < need) & (prefix + count >= need):
                s_scal[_THR_BIN] = base + cutlass.Int32(j)
                s_scal[_THR_BEFORE] = prefix
                s_scal[_THR_COUNT] = count
            prefix = prefix + count
        cute.arch.barrier()
        return s_scal[_THR_BIN], s_scal[_THR_BEFORE], s_scal[_THR_COUNT]

    @cute.jit
    def _emit_tie(self, s_out, s_scal, col, share, take):
        """Write `col` for the first `take` lanes with `share` (counted over the CTA)."""
        slot = self._claim(s_scal, _N_TIE, share)
        win = share & (slot < take)
        pos = self._claim(s_scal, _N_OUT, win)
        if win:
            s_out[pos] = col

    # ------------------------------------------------------------------ row of end > top_k
    @cute.jit
    def _select(self, score, end, pad, s_hist, s_out, s_lkey, s_lcol, s_warp, s_scal, tid):
        """The top_k of the row's columns and `pad` pads, given the histogram of digit 31..20
        (pads included) in `s_hist`. Writes the selected columns to `s_out`; returns their
        count."""
        lane = cute.arch.lane_idx()
        warp = tid // cutlass.Int32(32)
        top_k = cutlass.Int32(TOP_K)
        pad_key = cutlass.Uint32(PAD_KEY)
        t1, before, count = self._threshold(s_hist, s_warp, s_scal, D1_BINS, top_k, tid)
        need = top_k - before  # still to take among the keys sharing `prefix`
        prefix = cutlass.Uint32(t1) << D1_SHIFT
        shift = cutlass.Int32(D1_SHIFT)  # `prefix` holds the threshold key's bits from `shift` up
        pad_in = cutlass.Int32(0)  # pads sharing `prefix`
        if _digit(pad_key, D1_SHIFT, D1_MASK) == t1:
            pad_in = pad

        if count - pad_in > cutlass.Int32(LIST_CAPACITY):
            # ---- digit 19..10 over the row, inside the prefix ------------------
            self._clear(s_hist, D2_BINS, tid)
            i = tid
            while i < end:
                key = _key(score[i])
                if (key >> D1_SHIFT) == (prefix >> D1_SHIFT):
                    self._count(s_hist, _digit(key, D2_SHIFT, D2_MASK), cutlass.Int32(1))
                i = i + cutlass.Int32(self.T)
            if (tid == cutlass.Int32(0)) & (pad_in > cutlass.Int32(0)):
                self._count(s_hist, _digit(pad_key, D2_SHIFT, D2_MASK), pad_in)
            cute.arch.barrier()
            t2, before, count = self._threshold(s_hist, s_warp, s_scal, D2_BINS, need, tid)
            need = need - before
            prefix = prefix | (cutlass.Uint32(t2) << D2_SHIFT)
            shift = cutlass.Int32(D2_SHIFT)
            if _digit(pad_key, D2_SHIFT, D2_MASK) != t2:
                pad_in = cutlass.Int32(0)
            if count - pad_in > cutlass.Int32(LIST_CAPACITY):
                # ---- digit 9..0 over the row, inside the prefix ----------------
                self._clear(s_hist, D3_BINS, tid)
                i = tid
                while i < end:
                    key = _key(score[i])
                    if (key >> D2_SHIFT) == (prefix >> D2_SHIFT):
                        self._count(s_hist, _digit(key, 0, D3_MASK), cutlass.Int32(1))
                    i = i + cutlass.Int32(self.T)
                if (tid == cutlass.Int32(0)) & (pad_in > cutlass.Int32(0)):
                    self._count(s_hist, _digit(pad_key, 0, D3_MASK), pad_in)
                cute.arch.barrier()
                t3, before, count = self._threshold(s_hist, s_warp, s_scal, D3_BINS, need, tid)
                need = need - before
                prefix = prefix | cutlass.Uint32(t3)
                shift = cutlass.Int32(0)
                if _digit(pad_key, 0, D3_MASK) != t3:
                    pad_in = cutlass.Int32(0)

        # ---- one pass over the row: emit what ranks above the prefix ----------------
        # With the threshold key known (`shift == 0`) its tie group is taken here, columns
        # before pads; otherwise the columns sharing the prefix go to the list.
        take = need
        if count - pad_in < take:
            take = count - pad_in
        wbase = warp * cutlass.Int32(32)
        while wbase < end:
            c = wbase + lane
            live = c < end
            key = pad_key
            if live:
                key = _key(score[c])
            above = live & ((key >> shift) < (prefix >> shift))
            share = live & ((key >> shift) == (prefix >> shift))
            pos = self._claim(s_scal, _N_OUT, above)
            if above:
                s_out[pos] = c
            if shift == cutlass.Int32(0):
                self._emit_tie(s_out, s_scal, c, share, take)
            else:
                slot = self._claim(s_scal, _N_LIST, share)
                if share:
                    s_lkey[slot] = key
                    s_lcol[slot] = c
            wbase = wbase + cutlass.Int32(self.T)
        cute.arch.barrier()

        if shift != cutlass.Int32(0):
            n_list = s_scal[_N_LIST]
            if shift == cutlass.Int32(D1_SHIFT):
                # ---- digit 19..10 over the list --------------------------------
                self._clear(s_hist, D2_BINS, tid)
                j = tid
                while j < n_list:
                    self._count(s_hist, _digit(s_lkey[j], D2_SHIFT, D2_MASK), cutlass.Int32(1))
                    j = j + cutlass.Int32(self.T)
                if (tid == cutlass.Int32(0)) & (pad_in > cutlass.Int32(0)):
                    self._count(s_hist, _digit(pad_key, D2_SHIFT, D2_MASK), pad_in)
                cute.arch.barrier()
                t2, before, count = self._threshold(s_hist, s_warp, s_scal, D2_BINS, need, tid)
                need = need - before
                prefix = prefix | (cutlass.Uint32(t2) << D2_SHIFT)
                if _digit(pad_key, D2_SHIFT, D2_MASK) != t2:
                    pad_in = cutlass.Int32(0)
            # ---- digit 9..0 over the list ------------------------------------------
            self._clear(s_hist, D3_BINS, tid)
            j = tid
            while j < n_list:
                key = s_lkey[j]
                if (key >> D2_SHIFT) == (prefix >> D2_SHIFT):
                    self._count(s_hist, _digit(key, 0, D3_MASK), cutlass.Int32(1))
                j = j + cutlass.Int32(self.T)
            if (tid == cutlass.Int32(0)) & (pad_in > cutlass.Int32(0)):
                self._count(s_hist, _digit(pad_key, 0, D3_MASK), pad_in)
            cute.arch.barrier()
            t3, before, count = self._threshold(s_hist, s_warp, s_scal, D3_BINS, need, tid)
            need = need - before
            thr = prefix | cutlass.Uint32(t3)
            if _digit(pad_key, 0, D3_MASK) != t3:
                pad_in = cutlass.Int32(0)
            take = need
            if count - pad_in < take:
                take = count - pad_in
            # Above the threshold key: selected; equal to it: `take` of them.
            lbase = warp * cutlass.Int32(32)
            while lbase < n_list:
                j = lbase + lane
                live = j < n_list
                key = thr
                col = cutlass.Int32(-1)
                if live:
                    key = s_lkey[j]
                    col = s_lcol[j]
                above = live & (key < thr)
                share = live & (key == thr)
                pos = self._claim(s_scal, _N_OUT, above)
                if above:
                    s_out[pos] = col
                self._emit_tie(s_out, s_scal, col, share, take)
                lbase = lbase + cutlass.Int32(self.T)
            cute.arch.barrier()
        return s_scal[_N_OUT]

    # ------------------------------------------------------------------ emulation
    @cute.jit
    def _in_mask(self, s_mask, c):
        word = s_mask[c >> cutlass.Int32(5)]
        return ((word >> cutlass.Uint32(c & cutlass.Int32(31))) & cutlass.Uint32(1)) != 0

    @cute.jit
    def _mark_range(self, s_mask, lo, hi, tid):
        """Set the mask bits of columns [lo, hi), one word per thread at a time."""
        w = (lo >> cutlass.Int32(5)) + tid
        while w * cutlass.Int32(32) < hi:
            first = w * cutlass.Int32(32)
            a = lo - first
            if a < cutlass.Int32(0):
                a = cutlass.Int32(0)
            b = hi - first
            if b > cutlass.Int32(32):
                b = cutlass.Int32(32)
            upper = cutlass.Uint32(0xFFFFFFFF)
            if b < cutlass.Int32(32):
                upper = (cutlass.Uint32(1) << cutlass.Uint32(b)) - cutlass.Uint32(1)
            lower = (cutlass.Uint32(1) << cutlass.Uint32(a)) - cutlass.Uint32(1)
            s_mask[w] = s_mask[w] | (upper & (lower ^ cutlass.Uint32(0xFFFFFFFF)))
            w = w + cutlass.Int32(self.T)
        cute.arch.barrier()

    @cute.jit
    def _stage(
        self,
        score,
        lo,
        hi,
        use_mask: cutlass.Constexpr[bool],
        pad,
        emit: cutlass.Constexpr[bool],
        s_hist,
        s_out,
        s_mask,
        s_warp,
        s_scal,
        tid,
    ):
        """One selection of the op, as its kernel makes it: the top_k of the columns in
        [lo, hi) (with `use_mask`, only those whose mask bit is set) and `pad` pads.

        Its first pass bins every candidate by `_fp16_bin` and finds the threshold bin. When
        that bin holds at most OP_FINAL_ITEMS candidates the op takes every better bin whole
        and sorts the threshold bin by key; otherwise it selects by key over all candidates.
        Either way the key select runs here as three digit passes over the row. The winners
        are set in `s_mask` (`emit` False) or written to `s_out` (`emit` True)."""
        lane = cute.arch.lane_idx()
        warp = tid // cutlass.Int32(32)
        top_k = cutlass.Int32(TOP_K)
        pad_key = cutlass.Uint32(PAD_KEY)
        pad_bin = _fp16_bin(cutlass.Float32(-_FLT_MAX))
        if tid == cutlass.Int32(0):
            s_scal[_N_TIE] = cutlass.Int32(0)
        self._clear(s_hist, NUM_BINS, tid)
        c = lo + tid
        while c < hi:
            ok = cutlass.Boolean(True)
            if cutlass.const_expr(use_mask):
                ok = self._in_mask(s_mask, c)
            if ok:
                self._count(s_hist, _fp16_bin(score[c]), cutlass.Int32(1))
            c = c + cutlass.Int32(self.T)
        if (tid == cutlass.Int32(0)) & (pad > cutlass.Int32(0)):
            self._count(s_hist, pad_bin, pad)
        cute.arch.barrier()
        t0, before, count = self._threshold(s_hist, s_warp, s_scal, NUM_BINS, top_k, tid)
        resolved = count <= cutlass.Int32(OP_FINAL_ITEMS)
        need = top_k
        pad_in = pad
        if resolved:
            need = top_k - before
            if pad_bin != t0:
                pad_in = cutlass.Int32(0)

        prefix = cutlass.Uint32(0)
        for d in cutlass.range_constexpr(3):
            shift = cutlass.const_expr((21, 10, 0)[d])
            mask = cutlass.const_expr((0x7FF, 0x7FF, 0x3FF)[d])
            nbins = cutlass.const_expr((NUM_BINS, NUM_BINS, NUM_BINS // 2)[d])
            self._clear(s_hist, nbins, tid)
            c = lo + tid
            while c < hi:
                ok = cutlass.Boolean(True)
                if cutlass.const_expr(use_mask):
                    ok = self._in_mask(s_mask, c)
                if ok:
                    x = score[c]
                    key = _key(x)
                    if resolved:
                        ok = _fp16_bin(x) == t0
                    if cutlass.const_expr(d > 0):
                        above_shift = cutlass.const_expr((21, 10)[d - 1])
                        ok = ok & ((key >> above_shift) == (prefix >> above_shift))
                    if ok:
                        self._count(s_hist, _digit(key, shift, mask), cutlass.Int32(1))
                c = c + cutlass.Int32(self.T)
            if (tid == cutlass.Int32(0)) & (pad_in > cutlass.Int32(0)):
                self._count(s_hist, _digit(pad_key, shift, mask), pad_in)
            cute.arch.barrier()
            t, before, count = self._threshold(s_hist, s_warp, s_scal, nbins, need, tid)
            need = need - before
            prefix = prefix | (cutlass.Uint32(t) << shift)
            if _digit(pad_key, shift, mask) != t:
                pad_in = cutlass.Int32(0)
        take = need
        if count - pad_in < take:
            take = count - pad_in

        wbase = (lo & cutlass.Int32(-32)) + warp * cutlass.Int32(32)
        while wbase < hi:
            c = wbase + lane
            live = (c >= lo) & (c < hi)
            if cutlass.const_expr(use_mask):
                live = live & self._in_mask(s_mask, c)
            key = pad_key
            h = cutlass.Int32(0)
            if live:
                x = score[c]
                key = _key(x)
                h = _fp16_bin(x)
            inbin = live
            above = live & (key < prefix)
            if resolved:
                inbin = live & (h == t0)
                above = (live & (h < t0)) | (inbin & (key < prefix))
            tie = inbin & (key == prefix)
            slot = self._claim(s_scal, _N_TIE, tie)
            chosen = above | (tie & (slot < take))
            if cutlass.const_expr(emit):
                pos = self._claim(s_scal, _N_OUT, chosen)
                if chosen:
                    s_out[pos] = c
            else:
                bits = cute.arch.vote_ballot_sync(chosen)
                if lane == cutlass.Int32(0):
                    s_mask[wbase >> cutlass.Int32(5)] = s_mask[wbase >> cutlass.Int32(5)] | bits
            wbase = wbase + cutlass.Int32(self.T)
        cute.arch.barrier()

    @cute.jit
    def _emulate(self, score, end, blocks_per_row, pad, s_hist, s_out, s_mask, s_warp, s_scal, tid):
        """The op's two launches, stage by stage, for a row holding a negative NaN.

        That is the one value whose rank the op's fp16 first pass changes (the conversion
        drops its sign), so only such a row needs this path. Each block of more than top_k
        columns selects its top_k as the op's first launch does; a shorter one passes all its
        columns. The merge then selects over those and the pads. Returns the columns written."""
        top_k = cutlass.Int32(TOP_K)
        w = tid
        while w < cutlass.Int32(MAX_COLUMNS // 32):
            s_mask[w] = cutlass.Uint32(0)
            w = w + cutlass.Int32(self.T)
        cute.arch.barrier()
        block = end // blocks_per_row
        j = cutlass.Int32(0)
        while j < blocks_per_row:
            lo = j * block
            hi = lo + block
            if j == blocks_per_row - cutlass.Int32(1):
                hi = end
            if hi - lo <= top_k:
                self._mark_range(s_mask, lo, hi, tid)
            else:
                self._stage(
                    score, lo, hi, False, cutlass.Int32(0), False,
                    s_hist, s_out, s_mask, s_warp, s_scal, tid,
                )  # fmt: skip
            j = j + cutlass.Int32(1)
        self._stage(score, cutlass.Int32(0), end, True, pad, True, s_hist, s_out, s_mask, s_warp,
                    s_scal, tid)  # fmt: skip
        return s_scal[_N_OUT]

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        seq_lens: cute.Tensor,
        out: cute.Tensor,
        blocks_per_row: cutlass.Int32,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = tid // cutlass.Int32(32)
        next_n = cutlass.const_expr(self.next_n)
        top_k = cutlass.Int32(TOP_K)
        pad_key = cutlass.Uint32(PAD_KEY)
        neg_inf_key = cutlass.Uint32(NEG_INF_KEY)

        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((D1_BINS,)), 128)
        s_out = smem.allocate_tensor(cutlass.Int32, cute.make_layout((TOP_K,)), 128)
        s_lkey = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((LIST_CAPACITY,)), 128)
        s_lcol = smem.allocate_tensor(cutlass.Int32, cute.make_layout((LIST_CAPACITY,)), 128)
        s_mask = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((MAX_COLUMNS // 32,)), 128)
        s_warp = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.W,)), 128)
        s_scal = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_NUM_SCALARS,)), 128)

        # The causal end as the op computes it (compress ratio 1), held inside the row.
        cols = cute.size(logits, mode=[1])
        end = seq_lens[row // cutlass.Int32(next_n)] - cutlass.Int32(next_n)
        end = end + row % cutlass.Int32(next_n) + cutlass.Int32(1)
        if end > cols:
            end = cols
        if end < cutlass.Int32(0):
            end = cutlass.Int32(0)
        score = logits[row, None]
        dst = out[row, None]

        # The op's pad entries: each of its blocks of at most top_k columns pads to top_k.
        block = end // blocks_per_row
        last = end - block * (blocks_per_row - cutlass.Int32(1))
        pad = cutlass.Int32(0)
        if block < top_k:
            pad = (blocks_per_row - cutlass.Int32(1)) * (top_k - block)
        if last < top_k:
            pad = pad + top_k - last

        if tid < cutlass.Int32(_NUM_SCALARS):
            s_scal[tid] = cutlass.Int32(0)
        cute.arch.barrier()

        # Each path's first pass also counts the negative NaNs; a row with one goes through
        # the op's stages instead (`_emulate`).
        n_sel = cutlass.Int32(0)
        emulate = cutlass.Boolean(False)
        if end <= top_k:
            # At least top_k pads: every column whose key is at most key(-FLT_MAX) is kept,
            # in column order (`per` consecutive columns per thread); a column below it
            # loses to a pad.
            per = cutlass.const_expr(TOP_K // self.T)
            c0 = tid * cutlass.Int32(per)
            keep = cute.make_rmem_tensor(cute.make_layout((per,)), cutlass.Int32)
            neg = cutlass.Int32(0)
            local = cutlass.Int32(0)
            for e in cutlass.range_constexpr(per):
                keep[e] = cutlass.Int32(0)
                if c0 + cutlass.Int32(e) < end:
                    key = _key(score[c0 + cutlass.Int32(e)])
                    if key <= pad_key:
                        keep[e] = cutlass.Int32(1)
                    if key > neg_inf_key:
                        neg = neg + cutlass.Int32(1)
                local = local + keep[e]
            if neg > cutlass.Int32(0):
                self._count(s_scal, cutlass.Int32(_N_NEG_NAN), neg)
            inclusive = self._warp_scan(local)
            if lane == cutlass.Int32(31):
                s_warp[warp] = inclusive
            cute.arch.barrier()
            if s_scal[_N_NEG_NAN] == cutlass.Int32(0):
                pos0, total = self._warp_offsets(s_warp, warp)
                pos0 = pos0 + inclusive - local
                for e in cutlass.range_constexpr(per):
                    if keep[e] == cutlass.Int32(1):
                        s_out[pos0] = c0 + cutlass.Int32(e)
                    pos0 = pos0 + keep[e]
                n_sel = total
            else:
                emulate = cutlass.Boolean(True)
        else:
            # ---- digit 31..20 over the row -----------------------------------
            self._clear(s_hist, D1_BINS, tid)
            neg = cutlass.Int32(0)
            size = (end + cutlass.Int32(VEC - 1)) // cutlass.Int32(VEC) * cutlass.Int32(VEC)
            atom, t_src, t_crd, tiles = self._row_tiles(score, size, tid)
            t_val = cute.make_fragment_like(t_src[None, None, None, 0])
            for tile in range(tiles):
                pred = self._predicate(t_crd[None, None, None, tile], size)
                cute.copy(atom, t_src[None, None, None, tile], t_val, pred=pred[None, None, None])
                crd = t_crd[(0, None), None, None, tile]
                for v in cutlass.range(cute.size(t_val), unroll_full=True):
                    c = crd[v // VEC][1] + v % VEC
                    if pred[0, v // VEC, 0] & (c < end):
                        key = _key(t_val[v])
                        self._count(s_hist, _digit(key, D1_SHIFT, D1_MASK), cutlass.Int32(1))
                        if key > neg_inf_key:
                            neg = neg + cutlass.Int32(1)
            if neg > cutlass.Int32(0):
                self._count(s_scal, cutlass.Int32(_N_NEG_NAN), neg)
            if (tid == cutlass.Int32(0)) & (pad > cutlass.Int32(0)):
                self._count(s_hist, _digit(pad_key, D1_SHIFT, D1_MASK), pad)
            cute.arch.barrier()
            if s_scal[_N_NEG_NAN] != cutlass.Int32(0):
                emulate = cutlass.Boolean(True)
            else:
                n_sel = self._select(
                    score, end, pad, s_hist, s_out, s_lkey, s_lcol, s_warp, s_scal, tid
                )
        if emulate:
            n_sel = self._emulate(
                score, end, blocks_per_row, pad, s_hist, s_out, s_mask, s_warp, s_scal, tid
            )
        cute.arch.barrier()

        # The selected columns, then -1 for every selected pad.
        k = tid
        while k < top_k:
            v = cutlass.Int32(-1)
            if k < n_sel:
                v = s_out[k]
            dst[k] = v
            k = k + cutlass.Int32(self.T)

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        seq_lens: cute.Tensor,
        out: cute.Tensor,
        blocks_per_row: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        rows = cute.size(logits, mode=[0])
        self.kernel(logits, seq_lens, out, blocks_per_row).launch(
            grid=(rows, 1, 1), block=(self.T, 1, 1), stream=stream
        )
