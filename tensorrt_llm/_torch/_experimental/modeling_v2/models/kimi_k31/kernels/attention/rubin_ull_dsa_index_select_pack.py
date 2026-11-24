# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-shard local Top-L and pack in one CuTe DSL kernel (one CTA per decode row).

Replaces ``cute_dsl_indexer_topk_decode`` (local Top-L into ``key_shard_local_topk``) followed by
``KeyShardTopKMerge.pack`` (``_ks_pack_kernel``) of ``attention/backends/sparse/dsa/key_shard.py``.
The kernel writes exactly what the pack writes, in the merge workspace's layout and protocol:

- the row's ``L`` candidate keys, in ascending local position, ``INT64_MIN`` padded, into receive
  slot ``e & 1`` of every rank (``e`` = merge epoch + 1);
- the smallest sent key, the sent count and the row's local key count into every rank's header;
- after a CTA barrier, the row's flag ``e`` on every rank with a system-scope release.

Candidates: the exact Top-``L`` of the row's ``n`` local scores by ``(score desc, local index asc)``
on the order-preserving bit image of the FP32 score (``-0.0`` below ``+0.0``), i.e. the merge's own
key order restricted to this rank. Every exact Top-L by score gives the same merge output (the
merge re-selects a row from the local scores whenever a rank's unsent keys could win), so the
global Top-K is unchanged; with this tie rule the sent keys are also deterministic.

Work is bounded by the row's length ``n``, not by the workspace width. A row is read in chunks of
``threads x per`` scores, each thread holding consecutive scores; the first chunk stays in
registers, later chunks (rows longer than one chunk) are read again in every pass. A row within
one chunk runs on the fewest of 256, 512, ... threads holding it with at most 16 scores each (so a
wide workspace costs short rows nothing); the CTA's barriers are a named barrier over them.

Selection: MSB-first radix select over the 32-bit images (rounds of 11, 11 and 10 bits with
shared-memory histograms accumulated over the chunks; a round whose boundary bucket is taken whole
ends the search), then per chunk one scan of the per-thread (greater, equal) counts, which yields
each score's output position in index order and the tie budget (both counts are recomputed from
the images, not held: 1024-thread CTAs have 64 registers per thread).

The kernel waits (``griddepcontrol.wait``) for the logits grid before any load and triggers its
dependents (the merge) right after, as the pack does.

With ``top_k`` the same CTA then merges its row as ``_ks_merge_row`` does (``merge_row``): it waits for
the row's flags of every rank, selects the global Top-K of the received keys, checks the certificate,
re-selects from every rank's local scores when it fails, writes this rank's winners, count and
combine mask, and the last CTA advances the epoch; dependents then launch at the grid's end.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64

from ..common import rubin_ull_spin_trap as spin_trap
from . import rubin_ull_dsa_index_ptx as ptx

MAX_THREADS = 1024
#: Scores per thread in registers: up to 32 below 1024 threads, 16 at 1024 (64 registers per thread).
MAX_PER_THREAD = 32
WIDE_PER = 16
#: Warps of the smallest CTA; the histogram, its scans and the push run on these 256 threads.
MIN_WARPS = 8
TOKENS_PER_BLOCK = 64
#: Candidates (scores at or above the radix prefix) a row's later rounds and staging read from a
#: shared list once they fit, instead of the whole row.
LIST_KEYS = 4096
#: Rows up to this many keys stay on the registers (the list's copy costs more than it saves).
LIST_MIN_KEYS = 4096
SPIN_SCALE = 32
#: Timing probe points: start (after the logits), published, flags seen, threshold, written,
#: first chunk loaded, radix select done, keys staged. Before the exchange overwrites them, points
#: 1-3 hold the radix rounds' ends and 4 the candidate list's (select-only runs read them there).
TRACE_POINTS = 8

_INT64_MIN = 1 << 63
_INT64_MAX = (1 << 63) - 1

#: Radix rounds, most significant first (bits of a 32-bit word per round; a 64-bit merge key
#: takes them twice, high word first).
_ROUND_BITS = (11, 11, 10)
_HIST_WORDS = 1 << max(_ROUND_BITS)
_BASE_THREADS = 32 * MIN_WARPS

# Shared-memory layout (bytes).
_HIST = 0  # _HIST_WORDS x u32
_WSUM = _HIST + 4 * _HIST_WORDS  # 2 x 32 x u32 (per-warp scan totals, double-buffered)
_CTL = _WSUM + 256  # [bucket, above, count]
_WMIN = _CTL + 64  # 32 x u64 (per-warp minimum sent key)
# merge: [sent total, key total, failed ranks, block total, min selected high word]
_MCTL = _WMIN + 256
_CNT = _MCTL + 64  # merge: 32 x u32 (every rank's local key count)
_STAGE = _CNT + 128  # L x u64


def _max_per(threads: int) -> int:
    return WIDE_PER if threads >= MAX_THREADS else MAX_PER_THREAD


def launch_shape(local_width: int, threads: int | None = None, per: int | None = None):
    """(threads, scores per thread per chunk) for a workspace ``local_width`` keys wide.

    Default (fastest on GB300 under CUDA graphs): 256 threads with up to 8 scores each for up to
    2048 keys, else 512 threads with up to 16 (wider rows take several chunks).
    ``threads`` / ``per`` force a shape (benchmarks)."""

    def per_of(t):
        return min(_max_per(t), max(4, -(-(-(-local_width // t)) // 4) * 4))

    if threads:
        shape = (threads, per_of(threads))
    elif local_width <= 256 * 8:
        shape = (256, per_of(256))
    else:
        shape = (512, min(16, per_of(512)))
    if per:
        shape = (shape[0], per)
    return shape


def _digit(r: int, hi, lo, valid, prefix_hi, prefix_lo):
    """(active, digit) of a 64-bit key (``hi``, ``lo`` words of its unsigned image) in round ``r``
    of the merge's radix select: rounds 0-2 take the high word, 3-5 the low word."""
    bits = _ROUND_BITS[r % 3]
    before = sum(_ROUND_BITS[: r % 3])
    active = valid
    if r < 3:
        word = hi
        if before > 0:
            active = active & ptx.eq_u32(hi >> Uint32(32 - before), prefix_hi)
    else:
        word = lo
        active = active & ptx.eq_u32(hi, prefix_hi)
        if before > 0:
            active = active & ptx.eq_u32(lo >> Uint32(32 - before), prefix_lo)
    digit = (word >> Uint32(32 - before - bits)) & Uint32((1 << bits) - 1)
    return active, digit


#: ``kernels.spin_trap`` sites of the key-shard waits, a / b / c =
#: row / thread / epoch: the flag merge's peer-flag wait (``wait_flag_sys_if``), the local Top-k's
#: own LL slots, the exchange merge's received keys and header, the rescue's pushed keys
#: (``settle_ll_if``).
SITE_FLAGS, SITE_OWN, SITE_MERGE, SITE_RESCUE = 171, 172, 173, 174


class KeyShardSelectPackKernel:
    """``world`` ranks, ``local_k`` candidates per rank (a power of two), chunks of ``threads`` x
    ``per`` scores. ``top_k`` > 0: the CTA also merges its row (``merge_row``)."""

    def __init__(
        self,
        world: int,
        local_k: int,
        threads: int,
        per: int,
        top_k: int = 0,
        bounds64: bool = False,
        mask: bool = False,
        trace: bool = False,
    ):
        assert world & (world - 1) == 0 and 1 < world <= 32
        assert local_k & (local_k - 1) == 0 and 2 <= local_k <= 4096
        assert per % 4 == 0 and 0 < per <= _max_per(threads)
        assert threads in (256, 512, 1024) and (world * local_k // 2) % threads == 0
        # Per-chunk (greater, equal) counts are packed into 16-bit halves.
        assert threads * per < (1 << 16)
        assert world * local_k // 2 >= _BASE_THREADS
        assert top_k % threads == 0
        self.world, self.local_k, self.threads, self.per = world, local_k, threads, per
        self.top_k, self.bounds64, self.mask = top_k, bounds64, mask
        # Timing probe: per row, (globaltimer, clock64) of thread 0 at TRACE_POINTS points.
        self.trace = trace
        # ``kernels.spin_trap`` record address (0: a timed-out wait is counted in ``state[3]``
        # and the kernel goes on; else it also writes the trap record and traps).
        self.trap_record = 0
        # Candidate list: LIST_KEYS images, then their row indices.
        self.list_offset = _STAGE + 8 * local_k
        self.smem_bytes = self.list_offset + 8 * LIST_KEYS

    @cute.jit
    def __call__(
        self,
        state: cute.Tensor,
        logits_addr: Int64,
        logits_stride: Int32,
        rowlen_addr: Int64,
        keys_addr: Int64,
        i32_addr: Int64,
        dk_addr: Int64,
        d32_addr: Int64,
        rank: Int32,
        rows_max: Int32,
        rows: Int32,
        out_addr: Int64,
        out_stride: Int32,
        counts_addr: Int64,
        bounds_addr: Int64,
        mask_addr: Int64,
        max_spin: Int32,
        trace_addr: Int64,
        stream,
    ):
        self.kernel(
            state,
            logits_addr,
            logits_stride,
            rowlen_addr,
            keys_addr,
            i32_addr,
            dk_addr,
            d32_addr,
            rank,
            rows_max,
            rows,
            out_addr,
            out_stride,
            counts_addr,
            bounds_addr,
            mask_addr,
            max_spin,
            trace_addr,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def trap_late(self, late, site: cutlass.Constexpr[int], row, tid, epoch):
        """With a trap record: a thread whose wait timed out (``late``) records ``site`` and
        traps (``kernels.spin_trap.expired``)."""
        if cutlass.const_expr(self.trap_record != 0):
            spin_trap.expired(late, site, self.trap_record, row, tid, epoch)

    @cute.jit
    def trap_late2(
        self,
        late_a,
        site_a: cutlass.Constexpr[int],
        late_b,
        site_b: cutlass.Constexpr[int],
        row,
        tid,
        epoch,
    ):
        """``trap_late`` for two waits of a kernel at one trap point: the record's site is
        ``site_a`` if that wait timed out, else ``site_b``."""
        if cutlass.const_expr(self.trap_record != 0):
            site = ptx.sel(late_a, Uint32(site_a), Uint32(site_b))
            spin_trap.expired_at(late_a | late_b, site, self.trap_record, row, tid, epoch)

    @cute.jit
    def stamp(self, trace_addr, row, tid, point: cutlass.Constexpr[int]):
        if cutlass.const_expr(self.trace):
            ptx.stamp_if(
                ptx.eq_u32(tid, 0),
                Uint64(trace_addr)
                + (Uint64(row) * Uint64(TRACE_POINTS) + Uint64(point)) * Uint64(16),
            )

    def bar(self, warps):
        """Barrier over the CTA's first ``warps`` warps (the ones working on the row)."""
        ptx.bar_sync(Uint32(warps) * Uint32(32))

    @cute.jit
    def block_scan(self, value, lane, warp, wsum, warps, limit: cutlass.Constexpr[int]):
        """(inclusive scan of ``value`` over the first ``warps`` warps, their total), counting
        the totals of warps below ``limit`` only. Callers barrier before ``wsum`` is reused."""
        for offset in cutlass.range_constexpr(5):
            value = ptx.scan_step(value, 1 << offset)
        ptx.st_shared_u32_if(ptx.eq_u32(lane, 31), wsum + warp * 4, value)
        self.bar(warps)
        before = Uint32(0)
        total = Uint32(0)
        for w in cutlass.range_constexpr(limit):
            x = ptx.ld_shared_u32_if(ptx.lt_u32(w, warps), wsum + 4 * w)
            before = before + ptx.sel(ptx.lt_u32(w, warp), x, 0)
            total = total + x
        return value + before, total

    @cute.jit
    def clear_hist(self, tid, hist):
        for i in cutlass.range_constexpr(_HIST_WORDS // _BASE_THREADS):
            b = tid + Uint32(i * _BASE_THREADS)
            ptx.st_shared_u32_if(ptx.lt_u32(tid, _BASE_THREADS), hist + b * Uint32(4), 0)

    @cute.jit
    def resolve(
        self, tid, lane, warp, hist, wsum, ctl, remaining, bits: cutlass.Constexpr[int], warps
    ):
        """End of a radix round whose counts are in ``hist``: (boundary bucket, keys in higher
        buckets, keys in it) for the ``remaining`` largest active keys; leaves ``hist`` zeroed (it
        zeroes the round's 2^bits bins; the bins above are zero already, as rounds count below
        2^bits). Threads below 256 own the bins; ``warps`` warps take part."""
        bins = 1 << bits
        self.bar(warps)
        # Thread t < 256 owns the C words from bins-C*(t+1) up, bins bins-1-(t*C .. t*C+C-1)
        # descending: read with 16-byte loads (one word per thread and load would be an 8-way bank
        # conflict at C = 8), then zeroed for the next round (only the owner reads them).
        C = bins // _BASE_THREADS
        assert C % 4 == 0
        owner = ptx.lt_u32(tid, _BASE_THREADS)
        first = hist + (Uint32(bins - C) - tid * Uint32(C)) * Uint32(4)
        words = []
        for q in cutlass.range_constexpr(C // 4):
            quad = ptx.ld_shared_v4_if(owner, first + Uint32(16 * q))
            for k in cutlass.range_constexpr(4):
                words.append(quad[k])
        for q in cutlass.range_constexpr(C // 4):
            ptx.zero_shared_v4_if(owner, first + Uint32(16 * q))
        counts = []
        total = Uint32(0)
        for i in cutlass.range_constexpr(C):
            counts.append(words[C - 1 - i])
            total = total + words[C - 1 - i]
        inclusive, _ = self.block_scan(total, lane, warp, wsum, warps, MIN_WARPS)
        # The boundary bin (at most one thread holds it), stored once.
        running = inclusive - total
        held = Uint32(0)
        bucket = Uint32(0)
        higher = Uint32(0)
        inside = Uint32(0)
        for i in cutlass.range_constexpr(C):
            above = running
            running = running + counts[i]
            hit = (
                owner & ptx.lt_u32(above, remaining) & (Uint32(1) - ptx.lt_u32(running, remaining))
            )
            bucket = ptx.sel(hit, Uint32(bins - 1) - (tid * Uint32(C) + Uint32(i)), bucket)
            higher = ptx.sel(hit, above, higher)
            inside = ptx.sel(hit, counts[i], inside)
            held = held | hit
        ptx.st_shared_u32_if(held, ctl, bucket)
        ptx.st_shared_u32_if(held, ctl + Uint32(4), higher)
        ptx.st_shared_u32_if(held, ctl + Uint32(8), inside)
        self.bar(warps)
        return (
            ptx.ld_shared_u32(ctl),
            ptx.ld_shared_u32(ctl + Uint32(4)),
            ptx.ld_shared_u32(ctl + Uint32(8)),
        )

    def load_chunk(self, row_base, start, slots):
        """Images of this thread's ``slots`` consecutive scores from row index ``start`` (a
        multiple of 4), in ``per`` registers (the rest 0)."""
        images = []
        for q in range(self.per // 4):
            words = ptx.ld_global_v4_if(
                ptx.lt_u32(4 * q, slots), row_base + Uint64(start + Uint32(4 * q)) * Uint64(4)
            )
            for k in range(4):
                images.append(ptx.ordered_bits(words[k]))
        return images

    @cute.jit
    def list_chunk(
        self,
        images,
        start,
        slots,
        cut,
        prefix,
        listed,
        lane,
        warp,
        wsum,
        warps,
        lst,
    ):
        """Append a chunk's candidates (images whose top ``32 - cut`` bits are at least
        ``prefix``) to the shared list (images, then row indices) in index order after the
        ``listed`` earlier ones; returns the new length."""
        count = Uint32(0)
        for q in cutlass.range_constexpr(len(images) // 4):
            if Uint32(4 * q) < slots:
                for k in cutlass.range_constexpr(4):
                    j = 4 * q + k
                    count = count + (
                        ptx.lt_u32(j, slots)
                        & (Uint32(1) - ptx.lt_u32(ptx.shr_u32(images[j], cut), prefix))
                    )
        inclusive, total = self.block_scan(count, lane, warp, wsum, warps, self.threads // 32)
        position = listed + inclusive - count
        for q in cutlass.range_constexpr(len(images) // 4):
            if Uint32(4 * q) < slots:
                for k in cutlass.range_constexpr(4):
                    j = 4 * q + k
                    keep = ptx.lt_u32(j, slots) & (
                        Uint32(1) - ptx.lt_u32(ptx.shr_u32(images[j], cut), prefix)
                    )
                    ptx.st_shared_u32_if(keep, lst + position * Uint32(4), images[j])
                    ptx.st_shared_u32_if(
                        keep, lst + (Uint32(LIST_KEYS) + position) * Uint32(4), start + Uint32(j)
                    )
                    position = position + keep
        return listed + total

    @cute.jit
    def count_list(
        self,
        lst,
        listed,
        prefix,
        consumed: cutlass.Constexpr[int],
        bits: cutlass.Constexpr[int],
        hist,
        tid,
        active,
    ):
        """Round histogram of the listed images matching ``prefix`` on the top ``consumed`` bits."""
        shift = 32 - consumed - bits
        for i in cutlass.range(Int32((listed + active - Uint32(1)) // active), unroll=1):
            entry = Uint32(i) * active + tid
            inside = ptx.lt_u32(entry, listed)
            image = ptx.ld_shared_u32_if(inside, lst + entry * Uint32(4))
            match = inside & ptx.eq_u32(image >> Uint32(32 - consumed), prefix)
            digit = (image >> Uint32(shift)) & Uint32((1 << bits) - 1)
            ptx.count_shared_if(match, hist + digit * Uint32(4))

    @cute.jit
    def compact_list(
        self, lst, listed, cut, prefix, need, tid, lane, warp, wsum, warps, active, stage, rank_u
    ):
        """``compact_chunk`` over the listed candidates (thread t: a contiguous run of the list,
        which is in index order); returns the smallest staged key image."""
        W = self.world
        run = (listed + active - Uint32(1)) // active
        first = tid * run
        last = ptx.min_u32(listed, first + run)
        packed = Uint32(0)
        for i in cutlass.range(Int32(run), unroll=1):
            entry = first + Uint32(i)
            inside = ptx.lt_u32(entry, last)
            top = ptx.shr_u32(ptx.ld_shared_u32_if(inside, lst + entry * Uint32(4)), cut)
            packed = (
                packed
                + ((inside & ptx.lt_u32(prefix, top)) << Uint32(16))
                + (inside & ptx.eq_u32(top, prefix))
            )
        inclusive, _ = self.block_scan(packed, lane, warp, wsum, warps, self.threads // 32)
        exclusive = inclusive - packed
        ties = exclusive & Uint32(0xFFFF)
        position = (exclusive >> Uint32(16)) + ptx.min_u32(ties, need)
        smallest = Uint64(0xFFFFFFFFFFFFFFFF)
        for i in cutlass.range(Int32(run), unroll=1):
            entry = first + Uint32(i)
            inside = ptx.lt_u32(entry, last)
            image = ptx.ld_shared_u32_if(inside, lst + entry * Uint32(4))
            index = ptx.ld_shared_u32_if(inside, lst + (Uint32(LIST_KEYS) + entry) * Uint32(4))
            top = ptx.shr_u32(image, cut)
            greater = inside & ptx.lt_u32(prefix, top)
            equal = inside & ptx.eq_u32(top, prefix)
            take = greater | (equal & ptx.lt_u32(ties, need))
            global_index = (((index >> Uint32(6)) * Uint32(W) + rank_u) << Uint32(6)) | (
                index & Uint32(63)
            )
            low = global_index ^ Uint32(0xFFFFFFFF)
            ptx.st_shared_u64_if(
                take, stage + position * Uint32(8), ptx.pack64(low, image ^ Uint32(0x80000000))
            )
            smallest = ptx.min_u64(
                smallest, ptx.sel64(take, ptx.pack64(low, image), Uint64(0xFFFFFFFFFFFFFFFF))
            )
            position = position + take
            ties = ties + equal
        return smallest

    @cute.jit
    def count_chunk(
        self,
        images,
        slots,
        prefix,
        consumed: cutlass.Constexpr[int],
        bits: cutlass.Constexpr[int],
        hist,
    ):
        """Round histogram of a chunk's images matching ``prefix`` on the top ``consumed`` bits.
        Groups of 4 past ``slots`` are skipped."""
        shift = 32 - consumed - bits
        for q in cutlass.range_constexpr(len(images) // 4):
            if Uint32(4 * q) < slots:
                for k in cutlass.range_constexpr(4):
                    j = 4 * q + k
                    active = ptx.lt_u32(j, slots)
                    if cutlass.const_expr(consumed > 0):
                        active = active & ptx.eq_u32(images[j] >> Uint32(32 - consumed), prefix)
                    digit = (images[j] >> Uint32(shift)) & Uint32((1 << bits) - 1)
                    ptx.count_shared_if(active, hist + digit * Uint32(4))

    @cute.jit
    def compact_chunk(
        self,
        images,
        start,
        slots,
        cut,
        prefix,
        need,
        before_g,
        before_e,
        smallest,
        lane,
        warp,
        wsum,
        warps,
        stage,
        rank_u,
    ):
        """Stage a chunk's selected keys (images above ``prefix`` on the top ``32 - cut`` bits, and
        equal ones while the tie budget ``need`` lasts) at their positions in index order, given
        the greater / equal counts of the earlier chunks; returns the updated counts and the
        smallest staged key image. Each thread's scores are consecutive: one scan of the
        per-thread (greater, equal) counts. The flags are recomputed rather than held, and only
        the few scores at or above the prefix take the staging path."""
        W = self.world
        packed = Uint32(0)
        for q in cutlass.range_constexpr(len(images) // 4):
            if Uint32(4 * q) < slots:
                for k in cutlass.range_constexpr(4):
                    j = 4 * q + k
                    top = ptx.shr_u32(images[j], cut)
                    valid = ptx.lt_u32(j, slots)
                    packed = (
                        packed
                        + ((valid & ptx.lt_u32(prefix, top)) << Uint32(16))
                        + (valid & ptx.eq_u32(top, prefix))
                    )
        inclusive, total = self.block_scan(packed, lane, warp, wsum, warps, self.threads // 32)
        exclusive = inclusive - packed
        ties = before_e + (exclusive & Uint32(0xFFFF))
        position = before_g + (exclusive >> Uint32(16)) + ptx.min_u32(ties, need)
        for q in cutlass.range_constexpr(len(images) // 4):
            if Uint32(4 * q) < slots:
                for k in cutlass.range_constexpr(4):
                    j = 4 * q + k
                    top = ptx.shr_u32(images[j], cut)
                    if (ptx.lt_u32(j, slots) & (Uint32(1) - ptx.lt_u32(top, prefix))) != Uint32(0):
                        equal = ptx.eq_u32(top, prefix)
                        index = start + Uint32(j)
                        global_index = (
                            ((index >> Uint32(6)) * Uint32(W) + rank_u) << Uint32(6)
                        ) | (index & Uint32(63))
                        low = global_index ^ Uint32(0xFFFFFFFF)
                        take = (Uint32(1) - equal) | ptx.lt_u32(ties, need)
                        ptx.st_shared_u64_if(
                            take,
                            stage + position * Uint32(8),
                            ptx.pack64(low, images[j] ^ Uint32(0x80000000)),
                        )
                        # Unsigned image of the key (same order as the signed key).
                        smallest = ptx.min_u64(
                            smallest,
                            ptx.sel64(take, ptx.pack64(low, images[j]), Uint64(0xFFFFFFFFFFFFFFFF)),
                        )
                        position = position + take
                        ties = ties + equal
        return before_g + (total >> Uint32(16)), before_e + (total & Uint32(0xFFFF)), smallest

    @cute.kernel
    def kernel(
        self,
        state: cute.Tensor,
        logits_addr: Int64,
        logits_stride: Int32,
        rowlen_addr: Int64,
        keys_addr: Int64,
        i32_addr: Int64,
        dk_addr: Int64,
        d32_addr: Int64,
        rank: Int32,
        rows_max: Int32,
        rows: Int32,
        out_addr: Int64,
        out_stride: Int32,
        counts_addr: Int64,
        bounds_addr: Int64,
        mask_addr: Int64,
        max_spin: Int32,
        trace_addr: Int64,
    ):
        W = self.world
        L = self.local_k
        PER = self.per
        T = self.threads
        CAP = T * PER
        tid_i, _, _ = cute.arch.thread_idx()
        row_i, _, _ = cute.arch.block_idx()
        tid = Uint32(tid_i)
        row = Uint32(row_i)
        lane = tid & Uint32(31)
        warp = tid >> Uint32(5)

        allocator = cutlass.utils.SmemAllocator()
        buffer = allocator.allocate_tensor(
            cutlass.Uint8, cute.make_layout(self.smem_bytes), byte_alignment=16
        )
        base = Uint32(buffer.iterator.toint())
        hist = base + _HIST
        wmin = base + _WMIN
        mctl = base + _MCTL
        stage = base + _STAGE
        self.clear_hist(tid, hist)
        ptx.st_shared_u32_if(ptx.lt_u32(tid, 4), mctl + tid * Uint32(4), 0)
        ptx.st_shared_u32_if(ptx.eq_u32(tid, 4), mctl + Uint32(16), 0xFFFFFFFF)
        ptx.st_shared_u64_if(ptx.lt_u32(tid, 32), wmin + tid * Uint32(8), 0xFFFFFFFFFFFFFFFF)

        # Logits, row lengths and the merge epoch come from earlier grids. Without the merge the
        # dependent grid (the merge) launches now; with it, at this grid's end.
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.top_k == 0):
            cute.arch.griddepcontrol_launch_dependents()
        state_addr = Uint64(state.iterator.toint())
        epoch = ptx.ld_global_u32(state_addr) + Uint32(1)
        n = ptx.ld_global_u32(Uint64(rowlen_addr) + Uint64(row) * Uint64(4))
        target = ptx.min_u32(n, L)
        # The first chunk (min(n, CAP) scores) runs on the fewest of 256 / 512 / ... threads that
        # hold it with at most 16 scores each (the measured best CTA sizes), each thread on
        # `slots` consecutive scores; later chunks (rows longer than CAP) on every thread.
        first_len = ptx.min_u32(n, CAP)
        active = Uint32(T)
        if cutlass.const_expr(self.top_k == 0):
            # (The fused merge needs every thread.)
            for size in (512, 256):
                if cutlass.const_expr(size < T):
                    fits = size * min(16, PER)
                    active = ptx.sel(ptx.lt_u32(first_len, fits + 1), size, active)
        warps = active >> Uint32(5)
        rank_u = Uint32(rank)
        slot = epoch & Uint32(1)
        rw = Uint32(rows_max) * Uint32(W)
        keys = Uint64(keys_addr)
        words32 = Uint64(i32_addr)

        if tid < active:
            self.bar(warps)
            self.stamp(trace_addr, row, tid, 0)
            row_base = Uint64(logits_addr) + Uint64(row) * Uint64(logits_stride) * Uint64(4)
            self.select_row(
                row_base, n, target, active, tid, lane, warp, base, stage, rank_u, trace_addr, row
            )
            self.stamp(trace_addr, row, tid, 7)

            # Push the candidates (and padding) into receive slot e & 1 of every rank.
            cell = slot * rw + row * Uint32(W) + rank_u
            recv = Uint64(slot) * Uint64(rw) * Uint64(L) + Uint64(
                row * Uint32(W) + rank_u
            ) * Uint64(L)
            half = L // 2
            pusher = ptx.lt_u32(tid, _BASE_THREADS)
            for it in cutlass.range_constexpr(W * half // _BASE_THREADS):
                flat = Uint32(it * _BASE_THREADS) + tid
                peer = ptx.min_u32(flat // Uint32(half), W - 1)
                i = (flat % Uint32(half)) * Uint32(2)
                a = ptx.sel64(
                    ptx.lt_u32(i, target),
                    ptx.ld_shared_u64(stage + i * Uint32(8)),
                    Uint64(_INT64_MIN),
                )
                b = ptx.sel64(
                    ptx.lt_u32(i + Uint32(1), target),
                    ptx.ld_shared_u64(stage + (i + Uint32(1)) * Uint32(8)),
                    Uint64(_INT64_MIN),
                )
                delta = ptx.ld_global_u64_nc(Uint64(dk_addr) + Uint64(peer) * Uint64(8))
                ptx.st_global_v2_u64_if(pusher, keys + (delta + recv + Uint64(i)) * Uint64(8), a, b)
            header = ptx.lt_u32(tid, W)
            peer = ptx.min_u32(tid, W - 1)
            delta = ptx.ld_global_u64_nc(Uint64(dk_addr) + Uint64(peer) * Uint64(8))
            delta32 = ptx.ld_global_u64_nc(Uint64(d32_addr) + Uint64(peer) * Uint64(8))
            smallest = self.staged_min(wmin)
            # No key sent: INT64_MAX (the pack's empty minimum); else back to the signed key.
            boundary = ptx.sel64(
                ptx.eq_u32(target, 0),
                Uint64(_INT64_MAX),
                smallest ^ Uint64(0x8000000000000000),
            )
            ptx.st_global_u64_if(
                header,
                keys + (delta + Uint64(2) * Uint64(rw) * Uint64(L) + Uint64(cell)) * Uint64(8),
                boundary,
            )
            ptx.st_global_u32_if(header, words32 + (delta32 + Uint64(cell)) * Uint64(4), target)
            ptx.st_global_u32_if(
                header, words32 + (delta32 + Uint64(2 * rw) + Uint64(cell)) * Uint64(4), n
            )
            self.bar(warps)
            ptx.publish_sys_if(
                header, words32 + (delta32 + Uint64(4 * rw) + Uint64(cell)) * Uint64(4), epoch
            )
            self.stamp(trace_addr, row, tid, 1)

        if cutlass.const_expr(self.top_k > 0):
            self.merge_row(
                state_addr,
                logits_addr,
                logits_stride,
                keys,
                words32,
                d32_addr,
                rank_u,
                row,
                rw,
                slot,
                epoch,
                n,
                target,
                base,
                tid,
                lane,
                warp,
                out_addr,
                out_stride,
                counts_addr,
                bounds_addr,
                mask_addr,
                max_spin,
                rows,
                trace_addr,
            )

    @cute.jit
    def staged_min(self, wmin):
        """The smallest staged unsigned key image of :meth:`select_row` (all ones when none)."""
        smallest = Uint64(0xFFFFFFFFFFFFFFFF)
        for w in cutlass.range_constexpr(32):
            smallest = ptx.min_u64(smallest, ptx.ld_shared_u64(wmin + Uint32(8 * w)))
        return smallest

    @cute.jit
    def select_row(
        self,
        row_base,
        n,
        target,
        active,
        tid,
        lane,
        warp,
        base,
        stage,
        rank_u,
        trace_addr,
        row,
        probe: cutlass.Constexpr[bool] = True,
    ):
        """Stage the exact Top-``target`` of a row's ``n`` scores at ``row_base`` by key (score
        desc, local index asc) at ``stage`` as signed keys in ascending local position, on the
        first ``active`` threads (the caller keeps the others out); the per-warp minima of the
        staged unsigned key images go to the shared ``wmin`` words (:meth:`staged_min`). The
        shared histogram is clear on entry and on return."""
        PER = self.per
        T = self.threads
        CAP = T * PER
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        wmin = base + _WMIN
        lst = base + Uint32(self.list_offset)
        chunks = (n + Uint32(CAP - 1)) // Uint32(CAP)
        first_len = ptx.min_u32(n, CAP)
        warps = active >> Uint32(5)
        group = ((first_len + Uint32(3)) // Uint32(4) + active - Uint32(1)) // active
        span = group * Uint32(4)
        start0 = tid * span
        slots0 = ptx.min_u32(span, n - ptx.min_u32(n, start0))
        images0 = self.load_chunk(row_base, start0, slots0)
        if cutlass.const_expr(probe):
            self.stamp(trace_addr, row, tid, 5)

        # Radix select, most significant bits first: after the rounds the selected images are
        # those whose top `depth` bits exceed `prefix`, plus the first `need` (in index order)
        # of those equal to it. A round whose boundary bucket is taken whole ends the search.
        # Once the candidates (scores at or above the prefix) fit LIST_KEYS, they are copied
        # to a shared list in index order and the later rounds and the staging read only it.
        prefix = Uint32(0)
        remaining = target
        depth = Uint32(0)
        done = ptx.eq_u32(target, 0)
        listed = Uint32(0)
        filtered = Uint32(0)
        consumed = 0
        for r in cutlass.range_constexpr(len(_ROUND_BITS)):
            bits = _ROUND_BITS[r]
            if done == Uint32(0):
                if filtered == Uint32(0):
                    self.count_chunk(images0, slots0, prefix, consumed, bits, hist)
                    for c in cutlass.range(Int32(chunks) - Int32(1), unroll=1):
                        start = (Uint32(c) + Uint32(1)) * Uint32(CAP) + tid * Uint32(PER)
                        slots = ptx.min_u32(PER, n - ptx.min_u32(n, start))
                        self.count_chunk(
                            self.load_chunk(row_base, start, slots),
                            slots,
                            prefix,
                            consumed,
                            bits,
                            hist,
                        )
                else:
                    self.count_list(lst, listed, prefix, consumed, bits, hist, tid, active)
                bucket, above, count = self.resolve(
                    tid, lane, warp, hist, wsum, ctl, remaining, bits, warps
                )
                remaining = remaining - above
                prefix = (prefix << Uint32(bits)) | bucket
                depth = Uint32(consumed + bits)
                done = ptx.eq_u32(count, remaining)
                if cutlass.const_expr(probe):
                    self.stamp(trace_addr, row, tid, 1 + r)
                if cutlass.const_expr(r + 1 < len(_ROUND_BITS)):
                    candidates = target - remaining + count
                    if (
                        ptx.eq_u32(filtered, 0)
                        & ptx.eq_u32(done, 0)
                        & ptx.lt_u32(LIST_MIN_KEYS, n)
                        & ptx.lt_u32(candidates, LIST_KEYS + 1)
                    ) != Uint32(0):
                        list_cut = Uint32(32 - consumed - bits)
                        listed = self.list_chunk(
                            images0,
                            start0,
                            slots0,
                            list_cut,
                            prefix,
                            Uint32(0),
                            lane,
                            warp,
                            wsum,
                            warps,
                            lst,
                        )
                        for c in cutlass.range(Int32(chunks) - Int32(1), unroll=1):
                            index = Uint32(c) + Uint32(1)
                            start = index * Uint32(CAP) + tid * Uint32(PER)
                            slots = ptx.min_u32(PER, n - ptx.min_u32(n, start))
                            listed = self.list_chunk(
                                self.load_chunk(row_base, start, slots),
                                start,
                                slots,
                                list_cut,
                                prefix,
                                listed,
                                lane,
                                warp,
                                wsum + (index & Uint32(1)) * Uint32(128),
                                warps,
                                lst,
                            )
                        filtered = Uint32(1)
                        # The list is read by other threads from the next round on.
                        self.bar(warps)
                        if cutlass.const_expr(probe):
                            self.stamp(trace_addr, row, tid, 4)
            consumed += bits
        cut = Uint32(32) - depth
        need = remaining
        if cutlass.const_expr(probe):
            self.stamp(trace_addr, row, tid, 6)

        # Stage the selected keys at their positions in index order.
        smallest = Uint64(0xFFFFFFFFFFFFFFFF)
        if filtered != Uint32(0):
            smallest = self.compact_list(
                lst,
                listed,
                cut,
                prefix,
                need,
                tid,
                lane,
                warp,
                wsum,
                warps,
                active,
                stage,
                rank_u,
            )
        else:
            before_g, before_e, smallest = self.compact_chunk(
                images0,
                start0,
                slots0,
                cut,
                prefix,
                need,
                Uint32(0),
                Uint32(0),
                Uint64(0xFFFFFFFFFFFFFFFF),
                lane,
                warp,
                wsum,
                warps,
                stage,
                rank_u,
            )
            for c in cutlass.range(Int32(chunks) - Int32(1), unroll=1):
                index = Uint32(c) + Uint32(1)
                start = index * Uint32(CAP) + tid * Uint32(PER)
                slots = ptx.min_u32(PER, n - ptx.min_u32(n, start))
                # Alternate the warp-total buffers: one barrier separates a chunk's reads of
                # one buffer from the next chunk but one's writes.
                before_g, before_e, smallest = self.compact_chunk(
                    self.load_chunk(row_base, start, slots),
                    start,
                    slots,
                    cut,
                    prefix,
                    need,
                    before_g,
                    before_e,
                    smallest,
                    lane,
                    warp,
                    wsum + (index & Uint32(1)) * Uint32(128),
                    warps,
                    stage,
                    rank_u,
                )
        smallest = ptx.warp_min_u64(smallest)
        ptx.st_shared_u64_if(ptx.eq_u32(lane, 0), wmin + warp * Uint32(8), smallest)
        self.bar(warps)

    @cute.jit
    def merge_row(
        self,
        state_addr,
        logits_addr,
        logits_stride,
        keys,
        words32,
        d32_addr,
        rank_u,
        row,
        rw,
        slot,
        epoch,
        n,
        target,
        base,
        tid,
        lane,
        warp,
        out_addr,
        out_stride,
        counts_addr,
        bounds_addr,
        mask_addr,
        max_spin,
        rows,
        trace_addr,
    ):
        """This rank's share of the row's global Top-K, as ``_ks_merge_row`` computes it.

        Waits for the row's ``W`` flags in this rank's receive slot, selects the ``K`` largest of
        the ``W * L`` received keys (radix select over the 64-bit unsigned key images: rounds of
        11, 11, 10 bits on the high word, then on the low word while score ties remain), checks
        the certificate on the scores (a rank with unsent keys whose smallest sent score is not
        below the threshold's) and, if it fails, re-selects exactly from every rank's local scores
        (read from the peers' workspaces). Writes this rank's winners as local token indices in
        ascending local position, ``-1`` after them, their count, the Helix combine mask
        (optional) and advances the merge epoch at the last CTA.
        """
        W, L, T, K = self.world, self.local_k, self.threads, self.top_k
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        mctl = base + _MCTL
        cnts = base + _CNT
        stage = base + _STAGE
        mine = ptx.lt_u32(tid, W)
        peer = ptx.min_u32(tid, W - 1)
        cell = Uint64(slot * rw + row * Uint32(W) + peer)
        timed_out = ptx.wait_flag_sys_if(
            mine, words32 + (Uint64(4) * Uint64(rw) + cell) * Uint64(4), epoch, Uint32(max_spin)
        )
        ptx.red_add_global_if(timed_out, state_addr + Uint64(12), 1)
        self.trap_late(timed_out, SITE_FLAGS, row, tid, epoch)
        sent = ptx.ld_global_cv_u32_if(mine, words32 + cell * Uint64(4))
        count = ptx.ld_global_cv_u32_if(mine, words32 + (Uint64(2) * Uint64(rw) + cell) * Uint64(4))
        boundary = ptx.ld_global_cv_u64_if(
            mine, keys + (Uint64(2) * Uint64(rw) * Uint64(L) + cell) * Uint64(8)
        )
        ptx.add_shared_if(mine, mctl, sent)
        ptx.add_shared_if(mine, mctl + Uint32(4), count)
        ptx.st_shared_u32_if(mine, cnts + tid * Uint32(4), count)
        cute.arch.sync_threads()
        total = ptx.ld_shared_u32(mctl)
        total_all = ptx.ld_shared_u32(mctl + Uint32(4))
        self.stamp(trace_addr, row, tid, 2)

        # The received keys (unsigned images), two per load.
        recv_row = keys + (Uint64(slot) * Uint64(rw) + Uint64(row) * Uint64(W)) * Uint64(8 * L)
        his = []
        los = []
        valid = []
        for it in cutlass.range_constexpr(W * L // (2 * T)):
            pair = Uint32(it * T) + tid
            a, b = ptx.ld_global_cv_v2_u64(recv_row + Uint64(pair) * Uint64(16))
            for x in (a, b):
                lo, hi = ptx.unpack64(x)
                his.append(hi ^ Uint32(0x80000000))
                los.append(lo)
                valid.append(ptx.ne_u64(x, Uint64(_INT64_MIN)))
        # Fewer than K keys sent: all of them (threshold image 1, the merge's INT64_MIN + 1).
        short = ptx.lt_u32(total, K)
        thr = self.select64(
            False,
            his,
            los,
            valid,
            ptx.sel(short, 0, K),
            tid,
            lane,
            warp,
            hist,
            wsum,
            ctl,
            logits_addr,
            logits_stride,
            d32_addr,
            row,
            cnts,
        )
        thr = ptx.sel64(short, Uint64(1), thr)
        # Certificate threshold: the score (high word) of the K-th largest received key, i.e. the
        # smallest high word of the selected keys.
        lowest = Uint32(0xFFFFFFFF)
        for j in cutlass.range_constexpr(len(his)):
            chosen = valid[j] & ptx.ge_u64(ptx.pack64(los[j], his[j]), thr)
            lowest = ptx.min_u32(lowest, ptx.sel(chosen, his[j], 0xFFFFFFFF))
        lowest = ptx.warp_min_u32(lowest)
        ptx.min_shared_if(ptx.eq_u32(lane, 0), mctl + Uint32(16), lowest)
        cute.arch.sync_threads()
        thr_hi = ptx.ld_shared_u32(mctl + Uint32(16))
        _, boundary_hi = ptx.unpack64(boundary)
        failed = (
            mine
            & ptx.lt_u32(sent, count)
            & (Uint32(1) - ptx.lt_u32(boundary_hi ^ Uint32(0x80000000), thr_hi))
        )
        ptx.add_shared_if(failed, mctl + Uint32(8), 1)
        cute.arch.sync_threads()
        fail = ptx.ld_shared_u32(mctl + Uint32(8))
        self.stamp(trace_addr, row, tid, 3)

        orow = Uint64(out_addr) + Uint64(row) * Uint64(out_stride) * Uint64(4)
        first_thread = ptx.eq_u32(tid, 0)
        written = Uint32(0)
        if fail != Uint32(0):
            ptx.red_add_global_if(first_thread, state_addr + Uint64(8), 1)
            exact = self.select64(
                True,
                [],
                [],
                [],
                ptx.min_u32(total_all, K),
                tid,
                lane,
                warp,
                hist,
                wsum,
                ctl,
                logits_addr,
                logits_stride,
                d32_addr,
                row,
                cnts,
            )
            exact = ptx.sel64(ptx.lt_u32(total_all, K), Uint64(1), exact)
            written = self.write_scores(
                exact, n, logits_addr, logits_stride, rank_u, row, orow, tid, lane, warp, wsum, mctl
            )
        else:
            written = self.write_staged(thr, target, stage, orow, tid, lane, warp, wsum, mctl)
        for it in cutlass.range_constexpr(K // T):
            kk = Uint32(it * T) + tid
            ptx.st_global_u32_if(
                Uint32(1) - ptx.lt_u32(kk, written), orow + Uint64(kk) * Uint64(4), 0xFFFFFFFF
            )
        ptx.st_global_u32_if(first_thread, Uint64(counts_addr) + Uint64(row) * Uint64(4), written)
        if cutlass.const_expr(self.mask):
            if cutlass.const_expr(self.bounds64):
                bound = ptx.ld_global_u64_if(
                    first_thread, Uint64(bounds_addr) + Uint64(row) * Uint64(8)
                )
                zero = Uint32(1) - ptx.ne_u64(bound, Uint64(0))
            else:
                bound = ptx.ld_global_u32_if(
                    first_thread, Uint64(bounds_addr) + Uint64(row) * Uint64(4)
                )
                zero = ptx.eq_u32(bound, 0)
            ptx.st_global_u8_if(
                first_thread, Uint64(mask_addr) + Uint64(row), zero | ptx.eq_u32(written, 0)
            )
        # Every thread of every CTA read the epoch before its CTA takes a ticket.
        cute.arch.sync_threads()
        self.stamp(trace_addr, row, tid, 4)
        ptx.advance_epoch_if(first_thread, state_addr, epoch, Uint32(rows))

    @cute.jit
    def select64(
        self,
        streamed: cutlass.Constexpr[bool],
        his,
        los,
        valid,
        target,
        tid,
        lane,
        warp,
        hist,
        wsum,
        ctl,
        logits_addr,
        logits_stride,
        d32_addr,
        row,
        cnts,
    ):
        """Unsigned image ``t`` such that the ``target`` largest keys are exactly those ``>= t``
        (keys are unique; 0 when ``target`` is 0): of the keys in registers, or (``streamed``) of
        every rank's local scores of the row."""
        prefix_hi = Uint32(0)
        prefix_lo = Uint32(0)
        remaining = target
        depth = Uint32(0)
        done = ptx.eq_u32(target, 0)
        consumed = 0
        for r in cutlass.range_constexpr(2 * len(_ROUND_BITS)):
            bits = _ROUND_BITS[r % 3]
            if done == Uint32(0):
                if cutlass.const_expr(streamed):
                    self.count_scores(
                        r,
                        prefix_hi,
                        prefix_lo,
                        logits_addr,
                        logits_stride,
                        d32_addr,
                        row,
                        cnts,
                        tid,
                        hist,
                    )
                else:
                    for j in cutlass.range_constexpr(len(his)):
                        active, digit = _digit(r, his[j], los[j], valid[j], prefix_hi, prefix_lo)
                        ptx.count_shared_if(active, hist + digit * Uint32(4))
                bucket, above, count = self.resolve(
                    tid, lane, warp, hist, wsum, ctl, remaining, bits, Uint32(self.threads // 32)
                )
                remaining = remaining - above
                if cutlass.const_expr(r < 3):
                    prefix_hi = (prefix_hi << Uint32(bits)) | bucket
                else:
                    prefix_lo = (prefix_lo << Uint32(bits)) | bucket
                depth = Uint32(consumed + bits)
                done = ptx.eq_u32(count, remaining)
            consumed += bits
        # prefix_hi holds min(depth, 32) bits, prefix_lo the rest.
        shift = Uint32(64) - depth
        high = ptx.sel64(
            ptx.lt_u32(32, depth),
            ptx.pack64(0, prefix_hi),
            ptx.shl_u64(ptx.pack64(prefix_hi, 0), shift),
        )
        return high | ptx.shl_u64(ptx.pack64(prefix_lo, 0), shift)

    @cute.jit
    def count_scores(
        self,
        r: cutlass.Constexpr[int],
        prefix_hi,
        prefix_lo,
        logits_addr,
        logits_stride,
        d32_addr,
        row,
        cnts,
        tid,
        hist,
    ):
        """Round ``r`` histogram of every rank's keys of the row, streamed from the local scores."""
        W, T = self.world, self.threads
        step = 4 * T
        for w in cutlass.range(W, unroll=1):
            peer = Uint32(w)
            n = ptx.ld_shared_u32(cnts + peer * Uint32(4))
            delta = ptx.ld_global_u64_nc(Uint64(d32_addr) + Uint64(peer) * Uint64(8))
            src = Uint64(logits_addr) + (delta + Uint64(row) * Uint64(logits_stride)) * Uint64(4)
            chunks = (n + Uint32(step - 1)) // Uint32(step)
            for c in cutlass.range(Int32(chunks), unroll=1):
                first = Uint32(c) * Uint32(step) + tid * Uint32(4)
                words = ptx.ld_global_cv_v4_if(
                    ptx.lt_u32(first, n), src + Uint64(first) * Uint64(4)
                )
                for k in cutlass.range_constexpr(4):
                    index = first + Uint32(k)
                    global_index = (((index >> Uint32(6)) * Uint32(W) + peer) << Uint32(6)) | (
                        index & Uint32(63)
                    )
                    active, digit = _digit(
                        r,
                        ptx.ordered_bits(words[k]),
                        global_index ^ Uint32(0xFFFFFFFF),
                        ptx.lt_u32(index, n),
                        prefix_hi,
                        prefix_lo,
                    )
                    ptx.count_shared_if(active, hist + digit * Uint32(4))

    @cute.jit
    def write_scores(
        self,
        thr,
        n,
        logits_addr,
        logits_stride,
        rank_u,
        row,
        orow,
        tid,
        lane,
        warp,
        wsum,
        mctl,
        table=None,
    ):
        """Fallback: write the own keys ``>= thr`` (local scores) in ascending local position;
        returns their count. ``table`` (``(row's block-table row, columns, stride_rows,
        row_offset)``, see ``ptx.kv_row_if``): written as KV pool rows instead of local token
        indices."""
        W, T = self.world, self.threads
        step = 4 * T
        src = Uint64(logits_addr) + Uint64(row) * Uint64(logits_stride) * Uint64(4)
        chunks = (n + Uint32(step - 1)) // Uint32(step)
        written = Uint32(0)
        for c in cutlass.range(Int32(chunks), unroll=1):
            first = Uint32(c) * Uint32(step) + tid * Uint32(4)
            words = ptx.ld_global_v4_if(ptx.lt_u32(first, n), src + Uint64(first) * Uint64(4))
            takes = []
            values = []
            taken = Uint32(0)
            for k in cutlass.range_constexpr(4):
                index = first + Uint32(k)
                global_index = (((index >> Uint32(6)) * Uint32(W) + rank_u) << Uint32(6)) | (
                    index & Uint32(63)
                )
                image = ptx.pack64(global_index ^ Uint32(0xFFFFFFFF), ptx.ordered_bits(words[k]))
                take = self.kv_keep(table, ptx.lt_u32(index, n) & ptx.ge_u64(image, thr), index)
                takes.append(take)
                values.append(self.kv_value(table, take, index))
                taken = taken + take
            inclusive, _ = self.block_scan(taken, lane, warp, wsum, Uint32(T // 32), T // 32)
            position = written + inclusive - taken
            for k in cutlass.range_constexpr(4):
                ptx.st_global_u32_if(takes[k], orow + Uint64(position) * Uint64(4), values[k])
                position = position + takes[k]
            ptx.st_shared_u32_if(ptx.eq_u32(tid, T - 1), mctl + Uint32(12), inclusive)
            cute.arch.sync_threads()
            written = written + ptx.ld_shared_u32(mctl + Uint32(12))
        return written

    @cute.jit
    def kv_value(self, table, take, local):
        """What the merge writes for an own winner at ``local``: the local token index, or with
        ``table`` its KV pool row (``ptx.kv_row_if``; no load where not ``take``)."""
        value = local
        if cutlass.const_expr(table is not None):
            row_table, cols, stride_rows, row_offset = table
            value = ptx.kv_row_if(take, row_table, cols, local, stride_rows, row_offset)
        return value

    @cute.jit
    def kv_keep(self, table, take, local):
        """``take`` without the keys whose page is past the request's pages (``table``'s
        columns), for which ``kv_value`` has no KV pool row (``-1``): no ``-1`` row is written."""
        if cutlass.const_expr(table is not None):
            take = take & ptx.lt_u32(local >> Uint32(6), table[1])
        return take

    @cute.jit
    def write_staged(
        self,
        thr,
        target,
        stage,
        orow,
        tid,
        lane,
        warp,
        wsum,
        mctl,
        cap: cutlass.Constexpr[int] = 0,
        table=None,
    ):
        """Write the own keys ``>= thr`` of the ``target`` staged ones (ascending local position,
        ``cap`` slots, default ``local_k``) as local token indices, or with ``table`` as KV pool
        rows (``write_scores``); returns their count."""
        W, T = self.world, self.threads
        C = cap if cap else self.local_k
        E = max(1, C // T)
        shift = 6 + (W.bit_length() - 1)
        takes = []
        locals_ = []
        taken = Uint32(0)
        for k in cutlass.range_constexpr(E):
            i = tid * Uint32(E) + Uint32(k)
            key = ptx.ld_shared_u64(stage + ptx.min_u32(i, C - 1) * Uint32(8))
            lo, hi = ptx.unpack64(key)
            take = ptx.lt_u32(i, target) & ptx.ge_u64(ptx.pack64(lo, hi ^ Uint32(0x80000000)), thr)
            g = lo ^ Uint32(0xFFFFFFFF)
            local = ((g >> Uint32(shift)) << Uint32(6)) | (g & Uint32(63))
            take = self.kv_keep(table, take, local)
            locals_.append(self.kv_value(table, take, local))
            takes.append(take)
            taken = taken + take
        inclusive, _ = self.block_scan(taken, lane, warp, wsum, Uint32(T // 32), T // 32)
        position = inclusive - taken
        for k in cutlass.range_constexpr(E):
            ptx.st_global_u32_if(takes[k], orow + Uint64(position) * Uint64(4), locals_[k])
            position = position + takes[k]
        ptx.st_shared_u32_if(ptx.eq_u32(tid, T - 1), mctl + Uint32(12), inclusive)
        cute.arch.sync_threads()
        return ptx.ld_shared_u32(mctl + Uint32(12))


def _architecture(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


@functools.lru_cache(maxsize=None)
def compile_select_pack(
    world: int,
    local_k: int,
    threads: int,
    per: int,
    arch: str,
    top_k: int = 0,
    bounds64: bool = False,
    mask: bool = False,
    trace: bool = False,
    record: int = 0,
):
    kernel = KeyShardSelectPackKernel(world, local_k, threads, per, top_k, bounds64, mask, trace)
    kernel.trap_record = record
    state = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4,), stride_order=(0,))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        state,
        Int64(0),
        Int32(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int64(0),
        Int32(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(0),
        Int64(0),
        stream,
        options=f"--gpu-arch={arch} --enable-tvm-ffi",
    )


def cluster_shape(merge, threads: int, device: torch.device) -> tuple[int, int]:
    """(CTAs per row, local keys per share) of the exchange kernel's local select:
    ``local_topk.CLUSTER`` and ``local_topk.SHARE_KEYS``. One CTA per row where clusters are
    unavailable (before sm_90), no row of the workspace can split, or ``rows_max`` clusters could
    exceed one CTA per SM."""
    from . import rubin_ull_dsa_index_local_topk as local_topk

    cluster = local_topk.CLUSTER
    share_keys = max(local_topk.SHARE_KEYS, 4)
    props = torch.cuda.get_device_properties(device)
    if (
        cluster <= 1
        or cluster > local_topk.CLUSTER_MAX
        or props.major < 9
        or merge.local_width <= share_keys
        or merge.rows_max * cluster > props.multi_processor_count
    ):
        return 1, share_keys
    return cluster, share_keys


#: ``tops_from`` past every row: the exchange kernel does not read the vector maxima.
_NO_TOPS = 2**31 - 1

#: The exchange kernels' KV-row arguments (block table, pitch, columns, request indices, page
#: stride, row offset) without ``kv_rows``.
_NO_KV_ROWS = (0, 0, 0, 0, 0, 0)


def _kv_row_args(kv_rows, rows: int) -> tuple[int, int, int, int, int, int]:
    """The exchange kernels' KV-row arguments of ``kv_rows`` (``KvRows``) for ``rows`` rows."""
    table, table_pitch, table_cols = kv_rows.table_args(rows)
    return (
        table,
        table_pitch,
        table_cols,
        kv_rows.req_idx.data_ptr(),
        kv_rows.stride_rows,
        kv_rows.row_offset,
    )


def tops_shape(merge, threads: int, per: int, device: torch.device):
    """(The vector maxima buffer, int32 [rows_max, local_width / 8] holding uint32 images, or None;
    the local key count from which rows select from it): ``local_topk.TOPS_FROM``, else one
    register chunk plus one, ``threads * per + 1``; at least ``8 * local_k``). None where the
    workspace's vectors exceed what a CTA holds."""
    from . import rubin_ull_dsa_index_local_topk as local_topk

    tops_from = max(local_topk.TOPS_FROM or threads * per + 1, 8 * merge.local_k)
    if (
        merge.local_width % 8
        or merge.local_width // 8 > threads * local_topk.TOPS_PER
        or local_topk.HOT % threads
        or tops_from > merge.local_width
    ):
        return None, tops_from
    tops = torch.zeros((merge.rows_max, merge.local_width // 8), dtype=torch.int32, device=device)
    return tops, tops_from


class KeyShardSelectPack:
    """The select + pack (+ merge) launch over one ``KeyShardTopKMerge`` workspace."""

    def __init__(self, merge, threads: int | None = None, per: int | None = None):
        self.merge = merge
        self.threads, self.per = launch_shape(merge.local_width, threads, per)
        #: Optional int64 [rows_max, TRACE_POINTS, 2] probe buffer (benchmarks; set before use).
        self.trace = None
        self.device = merge.state.device
        self.arch = _architecture(self.device)
        self.compiled = self._compile(0, False, False)
        #: CTAs per row of the exchange kernel's local select (``local_topk.py``; 1: one CTA per
        #: row, ``exchange.py``) and the local key count from which a row splits over them.
        self.cluster, self.share_keys = cluster_shape(merge, self.threads, self.device)
        #: vLLM's structure: the local_topk select / push grid, then a merge grid launched early
        #: by PDL. On 16 GB300 ranks (nsys) scorer + tops with split kernels took 59.46 us per layer,
        #: fused 60.45, an earlier revision 62.75.
        self.split_kernels = True
        #: The scorer's vector maxima (``score.score_logits_pdl(tops=...)``), uint32 [rows_max,
        #: local_width / 8], from which the exchange kernel selects rows of at least
        #: ``tops_from`` local keys (``local_topk.select_tops``); None: not used.
        self.tops, self.tops_from = tops_shape(merge, self.threads, self.per, self.device)
        #: The speculative decode's threshold pairs: the split
        #: merge grid triggers its dependents at once and publishes each row's threshold pairs into
        #: ``pub`` (int32 [rows_max, 8]: ``exchange.merge_exchange``'s ``pub``); ``published``:
        #: whether the last call did (the cluster exchange kernel with split kernels ran).
        self.publish = self.split_kernels
        self.pub = (
            torch.zeros((merge.rows_max, 8), dtype=torch.int32, device=self.device)
            if self.publish
            else None
        )
        self.published = False
        #: Set by the caller for one call: publish if this call runs the split merge grid (the
        #: workspace keeps ``local_k`` = the speculative decode's candidates).
        self.want_publish = False

    def _publishing(self) -> bool:
        from .rubin_ull_dsa_spec_decode import CANDIDATES

        return self.publish and self.want_publish and self.merge.local_k == CANDIDATES

    def _compile(self, top_k: int, bounds64: bool, mask: bool):
        merge = self.merge
        with torch.cuda.device(self.device):
            return compile_select_pack(
                merge.world,
                merge.local_k,
                self.threads,
                self.per,
                self.arch,
                top_k,
                bounds64,
                mask,
                self.trace is not None,
                spin_trap.record_address(),
            )

    @staticmethod
    def supported(merge) -> bool:
        world, local_k = merge.world, merge.local_k
        shape = launch_shape(merge.local_width)
        return (
            1 < world <= 32
            and local_k & (local_k - 1) == 0
            and 2 <= local_k <= 4096
            and (world * local_k // 2) % shape[0] == 0
            and world * local_k // 2 >= _BASE_THREADS
        )

    def supports_merge(self) -> bool:
        """The fused merge reads the peers' scores 16 bytes at a time."""
        merge = self.merge
        base = merge.buffers[merge.rank].data_ptr()
        return merge.top_k % self.threads == 0 and all(
            (b.data_ptr() - base) % 16 == 0 for b in merge.buffers
        )

    def supports_exchange(self) -> bool:
        """The LL exchange (``exchange.py``): 16-byte pairs in every rank's workspace, every thread
        of the CTA on whole groups of the received keys and of the rescue keys."""
        merge = self.merge
        world, local_k, top_k, threads = merge.world, merge.local_k, merge.top_k, self.threads
        return (
            self.supports_merge()
            and getattr(merge, "ll", None) is not None
            and merge.ll.data_ptr() % 16 == 0
            and local_k <= top_k <= LIST_KEYS
            and (world * local_k) % threads == 0
            and (world * top_k) % threads == 0
        )

    def _compile_exchange(self, bounds64: bool, mask: bool, rows: bool = False):
        from .rubin_ull_dsa_index_exchange import compile_exchange

        merge = self.merge
        with torch.cuda.device(self.device):
            return compile_exchange(
                merge.world,
                merge.local_k,
                self.threads,
                self.per,
                self.arch,
                merge.top_k,
                bounds64,
                mask,
                self.trace is not None,
                rows,
                record=spin_trap.record_address(),
            )

    def _compile_cluster(
        self,
        bounds64: bool,
        mask: bool,
        cluster: int,
        select_only: bool = False,
        tops: bool = False,
        rows: bool = False,
        publish: bool = False,
    ):
        from .rubin_ull_dsa_index_local_topk import compile_cluster_exchange

        merge = self.merge
        with torch.cuda.device(self.device):
            return compile_cluster_exchange(
                merge.world,
                merge.local_k,
                self.threads,
                self.per,
                self.arch,
                merge.top_k,
                bounds64,
                mask,
                self.trace is not None,
                cluster,
                select_only,
                self.split_kernels,
                tops,
                rows,
                record=spin_trap.record_address(),
                publish=publish and not select_only,
            )

    def select_only(
        self,
        row_lens: torch.Tensor,
        rows: int,
        dump: torch.Tensor,
        cluster: int,
        share_keys: int,
        tops: torch.Tensor | None = None,
        tops_from: int | None = None,
    ) -> None:
        """Benchmarks and checks: the exchange kernel's local select alone on ``cluster`` CTAs per
        row (``share_keys``: ``local_topk``'s; ``tops``: the logits' vector maxima, from
        ``tops_from`` keys, one CTA per row); ``dump`` [rows, local_k + 1] int64 gets each row's
        staged signed keys (0 past its count) and the smallest unsigned key image."""
        merge = self.merge
        assert dump.dtype == torch.int64 and dump.is_contiguous()
        assert dump.shape[0] >= rows and dump.shape[1] == merge.local_k + 1
        use_tops = tops is not None
        if use_tops:
            from .rubin_ull_dsa_index_local_topk import TOPS_PER

            cluster = 1
            tops_from = self.tops_from if tops_from is None else tops_from
            assert tops.dtype == torch.int32 and tops.stride(1) == 1
            assert tops.stride(0) * 8 == merge.local_width and tops.shape[0] >= rows
            assert tops_from >= 8 * merge.local_k
            assert merge.local_width // 8 <= self.threads * TOPS_PER
        self._compile_cluster(False, False, cluster, True, use_tops)(
            merge.state,
            merge.local_logits.data_ptr(),
            merge.local_width,
            row_lens.data_ptr(),
            0,
            0,
            0,
            merge.rank,
            merge.rows_max,
            rows,
            dump.data_ptr(),
            0,
            0,
            0,
            0,
            0,
            0 if self.trace is None else self.trace.data_ptr(),
            share_keys,
            tops.data_ptr() if use_tops else 0,
            tops_from if use_tops else _NO_TOPS,
            *_NO_KV_ROWS,
            0,
        )

    def __call__(
        self,
        row_lens: torch.Tensor,
        rows: int,
        out: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
        bounds: torch.Tensor | None = None,
        mask_out: torch.Tensor | None = None,
        exchange: bool = False,
        tops: bool = False,
        kv_rows: "KvRows | None" = None,
    ) -> None:
        """Select and publish the local Top-L of ``merge.local_logits[:rows]`` whose valid lengths
        are ``row_lens[:rows]`` (int32). Without ``out`` the merge (``merge_owned``) follows on
        the stream; with it the kernel also merges, with ``merge_owned``'s arguments: through the
        flags and the receive slots, or with ``exchange`` through the LL exchange
        (``exchange.KeyShardExchangeKernel``), which with ``kv_rows`` writes ``out`` as KV pool
        rows (``KvRows``) instead of local token indices. ``tops``: ``self.tops`` holds this step's
        vector maxima (the exchange kernel then selects long rows from them)."""
        merge = self.merge
        assert row_lens.dtype == torch.int32 and row_lens.is_contiguous()
        assert 0 < rows <= merge.rows_max
        self.published = False
        compiled = self.compiled if self.trace is None else self._compile(0, False, False)
        out_addr = out_stride = counts_addr = bounds_addr = mask_addr = 0
        # The flag wait polls with plain acquire loads, far cheaper than the Triton merge's atomics:
        # SPIN_SCALE x its poll budget for a comparable wall-clock timeout.
        spin = min(merge.max_spin * SPIN_SCALE, 2**31 - 1)
        assert out is not None or not exchange
        assert kv_rows is None or exchange
        if out is not None:
            assert out.dtype == torch.int32 and out.stride(1) == 1
            assert out.shape[0] == rows and out.shape[1] == merge.top_k
            if counts is None:
                counts = merge._counts_scratch(rows)
            assert counts.dtype == torch.int32 and counts.is_contiguous()
            assert counts.shape[0] >= rows
            masked = mask_out is not None
            if masked:
                assert bounds is not None and bounds.dim() == 1 and bounds.is_contiguous()
                assert bounds.dtype in (torch.int32, torch.int64) and bounds.shape[0] >= rows
                assert mask_out.dtype == torch.bool and mask_out.is_contiguous()
                assert mask_out.shape[0] >= rows
                bounds_addr, mask_addr = bounds.data_ptr(), mask_out.data_ptr()
            bounds64 = masked and bounds.dtype == torch.int64
            out_addr, out_stride, counts_addr = out.data_ptr(), out.stride(0), counts.data_ptr()
            tops = tops and exchange and self.tops is not None
            kv_args = _NO_KV_ROWS if kv_rows is None else _kv_row_args(kv_rows, rows)
            if exchange and (self.cluster > 1 or tops):
                publish = self._publishing() and kv_rows is not None
                self._compile_cluster(
                    bounds64,
                    masked,
                    1 if tops else self.cluster,
                    False,
                    tops,
                    kv_rows is not None,
                    publish,
                )(
                    merge.state,
                    merge.local_logits.data_ptr(),
                    merge.local_width,
                    row_lens.data_ptr(),
                    merge.ll.data_ptr(),
                    merge.dk.data_ptr(),
                    merge.d32.data_ptr(),
                    merge.rank,
                    merge.rows_max,
                    rows,
                    out_addr,
                    out_stride,
                    counts_addr,
                    bounds_addr,
                    mask_addr,
                    min(merge.max_spin, 2**31 - 1),
                    0 if self.trace is None else self.trace.data_ptr(),
                    self.share_keys,
                    self.tops.data_ptr() if tops else 0,
                    self.tops_from if tops else _NO_TOPS,
                    *kv_args,
                    self.pub.data_ptr() if publish else 0,
                )
                self.published = publish
                return
            if exchange:
                # The LL polls' deadline: max_spin microseconds (2^26: about 67 s).
                self._compile_exchange(bounds64, masked, kv_rows is not None)(
                    merge.state,
                    merge.local_logits.data_ptr(),
                    merge.local_width,
                    row_lens.data_ptr(),
                    merge.ll.data_ptr(),
                    merge.dk.data_ptr(),
                    merge.d32.data_ptr(),
                    merge.rank,
                    merge.rows_max,
                    rows,
                    out_addr,
                    out_stride,
                    counts_addr,
                    bounds_addr,
                    mask_addr,
                    min(merge.max_spin, 2**31 - 1),
                    0 if self.trace is None else self.trace.data_ptr(),
                    *kv_args,
                )
                return
            compiled = self._compile(merge.top_k, bounds64, masked)
        compiled(
            merge.state,
            merge.local_logits.data_ptr(),
            merge.local_width,
            row_lens.data_ptr(),
            merge.keys.data_ptr(),
            merge.i32.data_ptr(),
            merge.dk.data_ptr(),
            merge.d32.data_ptr(),
            merge.rank,
            merge.rows_max,
            rows,
            out_addr,
            out_stride,
            counts_addr,
            bounds_addr,
            mask_addr,
            spin,
            0 if self.trace is None else self.trace.data_ptr(),
        )
