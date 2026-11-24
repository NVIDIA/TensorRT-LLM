# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-shard local Top-L, LL exchange and exact merge in one CuTe DSL kernel (one CTA per row).

The flag-free form of ``select_pack.KeyShardSelectPackKernel`` with ``top_k`` (the fused merge),
after vLLM's K3.1 DSA decode exchange (``IndexerLocalTopk`` push + ``ExchangeMerge``), at FP32:

- select: the exact local Top-``L`` of the row by key (``select_row``: score desc, local index asc
  on the order-preserving bits of the FP32 logit), staged in ascending local position;
- push: every candidate as one LL pair (high and low word of its 64-bit unsigned key image, each
  next to the call's epoch; ``KeyShardTopKMerge.ll_pairs``), plus two header pairs (the row's
  local key count, the smallest sent key), into every rank's LL slot ``[epoch & 1][row][rank]``.
  No system-scope release, no flag: a reader polls each pair until both epochs match;
- merge: the ``W * L`` received keys, settled pair by pair (all loads issued first), the global
  ``K``-th largest received key ``thr`` (64-bit radix select), and the certificate of the Triton
  merge with ``ties_exact`` (``_ks_short``): rank ``w`` is short if it has unsent keys and its
  smallest sent key is above ``thr``. Without a short rank this rank's winners are its staged
  keys ``>= thr``;
- rescue (rows with short ranks; their number is at most ``(K - 1) // L``: each one's ``L`` sent
  keys lie above ``thr``, among the ``K - 1`` selected keys above it): every short rank stages its
  exact local Top-``min(n, K)`` from its own logits and pushes it as LL pairs into rescue slot
  ``s`` (``s`` = its order among the short ranks) of every rank; every rank then selects the
  ``K``-th largest key of the candidates "received keys of the ranks that are not short" plus
  "the short ranks' pushed keys". Every global Top-K key is a candidate (a rank that is not short
  has no unsent key above ``thr``, and ``thr`` is at most the global ``K``-th key; a short rank's
  global winners are among its local Top-K), and every candidate is a real, distinct key, so
  that ``K``-th largest is the global ``K``-th key. Own winners: the staged keys at or above it
  (the rescue stage when this rank is short). Rows of fewer than ``K`` keys take every key.

The output is exactly ``KeyShardTopKMerge.merge_owned``'s (``ties_exact``): this rank's winners as
local token indices in ascending local position, ``-1`` after them, their count and the combine
mask. The ``rows`` variant writes each winner as its KV pool row instead, exactly what
``convertReqIndexToGlobal`` makes of that index (``ptx.kv_row_if``: the MLA block table of the
row's request, 64-token pages, the layer's page stride and offset as runtime arguments, so one
compiled kernel serves every layer). The merge counters keep their meaning (``state[2]`` rows with a short rank, ``state[3]``
LL polls past their deadline, ``counters[4]`` rows with more short ranks than rescue slots, which
the bound above rules out; they would re-select from every rank's logits).

Slot reuse: the slots alternate by epoch parity. A rank pushes call ``e + 2`` into a peer only
after its call ``e + 1`` merge, which waited for that peer's ``e + 1`` pushes, issued by a later
grid than the peer's call ``e`` reads (the epoch, read after ``griddepcontrol.wait``, advances when
every CTA of a call has merged). An epoch never repeats within 2^32 calls, so a stale pair never
matches.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64

from . import rubin_ull_dsa_index_ptx as ptx
from .rubin_ull_dsa_index_select_pack import (
    _CNT,
    _MCTL,
    _ROUND_BITS,
    _STAGE,
    _WMIN,
    _WSUM,
    _CTL,
    _HIST,
    SITE_MERGE,
    SITE_RESCUE,
    KeyShardSelectPackKernel,
    _digit,
)

_MSB = 0x8000000000000000


def block_pairs(world: int, local_k: int, top_k: int) -> int:
    """LL pairs per (parity, row) block of the exchange workspace: every rank's ``local_k`` keys,
    two header pairs per rank, then ``(top_k - 1) // local_k`` rescue slots of ``top_k`` keys."""
    return world * (local_k + 2) + (top_k - 1) // local_k * top_k


def first_thread_of(tid):
    """1 in thread 0, else 0."""
    return ptx.eq_u32(tid, 0)
_ALL = 0xFFFFFFFFFFFFFFFF
# Shared ``mctl`` words beyond the fused merge's five.
_MASK = 20  # the row's short ranks (bit per rank)


class KeyShardExchangeKernel(KeyShardSelectPackKernel):
    """``KeyShardSelectPackKernel`` with ``top_k`` merging through the LL exchange (module doc)."""

    def __init__(
        self,
        world: int,
        local_k: int,
        threads: int,
        per: int,
        top_k: int,
        bounds64: bool = False,
        mask: bool = False,
        trace: bool = False,
        rows: bool = False,
    ):
        super().__init__(world, local_k, threads, per, top_k, bounds64, mask, trace)
        #: Own winners written as KV pool rows (``ptx.kv_row_if``) instead of local token indices.
        self.kv_rows = rows
        assert 0 < top_k and local_k <= top_k and top_k % threads == 0
        assert (world * local_k) % threads == 0 and (world * top_k) % threads == 0
        self.slots = (top_k - 1) // local_k
        self.block = block_pairs(world, local_k, top_k)
        # Shared: the short ranks in rescue-slot order, then the rescue stage (K keys).
        self.owner_offset = self.smem_bytes
        self.rescue_offset = self.owner_offset + 4 * 32
        self.smem_bytes = self.rescue_offset + 8 * top_k

    @cute.jit
    def __call__(
        self,
        state: cute.Tensor,
        logits_addr: Int64,
        logits_stride: Int32,
        rowlen_addr: Int64,
        ll_addr: Int64,
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
        timeout_us: Int32,
        trace_addr: Int64,
        table_addr: Int64,
        table_pitch: Int32,
        table_cols: Int32,
        req_addr: Int64,
        stride_rows: Int32,
        row_offset: Int32,
        stream,
    ):
        self.kernel(
            state,
            logits_addr,
            logits_stride,
            rowlen_addr,
            ll_addr,
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
            timeout_us,
            trace_addr,
            table_addr,
            table_pitch,
            table_cols,
            req_addr,
            stride_rows,
            row_offset,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def peer_ll(self, ll, dk_addr, peer):
        """Byte address of ``peer``'s LL region (``ll``: this rank's)."""
        return ll + ptx.ld_global_u64_nc(Uint64(dk_addr) + Uint64(peer) * Uint64(8)) * Uint64(8)

    @cute.jit
    def push_keys(self, ll, dk_addr, block, stage, count, cap: cutlass.Constexpr[int], epoch, tid):
        """The first ``count`` of the ``cap`` keys staged (signed) at ``stage`` -> pairs
        ``block + i`` of every rank's LL region as unsigned key images (0 past ``count``: a
        candidate block is always written whole)."""
        W, T = self.world, self.threads
        whole = cap == self.local_k
        for it in cutlass.range_constexpr(W * cap // T):
            # Consecutive threads store consecutive pairs of one peer.
            if cutlass.const_expr(T % cap == 0):
                peer = Uint32(it * (T // cap)) + tid // Uint32(cap)
                i = tid % Uint32(cap)
            else:
                peer = Uint32(it // (cap // T))
                i = Uint32((it % (cap // T)) * T) + tid
            take = ptx.lt_u32(i, count)
            key = ptx.ld_shared_u64(stage + ptx.min_u32(i, cap - 1) * Uint32(8))
            lo, hi = ptx.unpack64(ptx.sel64(take, key ^ Uint64(_MSB), Uint64(0)))
            ptx.st_ll_pair_if(
                Uint32(1) if whole else take,
                self.peer_ll(ll, dk_addr, peer) + (Uint64(block) + Uint64(i)) * Uint64(16),
                hi,
                lo,
                epoch,
            )

    @cute.kernel
    def kernel(
        self,
        state: cute.Tensor,
        logits_addr: Int64,
        logits_stride: Int32,
        rowlen_addr: Int64,
        ll_addr: Int64,
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
        timeout_us: Int32,
        trace_addr: Int64,
        table_addr: Int64,
        table_pitch: Int32,
        table_cols: Int32,
        req_addr: Int64,
        stride_rows: Int32,
        row_offset: Int32,
    ):
        W, L, T, K = self.world, self.local_k, self.threads, self.top_k
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
        ptx.st_shared_u32_if(
            ptx.lt_u32(tid, 16), mctl + tid * Uint32(4), ptx.sel(ptx.eq_u32(tid, 4), 0xFFFFFFFF, 0)
        )
        ptx.st_shared_u64_if(ptx.lt_u32(tid, 32), wmin + tid * Uint32(8), Uint64(_ALL))

        # Logits, row lengths and the epoch come from earlier grids; dependents launch at the end.
        cute.arch.griddepcontrol_wait()
        state_addr = Uint64(state.iterator.toint())
        epoch = ptx.ld_global_u32(state_addr) + Uint32(1)
        n = ptx.ld_global_u32(Uint64(rowlen_addr) + Uint64(row) * Uint64(4))
        target = ptx.min_u32(n, L)
        deadline = ptx.globaltimer() + Uint64(timeout_us) * Uint64(1000)
        rank_u = Uint32(rank)
        ll = Uint64(ll_addr)
        block = ((epoch & Uint32(1)) * Uint32(rows_max) + row) * Uint32(self.block)
        row_base = Uint64(logits_addr) + Uint64(row) * Uint64(logits_stride) * Uint64(4)
        # The row's request (step metadata, possibly written by an earlier grid of the step). Every
        # read of the request index, the block table and the row lengths follows the wait.
        req = Uint32(0)
        if cutlass.const_expr(self.kv_rows):
            req = ptx.ld_global_u32(Uint64(req_addr) + Uint64(row) * Uint64(4))

        self.bar(Uint32(T // 32))
        self.stamp(trace_addr, row, tid, 0)
        self.select_row(
            row_base, n, target, Uint32(T), tid, lane, warp, base, stage, rank_u, trace_addr, row
        )
        self.stamp(trace_addr, row, tid, 7)
        smallest = self.staged_min(wmin)
        # Push: this rank's candidates and header into every rank's slot.
        self.push_keys(ll, dk_addr, block + rank_u * Uint32(L), stage, target, L, epoch, tid)
        header = ptx.lt_u32(tid, W)
        hpeer = ptx.min_u32(tid, W - 1)
        hdr = Uint64(block + Uint32(W * L) + rank_u * Uint32(2)) * Uint64(16)
        b_lo, b_hi = ptx.unpack64(smallest)
        dst = self.peer_ll(ll, dk_addr, hpeer) + hdr
        ptx.st_ll_pair_if(header, dst, n, b_hi, epoch)
        ptx.st_ll_pair_if(header, dst + Uint64(16), b_lo, 0, epoch)
        self.stamp(trace_addr, row, tid, 1)
        table = None
        if cutlass.const_expr(self.kv_rows):
            # The request's block-table row; its pages the row's keys can use go to L2 while the
            # merge waits for the peers.
            row_table = Uint64(table_addr) + Uint64(req) * Uint64(table_pitch) * Uint64(4)
            pages = ptx.min_u32((n + Uint32(63)) >> Uint32(6), Uint32(table_cols))
            first_line = row_table >> Uint64(7)
            last_line = (row_table + Uint64(pages) * Uint64(4) + Uint64(127)) >> Uint64(7)
            lines = Uint32(last_line - first_line)
            ptx.prefetch_l2_if(
                ptx.lt_u32(tid, lines), (first_line + Uint64(tid)) << Uint64(7)
            )
            table = (row_table, pages, Uint32(stride_rows), Uint32(row_offset))

        self.merge_exchange(
            state_addr,
            logits_addr,
            logits_stride,
            ll,
            dk_addr,
            d32_addr,
            rank_u,
            row,
            block,
            epoch,
            deadline,
            n,
            target,
            row_base,
            base,
            tid,
            lane,
            warp,
            out_addr,
            out_stride,
            counts_addr,
            bounds_addr,
            mask_addr,
            rows,
            trace_addr,
            table,
        )

    @cute.jit
    def merge_exchange(
        self,
        state_addr,
        logits_addr,
        logits_stride,
        ll,
        dk_addr,
        d32_addr,
        rank_u,
        row,
        block,
        epoch,
        deadline,
        n,
        target,
        row_base,
        base,
        tid,
        lane,
        warp,
        out_addr,
        out_stride,
        counts_addr,
        bounds_addr,
        mask_addr,
        rows,
        trace_addr,
        table=None,
        pub=None,
    ):
        """This rank's share of the row's global Top-K from its LL slot (module doc); with
        ``table`` (``write_scores``) written as KV pool rows. ``pub`` (the speculative decode's
        pairs, 32 bytes per row): as soon as ``thr`` and the short mask are known, a row without a
        short rank publishes the LL pair ``(thr high, thr low; epoch)`` at ``pub + 32 row``; a row
        with one publishes ``(ALL, ALL; epoch)`` (take the written winners) only after its
        winners. Every row ends with the pair ``(written, 0; epoch)`` at ``pub + 32 row + 16``
        behind a GPU fence: a reader of either pair after a fence sees everything this CTA wrote.
        With ``pub`` this grid takes no epoch ticket: the decode grid that reads the pairs does."""
        W, L, T, K = self.world, self.local_k, self.threads, self.top_k
        E = W * L // T
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        mctl = base + _MCTL
        cnts = base + _CNT
        stage = base + _STAGE
        # Every received key pair is loaded before any is checked; then each is settled.
        keys_at = ll + Uint64(block) * Uint64(16)
        raw = []
        for it in cutlass.range_constexpr(E):
            raw.append(
                ptx.ld_ll_pair_if(
                    Uint32(1), keys_at + Uint64(Uint32(it * T) + tid) * Uint64(16)
                )
            )
        mine = ptx.lt_u32(tid, W)
        peer = ptx.min_u32(tid, W - 1)
        hdr = keys_at + Uint64(Uint32(W * L) + peer * Uint32(2)) * Uint64(16)
        h0 = ptx.ld_ll_pair_if(mine, hdr)
        h1 = ptx.ld_ll_pair_if(mine, hdr + Uint64(16))
        his = []
        los = []
        valid = []
        late = Uint32(0)
        for it in cutlass.range_constexpr(E):
            w0, e0, w1, e1 = raw[it]
            hi, lo, timed_out = ptx.settle_ll_if(
                Uint32(1),
                keys_at + Uint64(Uint32(it * T) + tid) * Uint64(16),
                w0,
                e0,
                w1,
                e1,
                epoch,
                deadline,
            )
            late = late | timed_out
            his.append(hi)
            los.append(lo)
            valid.append(ptx.ne_u64(ptx.pack64(lo, hi), Uint64(0)))
        count, b_hi, t0 = ptx.settle_ll_if(mine, hdr, h0[0], h0[1], h0[2], h0[3], epoch, deadline)
        b_lo, _, t1 = ptx.settle_ll_if(
            mine, hdr + Uint64(16), h1[0], h1[1], h1[2], h1[3], epoch, deadline
        )
        late = late | t0 | t1
        ptx.red_add_global_if(late, state_addr + Uint64(12), 1)
        sent = ptx.min_u32(count, L)
        boundary = ptx.pack64(b_lo, b_hi)
        ptx.add_shared_if(mine, mctl, sent)
        ptx.add_shared_if(mine, mctl + Uint32(4), count)
        ptx.st_shared_u32_if(mine, cnts + tid * Uint32(4), count)
        cute.arch.sync_threads()
        total = ptx.ld_shared_u32(mctl)
        total_all = ptx.ld_shared_u32(mctl + Uint32(4))
        self.stamp(trace_addr, row, tid, 2)

        # thr: the K-th largest received key (fewer than K sent: all, image 1).
        few = ptx.lt_u32(total, K)
        thr = self.select64(
            False,
            his,
            los,
            valid,
            ptx.sel(few, 0, K),
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
        thr = ptx.sel64(few, Uint64(1), thr)
        # Short: unsent keys, and the smallest sent key above thr (whole keys: the candidates are
        # the exact local Top-L by key).
        short = mine & ptx.lt_u32(sent, count) & (Uint32(1) - ptx.ge_u64(thr, boundary))
        ptx.or_shared_if(short, mctl + Uint32(_MASK), Uint32(1) << peer)
        cute.arch.sync_threads()
        mask = ptx.ld_shared_u32(mctl + Uint32(_MASK))
        self.stamp(trace_addr, row, tid, 3)
        if cutlass.const_expr(pub is not None):
            thr_lo, thr_hi = ptx.unpack64(thr)
            pub_row = pub + Uint64(row) * Uint64(32)
            ptx.st_ll_pair_if(first_thread_of(tid) & ptx.eq_u32(mask, 0), pub_row, thr_hi, thr_lo, epoch)

        orow = Uint64(out_addr) + Uint64(row) * Uint64(out_stride) * Uint64(4)
        first_thread = ptx.eq_u32(tid, 0)
        written = Uint32(0)
        rescue_late = Uint32(0)
        if mask != Uint32(0):
            ptx.red_add_global_if(first_thread, state_addr + Uint64(8), 1)
            written, rescue_late = self.rescue(
                state_addr,
                logits_addr,
                logits_stride,
                ll,
                dk_addr,
                d32_addr,
                rank_u,
                row,
                block,
                epoch,
                deadline,
                n,
                target,
                row_base,
                base,
                tid,
                lane,
                warp,
                mask,
                total_all,
                his,
                los,
                valid,
                orow,
                trace_addr,
                table,
            )
        else:
            written = self.write_staged(
                thr, target, stage, orow, tid, lane, warp, wsum, mctl, table=table
            )
        self.trap_late2(late, SITE_MERGE, rescue_late, SITE_RESCUE, row, tid, epoch)
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
        if cutlass.const_expr(pub is None):
            ptx.advance_epoch_if(first_thread, state_addr, epoch, Uint32(rows))
        else:
            # The speculative decode's CTAs take the epoch tickets (they read the epoch too).
            pub_row = pub + Uint64(row) * Uint64(32)
            ptx.fence_acq_rel_gpu_if(first_thread)
            rescued = first_thread & (Uint32(1) - ptx.eq_u32(mask, 0))
            ptx.st_ll_pair_if(rescued, pub_row, 0xFFFFFFFF, 0xFFFFFFFF, epoch)
            ptx.st_ll_pair_if(first_thread, pub_row + Uint64(16), written, 0, epoch)

    @cute.jit
    def rescue(
        self,
        state_addr,
        logits_addr,
        logits_stride,
        ll,
        dk_addr,
        d32_addr,
        rank_u,
        row,
        block,
        epoch,
        deadline,
        n,
        target,
        row_base,
        base,
        tid,
        lane,
        warp,
        mask,
        total_all,
        his,
        los,
        valid,
        orow,
        trace_addr,
        table=None,
    ):
        """This rank's winners of a row with short ranks ``mask`` (module doc); returns their
        count and whether a wait for the pushed keys timed out. ``table``: as
        ``merge_exchange``."""
        W, L, T, K = self.world, self.local_k, self.threads, self.top_k
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        mctl = base + _MCTL
        cnts = base + _CNT
        stage = base + _STAGE
        owners = base + Uint32(self.owner_offset)
        rstage = base + Uint32(self.rescue_offset)
        own_short = (mask >> rank_u) & Uint32(1)
        nshort = ptx.popc(mask)
        target2 = ptx.min_u32(n, K)
        first_thread = ptx.eq_u32(tid, 0)
        written = Uint32(0)
        late = Uint32(0)
        if ptx.lt_u32(total_all, K) != Uint32(0):
            # Fewer than K keys in the row: every key is a winner.
            if own_short != Uint32(0):
                written = self.write_scores(
                    Uint64(1),
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
                    table,
                )
            else:
                written = self.write_staged(
                    Uint64(1), target, stage, orow, tid, lane, warp, wsum, mctl, table=table
                )
        elif ptx.lt_u32(self.slots, nshort) != Uint32(0):
            # More short ranks than rescue slots (ruled out by the bound): re-select from every
            # rank's logits.
            ptx.red_add_global_if(first_thread, state_addr + Uint64(16), 1)
            exact = self.select64(
                True,
                [],
                [],
                [],
                Uint32(K),
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
            written = self.write_scores(
                exact,
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
                table,
            )
        else:
            # The short ranks in rescue-slot order.
            mine = ptx.lt_u32(tid, W)
            peer = ptx.min_u32(tid, W - 1)
            listed = mine & ((mask >> peer) & Uint32(1))
            below = mask & ((Uint32(1) << peer) - Uint32(1))
            ptx.st_shared_u32_if(listed, owners + ptx.popc(below) * Uint32(4), peer)
            rescue_at = Uint32(W * (L + 2))
            if own_short != Uint32(0):
                # This rank's exact local Top-min(n, K) from its own logits, pushed into its
                # rescue slot of every rank.
                self.select_row(
                    row_base,
                    n,
                    target2,
                    Uint32(T),
                    tid,
                    lane,
                    warp,
                    base,
                    rstage,
                    rank_u,
                    trace_addr,
                    row,
                    False,
                )
                own_slot = ptx.popc(mask & ((Uint32(1) << rank_u) - Uint32(1)))
                self.push_keys(
                    ll,
                    dk_addr,
                    block + rescue_at + own_slot * Uint32(K),
                    rstage,
                    target2,
                    K,
                    epoch,
                    tid,
                )
            cute.arch.sync_threads()
            # Settle the short ranks' pushed keys (each thread the pairs it reads below).
            rescue_base = ll + Uint64(block + rescue_at) * Uint64(16)
            late = Uint32(0)
            for s in cutlass.range(Int32(nshort), unroll=1):
                src = ptx.ld_shared_u32(owners + Uint32(s) * Uint32(4))
                c = ptx.min_u32(ptx.ld_shared_u32(cnts + src * Uint32(4)), K)
                for i in cutlass.range_constexpr(K // T):
                    idx = Uint32(i * T) + tid
                    inside = ptx.lt_u32(idx, c)
                    at = rescue_base + (Uint64(s) * Uint64(K) + Uint64(idx)) * Uint64(16)
                    w0, e0, w1, e1 = ptx.ld_ll_pair_if(inside, at)
                    _, _, timed_out = ptx.settle_ll_if(inside, at, w0, e0, w1, e1, epoch, deadline)
                    late = late | timed_out
            ptx.red_add_global_if(late, state_addr + Uint64(12), 1)
            # Candidates: the received keys of the ranks that are not short (a short rank's are
            # among its pushed ones), and the pushed keys.
            kept = []
            for it in cutlass.range_constexpr(len(his)):
                src = (Uint32(it * T) + tid) // Uint32(L)
                kept.append(valid[it] & (Uint32(1) - (ptx.shr_u32(mask, src) & Uint32(1))))
            exact = self.select_rescue(
                his, los, kept, nshort, owners, cnts, rescue_base, tid, lane, warp, hist, wsum, ctl
            )
            if own_short != Uint32(0):
                written = self.write_staged(
                    exact, target2, rstage, orow, tid, lane, warp, wsum, mctl, K, table
                )
            else:
                written = self.write_staged(
                    exact, target, stage, orow, tid, lane, warp, wsum, mctl, table=table
                )
        return written, late

    @cute.jit
    def select_rescue(
        self, his, los, valid, nshort, owners, cnts, rescue_base, tid, lane, warp, hist, wsum, ctl
    ):
        """Unsigned image ``t``: the ``K`` largest of the candidates (the keys in registers where
        ``valid``, and the short ranks' pushed keys, re-read from this rank's rescue slots) are
        exactly those ``>= t``. The candidates are distinct and at least ``K``."""
        K, T = self.top_k, self.threads
        prefix_hi = Uint32(0)
        prefix_lo = Uint32(0)
        remaining = Uint32(K)
        depth = Uint32(0)
        done = Uint32(0)
        consumed = 0
        for r in cutlass.range_constexpr(2 * len(_ROUND_BITS)):
            bits = _ROUND_BITS[r % 3]
            if done == Uint32(0):
                for j in cutlass.range_constexpr(len(his)):
                    active, digit = _digit(r, his[j], los[j], valid[j], prefix_hi, prefix_lo)
                    ptx.count_shared_if(active, hist + digit * Uint32(4))
                for s in cutlass.range(Int32(nshort), unroll=1):
                    src = ptx.ld_shared_u32(owners + Uint32(s) * Uint32(4))
                    c = ptx.min_u32(ptx.ld_shared_u32(cnts + src * Uint32(4)), K)
                    for i in cutlass.range_constexpr(K // T):
                        idx = Uint32(i * T) + tid
                        inside = ptx.lt_u32(idx, c)
                        hi, _, lo, _ = ptx.ld_ll_pair_if(
                            inside,
                            rescue_base + (Uint64(s) * Uint64(K) + Uint64(idx)) * Uint64(16),
                        )
                        active, digit = _digit(r, hi, lo, inside, prefix_hi, prefix_lo)
                        ptx.count_shared_if(active, hist + digit * Uint32(4))
                bucket, above, count = self.resolve(
                    tid, lane, warp, hist, wsum, ctl, remaining, bits, Uint32(T // 32)
                )
                remaining = remaining - above
                if cutlass.const_expr(r < 3):
                    prefix_hi = (prefix_hi << Uint32(bits)) | bucket
                else:
                    prefix_lo = (prefix_lo << Uint32(bits)) | bucket
                depth = Uint32(consumed + bits)
                done = ptx.eq_u32(count, remaining)
            consumed += bits
        shift = Uint32(64) - depth
        high = ptx.sel64(
            ptx.lt_u32(32, depth),
            ptx.pack64(0, prefix_hi),
            ptx.shl_u64(ptx.pack64(prefix_hi, 0), shift),
        )
        return high | ptx.shl_u64(ptx.pack64(prefix_lo, 0), shift)


def _architecture(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


@functools.lru_cache(maxsize=None)
def compile_exchange(
    world: int,
    local_k: int,
    threads: int,
    per: int,
    arch: str,
    top_k: int,
    bounds64: bool = False,
    mask: bool = False,
    trace: bool = False,
    rows: bool = False,
    extra_options: str = "",
    record: int = 0,
):
    kernel = KeyShardExchangeKernel(
        world, local_k, threads, per, top_k, bounds64, mask, trace, rows
    )
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
        Int64(0),
        Int32(0),
        Int32(0),
        Int64(0),
        Int32(0),
        Int32(0),
        stream,
        options=f"--gpu-arch={arch} --enable-tvm-ffi {extra_options}".strip(),
    )
