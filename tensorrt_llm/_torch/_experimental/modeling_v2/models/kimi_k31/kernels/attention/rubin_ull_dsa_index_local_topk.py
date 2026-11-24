# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Adapted from vllm-project/vllm@e3ddbdbd60
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/dsa/indexer.py (IndexerLocalTopk:
# LOCAL_CLUSTER, LOCAL_SPLIT_FROM, cluster_kernel, select_split with its share bound and candidate
# lists, the DSMEM helpers). Changed: the keys are the order-preserving 32-bit images of the FP32
# logits with the inverted global index below them (score desc, local index asc), not 16-bit BF16
# keys; the shares write their lists straight into the leader's list (instead of the leader
# reading them), and the leader re-selects the gathered list with select_pack's radix rounds;
# rows whose lists overflow run select_pack.select_row on the leader (instead of passes summed
# over the cluster); the leader then pushes and merges in the same kernel (exchange.py), so there
# is no separate ExchangeMerge launch.
"""Key-shard local Top-L on a thread-block cluster per row, then the LL exchange and merge.

``KeyShardClusterExchangeKernel`` is ``exchange.KeyShardExchangeKernel`` with ``cluster`` CTAs per
row (grid ``rows * cluster``, cluster ``(cluster, 1, 1)``; cluster CTA 0 is the row's leader).
A row of ``n`` local keys splits into ``ways = min(cluster, ceil(n / share_keys))`` contiguous
shares (multiples of 4 keys), one per CTA (the cluster's other CTAs hold none), each held in
registers, if ``ways > 1`` and a share fits one chunk (``threads * per`` keys):

0. bound: the ``L``-th largest of the share's per-thread maxima is at most the row's ``L``-th
   largest key image (``L`` threads of the share hold a key at least as large), so every key of
   the row's Top-L is at or above the 11-bit bin of that thread maximum: the share's candidates
   are its keys at or above that bin (all its keys when fewer than ``L`` threads hold keys);
1. each CTA counts its candidates and publishes the count; after a cluster barrier (release /
   acquire) every CTA knows the lower shares' counts and the total. If the total fits the list
   (``LIST_KEYS``), each CTA writes its candidates (key image, row index) into the leader's list
   after the lower shares' (``st.shared::cluster``), so the list is in ascending local position;
   a second cluster barrier publishes it and the other CTAs exit;
2. the leader runs ``select_row``'s radix rounds (11, 11, 10 bits) and staging over the list.

A row whose lists overflow (e.g. long runs of equal logits) runs ``select_row`` on the leader
alone, as do shorter rows and rows whose shares exceed a chunk; the other CTAs exit at once.

The leader then pushes and merges exactly as ``KeyShardExchangeKernel`` does (rescue included,
whose re-selection runs ``select_row`` on the leader). The selected set (the exact local Top-L by
key: the list holds every key of it, so its Top-L by key is the row's) and the stage order
(ascending local position) do not depend on the split, so every output is the one-CTA kernel's.
With ``rows`` the merge (in the same kernel or the split merge grid) writes the own winners as KV
pool rows, exactly as ``KeyShardExchangeKernel``'s ``rows`` variant (``ptx.kv_row_if``).

``tops`` (one CTA per row; vLLM ``IndexerSelect``'s bound at FP32): the scorer also writes each 8
keys' largest image (``score.ScoreKernel(tops=True)``). The ``target``-th largest of a row's
vector maxima is at most its ``target``-th largest key (``target`` vectors each hold a key at least
as large), so every key of the Top-``target`` is at or above the bin of that maximum (11 bits; 22
after a second round when more than ``REFINE_FROM`` vectors reach the first bin). The hot vectors
(maximum at or above the bound) are listed in ascending position, their keys at or above the bound
gathered into the list (ascending local position), and the list selected in two rounds: the
22-bit prefix as an offset from the bound's (2048 bins, the last open-ended: then one more round
over the next 2048), then the low 10 bits. Rows of fewer than ``tops_from`` keys and rows with
more than ``HOT`` hot vectors or ``LIST_KEYS`` candidates (e.g. long runs of equal logits) take
``select_row``; a threshold past both ranges takes ``select_listed``'s three rounds. The staged
keys and their order are ``select_row``'s.

Assumptions: ``rows * cluster`` CTAs fit the device at once (checked by the host against the SM
count: at most one CTA per SM is assumed), cluster sizes up to 8 (the portable maximum).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64

from . import rubin_ull_dsa_index_ptx as ptx
from .rubin_ull_dsa_index_exchange import KeyShardExchangeKernel
from .rubin_ull_dsa_index_select_pack import _CTL, _HIST, _MCTL, _ROUND_BITS, _STAGE, _WMIN, _WSUM, LIST_KEYS, SITE_OWN

#: CTAs per row (one cluster per row) at 16 key-shard ranks; 1: one CTA per row (exchange.py).
CLUSTER = 4
#: Local keys per share: a row splits into min(CLUSTER, ceil(n / SHARE_KEYS)) shares (one: the
#: leader alone). One register chunk at the served shape (512 threads x 16); on GB300
#: shares of 7040 / 8192 keys beat one CTA by 0.6 / 4.6 us at 14080 / 32768 keys, 3520 lost 0.7.
SHARE_KEYS = 8192
#: Portable thread-block cluster size limit.
CLUSTER_MAX = 8
#: Bits of the bound's bins (the first radix round's).
BOUND_BITS = 11
#: ``tops``: vector maxima (8 keys each) a thread holds (the served width, 62464 keys, is 7808
#: vectors: at most 16 per thread of 512).
TOPS_PER = 16
#: ``tops``: hot vectors (maxima at or above the bound) the candidate list is gathered from; more:
#: ``select_row``.
HOT = 1024
#: ``tops``: above this many hot vectors after the first round on the maxima, a second round
#: tightens the bound.
REFINE_FROM = 512
#: ``tops``: rows of at least this many local keys select from the maxima (shorter: select_row);
#: None: rows longer than one register chunk (``threads * per`` keys). Shorter rows lose on the
#: scorer's logits (GB300): within one chunk ``select_row`` reads the row once, and the
#: ``L``-th largest of fewer maxima bounds loosely (at 4096 keys 7 of 8 rows took the fallback).
TOPS_FROM = None

_ALL = 0xFFFFFFFFFFFFFFFF


class KeyShardClusterExchangeKernel(KeyShardExchangeKernel):
    """``KeyShardExchangeKernel`` with ``cluster`` CTAs per row (module doc). ``select_only``
    (benchmarks and checks): the leader writes the staged keys and their minimum to ``out_addr``
    (``[rows, local_k + 1]`` u64) instead of pushing and merging. ``split_kernels``: vLLM's
    structure, the select (and push) grid triggers its dependents at once and a separate merge
    grid (one CTA per row, launched early by PDL) settles its own row's pushed keys into its stage
    and merges (``merge_kernel``); else the leader merges in the same kernel. ``rows``: the merge
    writes the own winners as KV pool rows (``KeyShardExchangeKernel``, ``kv_table``)."""

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
        cluster: int = CLUSTER,
        select_only: bool = False,
        split_kernels: bool = False,
        tops: bool = False,
        rows: bool = False,
        publish: bool = False,
    ):
        super().__init__(world, local_k, threads, per, top_k, bounds64, mask, trace, rows)
        assert 1 <= cluster <= CLUSTER_MAX
        assert not tops or (cluster == 1 and HOT % threads == 0 and TOPS_PER % 4 == 0)
        self.cluster = cluster
        self.select_only = select_only
        self.split_kernels = split_kernels and not select_only
        # ``publish`` (split merge grid only; the DSA stack's speculative decode): the merge grid
        # triggers its dependents at once and publishes each row's threshold (``merge_exchange``).
        assert not publish or self.split_kernels
        self.publish = publish
        self.tops = tops
        # The shares' candidate counts.
        self.xc_offset = self.smem_bytes
        self.smem_bytes = self.xc_offset + 4 * 32
        # ``tops``: the hot vectors' indices.
        self.hot_offset = self.smem_bytes
        if tops:
            self.smem_bytes = self.hot_offset + 4 * HOT

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
        share_keys: Int32,
        tops_addr: Int64,
        tops_from: Int32,
        table_addr: Int64,
        table_pitch: Int32,
        table_cols: Int32,
        req_addr: Int64,
        stride_rows: Int32,
        row_offset: Int32,
        pub_addr: Int64,
        stream,
    ):
        kernel = self.cluster_kernel(
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
            share_keys,
            tops_addr,
            tops_from,
            table_addr,
            table_pitch,
            table_cols,
            req_addr,
            stride_rows,
            row_offset,
        )
        if cutlass.const_expr(self.cluster > 1):
            kernel.launch(
                grid=(rows * self.cluster, 1, 1),
                block=(self.threads, 1, 1),
                cluster=(self.cluster, 1, 1),
                smem=self.smem_bytes,
                stream=stream,
                use_pdl=True,
            )
        else:
            kernel.launch(
                grid=(rows, 1, 1),
                block=(self.threads, 1, 1),
                smem=self.smem_bytes,
                stream=stream,
                use_pdl=True,
            )
        if cutlass.const_expr(self.split_kernels):
            self.merge_kernel(
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
                pub_addr,
            ).launch(
                grid=(rows, 1, 1),
                block=(self.threads, 1, 1),
                smem=self.smem_bytes,
                stream=stream,
                use_pdl=True,
            )

    @cute.kernel
    def cluster_kernel(
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
        share_keys: Int32,
        tops_addr: Int64,
        tops_from: Int32,
        table_addr: Int64,
        table_pitch: Int32,
        table_cols: Int32,
        req_addr: Int64,
        stride_rows: Int32,
        row_offset: Int32,
    ):
        W, L, T, C = self.world, self.local_k, self.threads, self.cluster
        CAP = T * self.per
        tid_i, _, _ = cute.arch.thread_idx()
        bid_i, _, _ = cute.arch.block_idx()
        tid = Uint32(tid_i)
        lane = tid & Uint32(31)
        warp = tid >> Uint32(5)
        row = Uint32(bid_i)
        crank = Uint32(0)
        if cutlass.const_expr(C > 1):
            row = Uint32(bid_i) // Uint32(C)
            crank = ptx.cluster_ctarank()

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

        # Logits, row lengths and the epoch come from earlier grids; dependents (a separate merge
        # grid) launch now, else at the end.
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.split_kernels):
            cute.arch.griddepcontrol_launch_dependents()
        n = ptx.ld_global_u32(Uint64(rowlen_addr) + Uint64(row) * Uint64(4))
        target = ptx.min_u32(n, L)
        rank_u = Uint32(rank)
        row_base = Uint64(logits_addr) + Uint64(row) * Uint64(logits_stride) * Uint64(4)
        leader = ptx.eq_u32(crank, 0)
        # min(C, ceil(n / share_keys)) shares of whole groups of 4 keys (CTAs past them hold none);
        # the row splits into two or more if a share fits a chunk (the same decision in every CTA of
        # the cluster).
        ways = ptx.min_u32((n + Uint32(share_keys) - Uint32(1)) // Uint32(share_keys), C)
        ways = ptx.max_u32(ways, 1)
        seg = ((n + ways - Uint32(1)) // ways + Uint32(3)) // Uint32(4) * Uint32(4)
        split = Uint32(0)
        if cutlass.const_expr(C > 1):
            split = ptx.lt_u32(1, ways) & ptx.lt_u32(seg, CAP + 1)

        self.bar(Uint32(T // 32))
        # Probes: the leader's thread 0 only.
        self.stamp(trace_addr, row, tid + crank * Uint32(T), 0)
        if cutlass.const_expr(C > 1):
            alone = leader
            if split != Uint32(0):
                listed = self.gather_split(
                    row_base, n, seg, tid, lane, warp, base, crank, trace_addr, row
                )
                if leader != Uint32(0):
                    if ptx.lt_u32(listed, LIST_KEYS + 1) != Uint32(0):
                        alone = Uint32(0)
                        self.select_listed(listed, target, tid, lane, warp, base, rank_u)
                else:
                    alone = Uint32(0)
            if alone != Uint32(0):
                self.select_row(
                    row_base,
                    n,
                    target,
                    Uint32(T),
                    tid,
                    lane,
                    warp,
                    base,
                    stage,
                    rank_u,
                    trace_addr,
                    row,
                )
        else:
            pending = Uint32(1)
            if cutlass.const_expr(self.tops):
                # (The host passes a tops_from past every row when the maxima are not written.)
                if ptx.lt_u32(n, Uint32(tops_from)) == Uint32(0):
                    tops_pitch = Uint32(logits_stride) >> Uint32(3)
                    tops_row = Uint64(tops_addr) + Uint64(row) * Uint64(tops_pitch) * Uint64(4)
                    pending = Uint32(1) - self.select_tops(
                        row_base, tops_row, n, target, tid, lane, warp, base, rank_u, trace_addr, row
                    )
            if pending != Uint32(0):
                self.select_row(
                    row_base,
                    n,
                    target,
                    Uint32(T),
                    tid,
                    lane,
                    warp,
                    base,
                    stage,
                    rank_u,
                    trace_addr,
                    row,
                )
        if leader != Uint32(0):
            self.stamp(trace_addr, row, tid, 7)
            smallest = self.staged_min(wmin)
            if cutlass.const_expr(self.select_only):
                dump = Uint64(out_addr) + Uint64(row) * Uint64(8 * (L + 1))
                for it in cutlass.range_constexpr((L + T - 1) // T):
                    i = Uint32(it * T) + tid
                    key = ptx.ld_shared_u64(stage + ptx.min_u32(i, L - 1) * Uint32(8))
                    ptx.st_global_u64_if(
                        ptx.lt_u32(i, L),
                        dump + Uint64(i) * Uint64(8),
                        ptx.sel64(ptx.lt_u32(i, target), key, Uint64(0)),
                    )
                ptx.st_global_u64_if(ptx.eq_u32(tid, 0), dump + Uint64(8 * L), smallest)
            else:
                state_addr = Uint64(state.iterator.toint())
                epoch = ptx.ld_global_u32(state_addr) + Uint32(1)
                deadline = ptx.globaltimer() + Uint64(timeout_us) * Uint64(1000)
                ll = Uint64(ll_addr)
                block = ((epoch & Uint32(1)) * Uint32(rows_max) + row) * Uint32(self.block)
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
                if cutlass.const_expr(not self.split_kernels):
                    table = self.kv_table(
                        req_addr,
                        table_addr,
                        table_pitch,
                        table_cols,
                        stride_rows,
                        row_offset,
                        row,
                        n,
                        tid,
                    )
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

    @cute.kernel
    def merge_kernel(
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
        pub_addr: Int64,
    ):
        """``split_kernels``: the merge of row ``blockIdx.x``, launched early by PDL. The epoch and
        the row lengths come from grids before the select grid (complete when it triggered); this
        rank's own pushed keys (ascending local position) are settled into the stage, then (after
        ``griddepcontrol.wait``: the select grid is complete, its logits visible to a rescue)
        ``merge_exchange`` runs as in the fused kernel. ``rows``: the request index, the block-table
        row and its entries are read after the wait (as in ``KeyShardExchangeKernel.kernel``): the
        request index is written in the step by a torch kernel (``_cached_req_idx_gen``) whose
        order against the select grid this kernel does not rely on. ``publish``: dependents launch
        at once (the speculative decode reads nothing of this grid but its published pairs)."""
        if cutlass.const_expr(self.publish):
            cute.arch.griddepcontrol_launch_dependents()
        L, T = self.local_k, self.threads
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
        mctl = base + _MCTL
        stage = base + _STAGE
        self.clear_hist(tid, hist)
        ptx.st_shared_u32_if(
            ptx.lt_u32(tid, 16), mctl + tid * Uint32(4), ptx.sel(ptx.eq_u32(tid, 4), 0xFFFFFFFF, 0)
        )
        state_addr = Uint64(state.iterator.toint())
        epoch = ptx.ld_global_u32(state_addr) + Uint32(1)
        n = ptx.ld_global_u32(Uint64(rowlen_addr) + Uint64(row) * Uint64(4))
        target = ptx.min_u32(n, L)
        rank_u = Uint32(rank)
        row_base = Uint64(logits_addr) + Uint64(row) * Uint64(logits_stride) * Uint64(4)
        deadline = ptx.globaltimer() + Uint64(timeout_us) * Uint64(1000)
        ll = Uint64(ll_addr)
        block = ((epoch & Uint32(1)) * Uint32(rows_max) + row) * Uint32(self.block)
        own = ll + Uint64(block + rank_u * Uint32(L)) * Uint64(16)
        late = Uint32(0)
        for it in cutlass.range_constexpr((L + T - 1) // T):
            i = Uint32(it * T) + tid
            inside = ptx.lt_u32(i, target)
            at = own + Uint64(i) * Uint64(16)
            w0, e0, w1, e1 = ptx.ld_ll_pair_if(inside, at)
            hi, lo, timed_out = ptx.settle_ll_if(inside, at, w0, e0, w1, e1, epoch, deadline)
            late = late | timed_out
            ptx.st_shared_u64_if(
                inside, stage + i * Uint32(8), ptx.pack64(lo, hi ^ Uint32(0x80000000))
            )
        ptx.red_add_global_if(late, state_addr + Uint64(12), 1)
        self.trap_late(late, SITE_OWN, row, tid, epoch)
        cute.arch.griddepcontrol_wait()
        # The request index after the wait; the table row's L2 prefetch overlaps the merge's
        # histogram rounds, its entries are read by the writes at the end.
        table = self.kv_table(
            req_addr, table_addr, table_pitch, table_cols, stride_rows, row_offset, row, n, tid
        )
        self.bar(Uint32(T // 32))
        # (Probes 0, 7, 1: the select grid's; 2-4: this merge's.)
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
            pub=Uint64(pub_addr) if cutlass.const_expr(self.publish) else None,
        )

    @cute.jit
    def kv_table(
        self, req_addr, table_addr, table_pitch, table_cols, stride_rows, row_offset, row, n, tid
    ):
        """``rows``: the merge's ``table`` argument (``KeyShardExchangeKernel.merge_exchange``)
        for row ``row`` of ``n`` keys, its block-table row's pages the row's keys can use sent to
        L2 (as ``KeyShardExchangeKernel.kernel``); else None."""
        table = None
        if cutlass.const_expr(self.kv_rows):
            req = ptx.ld_global_u32(Uint64(req_addr) + Uint64(row) * Uint64(4))
            row_table = Uint64(table_addr) + Uint64(req) * Uint64(table_pitch) * Uint64(4)
            pages = ptx.min_u32((n + Uint32(63)) >> Uint32(6), Uint32(table_cols))
            first_line = row_table >> Uint64(7)
            last_line = (row_table + Uint64(pages) * Uint64(4) + Uint64(127)) >> Uint64(7)
            lines = Uint32(last_line - first_line)
            ptx.prefetch_l2_if(ptx.lt_u32(tid, lines), (first_line + Uint64(tid)) << Uint64(7))
            table = (row_table, pages, Uint32(stride_rows), Uint32(row_offset))
        return table

    @cute.jit
    def gather_split(self, row_base, n, seg, tid, lane, warp, base, crank, trace_addr, row):
        """Steps 0 and 1 of the module doc on this CTA's share of a split row: returns the
        cluster's candidate total; when it fits ``LIST_KEYS`` the leader's list holds them in
        ascending local position. Every CTA of the cluster calls it."""
        C, T, L = self.cluster, self.threads, self.local_k
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        lst = base + Uint32(self.list_offset)
        xc = base + Uint32(self.xc_offset)
        warps = Uint32(T // 32)
        s0 = ptx.min_u32(crank * seg, n)
        m = ptx.min_u32(seg, n - s0)
        group = ((m + Uint32(3)) // Uint32(4) + Uint32(T - 1)) // Uint32(T)
        span = group * Uint32(4)
        start0 = tid * span
        slots0 = ptx.min_u32(span, m - ptx.min_u32(m, start0))
        images = self.load_chunk(row_base + Uint64(s0) * Uint64(4), start0, slots0)
        probe = tid + crank * Uint32(T)

        # 0: the bound's bin (0 when fewer than L threads hold keys: every key is a candidate).
        top = Uint32(0)
        for j in cutlass.range_constexpr(len(images)):
            top = ptx.max_u32(top, ptx.sel(ptx.lt_u32(j, slots0), images[j], 0))
        cut = Uint32(32 - BOUND_BITS)
        ptx.count_shared_if(
            Uint32(1) - ptx.lt_u32(slots0, 1), hist + ptx.shr_u32(top, cut) * Uint32(4)
        )
        ptx.st_shared_u32_if(ptx.eq_u32(tid, 0), ctl, 0)
        bound, _, _ = self.resolve(tid, lane, warp, hist, wsum, ctl, Uint32(L), BOUND_BITS, warps)
        self.stamp(trace_addr, row, probe, 2)

        # 1: the candidates, counted, then written into the leader's list after the lower shares'.
        count = Uint32(0)
        for j in cutlass.range_constexpr(len(images)):
            count = count + (
                ptx.lt_u32(j, slots0) & (Uint32(1) - ptx.lt_u32(ptx.shr_u32(images[j], cut), bound))
            )
        inclusive, total = self.block_scan(count, lane, warp, wsum, warps, T // 32)
        ptx.st_shared_u32_if(ptx.eq_u32(tid, 0), xc, total)
        self.stamp(trace_addr, row, probe, 3)
        ptx.cluster_sync()
        self.stamp(trace_addr, row, probe, 4)
        before = Uint32(0)
        listed = Uint32(0)
        for r in cutlass.range_constexpr(C):
            v = ptx.ld_cluster_u32(ptx.mapa_shared(xc, Uint32(r)))
            before = before + ptx.sel(ptx.lt_u32(r, crank), v, 0)
            listed = listed + v
        if ptx.lt_u32(listed, LIST_KEYS + 1) != Uint32(0):
            lead = ptx.mapa_shared(lst, Uint32(0))
            position = before + inclusive - count
            for j in cutlass.range_constexpr(len(images)):
                keep = ptx.lt_u32(j, slots0) & (
                    Uint32(1) - ptx.lt_u32(ptx.shr_u32(images[j], cut), bound)
                )
                ptx.st_cluster_u32_if(keep, lead + position * Uint32(4), images[j])
                ptx.st_cluster_u32_if(
                    keep,
                    lead + (Uint32(LIST_KEYS) + position) * Uint32(4),
                    s0 + start0 + Uint32(j),
                )
                position = position + keep
        self.stamp(trace_addr, row, probe, 5)
        # Release the lists to the leader; no CTA touches a peer after this.
        ptx.cluster_sync()
        self.stamp(trace_addr, row, probe, 6)
        return listed

    @cute.jit
    def select_listed(self, listed, target, tid, lane, warp, base, rank_u):
        """Step 2 of the module doc: ``select_row``'s rounds and staging over the ``listed``
        candidates of the leader's list (in ascending local position, row indices)."""
        T = self.threads
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        wmin = base + _WMIN
        stage = base + _STAGE
        lst = base + Uint32(self.list_offset)
        active = Uint32(T)
        warps = Uint32(T // 32)
        prefix = Uint32(0)
        remaining = target
        depth = Uint32(0)
        done = ptx.eq_u32(target, 0)
        consumed = 0
        for r in cutlass.range_constexpr(len(_ROUND_BITS)):
            bits = _ROUND_BITS[r]
            if done == Uint32(0):
                shift = 32 - consumed - bits
                for i in cutlass.range(Int32((listed + active - Uint32(1)) // active), unroll=1):
                    entry = Uint32(i) * active + tid
                    inside = ptx.lt_u32(entry, listed)
                    image = ptx.ld_shared_u32_if(inside, lst + entry * Uint32(4))
                    match = inside
                    if cutlass.const_expr(consumed > 0):
                        match = match & ptx.eq_u32(image >> Uint32(32 - consumed), prefix)
                    digit = (image >> Uint32(shift)) & Uint32((1 << bits) - 1)
                    ptx.count_shared_if(match, hist + digit * Uint32(4))
                bucket, above, count = self.resolve(
                    tid, lane, warp, hist, wsum, ctl, remaining, bits, warps
                )
                remaining = remaining - above
                prefix = (prefix << Uint32(bits)) | bucket
                depth = Uint32(consumed + bits)
                done = ptx.eq_u32(count, remaining)
            consumed += bits
        smallest = self.compact_list(
            lst,
            listed,
            Uint32(32) - depth,
            prefix,
            remaining,
            tid,
            lane,
            warp,
            wsum,
            warps,
            active,
            stage,
            rank_u,
        )
        smallest = ptx.warp_min_u64(smallest)
        ptx.st_shared_u64_if(ptx.eq_u32(lane, 0), wmin + warp * Uint32(8), smallest)
        self.bar(warps)

    @cute.jit
    def select_tops(
        self, row_base, tops_row, n, target, tid, lane, warp, base, rank_u, trace_addr, row
    ):
        """``tops``: stage the row's exact Top-``target`` from the scorer's vector maxima (module
        doc); returns 0 (nothing staged, the shared state as on entry) when the hot vectors or the
        candidates overflow, else 1. Needs ``n >= 8 * target``."""
        T = self.threads
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        lst = base + Uint32(self.list_offset)
        hotl = base + Uint32(self.hot_offset)
        warps = Uint32(T // 32)
        # Each thread: a run of whole groups of 4 consecutive vectors (in ascending position).
        nv = (n + Uint32(7)) >> Uint32(3)
        vrun = ((nv + Uint32(T - 1)) // Uint32(T) + Uint32(3)) // Uint32(4) * Uint32(4)
        v0 = tid * vrun
        vslots = ptx.min_u32(vrun, nv - ptx.min_u32(nv, v0))
        tops = []
        for q in cutlass.range_constexpr(TOPS_PER // 4):
            words = ptx.ld_global_v4_if(
                ptx.lt_u32(4 * q, vslots), tops_row + Uint64(v0 + Uint32(4 * q)) * Uint64(4)
            )
            for k in cutlass.range_constexpr(4):
                tops.append(words[k])
        self.stamp(trace_addr, row, tid, 5)

        # The bound: the bin (11 bits; with many hot vectors, 22) of the target-th largest
        # maximum. At least target maxima, so target keys, are at or above it.
        for j in cutlass.range_constexpr(TOPS_PER):
            ptx.count_shared_if(ptx.lt_u32(j, vslots), hist + ptx.shr_u32(tops[j], 21) * Uint32(4))
        b0, above, count = self.resolve(tid, lane, warp, hist, wsum, ctl, target, 11, warps)
        bound = b0 << Uint32(21)
        hot = above + count
        self.stamp(trace_addr, row, tid, 1)
        if ptx.lt_u32(REFINE_FROM, hot) != Uint32(0):
            for j in cutlass.range_constexpr(TOPS_PER):
                match = ptx.lt_u32(j, vslots) & ptx.eq_u32(ptx.shr_u32(tops[j], 21), b0)
                digit = ptx.shr_u32(tops[j], 10) & Uint32(2047)
                ptx.count_shared_if(match, hist + digit * Uint32(4))
            b1, above1, count1 = self.resolve(
                tid, lane, warp, hist, wsum, ctl, target - above, 11, warps
            )
            bound = bound | (b1 << Uint32(10))
            hot = above + above1 + count1
        self.stamp(trace_addr, row, tid, 2)

        staged = Uint32(0)
        if ptx.lt_u32(hot, HOT + 1) != Uint32(0):
            # The hot vectors' positions, ascending.
            count = Uint32(0)
            for j in cutlass.range_constexpr(TOPS_PER):
                count = count + (
                    ptx.lt_u32(j, vslots) & (Uint32(1) - ptx.lt_u32(tops[j], bound))
                )
            inclusive, _ = self.block_scan(count, lane, warp, wsum, warps, T // 32)
            position = inclusive - count
            for j in cutlass.range_constexpr(TOPS_PER):
                keep = ptx.lt_u32(j, vslots) & (Uint32(1) - ptx.lt_u32(tops[j], bound))
                ptx.st_shared_u32_if(keep, hotl + position * Uint32(4), v0 + Uint32(j))
                position = position + keep
            self.bar(warps)
            self.stamp(trace_addr, row, tid, 3)

            # Their keys at or above the bound (all of the Top-target), into the list in
            # ascending position: each thread loads a run of the hot vectors.
            hrun = (hot + Uint32(T - 1)) // Uint32(T)
            h0 = ptx.min_u32(tid * hrun, hot)
            h1 = ptx.min_u32(h0 + hrun, hot)
            # (Few live registers: per hot vector its first key index (n past this thread's run)
            # and its 8 images; the index and the row length mask the keys past the row.)
            images = []
            firsts = []
            for u in cutlass.range_constexpr(HOT // T):
                entry = h0 + Uint32(u)
                inside = ptx.lt_u32(entry, h1)
                v = ptx.ld_shared_u32_if(inside, hotl + entry * Uint32(4))
                at = row_base + Uint64(v) * Uint64(32)
                lo = ptx.ld_global_v4_if(inside, at)
                hi = ptx.ld_global_v4_if(inside, at + Uint64(16))
                first = ptx.sel(inside, v * Uint32(8), n)
                firsts.append(first)
                for k in cutlass.range_constexpr(8):
                    word = lo[k] if k < 4 else hi[k - 4]
                    images.append(ptx.ordered_bits(word))
            count = Uint32(0)
            for i in cutlass.range_constexpr(len(images)):
                keep = ptx.lt_u32(firsts[i // 8] + Uint32(i % 8), n) & (
                    Uint32(1) - ptx.lt_u32(images[i], bound)
                )
                count = count + keep
            inclusive, listed = self.block_scan(count, lane, warp, wsum, warps, T // 32)
            if ptx.lt_u32(listed, LIST_KEYS + 1) != Uint32(0):
                position = inclusive - count
                for i in cutlass.range_constexpr(len(images)):
                    index = firsts[i // 8] + Uint32(i % 8)
                    keep = ptx.lt_u32(index, n) & (Uint32(1) - ptx.lt_u32(images[i], bound))
                    ptx.st_shared_u32_if(keep, lst + position * Uint32(4), images[i])
                    ptx.st_shared_u32_if(
                        keep, lst + (Uint32(LIST_KEYS) + position) * Uint32(4), index
                    )
                    position = position + keep
                self.bar(warps)
                self.stamp(trace_addr, row, tid, 4)
                self.select_fine(listed, target, bound, tid, lane, warp, base, rank_u)
                staged = Uint32(1)
            else:
                # (wsum is read by the scan above; select_row's first scan rewrites it.)
                self.bar(warps)
        return staged

    @cute.jit
    def select_fine(self, listed, target, bound, tid, lane, warp, base, rank_u):
        """``tops``: the Top-``target`` of the ``listed`` candidates (every one at or above
        ``bound``) staged as ``select_listed`` does: the 22-bit prefix as an offset from the
        bound's in 2048 bins, the last open-ended (a threshold there: one more such round over the
        next 2048 offsets, and past those ``select_listed``'s three rounds), then the low 10 bits
        unless the threshold's bin is taken whole. Fine bins of 2^10 keep most thresholds' bins
        whole on the scorer's logits (2^11 cost 0.3-0.9 us more at 220k-512k on GB300)."""
        T = self.threads
        hist = base + _HIST
        wsum = base + _WSUM
        ctl = base + _CTL
        wmin = base + _WMIN
        stage = base + _STAGE
        lst = base + Uint32(self.list_offset)
        active = Uint32(T)
        warps = Uint32(T // 32)
        floor = ptx.shr_u32(bound, 10)
        steps = (listed + active - Uint32(1)) // active
        for i in cutlass.range(Int32(steps), unroll=1):
            entry = Uint32(i) * active + tid
            inside = ptx.lt_u32(entry, listed)
            image = ptx.ld_shared_u32_if(inside, lst + entry * Uint32(4))
            digit = ptx.min_u32(ptx.shr_u32(image, 10) - floor, 2047)
            ptx.count_shared_if(inside, hist + digit * Uint32(4))
        bucket, above, count = self.resolve(tid, lane, warp, hist, wsum, ctl, target, 11, warps)
        if ptx.eq_u32(bucket, 2047) != Uint32(0):
            # Every candidate above the open bin's start is in it (above == 0): its next 2048
            # offsets.
            for i in cutlass.range(Int32(steps), unroll=1):
                entry = Uint32(i) * active + tid
                inside = ptx.lt_u32(entry, listed)
                image = ptx.ld_shared_u32_if(inside, lst + entry * Uint32(4))
                offset = ptx.shr_u32(image, 10) - floor
                match = inside & (Uint32(1) - ptx.lt_u32(offset, 2047))
                digit = ptx.min_u32(offset - ptx.min_u32(offset, 2047), 2047)
                ptx.count_shared_if(match, hist + digit * Uint32(4))
            bucket, above, count = self.resolve(
                tid, lane, warp, hist, wsum, ctl, target, 11, warps
            )
            bucket = bucket + Uint32(2047)
        if ptx.eq_u32(bucket, 2 * 2047) != Uint32(0):
            self.select_listed(listed, target, tid, lane, warp, base, rank_u)
        else:
            remaining = target - above
            prefix = floor + bucket
            cut = Uint32(10)
            if ptx.eq_u32(count, remaining) == Uint32(0):
                for i in cutlass.range(Int32(steps), unroll=1):
                    entry = Uint32(i) * active + tid
                    inside = ptx.lt_u32(entry, listed)
                    image = ptx.ld_shared_u32_if(inside, lst + entry * Uint32(4))
                    match = inside & ptx.eq_u32(ptx.shr_u32(image, 10), prefix)
                    ptx.count_shared_if(match, hist + (image & Uint32(1023)) * Uint32(4))
                low, above2, _ = self.resolve(
                    tid, lane, warp, hist, wsum, ctl, remaining, 10, warps
                )
                remaining = remaining - above2
                prefix = (prefix << Uint32(10)) | low
                cut = Uint32(0)
            smallest = self.compact_list(
                lst,
                listed,
                cut,
                prefix,
                remaining,
                tid,
                lane,
                warp,
                wsum,
                warps,
                active,
                stage,
                rank_u,
            )
            smallest = ptx.warp_min_u64(smallest)
            ptx.st_shared_u64_if(ptx.eq_u32(lane, 0), wmin + warp * Uint32(8), smallest)
            self.bar(warps)


@functools.lru_cache(maxsize=None)
def compile_cluster_exchange(
    world: int,
    local_k: int,
    threads: int,
    per: int,
    arch: str,
    top_k: int,
    bounds64: bool = False,
    mask: bool = False,
    trace: bool = False,
    cluster: int = CLUSTER,
    select_only: bool = False,
    split_kernels: bool = False,
    tops: bool = False,
    rows: bool = False,
    extra_options: str = "",
    record: int = 0,
    publish: bool = False,
):
    kernel = KeyShardClusterExchangeKernel(
        world, local_k, threads, per, top_k, bounds64, mask, trace, cluster, select_only,
        split_kernels, tops, rows, publish,
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
        Int32(0),
        Int64(0),
        Int32(0),
        Int64(0),
        Int32(0),
        Int32(0),
        Int64(0),
        Int32(0),
        Int32(0),
        Int64(0),
        stream,
        options=f"--gpu-arch={arch} --enable-tvm-ffi {extra_options}".strip(),
    )
