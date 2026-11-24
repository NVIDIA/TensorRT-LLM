# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Adapted from vllm-project/vllm@e3ddbdbd60
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/dsa/attn_sparse.py (SparseDcpDecode, the
# vendored copy ``kernels/vendor/vllm_main/.../dsa/attn_sparse.py``). Changed: the kernel no
# longer waits for the key-shard merge grid; it starts on the row's staged candidates and masks
# them by the threshold the merge publishes (module doc); its ``stage`` / ``merge`` are the
# vendored class's.
"""Speculative sparse DCP decode of the K3.1 ULL DSA stack.

The vendored ``SparseDcpDecode`` waits (``griddepcontrol.wait``) for the key-shard merge grid
and reads its winners (``rows`` / ``valid``). Here the decode does not wait: the merge grid
(``local_topk.merge_kernel`` with ``publish``) triggers it at once and publishes, per row, the LL
pair ``(thr high, thr low; epoch)`` at ``pub + 32 row`` (``exchange.merge_exchange``). A row's
winners on this rank are its staged candidates (the exact local Top-``L`` by key, pushed by the
select grid into this rank's own LL slot) whose key image is at least ``thr``. So the cluster of
a (head tiles, row) gathers the KV rows of the candidates and computes QK while the merge settles
and selects:

- candidates: CTA ``s`` of the cluster takes staged keys ``[64 s, 64 s + 64)`` of the row's
  ``min(n, L)`` (``L`` = ``SPLIT * CHUNK`` = 256: one chunk per CTA), each settled from this
  rank's own LL slot (``ll + (block + rank L) 16``, ``block = ((epoch & 1) rows_max + row)
  block_pairs``), its local index ``((g >> (6 + log2 W)) << 6) | (g & 63)`` (``g`` = the low
  word inverted) turned into its KV pool row (``ptx.kv_row_if``; a key past the request's pages
  is not a candidate);
- QK as the vendored kernel, then the threshold pair (thread 0 settles it; one CTA barrier);
  a key enters the softmax only if its image is at least ``thr``; PV, the cluster merge and the
  push to the owners are the vendored kernel's;
- a row whose merge took the rescue path (a short rank) publishes ``(ALL, ALL)`` after its
  winners: the CTA drops the speculative chunk and runs the vendored loop over the written
  winners (``rows`` / ``valid``; after a GPU fence).

The epoch: the merge grid does not advance it (``publish``); every decode CTA takes a ticket at
its end, the last advances ``state[0]`` (``ptx.advance_epoch_if``), after it saw its row's
``done`` pair (``pub + 32 row + 16``, written by the merge CTA after everything it writes), so the
decode grid completing implies the merge grid's writes are complete.

Inputs without ``griddepcontrol.wait``: the step metadata (row lengths, block table, request
index: complete before the graph's layers), the q mailbox and the KV rows of this step's tokens,
the candidates and the threshold (LL pairs, by epoch). The query side joins this stream by a
programmatic edge (this kernel launches with PDL), so the q mailbox is safe only because its
``QueryWait`` is compiled with ``release_after_poll``: it releases this kernel after it saw every
chunk published (and after the query kernel that appends the KV rows completed). The q words are
read with ``ld.global.cg`` (L2, not a stale L1 line), and the end-of-kernel re-arm (sentinel stores)
then comes after the last poll (a dependent that skips ``griddepcontrol.wait`` must not read or
re-arm a buffer its primary is still filling).

Precision: the vendored kernel's per key; the key split over the cluster is by candidate
position instead of by winner position, so the P anchors (the running max of each CTA's share)
and the order of the merge's sums differ: not bitwise to the vendored decode.
"""

from __future__ import annotations

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..vendor.vllm_main import module as main_module
from . import rubin_ull_dsa_index_ptx as ptx

_sparse = main_module("cutedsl.dsa.attn_sparse")
_prim = main_module("cutedsl.common.primitives")
_host = main_module("cutedsl.common.host")

ROW, LATENT, SPLIT, CHUNK = _sparse.ROW, _sparse.LATENT, _sparse.SPLIT, _sparse.CHUNK
PITCH, P_PITCH, HEADS = _sparse.PITCH, _sparse.P_PITCH, _sparse.HEADS
QK_WARPS, QK_CHAINS, THREADS, WARPS, COLS = (
    _sparse.QK_WARPS,
    _sparse.QK_CHAINS,
    _sparse.THREADS,
    _sparse.WARPS,
    _sparse.COLS,
)
LOG2_P_SCALE = _sparse.LOG2_P_SCALE
Q_SENT = _sparse.Q_SENT
#: Staged candidates per row: one chunk per cluster CTA.
CANDIDATES = SPLIT * CHUNK
_ALL = 0xFFFFFFFF

alloc, at, copy_g2s, copy_s2cluster = _prim.alloc, _prim.at, _prim.copy_g2s, _prim.copy_s2cluster
ldmatrix, load_words, mma_into, push_float = (
    _prim.ldmatrix,
    _prim.load_words,
    _prim.mma_into,
    _prim.push_float,
)
store_words, wait, zeros = _prim.store_words, _prim.wait, _prim.zeros
store_fp8_pair_shared, transpose4 = _sparse.store_fp8_pair_shared, _sparse.transpose4


class SpecDcpDecode(_sparse.SparseDcpDecode):
    """``SparseDcpDecode`` (pushing; module doc) on the row's staged candidates. ``world``:
    key-shard ranks; ``local_k``: staged keys per row (``CANDIDATES``); ``block_pairs``: LL pairs
    per (parity, row) block of the merge workspace."""

    def __init__(self, batch, heads, padded, width, push, tiles, bulk, world, local_k, block_pairs):
        assert push is not None and local_k == CANDIDATES and world & (world - 1) == 0
        super().__init__(batch, heads, padded, width, push, tiles, bulk)
        self.world, self.local_k, self.block_pairs = world, local_k, block_pairs
        self.ctas = self.grid[0] * self.grid[1]

    @cute.jit
    def candidate(self, ll, rank, block, epoch, deadline, i, table, n):
        """Staged key ``i`` of the row (this rank's own LL slot): (image, KV pool row, timed out);
        image 0 and row 0 where it is not a candidate."""
        W, L = self.world, self.local_k
        shift = 6 + (W.bit_length() - 1)
        at_ = ll + Uint64(block + rank * Uint32(L) + i) * Uint64(16)
        w0, e0, w1, e1 = ptx.ld_ll_pair_if(Uint32(1), at_)
        hi, lo, late = ptx.settle_ll_if(Uint32(1), at_, w0, e0, w1, e1, epoch, deadline)
        g = lo ^ Uint32(0xFFFFFFFF)
        local = ((g >> Uint32(shift)) << Uint32(6)) | (g & Uint32(63))
        row_table, cols, stride_rows, row_offset = table
        keep = ptx.lt_u32(local >> Uint32(6), cols) & ptx.lt_u32(local, n)
        kv_row = ptx.kv_row_if(keep, row_table, cols, local, stride_rows, row_offset)
        keep = keep & (Uint32(1) - ptx.eq_u32(kv_row, 0xFFFFFFFF))
        image = ptx.sel64(keep, ptx.pack64(lo, hi), Uint64(0))
        return image, ptx.sel(keep, kv_row, 0), late

    @cute.kernel
    def kernel(self, args):
        query, kv, rows_in, valid_in = args[:4]
        scale_log2 = args[6]
        rank_i = args[11]
        ll_addr, pub_addr, state_addr, rowlen_addr = args[12:16]
        req_addr, table_addr, table_pitch, table_cols = args[16:20]
        stride_rows, row_offset, rows_max, timeout_us = args[20:24]
        T: cutlass.Constexpr = self.tiles
        CH: cutlass.Constexpr = self.cta_heads
        L: cutlass.Constexpr = self.local_k
        smem = cutlass.utils.SmemAllocator()
        kvs = alloc(smem, cutlass.Uint8, (CHUNK, PITCH), 128)
        qs = alloc(smem, cutlass.Uint8, (CH, PITCH), 128)
        ps = alloc(smem, cutlass.Uint8, (CH, P_PITCH), 128)
        red = alloc(smem, Float32, (2, QK_WARPS, CH))
        inbox = alloc(smem, Float32, (SPLIT, CH, COLS), 16)
        stats = alloc(smem, Float32, (SPLIT, 2, CH))
        if cutlass.const_expr(self.bulk):
            staged = alloc(smem, Float32, (SPLIT - 1, CH, COLS), 128)
        # The chunk's candidate key images, then the threshold and the redo flag.
        images = alloc(smem, cutlass.Uint64, (CHUNK + 2,), 16)
        bar = smem.allocate(cutlass.Uint64, 8)
        inbound = smem.allocate(cutlass.Uint64, 8)
        tid, _, _ = cute.arch.thread_idx()
        x, b, _ = cute.arch.block_idx()
        tile, split = x // SPLIT, x % SPLIT
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        qt, kw = 0, warp
        if cutlass.const_expr(T > 1):
            qt, kw = warp // QK_WARPS, warp % QK_WARPS
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(inbound, 1)
            cute.arch.mbarrier_init_fence()
            inbound_bytes = SPLIT * (CH * COLS + 2 * CH) * 4
            if cutlass.const_expr(self.bulk):
                inbound_bytes = ((SPLIT - 1) * CH * COLS + SPLIT * 2 * CH) * 4
            cute.arch.mbarrier_arrive_and_expect_tx(inbound, inbound_bytes)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()
        # No griddepcontrol.wait (module doc): the dependents may launch now.
        cute.arch.griddepcontrol_launch_dependents()
        utid = Uint32(tid)
        first_thread = ptx.eq_u32(utid, 0)
        row_u = Uint32(b)
        rank = Uint32(rank_i)
        state = Uint64(state_addr)
        epoch = ptx.ld_global_u32(state) + Uint32(1)
        deadline = ptx.globaltimer() + Uint64(Uint32(timeout_us)) * Uint64(1000)
        block = ((epoch & Uint32(1)) * Uint32(rows_max) + row_u) * Uint32(self.block_pairs)
        ll = Uint64(ll_addr)
        pub_row = Uint64(pub_addr) + Uint64(row_u) * Uint64(32)
        n_keys = ptx.ld_global_u32(Uint64(rowlen_addr) + Uint64(row_u) * Uint64(4))
        target = ptx.min_u32(n_keys, Uint32(L))
        req = ptx.ld_global_u32(Uint64(req_addr) + Uint64(row_u) * Uint64(4))
        row_table = Uint64(table_addr) + Uint64(req) * Uint64(Uint32(table_pitch)) * Uint64(4)
        pages = ptx.min_u32((n_keys + Uint32(63)) >> Uint32(6), Uint32(table_cols))
        table = (row_table, pages, Uint32(stride_rows), Uint32(row_offset))
        chunks16 = ROW // 16
        q_loads: cutlass.Constexpr = -(-CH * chunks16 // THREADS)
        q_words = []
        if cutlass.const_expr(T > 1):
            for i in cutlass.range_constexpr(q_loads):
                h_flat = min(tid + THREADS * i, Int32(CH * chunks16 - 1))
                h_row, h_chunk = h_flat // chunks16, h_flat % chunks16
                h_src = at(query, b, tile * CH + h_row, 16 * h_chunk)
                q_words.append(load_words(h_src, count=4, mutable=True))
        kv_base = kv.iterator.toint()
        neg_inf = Float32(-math.inf)
        m_run = zeros(2 * T, Float32)
        m_run.fill(neg_inf)
        l_run = zeros(2 * T, Float32)
        acc = zeros((T, 4, 4), Float32)
        a_row = lane % 8 + 8 * ((lane // 8) % 2)
        a_col = 16 * (lane // 16)
        b_row = 8 * kw + lane % 8
        b_col = 16 * (lane // 8)
        phase = Int32(0)
        late = Uint32(0)
        # Attempt 0: the speculative chunk (this CTA's 64 candidates); attempt 1 (a rescued row
        # only): the vendored loop over the merge's written winners.
        redo = Int32(0)
        for attempt in cutlass.range(2, unroll=1):
            if (attempt == 0) | (redo != 0):
                spec = attempt == 0
                first = Int32(0)
                mine = Int32(0)
                if spec:
                    first = Int32(ptx.min_u32(target, Uint32(split * CHUNK)))
                    mine = Int32(ptx.min_u32(target - Uint32(first), Uint32(CHUNK)))
                else:
                    # The rescued winners (behind the merge CTA's fence): as the vendored kernel.
                    ptx.fence_acq_rel_gpu_if(Uint32(1))
                    n = valid_in[b]
                    n = max(min(n, Int32(self.width)), Int32(0))
                    per = (n + SPLIT - 1) // SPLIT
                    first = min(n, split * per)
                    mine = min(n - first, per)
                    m_run.fill(neg_inf)
                    l_run.fill(0.0)
                    acc.fill(0.0)
                chunks = (mine + CHUNK - 1) // CHUNK
                for c in cutlass.range(0, chunks, unroll=1):
                    base = first + c * CHUNK
                    count = min(first + mine - base, Int32(CHUNK))
                    span = (count + 31) // 32 * 32
                    cute.arch.sync_threads()  # the previous chunk's (or attempt's) readers
                    if tid == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(bar, count * ROW)
                    if tid < count:
                        kv_row = Int32(0)
                        if spec:
                            image, crow, t_late = self.candidate(
                                ll, rank, block, epoch, deadline, Uint32(base + tid), table, n_keys
                            )
                            late = late | t_late
                            images[tid] = image
                            kv_row = Int32(crow)
                        else:
                            kv_row = rows_in[b, base + tid]
                        src = kv_base + Int64(kv_row) * ROW
                        copy_g2s(kvs.iterator + tid * PITCH, src, ROW, bar)
                    elif tid < span:
                        for k in cutlass.range_constexpr(LATENT // 16):
                            dst = kvs.iterator + tid * PITCH + 16 * k
                            store_words(dst, (Uint32(0),) * 4, shared=True)
                    if c == 0:  # (again on a rescued row: the first attempt may have had none)
                        for i in cutlass.range_constexpr(q_loads):
                            flat = tid + THREADS * i
                            if flat < CH * chunks16:
                                q_row, q_chunk = flat // chunks16, flat % chunks16
                                if cutlass.const_expr(T > 1):
                                    words = q_words[i]
                                else:
                                    src = at(query, b, tile * CH + q_row, 16 * q_chunk)
                                    words = load_words(src, count=4, mutable=True)
                                dst = qs.iterator + q_row * PITCH + 16 * q_chunk
                                store_words(dst, words, shared=True)
                    wait(bar, phase, scope="cta")
                    phase = phase ^ 1
                    cute.arch.sync_threads()
                    # ---- QK -----------------------------------------------------------
                    s = zeros(4, Float32)
                    if (warp < QK_WARPS * T) & (8 * kw < span):
                        parts = zeros((QK_CHAINS, 4), Float32)
                        for kp in cutlass.range_constexpr(ROW // 64):
                            k = ldmatrix(kvs.iterator + b_row * PITCH + 64 * kp + b_col)
                            for h in cutlass.range_constexpr(2):
                                ks = 2 * kp + h
                                qk_row = 16 * qt + a_row
                                a = ldmatrix(qs.iterator + qk_row * PITCH + 32 * ks + a_col)
                                b_frag = (k[2 * h], k[2 * h + 1])
                                mma_into(parts[ks % QK_CHAINS, None], a, b_frag, fp8=True)
                        for e in cutlass.range_constexpr(4):
                            total = parts[0, e]
                            for chain in cutlass.range_constexpr(1, QK_CHAINS):
                                total += parts[chain, e]
                            s[e] = total
                    # ---- the threshold (speculative chunk) --------------------------------
                    thr = Uint64(0)
                    if spec:
                        if tid == 0:
                            at_ = pub_row
                            w0, e0, w1, e1 = ptx.ld_ll_pair_if(Uint32(1), at_)
                            hi, lo, t_late = ptx.settle_ll_if(
                                Uint32(1), at_, w0, e0, w1, e1, epoch, deadline
                            )
                            late = late | t_late
                            images[CHUNK] = ptx.pack64(lo, hi)
                        cute.arch.sync_threads()
                        thr = images[CHUNK]
                        if thr == Uint64(0xFFFFFFFFFFFFFFFF):
                            redo = Int32(1)
                    top = zeros(2, Float32)
                    top.fill(neg_inf)
                    for e in cutlass.range_constexpr(4):
                        key = 8 * kw + 2 * quarter + e % 2
                        value = neg_inf
                        take = key < count
                        if spec:
                            take = take & (redo == 0)
                            if take:
                                take = ptx.ge_u64(images[key], thr) != Uint32(0)
                        if take:
                            value = s[e] * scale_log2
                        s[e] = value
                        top[e // 2] = max(top[e // 2], value)
                    for r in cutlass.range_constexpr(2):
                        for shift in cutlass.range_constexpr(2):
                            other = cute.arch.shuffle_sync_bfly(top[r], 1 << shift)
                            top[r] = max(top[r], other)
                    if (quarter == 0) & (warp < QK_WARPS * T):
                        red[0, kw, 16 * qt + group] = top[0]
                        red[0, kw, 16 * qt + group + 8] = top[1]
                    cute.arch.sync_threads()
                    alpha = zeros(2 * T, Float32)
                    alpha.fill(1.0)
                    m_new = zeros(2 * T, Float32)
                    for t in cutlass.range_constexpr(T):
                        for r in cutlass.range_constexpr(2):
                            i = 2 * t + r
                            chunk_max = red[0, 0, 16 * t + group + 8 * r]
                            for w in cutlass.range_constexpr(1, QK_WARPS):
                                chunk_max = max(chunk_max, red[0, w, 16 * t + group + 8 * r])
                            m_new[i] = max(m_run[i], chunk_max)
                            if m_run[i] != neg_inf:
                                alpha[i] = cute.math.exp2(m_run[i] - m_new[i], fastmath=True)
                    own = m_new
                    if cutlass.const_expr(T > 1):
                        own = zeros(2, Float32)
                        for t in cutlass.range_constexpr(T):
                            if qt == t:
                                for r in cutlass.range_constexpr(2):
                                    own[r] = m_new[2 * t + r]
                    # ---- P and its FP32 sums ------------------------------------------
                    sums = zeros(2, Float32)
                    if (warp < QK_WARPS * T) & (8 * kw < span):
                        for r in cutlass.range_constexpr(2):
                            bias = own[r] - Float32(LOG2_P_SCALE)
                            p = []
                            for e in cutlass.range_constexpr(2):
                                value = s[2 * r + e]
                                prob = Float32(0)
                                if value != neg_inf:
                                    prob = cute.math.exp2(value - bias, fastmath=True)
                                sums[r] += prob
                                p.append(prob)
                            key = 8 * kw + 2 * quarter
                            dst = ps.iterator + (16 * qt + group + 8 * r) * P_PITCH + key
                            store_fp8_pair_shared(dst, p[0], p[1])
                    for r in cutlass.range_constexpr(2):
                        for shift in cutlass.range_constexpr(2):
                            sums[r] += cute.arch.shuffle_sync_bfly(sums[r], 1 << shift)
                    if (quarter == 0) & (warp < QK_WARPS * T):
                        red[1, kw, 16 * qt + group] = sums[0]
                        red[1, kw, 16 * qt + group + 8] = sums[1]
                    cute.arch.sync_threads()
                    for t in cutlass.range_constexpr(T):
                        for r in cutlass.range_constexpr(2):
                            i = 2 * t + r
                            total = red[1, 0, 16 * t + group + 8 * r]
                            for w in cutlass.range_constexpr(1, QK_WARPS):
                                total += red[1, w, 16 * t + group + 8 * r]
                            l_run[i] = l_run[i] * alpha[i] + total
                            m_run[i] = m_new[i]
                            for j in cutlass.range_constexpr(4):
                                for e in cutlass.range_constexpr(2):
                                    acc[t, j, 2 * r + e] *= alpha[i]
                    # ---- PV -----------------------------------------------------------
                    col = 32 * warp + 4 * group
                    for ks in cutlass.range_constexpr(CHUNK // 32):
                        if 32 * ks < span:
                            a = ldmatrix(ps.iterator + a_row * P_PITCH + 32 * ks + a_col)
                            halves = []
                            for half in cutlass.range_constexpr(2):
                                words = []
                                for i in cutlass.range_constexpr(4):
                                    key = 32 * ks + 16 * half + 4 * quarter + i
                                    src = kvs.iterator + key * PITCH + col
                                    words.append(load_words(src, count=1, shared=True)[0])
                                halves.append(transpose4(words))
                            for t in cutlass.range_constexpr(T):
                                if cutlass.const_expr(t > 0):
                                    pv_row = 16 * t + a_row
                                    pv_at = pv_row * P_PITCH + 32 * ks + a_col
                                    a = ldmatrix(ps.iterator + pv_at)
                                for j in cutlass.range_constexpr(4):
                                    b_frag = (halves[0][j], halves[1][j])
                                    mma_into(acc[t, j, None], a, b_frag, fp8=True)
                if spec:
                    if mine == 0:
                        # No candidate in this CTA's share: still learn whether the row was
                        # rescued (its winners may fall in this CTA's share of them).
                        if tid == 0:
                            at_ = pub_row
                            w0, e0, w1, e1 = ptx.ld_ll_pair_if(Uint32(1), at_)
                            hi, lo, t_late = ptx.settle_ll_if(
                                Uint32(1), at_, w0, e0, w1, e1, epoch, deadline
                            )
                            late = late | t_late
                            images[CHUNK] = ptx.pack64(lo, hi)
                        cute.arch.sync_threads()
                        if images[CHUNK] == Uint64(0xFFFFFFFFFFFFFFFF):
                            redo = Int32(1)
        # ---- push this CTA's partials to the column owners (the vendored kernel's) ----------
        if cutlass.const_expr(self.bulk):
            self.stage(inbox, staged, acc, split, warp, group, quarter)
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.bulk):  # noqa: SIM102 (compile time)
            if tid < SPLIT - 1:
                dst_cta = tid
                if tid >= split:
                    dst_cta = tid + 1
                inbox_ptr, block_ptr = at(inbox, split, 0, 0), at(staged, tid, 0, 0)
                size = CH * COLS * 4
                copy_s2cluster(inbox_ptr, block_ptr, inbound, dst_cta, size=size)
        owner = warp // (COLS // 32)
        for t in cutlass.range_constexpr(T):
            for r in cutlass.range_constexpr(2):
                row = 16 * t + group + 8 * r
                if cutlass.const_expr(not self.bulk):
                    for e in cutlass.range_constexpr(2):
                        values = tuple(acc[t, j, 2 * r + e] for j in range(4))
                        column = 32 * (warp % (COLS // 32)) + 8 * quarter + 4 * e
                        dst = at(inbox, split, row, column)
                        push_float(dst, values, inbound, owner)
                if (warp == 0) & (quarter == 0):
                    for peer in cutlass.range_constexpr(SPLIT):
                        push_float(at(stats, split, 0, row), m_run[2 * t + r], inbound, peer)
                        push_float(at(stats, split, 1, row), l_run[2 * t + r], inbound, peer)
        wait(inbound, 0)
        if cutlass.const_expr(self.bulk):
            cute.arch.sync_threads()
            cute.arch.cluster_arrive_relaxed()
        for t in cutlass.range_constexpr(T):
            self.merge(args, inbox, stats, tile, split, b, tid, t)
        if split == 0:
            for i in range(tid, CH * chunks16, THREADS):
                q_row, q_chunk = i // chunks16, i % chunks16
                sentinel = (Uint32(Q_SENT),) * 4
                q_dst = at(query, b, tile * CH + q_row, 16 * q_chunk)
                store_words(q_dst, sentinel)
        if cutlass.const_expr(self.bulk):
            cute.arch.cluster_wait()
        # The merge CTA of this row is done (its last pair), then this CTA's epoch ticket.
        ptx.red_add_global_if(late, state + Uint64(12), 1)
        if tid == 0:
            at_ = pub_row + Uint64(16)
            w0, e0, w1, e1 = ptx.ld_ll_pair_if(Uint32(1), at_)
            _, _, t_late = ptx.settle_ll_if(Uint32(1), at_, w0, e0, w1, e1, epoch, deadline)
            ptx.red_add_global_if(t_late, state + Uint64(12), 1)
        cute.arch.sync_threads()
        ptx.advance_epoch_if(first_thread, state, epoch, Uint32(self.ctas))


def spec_supported_rows(rows: int, heads: int, padded: int, push, world: int) -> bool:
    """Whether the speculative decode serves ``rows`` rows of ``world`` key-shard ranks: the
    pushing sparse decode serves them (the merge workspace's ``local_k`` must be
    ``CANDIDATES``: checked where the merge is asked to publish)."""
    return (
        push is not None
        and _sparse.sparse_for(rows, heads, padded)
        and world > 1
        and world & (world - 1) == 0
    )


@cache
def compile_spec(batch, heads, padded, width, cap, device, push, world, local_k, block_pairs):
    """``SpecDcpDecode`` of ``batch`` rows (the q mailbox of ``cap`` rows; ``push``: (owner heads,
    DCP)), the head tiles / bulk exchange of the vendored layout."""
    tiles, bulk = _sparse.sparse_layout(batch, heads, push)
    fake = _host.fake
    query = fake((cap, padded, ROW), None, torch.uint8)
    kv = fake((cute.sym_int(), ROW), (ROW, 1), torch.uint8)
    rows = fake((cute.sym_int(), width), (width, 1), torch.int32, 4)
    valid = fake((cute.sym_int(),), (1,), torch.int32, 4)
    latent = fake((batch, padded, LATENT))
    lse = fake((batch, padded), None, torch.float32, 4)
    kernel = SpecDcpDecode(
        batch, heads, padded, width, (*push, cap), tiles, bulk, world, local_k, block_pairs
    )
    peers = fake((push[1],), None, torch.int64, 8)
    args: tuple = (query, kv, rows, valid, latent, lse, Float32(1.0), Float32(1.0))
    args += (peers, peers, Int32(0), Int32(0))
    args += (Int64(0), Int64(0), Int64(0), Int64(0))  # ll, pub, state, row lengths
    args += (Int64(0), Int64(0), Int32(0), Int32(0))  # request index, block table
    args += (Int32(0), Int32(0), Int32(0), Int32(0))  # stride, offset, rows_max, timeout
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


def run_spec(
    kernel,
    query,
    kv,
    rows,
    valid,
    latent,
    lse,
    scale_log2,
    out_scale,
    push,
    merge,
    pub,
    row_lens,
    kv_rows,
):
    """``SpecDcpDecode`` of one step: the vendored ``run_sparse`` arguments with ``push`` =
    (DcpMailbox, parity, rank), then the key-shard workspace ``merge`` (its LL buffer, state,
    rows_max, deadline), the merge grid's published pairs ``pub``, the row lengths and the
    ``KvRows`` remap."""
    to_cute = _host.to_cute
    mail, parity, rank = push
    table, table_pitch, table_cols = kv_rows.table_args(latent.shape[0])
    args: tuple = (
        to_cute(query.view(torch.uint8)),
        to_cute(kv.view(torch.uint8)),
        to_cute(rows, 4),
        to_cute(valid, 4),
        to_cute(latent),
        to_cute(lse, 4),
        Float32(scale_log2),
        Float32(out_scale),
        to_cute(mail.peers, 8),
        to_cute(mail.lse_peers, 8),
        Int32(parity),
        Int32(rank),
        Int64(merge.ll.data_ptr()),
        Int64(pub.data_ptr()),
        Int64(merge.state.data_ptr()),
        Int64(row_lens.data_ptr()),
        Int64(kv_rows.req_idx.data_ptr()),
        Int64(table),
        Int32(table_pitch),
        Int32(table_cols),
        Int32(kv_rows.stride_rows),
        Int32(kv_rows.row_offset),
        Int32(merge.rows_max),
        Int32(min(merge.max_spin, 2**31 - 1)),
    )
    kernel(*args, _host.stream(latent.device))
    return latent, lse
