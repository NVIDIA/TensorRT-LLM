# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rubin-class DCP MLA decode core at q = 1 (``DcpCore``): one launch, no split
reduction kernel.

Replaces tokenspeed's split_kv + reduction under DCP for the batches of
``RUBIN_CORE``. Each rank's KV shard is short (~ctx / DCP tokens per request), so
the core is latency-bound. CTA (j, g, b) owns 16 (or 32) query heads of request b
and the 64-token tiles j, j + splits, ... of its local KV:
- before the PDL wait (the pages do not depend on the predecessor) it reads the
  first ``buffers`` tiles (one table entry per page, unmasked 16 B loads), converts
  FP8 -> FP16 (exact) into shared memory and prefetches the next tiles' pages to
  L2; the request's last local row, which MlaQuery may insert this step, is read
  after the wait;
- then the query rows (FP8 -> FP16, exact; A fragments kept in registers);
- per tile: QK with FP16 mma.sync, FP32 accumulation (warp (kq, th): K quarter kq,
  token group th; the four quarter partials summed in order through shared memory),
  online softmax in FP32 (log2), P rounded to FP16 (the stock kernel rounds P to
  FP8), PV with ldmatrix.trans V fragments, FP32 accumulation;
- out: per (request, head, split) the unnormalized O times the output scale (FP32)
  and (max, sum); DcpValue (``fold``) LSE-merges the splits of each row before its
  push, with the one BF16 rounding of the stock rows.
mma.sync rather than FP8 MMAs: FP8 mma.sync is emulated on sm_100/107 (FP16 HMMA
plus conversions of both operands per instruction).
"""

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake
from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    asm,
    at,
    float_bits,
    launch,
    ldmatrix,
    load_volatile,
    load_words,
    mma,
    prefetch_bytes,
    published,
    store_words,
    zeros,
)

# Rubin-class devices, DCP core at q = 1: batch -> (splits per request and head
# group, 64-row pages per tile, tiles staged before the PDL wait[, warps, M16 tiles
# per CTA]). Batches not listed keep tokenspeed.
RUBIN_CORE = batch_table("8:4.1.2")
GROUP, PAGE_ROWS, KROW = 16, 64, 1168  # heads per CTA, rows per staged page, FP16 row B
EMPTY = -3.0e38  # running max before any token: finite (no inf - inf)
Q_SENT = 0xFFFFFFFF  # q mailbox sentinel (four FP8 0xFF bytes, never published)
LOG2E = 1.4426950408889634


@asm(Uint32, pure=True)
def f16x2(low, high):
    return "cvt.rn.f16x2.f32 $0,$2,$1;", "=r,f,f", (Float32(low), Float32(high))


@asm((Uint32, Uint32), pure=True)
def fp8x4_f16(word):
    """Four FP8 (E4M3) bytes -> two FP16x2 words, in order (exact)."""
    code = "{ .reg .b16 a,b; mov.b32 {a,b},$2; "
    code += "cvt.rn.f16x2.e4m3x2 $0,a; cvt.rn.f16x2.e4m3x2 $1,b; }"
    return code, "=r,=r,r", (Uint32(word),)


@asm()
def store_f32x2_shared(pointer, a, b):
    return "st.shared.v2.f32 [$0],{$1,$2};", "r,f,f,~{memory}", (pointer, a, b)


@asm((Float32,) * 2)
def load_f32x2_shared(pointer):
    return "ld.shared.v2.f32 {$0,$1},[$2];", "=f,=f,r,~{memory}", (pointer,)


@asm((Float32,) * 4)
def load_f32x4_shared(pointer):
    code = "ld.shared.v4.f32 {$0,$1,$2,$3},[$4];"
    return code, "=f,=f,=f,=f,r,~{memory}", (pointer,)


class DcpCore:
    """``heads`` (DCP * H) query heads over this rank's KV shard, ``batch`` requests,
    ``policy`` a RUBIN_CORE entry, KV pages of ``page`` rows (32 or a multiple of 64).
    ``stamps`` (benchmarks only): per-CTA %globaltimer phase stamps."""

    def __init__(self, heads, batch, policy, page, stamps=False, early=True):
        policy = tuple(policy) + (1, 2, 8, 1)[len(policy) - 1 :]
        splits, sub, buffers, warps, mt = policy
        assert warps in (8, 16) and mt in (1, 2)
        self.warps, self.threads, self.mt = warps, 32 * warps, mt
        self.group = GROUP * mt  # heads per CTA: mt M16 tiles
        assert heads % self.group == 0 and sub in (1, 2)
        assert page % PAGE_ROWS == 0 or PAGE_ROWS % page == 0
        self.heads, self.batch, self.splits, self.page = heads, batch, splits, page
        self.grid = (splits, heads // self.group, batch)
        tile = PAGE_ROWS * sub
        self.tq = warps // 4  # QK token groups (warp (kq, th): K quarter kq, group th)
        self.ntw = tile // (8 * self.tq)  # 8-token MMA tiles per warp
        self.dw = 512 // warps  # PV: latent columns per warp
        self.tpr = self.threads // self.group  # softmax threads per row
        self.tpr_bits = self.tpr.bit_length() - 1
        self.v = tile // self.tpr  # scores per softmax thread
        self.sub, self.tile, self.buffers = sub, PAGE_ROWS * sub, buffers
        self.s_pitch, self.p_pitch = self.tile + 4, (self.tile + 8) * 2
        # CTAs per SM (registers <= 128 at two)
        self.occupancy = 2 if buffers * sub == 1 and warps == 8 else 1
        self.stamps = stamps
        self.steps = -(-PAGE_ROWS * 36 // self.threads)  # 16 B chunks per thread
        # The first refilled tile is loaded into registers right after the PDL wait.
        self.early = early and sub == 1 and mt == 1  # (32 heads: registers)

    @cute.jit
    def __call__(self, *args):
        launch(
            self.kernel,
            args,
            self.grid,
            self.threads,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.jit
    def stamp(self, address, i: cutlass.Constexpr[int], tid):
        if cutlass.const_expr(self.stamps):  # noqa: SIM102 (compile-time)
            if tid == 0:
                j, g, b = cute.arch.block_idx()
                cta = (b * self.grid[1] + g) * self.grid[0] + j
                pointer = cute.make_ptr(
                    Int64, address, cute.AddressSpace.gmem, assumed_align=8
                )
                stamps = cute.make_tensor(pointer, cute.make_layout(1 << 20))
                stamps[cta * 12 + i] = Int64(_globaltimer())

    @cute.jit
    def row_base(self, kv, table, stride, b, pos):
        """Address of local token ``pos`` of request b (64-bit page offset)."""
        page = Int64(table[b * stride + pos // self.page])
        offset = page * Int64(kv.layout.stride[0])
        offset += Int64((pos % self.page) * kv.layout.stride[1])
        return kv.iterator + offset

    @cute.jit
    def stage(self, kv, table, stride, b, first, n, skip, dst, tid):
        """The 64 FP8 rows from local token ``first`` -> FP16 rows of ``dst`` (rows past
        the length zero, row ``skip`` not written)."""
        words = self.stage_load(kv, table, stride, b, first, n, tid)
        self.stage_store(words, first, n, skip, dst, tid)

    @cute.jit
    def stage_load(self, kv, table, stride, b, first, n, tid):
        """This thread's 16 B chunks of the 64 rows from ``first`` (FP8 words). One or
        two pages (64 or 32 rows) per tile: their table entries are read once, the
        row loads are unmasked."""
        P, rs = self.page, kv.layout.stride[1]
        two = P < PAGE_ROWS  # 32-row pages
        pages = []
        for u in cutlass.range_constexpr(2 if two else 1):
            page = Int64(0)
            if first + u * P < n:
                page = Int64(table[b * stride + first // P + u])
            pages.append(page)
        NT, chunks = self.threads, PAGE_ROWS * 36
        words = cute.make_rmem_tensor(4 * self.steps, Uint32)
        for i in cutlass.range_constexpr(self.steps):
            c = tid + NT * i
            if cutlass.const_expr(NT * (i + 1) > chunks):
                c = c % chunks  # the tail: an in-page address, the store is skipped
            r = c // 36
            page, row = pages[0], (first % P) + r
            if cutlass.const_expr(two):
                upper = Int64(r >= P)
                page = pages[0] + (pages[1] - pages[0]) * upper
                row = r - P * Int32(r >= P)
            offset = page * Int64(kv.layout.stride[0]) + Int64(row * rs + (c % 36) * 16)
            chunk = load_words(kv.iterator + offset, count=4, mutable=True)
            for e in cutlass.range_constexpr(4):
                words[4 * i + e] = chunk[e]
        return words

    @cute.jit
    def stage_store(self, words, first, n, skip, dst, tid):
        """``stage_load``'s words -> FP16 rows of ``dst`` (exact; rows past the length
        zero, row ``skip`` not written)."""
        NT, chunks = self.threads, PAGE_ROWS * 36
        for i in cutlass.range_constexpr(self.steps):
            c = tid + NT * i
            r, k = c // 36, c % 36
            pos = first + r
            keep = Uint32(0) - Uint32(pos < n)
            out = [fp8x4_f16(words[4 * i + e] & keep) for e in range(4)]
            if (pos != skip) & (c < chunks):
                target = dst + r * KROW + k * 32
                store_words(target, (*out[0], *out[1]), shared=True)
                store_words(target + 16, (*out[2], *out[3]), shared=True)

    @cute.kernel
    def kernel(self, args):
        query, kv, table_address, stride, lens, part, stats_g = args[0:7]
        scale, out_scale = args[7:9]
        st = args[9] if cutlass.const_expr(self.stamps) else Int64(0)
        tptr = cute.make_ptr(
            Int32, table_address, cute.AddressSpace.gmem, assumed_align=4
        )
        table = cute.make_tensor(tptr, cute.make_layout(1 << 30))
        NS, NB, T, SP, PP = (
            self.splits,
            self.buffers,
            self.tile,
            self.s_pitch,
            self.p_pitch,
        )
        smem = cutlass.utils.SmemAllocator()
        tiles_s = alloc(smem, Uint8, NB * T * KROW, 128)
        # The FP16 query rows, then the QK partials of the four K quarters.
        G, MT = self.group, self.mt
        query_s = alloc(smem, Uint8, max(G * KROW, 4 * G * SP * 4), 128)
        probs = alloc(smem, Uint8, G * PP, 128)
        alphas = alloc(smem, Float32, G)
        tid, _, _ = cute.arch.thread_idx()
        self.stamp(st, 0, tid)
        j, g, b = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        gid, t4 = lane // 4, lane % 4
        quad, r8 = lane // 8, lane % 8
        n = lens[b]
        tiles = (n + T - 1) // T
        mine = (tiles - j + NS - 1) // NS  # tiles j, j + NS, ... (j < NS)
        late = n - 1  # MlaQuery may insert this row this step: read after the wait
        for s in cutlass.range_constexpr(NB):
            if s < mine:
                for u in cutlass.range_constexpr(self.sub):
                    first = (j + s * NS) * T + PAGE_ROWS * u
                    buffer = tiles_s.iterator + (s * T + PAGE_ROWS * u) * KROW
                    self.stage(kv, table, stride, b, first, n, late, buffer, tid)
        # L2 prefetch of the pages of the next two tiles (refilled after the wait).
        for s in cutlass.range_constexpr(NB, NB + 2):
            for u in cutlass.range_constexpr(self.sub):
                ahead = (j + s * NS) * T + PAGE_ROWS * u
                if (tid == 32 * (2 * (s - NB) + u)) & (ahead < n):
                    for v in cutlass.range_constexpr(max(1, PAGE_ROWS // self.page)):
                        rows = min(PAGE_ROWS, self.page)
                        if ahead + v * rows < n:
                            pos = ahead + v * rows
                            origin = self.row_base(kv, table, stride, b, pos)
                            prefetch_bytes(origin, rows * kv.layout.stride[1])
        self.stamp(st, 1, tid)
        cute.arch.griddepcontrol_wait()
        self.stamp(st, 2, tid)
        ahead_words = cute.make_rmem_tensor(4 * self.steps, Uint32)
        if cutlass.const_expr(self.early):  # noqa: SIM102 (compile-time)
            if mine > NB:  # the first refill tile (its pages are prefetched to L2)
                first = (j + NB * NS) * T
                loaded = self.stage_load(kv, table, stride, b, first, n, tid)
                for e in cutlass.range_constexpr(4 * self.steps):
                    ahead_words[e] = loaded[e]
        last = late // T
        if (n > 0) & (last % NS == j) & (last // NS < NB) & (tid < 36):
            slot = tiles_s.iterator + ((last // NS) * T + late % T) * KROW
            origin = self.row_base(kv, table, stride, b, late) + tid * 16
            words = load_words(origin, count=4, mutable=True)
            out = [fp8x4_f16(words[e]) for e in range(4)]
            store_words(slot + tid * 32, (*out[0], *out[1]), shared=True)
            store_words(slot + tid * 32 + 16, (*out[2], *out[3]), shared=True)
        # MlaQuery does not wait for the gathered q (DCP): poll this CTA's G query rows
        # (16 B chunks, no word may be the sentinel), gated on chunk 0 of each row.
        if tid < G:
            first = at(query, b, g * G + tid, 0)
            words = load_volatile(first, count=4)
            while not published(words, sentinel=Q_SENT):
                words = load_volatile(first, count=4)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(-(-G * 36 // self.threads)):
            c = tid + self.threads * i  # G query rows x 36 16 B chunks
            if c < G * 36:
                r, k = c // 36, c % 36
                source = at(query, b, g * G + r, k * 16)
                words = load_volatile(source, count=4)
                while not published(words, sentinel=Q_SENT):
                    words = load_volatile(source, count=4)
                out = [fp8x4_f16(words[e]) for e in range(4)]
                target = query_s.iterator + r * KROW + k * 32
                store_words(target, (*out[0], *out[1]), shared=True)
                store_words(target + 16, (*out[2], *out[3]), shared=True)
        cute.arch.sync_threads()
        self.stamp(st, 3, tid)
        # Warp (kq, th): query K quarter kq (9 K16 steps), token group th of a tile.
        kq, th = warp % 4, warp // 4
        TQ, DW, TPR = self.tq, self.dw, self.tpr
        qa = cute.make_rmem_tensor(36 * MT, Uint32)
        for i in cutlass.range_constexpr(9):
            col = 16 * (9 * kq + i) + (quad // 2) * 8
            for mt in cutlass.range_constexpr(MT):
                qrow = 16 * mt + (quad % 2) * 8 + r8
                frag = ldmatrix(query_s.iterator + qrow * KROW + col * 2)
                for e in cutlass.range_constexpr(4):
                    qa[4 * (i * MT + mt) + e] = frag[e]
        cute.arch.sync_threads()  # the query rows become the QK partials
        partials = cute.recast_ptr(query_s.iterator, dtype=Float32)
        o = zeros(MT * DW // 2, Float32)  # [m tile][DW / 8 column tiles][c0..c3]
        mx = cute.make_rmem_tensor(1, Float32)
        mx[0] = Float32(EMPTY)
        sums = zeros(1, Float32)
        ninf = Float32(-math.inf)
        V = self.v
        row, cv = tid // TPR, (tid % TPR) * V
        for s in range(mine):
            if s >= NB:  # refill (after the PV of tile s - NB)
                for u in cutlass.range_constexpr(self.sub):
                    ahead = (j + s * NS) * T + PAGE_ROWS * u
                    buffer = tiles_s.iterator + ((s % NB) * T + PAGE_ROWS * u) * KROW
                    if cutlass.const_expr(self.early):
                        if s == NB:
                            self.stage_store(ahead_words, ahead, n, -1, buffer, tid)
                        else:
                            self.stage(kv, table, stride, b, ahead, n, -1, buffer, tid)
                    else:
                        self.stage(kv, table, stride, b, ahead, n, -1, buffer, tid)
                cute.arch.sync_threads()
            tile = tiles_s.iterator + (s % NB) * T * KROW
            # [m tile][NTW token tiles][c0..c3]
            acc = zeros(4 * self.ntw * self.mt, Float32)
            tile_first = (j + s * NS) * T
            if tile_first + (T // TQ) * th < n:  # skip a token group past the length
                for i in cutlass.range_constexpr(9):
                    col = 16 * (9 * kq + i) + (quad % 2) * 8
                    bws = []
                    for pair in cutlass.range_constexpr(self.ntw // 2):
                        token = (T // TQ) * th + 16 * pair + (quad // 2) * 8 + r8
                        bws.append(ldmatrix(tile + token * KROW + col * 2))
                    for mt in cutlass.range_constexpr(self.mt):
                        qf = tuple(qa[4 * (i * self.mt + mt) + e] for e in range(4))
                        for pair in cutlass.range_constexpr(self.ntw // 2):
                            base = 4 * (mt * self.ntw + 2 * pair)
                            _mma_at(acc, base, qf, (bws[pair][0], bws[pair][1]))
                            _mma_at(acc, base + 4, qf, (bws[pair][2], bws[pair][3]))
            for mt in cutlass.range_constexpr(self.mt):
                for nt in cutlass.range_constexpr(self.ntw):
                    column = (T // TQ) * th + 8 * nt + 2 * t4
                    for h in cutlass.range_constexpr(2):
                        qrow = kq * G + 16 * mt + gid + 8 * h
                        k0 = 4 * (mt * self.ntw + nt) + 2 * h
                        target = partials + qrow * SP + column
                        store_f32x2_shared(target, acc[k0], acc[k0 + 1])
            cute.arch.sync_threads()
            if s == 0:
                self.stamp(st, 8, tid)
            # Softmax: TPR threads per row, V tokens each.
            sc = cute.make_rmem_tensor(self.v, Float32)
            for e in cutlass.range_constexpr(self.v):
                sc[e] = Float32(0)
            for q in cutlass.range_constexpr(4):  # the K quarters, in order
                for v in cutlass.range_constexpr(max(1, self.v // 4)):
                    origin = partials + (q * G + row) * SP + cv + 4 * v
                    qkp = (
                        load_f32x4_shared(origin)
                        if cutlass.const_expr(self.v >= 4)
                        else load_f32x2_shared(origin)
                    )
                    for e in cutlass.range_constexpr(min(4, self.v)):
                        sc[4 * v + e] = sc[4 * v + e] + qkp[e]
            token0 = (j + s * NS) * T + cv
            for e in cutlass.range_constexpr(self.v):
                sc[e] = sc[e] * scale
                if token0 + e >= n:
                    sc[e] = ninf
            top = sc[0]
            for e in cutlass.range_constexpr(1, self.v):
                top = max(top, sc[e])
            for shift in cutlass.range_constexpr(self.tpr_bits):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            top = max(top, mx[0])
            alpha = cute.math.exp2(mx[0] - top, fastmath=True)
            mx[0] = top
            p = [cute.math.exp2(sc[e] - top, fastmath=True) for e in range(self.v)]
            tile_sum = Float32(0)
            for e in cutlass.range_constexpr(self.v):
                tile_sum = tile_sum + p[e]
            sums[0] = sums[0] * alpha + tile_sum
            pw = tuple(f16x2(p[2 * e], p[2 * e + 1]) for e in range(self.v // 2))
            store_words(probs.iterator + row * PP + cv * 2, pw, shared=True)
            if tid % TPR == 0:
                alphas[row] = alpha
            cute.arch.sync_threads()
            if s == 0:
                self.stamp(st, 9, tid)
            # PV: warp w owns latent columns 64 w..; V^T fragments by ldmatrix.trans.
            for mt in cutlass.range_constexpr(self.mt):
                a0, a1 = alphas[16 * mt + gid], alphas[16 * mt + gid + 8]
                for i in cutlass.range_constexpr(self.dw // 2):
                    k0 = mt * (self.dw // 2) + i
                    o[k0] = o[k0] * (a0 if i % 4 < 2 else a1)
            for kk in cutlass.range_constexpr(T // 16):
                if tile_first + 16 * kk < n:  # P is zero past the length
                    pas = []
                    for mt in cutlass.range_constexpr(self.mt):
                        prow = 16 * mt + (quad % 2) * 8 + r8
                        pp = probs.iterator + prow * PP
                        pas.append(ldmatrix(pp + (kk * 16 + (quad // 2) * 8) * 2))
                    token = 16 * kk + (quad % 2) * 8 + r8
                    for q in cutlass.range_constexpr(self.dw // 16):
                        col = DW * warp + 16 * q + (quad // 2) * 8
                        bt = ldmatrix(tile + token * KROW + col * 2, trans=True)
                        for mt in cutlass.range_constexpr(self.mt):
                            base = mt * (self.dw // 2) + 8 * q
                            _mma_at(o, base, pas[mt], (bt[0], bt[1]))
                            _mma_at(o, base + 4, pas[mt], (bt[2], bt[3]))
            cute.arch.sync_threads()
            if s == 0:
                self.stamp(st, 10, tid)
        self.stamp(st, 4, tid)
        cute.arch.griddepcontrol_launch_dependents()
        # This split's (max, sum) per row and its unnormalized partial O times the
        # output scale (FP32) for DcpValue's merge.
        total = sums[0]
        for shift in cutlass.range_constexpr(self.tpr_bits):
            total = total + cute.arch.shuffle_sync_bfly(total, 1 << shift)
        head0 = g * G
        if tid % TPR == 0:
            stats_g[b, head0 + row, 0, j] = mx[0]
            stats_g[b, head0 + row, 1, j] = total
        for mt in cutlass.range_constexpr(MT):  # idle splits store zeros
            for nt in cutlass.range_constexpr(DW // 8):
                d = DW * warp + 8 * nt + 2 * t4
                for h in cutlass.range_constexpr(2):
                    k0 = mt * (DW // 2) + 4 * nt + 2 * h
                    values = (o[k0] * out_scale, o[k0 + 1] * out_scale)
                    target = at(part, b, head0 + 16 * mt + gid + 8 * h, j, d)
                    store_words(target, tuple(float_bits(v) for v in values))
        self.stamp(st, 5, tid)


@asm(cutlass.Uint64)
def _globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@cute.jit
def _mma_at(acc, base: cutlass.Constexpr[int], a, b):
    """``acc[base : base + 4] += a @ b`` (FP16 m16n8k16, FP32 accumulation)."""
    result = mma(a, b, tuple(acc[base + e] for e in range(4)))
    for e in cutlass.range_constexpr(4):
        acc[base + e] = result[e]


@cache
def compile_dcp_core(heads, batch, policy, cap, cache_spec, device, stamps=False):
    """DcpCore of one batch (``policy``: a RUBIN_CORE entry); ``cache_spec``: the
    (shape, stride, dtype) of the uint8 paged cache view, ``cap``: the q mailbox's
    request capacity."""
    splits = policy[0]
    shape, stride, _ = cache_spec
    query = fake((cap, heads, 576), None, torch.uint8)
    kv = fake(shape, stride, torch.uint8)
    lens = fake((batch,), None, torch.int32, 4)
    part = fake((batch, heads, splits, 512), None, torch.float32)
    stats = fake((batch, heads, 2, splits), None, torch.float32, 4)
    args: tuple = (query, kv, Int64(0), Int32(0), lens, part, stats)
    args += (Float32(0), Float32(0))
    args += (Int64(0),) * stamps + (make_fake_stream(),)
    core = DcpCore(heads, batch, policy, shape[1], stamps=stamps)
    with torch.accelerator.device_index(device):
        return cute.compile(core, *args)


_SPLITS: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}


def split_workspace(device, heads, splits, batch=8):
    """(partial O [batch, heads, splits, 512], stats [batch, heads, 2, splits]) FP32,
    one per (device, heads, splits), shared by every MLA layer (stream ordered)."""
    key = device, heads, splits
    if key not in _SPLITS:
        f32 = dict(dtype=torch.float32, device=device)
        part = torch.zeros(batch, heads, splits, 512, **f32)
        stats = torch.zeros(batch, heads, 2, splits, **f32)
        _SPLITS[key] = part, stats
    return _SPLITS[key]


def core_args(table, softmax_scale, output_scale):
    """The block table (address, row stride) and scales of a DcpCore call."""
    table_args = (Int64(table.data_ptr()), Int32(table.stride(0)))
    return table_args, (Float32(softmax_scale * LOG2E), Float32(output_scale))
