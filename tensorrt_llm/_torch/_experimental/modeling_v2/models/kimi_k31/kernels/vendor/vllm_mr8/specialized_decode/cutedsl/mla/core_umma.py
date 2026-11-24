# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rubin DCP MLA decode core on tcgen05 (``DcpCoreUmma``) at q = 1.

CTA (j, c, b) takes split j of request b's local KV (an even share of its tokens, a
multiple of 16; up to ``TN`` per pass) and latent quarter c (128 of the 512 value
columns); all query heads (<= 128, zero padded) are the M = 128 rows of the MMAs:
- before the PDL wait: the split's FP8 rows (32 B loads, all in flight) into a K-major
  128B-swizzled image, and their value quarter converted FP8 -> FP16 (exact) into an
  MN-major one; rows past the length are zero. The request's last local row, which
  MlaQuery may insert this step, is read after the wait;
- after the wait: the FP8 query rows, stored by 128 B column blocks (one mbarrier
  each); QK = Q K^T with kind::f8f6f4 MMAs (exact products, FP32 accumulation in the
  tensor core) into TMEM, block by block;
- softmax: four warps per TMEM lane quarter, each with a quarter of its rows' scores;
  FP32, log2, ex2.approx (as DcpCore); P rounded to FP16 into shared memory (the stock
  kernel rounds P to FP8);
- PV with kind::f16 MMAs (FP16 P and V, FP32 accumulation) into TMEM; longer splits
  run several passes and rescale O in TMEM between them;
- out as DcpCore: per (request, head, split) the unnormalized O times the output scale
  (FP32, 32 B stores) and (max, sum), which DcpValue (``fold``) LSE-merges.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake
from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    asm,
    at,
    commit_asm,
    fence_tmem,
    fma,
    free_tmem,
    launch,
    load_volatile,
    load_words,
    named_barrier,
    published,
    shared_address,
    store_words,
    wait,
)
from .core import EMPTY, _globaltimer, f16x2, fp8x4_f16

# Rubin-class devices, DCP core at q = 1: batch -> splits per request.
RUBIN_UMMA = batch_table("8:5")
# Rubin-class devices: batches whose split workspace (O partials, (max, sum)) DcpValue
# polls word by word instead of waiting for this grid (``polled_split``). Two per-parity
# workspaces armed with SPLIT_SENT; each MLA layer's core arms the other parity's after
# its PDL wait (``arm_split``) and writes its own once.
POLLED_SPLIT = batch_table("8:1")
# Rubin-class devices, q = 1: batches whose core polls the q rows and runs every split
# not holding a request's last local row before its PDL wait (``prewait``).
PREWAIT = batch_table("8:1")
SPLIT_SENT = 0xFFFFFFFF  # a NaN pattern: FP32 arithmetic only writes the canonical one
# Tokens per pass (QK's N, PV's K): a split runs one pass up to TN tokens, so 5 splits
# cover 800 local tokens (ISL ~12.8k at DCP16) in one pass; each further pass costs
# ~2.5-3.7 us (serial tile load, QK, softmax, PV). S takes columns 0..TN-1 < TMEM_Q.
TN = 160
ROWS = 128  # MMA M: the query heads, zero padded
QBLOCK = ROWS * 128  # a 128 B column block of the query rows (also of P)
KBLOCK = TN * 128  # a 128 B column block of the K rows (also a V atom column)
OFF_K = 5 * QBLOCK  # five column blocks of the 576 B rows (the last half used)
OFF_V = OFF_K + 5 * KBLOCK
SMEM = OFF_V + 2 * KBLOCK
# S at columns 0.., the FP8 query rows (A of QK, 4 per column) at 160.., O at 384..
TMEM_Q, TMEM_O = 160, 384
NT = 512  # threads: PARTS warps per TMEM lane quarter
# Warp 3 (TMEM lanes 96..127: no query rows at <= 96 heads) allocates the TMEM and frees
# it once every warp is past its last tcgen05 operation (named barrier TMEM_DONE).
TMEM_WARP, TMEM_DONE = 3, 1
PARTS = NT // ROWS
COLS, OCOLS = TN // PARTS, 128 // PARTS  # score / output columns per thread
STEPS = -(-TN * 18 // NT)  # 32 B KV chunks per thread
Q_SENT = 0xFFFFFFFF  # q mailbox sentinel (four FP8 0xFF bytes, never published)
# FP32 D, E4M3 A (TMEM) and B (K-major), M = 128, K = 64 per MMA (bit 29: Rubin dense
# FP8); N per pass at bit 17.
QK_INSTR = 1 << 4 | 8 << 24 | 1 << 29
QBYTES = 576 // 4  # query bytes per thread: its row's column part
# FP32 D, FP16 A (K-major) and B (MN-major), N = 64 (one V atom), M = 128.
PV_INSTR = 1 << 4 | 1 << 16 | 8 << 17 | 8 << 24


@asm()
def mma_f8(d, a, b, instruction, accumulate, *, count):
    """``count`` K32 E4M3 MMAs over one 128B-swizzled column block (+32 B each)."""
    op = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    code = "{ .reg .pred p0,pt; .reg .b64 xa<4>,xb<4>; setp.eq.b32 pt,0,0; "
    code += f"setp.ne.b32 p0,$4,0; {op} [$0],$1,$2,$3,p0; "
    for i in range(1, count):
        code += f"add.s64 xa{i},$1,{2 * i}; add.s64 xb{i},$2,{2 * i}; "
        code += f"{op} [$0],xa{i},xb{i},$3,pt; "
    operands = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    return code + "}", "r,l,l,r,r", (*operands, Uint32(accumulate))


@asm()
def mma_f8_tmem(d, a, b, instruction, accumulate):
    """One E4M3 MMA with A (K-major, 4 per column) from TMEM."""
    code = "{ .reg .pred p; setp.ne.b32 p,$4,0; "
    code += "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0],[$1],$2,$3,p; }"
    operands = (Uint32(d), Uint32(a), Uint64(b), Uint32(instruction))
    return code, "r,r,l,r,r", (*operands, Uint32(accumulate))


@asm()
def store_tmem_words(address, words):
    """36 words of this lane into columns address.. (x32 + x4), waiting."""
    names = [f"${i + 1}" for i in range(36)]
    code = "{ .reg .b32 ta; add.u32 ta,$0,32; "
    code += f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$0],{{{','.join(names[:32])}}}; "
    code += f"tcgen05.st.sync.aligned.32x32b.x4.b32 [ta],{{{','.join(names[32:])}}}; "
    code += "tcgen05.wait::st.sync.aligned; }"
    return code, "r" + ",r" * 36 + ",~{memory}", (Uint32(address), *words)


@asm()
def mma_f16(d, a, b, instruction, accumulate):
    code = "{ .reg .pred p; setp.ne.b32 p,$4,0; "
    code += "tcgen05.mma.cta_group::1.kind::f16 [$0],$1,$2,$3,p; }"
    operands = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    return code, "r,l,l,r,r", (*operands, Uint32(accumulate))


@asm()
def load_tmem_cols(address, count):
    """``count`` (a multiple of 8) FP32 columns of this lane, one wait."""
    code, col = "{ .reg .b32 ta; ", 0
    while col < count:
        width = next(w for w in (128, 64, 32, 16, 8) if count - col >= w)
        names = ",".join(f"${col + i}" for i in range(width))
        code += f"add.u32 ta,${count},{col}; "
        code += f"tcgen05.ld.sync.aligned.32x32b.x{width}.b32 {{{names}}},[ta]; "
        col += width
    code += "tcgen05.wait::ld.sync.aligned; }"
    constraints = ",".join(["=f"] * count + ["r", "~{memory}"])
    return code, constraints, (Uint32(address),), (Float32,) * count


@asm(Float32, pure=True)
def fmax(a, b):
    return "max.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm()
def store_f32x8(pointer, values):
    code = "st.global.v8.f32 [$0],{$1,$2,$3,$4,$5,$6,$7,$8};"
    return code, "l" + ",f" * 8, (pointer, *values)


@asm()
def store_tmem_cols(address, values):
    count = len(values)
    names = ",".join(f"${i + 1}" for i in range(count))
    code = f"tcgen05.st.sync.aligned.32x32b.x{count}.b32 [$0],{{{names}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r" + ",f" * count + ",~{memory}", (Uint32(address), *values)


@cute.jit
def desc128(pointer, lbo: cutlass.Constexpr[int]):
    """128B-swizzled shared memory descriptor (SBO 1024 B: 8-row atoms); ``lbo`` is
    the MN-major atom stride (K-major: unused)."""
    address = shared_address(pointer)
    value = Uint64((address & 262143) >> 4) | Uint64(1) << 46
    value |= Uint64(lbo >> 4) << 16 | Uint64(64) << 32 | Uint64(2) << 61
    return value


def swizzled(row, chunk):
    """Byte offset of 16 B ``chunk`` of 128 B ``row`` in a 128B-swizzled image."""
    return row * 128 + ((chunk ^ (row % 8)) << 4)


@asm(Uint64)
def _clock():
    return "mov.u64 $0, %clock64;", "=l,~{memory}", ()


class DcpCoreUmma:
    """``heads`` (DCP * H <= 128) query heads over this rank's KV shard, ``batch``
    requests, ``splits`` per request, KV pages of ``page`` (>= 32) rows. ``stamps``
    (benchmarks only): per-CTA %globaltimer phase stamps."""

    def __init__(
        self,
        heads,
        batch,
        splits,
        page,
        stamps=False,
        rearm_x=False,
        polled_split=False,
        prewait=False,
    ):
        assert heads <= ROWS and page >= 32
        self.heads, self.batch, self.splits, self.page = heads, batch, splits, page
        self.grid = (splits, 4, batch)
        self.stamps = stamps
        # ``rearm_x`` (MlaQuery polls x, ``mla.query.POLLED_X``): after the PDL wait
        # the grid re-arms the in-projection's x columns (args[9], [batch, 1536] BF16)
        # with the 0xFFFF sentinel for the next MLA layer.
        self.rearm_x = rearm_x
        # ``polled_split`` (POLLED_SPLIT): DcpValue polls this layer's split workspace;
        # the next two arguments are the other parity's (part, stats), armed here.
        self.polled_split = polled_split
        # ``prewait`` (PREWAIT): the q rows are polled before the PDL wait. A split
        # that does not hold its request's last local row (n - 1, which this step's
        # MlaQuery may insert) runs QK, softmax, PV and its stores before the wait;
        # the one that holds it has warps 0-1 wait and load that row meanwhile.
        self.prewait = prewait
        # (column block, step) of each of a thread's 16 B query chunks
        self.qmap = [
            (kb, i)
            for kb in range(5)
            for i in range(-(-heads * (8 if kb < 4 else 4) // NT))
        ]

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, NT, min_blocks_per_mp=1)

    @cute.jit
    def stamp(self, address, i: cutlass.Constexpr[int], tid):
        if cutlass.const_expr(self.stamps):  # noqa: SIM102 (compile-time)
            if tid == 0:
                j, c, b = cute.arch.block_idx()
                cta = (b * self.grid[1] + c) * self.grid[0] + j
                pointer = cute.make_ptr(
                    Int64, address, cute.AddressSpace.gmem, assumed_align=8
                )
                stamps = cute.make_tensor(pointer, cute.make_layout(1 << 20))
                stamps[cta * 16 + i] = Int64(_clock())
                if cutlass.const_expr(i == 0):
                    stamps[cta * 16 + 15] = Int64(_globaltimer())

    @cute.jit
    def row_address(self, kv, page, pos):
        offset = Int64(page) * Int64(kv.layout.stride[0])
        offset += Int64((pos % self.page) * kv.layout.stride[1])
        return kv.iterator + offset

    @cute.jit
    def convert_value(self, words, row, chunk, values):
        """16 FP8 values (chunk ``chunk`` of the quarter) -> two FP16 chunks of the
        MN-major value image (exact)."""
        out = [fp8x4_f16(words[e]) for e in range(4)]
        atom = values + (chunk // 4) * KBLOCK
        low = 2 * (chunk % 4)
        store_words(atom + swizzled(row, low), (*out[0], *out[1]), shared=True)
        store_words(atom + swizzled(row, low + 1), (*out[2], *out[3]), shared=True)

    @cute.jit
    def fill_pages(self, table, stride, pages, b, p0, n, tid):
        """The (<= 8) table entries of the pages of rows p0.. into shared memory
        (threads 32..39: warp 0 sets up the barriers and TMEM meanwhile)."""
        if (tid >= 32) & (tid < 40):
            index = p0 // self.page + tid - 32
            entry = Int32(0)
            if index * self.page < n:
                entry = table[b * stride + index]
            pages[tid - 32] = entry

    @cute.jit
    def issue_tile(self, words, kv, pages, p0, rows, tid, skip=-1):
        """Loads of rows p0 .. (18 32 B chunks per row, all in flight; rows >= ``rows``
        and row ``skip`` read page 0, the null block, and are masked by
        ``store_tile``)."""
        for i in cutlass.range_constexpr(STEPS):
            q = tid + NT * i
            r = q // 18
            page = Int32(0)
            if (r < rows) & (p0 + r != skip):
                page = pages[(p0 + r) // self.page - p0 // self.page]
            source = self.row_address(kv, page, p0 + r) + (q % 18) * 32
            loaded = load_words(source, count=8, mutable=True, cache_policy=None)
            for e in cutlass.range_constexpr(8):
                words[8 * i + e] = loaded[e]

    @cute.jit
    def store_tile(self, words, p0, rows, n16, skip, ktile, values, c, tid):
        """``issue_tile``'s rows -> the K-major FP8 image, and their value quarter
        (column block c) -> FP16 (rows >= ``rows`` zero, row ``skip`` not written)."""
        for i in cutlass.range_constexpr(STEPS):
            q = tid + NT * i
            r, k = q // 18, q % 18
            keep = Uint32(0) - Uint32(r < rows)
            if (r < n16) & (p0 + r != skip):
                for h in cutlass.range_constexpr(2):
                    chunk = tuple(words[8 * i + 4 * h + e] & keep for e in range(4))
                    target = ktile + (k // 4) * KBLOCK + swizzled(r, (2 * k + h) % 8)
                    store_words(target, chunk, shared=True)
                    if k // 4 == c:
                        self.convert_value(chunk, r, (2 * k + h) % 8, values)

    @cute.jit
    def query_chunk(self, query, b, w: cutlass.Constexpr[int], tid):
        """Address of this thread's ``w``-th 16 B query chunk (by column block)."""
        kb, i = self.qmap[w]
        chunks = 8 if kb < 4 else 4
        q = (tid + NT * i) % (self.heads * chunks)
        return at(query, b, q // chunks, 128 * kb + 16 * (q % chunks))

    @cute.jit
    def load_query(self, words, query, b, tid):
        """This thread's query chunks (volatile, all in flight); true when no word is
        the sentinel."""
        ready = True
        for w in cutlass.range_constexpr(len(self.qmap)):
            chunk = load_volatile(self.query_chunk(query, b, w, tid), count=4)
            for e in cutlass.range_constexpr(4):
                words[4 * w + e] = chunk[e]
            ready = published(chunk, ready, sentinel=Q_SENT)
        return ready

    @cute.jit
    def issue_query(self, words, query, b, tid):
        """``load_query``'s loads only (checked by ``query_ready``)."""
        for w in cutlass.range_constexpr(len(self.qmap)):
            chunk = load_volatile(self.query_chunk(query, b, w, tid), count=4)
            for e in cutlass.range_constexpr(4):
                words[4 * w + e] = chunk[e]

    @cute.jit
    def query_ready(self, words):
        ready = True
        for w in cutlass.range_constexpr(len(self.qmap)):
            chunk = tuple(words[4 * w + e] for e in range(4))
            ready = published(chunk, ready, sentinel=Q_SENT)
        return ready

    @cute.jit
    def arm_split(self, part, stats, cta, tid):
        """This CTA's share of the other parity's split workspace <- SPLIT_SENT (32 B
        stores). Invariant (DcpValue ``polled_split`` polls its parity's workspace):
        the other parity's last reader is the previous MLA layer's DcpValue. It is
        complete after this kernel's PDL wait (every kernel between them waits before
        it completes); with ``prewait`` this may run before that wait, but this core
        starts only after that DcpValue fired its trigger, which follows its pushes
        (its only reads of the workspace). Its next writer is the next MLA layer's
        core, which starts only after this layer's DcpValue has fired its trigger,
        after its own PDL wait, so after this grid has completed; its next reader,
        that layer's DcpValue, polls only words that core wrote. The core never writes
        the sentinel (FP32 results: no 0xFFFFFFFF NaN). Allocation arms both."""
        ctas: cutlass.Constexpr = self.grid[0] * self.grid[1] * self.grid[2]
        sentinel = (Uint32(SPLIT_SENT),) * 8
        chunks: cutlass.Constexpr = self.batch * self.heads * self.splits * 64
        per: cutlass.Constexpr = -(-chunks // ctas)
        for i in cutlass.range_constexpr(-(-per // NT)):
            q = tid + NT * i
            if q < per:
                g = cta * per + q
                if g < chunks:
                    store_words(part.iterator + g * 8, sentinel)
        schunks: cutlass.Constexpr = self.batch * self.heads * self.splits // 4
        sper: cutlass.Constexpr = -(-schunks // ctas)
        if tid < sper:
            g = cta * sper + tid
            if g < schunks:
                store_words(stats.iterator + g * 8, sentinel)

    @cute.jit
    def await_insert(self):
        """``prewait``: the wait before the last local row is read (MlaQuery's KV
        insert, and anything else of this step's MlaQuery or in-projection).

        Before it a ``prewait`` core reads only (a) the q rows, through the sentinel
        poll: DcpValue re-arms this parity's mailbox after its own PDL wait (so after
        this layer's core grid has completed) and fires launch_dependents only after
        that, and this core can start only once the previous MLA layer's DcpValue has
        fired it (every kernel between them starts at its predecessor's trigger), so
        the re-arm two layers back is complete and visible here; each word is then
        either the sentinel or this layer's q; (b) KV rows other than each request's
        last local one: written by MlaQuery in earlier steps (earlier graph launches);
        (c) the block table and lengths (step inputs, written before the launch).
        Its stores before the wait (split workspace, POLLED_SPLIT arming) are ordered
        after the previous MLA layer's DcpValue push, their only reader, the same way
        (DcpValue pushes before its trigger)."""
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def store_stats(self, stats_g, b, row, j, top, total):
        stats_g[b, row, 0, j] = top
        stats_g[b, row, 1, j] = total

    @cute.jit
    def issue_pv(self, tmem, base, values, width, s, bo):
        """PV MMAs into O columns 0..63 (V atom 0), their commit, then into 64..127
        (atom 1) and theirs: the epilogue of the first half overlaps the second."""
        steps = width // 16
        for it in range(2 * steps):  # O columns 0..63 first (committed), then 64..127
            h = Int32(it >= steps)
            kk = it - h * steps
            pdesc = desc128(base + OFF_K + (kk // 4) * QBLOCK, 16)
            pdesc = pdesc + Uint64(2 * (kk % 4))
            vdesc = desc128(values, KBLOCK) + Uint64(128 * kk + h * (KBLOCK >> 4))
            column = tmem + TMEM_O + Uint32(64 * h)
            mma_f16(column, pdesc, vdesc, PV_INSTR, (kk > 0) | (s > 0))
            if it == steps - 1:
                commit_asm(bo)
        commit_asm(bo + 1)

    @cute.jit
    def arm_x(self, x, cta, tid):
        """This CTA's share of x's 16 B chunks <- the sentinel. Invariant (MlaQuery's
        x poll, ``mla.query.MlaQuery.lean_kernel``): x is armed here, after this
        kernel's PDL wait, so MlaQuery (this layer's only x reader) is complete; the
        next writer is the next MLA layer's in-projection, which stores x only after
        its own PDL wait, and grid completion is transitive along the stream, so it
        runs after this grid completes. Allocation arms the buffer once."""
        chunks: cutlass.Constexpr = self.batch * 1536 // 8
        ctas: cutlass.Constexpr = self.grid[0] * self.grid[1] * self.grid[2]
        per: cutlass.Constexpr = -(-chunks // ctas)
        if tid < per:
            i = cta * per + tid
            if i < chunks:
                sentinel = Uint32(0xFFFFFFFF)
                store_words(at(x, i // 192, (i % 192) * 8), (sentinel,) * 4)

    @cute.kernel
    def kernel(self, args):
        query, kv, table_address, stride, lens, part_g, stats_g = args[0:7]
        scale, out_scale = args[7:9]
        st = args[-1] if cutlass.const_expr(self.stamps) else Int64(0)
        tptr = cute.make_ptr(
            Int32, table_address, cute.AddressSpace.gmem, assumed_align=4
        )
        table = cute.make_tensor(tptr, cute.make_layout(1 << 30))
        smem = cutlass.utils.SmemAllocator()
        buffer = alloc(smem, Uint8, SMEM, 1024)
        bars = alloc(smem, Uint64, 8, 8)  # query blocks, S, O column halves
        pages = alloc(smem, Int32, 8, 16)
        partial = alloc(smem, Float32, 2 * PARTS * ROWS, 16)  # row (max, sum) parts
        tslot = alloc(smem, Uint32, 1, 4)
        tid, _, _ = cute.arch.thread_idx()
        self.stamp(st, 0, tid)
        warp = tid // 32
        quarter, part = warp % 4, warp // 4  # TMEM lanes 32 quarter.., column part
        row = tid % ROWS
        j, c, b = cute.arch.block_idx()
        H, NS = self.heads, self.splits
        base = buffer.iterator
        ktile, values = base + OFF_K, base + OFF_V
        bq, bs, bo = bars.iterator, bars.iterator + 5, bars.iterator + 6
        n = lens[b]
        per = (n + 16 * NS - 1) // (16 * NS) * 16
        p0 = j * per
        count = min(n - p0, per)  # <= 0: an idle split
        passes = (max(count, 0) + TN - 1) // TN
        late = n - 1  # MlaQuery may insert this row this step: read after the wait
        rows = min(count, TN)
        n16 = (rows + 15) // 16 * 16
        kvw = cute.make_rmem_tensor(8 * STEPS, Uint32)
        if passes > 0:
            self.fill_pages(table, stride, pages, b, p0, n, tid)
        if tid == 0:
            for k in cutlass.range_constexpr(PARTS):  # query column parts in TMEM
                cute.arch.mbarrier_init(bq + k, NT // PARTS)
            cute.arch.mbarrier_init(bs, 1)
            for h in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(bo + h, 1)
            cute.arch.mbarrier_init_fence()
        if warp == TMEM_WARP:
            cute.arch.alloc_tmem(512, tslot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        pad = (ROWS - H) * 40  # zero query rows H.. (16 B chunks of 5 blocks)
        for i in cutlass.range_constexpr(-(-pad // NT)):
            q = tid + NT * i
            if q < pad:
                qrow, rest = H + q // 40, q % 40
                target = base + (rest // 8) * QBLOCK + qrow * 128 + (rest % 8) * 16
                store_words(target, (Uint32(0),) * 4, shared=True)
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tslot[0]
        self.stamp(st, 1, tid)
        if passes > 0:
            self.issue_tile(kvw, kv, pages, p0, rows, tid, late)
            self.store_tile(kvw, p0, rows, n16, late, ktile, values, c, tid)
        cute.arch.fence_proxy("async.shared", space="cta")
        self.stamp(st, 5, tid)
        cta = (b * 4 + c) * NS + j
        ia: cutlass.Constexpr = 10 if self.rearm_x else 9  # the other parity's split
        holds = (late >= p0) & (late < p0 + count)  # this split holds row n - 1
        if cutlass.const_expr(not self.prewait):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.rearm_x):
                self.arm_x(args[9], cta, tid)
        self.stamp(st, 6, tid)
        top = Float32(EMPTY)
        total = Float32(0)
        if passes > 0:
            # The last local row (MlaQuery may have inserted it).
            mine = (late >= p0) & (late < p0 + rows) & (tid < 36)
            lrow = cute.make_rmem_tensor(4, Uint32)
            if cutlass.const_expr(not self.prewait):  # noqa: SIM102
                if mine:
                    page = table[b * stride + late // self.page]
                    source = self.row_address(kv, page, late) + tid * 16
                    fetched = load_words(source, count=4, mutable=True)
                    for e in cutlass.range_constexpr(4):
                        lrow[e] = fetched[e]
            # MlaQuery does not wait for the gathered q (DCP): every 16 B chunk of
            # the rows is loaded (all loads in flight) until no word is the sentinel.
            qwords = cute.make_rmem_tensor(4 * len(self.qmap), Uint32)
            ready = True
            if cutlass.const_expr(self.polled_split or self.prewait):
                # the first round's loads in flight, then this CTA's arming stores
                self.issue_query(qwords, query, b, tid)
                if cutlass.const_expr(self.prewait):  # noqa: SIM102
                    if holds & (tid < 64):  # (warp-uniform)
                        self.await_insert()
                        self.stamp(st, 2, tid)
                        if mine:
                            page = table[b * stride + late // self.page]
                            source = self.row_address(kv, page, late) + tid * 16
                            fetched = load_words(source, count=4, mutable=True)
                            for e in cutlass.range_constexpr(4):
                                lrow[e] = fetched[e]
                if cutlass.const_expr(self.polled_split):
                    self.arm_split(args[ia], args[ia + 1], cta, tid)
                ready = self.query_ready(qwords)
            else:
                ready = self.load_query(qwords, query, b, tid)
            while not ready:
                ready = self.load_query(qwords, query, b, tid)
            w = 0
            for kb in cutlass.range_constexpr(5):
                chunks = 8 if kb < 4 else 4
                for i in cutlass.range_constexpr(-(-H * chunks // NT)):
                    q = tid + NT * i
                    full = cutlass.const_expr(NT * (i + 1) <= H * chunks)
                    if full or q < H * chunks:
                        qrow, chunk = q // chunks, q % chunks
                        target = base + kb * QBLOCK + swizzled(qrow, chunk)
                        words = tuple(qwords[4 * w + e] for e in range(4))
                        store_words(target, words, shared=True)
                    w += 1
                if cutlass.const_expr(kb == 0):  # noqa: SIM102 (compile-time)
                    if mine:
                        r, chunk = late - p0, tid % 8
                        words = tuple(lrow[e] for e in range(4))
                        target = ktile + (tid // 8) * KBLOCK + swizzled(r, chunk)
                        store_words(target, words, shared=True)
                        if tid // 8 == c:
                            self.convert_value(words, r, chunk, values)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            # Thread (quarter, part): its row's column part -> TMEM (A of QK).
            qbits = cute.make_rmem_tensor(QBYTES // 4, Uint32)
            for m in cutlass.range_constexpr(QBYTES // 16):
                g = (QBYTES // 16) * part + m  # 16 B chunk of the row
                origin = base + (g // 8) * QBLOCK + swizzled(row, g % 8)
                got = load_words(origin, count=4, shared=True)
                for e in cutlass.range_constexpr(4):
                    qbits[4 * m + e] = got[e]
            lanes = Uint32(32 * quarter) << 16
            words36 = tuple(qbits[e] for e in range(QBYTES // 4))
            store_tmem_words(tmem + lanes + TMEM_Q + QBYTES // 4 * part, words36)
            fence_tmem(before=True)
            cute.arch.mbarrier_arrive(bq + part)
            self.stamp(st, 7, tid)
            for s in range(passes):
                q0 = p0 + s * TN
                live = min(count - s * TN, TN)
                width = (live + 15) // 16 * 16
                if s > 0:  # the previous PV has read P (in the K image) and V
                    wait(bo + 1, (s - 1) % 2, scope="cta")  # (commits in order)
                    fence_tmem()
                    self.fill_pages(table, stride, pages, b, q0, n, tid)
                    cute.arch.sync_threads()
                    self.issue_tile(kvw, kv, pages, q0, live, tid)
                    self.store_tile(kvw, q0, live, width, -1, ktile, values, c, tid)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_threads()
                if tid == 0:
                    instruction = Uint32(QK_INSTR) | Uint32(width >> 3) << 17
                    cute.arch.fence_proxy("async.shared", space="cta")
                    fence_tmem()
                    for k in cutlass.range_constexpr(9):  # K64 steps
                        lo: cutlass.Constexpr = (
                            0 if k == 0 else (64 * k - 1) // QBYTES + 1
                        )
                        hi: cutlass.Constexpr = (64 * k + 63) // QBYTES + 1
                        for q in cutlass.range_constexpr(lo, hi):
                            if s == 0:
                                wait(bq + q, 0, scope="cta")
                            fence_tmem()
                        bk = desc128(ktile + (k // 2) * KBLOCK, 16) + Uint64(
                            4 * (k % 2)
                        )
                        mma_f8_tmem(tmem, tmem + TMEM_Q + 16 * k, bk, instruction, k)
                    commit_asm(bs)
                    if s == 0:
                        self.stamp(st, 8, tid)
                wait(bs, s % 2, scope="cta")
                fence_tmem()
                if s == 0:
                    self.stamp(st, 9, tid)
                # Softmax: warp (quarter, part) holds columns COLS * part.. of rows
                # 32 quarter..; max(raw) * scale = max(raw * scale) (scale > 0).
                lanes = Uint32(32 * quarter) << 16
                limit = live - COLS * part
                ninf = Float32(-float("inf"))
                raw = cute.make_rmem_tensor(COLS, Float32)
                peaks = [ninf] * 8
                if 32 * quarter < H:  # warp-uniform (tcgen05.ld)
                    loaded = load_tmem_cols(tmem + lanes + COLS * part, COLS)
                    for e in cutlass.range_constexpr(COLS):
                        x = loaded[e]
                        if e >= limit:
                            x = ninf
                        raw[e] = x
                        peaks[e % 8] = fmax(peaks[e % 8], x)
                    for w2 in (4, 2, 1):
                        for e in cutlass.range_constexpr(w2):
                            peaks[e] = fmax(peaks[e], peaks[e + w2])
                    partial[part * ROWS + row] = peaks[0]
                cute.arch.sync_threads()
                if s == 0:
                    self.stamp(st, 10, tid)
                if 32 * quarter < H:
                    peak = partial[row]
                    for k in cutlass.range_constexpr(1, PARTS):
                        peak = fmax(peak, partial[k * ROWS + row])
                    best = max(top, peak * scale)
                    alpha = cute.math.exp2(top - best, fastmath=True)
                    top = best
                    sums = cute.make_rmem_tensor(8, Float32)
                    sums.fill(0)
                    p = cute.make_rmem_tensor(COLS, Float32)
                    p.fill(0)
                    for g in cutlass.range_constexpr(COLS // 8):
                        if 8 * g < limit:  # (warp-uniform; exp2(-inf) = 0 past it)
                            for e in cutlass.range_constexpr(8 * g, 8 * g + 8):
                                p[e] = cute.math.exp2(
                                    fma(raw[e], scale, -best), fastmath=True
                                )
                                sums[e % 8] = sums[e % 8] + p[e]
                    for m in cutlass.range_constexpr(COLS // 8):
                        token = COLS * part + 8 * m
                        if token < width:
                            words = tuple(
                                f16x2(p[8 * m + 2 * e], p[8 * m + 2 * e + 1])
                                for e in range(4)
                            )
                            block = base + OFF_K + (token // 64) * QBLOCK
                            target = block + swizzled(row, (token % 64) // 8)
                            store_words(target, words, shared=True)
                    for w2 in (4, 2, 1):
                        for e in cutlass.range_constexpr(w2):
                            sums[e] = sums[e] + sums[e + w2]
                    total = total * alpha + sums[0]
                    partial[(PARTS + part) * ROWS + row] = total
                    if s == 0:
                        self.stamp(st, 11, tid)
                    if s > 0:  # rescale O (the previous PV is complete)
                        for ch in cutlass.range_constexpr(OCOLS // 32):
                            column = tmem + lanes + TMEM_O + OCOLS * part + 32 * ch
                            o = load_tmem_cols(column, 32)
                            store_tmem_cols(column, tuple(v * alpha for v in o))
                cute.arch.fence_proxy("async.shared", space="cta")
                fence_tmem(before=True)
                cute.arch.sync_threads()
                if s == 0:
                    self.stamp(st, 12, tid)
                if tid == 0:
                    fence_tmem()
                    self.issue_pv(tmem, base, values, width, s, bo)
            cute.arch.griddepcontrol_launch_dependents()
            wait(bo + part // 2, (passes - 1) % 2, scope="cta")  # (this part's half)
            fence_tmem()
        else:
            if cutlass.const_expr(self.polled_split):
                self.arm_split(args[ia], args[ia + 1], cta, tid)
            cute.arch.griddepcontrol_launch_dependents()
        self.stamp(st, 13, tid)
        # This split's unnormalized O times the output scale (warp (quarter, part):
        # columns OCOLS * part.. of its rows, 32 B stores) and (max, sum) per head.
        o = cute.make_rmem_tensor(OCOLS, Float32)
        o.fill(0)
        if (32 * quarter < H) & (passes > 0):
            lanes = Uint32(32 * quarter) << 16
            column = tmem + lanes + TMEM_O + OCOLS * part
            loaded = load_tmem_cols(column, OCOLS)
            for e in cutlass.range_constexpr(OCOLS):
                o[e] = loaded[e] * out_scale
        # Past the last tcgen05 operation: warp TMEM_WARP frees the TMEM while the
        # others store (no CTA-wide teardown barrier behind the slowest store issue).
        fence_tmem(before=True)
        if warp == TMEM_WARP:
            named_barrier(TMEM_DONE, NT)
            fence_tmem()
            free_tmem(tmem, 512)
        else:
            named_barrier(TMEM_DONE, NT, arrive=True)
        if (32 * quarter < H) & (row < H):
            stats = (c == 0) & (part == 0)
            total = Float32(0)
            if stats & (passes > 0):  # the parts' sums, in order
                for k in cutlass.range_constexpr(PARTS):
                    total = total + partial[(PARTS + k) * ROWS + row]
            if cutlass.const_expr(self.polled_split):  # noqa: SIM102 (LSE push first)
                if stats:
                    self.store_stats(stats_g, b, row, j, top, total)
            for k in cutlass.range_constexpr(OCOLS // 8):
                target = at(part_g, b, row, j, 128 * c + OCOLS * part + 8 * k)
                store_f32x8(target, tuple(o[8 * k + e] for e in range(8)))
            if cutlass.const_expr(not self.polled_split):  # noqa: SIM102
                if stats:
                    self.store_stats(stats_g, b, row, j, top, total)
        self.stamp(st, 14, tid)
        if cutlass.const_expr(self.prewait):
            # every grid waits before it completes (completion stays transitive)
            cute.arch.griddepcontrol_wait()
            self.stamp(st, 15, tid)
            if cutlass.const_expr(self.rearm_x):
                self.arm_x(args[9], cta, tid)


_POLLED_SPLITS: dict = {}


def polled_split_workspace(device, heads, splits, parity, batch=8):
    """(part [batch, heads, splits, 512], stats [batch, heads, 2, splits]) FP32 of one
    parity, armed with SPLIT_SENT, shared by every MLA layer of that parity."""
    key = device, heads, splits, parity
    if key not in _POLLED_SPLITS:
        i32 = dict(dtype=torch.int32, device=device)
        part = torch.full((batch, heads, splits, 512), -1, **i32)
        stats = torch.full((batch, heads, 2, splits), -1, **i32)
        _POLLED_SPLITS[key] = part.view(torch.float32), stats.view(torch.float32)
    return _POLLED_SPLITS[key]


@cache
def compile_dcp_core_umma(
    heads,
    batch,
    splits,
    cap,
    cache_spec,
    device,
    stamps=False,
    x_pitch=0,
    polled_split=False,
    prewait=False,
):
    """DcpCoreUmma of one batch; ``cache_spec``: the (shape, stride, dtype) of the
    uint8 paged cache view, ``cap``: the q mailbox's request capacity; ``x_pitch``
    (nonzero: MlaQuery polls x): the in-projection rows' pitch, x is re-armed;
    ``polled_split``: DcpValue polls the split workspace (POLLED_SPLIT), the other
    parity's (part, stats) follow and are armed; ``prewait``: PREWAIT."""
    shape, stride, _ = cache_spec
    query = fake((cap, heads, 576), None, torch.uint8)
    kv = fake(shape, stride, torch.uint8)
    lens = fake((batch,), None, torch.int32, 4)
    part = fake((batch, heads, splits, 512), None, torch.float32)
    stats = fake((batch, heads, 2, splits), None, torch.float32, 4)
    args: tuple = (query, kv, Int64(0), Int32(0), lens, part, stats)
    args += (Float32(0), Float32(0))
    if x_pitch:
        args += (fake((batch, 1536), (x_pitch, 1)),)
    if polled_split:
        args += (part, stats)
    args += (Int64(0),) * stamps + (make_fake_stream(),)
    core = DcpCoreUmma(
        heads, batch, splits, shape[1], stamps, bool(x_pitch), polled_split, prewait
    )
    with torch.accelerator.device_index(device):
        return cute.compile(core, *args)
