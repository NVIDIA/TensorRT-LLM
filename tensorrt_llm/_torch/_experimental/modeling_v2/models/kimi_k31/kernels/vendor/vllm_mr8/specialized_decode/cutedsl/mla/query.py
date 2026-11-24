# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cluster MLA query RMSNorm, Q_B projection, W_UK absorption and paged FP8 KV insert.
Static weights load before the PDL wait; current-token data is consumed after it."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    copy_evict,
    copy_g2s,
    copy_s2cluster,
    float_bits,
    fma,
    fma2,
    fp8_bits,
    launch,
    ldmatrix,
    load_bf16x8,
    load_volatile,
    load_words,
    mma_into,
    mul2,
    multicast_words,
    pack_fp8,
    prefetch_l2,
    published,
    push_float,
    query_norm_rsqrt,
    sleep_ns,
    smem_words,
    store_fp8_pair,
    store_shared_words,
    unpack_bf16,
    view,
    wait,
    warp_sum,
    zeros,
)
from .qvalid import off_sentinel, push_words

KP = 1536 + 32  # padded Q_B / x row (elements): 3136 B, conflict-free 16 B fragments
# The smallest batch on the tiled kernel (per head count):
# its 16-request tiles beat FP32 FMAs from B4 at 12 heads and from B6 at 6 heads.
TILED_FROM = {12: 4, 6: 6}
# The DSpark drafter's MLA layers (4 heads per rank at TP16, 8 at TP8; ``rope``): the
# lean tile from one row (MlaQuery(rope=True) needs the lean kernel).
DRAFT_TILED_FROM = {4: 1, 8: 1}
Q_SENT = 0xFFFFFFFF  # q mailbox sentinel: four FP8 0xFF bytes, never written
# Rubin-class devices (more SMs than GB300's 152), DCP16, q = 1: batches whose gathered
# q poll is gated on one chunk per source rank (``MlaQuery.gate``). 16-rank MLA chain
# (MlaQuery -> tokenspeed -> DcpValue, ctx 8k), us/layer ungated vs gated: B1 21.20
# vs 18.68, B4 21.50 vs 21.51, B8 25.26 vs 24.70-24.75; checks vs stock unchanged.
GATED_GATHER = {16: batch_table("1-8:1")}
# Rubin-class devices, DCP16, spec verify (q > 1): row batches whose gathered q poll is
# gated the same way (``MlaQuery.gate``; 2: and re-polls every late chunk of a pass
# together). MLA chain (in-proj -> MlaQuery -> tokenspeed -> DCP value), B8 x q8, ctx
# 8800, us/layer, 16 ranks in one process: ungated 62.45, gate 57.36, gate + batched
# re-polls 57.01 (with dcp_uv 60.48 / 55.39 / 54.92); outputs bitwise identical.
GATED_VERIFY = {16: batch_table("16-64:2")}

GB300_SMS = 152  # Rubin-class schedules key on more SMs than this (as mla.dcp)
# Rubin, DCP16 (TP16), q = 1, with the tcgen05 core (DcpCoreUmma): batches whose lean
# kernel polls x (the in-projection's q columns) as a sentinel row instead of waiting
# for the in-projection's grid completion (see ``MlaQuery.poll``).
POLLED_X = {16: batch_table("8:1")}
X_SENT = 0xFFFF  # x sentinel per BF16 element (a NaN the in-projection never writes)
# Rubin, tiled (not lean, not the drafter's rope): batches whose RMSNorm pass only
# computes each row's rsqrt factor; x * rsqrt * gamma (BF16) is applied to the Q_B A
# fragments as they load (the same packed operations per element, without the
# write-back pass and its barrier).
FUSED_NORM = batch_table("9-64:1")
STAGE = 96  # staged FP8 query bytes per request: 64 latent, then this CTA's R rope rows
# Rubin, tiled (not the drafter's rope): requests per CTA tile (``tokens``) by heads
# per rank and batch rows, instead of the default 16 / 32 / 64. The grid is 8-CTA
# clusters x heads x row tiles at one CTA per SM, and ~21-23 clusters run at once, so
# the tile needing the fewest cluster rounds wins. Same arithmetic per row. 6 heads
# (TP16), 4 ranks, MlaQuery alone, us, 32 -> 48: M112 10.14 -> 8.22, M128 10.57 ->
# 8.29 (two rounds -> one), M256 16.05 -> 15.30 (three -> two); M96 / M168 / M192 /
# M224 keep 32. A tile is a multiple of 16 requests, at most 8 x heads (the tile's CTAs
# insert its KV rows, one request each). 17-32 rows: two 16-request tiles (96 CTAs)
# instead of one of 32 halve each CTA's RMSNorm and Q_B MMAs; 1 GPU behind the real
# 32-row in-projection, MlaQuery chained 6.09 -> 4.47 us (bitwise: q mailbox, KV rows).
WIDE_TOKENS = {6: batch_table("17-32:16,97-144:48,225-256:48")}
# Rubin, DCP, tiled with staged output: row batches whose q publish keeps mla.qvalid's
# contract (``MlaQuery(valid=True)``) for a validity-polled reader. LSU readers are
# indifferent to it (the same bytes, whole 16 B chunks). The B1 lean batches follow
# mla.dcp.FORK_VALID.
VALID_PUBLISH = batch_table("")
# Rubin, lean with a polled x: row batches whose x poll is one warp-uniform loop (a
# vote ends it; each pass re-polls the warp's x row) instead of each lane spinning on
# its own chunks: the norm's shuffles then start without the warp reconverging after
# its last lane. The same loads and values. 16 ranks, B1 rows, us/layer at 8.8k:
# rows chain 19.63 -> 19.53; with the attention-residual front 26.61 -> 25.94.
QUERY_X_UNIFORM = batch_table("8:1")
# False only in the tests' positive control (QUERY_X_UNIFORM: x read once)
POLL_X = True


@asm()
def store_fp8_pair_shared(pointer, low, high):
    """FP8 E4M3 (satfinite) of ``low`` and ``high`` as the two shared bytes at
    ``pointer`` (as store_fp8_pair)."""
    code = "{ .reg .b16 pair; cvt.rn.satfinite.e4m3x2.f32 pair,$2,$1; "
    code += "st.shared.b16 [$0],pair; }"
    return code, "r,f,f,~{memory}", (pointer, low, high)


@cute.jit
def rotate(values, position, cos_sin, pair0):
    """GPT-J RoPE of consecutive (x, y) pairs ``values`` (pair indices pair0..) at
    ``position`` in FP32, as the stock fused kernel: x cos - y sin, x sin + y cos."""
    out = list(values)
    for j in cutlass.range_constexpr(len(values) // 2):
        cos = cos_sin[position, pair0 + j]
        sin = cos_sin[position, 32 + pair0 + j]
        x, y = values[2 * j], values[2 * j + 1]
        out[2 * j] = fma(x, cos, -(y * sin))
        out[2 * j + 1] = fma(x, sin, y * cos)
    return out


def x_ready(chunks):
    """No BF16 element of the 16 B ``chunks`` is the x sentinel."""
    ready = True
    for chunk in chunks:
        for word in chunk:
            ready &= (word & Uint32(X_SENT)) != Uint32(X_SENT)
            ready &= (word >> 16) != Uint32(X_SENT)
    return ready


class MlaQuery:
    """A cluster splits one head's 192 Q_B rows; each CTA then absorbs its share of the
    512 latent columns. Small batches (below TILED_FROM) are served by one CTA per
    cluster part with FP32 FMAs; wider batches tile 16-64 requests per CTA (one wave
    of 96 CTAs at B64 for both head counts) and use BF16 MMAs (FP32 accumulation).

    ``dcp`` (decode context parallel size; args then end with this parity's local q
    mailbox, its multicast address and the rank): each CTA also multicasts the query
    bytes it wrote into head block ``rank`` of every rank's Lamport q mailbox
    (mla.dcp.DcpBox), then polls its share of the gathered mailbox until no word is
    the sentinel, so the next kernel reads the gathered query."""

    def __init__(
        self,
        heads: int,
        batch: int,
        *,
        q_b_cp_async: bool = False,
        dcp: int = 0,
        gate: bool = False,
        gather: bool = True,
        sms: int = 148,
        poll_x: bool = False,
        rope: bool = False,
        tokens: int = 0,
        valid: bool = False,
        x_uniform: bool = False,
    ):
        # ``rope``: GPT-J RoPE of the query and key rope columns at the rows'
        # positions (args then end with positions [B] int64 and the FP32 cos/sin
        # cache [max position, 64]), as the stock fused query concat + cache insert:
        # FP32 rotation of the BF16 values, then the FP8 quantization (the DSpark
        # drafter; the target's MLA layers have no RoPE).
        self.rope = rope
        tiled = (DRAFT_TILED_FROM if rope else TILED_FROM)[heads]
        self.heads, self.batch, self.wide = heads, batch, batch >= tiled
        self.dcp = dcp
        # ``gate``: lanes 0..DCP-1 wait for one chunk per source rank before the whole
        # grid polls its share of the gathered mailbox (every word is still checked);
        # 2: each pass's late chunks are re-polled together (one round trip per pass,
        # not one per late chunk).
        self.gate = bool(gate) and dcp > 0
        self.batched_gather = self.gate and gate == 2
        # ``gather`` False: the core polls the gathered q rows it reads (Rubin DcpCore),
        # so the query kernel ends after its publish (and KV insert).
        self.gather = gather
        # Rubin-class devices (more SMs than GB300's 152): the tiled kernel stages its
        # Q_B rows in shared memory before the PDL wait (bulk copies; L2 prefetches can
        # be dropped while the input projection streams) and, under DCP, runs the KV
        # insert after the q publish, beside the gather poll. GB300 keeps the
        # original schedule.
        rubin = sms > GB300_SMS
        self.stage_qb = rubin and self.wide
        self.late_insert = rubin and self.wide and bool(dcp)
        # Rubin, tiled, DCP: the CTA's FP8 query bytes are staged in shared memory and
        # published from there (no global write + read-back; under DCP the core reads
        # the gathered mailbox, never the local query rows).
        self.stage_out = rubin and self.wide and bool(dcp)
        # Rubin, one 16-request tile holding at most 8 requests: the RMSNorm skips the
        # tile's dead upper rows (their MMA rows only feed outputs that are not stored).
        self.half_tile = rubin and self.wide and batch <= 8
        # Rubin, tiled, DCP, at most 8 requests: ``lean`` (same arithmetic as ``tiled``,
        # fewer barrier phases after the PDL wait).
        self.lean = self.stage_out and batch <= 8
        self.fused_norm = rubin and self.wide and not self.lean and not rope
        self.fused_norm = self.fused_norm and batch in FUSED_NORM
        # ``poll_x``: the lean kernel polls x (``poll``); the core re-arms it.
        self.poll_x = poll_x and self.lean and not gather
        self.x_uniform = bool(x_uniform) and self.poll_x  # QUERY_X_UNIFORM
        # ``valid`` (Rubin DCP: lean, or tiled with staged output): every q mailbox
        # chunk is one 16 B store whose first word is never 0x80000000, for readers
        # that poll with validity-reporting copies (mla.qvalid; the fork's FORK_VALID).
        # The rope chunk that straddles two cluster parts is joined through DSMEM.
        # Same arithmetic (bitwise).
        assert not valid or self.stage_out, "MlaQuery(valid=True): Rubin DCP only"
        self.valid = bool(valid)
        assert not rope or self.lean or self.stage_out, "MlaQuery rope: Rubin DCP"
        self.cluster = 16 if heads == 6 and not self.wide else 8
        self.rows, self.latent = 192 // self.cluster, 512 // self.cluster
        assert not valid or self.rows == 24, "MlaQuery(valid=True): 24 Q_B rows per CTA"
        self.groups, self.p_per_group = 256 // self.latent, self.latent // 2
        self.tokens = 1 if batch == 2 else min(batch, 8)
        if self.wide:
            self.tokens = 16 if batch <= 16 else 32 if batch <= 32 or heads < 8 else 64
            if sms > GB300_SMS and not rope:
                self.tokens = WIDE_TOKENS.get(heads, {}).get(batch, self.tokens)
            # ``tokens``: the caller's tile (16, 32 or 64 requests per CTA).
            self.tokens = tokens or self.tokens
            # CTA j of a tile inserts request j's KV row: the tile's CTAs cover it.
            assert self.tokens <= 8 * heads  # (the wide cluster is 8)
            assert self.tokens % 16 == 0  # (16-row MMA tiles)
        self.grid = (self.cluster, heads, (batch + self.tokens - 1) // self.tokens)
        carveout = 100 if batch > 1 else None
        self.options = dict(cluster=self.cluster, preferred_smem_carveout=carveout)
        self.q_b_cp_async = q_b_cp_async

    @cute.jit
    def __call__(self, *args):
        kernel = self.tiled if self.wide else self.kernel
        kernel = self.lean_kernel if self.lean else kernel
        launch(kernel, args, self.grid, 256, **self.options)

    @cute.jit
    def live(self, token):
        return cutlass.const_expr(not self.wide) or token < self.batch

    @cute.jit
    def publish(self, args, tok0, tid, staged=None):
        """DCP: this CTA's query bytes (latent columns L*part.. as 16 B, the rope rows
        of Q_B rows R*part.. as 4 B multicast stores) -> every rank's q mailbox (FP8
        bytes are never 0xFF: no word is the sentinel)."""
        output, (box_mc, rank) = args[10], args[12:14]
        part, head, _ = cute.arch.block_idx()
        R, L, D = self.rows, self.latent, self.dcp
        chunks: cutlass.Constexpr = L // 16
        items: cutlass.Constexpr = chunks + R // 4
        cute.arch.sync_threads()
        for i in range(tid, self.tokens * items, 256):
            t, w = i // items, i % items
            p = R * part + 4 * (w - chunks)  # Q_B row of a rope word
            row = ((tok0 + t) * D + rank) * self.heads + head
            if w < chunks:
                byte = L * part + 16 * w
                if self.live(tok0 + t):
                    if cutlass.const_expr(staged is not None):
                        words = smem_words(staged.iterator + t * STAGE + 16 * w)
                    else:
                        words = load_words(
                            at(output, tok0 + t, head, byte), count=4, mutable=True
                        )
                    multicast_words(box_mc + Int64(row) * 576 + byte, words)
            elif self.live(tok0 + t) & (p >= 128):
                byte = 512 + p - 128
                if cutlass.const_expr(staged is not None):
                    rope = staged.iterator + t * STAGE + 64 + 4 * (w - chunks)
                    word = load_words(rope, count=1, shared=True)
                else:
                    word = load_words(
                        at(output, tok0 + t, head, byte), count=1, mutable=True
                    )
                multicast_words(box_mc + Int64(row) * 576 + byte, word, kind="b32")

    @cute.jit
    def publish_valid(self, args, tok0, tid, staged, halves):
        """``publish`` under mla.qvalid's contract (``valid``, staged output, 24 Q_B
        rows per CTA): each 16 B chunk of the CTA's query bytes is ONE multicast store
        whose first word is never 0x80000000; never split back into 4 / 8 B pieces (a
        validity-polled reader checks only the first word). Rope chunks: the CTA
        holding a chunk's last byte stores it; the chunk whose first 8 B are the
        previous part's takes them from ``halves`` (pushed before the barq wait)."""
        box_mc, rank = args[12:14]
        part, head, _ = cute.arch.block_idx()
        R, L, D = self.rows, self.latent, self.dcp
        o = R * part - 128  # rope byte offset of this CTA's first Q_B row
        chunks: cutlass.Constexpr = L // 16
        items: cutlass.Constexpr = chunks + 2  # latent, then up to two rope chunks
        cute.arch.sync_threads()
        for i in range(tid, self.tokens * items, 256):
            t, w = i // items, i % items
            row = ((tok0 + t) * D + rank) * self.heads + head
            base = staged.iterator + t * STAGE
            if self.live(tok0 + t):
                if w < chunks:
                    q16 = list(smem_words(base + 16 * w))
                    q16[0] = off_sentinel(q16[0])
                    at_q = box_mc + Int64(row) * 576 + L * part + 16 * w
                    multicast_words(at_q, tuple(q16))
                else:
                    # rope chunk starting at byte s (a multiple of 16) whose last byte
                    # is this CTA's: the k-th such chunk after the first one
                    s = ((o + 15) // 16) * 16 + 16 * (w - chunks)
                    if self.rope_tail(part):
                        s = o - 8 + 16 * (w - chunks)
                    if (o >= -R) & (s >= 0) & (s + 15 < o + R) & (s + 15 >= o):
                        lo = load_words(base + 64 + s - o + 8, count=2, shared=True)
                        hi = lo
                        if s >= o:
                            hi = load_words(base + 64 + s - o, count=2, shared=True)
                        else:  # the previous part's half
                            hi = load_words(
                                halves.iterator + 2 * t, count=2, shared=True
                            )
                        rope16 = (off_sentinel(hi[0]), hi[1], lo[0], lo[1])
                        at_r = box_mc + Int64(row) * 576 + 512 + s
                        multicast_words(at_r, rope16)

    @cute.jit
    def gathered(self, args, tid):
        """DCP: poll this CTA's share of the gathered mailbox (16 B chunks of the
        B x DCP*H x 576 bytes) until every rank's bytes have landed."""
        mailbox = args[11]
        part, head, tile = cute.arch.block_idx()
        if cutlass.const_expr(self.gate):
            if tid < self.dcp:
                row = ((self.batch - 1) * self.dcp + tid) * self.heads + head
                ptr = mailbox.iterator + row * 576
                words = load_volatile(ptr, count=4)
                while not published(words, sentinel=Q_SENT):
                    words = load_volatile(ptr, count=4)
            cute.arch.sync_threads()
        threads: cutlass.Constexpr = self.grid[0] * self.grid[1] * self.grid[2] * 256
        chunks: cutlass.Constexpr = self.batch * self.dcp * self.heads * 36
        first = ((tile * self.heads + head) * self.cluster + part) * 256 + tid
        last = chunks - 1  # threads past the end re-poll it
        for i0 in cutlass.range_constexpr(0, -(-chunks // threads), 8):
            n: cutlass.Constexpr = min(8, -(-chunks // threads) - i0)
            sources = [
                mailbox.iterator + min(first + (i0 + i) * threads, last) * 16
                for i in range(n)
            ]
            loads = [load_volatile(source, count=4) for source in sources]
            if cutlass.const_expr(self.batched_gather):
                pending = Int32(0)
                for i in cutlass.range_constexpr(n):
                    if not published(loads[i], sentinel=Q_SENT):
                        pending |= 1 << i
                while pending != 0:
                    again = Int32(0)
                    for i in cutlass.range_constexpr(n):
                        if (pending & (1 << i)) != 0:
                            words = load_volatile(sources[i], count=4)
                            if not published(words, sentinel=Q_SENT):
                                again |= 1 << i
                    pending = again
            else:
                for i in cutlass.range_constexpr(n):  # the loads above are in flight
                    words = loads[i]
                    while not published(words, sentinel=Q_SENT):
                        words = load_volatile(sources[i], count=4)

    @cute.kernel
    def kernel(self, args):
        qkv, w_qa, w_kva, w_qb, w_uk, cache, slots, q_scale_inv = args[0:8]
        k_scale_inv, eps, output = args[8:11]
        tid, _, _ = cute.arch.thread_idx()
        part, head, tok0 = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        B, R, L, G = self.tokens, self.rows, self.latent, self.groups
        smem = cutlass.utils.SmemAllocator()
        qb = alloc(smem, BFloat16, (R, 1536), 128)
        uk = alloc(smem, BFloat16, (128, L))
        qa = alloc(smem, BFloat16, 1536)
        xq = alloc(smem, BFloat16, (B, 1536))
        qrow = alloc(smem, Float32, (R, B))
        qn = alloc(smem, Float32, (B, 128))
        red = alloc(smem, Float32, (G, B, L))
        ss = alloc(smem, Float32, (8, max(B, 2)))
        if cutlass.const_expr(not self.q_b_cp_async):
            barw = smem.allocate(cutlass.Uint64, 8)
        barq = smem.allocate(cutlass.Uint64, 8)
        if tid == 0:
            if cutlass.const_expr(not self.q_b_cp_async):
                cute.arch.mbarrier_init(barw, 1)
            cute.arch.mbarrier_init(barq, 1)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(not self.q_b_cp_async):
                cute.arch.mbarrier_arrive_and_expect_tx(barw, R * 1536 * 2)
                copy_evict(
                    qb.iterator,
                    w_qb.iterator + (head * 192 + part * R) * 1536,
                    size=R * 1536 * 2,
                    barrier=barw,
                )
            cute.arch.mbarrier_arrive_and_expect_tx(barq, B * 128 * 4)
        cute.arch.cluster_arrive_relaxed()
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            copy_evict(uk.iterator + p * L + j * 8, at(w_uk, head, p, part * L + j * 8))
        if tid < 192:
            copy_evict(qa.iterator + tid * 8, w_qa.iterator + tid * 8)
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(self.q_b_cp_async):
            for c in range(tid, R * 1536 // 8, 256):
                dst = qb.iterator + c * 8
                src = w_qb.iterator + (head * 192 + part * R) * 1536 + c * 8
                copy_evict(dst, src)
            cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        qsi = q_scale_inv[0]
        xv = cute.make_rmem_tensor((B, 8), Float32)
        for b in cutlass.range_constexpr(B):
            total = Float32(0)
            if (tid < 192) & self.live(tok0 + b):
                values = load_bf16x8(at(qkv, tok0 + b, tid * 8), mutable=True)
                for i in cutlass.range_constexpr(8):
                    xv[b, i] = values[i]
                    total = fma(values[i], values[i], total)
            total = warp_sum(total)
            if lane == 0:
                ss[warp, b] = total
        cute.arch.cp_async_wait_group(1 if self.q_b_cp_async else 0)
        cute.arch.sync_threads()
        if tid < 192:
            weights = load_bf16x8(qa.iterator + tid * 8, shared=True)
            for b in cutlass.range_constexpr(B):
                total = Float32(0)
                for w in cutlass.range_constexpr(8):
                    total += ss[w, b]
                rr = query_norm_rsqrt(total, eps)
                for i in cutlass.range_constexpr(8):
                    xq[b, tid * 8 + i] = BFloat16(xv[b, i] * rr * weights[i])
        cute.arch.sync_threads()
        if cutlass.const_expr(self.q_b_cp_async):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        else:
            wait(barw)
        for row in range(warp, R, 8):
            acc = zeros((B, 2), Float32)
            for chunk in cutlass.range_constexpr(6):
                k = (chunk * 32 + lane) * 8
                weights = load_bf16x8(qb.iterator + row * 1536 + k, shared=True)
                for b in cutlass.range_constexpr(B):
                    values = load_bf16x8(xq.iterator + b * 1536 + k, shared=True)
                    for i in cutlass.range_constexpr(8):
                        acc[b, i % 2] = fma(weights[i], values[i], acc[b, i % 2])
            for b in cutlass.range_constexpr(B):
                value = warp_sum(acc[b, 0] + acc[b, 1])
                if lane == 0:
                    qrow[row, b] = Float32(BFloat16(value))
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        nrows = min(R, max(0, 128 - R * part))
        for i in range(tid, nrows * B * self.cluster, 256):
            dst, jb = i % self.cluster, i // self.cluster
            j, b = jb % nrows, jb // nrows
            cute.arch.store_async_dsmem(
                qn.iterator + b * 128 + R * part + j, float_bits(qrow[j, b]), barq, dst
            )
        for i in range(tid, R * B, 256):
            j, b = i % R, i // R
            p = R * part + j
            if (p >= 128) & self.live(tok0 + b):
                output[tok0 + b, head, 512 + p - 128] = fp8_bits(qrow[j, b] * qsi)
        wait(barq)
        latent_col, group = tid % L, tid // L
        acc = zeros((B, 2), Float32)
        for i in cutlass.range_constexpr(self.p_per_group):
            p = group * self.p_per_group + i
            w = Float32(uk[p, latent_col])
            for b in cutlass.range_constexpr(B):
                acc[b, i % 2] = fma(qn[b, p], w, acc[b, i % 2])
        for b in cutlass.range_constexpr(B):
            red[group, b, latent_col] = acc[b, 0] + acc[b, 1]
        cute.arch.sync_threads()
        for i in range(tid, L * B, 256):
            b, ll = i // L, i % L
            value = Float32(0)
            for g in cutlass.range_constexpr(G):
                value += red[g, b, ll]
            if self.live(tok0 + b):
                rounded = Float32(BFloat16(value))
                output[tok0 + b, head, L * part + ll] = fp8_bits(rounded * qsi)
        if cutlass.const_expr(self.dcp):
            self.publish(args, tok0, tid)
        valid = (head == 0) & (part < B) & self.live(tok0 + part)
        self.insert(args, ss, tok0, part, valid, tid)
        if cutlass.const_expr(self.dcp and self.gather):
            self.gathered(args, tid)

    @cute.jit
    def insert(self, args, ss, tok0, j, valid, tid):
        """RMSNorm the current latent KV row of request tok0 + j and write it, with
        k_pe, to its FP8 cache slot."""
        qkv, w_kva, (cache, slots), (k_scale_inv, eps) = (
            args[0],
            args[2],
            args[5:7],
            args[8:10],
        )
        lane, warp = tid % 32, tid // 32
        slot = cutlass.Int64(-1)
        if valid:
            slot = slots[tok0 + j]
        if slot >= 0:
            ksi = k_scale_inv[0]
            xx = cute.make_rmem_tensor(8, Float32)
            total = Float32(0)
            if tid < 64:
                values = load_bf16x8(at(qkv, tok0 + j, 1536 + tid * 8), mutable=True)
                for i in cutlass.range_constexpr(8):
                    xx[i] = values[i]
                    total = fma(values[i], values[i], total)
            total = warp_sum(total)
            cute.arch.sync_threads()
            if (lane == 0) & (warp < 2):
                ss[warp, 0] = total
            cute.arch.sync_threads()
            page, offset = slot // cache.shape[1], slot % cache.shape[1]
            if tid < 64:
                mean = fma(ss[0, 0] + ss[1, 0], Float32(1 / 512), eps)
                rr = cute.math.rsqrt(mean, fastmath=True)
                weights = load_bf16x8(w_kva.iterator + tid * 8)
                for i in cutlass.range_constexpr(8):
                    normed = Float32(BFloat16(xx[i] * rr * weights[i]))
                    cache[page, offset, tid * 8 + i] = fp8_bits(normed * ksi)
            elif tid < 72:
                c = tid - 64
                values = load_bf16x8(at(qkv, tok0 + j, 2048 + c * 8), mutable=True)
                pe = list(values)
                if cutlass.const_expr(self.rope):
                    pe = rotate(pe, args[14][tok0 + j], args[15], c * 4)
                for i in cutlass.range_constexpr(8):
                    cache[page, offset, 512 + c * 8 + i] = fp8_bits(pe[i] * ksi)

    @cute.kernel
    def tiled(self, args):
        """B>8: T requests per CTA. Q_B fragments load to registers before the PDL
        wait; in-place RMSNorm of the staged rows, split-K Q_B MMAs over the warps,
        nope rows gathered from the six producing CTAs by bulk DSMEM copies (into the
        dead x rows), then W_UK MMAs (warp w owns latent columns 8w..8w+7)."""
        qkv, w_qa, w_qb, w_uk, q_scale_inv = (args[i] for i in (0, 1, 3, 4, 7))
        eps, output = args[9], args[10]
        tid, _, _ = cute.arch.thread_idx()
        part, head, tok0 = cute.arch.block_idx()
        tok0 *= self.tokens
        smem = cutlass.utils.SmemAllocator()
        T, R, L = self.tokens, self.rows, self.latent
        MT, senders = T // 16, (128 + R - 1) // R
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        xq = alloc(smem, BFloat16, (T, KP), 128)
        uk = alloc(smem, BFloat16, (128, L + 8), 128)  # 144 B rows for ldmatrix
        qa = alloc(smem, BFloat16, 1536)
        ss = alloc(smem, Float32, (8, 2))
        rrs = alloc(smem, Float32, T) if self.fused_norm else None  # rsqrt per row
        barq = smem.allocate(cutlass.Uint64, 8)
        if cutlass.const_expr(self.stage_qb):
            qbs = alloc(smem, BFloat16, (R, KP), 128)  # padded rows as x
            barb = smem.allocate(cutlass.Uint64, 8)
        qo = None
        if cutlass.const_expr(self.stage_out):
            qo = alloc(smem, cutlass.Uint8, (T, STAGE), 16)
        if cutlass.const_expr(self.valid):
            halves = alloc(smem, Uint32, (T, 2))  # the previous part's rope half
        # After the Q_B MMAs the x rows hold the split-K partials, then our nope rows
        # and the six producers' (peers write them only after our cluster arrival).
        red = view(xq.iterator, 0, Float32, (8, T * R // 32, 32), (T * R, 32, 1))
        qblk = view(xq.iterator, 8 * T * R * 4, BFloat16, (T, R), (R, 1))
        qn = view(xq.iterator, 9 * T * R * 4, BFloat16, (senders, T, R), (T * R, R, 1))
        if tid == 0:
            cute.arch.mbarrier_init(barq, 1)
            if cutlass.const_expr(self.stage_qb):
                cute.arch.mbarrier_init(barb, 1)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(self.valid):
                # [valid] + the previous part's 8 B half of a straddling rope chunk
                extra = Int32(0)
                if self.rope_tail(part):
                    extra = Int32(T * 8)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    barq, senders * T * R * 2 + extra
                )
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(barq, senders * T * R * 2)
            if cutlass.const_expr(self.stage_qb):  # one bulk copy per Q_B row
                cute.arch.mbarrier_arrive_and_expect_tx(barb, R * 1536 * 2)
                for r in cutlass.range_constexpr(R):
                    src = w_qb.iterator + (head * 192 + part * R + r) * 1536
                    copy_evict(qbs.iterator + r * KP, src, size=3072, barrier=barb)
        # Split-K Q_B: warp w owns k = 192w..192w+191 (8 values per lane and 32-wide
        # chunk; the x fragments use the same permutation). L2 now, registers later
        # (Rubin: from the staged rows).
        rows = head * 192 + part * R + group
        qb_at = [
            [
                at(w_qb, rows + 8 * n, (6 * warp + cc) * 32 + 8 * quarter)
                for n in range(3)
            ]
            for cc in range(6)
        ]
        if cutlass.const_expr(not self.stage_qb):
            for chunk in qb_at:
                for address in chunk:
                    prefetch_l2(address)
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            dst = uk.iterator + p * (L + 8) + j * 8
            copy_evict(dst, at(w_uk, head, p, part * L + j * 8))
        if tid < 192:
            copy_evict(qa.iterator + tid * 8, w_qa.iterator + tid * 8)
        cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for c in range(tid, T * 192, 256):
            t, j = c // 192, c % 192
            if self.live(tok0 + t):
                copy_g2s(xq.iterator + t * KP + j * 8, at(qkv, tok0 + t, j * 8))
        cute.arch.cp_async_commit_group()
        qsi = q_scale_inv[0]
        # The KV insert overlaps the staging of x (CTA j of the tile owns request j;
        # Rubin under DCP: after the publish).
        j = head * self.cluster + part
        if cutlass.const_expr(not self.late_insert):
            self.insert(args, ss, tok0, j, (j < T) & self.live(tok0 + j), tid)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # Packed FP32 pairs: the same IEEE ops in the same order as the scalar form.
        gamma = []
        for i in cutlass.range_constexpr(6):
            gamma += list(load_bf16x8(qa.iterator + 256 * i + 8 * lane, shared=True))
        for pair in cutlass.range_constexpr(T // 16):  # requests warp + 8k, two at once
            xrows = [xq.iterator + (warp + 16 * pair + 8 * h) * KP for h in range(2)]
            values: list = []
            rr: list = []
            rows_n = 2 - int(self.half_tile)  # half_tile: rows warp + 8 are dead
            for h in cutlass.range_constexpr(rows_n):
                values.append([])
                for i in cutlass.range_constexpr(6):
                    for word in smem_words(xrows[h] + 256 * i + 8 * lane):
                        values[h] += list(unpack_bf16(word))
            for h in cutlass.range_constexpr(rows_n):
                zero = Float32(0), Float32(0)
                chains = [zero, zero]  # (0, 1), (2, 3): element e feeds chain e % 4
                for e in cutlass.range_constexpr(0, 48, 2):
                    x = values[h][e], values[h][e + 1]
                    chains[e % 4 // 2] = fma2(x, x, chains[e % 4 // 2])
                (c0, c1), (c2, c3) = chains
                rr.append(query_norm_rsqrt(warp_sum((c0 + c1) + (c2 + c3)), eps))
                if cutlass.const_expr(self.fused_norm):  # noqa: SIM102
                    if lane == 0:
                        rrs[warp + 16 * pair + 8 * h] = rr[h]
            for h in cutlass.range_constexpr(0 if self.fused_norm else rows_n):
                for i in cutlass.range_constexpr(6):
                    words = []
                    for e in cutlass.range_constexpr(8 * i, 8 * i + 8, 2):
                        x = values[h][e], values[h][e + 1]
                        y = mul2(mul2(x, (rr[h], rr[h])), (gamma[e], gamma[e + 1]))
                        words.append(bf16x2(*y))
                    store_shared_words(xrows[h] + 256 * i + 8 * lane, tuple(words))
        if cutlass.const_expr(self.stage_qb):
            wait(barb)
            qbr = [
                [
                    smem_words(
                        qbs.iterator
                        + (group + 8 * n) * KP
                        + (6 * warp + cc) * 32
                        + 8 * quarter
                    )
                    for n in range(3)
                ]
                for cc in range(6)
            ]
        else:
            qbr = [[load_words(a, count=4) for a in chunk] for chunk in qb_at]
        cute.arch.sync_threads()
        acc = zeros((MT * 3, 4), Float32)
        for cc in cutlass.range_constexpr(6):
            k = (6 * warp + cc) * 32 + 8 * quarter
            z = qbr[cc]
            for m in cutlass.range_constexpr(MT):
                x, y = (
                    load_words(xq.iterator + r * KP + k, count=4, shared=True)
                    for r in (16 * m + group, 16 * m + group + 8)
                )
                if cutlass.const_expr(self.fused_norm):  # x * rsqrt * gamma here
                    g = load_bf16x8(qa.iterator + k, shared=True)
                    x, y = (
                        self.scaled(words, rrs[r], g)
                        for words, r in ((x, 16 * m + group), (y, 16 * m + group + 8))
                    )
                for n in cutlass.range_constexpr(3):
                    for s in cutlass.range_constexpr(2):
                        a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                        b = (z[n][2 * s], z[n][2 * s + 1])
                        mma_into(acc[m * 3 + n, None], a, b, bf16=True)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()  # x is consumed: peers may fill qn
        for i in cutlass.range_constexpr(MT * 12):
            red[warp, i, lane] = acc[i // 4, i % 4]
        cute.arch.sync_threads()
        if cutlass.const_expr(self.rope):
            self.rope_rows(args, red, qblk, qo, tok0, tid, qsi)
        else:
            for e in range(tid, T * R, 256):
                i, ln = e // 32, e % 32
                value = Float32(0)
                for w in cutlass.range_constexpr(8):
                    value += red[w, i, ln]
                value = Float32(BFloat16(value))
                tile, c = i // 4, i % 4
                t = 16 * (tile // 3) + ln // 4 + 8 * (c // 2)
                r = 8 * (tile % 3) + 2 * (ln % 4) + c % 2
                p = R * part + r
                if p < 128:
                    qblk[t, r] = BFloat16(value)
                elif cutlass.const_expr(self.stage_out):
                    qo[t, 64 + r] = fp8_bits(value * qsi)
                elif self.live(tok0 + t):
                    output[tok0 + t, head, 512 + p - 128] = fp8_bits(value * qsi)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        if (part < senders) & (tid < self.cluster):
            copy_s2cluster(
                qn.iterator + part * T * R, qblk.iterator, barq, tid, size=T * R * 2
            )
        if cutlass.const_expr(self.valid):  # noqa: SIM102
            # [valid] the first half of the rope chunk that ends in the next part
            assert qo is not None
            if self.rope_lead(part) & (tid < T):
                pair = load_words(
                    qo.iterator + tid * STAGE + 64 + R - 8, count=2, shared=True
                )
                push_words(halves.iterator + 2 * tid, pair, barq, part + 1)
        wait(barq)
        cute.arch.cluster_arrive_relaxed()  # peers may exit once qblk is delivered
        b_frags = [
            ldmatrix(uk.iterator + (32 * kp + lane) * (L + 8) + 8 * warp, trans=True)
            for kp in range(4)
        ]
        for m in cutlass.range_constexpr(MT):
            out = zeros(4, Float32)
            t = 16 * m + lane % 8 + 8 * ((lane // 8) % 2)
            for ks in cutlass.range_constexpr(8):
                p = 16 * ks + 8 * (lane // 16)
                a = ldmatrix(qn.iterator + ((p // R) * T + t) * R + p % R)
                b = b_frags[ks // 2][2 * (ks % 2) : 2 * (ks % 2) + 2]
                mma_into(out, a, b, bf16=True)
            for h in cutlass.range_constexpr(2):  # rows group, group + 8: 2 bytes
                row = 16 * m + group + 8 * h
                if self.live(tok0 + row):
                    lo, hi = (Float32(BFloat16(out[2 * h + e])) * qsi for e in range(2))
                    col = L * part + 8 * warp + 2 * quarter
                    if cutlass.const_expr(self.stage_out):
                        staged = qo.iterator + row * STAGE + 8 * warp + 2 * quarter
                        store_fp8_pair_shared(staged, lo, hi)
                    else:
                        store_fp8_pair(at(output, tok0 + row, head, col), lo, hi)
        if cutlass.const_expr(self.valid):
            self.publish_valid(args, tok0, tid, qo, halves)
        elif cutlass.const_expr(self.dcp):
            self.publish(args, tok0, tid, qo)
        if cutlass.const_expr(self.late_insert):
            self.insert(args, ss, tok0, j, (j < T) & self.live(tok0 + j), tid)
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.dcp and self.gather):
            self.gathered(args, tid)

    @cute.jit
    def rope_rows(self, args, red, qblk, qo, tok0, tid, qsi):
        """``tiled``'s Q_B row epilogue with RoPE: each thread sums (warp order) and
        rounds an adjacent (even, odd) pair of rows of one request, then nope pairs go
        to ``qblk`` and rope pairs are rotated at the request's position (FP32, as
        ``rotate``) and quantized into the staged query bytes (``stage_out``)."""
        part, _, _ = cute.arch.block_idx()
        T, R = self.tokens, self.rows
        for e in range(tid, T * R // 2, 256):
            pair, ln = e // 32, e % 32
            i = 2 * pair  # elements i, i + 1: the same tile and token, rows r, r + 1
            values = []
            for h in cutlass.range_constexpr(2):
                value = Float32(0)
                for w in cutlass.range_constexpr(8):
                    value += red[w, i + h, ln]
                values.append(Float32(BFloat16(value)))
            tile, c = i // 4, i % 4
            t = 16 * (tile // 3) + ln // 4 + 8 * (c // 2)
            r = 8 * (tile % 3) + 2 * (ln % 4)
            p = R * part + r
            if p < 128:
                qblk[t, r] = BFloat16(values[0])
                qblk[t, r + 1] = BFloat16(values[1])
            else:  # rows past the batch rotate at the last row's position (unused)
                position = args[14][min(tok0 + t, self.batch - 1)]
                pe = rotate(values, position, args[15], (p - 128) // 2)
                qo[t, 64 + r] = fp8_bits(pe[0] * qsi)
                qo[t, 64 + r + 1] = fp8_bits(pe[1] * qsi)

    @cute.jit
    def rope_lead(self, part):
        """[valid] this part's last 8 rope bytes start a 16 B chunk that ends in the
        next part (they are pushed there)."""
        end = self.rows * part + self.rows - 128  # rope byte offset past this CTA
        return (end > 8) & (end < 64) & (end % 16 == 8)

    @cute.jit
    def rope_tail(self, part):
        """[valid] this part's rope rows start in the middle of a 16 B chunk (its
        first half is the previous part's)."""
        o = self.rows * part - 128
        return (o >= 8) & (o % 16 == 8)

    @cute.jit
    def rope_words(self, args, totals, t, o, qsi):
        """[valid] FP8 words of rope rows (byte offset ``o``) of request ``t`` from
        their FP32 Q_B sums: BF16, the drafter's RoPE, the query scale (as the 8 B
        path, per element)."""
        pe = [Float32(BFloat16(v)) for v in totals]
        if cutlass.const_expr(self.rope):
            # Every request lane runs this for a chunk straddling two parts (the halves
            # must all arrive), but only the batch's rows have a position: a padded
            # lane reads the last row's (its words are never stored).
            position = args[14][min(t, self.batch - 1)]
            pe = rotate(pe, position, args[15], o // 2)
        rope = [v * qsi for v in pe]
        return [pack_fp8(tuple(rope[4 * i : 4 * i + 4])) for i in range(len(rope) // 4)]

    @cute.jit
    def scaled(self, words, rr, g):
        """Four BF16-pair ``words`` of one x row scaled as ``tiled``'s RMSNorm writes
        them back: BF16(x * rr * gamma) per element, packed pairs in the same order."""
        out = []
        for w in cutlass.range_constexpr(4):
            x = unpack_bf16(words[w])
            y = mul2(mul2(x, (rr, rr)), (g[2 * w], g[2 * w + 1]))
            out.append(bf16x2(*y))
        return tuple(out)

    @cute.jit
    def x_chunks(self, qkv, token, lane):
        """Volatile loads of request ``token``'s x row slice of this lane (6 x 16 B,
        ``tiled``'s norm layout)."""
        return [
            load_words(at(qkv, token, 256 * i + 8 * lane), count=4, volatile=True)
            for i in range(6)
        ]

    @cute.jit
    def poll(self, qkv, token, lane, words):
        """``words`` (``x_chunks``) re-polled with a short backoff until no BF16
        element is X_SENT.

        Invariant (Rubin B8, POLLED_X): the x columns [0, 1536) of the in-projection
        output are a Lamport row. Armed (X_SENT) at allocation and by the DcpCoreUmma
        grid after its PDL wait (``DcpCoreUmma.arm_x``), i.e. after this kernel, the
        layer's only x reader, completed; written once per layer by the in-projection
        after its own PDL wait (its epilogue never stores X_SENT: NaN results are
        canonical), so the write follows the re-arm by transitive grid completion; read
        here before this kernel's PDL wait. Everything else this kernel reads from
        the step (the KV columns, the slots) is read after its PDL wait."""
        if cutlass.const_expr(self.x_uniform):  # the warp leaves the loop together
            if cutlass.const_expr(POLL_X):
                ready = cute.arch.vote_all_sync(x_ready(words))
                while not ready:
                    sleep_ns(64)
                    words = self.x_chunks(qkv, token, lane)
                    ready = cute.arch.vote_all_sync(x_ready(words))
        else:
            while not x_ready(words):
                sleep_ns(64)
                words = self.x_chunks(qkv, token, lane)
        return words

    @cute.kernel
    def lean_kernel(self, args):
        """``tiled``'s arithmetic (bitwise) for one tile of at most 8 requests (Rubin,
        DCP) with fewer phases after the PDL wait. Before it: Q_B rows and W_UK staged,
        gamma loaded, the cluster barrier. After it: warp w loads request w's x row
        into registers while the Q_B rows become MMA A fragments, normalizes it
        (``tiled``'s chains), one barrier, the split-K Q_B MMAs with x as the B
        operand (N = 8 requests), the FP32 partials summed in warp order by 24 threads
        that push each 16 B nope chunk into every cluster CTA (``st.async``, no CTA or
        cluster barrier) and multicast the rope bytes, then W_UK, whose FP8 bytes are
        multicast from registers (8 B per request and warp), and (``gather``) the
        gathered q poll."""
        qkv, w_qa, w_qb, w_uk, q_scale_inv = (args[i] for i in (0, 1, 3, 4, 7))
        eps = args[9]
        tid, _, _ = cute.arch.thread_idx()
        part, head, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        T, R, L = 8, self.rows, self.latent
        senders = (128 + R - 1) // R
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        # Warp w: request w's row (warps past the batch normalize a copy of the last).
        token = min(warp, self.batch - 1)
        if cutlass.const_expr(self.poll_x):  # x's first poll round is in flight early
            cute.arch.griddepcontrol_launch_dependents()
            words = self.x_chunks(qkv, token, lane)
        RP = R + 4  # request stride of the FP32 partials: conflict-free fragment stores
        xn = alloc(smem, BFloat16, (T, KP), 128)
        uk = alloc(smem, BFloat16, (128, L + 8), 128)  # 144 B rows for ldmatrix
        red = alloc(smem, Float32, (8, T, RP))
        qn = alloc(smem, BFloat16, senders * T * R)  # [sender][request][row]
        ss = alloc(smem, Float32, (8, 2))
        qbs = alloc(smem, BFloat16, (R, KP), 128)  # padded rows as x
        barq = smem.allocate(cutlass.Uint64, 8)
        barb = smem.allocate(cutlass.Uint64, 8)
        if cutlass.const_expr(self.valid):
            halves = alloc(smem, Uint32, (T, 2))  # the previous part's rope half
        if tid == 0:
            cute.arch.mbarrier_init(barq, 1)
            cute.arch.mbarrier_init(barb, 1)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(self.valid):
                # [valid] a part whose rope rows start mid-chunk also receives the
                # previous part's 8 B half of that chunk, per request, on barq
                extra = Int32(0)
                if self.rope_tail(part):
                    extra = Int32(T * 8)
                cute.arch.mbarrier_arrive_and_expect_tx(barq, 128 * T * 2 + extra)
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(barq, 128 * T * 2)
            cute.arch.mbarrier_arrive_and_expect_tx(barb, R * 1536 * 2)
            for r in cutlass.range_constexpr(R):  # one bulk copy per Q_B row
                src = w_qb.iterator + (head * 192 + part * R + r) * 1536
                copy_evict(qbs.iterator + r * KP, src, size=3072, barrier=barb)
        cute.arch.cluster_arrive_relaxed()
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            dst = uk.iterator + p * (L + 8) + j * 8
            copy_evict(dst, at(w_uk, head, p, part * L + j * 8))
        cute.arch.cp_async_commit_group()
        gamma_words = [
            load_words(w_qa.iterator + 256 * i + 8 * lane, count=4, mutable=True)
            for i in range(6)
        ]
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.poll_x):
            words = self.poll(qkv, token, lane, words)
        else:
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            words = [
                load_words(at(qkv, token, 256 * i + 8 * lane), count=4, mutable=True)
                for i in range(6)
            ]
        qsi = q_scale_inv[0]
        box_mc, rank = args[12], args[13]
        # The staged Q_B rows (usually landed by now) -> A fragments while x is in
        # flight: rows group, group + 8 (tile 0), 16 + group (tile 1), tiled's K
        # permutation (warp w: k = 192w..192w+191, 8 values per lane and chunk).
        wait(barb, scope="cta")
        k0 = 6 * warp * 32 + 8 * quarter
        rows = [qbs.iterator + (group + r) * KP + k0 for r in (0, 8, 16)]
        qb = [[smem_words(row + 32 * cc) for row in rows] for cc in range(6)]
        values: list = []
        for i in cutlass.range_constexpr(6):
            for word in words[i]:
                values += list(unpack_bf16(word))
        zero = Float32(0), Float32(0)
        chains = [zero, zero]  # (0, 1), (2, 3): element e feeds chain e % 4
        for e in cutlass.range_constexpr(0, 48, 2):
            x = values[e], values[e + 1]
            chains[e % 4 // 2] = fma2(x, x, chains[e % 4 // 2])
        (c0, c1), (c2, c3) = chains
        rr = query_norm_rsqrt(warp_sum((c0 + c1) + (c2 + c3)), eps)
        gamma: list = []
        for i in cutlass.range_constexpr(6):
            for word in gamma_words[i]:
                gamma += list(unpack_bf16(word))
        for i in cutlass.range_constexpr(6):
            normed = []
            for e in cutlass.range_constexpr(8 * i, 8 * i + 8, 2):
                x = values[e], values[e + 1]
                y = mul2(mul2(x, (rr, rr)), (gamma[e], gamma[e + 1]))
                normed.append(bf16x2(*y))
            row = xn.iterator + warp * KP + 256 * i + 8 * lane
            store_shared_words(row, tuple(normed))
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # W_UK B fragments: warp w's 8 columns; ``valid``: 16 columns per warp (warps
        # 4-7 repeat warps 0-3 and store nothing), so a request's 16 latent bytes leave
        # in one store
        cols = [8 * warp]
        if cutlass.const_expr(self.valid):
            cols = [16 * (warp % 4), 16 * (warp % 4) + 8]
        b_frags = [
            [
                ldmatrix(uk.iterator + (32 * kp + lane) * (L + 8) + col, trans=True)
                for kp in range(4)
            ]
            for col in cols
        ]
        acc = zeros((2, 4), Float32)
        nil = Uint32(0)
        for cc in cutlass.range_constexpr(6):
            k = (6 * warp + cc) * 32 + 8 * quarter
            xw = smem_words(xn.iterator + group * KP + k)
            w0, w8, w16 = qb[cc]
            for s in cutlass.range_constexpr(2):
                b = (xw[2 * s], xw[2 * s + 1])
                a = (w0[2 * s], w8[2 * s], w0[2 * s + 1], w8[2 * s + 1])
                mma_into(acc[0, None], a, b, bf16=True)
                a = (w16[2 * s], nil, w16[2 * s + 1], nil)
                mma_into(acc[1, None], a, b, bf16=True)
        # partials [warp][request][row]; D rows (group, group + 8, 16 + group) x
        # requests (2 quarter, 2 quarter + 1)
        for e in cutlass.range_constexpr(6):
            r = group + 8 * (e // 2)
            red[warp, 2 * quarter + e % 2, r] = acc[e // 4, e % 4]
        cute.arch.sync_threads()
        tail0, tail1 = Uint32(0), Uint32(0)  # [valid] a rope chunk's second half
        if (warp < R // 8) & (lane < T):  # 8 rows (chunk warp) of request lane
            t = lane
            total = [Float32(0)] * 8
            for w in cutlass.range_constexpr(8):
                for h in cutlass.range_constexpr(2):
                    four = load_words(
                        red.iterator + (w * T + t) * RP + 8 * warp + 4 * h,
                        count=4,
                        shared=True,
                        result_type=Float32,
                        word_type="f32",
                    )
                    for j in cutlass.range_constexpr(4):
                        total[4 * h + j] += four[j]
            if cutlass.const_expr(self.valid):
                # [valid] the next 8 rows (clamped in the CTA; used only when this
                # group starts a 16-row rope chunk that the CTA holds whole)
                nxt = min(8 * warp + 8, R - 8)
                total2 = [Float32(0)] * 8
                for w in cutlass.range_constexpr(8):
                    for h in cutlass.range_constexpr(2):
                        four = load_words(
                            red.iterator + (w * T + t) * RP + nxt + 4 * h,
                            count=4,
                            shared=True,
                            result_type=Float32,
                            word_type="f32",
                        )
                        for j in cutlass.range_constexpr(4):
                            total2[4 * h + j] += four[j]
            if R * part + 8 * warp < 128:  # nope rows -> every CTA's qn
                pairs = [bf16x2(total[2 * j], total[2 * j + 1]) for j in range(4)]
                chunk = tuple(bits_float(pair) for pair in pairs)
                dst = qn.iterator + (part * T + t) * R + 8 * warp
                for peer in cutlass.range_constexpr(self.cluster):
                    push_float(dst, chunk, barq, peer)
            elif cutlass.const_expr(self.valid):
                # [valid] rope rows as whole 16 B chunks (mla.qvalid's contract)
                o = R * part + 8 * warp - 128  # rope byte offset of this group
                qrow = (t * self.dcp + rank) * self.heads + head
                if (o % 16 == 0) & (warp + 1 < R // 8):  # the whole chunk is here
                    if self.live(t):
                        whole = self.rope_words(args, total + total2, t, o, qsi)
                        whole[0] = off_sentinel(whole[0])
                        rope_at = box_mc + Int64(qrow) * 576 + 512 + o
                        multicast_words(rope_at, tuple(whole))
                elif o % 16 == 0:  # its first half: to the next part (all requests)
                    lead = self.rope_words(args, total, t, o, qsi)
                    push_words(halves.iterator + 2 * t, tuple(lead), barq, part + 1)
                elif warp == 0:  # its second half: stored after the first lands
                    second = self.rope_words(args, total, t, o, qsi)
                    tail0, tail1 = second[0], second[1]
            elif self.live(t):  # rope rows: 8 FP8 bytes straight to every q mailbox
                pe = [Float32(BFloat16(v)) for v in total]
                if cutlass.const_expr(self.rope):
                    pair0 = (R * part + 8 * warp - 128) // 2
                    pe = rotate(pe, args[14][t], args[15], pair0)
                rope = [v * qsi for v in pe]
                staged = (pack_fp8(tuple(rope[0:4])), pack_fp8(tuple(rope[4:8])))
                qrow = (t * self.dcp + rank) * self.heads + head
                byte = 512 + R * part + 8 * warp - 128
                multicast_words(box_mc + Int64(qrow) * 576 + byte, staged)
        wait(barq)
        cute.arch.cluster_arrive_relaxed()  # every push into this CTA has landed
        if cutlass.const_expr(self.valid):  # noqa: SIM102
            # [valid] the rope chunk whose first half came from the previous part
            if (warp == 0) & (lane < T) & self.rope_tail(part) & self.live(lane):
                pushed = load_words(halves.iterator + 2 * lane, count=2, shared=True)
                joined = (off_sentinel(pushed[0]), pushed[1], tail0, tail1)
                trow = (lane * self.dcp + rank) * self.heads + head
                tail_at = box_mc + Int64(trow) * 576 + 512 + R * part - 128 - 8
                multicast_words(tail_at, joined)
        outs = [zeros(4, Float32) for _ in cols]
        for ks in cutlass.range_constexpr(8):  # requests lane % 8; rows 8-15 are zero
            p = 16 * ks + 8 * ((lane // 8) % 2)
            a = ldmatrix(qn.iterator + ((p // R) * T + lane % 8) * R + p % R, count=2)
            for c in cutlass.range_constexpr(len(cols)):
                b = b_frags[c][ks // 2][2 * (ks % 2) : 2 * (ks % 2) + 2]
                mma_into(outs[c], (a[0], nil, a[1], nil), b, bf16=True)
        # Latent bytes of request group, columns 8 warp .. (``valid``: 16 (warp % 4)
        # ..): each tile's quad of four FP8 pairs -> lane quarter 0, one 8 B (16 B)
        # multicast store.
        latent: list = []
        for out in outs:
            lo, hi = (Float32(BFloat16(out[e])) * qsi for e in range(2))
            pair = pack_fp8((lo, hi, Float32(0), Float32(0))) & Uint32(0xFFFF)
            pair |= Uint32(cute.arch.shuffle_sync_down(pair, 1)) << 16
            other = Uint32(cute.arch.shuffle_sync_down(pair, 2))
            latent += [pair, other]
        if cutlass.const_expr(self.valid):
            # mla.qvalid's contract: ONE 16 B store per chunk, first word never the
            # validity sentinel (never split it back into 8 B halves)
            latent[0] = off_sentinel(latent[0])
            if (quarter == 0) & self.live(group) & (warp < 4):
                lrow = (group * self.dcp + rank) * self.heads + head
                latent_at = box_mc + Int64(lrow) * 576 + L * part + 16 * warp
                multicast_words(latent_at, tuple(latent))
        elif (quarter == 0) & self.live(group):
            qrow = (group * self.dcp + rank) * self.heads + head
            byte = L * part + 8 * warp
            multicast_words(box_mc + Int64(qrow) * 576 + byte, tuple(latent))
        j = head * self.cluster + part
        if cutlass.const_expr(self.poll_x):
            cute.arch.griddepcontrol_wait()  # the KV columns of x's rows, the slots
        self.insert(args, ss, 0, j, (j < self.tokens) & self.live(j), tid)
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.gather):
            self.gathered(args, tid)
