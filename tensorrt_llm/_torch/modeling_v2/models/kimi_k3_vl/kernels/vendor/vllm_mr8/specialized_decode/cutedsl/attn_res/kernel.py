# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seven-CTA cluster AttnRes: one cluster per token, 1024 hidden columns per CTA.

The delta is a tensor (read after the PDL wait) or ``slots`` Lamport mailbox slots
summed in slot order, rounded to BF16 and re-armed. The final boundary rounds the
mixture to BF16 before its RMSNorm. Prefix and saved blocks must be complete at launch.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from ..common.mailbox import poll, poll_warp, rearm
from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    bf16,
    bf16x2,
    dot,
    fma,
    group_barrier,
    launch,
    load_words,
    multicast_words,
    no_neg_zero,
    push_float,
    scatter_reduce,
    store_words,
    wait,
    zeros,
)


@asm()
def store_byte(pointer, value):
    """TensorRT-LLM local (``ClusterAttnRes.mx_out``): the low byte of ``value``."""
    return "st.global.u8 [$0], $1;", "l,r,~{memory}", (pointer, Int32(value))


def _mx_ops():
    """TensorRT-LLM local (``ClusterAttnRes.mx_out``): the K3 low-M MXFP8 GEMM's own amax /
    exponent / E4M3 conversion (``k3_fp8w_lowm_mxfp8``), so the stored operands are bitwise those
    of its in-kernel quantization of the BF16 output."""
    from ......k3r_gemm import k3_fp8w_lowm_mxfp8_ull as k3_fp8w_lowm_mxfp8

    return k3_fp8w_lowm_mxfp8


@cute.jit
def round_pair(t, row: cutlass.Constexpr, j: cutlass.Constexpr):
    word = bf16x2(t[row, 2 * j], t[row, 2 * j + 1])
    t[row, 2 * j], t[row, 2 * j + 1] = bf16(word, 0), bf16(word, 1)


@cute.jit
def unpack(words, out, row: cutlass.Constexpr):
    for i in cutlass.range_constexpr(8):
        out[row, i] = bf16(words[i // 2], i % 2)


def block_ptr(blocks, token, block, col):
    """``&blocks[token, block, col]`` from the tensor's strides."""
    return blocks.iterator + token * blocks.stride[0] + block * blocks.stride[1] + col


@cute.jit
def rstd(sum_sq, eps):
    """Reciprocal RMS of a 7168-wide row from its sum of squares."""
    return cute.math.rsqrt(fma(sum_sq, Float32(1 / 7168), eps), fastmath=True)


@cute.jit
def dot8(a, ar: cutlass.Constexpr, b, br: cutlass.Constexpr):
    return dot((a[ar, i] for i in range(8)), (b[br, i] for i in range(8)), Float32(0))


@cute.jit
def arm_exchanges(bars, counts):
    """Thread 0 arms barrier ``r`` for ``counts[r]`` FP32 pushes from each of 7 CTAs."""
    if cute.arch.thread_idx()[0] == 0:
        for r, count in enumerate(counts):
            if cutlass.const_expr(count > 0):
                cute.arch.mbarrier_init(bars.iterator + r, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + r, 28 * count)
        cute.arch.mbarrier_init_fence()


# Shared memory of one ClusterAttnRes CTA (``setup``'s buffers). The in-projections
# that follow an AttnRes (KdaHeadWide / KdaHeadVerify, SplitKTmemGemm, SplitKTmemGemm64)
# load their weights beside its CTAs (carve-out 100): BESIDE_SMEM is the largest
# request (a KB multiple) that fits beside one AttnRes CTA in an SM's 228 KB, with the
# driver's 1 KB per CTA for each; they assert it where they size themselves. One byte
# more and their clusters wait for the AttnRes CTAs to exit (KdaHeadVerify at 225 KB:
# +1.5 us per KDA layer at 16 ranks).
SMEM_BYTES = 64 * 4 + 3 * 112 * 4 + 3 * 16 * 4 + 3 * 8
BESIDE_SMEM = (228 * 1024 - 2 * 1024 - SMEM_BYTES) // 1024 * 1024


class ClusterAttnRes:
    """``sanitize``: multi-slot sums map -0 to +0 after rounding, as the framework
    all-reduce they replace does (the Rubin dense ring)."""

    def __init__(
        self, nb, *, slots=1, has_delta=True, write=-1, final=False, sanitize=False,
        stores_first=False, spread_push=False, poll_warp=False, min_blocks=0,
        out_mc=False, aux_mix=False, mx_out=False,
    ):  # fmt: skip
        # TensorRT-LLM local change (K3 chain -> the MLA layer after a run): ``mx_out``: the
        # normalized output row is also stored in MXFP8 (E4M3 ``xq [M, 7168]`` and one UE8M0
        # exponent per 1x32 group, ``xs [M, 224]``; two trailing Int64 addresses), the operands
        # of the MLA layer's low-M FP8 projection. Off: the vLLM MR8 kernel unchanged.
        if mx_out and out_mc:
            raise ValueError("mx_out with out_mc")
        self.mx_out = mx_out
        # TensorRT-LLM local change (K3 chain, DSpark taps): ``aux_mix``: the aux stream
        # gets the pre-norm mixture (the normalized output before its RMSNorm) instead
        # of the updated prefix row, the tap value of TensorRT-LLM's default
        # KIMI_K3_AUX_ATTN_RES_STREAM=1. Off: the vLLM MR8 kernel unchanged.
        self.aux_mix = aux_mix
        # ``out_mc`` (row-sharded AttnRes, M >= 256, ``attn_res.sharded``): the output
        # row (and the aux rows) go to every rank's Lamport all-gather row row0 + token
        # at the trailing multicast address (the aux args are multicast addresses too)
        # instead of ``output`` (-0 halves made +0: the sentinel's word).
        self.out_mc = out_mc
        # ``poll_warp`` (Rubin): the delta's Lamport poll exits by a warp vote
        # (``common.mailbox.poll_warp``), so the exchange's shuffles that follow do not
        # wait for per-lane poll loops to reconverge. Same words.
        self.poll_warp = poll_warp
        # ``min_blocks``: launch bound (CTAs per SM) that caps the registers per thread
        # at 64 K / (128 x min_blocks), so a variant keeps its register allocation and
        # the in-projection CTAs that load beside it stay co-resident. Same code.
        self.min_blocks = min_blocks
        self.nb, self.slots, self.has_delta = nb, slots, has_delta
        self.write, self.final, self.sanitize = write, final, sanitize
        self.stores_first = stores_first
        # ``spread_push``: the exchange's 7 x count DSMEM pushes come from 7 x count
        # threads (one each) instead of count threads pushing to every peer.
        self.spread_push = spread_push
        self.npre = 2 * nb
        self.npost = (2 if final else 4) if nb else 1
        self.kpre = 1 << max(self.npre - 1, 0).bit_length()
        self.kpost = 1 << max(self.npost - 1, 0).bit_length()

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(self.min_blocks > 0):
            grid, bound = (7, args[1].shape[0], 1), self.min_blocks
            launch(self.kernel, args, grid, 128, 7, min_blocks_per_mp=bound)
        else:
            launch(self.kernel, args, (7, args[1].shape[0], 1), 128, 7)

    @cute.jit
    def exchange(
        self, values, count: cutlass.Constexpr, kpad: cutlass.Constexpr, bufs, r, ids
    ):
        """Cluster-sum ``count`` per-thread partials into round ``r``'s total row."""
        wb, peer_buf, total, bars = bufs
        peer_buf, total, bar = peer_buf[r, None], total[r, None], bars.iterator + r
        rank, _, tid = ids
        lane, warp = tid % 32, tid // 32
        reduced, index = scatter_reduce(values, lane, kpad)
        if (lane & (32 // kpad - 1)) == 0 and index < count:
            wb[warp * 16 + index] = reduced
        group_barrier(0)
        if cutlass.const_expr(self.spread_push):
            if tid < 7 * count:
                i, peer = tid % count, tid // count
                value = (wb[i] + wb[16 + i]) + (wb[32 + i] + wb[48 + i])
                push_float(peer_buf.iterator + rank * 16 + i, value, bar, peer)
        elif tid < count:
            value = (wb[tid] + wb[16 + tid]) + (wb[32 + tid] + wb[48 + tid])
            for peer in cutlass.range_constexpr(7):
                push_float(peer_buf.iterator + rank * 16 + tid, value, bar, Int32(peer))
        wait(bar)
        if tid < count:
            sums = [Float32(0), Float32(0)]
            for peer in cutlass.range_constexpr(7):
                sums[peer % 2] += peer_buf[peer * 16 + tid]
            total[tid] = sums[0] + sums[1]
        group_barrier(0)

    @cute.kernel
    def kernel(self, args):
        delta, prefix, blocks, norm, query, out_norm, output, eps, out_eps = args[0:9]
        rank, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        bufs = self.setup(cutlass.utils.SmemAllocator())
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()
        state = self.state()
        rows, ids = (prefix, blocks, norm, query, out_norm), (rank, token, tid)
        self.pre(state, bufs, rows, eps, ids)
        extra = args[9:]
        mx = None
        if cutlass.const_expr(self.mx_out):
            mx, extra = extra[-2:], extra[:-2]  # (xq, xs addresses)
        if cutlass.const_expr(self.out_mc):
            output, extra = extra[-2:], extra[:-2]  # (multicast address, row0)
        self.post(state, bufs, delta, rows, output, eps, out_eps, ids, extra, mx)
        cute.arch.griddepcontrol_wait()

    def state(self):
        """Carried from pre() to post(); the rest reloads so a GEMM can run between."""
        return cute.make_rmem_tensor((1, 8), Float32), cute.make_rmem_tensor(2, Float32)

    @cute.jit
    def setup(self, smem):  # SMEM_BYTES
        warp_buf = alloc(smem, Float32, 64)
        peer_buf = alloc(smem, Float32, (3, 112))
        total = alloc(smem, Float32, (3, 16))
        bars = alloc(smem, Uint64, 3, 8)
        arm_exchanges(bars, (self.npre, self.npost, int(self.final)))
        return warp_buf, peer_buf, total, bars

    @cute.jit
    def pre(self, state, bufs, rows, eps: Float32, ids):
        """Inputs complete at launch, then the saved-block statistics exchange."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        warp_buf, peer_buf, total, bars = bufs
        mixture, scal = state
        # Rows: saved blocks, combined weight, gamma, old prefix, new prefix.
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        col = (rank * 128 + tid) * 8
        W = self.nb
        for source in cutlass.range_constexpr(self.nb):
            words = load_words(
                block_ptr(blocks, token, source, col), count=4, mutable=True
            )
            unpack(words, v, source)
        self.weights(v, norm, query, col)
        cute.arch.cluster_wait()
        maximum, exp_sum = Float32(0), Float32(0)
        mixture.fill(0)
        if cutlass.const_expr(self.nb > 0):
            values = zeros(self.kpre, Float32)
            for source in cutlass.range_constexpr(self.nb):
                values[source] = dot8(v, source, v, source)
                values[self.nb + source] = dot8(v, source, v, W)
            self.exchange(values, self.npre, self.kpre, bufs, 0, ids)
            logits = cute.make_rmem_tensor(self.nb, Float32)
            for b in cutlass.range_constexpr(self.nb):
                logits[b] = total[0, self.nb + b] * rstd(total[0, b], eps)
            maximum = logits[0]
            for source in cutlass.range_constexpr(1, self.nb):
                maximum = cute.arch.fmax(maximum, logits[source])
            for source in cutlass.range_constexpr(self.nb):
                e = cute.math.exp(logits[source] - maximum, fastmath=True)
                exp_sum += e
                for i in cutlass.range_constexpr(8):
                    mixture[0, i] = fma(e, v[source, i], mixture[0, i])
        scal[0], scal[1] = maximum, exp_sum

    @cute.jit
    def weights(self, v, norm, query, col):
        if cutlass.const_expr(self.nb > 0):
            nw = load_words(norm.iterator + col, count=4)
            qw = load_words(query.iterator + col, count=4)
            for i in cutlass.range_constexpr(8):
                v[self.nb, i] = bf16(nw[i // 2], i % 2) * bf16(qw[i // 2], i % 2)

    @cute.jit
    def store_rows(self, row, prefix, blocks, aux, token, col, row0=None):
        """The updated prefix row (and its ``aux`` copies and saved block)."""
        store_words(at(prefix, token, col), row)
        if cutlass.const_expr(not self.aux_mix):
            self.store_aux(row, aux, token, col, row0)
        if cutlass.const_expr(self.write >= 0):
            store_words(block_ptr(blocks, token, self.write, col), row)

    @cute.jit
    def store_aux(self, row, aux, token, col, row0=None):
        """``row`` (four BF16x2 words) into the ``aux`` streams."""
        for stream_out in aux:
            if cutlass.const_expr(self.out_mc):
                offset = Int64((row0 + token) * 7168 + col) * 2
                multicast_words(stream_out + offset, tuple(no_neg_zero(w) for w in row))
            else:
                store_words(at(stream_out, token, col), row)

    @cute.jit
    def post(
        self, state, bufs, delta, rows, output, eps: Float32, out_eps: Float32, ids, aux,
        mx=None,
    ):  # fmt: skip
        """Add the delta, update prefix (and its ``aux`` copy) and saved block, write
        the normalized output (``mx``: also its MXFP8 form, ``mx_out``)."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        warp_buf, peer_buf, total, bars = bufs
        mixture, scal = state
        maximum, exp_sum = scal[0], scal[1]
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        col = (rank * 128 + tid) * 8
        W, G, OLD, NEW = self.nb, self.nb + 1, self.nb + 2, self.nb + 3
        self.weights(v, norm, query, col)
        unpack(load_words(out_norm.iterator + col, count=4), v, G)
        if cutlass.const_expr(self.has_delta):
            unpack(load_words(at(prefix, token, col), count=4, mutable=True), v, OLD)
        cur = cute.make_rmem_tensor((1, 8), Float32)
        if cutlass.const_expr(self.slots == 0):
            cute.arch.griddepcontrol_wait()
            unpack(load_words(at(delta, token, col), count=4, mutable=True), cur, 0)
        else:
            cur.fill(0)
            base = delta.iterator + token * 7168 + col
            if cutlass.const_expr(self.poll_warp):
                got = poll_warp(base, delta.shape[1] * 7168, self.slots)
            else:
                got = poll(base, delta.shape[1] * 7168, self.slots)
            for slot in cutlass.range_constexpr(self.slots):
                rearm(base + slot * delta.shape[1] * 7168)
                part = cute.make_rmem_tensor((1, 8), Float32)
                unpack(tuple(got[slot, k] for k in range(4)), part, 0)
                for i in cutlass.range_constexpr(8):
                    cur[0, i] = cur[0, i] + part[0, i]
            if cutlass.const_expr(self.slots > 1):
                for j in cutlass.range_constexpr(4):  # all-reduce output boundary
                    round_pair(cur, 0, j)
            if cutlass.const_expr(self.slots > 1 and self.sanitize):
                # No negative zero, as the framework all-reduce returns (-0 + 0 = +0).
                for i in cutlass.range_constexpr(8):
                    cur[0, i] = add_rn(cur[0, i], Float32(0))
        words = cute.make_rmem_tensor(4, Uint32)
        for j in cutlass.range_constexpr(4):
            lo, hi = cur[0, 2 * j], cur[0, 2 * j + 1]
            if cutlass.const_expr(self.has_delta):
                lo, hi = v[OLD, 2 * j] + lo, v[OLD, 2 * j + 1] + hi
            words[j] = bf16x2(lo, hi)
        unpack(words, v, NEW)
        values = zeros(self.kpost, Float32)
        values[0] = dot8(v, NEW, v, NEW)
        if cutlass.const_expr(self.nb > 0):
            values[1] = dot8(v, NEW, v, W)
            if cutlass.const_expr(not self.final):
                values[2] = dot8(mixture, 0, v, NEW)
                values[3] = dot8(mixture, 0, mixture, 0)
        row = (words[0], words[1], words[2], words[3])
        row0 = output[1] if cutlass.const_expr(self.out_mc) else None
        if cutlass.const_expr(self.stores_first):
            self.store_rows(row, prefix, blocks, aux, token, col, row0)
        self.exchange(values, self.npost, self.kpost, bufs, 1, ids)
        if cutlass.const_expr(not self.stores_first):
            self.store_rows(row, prefix, blocks, aux, token, col, row0)
        result = cute.make_rmem_tensor((1, 8), Float32)
        if cutlass.const_expr(self.nb == 0):
            if cutlass.const_expr(self.aux_mix):
                self.store_aux(row, aux, token, col, row0)
            scale = rstd(total[1, 0], out_eps)
            for i in cutlass.range_constexpr(8):
                result[0, i] = v[NEW, i] * scale * v[G, i]
        else:
            ss, dot = total[1, 0], total[1, 1]
            logit = dot * rstd(ss, eps)
            top = cute.arch.fmax(maximum, logit)
            alpha = cute.math.exp(maximum - top, fastmath=True)
            ep = cute.math.exp(logit - top, fastmath=True)
            denominator = fma(alpha, exp_sum, ep)
            if cutlass.const_expr(self.final):
                inverse = Float32(1) / denominator
                for j in cutlass.range_constexpr(4):
                    for i in (2 * j, 2 * j + 1):
                        result[0, i] = fma(
                            alpha * inverse, mixture[0, i], ep * inverse * v[NEW, i]
                        )
                    round_pair(result, 0, j)
                if cutlass.const_expr(self.aux_mix):
                    mix = tuple(bf16x2(result[0, 2 * j], result[0, 2 * j + 1]) for j in range(4))
                    self.store_aux(mix, aux, token, col, row0)
                fin = cute.make_rmem_tensor(1, Float32)
                fin[0] = dot8(result, 0, result, 0)
                self.exchange(fin, 1, 1, bufs, 2, ids)
                scale = rstd(total[2, 0], out_eps)
                for i in cutlass.range_constexpr(8):
                    result[0, i] = result[0, i] * scale * v[G, i]
            else:
                a2, t3 = alpha * alpha, total[1, 3]
                ae2 = Float32(2) * alpha * ep
                quad = fma(a2, t3, fma(ae2, total[1, 2], ep * ep * ss))
                scale = rstd(quad, out_eps * denominator * denominator)
                if cutlass.const_expr(self.aux_mix):
                    inverse = Float32(1) / denominator
                    ma, mp = alpha * inverse, ep * inverse
                    mix = tuple(
                        bf16x2(
                            fma(ma, mixture[0, 2 * j], mp * v[NEW, 2 * j]),
                            fma(ma, mixture[0, 2 * j + 1], mp * v[NEW, 2 * j + 1]),
                        )
                        for j in range(4)
                    )
                    self.store_aux(mix, aux, token, col, row0)
                ca, cp = alpha * scale, ep * scale
                for i in cutlass.range_constexpr(8):
                    result[0, i] = fma(ca, mixture[0, i], cp * v[NEW, i]) * v[G, i]
        for j in cutlass.range_constexpr(4):
            words[j] = bf16x2(result[0, 2 * j], result[0, 2 * j + 1])
        if cutlass.const_expr(self.out_mc):
            mc, row0 = output
            offset = Int64((row0 + token) * 7168 + col) * 2
            multicast_words(mc + offset, tuple(no_neg_zero(words[j]) for j in range(4)))
        else:
            store_words(at(output, token, col), tuple(words[j] for j in range(4)))
        if cutlass.const_expr(self.mx_out):
            self.store_mx(mx, token, col, tid, words)

    @cute.jit
    def store_mx(self, mx, token, col, tid, words):
        """TensorRT-LLM local (``mx_out``): the MXFP8 form of this thread's eight output values;
        four consecutive threads hold one 1x32 group (``col`` = 8 x thread)."""
        ops = _mx_ops()
        amax = ops.amax_bf16x8_words(words[0], words[1], words[2], words[3])
        amax = ops.amax_pair(amax, cute.arch.shuffle_sync_bfly(amax, 1))
        amax = ops.amax_pair(amax, cute.arch.shuffle_sync_bfly(amax, 2))
        exponent = ops.ue8m0_of_amax(amax)
        inv = ops.bf16x2_inv_scale(exponent)
        q0, q1 = ops.quant_bf16x8_to_e4m3(words[0], words[1], words[2], words[3], inv)
        xq, xs = mx
        store_words(xq + Int64(token * 7168 + col), (q0, q1))
        if tid % 4 == 0:
            store_byte(xs + Int64(token * 224 + col // 32), exponent)
