# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA input projection and spec verify recurrence in one kernel: one request x q verify
tokens (B1, q <= 8) on ``KdaHeadWide``'s 16-CTA clusters (Rubin TP16).

The projection is ``KdaHeadWide``'s: 128-row Q / K / V / gate tiles split over K by
four CTAs, the weights streamed (and staged in tensor memory) before the PDL wait, the
FP32 partials of verify token b pushed to CTAs 2 b and 2 b + 1, which sum them in rank
order and round to BF16 (bitwise the split-K projection's (128, 8, 4) tactic). The
recurrence then needs every token of the request in one CTA, so a second DSMEM hop
redistributes the rounded inputs: every CTA sends its token's Q / K channels (half of
them per CTA of the pair) to the eight KDA CTAs, its V rows to the KDA CTA owning them,
and the output gate to the token's owner. KDA CTA r (r < 8) owns state rows
[16 r, 16 r + 16) and runs ``KdaSplit``'s verify recurrence with its B1 schedule
(lanes 8, eight CTAs per head): the convolution, normalization, decays (the gate warp
projects F_B for its 16 features from ``SignalGemm``'s F_A rows, polled as in
``KdaHeadWide``), token loop, checkpoints and outputs are ``KdaSplit``'s, so given the
same projected inputs the outputs and checkpoints are bit for bit ``KdaSplit``'s. CTA t
applies the gated RMS norm of token t; with ``polled_out`` its output is the polled
buffer of ``self.vkda.head.POLLED_OUT`` (canonical stores; warp 1 re-arms the previous
KDA layer's buffer after its PDL wait)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.arch import store_async_dsmem

from ..attn_res.kernel import BESIDE_SMEM
from ..common.primitives import (
    at,
    float_bits,
    fma,
    load_f32x4,
    load_words,
    mma_into,
    named_barrier,
    push_float,
    sigmoid,
    wait,
    warp_sum,
    zeros,
)
from ..gemm.splitk import _smem
from .head import (
    CLUSTER,
    INPUT_BYTES,
    SECTIONS,
    KdaHeadWide,
    load_bf16x4,
    store_bf16_canonical,
)
from .kernel import KdaSplit, sigmoids

KDA_CTAS = 8  # CTAs that run the recurrence (KdaSplit's B1 verify: cluster 8)
ROWS = 128 // KDA_CTAS  # state rows per KDA CTA


class KdaHeadVerify(KdaHeadWide):
    """``KdaHeadWide`` for one request x ``tokens`` verify tokens; extra operand:
    ``accepted`` [1] after F_B; ``slots`` is [1, >= tokens] (vLLM's spec slots)."""

    def __init__(
        self, heads, *, tokens=8, stages=10, cta_k=64, tmem_tiles=12, early_loads=17,
        bf16_state=True, has_bias=False, hidden=7168,
    ):  # fmt: skip
        super().__init__(
            heads, stages=stages, cta_k=cta_k, tmem_tiles=tmem_tiles,
            early_loads=early_loads, bf16_state=bf16_state, has_bias=has_bias,
            hidden=hidden,
        )  # fmt: skip
        assert tokens == 8, "B1 x q 8: every CTA pair of the cluster holds one token"
        self.tokens = tokens
        # KdaSplit's B1 verify geometry (schedule 1.8.x.8): 16 rows, 8 lanes per row.
        self.vkda = KdaSplit(heads, 1, 0, has_bias, (1, 8, 0, KDA_CTAS), 1, bf16_state)
        assert self.vkda.rows == ROWS and self.vkda.passes == 1
        T = tokens
        extra = 2 * T * 128 * 2 + T * ROWS * 2 + 128 * 2  # hop 2: raw Q / K, V, gate
        extra += 448 * 2  # this CTA's rounded token inputs (Q | K | V 64 | gate)
        extra += 3 * T * 128 * 4 + T * ROWS * 4 + 128 * 4  # q, k, decays, v, out
        extra += 8 * 4 + 8 * 4 + 5 * 8  # norms (one token), betas, barriers
        extra += T * ROWS * 4 + T * 4  # outputs (pushed after the loop), slots
        # KdaHeadWide's bytes without its q = 1 buffers (q, k, decay, v, out, norms,
        # betas, barriers), which this kernel allocates token-major instead (``extra``),
        # plus KdaHeadWide's alignment slack.
        wide_only = (3 * 128 + 64 + 128) * 4 + 8 * 4 + 4 * 4 + 3 * 8
        used = self.smem_used - wide_only + extra
        self.smem_bytes = (used + 2560 + 1023) // 1024 * 1024
        assert self.smem_bytes <= BESIDE_SMEM  # beside an AttnRes CTA

    @cute.experimental.jit
    def __call__(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, accepted: cute.Tensor,
        lower_bound: Float32, scale: Float32, eps: Float32, counter: cutlass.Int64,
        target: Int32, out_rearm: cutlass.Int64, stream,
    ):  # fmt: skip
        self.kernel(
            mA, mB, a_map, conv_w, bias, conv_state, fa_beta, a_log, dt_bias, slots,
            state, output, norm_w, f_b, accepted, lower_bound, scale, eps, counter,
            target, out_rearm,
        ).launch(
            grid=(CLUSTER, self.heads, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )  # fmt: skip

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, a_map: cutlass.Int64,
        conv_w: cute.Tensor, bias: cute.Tensor, conv_state: cute.Tensor,
        fa_beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
        slots: cute.Tensor, state: cute.Tensor, output: cute.Tensor,
        norm_w: cute.Tensor, f_b: cute.Tensor, accepted: cute.Tensor,
        lower_bound: Float32, scale: Float32, eps: Float32, counter: cutlass.Int64,
        target: Int32, out_rearm: cutlass.Int64,
    ):  # fmt: skip
        T: cutlass.Constexpr = self.tokens
        stages, staged = self.num_ab_stage, self.tmem_tiles > 0
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        rank, head, _ = cute.arch.block_idx()
        section, split_rank = rank % SECTIONS, rank // SECTIONS
        row0 = section * self.dim + head * 128
        token, partition = rank // 2, rank % 2  # hop 1: this CTA's token inputs
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        a_shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, self.k_tiles if staged else stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*a_shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*b_shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        b_full = _smem(cutlass.Int64, self.k_tiles if staged else 1, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        partial = _smem(Float32, None, 128, tile_layout)
        x_in = _smem(Float32, SECTIONS * INPUT_BYTES // 2, 128)
        # hop 2 (BF16): raw Q, K [T, 128], this KDA CTA's V rows [T, 16], the owned
        # token's output gate [128]; this CTA's rounded token inputs [448]
        raw_qk = _smem(BFloat16, 2 * T * 128, 128)
        raw_v = _smem(BFloat16, T * ROWS, 16)
        raw_g = _smem(BFloat16, 128, 16)
        mine = _smem(BFloat16, 448, 16)
        shared_q = _smem(Float32, T * 128, 16)
        shared_k = _smem(Float32, T * 128, 16)
        shared_decay = _smem(Float32, T * 128, 16)
        shared_v = _smem(Float32, T * ROWS, 16)
        shared_o = _smem(Float32, 128, 16)
        local_o = _smem(Float32, T * ROWS, 16)
        slot_rows = _smem(Int32, T, 16)
        reductions = _smem(Float32, 8, 16)
        shared_beta = _smem(Float32, 8, 16)
        bar_in = _smem(cutlass.Int64, 1, 8).iterator
        bar_hop = _smem(cutlass.Int64, 1, 8).iterator
        bar_gate = _smem(cutlass.Int64, 1, 8).iterator
        bar_out = _smem(cutlass.Int64, 1, 8).iterator
        bar_qk = _smem(cutlass.Int64, 1, 8).iterator
        kda_cta = rank < KDA_CTAS

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 1 if staged else 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                if cutlass.const_expr(staged):
                    for i in range(self.k_tiles):
                        cute.arch.mbarrier_init(b_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
        if tid == 128:
            cute.arch.mbarrier_init(bar_in, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_in, SECTIONS * INPUT_BYTES * 2)
            if kda_cta:
                # Q / K: 16 CTAs x 64 channels x 2 x T; V: T x 16 rows; gate: 128
                hop = 2 * T * 128 * 2 + T * ROWS * 2 + 256
                cute.arch.mbarrier_init(bar_hop, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bar_hop, hop)
                cute.arch.mbarrier_init(bar_gate, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_gate, KDA_CTAS * ROWS * T * 4 + T * 4
                )
                cute.arch.mbarrier_init(bar_out, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bar_out, 512)
                # every KDA CTA's normalized q / k of its token (DISTRIBUTED_QK)
                cute.arch.mbarrier_init(bar_qk, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bar_qk, KDA_CTAS * 2 * 128 * 4)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        tiles = (split_rank * k_tile_count, k_tile_count)
        offset = accepted[0] - 1
        slot = slots[0, offset]
        if warp_idx < 3:
            cute.arch.cluster_wait()
            if warp_idx == 0:
                self.weight_warp(bar_full, bar_empty, sA, a_map, row0, *tiles)
            elif warp_idx == 1:
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                if cutlass.const_expr(staged):
                    self.resident_b(b_full, b_tile, sB, b_map, tiles[0], 0)
                else:
                    self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
                if cutlass.const_expr(self.polled_out):  # after the PDL wait above
                    self.rearm_out(out_rearm, output.shape[0], head * CLUSTER + rank)
            elif cutlass.const_expr(staged):
                self.staged_mma_warp(
                    bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc, sA, sB,
                    tmem_base_ptr,
                )  # fmt: skip
            else:
                k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
                self.mma_warp(
                    bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA,
                    sB, tmem_base_ptr, acc_layout, k_blocks, tiles[1],
                )  # fmt: skip
            cute.arch.cluster_arrive_relaxed()
        elif warp_idx == 3:
            gate_args = (fa_beta, a_log, dt_bias, f_b, lower_bound, counter, target)
            self.verify_gate(
                gate_args, (head, rank, slot), shared_decay, shared_beta, bar_gate
            )
            cute.arch.cluster_arrive_relaxed()
        else:
            kt = tid - 128
            tensors = (conv_w, bias, conv_state, state, output, norm_w)
            coords = (head, rank, slot, offset)
            early = self.verify_early(kt, tensors, coords, slots)
            cute.arch.cluster_wait()
            if cutlass.const_expr(staged):  # the other B_ISSUERS - 1 warps of x's tiles
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (0, None, 0))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                self.resident_b(b_full, b_tile, sB, b_map, tiles[0], warp_idx - 3)
            self.push_partials(
                kt, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, partial,
                section, split_rank, x_in, bar_in,
            )  # fmt: skip
            wait(bar_in)
            inputs, gate_in = self.read_inputs(kt, x_in)
            # this CTA's rounded token inputs, then hop 2
            mine[kt] = inputs[0]
            mine[128 + kt] = inputs[1]
            if kt < 64:
                mine[256 + kt] = inputs[2]
            mine[320 + kt] = gate_in
            named_barrier(1, 128)
            self.hop(kt, token, partition, mine, raw_qk, raw_v, raw_g, bar_hop)
            cute.arch.cluster_arrive_relaxed()  # hop 1 done; hop 2 issued
            if kda_cta:
                rows = (
                    shared_q,
                    shared_k,
                    shared_decay,
                    shared_v,
                    shared_o,
                    reductions,
                    local_o,
                    slot_rows,
                )
                raws = (raw_qk, raw_v, raw_g, bar_hop)
                bars = (bar_gate, bar_out, shared_beta, bar_qk)
                self.verify_main(
                    kt, early, raws, rows, bars, tensors, coords, slots, scale, eps
                )
        cute.arch.cluster_wait()  # no CTA leaves while a peer may write it

    @cute.jit
    def hop(self, kt, token, partition, mine, raw_qk, raw_v, raw_g, bar_hop):
        """This CTA's rounded inputs of ``token`` to the KDA CTAs, 16 bytes per thread:
        Q / K channels [64 p, 64 p + 64) to all eight, V rows [64 p + 16 j, + 16) to
        KDA CTA 4 p + j, the output gate channels [64 p, + 64) to CTA ``token``."""
        T: cutlass.Constexpr = self.tokens
        smem = cute.AddressSpace.smem
        dest, c = kt // 16, kt % 16
        sec, sub = c // 8, c % 8
        channel = 64 * partition + 8 * sub
        src = mine.iterator + 128 * sec + channel
        dst = raw_qk.iterator + (sec * T + token) * 128 + channel
        words = load_words(src, count=4, shared=True)
        dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
        store_async_dsmem(dst, words, bar_hop, dest)
        if kt < 8:  # V: KDA CTA 4 p + j, rows 8 h .. 8 h + 7 of its 16
            j, h = kt // 2, kt % 2
            src = mine.iterator + 256 + 16 * j + 8 * h
            dst = raw_v.iterator + token * ROWS + 8 * h
            words = load_words(src, count=4, shared=True)
            dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
            store_async_dsmem(dst, words, bar_hop, 4 * partition + j)
        elif kt < 16:  # output gate: the token's owner
            sub = kt - 8
            channel = 64 * partition + 8 * sub
            src = mine.iterator + 320 + channel
            dst = raw_g.iterator + channel
            words = load_words(src, count=4, shared=True)
            dst = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
            store_async_dsmem(dst, words, bar_hop, token)

    @cute.jit
    def verify_early(self, kt, tensors, coords, slots):
        """Loads before the cluster sync (KDA CTAs): state rows from the accepted slot,
        conv taps and weights of channel ``kt`` (and V row 16 r + kt), the norm
        weight."""
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, slot, offset = coords
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base = rank * ROWS
        safe = slot
        if (safe < 0) | (rank >= KDA_CTAS):
            safe = Int32(0)
        state_page = state[Int64(safe), None, None, None]
        history = zeros((self.vkda.passes, self.vkda.chunks, 4), Float32)
        for j in cutlass.range_constexpr(self.vkda.chunks):  # (non-KDA CTAs: slot 0)
            column = 4 * (group_lane + self.vkda.lanes_per_row * j)
            pointer = at(state_page, head, value_base + group, column)
            values = self.vkda.load_state(pointer)
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = values[e]
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), conv_w.element_type)
        accum = zeros(3 * self.tokens, Float32)
        if (rank < KDA_CTAS) & (slot > 0):
            # The conv window (written by this layer's previous step) and weights of
            # channel kt (and V row 16 r + kt), loaded before the cluster sync so the
            # convolution does not wait for them after hop 2.
            dim = self.dim
            qk_channel = head * 128 + kt
            value_channel = head * 128 + value_base + kt
            conv_page = conv_state[Int64(slots[0, 0]), None, None]
            for s in cutlass.range_constexpr(3):
                channel = value_channel if s == 2 else qk_channel
                if (s != 2) | (kt < ROWS):
                    for t in cutlass.range_constexpr(3):
                        taps[s, t] = conv_page[s * dim + channel, offset + t]
                    for t in cutlass.range_constexpr(4):
                        coeff[s, t] = conv_w[s, t, channel]
                    if cutlass.const_expr(self.has_bias):
                        accum[s] = bias[s * dim + channel]
                        for t in cutlass.range_constexpr(1, self.tokens):
                            accum[3 * t + s] = accum[s]
        return history, taps, coeff, accum, norm_w[kt]

    @cute.jit
    def verify_gate(self, args, coords, shared_decay, shared_beta, barrier):
        """KDA CTAs: ``KdaSplit``'s verify gate warp for features [16 r, 16 r + 16) (one
        16-row tile, mode 0: one chain of eight MMAs; MMA column t is token t) with F_A
        polled from ``fa_beta``; the decays and the tokens' beta sigmoids go to every
        KDA CTA. Every CTA reads F_A (the counter protocol counts all of them)."""
        fa_beta, a_log, dt_bias, weight, lower_bound, counter, target = args
        head, rank, slot = coords
        T: cutlass.Constexpr = self.tokens
        lane = cute.arch.lane_idx()
        group, quarter = lane >> 2, lane & 3
        value_base = (rank % KDA_CTAS) * ROWS
        features = [group, group + 8]
        dt = [dt_bias[head * 128 + value_base + f] for f in features]
        wr = cute.make_rmem_tensor((8, 4), Uint32)
        row = head * 128 + value_base + group
        for chunk in cutlass.range_constexpr(8):
            col = 16 * chunk + 4 * quarter
            low = load_bf16x4(at(weight, row, col))
            high = load_bf16x4(at(weight, row + 8, col))
            wr[chunk, 0], wr[chunk, 2] = low
            wr[chunk, 1], wr[chunk, 3] = high
        a = a_log[head]
        cute.arch.cluster_wait()
        ar, beta = self.poll_fa(fa_beta, counter, target, group % T, quarter, head)
        cute.arch.griddepcontrol_launch_dependents()
        if (rank < KDA_CTAS) & (slot > 0):
            acc = zeros(4, Float32)
            for chunk in cutlass.range_constexpr(8):
                wa = tuple(wr[chunk, e] for e in range(4))
                mma_into(acc, wa, ar[chunk], bf16=True)
            for i in cutlass.range_constexpr(2):
                for column in cutlass.range_constexpr(2):
                    raw_g = Float32(BFloat16(acc[2 * i + column]))
                    gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                    gate = sigmoid(gate, False)
                    decay = float_bits(cute.math.exp(lower_bound * gate, fastmath=True))
                    t = 2 * quarter + column
                    pointer = shared_decay.iterator + t * 128 + value_base + features[i]
                    if t < T:
                        for dest in cutlass.range_constexpr(KDA_CTAS):
                            store_async_dsmem(pointer, decay, barrier, dest)
            if (quarter == 0) & (group < T):  # this CTA's betas (bytes on its barrier)
                beta_gate = sigmoid(Float32(beta), False)
                push_float(shared_beta.iterator + group, beta_gate, barrier, rank)
        self.rearm(counter, target, lane, fa_beta)

    @cute.jit
    def load_operands(self, ops, sources, t, group_lane):
        """Token t's q, k and decays of this lane's columns -> ops[0 / 1 / 2]."""
        for j in cutlass.range_constexpr(self.vkda.chunks):
            column = 4 * (group_lane + self.vkda.lanes_per_row * j)
            for i in cutlass.range_constexpr(3):
                pointer = sources[i].iterator + 128 * t + column
                v4 = load_f32x4(pointer, shared=True)
                for e in cutlass.range_constexpr(4):
                    ops[i, j, e] = v4[e]

    @cute.jit
    def store_checkpoint(self, history, state, slot, head, value_base, lanes):
        """``KdaSplit.store_rows`` of pass 0 into state slot ``slot`` (if > 0)."""
        vk = self.vkda
        group, group_lane = lanes
        page = state[Int64(slot), None, None, None]
        for j in cutlass.range_constexpr(vk.chunks):
            column = 4 * (group_lane + vk.lanes_per_row * j)
            values = [history[0, j, e] for e in range(4)]
            pointer = at(page, head, value_base + group % vk.rows, column)
            vk.store_state(pointer, values, (slot > 0) & (group < vk.rows))

    @cute.jit
    def rolled_update(self, history, ops, other, sources, t, group_lane, v_row, beta):
        """``KdaSplit.update`` of pass 0 for token t from register operands (``ops``: q,
        k, decays), software-pipelined: the previous token's query dot (``other``'s q
        against the state before this token's decay, i.e. after the previous token's
        update) is reduced across the row's lanes together with this token's key dot;
        ``other`` then receives token t + 1's operands. Returns the previous token's
        output; ``query_dot`` gives the last token's."""
        vk = self.vkda
        kr = ops[1, None, None]
        dot_q = vk.row_dot(history, other[0, None, None], 0)
        if t + 1 < self.tokens:
            self.load_operands(other, sources, t + 1, group_lane)
        for j in cutlass.range_constexpr(vk.chunks):
            for e in cutlass.range_constexpr(4):
                history[0, j, e] *= ops[2, j, e]
        dot_k = vk.row_dot(history, kr, 0)
        for shift in reversed(range(vk.lanes_per_row.bit_length() - 1)):
            dot_k += cute.arch.shuffle_sync_bfly(dot_k, 1 << shift)
            dot_q += cute.arch.shuffle_sync_bfly(dot_q, 1 << shift)
        v_new = (v_row - dot_k) * beta
        for j in cutlass.range_constexpr(vk.chunks):
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = fma(
                    kr[j, e], v_new, history[0, j, e], ftz=vk.fast_math
                )
        return dot_q

    @cute.jit
    def query_dot(self, history, qr):
        """``KdaSplit.update``'s query dot of the current state, row-reduced."""
        vk = self.vkda
        dot_q = vk.row_dot(history, qr, 0)
        for shift in reversed(range(vk.lanes_per_row.bit_length() - 1)):
            dot_q += cute.arch.shuffle_sync_bfly(dot_q, 1 << shift)
        return dot_q

    @cute.jit
    def verify_main(
        self, kt, early, raws, rows, bars, tensors, coords, slots, scale, eps
    ):
        """``KdaSplit``'s verify main threads (B1 schedule) on the hop-2 inputs."""
        T: cutlass.Constexpr = self.tokens
        dim = self.dim
        history, taps, coeff, accum, nw = early
        raw_qk, raw_v, raw_g, bar_hop = raws
        shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = rows[:6]
        local_o, slot_rows = rows[6:]
        bar_gate, bar_out, shared_beta, bar_qk = bars
        conv_w, bias, conv_state, state, output, norm_w = tensors
        head, rank, slot, offset = coords
        lane, warp = kt % 32, kt // 32
        group, group_lane = kt // self.vkda.lanes_per_row, kt % self.vkda.lanes_per_row
        value_base, qk_channel = rank * ROWS, head * 128 + kt
        value_channel = head * 128 + value_base + kt
        partition = rank
        conv_slot = slots[0, 0]
        if slot <= 0:
            output[partition, qk_channel] = BFloat16(0)
        else:
            conv_page = conv_state[Int64(conv_slot), None, None]
            wait(bar_hop)
            inputs = cute.make_rmem_tensor(3 * T, BFloat16)
            for t in cutlass.range_constexpr(T):
                inputs[3 * t] = raw_qk[t * 128 + kt]
                inputs[3 * t + 1] = raw_qk[(T + t) * 128 + kt]
                inputs[3 * t + 2] = BFloat16(0)
            if kt < ROWS:
                for t in cutlass.range_constexpr(T):
                    inputs[3 * t + 2] = raw_v[t * ROWS + kt]
            gate_owned = raw_g[kt]
            for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        cf = Float32(coeff[s, i])
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), cf, acc)
            # KDA CTA r activates and normalizes only token r's Q / K channels (the
            # same operations as for every token before) and all-gathers them to the
            # eight KDA CTAs; its V rows stay local (threads kt < 16, every token).
            pre_q, pre_k = accum[0], accum[1]
            for t in cutlass.range_constexpr(1, T):
                pre_q = Float32(cutlass.select_(rank == t, accum[3 * t], pre_q))
                pre_k = Float32(cutlass.select_(rank == t, accum[3 * t + 1], pre_k))
            if kt < ROWS:  # V pre-activations; thread kt activates (row kt % 16, token
                for t in cutlass.range_constexpr(T):  # kt // 16) after the barrier
                    shared_v[t * ROWS + kt] = accum[3 * t + 2]
            act = sigmoids([pre_q, pre_k, Float32(gate_owned)])
            q, k, out_gate = pre_q * act[0], pre_k * act[1], act[2]
            square_q, square_k = warp_sum(q * q), warp_sum(k * k)
            if lane == 0:
                reductions[warp] = square_q
                reductions[warp + 4] = square_k
            named_barrier(1, 128)
            pre_v = shared_v[kt]
            shared_v[kt] = pre_v * sigmoids([pre_v])[0]
            qn = (reductions[0] + reductions[2]) + (reductions[1] + reductions[3])
            kn = (reductions[4] + reductions[6]) + (reductions[5] + reductions[7])
            fq = cute.math.rsqrt(qn + 1.0e-6, fastmath=True) * scale
            fk = cute.math.rsqrt(kn + 1.0e-6, fastmath=True)
            q_ptr = shared_q.iterator + rank * 128 + kt
            k_ptr = shared_k.iterator + rank * 128 + kt
            for dest in cutlass.range_constexpr(KDA_CTAS):
                push_float(q_ptr, q * fq, bar_qk, dest)
                push_float(k_ptr, k * fk, bar_qk, dest)
            if kt < T:  # the tokens' checkpoint slots, for the rolled loop
                slot_rows[kt] = slots[0, kt]
            named_barrier(1, 128)  # shared_v, slot_rows
            wait(bar_qk)
            wait(bar_gate)
            # The token loop is a rolled loop (the next token's operands double-buffered
            # in registers): one ~130-instruction body instead of ~1000 instructions of
            # straight-line code fetched once per layer, which the serving neighbours'
            # code evicts from the instruction caches (KDA layer proxy: 807 cycles per
            # token unrolled vs 505 in the bare chain). KdaSplit's operations in its
            # order, so the same states and outputs.
            lanes = (group, group_lane)
            chunks: cutlass.Constexpr = self.vkda.chunks
            # Two operand buffers (q, k, decays of a token) alternate roles over a
            # body of two tokens: token t uses one and the other still holds token
            # t - 1's q for the pipelined query dot, then receives token t + 1's.
            buffers = (
                cute.make_rmem_tensor((3, chunks, 4), Float32),
                cute.make_rmem_tensor((3, chunks, 4), Float32),
            )
            for j in cutlass.range_constexpr(chunks):
                for e in cutlass.range_constexpr(4):
                    buffers[1][0, j, e] = Float32(0)  # (token -1's q: output unused)
            sources = (shared_q, shared_k, shared_decay)
            self.load_operands(buffers[0], sources, 0, group_lane)
            for pair in cutlass.range(T // 2, unroll=1):
                for half in cutlass.range_constexpr(2):
                    t = 2 * pair + half
                    cur, other = buffers[half], buffers[1 - half]
                    v_row = shared_v[t * ROWS + group % ROWS]
                    beta = shared_beta[t]
                    out_prev = self.rolled_update(
                        history, cur, other, sources, t, group_lane, v_row, beta
                    )
                    if (group_lane == 0) & (t > 0):  # (pushed together after the loop)
                        local_o[(t - 1) * ROWS + group] = out_prev
                    slot_t = slot_rows[t]
                    self.store_checkpoint(
                        history, state, slot_t, head, value_base, lanes
                    )
            out_last = self.query_dot(history, buffers[1][0, None, None])
            if group_lane == 0:
                local_o[(T - 1) * ROWS + group] = out_last
            # The outputs of every token: thread kt pushes row kt % 16 of token kt // 16
            # to that token's owner (one remote store per thread instead of eight
            # dependent ones per row lane inside the loop).
            named_barrier(1, 128)
            position = self.vkda.output_position(value_base + kt % ROWS)
            push_float(shared_o.iterator + position, local_o[kt], bar_out, kt // ROWS)
            if kt < ROWS:
                channel = 2 * dim + value_channel
                window = (
                    taps[2, 1],
                    taps[2, 2],
                    *(inputs[3 * t + 2] for t in range(T)),
                )
                for i, value in enumerate(window):
                    conv_page[channel, i] = value
            if partition < T:  # token ``partition``: the gated RMS norm
                wait(bar_out)
                raw_o = shared_o[self.vkda.output_position(kt)]
                sums = cute.make_rmem_tensor(4, Float32)
                for g in cutlass.range_constexpr(4):
                    base = lane % 8 * 16 + g * 4
                    o4 = load_f32x4(shared_o.iterator + base, shared=True)
                    s0, s1, s2, s3 = (o * o for o in o4)
                    sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
                sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
                rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
                result = raw_o * rstd * nw * out_gate
                if cutlass.const_expr(self.polled_out):
                    store_bf16_canonical(at(output, partition, qk_channel), result)
                else:
                    output[partition, qk_channel] = BFloat16(result)
                if partition == 0:
                    for s in cutlass.range_constexpr(2):
                        channel = s * dim + qk_channel
                        window = (taps[s, 1], taps[s, 2])
                        window += tuple(inputs[3 * u + s] for u in range(T))
                        for i, value in enumerate(window):
                            conv_page[channel, i] = value


@cache
def compile_head_verify(heads, bf16_state, has_bias, specs, arch, tile, hidden, polled):
    """``KdaHeadVerify`` for ``specs`` (the tensors of ``PreparedKdaHead._tensors`` plus
    accepted); ``tile``: ``KdaHeadWide``'s (stages, K per stage, ...): its K per stage,
    with VERIFY_STAGES ring stages, VERIFY_TMEM_TILES tensor-memory tiles and
    VERIFY_EARLY_LOADS A tiles before the PDL wait."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from ..common.host import fake

    def matrix(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    tensors = [fake(shape, stride, dtype) for shape, stride, dtype in specs]
    scalars = (Float32(-5), Float32(128**-0.5), Float32(1e-6))
    cta_k = tile[1]
    kernel = KdaHeadVerify(
        heads, stages=VERIFY_STAGES, cta_k=cta_k, tmem_tiles=VERIFY_TMEM_TILES,
        early_loads=VERIFY_EARLY_LOADS, hidden=hidden, bf16_state=bf16_state,
        has_bias=has_bias,
    )  # fmt: skip
    kernel.polled_out = polled
    args = (matrix(1), matrix(1), cutlass.Int64(0), *tensors, *scalars)
    args += (cutlass.Int64(0), Int32(0), cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


# ``KdaHeadVerify``'s ring stages, tensor-memory K tiles and A tiles loaded before the
# PDL wait (KdaHeadWide's HEAD_TILE: 11 / 12 / 17). The verify buffers need the shared
# memory of one ring stage. TMEM holds 15 of the 28 K tiles (480 + 8 of 512 columns).
# In serving (B1) the head is resident ~6 us before the AttnRes ends, beside the
# AttnRes's CTAs, so its pre-wait weight stream delays the AttnRes's latency-bound
# reads: 21 early tiles instead of 25 shorten the AttnRes ~0.2 us and cost the head
# ~0.06. 16-rank KDA layer proxy (tail -> AttnRes -> head -> polled o-proj -> MoE front
# -> experts, 24 layers, arms alternated in one process, 5 processes x 8 rounds): -0.10
# .. -0.21 us/layer; 17 early tiles lose (+0.27 at 4 ranks). Outputs bitwise.
VERIFY_STAGES, VERIFY_TMEM_TILES, VERIFY_EARLY_LOADS = 10, 15, 21
