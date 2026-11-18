# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaSplit``'s spec verify (q > 1 tokens per request) with one CTA per (request,
head) for many requests: no cluster and no DSMEM, every operand of the head computed
once in the CTA, and the recurrence transposed over 16 warps.

- Warps 0-3, thread c = channel c: the 4-tap convolution of Q / K / V channel c for the
  q tokens, the activations, the q / k L2 norms (each warp's ``warp_sum``, then
  ((r0 + r2) + (r1 + r3)) over the 4 warps), the conv window, the tokens' betas and
  output gates (``KdaSplit``'s prologue, every row of the head in one CTA).
- Warps 4-11, warp g: the decays of features [16 g, 16 g + 16) for the q tokens (one
  16-row tile of ``KdaSplit``'s gate mode 0: one chain of eight MMAs, MMA column t is
  token t).
- All 16 warps then run the token loop: warp w holds state rows 8 w .. 8 w + 7, lane l
  columns 4 l .. 4 l + 3 of them (32 FP32 per thread). A token's q, k and decays are
  one 16-byte shared load each per lane; each lane's 4-term FMA chains, then
  ``scatter_reduce`` of the 16 values (8 query dots against the previous state, 8 key
  dots after the decay) over lanes xor 16, 8, 4, 2, 1, and 8 broadcasts of the key
  dots. Lane l's columns are the 8-lane layout's chunk l // 8 of lane l % 8, so xor 16
  and xor 8 add the chunks as its chunk tree (c0 + c2) + (c1 + c3) does and xor 4, 2,
  1 are its butterfly (``KdaVerifyCols``' argument).
- Warps 0-15 in four groups of 128 threads: each token's gated RMS norm (all 128
  outputs of the head are in the CTA).

The operations and their orders are ``KdaSplit``'s, so the outputs, every state
checkpoint and the conv window are bit for bit ``KdaSplit``'s verify. Why: above 16
requests ``KdaSplit``'s CTAs (16-32 rows each) repeat the head's prologue per CTA, run
a gate warp each and exchange decays and outputs over DSMEM, and waves of them run
latency-bound; one CTA of 16 warps per head streams the state and its checkpoints."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    at,
    dot,
    fma,
    launch,
    load_f32x4,
    mma_into,
    named_barrier,
    push_float,
    scatter_reduce,
    sigmoid,
    wait,
    warp_sum,
    zeros,
)
from .kernel import KdaSplit, load_activation_bf16x4, load_bf16x4, sigmoids

WARPS = 16  # recurrence warps (8 state rows each)
THREADS = 32 * WARPS


class KdaVerifyWide(KdaSplit):
    """``KdaSplit`` verify (``tokens`` q in 2-8) of B requests: grid (1, heads, B),
    512 threads, no cluster. Operands and launch arguments are ``KdaSplit``'s."""

    def __init__(self, heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state):
        # one CTA per head: 128 state rows; lanes 8 (the chunk tree the transposed
        # reduction reproduces), early (predecessor-independent loads before the wait)
        super().__init__(
            heads, batch, gate_mode, has_bias, (1, 8, 0, 1), tokens, bf16_state
        )
        assert 1 < tokens <= 8 and self.rows == 128 and gate_mode == 0
        assert not self.fast_math

    @cute.jit
    def __call__(self, *args):
        grid = (1, self.heads, args[0].shape[0] // self.tokens)
        # 512 threads at most 128 registers each (one CTA per SM)
        launch(self.kernel, args, grid, THREADS, max_number_threads=[THREADS, 1, 1])

    @cute.kernel
    def kernel(self, args):
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted = args[15], args[17]
        T: cutlass.Constexpr = self.tokens
        thread, _, _ = cute.arch.thread_idx()
        _, head, request = cute.arch.block_idx()
        token = request * T
        lane, warp = thread % 32, thread // 32
        offset = accepted[request] - 1
        slot, conv_slot = slots[request, offset], slots[request, 0]
        smem = cutlass.utils.SmemAllocator()
        shared_q, shared_k, shared_decay, shared_v, shared_o = (
            alloc(smem, Float32, 128 * T) for _ in range(5)
        )
        reductions = alloc(smem, Float32, 8 * T)
        shared_beta = alloc(smem, Float32, T)
        slot_rows = alloc(smem, Int32, T)
        pairs: cutlass.Constexpr = (T + 1) // 2
        # operands of token pair p (q, k, v, betas, slots: the 128 prologue threads;
        # decays: the 256 gate-warp threads) -> ready[p]
        ready = alloc(smem, Int64, pairs, 8).iterator
        if slot <= 0:  # a padding request
            cute.arch.griddepcontrol_wait()
            if thread < 128:
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * 128 + thread] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            # The CTA's state rows (predecessor-independent: this layer's previous
            # step), as KdaSplit's early loads.
            page = state[Int64(slot), None, None, None]
            history = zeros((8, 4), Float32)  # [row 8 warp + r, column 4 lane + e]
            for r in cutlass.range_constexpr(8):
                values = self.load_state(at(page, head, 8 * warp + r, 4 * lane))
                for e in cutlass.range_constexpr(4):
                    history[r, e] = values[e]
            if thread < T:
                slot_rows[thread] = slots[request, thread]
            if thread == 0:
                for p in cutlass.range_constexpr(pairs):
                    cute.arch.mbarrier_init(ready + p, 128 + 32 * 8)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()  # (before the PDL wait)
            # The gated norm of token thread // 64 (channels thread % 64 and + 64):
            # its norm weights, and its raw output gates loaded right after the
            # thread's PDL wait (activated in the norm).
            nw = [norm_weight[thread % 64 + 64 * h] for h in range(2)]
            own_gates = zeros(2, gate.element_type)
            if thread < 128:
                self.prologue(args, thread, token, offset, conv_slot, shared_q,
                              shared_k, shared_v, reductions, shared_beta,
                              own_gates, ready)  # fmt: skip
            elif thread < 128 + 32 * 8:
                self.gate(args, thread // 32 - 4, lane, head, token, shared_decay,
                          own_gates)  # fmt: skip
                for p in cutlass.range_constexpr(pairs):
                    cute.arch.mbarrier_arrive(ready + p)
            else:
                cute.arch.griddepcontrol_wait()
                self.load_own_gates(args, token, thread // 64, thread % 64, own_gates)
            cute.arch.griddepcontrol_launch_dependents()

            # Rolled token loop (as KdaVerifyCols'): two register operand buffers (q,
            # k, decays of a token's 4 columns) alternate; token t's query dots are
            # reduced with token t + 1's key dots. Each pair's operands are awaited
            # before their first read: warps 4-15 start on pair 0 while warps 0-3
            # still compute the later pairs (and join the loop after).
            buffers = (
                cute.make_rmem_tensor((3, 4), Float32),
                cute.make_rmem_tensor((3, 4), Float32),
            )
            for e in cutlass.range_constexpr(4):
                buffers[1][0, e] = Float32(0)  # (token -1's q: unused)
            sources = (shared_q, shared_k, shared_decay)
            cute.arch.mbarrier_wait(ready, 0)
            self.wide_operands(buffers[0], sources, 0, lane)
            for pair in cutlass.range(T // 2, unroll=1):
                for half in cutlass.range_constexpr(2):
                    t = 2 * pair + half
                    if cutlass.const_expr(half == 1):  # noqa: SIM102 (reads t + 1's)
                        if t + 1 < T:
                            cute.arch.mbarrier_wait(ready + pair + 1, 0)
                    self.wide_token(history, buffers[half], buffers[1 - half], sources,
                                    t, lane, warp, shared_v, shared_beta, shared_o,
                                    state, slot_rows, head)  # fmt: skip
            if cutlass.const_expr(T % 2 == 1):  # the last token, after the pairs
                self.wide_token(history, buffers[0], buffers[1], sources, T - 1, lane,
                                warp, shared_v, shared_beta, shared_o, state,
                                slot_rows, head)  # fmt: skip
            q_last = [buffers[(T + 1) % 2][0, e] for e in range(4)]
            dots = [dot([history[r, e] for e in range(4)], q_last) for r in range(8)]
            out_last, _ = scatter_reduce(dots, lane, 8)  # lane 4 r: row r
            if lane % 4 == 0:
                row = 8 * warp + lane // 4
                shared_o[(T - 1) * 128 + self.output_position(row)] = out_last
            cute.arch.sync_threads()

            # Each token's gated RMS norm: 64 threads per token (all tokens at once),
            # thread c of token t: channels c and c + 64.
            t, c, glane = thread // 64, thread % 64, thread % 32
            if t < T:
                self.gated_norm(t, c, glane, shared_o, own_gates, nw, eps, output,
                                token, head)  # fmt: skip

    @cute.jit
    def gated_norm(self, t, c, glane, shared_o, own_gates, nw, eps, output, token,
                   head):  # fmt: skip
        """Token t's gated RMS norm of channels c and c + 64 (of its 128 outputs in
        ``shared_o``): each 8-lane group sums the 128 squares, as KdaSplit."""
        row_o = shared_o.iterator + 128 * t
        raw_o = [shared_o[128 * t + self.output_position(c + 64 * h)] for h in range(2)]
        sums = cute.make_rmem_tensor(4, Float32)
        for g4 in cutlass.range_constexpr(4):
            base = glane % 8 * 16 + g4 * 4
            o4 = load_f32x4(row_o + base, shared=True)
            s0, s1, s2, s3 = (o * o for o in o4)
            sums[g4] = warp_sum((s0 + s2) + (s1 + s3), 8)
        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
        rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
        activated = sigmoids([Float32(own_gates[h]) for h in range(2)])
        for h in cutlass.range_constexpr(2):
            result = raw_o[h] * rstd * nw[h] * activated[h]
            output[token + t, head * 128 + c + 64 * h] = BFloat16(result)

    @cute.jit
    def load_own_gates(self, args, token, t, c, own_gates):
        """The raw output gates of the thread's norm token t (if t < q), channels c
        and c + 64 of the head."""
        gate = args[11]
        _, head, _ = cute.arch.block_idx()
        if t < self.tokens:
            for h in cutlass.range_constexpr(2):
                own_gates[h] = gate[token + t, head * 128 + c + 64 * h]

    @cute.jit
    def prologue(self, args, thread, token, offset, conv_slot, shared_q, shared_k,
                 shared_v, reductions, shared_beta, own_gates, ready):  # fmt: skip
        """Channel ``thread``: KdaSplit's convolution, activations and q / k norms
        (into shared q, k, v [token, 128]) per token pair p, then an arrival on
        ``ready[p]``; thread t < q token t's beta (with pair 0); the conv window; its
        raw output gates (``load_own_gates``)."""
        x, weights, bias, conv_state, f_a, beta = args[:6]
        scale = args[14]
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * 128
        _, head, _ = cute.arch.block_idx()
        lane, warp = thread % 32, thread // 32
        channel = head * 128 + thread
        conv_page = conv_state[Int64(conv_slot), None, None]
        taps = zeros((3, 3), BFloat16)
        coeff = zeros((3, 4), weights.element_type)
        accum = zeros(3 * T, Float32)
        for s in cutlass.range_constexpr(3):
            for t in cutlass.range_constexpr(3):
                taps[s, t] = conv_page[s * dim + channel, offset + t]
            for t in cutlass.range_constexpr(4):
                coeff[s, t] = weights[s, t, channel]
            if cutlass.const_expr(self.has_bias):
                accum[s] = bias[s * dim + channel]
                for t in cutlass.range_constexpr(1, T):
                    accum[3 * t + s] = accum[s]
        cute.arch.griddepcontrol_wait()
        inputs = cute.make_rmem_tensor(3 * T, BFloat16)
        for t in cutlass.range_constexpr(T):
            for s in cutlass.range_constexpr(3):
                inputs[3 * t + s] = x[token + t, s * dim + channel]
        beta_in = beta[token + thread % T, head]
        self.load_own_gates(args, token, thread // 64, thread % 64, own_gates)
        for p in cutlass.range_constexpr((T + 1) // 2):
            tokens = [t for t in (2 * p, 2 * p + 1) if t < T]
            for t in tokens:  # window: 3 taps, then the inputs
                for s in cutlass.range_constexpr(3):
                    for i in cutlass.range_constexpr(4):
                        w = t + i
                        term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                        acc = accum[3 * t + s]
                        accum[3 * t + s] = fma(Float32(term), Float32(coeff[s, i]), acc)
            raw = [accum[3 * t + s] for t in tokens for s in range(3)]
            act = sigmoids(raw + [beta_in] if p == 0 else raw)
            if cutlass.const_expr(p == 0):  # noqa: SIM102
                if thread < T:
                    shared_beta[thread] = act[len(raw)]
            qks = []
            for j, t in enumerate(tokens):
                q = accum[3 * t] * act[3 * j]
                k = accum[3 * t + 1] * act[3 * j + 1]
                shared_v[128 * t + thread] = accum[3 * t + 2] * act[3 * j + 2]
                qks += [q, k]
                square_q, square_k = warp_sum(q * q), warp_sum(k * k)
                if lane == 0:
                    reductions[8 * t + warp] = square_q
                    reductions[8 * t + warp + 4] = square_k
            named_barrier(1, 128)
            for j, t in enumerate(tokens):
                r = [reductions[8 * t + i] for i in range(8)]
                qn = (r[0] + r[2]) + (r[1] + r[3])
                kn = (r[4] + r[6]) + (r[5] + r[7])
                fq = cute.math.rsqrt(qn + 1.0e-6, fastmath=True) * scale
                fk = cute.math.rsqrt(kn + 1.0e-6, fastmath=True)
                shared_q[128 * t + thread] = qks[2 * j] * fq
                shared_k[128 * t + thread] = qks[2 * j + 1] * fk
            cute.arch.mbarrier_arrive(ready + p)
        for s in cutlass.range_constexpr(3):  # the conv window
            window = (taps[s, 1], taps[s, 2], *(inputs[3 * t + s] for t in range(T)))
            for i, value in enumerate(window):
                conv_page[s * dim + channel, i] = value

    @cute.jit
    def gate(self, args, g, lane, head, token, shared_decay, own_gates):
        """Warp ``g`` of 8: ``KdaSplit.gate_warp`` (mode 0, q > 1) for features
        [16 g, 16 g + 16), its decays into the CTA's shared [token, 128]; first its
        threads' raw output gates (``load_own_gates``)."""
        f_a, a_log, dt_bias, lower_bound, weight = (args[i] for i in (4, 6, 7, 13, 16))
        T: cutlass.Constexpr = self.tokens
        group, quarter = lane >> 2, lane & 3
        base = 16 * g
        features = [group, group + 8]
        dt = [dt_bias[head * 128 + base + f] for f in features]
        wr = cute.make_rmem_tensor((8, 4), Uint32)
        row = head * 128 + base + group
        for chunk in cutlass.range_constexpr(8):
            col = 16 * chunk + 4 * quarter
            low = load_bf16x4(at(weight, row, col))
            high = load_bf16x4(at(weight, row + 8, col))
            wr[chunk, 0], wr[chunk, 2] = low
            wr[chunk, 1], wr[chunk, 3] = high
        a = a_log[head]
        cute.arch.griddepcontrol_wait()
        thread = 128 + 32 * g + lane
        self.load_own_gates(args, token, thread // 64, thread % 64, own_gates)
        fa_row = token + group % T  # the lane's token column
        ar = [
            load_activation_bf16x4(at(f_a, fa_row, 16 * c + 4 * quarter))
            for c in range(8)
        ]
        acc = zeros(4, Float32)
        for chunk in cutlass.range_constexpr(8):
            wa = tuple(wr[chunk, e] for e in range(4))
            mma_into(acc, wa, ar[chunk], bf16=True)
        for i in cutlass.range_constexpr(2):
            for column in cutlass.range_constexpr(2):
                raw_g = Float32(BFloat16(acc[2 * i + column]))
                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                gate = sigmoid(gate, False)
                decay = cute.math.exp(lower_bound * gate, fastmath=True)
                t = 2 * quarter + column
                if t < T:
                    shared_decay[128 * t + base + features[i]] = decay

    @cute.jit
    def wide_operands(self, ops, sources, t, lane):
        """Token t's q, k and decays of this lane's 4 columns -> ops[0 / 1 / 2]."""
        for i in cutlass.range_constexpr(3):
            v4 = load_f32x4(sources[i].iterator + 128 * t + 4 * lane, shared=True)
            for e in cutlass.range_constexpr(4):
                ops[i, e] = v4[e]

    @cute.jit
    def wide_token(self, history, ops, other, sources, t, lane, warp, shared_v,
                   shared_beta, shared_o, state, slot_rows, head):  # fmt: skip
        """Token t of the warp's 8 rows: the previous token's query dots (``other``'s
        q against the state after its update) and this token's key dots (after the
        decay) reduced across the warp together; ``other`` then receives token t + 1's
        operands; the update, the previous token's outputs and this token's
        checkpoint."""
        kr = [ops[1, e] for e in range(4)]
        qr = [other[0, e] for e in range(4)]
        dots = [dot([history[r, e] for e in range(4)], qr) for r in range(8)]
        if t + 1 < self.tokens:
            self.wide_operands(other, sources, t + 1, lane)
        for r in cutlass.range_constexpr(8):
            for e in cutlass.range_constexpr(4):
                history[r, e] *= ops[2, e]
        for r in cutlass.range_constexpr(8):
            dots.append(dot([history[r, e] for e in range(4)], kr))
        owned, _ = scatter_reduce(dots, lane, 16)  # lane 2 r: q row r; 16 + 2 r: k
        beta = shared_beta[t]
        for r in cutlass.range_constexpr(8):
            dot_k = cute.arch.shuffle_sync(owned, 16 + 2 * r)
            v_new = (shared_v[128 * t + 8 * warp + r] - dot_k) * beta
            for e in cutlass.range_constexpr(4):
                history[r, e] = fma(kr[e], v_new, history[r, e])
        if (lane < 16) & (lane % 2 == 0) & (t > 0):
            row = 8 * warp + lane // 2
            shared_o[(t - 1) * 128 + self.output_position(row)] = owned
        slot = slot_rows[t]
        page = state[Int64(slot), None, None, None]
        for r in cutlass.range_constexpr(8):
            values = [history[r, e] for e in range(4)]
            pointer = at(page, head, 8 * warp + r, 4 * lane)
            self.store_state(pointer, values, slot > 0)


HALF_WARPS = 8  # each CTA of the pair: 64 state rows, 8 rows per warp


class KdaVerifyHalves(KdaVerifyWide):
    """``KdaVerifyWide`` over a cluster of two CTAs per (request, head): CTA p holds
    state rows [64 p, 64 p + 64) on 8 warps (rows 64 p + 8 w + r, columns 4 l ..), so
    two CTAs of 256 threads (at most 128 registers) share an SM and one CTA's prologue
    and norms overlap another's token loop. Each CTA computes the head's q / k (warps
    0-3 Q, warps 4-7 K: the same per-warp ``warp_sum`` and 4-warp tree), its V rows
    (warps 0-1) and all 128 decays (warp w: features [16 w, 16 w + 16)); CTA t % 2
    owns token t's gated norm and receives the other CTA's 64 outputs of it over DSMEM.
    The same operations in the same orders as ``KdaSplit``: bitwise."""

    @cute.jit
    def __call__(self, *args):
        grid = (2, self.heads, args[0].shape[0] // self.tokens)
        threads = 32 * HALF_WARPS
        launch(
            self.kernel, args, grid, threads, (2, 1, 1),
            max_number_threads=[threads, 1, 1], min_blocks_per_mp=2,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, args):
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted = args[15], args[17]
        T: cutlass.Constexpr = self.tokens
        thread, _, _ = cute.arch.thread_idx()
        half, head, request = cute.arch.block_idx()
        token = request * T
        lane, warp = thread % 32, thread // 32
        row0 = 64 * half  # this CTA's first state row
        offset = accepted[request] - 1
        slot, conv_slot = slots[request, offset], slots[request, 0]
        smem = cutlass.utils.SmemAllocator()
        shared_q, shared_k, shared_decay, shared_o = (
            alloc(smem, Float32, 128 * T) for _ in range(4)
        )
        shared_v = alloc(smem, Float32, 64 * T)
        reductions = alloc(smem, Float32, 8 * T)
        shared_beta = alloc(smem, Float32, T)
        slot_rows = alloc(smem, Int32, T)
        out_bar = smem.allocate(cutlass.Uint64, 8)
        if slot <= 0:  # a padding request (both CTAs of the pair)
            cute.arch.griddepcontrol_wait()
            if (half == 0) & (thread < 128):
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * 128 + thread] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            if thread == 0:  # the peer's 64 outputs of each token this CTA owns
                owned = (T + 1 - half) // 2
                cute.arch.mbarrier_init(out_bar, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(out_bar, owned * 64 * 4)
                cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            page = state[Int64(slot), None, None, None]
            history = zeros(
                (8, 4), Float32
            )  # [row row0 + 8 warp + r, column 4 lane + e]
            for r in cutlass.range_constexpr(8):
                row = row0 + 8 * warp + r
                values = self.load_state(at(page, head, row, 4 * lane))
                for e in cutlass.range_constexpr(4):
                    history[r, e] = values[e]
            if thread < T:
                slot_rows[thread] = slots[request, thread]
            # The gated norm of owned token 2 (thread // 64) + half (channels
            # thread % 64 and + 64): its norm weights and raw output gates (loaded in
            # the prologue right after its PDL wait, activated in the norm).
            nw = [norm_weight[thread % 64 + 64 * h] for h in range(2)]
            own_gates = zeros(2, gate.element_type)
            self.half_prologue(args, thread, token, offset, conv_slot, half, shared_q,
                               shared_k, shared_v, shared_decay, reductions,
                               shared_beta, own_gates)  # fmt: skip
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.cluster_wait()  # the peer's out_bar is armed

            buffers = (
                cute.make_rmem_tensor((3, 4), Float32),
                cute.make_rmem_tensor((3, 4), Float32),
            )
            for e in cutlass.range_constexpr(4):
                buffers[1][0, e] = Float32(0)  # (token -1's q: unused)
            sources = (shared_q, shared_k, shared_decay)
            self.wide_operands(buffers[0], sources, 0, lane)
            outs = (shared_o, out_bar, half, row0)
            for pair in cutlass.range(T // 2, unroll=1):
                for h in cutlass.range_constexpr(2):
                    t = 2 * pair + h
                    self.half_token(history, buffers[h], buffers[1 - h], sources, t,
                                    lane, warp, shared_v, shared_beta, outs, state,
                                    slot_rows, head)  # fmt: skip
            if cutlass.const_expr(T % 2 == 1):  # the last token, after the pairs
                self.half_token(history, buffers[0], buffers[1], sources, T - 1, lane,
                                warp, shared_v, shared_beta, outs, state, slot_rows,
                                head)  # fmt: skip
            q_last = [buffers[(T + 1) % 2][0, e] for e in range(4)]
            dots = [dot([history[r, e] for e in range(4)], q_last) for r in range(8)]
            out_last, _ = scatter_reduce(dots, lane, 8)  # lane 4 r: row r
            if lane % 4 == 0:
                self.half_output(outs, T - 1, row0 + 8 * warp + lane // 4, out_last)
            cute.arch.sync_threads()  # this CTA's own outputs in shared_o
            wait(out_bar)  # the peer's
            # 64 threads per owned token (all at once), thread c: channels c, c + 64
            t, c, glane = 2 * (thread // 64) + half, thread % 64, thread % 32
            if t < T:
                self.gated_norm(t, c, glane, shared_o, own_gates, nw, eps, output,
                                token, head)  # fmt: skip
            cute.arch.cluster_arrive_relaxed()  # no CTA leaves while its peer may push
            cute.arch.cluster_wait()

    @cute.jit
    def half_prologue(self, args, thread, token, offset, conv_slot, half, shared_q,
                      shared_k, shared_v, shared_decay, reductions, shared_beta,
                      own_gates):  # fmt: skip
        """Thread c of warps 0-3: Q channel c, of warps 4-7: K channel c (KdaSplit's
        conv, activation, norm); threads < 64 also V row 64 half + c; warp w the
        decays of features [16 w, + 16); thread t < q token t's beta; CTA 0 the Q / K
        conv windows, each CTA its V rows'; the raw output gates of its gated norm
        (``load_own_gates``: owned token 2 (thread // 64) + half)."""
        x, weights, bias, conv_state, f_a, beta = args[:6]
        a_log, dt_bias, lower_bound, scale, f_b = (args[i] for i in (6, 7, 13, 14, 16))
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * 128
        _, head, _ = cute.arch.block_idx()
        lane, warp = thread % 32, thread // 32
        section = thread // 128  # 0: Q, 1: K
        c = thread % 128
        channels = (
            section * dim + head * 128 + c,
            2 * dim + head * 128 + 64 * half + c,
        )
        sections = (section, 2)
        conv_page = conv_state[Int64(conv_slot), None, None]
        has_v = thread < 64
        taps = zeros((2, 3), BFloat16)
        coeff = zeros((2, 4), weights.element_type)
        accum = zeros(2 * T, Float32)  # [2 t]: Q or K, [2 t + 1]: V
        for s in cutlass.range_constexpr(2):
            if (s == 0) | has_v:
                for t in cutlass.range_constexpr(3):
                    taps[s, t] = conv_page[channels[s], offset + t]
                for t in cutlass.range_constexpr(4):
                    coeff[s, t] = weights[
                        sections[s], t, channels[s] - sections[s] * dim
                    ]
                if cutlass.const_expr(self.has_bias):
                    accum[s] = bias[channels[s]]
                    for t in cutlass.range_constexpr(1, T):
                        accum[2 * t + s] = accum[s]
        # the gate tile's F_B rows (weights: before the wait)
        group, quarter = lane >> 2, lane & 3
        base = 16 * warp
        features = [group, group + 8]
        dt = [dt_bias[head * 128 + base + f] for f in features]
        wr = cute.make_rmem_tensor((8, 4), Uint32)
        frow = head * 128 + base + group
        for chunk in cutlass.range_constexpr(8):
            col = 16 * chunk + 4 * quarter
            low = load_bf16x4(at(f_b, frow, col))
            high = load_bf16x4(at(f_b, frow + 8, col))
            wr[chunk, 0], wr[chunk, 2] = low
            wr[chunk, 1], wr[chunk, 3] = high
        a = a_log[head]
        cute.arch.griddepcontrol_wait()
        inputs = cute.make_rmem_tensor(2 * T, BFloat16)
        for t in cutlass.range_constexpr(T):
            inputs[2 * t] = x[token + t, channels[0]]
            inputs[2 * t + 1] = BFloat16(0)
            if has_v:
                inputs[2 * t + 1] = x[token + t, channels[1]]
        fa_row = token + group % T
        ar = [
            load_activation_bf16x4(at(f_a, fa_row, 16 * k + 4 * quarter))
            for k in range(8)
        ]
        beta_in = beta[token + thread % T, head]
        owned = 2 * (thread // 64) + half
        self.load_own_gates(args, token, owned, thread % 64, own_gates)
        # decays of the warp's 16 features (KdaSplit's gate mode 0 tile)
        acc = zeros(4, Float32)
        for chunk in cutlass.range_constexpr(8):
            wa = tuple(wr[chunk, e] for e in range(4))
            mma_into(acc, wa, ar[chunk], bf16=True)
        for i in cutlass.range_constexpr(2):
            for column in cutlass.range_constexpr(2):
                raw_g = Float32(BFloat16(acc[2 * i + column]))
                g = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                g = sigmoid(g, False)
                decay = cute.math.exp(lower_bound * g, fastmath=True)
                t = 2 * quarter + column
                if t < T:
                    shared_decay[128 * t + base + features[i]] = decay
        for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
            for s in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(4):
                    w = t + i
                    term = taps[s, w] if w < 3 else inputs[2 * (w - 3) + s]
                    acc_s = accum[2 * t + s]
                    accum[2 * t + s] = fma(Float32(term), Float32(coeff[s, i]), acc_s)
        act = sigmoids([accum[i] for i in range(2 * T)] + [beta_in])
        if thread < T:
            shared_beta[thread] = act[2 * T]
        xs = []
        for t in cutlass.range_constexpr(T):
            activated = accum[2 * t] * act[2 * t]
            if has_v:
                shared_v[64 * t + c] = accum[2 * t + 1] * act[2 * t + 1]
            xs.append(activated)
            square = warp_sum(activated * activated)
            if lane == 0:
                reductions[8 * t + warp] = square  # Q: warps 0-3, K: warps 4-7
        cute.arch.sync_threads()
        for t in cutlass.range_constexpr(T):
            r = [reductions[8 * t + 4 * section + i] for i in range(4)]
            n = (r[0] + r[2]) + (r[1] + r[3])
            factor = cute.math.rsqrt(n + 1.0e-6, fastmath=True)
            if section == 0:
                shared_q[128 * t + c] = xs[t] * (factor * scale)
            else:
                shared_k[128 * t + c] = xs[t] * factor
        if half == 0:  # Q / K conv windows (CTA 0)
            window = (taps[0, 1], taps[0, 2], *(inputs[2 * t] for t in range(T)))
            for i, value in enumerate(window):
                conv_page[channels[0], i] = value
        if has_v:  # this CTA's V rows' windows
            window = (taps[1, 1], taps[1, 2], *(inputs[2 * t + 1] for t in range(T)))
            for i, value in enumerate(window):
                conv_page[channels[1], i] = value
        cute.arch.sync_threads()  # q, k, v, decays, betas

    @cute.jit
    def half_output(self, outs, t, row, value):
        """Token t's output of state row ``row`` to its owner CTA t % 2."""
        shared_o, out_bar, half, row0 = outs
        pointer = shared_o.iterator + 128 * t + self.output_position(row)
        if t % 2 == half:
            shared_o[128 * t + self.output_position(row)] = value
        else:
            push_float(pointer, value, out_bar, 1 - half)

    @cute.jit
    def half_token(self, history, ops, other, sources, t, lane, warp, shared_v,
                   shared_beta, outs, state, slot_rows, head):  # fmt: skip
        """``wide_token`` over this CTA's 64 rows (V rows local: 64 per token)."""
        row0 = outs[3]
        kr = [ops[1, e] for e in range(4)]
        qr = [other[0, e] for e in range(4)]
        dots = [dot([history[r, e] for e in range(4)], qr) for r in range(8)]
        if t + 1 < self.tokens:
            self.wide_operands(other, sources, t + 1, lane)
        for r in cutlass.range_constexpr(8):
            for e in cutlass.range_constexpr(4):
                history[r, e] *= ops[2, e]
        for r in cutlass.range_constexpr(8):
            dots.append(dot([history[r, e] for e in range(4)], kr))
        owned, _ = scatter_reduce(dots, lane, 16)  # lane 2 r: q row r; 16 + 2 r: k
        beta = shared_beta[t]
        for r in cutlass.range_constexpr(8):
            dot_k = cute.arch.shuffle_sync(owned, 16 + 2 * r)
            v_new = (shared_v[64 * t + 8 * warp + r] - dot_k) * beta
            for e in cutlass.range_constexpr(4):
                history[r, e] = fma(kr[e], v_new, history[r, e])
        if (lane < 16) & (lane % 2 == 0) & (t > 0):
            self.half_output(outs, t - 1, row0 + 8 * warp + lane // 2, owned)
        slot = slot_rows[t]
        page = state[Int64(slot), None, None, None]
        for r in cutlass.range_constexpr(8):
            values = [history[r, e] for e in range(4)]
            pointer = at(page, head, row0 + 8 * warp + r, 4 * lane)
            self.store_state(pointer, values, slot > 0)
