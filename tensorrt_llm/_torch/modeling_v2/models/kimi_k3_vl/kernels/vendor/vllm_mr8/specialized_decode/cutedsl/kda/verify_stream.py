# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaSplit``'s spec verify (q in 2-4 tokens per request) for many requests as one
chunked (WY) step per (request, head): the base state is read once, the q tokens' dot
bases are dots of its rows, and every position's state is materialized elementwise
and stored. Bandwidth-oriented: 8 warps, no cluster, the base state staged in shared
memory (several CTAs per SM).

Per head, from the base state S0 (BF16, slot[b, accepted - 1]) and the step's
operands (``KdaSplit``'s prologue, bit for bit: q / k / v / decays a_t / betas):

- D_t = a_0 ... a_t (cumulative decays per key column), E_it = a_(i+1) ... a_t.
- P_t = S0 (D_t k_t), R_t = S0 (D_t q_t): IEEE FP32 dots of each state row (an
  8-FMA chain per 8-column chunk in packed FP32x2 FMAs, the lane's 4 chunks summed in
  order, then the 4 lanes of the row: xor 1, xor 2).
- M_ti = sum_c k_i E_it k_t (i < t), N_ti = sum_c k_i E_it q_t (i <= t): FP32 dots.
- Per state row: u_t = (v_t - (P_t + sum_i<t M_ti u_i)) beta_t (KdaSplit's v_new),
  o_t = R_t + sum_i<=t N_ti u_i (KdaSplit's query dot of the updated state).
- S_t = fma(k_t, u_t, S_(t-1) * a_t) per element, KdaSplit's update formula, rounded
  to nearest-even BF16 once at its store to slot[b, t] (the state stays FP32 across
  the step's tokens); then each token's gated RMS norm.

Not bitwise vs ``KdaSplit``: u_t and o_t are the same quantities formed from dots
against S0 plus a q x q correction instead of dots against each decayed state (a
stated reordering of FP32 arithmetic; the conv window, q, k, v, decays and betas are
``KdaSplit``'s, bit for bit).

Lane (g, c) of warp w (g = lane // 4, c = lane % 4) owns state rows 16 w + g and
16 w + g + 8, columns 32 p + 8 c + [0, 8) for p < 4: 16-byte shared loads of the
staged rows, 16-byte checkpoint stores (64 contiguous bytes per row per warp)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    load_f32x4,
    load_words,
    mma_into,
    mul2,
    named_barrier,
    sigmoid,
    unpack_bf16,
    wait,
    warp_sum,
    zeros,
)
from .kernel import load_activation_bf16x4, load_bf16x4, sigmoids
from .verify_wide import KdaVerifyWide

THREADS = 256  # 8 warps: 16 state rows each


def load_shared_u32x4(pointer):
    return load_words(pointer, count=4, shared=True, mutable=True, word_type="u32")


def load_shared_f32x8(pointer):
    """Eight consecutive FP32 values of shared memory (two 16-byte loads)."""
    return (*load_f32x4(pointer, shared=True), *load_f32x4(pointer + 4, shared=True))


@asm()
def store_bf16x8(pointer, values, *, evict=False, when=None):
    """Eight FP32 values rounded to nearest-even BF16 in one 16-byte store; ``evict``:
    L2 evict-first; ``when``: predicated on it."""
    head = "{ .reg .b32 w0,w1,w2,w3; "
    hint = tail = guard = ""
    if evict:
        head += ".reg .b64 pol; "
        guard = "createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
        hint, tail = ".L2::cache_hint", ",pol"
    constraints, operands = "l" + ",f" * 8, (pointer, *values)
    if when is not None:
        head += ".reg .pred p; "
        guard += "setp.ne.b32 p,$9,0; @p "
        constraints, operands = constraints + ",r", (*operands, Int32(when))
    code = head
    for i in range(4):
        code += f"cvt.rn.bf16x2.f32 w{i},${2 * i + 2},${2 * i + 1}; "
    code += guard + f"st.global{hint}.v4.b32 [$0],{{w0,w1,w2,w3}}{tail}; }}"
    return code, constraints, operands


class KdaVerifyStream(KdaVerifyWide):
    """``KdaSplit`` verify (``tokens`` q in 2-4, BF16 state) of B requests as a chunked
    step: grid (1, heads, B), 256 threads, no cluster. Operands and launch arguments
    are ``KdaSplit``'s. ``occupancy``: minimum CTAs per SM (register bound);
    ``evict``: checkpoint stores and the base-state copy L2 evict-first."""

    def __init__(self, heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state,
                 occupancy=2, evict=True):  # fmt: skip
        super().__init__(
            heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state
        )
        assert tokens <= 4 and bf16_state
        self.min_ctas, self.evict = occupancy, evict
        # (no shared-memory carve-out override: a 100% carve-out measured slower, the
        # prologue's loads use L1)

    @cute.jit
    def __call__(self, *args):
        grid = (1, self.heads, args[0].shape[0] // self.tokens)
        launch(
            self.kernel, args, grid, THREADS, max_number_threads=[THREADS, 1, 1],
            min_blocks_per_mp=self.min_ctas,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, args):
        slots, state, output, gate, norm_weight = (args[i] for i in (8, 9, 10, 11, 12))
        eps, accepted = args[15], args[17]
        T: cutlass.Constexpr = self.tokens
        thread, _, _ = cute.arch.thread_idx()
        _, head, request = cute.arch.block_idx()
        token = request * T
        lane, warp = thread % 32, thread // 32
        g, c = lane // 4, lane % 4
        offset = accepted[request] - 1
        slot, conv_slot = slots[request, offset], slots[request, 0]
        smem = cutlass.utils.SmemAllocator()
        staged = alloc(smem, Uint32, (128, 64), align=128)  # S0 rows, BF16 pairs
        # D_t k_t (t), D_t q_t (4 + t) of column 32 p + 8 c + e at row 32 p + 4 e + c
        bases = alloc(smem, Float32, (128, 8))
        shared_q, shared_k, shared_decay, shared_v = (
            alloc(smem, Float32, 128 * T) for _ in range(4)
        )
        reductions = alloc(smem, Float32, 8 * T)
        shared_beta = alloc(smem, Float32, T)
        scalars = alloc(smem, Float32, (2, 4, 4))  # [M / N, t, i]
        partials = alloc(smem, Float32, (8, 4))  # per warp, per token: sum of squares
        slot_rows = alloc(smem, Int32, 4)
        state_bar = smem.allocate(cutlass.Uint64, 8)
        if slot <= 0:  # a padding request
            cute.arch.griddepcontrol_wait()
            if thread < 128:
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * 128 + thread] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            # The base state (predecessor-independent: this layer's previous step)
            # into shared memory before the PDL wait, as KdaSplit's early 2.
            if thread == 0:
                cute.arch.mbarrier_init(state_bar, 1)
                cute.arch.mbarrier_init_fence()
                size = 128 * 256
                cute.arch.mbarrier_arrive_and_expect_tx(state_bar, size)
                src = at(state[Int64(slot), None, None, None], head, 0, 0)
                copy_g2s(staged.iterator, src, size, state_bar, evict_first=self.evict)
            if thread < T:
                slot_rows[thread] = slots[request, thread]
            rows = [16 * warp + g + 8 * r for r in range(2)]  # the lane's state rows
            nw = [norm_weight[row] for row in rows]
            own_gates = zeros(2, gate.element_type)
            cute.arch.sync_threads()  # (before the PDL wait)
            self.stream_prologue(args, thread, token, offset, conv_slot, shared_q,
                                 shared_k, shared_v, shared_decay, reductions,
                                 shared_beta)  # fmt: skip
            for r in cutlass.range_constexpr(2):  # its norm token c's raw gates
                own_gates[r] = gate[token + c % T, head * 128 + rows[r]]
            cute.arch.griddepcontrol_launch_dependents()
            if thread < 128:
                self.dot_bases(thread, shared_q, shared_k, shared_decay, bases)
            else:
                self.head_scalars(warp - 4, lane, shared_q, shared_k, shared_decay,
                                  scalars)  # fmt: skip
            cute.arch.sync_threads()

            # P_t / R_t of rows g, g + 8: chunk partials (8-FMA chains), the 4
            # chunks' tree, then the quad's (lanes c) butterfly
            wait(state_bar)
            dots = cute.make_rmem_tensor((2, 8), Float32)
            for n in cutlass.range_constexpr(8):
                for r in cutlass.range_constexpr(2):
                    dots[r, n] = Float32(0)
            pairs = [j for j in range(4) if (2 * j) % 4 < T]  # vector pairs in use
            for p in cutlass.range(4, unroll=1):
                column = 32 * p + 8 * c
                values = self.staged_rows(staged, rows, column)
                chunk = cute.make_rmem_tensor((2, 8), Float32)
                for e in cutlass.range_constexpr(8):
                    w8 = load_shared_f32x8(bases.iterator + (32 * p + 4 * e + c) * 8)
                    for r in cutlass.range_constexpr(2):
                        v = values[r, e]
                        for j in pairs:
                            w = (w8[2 * j], w8[2 * j + 1])
                            if cutlass.const_expr(e == 0):
                                pair = mul2((v, v), w)
                            else:
                                pair = fma2(
                                    (v, v), w, (chunk[r, 2 * j], chunk[r, 2 * j + 1])
                                )
                            chunk[r, 2 * j], chunk[r, 2 * j + 1] = pair
                for r in cutlass.range_constexpr(2):
                    for n in cutlass.range_constexpr(8):
                        if cutlass.const_expr(n % 4 < T):
                            dots[r, n] = dots[r, n] + chunk[r, n]
            for r in cutlass.range_constexpr(2):
                for n in cutlass.range_constexpr(8):
                    if cutlass.const_expr(n % 4 < T):
                        s = dots[r, n]
                        s += cute.arch.shuffle_sync_bfly(s, 1)
                        s += cute.arch.shuffle_sync_bfly(s, 2)
                        dots[r, n] = s
            u = cute.make_rmem_tensor((2, 4), Float32)
            outs = cute.make_rmem_tensor((2, 4), Float32)
            for r in cutlass.range_constexpr(2):
                for t in cutlass.range_constexpr(T):
                    kd = dots[r, t]
                    for i in cutlass.range_constexpr(t):
                        kd = fma(scalars[0, t, i], u[r, i], kd)
                    u[r, t] = (shared_v[128 * t + rows[r]] - kd) * shared_beta[t]
                for t in cutlass.range_constexpr(T):
                    o = dots[r, 4 + t]
                    for i in cutlass.range_constexpr(t + 1):
                        o = fma(scalars[1, t, i], u[r, i], o)
                    outs[r, t] = o

            # The checkpoints: rows g, g + 8, columns 32 p + 8 c + [0, 8)
            self.materialize(state, staged, rows, c, u, shared_k, shared_decay,
                             slot_rows, head)  # fmt: skip

            # Each token's gated RMS norm: lane c normalizes token c of its rows
            mine = cute.make_rmem_tensor(2, Float32)
            for r in cutlass.range_constexpr(2):
                mine[r] = outs[r, 0]
            for t in cutlass.range_constexpr(1, T):
                if c == t:
                    for r in cutlass.range_constexpr(2):
                        mine[r] = outs[r, t]
            square = mine[0] * mine[0] + mine[1] * mine[1]
            for shift in cutlass.range_constexpr(3):
                square += cute.arch.shuffle_sync_bfly(square, 4 << shift)
            if (g == 0) & (c < T):
                partials[warp, c] = square
            named_barrier(1, THREADS)
            if c < T:
                pw = [partials[w, c] for w in range(8)]
                total = ((pw[0] + pw[1]) + (pw[2] + pw[3])) + (
                    (pw[4] + pw[5]) + (pw[6] + pw[7])
                )
                rstd = cute.math.rsqrt(total * (1.0 / 128) + eps, fastmath=True)
                activated = sigmoids([Float32(own_gates[r]) for r in range(2)])
                for r in cutlass.range_constexpr(2):
                    result = mine[r] * rstd * nw[r] * activated[r]
                    output[token + c, head * 128 + rows[r]] = BFloat16(result)

    @cute.jit
    def staged_rows(self, staged, rows, column):
        """The base state's 8 values (FP32, exact) of rows ``rows`` at columns
        ``column`` + [0, 8) from the staged BF16 rows."""
        values = cute.make_rmem_tensor((2, 8), Float32)
        for r in cutlass.range_constexpr(2):
            words = load_shared_u32x4(staged.iterator + (rows[r] * 64 + (column >> 1)))
            for e in cutlass.range_constexpr(4):
                low, high = unpack_bf16(words[e])
                values[r, 2 * e] = low
                values[r, 2 * e + 1] = high
        return values

    @cute.jit
    def materialize(self, state, staged, rows, c, u, shared_k, shared_decay,
                    slot_rows, head):  # fmt: skip
        """S_t = fma(k_t, u_t, S_(t - 1) a_t) of the lane's 2 x 32 elements (rows,
        columns 32 p + 8 c + [0, 8)), each token's rounded to BF16 at slot[b, t]."""
        T: cutlass.Constexpr = self.tokens
        writes = [slot_rows[t] for t in range(T)]
        for p in cutlass.range(4, unroll=1):
            column = 32 * p + 8 * c
            values = self.staged_rows(staged, rows, column)
            for t in cutlass.range_constexpr(T):
                a8 = load_shared_f32x8(shared_decay.iterator + (128 * t + column))
                k8 = load_shared_f32x8(shared_k.iterator + (128 * t + column))
                for r in cutlass.range_constexpr(2):
                    ur = u[r, t]
                    for e in cutlass.range_constexpr(4):
                        pair = (values[r, 2 * e], values[r, 2 * e + 1])
                        decayed = mul2(pair, (a8[2 * e], a8[2 * e + 1]))
                        updated = fma2((k8[2 * e], k8[2 * e + 1]), (ur, ur), decayed)
                        values[r, 2 * e] = updated[0]
                        values[r, 2 * e + 1] = updated[1]
                    page = state[Int64(writes[t]), None, None, None]
                    pointer = at(page, head, rows[r], column)
                    out = [values[r, e] for e in range(8)]
                    store_bf16x8(pointer, out, evict=self.evict, when=writes[t] > 0)

    @cute.jit
    def dot_bases(self, column, shared_q, shared_k, shared_decay, bases):
        """Thread ``column`` < 128: D_t k_t and D_t q_t of its key column (bases t
        and 4 + t, t < q)."""
        T: cutlass.Constexpr = self.tokens
        index = (column & ~31) + 4 * (column & 7) + ((column >> 3) & 3)
        cumulative = shared_decay[column]
        for t in cutlass.range_constexpr(4):
            if cutlass.const_expr(t < T):
                if cutlass.const_expr(t > 0):
                    cumulative = cumulative * shared_decay[128 * t + column]
                bases[index, t] = cumulative * shared_k[128 * t + column]
                bases[index, 4 + t] = cumulative * shared_q[128 * t + column]
            else:
                bases[index, t] = Float32(0)
                bases[index, 4 + t] = Float32(0)

    @cute.jit
    def head_scalars(self, w, lane, shared_q, shared_k, shared_decay, scalars):
        """Warp ``w`` of 4: token t = w's scalars M_ti (i < t) and N_ti (i <= t),
        FP32 over the lane's 4 key columns (E_it applied to k_i left to right), then
        a butterfly over the warp."""
        T: cutlass.Constexpr = self.tokens
        for t in cutlass.range_constexpr(T):
            if w == t:
                sums = cute.make_rmem_tensor(2 * t + 1, Float32)  # M_t0.., N_t0..
                for e in cutlass.range_constexpr(4):
                    col = 4 * lane + e
                    kt = shared_k[128 * t + col]
                    qt = shared_q[128 * t + col]
                    for i in cutlass.range_constexpr(t + 1):
                        term = shared_k[128 * i + col]
                        for m in cutlass.range_constexpr(i + 1, t + 1):
                            term = term * shared_decay[128 * m + col]
                        if cutlass.const_expr(i < t):
                            if cutlass.const_expr(e == 0):
                                sums[i] = term * kt
                            else:
                                sums[i] = fma(term, kt, sums[i])
                        if cutlass.const_expr(e == 0):
                            sums[t + i] = term * qt
                        else:
                            sums[t + i] = fma(term, qt, sums[t + i])
                for j in cutlass.range_constexpr(2 * t + 1):
                    for shift in cutlass.range_constexpr(5):
                        sums[j] += cute.arch.shuffle_sync_bfly(sums[j], 16 >> shift)
                if lane == 0:
                    for i in cutlass.range_constexpr(t):
                        scalars[0, t, i] = sums[i]
                    for i in cutlass.range_constexpr(t + 1):
                        scalars[1, t, i] = sums[t + i]

    @cute.jit
    def stream_prologue(self, args, thread, token, offset, conv_slot, shared_q,
                        shared_k, shared_v, shared_decay, reductions,
                        shared_beta):  # fmt: skip
        """``KdaVerifyHalves.half_prologue`` with every V row in the CTA: thread c of
        warps 0-3: Q channel c and V row c, of warps 4-7: K channel c (KdaSplit's
        conv, activation, norm); warp w the decays of features [16 w, + 16); thread
        t < q token t's beta; the Q / K / V conv windows."""
        x, weights, bias, conv_state, f_a, beta = args[:6]
        a_log, dt_bias, lower_bound, scale, f_b = (args[i] for i in (6, 7, 13, 14, 16))
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * 128
        _, head, _ = cute.arch.block_idx()
        lane, warp = thread % 32, thread // 32
        section = thread // 128  # 0: Q, 1: K
        c = thread % 128
        channels = (section * dim + head * 128 + c, 2 * dim + head * 128 + c)
        sections = (section, 2)
        conv_page = conv_state[Int64(conv_slot), None, None]
        has_v = thread < 128
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
        # decays of the warp's 16 features (KdaSplit's gate mode 0 tile)
        acc = zeros(4, Float32)
        for chunk in cutlass.range_constexpr(8):
            wa = tuple(wr[chunk, e] for e in range(4))
            mma_into(acc, wa, ar[chunk], bf16=True)
        for i in cutlass.range_constexpr(2):
            for column in cutlass.range_constexpr(2):
                raw_g = Float32(BFloat16(acc[2 * i + column]))
                gt = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                gt = sigmoid(gt, False)
                decay = cute.math.exp(lower_bound * gt, fastmath=True)
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
                shared_v[128 * t + c] = accum[2 * t + 1] * act[2 * t + 1]
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
        window = (taps[0, 1], taps[0, 2], *(inputs[2 * t] for t in range(T)))
        for i, value in enumerate(window):  # Q / K conv windows
            conv_page[channels[0], i] = value
        if has_v:  # V windows
            window = (taps[1, 1], taps[1, 2], *(inputs[2 * t + 1] for t in range(T)))
            for i, value in enumerate(window):
                conv_page[channels[1], i] = value
        cute.arch.sync_threads()  # q, k, v, decays, betas
