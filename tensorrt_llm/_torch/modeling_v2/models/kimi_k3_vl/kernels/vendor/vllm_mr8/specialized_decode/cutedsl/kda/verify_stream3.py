# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaVerifyStream`` with its row dots on tensor cores and a materialization tile
that keeps each lane's decays and keys in registers (fewer shared-memory operations:
the kernel is bound by the SM's memory-instruction pipe, not by FP32 issue).

- P_t / R_t: warp w's 16 state rows x the 8 vectors D_t k_t / D_t q_t (t < 4) as
  m16n8k16 BF16 mma.sync with FP32 accumulation. S0 is exact in BF16; each FP32 vector
  element enters as three BF16 terms hi + mid + lo (hi = RNE(x), mid = RNE(x - hi), lo
  = RNE(x - hi - mid): exactly x), one accumulator chain per term, summed (lo + mid) +
  hi. The K
  (key column) index is permuted per lane so each operand fragment is one 16-byte
  shared load: thread (g, c) supplies columns 32 p + 8 c + 4 h + [0, 4) for k-step
  2 p + h of rows g and g + 8 (A) and vector g (B); a permutation of the summed index,
  the same for A and B.
- u_t, o_t per row as ``KdaVerifyStream`` (the quad of lanes holding a row computes it
  redundantly); u goes to shared memory for the materialization.
- Materialization: lane (half, cg) of warp w owns rows 16 w + 8 half + [0, 8), columns
  8 cg + [0, 8); its a_t / k_t (8 columns x q tokens) stay in registers; each row is
  one 16-byte S0 load, one 16-byte u load and q 16-byte checkpoint stores (a half-warp
  stores a whole 256-byte state row).

Same update formula, rounding point and stores as ``KdaVerifyStream``; u_t / o_t come
from tensor-core dots (a reordering of their FP32 sums)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    at,
    copy_g2s,
    fma,
    fma2,
    load_f32x4,
    mma_into,
    mul2,
    named_barrier,
    unpack_bf16,
    wait,
    zeros,
)
from .kernel import sigmoids
from .verify_stream import (
    THREADS,
    KdaVerifyStream,
    load_shared_f32x8,
    load_shared_u32x4,
    store_bf16x8,
)


class KdaVerifyStream3(KdaVerifyStream):
    """``KdaVerifyStream`` (q in 2-4, BF16 state) with tensor-core row dots: grid (1,
    heads, B), 256 threads."""

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
        # B operand: [term lo / mid / hi][chunk p][vector n][32 columns] BF16
        vectors = alloc(smem, BFloat16, (3, 4, 8, 32), align=128)
        shared_q, shared_k, shared_decay, shared_v = (
            alloc(smem, Float32, 128 * T) for _ in range(4)
        )
        shared_u = alloc(smem, Float32, (128, 4), align=16)  # u_t of each state row
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
            if thread == 0:  # the base state into shared memory (KdaSplit's early 2)
                cute.arch.mbarrier_init(state_bar, 1)
                cute.arch.mbarrier_init_fence()
                size = 128 * 256
                cute.arch.mbarrier_arrive_and_expect_tx(state_bar, size)
                src = at(state[Int64(slot), None, None, None], head, 0, 0)
                copy_g2s(staged.iterator, src, size, state_bar, evict_first=self.evict)
            if thread < T:
                slot_rows[thread] = slots[request, thread]
            rows = [16 * warp + g + 8 * r for r in range(2)]  # the quad's state rows
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
                self.split_bases(thread, shared_q, shared_k, shared_decay, vectors)
            else:
                self.head_scalars(warp - 4, lane, shared_q, shared_k, shared_decay,
                                  scalars)  # fmt: skip
            cute.arch.sync_threads()
            wait(state_bar)

            dots = self.mma_dots(staged, vectors, warp, g, c)
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
                if c == 0:
                    for t in cutlass.range_constexpr(T):
                        shared_u[rows[r], t] = u[r, t]
            # each token's sum of squares (lane c: token c of its rows)
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
            named_barrier(1, THREADS)  # u and the partial sums

            if c < T:  # the gated RMS norms
                pw = [partials[w, c] for w in range(8)]
                total = ((pw[0] + pw[1]) + (pw[2] + pw[3])) + (
                    (pw[4] + pw[5]) + (pw[6] + pw[7])
                )
                rstd = cute.math.rsqrt(total * (1.0 / 128) + eps, fastmath=True)
                activated = sigmoids([Float32(own_gates[r]) for r in range(2)])
                for r in cutlass.range_constexpr(2):
                    result = mine[r] * rstd * nw[r] * activated[r]
                    output[token + c, head * 128 + rows[r]] = BFloat16(result)
            self.materialize3(state, staged, shared_u, shared_k, shared_decay,
                              slot_rows, head, warp, lane)  # fmt: skip

    @cute.jit
    def mma_dots(self, staged, vectors, warp, g, c):
        """P_t (vectors t) and R_t (4 + t) of rows 16 warp + g + 8 r for every lane of
        the quad: [r, n]."""
        words = cute.make_rmem_tensor((2, 4, 4), Uint32)  # [row r, chunk p, word]
        for r in cutlass.range_constexpr(2):
            row = 16 * warp + g + 8 * r
            for p in cutlass.range_constexpr(4):
                w4 = load_shared_u32x4(staged.iterator + (row * 64 + 16 * p + 4 * c))
                for e in cutlass.range_constexpr(4):
                    words[r, p, e] = w4[e]
        acc = zeros((3, 4), Float32)  # per term (lo, mid, hi): rows g / g + 8, n 2c..
        for term in cutlass.range_constexpr(3):
            for p in cutlass.range_constexpr(4):
                base = vectors.iterator + ((term * 4 + p) * 8 + g) * 32 + 8 * c
                b4 = load_shared_u32x4(cute.recast_ptr(base, dtype=Uint32))
                for h in cutlass.range_constexpr(2):
                    a = (
                        words[0, p, 2 * h],
                        words[1, p, 2 * h],
                        words[0, p, 2 * h + 1],
                        words[1, p, 2 * h + 1],
                    )
                    mma_into(acc[term, None], a, (b4[2 * h], b4[2 * h + 1]), bf16=True)
        total = cute.make_rmem_tensor(4, Float32)
        for e in cutlass.range_constexpr(4):
            total[e] = (acc[0, e] + acc[1, e]) + acc[2, e]
        dots = cute.make_rmem_tensor((2, 8), Float32)
        for source in cutlass.range_constexpr(4):
            for r in cutlass.range_constexpr(2):
                for e in cutlass.range_constexpr(2):
                    value = cute.arch.shuffle_sync(total[2 * r + e], 4 * g + source)
                    dots[r, 2 * source + e] = value
        return dots

    @cute.jit
    def materialize3(self, state, staged, shared_u, shared_k, shared_decay, slot_rows,
                     head, warp, lane):  # fmt: skip
        """Lane (half, cg): rows 16 warp + 8 half + [0, 8), columns 8 cg + [0, 8); its
        a_t / k_t in registers; S_t = fma(k_t, u_t, S_(t-1) a_t), RNE BF16 stores."""
        T: cutlass.Constexpr = self.tokens
        half, cg = lane // 16, lane % 16
        column = 8 * cg
        writes = [slot_rows[t] for t in range(T)]
        a8 = [
            load_shared_f32x8(shared_decay.iterator + (128 * t + column))
            for t in range(T)
        ]
        k8 = [
            load_shared_f32x8(shared_k.iterator + (128 * t + column)) for t in range(T)
        ]
        for r in cutlass.range(8, unroll=1):
            row = 16 * warp + 8 * half + r
            words = load_shared_u32x4(staged.iterator + (row * 64 + 4 * cg))
            values = cute.make_rmem_tensor(8, Float32)
            for e in cutlass.range_constexpr(4):
                low, high = unpack_bf16(words[e])
                values[2 * e] = low
                values[2 * e + 1] = high
            u4 = load_f32x4(shared_u.iterator + row * 4, shared=True)
            for t in cutlass.range_constexpr(T):
                ur = u4[t]
                for e in cutlass.range_constexpr(4):
                    pair = (values[2 * e], values[2 * e + 1])
                    decayed = mul2(pair, (a8[t][2 * e], a8[t][2 * e + 1]))
                    updated = fma2((k8[t][2 * e], k8[t][2 * e + 1]), (ur, ur), decayed)
                    values[2 * e] = updated[0]
                    values[2 * e + 1] = updated[1]
                page = state[Int64(writes[t]), None, None, None]
                pointer = at(page, head, row, column)
                store_bf16x8(pointer, [values[e] for e in range(8)], evict=self.evict,
                             when=writes[t] > 0)  # fmt: skip

    @cute.jit
    def split_bases(self, column, shared_q, shared_k, shared_decay, vectors):
        """Thread ``column`` < 128: D_t k_t (vector t) and D_t q_t (4 + t) of its key
        column, each as three BF16 terms lo / mid / hi (zeros for t >= q)."""
        T: cutlass.Constexpr = self.tokens
        p, index = column // 32, column % 32
        cumulative = shared_decay[column]
        for t in cutlass.range_constexpr(4):
            pair = (Float32(0), Float32(0))
            if cutlass.const_expr(t < T):
                if cutlass.const_expr(t > 0):
                    cumulative = cumulative * shared_decay[128 * t + column]
                pair = (
                    cumulative * shared_k[128 * t + column],
                    cumulative * shared_q[128 * t + column],
                )
            for n, value in ((t, pair[0]), (4 + t, pair[1])):
                hi = BFloat16(value)
                rest = value - Float32(hi)
                mid = BFloat16(rest)
                lo = BFloat16(rest - Float32(mid))
                vectors[0, p, n, index] = lo
                vectors[1, p, n, index] = mid
                vectors[2, p, n, index] = hi
