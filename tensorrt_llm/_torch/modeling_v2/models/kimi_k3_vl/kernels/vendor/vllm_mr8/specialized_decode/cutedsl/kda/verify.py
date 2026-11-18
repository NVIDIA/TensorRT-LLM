# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KdaSplit``'s spec verify recurrence (q > 1) on prepared operands: the input
projection's epilogue has already run the convolution, the activations and the q / k
norms (``kda.inproj64``), so this kernel only decays, updates, checkpoints and writes
the gated outputs."""

from functools import reduce
from operator import or_

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    dot,
    fma,
    front_dirty,
    launch,
    load_f32x4,
    load_volatile,
    load_words,
    mma_into,
    named_barrier,
    push_float,
    scatter_reduce,
    sleep_ns,
    store_words,
    wait,
    warp_sum,
    zeros,
)
from .kernel import KdaSplit, load_bf16x4

FA_SENT = 0xFFFF  # an F_A element not yet written (the in-projection stores canonical)
FA_BACKOFF_NS = 128  # between the light poll's looks


@asm(Uint32)
def _count_reader(address):
    return "atom.acq_rel.gpu.global.add.u32 $0,[$1],1;", "=r,l,~{memory}", (address,)


# ``prepared`` row sections (FP32): the per-head scale is already applied to q.
Q, K, V, GATE, BETA = range(5)  # x 768 (TP16): q, k, v, sigmoid(gate), sigmoid(beta)


class KdaVerifyPrepared(KdaSplit):
    """``KdaSplit`` verify (``tokens`` q > 1, one CTA per token: cluster == q) whose
    first operand is ``prepared`` [B * q, >= 4 * dim + heads] FP32 instead of the BF16
    projection: per token, q (normalized, x scale), k (normalized), v (SiLU), the
    sigmoid of the output gate and the sigmoids of the heads' betas. The gate warp still
    projects F_B from the BF16 projection's F_A, polled: the in-projection stores those
    rows canonical (never ``FA_SENT``) and they start armed, so the gate warp reads them
    as they land instead of after the PDL wait; extra operand (last) ``readers`` [>= B]
    Int32, zeros: the last of a request's heads x cluster readers re-arms its rows and
    resets its count (the next writer is the next KDA layer's in-projection, after its
    PDL wait). The conv weights, bias, conv state, beta and gate operands are not read
    (the conv state is the epilogue's). The rest is ``KdaSplit``'s verify (its
    operations in its order, so the same states, checkpoints and outputs), with the
    token loop rolled and every output pushed once after it."""

    def __init__(self, heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state):
        super().__init__(
            heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state
        )
        # One owned token per CTA, one pass of rows, rolled in token pairs.
        assert tokens > 1 and tokens % 2 == 0 and self.cluster == tokens
        assert self.passes == 1 and self.early and not self.staged
        self.fa_polled = True

    @cute.kernel
    def kernel(self, args):
        prepared, slots, state, output, norm_weight = (
            args[i] for i in (0, 8, 9, 10, 12)
        )
        eps = args[15]
        T: cutlass.Constexpr = self.tokens
        ROWS: cutlass.Constexpr = self.rows  # x T == 128: one (token, row) per thread
        dim: cutlass.Constexpr = self.heads * 128
        thread, _, _ = cute.arch.thread_idx()
        partition, head, request = cute.arch.block_idx()
        token = request * T
        slot = slots[request, args[17][request] - 1]
        if slot <= 0:
            cute.arch.griddepcontrol_wait()
            if (partition == 0) & (thread < 128):
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * 128 + thread] = BFloat16(0)
            if (thread >= 128) & (thread < 160):  # a reader of F_A's rows all the same
                self.release_fa(args, request, thread - 128)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            lane = thread % 32
            group = thread // self.lanes_per_row
            group_lane = thread % self.lanes_per_row
            value_base, qk_channel = partition * ROWS, head * 128 + thread

            smem = cutlass.utils.SmemAllocator()
            shared_q, shared_k, shared_decay = (
                alloc(smem, Float32, 128 * T) for _ in range(3)
            )
            shared_v, local_o, shared_o = (
                alloc(smem, Float32, size) for size in (128, 128, 128)
            )
            shared_beta = alloc(smem, Float32, T)
            slot_rows = alloc(smem, Int32, T)
            barrier, gate_barrier = (smem.allocate(cutlass.Uint64, 8) for _ in range(2))
            if cutlass.const_expr(self.trigger_at_entry):
                cute.arch.griddepcontrol_launch_dependents()
            if thread >= 128:
                cute.arch.cluster_arrive_relaxed()
                coords = (head, token, value_base, lane)
                self.gate_warp(args, shared_decay, gate_barrier, coords, None)

            if thread < 128:
                self.setup(barrier, gate_barrier, partition, thread)
                page = state[Int64(slot), None, None, None]
                history = zeros((1, self.chunks, 4), Float32)
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (group_lane + self.lanes_per_row * j)
                    pointer = at(page, head, value_base + group, column)
                    values = self.load_state(pointer)
                    for e in cutlass.range_constexpr(4):
                        history[0, j, e] = values[e]
                if thread < T:  # the tokens' checkpoint slots, for the rolled loop
                    slot_rows[thread] = slots[request, thread]
                nw = norm_weight[thread]

                cute.arch.griddepcontrol_wait()
                # q / k of every token (4 KB each) by 16-byte cp.async; thread i also
                # loads V row i % 16 of token i // 16, and threads < q the betas.
                for c in cutlass.range_constexpr(4):
                    index = thread + 128 * (c % 2)
                    t, column = index // 32, 4 * (index % 32)
                    target = (shared_q, shared_k)[c // 2].iterator + 128 * t + column
                    section = (c // 2) * dim + head * 128 + column
                    copy_g2s(target, at(prepared, token + t, section))
                cute.arch.cp_async_commit_group()
                v_token, v_row = thread // ROWS, thread % ROWS
                v_channel = V * dim + head * 128 + value_base + v_row
                v_value = prepared[token + v_token, v_channel]
                out_gate = prepared[token + partition, GATE * dim + qk_channel]
                beta = Float32(0)
                if thread < T:
                    beta = prepared[token + thread, BETA * dim + head]
                cute.arch.griddepcontrol_launch_dependents()
                shared_v[thread] = v_value
                if thread < T:
                    shared_beta[thread] = beta
                cute.arch.cp_async_wait_group(0)
                named_barrier(1, 128)  # q, k, v, betas, slot_rows
                wait(gate_barrier)

                # Rolled token loop (as KdaHeadVerify's): two register operand buffers
                # (q, k, decays of a token) alternate roles over a body of two tokens;
                # token t's query dot is reduced with token t + 1's key dot.
                lanes = (group, group_lane)
                buffers = (
                    cute.make_rmem_tensor((3, self.chunks, 4), Float32),
                    cute.make_rmem_tensor((3, self.chunks, 4), Float32),
                )
                for j in cutlass.range_constexpr(self.chunks):
                    for e in cutlass.range_constexpr(4):
                        buffers[1][0, j, e] = Float32(0)  # (token -1's q: unused)
                sources = (shared_q, shared_k, shared_decay)
                self.load_operands(buffers[0], sources, 0, group_lane)
                for pair in cutlass.range(T // 2, unroll=1):
                    for half in cutlass.range_constexpr(2):
                        t = 2 * pair + half
                        cur, other = buffers[half], buffers[1 - half]
                        v_t = shared_v[t * ROWS + group]
                        beta_t = shared_beta[t]
                        out_prev = self.rolled_update(
                            history, cur, other, sources, t, group_lane, v_t, beta_t
                        )
                        if (group_lane == 0) & (t > 0):
                            local_o[(t - 1) * ROWS + group] = out_prev
                        self.store_checkpoint(
                            history, state, slot_rows[t], head, value_base, lanes
                        )
                out_last = self.query_dot(history, buffers[1][0, None, None])
                if group_lane == 0:
                    local_o[(T - 1) * ROWS + group] = out_last
                # Thread i pushes row i % 16 of token i // 16 to the token's owner.
                named_barrier(1, 128)
                cute.arch.cluster_wait()
                position = self.output_position(value_base + thread % ROWS)
                owner = thread // ROWS
                push_float(
                    shared_o.iterator + position, local_o[thread], barrier, owner
                )

                wait(barrier)  # token ``partition``: the gated RMS norm
                raw_o = shared_o[self.output_position(thread)]
                sums = cute.make_rmem_tensor(4, Float32)
                for g in cutlass.range_constexpr(4):
                    base = lane % 8 * 16 + g * 4
                    o4 = load_f32x4(shared_o.iterator + base, shared=True)
                    s0, s1, s2, s3 = (o * o for o in o4)
                    sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
                sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
                rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
                result = raw_o * rstd * nw * out_gate
                output[token + partition, qk_channel] = BFloat16(result)

    @cute.jit
    def polled_fa(self, f_a, token, quarter, args):
        """The gate warp's F_A operands (``KdaSplit.gate_warp``: 8 chunks of 4 BF16 of
        this lane's token, columns 16 c + 4 quarter), polled until no element is
        ``FA_SENT``. Light first (every CTA of a request polls the same rows): one lane
        per token watches one word of each in-projection owner's 32 columns, backing
        off between looks; then every lane loads its chunks, re-polling while any is
        unwritten. (Its count, ``release_fa``, follows the decays' multicast.)"""
        if quarter == 0:
            waiting = Int32(1)
            while waiting != 0:
                pointers = [at(f_a, token, 32 * owner + 30) for owner in range(4)]
                watched = [load_volatile(p)[0] for p in pointers]
                waiting = Int32(reduce(or_, map(front_dirty, watched)))
                if waiting != 0:
                    sleep_ns(FA_BACKOFF_NS)
        cute.arch.sync_warp()
        values = cute.make_rmem_tensor((8, 2), Uint32)
        dirty = Int32(1)
        while dirty != 0:
            for c in cutlass.range_constexpr(8):
                pointer = at(f_a, token, 16 * c + 4 * quarter)
                low, high = load_words(pointer, count=2, volatile=True)
                values[c, 0] = low
                values[c, 1] = high
            words = [values[c, k] for c in range(8) for k in range(2)]
            dirty = Int32(cute.arch.vote_any_sync(reduce(or_, map(front_dirty, words))))
        return [(values[c, 0], values[c, 1]) for c in range(8)]

    @cute.jit
    def release_fa(self, args, request, lane):
        """The gate warp, after the multicast (its F_A reads long done): count this
        reader; the last of the request's heads x cluster re-arms its tokens' 128 F_A
        columns to ``FA_SENT`` and zeroes the count."""
        f_a, readers = args[4], args[18]
        cute.arch.sync_warp()
        last = Int32(0)
        if lane == 0:
            before = _count_reader(readers.iterator + request)
            last = Int32(before == Uint32(self.heads * self.cluster - 1))
        last = cute.arch.shuffle_sync(last, 0)
        cute.arch.sync_warp()  # the other lanes' stores after lane 0's acquire
        if last != 0:
            sentinel = Uint32(FA_SENT | FA_SENT << 16)
            chunks: cutlass.Constexpr = self.tokens * 128 // 8  # 16-byte chunks
            for i in cutlass.range_constexpr(-(-chunks // 32)):
                chunk = lane + 32 * i
                row, column = request * self.tokens + chunk // 16, chunk % 16 * 8
                if cutlass.const_expr(chunks % 32 == 0):
                    store_words(at(f_a, row, column), (sentinel,) * 4)
                else:  # (q = 5: 80 chunks; written as else-if, not a merged or)
                    if chunk < chunks:
                        store_words(at(f_a, row, column), (sentinel,) * 4)
            if lane == 0:
                readers[request] = Int32(0)

    @cute.jit
    def load_operands(self, ops, sources, t, group_lane):
        """Token t's q, k and decays of this lane's columns -> ops[0 / 1 / 2]."""
        for j in cutlass.range_constexpr(self.chunks):
            column = 4 * (group_lane + self.lanes_per_row * j)
            for i in cutlass.range_constexpr(3):
                pointer = sources[i].iterator + 128 * t + column
                v4 = load_f32x4(pointer, shared=True)
                for e in cutlass.range_constexpr(4):
                    ops[i, j, e] = v4[e]

    @cute.jit
    def store_checkpoint(self, history, state, slot, head, value_base, lanes):
        """``store_rows`` of the row into state slot ``slot`` (if > 0)."""
        group, group_lane = lanes
        page = state[Int64(slot), None, None, None]
        for j in cutlass.range_constexpr(self.chunks):
            column = 4 * (group_lane + self.lanes_per_row * j)
            values = [history[0, j, e] for e in range(4)]
            pointer = at(page, head, value_base + group % self.rows, column)
            self.store_state(pointer, values, (slot > 0) & (group < self.rows))

    @cute.jit
    def rolled_update(self, history, ops, other, sources, t, group_lane, v_row, beta):
        """``update`` of the row for token t from register operands (``ops``: q, k,
        decays), software-pipelined: the previous token's query dot (``other``'s q
        against the state after the previous token's update) is reduced across the
        row's lanes together with this token's key dot; ``other`` then receives token
        t + 1's operands. Returns the previous token's output; ``query_dot`` gives the
        last token's."""
        kr = ops[1, None, None]
        dot_q = self.row_dot(history, other[0, None, None], 0)
        if t + 1 < self.tokens:
            self.load_operands(other, sources, t + 1, group_lane)
        for j in cutlass.range_constexpr(self.chunks):
            for e in cutlass.range_constexpr(4):
                history[0, j, e] *= ops[2, j, e]
        dot_k = self.row_dot(history, kr, 0)
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            dot_k += cute.arch.shuffle_sync_bfly(dot_k, 1 << shift)
            dot_q += cute.arch.shuffle_sync_bfly(dot_q, 1 << shift)
        v_new = (v_row - dot_k) * beta
        for j in cutlass.range_constexpr(self.chunks):
            for e in cutlass.range_constexpr(4):
                history[0, j, e] = fma(
                    kr[j, e], v_new, history[0, j, e], ftz=self.fast_math
                )
        return dot_q

    @cute.jit
    def query_dot(self, history, qr):
        """``update``'s query dot of the current state, row-reduced."""
        dot_q = self.row_dot(history, qr, 0)
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            dot_q += cute.arch.shuffle_sync_bfly(dot_q, 1 << shift)
        return dot_q


class KdaVerifyRows(KdaVerifyPrepared):
    """``KdaVerifyPrepared`` with ``per_thread`` (R = 2 or 4) state rows per recurrence
    thread: 128 / R threads (8 lanes per row, 16 / R row groups; group g holds rows
    g + 16 / R * r) and the gate warp after them. A token's q, k and decays are loaded
    from shared memory once per thread for its R rows, which share the columns. Each
    row runs the same operations in the same order as ``KdaVerifyPrepared`` (bitwise);
    the copies, V rows, output pushes and output channels are spread over the fewer
    threads. Why: at three CTAs per SM the token loop is bound by shared-memory
    bandwidth, and a warp's row groups load the same columns (Rubin TP16 B8 x q8: the
    loop without the per-token operand loads takes 3673 instead of 7808 cycles)."""

    def __init__(
        self,
        heads,
        batch,
        gate_mode,
        has_bias,
        schedule,
        tokens,
        bf16_state,
        per_thread,
        spare=0,
    ):
        """``spare``: idle warps after the gate warp, to cap CTAs per SM by registers
        (the fewer threads would let more pack)."""
        super().__init__(
            heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state
        )
        assert per_thread in (2, 4) and self.lanes_per_row == 8 and self.rows == 16
        assert self.gate_mode in (0, 1) and not self.trigger_at_entry
        self.per_thread, self.workers = per_thread, 128 // per_thread
        self.spare = spare

    @cute.jit
    def __call__(self, *args):
        grid = (self.cluster, self.heads, args[0].shape[0] // self.tokens)
        bound = dict(min_blocks_per_mp=self.occupancy) if self.occupancy else {}
        threads = self.workers + 32 * (1 + self.spare)
        launch(self.kernel, args, grid, threads, self.cluster, **bound)

    @cute.kernel
    def kernel(self, args):
        prepared, slots, state, output, norm_weight = (
            args[i] for i in (0, 8, 9, 10, 12)
        )
        eps = args[15]
        T: cutlass.Constexpr = self.tokens
        ROWS: cutlass.Constexpr = self.rows  # x T == 128
        R: cutlass.Constexpr = self.per_thread
        W: cutlass.Constexpr = self.workers  # recurrence threads
        G: cutlass.Constexpr = W // 8  # row groups: group g holds rows g + G r
        dim: cutlass.Constexpr = self.heads * 128
        thread, _, _ = cute.arch.thread_idx()
        partition, head, request = cute.arch.block_idx()
        token = request * T
        lane = thread % 32
        group, group_lane = thread // 8, thread % 8
        value_base = partition * ROWS
        smem = cutlass.utils.SmemAllocator()
        shared_q, shared_k, shared_decay = (
            alloc(smem, Float32, 128 * T) for _ in range(3)
        )
        shared_v, local_o, shared_o = (
            alloc(smem, Float32, size) for size in (128, 128, 128)
        )
        shared_beta = alloc(smem, Float32, T)
        slot_rows = alloc(smem, Int32, T)
        barrier, gate_barrier = (smem.allocate(cutlass.Uint64, 8) for _ in range(2))
        if (thread >= W) & (thread < W + 32):  # the gate warp: its slot loads overlap
            coords = (head, token, value_base, lane)
            self.gate_rows(args, shared_decay, gate_barrier, coords, request)
        else:
            slot = slots[request, args[17][request] - 1]
            if slot <= 0:
                cute.arch.griddepcontrol_wait()
                if (partition == 0) & (thread < W):
                    for t in cutlass.range_constexpr(T):
                        for k in cutlass.range_constexpr(R):
                            channel = head * 128 + thread + W * k
                            output[token + t, channel] = BFloat16(0)
                cute.arch.griddepcontrol_launch_dependents()
            elif thread >= W + 32:  # spare warps: the cluster barrier only
                cute.arch.cluster_arrive_relaxed()
            else:
                self.setup(barrier, gate_barrier, partition, thread)
                page = state[Int64(slot), None, None, None]
                history = zeros((R, self.chunks, 4), Float32)
                for r in cutlass.range_constexpr(R):
                    for j in cutlass.range_constexpr(self.chunks):
                        column = 4 * (group_lane + 8 * j)
                        row = value_base + group + G * r
                        values = self.load_state(at(page, head, row, column))
                        for e in cutlass.range_constexpr(4):
                            history[r, j, e] = values[e]
                if thread < T:  # the tokens' checkpoint slots, for the rolled loop
                    slot_rows[thread] = slots[request, thread]
                nw = [norm_weight[thread + W * k] for k in range(R)]

                cute.arch.griddepcontrol_wait()
                # q / k of every token (4 KB each) by 16-byte cp.async; thread i also
                # loads V rows (i + W k) % 16 of tokens (i + W k) // 16, threads < q the
                # betas.
                for section in cutlass.range_constexpr(2):
                    for c in cutlass.range_constexpr(256 // W):
                        index = thread + W * c
                        t, column = index // 32, 4 * (index % 32)
                        target = (shared_q, shared_k)[section].iterator
                        source = section * dim + head * 128 + column
                        copy_g2s(
                            target + 128 * t + column, at(prepared, token + t, source)
                        )
                cute.arch.cp_async_commit_group()
                v_values, out_gate = [], []
                for k in cutlass.range_constexpr(R):
                    index = thread + W * k
                    v_channel = V * dim + head * 128 + value_base + index % ROWS
                    v_values.append(prepared[token + index // ROWS, v_channel])
                    channel = GATE * dim + head * 128 + index
                    out_gate.append(prepared[token + partition, channel])
                beta = Float32(0)
                if thread < T:
                    beta = prepared[token + thread, BETA * dim + head]
                cute.arch.griddepcontrol_launch_dependents()
                for k in cutlass.range_constexpr(R):
                    shared_v[thread + W * k] = v_values[k]
                if thread < T:
                    shared_beta[thread] = beta
                cute.arch.cp_async_wait_group(0)
                named_barrier(1, W)  # q, k, v, betas, slot_rows
                wait(gate_barrier)

                # The rolled loop of KdaVerifyPrepared, each step over the R rows.
                buffers = (
                    cute.make_rmem_tensor((3, self.chunks, 4), Float32),
                    cute.make_rmem_tensor((3, self.chunks, 4), Float32),
                )
                for j in cutlass.range_constexpr(self.chunks):
                    for e in cutlass.range_constexpr(4):
                        buffers[1][0, j, e] = Float32(0)  # (token -1's q: unused)
                sources = (shared_q, shared_k, shared_decay)
                self.load_operands(buffers[0], sources, 0, group_lane)
                for pair in cutlass.range(T // 2, unroll=1):
                    for half in cutlass.range_constexpr(2):
                        t = 2 * pair + half
                        cur, other = buffers[half], buffers[1 - half]
                        v_t = [shared_v[t * ROWS + group + G * r] for r in range(R)]
                        beta_t = shared_beta[t]
                        out_prev = self.rows_update(
                            history, cur, other, sources, t, group_lane, v_t, beta_t
                        )
                        if (group_lane == 0) & (t > 0):
                            for r in cutlass.range_constexpr(R):
                                local_o[(t - 1) * ROWS + group + G * r] = out_prev[r]
                        self.store_checkpoints(
                            history, state, slot_rows[t], head, value_base, group_lane
                        )
                out_last = self.rows_query_dot(history, buffers[1][0, None, None])
                if group_lane == 0:
                    for r in cutlass.range_constexpr(R):
                        local_o[(T - 1) * ROWS + group + G * r] = out_last[r]
                # Index i = thread + W k pushes row i % 16 of token i // 16 to the
                # token's owner.
                named_barrier(1, W)
                cute.arch.cluster_wait()
                for k in cutlass.range_constexpr(R):
                    index = thread + W * k
                    position = self.output_position(value_base + index % ROWS)
                    pointer = shared_o.iterator + position
                    push_float(pointer, local_o[index], barrier, index // ROWS)

                wait(barrier)  # token ``partition``: the gated RMS norm
                sums = cute.make_rmem_tensor(4, Float32)
                for g in cutlass.range_constexpr(4):
                    base = lane % 8 * 16 + g * 4
                    o4 = load_f32x4(shared_o.iterator + base, shared=True)
                    s0, s1, s2, s3 = (o * o for o in o4)
                    sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
                sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
                rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
                for k in cutlass.range_constexpr(R):
                    channel = thread + W * k
                    raw_o = shared_o[self.output_position(channel)]
                    result = raw_o * rstd * nw[k] * out_gate[k]
                    output[token + partition, head * 128 + channel] = BFloat16(result)

    @cute.jit
    def gate_rows(self, args, shared_decay, gate_barrier, coords, request):
        """``gate_warp`` (q > 1, MMA gate modes, ``polled_fa``) started before the
        request's slot is known: the F_B weights and a first look at all of F_A's rows
        load with the slot. A look that finds every element written skips the light
        poll (``finish_fa``). A padding request re-arms F_A after the PDL wait."""
        f_a, a_log, dt_bias, lower_bound, weight = (args[i] for i in (4, 6, 7, 13, 16))
        slots, accepted = args[8], args[17]
        head, token, value_base, lane = coords
        rows, mode = self.rows, self.gate_mode
        tiles: cutlass.Constexpr = rows // 16
        chains: cutlass.Constexpr = 4 if mode == 1 else 1
        group, quarter = lane >> 2, lane & 3
        a = a_log[head]
        token += group % self.tokens  # the lane's token column, its features
        features = [16 * (i // 2) + group + 8 * (i % 2) for i in range(rows // 8)]
        dt = [dt_bias[head * 128 + value_base + f] for f in features]
        wr = cute.make_rmem_tensor((tiles, 8, 4), Uint32)
        for tile in cutlass.range_constexpr(tiles):
            row = head * 128 + value_base + 16 * tile + group
            for chunk in cutlass.range_constexpr(8):
                col = 16 * chunk + 4 * quarter
                low = load_bf16x4(at(weight, row, col))
                high = load_bf16x4(at(weight, row + 8, col))
                wr[tile, chunk, 0], wr[tile, chunk, 2] = low
                wr[tile, chunk, 1], wr[tile, chunk, 3] = high
        values = cute.make_rmem_tensor((8, 2), Uint32)
        self.look_fa(f_a, token, quarter, values)
        slot = slots[request, accepted[request] - 1]
        if slot <= 0:
            cute.arch.griddepcontrol_wait()
            self.release_fa(args, request, lane)  # a reader of F_A's rows all the same
            cute.arch.griddepcontrol_launch_dependents()
        else:
            cute.arch.cluster_arrive_relaxed()
            ar = self.finish_fa(f_a, token, quarter, values)
            acc = zeros((chains, tiles, 4), Float32)
            for chunk in cutlass.range_constexpr(8):
                chain: cutlass.Constexpr = chunk // 2 if mode == 1 else 0
                for tile in cutlass.range_constexpr(tiles):
                    wa = tuple(wr[tile, chunk, e] for e in range(4))
                    mma_into(acc[chain, tile, None], wa, ar[chunk], bf16=True)
            gates = (a, lower_bound, dt, features)
            targets = (shared_decay, gate_barrier, value_base)
            self.multicast(acc, gates, targets, lane)
            self.release_fa(args, request, lane)

    @cute.jit
    def look_fa(self, f_a, token, quarter, values):
        """One look at the lane's F_A chunks (``polled_fa``'s full loads)."""
        for c in cutlass.range_constexpr(8):
            pointer = at(f_a, token, 16 * c + 4 * quarter)
            low, high = load_words(pointer, count=2, volatile=True)
            values[c, 0] = low
            values[c, 1] = high

    @cute.jit
    def finish_fa(self, f_a, token, quarter, values):
        """``polled_fa`` after the first full look (``values``): done if no element is
        ``FA_SENT``; else its light poll, then full looks until none is."""
        words = [values[c, k] for c in range(8) for k in range(2)]
        dirty = Int32(cute.arch.vote_any_sync(reduce(or_, map(front_dirty, words))))
        if dirty != 0:
            if quarter == 0:
                waiting = Int32(1)
                while waiting != 0:
                    pointers = [at(f_a, token, 32 * owner + 30) for owner in range(4)]
                    watched = [load_volatile(p)[0] for p in pointers]
                    waiting = Int32(reduce(or_, map(front_dirty, watched)))
                    if waiting != 0:
                        sleep_ns(FA_BACKOFF_NS)
            cute.arch.sync_warp()
            while dirty != 0:
                self.look_fa(f_a, token, quarter, values)
                words = [values[c, k] for c in range(8) for k in range(2)]
                dirty = Int32(
                    cute.arch.vote_any_sync(reduce(or_, map(front_dirty, words)))
                )
        return [(values[c, 0], values[c, 1]) for c in range(8)]

    @cute.jit
    def store_checkpoints(self, history, state, slot, head, value_base, group_lane):
        """``store_checkpoint`` of each of the thread's R rows into slot ``slot``."""
        G: cutlass.Constexpr = self.workers // 8
        group = cute.arch.thread_idx()[0] // 8
        page = state[Int64(slot), None, None, None]
        for r in cutlass.range_constexpr(self.per_thread):
            for j in cutlass.range_constexpr(self.chunks):
                column = 4 * (group_lane + 8 * j)
                values = [history[r, j, e] for e in range(4)]
                pointer = at(page, head, value_base + group + G * r, column)
                self.store_state(pointer, values, slot > 0)

    @cute.jit
    def rows_update(self, history, ops, other, sources, t, group_lane, v_rows, beta):
        """``rolled_update`` of each of the thread's R rows, one operand load."""
        R: cutlass.Constexpr = self.per_thread
        kr = ops[1, None, None]
        dot_q = [self.row_dot(history, other[0, None, None], r) for r in range(R)]
        if t + 1 < self.tokens:
            self.load_operands(other, sources, t + 1, group_lane)
        for r in cutlass.range_constexpr(R):
            for j in cutlass.range_constexpr(self.chunks):
                for e in cutlass.range_constexpr(4):
                    history[r, j, e] *= ops[2, j, e]
        dot_k = [self.row_dot(history, kr, r) for r in range(R)]
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            for r in cutlass.range_constexpr(R):
                dot_k[r] += cute.arch.shuffle_sync_bfly(dot_k[r], 1 << shift)
                dot_q[r] += cute.arch.shuffle_sync_bfly(dot_q[r], 1 << shift)
        for r in cutlass.range_constexpr(R):
            v_new = (v_rows[r] - dot_k[r]) * beta
            for j in cutlass.range_constexpr(self.chunks):
                for e in cutlass.range_constexpr(4):
                    history[r, j, e] = fma(
                        kr[j, e], v_new, history[r, j, e], ftz=self.fast_math
                    )
        return dot_q

    @cute.jit
    def rows_query_dot(self, history, qr):
        """``query_dot`` of each of the thread's R rows."""
        dot_q = [self.row_dot(history, qr, r) for r in range(self.per_thread)]
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            for r in cutlass.range_constexpr(self.per_thread):
                dot_q[r] += cute.arch.shuffle_sync_bfly(dot_q[r], 1 << shift)
        return dot_q


class KdaVerifyCols(KdaVerifyRows):
    """``KdaVerifyRows``' gate warp and launch with the recurrence transposed: warp w of
    the four recurrence warps holds state rows 4 w + r (r < 4), lane l columns
    4 l + [0, 4) of them (16 FP32 per thread, as one row of 16 columns per thread does).
    A token's q, k and decays are one 16-byte shared load each per lane (8 lanes per
    row loaded every column 4 times per warp), the checkpoint is one coalesced 256-byte
    row per store, and the row dots reduce across the warp: each lane's 4-term FMA
    chain, then ``scatter_reduce`` of the 8 values (4 query dots against the previous
    state, 4 key dots after the decay) over lanes xor 16, 8, 4, 2, 1, and 4 broadcasts
    of the key dots. Bitwise: lane l's chunk is ``KdaVerifyPrepared``'s chunk l (lane
    l % 8, chunk l // 8, columns 4 (l % 8 + 8 (l // 8))), so lanes xor 16 then xor 8
    add the chunks as its chunk tree (c0 + c2) + (c1 + c3) does, and xor 4, 2, 1 are its
    8-lane butterfly; the decays, v_new, the update, its BF16 rounding and the norm are
    the same operations.

    Any verify length q <= 8 (the cluster): CTAs t < q own token t's gated norm and
    output, the others none; an odd q's last token runs after the token pairs."""

    def __init__(
        self, heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state, spare=0
    ):
        KdaSplit.__init__(
            self, heads, batch, gate_mode, has_bias, schedule, tokens, bf16_state
        )
        # At most one owned token per CTA, one pass of rows, rolled in token pairs.
        assert 1 < tokens <= self.cluster and self.rows == 16
        # No spare warp is unvalidated: 16 x q4 at three CTAs per SM faulted
        # (unspecified launch failure) in a 1-GPU timing run, cause not found.
        assert spare >= 1, "KdaVerifyCols needs a spare warp (spare 0 faulted)"
        assert self.passes == 1 and self.early and not self.staged
        assert self.gate_mode in (0, 1) and not self.trigger_at_entry
        self.fa_polled = True
        self.per_thread, self.workers, self.spare = 4, 128, spare

    @cute.kernel
    def kernel(self, args):
        prepared, slots, state, output, norm_weight = (
            args[i] for i in (0, 8, 9, 10, 12)
        )
        T: cutlass.Constexpr = self.tokens
        ROWS: cutlass.Constexpr = self.rows  # x T == 128
        dim: cutlass.Constexpr = self.heads * 128
        thread, _, _ = cute.arch.thread_idx()
        partition, head, request = cute.arch.block_idx()
        token = request * T
        lane, warp = thread % 32, thread // 32
        value_base = partition * ROWS
        smem = cutlass.utils.SmemAllocator()
        shared_q, shared_k, shared_decay = (
            alloc(smem, Float32, 128 * T) for _ in range(3)
        )
        shared_v, local_o, shared_o = (
            alloc(smem, Float32, size) for size in (128, 128, 128)
        )
        shared_beta = alloc(smem, Float32, T)
        slot_rows = alloc(smem, Int32, T)
        barrier, gate_barrier = (smem.allocate(cutlass.Uint64, 8) for _ in range(2))
        if (thread >= 128) & (thread < 160):  # the gate warp (KdaVerifyRows)
            coords = (head, token, value_base, lane)
            self.gate_rows(args, shared_decay, gate_barrier, coords, request)
        else:
            slot = slots[request, args[17][request] - 1]
            if slot <= 0:
                cute.arch.griddepcontrol_wait()
                if (partition == 0) & (thread < 128):
                    for t in cutlass.range_constexpr(T):
                        output[token + t, head * 128 + thread] = BFloat16(0)
                cute.arch.griddepcontrol_launch_dependents()
            elif thread >= 160:  # spare warps: the cluster barrier only
                cute.arch.cluster_arrive_relaxed()
            else:
                self.setup(barrier, gate_barrier, partition, thread)
                page = state[Int64(slot), None, None, None]
                history = zeros((4, 4), Float32)  # [row 4 warp + r, column 4 lane + e]
                for r in cutlass.range_constexpr(4):
                    row = value_base + 4 * warp + r
                    values = self.load_state(at(page, head, row, 4 * lane))
                    for e in cutlass.range_constexpr(4):
                        history[r, e] = values[e]
                if thread < T:  # the tokens' checkpoint slots, for the rolled loop
                    slot_rows[thread] = slots[request, thread]
                nw = norm_weight[thread]

                cute.arch.griddepcontrol_wait()
                # q / k of every token (4 KB each) by 16-byte cp.async; thread i also
                # loads V row i % 16 of token i // 16, and threads < q the betas.
                for c in cutlass.range_constexpr(4):
                    index = thread + 128 * (c % 2)
                    t, column = index // 32, 4 * (index % 32)
                    target = (shared_q, shared_k)[c // 2].iterator + 128 * t + column
                    section = (c // 2) * dim + head * 128 + column
                    if cutlass.const_expr(T == 8):
                        copy_g2s(target, at(prepared, token + t, section))
                    else:  # this request's tokens (warp-uniform)
                        if t < T:
                            copy_g2s(target, at(prepared, token + t, section))
                cute.arch.cp_async_commit_group()
                v_token, v_row = thread // ROWS, thread % ROWS
                v_channel = V * dim + head * 128 + value_base + v_row
                gate_channel = GATE * dim + head * 128 + thread
                if cutlass.const_expr(T == 8):
                    v_value = prepared[token + v_token, v_channel]
                    out_gate = prepared[token + partition, gate_channel]
                else:  # rows of this request only
                    v_value, out_gate = Float32(0), Float32(0)
                    if v_token < T:
                        v_value = prepared[token + v_token, v_channel]
                    if partition < T:
                        out_gate = prepared[token + partition, gate_channel]
                beta = Float32(0)
                if thread < T:
                    beta = prepared[token + thread, BETA * dim + head]
                cute.arch.griddepcontrol_launch_dependents()
                shared_v[thread] = v_value
                if thread < T:
                    shared_beta[thread] = beta
                cute.arch.cp_async_wait_group(0)
                named_barrier(1, 128)  # q, k, v, betas, slot_rows
                wait(gate_barrier)

                # Rolled token loop: two register operand buffers (q, k, decays of a
                # token's 4 columns) alternate; token t's query dot is reduced with
                # token t + 1's key dot.
                buffers = (
                    cute.make_rmem_tensor((3, 4), Float32),
                    cute.make_rmem_tensor((3, 4), Float32),
                )
                for e in cutlass.range_constexpr(4):
                    buffers[1][0, e] = Float32(0)  # (token -1's q: unused)
                sources = (shared_q, shared_k, shared_decay)
                self.cols_operands(buffers[0], sources, 0, lane)
                for pair in cutlass.range(T // 2, unroll=1):
                    for half in cutlass.range_constexpr(2):
                        t = 2 * pair + half
                        cur, other = buffers[half], buffers[1 - half]
                        v_t = [shared_v[t * ROWS + 4 * warp + r] for r in range(4)]
                        beta_t = shared_beta[t]
                        out_prev = self.cols_update(
                            history, cur, other, sources, t, lane, v_t, beta_t
                        )
                        # lane 4 r holds row r's query dot (scatter_reduce's owner)
                        if (lane < 16) & (lane % 4 == 0) & (t > 0):
                            local_o[(t - 1) * ROWS + 4 * warp + lane // 4] = out_prev
                        self.cols_checkpoint(
                            history, state, slot_rows[t], head, value_base, lane
                        )
                if cutlass.const_expr(T % 2 == 1):  # the last token, after the pairs
                    t = T - 1
                    v_t = [shared_v[t * ROWS + 4 * warp + r] for r in range(4)]
                    out_prev = self.cols_update(
                        history, buffers[0], buffers[1], sources, t, lane, v_t,
                        shared_beta[t],
                    )  # fmt: skip
                    if (lane < 16) & (lane % 4 == 0):
                        local_o[(t - 1) * ROWS + 4 * warp + lane // 4] = out_prev
                    self.cols_checkpoint(
                        history, state, slot_rows[t], head, value_base, lane
                    )
                q_last = [buffers[(T + 1) % 2][0, e] for e in range(4)]
                dots = [
                    dot([history[r, e] for e in range(4)], q_last) for r in range(4)
                ]
                out_last, _ = scatter_reduce(dots, lane, 4)
                if lane % 8 == 0:  # lane 8 r: row r
                    local_o[(T - 1) * ROWS + 4 * warp + lane // 8] = out_last
                # Thread i pushes row i % 16 of token i // 16 to the token's owner.
                named_barrier(1, 128)
                cute.arch.cluster_wait()
                position = self.output_position(value_base + thread % ROWS)
                owner = thread // ROWS
                pointer = shared_o.iterator + position
                if cutlass.const_expr(T == 8):
                    push_float(pointer, local_o[thread], barrier, owner)
                else:  # owners t < q
                    if owner < T:
                        push_float(pointer, local_o[thread], barrier, owner)
                # token ``partition``: the gated RMS norm
                operands = (thread, lane, nw, out_gate)
                if cutlass.const_expr(T == 8):
                    self.cols_norm(shared_o, barrier, operands, args)
                else:
                    if partition < T:
                        self.cols_norm(shared_o, barrier, operands, args)

    @cute.jit
    def cols_norm(self, shared_o, barrier, values, args):
        """Token ``partition``'s gated RMS norm and output channel ``thread``."""
        thread, lane, nw, out_gate = values
        output, eps = args[10], args[15]
        partition, head, request = cute.arch.block_idx()
        token = request * self.tokens
        wait(barrier)  # token ``partition``: the gated RMS norm
        raw_o = shared_o[self.output_position(thread)]
        sums = cute.make_rmem_tensor(4, Float32)
        for g in cutlass.range_constexpr(4):
            base = lane % 8 * 16 + g * 4
            o4 = load_f32x4(shared_o.iterator + base, shared=True)
            s0, s1, s2, s3 = (o * o for o in o4)
            sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
        rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
        result = raw_o * rstd * nw * out_gate
        output[token + partition, head * 128 + thread] = BFloat16(result)

    @cute.jit
    def cols_operands(self, ops, sources, t, lane):
        """Token t's q, k and decays of this lane's 4 columns -> ops[0 / 1 / 2]."""
        for i in cutlass.range_constexpr(3):
            v4 = load_f32x4(sources[i].iterator + 128 * t + 4 * lane, shared=True)
            for e in cutlass.range_constexpr(4):
                ops[i, e] = v4[e]

    @cute.jit
    def cols_update(self, history, ops, other, sources, t, lane, v_rows, beta):
        """Token t's update of the warp's 4 rows: the previous token's query dots
        (``other``'s q against the state after its update) and this token's key dots
        (after the decay) reduced across the warp together; ``other`` then receives
        token t + 1's operands. Returns this lane's reduced value: row lane // 4's
        query dot on lanes 4 r (r < 4)."""
        kr = [ops[1, e] for e in range(4)]
        qr = [other[0, e] for e in range(4)]
        dots = [dot([history[r, e] for e in range(4)], qr) for r in range(4)]
        if t + 1 < self.tokens:
            self.cols_operands(other, sources, t + 1, lane)
        for r in cutlass.range_constexpr(4):
            for e in cutlass.range_constexpr(4):
                history[r, e] *= ops[2, e]
        for r in cutlass.range_constexpr(4):
            dots.append(dot([history[r, e] for e in range(4)], kr))
        owned, _ = scatter_reduce(dots, lane, 8)  # lane 16 + 4 r: row r's key dot
        for r in cutlass.range_constexpr(4):
            dot_k = cute.arch.shuffle_sync(owned, 16 + 4 * r)
            v_new = (v_rows[r] - dot_k) * beta
            for e in cutlass.range_constexpr(4):
                history[r, e] = fma(kr[e], v_new, history[r, e], ftz=self.fast_math)
        return owned

    @cute.jit
    def cols_checkpoint(self, history, state, slot, head, value_base, lane):
        """The warp's 4 rows into state slot ``slot`` (if > 0): a 256-byte row per
        store."""
        page = state[Int64(slot), None, None, None]
        warp = cute.arch.thread_idx()[0] // 32
        for r in cutlass.range_constexpr(4):
            values = [history[r, e] for e in range(4)]
            pointer = at(page, head, value_base + 4 * warp + r, 4 * lane)
            self.store_state(pointer, values, slot > 0)
