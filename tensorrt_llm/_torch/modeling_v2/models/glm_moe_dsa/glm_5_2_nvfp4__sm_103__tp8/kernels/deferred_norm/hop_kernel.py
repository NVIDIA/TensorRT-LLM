# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CuTe DSL kernels of the deferred-norm post-MoE hop.

``DeferredNormHopKernel`` is the post_moe_hop cluster kernel (``post_moe_hop/cute_kernel.py``) with
the RMSNorm deferred to the next layer's projection. Steps 1-5 and 7 are post_moe_hop's
(finalize, multicast publish, clear of the previous call, Lamport poll,
rank-ordered fp32 sum, bf16, residual add, arrival), on the same post_moe_hop
workspace and with the same workspace protocol, so deferred_norm and post_moe_hop hops may
follow each other on one workspace:

* state words, epoch / previous rows / arrivals / call count (post_moe_hop's word
  indices, ``ptx.py``);
* error handling: every Lamport poll is bounded by the hop's ``timeout_ns``
  deadline; on expiry the thread records the Lamport-timeout bit and its
  poll count and writes a NaN residual (so the row's square sum, ``r`` and
  every value derived from the packet are NaN); a permutation entry at or
  beyond ``fc2_rows`` is skipped and records the out-of-range bit. Error
  bits go into the error word of the call's Lamport buffer. A call reads the
  previous buffer's error word (complete at the kernel boundary) and, when
  it is nonzero, publishes NaN and carries the bits into its own word, so
  every later call of the rank -- post_moe_hop or deferred_norm -- publishes NaN until the
  collective workspace reset, and ``post_moe_hop.check_health`` sees deferred_norm's errors.

Step 6 differs:

* every thread forms post_moe_hop's per-thread sum of bf16-rounded squares of its 8
  columns and the warp butterfly (16, 8, 4, 2, 1); lane 0 of every warp
  stores the warp sum to ``sumsq[row, cta * WARPS + warp]`` (fp32). The
  consumer sums the ``CTAS * WARPS`` values in index order, which is the
  order of post_moe_hop's distributed-shared-memory reduction, so its ``rsqrt``
  operand is bit-identical to post_moe_hop's;
* every thread stores the gain-folded activation ``a = bf16(x * gamma)`` of
  its 8 columns instead of the normalized output, in the MMA B-fragment
  layout of ``packet.py`` (``scaled`` is the ``a_frag`` buffer), so that the
  projection reads it with one coalesced 8-byte load per lane and K step.

There is no cross-CTA dependency left inside a row, so the kernel is not a
cluster launch. With ``trigger_early`` the PDL dependents (the next layer's
projection, which reads only weights before its own dependency wait) are
released right after this kernel's dependency wait instead of after the
reduction.

With ``routed_input`` (deferred_norm composed with routed_moe_into_hop) step 1 reads the
routed input routed_moe wrote into the post_moe_hop's routed-input buffer instead of the
raw tuple, with the same instructions as post_moe_hop's routed_moe_into_hop variant of the cluster
kernel for each mode:

* ``"row"``: the finalized bf16 row ``[R, H]``;
* ``"identity"``: raw FC2 rows ``t * 8 + k`` with bf16 scales ``[R, 8]``, the
  fp32 FMA chain in k order without the permutation;
* ``"fp32"``: the FP32 accumulator ``[R, H]``, rounded to bf16; the thread
  clears the 8 values it read (the buffer is zero before every call).

Everything after step 1 is the same code as the raw-tuple variant, so the
packet's materialization equals ``PostMoeHop.hop_routed``'s output bit
for bit on the same routed input.

``DeferredNormMaterializeKernel`` turns a deferred packet back into post_moe_hop's normalized
output ``bf16((x * r) * gamma)`` (same instructions as post_moe_hop's step 6); it is
the fallback when a consumer cannot take the packet.

Every inline-PTX helper comes from ``ptx.py`` (owned by deferred_norm).
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from ..post_moe_hop.kernels import ERROR_PERM_RANGE, MAX_CHUNKS
from .ptx import (
    FP32_SENTINEL,
    SENTINEL_WORD,
    _asm,
    _v,
    add,
    arrive,
    bits_f32,
    error_word_address,
    fault_delay,
    flag_error,
    flag_timeout,
    fma,
    inv_rms,
    ld_v4,
    load_state,
    mul,
    pack_bf16x2,
    perm_check,
    poison,
    poll8,
    round_bf16,
    sanitize,
    select3,
    select_f32,
    shfl_xor,
    st_v4,
    unpack_bf16x2,
)


@dsl_user_op
def _st_f32_if(addr: Int64, value: Float32, enable: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, loc, ip), _v(value, loc, ip), _v(enable, loc, ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\n@p st.global.f32 [$0], $1;\n}\n",
        "l,f,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_b32(addr: Int64, word: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, loc, ip), _v(word, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_v(addr, loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _globaltimer_st(addr: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, loc, ip)],
        "{\n.reg .u64 t;\nmov.u64 t, %globaltimer;\nst.global.u64 [$0], t;\n}\n",
        "l",
        loc=loc,
        ip=ip,
    )


@cute.jit
def _hop_stamp(
    enabled: cutlass.Constexpr[bool], ts: Int64, sid: Int32, tid: Int32, k: cutlass.Constexpr[int]
):
    if cutlass.const_expr(enabled):
        if tid == Int32(0):
            _globaltimer_st(ts + Int64((sid * Int32(8) + Int32(k)) * Int32(8)))


class DeferredNormHopKernel:
    """Compile-time configuration of the deferred hop for one shape.

    ``fault_delay_ns`` (tests only; the served hop keeps 0) delays about one
    thread in four by a hashed amount below this bound before its multicast
    publish, before its share of the clear and before the arrival (the sites
    of post_moe_hop's fault injection), so the individual stores of a call land in a
    scrambled order.

    ``routed_input`` (``None``: the raw tuple; ``"row"``, ``"identity"`` or
    ``"fp32"``: routed_moe_into_hop's routed input, module docstring) selects step 1; then
    ``fc2`` is the routed buffer, ``fc2_row_bytes`` its row pitch, ``scales``
    the identity-mode scales and ``perm`` / ``fc2_rows`` are unused.
    """

    def __init__(
        self,
        hidden: int,
        top_k: int,
        world: int,
        max_rows: int,
        ctas: int,
        has_shared: bool,
        scale_fp32: bool,
        use_pdl: bool,
        trigger_early: bool,
        carveout: int = -1,
        debug_ts: bool = False,
        fault_delay_ns: int = 0,
        routed_input: Optional[str] = None,
    ):
        assert top_k == 8 and world == 8, "the deferred_norm hop is specialized to top-8 and TP8"
        assert routed_input in ROUTED_MODES, routed_input
        assert hidden % (8 * 32 * ctas) == 0
        self.hidden = hidden
        self.top_k = top_k
        self.world = world
        self.max_rows = max_rows
        self.ctas = ctas
        self.threads = hidden // (8 * ctas)
        self.warps = self.threads // 32
        self.has_shared = has_shared
        self.scale_fp32 = scale_fp32
        self.use_pdl = use_pdl
        self.trigger_early = trigger_early
        # Preferred shared-memory carveout (percent; -1: driver default).
        self.carveout = carveout
        # Tests only: %globaltimer stamps per CTA into ``ts`` [R * CTAS, 8]
        # (0 entry, 1 after the dependency wait, 2 published, 3 poll done,
        # 4 exit).
        self.debug_ts = debug_ts
        self.fault_delay_ns = fault_delay_ns
        self.routed_input = routed_input

    @property
    def partial_count(self) -> int:
        """Number of fp32 square-sum partials per row."""
        return self.ctas * self.warps

    @cute.jit
    def __call__(
        self,
        fc2: Int64,
        fc2_row_bytes: Int64,
        fc2_rows: Int32,
        scales: Int64,
        perm: Int64,
        shared: Int64,
        residual: Int64,
        gamma: Int64,
        scaled: Int64,
        residual_out: Int64,
        sumsq: Int64,
        mc: Int64,
        uc: Int64,
        partials: Int64,
        ts: Int64,
        state: cute.Tensor,
        rank: Int32,
        timeout_ns: Int64,
        num_rows: Int32,
        stream: CUstream,
    ):
        # ``state`` is passed as a GPU tensor so the TVM-FFI launcher can bind
        # the caller's current stream; every other buffer is a raw address.
        if cutlass.const_expr(self.carveout >= 0):
            self.kernel(
                fc2,
                fc2_row_bytes,
                fc2_rows,
                scales,
                perm,
                shared,
                residual,
                gamma,
                scaled,
                residual_out,
                sumsq,
                mc,
                uc,
                partials,
                ts,
                state,
                rank,
                timeout_ns,
                num_rows,
            ).launch(
                grid=[self.ctas, num_rows, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
                preferred_smem_carveout=self.carveout,
            )
        else:
            self.kernel(
                fc2,
                fc2_row_bytes,
                fc2_rows,
                scales,
                perm,
                shared,
                residual,
                gamma,
                scaled,
                residual_out,
                sumsq,
                mc,
                uc,
                partials,
                ts,
                state,
                rank,
                timeout_ns,
                num_rows,
            ).launch(
                grid=[self.ctas, num_rows, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.kernel
    def kernel(
        self,
        fc2: Int64,
        fc2_row_bytes: Int64,
        fc2_rows: Int32,
        scales: Int64,
        perm: Int64,
        shared: Int64,
        residual: Int64,
        gamma: Int64,
        scaled: Int64,
        residual_out: Int64,
        sumsq: Int64,
        mc: Int64,
        uc: Int64,
        partials: Int64,
        ts: Int64,
        state_tensor: cute.Tensor,
        rank: Int32,
        timeout_ns: Int64,
        num_rows: Int32,
    ):
        hidden = self.hidden
        world = self.world
        threads = self.threads
        warps = self.warps
        ctas = self.ctas
        buffer_bytes = self.max_rows * world * hidden * 2

        tidx, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        state = Int64(state_tensor.iterator.toint())
        sid = row * Int32(ctas) + cta
        _hop_stamp(self.debug_ts, ts, sid, tidx, 0)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.trigger_early):
                cute.arch.griddepcontrol_launch_dependents()
        _hop_stamp(self.debug_ts, ts, sid, tidx, 1)

        # State words of the previous call (complete at the kernel boundary).
        epoch, prev_rows, err_words = load_state(state)
        cur = epoch % Int32(3)
        dirty = (epoch + Int32(2)) % Int32(3)
        # Bits of the previous call's buffer: when set, this call publishes
        # NaN and carries them into its own error word.
        sticky = select3(err_words, dirty)
        err_cur = error_word_address(state, cur)

        col = (cta * Int32(threads) + tidx) * Int32(8)
        row_off = Int64(row) * Int64(hidden * 2) + Int64(col) * Int64(2)
        res_w = ld_v4(residual + row_off)
        gamma_w = ld_v4(gamma + Int64(col) * Int64(2))
        if cutlass.const_expr(self.has_shared):
            sh_w = ld_v4(shared + row_off)

        # 1. Finalize (fp32 FMA chain in k order), bf16, + shared, sanitize.
        if cutlass.const_expr(self.routed_input == "row"):
            # routed_moe_into_hop: routed_moe already finalized this rank's routed output.
            r_w = ld_v4(fc2 + Int64(row) * fc2_row_bytes + Int64(col) * Int64(2))
            local = [r_w[j] for j in range(4)]
            out_of_range = Int32(0)
        elif cutlass.const_expr(self.routed_input == "fp32"):
            # routed_moe_into_hop: FP32 accumulator of the top-k contributions; round, then
            # clear it for the routed kernel of the next call.
            a_addr = fc2 + Int64(row) * fc2_row_bytes + Int64(col) * Int64(4)
            a_w = ld_v4(a_addr) + ld_v4(a_addr + Int64(16))
            a_f = [bits_f32(w) for w in a_w]
            local = [pack_bf16x2(a_f[2 * j], a_f[2 * j + 1]) for j in range(4)]
            zero = Int32(0)
            st_v4(a_addr, zero, zero, zero, zero)
            st_v4(a_addr + Int64(16), zero, zero, zero, zero)
            out_of_range = Int32(0)
        elif cutlass.const_expr(self.routed_input == "identity"):
            # routed_moe_into_hop: raw rows t * 8 + k, every slot valid; the finalize without
            # the permutation round trip.
            s_w = ld_v4(scales + Int64(row) * Int64(16))
            weights = []
            for w in s_w:
                lo, hi = unpack_bf16x2(w)
                weights += [lo, hi]
            acc = [Float32(0.0)] * 8
            for k in cutlass.range_constexpr(8):
                x_w = ld_v4(
                    fc2 + (Int64(row) * Int64(8) + Int64(k)) * fc2_row_bytes + Int64(col) * Int64(2)
                )
                for j in cutlass.range_constexpr(4):
                    x_lo, x_hi = unpack_bf16x2(x_w[j])
                    acc[2 * j] = fma(weights[k], x_lo, acc[2 * j])
                    acc[2 * j + 1] = fma(weights[k], x_hi, acc[2 * j + 1])
            local = [pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]
            out_of_range = Int32(0)
        else:
            perm_lo = ld_v4(perm + Int64(row) * Int64(32))
            perm_hi = ld_v4(perm + Int64(row) * Int64(32) + Int64(16))
            perm_k = perm_lo + perm_hi
            if cutlass.const_expr(self.scale_fp32):
                s_lo = ld_v4(scales + Int64(row) * Int64(32))
                s_hi = ld_v4(scales + Int64(row) * Int64(32) + Int64(16))
                weights = [bits_f32(w) for w in s_lo + s_hi]
            else:
                s_w = ld_v4(scales + Int64(row) * Int64(16))
                weights = []
                for w in s_w:
                    lo, hi = unpack_bf16x2(w)
                    weights += [lo, hi]
            acc = [Float32(0.0)] * 8
            out_of_range = Int32(0)
            for k in cutlass.range_constexpr(8):
                valid, safe, oob = perm_check(perm_k[k], fc2_rows)
                out_of_range = out_of_range + oob
                x_w = ld_v4(fc2 + Int64(safe) * fc2_row_bytes + Int64(col) * Int64(2))
                for j in cutlass.range_constexpr(4):
                    x_lo, x_hi = unpack_bf16x2(x_w[j])
                    acc[2 * j] = select_f32(valid, fma(weights[k], x_lo, acc[2 * j]), acc[2 * j])
                    acc[2 * j + 1] = select_f32(
                        valid, fma(weights[k], x_hi, acc[2 * j + 1]), acc[2 * j + 1]
                    )
            local = [pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]
        if cutlass.const_expr(self.has_shared):
            summed = []
            for j in cutlass.range_constexpr(4):
                r_lo, r_hi = unpack_bf16x2(local[j])
                s_lo_f, s_hi_f = unpack_bf16x2(sh_w[j])
                summed.append(pack_bf16x2(add(s_lo_f, r_lo), add(s_hi_f, r_hi)))
            local = summed
        # After an earlier error on this rank, publish NaN until the reset.
        local = [poison(sanitize(w), sticky) for w in local]

        # 2. Publish through multicast into slot [row][rank].
        if cutlass.const_expr(self.fault_delay_ns > 0):
            fault_delay(
                tidx + cta * Int32(1024),
                row + rank * Int32(64),
                epoch * Int32(4),
                self.fault_delay_ns,
                0,
            )
        slot = (
            Int64(cur) * Int64(buffer_bytes)
            + Int64(row) * Int64(world * hidden * 2)
            + Int64(rank) * Int64(hidden * 2)
            + Int64(col) * Int64(2)
        )
        st_v4(mc + slot, local[0], local[1], local[2], local[3])
        _hop_stamp(self.debug_ts, ts, sid, tidx, 2)
        flag_error(err_cur, out_of_range, Int32(ERROR_PERM_RANGE))
        flag_error(err_cur, sticky, sticky)

        # 3. Clear the previous call's Lamport rows and partial slice.
        if cutlass.const_expr(self.fault_delay_ns > 0):
            fault_delay(
                tidx + cta * Int32(1024),
                row + rank * Int32(64),
                epoch * Int32(4) + Int32(1),
                self.fault_delay_ns,
                1,
            )
        total_threads = num_rows * Int32(ctas * threads)
        gtid = (row * Int32(ctas) + cta) * Int32(threads) + tidx
        clear_vecs = prev_rows * Int32(world * hidden // 8)
        sentinel = Int32(SENTINEL_WORD)
        dirty_base = uc + Int64(dirty) * Int64(buffer_bytes)
        for v in cutlass.range(gtid, clear_vecs, total_threads):
            st_v4(dirty_base + Int64(v) * Int64(16), sentinel, sentinel, sentinel, sentinel)
        part_vecs = prev_rows * Int32(MAX_CHUNKS // 4)
        part_base = partials + Int64(dirty) * Int64(self.max_rows * MAX_CHUNKS * 4)
        fp32_sentinel = Int32(FP32_SENTINEL)
        for v in cutlass.range(gtid, part_vecs, total_threads):
            st_v4(
                part_base + Int64(v) * Int64(16),
                fp32_sentinel,
                fp32_sentinel,
                fp32_sentinel,
                fp32_sentinel,
            )

        # 4. Per-thread Lamport poll (timer deadline), rank-ordered fp32 sum.
        poll_base = (
            uc
            + Int64(cur) * Int64(buffer_bytes)
            + Int64(row) * Int64(world * hidden * 2)
            + Int64(col) * Int64(2)
        )
        polled = poll8(poll_base, Int64(hidden * 2), timeout_ns)
        timed_out = polled[32]
        flag_timeout(err_cur, state, timed_out, polled[33])
        _hop_stamp(self.debug_ts, ts, sid, tidx, 3)
        reduced = [Float32(0.0)] * 8
        for r in cutlass.range_constexpr(8):
            for j in cutlass.range_constexpr(4):
                v_lo, v_hi = unpack_bf16x2(polled[4 * r + j])
                reduced[2 * j] = add(reduced[2 * j], v_lo)
                reduced[2 * j + 1] = add(reduced[2 * j + 1], v_hi)
        if cutlass.const_expr(self.use_pdl and not self.trigger_early):
            cute.arch.griddepcontrol_launch_dependents()

        # 5. Residual added once, after the reduction; NaN if this poll timed out.
        new_res = []
        x = []
        for j in cutlass.range_constexpr(4):
            rb = pack_bf16x2(reduced[2 * j], reduced[2 * j + 1])
            a_lo, a_hi = unpack_bf16x2(rb)
            r_lo, r_hi = unpack_bf16x2(res_w[j])
            word = poison(pack_bf16x2(add(a_lo, r_lo), add(a_hi, r_hi)), timed_out)
            new_res.append(word)
            n_lo, n_hi = unpack_bf16x2(word)
            x += [n_lo, n_hi]
        st_v4(residual_out + row_off, new_res[0], new_res[1], new_res[2], new_res[3])

        # 6. Deferred RMSNorm: post_moe_hop's per-thread square sum and warp butterfly,
        #    one fp32 partial per warp; gain-folded activation.
        thread_sum = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            thread_sum = add(thread_sum, round_bf16(mul(x[i], x[i])))
        for mask in (16, 8, 4, 2, 1):
            thread_sum = add(thread_sum, shfl_xor(thread_sum, mask))
        slot_idx = Int64(row) * Int64(ctas * warps) + Int64(cta * Int32(warps) + warp)
        _st_f32_if(sumsq + slot_idx * Int64(4), thread_sum, Int32(lane == Int32(0)))
        # a = bf16(x * gamma) in the MMA B-fragment layout of packet.py: this
        # thread's columns [col, col + 8) are half (col / 8) % 2 of 16-column
        # block col / 16, pair j of lane 4 * row + j.
        blk = col // Int32(16)
        half = (col // Int32(8)) % Int32(2)
        for j in cutlass.range_constexpr(4):
            g_lo, g_hi = unpack_bf16x2(gamma_w[j])
            word = pack_bf16x2(mul(x[2 * j], g_lo), mul(x[2 * j + 1], g_hi))
            lane_slot = blk * Int32(32) + row * Int32(4) + Int32(j)
            _st_b32(scaled + Int64(lane_slot * Int32(2) + half) * Int64(4), word)

        # 7. Arrival; the last CTA publishes the next epoch.
        if cutlass.const_expr(self.fault_delay_ns > 0):
            fault_delay(
                tidx + cta * Int32(1024),
                row + rank * Int32(64),
                epoch * Int32(4) + Int32(2),
                self.fault_delay_ns,
                2,
            )
        cute.arch.sync_threads()
        arrive(
            state,
            Int32(tidx == Int32(0)),
            num_rows * Int32(ctas),
            num_rows,
            (epoch + Int32(1)) % Int32(3),
        )
        _hop_stamp(self.debug_ts, ts, sid, tidx, 4)


class DeferredNormMaterializeKernel:
    """``h = bf16((x * r) * gamma)`` from a deferred packet (post_moe_hop's step 6 arithmetic)."""

    def __init__(self, hidden: int, ctas: int, partials_per_row: int, use_pdl: bool):
        assert hidden % (8 * 32 * ctas) == 0
        self.hidden = hidden
        self.ctas = ctas
        self.threads = hidden // (8 * ctas)
        self.partials_per_row = partials_per_row
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        residual_out: Int64,
        gamma: Int64,
        sumsq: Int64,
        out: Int64,
        anchor: cute.Tensor,
        eps: Float32,
        num_rows: Int32,
        stream: CUstream,
    ):
        self.kernel(residual_out, gamma, sumsq, out, anchor, eps).launch(
            grid=[self.ctas, num_rows, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        residual_out: Int64,
        gamma: Int64,
        sumsq: Int64,
        out: Int64,
        anchor: cute.Tensor,
        eps: Float32,
    ):
        hidden = self.hidden
        tidx, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        col = (cta * Int32(self.threads) + tidx) * Int32(8)
        row_off = Int64(row) * Int64(hidden * 2) + Int64(col) * Int64(2)
        x_w = ld_v4(residual_out + row_off)
        gamma_w = ld_v4(gamma + Int64(col) * Int64(2))
        square_sum = Float32(0.0)
        base = sumsq + Int64(row) * Int64(self.partials_per_row * 4)
        for i in cutlass.range_constexpr(self.partials_per_row):
            square_sum = add(square_sum, _ld_f32(base + Int64(4 * i)))
        r = inv_rms(square_sum, Float32(float(hidden)), eps)
        out_w = []
        for j in cutlass.range_constexpr(4):
            x_lo, x_hi = unpack_bf16x2(x_w[j])
            g_lo, g_hi = unpack_bf16x2(gamma_w[j])
            out_w.append(pack_bf16x2(mul(mul(x_lo, r), g_lo), mul(mul(x_hi, r), g_hi)))
        st_v4(out + row_off, out_w[0], out_w[1], out_w[2], out_w[3])
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


_COMPILED = {}
# Step-1 variants: the raw tuple (None) and routed_moe_into_hop's routed-input modes.
ROUTED_MODES = (None, "row", "identity", "fp32")


def compiled_hop(
    hidden: int,
    top_k: int,
    world: int,
    max_rows: int,
    ctas: int,
    has_shared: bool,
    scale_fp32: bool,
    use_pdl: bool,
    trigger_early: bool,
    state_words: int,
    carveout: int = -1,
    debug_ts: bool = False,
    fault_delay_ns: int = 0,
    routed_input: Optional[str] = None,
):
    """Return the compiled deferred hop for one shape (compiles on first use, no GPU work).

    ``routed_input`` selects a routed_moe_into_hop routed-input variant (``DeferredNormHopKernel``);
    the raw-tuple variants keep their key.
    """
    key = (
        "hop",
        hidden,
        top_k,
        world,
        max_rows,
        ctas,
        has_shared,
        scale_fp32,
        use_pdl,
        trigger_early,
        state_words,
        carveout,
        debug_ts,
        fault_delay_ns,
    ) + ((routed_input,) if routed_input else ())
    compiled = _COMPILED.get(key)
    if compiled is None:
        from cutlass.cute import runtime as cute_runtime

        kernel = DeferredNormHopKernel(
            hidden,
            top_k,
            world,
            max_rows,
            ctas,
            has_shared,
            scale_fp32,
            use_pdl,
            trigger_early,
            carveout,
            debug_ts,
            fault_delay_ns,
            routed_input,
        )
        zero64, zero32 = Int64(0), Int32(0)
        state_fake = cute_runtime.make_fake_compact_tensor(
            Int32, (state_words,), stride_order=(0,), assumed_align=16
        )
        compiled = cute.compile(
            kernel,
            zero64,
            zero64,
            zero32,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            zero64,
            state_fake,
            zero32,
            zero64,
            zero32,
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    return compiled


def compiled_materialize(hidden: int, ctas: int, partials_per_row: int, use_pdl: bool):
    key = ("materialize", hidden, ctas, partials_per_row, use_pdl)
    compiled = _COMPILED.get(key)
    if compiled is None:
        from cutlass.cute import runtime as cute_runtime

        kernel = DeferredNormMaterializeKernel(hidden, ctas, partials_per_row, use_pdl)
        zero64 = Int64(0)
        anchor = cute_runtime.make_fake_compact_tensor(
            Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
        )
        compiled = cute.compile(
            kernel,
            zero64,
            zero64,
            zero64,
            zero64,
            anchor,
            Float32(0.0),
            Int32(0),
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    return compiled


def hop_kernel_supported(hidden: int, top_k: int, world: int, ctas: int) -> Optional[str]:
    """Return why the deferred hop cannot serve a shape, or ``None``."""
    if top_k != 8 or world != 8:
        return "deferred hop specialized to top-8 / TP8"
    if ctas < 1 or ctas > 8 or hidden % (8 * 32 * ctas):
        return f"hidden {hidden} does not split into {ctas} CTAs of whole warps"
    if hidden // (8 * ctas) > 1024:
        return "too many threads per CTA"
    return None
