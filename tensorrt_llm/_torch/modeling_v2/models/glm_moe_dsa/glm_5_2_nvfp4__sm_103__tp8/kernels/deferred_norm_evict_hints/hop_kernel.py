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
"""CuTe DSL kernel of the deferred_norm_evict_hints deferred-norm post-MoE hop.

``EvictHintsHopKernel`` is deferred_norm's ``DeferredNormHopKernel``
(``deferred_norm/hop_kernel.py``) with the same instructions for steps 1-7 on the same post_moe_hop
workspace (so its packet, residual and materialization are bit-identical to deferred_norm's), plus
L2 cache hints that keep the hop's Lamport lines out of the next layer's weight
stream:

* ``poll_prefetch``: ``cp.async.bulk.prefetch.L2`` of the CTA's poll slice
  (8 rank slots x ``threads * 16`` bytes of its row):

  - ``"cur"``: the current buffer, after the state words are loaded;
  - ``"wait3"``: the slice of all three Lamport buffers right after the
    dependency wait (no state needed), before the PDL trigger;
  - ``"pre3"``: the same before the dependency wait (exception arm: the
    addresses are producer-owned; a prefetch returns no value);
  - ``"tcur"`` / ``"t3"`` / ``"tpre3"``: the same three placements with a
    per-thread ``prefetch.global.L2`` of the thread's own 8 poll addresses
    (per buffer) instead of the bulk prefetch, whose uniform operands make
    the compiler serialize one issue per thread in warp 0;

* ``clear_hint="evict_last"``: the clear of the previous call's buffer
  (which is the buffer of call n + 2) stores with an ``L2::evict_last``
  policy;
* ``carveout``: the preferred shared-memory carveout of the launch, so an
  SM that holds a hop CTA can also take a projection CTA;
* ``dyn_smem``: an unused dynamic shared-memory allocation of the launch
  (at most ``MAX_DYN_SMEM`` bytes, so that a projection CTA still fits next
  to the hop CTA in the 228 KB configuration), for drivers that apply the
  carveout only to a kernel that uses shared memory.

None of these changes a value or the order of the hop's memory operations.
``debug_ts`` adds the CTA's ``%smid`` in stamp slot 5.

The carveout is a CUDA kernel attribute. The DSL records
``preferred_smem_carveout`` but its CUDA-library loader does not apply it,
so a DSL kernel runs with the driver's default configuration (32 KB of
shared memory on the hop's SMs; a projection CTA, which needs the 228 KB
configuration, cannot start there until the hop CTA exits).
``compiled_evict_hints_hop`` therefore sets ``CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_
MEMORY_CARVEOUT`` itself on every kernel of the compiled function's CUDA
library for the current device (``set_library_carveout``).
``ffi=False`` compiles ``call_raw`` (the state words and the stream passed
explicitly) for the DSL executor instead of the TVM-FFI launcher.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64

from ..deferred_norm.hop_kernel import (
    ROUTED_MODES,
    DeferredNormHopKernel,
    _hop_stamp,
    _st_b32,
    _st_f32_if,
)
from ..deferred_norm.ptx import (
    FP32_SENTINEL,
    SENTINEL_WORD,
    add,
    arrive,
    bits_f32,
    error_word_address,
    fault_delay,
    flag_error,
    flag_timeout,
    fma,
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
from ..post_moe_hop.kernels import ERROR_PERM_RANGE, MAX_CHUNKS
from .ptx import policy_evict_last, prefetch_l2, prefetch_line_l2, smid, st_b64, st_v4_hint

POLL_PREFETCH_MODES = ("none", "cur", "wait3", "pre3", "tcur", "t3", "tpre3")
CLEAR_HINTS = ("none", "evict_last")
# 228 KB configuration - (projection CTA 222,080 B + 1 KB reserved) - 1 KB reserved.
MAX_DYN_SMEM = 233472 - (222080 + 1024) - 1024


@cute.jit
def _smid_stamp(enabled: cutlass.Constexpr[bool], ts: Int64, sid: Int32, tid: Int32):
    if cutlass.const_expr(enabled):
        if tid == Int32(0):
            st_b64(ts + Int64((sid * Int32(8) + Int32(5)) * Int32(8)), Int64(smid()))


class EvictHintsHopKernel(DeferredNormHopKernel):
    """deferred_norm's deferred hop with deferred_norm_evict_hints's L2 hints (module docstring)."""

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
        poll_prefetch: str = "none",
        clear_hint: str = "none",
        dyn_smem: int = 0,
    ):
        super().__init__(
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
        assert poll_prefetch in POLL_PREFETCH_MODES, poll_prefetch
        assert clear_hint in CLEAR_HINTS, clear_hint
        assert 0 <= dyn_smem <= MAX_DYN_SMEM and dyn_smem % 16 == 0, dyn_smem
        self.dyn_smem = dyn_smem
        self.poll_prefetch = poll_prefetch
        self.clear_hint = clear_hint

    @cute.jit
    def _prefetch_poll(self, uc: Int64, buf: Int32, row: Int32, cta: Int32, slot: Int32):
        """L2 prefetch of rank slot ``slot`` of this CTA's columns in buffer ``buf``."""
        hidden = self.hidden
        base = (
            uc
            + Int64(buf) * Int64(self.max_rows * self.world * hidden * 2)
            + Int64(row) * Int64(self.world * hidden * 2)
            + Int64(slot) * Int64(hidden * 2)
            + Int64(cta) * Int64(self.threads * 16)
        )
        prefetch_l2(base, Int32(self.threads * 16))

    @cute.jit
    def _prefetch_own(self, uc: Int64, buf: Int32, row: Int32, cta: Int32, tidx: Int32):
        """Per-thread L2 prefetch of this thread's 8 poll addresses in buffer ``buf``."""
        hidden = self.hidden
        base = (
            uc
            + Int64(buf) * Int64(self.max_rows * self.world * hidden * 2)
            + Int64(row) * Int64(self.world * hidden * 2)
            + Int64((cta * Int32(self.threads) + tidx) * Int32(16))
        )
        for r in cutlass.range_constexpr(8):
            prefetch_line_l2(base + Int64(r * hidden * 2))

    @cute.jit
    def _prefetch_own3(self, uc: Int64, row: Int32, cta: Int32, tidx: Int32):
        for b in cutlass.range_constexpr(3):
            self._prefetch_own(uc, Int32(b), row, cta, tidx)

    @cute.jit
    def _prefetch_three(self, uc: Int64, row: Int32, cta: Int32, tidx: Int32):
        """Threads 0..23: slot ``tidx % 8`` of buffer ``tidx // 8``."""
        if tidx < Int32(24):
            self._prefetch_poll(uc, tidx // Int32(8), row, cta, tidx % Int32(8))

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
        """TVM-FFI entry (``state`` as a GPU tensor binds the caller's stream)."""
        self._launch_kernel(
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
            Int64(state.iterator.toint()),
            rank,
            timeout_ns,
            num_rows,
            stream,
        )

    @cute.jit
    def call_raw(
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
        state: Int64,
        rank: Int32,
        timeout_ns: Int64,
        num_rows: Int32,
        stream: CUstream,
    ):
        """Executor entry (state-word address and stream explicit)."""
        self._launch_kernel(
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
            stream,
        )

    @cute.jit
    def _launch_kernel(
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
        state: Int64,
        rank: Int32,
        timeout_ns: Int64,
        num_rows: Int32,
        stream: CUstream,
    ):
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
                smem=self.dyn_smem if self.dyn_smem else None,
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
                smem=self.dyn_smem if self.dyn_smem else None,
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
        state: Int64,
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

        sid = row * Int32(ctas) + cta
        _hop_stamp(self.debug_ts, ts, sid, tidx, 0)
        _smid_stamp(self.debug_ts, ts, sid, tidx)
        if cutlass.const_expr(self.poll_prefetch == "pre3"):
            self._prefetch_three(uc, row, cta, tidx)
        if cutlass.const_expr(self.poll_prefetch == "tpre3"):
            self._prefetch_own3(uc, row, cta, tidx)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.poll_prefetch == "wait3"):
                self._prefetch_three(uc, row, cta, tidx)
            if cutlass.const_expr(self.poll_prefetch == "t3"):
                self._prefetch_own3(uc, row, cta, tidx)
            if cutlass.const_expr(self.trigger_early):
                cute.arch.griddepcontrol_launch_dependents()
        else:
            if cutlass.const_expr(self.poll_prefetch == "wait3"):
                self._prefetch_three(uc, row, cta, tidx)
            if cutlass.const_expr(self.poll_prefetch == "t3"):
                self._prefetch_own3(uc, row, cta, tidx)
        _hop_stamp(self.debug_ts, ts, sid, tidx, 1)

        # State words of the previous call (complete at the kernel boundary).
        epoch, prev_rows, err_words = load_state(state)
        cur = epoch % Int32(3)
        dirty = (epoch + Int32(2)) % Int32(3)
        if cutlass.const_expr(self.poll_prefetch == "cur"):
            if tidx < Int32(8):
                self._prefetch_poll(uc, cur, row, cta, tidx)
        if cutlass.const_expr(self.poll_prefetch == "tcur"):
            self._prefetch_own(uc, cur, row, cta, tidx)
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
            r_w = ld_v4(fc2 + Int64(row) * fc2_row_bytes + Int64(col) * Int64(2))
            local = [r_w[j] for j in range(4)]
            out_of_range = Int32(0)
        elif cutlass.const_expr(self.routed_input == "fp32"):
            a_addr = fc2 + Int64(row) * fc2_row_bytes + Int64(col) * Int64(4)
            a_w = ld_v4(a_addr) + ld_v4(a_addr + Int64(16))
            a_f = [bits_f32(w) for w in a_w]
            local = [pack_bf16x2(a_f[2 * j], a_f[2 * j + 1]) for j in range(4)]
            zero = Int32(0)
            st_v4(a_addr, zero, zero, zero, zero)
            st_v4(a_addr + Int64(16), zero, zero, zero, zero)
            out_of_range = Int32(0)
        elif cutlass.const_expr(self.routed_input == "identity"):
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
        if cutlass.const_expr(self.clear_hint == "evict_last"):
            policy = policy_evict_last()
            for v in cutlass.range(gtid, clear_vecs, total_threads):
                st_v4_hint(
                    dirty_base + Int64(v) * Int64(16),
                    sentinel,
                    sentinel,
                    sentinel,
                    sentinel,
                    policy,
                )
        else:
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

        # 6. Deferred RMSNorm: one fp32 partial per warp; gain-folded activation.
        thread_sum = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            thread_sum = add(thread_sum, round_bf16(mul(x[i], x[i])))
        for mask in (16, 8, 4, 2, 1):
            thread_sum = add(thread_sum, shfl_xor(thread_sum, mask))
        slot_idx = Int64(row) * Int64(ctas * warps) + Int64(cta * Int32(warps) + warp)
        _st_f32_if(sumsq + slot_idx * Int64(4), thread_sum, Int32(lane == Int32(0)))
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


_COMPILED = {}


def compiled_evict_hints_hop(
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
    poll_prefetch: str = "none",
    clear_hint: str = "none",
    ffi: bool = True,
    dyn_smem: int = 0,
):
    """Return the compiled deferred_norm_evict_hints hop for one shape (compiles on first use, no GPU work).

    ``carveout >= 0`` is set on the compiled kernels (module docstring).
    ``ffi=False`` compiles ``call_raw`` for the DSL's executor; call it with
    ``(..., state_address, rank, timeout_ns, num_rows, CUstream)``.
    """
    assert routed_input in ROUTED_MODES, routed_input
    key = (
        "evict_hints_hop",
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
        routed_input,
        poll_prefetch,
        clear_hint,
        ffi,
        dyn_smem,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        from cutlass.cute import runtime as cute_runtime

        kernel = EvictHintsHopKernel(
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
            poll_prefetch,
            clear_hint,
            dyn_smem,
        )
        zero64, zero32 = Int64(0), Int32(0)
        if ffi:
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
        else:
            import torch

            stream = CUstream(torch.cuda.current_stream().cuda_stream)
            compiled = cute.compile(
                kernel.call_raw,
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
                zero64,
                zero32,
                zero64,
                zero32,
                stream,
            )
        if carveout >= 0:
            LIBRARY_CARVEOUT[key] = set_library_carveout(compiled, carveout)
        _COMPILED[key] = compiled
    return compiled


# compile key -> number of kernels whose carveout was set, or the reason it was not.
LIBRARY_CARVEOUT = {}


def set_library_carveout(compiled, carveout: int):
    """Set the preferred shared-memory carveout of every kernel in the CUDA
    library of a compiled DSL function, for the current device.

    Returns the number of kernels set, or a string with the reason the
    library could not be reached (the kernels then keep the driver default).
    """
    import torch
    from cuda.bindings import driver as cu

    try:
        library = compiled.library
    except (AttributeError, RuntimeError) as exc:
        return f"no CUDA library: {exc!r}"[:200]
    lib = cu.CUlibrary(int(library))
    err, count = cu.cuLibraryGetKernelCount(lib)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    err, kernels = cu.cuLibraryEnumerateKernels(count, lib)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    device = cu.CUdevice(torch.cuda.current_device())
    attr = cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    for kernel in kernels:
        (err,) = cu.cuKernelSetAttribute(attr, carveout, kernel, device)
        assert err == cu.CUresult.CUDA_SUCCESS, err
    return len(kernels)
