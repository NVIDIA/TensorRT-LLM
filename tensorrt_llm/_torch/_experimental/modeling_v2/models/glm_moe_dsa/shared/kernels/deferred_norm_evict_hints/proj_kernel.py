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
"""CuTe DSL kernel of the deferred_norm_evict_hints next-layer projection.

``EvictHintsProjKernel`` is deferred_norm's staged projection (``deferred_norm/proj_kernel.py``)
with the same per-warp MMA chains, warp-order reduction and ``r`` scaling (so its
outputs are bit-identical to deferred_norm's), and three schedule knobs:

* ``maxrreg`` / ``maxntid``: a register cap (``ptxas --maxrregcount``, or a
  ``maxntid`` launch bound) so that a projection CTA fits in the register
  file next to a hop CTA: the projection then stages its weights on every
  SM while the hop exchanges, instead of starting on the hop's SMs only when
  the hop exits;
* ``copy_hint="evict_first"``: the weight bulk copies carry an
  ``L2::evict_first`` policy (the weights are read once per iteration), so
  they do not push the hop's Lamport lines out of L2;
* ``split_chunk`` / ``split_delay_ns``: thread 0 issues the copies of K
  chunks ``>= split_chunk`` only ``split_delay_ns`` after its first copy
  (a ``%globaltimer`` spin; the copies still read only immutable weights);
* ``copy_warps``: lane 0 of warp ``w`` issues the copies of rows ``w``,
  ``w + NW``, ... (chunk-major) instead of thread 0 issuing all of them.
  A bulk copy takes uniform operands, so one thread issues its copies one
  at a time; 16 issuing warps stage a CTA's rows at the copy engine's rate.
  Thread 0 still posts every chunk's byte count (the mbarrier transaction
  count may go below zero transiently; the phase completes only after the
  single arrival and all bytes);
* ``prod_warp``: the CTA gets a 17th warp whose lane 0 issues every bulk
  copy (with ``split_chunk`` / ``split_delay_ns``: the chunks below
  ``split_chunk`` at entry, the rest ``split_delay_ns`` after its first
  copy) and then exits. The 16 MMA warps issue nothing, so a deferred
  chunk never holds an MMA warp before its dependency wait; they run deferred_norm's
  body (same K steps, MMA chains, warp-order reduction and ``r``) with the
  two post-loop CTA barriers over the 512 MMA threads only (``bar.sync 1,
  512``). The producer reads only immutable weights and never waits on the
  preceding grid; every chunk's mbarrier is waited on by every MMA warp, so
  the CTA outlives every copy. ``defer_tail`` defers every chunk of the
  last ``defer_tail * R`` CTAs, and ``defer_wait`` issues deferred copies
  after the producer's own ``griddepcontrol.wait`` instead of the timer.
  Debug stamps: slot 7 = first-phase copies issued, slot 6 = every copy
  issued (producer lane 0).

Every change is to the copy issue, the barrier membership or the compile;
every value comes from deferred_norm's instruction sequence.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from cutlass.utils.smem_allocator import SmemAllocator

from ..deferred_norm.proj_kernel import (
    MAX_TOKENS,
    DeferredNormProjKernel,
    _bulk_g2s,
    _ld_b32,
    _ld_shared_f32,
    _ld_v2_if,
    _ldsm_x2,
    _mbar_expect_tx,
    _mbar_fence_init,
    _mbar_init,
    _mbar_wait,
    _min_i32,
    _prefetch_l2,
    _row_src,
    _sel_i32,
    _sel_i64,
    _shfl_idx_f32,
    _st_shared_f32,
    _stamp,
    _store_out,
)
from ..deferred_norm.ptx import add, bits_f32, inv_rms, ldsm_x4, mma_bf16, mul, saddr, st_shared_v4
from .ptx import bar_sync_named, bulk_g2s_hint, globaltimer, policy_evict_first, spin_until, st_b64

COPY_HINTS = ("none", "evict_first")


class EvictHintsProjKernel(DeferredNormProjKernel):
    """deferred_norm's staged projection with deferred_norm_evict_hints's copy-issue knobs (module docstring)."""

    def __init__(
        self,
        maxrreg: int = 0,
        maxntid: int = 0,
        copy_hint: str = "none",
        split_chunk: int = 0,
        split_delay_ns: int = 0,
        copy_warps: bool = False,
        prod_warp: bool = False,
        defer_tail: int = 0,
        defer_wait: bool = False,
        **deferred_norm_cfg,
    ):
        super().__init__(**deferred_norm_cfg)
        assert copy_hint in COPY_HINTS, copy_hint
        if copy_hint != "none" or split_chunk or copy_warps or prod_warp:
            assert self.copy_method == "bulk" and self.copy_threads == 1, (
                "copy hints, split staging and warp-parallel copies need the bulk copy path"
            )
        assert not (copy_warps and split_chunk), "split staging issues from thread 0 only"
        assert prod_warp or not (defer_tail or defer_wait), "deferral needs the producer warp"
        if prod_warp:
            assert not copy_warps, "the producer warp issues every copy"
            assert self.stage_before_wait and not self.wait_staged_first
            assert self.overflow == 0, "the producer warp needs every row staged"
        self.copy_warps = copy_warps
        self.prod_warp = prod_warp
        self.defer_tail = defer_tail
        self.defer_wait = defer_wait
        # Threads per CTA: the MMA warps plus the producer warp.
        self.nt_launch = self.nt + (32 if prod_warp else 0)
        assert 0 <= split_chunk < self.nchunk
        assert maxntid == 0 or maxntid >= self.nt_launch
        self.maxrreg = maxrreg
        self.maxntid = maxntid
        self.copy_hint = copy_hint
        self.split_chunk = split_chunk
        self.split_delay_ns = split_delay_ns

    @cute.jit
    def __call__(
        self,
        act: Int64,
        sumsq: Int64,
        w0: Int64,
        w1: Int64,
        out0: Int64,
        out1: Int64,
        ts: Int64,
        pf0: Int64,
        pf0_bytes: Int64,
        pf1: Int64,
        pf1_bytes: Int64,
        pf2: Int64,
        pf2_bytes: Int64,
        anchor: cute.Tensor,
        rows: Int32,
        eps: Float32,
        stream: CUstream,
    ):
        if cutlass.const_expr(self.prod_warp):
            if cutlass.const_expr(self.maxntid > 0):
                self.kernel_pw(
                    act,
                    sumsq,
                    w0,
                    w1,
                    out0,
                    out1,
                    ts,
                    pf0,
                    pf0_bytes,
                    pf1,
                    pf1_bytes,
                    pf2,
                    pf2_bytes,
                    anchor,
                    rows,
                    eps,
                ).launch(
                    grid=[self.ctas, 1, 1],
                    block=[self.nt_launch, 1, 1],
                    stream=stream,
                    use_pdl=self.use_pdl,
                    max_number_threads=[self.maxntid, 1, 1],
                )
            else:
                self.kernel_pw(
                    act,
                    sumsq,
                    w0,
                    w1,
                    out0,
                    out1,
                    ts,
                    pf0,
                    pf0_bytes,
                    pf1,
                    pf1_bytes,
                    pf2,
                    pf2_bytes,
                    anchor,
                    rows,
                    eps,
                ).launch(
                    grid=[self.ctas, 1, 1],
                    block=[self.nt_launch, 1, 1],
                    stream=stream,
                    use_pdl=self.use_pdl,
                )
        elif cutlass.const_expr(self.maxntid > 0):
            self.kernel(
                act,
                sumsq,
                w0,
                w1,
                out0,
                out1,
                ts,
                pf0,
                pf0_bytes,
                pf1,
                pf1_bytes,
                pf2,
                pf2_bytes,
                anchor,
                rows,
                eps,
            ).launch(
                grid=[self.ctas, 1, 1],
                block=[self.nt, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
                max_number_threads=[self.maxntid, 1, 1],
            )
        else:
            self.kernel(
                act,
                sumsq,
                w0,
                w1,
                out0,
                out1,
                ts,
                pf0,
                pf0_bytes,
                pf1,
                pf1_bytes,
                pf2,
                pf2_bytes,
                anchor,
                rows,
                eps,
            ).launch(
                grid=[self.ctas, 1, 1], block=[self.nt, 1, 1], stream=stream, use_pdl=self.use_pdl
            )

    @cute.jit
    def _copy_chunk(
        self,
        sW: Int32,
        sMbar: Int32,
        w0: Int64,
        w1: Int64,
        row0: Int32,
        nsm: Int32,
        j: cutlass.Constexpr[int],
        policy: Int64,
    ):
        chunk_bytes = Int32(self.chunk_k * 2)
        _mbar_expect_tx(sMbar + Int32(8 * j), nsm * chunk_bytes)
        for i in cutlass.range(nsm):
            dst = sW + i * Int32(self.stride) + Int32(j * self.chunk_k * 2)
            src = _row_src(w0, w1, row0 + i, self.k, self.n0, self.n1) + Int64(j * self.chunk_k * 2)
            if cutlass.const_expr(self.copy_hint == "evict_first"):
                bulk_g2s_hint(dst, src, chunk_bytes, sMbar + Int32(8 * j), policy)
            else:
                _bulk_g2s(dst, src, chunk_bytes, sMbar + Int32(8 * j))

    @cute.jit
    def _copy_one(
        self,
        sW: Int32,
        sMbar: Int32,
        w0: Int64,
        w1: Int64,
        row0: Int32,
        r: Int32,
        j: cutlass.Constexpr[int],
        policy: Int64,
    ):
        chunk_bytes = Int32(self.chunk_k * 2)
        dst = sW + r * Int32(self.stride) + Int32(j * self.chunk_k * 2)
        src = _row_src(w0, w1, row0 + r, self.k, self.n0, self.n1) + Int64(j * self.chunk_k * 2)
        if cutlass.const_expr(self.copy_hint == "evict_first"):
            bulk_g2s_hint(dst, src, chunk_bytes, sMbar + Int32(8 * j), policy)
        else:
            _bulk_g2s(dst, src, chunk_bytes, sMbar + Int32(8 * j))

    @cute.jit
    def _issue_copies(
        self, sW: Int32, sMbar: Int32, w0: Int64, w1: Int64, row0: Int32, nsm: Int32, tid: Int32
    ):
        if cutlass.const_expr(self.copy_warps):
            chunk_bytes = Int32(self.chunk_k * 2)
            if tid == Int32(0):
                for j in cutlass.range_constexpr(self.nchunk):
                    _mbar_expect_tx(sMbar + Int32(8 * j), nsm * chunk_bytes)
            if tid % Int32(32) == Int32(0):
                warp = tid // Int32(32)
                policy = Int64(0)
                if cutlass.const_expr(self.copy_hint == "evict_first"):
                    policy = policy_evict_first()
                per_warp = (self.smem_rows + self.nw - 1) // self.nw
                for j in cutlass.range_constexpr(self.nchunk):
                    for q in cutlass.range_constexpr(per_warp):
                        r = warp + Int32(q * self.nw)
                        if r < nsm:
                            self._copy_one(sW, sMbar, w0, w1, row0, r, j, policy)
        elif cutlass.const_expr(self.copy_hint == "none" and self.split_chunk == 0):
            DeferredNormProjKernel._issue_copies(self, sW, sMbar, w0, w1, row0, nsm, tid)
        else:
            if tid == Int32(0):
                policy = Int64(0)
                if cutlass.const_expr(self.copy_hint == "evict_first"):
                    policy = policy_evict_first()
                t0 = Int64(0)
                if cutlass.const_expr(self.split_chunk > 0):
                    t0 = globaltimer()
                for j in cutlass.range_constexpr(self.nchunk):
                    if cutlass.const_expr(self.split_chunk > 0 and j == self.split_chunk):
                        spin_until(t0 + Int64(self.split_delay_ns))
                    self._copy_chunk(sW, sMbar, w0, w1, row0, nsm, j, policy)

    @cute.jit
    def _pw_stamp(self, ts: Int64, bid: Int32, k: cutlass.Constexpr[int]):
        """Tests only: ``%globaltimer`` of CTA ``bid`` at point ``k`` (producer lane 0)."""
        if cutlass.const_expr(self.debug_ts):
            st_b64(ts + Int64((bid * Int32(8) + Int32(k)) * Int32(8)), globaltimer())

    @cute.jit
    def _produce(
        self,
        sW: Int32,
        sMbar: Int32,
        w0: Int64,
        w1: Int64,
        row0: Int32,
        nsm: Int32,
        ts: Int64,
        bid: Int32,
        lane: Int32,
        rows: Int32,
    ):
        """Producer warp: lane 0 posts each chunk's byte count and issues its
        copies. Deferred copies: the chunks ``>= split_chunk`` of every CTA,
        and every chunk of the last ``defer_tail * rows`` CTAs (the CTA count
        of the preceding hop at ``defer_tail`` CTAs per token row). They are
        issued after ``split_delay_ns`` (``%globaltimer``) or, with
        ``defer_wait``, after the producer's own ``griddepcontrol.wait`` (the
        preceding grid's completion; the producer still reads no data it
        wrote). Every lane executes the PDL trigger no later than deferred_norm's
        thread 0 would (after the copies issued at entry)."""
        if lane == Int32(0):
            policy = Int64(0)
            if cutlass.const_expr(self.copy_hint == "evict_first"):
                policy = policy_evict_first()
            t0 = globaltimer()
            late = Int32(0)
            if cutlass.const_expr(self.defer_tail > 0):
                late = Int32(bid >= Int32(self.ctas) - Int32(self.defer_tail) * rows)
            first = self.split_chunk if self.split_chunk > 0 else self.nchunk
            if late == Int32(0):
                for j in cutlass.range_constexpr(first):
                    self._copy_chunk(sW, sMbar, w0, w1, row0, nsm, j, policy)
            self._pw_stamp(ts, bid, 7)
            if cutlass.const_expr(self.use_pdl and self.trigger == "start"):
                cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(self.split_chunk > 0 or self.defer_tail > 0):
                if cutlass.const_expr(self.split_chunk > 0):
                    self._defer_point(t0)
                    for j in cutlass.range_constexpr(self.nchunk):
                        if cutlass.const_expr(j < first):
                            if late != Int32(0):
                                self._copy_chunk(sW, sMbar, w0, w1, row0, nsm, j, policy)
                        else:
                            self._copy_chunk(sW, sMbar, w0, w1, row0, nsm, j, policy)
                else:
                    if late != Int32(0):
                        self._defer_point(t0)
                        for j in cutlass.range_constexpr(self.nchunk):
                            self._copy_chunk(sW, sMbar, w0, w1, row0, nsm, j, policy)
            self._pw_stamp(ts, bid, 6)
        else:
            if cutlass.const_expr(self.use_pdl and self.trigger == "start"):
                cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _defer_point(self, t0: Int64):
        """Where the producer issues its deferred copies (bounded either way:
        the timer, or the preceding grid's completion)."""
        if cutlass.const_expr(self.defer_wait and self.use_pdl):
            cute.arch.griddepcontrol_wait()
        else:
            spin_until(t0 + Int64(self.split_delay_ns))

    @cute.jit
    def _compute(
        self,
        act: Int64,
        sumsq: Int64,
        w0: Int64,
        w1: Int64,
        out0: Int64,
        out1: Int64,
        ts: Int64,
        pf0: Int64,
        pf0_bytes: Int64,
        pf1: Int64,
        pf1_bytes: Int64,
        pf2: Int64,
        pf2_bytes: Int64,
        rows: Int32,
        eps: Float32,
        tid: Int32,
        bid: Int32,
        warp: Int32,
        lane: Int32,
        sW: Int32,
        sMisc: Int32,
        row0: Int32,
        nrows: Int32,
        nsm: Int32,
    ):
        """The 16 MMA warps: deferred_norm's kernel body after its copy issue, with the
        post-loop barriers over the MMA threads only (no overflow row)."""
        k = self.k
        nw = self.nw
        stride = self.stride
        sZero = sMisc
        sMbar = sMisc + Int32(64)
        sR = sMisc + Int32(256)
        if cutlass.const_expr(self.prefetch_next):
            if tid < Int32(3):
                pf_addr = _sel_i64(
                    Int32(tid == Int32(0)), pf0, _sel_i64(Int32(tid == Int32(1)), pf1, pf2)
                )
                pf_bytes = _sel_i64(
                    Int32(tid == Int32(0)),
                    pf0_bytes,
                    _sel_i64(Int32(tid == Int32(1)), pf1_bytes, pf2_bytes),
                )
                self._prefetch_slice(pf_addr, pf_bytes, bid)
        if cutlass.const_expr(self.l2_prefetch):
            # Rows that are not staged (none: the producer variant stages every row).
            if tid >= nsm:
                if tid < nrows:
                    _prefetch_l2(_row_src(w0, w1, row0 + tid, k, self.n0, self.n1), Int32(k * 2))
        if cutlass.const_expr(self.use_pdl and self.trigger == "start"):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        _stamp(self.debug_ts, ts, bid, tid, 1)

        g = lane // Int32(4)
        t4 = lane % Int32(4)
        tok_ok = Int32(g < rows)
        safe_g = _min_i32(g, rows - Int32(1))
        pq = []
        if cutlass.const_expr(self.deferred):
            quarter = self.partials_per_row // 4
            p_ok = Int32(warp == Int32(nw - 1)) * tok_ok
            pbase = sumsq + Int64(
                (g * Int32(self.partials_per_row) + t4 * Int32(quarter)) * Int32(4)
            )
            for j in cutlass.range_constexpr(quarter // 2):
                lo, hi = _ld_v2_if(pbase + Int64(8 * j), p_ok)
                pq.append(bits_f32(lo))
                pq.append(bits_f32(hi))
        bfr = []
        if cutlass.const_expr(self.b_layout == "frag"):
            a_lane = act + Int64(lane * Int32(8))
            for m in cutlass.range_constexpr(self.rounds):
                bfr.append(_ld_v2_if(a_lane + Int64((warp + Int32(nw * m)) * Int32(256)), tok_ok))
        else:
            a_row = act + Int64(safe_g * Int32(k) * Int32(2) + t4 * Int32(4))
            for m in cutlass.range_constexpr(self.rounds):
                off = Int64((warp + Int32(nw * m)) * Int32(32))
                b0 = _ld_b32(a_row + off)
                b1 = _ld_b32(a_row + off + Int64(16))
                bfr.append((_sel_i32(tok_ok, b0, Int32(0)), _sel_i32(tok_ok, b1, Int32(0))))
        _stamp(self.debug_ts, ts, bid, tid, 2)

        r16 = lane % Int32(16)
        a_lane0 = sW + r16 * Int32(stride) + (lane // Int32(16)) * Int32(16)
        ok0 = Int32(r16 < nsm)
        r8 = Int32(16) + lane % Int32(8)
        a_lane1 = sW + r8 * Int32(stride) + ((lane // Int32(8)) % Int32(2)) * Int32(16)
        ok1 = Int32(r8 < nsm)

        acc0 = [Float32(0.0)] * 4
        acc1 = [Float32(0.0)] * 4
        zero = Int32(0)
        rpc = self.rounds_per_chunk

        def load_a(n):
            if cutlass.const_expr(n % rpc == 0):
                _mbar_wait(sMbar + Int32(8 * (n // rpc)), Int32(0))
            kb = (warp + Int32(nw * n)) * Int32(32)
            return (
                ldsm_x4(_sel_i32(ok0, a_lane0 + kb, sZero)),
                _ldsm_x2(_sel_i32(ok1, a_lane1 + kb, sZero)),
            )

        frags = [load_a(n) for n in range(min(self.dist, self.rounds))]
        for m in cutlass.range_constexpr(self.rounds):
            if cutlass.const_expr(m + self.dist < self.rounds):
                frags.append(load_a(m + self.dist))
            fa, fb = frags[m]
            b0, b1 = bfr[m]
            acc0[0], acc0[1], acc0[2], acc0[3] = mma_bf16(
                fa[0], fa[1], fa[2], fa[3], b0, b1, acc0[0], acc0[1], acc0[2], acc0[3]
            )
            acc1[0], acc1[1], acc1[2], acc1[3] = mma_bf16(
                fb[0], zero, fb[1], zero, b0, b1, acc1[0], acc1[1], acc1[2], acc1[3]
            )
        _stamp(self.debug_ts, ts, bid, tid, 3)

        if cutlass.const_expr(self.deferred):
            if warp == Int32(nw - 1):
                s = Float32(0.0)
                for q in cutlass.range_constexpr(4):
                    mine = s
                    for v in pq:
                        mine = add(mine, v)
                    s = _shfl_idx_f32(mine, g * Int32(4) + Int32(q))
                if t4 == Int32(0):
                    _st_shared_f32(sR + g * Int32(4), inv_rms(s, Float32(float(k)), eps))
        else:
            if tid < Int32(MAX_TOKENS):
                _st_shared_f32(sR + tid * Int32(4), Float32(1.0))

        # Cross-warp reduction in warp order (every MMA warp has waited on
        # every chunk, so every copy has landed before a row is overwritten).
        bar_sync_named(1, self.nt)
        sRed = sW
        wbase = sRed + warp * Int32(24 * MAX_TOKENS * 4)
        c0 = Int32(2) * t4
        _st_shared_f32(wbase + (g * Int32(MAX_TOKENS) + c0) * Int32(4), acc0[0])
        _st_shared_f32(wbase + (g * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc0[1])
        _st_shared_f32(wbase + ((g + Int32(8)) * Int32(MAX_TOKENS) + c0) * Int32(4), acc0[2])
        _st_shared_f32(
            wbase + ((g + Int32(8)) * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc0[3]
        )
        _st_shared_f32(wbase + ((g + Int32(16)) * Int32(MAX_TOKENS) + c0) * Int32(4), acc1[0])
        _st_shared_f32(
            wbase + ((g + Int32(16)) * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc1[1]
        )
        bar_sync_named(1, self.nt)
        if tid < Int32(24 * MAX_TOKENS):
            row = tid // Int32(MAX_TOKENS)
            tok = tid % Int32(MAX_TOKENS)
            if row < nsm:
                if tok < rows:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(nw):
                        s = add(
                            s,
                            _ld_shared_f32(sRed + Int32(w * 24 * MAX_TOKENS * 4) + tid * Int32(4)),
                        )
                    _store_out(
                        out0,
                        out1,
                        row0 + row,
                        tok,
                        mul(_ld_shared_f32(sR + tok * Int32(4)), s),
                        self.n0,
                        self.n1,
                    )
        _stamp(self.debug_ts, ts, bid, tid, 4)
        if cutlass.const_expr(self.use_pdl and self.trigger == "end"):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel_pw(
        self,
        act: Int64,
        sumsq: Int64,
        w0: Int64,
        w1: Int64,
        out0: Int64,
        out1: Int64,
        ts: Int64,
        pf0: Int64,
        pf0_bytes: Int64,
        pf1: Int64,
        pf1_bytes: Int64,
        pf2: Int64,
        pf2_bytes: Int64,
        anchor: cute.Tensor,
        rows: Int32,
        eps: Float32,
    ):
        """deferred_norm's projection with a producer warp (``prod_warp``; module docstring)."""
        stride = self.stride
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        _stamp(self.debug_ts, ts, bid, tid, 0)

        smem = SmemAllocator()
        sW = saddr(smem.allocate(self.smem_rows * stride, byte_alignment=128))
        sMisc = saddr(smem.allocate(512, byte_alignment=128))
        sMbar = sMisc + Int32(64)

        base = Int32(self.base_rows)
        rem = Int32(self.rem_rows)
        extra = Int32(bid < rem)
        nrows = base + extra
        row0 = bid * base + _min_i32(bid, rem)
        nsm = _min_i32(nrows, Int32(self.smem_rows))

        if tid < Int32(2):
            st_shared_v4(sMisc + tid * Int32(16), Int32(0), Int32(0), Int32(0), Int32(0))
        if tid == Int32(0):
            for j in cutlass.range_constexpr(self.nchunk + 1):
                _mbar_init(sMbar + Int32(8 * j), Int32(1))
            _mbar_fence_init()
        cute.arch.sync_threads()

        if warp == Int32(self.nw):
            self._produce(sW, sMbar, w0, w1, row0, nsm, ts, bid, lane, rows)
        else:
            self._compute(
                act,
                sumsq,
                w0,
                w1,
                out0,
                out1,
                ts,
                pf0,
                pf0_bytes,
                pf1,
                pf1_bytes,
                pf2,
                pf2_bytes,
                rows,
                eps,
                tid,
                bid,
                warp,
                lane,
                sW,
                sMisc,
                row0,
                nrows,
                nsm,
            )
        _stamp(self.debug_ts, ts, bid, tid, 5)


_COMPILED = {}


def is_compiled(**cfg) -> bool:
    return tuple(sorted(cfg.items())) in _COMPILED


def compiled_evict_hints_projection(**cfg):
    """Return the compiled deferred_norm_evict_hints projection (deferred_norm keys plus
    ``projection.EVICT_HINTS_CONFIG_KEYS``); compiles once."""
    key = tuple(sorted(cfg.items()))
    compiled = _COMPILED.get(key)
    if compiled is None:
        from cutlass.cute import runtime as cute_runtime

        kernel = EvictHintsProjKernel(**cfg)
        options = "--enable-tvm-ffi"
        if kernel.maxrreg:
            options += f" --ptxas-options=--maxrregcount={kernel.maxrreg}"
        z64 = Int64(0)
        anchor = cute_runtime.make_fake_compact_tensor(
            Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
        )
        compiled = cute.compile(
            kernel,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            anchor,
            Int32(1),
            Float32(1e-5),
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=options,
        )
        _COMPILED[key] = compiled
    return compiled
