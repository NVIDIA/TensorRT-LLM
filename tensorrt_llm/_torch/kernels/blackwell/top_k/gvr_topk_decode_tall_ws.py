# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent warp-specialized GVR top-K decode for tall K = 512 / 1024 / 2048 batches (Hopper).

The host routes tall decode batches with long rows here on 132-SM Hopper parts
(``gvr_topk_decode_self_sampling_host._tallws_gate``). Each of the ``grid_of(rows, SMs)`` CTAs
owns the rows ``blockIdx.x + i * gridDim.x``, so one CTA streams its next row while it selects
the current one; CTAs never wait on each other. A CTA has 576 threads in three roles:

* warp 0, the producer: one lane streams the CTA's rows into a ring of 32 KB shared-memory
  stages (four; three for FP32 at K <= 1024, whose sample buffer is larger) with 1-D
  ``cp.async.bulk`` copies under an L2 evict_first policy. A stage's ``full`` mbarrier counts its
  transaction bytes, its ``empty`` mbarrier one arrive per consumer warp, and the ring state runs
  on across rows. Rows stream back to front: their survivors cluster at the row end (recent
  tokens), so the emit-heavy chunks arrive while the ring is still filling.
* warp 1, the planner: one row ahead of the consumers it gathers the row's strided 16-B samples
  (the sampling ladder of ``GvrMainKernel``), builds a 256-bin sample histogram, derives the
  rungs (threshold T, histogram cut HIC, retry rung t5, the sampled K-th code tkc) and the first
  attempt's window, and publishes them through two mbarrier-guarded slots. Row 0's first gathers
  leave before the producer's first ring loads (named barrier ``BAR_EI``), and the consumers wait
  for row 0's slot on the named barrier ``BAR_S0``, which the planner passes once it published it.
* warps 2-17, 512 consumers: per row they classify every stage against the window, stage each
  survivor's {value, index} and count it in the window histogram, and verify that the window
  holds K scores; a short window retries at the next rung straight from global memory while the
  ring streams the next row. They then select with the BLK = 512 staged path of
  ``GvrMainKernel`` (cursor emit, crossing-bin refinement, degenerate rows) and copy the output
  row out with 16-B stores. Every CTA-wide barrier of that path is the named barrier
  ``BAR_CONS`` over the consumers.

Window bins: at K = 2048 BF16 one 16-bit code per bin over 256 bins and FP16 one code per bin
over 2,048 bins anchored 1,023 codes below the sampled K-th; FP32 and K <= 1024 256 linear bins.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute import runtime as _crt
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from . import gvr_topk_decode_self_sampling as C
from .gvr_topk_decode_self_sampling import (
    _NEG_INF,
    GvrMainKernel,
    QUADC_CLUS__main,
    _bf16_emit_prefix_agg,
    _dtype_of,
    _fmaf,
    _ld_g_nc_u16,
    _lds_v4_u32,
    _smem_addr_reg,
    _st_g_v4_b32,
    bf16x2_ge_or,
    ceil16x2_of_f32,
    f32_of_u32,
    fkey16_of_code16,
    ge16x2_or,
    key16_of_f32,
    key16_of_pair16,
    key16_rz_of_f32,
    ldg_16,
    ldg_16_code,
    widen_f16,
    widen_f16_hi,
)

WS_KS = (512, 1024, 2048)  # the top-K values the engine serves
WS_NCT = 512  # consumer threads (the BLK = 512 selection)
WS_NCW = WS_NCT // 32  # consumer warps: the arrive count of a ring stage's empty mbarrier
WS_SCPB = 8192  # survivor staging slots ({value bits, idx}, 8 B); the ladder clamps aim to SCPB / 2
WS_CMPB = 1024  # crossing-bin candidate slots (s_ck64)
WS_STAGE_BYTES = 32768  # bytes per ring stage (one stream chunk per stage)
# Sampling ladder per K (aim_base, sfac, sd_en): aim 2K with sample factor 64 (32 at K = 512);
# at K = 2048 rows of 2K < n <= SCPB scores are sampled as well.
WS_LADDER = {512: (1024, 32, 0), 1024: (2048, 64, 0), 2048: (4096, 64, 1)}
# The ladder's largest sample set (samples of 8 scores) over rows of up to 262,144 scores: the
# planner's sample buffer holds it, so such rows gather in one round; longer rows gather in rounds.
_WS_MAX_SAMPLES = {512: 853, 1024: 1043, 2048: 512}
NB_PLAN = 256  # planner sample-histogram bins
BAR_CONS = 1  # consumer named barrier (512 threads); barrier 0 is the CTA prologue only
BAR_S0 = 2  # planner warp + consumers: the planner published row 0's slot
BAR_EI = 3  # producer + planner warps: row 0's sample gathers leave before the first ring loads
WS_SMEM_MAX = 227 * 1024  # opt-in dynamic shared memory per block
_PRODUCER_WARP = 0
_PLANNER_WARP = 1
_CONSUMER_WARP0 = 2

# s_lad slot words (per row, written by the planner, read by the consumers)
L_N, L_NROW, L_SHORT, L_T, L_HIC, L_TKC, L_T5, L_TF, L_SC, L_ANCH = range(10)
LADW = 16

_SENT_LO = C.SENT_LO
_SENT_HI = C.SENT_HI


@dsl_user_op
def _lds_u16(saddr, *, loc=None, ip=None):
    """ld.shared.u16 at a plain Int32 byte address, zero-extended to Uint32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [saddr.ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 h; ld.shared.u16 h, [$1]; cvt.u32.u16 $0, h; }",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bulk_g2s_hint(saddr, gaddr, nbytes, mbar, pol, *, loc=None, ip=None):
    """1-D bulk copy of ``nbytes`` (a multiple of 16) from global to shared memory under the L2
    cache policy ``pol``, completing ``nbytes`` of transaction count on the mbarrier at shared
    address ``mbar``."""
    llvm.inline_asm(
        res=None,
        operands_=[v.ir_value(loc=loc, ip=ip) for v in (saddr, gaddr, nbytes, mbar, pol)],
        asm_string=(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [$0], [$1], $2, [$3], $4;"
        ),
        constraints="r,l,r,r,l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_u32(saddr, *, loc=None, ip=None):
    """ld.shared.u32 at a plain Int32 byte address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [saddr.ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _bar_cons():
    """Consumer named barrier: every CTA-wide barrier of the ported selection."""
    cute.arch.barrier(barrier_id=BAR_CONS, number_of_threads=WS_NCT)


@cute.jit
def _scan_cross_w_cons(s_hist, s_ws, target, ctid, s_res, nb: cutlass.Constexpr):
    """Block suffix scan over ``nb`` >= 512 bins (``nb / 512`` per consumer thread) on the consumer
    barrier: leaves ``s_hist[j]`` = the count in bins above j (the emit cursor) and pins the
    crossing of ``target`` in ``s_res`` (B, above, m, tot). One internal consumer barrier; the
    caller pays the barrier that publishes the result."""
    BPT = nb // WS_NCT
    assert BPT in (2, 4), "vectorized per-thread bin span"
    NW = WS_NCT // 32
    lane = ctid & Int32(31)
    wid = ctid >> Int32(5)
    loc = cute.make_rmem_tensor((BPT,), Int32)
    base = ctid * Int32(BPT)
    sm = Int32(0)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=32 * BPT)
    hist_span = cute.make_tensor(
        cute.make_ptr(
            Int32,
            s_hist.iterator.toint() + base * Int32(4),
            cute.AddressSpace.smem,
            assumed_align=4 * BPT,
        ),
        cute.make_layout((BPT,)),
    )
    cute.copy(atom, hist_span, loc)
    for i in cutlass.range_constexpr(BPT):
        sm = sm + loc[i]
    w = C.warp_incl_scan_add(sm, lane)
    if lane == Int32(31):
        s_ws[wid] = w
    _bar_cons()  # ---- the one internal barrier ----
    vv = Int32(0)
    if lane < Int32(NW):
        vv = s_ws[lane]
    tot = Int32(C.warp_add_i32(vv))
    sel = Int32(0)
    if lane < wid:
        sel = vv
    off = Int32(C.warp_add_i32(sel))
    after = tot - (off + w)
    if ctid == Int32(0):
        s_res[C.RES_TOT] = tot
    for i in cutlass.range_constexpr(BPT - 1, -1, -1):
        cq = loc[i]
        loc[i] = after
        gb = base + Int32(i)
        cross = Int32(0)
        if after < target:
            if (after + cq) >= target:
                cross = Int32(1)
            if gb == Int32(0):
                cross = Int32(1)
        if cross != Int32(0):
            s_res[C.RES_B] = gb
            s_res[C.RES_ABOVE] = after
            s_res[C.RES_M] = cq
        after = after + cq
    cute.copy(atom, loc, hist_span)


class GvrTallWsKernel(GvrMainKernel):
    """Persistent warp-specialized top-K decode for tall batches (see the module docstring).

    Args:
        dtype: cutlass.Float32, cutlass.BFloat16 or cutlass.Float16 logits.
        k: the top-K; 512, 1024 or 2048.
        next_n: rows per request. Row ``r`` holds ``n = (kv_lens[r // next_n] - next_n +
            r % next_n + 1) >> cr_shift`` scores, clamped to ``[0, npad]``.
        cr_shift: log2 of the KV compression ratio (0 or 2).
    """

    def __init__(
        self, dtype=cutlass.BFloat16, k: int = 2048, next_n: int = 1, cr_shift: int = 0
    ) -> None:
        assert k in WS_KS, f"the tall-batch engine serves K in {WS_KS}"
        fp32 = dtype is cutlass.Float32
        # 16-bit K = 2048 windows hold one code per bin (FP16: 4 x 512 bins); the others are linear
        dense = k == 2048 and not fp32
        wide = 4 if dense and dtype is cutlass.Float16 else 1
        super().__init__(
            512,
            2,
            1,
            256,
            4,
            False,
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            r_const=1,
            hint_free=True,
            warp_rank=k == 2048,
            staged_k=0,
            dtype=dtype,
            dense=dense,
        )
        # The output row has its own buffer, so the staging keeps the full survivor budget.
        self.scpb = WS_SCPB
        self.sample_scpb = WS_SCPB
        self.sampling_cap = WS_SCPB
        self.cmpb = WS_CMPB
        if self.dense and dtype is cutlass.Float16:
            # FP16: 512 * wide one-code bins, always anchored half a window below the sampled K-th
            self.nbd = 2 * self.nbs * wide
            self.hb = self.nbd
            self.anchor_lo = self.nbd // 2 - 1
        assert self.nbs == NB_PLAN and self.vstg and not self.split
        self.kk = int(k)
        self.nt = 32 * _CONSUMER_WARP0 + WS_NCT
        self.eb = 4 if dtype is cutlass.Float32 else 2  # element bytes
        self.ve = 16 // self.eb  # elements per 16-B vector
        self.lve = self.ve.bit_length() - 1
        self.samp_bytes = 8 * self.eb  # one sample = 8 elements
        # the larger FP32 sample buffers of K <= 1024 leave room for three stages
        self.ring_stages = 3 if fp32 and k <= 1024 else 4
        self.stage_bytes = WS_STAGE_BYTES
        self.chv = self.stage_bytes // 16  # vectors per stage
        self.vpt = self.chv // WS_NCT  # vectors per consumer thread and stage
        assert self.vpt * self.ve <= 32, "one Int32 survivor mask per thread and stage"
        # the sample buffer, rounded up to 1 KB
        self.sb_bytes = -(-_WS_MAX_SAMPLES[self.kk] * self.samp_bytes // 1024) * 1024
        self.spr = self.sb_bytes // self.samp_bytes  # samples per gather round
        self.smem_bytes = self._smem_plan()
        assert self.smem_bytes <= WS_SMEM_MAX, f"dynamic smem {self.smem_bytes} B > {WS_SMEM_MAX} B"

    def _smem_plan(self) -> int:
        """Dynamic shared memory bytes of ``kern_ws``: its allocations in order, with alignment."""
        parts = (
            (4 * self.hb, 128),  # s_hist
            (32, 16),  # s_res
            (16, 16),  # s_scal
            (8, 8),  # s_kmm
            (4 * WS_NCW, 16),  # s_ws
            (8 * (self.scpb + 4), 16),  # s_cbuf2
            (8 * (self.cmpb + 1), 16),  # s_ck64
            (4 * self.kk, 16),  # out_row
            (self.ring_stages * self.stage_bytes, 128),  # ring
            (4 * NB_PLAN, 128),  # phist
            (32, 16),  # s_resp
            (self.sb_bytes, 128),  # sbuf
            (4 * 2 * LADW, 16),  # s_lad
            (8 * (4 + 2 * self.ring_stages), 8),  # mbarriers
        )
        off = 0
        for size, align in parts:
            off = -(-off // align) * align + size
        return off

    # ------------------------------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def _rcp(self, x):
        """Correctly rounded 1 / x."""
        r = cutlass.Float32(1.0) / x
        return r

    @cute.jit
    def _clear_hist(self, s_hist, ctid):
        """Zero the window histogram (512 consumer threads)."""
        for z in cutlass.range_constexpr((self.hb + WS_NCT - 1) // WS_NCT):
            if cutlass.const_expr((z + 1) * WS_NCT <= self.hb):
                s_hist[ctid + Int32(z * WS_NCT)] = Int32(0)
            else:
                if ctid + Int32(z * WS_NCT) < Int32(self.hb):
                    s_hist[ctid + Int32(z * WS_NCT)] = Int32(0)

    @cute.jit
    def _row_len(self, kv_lens, row, npad):
        """Valid length of ``row``: request ``row // next_n``, ``n = (kv_lens[req] - next_n +
        row % next_n + 1) >> cr_shift`` clamped to ``[0, npad]``. The producer and the planner
        each derive it, so neither waits on the other for it."""
        req = row // Int32(self.next_n)
        rr = row - req * Int32(self.next_n)
        nv = (kv_lens[req] - Int32(self.next_n) + rr + Int32(1)) >> Int32(self.cr_shift)
        if nv < Int32(0):
            nv = Int32(0)
        if nv > npad:
            nv = npad
        return nv

    def _row_chunks(self, n):
        """(16-B vectors, stream chunks) of a row of n scores: ceil(n / VE) vectors in chunks of
        one stage. The producer issues exactly these chunks per row; rows with n <= K (identity
        output) are not streamed."""
        nvec = (n + Int32(self.ve - 1)) >> Int32(self.lve)
        nch = (nvec + Int32(self.chv - 1)) >> Int32(self.chv.bit_length() - 1)
        return nvec, nch

    def _vals8(self, words):
        """Float32 values of one 8-element sample from its 32-bit words (4 for 16-bit, 8 for FP32)."""
        if self.dtype is cutlass.Float32:
            return [C.f32_of_i32(w) for w in words]
        out = []
        for w in words:
            u = Uint32(w)
            if self.dtype is cutlass.BFloat16:
                out.append(f32_of_u32(u << Uint32(16)))
                out.append(f32_of_u32(u & Uint32(0xFFFF0000)))
            else:
                out.append(widen_f16(u))
                out.append(widen_f16_hi(u))
        return out

    def _samp_words(self, sbuf_a, jj):
        if self.dtype is cutlass.Float32:
            a0, a1, a2, a3 = _lds_v4_u32(sbuf_a + jj * Int32(32))
            b0, b1, b2, b3 = _lds_v4_u32(sbuf_a + jj * Int32(32) + Int32(16))
            return [a0, a1, a2, a3, b0, b1, b2, b3]
        a0, a1, a2, a3 = _lds_v4_u32(sbuf_a + jj * Int32(16))
        return [a0, a1, a2, a3]

    # ------------------------------------------------------------------------------------------
    # planner
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def _ladder(self, n, aim_base, sfac, sd_en):
        """Sampling ladder of ``GvrMainKernel`` (one CTA per row, sample cap SCPB): the sample count
        SMP, stride SS2, histogram targets TGT / TGT2 and the aim."""
        k = Int32(self.kk)
        SMP = Int32(0)
        SS2 = Int32(1)
        TGT = Int32(0)
        TGT2 = Int32(0)
        x6 = Int32(6) * n
        ri = Int32(cmath.sqrt(cutlass.Float32(x6)))
        if ri * ri > x6:
            ri = ri - Int32(1)
        if (ri + Int32(1)) * (ri + Int32(1)) <= x6:
            ri = ri + Int32(1)
        r6 = ri
        if x6 - ri * ri > ri:
            r6 = ri + Int32(1)
        aim = aim_base
        if r6 > aim:
            aim = r6
        scap_c = Int32(self.sample_scpb)
        if aim > (scap_c >> Int32(1)):
            aim = scap_c >> Int32(1)
        if aim < k:
            aim = k
        n4w = n >> Int32(2)
        gate = Int32(0)
        if n > scap_c:
            gate = Int32(1)
        if sd_en != Int32(0):
            if n <= scap_c:
                if n > (k << Int32(1)):
                    gate = Int32(1)
        if n4w < Int32(4):
            gate = Int32(0)
        if gate != Int32(0):
            sel = Int32(cutlass.Float32(sfac * n) * self._rcp(cutlass.Float32(aim)))
            if sel < Int32(256):
                sel = Int32(256)
            nh = n >> Int32(1)
            if sel > nh:
                sel = nh
            pairs = sel >> Int32(3)
            if pairs < Int32(1):
                pairs = Int32(1)
            half = n4w >> Int32(1)
            if half < Int32(1):
                half = Int32(1)
            if pairs > half:
                pairs = half
            SS2 = Int32(cutlass.Float32(half) * self._rcp(cutlass.Float32(pairs)))
            if SS2 < Int32(1):
                SS2 = Int32(1)
            SMP = Int32(cutlass.Float32(half) * self._rcp(cutlass.Float32(SS2)))
            if SMP * SS2 > half:
                SMP = SMP - Int32(1)
            if SMP < Int32(1):
                SMP = Int32(1)
            rn_ = self._rcp(cutlass.Float32(n))
            smp8f = cutlass.Float32(SMP) * cutlass.Float32(8.0)
            TGT = Int32(cutlass.Float32(aim) * smp8f * rn_)
            if TGT < Int32(1):
                TGT = Int32(1)
            TGT2 = Int32(cutlass.Float32(k) * smp8f * rn_)
            if TGT2 < Int32(1):
                TGT2 = Int32(1)
        return SMP, SS2, TGT, TGT2, aim

    @cute.jit
    def _gather(self, x_addr, sbuf_a, j0, jn, ss2, lane):
        """Gather the samples [j0, j0 + jn) into the buffer and wait for them."""
        self._gather_issue(x_addr, sbuf_a, j0, jn, ss2, lane)
        self._gather_wait()

    @cute.jit
    def _gather_wait(self):
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()

    @cute.jit
    def _gather_issue(self, x_addr, sbuf_a, j0, jn, ss2, lane):
        """cp.async (and commit) the samples [j0, j0 + jn) into the buffer without waiting; sample
        j is the 8 scores at element j * SS2 * 8."""
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            Int32,
            num_bits_per_copy=128,
        )
        SB = self.samp_bytes
        for jj in cutlass.range(lane, jn, 32, unroll=1):
            g = x_addr + Int64(j0 + jj) * Int64(ss2) * Int64(SB)
            for h in cutlass.range_constexpr(SB // 16):
                src = cute.make_tensor(
                    cute.make_ptr(
                        Int32, g + Int64(16 * h), cute.AddressSpace.gmem, assumed_align=16
                    ),
                    cute.make_layout((4,)),
                )
                dst = cute.make_tensor(
                    cute.make_ptr(
                        Int32,
                        sbuf_a + jj * Int32(SB) + Int32(16 * h),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                )
                cute.copy(atom, src, dst)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _plan_row(self, x_addr, n, aim_base, sfac, sd_en, lane, sbuf_a, phist, s_resp, pre):
        """Planner math for one row of n > K scores: ladder, samples, sample histogram and the
        rungs (T, HIC, t5, tkc). ``pre`` != 0: the first gather round is already in flight."""
        NBS = NB_PLAN
        SPR = self.spr
        SMP, SS2, TGT, TGT2, aim = self._ladder(n, aim_base, sfac, sd_en)
        phist_a = phist.iterator.toint()
        if lane == Int32(0):
            s_resp[C.RES_B] = Int32(0)
            s_resp[C.RES_B2] = Int32(-1)
            s_resp[C.RES_B3] = Int32(-1)
        cute.arch.sync_warp()
        nround = (SMP + Int32(SPR - 1)) // Int32(SPR)
        smn = cutlass.Float32(float("inf"))
        smx = cutlass.Float32(float("-inf"))
        for r in cutlass.range(nround, unroll=1):
            j0 = r * Int32(SPR)
            jn = SMP - j0
            if jn > Int32(SPR):
                jn = Int32(SPR)
            issue = Int32(1)
            if r == Int32(0):
                if pre != Int32(0):
                    issue = Int32(0)
            if issue != Int32(0):
                self._gather_issue(x_addr, sbuf_a, j0, jn, SS2, lane)
            self._gather_wait()
            for jj in cutlass.range(lane, jn, 32, unroll=1):
                for v in self._vals8(self._samp_words(sbuf_a, jj)):
                    smn = C.fmin_f32(smn, v)
                    smx = C.fmax_f32(smx, v)
            cute.arch.sync_warp()
        SMIN = C.invkey(C.warp_min_u32(C.fkey(smn)))
        SMAX = C.invkey(C.warp_max_u32(C.fkey(smx)))
        w = cutlass.Float32(0.0)
        sok = Int32(0)
        if SMP > Int32(0):
            if SMAX > SMIN:
                sok = Int32(1)
        if sok != Int32(0):
            w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
            sc_s = self._rcp(w)
            # histogram pass, last round first: it is still in the buffer, so a row of R rounds
            # re-gathers R - 1 of them
            for r in cutlass.range(nround, unroll=1):
                j0 = (nround - Int32(1) - r) * Int32(SPR)
                jn = SMP - j0
                if jn > Int32(SPR):
                    jn = Int32(SPR)
                if r > Int32(0):
                    self._gather(x_addr, sbuf_a, j0, jn, SS2, lane)
                for jj in cutlass.range(lane, jn, 32, unroll=1):
                    for v in self._vals8(self._samp_words(sbuf_a, jj)):
                        bq = C.f2s_rz((v - SMIN) * sc_s)
                        if bq > Int32(NBS - 1):
                            bq = Int32(NBS - 1)
                        if bq < Int32(0):
                            bq = Int32(0)
                        C._red_shared_add1(phist_a + (bq << Int32(2)))
                cute.arch.sync_warp()
        # triple-target scan TGT / TGT2 / 2 TGT; zero=True leaves the histogram cleared
        C.scan_cross0(
            phist,
            TGT,
            lane,
            s_resp,
            TGT2,
            TGT * Int32(2),
            phist,
            nb=NBS,
            zero=True,
            two=True,
            three=True,
        )
        cute.arch.sync_warp()
        tot0 = s_resp[C.RES_TOT]
        b1v = s_resp[C.RES_B]
        b2v = s_resp[C.RES_B2]
        b3v = s_resp[C.RES_B3]
        cute.arch.sync_warp()
        GMIN = cutlass.Float32(_SENT_LO)
        T = cutlass.Float32(_NEG_INF)
        HIC = cutlass.Float32(_NEG_INF)
        tkc = Int32(-1)
        t5 = cutlass.Float32(_NEG_INF)
        if sok != Int32(0):
            if tot0 >= TGT:
                T = _fmaf(cutlass.Float32(b1v), w, SMIN)
        Trung = T
        needg = Int32(1)
        if T > cutlass.Float32(_NEG_INF):
            needg = Int32(0)
        if needg != Int32(0):
            T = GMIN  # hint-free: no sampled rung, the floor
        if sok != Int32(0):
            if tot0 >= TGT:
                if b2v >= Int32(0):
                    Tk = _fmaf(cutlass.Float32(b2v), w, SMIN)
                    if cutlass.const_expr(self.dense):
                        tkc = Int32(key16_rz_of_f32(self.dtype, Tk))
                    anch_ = C.fmin_f32(T, Trung)
                    d_ = C.fmax_f32(Tk - anch_, cutlass.Float32(0.0))
                    HIC = C.fmax_f32(
                        _fmaf(cutlass.Float32(4.0), d_, T), _fmaf(cutlass.Float32(8.0), w, T)
                    )
            if tot0 >= TGT * Int32(2):
                if b3v >= Int32(0):
                    if T > GMIN:
                        T3 = _fmaf(cutlass.Float32(b3v), w, SMIN)
                        if T3 < T:
                            t5 = T3
        return T, HIC, t5, tkc

    @cute.jit
    def _window(self, T, HIC, tkc):
        """Attempt window at rung T: classify threshold TF, bin scale (linear) or base code
        (dense) SC, the packed 16-bit threshold pair and whether the dense window re-anchored."""
        TF = T
        tf_pair = Uint32(0)
        if cutlass.const_expr(self.dtype != cutlass.Float32):
            tf_pair = ceil16x2_of_f32(self.dtype, TF)
        hi = C.fmax_f32(cutlass.Float32(_SENT_HI), T)
        if HIC > T:
            if HIC < hi:
                hi = HIC
        WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
        wdok = Int32(0)
        if WD > cutlass.Float32(0.0):
            wdok = Int32(1)
        if wdok == Int32(0):
            WD = cutlass.Float32(1e-30)
        SC = cutlass.Float32(1.0)
        anch = Int32(0)
        if cutlass.const_expr(self.dense):
            base16 = Int32(key16_of_pair16(self.dtype, tf_pair))
            if tkc >= Int32(0):
                if cutlass.const_expr(self.dtype == cutlass.Float16):
                    # FP16: the sampled K-th code sits in bin anchor_lo
                    base16 = tkc - Int32(self.anchor_lo)
                    anch = Int32(1)
                elif tkc - base16 > Int32(self.nbd - 2):
                    base16 = tkc - Int32(self.anchor_lo)
                    anch = Int32(1)
            SC = cutlass.Float32(base16)
        else:
            SC = self._rcp(WD)
        if cutlass.const_expr(self.dtype != cutlass.Float32 and not self.dense):
            # 16-bit linear bins classify one bin below the rung (rounding knife edge)
            if wdok != Int32(0):
                TF = T - WD
                tf_pair = ceil16x2_of_f32(self.dtype, TF)
        return TF, SC, tf_pair, anch

    # ------------------------------------------------------------------------------------------
    # consumers: ring stream, retries from global memory
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def _vec_mask(self, w0, w1, w2, w3, TF, tf_pair):
        """Survivor bits of one 16-B vector (packed 16x2 compares against ceil16(TF); FP32 x >= TF)."""
        mm = Int32(0)
        if cutlass.const_expr(self.dtype == cutlass.BFloat16):
            for q, wq in enumerate((w0, w1, w2, w3)):
                mm = bf16x2_ge_or(mm, Uint32(wq), tf_pair, 2 * q)
        elif cutlass.const_expr(self.dtype == cutlass.Float16):
            for q, wq in enumerate((w0, w1, w2, w3)):
                mm = Int32(ge16x2_or(self.dtype, Uint32(mm), Uint32(wq), tf_pair, 2 * q))
        else:
            for q, wq in enumerate((w0, w1, w2, w3)):
                mm = mm | (Int32(C.f32_of_i32(wq) >= TF) << Int32(q))
        return mm

    @cute.jit
    def _reserve(self, m_, lane, s_scal):
        """Warp-aggregated staging reservation: the thread's first slot."""
        cnt = Int32(C.popc(m_))
        inc = _bf16_emit_prefix_agg(cnt)  # predicated shfl.up inclusive scan
        bpos = Int32(0)
        if lane == Int32(31):
            if inc != Int32(0):
                bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
        return cute.arch.shuffle_sync(bpos, Int32(31)) + (inc - cnt)

    @cute.jit
    def _consume_stage(
        self,
        stage_a,
        v0,
        nvc,
        n,
        ctid,
        lane,
        TF,
        SC,
        tf_pair,
        hb_pin,
        cb2_pin,
        s_hist,
        s_cbuf,
        s_cbuf2,
        s_scal,
    ):
        """Classify the thread's vectors of one stage (scores >= n masked), reserve staging slots
        warp-aggregated, then stage each survivor (``_emitc``: {value, idx} and its bin) with its
        value read back from the stage. Whole stages skip the bound masks."""
        VE = self.ve
        m_ = Int32(0)
        whole = Int32(0)
        if nvc == Int32(self.chv):
            if (v0 + nvc) * Int32(VE) <= n:
                whole = Int32(1)
        if whole != Int32(0):
            for j in cutlass.range_constexpr(self.vpt):
                v = ctid + Int32(j * WS_NCT)
                w0, w1, w2, w3 = _lds_v4_u32(stage_a + v * Int32(16))
                m_ = m_ | (self._vec_mask(w0, w1, w2, w3, TF, tf_pair) << Int32(j * VE))
        else:
            for j in cutlass.range_constexpr(self.vpt):
                v = ctid + Int32(j * WS_NCT)
                if v < nvc:
                    w0, w1, w2, w3 = _lds_v4_u32(stage_a + v * Int32(16))
                    mm = self._vec_mask(w0, w1, w2, w3, TF, tf_pair)
                    lim = n - (v0 + v) * Int32(VE)
                    if lim < Int32(VE):
                        mm = mm & ((Int32(1) << lim) - Int32(1))
                    m_ = m_ | (mm << Int32(j * VE))
        pos = self._reserve(m_, lane, s_scal)
        while m_ != Int32(0):
            bp = C.ffs_m1(m_)
            m_ = m_ & (m_ - Int32(1))
            jv = bp >> Int32(self.lve)
            e = bp & Int32(VE - 1)
            v = ctid + jv * Int32(WS_NCT)
            idx = ((v0 + v) << Int32(self.lve)) + e
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                xv = C.f32_of_i32(_lds_u32(stage_a + v * Int32(16) + (e << Int32(2))))
                pos = self._emitc(
                    xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2, None
                )
            else:
                code = _lds_u16(stage_a + v * Int32(16) + (e << Int32(1)))
                if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                    xv = f32_of_u32(code << Uint32(16))
                    pos = self._emitc(
                        xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2, None
                    )
                else:
                    xv = widen_f16(code)
                    pos = self._emitc(
                        xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2, code
                    )

    @cute.jit
    def _consume_gmem(
        self,
        x_addr,
        v0,
        nvc,
        n,
        ctid,
        lane,
        TF,
        SC,
        tf_pair,
        hb_pin,
        cb2_pin,
        s_hist,
        s_cbuf,
        s_cbuf2,
        s_scal,
    ):
        """Retry pass over one chunk straight from global memory: 16-B loads, the same classify and
        reservation, survivor values re-loaded (L1 / L2 hits of the vector just read)."""
        VE = self.ve
        m_ = Int32(0)
        for j in cutlass.range_constexpr(self.vpt):
            v = ctid + Int32(j * WS_NCT)
            if v < nvc:
                w0, w1, w2, w3 = C._ld_g_nc_v4_b32(x_addr + Int64(v0 + v) * Int64(16))
                mm = self._vec_mask(w0, w1, w2, w3, TF, tf_pair)
                lim = n - (v0 + v) * Int32(VE)
                if lim < Int32(VE):
                    mm = mm & ((Int32(1) << lim) - Int32(1))
                m_ = m_ | (mm << Int32(j * VE))
        pos = self._reserve(m_, lane, s_scal)
        while m_ != Int32(0):
            bp = C.ffs_m1(m_)
            m_ = m_ & (m_ - Int32(1))
            jv = bp >> Int32(self.lve)
            e = bp & Int32(VE - 1)
            v = ctid + jv * Int32(WS_NCT)
            idx = ((v0 + v) << Int32(self.lve)) + e
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                xv = C.ldg_f32(x_addr, idx)
                pos = self._emitc(
                    xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2, None
                )
            else:
                xv, xtc = ldg_16_code(self.dtype, x_addr, idx)
                pos = self._emitc(
                    xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2, xtc
                )

    @cute.jit
    def _chunk_v0(self, c, nch, nvec):
        """(first vector, vector count) of the chunk at stream step c of a row of nch chunks /
        nvec vectors: physical chunk nch - 1 - c (back to front; the partial last chunk first)."""
        p = nch - Int32(1) - c
        v0 = p * Int32(self.chv)
        nvc = nvec - v0
        if nvc > Int32(self.chv):
            nvc = Int32(self.chv)
        return v0, nvc

    @cute.jit
    def _issue_chunk(self, x_addr, nvec, nch, c, stage_a, full_p, pol):
        """Producer lane: arm ``full_p`` with the chunk's bytes and issue its 1-D bulk copy."""
        v0, nvc = self._chunk_v0(c, nch, nvec)
        nb = nvc * Int32(16)
        cute.arch.mbarrier_arrive_and_expect_tx(full_p, nb)
        ga = x_addr + Int64(v0) * Int64(16)
        _bulk_g2s_hint(stage_a, ga, nb, full_p.toint(), pol)

    @cute.jit
    def _issue_first(self, kv_lens, bid, npad, rcta, rowb, x_base, ring, ring_full):
        """Producer lane, before the CTA barrier: the first min(stages, chunks) ring loads of the
        CTA's row 0; returns how many."""
        s0 = Int32(0)
        if rcta > Int32(0):
            nv0p = self._row_len(kv_lens, bid, npad)
            if nv0p > Int32(self.kk):
                nvec0, nch0 = self._row_chunks(nv0p)
                s0 = nch0
                if s0 > Int32(self.ring_stages):
                    s0 = Int32(self.ring_stages)
                pol0 = C._l2_evict_first_policy()
                ring_p0 = ring.iterator.toint()
                xa0 = x_base + Int64(bid) * rowb
                for q in cutlass.range(s0, unroll=1):
                    self._issue_chunk(
                        xa0,
                        nvec0,
                        nch0,
                        q,
                        ring_p0 + q * Int32(self.stage_bytes),
                        ring_full + q,
                        pol0,
                    )
        return s0

    @cute.jit
    def _stream_ring(
        self,
        n,
        ctid,
        lane,
        TF,
        SC,
        tf_pair,
        hb_pin,
        cb2_pin,
        ring_a,
        rs,
        rph,
        s_hist,
        s_cbuf,
        s_cbuf2,
        s_scal,
        ring_full,
        ring_empty,
    ):
        """First attempt: the row chunk by chunk through the ring. Per chunk: wait ``full[rs]``
        (parity rph), consume the stage, then lane 0 of each consumer warp releases it on
        ``empty[rs]``; (rs, rph) advance once per chunk and carry across rows."""
        nvec, nch = self._row_chunks(n)
        for c in cutlass.range(nch, unroll=1):
            v0, nvc = self._chunk_v0(c, nch, nvec)
            stage_a = ring_a + rs * Int32(self.stage_bytes)
            cute.arch.mbarrier_wait(ring_full + rs, rph)
            self._consume_stage(
                stage_a,
                v0,
                nvc,
                n,
                ctid,
                lane,
                TF,
                SC,
                tf_pair,
                hb_pin,
                cb2_pin,
                s_hist,
                s_cbuf,
                s_cbuf2,
                s_scal,
            )
            cute.arch.sync_warp()
            if lane == Int32(0):
                cute.arch.mbarrier_arrive(ring_empty + rs)
            rs = rs + Int32(1)
            if rs == Int32(self.ring_stages):
                rs = Int32(0)
                rph = rph ^ Int32(1)
        return rs, rph

    @cute.jit
    def _stream_gmem(
        self,
        x_addr,
        n,
        ctid,
        lane,
        TF,
        SC,
        tf_pair,
        hb_pin,
        cb2_pin,
        s_hist,
        s_cbuf,
        s_cbuf2,
        s_scal,
    ):
        """Retry pass from global memory; the ring keeps streaming the CTA's next row meanwhile."""
        CHV = self.chv
        nvec, nch = self._row_chunks(n)
        for c in cutlass.range(nch, unroll=1):
            v0 = c * Int32(CHV)
            nvc = nvec - v0
            if nvc > Int32(CHV):
                nvc = Int32(CHV)
            self._consume_gmem(
                x_addr,
                v0,
                nvc,
                n,
                ctid,
                lane,
                TF,
                SC,
                tf_pair,
                hb_pin,
                cb2_pin,
                s_hist,
                s_cbuf,
                s_cbuf2,
                s_scal,
            )

    @cute.jit
    def _attempts(
        self,
        x_addr,
        n,
        T,
        HIC,
        tkc,
        t5,
        TF0,
        SC0,
        anch0,
        ctid,
        lane,
        hb_pin,
        cb2_pin,
        ring_a,
        rs,
        rph,
        s_hist,
        s_cbuf,
        s_cbuf2,
        s_scal,
        s_res,
        s_ws,
        ring_full,
        ring_empty,
    ):
        """Attempt loop: stream and classify, verify that the window holds K scores, else retry at
        the TSH rung t5 (first attempt only) or at the floor."""
        NBS = self.nbs
        NBD = self.nbd
        SCPB = self.scpb
        NATT = self.natt
        k = Int32(self.kk)
        GMIN = cutlass.Float32(_SENT_LO)
        listN = Int32(0)
        above = Int32(0)
        m = Int32(0)
        need = Int32(0)
        B = Int32(0)
        TF = TF0
        SC = SC0
        anch = anch0
        tf_pair = Uint32(0)
        if cutlass.const_expr(self.dtype != cutlass.Float32):
            tf_pair = ceil16x2_of_f32(self.dtype, TF0)
        complete = Int32(0)
        valid = Int32(0)
        att = Int32(0)
        running = Int32(1)
        while running != Int32(0):
            if att > Int32(0):
                self._clear_hist(s_hist, ctid)
                if ctid == Int32(0):
                    s_scal[0] = Int32(0)
                _bar_cons()  # ---- barrier (retry reset) ----
                TF, SC, tf_pair, anch = self._window(T, HIC, tkc)
                self._stream_gmem(
                    x_addr,
                    n,
                    ctid,
                    lane,
                    TF,
                    SC,
                    tf_pair,
                    hb_pin,
                    cb2_pin,
                    s_hist,
                    s_cbuf,
                    s_cbuf2,
                    s_scal,
                )
            else:
                rs, rph = self._stream_ring(
                    n,
                    ctid,
                    lane,
                    TF,
                    SC,
                    tf_pair,
                    hb_pin,
                    cb2_pin,
                    ring_a,
                    rs,
                    rph,
                    s_hist,
                    s_cbuf,
                    s_cbuf2,
                    s_scal,
                    ring_full,
                    ring_empty,
                )
            _bar_cons()  # ---- barrier (row pass) ----
            myn = s_scal[0]
            if cutlass.const_expr(NBD > NBS):
                # wide dense window: block scan over NBD >= 512 bins (one internal barrier)
                _scan_cross_w_cons(s_hist, s_ws, k, ctid, s_res, nb=NBD)
            else:
                C.scan_cross0(
                    s_hist, k, ctid, s_res, Int32(0), Int32(0), s_hist, nb=NBD, zero=False
                )
            _bar_cons()  # ---- barrier (verify scan) ----
            tot = s_res[C.RES_TOT]
            if tot >= k:
                valid = Int32(1)
                complete = Int32(0)
                if myn <= Int32(SCPB):
                    complete = Int32(1)
                listN = myn
                above = s_res[C.RES_ABOVE]
                m = s_res[C.RES_M]
                need = k - above
                B = s_res[C.RES_B]
                running = Int32(0)
            else:
                if att == Int32(NATT - 1):
                    running = Int32(0)
                else:
                    tshtaken = Int32(0)
                    if att == Int32(0):
                        if t5 > cutlass.Float32(_NEG_INF):
                            if t5 < TF:
                                T = t5
                                tshtaken = Int32(1)
                    if tshtaken != Int32(0):
                        _bar_cons()  # ---- barrier (TSH retry) ----
                    else:
                        floorhit = Int32(1)
                        if T > GMIN:
                            floorhit = Int32(0)
                        if floorhit != Int32(0):
                            running = Int32(0)
                        else:
                            T = GMIN
                            _bar_cons()  # ---- barrier (floor retry) ----
            att = att + Int32(1)
        return listN, above, m, need, B, TF, SC, anch, complete, valid, rs, rph

    # ------------------------------------------------------------------------------------------
    # selection: the BLK = 512 staged path of GvrMainKernel (non-split, hint-free)
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def _select(
        self,
        x_addr,
        n,
        TF,
        SC,
        anch,
        B,
        m,
        above,
        need,
        listN,
        valid,
        complete,
        ctid,
        lane,
        s_hist,
        s_cbuf2,
        s_ck64,
        s_res,
        s_scal,
        s_kmm,
        out_row,
    ):
        """Emit the row's top-K indices into ``out_row``: bins above the crossing bin B whole,
        then the crossing bin refined (warp-ballot rank at K = 2048, quadratic rank or radix
        narrowing); rows without a valid window (degen) narrow over the staged survivors (A) or
        over the whole row from global memory (B)."""
        BLK = WS_NCT
        NBS = self.nbs
        NBD = self.nbd
        CMPB = self.cmpb
        k = Int32(self.kk)
        ck_addr = s_ck64.iterator.toint()
        # ============ classification ============
        whole = Int32(0)
        if valid != Int32(0):
            if need >= m:
                whole = Int32(1)
        if cutlass.const_expr(self.dense):
            if valid != Int32(0):
                if B < Int32(NBD - 1):
                    # every interior bin holds one exact 16-bit value (bin 0 of a re-anchored
                    # window holds every survivor below the window)
                    okw = Int32(1)
                    if anch != Int32(0):
                        if B == Int32(0):
                            okw = Int32(0)
                    if okw != Int32(0):
                        whole = Int32(1)
        lim1 = above
        if whole != Int32(0):
            if cutlass.const_expr(self.dense):
                lim1 = above + need
            else:
                lim1 = above + m
        degen = Int32(0)
        if valid == Int32(0):
            degen = Int32(1)
        if m > Int32(CMPB):
            if cutlass.const_expr(self.dense):
                if whole == Int32(0):
                    degen = Int32(1)
            else:
                degen = Int32(1)
        mc = Int32(0)
        if degen == Int32(0):
            mc = m
        dga = Int32(0)
        if degen == Int32(0):
            # ---- P5 cursor emit ----
            if complete != Int32(0):
                i = ctid
                while i < listN:
                    pk64 = s_cbuf2[i]
                    vx = Int32(Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF)))
                    vy = Int32(pk64 >> cutlass.Uint64(32))
                    xv = C.f32_of_i32(vx)
                    idv = vy
                    bq = Int32(0)
                    if cutlass.const_expr(self.dense):
                        bq = self._dense_bin(xv, SC)
                    else:
                        bq = C.f2s_rz((xv - TF) * SC)
                    if bq > Int32(NBD - 1):
                        bq = Int32(NBD - 1)
                    if bq >= B:
                        p = C.atomic_add_cta(s_hist.iterator + bq, Int32(1))
                        if p < lim1:
                            out_row[p] = idv
                        else:
                            if whole == Int32(0):
                                q2 = p - above
                                if q2 < Int32(CMPB):
                                    kk = C.fkey(xv)
                                    s_ck64[q2] = (
                                        cutlass.Uint64(kk) << cutlass.Uint64(32)
                                    ) | cutlass.Uint64(Uint32(idv))
                    i = i + Int32(BLK)
            else:
                # incomplete list (myn > SCPB): re-sweep the row (x >= TF) from global memory
                i_ = ctid
                while i_ < n:
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        xtc = None
                        x = C.ldg_f32(x_addr, i_)
                    else:
                        x, xtc = ldg_16_code(self.dtype, x_addr, i_)
                    if x >= TF:
                        bq = Int32(0)
                        if cutlass.const_expr(self.dense):
                            bq = self._dense_bin(x, SC, xtc)
                        else:
                            bq = C.f2s_rz((x - TF) * SC)
                        if bq > Int32(NBD - 1):
                            bq = Int32(NBD - 1)
                        if bq >= B:
                            p = C.atomic_add_cta(s_hist.iterator + bq, Int32(1))
                            if p < lim1:
                                out_row[p] = i_
                            else:
                                if whole == Int32(0):
                                    q2 = p - above
                                    if q2 < Int32(CMPB):
                                        s_ck64[q2] = (
                                            cutlass.Uint64(C.fkey(x)) << cutlass.Uint64(32)
                                        ) | cutlass.Uint64(Uint32(i_))
                    i_ = i_ + Int32(BLK)

            # ---- P6 refine ----
            if whole == Int32(0):
                _bar_cons()  # ---- barrier (emit done) ----
                if mc <= Int32(QUADC_CLUS__main):
                    warp_quad = Int32(0)
                    if cutlass.const_expr(self.warp_rank):
                        if mc <= Int32(32):
                            warp_quad = Int32(1)
                    if warp_quad != Int32(0):
                        rank_lane = lane
                        rank_i = ctid >> Int32(5)
                        if rank_i < mc:
                            vote_hi = Uint32(0)
                            vote_lo = Uint32(0)
                            if rank_lane < mc:
                                vote64 = cutlass.Uint64(s_ck64[rank_lane])
                                vote_hi = Uint32(vote64 >> cutlass.Uint64(32))
                                vote_lo = Uint32(vote64 & cutlass.Uint64(0xFFFFFFFF))
                            while rank_i < mc:
                                target_hi = cute.arch.shuffle_sync(vote_hi, rank_i)
                                target_lo = cute.arch.shuffle_sync(vote_lo, rank_i)
                                rank_vote = Int32(0)
                                if rank_lane < mc:
                                    if vote_hi > target_hi:
                                        rank_vote = Int32(1)
                                    if vote_hi == target_hi and vote_lo > target_lo:
                                        rank_vote = Int32(1)
                                rank_count = C.popc(C.ballot(rank_vote != Int32(0)))
                                if rank_lane == Int32(0) and rank_count < need:
                                    out_row[above + rank_count] = Int32(target_lo)
                                rank_i = rank_i + Int32(BLK // 32)
                    else:
                        mc2 = mc & Int32(~1)
                        i = ctid
                        while i < mc:
                            u64v = s_ck64[i]
                            r_ = Int32(0)
                            jq = Int32(0)
                            while jq < mc2:  # ulonglong2 16B reads
                                vlo, vhi = C._lds_v2_u64(ck_addr + jq * Int32(8))
                                r_ = (
                                    r_
                                    + Int32(vlo > cutlass.Uint64(u64v))
                                    + Int32(vhi > cutlass.Uint64(u64v))
                                )
                                jq = jq + Int32(2)
                            if mc2 < mc:  # odd tail
                                r_ = r_ + Int32(cutlass.Uint64(s_ck64[mc2]) > cutlass.Uint64(u64v))
                            if r_ < need:
                                out_row[above + r_] = Int32(
                                    Uint32(cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF))
                                )
                            i = i + Int32(BLK)
                else:
                    if ctid == Int32(0):
                        s_kmm[0] = Uint32(0xFFFFFFFF)
                        s_kmm[1] = Uint32(0)
                    if ctid < Int32(NBS):
                        s_hist[ctid] = Int32(0)
                    _bar_cons()  # ---- barrier (narrowing init) ----
                    i = ctid
                    while i < mc:
                        kk = Uint32(s_ck64[i] >> cutlass.Uint64(32 + self.ck_key_shift))
                        C.atomic_min_cta(s_kmm.iterator + 0, kk)
                        C.atomic_max_cta(s_kmm.iterator + 1, kk)
                        i = i + Int32(BLK)
                    _bar_cons()  # ---- barrier (key range) ----
                    rlo = s_kmm[0]
                    rhi = s_kmm[1]
                    ethr = Int64(Uint32(rlo))
                    aboveC = Int32(0)
                    needC = need
                    mm = mc
                    brk = Int32(0)
                    lev = Int32(0)
                    while brk == Int32(0):
                        if needC == mm:
                            ethr = Int64(Uint32(rlo)) - Int64(1)
                            aboveC = aboveC + mm
                            needC = Int32(0)
                            brk = Int32(1)
                        elif Uint32(rlo) >= Uint32(rhi):
                            ethr = Int64(Uint32(rlo))
                            brk = Int32(1)
                        elif lev >= Int32(6):
                            ethr = Int64(Uint32(rlo))
                            brk = Int32(1)
                        else:
                            d2 = Uint32(rhi) - Uint32(rlo)
                            b2_ = Int32(32) - C.clz_i32(Int32(d2 | Uint32(1)))
                            sh2 = b2_ - Int32(self.lb)
                            if sh2 < Int32(0):
                                sh2 = Int32(0)
                            sh2u = Uint32(sh2)
                            i = ctid
                            while i < mc:  # re-bin
                                uq = Uint32(s_ck64[i] >> cutlass.Uint64(32 + self.ck_key_shift))
                                if uq >= Uint32(rlo):
                                    if uq <= Uint32(rhi):
                                        du = (uq - Uint32(rlo)) >> sh2u
                                        if du > Uint32(NBS - 1):
                                            du = Uint32(NBS - 1)
                                        C.atomic_add_cta(s_hist.iterator + Int32(du), Int32(1))
                                i = i + Int32(BLK)
                            _bar_cons()  # ---- barrier (level hist) ----
                            C.scan_cross0(
                                s_hist,
                                needC,
                                ctid,
                                s_res,
                                Int32(0),
                                Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=True,
                            )
                            _bar_cons()  # ---- barrier (level scan) ----
                            aboveC = aboveC + s_res[C.RES_ABOVE]
                            needC = needC - s_res[C.RES_ABOVE]
                            mm = s_res[C.RES_M]
                            sB = s_res[C.RES_B]
                            nlo = Uint32(rlo) + (Uint32(sB) << sh2u)
                            if sB != Int32(NBS - 1):
                                rhi = nlo + ((Uint32(1) << sh2u) - Uint32(1))
                            rlo = nlo
                            lev = lev + Int32(1)
                    if ctid == Int32(0):
                        s_scal[1] = Int32(0)
                        s_scal[2] = Int32(0)
                    _bar_cons()  # ---- barrier (emit counters) ----
                    it2 = (mc + Int32(BLK - 1)) // Int32(BLK)
                    it = Int32(0)
                    while it < it2:  # ballot emit
                        i = it * Int32(BLK) + ctid
                        p1 = Int32(0)
                        p2 = Int32(0)
                        idv = Int32(0)
                        if i < mc:
                            w64 = s_ck64[i]
                            iu = Int64(Uint32(w64 >> cutlass.Uint64(32 + self.ck_key_shift)))
                            idv = Int32(Uint32(w64 & cutlass.Uint64(0xFFFFFFFF)))
                            if iu > ethr:
                                p1 = Int32(1)
                            if iu == ethr:
                                p2 = Int32(1)
                        self._ballot_pair_emit(
                            p1, p2, idv, above, aboveC, above + aboveC, needC, out_row, s_scal, lane
                        )
                        it = it + Int32(1)
        else:
            if valid != Int32(0):
                if complete != Int32(0):
                    dga = Int32(1)
            if dga != Int32(0):
                # ---- degen A: narrowing over the staged candidates ----
                rlo = Uint32(0)
                if cutlass.const_expr(self.k16):
                    rhi = Uint32(0xFFFF)
                else:
                    rhi = Uint32(0xFFFFFFFF)
                above2 = Int32(0)
                need2 = k
                m2 = listN
                ethr = Int64(0)
                tie_m = Int32(1)
                if ctid < Int32(NBS):
                    s_hist[ctid] = Int32(0)
                _bar_cons()  # ---- barrier (degen A init) ----
                brk = Int32(0)
                lev = Int32(0)
                while brk == Int32(0):
                    if need2 == m2:
                        ethr = Int64(Uint32(rlo)) - Int64(1)
                        above2 = above2 + m2
                        need2 = Int32(0)
                        tie_m = Int32(0)
                        brk = Int32(1)
                    elif Uint32(rlo) >= Uint32(rhi):
                        ethr = Int64(Uint32(rlo))
                        brk = Int32(1)
                    elif lev >= Int32(8):
                        ethr = Int64(Uint32(rlo))
                        brk = Int32(1)
                    else:
                        d2 = Uint32(rhi) - Uint32(rlo)
                        b2_ = Int32(32) - C.clz_i32(Int32(d2 | Uint32(1)))
                        sh2 = b2_ - Int32(self.lb)
                        if sh2 < Int32(0):
                            sh2 = Int32(0)
                        sh2u = Uint32(sh2)
                        i = ctid
                        while i < listN:
                            pk64 = s_cbuf2[i]
                            vx = Int32(Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF)))
                            uq = Uint32(0)
                            if cutlass.const_expr(self.k16):
                                uq = key16_of_f32(self.dtype, C.f32_of_i32(vx))
                            else:
                                uq = C.fkey_bits(Uint32(vx))
                            if uq >= Uint32(rlo):
                                if uq <= Uint32(rhi):
                                    du = (uq - Uint32(rlo)) >> sh2u
                                    if du > Uint32(NBS - 1):
                                        du = Uint32(NBS - 1)
                                    C.atomic_add_cta(s_hist.iterator + Int32(du), Int32(1))
                            i = i + Int32(BLK)
                        _bar_cons()  # ---- barrier (level hist) ----
                        C.scan_cross0(
                            s_hist,
                            need2,
                            ctid,
                            s_res,
                            Int32(0),
                            Int32(0),
                            s_hist,
                            nb=NBS,
                            zero=True,
                        )
                        _bar_cons()  # ---- barrier (level scan) ----
                        above2 = above2 + s_res[C.RES_ABOVE]
                        need2 = need2 - s_res[C.RES_ABOVE]
                        m2 = s_res[C.RES_M]
                        sB = s_res[C.RES_B]
                        nlo = Uint32(rlo) + (Uint32(sB) << sh2u)
                        if sB != Int32(NBS - 1):
                            rhi = nlo + ((Uint32(1) << sh2u) - Uint32(1))
                        rlo = nlo
                        lev = lev + Int32(1)
                if ctid == Int32(0):
                    s_scal[1] = Int32(0)
                    s_scal[2] = Int32(0)
                _bar_cons()  # ---- barrier (emit counters) ----
                nA = k
                nT = Int32(0)
                if tie_m != Int32(0):
                    nA = above2
                    nT = need2
                it2 = (listN + Int32(BLK - 1)) // Int32(BLK)
                it = Int32(0)
                while it < it2:
                    i = it * Int32(BLK) + ctid
                    p1 = Int32(0)
                    p2 = Int32(0)
                    idv = Int32(0)
                    if i < listN:
                        pk64 = s_cbuf2[i]
                        vx = Int32(Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF)))
                        vy = Int32(pk64 >> cutlass.Uint64(32))
                        uq = Uint32(0)
                        if cutlass.const_expr(self.k16):
                            uq = key16_of_f32(self.dtype, C.f32_of_i32(vx))
                        else:
                            uq = C.fkey_bits(Uint32(vx))
                        idv = vy
                        iu = Int64(uq)
                        if iu > ethr:
                            p1 = Int32(1)
                        if tie_m != Int32(0):
                            if iu == ethr:
                                p2 = Int32(1)
                    self._ballot_pair_emit(p1, p2, idv, Int32(0), nA, nA, nT, out_row, s_scal, lane)
                    it = it + Int32(1)
            else:
                # ---- degen B: whole-row narrowing (keys re-read from global memory) ----
                rlo = Uint32(0)
                if cutlass.const_expr(self.k16):
                    rhi = Uint32(0xFFFF)
                else:
                    rhi = Uint32(0xFFFFFFFF)
                above2 = Int32(0)
                need2 = k
                m2 = n
                ethr = Int64(0)
                tie_m = Int32(1)
                if ctid < Int32(NBS):
                    s_hist[ctid] = Int32(0)
                _bar_cons()  # ---- barrier (degen B init) ----
                brk = Int32(0)
                lev = Int32(0)
                while brk == Int32(0):
                    if need2 == m2:
                        ethr = Int64(Uint32(rlo)) - Int64(1)
                        above2 = above2 + m2
                        need2 = Int32(0)
                        tie_m = Int32(0)
                        brk = Int32(1)
                    elif Uint32(rlo) >= Uint32(rhi):
                        ethr = Int64(Uint32(rlo))
                        brk = Int32(1)
                    elif lev >= Int32(8):
                        ethr = Int64(Uint32(rlo))
                        brk = Int32(1)
                    else:
                        d2 = Uint32(rhi) - Uint32(rlo)
                        b2_ = Int32(32) - C.clz_i32(Int32(d2 | Uint32(1)))
                        sh2 = b2_ - Int32(self.lb)
                        if sh2 < Int32(0):
                            sh2 = Int32(0)
                        sh2u = Uint32(sh2)
                        i = ctid
                        while i < n:  # whole row
                            uq = Uint32(0)
                            if cutlass.const_expr(self.dtype == cutlass.Float32):
                                uq = C.fkey(C.ldg_f32(x_addr, i))
                            elif cutlass.const_expr(self.k16):
                                uq = fkey16_of_code16(_ld_g_nc_u16(x_addr + Int64(i) * Int64(2)))
                            else:
                                uq = C.fkey(ldg_16(self.dtype, x_addr, i))
                            if uq >= Uint32(rlo):
                                if uq <= Uint32(rhi):
                                    du = (uq - Uint32(rlo)) >> sh2u
                                    if du > Uint32(NBS - 1):
                                        du = Uint32(NBS - 1)
                                    C.atomic_add_cta(s_hist.iterator + Int32(du), Int32(1))
                            i = i + Int32(BLK)
                        _bar_cons()  # ---- barrier (level hist) ----
                        C.scan_cross0(
                            s_hist,
                            need2,
                            ctid,
                            s_res,
                            Int32(0),
                            Int32(0),
                            s_hist,
                            nb=NBS,
                            zero=True,
                        )
                        _bar_cons()  # ---- barrier (level scan) ----
                        above2 = above2 + s_res[C.RES_ABOVE]
                        need2 = need2 - s_res[C.RES_ABOVE]
                        m2 = s_res[C.RES_M]
                        sB = s_res[C.RES_B]
                        nlo = Uint32(rlo) + (Uint32(sB) << sh2u)
                        if sB != Int32(NBS - 1):
                            rhi = nlo + ((Uint32(1) << sh2u) - Uint32(1))
                        rlo = nlo
                        lev = lev + Int32(1)
                if ctid == Int32(0):
                    s_scal[1] = Int32(0)
                    s_scal[2] = Int32(0)
                _bar_cons()  # ---- barrier (emit counters) ----
                nA = k
                nT = Int32(0)
                if tie_m != Int32(0):
                    nA = above2
                    nT = need2
                it2 = (n + Int32(BLK - 1)) // Int32(BLK)
                it = Int32(0)
                while it < it2:
                    i = it * Int32(BLK) + ctid
                    p1 = Int32(0)
                    p2 = Int32(0)
                    if i < n:
                        uq = Uint32(0)
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            uq = C.fkey(C.ldg_f32(x_addr, i))
                        elif cutlass.const_expr(self.k16):
                            uq = fkey16_of_code16(_ld_g_nc_u16(x_addr + Int64(i) * Int64(2)))
                        else:
                            uq = C.fkey(ldg_16(self.dtype, x_addr, i))
                        iu = Int64(uq)
                        if iu > ethr:
                            p1 = Int32(1)
                        if tie_m != Int32(0):
                            if iu == ethr:
                                p2 = Int32(1)
                    self._ballot_pair_emit(p1, p2, i, Int32(0), nA, nA, nT, out_row, s_scal, lane)
                    it = it + Int32(1)

    # ------------------------------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------------------------------
    @cute.kernel
    def kern_ws(
        self,
        logits: cute.Tensor,
        out: cute.Tensor,
        kv_lens: cute.Tensor,
        npad: Int32,
        nrows: Int32,
        aim_base: Int32,
        sfac: Int32,
        sd_en: Int32,
    ):
        K = self.kk
        k = Int32(K)
        NCT = WS_NCT
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rcta = Int32(0)  # rows of this CTA: bid, bid + G, ...
        if bid < nrows:
            rcta = (nrows - bid + gdim - Int32(1)) // gdim

        # ---- shared memory ----
        smem = SmemAllocator()
        # consumers
        s_hist = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((self.hb,), order=(0,)), byte_alignment=128
        )
        s_res = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        s_scal = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        s_kmm = smem.allocate_tensor(
            Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        s_ws = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((WS_NCW,), order=(0,)), byte_alignment=16
        )
        s_cbuf2 = smem.allocate_tensor(
            cutlass.Uint64,
            cute.make_ordered_layout((self.scpb + 4,), order=(0,)),
            byte_alignment=16,
        )
        s_cbuf = cute.make_tensor(  # Int32 view (the _emitc signature)
            cute.make_ptr(
                Int32, s_cbuf2.iterator.toint(), cute.AddressSpace.smem, assumed_align=16
            ),
            cute.make_layout((self.scpb + 4,)),
        )
        s_ck64 = smem.allocate_tensor(
            cutlass.Uint64,
            cute.make_ordered_layout((self.cmpb + 1,), order=(0,)),
            byte_alignment=16,
        )
        out_row = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((K,), order=(0,)), byte_alignment=16
        )
        ring = smem.allocate_tensor(
            cutlass.Int8,
            cute.make_ordered_layout((self.ring_stages * self.stage_bytes,), order=(0,)),
            byte_alignment=128,
        )
        # planner
        phist = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((NB_PLAN,), order=(0,)), byte_alignment=128
        )
        s_resp = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        sbuf = smem.allocate_tensor(
            cutlass.Int8, cute.make_ordered_layout((self.sb_bytes,), order=(0,)), byte_alignment=128
        )
        # planner -> consumer slots + mbarriers (thr_full[2], thr_empty[2], ring full[S], empty[S])
        s_lad = smem.allocate_tensor(
            Int32, cute.make_ordered_layout((2 * LADW,), order=(0,)), byte_alignment=16
        )
        mbar = smem.allocate_tensor(
            Int64,
            cute.make_ordered_layout((4 + 2 * self.ring_stages,), order=(0,)),
            byte_alignment=8,
        )
        thr_full = mbar.iterator
        thr_empty = mbar.iterator + 2
        S = self.ring_stages
        ring_full = mbar.iterator + 4
        ring_empty = mbar.iterator + 4 + S

        x_base = logits.iterator.toint()
        o_base = out.iterator.toint()
        rowb = Int64(npad) * Int64(self.eb)  # row stride in bytes
        s0 = Int32(0)  # ring loads issued before the CTA barrier (producer lane)
        # the planner's row-0 sample gathers leave before the producer's first ring loads
        if warp == Int32(_PLANNER_WARP):
            if rcta > Int32(0):
                nv0 = self._row_len(kv_lens, bid, npad)
                if nv0 > k:
                    smp0, ss20, _t0, _t1, _t2 = self._ladder(nv0, aim_base, sfac, sd_en)
                    jn0 = smp0
                    if jn0 > Int32(self.spr):
                        jn0 = Int32(self.spr)
                    self._gather_issue(
                        x_base + Int64(bid) * rowb,
                        sbuf.iterator.toint(),
                        Int32(0),
                        jn0,
                        ss20,
                        tidx & Int32(31),
                    )
            cute.arch.barrier_arrive(barrier_id=BAR_EI, number_of_threads=64)
        if warp == Int32(_PRODUCER_WARP):
            cute.arch.barrier(barrier_id=BAR_EI, number_of_threads=64)

        if tidx < Int32(NB_PLAN):
            phist[tidx] = Int32(0)
        for z in cutlass.range_constexpr((self.hb + 255) // 256):
            if tidx < Int32(min(256, self.hb - 256 * z)):
                s_hist[tidx + Int32(256 * z)] = Int32(0)
        if tidx < Int32(8):
            s_res[tidx] = Int32(0)
            s_resp[tidx] = Int32(0)
        if tidx == Int32(0):
            s_scal[0] = Int32(0)
            for q in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(thr_full + q, 1)
                cute.arch.mbarrier_init(thr_empty + q, 1)
            for q in cutlass.range_constexpr(S):
                cute.arch.mbarrier_init(ring_full + q, 1)
                cute.arch.mbarrier_init(ring_empty + q, WS_NCW)
            cute.arch.mbarrier_init_fence()
            s0 = self._issue_first(kv_lens, bid, npad, rcta, rowb, x_base, ring, ring_full)
        cute.arch.barrier()  # the only CTA-wide barrier

        # ================================ producer ================================
        if warp == Int32(_PRODUCER_WARP):
            if tidx == Int32(0):
                pol = C._l2_evict_first_policy()
                ring_p = ring.iterator.toint()
                ps = (
                    s0  # ring stage / phase of the next load (s0 loads went out before the barrier)
                )
                pph = Int32(0)
                if ps == Int32(S):
                    ps = Int32(0)
                    pph = Int32(1)
                for i in cutlass.range(rcta, unroll=1):
                    row = bid + i * gdim
                    nv = self._row_len(kv_lens, row, npad)
                    if nv > k:
                        nvec, nch = self._row_chunks(nv)
                        c0 = Int32(0)
                        if i == Int32(0):
                            c0 = s0
                        xa = x_base + Int64(row) * rowb
                        for c in cutlass.range(c0, nch, unroll=1):
                            cute.arch.mbarrier_wait(ring_empty + ps, pph ^ Int32(1))
                            self._issue_chunk(
                                xa,
                                nvec,
                                nch,
                                c,
                                ring_p + ps * Int32(self.stage_bytes),
                                ring_full + ps,
                                pol,
                            )
                            ps = ps + Int32(1)
                            if ps == Int32(S):
                                ps = Int32(0)
                                pph = pph ^ Int32(1)

        # ================================ planner ================================
        if warp == Int32(_PLANNER_WARP):
            lane = tidx & Int32(31)
            sbuf_a = sbuf.iterator.toint()
            for i in cutlass.range(rcta, unroll=1):
                row = bid + i * gdim
                slot = i & Int32(1)
                use = i >> Int32(1)
                nv = self._row_len(kv_lens, row, npad)
                short = Int32(0)
                if nv <= k:
                    short = Int32(1)
                x_addr = x_base + Int64(row) * Int64(npad) * Int64(self.eb)
                T = cutlass.Float32(_SENT_LO)
                HIC = cutlass.Float32(_NEG_INF)
                tkc = Int32(-1)
                t5 = cutlass.Float32(_NEG_INF)
                TF0 = cutlass.Float32(0.0)
                SC0 = cutlass.Float32(1.0)
                anch0 = Int32(0)
                if short == Int32(0):
                    pre = Int32(0)
                    if i == Int32(0):
                        pre = Int32(1)  # row 0's first gather round went out in the prologue
                    T, HIC, t5, tkc = self._plan_row(
                        x_addr, nv, aim_base, sfac, sd_en, lane, sbuf_a, phist, s_resp, pre
                    )
                    TF0, SC0, _tf_pair, anch0 = self._window(T, HIC, tkc)
                if lane == Int32(0):
                    cute.arch.mbarrier_wait(thr_empty + slot, (use & Int32(1)) ^ Int32(1))
                    lb = slot * Int32(LADW)
                    s_lad[lb + Int32(L_N)] = nv
                    s_lad[lb + Int32(L_NROW)] = nv
                    s_lad[lb + Int32(L_SHORT)] = short
                    s_lad[lb + Int32(L_T)] = C.i32_of_f32(T)
                    s_lad[lb + Int32(L_HIC)] = C.i32_of_f32(HIC)
                    s_lad[lb + Int32(L_TKC)] = tkc
                    s_lad[lb + Int32(L_T5)] = C.i32_of_f32(t5)
                    s_lad[lb + Int32(L_TF)] = C.i32_of_f32(TF0)
                    s_lad[lb + Int32(L_SC)] = C.i32_of_f32(SC0)
                    s_lad[lb + Int32(L_ANCH)] = anch0
                    cute.arch.mbarrier_arrive(thr_full + slot)
                cute.arch.sync_warp()
                if i == Int32(0):
                    # row 0's slot is published: release the consumers waiting on BAR_S0
                    cute.arch.barrier_arrive(barrier_id=BAR_S0, number_of_threads=WS_NCT + 32)

        # ================================ consumers ================================
        if warp >= Int32(_CONSUMER_WARP0):
            ctid = tidx - Int32(32 * _CONSUMER_WARP0)
            lane = ctid & Int32(31)
            hb_pin = _smem_addr_reg(s_hist.iterator.toint())
            cb2_pin = _smem_addr_reg(s_cbuf2.iterator.toint())
            ring_a = _smem_addr_reg(ring.iterator.toint())
            out_a = out_row.iterator.toint()
            rs = Int32(0)  # ring stage / phase of the next chunk, carried across the CTA's rows
            rph = Int32(0)
            for i in cutlass.range(rcta, unroll=1):
                row = bid + i * gdim
                slot = i & Int32(1)
                use = i >> Int32(1)
                if i == Int32(0):
                    # row 0's thresholds take the planner several microseconds: wait for them on a
                    # hardware barrier rather than in the mbarrier retry loop
                    cute.arch.barrier(barrier_id=BAR_S0, number_of_threads=WS_NCT + 32)
                cute.arch.mbarrier_wait(thr_full + slot, use & Int32(1))
                lb = slot * Int32(LADW)
                n = s_lad[lb + Int32(L_N)]
                n_row = s_lad[lb + Int32(L_NROW)]
                short = s_lad[lb + Int32(L_SHORT)]
                T = C.f32_of_i32(s_lad[lb + Int32(L_T)])
                HIC = C.f32_of_i32(s_lad[lb + Int32(L_HIC)])
                tkc = s_lad[lb + Int32(L_TKC)]
                t5 = C.f32_of_i32(s_lad[lb + Int32(L_T5)])
                TF0 = C.f32_of_i32(s_lad[lb + Int32(L_TF)])
                SC0 = C.f32_of_i32(s_lad[lb + Int32(L_SC)])
                anch0 = s_lad[lb + Int32(L_ANCH)]
                self._clear_hist(s_hist, ctid)
                if ctid == Int32(0):
                    s_scal[0] = Int32(0)
                _bar_cons()  # ---- barrier (row start: slot read, histogram cleared) ----
                if ctid == Int32(0):
                    cute.arch.mbarrier_arrive(thr_empty + slot)
                x_addr = x_base + Int64(row) * rowb
                if short == Int32(0):
                    (listN, above, m, need, B, TF, SC, anch, complete, valid, rs, rph) = (
                        self._attempts(
                            x_addr,
                            n,
                            T,
                            HIC,
                            tkc,
                            t5,
                            TF0,
                            SC0,
                            anch0,
                            ctid,
                            lane,
                            hb_pin,
                            cb2_pin,
                            ring_a,
                            rs,
                            rph,
                            s_hist,
                            s_cbuf,
                            s_cbuf2,
                            s_scal,
                            s_res,
                            s_ws,
                            ring_full,
                            ring_empty,
                        )
                    )
                    self._select(
                        x_addr,
                        n,
                        TF,
                        SC,
                        anch,
                        B,
                        m,
                        above,
                        need,
                        listN,
                        valid,
                        complete,
                        ctid,
                        lane,
                        s_hist,
                        s_cbuf2,
                        s_ck64,
                        s_res,
                        s_scal,
                        s_kmm,
                        out_row,
                    )
                else:
                    # n <= K: identity indices, -1 padding
                    i2 = ctid
                    while i2 < n_row:
                        out_row[i2] = i2
                        i2 = i2 + Int32(NCT)
                    j2 = n_row + ctid
                    while j2 < k:
                        out_row[j2] = Int32(-1)
                        j2 = j2 + Int32(NCT)
                # ---- staged-K copy-out: one 16-B store per thread and step ----
                _bar_cons()
                o_addr = o_base + Int64(row) * Int64(K * 4)
                oi = ctid * Int32(4)
                while oi + Int32(3) < k:
                    a, b, c, d = _lds_v4_u32(out_a + oi * Int32(4))
                    _st_g_v4_b32(o_addr + Int64(oi) * Int64(4), a, b, c, d)
                    oi = oi + Int32(4 * NCT)
                oi_tail = (k & Int32(~3)) + ctid
                if oi_tail < k:
                    C._st_g_u32(o_addr + Int64(oi_tail) * Int64(4), out_row[oi_tail])

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        out: cute.Tensor,
        kv_lens: cute.Tensor,
        grid: Int32,
        npad: Int32,
        nrows: Int32,
        aim_base: Int32,
        sfac: Int32,
        sd_en: Int32,
        stream,
    ):
        self.kern_ws(logits, out, kv_lens, npad, nrows, aim_base, sfac, sd_en).launch(
            grid=(grid, 1, 1), block=(self.nt, 1, 1), stream=stream, min_blocks_per_mp=1
        )


_TALLWS_CACHE = {}


def grid_of(rows: int, num_sms: int) -> int:
    """CTAs of one launch: ``ceil(rows / ceil(rows / num_sms))``, each owning the same number of
    rows up to one."""
    return -(-rows // -(-rows // num_sms))


def get_compiled__tallws(dtype: str, k: int, next_n: int = 1, cr_shift: int = 0):
    """Compiled ``GvrTallWsKernel`` launcher for logits dtype tag ``dtype`` ('f32', 'bf16' or 'f16'),
    K, next_n and cr_shift, cached per key.

    Call: ``fn(logits [rows, npad], out [rows, K] int32, kv_lens [rows // next_n] int32, grid,
    npad, rows, aim_base, sfac, sd_en)`` with ``grid = grid_of(rows, SMs)`` and
    ``(aim_base, sfac, sd_en) = WS_LADDER[K]``; ``npad`` is a multiple of the 16-B vector and the
    logits and out bases are 16-B aligned. The build depends on nothing else: the ring, sample
    buffer, window bins and crossing-bin rank follow from (dtype, K).
    """
    key = (dtype, int(k), int(next_n), int(cr_shift))
    hit = _TALLWS_CACHE.get(key)
    if hit is not None:
        return hit
    kern = GvrTallWsKernel(_dtype_of(dtype), k, next_n=next_n, cr_shift=cr_shift)
    r0, c0 = cute.sym_int(), cute.sym_int()
    r2 = cute.sym_int()
    logits_fake = _crt.make_fake_compact_tensor(
        kern.dtype, (r0, c0), stride_order=(1, 0), assumed_align=16
    )
    out_fake = _crt.make_fake_compact_tensor(Int32, (r2, k), stride_order=(1, 0), assumed_align=16)
    kv_fake = _crt.make_fake_compact_tensor(
        Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
    )
    compiled = cute.compile(
        kern,
        logits_fake,
        out_fake,
        kv_fake,
        *([Int32(0)] * 6),
        stream=_crt.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _TALLWS_CACHE[key] = compiled
    return compiled


__all__ = ["WS_KS", "WS_LADDER", "GvrTallWsKernel", "get_compiled__tallws", "grid_of"]
