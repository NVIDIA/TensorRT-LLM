# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# ===========================================================================
# ==== family: clus ============================================
# ===========================================================================
"""gvr_clus — clustered streaming self-sampling GVR; per-CTA stream mirrors
gvr_main.

Ctor knobs (compile-time, mirror of the CUDA template params):
    BLK = 1024, U ∈ {1,2,4,8}, MINB = 1, NBS = 256, CS ∈ {2,4,8}
    (+ scap/cmp smem-extent knobs: every reachable route has 8192/2048 —
     SCAP/CMP stay LIVE runtime args for all value logic, ABI parity).
Derived: HB=NBS, STEPC=BLK*U, PFD=min(U,4).

Signature (ABI parity with the CUDA form; Q is dead in-kernel):
    kern(..., n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2)
Grid dim3(CS, b) native 2-D + cluster (CS,1,1); block 1024;
min_blocks_per_mp=1 (64-register wall); smem one SmemAllocator blob
mirroring the CUDA dynamic map hist|cbuf|ck64c|mrg, dyn-equivalent bytes ==
the host dispatch formula.

int2 staging convention (same as gvr_main): int2(value bits, index) is ONE
little-endian Uint64 = (idx << 32) | value_bits — single u64 smem ld/st.

Barrier / cluster-op placement mirrors the CUDA source one-for-one:
    sample redux publish, sample hist, scan publish,
    [degenerate sample: 2 inside gather_hint],
    retry preamble: clus.sync + __syncthreads,
    clus.sync (merge), __syncthreads (merge publish),
    [ladder gather: 2 inside gather_hint],
    clus.sync (EXIT RENDEZVOUS — the only one; rank!=0 falls through),
    narrowing / degen per-level barriers (+1 INSIDE scan_cross).
    NO loop-tail ladder barriers (gvr_clus has none — unlike gvr_main).
    All clus.sync = releasing aligned arrive+wait, never relaxed.
Cluster ops: merge = _merge_scan0_local, a LOCAL patched copy of
    merge_scan0 that rematerializes mapa per (q, r) like the CUDA
    (register-pressure fix); ONE packed u64 st.shared::cluster candidate
    push to rank-0 ck64c (never split into 4B stores); mapa of ck64c to
    rank 0.
Every rung/ladder decision is cluster-uniform by construction (identical
sample locations on every rank; merged tot; block-uniform gather) — the
conditional retry clus.sync cannot deadlock.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cute import runtime as _crt
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

# The shared device units, the main family and the families before this one, from the module that
# re-exports this one (``C`` as in that module, where it is the module itself).
try:
    from . import gvr_topk_decode as C
except ImportError:  # standalone dir
    import gvr_topk_decode as C

_lds_v2_u32 = C._lds_v2_u32
_smem_addr_reg = C._smem_addr_reg


QUADC_CLUS__clus = C.QUADC_CLUS

_NEG_INF__clus = float("-inf")


# ---------------------------------------------------------------------------
# single-rounding fma.rn.f32, used at the T / Tk / T3 / HIC sites.
# (x-TF)*SC classify shapes stay plain sub+mul (uncontractible).
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf__clus(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        mlir_math.fma(
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            fastmath=mlir_arith.FastMathFlags.none,
            loc=loc,
            ip=ip,
        )
    )


class GvrClusKernel:
    """gvr_clus<BLK, U, MINB, NBS, CS> — clustered streaming GVR."""

    def __init__(
        self,
        blk: int,
        u: int,
        minb: int,
        nbs: int,
        cs: int,
        scap: int = 8192,
        cmp_: int = 2048,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        hint_free: bool = False,
        det: bool = False,
    ) -> None:
        self.det = bool(det)
        assert blk == 1024, "gvr_clus is always BLK=1024"
        assert minb == 1, "gvr_clus is __launch_bounds__(BLK, 1)"
        assert nbs == 256, "SNB must stay 256"
        assert u in (1, 2, 4, 8) and cs in (2, 4, 8)
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel / GvrRegClusKernel): n and the sampling-
        # ladder scalars are re-derived PER ROW in-kernel from a device
        # kv_lens tensor; the scalar launch args become the envelope clamp
        # bound (n) and dead slots (SMP/TGT/Q/SS2/TGT2).
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        self.hint_free = bool(hint_free)  # hint-free: gather_hint sites compiled out
        self.lcs = cs.bit_length() - 1  # log2(CS) for the per-row Q shift
        self.blk = blk
        self.u = u
        self.minb = minb
        self.nbs = nbs
        self.cs = cs
        self.scap = scap  # smem extents only —
        self.cmp = cmp_  # value logic uses rt args
        self.hb = nbs
        self.stepc = blk * u
        self.pfd = u if u < 4 else 4  # PFD=min(U,4)
        self.lb = nbs.bit_length() - 1  # log2(NBS)=8
        # dynamic-region byte map: hist | cbuf(int2) | ck64c | mrg
        self.cbuf_bytes = (scap + 4) * 8
        assert self.cbuf_bytes % 16 == 0
        self.ck_off = self.cbuf_bytes  # inside the blob
        self.dyn_bytes = nbs * 4 + self.cbuf_bytes + cmp_ * 8 + nbs * 4
        # == host smc = SNB*8 + (SCAP+4)*8 + CMP*8

    # ------------------------------------------------------------------
    # GVR_EMITK: classify+stage one survivor.
    # bn via UNSIGNED saturating convert (f2u_rz); staging store is ONE
    # u64; branchless trash slot min(pos, SCAP) (runtime SCAP). Returns pos+1.
    # ------------------------------------------------------------------
    @cute.jit
    def _emitk(self, xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2):
        NBS = self.nbs
        bn_u = C.f2u_rz((xv - TF) * SC)
        if bn_u > cutlass.Uint32(NBS - 1):
            bn_u = cutlass.Uint32(NBS - 1)
        bn = cutlass.Int32(bn_u)
        C.atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
        ps = pos
        if ps > SCAP:
            ps = SCAP  # trash slot (IMNMX)
        if cutlass.const_expr(self.det):
            s_cbuf2[ps] = (
                cutlass.Uint64(cutlass.Uint32(idx) | (cutlass.Uint32(bn_u) << cutlass.Uint32(24)))
                << cutlass.Uint64(32)
            ) | cutlass.Uint64(C.u32_of_f32(xv))  # CLUS FAST: bin in bits 24..31
        else:
            s_cbuf2[ps] = (
                cutlass.Uint64(cutlass.Uint32(idx)) << cutlass.Uint64(32)
            ) | cutlass.Uint64(C.u32_of_f32(xv))
        return pos + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # P5 emit step: bn via SIGNED rz convert (__float2int_rz); bn>=B gate;
    # LOCAL mrg atomicAdd whose result is a CLUSTER-GLOBAL position
    # (prefix-biased cursors from merge_scan0); overflow -> ONE packed u64
    # DSMEM store to rank-0 ck64c (never split into 4B stores).
    # ------------------------------------------------------------------
    @cute.jit
    def _p5_emit(self, xv, idv, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64):
        NBS = self.nbs
        bn = C.f2s_rz((xv - TF) * SC)
        if bn > cutlass.Int32(NBS - 1):
            bn = cutlass.Int32(NBS - 1)
        if bn >= B:
            p = C.atomic_add_cta(s_mrg.iterator + bn, cutlass.Int32(1))
            if p < lim1:
                out_row[p] = idv
            else:
                if whole == cutlass.Int32(0):
                    q2 = p - above
                    if q2 < CMP:
                        C._st_shared_cluster_u64(
                            rk64 + q2 * cutlass.Int32(8),
                            (cutlass.Uint64(C.fkey(xv)) << cutlass.Uint64(32))
                            | cutlass.Uint64(cutlass.Uint32(idv)),
                        )

    # ------------------------------------------------------------------
    # LOCAL patched copy of merge_scan0: rematerializes mapa per (q, r)
    # exactly like the CUDA instead of holding CS mapped base addresses
    # across the whole merge. The hoisted-array form costs CS extra
    # long-lived registers; with the U>=4 sixteen-register pf prime batch
    # it tips ptxas into spilling the batch across the rung phase.
    # Semantics, the DSMEM v4 load spelling, the register accumulation and
    # the prefix-biased STS.128 cursor write are IDENTICAL to merge_scan0.
    # NO barrier inside (caller pays the merge-publish barrier).
    # ------------------------------------------------------------------
    @cute.jit
    def _merge_scan0_local(self, s_hist, s_mrg, rank, target, tidx, s_res):
        NBS = self.nbs
        CS = self.cs
        BPT = NBS // 32
        NV = BPT // 4
        if tidx < cutlass.Int32(32):
            lane = tidx
            atom = C.smem_atom_i32_128()
            hbase = s_hist.iterator.toint()
            # pass 1: remote v4 accumulation of tot/pre per vector
            tot_r = []
            pre_r = []
            sm = cutlass.Int32(0)
            for q in cutlass.range_constexpr(NV):
                boff = (lane * cutlass.Int32(BPT) + cutlass.Int32(4 * q)) * cutlass.Int32(4)
                t = [cutlass.Int32(0)] * 4
                p = [cutlass.Int32(0)] * 4
                for r in cutlass.range_constexpr(CS):
                    mapped = C._mapa_shared_cluster_addr(
                        hbase + boff, cutlass.Int32(r)
                    )  # per-use mapa
                    v0, v1, v2, v3 = C._ld_shared_cluster_v4_u32(mapped)
                    t[0] = t[0] + v0
                    t[1] = t[1] + v1
                    t[2] = t[2] + v2
                    t[3] = t[3] + v3
                    if cutlass.Int32(r) < rank:  # predicated adds
                        p[0] = p[0] + v0
                        p[1] = p[1] + v1
                        p[2] = p[2] + v2
                        p[3] = p[3] + v3
                tot_r.append(t)
                pre_r.append(p)
                sm = sm + t[0] + t[1] + t[2] + t[3]
            # inclusive scan + totals
            w = C.warp_incl_scan_add(sm, lane)
            tt = cute.arch.shuffle_sync(w, cutlass.Int32(31))
            after = tt - w
            if lane == cutlass.Int32(0):
                s_res[C.RES_TOT] = tt
            base = lane * cutlass.Int32(BPT)
            # descending walk: crossing pin + prefix-biased cursors into mrg
            for q in cutlass.range_constexpr(NV - 1, -1, -1):
                o4 = cute.make_rmem_tensor((4,), cutlass.Int32)
                for j in cutlass.range_constexpr(3, -1, -1):
                    cq = tot_r[q][j]
                    o4[j] = after + pre_r[q][j]
                    gb = base + cutlass.Int32(4 * q + j)
                    cross = cutlass.Int32(0)
                    if after < target:
                        if (after + cq) >= target:
                            cross = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross = cutlass.Int32(1)
                    if cross != cutlass.Int32(0):
                        s_res[C.RES_B] = gb
                        s_res[C.RES_ABOVE] = after
                        s_res[C.RES_M] = cq
                    after = after + cq
                boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                C.sts128_i32(atom, o4, s_mrg.iterator.toint(), boff)

    # ------------------------------------------------------------------
    # two-predicate warp-ballot emit step (narrowing and degen emits) —
    # same helper as the main family. s_scal[1]=s_o1, s_scal[2]=s_o2.
    # ------------------------------------------------------------------
    @cute.jit
    def _ballot_pair_emit(self, p1, p2, idv, base1, cap1, base2, cap2, out_row, s_scal, lane):
        n1 = C.ballot(p1 != cutlass.Int32(0))
        n2 = C.ballot(p2 != cutlass.Int32(0))
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        if lane == cutlass.Int32(0):
            if n1 != cutlass.Int32(0):
                b1 = C.atomic_add_cta(s_scal.iterator + 1, cutlass.Int32(C.popc(n1)))
            if n2 != cutlass.Int32(0):
                b2 = C.atomic_add_cta(s_scal.iterator + 2, cutlass.Int32(C.popc(n2)))
        b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
        b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
        lm = cutlass.Int32(cute.arch.lanemask_lt())
        if p1 != cutlass.Int32(0):
            p = b1 + cutlass.Int32(C.popc(n1 & lm))
            if p < cap1:
                out_row[base1 + p] = idv
        if p2 != cutlass.Int32(0):
            p = b2 + cutlass.Int32(C.popc(n2 & lm))
            if p < cap2:
                out_row[base2 + p] = idv

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        SCAP: cutlass.Int32,
        CMP: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        bigf: cutlass.Int32,
    ):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        CS = self.cs
        PFD = self.pfd
        STEPC = self.stepc
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()  # (rank, row)
        rank = bx
        row = by
        lane = tidx & cutlass.Int32(31)
        if cutlass.const_expr(self.det):
            wid = tidx >> cutlass.Int32(5)  # CLUS FAST

        # ============ per-row varlen prologue — shared contract lives in ======
        # GvrMainKernel's prologue (per-row n from kv_lens; ladder scalars are
        # dead launch args, re-derived here by the route_dynamic() clus
        # formulas). Clus deviation: QUAD sample geometry runs for every
        # non-short row (host never launches clus at n <= SCAP, so SMP == 0 is
        # untested; sampling only steers the rung — exactness is
        # schedule-invariant). All values are pure functions of `row`, so the
        # whole-body guard and the cluster barriers stay cluster-uniform.
        # Short rows (n <= k): rank 0 emits identity + (-1); body is SKIPped.
        short = cutlass.Int32(0)
        prow = row
        if cutlass.const_expr(self.varlen):
            kq = cutlass.Int32(pre_idx.shape[1])
            req = row // cutlass.Int32(self.next_n)
            rr = row % cutlass.Int32(self.next_n)
            prow = req
            kvl = kv_lens[req]
            nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                self.cr_shift
            )
            if nv < cutlass.Int32(0):
                nv = cutlass.Int32(0)
            if nv > n:
                nv = n
            if nv <= kq:
                short = cutlass.Int32(1)
            if short == cutlass.Int32(0):
                n = nv
                # ---- aim ladder (cheap mirror, all-thread) ----
                # Schedule quantities only (exactness is schedule-invariant,
                # same argument as the always-sample deviation above): divides
                # become MUFU.RCP multiplies and the isqrt fixup loops collapse
                # to single steps (f32 sqrt of an exactly-representable int
                # (6n <= 2^23) is within 1 of isqrt).  All-thread on purpose:
                # this family's mirror redundancy is small, and a
                # warp0+barrier hoist EXPOSES the chain's serial latency at
                # ~1 CTA/SM, while the redundant form hides it across warps.
                # Q (chunk ownership) keeps its exact shift form.
                x6 = cutlass.Int32(6) * nv
                ri = cutlass.Int32(cmath.sqrt(cutlass.Float32(x6)))
                if ri * ri > x6:
                    ri = ri - cutlass.Int32(1)
                if (ri + cutlass.Int32(1)) * (ri + cutlass.Int32(1)) <= x6:
                    ri = ri + cutlass.Int32(1)
                r6 = ri
                if x6 - ri * ri > ri:
                    r6 = ri + cutlass.Int32(1)
                # aim_base: R = CS > 1 always for this family; bigf is the
                # launch-computed occupancy flag (num_rows * CS <= 148).
                aim = k << cutlass.Int32(1)
                if bigf == cutlass.Int32(0):
                    if k >= cutlass.Int32(1024):
                        aim = (cutlass.Int32(11) * k) >> cutlass.Int32(3)
                    else:
                        aim = (cutlass.Int32(3) * k) >> cutlass.Int32(1)
                if r6 > aim:
                    aim = r6
                amin = cutlass.Int32(3) * k
                if cutlass.const_expr(self.cs != 2):
                    amin = (cutlass.Int32(7) * k) >> cutlass.Int32(1)
                if aim < amin:
                    aim = amin
                if aim > (SCAP >> cutlass.Int32(1)):
                    aim = SCAP >> cutlass.Int32(1)
                if aim < k:
                    aim = k
                # ---- QUAD sample geometry (route_dynamic clus override,
                # always-on per the deviation note above; k <= 1024 for this
                # family so sfac has no k > 1024 arm).
                n4v = nv >> cutlass.Int32(2)
                sel = cutlass.Int32(
                    cutlass.Float32(nv)
                    * cutlass.Float32(32.0 if self.cs == 2 else 16.0)
                    * cute.arch.rcp_approx(cutlass.Float32(aim))
                )
                if sel < cutlass.Int32(256):
                    sel = cutlass.Int32(256)
                nh = nv >> cutlass.Int32(1)
                if sel > nh:
                    sel = nh
                quads = sel >> cutlass.Int32(4)
                if quads < cutlass.Int32(1):
                    quads = cutlass.Int32(1)
                quarter = n4v >> cutlass.Int32(2)
                if quarter < cutlass.Int32(1):
                    quarter = cutlass.Int32(1)
                if quads > quarter:
                    quads = quarter
                SS2 = cutlass.Int32(
                    cutlass.Float32(quarter) * cute.arch.rcp_approx(cutlass.Float32(quads))
                )
                if SS2 < cutlass.Int32(1):
                    SS2 = cutlass.Int32(1)
                SMP = cutlass.Int32(
                    cutlass.Float32(quarter) * cute.arch.rcp_approx(cutlass.Float32(SS2))
                )
                # sample-window guard: P1 indexes up to ~SMP*SS2*4 lines; keep
                # SMP*SS2 <= quarter (approx error <= +1, one step closes it)
                if SMP * SS2 > quarter:
                    SMP = SMP - cutlass.Int32(1)
                if SMP < cutlass.Int32(1):
                    SMP = cutlass.Int32(1)
                rn_ = cute.arch.rcp_approx(cutlass.Float32(nv))
                smp16f = cutlass.Float32(SMP) * cutlass.Float32(16.0)
                TGT = cutlass.Int32(cutlass.Float32(aim) * smp16f * rn_)
                if TGT < cutlass.Int32(1):
                    TGT = cutlass.Int32(1)
                TGT2 = cutlass.Int32(cutlass.Float32(k) * smp16f * rn_)
                if TGT2 < cutlass.Int32(1):
                    TGT2 = cutlass.Int32(1)
                # NB: the Q launch slot is dead in this family (no in-kernel
                # consumer); chunk ownership derives from n4/STEPC below.
            if short != cutlass.Int32(0):
                if rank == cutlass.Int32(0):
                    if tidx < kq:
                        ov = cutlass.Int32(-1)
                        if tidx < nv:
                            ov = tidx
                        out[row, tidx] = ov

        # ---- shared memory (CUDA dynamic map order, then static allocs) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(  # hist[NBS] @ blob start
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)), byte_alignment=128
        )
        blob = smem.allocate_tensor(  # cbuf(int2) | ck64c
            cutlass.Int8,
            cute.make_ordered_layout((self.cbuf_bytes + self.cmp * 8,), order=(0,)),
            byte_alignment=16,
        )
        s_mrg = smem.allocate_tensor(  # mrg[NBS]
            cutlass.Int32, cute.make_ordered_layout((self.nbs,), order=(0,)), byte_alignment=16
        )
        s_ws = smem.allocate_tensor(  # degen scan only
            cutlass.Int32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        # scan_cross predeclares its second-stage partial as Int32 and reads
        # s_ws inside a dynamic if — a Uint32 ws tensor trips the DSL type-
        # stability check (counts < 2^31 so Int32 is exact).
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_res = smem.allocate_tensor(  # shared slot map
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        s_tsh = smem.allocate_tensor(
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
        )
        if cutlass.const_expr(self.det):
            s_wcnt = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((5 * NW,), order=(0,)), byte_alignment=16
            )  # CLUS FAST: per-warp counts / class counts / rank-base partials
            s_prebin = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((NBS,), order=(0,)), byte_alignment=16
            )  # CLUS FAST: lower-rank counts per bin
        s_kmm = smem.allocate_tensor(  # [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        sbase = blob.iterator.toint()
        if cutlass.const_expr(self.det):
            wcap_ = cutlass.Int32(0)  # CLUS FAST
            wbase = cutlass.Int32(0)
            wend = cutlass.Int32(0)
            wovf = cutlass.Int32(0)
        s_cbuf2 = cute.make_tensor(  # int2 staged as u64
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.scap + 4,)),
        )
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.cmp,)),
        )

        # ---- whole-body short-row guard (cluster-uniform: `short` is a pure
        # function of `row`, identical across all CS ranks and threads, so
        # every cluster barrier below stays aligned; short rows already
        # emitted identity + -1 tail in the prologue) ----
        if short == cutlass.Int32(0):
            # ---- row bases (pre_idx is request-level under varlen) ----
            row64 = cutlass.Int64(row)
            x_addr = logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(4)
            p_addr = pre_idx.iterator.toint() + cutlass.Int64(prow) * cutlass.Int64(
                k
            ) * cutlass.Int64(4)
            out_row = out[row, None]

            # ---- interleaved chunk ownership ----
            n4 = n >> cutlass.Int32(2)
            nCh = (n4 + cutlass.Int32(STEPC - 1)) // cutlass.Int32(STEPC)
            nFullG = n4 // cutlass.Int32(STEPC)
            tail0 = n4 << cutlass.Int32(2)
            tailn = cutlass.Int32(0)
            if rank == cutlass.Int32(0):
                tailn = n - tail0

            if tidx == cutlass.Int32(0):
                s_res[C.RES_B2] = cutlass.Int32(-1)
                s_res[C.RES_B3] = cutlass.Int32(-1)
                s_scal[0] = cutlass.Int32(0)  # s_bufn
                if cutlass.const_expr(self.det):
                    s_scal[3] = cutlass.Int32(0)  # CLUS FAST: segment overflow
            if tidx < cutlass.Int32(self.hb):  # HB<=BLK
                s_hist[tidx] = cutlass.Int32(0)

            # ============ P1: QUAD sample (hint gather LAZY) =====================
            # one 64B line = 4 float4 per location, TWO threads: tid takes the
            # lower pair at p4, tid+SMP the upper pair at p4+2.
            atom128 = C.g2r_atom_f32(128, invariant=True)
            fsa = cute.make_rmem_tensor((4,), cutlass.Float32)
            fsb = cute.make_rmem_tensor((4,), cutlass.Float32)
            smp2 = SMP * cutlass.Int32(2)
            shas = cutlass.Int32(0)
            if tidx < smp2:
                shas = cutlass.Int32(1)
            if shas != cutlass.Int32(0):
                p4 = tidx * SS2 * cutlass.Int32(4)
                if tidx >= SMP:
                    p4 = (tidx - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                C.ld_g_f32x4(atom128, x_addr, p4, fsa)
                C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)

            # ============ P2: quantile rung, redundant per CTA ===================
            smn = cutlass.Float32(float("inf"))
            smx = cutlass.Float32(float("-inf"))
            if shas != cutlass.Int32(0):
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fsa[t])
                    smx = C.fmax_f32(smx, fsa[t])
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fsb[t])
                    smx = C.fmax_f32(smx, fsb[t])
            fma_ = cute.make_rmem_tensor((4,), cutlass.Float32)  # mop-up pair bufs
            fmb_ = cute.make_rmem_tensor((4,), cutlass.Float32)
            j = tidx + cutlass.Int32(BLK)  # mop-up
            while j < smp2:
                p4 = j * SS2 * cutlass.Int32(4)
                if j >= SMP:
                    p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fma_[t])
                    smx = C.fmax_f32(smx, fma_[t])
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fmb_[t])
                    smx = C.fmax_f32(smx, fmb_[t])
                j = j + cutlass.Int32(BLK)
            a0 = C.warp_min_u32(C.fkey(smn))
            c0m = C.warp_max_u32(C.fkey(smx))
            if lane == cutlass.Int32(0):
                s_wmn[tidx >> cutlass.Int32(5)] = a0
                s_wmx[tidx >> cutlass.Int32(5)] = c0m
            cute.arch.barrier()  # ---- barrier (sample redux publish) ----

            # PRIME-LATE: every rank's sample has landed; prime NOW.
            lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
            pf = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(PFD)]
            for uu in cutlass.range_constexpr(PFD):  # clamped prime
                i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                ic = i_
                if ic >= n4:
                    ic = lim4
                C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
            # asm prefetch gate: DEEP rows only; empty for U<=PFD
            if cutlass.const_expr(U > PFD):
                gpp = cutlass.Int32(0)
                if n4 >= cutlass.Int32(32768):
                    if (rank + cutlass.Int32(1)) * cutlass.Int32(STEPC) <= n4:
                        gpp = cutlass.Int32(1)
                if gpp != cutlass.Int32(0):
                    for uu in cutlass.range_constexpr(PFD, U):
                        C._prefetch_l2(
                            x_addr
                            + cutlass.Int64(
                                rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                            )
                            * cutlass.Int64(16)
                        )

            # cross-warp sample reduce
            av = cutlass.Uint32(0xFFFFFFFF)
            cv = cutlass.Uint32(0)
            if lane < cutlass.Int32(NW):
                av = s_wmn[lane]
                cv = s_wmx[lane]
            SMIN = C.invkey(C.warp_min_u32(av))
            SMAX = C.invkey(C.warp_max_u32(cv))

            GMIN = cutlass.Float32(C.SENT_LO)  # sentinels
            GMAX = cutlass.Float32(C.SENT_HI)
            T = cutlass.Float32(_NEG_INF__clus)
            HIC = cutlass.Float32(_NEG_INF__clus)
            w = cutlass.Float32(0.0)
            sok = cutlass.Int32(0)
            if SMP > cutlass.Int32(0):
                if SMAX > SMIN:
                    sok = cutlass.Int32(1)
            if sok != cutlass.Int32(0):  # sample hist
                w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
                # CUDA --use_fast_math lowers `1.0f / w` to a bare MUFU.RCP;
                # a plain `/` would emit the IEEE div.rn rcp+Newton+CALL
                # chain. w > 0 by the sok guard; bucketing is SC-invariant.
                sc_s = cute.arch.rcp_approx(w)
                if shas != cutlass.Int32(0):
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fsa[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fsb[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                j = tidx + cutlass.Int32(BLK)  # mop-up reloads
                while j < smp2:
                    p4 = j * SS2 * cutlass.Int32(4)
                    if j >= SMP:
                        p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                    C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                    C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fma_[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fmb_[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    j = j + cutlass.Int32(BLK)
            cute.arch.barrier()  # ---- barrier (sample histogram) ----
            # triple-target ZERO scan: TGT / TGT2 / 2*TGT
            C.scan_cross0(
                s_hist,
                TGT,
                tidx,
                s_res,
                TGT2,
                TGT * cutlass.Int32(2),
                s_hist,
                nb=NBS,
                zero=True,
                two=True,
                three=True,
            )
            cute.arch.barrier()  # ---- barrier (scan publish) ----

            tot0 = s_res[C.RES_TOT]
            b1v = s_res[C.RES_B]
            if sok != cutlass.Int32(0):
                if tot0 >= TGT:
                    T = _fmaf__clus(cutlass.Float32(b1v), w, SMIN)
            needg = cutlass.Int32(1)
            if T > cutlass.Float32(_NEG_INF__clus):
                needg = cutlass.Int32(0)
            if needg != cutlass.Int32(0):
                # degenerate sample: identical on every rank of the cluster
                if cutlass.const_expr(not self.hint_free):
                    GMIN, GMAX = C.gather_hint(
                        x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=1
                    )  # 2 barriers
                T = GMIN
            if sok != cutlass.Int32(0):  # HIC tighten
                if tot0 >= TGT:
                    b2v = s_res[C.RES_B2]
                    if b2v >= cutlass.Int32(0):
                        Tk = _fmaf__clus(cutlass.Float32(b2v), w, SMIN)
                        up = C.fmax_f32(Tk - T, cutlass.Float32(0.0))
                        # heavy-tail cap by T - T3 (rank-TGT..rank-2TGT distance)
                        if tot0 >= TGT * cutlass.Int32(2):
                            b3v = s_res[C.RES_B3]
                            if b3v >= cutlass.Int32(0):
                                T3 = _fmaf__clus(cutlass.Float32(b3v), w, SMIN)
                                if T > T3:
                                    up = C.fmin_f32(up, cutlass.Float32(2.0) * (T - T3))
                        HIC = C.fmax_f32(
                            _fmaf__clus(cutlass.Float32(4.0), up, T),
                            _fmaf__clus(cutlass.Float32(8.0), w, T),
                        )
            # ladder floor kept in SHARED (64-register wall)
            if tidx == cutlass.Int32(0):
                t5 = cutlass.Float32(_NEG_INF__clus)
                if sok != cutlass.Int32(0):
                    if tot0 >= TGT * cutlass.Int32(2):
                        b3v = s_res[C.RES_B3]
                        if b3v >= cutlass.Int32(0):
                            if T > GMIN:
                                T3 = _fmaf__clus(cutlass.Float32(b3v), w, SMIN)
                                if T3 < T:
                                    t5 = T3
                s_tsh[0] = t5

            # ============ attempt loop — MUST NOT unroll ===========
            if cutlass.const_expr(not self.det):
                listN = cutlass.Int32(0)
            above = cutlass.Int32(0)
            m = cutlass.Int32(0)
            need = cutlass.Int32(0)
            B = cutlass.Int32(0)
            SC = cutlass.Float32(1.0)
            TF = T
            complete = cutlass.Int32(0)
            valid = cutlass.Int32(0)

            fr = [
                cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(U - PFD)
            ]  # explicit batch
            # (empty for U<=PFD — every row-pass float4 then comes from pf[])
            att = cutlass.Int32(0)
            running = cutlass.Int32(1)
            while running != cutlass.Int32(0):
                if att > cutlass.Int32(0):  # retry preamble
                    # exactness: re-prime pf[] (holds stale roll data)
                    if rank < nFullG:
                        for uu in cutlass.range_constexpr(PFD):
                            C.ld_g_f32x4(
                                atom128,
                                x_addr,
                                rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK),
                                pf[uu],
                            )
                    else:
                        for uu in cutlass.range_constexpr(PFD):
                            i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= n4:
                                ic = lim4
                            C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                    C._cluster_sync_aligned()  # ==== clus.sync (retry) ====
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                        if cutlass.const_expr(self.det):
                            s_scal[3] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (retry reset) ----

                TF = T  # window
                hi = C.fmax_f32(GMAX, T)
                if HIC > T:
                    if HIC < hi:
                        hi = HIC
                WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
                wdok = cutlass.Int32(0)
                if WD > cutlass.Float32(0.0):
                    wdok = cutlass.Int32(1)
                if wdok == cutlass.Int32(0):
                    WD = cutlass.Float32(1e-30)
                # MUFU.RCP spelling: WD >= 1e-30 finite by the wdok clamp;
                # classify bucketing is SC-invariant for any SC > 0.
                SC = cute.arch.rcp_approx(WD)
                if cutlass.const_expr(self.det):
                    wcap_ = ((SCAP >> cutlass.Int32(5)) >> cutlass.Int32(5)) << cutlass.Int32(
                        5
                    )  # CLUS FAST: per-warp segment (runtime SCAP)
                    wbase = wid * wcap_
                    wend = wbase + wcap_
                    wovf = cutlass.Int32(0)

                # ---- P3 row pass over OWNED CHUNKS ----
                g = rank + cutlass.Int32(0)
                if cutlass.const_expr(self.det):
                    while g < nCh:
                        i0 = g * cutlass.Int32(STEPC) + tidx
                        M = cutlass.Int32(0)
                        isfull = cutlass.Int32(0)
                        if g < nFullG:
                            isfull = cutlass.Int32(1)
                        if isfull != cutlass.Int32(0):  # full body
                            for uu in cutlass.range_constexpr(PFD, U):
                                C.ld_g_f32x4(
                                    atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD]
                                )
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                for q in cutlass.range_constexpr(4):
                                    M = M | (
                                        cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                    )
                        else:  # partial body
                            for uu in cutlass.range_constexpr(PFD, U):
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= n4:
                                    ic = lim4  # clamp in [n, npad)
                                C.ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                okq = cutlass.Int32(0)
                                if i_ < n4:
                                    okq = cutlass.Int32(1)
                                if okq != cutlass.Int32(0):  # +inf-pad escape, ok-gated
                                    for q in cutlass.range_constexpr(4):
                                        M = M | (
                                            cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                        )
                        g2 = g + cutlass.Int32(CS)
                        if g2 < nCh:
                            j0 = g2 * cutlass.Int32(STEPC) + tidx
                            infull = cutlass.Int32(0)
                            if g2 < nFullG:
                                infull = cutlass.Int32(1)
                            if infull != cutlass.Int32(0):
                                for uu in cutlass.range_constexpr(PFD):
                                    C.ld_g_f32x4(
                                        atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    j_ = j0 + cutlass.Int32(uu * BLK)
                                    jc = j_
                                    if jc >= n4:
                                        jc = lim4
                                    C.ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                        cnt = cutlass.Int32(C.popc(M))
                        inc = C.warp_incl_scan_add(cnt, lane)
                        pos = wbase + (
                            inc - cnt
                        )  # CLUS FAST: per-warp private segment, program order
                        wtot_ = cute.arch.shuffle_sync(inc, cutlass.Int32(31))
                        wbase = wbase + wtot_
                        if wbase > wend:
                            wovf = cutlass.Int32(1)
                        if pos + cnt > wend:
                            pos = SCAP  # trash slot
                        if M != cutlass.Int32(0):
                            bp = C.ffs_m1(M)
                            M = M & (M - cutlass.Int32(1))
                            idx = (
                                (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                << cutlass.Int32(2)
                            ) + (bp & cutlass.Int32(3))
                            xv = C.ldg_f32(x_addr, idx)
                            while M != cutlass.Int32(0):
                                bp2 = C.ffs_m1(M)
                                M = M & (M - cutlass.Int32(1))
                                idx2 = (
                                    (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                    << cutlass.Int32(2)
                                ) + (bp2 & cutlass.Int32(3))
                                xv2 = C.ldg_f32(x_addr, idx2)
                                pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                                idx = idx2
                                xv = xv2
                            pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                        g = g + cutlass.Int32(CS)
                    if wid == cutlass.Int32(0):
                        ti = cutlass.Int32(0)
                        tx = cutlass.Float32(_NEG_INF__clus)
                        tsel = cutlass.Int32(0)
                        nt = cutlass.Int32(0)
                        post = cutlass.Int32(0)
                        while ti < tailn:
                            tx = cutlass.Float32(_NEG_INF__clus)
                            tsel = cutlass.Int32(0)
                            if ti + lane < tailn:
                                tx = C.ldg_f32(x_addr, tail0 + ti + lane)
                                if tx >= TF:
                                    tsel = cutlass.Int32(1)
                            nt = C.ballot(tsel == cutlass.Int32(1))
                            if tsel == cutlass.Int32(1):
                                post = wbase + cutlass.Int32(
                                    C.popc(nt & cutlass.Int32(cute.arch.lanemask_lt()))
                                )
                                if post >= wend:
                                    post = SCAP
                                post = self._emitk(
                                    tx, tail0 + ti + lane, post, TF, SC, SCAP, s_hist, s_cbuf2
                                )
                            wbase = wbase + cutlass.Int32(C.popc(nt))
                            if wbase > wend:
                                wovf = cutlass.Int32(1)
                            ti = ti + cutlass.Int32(32)
                    if lane == cutlass.Int32(0):
                        s_wcnt[wid] = wbase - wid * wcap_
                        C.atomic_add_cta(s_scal.iterator + 0, wbase - wid * wcap_)
                        if wovf != cutlass.Int32(0):
                            C.atomic_add_cta(s_scal.iterator + 3, cutlass.Int32(1))
                else:
                    while g < nCh:
                        i0 = g * cutlass.Int32(STEPC) + tidx
                        M = cutlass.Int32(0)
                        isfull = cutlass.Int32(0)
                        if g < nFullG:
                            isfull = cutlass.Int32(1)
                        if isfull != cutlass.Int32(0):  # full body
                            for uu in cutlass.range_constexpr(PFD, U):
                                C.ld_g_f32x4(
                                    atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD]
                                )
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                for q in cutlass.range_constexpr(4):
                                    M = M | (
                                        cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                    )
                        else:  # partial body
                            for uu in cutlass.range_constexpr(PFD, U):
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= n4:
                                    ic = lim4  # clamp in [n, npad)
                                C.ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                            for uu in cutlass.range_constexpr(U):
                                if cutlass.const_expr(uu < PFD):
                                    vv = pf[uu]
                                else:
                                    vv = fr[uu - PFD]
                                i_ = i0 + cutlass.Int32(uu * BLK)
                                okq = cutlass.Int32(0)
                                if i_ < n4:
                                    okq = cutlass.Int32(1)
                                if okq != cutlass.Int32(0):  # +inf-pad escape, ok-gated
                                    for q in cutlass.range_constexpr(4):
                                        M = M | (
                                            cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                        )
                        # ROLL THE PREFETCH FORWARD: next OWNED chunk, issued
                        # before the reservation and the survivor walk.
                        g2 = g + cutlass.Int32(CS)
                        if g2 < nCh:
                            j0 = g2 * cutlass.Int32(STEPC) + tidx
                            infull = cutlass.Int32(0)
                            if g2 < nFullG:
                                infull = cutlass.Int32(1)
                            if infull != cutlass.Int32(0):
                                for uu in cutlass.range_constexpr(PFD):
                                    C.ld_g_f32x4(
                                        atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu]
                                    )
                            else:
                                for uu in cutlass.range_constexpr(PFD):
                                    j_ = j0 + cutlass.Int32(uu * BLK)
                                    jc = j_
                                    if jc >= n4:
                                        jc = lim4
                                    C.ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                        # warp-aggregated slot reservation
                        cnt = cutlass.Int32(C.popc(M))
                        inc = C.warp_incl_scan_add(cnt, lane)
                        bpos = cutlass.Int32(0)
                        if lane == cutlass.Int32(31):
                            if inc != cutlass.Int32(0):
                                bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
                        pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                        # survivor bit-walk, software-pipelined ONE deep;
                        # reload X[idx] — never hold the U float4s across the walk
                        if M != cutlass.Int32(0):
                            bp = C.ffs_m1(M)
                            M = M & (M - cutlass.Int32(1))
                            idx = (
                                (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                << cutlass.Int32(2)
                            ) + (bp & cutlass.Int32(3))
                            xv = C.ldg_f32(x_addr, idx)
                            while M != cutlass.Int32(0):
                                bp2 = C.ffs_m1(M)
                                M = M & (M - cutlass.Int32(1))
                                idx2 = (
                                    (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                    << cutlass.Int32(2)
                                ) + (bp2 & cutlass.Int32(3))
                                xv2 = C.ldg_f32(x_addr, idx2)
                                pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                                idx = idx2
                                xv = xv2
                            pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                        g = g + cutlass.Int32(CS)
                    # rank-0 scalar tail: per-thread atomics, bound-check
                    i = tidx
                    while i < tailn:
                        x = C.ldg_f32(x_addr, tail0 + i)
                        if x >= TF:
                            bq = C.f2s_rz((x - TF) * SC)  # signed form
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                            post = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                            if post < SCAP:
                                s_cbuf2[post] = (
                                    cutlass.Uint64(cutlass.Uint32(tail0 + i)) << cutlass.Uint64(32)
                                ) | cutlass.Uint64(C.u32_of_f32(x))
                        i = i + cutlass.Int32(BLK)

                # ---- cluster merge ----
                C._cluster_sync_aligned()  # ==== clus.sync (merge) ====
                myn = s_scal[0]
                if cutlass.const_expr(self.det):
                    if tidx >= cutlass.Int32(32):
                        if tidx < cutlass.Int32(32 + NBS // 4):
                            tb_ = tidx - cutlass.Int32(32)
                            hoff_ = s_hist.iterator.toint() + tb_ * cutlass.Int32(16)
                            q0_ = cutlass.Int32(0)
                            q1_ = cutlass.Int32(0)
                            q2_ = cutlass.Int32(0)
                            q3_ = cutlass.Int32(0)
                            for r_ in cutlass.range_constexpr(CS - 1):
                                if cutlass.Int32(r_) < rank:
                                    h0_, h1_, h2_, h3_ = C._ld_shared_cluster_v4_u32(
                                        C._mapa_shared_cluster_addr(hoff_, cutlass.Int32(r_))
                                    )
                                    q0_ = q0_ + h0_
                                    q1_ = q1_ + h1_
                                    q2_ = q2_ + h2_
                                    q3_ = q3_ + h3_
                            s_prebin[tb_ * cutlass.Int32(4)] = q0_
                            s_prebin[tb_ * cutlass.Int32(4) + cutlass.Int32(1)] = q1_
                            s_prebin[tb_ * cutlass.Int32(4) + cutlass.Int32(2)] = q2_
                            s_prebin[tb_ * cutlass.Int32(4) + cutlass.Int32(3)] = q3_
                self._merge_scan0_local(s_hist, s_mrg, rank, k, tidx, s_res)
                cute.arch.barrier()  # ---- barrier (merge publish) ----
                tot = s_res[C.RES_TOT]
                acc = cutlass.Int32(0)
                if tot >= k:
                    acc = cutlass.Int32(1)
                if acc != cutlass.Int32(0):  # accept
                    valid = cutlass.Int32(1)
                    complete = cutlass.Int32(0)
                    if myn <= SCAP:
                        complete = cutlass.Int32(1)
                    if cutlass.const_expr(self.det):
                        if s_scal[3] != cutlass.Int32(
                            0
                        ):  # CLUS FAST: a warp segment overflowed -> deterministic re-sweep
                            complete = cutlass.Int32(0)
                    else:
                        listN = myn
                    above = s_res[C.RES_ABOVE]
                    m = s_res[C.RES_M]
                    need = k - s_res[C.RES_ABOVE]
                    B = s_res[C.RES_B]
                    running = cutlass.Int32(0)
                else:
                    if att == cutlass.Int32(2):  # ladder exhausted
                        running = cutlass.Int32(0)
                    else:
                        # rung ladder — cluster-uniform on every arm
                        tshtaken = cutlass.Int32(0)
                        if att == cutlass.Int32(0):
                            T5 = s_tsh[0]
                            if T5 > cutlass.Float32(_NEG_INF__clus):
                                if T5 < TF:
                                    T = T5
                                    tshtaken = cutlass.Int32(1)
                        if tshtaken == cutlass.Int32(0):
                            # LAZY GATHER — every rank computes identical GMIN
                            if GMIN == cutlass.Float32(C.SENT_LO):
                                if cutlass.const_expr(not self.hint_free):
                                    GMIN, GMAX = C.gather_hint(
                                        x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=1
                                    )  # 2 barriers inside
                            floorhit = cutlass.Int32(1)
                            if T > GMIN:
                                floorhit = cutlass.Int32(0)
                            if floorhit != cutlass.Int32(0):
                                running = cutlass.Int32(0)
                            else:
                                T = GMIN
                att = att + cutlass.Int32(1)

            # ============ classification ============
            whole = cutlass.Int32(0)
            if valid != cutlass.Int32(0):
                if need >= m:
                    whole = cutlass.Int32(1)
            if cutlass.const_expr(self.det):
                if whole != cutlass.Int32(0):
                    pass
            else:
                lim1 = above
                if whole != cutlass.Int32(0):
                    lim1 = above + m
            degen = cutlass.Int32(0)
            if valid == cutlass.Int32(0):
                degen = cutlass.Int32(1)
            if m > CMP:
                degen = cutlass.Int32(1)
            mc = cutlass.Int32(0)
            if degen == cutlass.Int32(0):
                mc = m
            # crossing candidates land in RANK 0's ck64c via DSMEM
            rk64 = C._mapa_shared_cluster_addr(ck_addr, cutlass.Int32(0))

            if degen == cutlass.Int32(0):
                if complete != cutlass.Int32(0):
                    # ---- P5 emit from staged cbuf ----
                    if cutlass.const_expr(self.det):
                        bacc_ = B - whole
                        bcrs_ = B + whole * cutlass.Int32(-1 - B)
                        pv_ = cutlass.Int32(0)
                        if tidx < cutlass.Int32(NBS):
                            pv_ = s_prebin[tidx]
                        pa_ = cutlass.Int32(
                            C.warp_add_i32(pv_ * cutlass.Int32(tidx > bacc_))
                        )  # per-warp partial of A_r (warps 0..NBS/32-1)
                        pc_ = cutlass.Int32(C.warp_add_i32(pv_ * cutlass.Int32(tidx == bcrs_)))
                        if lane == cutlass.Int32(0):
                            s_wcnt[cutlass.Int32(3 * NW) + wid] = pa_
                            s_wcnt[cutlass.Int32(4 * NW) + wid] = pc_
                        ra_ = cutlass.Int32(0)
                        rc_ = cutlass.Int32(0)
                        lt_ = cutlass.Int32(cute.arch.lanemask_lt())
                        c1_ = cutlass.Int32(0)
                        c2_ = cutlass.Int32(0)
                        p1b = cutlass.Int32(0)
                        p2b = cutlass.Int32(0)
                        wcnt_ = s_wcnt[wid]
                        wlast_ = cutlass.max(wcnt_ - cutlass.Int32(1), cutlass.Int32(0))
                        ne_ = (wcnt_ + cutlass.Int32(31)) >> cutlass.Int32(5)  # entries per lane
                        seg_a = _smem_addr_reg(s_cbuf2.iterator.toint()) + (
                            wid * wcap_
                        ) * cutlass.Int32(8)
                        i_ = cutlass.Int32(0)
                        jj_ = cutlass.Int32(0)
                        lo_ = cutlass.Int32(0)
                        hi_ = cutlass.Int32(0)
                        ix_ = cutlass.Int32(0)
                        bq_ = cutlass.Int32(0)
                        e1_ = cutlass.Int32(0)
                        e2_ = cutlass.Int32(0)
                        t1_ = cutlass.Int32(0)
                        t2_ = cutlass.Int32(0)
                        xv_ = cutlass.Float32(0.0)
                        while i_ < ne_:  # pass 1: per-lane class counts
                            jj_ = lane * ne_ + i_
                            lo_, hi_ = _lds_v2_u32(
                                seg_a + (cutlass.min(jj_, wlast_) << cutlass.Int32(3))
                            )
                            bq_ = cutlass.Int32(cutlass.Uint32(hi_) >> cutlass.Uint32(24))
                            ix_ = hi_ & cutlass.Int32(0xFFFFFF)
                            if jj_ >= wcnt_:
                                bq_ = cutlass.Int32(-1)
                            c1_ = c1_ + cutlass.Int32(bq_ > bacc_)
                            c2_ = c2_ + cutlass.Int32(bq_ == bcrs_)
                            i_ = i_ + cutlass.Int32(1)
                        e1_, e2_ = C.warp_incl_scan_add2(c1_, c2_, lane)
                        t1_ = cute.arch.shuffle_sync(e1_, cutlass.Int32(31))
                        t2_ = cute.arch.shuffle_sync(e2_, cutlass.Int32(31))
                        e1_ = e1_ - c1_
                        e2_ = e2_ - c2_
                        c1_ = t1_
                        c2_ = t2_
                        if lane == cutlass.Int32(0):
                            s_wcnt[cutlass.Int32(NW) + wid] = c1_
                            s_wcnt[cutlass.Int32(2 * NW) + wid] = c2_
                        cute.arch.barrier()  # ---- barrier (CLUS FAST class counts) ----
                        v1_ = cutlass.Int32(0)
                        v2_ = cutlass.Int32(0)
                        if lane < wid:  # exclusive warp prefix through one redux per class
                            v1_ = s_wcnt[cutlass.Int32(NW) + lane]
                            v2_ = s_wcnt[cutlass.Int32(2 * NW) + lane]
                        if lane < cutlass.Int32(
                            NW
                        ):  # rank bases: sum of the per-warp A_r / C_r partials
                            ra_ = s_wcnt[cutlass.Int32(3 * NW) + lane]
                            rc_ = s_wcnt[cutlass.Int32(4 * NW) + lane]
                        ra_ = cutlass.Int32(C.warp_add_i32(ra_))
                        rc_ = cutlass.Int32(C.warp_add_i32(rc_))
                        ra_ = ra_ + whole * rc_
                        p1b = ra_ + cutlass.Int32(C.warp_add_i32(v1_))
                        p2b = rc_ + cutlass.Int32(C.warp_add_i32(v2_))
                        p1b = p1b + e1_
                        p2b = p2b + e2_
                        i_ = cutlass.Int32(0)
                        while i_ < ne_:  # pass 2: emit in slice order
                            jj_ = lane * ne_ + i_
                            lo_, hi_ = _lds_v2_u32(
                                seg_a + (cutlass.min(jj_, wlast_) << cutlass.Int32(3))
                            )
                            bq_ = cutlass.Int32(cutlass.Uint32(hi_) >> cutlass.Uint32(24))
                            ix_ = hi_ & cutlass.Int32(0xFFFFFF)
                            if jj_ >= wcnt_:
                                bq_ = cutlass.Int32(-1)
                            if bq_ > bacc_:
                                out_row[p1b] = ix_
                                p1b = p1b + cutlass.Int32(1)
                            if bq_ == bcrs_:
                                if p2b < CMP:
                                    xv_ = C.f32_of_i32(lo_)
                                    C._st_shared_cluster_u64(
                                        rk64 + p2b * cutlass.Int32(8),
                                        (cutlass.Uint64(C.fkey(xv_)) << cutlass.Uint64(32))
                                        | cutlass.Uint64(cutlass.Uint32(ix_)),
                                    )
                                p2b = p2b + cutlass.Int32(1)
                            i_ = i_ + cutlass.Int32(1)
                    else:
                        i = tidx
                        while i < listN:
                            pk64 = s_cbuf2[i]
                            vx = cutlass.Int32(cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF)))
                            idv = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                            xv = C.f32_of_i32(vx)
                            self._p5_emit(
                                xv, idv, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                            )
                            i = i + cutlass.Int32(BLK)
                else:
                    if cutlass.const_expr(self.det):
                        bacc_ = B - whole
                        bcrs_ = B + whole * cutlass.Int32(-1 - B)
                        pv_ = cutlass.Int32(0)
                        if tidx < cutlass.Int32(NBS):
                            pv_ = s_prebin[tidx]
                        pa_ = cutlass.Int32(
                            C.warp_add_i32(pv_ * cutlass.Int32(tidx > bacc_))
                        )  # per-warp partial of A_r (warps 0..NBS/32-1)
                        pc_ = cutlass.Int32(C.warp_add_i32(pv_ * cutlass.Int32(tidx == bcrs_)))
                        if lane == cutlass.Int32(0):
                            s_wcnt[cutlass.Int32(3 * NW) + wid] = pa_
                            s_wcnt[cutlass.Int32(4 * NW) + wid] = pc_
                        ra_ = cutlass.Int32(0)
                        rc_ = cutlass.Int32(0)
                        lt_ = cutlass.Int32(cute.arch.lanemask_lt())
                        c1_ = cutlass.Int32(0)
                        c2_ = cutlass.Int32(0)
                        p1b = cutlass.Int32(0)
                        p2b = cutlass.Int32(0)
                        ib_ = cutlass.Int32(0)
                        ii_ = cutlass.Int32(0)
                        bq_ = cutlass.Int32(0)
                        ma_ = cutlass.Int32(0)
                        mc_ = cutlass.Int32(0)
                        sl_ = cutlass.Int32(0)
                        ti = cutlass.Int32(0)
                        xv_ = cutlass.Float32(0.0)
                    g = rank + cutlass.Int32(0)
                    if cutlass.const_expr(self.det):
                        while g < nCh:
                            lo2 = (g * cutlass.Int32(STEPC)) << cutlass.Int32(2)
                            e4 = (g + cutlass.Int32(1)) * cutlass.Int32(STEPC)
                            if e4 > n4:
                                e4 = n4
                            hi2 = e4 << cutlass.Int32(2)
                            ib_ = lo2 + (wid << cutlass.Int32(5))
                            while ib_ < hi2:
                                ii_ = ib_ + lane
                                bq_ = cutlass.Int32(-1)
                                xv_ = cutlass.Float32(0.0)
                                if ii_ < hi2:
                                    xv_ = C.ldg_f32(x_addr, ii_)
                                    if xv_ >= TF:
                                        bq_ = cutlass.min(
                                            C.f2s_rz((xv_ - TF) * SC), cutlass.Int32(NBS - 1)
                                        )
                                c1_ = c1_ + cutlass.Int32(C.popc(C.ballot(bq_ > bacc_)))
                                c2_ = c2_ + cutlass.Int32(C.popc(C.ballot(bq_ == bcrs_)))
                                ib_ = ib_ + cutlass.Int32(BLK)
                            g = g + cutlass.Int32(CS)
                        if wid == cutlass.Int32(0):
                            ti = cutlass.Int32(0)
                            while ti < tailn:
                                ii_ = tail0 + ti + lane
                                bq_ = cutlass.Int32(-1)
                                xv_ = cutlass.Float32(0.0)
                                if ti + lane < tailn:
                                    xv_ = C.ldg_f32(x_addr, ii_)
                                    if xv_ >= TF:
                                        bq_ = cutlass.min(
                                            C.f2s_rz((xv_ - TF) * SC), cutlass.Int32(NBS - 1)
                                        )
                                c1_ = c1_ + cutlass.Int32(C.popc(C.ballot(bq_ > bacc_)))
                                c2_ = c2_ + cutlass.Int32(C.popc(C.ballot(bq_ == bcrs_)))
                                ti = ti + cutlass.Int32(32)
                        if lane == cutlass.Int32(0):
                            s_wcnt[cutlass.Int32(NW) + wid] = c1_
                            s_wcnt[cutlass.Int32(2 * NW) + wid] = c2_
                        cute.arch.barrier()  # ---- barrier (CLUS FAST class counts) ----
                        v1_ = cutlass.Int32(0)
                        v2_ = cutlass.Int32(0)
                        if lane < wid:  # exclusive warp prefix through one redux per class
                            v1_ = s_wcnt[cutlass.Int32(NW) + lane]
                            v2_ = s_wcnt[cutlass.Int32(2 * NW) + lane]
                        if lane < cutlass.Int32(
                            NW
                        ):  # rank bases: sum of the per-warp A_r / C_r partials
                            ra_ = s_wcnt[cutlass.Int32(3 * NW) + lane]
                            rc_ = s_wcnt[cutlass.Int32(4 * NW) + lane]
                        ra_ = cutlass.Int32(C.warp_add_i32(ra_))
                        rc_ = cutlass.Int32(C.warp_add_i32(rc_))
                        ra_ = ra_ + whole * rc_
                        p1b = ra_ + cutlass.Int32(C.warp_add_i32(v1_))
                        p2b = rc_ + cutlass.Int32(C.warp_add_i32(v2_))
                        g = rank + cutlass.Int32(0)
                        while g < nCh:
                            lo2 = (g * cutlass.Int32(STEPC)) << cutlass.Int32(2)
                            e4 = (g + cutlass.Int32(1)) * cutlass.Int32(STEPC)
                            if e4 > n4:
                                e4 = n4
                            hi2 = e4 << cutlass.Int32(2)
                            ib_ = lo2 + (wid << cutlass.Int32(5))
                            while ib_ < hi2:
                                ii_ = ib_ + lane
                                bq_ = cutlass.Int32(-1)
                                xv_ = cutlass.Float32(0.0)
                                if ii_ < hi2:
                                    xv_ = C.ldg_f32(x_addr, ii_)
                                    if xv_ >= TF:
                                        bq_ = cutlass.min(
                                            C.f2s_rz((xv_ - TF) * SC), cutlass.Int32(NBS - 1)
                                        )
                                ma_ = C.ballot(bq_ > bacc_)
                                mc_ = C.ballot(bq_ == bcrs_)
                                if bq_ > bacc_:
                                    out_row[p1b + cutlass.Int32(C.popc(ma_ & lt_))] = ii_
                                if bq_ == bcrs_:
                                    sl_ = p2b + cutlass.Int32(C.popc(mc_ & lt_))
                                    if sl_ < CMP:
                                        C._st_shared_cluster_u64(
                                            rk64 + sl_ * cutlass.Int32(8),
                                            (cutlass.Uint64(C.fkey(xv_)) << cutlass.Uint64(32))
                                            | cutlass.Uint64(cutlass.Uint32(ii_)),
                                        )
                                p1b = p1b + cutlass.Int32(C.popc(ma_))
                                p2b = p2b + cutlass.Int32(C.popc(mc_))
                                ib_ = ib_ + cutlass.Int32(BLK)
                            g = g + cutlass.Int32(CS)
                        if wid == cutlass.Int32(0):
                            ti = cutlass.Int32(0)
                            while ti < tailn:
                                ii_ = tail0 + ti + lane
                                bq_ = cutlass.Int32(-1)
                                xv_ = cutlass.Float32(0.0)
                                if ti + lane < tailn:
                                    xv_ = C.ldg_f32(x_addr, ii_)
                                    if xv_ >= TF:
                                        bq_ = cutlass.min(
                                            C.f2s_rz((xv_ - TF) * SC), cutlass.Int32(NBS - 1)
                                        )
                                ma_ = C.ballot(bq_ > bacc_)
                                mc_ = C.ballot(bq_ == bcrs_)
                                if bq_ > bacc_:
                                    out_row[p1b + cutlass.Int32(C.popc(ma_ & lt_))] = ii_
                                if bq_ == bcrs_:
                                    sl_ = p2b + cutlass.Int32(C.popc(mc_ & lt_))
                                    if sl_ < CMP:
                                        C._st_shared_cluster_u64(
                                            rk64 + sl_ * cutlass.Int32(8),
                                            (cutlass.Uint64(C.fkey(xv_)) << cutlass.Uint64(32))
                                            | cutlass.Uint64(cutlass.Uint32(ii_)),
                                        )
                                p1b = p1b + cutlass.Int32(C.popc(ma_))
                                p2b = p2b + cutlass.Int32(C.popc(mc_))
                                ti = ti + cutlass.Int32(32)
                    else:
                        while g < nCh:
                            lo2 = (g * cutlass.Int32(STEPC)) << cutlass.Int32(2)
                            e4 = (g + cutlass.Int32(1)) * cutlass.Int32(STEPC)
                            if e4 > n4:
                                e4 = n4
                            hi2 = e4 << cutlass.Int32(2)
                            i = lo2 + tidx
                            while i < hi2:
                                x = C.ldg_f32(x_addr, i)
                                if x >= TF:
                                    self._p5_emit(
                                        x,
                                        i,
                                        TF,
                                        SC,
                                        B,
                                        above,
                                        lim1,
                                        whole,
                                        CMP,
                                        s_mrg,
                                        out_row,
                                        rk64,
                                    )
                                i = i + cutlass.Int32(BLK)
                            g = g + cutlass.Int32(CS)
                        t2 = tidx
                        while t2 < tailn:
                            ii = tail0 + t2
                            x = C.ldg_f32(x_addr, ii)
                            if x >= TF:
                                self._p5_emit(
                                    x, ii, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                                )
                            t2 = t2 + cutlass.Int32(BLK)

            # ============ EXIT RENDEZVOUS ============
            # all DSMEM traffic retired; the ONLY exit rendezvous. rank!=0
            # falls through to the kernel end (post-barrier asymmetric exit);
            # NO later cluster barrier.
            C._cluster_sync_aligned()  # ==== clus.sync (exit) ====

            if rank == cutlass.Int32(0):
                if degen == cutlass.Int32(0):
                    if whole == cutlass.Int32(0):
                        # ---- P6 rank-0 refine ----
                        if mc <= cutlass.Int32(QUADC_CLUS__clus):  # O(mc^2)
                            mc2 = mc & cutlass.Int32(~1)
                            i = tidx
                            while i < mc:
                                # re-assert Uint64 at every unsigned compare
                                # in/after dynamic loops.
                                u64v = s_ck64[i]
                                if cutlass.const_expr(self.det):
                                    u64f = (cutlass.Uint64(u64v) | cutlass.Uint64(0xFFFFFFFF)) - (
                                        cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF)
                                    )
                                r_ = cutlass.Int32(0)
                                jq = cutlass.Int32(0)
                                while jq < mc2:  # ulonglong2 16B reads
                                    vlo, vhi = C._lds_v2_u64(ck_addr + jq * cutlass.Int32(8))
                                    if cutlass.const_expr(self.det):
                                        vlf = (cutlass.Uint64(vlo) | cutlass.Uint64(0xFFFFFFFF)) - (
                                            cutlass.Uint64(vlo) & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                        vhf = (cutlass.Uint64(vhi) | cutlass.Uint64(0xFFFFFFFF)) - (
                                            cutlass.Uint64(vhi) & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                        r_ = (
                                            r_
                                            + cutlass.Int32(
                                                cutlass.Uint64(vlf) > cutlass.Uint64(u64f)
                                            )
                                            + cutlass.Int32(
                                                cutlass.Uint64(vhf) > cutlass.Uint64(u64f)
                                            )
                                        )
                                    else:
                                        r_ = (
                                            r_
                                            + cutlass.Int32(vlo > cutlass.Uint64(u64v))
                                            + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                        )
                                    jq = jq + cutlass.Int32(2)
                                if mc2 < mc:  # odd tail
                                    if cutlass.const_expr(self.det):
                                        vtl = cutlass.Uint64(s_ck64[mc2])
                                        vtf = (cutlass.Uint64(vtl) | cutlass.Uint64(0xFFFFFFFF)) - (
                                            cutlass.Uint64(vtl) & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                        r_ = r_ + cutlass.Int32(
                                            cutlass.Uint64(vtf) > cutlass.Uint64(u64f)
                                        )
                                    else:
                                        r_ = r_ + cutlass.Int32(
                                            cutlass.Uint64(s_ck64[mc2]) > cutlass.Uint64(u64v)
                                        )
                                if r_ < need:
                                    out_row[above + r_] = cutlass.Int32(
                                        cutlass.Uint32(
                                            cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                    )
                                i = i + cutlass.Int32(BLK)
                        else:
                            if tidx == cutlass.Int32(0):
                                s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                                s_kmm[1] = cutlass.Uint32(0)
                            if tidx < cutlass.Int32(NBS):  # cleared ONCE
                                s_hist[tidx] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (narrowing init) ----
                            i = tidx
                            while i < mc:
                                kk = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                C.atomic_min_cta(s_kmm.iterator + 0, kk)
                                C.atomic_max_cta(s_kmm.iterator + 1, kk)
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (key range) ----
                            rlo = s_kmm[0]
                            rhi = s_kmm[1]
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            aboveC = cutlass.Int32(0)
                            needC = need
                            mm = mc
                            brk = cutlass.Int32(0)
                            lev = cutlass.Int32(0)
                            while brk == cutlass.Int32(0):  # <=6 levels
                                if needC == mm:
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                    aboveC = aboveC + mm
                                    needC = cutlass.Int32(0)
                                    brk = cutlass.Int32(1)
                                elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    brk = cutlass.Int32(1)
                                elif lev >= cutlass.Int32(6):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    brk = cutlass.Int32(1)
                                else:
                                    d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                    b2_ = cutlass.Int32(32) - C.clz_i32(
                                        cutlass.Int32(d2 | cutlass.Uint32(1))
                                    )
                                    sh2 = b2_ - cutlass.Int32(self.lb)
                                    if sh2 < cutlass.Int32(0):
                                        sh2 = cutlass.Int32(0)
                                    sh2u = cutlass.Uint32(sh2)
                                    i = tidx
                                    while i < mc:  # re-bin
                                        uq = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                        if uq >= cutlass.Uint32(rlo):
                                            if uq <= cutlass.Uint32(rhi):
                                                du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                                if du > cutlass.Uint32(NBS - 1):
                                                    du = cutlass.Uint32(NBS - 1)
                                                C.atomic_add_cta(
                                                    s_hist.iterator + cutlass.Int32(du),
                                                    cutlass.Int32(1),
                                                )
                                        i = i + cutlass.Int32(BLK)
                                    cute.arch.barrier()  # ---- barrier (level hist) ----
                                    C.scan_cross0(
                                        s_hist,
                                        needC,
                                        tidx,
                                        s_res,
                                        cutlass.Int32(0),
                                        cutlass.Int32(0),
                                        s_hist,
                                        nb=NBS,
                                        zero=True,
                                    )
                                    cute.arch.barrier()  # ---- barrier (level scan) ----
                                    aboveC = aboveC + s_res[C.RES_ABOVE]
                                    needC = needC - s_res[C.RES_ABOVE]
                                    mm = s_res[C.RES_M]
                                    sB = s_res[C.RES_B]
                                    nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                    if sB != cutlass.Int32(NBS - 1):
                                        rhi = nlo + (
                                            (cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1)
                                        )
                                    rlo = nlo
                                    lev = lev + cutlass.Int32(1)
                            if tidx == cutlass.Int32(0):
                                s_scal[1] = cutlass.Int32(0)
                                s_scal[2] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (emit counters) ----
                            it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                            if cutlass.const_expr(self.det):
                                det_lo = cutlass.Uint32(0xFFFFFFFF) - (
                                    cutlass.Uint32(n) - cutlass.Uint32(1)
                                )  # ~idx >= 2^32 - n
                                det_hi = cutlass.Uint32(0xFFFFFFFF)
                                det_need = needC
                                det_thr = cutlass.Int64(-1)
                                det_sh = cutlass.Uint32(0)
                                det_ph = cutlass.Int32(1)
                                if needC > cutlass.Int32(0):
                                    if mm > needC:
                                        det_ph = cutlass.Int32(0)
                                while det_ph <= cutlass.Int32(1):
                                    if det_ph == cutlass.Int32(0):
                                        det_d = cutlass.Uint32(det_hi) - cutlass.Uint32(det_lo)
                                        det_bw = cutlass.Int32(32) - C.clz_i32(
                                            cutlass.Int32(det_d | cutlass.Uint32(1))
                                        )
                                        det_s = det_bw - cutlass.Int32(self.lb)
                                        if det_s < cutlass.Int32(0):
                                            det_s = cutlass.Int32(0)
                                        det_sh = cutlass.Uint32(det_s)
                                    dc1 = cutlass.Int32(0)  # DET CLUS: class counts, pass 1
                                    dc2 = cutlass.Int32(0)
                                    it = cutlass.Int32(0)
                                    while it < it2:  # ballot emit
                                        i = it * cutlass.Int32(BLK) + tidx
                                        p1 = cutlass.Int32(0)
                                        p2 = cutlass.Int32(0)
                                        idv = cutlass.Int32(0)
                                        if i < mc:
                                            w64 = s_ck64[i]
                                            iu = cutlass.Int64(
                                                cutlass.Uint32(w64 >> cutlass.Uint64(32))
                                            )
                                            idv = cutlass.Int32(
                                                cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF))
                                            )
                                            if iu > ethr:
                                                p1 = cutlass.Int32(1)
                                            if iu == ethr:
                                                det_u = cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(
                                                    idv
                                                )
                                                if det_ph == cutlass.Int32(0):
                                                    if cutlass.Uint32(det_u) >= cutlass.Uint32(
                                                        det_lo
                                                    ):
                                                        if cutlass.Uint32(det_u) <= cutlass.Uint32(
                                                            det_hi
                                                        ):
                                                            det_b = (
                                                                cutlass.Uint32(det_u)
                                                                - cutlass.Uint32(det_lo)
                                                            ) >> cutlass.Uint32(det_sh)
                                                            if det_b > cutlass.Uint32(NBS - 1):
                                                                det_b = cutlass.Uint32(NBS - 1)
                                                            C.atomic_add_cta(
                                                                s_hist.iterator
                                                                + cutlass.Int32(det_b),
                                                                cutlass.Int32(1),
                                                            )
                                                else:
                                                    if (
                                                        cutlass.Int64(cutlass.Uint32(det_u))
                                                        > det_thr
                                                    ):
                                                        p2 = cutlass.Int32(1)
                                        if det_ph == cutlass.Int32(1):
                                            dc1 = dc1 + p1
                                            dc2 = dc2 + p2
                                        it = it + cutlass.Int32(1)
                                    if det_ph == cutlass.Int32(1):
                                        s1_, s2_ = C.warp_incl_scan_add2(dc1, dc2, lane)
                                        if lane == cutlass.Int32(31):
                                            s_ws[wid] = s1_
                                            s_wmn[wid] = cutlass.Uint32(s2_)
                                        cute.arch.barrier()  # DET CLUS: warp totals published
                                        v1_ = cutlass.Int32(0)
                                        v2_ = cutlass.Int32(0)
                                        if lane < cutlass.Int32(NW):
                                            v1_ = s_ws[lane]
                                            v2_ = cutlass.Int32(s_wmn[lane])
                                        i1_, i2_ = C.warp_incl_scan_add2(v1_, v2_, lane)
                                        p1e = (
                                            cute.arch.shuffle_sync(i1_, wid)
                                            - cute.arch.shuffle_sync(v1_, wid)
                                            + (s1_ - dc1)
                                        )
                                        p2e = (
                                            cute.arch.shuffle_sync(i2_, wid)
                                            - cute.arch.shuffle_sync(v2_, wid)
                                            + (s2_ - dc2)
                                        )
                                        it = cutlass.Int32(0)
                                        while it < it2:
                                            i = it * cutlass.Int32(BLK) + tidx
                                            p1 = cutlass.Int32(0)
                                            p2 = cutlass.Int32(0)
                                            idv = cutlass.Int32(0)
                                            if i < mc:
                                                w64 = s_ck64[i]
                                                iu = cutlass.Int64(
                                                    cutlass.Uint32(w64 >> cutlass.Uint64(32))
                                                )
                                                idv = cutlass.Int32(
                                                    cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF))
                                                )
                                                if iu > ethr:
                                                    p1 = cutlass.Int32(1)
                                                if iu == ethr:
                                                    det_u = cutlass.Uint32(
                                                        0xFFFFFFFF
                                                    ) - cutlass.Uint32(idv)
                                                    if det_ph == cutlass.Int32(0):
                                                        if cutlass.Uint32(det_u) >= cutlass.Uint32(
                                                            det_lo
                                                        ):
                                                            if cutlass.Uint32(
                                                                det_u
                                                            ) <= cutlass.Uint32(det_hi):
                                                                det_b = (
                                                                    cutlass.Uint32(det_u)
                                                                    - cutlass.Uint32(det_lo)
                                                                ) >> cutlass.Uint32(det_sh)
                                                                if det_b > cutlass.Uint32(NBS - 1):
                                                                    det_b = cutlass.Uint32(NBS - 1)
                                                                C.atomic_add_cta(
                                                                    s_hist.iterator
                                                                    + cutlass.Int32(det_b),
                                                                    cutlass.Int32(1),
                                                                )
                                                    else:
                                                        if (
                                                            cutlass.Int64(cutlass.Uint32(det_u))
                                                            > det_thr
                                                        ):
                                                            p2 = cutlass.Int32(1)
                                            if p1 != cutlass.Int32(0):
                                                if p1e < aboveC:
                                                    out_row[above + p1e] = idv
                                                p1e = p1e + cutlass.Int32(1)
                                            if p2 != cutlass.Int32(0):
                                                if p2e < needC:
                                                    out_row[above + aboveC + p2e] = idv
                                                p2e = p2e + cutlass.Int32(1)
                                            it = it + cutlass.Int32(1)
                                    if det_ph == cutlass.Int32(0):
                                        cute.arch.barrier()  # ---- barrier (det level hist) ----
                                        C.scan_cross0(
                                            s_hist,
                                            det_need,
                                            tidx,
                                            s_res,
                                            cutlass.Int32(0),
                                            cutlass.Int32(0),
                                            s_hist,
                                            nb=NBS,
                                            zero=True,
                                        )
                                        cute.arch.barrier()  # ---- barrier (det level scan) ----
                                        det_need = det_need - s_res[C.RES_ABOVE]
                                        det_m = s_res[C.RES_M]
                                        det_bin = s_res[C.RES_B]
                                        det_nlo = cutlass.Uint32(det_lo) + (
                                            cutlass.Uint32(det_bin) << cutlass.Uint32(det_sh)
                                        )
                                        if det_bin != cutlass.Int32(NBS - 1):
                                            det_top = det_nlo + (
                                                (cutlass.Uint32(1) << cutlass.Uint32(det_sh))
                                                - cutlass.Uint32(1)
                                            )
                                            # clamped top edge: the ~index range starts at 2^32 - n,
                                            # so the top bin can wrap past 2^32 - 1
                                            if cutlass.Uint32(det_top) >= cutlass.Uint32(det_nlo):
                                                if cutlass.Uint32(det_top) < cutlass.Uint32(det_hi):
                                                    det_hi = det_top
                                        det_lo = det_nlo
                                        if det_need == det_m:
                                            det_thr = cutlass.Int64(
                                                cutlass.Uint32(det_lo)
                                            ) - cutlass.Int64(1)
                                            det_ph = cutlass.Int32(1)
                                        elif cutlass.Uint32(det_lo) >= cutlass.Uint32(det_hi):
                                            det_thr = cutlass.Int64(
                                                cutlass.Uint32(det_lo)
                                            ) - cutlass.Int64(1)
                                            det_ph = cutlass.Int32(1)
                                    else:
                                        det_ph = cutlass.Int32(2)
                            else:
                                it = cutlass.Int32(0)
                                while it < it2:  # ballot emit
                                    i = it * cutlass.Int32(BLK) + tidx
                                    p1 = cutlass.Int32(0)
                                    p2 = cutlass.Int32(0)
                                    idv = cutlass.Int32(0)
                                    if i < mc:
                                        w64 = s_ck64[i]
                                        iu = cutlass.Int64(
                                            cutlass.Uint32(w64 >> cutlass.Uint64(32))
                                        )
                                        idv = cutlass.Int32(
                                            cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF))
                                        )
                                        if iu > ethr:
                                            p1 = cutlass.Int32(1)
                                        if iu == ethr:
                                            p2 = cutlass.Int32(1)
                                    self._ballot_pair_emit(
                                        p1,
                                        p2,
                                        idv,
                                        above,
                                        aboveC,
                                        above + aboveC,
                                        needC,
                                        out_row,
                                        s_scal,
                                        lane,
                                    )
                                    it = it + cutlass.Int32(1)
                else:
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = n
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):  # <=8 levels
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        elif lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - C.clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1))
                            )
                            sh2 = b2_ - cutlass.Int32(self.lb)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            if tidx < cutlass.Int32(NBS):  # per-level clear
                                s_hist[tidx] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (level clear) ----
                            i = tidx
                            while i < n:  # whole row
                                uq = C.fkey(C.ldg_f32(x_addr, i))
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        C.atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (level hist) ----
                            # block-parallel scan (ONE internal barrier; only
                            # use of ws in this kernel)
                            C.scan_cross(
                                s_hist,
                                s_ws,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                blk=BLK,
                                nb=NBS,
                                two=False,
                            )
                            cute.arch.barrier()  # ---- barrier (level scan) ----
                            above2 = above2 + s_res[C.RES_ABOVE]
                            need2 = need2 - s_res[C.RES_ABOVE]
                            m2 = s_res[C.RES_M]
                            sB = s_res[C.RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (emit counters) ----
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    if cutlass.const_expr(self.det):
                        det_lo = cutlass.Uint32(0xFFFFFFFF) - (
                            cutlass.Uint32(n) - cutlass.Uint32(1)
                        )  # ~idx >= 2^32 - n
                        det_hi = cutlass.Uint32(0xFFFFFFFF)
                        det_need = need2
                        det_thr = cutlass.Int64(-1)
                        det_sh = cutlass.Uint32(0)
                        det_ph = cutlass.Int32(1)
                        if need2 > cutlass.Int32(0):
                            if m2 > need2:
                                det_ph = cutlass.Int32(0)
                        if det_ph == cutlass.Int32(0):
                            if m2 == n:  # every valid position ties: take the first L positions
                                det_thr = cutlass.Int64(
                                    cutlass.Uint32(0xFFFFFFFF)
                                    - cutlass.Uint32(cutlass.Int32(0) + need2 - cutlass.Int32(1))
                                ) - cutlass.Int64(1)
                                det_ph = cutlass.Int32(1)
                        while det_ph <= cutlass.Int32(1):
                            if det_ph == cutlass.Int32(0):
                                det_d = cutlass.Uint32(det_hi) - cutlass.Uint32(det_lo)
                                det_bw = cutlass.Int32(32) - C.clz_i32(
                                    cutlass.Int32(det_d | cutlass.Uint32(1))
                                )
                                det_s = det_bw - cutlass.Int32(self.lb)
                                if det_s < cutlass.Int32(0):
                                    det_s = cutlass.Int32(0)
                                det_sh = cutlass.Uint32(det_s)
                                if tidx < cutlass.Int32(NBS):
                                    s_hist[tidx] = cutlass.Int32(0)
                                cute.arch.barrier()  # ---- barrier (det level clear) ----
                            dc1 = cutlass.Int32(0)  # CLUS FAST: class counts, pass 1
                            dc2 = cutlass.Int32(0)
                            it = cutlass.Int32(0)
                            while it < it2:
                                i = it * cutlass.Int32(BLK) + tidx
                                p1 = cutlass.Int32(0)
                                p2 = cutlass.Int32(0)
                                if i < n:
                                    uq = C.fkey(C.ldg_f32(x_addr, i))
                                    iu = cutlass.Int64(uq)
                                    if iu > ethr:
                                        p1 = cutlass.Int32(1)
                                    if tie_m != cutlass.Int32(0):
                                        if iu == ethr:
                                            det_u = cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(i)
                                            if det_ph == cutlass.Int32(0):
                                                if cutlass.Uint32(det_u) >= cutlass.Uint32(det_lo):
                                                    if cutlass.Uint32(det_u) <= cutlass.Uint32(
                                                        det_hi
                                                    ):
                                                        det_b = (
                                                            cutlass.Uint32(det_u)
                                                            - cutlass.Uint32(det_lo)
                                                        ) >> cutlass.Uint32(det_sh)
                                                        if det_b > cutlass.Uint32(NBS - 1):
                                                            det_b = cutlass.Uint32(NBS - 1)
                                                        C.atomic_add_cta(
                                                            s_hist.iterator + cutlass.Int32(det_b),
                                                            cutlass.Int32(1),
                                                        )
                                            else:
                                                if cutlass.Int64(cutlass.Uint32(det_u)) > det_thr:
                                                    p2 = cutlass.Int32(1)
                                if det_ph == cutlass.Int32(1):
                                    dc1 = dc1 + p1
                                    dc2 = dc2 + p2
                                it = it + cutlass.Int32(1)
                            if det_ph == cutlass.Int32(1):
                                s1_, s2_ = C.warp_incl_scan_add2(dc1, dc2, lane)
                                if lane == cutlass.Int32(31):
                                    s_ws[wid] = s1_
                                    s_wmn[wid] = cutlass.Uint32(s2_)
                                cute.arch.barrier()  # DET CLUS: warp totals published
                                v1_ = cutlass.Int32(0)
                                v2_ = cutlass.Int32(0)
                                if lane < cutlass.Int32(NW):
                                    v1_ = s_ws[lane]
                                    v2_ = cutlass.Int32(s_wmn[lane])
                                i1_, i2_ = C.warp_incl_scan_add2(v1_, v2_, lane)
                                p1e = (
                                    cute.arch.shuffle_sync(i1_, wid)
                                    - cute.arch.shuffle_sync(v1_, wid)
                                    + (s1_ - dc1)
                                )
                                p2e = (
                                    cute.arch.shuffle_sync(i2_, wid)
                                    - cute.arch.shuffle_sync(v2_, wid)
                                    + (s2_ - dc2)
                                )
                                it = cutlass.Int32(0)
                                while it < it2:
                                    i = it * cutlass.Int32(BLK) + tidx
                                    p1 = cutlass.Int32(0)
                                    p2 = cutlass.Int32(0)
                                    if i < n:
                                        uq = C.fkey(C.ldg_f32(x_addr, i))
                                        iu = cutlass.Int64(uq)
                                        if iu > ethr:
                                            p1 = cutlass.Int32(1)
                                        if tie_m != cutlass.Int32(0):
                                            if iu == ethr:
                                                det_u = cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(
                                                    i
                                                )
                                                if det_ph == cutlass.Int32(0):
                                                    if cutlass.Uint32(det_u) >= cutlass.Uint32(
                                                        det_lo
                                                    ):
                                                        if cutlass.Uint32(det_u) <= cutlass.Uint32(
                                                            det_hi
                                                        ):
                                                            det_b = (
                                                                cutlass.Uint32(det_u)
                                                                - cutlass.Uint32(det_lo)
                                                            ) >> cutlass.Uint32(det_sh)
                                                            if det_b > cutlass.Uint32(NBS - 1):
                                                                det_b = cutlass.Uint32(NBS - 1)
                                                            C.atomic_add_cta(
                                                                s_hist.iterator
                                                                + cutlass.Int32(det_b),
                                                                cutlass.Int32(1),
                                                            )
                                                else:
                                                    if (
                                                        cutlass.Int64(cutlass.Uint32(det_u))
                                                        > det_thr
                                                    ):
                                                        p2 = cutlass.Int32(1)
                                    if p1 != cutlass.Int32(0):
                                        if p1e < nA:
                                            out_row[p1e] = i
                                        p1e = p1e + cutlass.Int32(1)
                                    if p2 != cutlass.Int32(0):
                                        if p2e < nT:
                                            out_row[nA + p2e] = i
                                        p2e = p2e + cutlass.Int32(1)
                                    it = it + cutlass.Int32(1)
                            if det_ph == cutlass.Int32(0):
                                cute.arch.barrier()  # ---- barrier (det level hist) ----
                                C.scan_cross(
                                    s_hist,
                                    s_ws,
                                    det_need,
                                    tidx,
                                    s_res,
                                    cutlass.Int32(0),
                                    blk=BLK,
                                    nb=NBS,
                                    two=False,
                                )
                                cute.arch.barrier()  # ---- barrier (det level scan) ----
                                det_need = det_need - s_res[C.RES_ABOVE]
                                det_m = s_res[C.RES_M]
                                det_bin = s_res[C.RES_B]
                                det_nlo = cutlass.Uint32(det_lo) + (
                                    cutlass.Uint32(det_bin) << cutlass.Uint32(det_sh)
                                )
                                if det_bin != cutlass.Int32(NBS - 1):
                                    det_top = det_nlo + (
                                        (cutlass.Uint32(1) << cutlass.Uint32(det_sh))
                                        - cutlass.Uint32(1)
                                    )
                                    # clamped top edge: the ~index range starts at 2^32 - n,
                                    # so the top bin can wrap past 2^32 - 1
                                    if cutlass.Uint32(det_top) >= cutlass.Uint32(det_nlo):
                                        if cutlass.Uint32(det_top) < cutlass.Uint32(det_hi):
                                            det_hi = det_top
                                det_lo = det_nlo
                                if det_need == det_m:
                                    det_thr = cutlass.Int64(cutlass.Uint32(det_lo)) - cutlass.Int64(
                                        1
                                    )
                                    det_ph = cutlass.Int32(1)
                                elif cutlass.Uint32(det_lo) >= cutlass.Uint32(det_hi):
                                    det_thr = cutlass.Int64(cutlass.Uint32(det_lo)) - cutlass.Int64(
                                        1
                                    )
                                    det_ph = cutlass.Int32(1)
                            else:
                                det_ph = cutlass.Int32(2)
                    else:
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            if i < n:
                                uq = C.fkey(C.ldg_f32(x_addr, i))
                                iu = cutlass.Int64(uq)
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if tie_m != cutlass.Int32(0):
                                    if iu == ethr:
                                        p2 = cutlass.Int32(1)
                            self._ballot_pair_emit(
                                p1, p2, i, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                            )
                            it = it + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # host launcher: grid dim3(CS, b) + cluster (CS,1,1);
    # min_blocks_per_mp=1 == __launch_bounds__(1024, 1) 64-register wall.
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        SCAP: cutlass.Int32,
        CMP: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        stream,
    ):
        # varlen grid rows = out.shape[0] (logits row count == out row count in
        # both modes); bigf = the route() `big` occupancy flag, a pure function
        # of (rows, CS) so it is launch-computed, not an ABI scalar.
        b = out.shape[0]
        bigf = cutlass.Int32(0)
        if b * cutlass.Int32(self.cs) <= cutlass.Int32(148):
            bigf = cutlass.Int32(1)
        self.kern(
            logits, pre_idx, kv_lens, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2, bigf
        ).launch(
            grid=(self.cs, b, 1),
            block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.minb,
        )


# ---------------------------------------------------------------------------
# compile cache + torch-facing entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE__clus = {}


def get_compiled__clus(
    tpl: tuple,
    scap: int = 8192,
    cmp_: int = 2048,
    options_extra: str = "",
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
    hint_free: bool = False,
    det: bool = False,
) -> Any:
    """Compile (or fetch) the gvr_clus variant for constexpr tuple
    tpl = (BLK, U, MINB, NBS, CS); scap/cmp are smem-extent keys (every
    reachable route has 8192/2048)."""
    key = (
        tuple(tpl),
        scap,
        cmp_,
        options_extra,
        bool(varlen),
        int(next_n),
        int(cr_shift),
        bool(hint_free),
        bool(det),
    )
    hit = _COMPILE_CACHE__clus.get(key)
    if hit is not None:
        return hit
    blk, u, minb, nbs, cs = tpl
    kern = GvrClusKernel(
        blk,
        u,
        minb,
        nbs,
        cs,
        scap=scap,
        cmp_=cmp_,
        varlen=varlen,
        next_n=next_n,
        cr_shift=cr_shift,
        hint_free=hint_free,
        det=det,
    )
    r0, c0 = cute.sym_int(), cute.sym_int()
    r1, c1 = cute.sym_int(), cute.sym_int()
    r2, c2 = cute.sym_int(), cute.sym_int()
    logits_fake = _crt.make_fake_compact_tensor(
        cutlass.Float32, (r0, c0), stride_order=(1, 0), assumed_align=16
    )
    pre_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r1, c1), stride_order=(1, 0), assumed_align=16
    )
    out_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r2, c2), stride_order=(1, 0), assumed_align=16
    )
    v0 = cute.sym_int()
    kv_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (v0,), stride_order=(0,), assumed_align=4
    )
    fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kern,
        logits_fake,
        pre_fake,
        kv_fake,
        out_fake,
        *([cutlass.Int32(0)] * 10),
        stream=fake_stream,
        options=("--enable-tvm-ffi " + options_extra).strip(),
    )
    _COMPILE_CACHE__clus[key] = compiled
    return compiled
