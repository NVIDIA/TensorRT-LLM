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

"""Hybrid-sampling clustered-register BF16 GVR decode specialization.

The sample combines 128 globally distributed values with 896 prefix values.
The 512-bin histogram and 32-bit exact refinement retain independent cluster
collectives for this sampling strategy.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .gvr_topk_decode_self_sampling_bf16 import (
    _bf16_rne_bits__reg,
    _cluster_sync_aligned,
    _dtype_of,
    _fabsf__regclus,
    _fmaf__regclus,
    _ld_shared_cluster_i32,
    _mapa_shared_cluster_addr,
    _smem_view__regclus,
    _st_shared_cluster_i32,
    _umin_u32__regclus,
    _val8__regclus,
    _val8_pk__regclus,
    _val__regclus,
    atomic_add_cta,
    atomic_max_cta,
    atomic_min_cta,
    ballot,
    clz_i32,
    f2u_rz,
    f32_of_u32,
    find_cross,
    fkey,
    g2r_atom_f32,
    invkey,
    ld_g_bf16x8,
    ld_g_bf16x8_pk,
    ld_g_f32x4,
    ld_g_i32,
    ldg_bf16,
    ldg_f32,
    popc,
    scan_cross,
    scan_cross_w,
    warp_max_u32,
    warp_min_u32,
)

SENT_LO = -3.0e38


SENT_HI = 3.0e38


RES_B = 0


RES_M = 1


RES_ABOVE = 2


@dsl_user_op
def _red_shared_cluster_min_u32(mapped_addr, val, *, loc=None, ip=None):
    """Remote CTA smem reduction min.u32, discarding the old value."""
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="red.relaxed.cluster.shared::cluster.min.u32 [$0], $1;",
        constraints="r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_shared_cluster_max_u32(mapped_addr, val, *, loc=None, ip=None):
    """Remote CTA smem reduction max.u32, discarding the old value."""
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="red.relaxed.cluster.shared::cluster.max.u32 [$0], $1;",
        constraints="r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


NB__regclus = 1024  # histogram bins; == BLKC here


LNB = 10  # log2(NB__regclus) — reg_clus narrowing shift


QUADC__regclus = 96  # O(mc^2) rank gate


CMPC = 4096  # crossing slots PER CTA (pow2)


LCMPC = 12  # log2(CMPC)


BLKC = 1024  # CTA size


STATIC_WORDS__regclus = 128  # DSL smem prelude (static-__shared__ mirror)


STATIC_BYTES__regclus = STATIC_WORDS__regclus * 4


DYN_SMEM_BYTES = (3 * NB__regclus + 2 * CMPC) * 4  # 45,056


SMEM_BYTES = STATIC_BYTES__regclus + DYN_SMEM_BYTES  # 45,568


W_HIST = STATIC_WORDS__regclus


W_MRG = STATIC_WORDS__regclus + NB__regclus


W_HOFF = STATIC_WORDS__regclus + 2 * NB__regclus


W_CK = STATIC_WORDS__regclus + 3 * NB__regclus


W_CI = STATIC_WORDS__regclus + 3 * NB__regclus + CMPC


_NEG_INF__regclus = float("-inf")


class GvrRegClusKernel:
    """gvr_reg_clus<BLK, VPT, CS>."""

    def __init__(
        self,
        blk: int,
        vpt: int,
        cs: int,
        pdl: bool = False,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        hint_free: bool = False,
        dtype=cutlass.Float32,
        oneq_enabled: bool = False,
    ) -> None:
        assert blk == BLKC, "all instantiations BLK=BLKC=1024"
        # CS=16 is a bf16-route-only widening: the merge/striping phases are
        # already CS-parametric (range_constexpr(CS) mapa loops, rank slabs
        # q2i >> LCMPC with CS*CMPC capacity) and 16-CTA clusters are within
        # the SM100 nonportable limit. fp32 route() never emits cs > 8.
        assert vpt in (1, 2, 4) and cs in (2, 4, 8, 16)
        # constexpr logits dtype (Float32 verbatim arm / BFloat16 native arm)
        self.dtype = dtype
        self.oneq_enabled = bool(oneq_enabled)
        assert dtype in (cutlass.Float32, cutlass.BFloat16)
        self.blk = blk
        self.vpt = vpt
        self.cs = cs
        self.pdl = bool(pdl)
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel): n is re-derived PER ROW in-kernel from
        # a device kv_lens tensor; the scalar n launch arg becomes the
        # envelope clamp bound. next_n / cr_shift are compile-time.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        # hint-free: P0 samples the first k row elements (coalesced) instead of the hint
        self.hint_free = bool(hint_free)
        if dtype is cutlass.Float32:
            self.S = vpt * 4
        else:
            self.S = vpt * 8  # bf16: 8 elems per 16B vector, same element count
        self.span = blk * vpt  # float4 per CTA

    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
    ):
        BLK = cutlass.const_expr(self.blk)
        VPT = cutlass.const_expr(self.vpt)
        CS = cutlass.const_expr(self.cs)
        S = cutlass.const_expr(self.S)
        NW = cutlass.const_expr(self.blk // 32)

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # knob default off

        tid, _, _ = cute.arch.thread_idx()
        rank, row, _ = cute.arch.block_idx()  # bx=rank
        lane = tid & cutlass.Int32(31)

        # ---- bf16 varlen head-latency hoist (see GvrTopkRegKernel): the P1
        # row batch is address-independent of kv_lens (the envelope arg n
        # bounds it inside the padded row) but in warp issue order it sits
        # BEHIND the kv_lens-dependent P0 bracket gather, so the whole batch
        # serializes on the kv_lens round trip. Issue it before the varlen
        # prologue; lanes at/beyond the per-row window are overwritten with
        # packed/widened -inf at P1 (register state identical to the
        # unhoisted batch; pad contents never consulted). fp32 arm:
        # constexpr-eliminated, codegen untouched.
        if cutlass.const_expr(self.dtype != cutlass.Float32 and self.varlen):
            hcx_addr = logits[row, None].iterator.toint()
            hcatom = g2r_atom_f32(128, invariant=True)
            envc8 = n >> cutlass.Int32(3)
            if envc8 > cutlass.Int32(self.cs * self.span):
                envc8 = cutlass.Int32(self.cs * self.span)
            hcbase = rank * cutlass.Int32(self.span)
            if cutlass.const_expr(VPT >= 2):
                hcfrags = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(VPT)]
                for u in cutlass.range_constexpr(VPT):
                    hci = hcbase + tid + cutlass.Int32(u * self.blk)
                    if hci < envc8:
                        ld_g_bf16x8_pk(hcatom, hcx_addr, hci, hcfrags[u])
            else:
                hcfrags = [cute.make_rmem_tensor((8,), cutlass.Float32) for _ in range(VPT)]
                for u in cutlass.range_constexpr(VPT):
                    hci = hcbase + tid + cutlass.Int32(u * self.blk)
                    if hci < envc8:
                        ld_g_bf16x8(hcatom, hcx_addr, hci, hcfrags[u])
            # P0/P3 bracket gather hoist: the sample address is a pure
            # function of envelope-only quantities (tid, k, grid rows, the
            # envelope arg n), so the gather — the ladder's critical-path
            # load — issues here instead of serializing behind kv_lens.
            # Contribution at P3 keeps the per-row `< n` guard, so rows
            # shorter than the envelope contribute a SUBSET of the original
            # sample set (never a pad value); the bracket only initializes
            # the ladder, count-crossing still proves exactness.
            hpv = cutlass.Int32(-1)
            hgv = cutlass.Float32(0.0)
            if cutlass.const_expr(self.hint_free):
                if cutlass.const_expr(self.cs >= 16):
                    hpv = tid * (n >> cutlass.Int32(10))
                else:
                    _, hgrows, _ = cute.arch.grid_dim()
                    khq = cutlass.Int32(pre_idx.shape[1])
                    if khq > cutlass.Int32(self.blk):
                        hpv = tid * (n >> cutlass.Int32(10))
                        if hgrows == cutlass.Int32(1):
                            # Retain prefix locality while sampling the full row.
                            hpv = tid
                            if tid < cutlass.Int32(128):
                                hpv = tid * (n >> cutlass.Int32(7))
                    else:
                        if tid < khq:
                            hpv = tid
                if cutlass.Uint32(hpv) < cutlass.Uint32(n):
                    hgv = ldg_bf16(hcx_addr, hpv)

        # ============ per-row varlen prologue — shared contract lives in ======
        # GvrMainKernel's prologue (per-row n from kv_lens, clamped to the
        # envelope arg; the launcher admits this family only when the envelope
        # fits its capacity window). Pure functions of `row` keep the body
        # guard and cluster barriers cluster-uniform. Short rows (n <= k):
        # rank 0 emits identity + (-1) and the body is SKIPped — a zero-work
        # pass would reach the crossing-overflow emitter and poison the output.
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
            if short != cutlass.Int32(0):
                if rank == cutlass.Int32(0):
                    # BF16 admits k > BLK; rank 0 must cover every output
                    # column, including the identity and -1 padding tail.
                    short_idx = tid
                    while short_idx < kq:
                        ov = cutlass.Int32(-1)
                        if short_idx < nv:
                            ov = short_idx
                        out[row, short_idx] = ov
                        short_idx = short_idx + cutlass.Int32(BLK)

        # ------------------------------------------------------------------
        # Predeclarations (DSL AST rule: every scalar (re)assigned under a
        # dynamic if/while must pre-exist with a stable type; constant inits
        # are dead-coded).
        # ------------------------------------------------------------------
        i = cutlass.Int32(0)
        j = cutlass.Int32(0)
        rnk = cutlass.Int32(0)
        tinc = cutlass.Int32(0)
        mc = cutlass.Int32(0)
        p = cutlass.Int32(0)
        q2i = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
        lim8q = cutlass.Int32(0)
        aboveC = cutlass.Int32(0)
        needC = cutlass.Int32(0)
        mm = cutlass.Int32(0)
        lev = cutlass.Int32(0)
        done = cutlass.Int32(0)
        b2w = cutlass.Int32(0)
        sh2 = cutlass.Int32(0)
        b_lv = cutlass.Int32(0)
        it = cutlass.Int32(0)
        it2 = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        q1f = cutlass.Int32(0)
        q2f = cutlass.Int32(0)
        n1 = cutlass.Int32(0)
        n2 = cutlass.Int32(0)
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        p1e = cutlass.Int32(0)
        p2e = cutlass.Int32(0)
        lml = cutlass.Int32(0)
        nA = cutlass.Int32(0)
        nT = cutlass.Int32(0)
        tie_m = cutlass.Int32(0)
        pv0 = cutlass.Int32(-1)
        okc = cutlass.Int32(0)
        whole = cutlass.Int32(0)
        degen = cutlass.Int32(0)
        pre_a = cutlass.Int32(0)
        tot_a = cutlass.Int32(0)
        uk = cutlass.Uint32(0)
        uq = cutlass.Uint32(0)
        vq = cutlass.Uint32(0)
        kv = cutlass.Uint32(0)
        rlo = cutlass.Uint32(0)
        rhi = cutlass.Uint32(0)
        d2 = cutlass.Uint32(0)
        unar = cutlass.Uint32(0)
        bnn = cutlass.Uint32(0)
        nlo = cutlass.Uint32(0)
        uke = cutlass.Uint32(0)
        bn = cutlass.Uint32(0)
        ethr = cutlass.Int64(0)
        tval = cutlass.Float32(_NEG_INF__regclus)
        LOQ = cutlass.Float32(0.0)
        qv = cutlass.Float32(0.0)

        if short == cutlass.Int32(0):
            npad = cutlass.Int32(logits.shape[1])  # noqa: F841
            k = cutlass.Int32(pre_idx.shape[1])
            out_row = out[row, None]
            x_addr = logits[row, None].iterator.toint()  # Int64 gmem byte base
            p_addr = pre_idx[prow, None].iterator.toint()  # request-level under varlen

            # ---- shared-memory window (map in module docstring) ----
            sptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
            sbase = sptr.toint()  # Int32 shared addr

            s_res = _smem_view__regclus(cutlass.Int32, sbase, 0, 6)
            s_cnt = _smem_view__regclus(cutlass.Int32, sbase, 6, 2)  # [0]=s_o1 [1]=s_o2
            s_kmm = _smem_view__regclus(cutlass.Uint32, sbase, 8, 2)  # [0]=s_kmin [1]=s_kmax
            s_ws = _smem_view__regclus(cutlass.Int32, sbase, 16, 32)
            s_wmn = _smem_view__regclus(cutlass.Uint32, sbase, 48, 32)
            s_wmx = _smem_view__regclus(cutlass.Uint32, sbase, 80, 32)
            s_hist = _smem_view__regclus(cutlass.Int32, sbase, W_HIST, NB__regclus)
            s_mrg = _smem_view__regclus(cutlass.Int32, sbase, W_MRG, NB__regclus)
            s_hoff = _smem_view__regclus(cutlass.Int32, sbase, W_HOFF, NB__regclus)
            s_ck = _smem_view__regclus(cutlass.Uint32, sbase, W_CK, CMPC)
            s_ci = _smem_view__regclus(cutlass.Int32, sbase, W_CI, CMPC, align=4)
            # raw byte bases for DSMEM (mapa) addressing
            hist_addr = sbase + cutlass.Int32(W_HIST * 4)
            ck_addr = sbase + cutlass.Int32(W_CK * 4)
            ci_addr = sbase + cutlass.Int32(W_CI * 4)
            FOLDQ = cutlass.const_expr(
                self.oneq_enabled
                and self.dtype != cutlass.Float32
                and NB__regclus == 512
                and self.cs >= 2
                and (QUADC__regclus == 192 or QUADC__regclus == 384)
                and self.vpt == 1
            )
            if cutlass.const_expr(FOLDQ):
                kmm_addr = sbase + cutlass.Int32(8 * 4)
                kmn_st = cutlass.Uint32(0xFFFFFFFF)
                kmx_st = cutlass.Uint32(0)
                kq_f = cutlass.Uint32(0)
                kq_t = cutlass.Uint32(0)
                kmn_w = cutlass.Uint32(0xFFFFFFFF)
                kmx_w = cutlass.Uint32(0)

            if cutlass.const_expr(self.dtype == cutlass.Float32):
                n4 = n >> cutlass.Int32(2)
                ntail = n - (n4 << cutlass.Int32(2))
                base4 = rank * cutlass.Int32(self.span)
                tix = (n4 << cutlass.Int32(2)) + tid  # CUDA `tidx`
            else:
                # bf16 slot-exact window (see GvrTopkRegKernel): cap the
                # vector window at the cluster's register capacity (cs * span
                # vectors) and hand the (<= BLK, route-enforced) element
                # overflow to the existing rank-0 scalar tail — the tail is
                # value-generic (trash-bin histogram, tval==-inf elsewhere),
                # so the bf16 route may pick (v, cs) that under-cover by up
                # to BLK elements instead of doubling v or cs for a 64-pad
                # envelope overflow.
                n4 = n >> cutlass.Int32(3)  # bf16: 8-elem 16B vectors
                if n4 > cutlass.Int32(self.cs * self.span):
                    n4 = cutlass.Int32(self.cs * self.span)
                ntail = n - (n4 << cutlass.Int32(3))
                base4 = rank * cutlass.Int32(self.span)
                tix = (n4 << cutlass.Int32(3)) + tid

            # ---- P0: redundant hint gather, EVERY CTA (k<=BLK by dispatch
            # gate). One coalesced word per thread, NO cluster barrier —
            # GMIN/GMAX identical everywhere by construction.
            if cutlass.const_expr(self.hint_free):
                if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                    _, grid_rows, _ = cute.arch.grid_dim()
                    if cutlass.const_expr(self.cs >= 16):
                        # 16-CTA envelopes (163K-262K columns): the first-K
                        # prefix covers < 1.3% of the row, and a collapsed
                        # bracket sends every row through the striped-slab
                        # narrowing (probe: pro_1024k_bs2 17.9us prefix vs the
                        # 11.9us split baseline). Whole-row 1/1024 stride
                        # restores a full-range bracket; the sample only
                        # initializes the ladder, count crossing stays exact.
                        pv0 = tid * (n >> cutlass.Int32(10))
                    else:
                        if k > cutlass.Int32(self.blk):
                            # Multi-row K=2048 captures need a whole-row bracket:
                            # their first-K prefix collapses the upper tail into
                            # one large crossing bin. The sample only initializes
                            # the ladder; count crossing still proves exactness.
                            pv0 = tid * (n >> cutlass.Int32(10))
                            if grid_rows == cutlass.Int32(1):
                                pv0 = tid
                                if tid < cutlass.Int32(128):
                                    pv0 = tid * (n >> cutlass.Int32(7))
                        else:
                            if tid < k:
                                pv0 = tid
                else:
                    if tid < k:
                        pv0 = tid
            else:
                if tid < k:
                    pv0 = ld_g_i32(p_addr, tid)

            # ---- P1: row load — predicated flat float4[VPT] batch (the CUDA
            # has NO exact-fit peel here, guard is per-load). Issue all loads
            # first, then -INFINITY-fill missed slots.
            atom128 = g2r_atom_f32(128, invariant=True)
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(VPT)]
                for u in cutlass.range_constexpr(VPT):
                    i = base4 + tid + cutlass.Int32(u * self.blk)
                    if i < n4:
                        ld_g_f32x4(atom128, x_addr, i, frags[u])
                for u in cutlass.range_constexpr(VPT):
                    i = base4 + tid + cutlass.Int32(u * self.blk)
                    if i >= n4:  # -INFINITY fill
                        for z in cutlass.range_constexpr(4):
                            frags[u][z] = cutlass.Float32(_NEG_INF__regclus)
            else:
                if cutlass.const_expr(self.varlen):
                    # Batch already in flight from the kernel-head hoist:
                    # adopt it and overwrite lanes at/beyond the per-row
                    # window (identical register state to the unhoisted
                    # predicated batches below).
                    frags = hcfrags
                    if cutlass.const_expr(VPT >= 2):
                        for u in cutlass.range_constexpr(VPT):
                            hcj = base4 + tid + cutlass.Int32(u * self.blk)
                            if hcj >= n4:  # -INFINITY fill (packed bf16x2 -inf)
                                for z in cutlass.range_constexpr(4):
                                    frags[u][z] = cutlass.Uint32(0xFF80FF80)
                    else:
                        for u in cutlass.range_constexpr(VPT):
                            hcj = base4 + tid + cutlass.Int32(u * self.blk)
                            if hcj >= n4:  # -INFINITY fill
                                for z in cutlass.range_constexpr(8):
                                    frags[u][z] = cutlass.Float32(_NEG_INF__regclus)
                elif cutlass.const_expr(VPT >= 2):
                    # packed retention: bf16x8 vector = FOUR raw u32 pairs
                    # (not eight widened f32) held live across the classify
                    # phases; unpack (shift/mask, exact) happens per use in
                    # _val8_pk__regclus. Halves the long-lived value
                    # registers — the VPT=4 batch carries 16 u32 instead of
                    # 32 f32, staying under the 64-register launch wall.
                    frags = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(VPT)]
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i < n4:
                            ld_g_bf16x8_pk(atom128, x_addr, i, frags[u])
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i >= n4:  # -INFINITY fill (packed bf16x2 -inf)
                            for z in cutlass.range_constexpr(4):
                                frags[u][z] = cutlass.Uint32(0xFF80FF80)
                else:
                    frags = [cute.make_rmem_tensor((8,), cutlass.Float32) for _ in range(VPT)]
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i < n4:
                            ld_g_bf16x8(atom128, x_addr, i, frags[u])
                    for u in cutlass.range_constexpr(VPT):
                        i = base4 + tid + cutlass.Int32(u * self.blk)
                        if i >= n4:  # -INFINITY fill
                            for z in cutlass.range_constexpr(8):
                                frags[u][z] = cutlass.Float32(_NEG_INF__regclus)
            # tail element: rank 0 only
            if rank == cutlass.Int32(0):
                if tid < ntail:
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        tval = ldg_f32(x_addr, tix)
                    else:
                        tval = ldg_bf16(x_addr, tix)

            # ---- P2: init. NB__regclus == BLK -> single-pass hist clear.
            if tid == cutlass.Int32(0):
                s_cnt[0] = cutlass.Int32(0)
                s_cnt[1] = cutlass.Int32(0)
            if cutlass.const_expr(FOLDQ):
                if rank == cutlass.Int32(0):
                    if tid == cutlass.Int32(0):
                        s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                        s_kmm[1] = cutlass.Uint32(0)
            if cutlass.const_expr(NB__regclus < self.blk):
                if tid < cutlass.Int32(NB__regclus):
                    s_hist[tid] = cutlass.Int32(0)
            else:
                for z in cutlass.range_constexpr(NB__regclus // self.blk):
                    s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

            # ---- P3: GMIN/GMAX from the hint, ONE barrier fold.
            lmin = cutlass.Uint32(0xFFFFFFFF)
            lmax = cutlass.Uint32(0)
            if cutlass.const_expr(
                self.dtype == cutlass.BFloat16 and self.varlen and self.hint_free
            ):
                # value preloaded at the kernel head (see gather hoist);
                # contribution stays per-row-guarded so short rows sample a
                # subset of the original set, never a pad column.
                if cutlass.Uint32(hpv) < cutlass.Uint32(n):
                    uk = fkey(hgv)
                    lmin = uk
                    lmax = uk
            else:
                if cutlass.Uint32(pv0) < cutlass.Uint32(n):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        uk = fkey(ldg_f32(x_addr, pv0))  # __ldg(X+pv0)
                    else:
                        uk = fkey(ldg_bf16(x_addr, pv0))
                    lmin = uk
                    lmax = uk
            lmin = warp_min_u32(lmin)
            lmax = warp_max_u32(lmax)
            if lane == cutlass.Int32(0):
                s_wmn[tid >> cutlass.Int32(5)] = lmin
                s_wmx[tid >> cutlass.Int32(5)] = lmax
            cute.arch.barrier()  # warp partials published
            a = cutlass.Uint32(0xFFFFFFFF)
            c = cutlass.Uint32(0)
            if lane < cutlass.Int32(NW):
                a = cutlass.Uint32(s_wmn[lane])
                c = cutlass.Uint32(s_wmx[lane])
            lmin = warp_min_u32(a)
            lmax = warp_max_u32(c)
            Tv = invkey(lmin)
            GMAX = invkey(lmax)

            # ---- collapse guard, NaN-safe. Reject infinite width as well:
            # otherwise SC becomes zero and can collapse +inf into a finite
            # histogram bin.
            okc = cutlass.Int32(0)
            if Tv < GMAX:
                w_ = GMAX - Tv
                if w_ > cutlass.Float32(1e-30) and w_ < cutlass.Float32(3.5e38):
                    okc = cutlass.Int32(1)
            if okc == cutlass.Int32(0):
                Tv = cutlass.Float32(SENT_LO)
                GMAX = cutlass.Float32(SENT_HI)

            # ---- bin transform constants: branchless trash bin.
            WD = (GMAX - Tv) * cutlass.Float32(1.0 / float(NB__regclus - 2))
            wsel = cutlass.Float32(1e-30)
            if WD > cutlass.Float32(0.0):
                wsel = WD
            # MUFU.RCP spelling (mirrors the reg family's site):
            # wsel >= 1e-30 finite; bucketing is SC-invariant for any SC > 0.
            SC = cute.arch.rcp_approx(wsel)
            CQ0 = cutlass.Float32(1.0) - Tv * SC
            CQ = CQ0 + cutlass.Float32(1e-6) * (_fabsf__regclus(CQ0) + cutlass.Float32(1.0))

            # ---- P4: histogram; tval add UNCONDITIONAL (trash bin swallows
            # -INFINITY via the saturating cvt).
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                for s in cutlass.range_constexpr(S):
                    qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)
                    bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                    atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
            else:
                if cutlass.const_expr(self.cs >= 16):
                    if base4 >= n4:
                        # Fully-idle rank (window entirely beyond the row, only
                        # reachable on the slack ranks of the widened envelopes):
                        # every slot carries the -inf fill, so all S*BLK adds land
                        # in the saturating trash bin. One store writes the exact
                        # same histogram (hist was just cleared, no other writer
                        # before the cluster merge) without S*BLK same-address
                        # smem atomics convoying the P5 rendezvous.
                        if tid == cutlass.Int32(0):
                            s_hist[0] = cutlass.Int32(S * self.blk)
                    else:
                        for s in cutlass.range_constexpr(S):
                            if cutlass.const_expr(VPT >= 2):
                                qv = _fmaf__regclus(_val8_pk__regclus(frags, s), SC, CQ)
                            else:
                                qv = _fmaf__regclus(_val8__regclus(frags, s), SC, CQ)
                            bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                            atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                else:
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(VPT >= 2):
                            qv = _fmaf__regclus(_val8_pk__regclus(frags, s), SC, CQ)
                        else:
                            qv = _fmaf__regclus(_val8__regclus(frags, s), SC, CQ)
                        bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                        atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                qv = _fmaf__regclus(tval, SC, CQ)
                bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
            else:
                if rank == cutlass.Int32(0):
                    if tid < ntail:
                        qv = _fmaf__regclus(tval, SC, CQ)
                        bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                        atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))

            # ---- P5: cluster merge
            _cluster_sync_aligned()  # histograms complete on every rank
            if cutlass.const_expr(NB__regclus < self.blk):
                if tid < cutlass.Int32(NB__regclus):
                    i = tid
                    hvals = []
                    for r in cutlass.range_constexpr(CS):
                        ma = _mapa_shared_cluster_addr(
                            hist_addr + (i << cutlass.Int32(2)), cutlass.Int32(r)
                        )
                        hvals.append(_ld_shared_cluster_i32(ma))
                    tot_a = cutlass.Int32(0)
                    pre_a = cutlass.Int32(0)
                    for r in cutlass.range_constexpr(CS):
                        if cutlass.Int32(r) < rank:
                            pre_a = pre_a + hvals[r]
                        tot_a = tot_a + hvals[r]
                    s_mrg[i] = tot_a
                    s_hoff[i] = pre_a
            else:
                for z in cutlass.range_constexpr(NB__regclus // self.blk):
                    i = tid + cutlass.Int32(z * self.blk)
                    # CS-unrolled remote u32 loads: batch-issue, then fold
                    # (one mapa per (i, r) exactly like the CUDA)
                    hvals = []
                    for r in cutlass.range_constexpr(CS):
                        ma = _mapa_shared_cluster_addr(
                            hist_addr + (i << cutlass.Int32(2)), cutlass.Int32(r)
                        )
                        hvals.append(_ld_shared_cluster_i32(ma))
                    tot_a = cutlass.Int32(0)
                    pre_a = cutlass.Int32(0)
                    for r in cutlass.range_constexpr(CS):
                        if cutlass.Int32(r) < rank:
                            pre_a = pre_a + hvals[r]  # rank-exclusive
                        tot_a = tot_a + hvals[r]
                    s_mrg[i] = tot_a
                    s_hoff[i] = pre_a

            # ---- P6: scan
            cute.arch.barrier()  # merge published
            if cutlass.const_expr(NB__regclus < self.blk):
                scan_cross(
                    s_mrg,
                    s_ws,
                    k,
                    tid,
                    s_res,
                    cutlass.Int32(0),
                    blk=self.blk,
                    nb=NB__regclus,
                    two=False,
                )
            else:
                scan_cross_w(s_mrg, s_ws, k, tid, s_res, blk=self.blk, nb=NB__regclus)
            if cutlass.const_expr(self.dtype != cutlass.Float32):
                if cutlass.const_expr(NB__regclus < self.blk):
                    if tid < cutlass.Int32(NB__regclus):
                        s_mrg[tid] = s_mrg[tid] + s_hoff[tid]
                else:
                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                        i = tid + cutlass.Int32(z * self.blk)
                        s_mrg[i] = s_mrg[i] + s_hoff[i]  # global cursor
            cute.arch.barrier()  # scan published; all bf16 cursors published
            above = s_res[RES_ABOVE]
            m = s_res[RES_M]
            Bv = s_res[RES_B]
            need = k - above
            whole = cutlass.Int32(0)
            if need >= m:
                whole = cutlass.Int32(1)
            degen = cutlass.Int32(0)
            if m > cutlass.Int32(CS * CMPC):
                degen = cutlass.Int32(1)
            # Global per-bin cursors permit a singleton crossing bin to
            # emit any needed unique ties without DSM candidate staging.
            if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                if whole == cutlass.Int32(0) and m > cutlass.Int32(0):
                    if Bv > cutlass.Int32(0) and Bv < cutlass.Int32(NB__regclus - 1):
                        if SC > cutlass.Float32(0.0):
                            center = (
                                cutlass.Float32(Bv) + cutlass.Float32(0.5) - CQ
                            ) * cute.arch.rcp_approx(SC)
                            center_bits = _bf16_rne_bits__reg(center)
                            magnitude = center_bits & cutlass.Uint32(0x7FFF)
                            # Exclude signed zero, subnormal boundaries, and
                            # the maximal finite encodings next to infinity.
                            if magnitude > cutlass.Uint32(0x0080) and magnitude < cutlass.Uint32(
                                0x7F7F
                            ):
                                previous_bits = center_bits - cutlass.Uint32(1)
                                next_bits = center_bits + cutlass.Uint32(1)
                                if (center_bits & cutlass.Uint32(0x8000)) != cutlass.Uint32(0):
                                    previous_bits = center_bits + cutlass.Uint32(1)
                                    next_bits = center_bits - cutlass.Uint32(1)
                                previous = f32_of_u32(previous_bits << cutlass.Uint32(16))
                                following = f32_of_u32(next_bits << cutlass.Uint32(16))
                                # Interior bin boundaries are exactly representable.
                                # These FMA bounds imply the original f2u/clamp
                                # classifications while avoiding three conversions.
                                previous_q = _fmaf__regclus(previous, SC, CQ)
                                next_q = _fmaf__regclus(following, SC, CQ)
                                bin_low = cutlass.Float32(Bv)
                                bin_high = cutlass.Float32(Bv + cutlass.Int32(1))
                                # The bin is nonempty (m > 0). Monotonicity
                                # and these neighbor bounds restrict it to
                                # the candidate value, so checking that value
                                # itself would repeat an established fact.
                                if previous_q < bin_low and next_q >= bin_high:
                                    whole = cutlass.Int32(1)
                                    degen = cutlass.Int32(0)

            # The sentinel bracket also has infinite width. Bypass its
            # collapsed histogram and enter the exact whole-row key-space
            # fallback on rank 0, where fkey(+inf) is the maximum key.
            if okc == cutlass.Int32(0):
                whole = cutlass.Int32(0)
                degen = cutlass.Int32(1)
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                if cutlass.const_expr(NB__regclus < self.blk):
                    if tid < cutlass.Int32(NB__regclus):
                        s_mrg[tid] = s_mrg[tid] + s_hoff[tid]
                else:
                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                        i = tid + cutlass.Int32(z * self.blk)
                        s_mrg[i] = s_mrg[i] + s_hoff[i]  # global cursor
                cute.arch.barrier()  # cursors published

            # ---- P7: register sweep emit (!degen)
            if degen == cutlass.Int32(0):
                LOQ = cutlass.Float32(Bv)
                lim1 = above
                if whole == cutlass.Int32(1):
                    lim1 = above + m
                    if cutlass.const_expr(self.dtype == cutlass.BFloat16):
                        if lim1 > k:
                            lim1 = k
                if cutlass.const_expr(FOLDQ):
                    kmn_st = cutlass.Uint32(0xFFFFFFFF)
                    kmx_st = cutlass.Uint32(0)
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    for s in cutlass.range_constexpr(S):
                        qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)  # bit-identical
                        if qv >= LOQ:
                            bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                            p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                            idx = (
                                (base4 + tid + cutlass.Int32((s // 4) * self.blk))
                                << cutlass.Int32(2)
                            ) + cutlass.Int32(s % 4)
                            if p < lim1:
                                out_row[p] = idx
                            else:
                                if whole == cutlass.Int32(0):
                                    # crossing overflow -> striped DSMEM slabs; TWO
                                    # separate u32 remote stores (NOT packed)
                                    q2i = p - above
                                    rnk = q2i >> cutlass.Int32(LCMPC)
                                    j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ck_addr + j, rnk),
                                        fkey(_val__regclus(frags, s)),
                                    )
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ci_addr + j, rnk), idx
                                    )
                else:
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(VPT >= 2):
                            xv = _val8_pk__regclus(frags, s)
                        else:
                            xv = _val8__regclus(frags, s)
                        qv = _fmaf__regclus(xv, SC, CQ)  # bit-identical
                        if qv >= LOQ:
                            bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                            p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                            idx = (
                                (base4 + tid + cutlass.Int32((s // 8) * self.blk))
                                << cutlass.Int32(3)
                            ) + cutlass.Int32(s % 8)
                            if p < lim1:
                                out_row[p] = idx
                            else:
                                if whole == cutlass.Int32(0):
                                    # crossing overflow -> striped DSMEM slabs; TWO
                                    # separate u32 remote stores (NOT packed)
                                    q2i = p - above
                                    rnk = q2i >> cutlass.Int32(LCMPC)
                                    j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ck_addr + j, rnk),
                                        fkey(xv),
                                    )
                                    _st_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(ci_addr + j, rnk), idx
                                    )
                                    if cutlass.const_expr(FOLDQ):
                                        kq_f = fkey(xv)
                                        if kq_f < kmn_st:
                                            kmn_st = kq_f
                                        if kq_f > kmx_st:
                                            kmx_st = kq_f
                # tail element: tval == -INF fails q>=LOQ elsewhere
                qv = _fmaf__regclus(tval, SC, CQ)
                if qv >= LOQ:
                    bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                    p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                    if p < lim1:
                        out_row[p] = tix
                    else:
                        if whole == cutlass.Int32(0):
                            q2i = p - above
                            rnk = q2i >> cutlass.Int32(LCMPC)
                            j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                            _st_shared_cluster_i32(
                                _mapa_shared_cluster_addr(ck_addr + j, rnk), fkey(tval)
                            )
                            _st_shared_cluster_i32(_mapa_shared_cluster_addr(ci_addr + j, rnk), tix)
                            if cutlass.const_expr(FOLDQ):
                                kq_t = fkey(tval)
                                if kq_t < kmn_st:
                                    kmn_st = kq_t
                                if kq_t > kmx_st:
                                    kmx_st = kq_t

                if cutlass.const_expr(FOLDQ):
                    kmn_w = warp_min_u32(kmn_st)
                    kmx_w = warp_max_u32(kmx_st)
                    if lane == cutlass.Int32(0):
                        if kmn_w <= kmx_w:
                            _red_shared_cluster_min_u32(
                                _mapa_shared_cluster_addr(kmm_addr, cutlass.Int32(0)),
                                kmn_w,
                            )
                            _red_shared_cluster_max_u32(
                                _mapa_shared_cluster_addr(
                                    kmm_addr + cutlass.Int32(4), cutlass.Int32(0)
                                ),
                                kmx_w,
                            )

            # ---- P8: release staging to rank 0
            if cutlass.const_expr(
                self.oneq_enabled
                and self.dtype != cutlass.Float32
                and NB__regclus == 512
                and QUADC__regclus == 96
                and CS == 8
                and VPT == 2
            ):
                if rank == cutlass.Int32(0):
                    if tid == cutlass.Int32(0):
                        # Publish the existing range seeds under P8's barrier.
                        s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                        s_kmm[1] = cutlass.Uint32(0)
            # The ordered cluster arrive below already publishes every P7
            # shared-memory write before any rank leaves the rendezvous.  The
            # extra CTA barrier is therefore needed only by the byte-identical
            # fp32 specialization; omit it from the latency-bound bf16 arm.
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                cute.arch.barrier()
            _cluster_sync_aligned()

            # ---- P9: rank-0 selection
            if rank == cutlass.Int32(0):
                if whole == cutlass.Int32(0):
                    mc = m
                    if degen == cutlass.Int32(1):
                        mc = cutlass.Int32(0)
                    if degen == cutlass.Int32(0):
                        if mc <= cutlass.Int32(QUADC__regclus):
                            if cutlass.const_expr(
                                self.oneq_enabled
                                and self.dtype != cutlass.Float32
                                and NB__regclus == 512
                                and CS >= 2
                                and (
                                    ((QUADC__regclus == 96 or QUADC__regclus == 192) and VPT == 2)
                                    or (
                                        (QUADC__regclus == 192 or QUADC__regclus == 384)
                                        and VPT == 1
                                    )
                                )
                            ):
                                # All gated candidates are local to rank 0. A
                                # single bf16 key class needs no slot-order
                                # O(mc^2) rank: any `need` unique members pass.
                                kmn_q = cutlass.Uint32(0xFFFFFFFF)
                                kmx_q = cutlass.Uint32(0)
                                kv_q = cutlass.Uint32(0)
                                qi_q = cutlass.Int32(0)
                                oneq = cutlass.Int32(0)
                                if cutlass.const_expr(
                                    VPT == 2 and (QUADC__regclus == 96 or QUADC__regclus == 192)
                                ):
                                    # On the wide VPT2 routes every warp scans
                                    # the small local slab independently. This
                                    # trades replicated shared loads for removal
                                    # of one latency-dominant CTA barrier.
                                    for z in cutlass.range_constexpr(QUADC__regclus // 32):
                                        qi_q = lane + cutlass.Int32(32 * z)
                                        kv_q = cutlass.Uint32(0)
                                        if qi_q < mc:
                                            kv_q = cutlass.Uint32(s_ck[qi_q])
                                            if kv_q < kmn_q:
                                                kmn_q = kv_q
                                            if kv_q > kmx_q:
                                                kmx_q = kv_q
                                    kmn_q = warp_min_u32(kmn_q)
                                    kmx_q = warp_max_u32(kmx_q)
                                    if mc > cutlass.Int32(0):
                                        if kmn_q == kmx_q:
                                            oneq = cutlass.Int32(1)
                                else:
                                    kmn_q = cutlass.Uint32(s_kmm[0])
                                    kmx_q = cutlass.Uint32(s_kmm[1])
                                    if mc > cutlass.Int32(0):
                                        if kmn_q == kmx_q:
                                            oneq = cutlass.Int32(1)
                                i = tid
                                uq = cutlass.Uint32(0)
                                rnk = cutlass.Int32(0)
                                j = cutlass.Int32(0)
                                vq = cutlass.Uint32(0)
                                tinc = cutlass.Int32(0)
                                if oneq == cutlass.Int32(1):
                                    while i < need:
                                        out_row[above + i] = s_ci[i]
                                        i = i + cutlass.Int32(BLK)
                                else:
                                    # Multi-class crossing bin: same slot-order
                                    # tie-broken rank. Small bins keep the slim
                                    # serial scan (the 8-wide body's setup and
                                    # bulk cost more than it saves under ~6
                                    # iterations); large bins scan 8-wide so
                                    # the eight LDS issue independently instead
                                    # of one load-use chain per key (mc <= QC
                                    # <= 384 -> one candidate per thread; the
                                    # scalar tail covers mc % 8). The mc gate
                                    # is CTA-uniform: no divergence.
                                    if mc < cutlass.Int32(40):
                                        while i < mc:
                                            uq = cutlass.Uint32(s_ck[i])
                                            rnk = cutlass.Int32(0)
                                            j = cutlass.Int32(0)
                                            while j < mc:
                                                vq = cutlass.Uint32(s_ck[j])
                                                tinc = cutlass.Int32(0)
                                                if vq > uq:
                                                    tinc = cutlass.Int32(1)
                                                if vq == uq:
                                                    if j < i:
                                                        tinc = cutlass.Int32(1)
                                                rnk = rnk + tinc
                                                j = j + cutlass.Int32(1)
                                            if rnk < need:
                                                out_row[above + rnk] = s_ci[i]
                                            i = i + cutlass.Int32(BLK)
                                    else:
                                        lim8q = mc & cutlass.Int32(~7)
                                        while i < mc:
                                            uq = cutlass.Uint32(s_ck[i])
                                            rnk = cutlass.Int32(0)
                                            j = cutlass.Int32(0)
                                            while j < lim8q:
                                                for z8 in cutlass.range_constexpr(8):
                                                    vq = cutlass.Uint32(s_ck[j + cutlass.Int32(z8)])
                                                    tinc = cutlass.Int32(0)
                                                    if vq > uq:
                                                        tinc = cutlass.Int32(1)
                                                    if vq == uq:
                                                        if j + cutlass.Int32(z8) < i:
                                                            tinc = cutlass.Int32(1)
                                                    rnk = rnk + tinc
                                                j = j + cutlass.Int32(8)
                                            while j < mc:
                                                vq = cutlass.Uint32(s_ck[j])
                                                tinc = cutlass.Int32(0)
                                                if vq > uq:
                                                    tinc = cutlass.Int32(1)
                                                if vq == uq:
                                                    if j < i:
                                                        tinc = cutlass.Int32(1)
                                                rnk = rnk + tinc
                                                j = j + cutlass.Int32(1)
                                            if rnk < need:
                                                out_row[above + rnk] = s_ci[i]
                                            i = i + cutlass.Int32(BLK)
                            else:
                                # (1) quad-96: all candidates LOCAL (96 < CMPC),
                                # O(mc^2) slot-order tie-broken rank
                                if cutlass.const_expr(self.dtype == cutlass.Float32):
                                    i = tid
                                    while i < mc:
                                        uq = cutlass.Uint32(s_ck[i])
                                        rnk = cutlass.Int32(0)
                                        j = cutlass.Int32(0)
                                        while j < mc:
                                            vq = cutlass.Uint32(s_ck[j])
                                            tinc = cutlass.Int32(0)
                                            if vq > uq:
                                                tinc = cutlass.Int32(1)
                                            if vq == uq:
                                                if j < i:
                                                    tinc = cutlass.Int32(1)
                                            rnk = rnk + tinc
                                            j = j + cutlass.Int32(1)
                                        if rnk < need:
                                            out_row[above + rnk] = s_ci[i]
                                        i = i + cutlass.Int32(BLK)
                                else:
                                    # bf16 twin: identical slot-order rank —
                                    # slim serial scan for small bins, 8-wide
                                    # inner scan for large ones (the non-oneq
                                    # K<=1024 routes run this for every
                                    # non-whole crossing bin, one-class or
                                    # not). The mc gate is CTA-uniform.
                                    if mc < cutlass.Int32(40):
                                        i = tid
                                        while i < mc:
                                            uq = cutlass.Uint32(s_ck[i])
                                            rnk = cutlass.Int32(0)
                                            j = cutlass.Int32(0)
                                            while j < mc:
                                                vq = cutlass.Uint32(s_ck[j])
                                                tinc = cutlass.Int32(0)
                                                if vq > uq:
                                                    tinc = cutlass.Int32(1)
                                                if vq == uq:
                                                    if j < i:
                                                        tinc = cutlass.Int32(1)
                                                rnk = rnk + tinc
                                                j = j + cutlass.Int32(1)
                                            if rnk < need:
                                                out_row[above + rnk] = s_ci[i]
                                            i = i + cutlass.Int32(BLK)
                                    else:
                                        lim8q = mc & cutlass.Int32(~7)
                                        i = tid
                                        while i < mc:
                                            uq = cutlass.Uint32(s_ck[i])
                                            rnk = cutlass.Int32(0)
                                            j = cutlass.Int32(0)
                                            while j < lim8q:
                                                for z8 in cutlass.range_constexpr(8):
                                                    vq = cutlass.Uint32(s_ck[j + cutlass.Int32(z8)])
                                                    tinc = cutlass.Int32(0)
                                                    if vq > uq:
                                                        tinc = cutlass.Int32(1)
                                                    if vq == uq:
                                                        if j + cutlass.Int32(z8) < i:
                                                            tinc = cutlass.Int32(1)
                                                    rnk = rnk + tinc
                                                j = j + cutlass.Int32(8)
                                            while j < mc:
                                                vq = cutlass.Uint32(s_ck[j])
                                                tinc = cutlass.Int32(0)
                                                if vq > uq:
                                                    tinc = cutlass.Int32(1)
                                                if vq == uq:
                                                    if j < i:
                                                        tinc = cutlass.Int32(1)
                                                rnk = rnk + tinc
                                                j = j + cutlass.Int32(1)
                                            if rnk < need:
                                                out_row[above + rnk] = s_ci[i]
                                            i = i + cutlass.Int32(BLK)
                        else:
                            # (2) key-space narrowing over striped DSMEM slabs:
                            # slot = i & (CMPC-1), rank = i >> LCMPC
                            if cutlass.const_expr(
                                not (
                                    self.oneq_enabled
                                    and self.dtype != cutlass.Float32
                                    and NB__regclus == 512
                                    and QUADC__regclus == 96
                                    and CS == 8
                                    and VPT == 2
                                )
                            ):
                                if tid == cutlass.Int32(0):
                                    s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                                    s_kmm[1] = cutlass.Uint32(0)
                                cute.arch.barrier()  # kmm init
                            i = tid
                            while i < mc:
                                if cutlass.const_expr(self.dtype == cutlass.Float32):
                                    kv = cutlass.Uint32(
                                        _ld_shared_cluster_i32(
                                            _mapa_shared_cluster_addr(
                                                ck_addr
                                                + (
                                                    (i & cutlass.Int32(CMPC - 1))
                                                    << cutlass.Int32(2)
                                                ),
                                                i >> cutlass.Int32(LCMPC),
                                            )
                                        )
                                    )
                                elif cutlass.const_expr(
                                    self.oneq_enabled
                                    and NB__regclus == 512
                                    and QUADC__regclus == 96
                                    and CS == 8
                                    and VPT == 2
                                ):
                                    if mc <= cutlass.Int32(CMPC):
                                        # P9 runs on rank 0 and P7 maps q2i<CMPC
                                        # to rank 0, so this slab is local.
                                        kv = cutlass.Uint32(s_ck[i])
                                    else:
                                        kv = cutlass.Uint32(
                                            _ld_shared_cluster_i32(
                                                _mapa_shared_cluster_addr(
                                                    ck_addr
                                                    + (
                                                        (i & cutlass.Int32(CMPC - 1))
                                                        << cutlass.Int32(2)
                                                    ),
                                                    i >> cutlass.Int32(LCMPC),
                                                )
                                            )
                                        )
                                else:
                                    kv = cutlass.Uint32(
                                        _ld_shared_cluster_i32(
                                            _mapa_shared_cluster_addr(
                                                ck_addr
                                                + (
                                                    (i & cutlass.Int32(CMPC - 1))
                                                    << cutlass.Int32(2)
                                                ),
                                                i >> cutlass.Int32(LCMPC),
                                            )
                                        )
                                    )
                                atomic_min_cta(s_kmm.iterator, kv)
                                atomic_max_cta(s_kmm.iterator + 1, kv)
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # key range published
                            rlo = cutlass.Uint32(s_kmm[0])
                            rhi = cutlass.Uint32(s_kmm[1])
                            if cutlass.const_expr(
                                self.oneq_enabled
                                and self.dtype != cutlass.Float32
                                and NB__regclus == 512
                                and QUADC__regclus == 96
                                and CS == 8
                                and VPT == 2
                            ):
                                done = cutlass.Int32(0)
                            if cutlass.const_expr(
                                self.oneq_enabled
                                and self.dtype != cutlass.Float32
                                and NB__regclus == 512
                                and QUADC__regclus == 96
                                and CS == 8
                                and VPT == 2
                            ):
                                if mc <= cutlass.Int32(CMPC):
                                    if mc > cutlass.Int32(0):
                                        if rlo == rhi:
                                            # Existing range proof says every
                                            # staged key is one BF16 tie class.
                                            # Any `need` local indices are valid.
                                            i = tid
                                            while i < need:
                                                out_row[above + i] = s_ci[i]
                                                i = i + cutlass.Int32(BLK)
                                            # Skip the generic narrowing loop;
                                            # mc=0 also leaves its ballot tail empty.
                                            mc = cutlass.Int32(0)
                                            done = cutlass.Int32(1)
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            aboveC = cutlass.Int32(0)
                            needC = need
                            mm = mc
                            lev = cutlass.Int32(0)
                            rng16d = cutlass.Int32(0)
                            if cutlass.const_expr(
                                not (
                                    self.oneq_enabled
                                    and self.dtype != cutlass.Float32
                                    and NB__regclus == 512
                                    and QUADC__regclus == 96
                                    and CS == 8
                                    and VPT == 2
                                )
                            ):
                                done = cutlass.Int32(0)
                            while done == cutlass.Int32(0):  # <=6 levels
                                if needC == mm:
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                    aboveC = aboveC + mm
                                    needC = cutlass.Int32(0)
                                    done = cutlass.Int32(1)
                                if done == cutlass.Int32(0):
                                    if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                        done = cutlass.Int32(1)
                                    if lev >= cutlass.Int32(6):
                                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                        done = cutlass.Int32(1)
                                if cutlass.const_expr(self.dtype != cutlass.Float32):
                                    if done == cutlass.Int32(0):
                                        if (
                                            cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                        ) < cutlass.Uint32(65536):
                                            # bf16 one-class exit: 0x10000 key spacing means
                                            # this bracket holds ONE value class -- emit ties
                                            # by key range (rng16d).
                                            ethr = cutlass.Int64(cutlass.Uint32(rhi))
                                            rng16d = cutlass.Int32(1)
                                            done = cutlass.Int32(1)
                                if done == cutlass.Int32(0):
                                    d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                    b2w = cutlass.Int32(32) - clz_i32(
                                        cutlass.Int32(d2 | cutlass.Uint32(1))
                                    )
                                    sh2 = cutlass.Int32(0)
                                    if b2w > cutlass.Int32(LNB):
                                        sh2 = b2w - cutlass.Int32(LNB)
                                    if cutlass.const_expr(NB__regclus < self.blk):
                                        if tid < cutlass.Int32(NB__regclus):
                                            s_hist[tid] = cutlass.Int32(0)
                                    else:
                                        for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                            s_hist[tid + cutlass.Int32(z * self.blk)] = (
                                                cutlass.Int32(0)
                                            )
                                    cute.arch.barrier()  # level clear
                                    i = tid
                                    while i < mc:
                                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                                            unar = cutlass.Uint32(
                                                _ld_shared_cluster_i32(
                                                    _mapa_shared_cluster_addr(
                                                        ck_addr
                                                        + (
                                                            (i & cutlass.Int32(CMPC - 1))
                                                            << cutlass.Int32(2)
                                                        ),
                                                        i >> cutlass.Int32(LCMPC),
                                                    )
                                                )
                                            )
                                        elif cutlass.const_expr(
                                            self.oneq_enabled
                                            and NB__regclus == 512
                                            and QUADC__regclus == 96
                                            and CS == 8
                                            and VPT == 2
                                        ):
                                            if mc <= cutlass.Int32(CMPC):
                                                unar = cutlass.Uint32(s_ck[i])
                                            else:
                                                unar = cutlass.Uint32(
                                                    _ld_shared_cluster_i32(
                                                        _mapa_shared_cluster_addr(
                                                            ck_addr
                                                            + (
                                                                (i & cutlass.Int32(CMPC - 1))
                                                                << cutlass.Int32(2)
                                                            ),
                                                            i >> cutlass.Int32(LCMPC),
                                                        )
                                                    )
                                                )
                                        else:
                                            unar = cutlass.Uint32(
                                                _ld_shared_cluster_i32(
                                                    _mapa_shared_cluster_addr(
                                                        ck_addr
                                                        + (
                                                            (i & cutlass.Int32(CMPC - 1))
                                                            << cutlass.Int32(2)
                                                        ),
                                                        i >> cutlass.Int32(LCMPC),
                                                    )
                                                )
                                            )
                                        if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                            if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                                bnn = (
                                                    cutlass.Uint32(unar) - cutlass.Uint32(rlo)
                                                ) >> cutlass.Uint32(sh2)
                                                bnn = _umin_u32__regclus(
                                                    bnn, cutlass.Uint32(NB__regclus - 1)
                                                )
                                                atomic_add_cta(
                                                    s_hist.iterator + cutlass.Int32(bnn),
                                                    cutlass.Int32(1),
                                                )
                                        i = i + cutlass.Int32(BLK)
                                    cute.arch.barrier()  # level hist
                                    find_cross(s_hist, needC, tid, s_res, nb=NB__regclus)
                                    cute.arch.barrier()  # level scan
                                    aboveC = aboveC + s_res[RES_ABOVE]
                                    needC = needC - s_res[RES_ABOVE]
                                    mm = s_res[RES_M]
                                    b_lv = s_res[RES_B]
                                    nlo = cutlass.Uint32(rlo) + (
                                        cutlass.Uint32(b_lv) << cutlass.Uint32(sh2)
                                    )
                                    if b_lv != cutlass.Int32(NB__regclus - 1):
                                        rhi = nlo + (
                                            (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                            - cutlass.Uint32(1)
                                        )
                                    rlo = nlo
                                    lev = lev + cutlass.Int32(1)
                            if mc > cutlass.Int32(0):
                                cute.arch.barrier()  # narrowing done
                            # two-predicate ballot emit over the striped slabs
                            lml = cutlass.Int32(cute.arch.lanemask_lt())
                            it2 = (mc + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                            it = cutlass.Int32(0)
                            while it < it2:
                                i = it * cutlass.Int32(BLK) + tid
                                uke = cutlass.Uint32(0)
                                idv = cutlass.Int32(0)
                                if i < mc:
                                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                                        uke = cutlass.Uint32(
                                            _ld_shared_cluster_i32(
                                                _mapa_shared_cluster_addr(
                                                    ck_addr
                                                    + (
                                                        (i & cutlass.Int32(CMPC - 1))
                                                        << cutlass.Int32(2)
                                                    ),
                                                    i >> cutlass.Int32(LCMPC),
                                                )
                                            )
                                        )
                                        idv = _ld_shared_cluster_i32(
                                            _mapa_shared_cluster_addr(
                                                ci_addr
                                                + (
                                                    (i & cutlass.Int32(CMPC - 1))
                                                    << cutlass.Int32(2)
                                                ),
                                                i >> cutlass.Int32(LCMPC),
                                            )
                                        )
                                    elif cutlass.const_expr(
                                        self.oneq_enabled
                                        and NB__regclus == 512
                                        and QUADC__regclus == 96
                                        and CS == 8
                                        and VPT == 2
                                    ):
                                        if mc <= cutlass.Int32(CMPC):
                                            uke = cutlass.Uint32(s_ck[i])
                                            idv = s_ci[i]
                                        else:
                                            uke = cutlass.Uint32(
                                                _ld_shared_cluster_i32(
                                                    _mapa_shared_cluster_addr(
                                                        ck_addr
                                                        + (
                                                            (i & cutlass.Int32(CMPC - 1))
                                                            << cutlass.Int32(2)
                                                        ),
                                                        i >> cutlass.Int32(LCMPC),
                                                    )
                                                )
                                            )
                                            idv = _ld_shared_cluster_i32(
                                                _mapa_shared_cluster_addr(
                                                    ci_addr
                                                    + (
                                                        (i & cutlass.Int32(CMPC - 1))
                                                        << cutlass.Int32(2)
                                                    ),
                                                    i >> cutlass.Int32(LCMPC),
                                                )
                                            )
                                    else:
                                        uke = cutlass.Uint32(
                                            _ld_shared_cluster_i32(
                                                _mapa_shared_cluster_addr(
                                                    ck_addr
                                                    + (
                                                        (i & cutlass.Int32(CMPC - 1))
                                                        << cutlass.Int32(2)
                                                    ),
                                                    i >> cutlass.Int32(LCMPC),
                                                )
                                            )
                                        )
                                        idv = _ld_shared_cluster_i32(
                                            _mapa_shared_cluster_addr(
                                                ci_addr
                                                + (
                                                    (i & cutlass.Int32(CMPC - 1))
                                                    << cutlass.Int32(2)
                                                ),
                                                i >> cutlass.Int32(LCMPC),
                                            )
                                        )
                                q1f = cutlass.Int32(0)
                                q2f = cutlass.Int32(0)
                                if i < mc:
                                    if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                        q1f = cutlass.Int32(1)
                                    if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                        q2f = cutlass.Int32(1)
                                    if cutlass.const_expr(self.dtype != cutlass.Float32):
                                        if rng16d != cutlass.Int32(0):
                                            q2f = cutlass.Int32(0)
                                            if cutlass.Int64(cutlass.Uint32(uke)) >= cutlass.Int64(
                                                cutlass.Uint32(rlo)
                                            ):
                                                if cutlass.Int64(cutlass.Uint32(uke)) <= ethr:
                                                    q2f = cutlass.Int32(1)
                                n1 = ballot(q1f == cutlass.Int32(1))
                                n2 = ballot(q2f == cutlass.Int32(1))
                                b1 = cutlass.Int32(0)
                                b2 = cutlass.Int32(0)
                                if lane == cutlass.Int32(0):
                                    if n1 != cutlass.Int32(0):
                                        b1 = atomic_add_cta(s_cnt.iterator, popc(n1))
                                    if n2 != cutlass.Int32(0):
                                        b2 = atomic_add_cta(s_cnt.iterator + 1, popc(n2))
                                b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                                b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                                p1e = b1 + popc(n1 & lml)
                                p2e = b2 + popc(n2 & lml)
                                if q1f == cutlass.Int32(1):
                                    if p1e < aboveC:
                                        out_row[above + p1e] = idv
                                if q2f == cutlass.Int32(1):
                                    if p2e < needC:
                                        out_row[above + aboveC + p2e] = idv
                                it = it + cutlass.Int32(1)
                    else:
                        # (3) degen safety net: crossing bin larger than the
                        # whole cluster buffer -> exact whole-row key-space
                        # narrowing by rank 0 alone, <=8 levels.
                        rlo = cutlass.Uint32(0)
                        rhi = cutlass.Uint32(0xFFFFFFFF)
                        aboveC = cutlass.Int32(0)  # above2
                        needC = k  # need2
                        mm = n  # m2
                        ethr = cutlass.Int64(0)
                        tie_m = cutlass.Int32(1)
                        lev = cutlass.Int32(0)
                        rng16n = cutlass.Int32(0)
                        done = cutlass.Int32(0)
                        while done == cutlass.Int32(0):
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                tie_m = cutlass.Int32(0)
                                done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                                if lev >= cutlass.Int32(8):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                            if cutlass.const_expr(self.dtype != cutlass.Float32):
                                if done == cutlass.Int32(0):
                                    if (cutlass.Uint32(rhi) - cutlass.Uint32(rlo)) < cutlass.Uint32(
                                        65536
                                    ):
                                        # bf16 one-class exit: 0x10000 key spacing means
                                        # this bracket holds ONE value class -- emit ties
                                        # by key range (rng16n).
                                        ethr = cutlass.Int64(cutlass.Uint32(rhi))
                                        rng16n = cutlass.Int32(1)
                                        done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2w = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = cutlass.Int32(0)
                                if b2w > cutlass.Int32(LNB):
                                    sh2 = b2w - cutlass.Int32(LNB)
                                if cutlass.const_expr(NB__regclus < self.blk):
                                    if tid < cutlass.Int32(NB__regclus):
                                        s_hist[tid] = cutlass.Int32(0)
                                else:
                                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                        s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                cute.arch.barrier()  # level clear
                                i = tid
                                while i < n:  # whole-row bin
                                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                                        unar = fkey(ldg_f32(x_addr, i))
                                    else:
                                        unar = fkey(ldg_bf16(x_addr, i))
                                    if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                        if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                            bnn = (
                                                cutlass.Uint32(unar) - cutlass.Uint32(rlo)
                                            ) >> cutlass.Uint32(sh2)
                                            bnn = _umin_u32__regclus(
                                                bnn, cutlass.Uint32(NB__regclus - 1)
                                            )
                                            atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(bnn),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # level hist
                                find_cross(s_hist, needC, tid, s_res, nb=NB__regclus)
                                cute.arch.barrier()  # level scan
                                aboveC = aboveC + s_res[RES_ABOVE]
                                needC = needC - s_res[RES_ABOVE]
                                mm = s_res[RES_M]
                                b_lv = s_res[RES_B]
                                nlo = cutlass.Uint32(rlo) + (
                                    cutlass.Uint32(b_lv) << cutlass.Uint32(sh2)
                                )
                                if b_lv != cutlass.Int32(NB__regclus - 1):
                                    rhi = nlo + (
                                        (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                        - cutlass.Uint32(1)
                                    )
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        cute.arch.barrier()  # narrowing done
                        nA = k  # tie_m ? above2 : k
                        if tie_m == cutlass.Int32(1):
                            nA = aboveC
                        nT = cutlass.Int32(0)
                        if tie_m == cutlass.Int32(1):
                            nT = needC
                        lml = cutlass.Int32(cute.arch.lanemask_lt())
                        it2 = (n + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tid
                            uke = cutlass.Uint32(0)
                            if i < n:
                                if cutlass.const_expr(self.dtype == cutlass.Float32):
                                    uke = fkey(ldg_f32(x_addr, i))
                                else:
                                    uke = fkey(ldg_bf16(x_addr, i))
                            q1f = cutlass.Int32(0)
                            q2f = cutlass.Int32(0)
                            if i < n:
                                if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                    q1f = cutlass.Int32(1)
                                if tie_m == cutlass.Int32(1):
                                    if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                        q2f = cutlass.Int32(1)
                                    if cutlass.const_expr(self.dtype != cutlass.Float32):
                                        if rng16n != cutlass.Int32(0):
                                            q2f = cutlass.Int32(0)
                                            if cutlass.Int64(cutlass.Uint32(uke)) >= cutlass.Int64(
                                                cutlass.Uint32(rlo)
                                            ):
                                                if cutlass.Int64(cutlass.Uint32(uke)) <= ethr:
                                                    q2f = cutlass.Int32(1)
                            n1 = ballot(q1f == cutlass.Int32(1))
                            n2 = ballot(q2f == cutlass.Int32(1))
                            b1 = cutlass.Int32(0)
                            b2 = cutlass.Int32(0)
                            if lane == cutlass.Int32(0):
                                if n1 != cutlass.Int32(0):
                                    b1 = atomic_add_cta(s_cnt.iterator, popc(n1))
                                if n2 != cutlass.Int32(0):
                                    b2 = atomic_add_cta(s_cnt.iterator + 1, popc(n2))
                            b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                            b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                            p1e = b1 + popc(n1 & lml)
                            p2e = b2 + popc(n2 & lml)
                            if q1f == cutlass.Int32(1):
                                if p1e < nA:
                                    out_row[p1e] = i
                            if q2f == cutlass.Int32(1):
                                if p2e < nT:
                                    out_row[nA + p2e] = i
                            it = it + cutlass.Int32(1)

            # ---- P10: FINAL cluster rendezvous — ALL ranks reach it;
            # keeps peers resident until rank 0 has read their ck/ci.
            _cluster_sync_aligned()

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        stream,
    ):
        b = out.shape[0]
        self.kern(logits, pre_idx, kv_lens, out, n).launch(
            grid=(self.cs, b, 1),
            block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1),
            stream=stream,
            smem=SMEM_BYTES,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )


_COMPILE_CACHE__regclus: dict = {}


def get_compiled__regclus(
    tpl: tuple,
    dump_dir: str | None = None,
    pdl: bool = False,
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
    hint_free: bool = False,
    dtype: str = "f32",
    nbh: int = 1024,
    quadc: int = 384,
    oneq_enabled: bool = False,
) -> Any:
    """Compile (or fetch) the variant for constexpr tuple (BLK, VPT, CS)."""
    key = (
        tuple(tpl),
        bool(pdl),
        bool(varlen),
        int(next_n),
        int(cr_shift),
        bool(hint_free),
        str(dtype),
        int(nbh),
        int(quadc),
        bool(oneq_enabled),
    )
    compiled = _COMPILE_CACHE__regclus.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt

        blk, vpt, cs = tpl
        kernel = GvrRegClusKernel(
            blk,
            vpt,
            cs,
            pdl=pdl,
            varlen=varlen,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=hint_free,
            dtype=_dtype_of(dtype),
            oneq_enabled=oneq_enabled,
        )
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        if dtype == "f32":
            lg_fake = _crt.make_fake_compact_tensor(
                cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16
            )
        else:
            lg_fake = _crt.make_fake_compact_tensor(
                cutlass.BFloat16, (nb_, nc_), stride_order=(1, 0), assumed_align=16
            )
        pi_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb2_, nc2_), stride_order=(1, 0), assumed_align=16
        )
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb3_, nc3_), stride_order=(1, 0), assumed_align=16
        )
        v0_ = cute.sym_int()
        kv_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (v0_,), stride_order=(0,), assumed_align=4
        )
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        opts = "--enable-tvm-ffi"
        if dump_dir:
            opts += f" --keep-ptx --keep-cubin --dump-dir {dump_dir}"
        assert nbh in (512, 1024)
        assert quadc in (96, 384)
        if dtype == "f32":
            assert nbh == 1024
        global NB__regclus, LNB, DYN_SMEM_BYTES, SMEM_BYTES
        global W_MRG, W_HOFF, W_CK, W_CI, QUADC__regclus
        _cfg_saved = (
            NB__regclus,
            LNB,
            DYN_SMEM_BYTES,
            SMEM_BYTES,
            W_MRG,
            W_HOFF,
            W_CK,
            W_CI,
            QUADC__regclus,
        )
        if dtype != "f32" and nbh == 512:
            NB__regclus = 512
            LNB = 9
            DYN_SMEM_BYTES = (3 * NB__regclus + 2 * CMPC) * 4
            SMEM_BYTES = STATIC_BYTES__regclus + DYN_SMEM_BYTES
            W_MRG = STATIC_WORDS__regclus + NB__regclus
            W_HOFF = STATIC_WORDS__regclus + 2 * NB__regclus
            W_CK = STATIC_WORDS__regclus + 3 * NB__regclus
            W_CI = STATIC_WORDS__regclus + 3 * NB__regclus + CMPC
            QUADC__regclus = quadc
        try:
            compiled = cute.compile(
                kernel,
                lg_fake,
                pi_fake,
                kv_fake,
                out_fake,
                cutlass.Int32(0),
                stream=fake_stream,
                options=opts,
            )
        finally:
            (
                NB__regclus,
                LNB,
                DYN_SMEM_BYTES,
                SMEM_BYTES,
                W_MRG,
                W_HOFF,
                W_CK,
                W_CI,
                QUADC__regclus,
            ) = _cfg_saved
        _COMPILE_CACHE__regclus[key] = compiled
    return compiled


__all__ = ["GvrRegClusKernel", "get_compiled__regclus"]
