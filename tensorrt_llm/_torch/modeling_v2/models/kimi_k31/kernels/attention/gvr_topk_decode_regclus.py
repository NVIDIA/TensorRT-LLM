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
# ==== family: regclus =========================================
# ===========================================================================
"""gvr_reg_clus — CLUSTERED register-resident GVR: the register algorithm
(T = GMIN directly, one float-space histogram, one register sweep) run
across a cluster of CS CTAs; per-CTA instruction stream intentionally
identical to the single-CTA reg path plus two hardware cluster barriers and
CS DSMEM reads per bin. Signedness rule applied at every unsigned
compare/shift in/after dynamic loops.

Template knobs (CUDA `gvr_reg_clus<BLK,VPT,CS>`, all instantiations
BLK=BLKC=1024): ctor args of :class:`GvrRegClusKernel`. Runtime args mirror
the CUDA `(n, npad, k)` — npad/k come from tensor shapes, so only `n` crosses
the ABI. Launch: grid=(CS, b), cluster=(CS,1,1), block=1024, dynamic smem
45,056 B (+512 B static-mirror prelude), `__launch_bounds__(1024,1)` ==
min_blocks_per_mp=1 -> 64-register wall.

Shared-memory map (single dynamic window, word offsets; the CUDA static
__shared__ block folded into the first 512 B — byte-identical layout in every
CTA, a mapa/DSMEM requirement):

    [0..5]        s_res   (shared slot map RES_B/M/ABOVE/TOT/B2/B3)
    [6..7]        s_cnt   (s_o1, s_o2)
    [8..9]        s_kmm   (s_kmin, s_kmax — Uint32)
    [16..16+32)   ws      (scan_cross_w workspace)
    [48..48+32)   wmn     (Uint32 warp min partials)
    [80..80+32)   wmx     (Uint32 warp max partials)
    [128..1152)   hist    (this CTA's raw counts)
    [1152..2176)  mrg     (cluster totals -> per-CTA global write cursors)
    [2176..3200)  hoff    (this CTA's rank-exclusive bin offset)
    [3200..7296)  ck      (crossing keys, Uint32, CMPC=4096 slots)
    [7296..11392) ci      (crossing indices, Int32, CMPC slots)

Launch smem = 45,568 B (compile-time constant -> plain int at .launch();
MINB==1 so the _build_kernel_attrs carveout path is not taken and the reg
family's _no_carveout workaround is unnecessary here).
"""

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import dsl_user_op

# The shared device units, the main family and the families before this one, from the module that
# re-exports this one (``C`` as in that module, where it is the module itself).
try:
    from . import gvr_topk_decode as C
except ImportError:  # standalone dir
    import gvr_topk_decode as C

RES_ABOVE = C.RES_ABOVE
RES_B = C.RES_B
RES_M = C.RES_M
SENT_HI = C.SENT_HI
SENT_LO = C.SENT_LO
_POS_INF = C._POS_INF
_cluster_sync_aligned = C._cluster_sync_aligned
_ld_shared_cluster_i32 = C._ld_shared_cluster_i32
_mapa_shared_cluster_addr = C._mapa_shared_cluster_addr
_st_shared_cluster_i32 = C._st_shared_cluster_i32
atomic_add_cta = C.atomic_add_cta
atomic_max_cta = C.atomic_max_cta
atomic_min_cta = C.atomic_min_cta
ballot = C.ballot
clz_i32 = C.clz_i32
f2u_rz = C.f2u_rz
f32_of_u32 = C.f32_of_u32
ffs_m1 = C.ffs_m1
find_cross = C.find_cross
fkey = C.fkey
g2r_atom_f32 = C.g2r_atom_f32
invkey = C.invkey
ld_g_f32x4 = C.ld_g_f32x4
ld_g_i32 = C.ld_g_i32
ldg_f32 = C.ldg_f32
popc = C.popc
scan_cross_w = C.scan_cross_w
u32_of_f32 = C.u32_of_f32
warp_add_i32 = C.warp_add_i32
warp_max_u32 = C.warp_max_u32
warp_min_u32 = C.warp_min_u32


# ---- constants --------------------------------------------------------------
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

# word offsets into the shared window (module docstring)
W_HIST = STATIC_WORDS__regclus
W_MRG = STATIC_WORDS__regclus + NB__regclus
W_HOFF = STATIC_WORDS__regclus + 2 * NB__regclus
W_CK = STATIC_WORDS__regclus + 3 * NB__regclus
W_CI = STATIC_WORDS__regclus + 3 * NB__regclus + CMPC

_NEG_INF__regclus = float("-inf")


# ---------------------------------------------------------------------------
# module-local FP/util spellings (same forms as the reg family)
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf__regclus(a, b, c, *, loc=None, ip=None):
    """CUDA fmaf: single fma.rn.f32 (classify == emit bit-exact)."""
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


@cute.jit
def _umin_u32__regclus(a, b):
    """unsigned min(a, b) — CUDA min() on the bin clamp (IMNMX)."""
    r = a
    if b < a:
        r = b
    return r


@cute.jit
def _fabsf__regclus(x):
    """|x| via sign-bit clear (exact, matches fabsf)."""
    return f32_of_u32(u32_of_f32(x) & cutlass.Uint32(0x7FFFFFFF))


def _smem_view__regclus(dtype, sbase, word_off: int, length: int, align: int = 16):
    """Typed tensor view at a constexpr word offset into the smem window."""
    p = cute.make_ptr(
        dtype, sbase + cutlass.Int32(word_off * 4), cute.AddressSpace.smem, assumed_align=align
    )
    return cute.make_tensor(p, cute.make_layout((length,)))


def _val__regclus(frags, s: int):
    """val[s] accessor over the float4[VPT] register batch (constexpr s)."""
    return frags[s // 4][s % 4]


class GvrRegClusKernel:
    """gvr_reg_clus<BLK, VPT, CS, K>; K specializes only hint-slot unrolling."""

    def __init__(
        self,
        tpl: tuple[int, int, int, int],
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        hint_free: bool = False,
    ) -> None:
        blk, vpt, cs, k = tpl
        assert 0 < k <= 2 * blk, "reg_clus supports at most two hint slots per thread"
        self.top_k = k
        assert blk == BLKC, "all instantiations BLK=BLKC=1024"
        assert vpt in (1, 2, 4) and cs in (2, 4, 8)
        self.blk = blk
        self.vpt = vpt
        self.cs = cs
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel): n is re-derived PER ROW in-kernel from
        # a device kv_lens tensor; the scalar n launch arg becomes the
        # envelope clamp bound. next_n / cr_shift are compile-time.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        # hint-free: P0 samples K values in uniform contiguous blocks instead of the hint
        self.hint_free = bool(hint_free)
        self.S = vpt * 4
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
        # Keep tensor shapes dynamic: specializing selection arithmetic can
        # regress the exact-rescue path. Static K only controls slot unrolling.
        K = cutlass.const_expr(self.top_k)
        HINT_SLOTS = cutlass.const_expr((K + self.blk - 1) // self.blk)

        tid, _, _ = cute.arch.thread_idx()
        rank, row, _ = cute.arch.block_idx()  # bx=rank
        lane = tid & cutlass.Int32(31)
        dw = tid >> cutlass.Int32(5)  # DET SLOTS: warp index

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
                    for h in cutlass.range_constexpr(HINT_SLOTS):
                        th = tid + cutlass.Int32(h * self.blk)
                        if th < kq:
                            ov = cutlass.Int32(-1)
                            if th < nv:
                                ov = th
                            out[row, th] = ov

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
        idx = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
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
        pv1 = cutlass.Int32(-1)
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
            s_dw = _smem_view__regclus(
                cutlass.Int32, sbase, 16, 64
            )  # DET SLOTS: per-warp class counts (aliases ws+wmn)
            s_dx = _smem_view__regclus(
                cutlass.Int32, sbase, 80, 32
            )  # DET SLOTS: per-warp hoff partials (aliases wmx)
            s_hist = _smem_view__regclus(cutlass.Int32, sbase, W_HIST, NB__regclus)
            s_mrg = _smem_view__regclus(cutlass.Int32, sbase, W_MRG, NB__regclus)
            s_hoff = _smem_view__regclus(cutlass.Int32, sbase, W_HOFF, NB__regclus)
            s_ck = _smem_view__regclus(cutlass.Uint32, sbase, W_CK, CMPC)
            s_ci = _smem_view__regclus(cutlass.Int32, sbase, W_CI, CMPC, align=4)
            # raw byte bases for DSMEM (mapa) addressing
            hist_addr = sbase + cutlass.Int32(W_HIST * 4)
            ck_addr = sbase + cutlass.Int32(W_CK * 4)
            ci_addr = sbase + cutlass.Int32(W_CI * 4)

            n4 = n >> cutlass.Int32(2)
            ntail = n - (n4 << cutlass.Int32(2))
            base4 = rank * cutlass.Int32(self.span)
            tix = (n4 << cutlass.Int32(2)) + tid  # CUDA `tidx`

            # ---- P0: redundant hint gather, EVERY CTA (ceil(K/BLK) slots).
            # One coalesced word per thread per slot, NO cluster barrier —
            # GMIN/GMAX identical everywhere by construction.
            if cutlass.const_expr(self.hint_free):
                # Exactly K distinct values in endpoint-inclusive uniform blocks.
                if tid < k:
                    pv0 = tid
                if cutlass.const_expr(HINT_SLOTS == 2):
                    if tid + cutlass.Int32(self.blk) < k:
                        pv1 = tid + cutlass.Int32(self.blk)
                if k == cutlass.Int32(512) or k == cutlass.Int32(1024) or k == cutlass.Int32(2048):
                    sample_blocks = k // cutlass.Int32(64)
                    if tid < k:
                        pv0 = ((tid // cutlass.Int32(64)) * (n - cutlass.Int32(64))) // (
                            sample_blocks - cutlass.Int32(1)
                        ) + tid % cutlass.Int32(64)
                    if cutlass.const_expr(HINT_SLOTS == 2):
                        sample_id = tid + cutlass.Int32(self.blk)
                        if sample_id < k:
                            pv1 = ((sample_id // cutlass.Int32(64)) * (n - cutlass.Int32(64))) // (
                                sample_blocks - cutlass.Int32(1)
                            ) + sample_id % cutlass.Int32(64)
            else:
                if tid < k:
                    pv0 = ld_g_i32(p_addr, tid)
                if cutlass.const_expr(HINT_SLOTS == 2):
                    if tid + cutlass.Int32(self.blk) < k:
                        pv1 = ld_g_i32(p_addr, tid + cutlass.Int32(self.blk))

            # ---- P1: row load — predicated flat float4[VPT] batch (the CUDA
            # has NO exact-fit peel here, guard is per-load). Issue all loads
            # first, then -INFINITY-fill missed slots.
            atom128 = g2r_atom_f32(128, invariant=True)
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
            # tail element: rank 0 only
            if rank == cutlass.Int32(0):
                if tid < ntail:
                    tval = ldg_f32(x_addr, tix)

            # ---- P2: init. NB__regclus == BLK -> single-pass hist clear.
            if tid == cutlass.Int32(0):
                s_cnt[0] = cutlass.Int32(0)
                s_cnt[1] = cutlass.Int32(0)
            for z in cutlass.range_constexpr(NB__regclus // self.blk):
                s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

            # ---- P3: GMIN/GMAX from the hint, ONE barrier fold.
            lmin = cutlass.Uint32(0xFFFFFFFF)
            lmax = cutlass.Uint32(0)
            if cutlass.Uint32(pv0) < cutlass.Uint32(n):
                uk = fkey(ldg_f32(x_addr, pv0))  # __ldg(X+pv0)
                lmin = uk
                lmax = uk
            if cutlass.const_expr(HINT_SLOTS == 2):
                if cutlass.Uint32(pv1) < cutlass.Uint32(n):
                    uk = fkey(ldg_f32(x_addr, pv1))
                    if uk < lmin:
                        lmin = uk
                    if uk > lmax:
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

            if okc != cutlass.Int32(0) or cutlass.const_expr(not self.hint_free):
                # ---- P4: histogram; tval add UNCONDITIONAL (trash bin swallows
                # -INFINITY via the saturating cvt).
                for s in cutlass.range_constexpr(S):
                    qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)
                    bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                    atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                qv = _fmaf__regclus(tval, SC, CQ)
                bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))

            else:
                # Publish a scan sentinel; exact rescue uses all valid row keys.
                if rank == cutlass.Int32(0) and tid == cutlass.Int32(0):
                    s_hist[NB__regclus - 1] = k

            # ---- P5: cluster merge
            _cluster_sync_aligned()  # histograms complete on every rank
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
            scan_cross_w(s_mrg, s_ws, k, tid, s_res, blk=self.blk, nb=NB__regclus)
            cute.arch.barrier()  # scan published
            above = s_res[RES_ABOVE]
            m = s_res[RES_M]
            Bv = s_res[RES_B]
            need = k - above
            whole = cutlass.Int32(0)
            if need >= m:
                whole = cutlass.Int32(1)
            degen = cutlass.Int32(0)
            rescue_limit = cutlass.Int32(CS * CMPC)
            if cutlass.const_expr(self.hint_free):
                # Serial refinement becomes expensive before the candidate buffer fills.
                # Cap work independently of K so K=2048 also takes the parallel escape.
                rescue_limit = min(rescue_limit, cutlass.Int32(32) * k)
                rescue_limit = min(rescue_limit, cutlass.Int32(16384))
            if m > rescue_limit:
                degen = cutlass.Int32(1)
            # The sentinel bracket also has infinite width. Bypass its
            # collapsed histogram and enter the exact whole-row key-space
            # fallback, where fkey(+inf) is the maximum key.
            if okc == cutlass.Int32(0):
                whole = cutlass.Int32(0)
                degen = cutlass.Int32(1)
            dsp = cutlass.Int32(0)
            for z in cutlass.range_constexpr(NB__regclus // self.blk):
                i = tid + cutlass.Int32(z * self.blk)
                if i > Bv:
                    dsp = dsp + s_hoff[i]
            dsp = cutlass.Int32(warp_add_i32(dsp))
            if lane == cutlass.Int32(0):
                s_dx[dw] = dsp
            cute.arch.barrier()  # cursors published

            # ---- P7: register sweep emit (!degen)
            v1 = cutlass.Int32(0)
            if lane < cutlass.Int32(NW):
                v1 = s_dx[lane]
            dR1 = cutlass.Int32(warp_add_i32(v1))
            dR2 = s_hoff[Bv]
            if whole == cutlass.Int32(1):
                dR1 = dR1 + dR2
            if degen == cutlass.Int32(0):
                HIf = cutlass.Float32(_POS_INF)
                LOf = cutlass.Float32(_POS_INF)
                if whole == cutlass.Int32(1):
                    HIf = cutlass.Float32(Bv)
                else:
                    if Bv < cutlass.Int32(NB__regclus - 1):
                        HIf = cutlass.Float32(Bv + cutlass.Int32(1))
                    LOf = cutlass.Float32(Bv)
                m1 = cutlass.Int32(0)
                m2 = cutlass.Int32(0)
                for s in cutlass.range_constexpr(S):
                    qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)  # bit-identical
                    if qv >= HIf:
                        m1 = m1 | cutlass.Int32(1 << s)
                    else:
                        if qv >= LOf:
                            m2 = m2 | cutlass.Int32(1 << s)
                qv = _fmaf__regclus(tval, SC, CQ)  # tail (rank 0 only; -INF elsewhere)
                t1 = cutlass.Int32(0)
                t2 = cutlass.Int32(0)
                if qv >= HIf:
                    t1 = cutlass.Int32(1)
                else:
                    if qv >= LOf:
                        t2 = cutlass.Int32(1)
                lml = cutlass.Int32(cute.arch.lanemask_lt())
                c1 = popc(m1) + t1
                c2 = popc(m2) + t2
                w1 = cutlass.Int32(warp_add_i32(c1))
                w2 = cutlass.Int32(warp_add_i32(c2))
                x1 = cutlass.Int32(0)
                x1 = x1 + (
                    popc(
                        ballot(((c1 >> cutlass.Int32(0)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(0)
                )
                x1 = x1 + (
                    popc(
                        ballot(((c1 >> cutlass.Int32(1)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(1)
                )
                x1 = x1 + (
                    popc(
                        ballot(((c1 >> cutlass.Int32(2)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(2)
                )
                x1 = x1 + (
                    popc(
                        ballot(((c1 >> cutlass.Int32(3)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(3)
                )
                x1 = x1 + (
                    popc(
                        ballot(((c1 >> cutlass.Int32(4)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(4)
                )
                if cutlass.const_expr(S + 1 >= 32):
                    x1 = x1 + (
                        popc(
                            ballot(
                                ((c1 >> cutlass.Int32(5)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(5)
                    )
                x2 = cutlass.Int32(0)
                x2 = x2 + (
                    popc(
                        ballot(((c2 >> cutlass.Int32(0)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(0)
                )
                x2 = x2 + (
                    popc(
                        ballot(((c2 >> cutlass.Int32(1)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(1)
                )
                x2 = x2 + (
                    popc(
                        ballot(((c2 >> cutlass.Int32(2)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(2)
                )
                x2 = x2 + (
                    popc(
                        ballot(((c2 >> cutlass.Int32(3)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(3)
                )
                x2 = x2 + (
                    popc(
                        ballot(((c2 >> cutlass.Int32(4)) & cutlass.Int32(1)) == cutlass.Int32(1))
                        & lml
                    )
                    << cutlass.Int32(4)
                )
                if cutlass.const_expr(S + 1 >= 32):
                    x2 = x2 + (
                        popc(
                            ballot(
                                ((c2 >> cutlass.Int32(5)) & cutlass.Int32(1)) == cutlass.Int32(1)
                            )
                            & lml
                        )
                        << cutlass.Int32(5)
                    )
                if lane == cutlass.Int32(0):
                    s_dw[dw] = w1
                    s_dw[32 + dw] = w2
                cute.arch.barrier()  # DET SLOTS: warp totals published
                v1 = cutlass.Int32(0)
                v2 = cutlass.Int32(0)
                if lane < cutlass.Int32(NW):
                    v1 = s_dw[lane]
                    v2 = s_dw[32 + lane]
                tot1 = cutlass.Int32(warp_add_i32(v1))
                tot2 = cutlass.Int32(warp_add_i32(v2))
                if lane >= dw:
                    v1 = cutlass.Int32(0)
                    v2 = cutlass.Int32(0)
                b1 = cutlass.Int32(warp_add_i32(v1))
                b2 = cutlass.Int32(warp_add_i32(v2))
                p1 = dR1 + b1 + x1
                p2 = dR2 + b2 + x2
                lim1 = above
                if whole == cutlass.Int32(1):
                    lim1 = above + m
                ewm = m1
                while ewm != cutlass.Int32(0):
                    esd = ffs_m1(ewm)
                    idx = (
                        ((base4 + tid) + (esd >> cutlass.Int32(2)) * cutlass.Int32(self.blk))
                        << cutlass.Int32(2)
                    ) + (esd & cutlass.Int32(3))
                    if p1 < lim1:
                        out_row[p1] = idx
                    p1 = p1 + cutlass.Int32(1)
                    ewm = ewm & (ewm - cutlass.Int32(1))
                if t1 == cutlass.Int32(1):
                    if p1 < lim1:
                        out_row[p1] = tix
                    p1 = p1 + cutlass.Int32(1)
                if m2 != cutlass.Int32(0):  # static-unrolled: striped DSMEM slabs
                    for s in cutlass.range_constexpr(S):
                        if (m2 & cutlass.Int32(1 << s)) != cutlass.Int32(0):
                            idx = (
                                (base4 + tid + cutlass.Int32((s // 4) * self.blk))
                                << cutlass.Int32(2)
                            ) + cutlass.Int32(s % 4)
                            rnk = p2 >> cutlass.Int32(LCMPC)
                            j = (p2 & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                            _st_shared_cluster_i32(
                                _mapa_shared_cluster_addr(ck_addr + j, rnk),
                                fkey(_val__regclus(frags, s)),
                            )
                            _st_shared_cluster_i32(_mapa_shared_cluster_addr(ci_addr + j, rnk), idx)
                            p2 = p2 + cutlass.Int32(1)
                if t2 == cutlass.Int32(1):
                    rnk = p2 >> cutlass.Int32(LCMPC)
                    j = (p2 & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                    _st_shared_cluster_i32(_mapa_shared_cluster_addr(ck_addr + j, rnk), fkey(tval))
                    _st_shared_cluster_i32(_mapa_shared_cluster_addr(ci_addr + j, rnk), tix)
                    p2 = p2 + cutlass.Int32(1)

            # ---- P8: release staging to rank 0
            cute.arch.barrier()
            _cluster_sync_aligned()

            if cutlass.const_expr(self.hint_free):
                # All cluster CTAs retain their register fragments for exact rescue.
                # Only complete float4 loads are resident; rank 0 owns the scalar tail.
                if degen != cutlass.Int32(0):
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    needC = k
                    aboveC = cutlass.Int32(0)
                    mm = n
                    done = cutlass.Int32(0)
                    tie_m = cutlass.Int32(1)
                    ethr = cutlass.Int64(0)
                    for level in cutlass.range_constexpr(4):
                        if done == cutlass.Int32(0):
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                aboveC = k
                                needC = cutlass.Int32(0)
                                tie_m = cutlass.Int32(0)
                                done = cutlass.Int32(1)

                        if done == cutlass.Int32(0):
                            sh2 = cutlass.Int32(0)
                            if cutlass.const_expr(level < 3):
                                sh2 = cutlass.Int32(22 - level * 10)
                            for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                            cute.arch.barrier()
                            for s in cutlass.range_constexpr(S):
                                idx = (
                                    (base4 + tid + cutlass.Int32((s // 4) * self.blk))
                                    << cutlass.Int32(2)
                                ) + cutlass.Int32(s % 4)
                                uk = fkey(_val__regclus(frags, s))
                                if (
                                    idx < (n4 << cutlass.Int32(2))
                                    and cutlass.Uint32(uk) >= cutlass.Uint32(rlo)
                                    and cutlass.Uint32(uk) <= cutlass.Uint32(rhi)
                                ):
                                    bn = (
                                        cutlass.Uint32(uk) - cutlass.Uint32(rlo)
                                    ) >> cutlass.Uint32(sh2)
                                    atomic_add_cta(
                                        s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1)
                                    )
                            if rank == cutlass.Int32(0) and tid < ntail:
                                uk = fkey(tval)
                                if cutlass.Uint32(uk) >= cutlass.Uint32(rlo) and cutlass.Uint32(
                                    uk
                                ) <= cutlass.Uint32(rhi):
                                    bn = (
                                        cutlass.Uint32(uk) - cutlass.Uint32(rlo)
                                    ) >> cutlass.Uint32(sh2)
                                    atomic_add_cta(
                                        s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1)
                                    )
                            _cluster_sync_aligned()
                            for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                i = tid + cutlass.Int32(z * self.blk)
                                tot_a = cutlass.Int32(0)
                                for r in cutlass.range_constexpr(CS):
                                    ma = _mapa_shared_cluster_addr(
                                        hist_addr + (i << cutlass.Int32(2)), cutlass.Int32(r)
                                    )
                                    tot_a = tot_a + _ld_shared_cluster_i32(ma)
                                s_mrg[i] = tot_a
                            cute.arch.barrier()
                            find_cross(s_mrg, needC, tid, s_res, nb=NB__regclus)
                            cute.arch.barrier()
                            aboveC = aboveC + s_res[RES_ABOVE]
                            needC = needC - s_res[RES_ABOVE]
                            mm = s_res[RES_M]
                            rlo = cutlass.Uint32(rlo) + (
                                cutlass.Uint32(s_res[RES_B]) << cutlass.Uint32(sh2)
                            )
                            rhi = cutlass.Uint32(rlo) + (
                                (cutlass.Uint32(1) << cutlass.Uint32(sh2)) - cutlass.Uint32(1)
                            )
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            _cluster_sync_aligned()
                    # Deterministic rescue: count full fragments separately from the
                    # tail owned by rank 0, whose logical indices follow every rank.
                    masks_a = cutlass.Int32(0)
                    masks_t = cutlass.Int32(0)
                    for s in cutlass.range_constexpr(S):
                        idx = (
                            (base4 + tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                        ) + cutlass.Int32(s % 4)
                        uk = fkey(_val__regclus(frags, s))
                        if idx < (n4 << cutlass.Int32(2)):
                            if cutlass.Int64(cutlass.Uint32(uk)) > ethr:
                                masks_a = masks_a | cutlass.Int32(1 << s)
                            if (
                                tie_m != cutlass.Int32(0)
                                and cutlass.Int64(cutlass.Uint32(uk)) == ethr
                            ):
                                masks_t = masks_t | cutlass.Int32(1 << s)
                    count_a = cutlass.Int32(warp_add_i32(popc(masks_a)))
                    count_t = cutlass.Int32(warp_add_i32(popc(masks_t)))
                    if lane == cutlass.Int32(0):
                        s_dw[dw] = count_a
                        s_dw[32 + dw] = count_t
                    cute.arch.barrier()
                    ca = cutlass.Int32(0)
                    ct = cutlass.Int32(0)
                    if lane < cutlass.Int32(NW):
                        ca = s_dw[lane]
                        ct = s_dw[32 + lane]
                    ca = cutlass.Int32(warp_add_i32(ca))
                    ct = cutlass.Int32(warp_add_i32(ct))
                    if tid == cutlass.Int32(0):
                        s_cnt[0] = ca
                        s_cnt[1] = ct
                    _cluster_sync_aligned()
                    rank_a = cutlass.Int32(0)
                    rank_t = cutlass.Int32(0)
                    full_a = cutlass.Int32(0)
                    full_t = cutlass.Int32(0)
                    for r in cutlass.range_constexpr(CS):
                        ma = _mapa_shared_cluster_addr(
                            sbase + cutlass.Int32(6 * 4), cutlass.Int32(r)
                        )
                        ca = _ld_shared_cluster_i32(ma)
                        ma = _mapa_shared_cluster_addr(
                            sbase + cutlass.Int32(7 * 4), cutlass.Int32(r)
                        )
                        ct = _ld_shared_cluster_i32(ma)
                        if cutlass.Int32(r) < rank:
                            rank_a = rank_a + ca
                            rank_t = rank_t + ct
                        full_a = full_a + ca
                        full_t = full_t + ct
                    _cluster_sync_aligned()
                    run_a = rank_a
                    run_t = rank_t
                    lml = cutlass.Int32(cute.arch.lanemask_lt())
                    # Order is rank, float4 fragment, thread, then component.
                    # Thus the first needC equal keys have the smallest indices.
                    for u in cutlass.range_constexpr(VPT):
                        bits_a = (masks_a >> cutlass.Int32(4 * u)) & cutlass.Int32(15)
                        bits_t = (masks_t >> cutlass.Int32(4 * u)) & cutlass.Int32(15)
                        ca = popc(bits_a)
                        ct = popc(bits_t)
                        wa = cutlass.Int32(warp_add_i32(ca))
                        wt = cutlass.Int32(warp_add_i32(ct))
                        xa = cutlass.Int32(0)
                        xt = cutlass.Int32(0)
                        for bit in cutlass.range_constexpr(3):
                            ba = ballot(
                                ((ca >> cutlass.Int32(bit)) & cutlass.Int32(1)) != cutlass.Int32(0)
                            )
                            bt = ballot(
                                ((ct >> cutlass.Int32(bit)) & cutlass.Int32(1)) != cutlass.Int32(0)
                            )
                            xa = xa + (popc(ba & lml) << cutlass.Int32(bit))
                            xt = xt + (popc(bt & lml) << cutlass.Int32(bit))
                        if lane == cutlass.Int32(0):
                            s_dw[dw] = wa
                            s_dw[32 + dw] = wt
                        cute.arch.barrier()
                        va = cutlass.Int32(0)
                        vt = cutlass.Int32(0)
                        if lane < cutlass.Int32(NW):
                            va = s_dw[lane]
                            vt = s_dw[32 + lane]
                        total_a = cutlass.Int32(warp_add_i32(va))
                        total_t = cutlass.Int32(warp_add_i32(vt))
                        if lane >= dw:
                            va = cutlass.Int32(0)
                            vt = cutlass.Int32(0)
                        pa = run_a + cutlass.Int32(warp_add_i32(va)) + xa
                        pt = run_t + cutlass.Int32(warp_add_i32(vt)) + xt
                        cute.arch.barrier()
                        for z in cutlass.range_constexpr(4):
                            idx = (
                                (base4 + tid + cutlass.Int32(u * self.blk)) << cutlass.Int32(2)
                            ) + cutlass.Int32(z)
                            if (bits_a & cutlass.Int32(1 << z)) != cutlass.Int32(0):
                                if pa < aboveC:
                                    out_row[pa] = idx
                                pa = pa + cutlass.Int32(1)
                            if (bits_t & cutlass.Int32(1 << z)) != cutlass.Int32(0):
                                if pt < needC:
                                    out_row[aboveC + pt] = idx
                                pt = pt + cutlass.Int32(1)
                        run_a = run_a + total_a
                        run_t = run_t + total_t
                    if rank == cutlass.Int32(0):
                        uk = fkey(tval)
                        qa = cutlass.Int32(0)
                        qt = cutlass.Int32(0)
                        if tid < ntail:
                            qa = cutlass.Int32(cutlass.Int64(cutlass.Uint32(uk)) > ethr)
                            qt = cutlass.Int32(
                                tie_m != cutlass.Int32(0)
                                and cutlass.Int64(cutlass.Uint32(uk)) == ethr
                            )
                        ba = ballot(qa != cutlass.Int32(0))
                        bt = ballot(qt != cutlass.Int32(0))
                        pa = full_a + popc(ba & lml)
                        pt = full_t + popc(bt & lml)
                        if qa != cutlass.Int32(0) and pa < aboveC:
                            out_row[pa] = tix
                        if qt != cutlass.Int32(0) and pt < needC:
                            out_row[aboveC + pt] = tix

            # ---- P9: rank-0 selection
            if rank == cutlass.Int32(0) and (
                degen == cutlass.Int32(0) or cutlass.const_expr(not self.hint_free)
            ):
                if whole == cutlass.Int32(0):
                    mc = m
                    if degen == cutlass.Int32(1):
                        mc = cutlass.Int32(0)
                    if degen == cutlass.Int32(0):
                        if mc <= cutlass.Int32(QUADC__regclus):
                            # (1) quad-96: all candidates LOCAL (96 < CMPC),
                            # O(mc^2) slot-order tie-broken rank
                            i = tid
                            while i < mc:
                                uq = cutlass.Uint32(s_ck[i])
                                uci = s_ci[i]  # DET: logical-index tie-break
                                rnk = cutlass.Int32(0)
                                j = cutlass.Int32(0)
                                while j < mc:
                                    vq = cutlass.Uint32(s_ck[j])
                                    tinc = cutlass.Int32(0)
                                    if vq > uq:
                                        tinc = cutlass.Int32(1)
                                    if vq == uq:
                                        if s_ci[j] < uci:
                                            tinc = cutlass.Int32(1)
                                    rnk = rnk + tinc
                                    j = j + cutlass.Int32(1)
                                if rnk < need:
                                    out_row[above + rnk] = uci
                                i = i + cutlass.Int32(BLK)
                        else:
                            if tid == cutlass.Int32(0):
                                s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                                s_kmm[1] = cutlass.Uint32(0)
                            cute.arch.barrier()  # kmm init
                            i = tid
                            while i < mc:
                                kv = cutlass.Uint32(
                                    _ld_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(
                                            ck_addr
                                            + ((i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)),
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
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            aboveC = cutlass.Int32(0)
                            needC = need
                            mm = mc
                            lev = cutlass.Int32(0)
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
                                if done == cutlass.Int32(0):
                                    d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                    b2w = cutlass.Int32(32) - clz_i32(
                                        cutlass.Int32(d2 | cutlass.Uint32(1))
                                    )
                                    sh2 = cutlass.Int32(0)
                                    if b2w > cutlass.Int32(LNB):
                                        sh2 = b2w - cutlass.Int32(LNB)
                                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                        s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                    cute.arch.barrier()  # level clear
                                    i = tid
                                    while i < mc:
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
                            cute.arch.barrier()  # narrowing done
                            # two-predicate ballot emit over the striped slabs
                            lml = cutlass.Int32(cute.arch.lanemask_lt())
                            it2 = (mc + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                            run1 = cutlass.Int32(0)  # DET SLOTS
                            run2 = cutlass.Int32(0)
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
                                    det_bw = cutlass.Int32(32) - clz_i32(
                                        cutlass.Int32(det_d | cutlass.Uint32(1))
                                    )
                                    det_s = det_bw - cutlass.Int32(LNB)
                                    if det_s < cutlass.Int32(0):
                                        det_s = cutlass.Int32(0)
                                    det_sh = cutlass.Uint32(det_s)
                                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                        s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                    cute.arch.barrier()  # ---- barrier (det level clear) ----
                                it = cutlass.Int32(0)
                                while it < it2:
                                    i = it * cutlass.Int32(BLK) + tid
                                    uke = cutlass.Uint32(0)
                                    idv = cutlass.Int32(0)
                                    if i < mc:  # predicated remote
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
                                            det_u = cutlass.Uint32(0xFFFFFFFF) - cutlass.Uint32(idv)
                                            if det_ph == cutlass.Int32(0):
                                                if cutlass.Uint32(det_u) >= cutlass.Uint32(det_lo):
                                                    if cutlass.Uint32(det_u) <= cutlass.Uint32(
                                                        det_hi
                                                    ):
                                                        det_b = (
                                                            cutlass.Uint32(det_u)
                                                            - cutlass.Uint32(det_lo)
                                                        ) >> cutlass.Uint32(det_sh)
                                                        if det_b > cutlass.Uint32(NB__regclus - 1):
                                                            det_b = cutlass.Uint32(NB__regclus - 1)
                                                        atomic_add_cta(
                                                            s_hist.iterator + cutlass.Int32(det_b),
                                                            cutlass.Int32(1),
                                                        )
                                            else:
                                                if cutlass.Int64(cutlass.Uint32(det_u)) > det_thr:
                                                    q2f = cutlass.Int32(1)
                                    if det_ph == cutlass.Int32(1):
                                        n1 = ballot(q1f == cutlass.Int32(1))
                                        n2 = ballot(q2f == cutlass.Int32(1))
                                        if lane == cutlass.Int32(0):
                                            s_dw[dw] = popc(n1)
                                            s_dw[32 + dw] = popc(n2)
                                        cute.arch.barrier()  # DET SLOTS: warp totals published
                                        v1 = cutlass.Int32(0)
                                        v2 = cutlass.Int32(0)
                                        if lane < cutlass.Int32(NW):
                                            v1 = s_dw[lane]
                                            v2 = s_dw[32 + lane]
                                        tot1 = cutlass.Int32(warp_add_i32(v1))
                                        tot2 = cutlass.Int32(warp_add_i32(v2))
                                        if lane >= dw:
                                            v1 = cutlass.Int32(0)
                                            v2 = cutlass.Int32(0)
                                        b1 = cutlass.Int32(warp_add_i32(v1))
                                        b2 = cutlass.Int32(warp_add_i32(v2))
                                        b1 = run1 + b1
                                        b2 = run2 + b2
                                        run1 = run1 + tot1
                                        run2 = run2 + tot2
                                        if it + cutlass.Int32(1) < it2:
                                            cute.arch.barrier()  # DET SLOTS: scratch reusable by the next it
                                        p1e = b1 + popc(n1 & lml)
                                        p2e = b2 + popc(n2 & lml)
                                        if q1f == cutlass.Int32(1):
                                            if p1e < aboveC:
                                                out_row[above + p1e] = idv
                                        if q2f == cutlass.Int32(1):
                                            if p2e < needC:
                                                out_row[above + aboveC + p2e] = idv
                                    it = it + cutlass.Int32(1)
                                if det_ph == cutlass.Int32(0):
                                    cute.arch.barrier()  # ---- barrier (det level hist) ----
                                    find_cross(s_hist, det_need, tid, s_res, nb=NB__regclus)
                                    cute.arch.barrier()  # ---- barrier (det level scan) ----
                                    det_need = det_need - s_res[RES_ABOVE]
                                    det_m = s_res[RES_M]
                                    det_bin = s_res[RES_B]
                                    det_nlo = cutlass.Uint32(det_lo) + (
                                        cutlass.Uint32(det_bin) << cutlass.Uint32(det_sh)
                                    )
                                    if det_bin != cutlass.Int32(NB__regclus - 1):
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
                        rlo = cutlass.Uint32(0)
                        rhi = cutlass.Uint32(0xFFFFFFFF)
                        aboveC = cutlass.Int32(0)  # above2
                        needC = k  # need2
                        mm = n  # m2
                        ethr = cutlass.Int64(0)
                        tie_m = cutlass.Int32(1)
                        lev = cutlass.Int32(0)
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
                            if done == cutlass.Int32(0):
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2w = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = cutlass.Int32(0)
                                if b2w > cutlass.Int32(LNB):
                                    sh2 = b2w - cutlass.Int32(LNB)
                                for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                    s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                cute.arch.barrier()  # level clear
                                i = tid
                                while i < n:  # whole-row bin
                                    unar = fkey(ldg_f32(x_addr, i))
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
                        run1 = cutlass.Int32(0)  # DET SLOTS
                        run2 = cutlass.Int32(0)
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
                        det_direct = cutlass.Int32(
                            0
                        )  # DET SLOTS: all positions tie -> emit [0, L) directly
                        if det_ph == cutlass.Int32(0):
                            if mm == n:  # every valid position ties: take the first L positions
                                if nA == cutlass.Int32(0):
                                    det_direct = cutlass.Int32(1)
                                det_thr = cutlass.Int64(
                                    cutlass.Uint32(0xFFFFFFFF)
                                    - cutlass.Uint32(cutlass.Int32(0) + needC - cutlass.Int32(1))
                                ) - cutlass.Int64(1)
                                det_ph = cutlass.Int32(1)
                        if det_direct == cutlass.Int32(0):
                            while det_ph <= cutlass.Int32(1):
                                if det_ph == cutlass.Int32(0):
                                    det_d = cutlass.Uint32(det_hi) - cutlass.Uint32(det_lo)
                                    det_bw = cutlass.Int32(32) - clz_i32(
                                        cutlass.Int32(det_d | cutlass.Uint32(1))
                                    )
                                    det_s = det_bw - cutlass.Int32(LNB)
                                    if det_s < cutlass.Int32(0):
                                        det_s = cutlass.Int32(0)
                                    det_sh = cutlass.Uint32(det_s)
                                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                        s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                    cute.arch.barrier()  # ---- barrier (det level clear) ----
                                it = cutlass.Int32(0)
                                while it < it2:
                                    i = it * cutlass.Int32(BLK) + tid
                                    uke = cutlass.Uint32(0)
                                    if i < n:
                                        uke = fkey(ldg_f32(x_addr, i))
                                    q1f = cutlass.Int32(0)
                                    q2f = cutlass.Int32(0)
                                    if i < n:
                                        if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                            q1f = cutlass.Int32(1)
                                        if tie_m == cutlass.Int32(1):
                                            if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
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
                                                            if det_b > cutlass.Uint32(
                                                                NB__regclus - 1
                                                            ):
                                                                det_b = cutlass.Uint32(
                                                                    NB__regclus - 1
                                                                )
                                                            atomic_add_cta(
                                                                s_hist.iterator
                                                                + cutlass.Int32(det_b),
                                                                cutlass.Int32(1),
                                                            )
                                                else:
                                                    if (
                                                        cutlass.Int64(cutlass.Uint32(det_u))
                                                        > det_thr
                                                    ):
                                                        q2f = cutlass.Int32(1)
                                    if det_ph == cutlass.Int32(1):
                                        n1 = ballot(q1f == cutlass.Int32(1))
                                        n2 = ballot(q2f == cutlass.Int32(1))
                                        if lane == cutlass.Int32(0):
                                            s_dw[dw] = popc(n1)
                                            s_dw[32 + dw] = popc(n2)
                                        cute.arch.barrier()  # DET SLOTS: warp totals published
                                        v1 = cutlass.Int32(0)
                                        v2 = cutlass.Int32(0)
                                        if lane < cutlass.Int32(NW):
                                            v1 = s_dw[lane]
                                            v2 = s_dw[32 + lane]
                                        tot1 = cutlass.Int32(warp_add_i32(v1))
                                        tot2 = cutlass.Int32(warp_add_i32(v2))
                                        if lane >= dw:
                                            v1 = cutlass.Int32(0)
                                            v2 = cutlass.Int32(0)
                                        b1 = cutlass.Int32(warp_add_i32(v1))
                                        b2 = cutlass.Int32(warp_add_i32(v2))
                                        b1 = run1 + b1
                                        b2 = run2 + b2
                                        run1 = run1 + tot1
                                        run2 = run2 + tot2
                                        if it + cutlass.Int32(1) < it2:
                                            cute.arch.barrier()  # DET SLOTS: scratch reusable by the next it
                                        p1e = b1 + popc(n1 & lml)
                                        p2e = b2 + popc(n2 & lml)
                                        if q1f == cutlass.Int32(1):
                                            if p1e < nA:
                                                out_row[p1e] = i
                                        if q2f == cutlass.Int32(1):
                                            if p2e < nT:
                                                out_row[nA + p2e] = i
                                    it = it + cutlass.Int32(1)
                                if det_ph == cutlass.Int32(0):
                                    cute.arch.barrier()  # ---- barrier (det level hist) ----
                                    find_cross(s_hist, det_need, tid, s_res, nb=NB__regclus)
                                    cute.arch.barrier()  # ---- barrier (det level scan) ----
                                    det_need = det_need - s_res[RES_ABOVE]
                                    det_m = s_res[RES_M]
                                    det_bin = s_res[RES_B]
                                    det_nlo = cutlass.Uint32(det_lo) + (
                                        cutlass.Uint32(det_bin) << cutlass.Uint32(det_sh)
                                    )
                                    if det_bin != cutlass.Int32(NB__regclus - 1):
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
                            i = tid
                            while i < nT:
                                out_row[nA + i] = i
                                i = i + cutlass.Int32(BLK)

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
        )


# ---------------------------------------------------------------------------
# host wrapper: compile cache + route()-driven entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE__regclus: dict = {}


def get_compiled__regclus(
    tpl: tuple,
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
    hint_free: bool = False,
) -> Any:
    """Compile (or fetch) the variant for constexpr tuple (BLK, VPT, CS, K)."""
    key = (
        tuple(tpl),
        bool(varlen),
        int(next_n),
        int(cr_shift),
        bool(hint_free),
    )
    compiled = _COMPILE_CACHE__regclus.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt

        kernel = GvrRegClusKernel(
            tpl,
            varlen=varlen,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=hint_free,
        )
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        lg_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16
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
        compiled = cute.compile(
            kernel,
            lg_fake,
            pi_fake,
            kv_fake,
            out_fake,
            cutlass.Int32(0),
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )
        _COMPILE_CACHE__regclus[key] = compiled
    return compiled
