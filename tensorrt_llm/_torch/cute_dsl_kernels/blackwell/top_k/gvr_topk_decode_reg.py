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

"""Register-resident (reg) GVR Top-K tier — CuTe DSL, Blackwell SM100.

CuTe DSL translation of the CUDA
``gvr_topk_reg<CS,TB,MAXV,AR>`` register-resident GVR top-K kernel
(tuned CUDA head), adapted for the production
``trtllm::cute_dsl_gvr_topk_decode`` contract (see the module docstring of
``gvr_topk_decode_tp`` for the shared adaptation inventory: ragged-N
masking, pre_idx clamping, per-row degenerate identity emit).

The row is loaded ONCE from GMEM into per-thread registers (MAXV float4s per
thread; out-of-slice lanes AND lanes beyond the row's N_eff are filled
-FLT_MAX — the ragged-N adaptation extends the existing OOR-lane idiom, so a
single mask at load time covers every downstream count / max-below / plateau
emit / collect over the register array). All subsequent passes are pure ALU —
global traffic pinned at 4*npad bytes/row.

Phase skeleton (shared with the tp tier — helpers REUSED from
``gvr_topk_decode_tp``):
  P1 : hint gather + two-stage 64-bin histogram -> rung ladder (tp.phase1)
  P2 : full AR-rung register count + cluster exchange, secant refine /
       max-below plateau descent                              (count_reg here)
  P3 : rank-scatter collect (NO atomics): cached per-thread counts -> warp
       inclusive scan + block scan + cluster-peer prefix -> deterministic
       scatter position; keys f2u'd at push; P4's round-0 radix histogram is
       built IN FLIGHT with LOCAL smem atomics, non-rank0 CTAs merge their
       histograms into CTA0 with <=256 remote atomics.
  P4 : CTA0-solo radix select starting directly at bin-select for round 0,
       then tie-aware ticketed emit.

``__launch_bounds__(TB, 1)`` -> ``.launch(min_blocks_per_mp=1)``.
CS=16 relies on CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, which the
CuTe DSL sets unconditionally on every kernel — the dispatcher additionally
verifies the queried hardware max cluster size and falls back to the in-tree
kernel when the tier's CS exceeds it (never degrades silently).

CRITICAL: CuTe DSL cluster kernels need EXPLICIT trailing cluster
rendezvous where remote (DSMEM) ops can still be in flight — nvcc inserts an
implicit cluster barrier before ret in DSMEM kernels, the DSL does not
(missing it => timing-dependent CUDA 719 faults).

Convergence notes preserved in this port (do not "simplify away"):
1. ``cutlass.utils.distributed.atomicAdd`` lowers to sys-scoped
   ``atom.relaxed.SYS.shared`` which blocks ptxas' warp aggregation; the
   CTA-scope relaxed ``_atomic_add_cta`` restores the fast path.
2. ``cute.arch.cluster_arrive/wait`` emit the NON-aligned barrier forms;
   ``cg::cluster.sync()`` parity needs the ALIGNED pair
   (``_cluster_sync_aligned``).
3. Candidates are a single packed (key<<32|idx) u64 array — one 8B DSMEM
   push per candidate (halves remote-store transactions at CS>1).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import nvvm
from cutlass.cute import runtime as _crt
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from .gvr_topk_decode_tp import (
    FLT_MAX,
    INF,
    MAXPASS,
    RUNGS,
    GvrTpKernel,
    _atom_shared_cluster_add_i32,
    _exp2i,
    _f2u_bits,
    _f32_bits_u32,
    _ld_shared_cluster_f32,
    _ld_shared_cluster_i32,
    _mapa_shared_cluster,
    _shfl_down_add,
    _shfl_up_add,
    _st_shared_cluster_u64,
)


@dsl_user_op
def _atomic_add_cta(dst_ptr, val, *, loc=None, ip=None):
    """CTA-scope relaxed atomicAdd on shared memory; returns the old value.
    Drop-in for cutlass.utils.distributed.atomicAdd (which is sys-scoped and
    blocks ptxas warp-aggregation — see module docstring note 1)."""
    return cute.arch.atomic_add(
        ptr=dst_ptr.llvm_ptr, val=val, sem="relaxed", scope="cta", loc=loc, ip=ip
    )


@dsl_user_op
def _cluster_arrive_aligned(*, loc=None, ip=None):
    nvvm.cluster_arrive(aligned=True, loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait_aligned(*, loc=None, ip=None):
    nvvm.cluster_wait(aligned=True, loc=loc, ip=ip)


@cute.jit
def _cluster_sync_aligned():
    _cluster_arrive_aligned()
    _cluster_wait_aligned()


class GvrRegKernel(GvrTpKernel):
    """CuTe DSL port of gvr_topk_reg<CS, TB, MAXV, AR> (fp32, B200/B300).

    Inherits phase1 (already N_eff-aware), the DSMEM/scan idioms and
    ``_row_n_eff`` from the tp port (identical CUDA source); adds
    register-resident count/max-below/collect and the rank-scatter +
    in-flight-histogram P3/P4.
    """

    def __init__(
        self,
        top_k: int,
        kC: int,
        cluster_size: int = 1,
        ar: int = RUNGS,
        maxv: int = 8,
        num_threads: int = 512,
        next_n: int = 1,
        compress_ratio: int = 4,
    ):
        assert num_threads % 32 == 0
        assert ar in (6, 8)
        assert cluster_size in (1, 2, 4, 8, 16)
        assert 1 <= maxv <= 8
        self.top_k = top_k
        self.kC = kC
        self.cluster_size = cluster_size
        self.ar = ar
        self.uf = 4  # unused by the reg path; kept for inherited helpers
        self.maxv = maxv
        self.num_threads = num_threads
        self.num_warps = num_threads // 32
        self.next_n = next_n
        self.compress_ratio = compress_ratio

    # ------------------------------------------------------------------
    # count_reg<TB, R, MAXV>: R-rung count over the register array.
    # Per-thread counts -> s_ptcnt[r*TB + tid] (P3 rank-scatter offsets).
    # Dummy (-FLT_MAX) lanes — both out-of-slice AND ragged-N-masked —
    # only pass a rung == -FLT_MAX, which can never be `chosen` (its count
    # > kC by construction since npad > kC).
    # ------------------------------------------------------------------
    @cute.jit
    def count_reg(self, R: cutlass.Constexpr, a, tidx, s_rungs, s_ptcnt):
        TB = cutlass.const_expr(self.num_threads)
        MAXV = cutlass.const_expr(self.maxv)
        tr = cute.make_rmem_tensor((R,), cutlass.Float32)
        cnt = cute.make_rmem_tensor((R,), cutlass.Int32)
        for r in cutlass.range_constexpr(R):
            tr[r] = s_rungs[r]
            cnt[r] = cutlass.Int32(0)
        for u in cutlass.range_constexpr(MAXV):
            for q in cutlass.range_constexpr(4):
                v = a[4 * u + q]
                for r in cutlass.range_constexpr(R):
                    cnt[r] = cnt[r] + cutlass.Int32(v >= tr[r])
        for r in cutlass.range_constexpr(R):
            s_ptcnt[r * TB + tidx] = cnt[r]

    # ------------------------------------------------------------------
    # exchange_counts override: identical to the tp tier's, but the cluster
    # sync is the ALIGNED barrier pair (cg::cluster.sync() parity — module
    # docstring note 2). The tp tier keeps the non-aligned form it was
    # validated with.
    # ------------------------------------------------------------------
    @cute.jit
    def exchange_counts(
        self, R: cutlass.Constexpr, par, tidx, rank, s_ptcnt, s_rcnt, s_rpre, s_ipartial
    ):
        TB = cutlass.const_expr(self.num_threads)
        CS = cutlass.const_expr(self.cluster_size)
        cute.arch.barrier()  # ptcnt final
        lane = tidx & cutlass.Int32(31)
        wid = tidx >> cutlass.Int32(5)
        if wid < cutlass.Int32(R):
            s = cutlass.Int32(0)
            for k in cutlass.range_constexpr(TB // 32):
                s = s + s_ptcnt[wid * TB + lane + cutlass.Int32(32 * k)]
            s = cute.arch.warp_redux_sync(s, "add")
            if cutlass.const_expr(CS == 1):
                if lane == cutlass.Int32(0):
                    s_rcnt[wid] = s
                    s_rpre[wid] = cutlass.Int32(0)
            else:
                if lane == cutlass.Int32(0):
                    s_ipartial[par * cutlass.Int32(RUNGS) + wid] = s
        if cutlass.const_expr(CS == 1):
            cute.arch.barrier()
        else:
            _cluster_sync_aligned()
            if tidx < cutlass.Int32(R):
                tot = cutlass.Int32(0)
                pre = cutlass.Int32(0)
                local_ptr = s_ipartial.iterator + (par * cutlass.Int32(RUNGS) + tidx)
                for rr in cutlass.range_constexpr(CS):
                    a = _mapa_shared_cluster(local_ptr, cutlass.Int32(rr))
                    v = _ld_shared_cluster_i32(a)
                    tot = tot + v
                    if cutlass.Int32(rr) < rank:
                        pre = pre + v
                s_rcnt[tidx] = tot
                s_rpre[tidx] = pre
            cute.arch.barrier()

    # ------------------------------------------------------------------
    # max_below_reg<TB, CS, MAXV>: largest value strictly below t_hi_bound across the
    # whole row (cluster-reduced). No index gate needed: dummy lanes hold
    # -FLT_MAX which never raises the max.
    # ------------------------------------------------------------------
    @cute.jit
    def max_below_reg(self, a, t_hi_bound, par, tidx, s_fwred, s_fpartial):
        CS = cutlass.const_expr(self.cluster_size)
        MAXV = cutlass.const_expr(self.maxv)
        NWARP = cutlass.const_expr(self.num_warps)
        m = cutlass.Float32(-FLT_MAX)
        for u in cutlass.range_constexpr(MAXV):
            for q in cutlass.range_constexpr(4):
                v = a[4 * u + q]
                if v < t_hi_bound:
                    m = cute.arch.fmax(m, v)
        m = cute.arch.warp_redux_sync(m, "fmax")
        lane = tidx & cutlass.Int32(31)
        wid = tidx >> cutlass.Int32(5)
        if lane == cutlass.Int32(0):
            s_fwred[wid] = m
        cute.arch.barrier()
        if tidx == cutlass.Int32(0):
            mm = cutlass.Float32(-FLT_MAX)
            for w in cutlass.range_constexpr(NWARP):
                mm = cute.arch.fmax(mm, s_fwred[w])
            s_fpartial[par] = mm
        res = cutlass.Float32(-FLT_MAX)
        if cutlass.const_expr(CS == 1):
            cute.arch.barrier()
            res = s_fpartial[par]
        else:
            _cluster_sync_aligned()
            local_ptr = s_fpartial.iterator + par
            for rr in cutlass.range_constexpr(CS):
                ad = _mapa_shared_cluster(local_ptr, cutlass.Int32(rr))
                res = cute.arch.fmax(res, _ld_shared_cluster_f32(ad))
        return res

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def gvr_reg_kernel(
        self, logits: cute.Tensor, pre_idx: cute.Tensor, seq_lens: cute.Tensor, out_idx: cute.Tensor
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        TB = cutlass.const_expr(self.num_threads)
        CS = cutlass.const_expr(self.cluster_size)
        AR = cutlass.const_expr(self.ar)
        MAXV = cutlass.const_expr(self.maxv)
        K = cutlass.const_expr(self.top_k)
        kC = cutlass.const_expr(self.kC)
        NWARP = cutlass.const_expr(self.num_warps)

        if cutlass.const_expr(CS > 1):
            row = bidx // cutlass.Int32(CS)
            rank = cute.arch.block_idx_in_cluster()
        else:
            row = bidx
            rank = cutlass.Int32(0)

        npad = cutlass.Int32(logits.shape[1])
        logits_row = logits[row, None]
        # Hint sharing: pre_idx is request-level ([num_rows // next_n, K]) —
        # see the tp tier for the in-tree run_one_row parity notes.
        if cutlass.const_expr(self.next_n == 1):
            pre_idx_row = pre_idx[row, None]
        else:
            pre_idx_row = pre_idx[row // cutlass.Int32(self.next_n), None]
        out_row = out_idx[row, None]
        row_addr = logits_row.iterator.toint()

        # Ragged N: per-row valid length from seq_lens (inherited helper).
        n_eff = self._row_n_eff(seq_lens, row)

        # ---- shared memory (order must be identical across CTAs for mapa) ----
        smem = SmemAllocator()
        # Single packed (key<<32 | idx) u64 candidate array — CUDA's
        # `unsigned long long cand[kC]`; one 8B DSMEM push per candidate in
        # the P3 scatter (halves remote-store transactions under 8-cluster
        # contention at BS>=8 — measured).
        s_cand = smem.allocate_tensor(
            element_type=cutlass.Uint64,
            layout=cute.make_ordered_layout((kC,), order=(0,)),
            byte_alignment=128,
        )
        s_ptcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((RUNGS * self.num_threads,), order=(0,)),
            byte_alignment=128,
        )
        s_hist = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((256,), order=(0,)),
            byte_alignment=128,
        )
        s_rungs = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_rcnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_rpre = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_ipartial = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2 * RUNGS,), order=(0,)),
            byte_alignment=32,
        )
        s_fpartial = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=16,
        )
        s_fwred = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((2 * self.num_warps,), order=(0,)),
            byte_alignment=64,
        )
        s_hminmax = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=16,
        )
        # iscalars: [0]=sel_bin [1]=sel_above [2]=sel_count [3]=cnt_m [4]=cnt_t
        s_isc = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=32,
        )
        # P3 rank-scatter block-scan scratch (reg path only)
        s_iwred = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.num_warps,), order=(0,)),
            byte_alignment=64,
        )

        # ---- Degenerate rows (N_eff <= K): identity emit + -1 pad. ----
        # Cluster-uniform branch (all CTAs own the same row); no DSMEM op is
        # issued on this path, so no exit rendezvous is required either.
        if n_eff <= cutlass.Int32(K):
            if rank == cutlass.Int32(0):
                jd = cutlass.Int32(tidx)
                while jd < n_eff:
                    out_row[jd] = jd
                    jd = jd + cutlass.Int32(TB)
                jp = n_eff + cutlass.Int32(tidx)
                if jp < cutlass.Int32(0):
                    jp = cutlass.Int32(tidx)  # n_eff < 0 (defensive): pad all
                while jp < cutlass.Int32(K):
                    out_row[jp] = cutlass.Int32(-1)
                    jp = jp + cutlass.Int32(TB)
        else:
            # ---- slice + one-time register row load (launcher: vpc <= MAXV*TB)
            V4 = npad >> cutlass.Int32(2)
            vpc = (V4 + cutlass.Int32(CS) - cutlass.Int32(1)) // cutlass.Int32(CS)
            v0 = rank * vpc
            v1 = v0 + vpc
            if v1 > V4:
                v1 = V4

            copy_atom = self._copy_atom()
            a = cute.make_rmem_tensor((MAXV * 4,), cutlass.Float32)
            frag4 = cute.make_rmem_tensor((4,), cutlass.Float32)
            for u in cutlass.range_constexpr(MAXV):
                i = v0 + tidx + cutlass.Int32(u * TB)
                if i < v1:
                    self._ld_float4(copy_atom, row_addr, i, frag4)
                    gi = i << cutlass.Int32(2)
                    for q in cutlass.range_constexpr(4):
                        # Ragged N: extend the OOR-lane idiom to the valid
                        # length — elements at global index >= N_eff become
                        # -FLT_MAX dummy lanes, exactly like out-of-slice
                        # lanes. One mask here covers ALL downstream register
                        # passes (count/max-below/plateau emit/P3 collect).
                        val = frag4[q]
                        if gi + cutlass.Int32(q) >= n_eff:
                            val = cutlass.Float32(-FLT_MAX)
                        a[4 * u + q] = val
                else:
                    for q in cutlass.range_constexpr(4):
                        a[4 * u + q] = cutlass.Float32(-FLT_MAX)

            xch = cutlass.Int32(0)
            thr = cutlass.Float32(0.0)
            chosen = cutlass.Int32(-1)
            C = cutlass.Int32(0)
            cbase = cutlass.Int32(0)
            m_gt = cutlass.Int32(-1)

            # cr==1 hint temporal shift (in-tree pre_idx_offset parity); the
            # cr>1 branch keeps the exact pre-MTP trace — see the tp tier.
            if cutlass.const_expr(self.compress_ratio == 1):
                hint_off = (row % cutlass.Int32(self.next_n)) + cutlass.Int32(1)
                self.phase1(
                    logits_row,
                    pre_idx_row,
                    n_eff,
                    tidx,
                    s_hist,
                    s_fwred,
                    s_hminmax,
                    s_rungs,
                    hint_off=hint_off,
                )
            else:
                self.phase1(
                    logits_row, pre_idx_row, n_eff, tidx, s_hist, s_fwred, s_hminmax, s_rungs
                )
            # Pre-zero the P4 round-0 radix histogram now that P1 is done with
            # hist. The first exchange's cluster barrier (release) publishes it
            # before any CTA reaches the P3 collect that increments it in flight.
            if tidx < cutlass.Int32(256):
                s_hist[tidx] = cutlass.Int32(0)
            self.count_reg(AR, a, tidx, s_rungs, s_ptcnt)
            self.exchange_counts(
                AR, xch & cutlass.Int32(1), tidx, rank, s_ptcnt, s_rcnt, s_rpre, s_ipartial
            )
            xch = xch + cutlass.Int32(1)

            span0 = cute.arch.fmax(s_hminmax[1] - s_hminmax[0], cutlass.Float32(1e-3))

            # ---- P2: secant refine driver (redundant on every thread) ----
            t_lo = cutlass.Float32(-FLT_MAX)
            t_hi = cutlass.Float32(INF)
            c_hi = cutlass.Int32(0)
            Rcur = cutlass.Int32(AR)
            passno = cutlass.Int32(0)
            running = cutlass.Int32(1)
            descend_break = cutlass.Int32(0)
            while running != cutlass.Int32(0):
                # first rung index j with rcnt[j] >= K (rcnt ascending in j)
                j = Rcur
                for r_ in range(AR - 1, -1, -1):
                    if cutlass.Int32(r_) < Rcur:
                        if s_rcnt[r_] >= cutlass.Int32(K):
                            j = cutlass.Int32(r_)
                jj = j
                if jj > Rcur - cutlass.Int32(1):
                    jj = Rcur - cutlass.Int32(1)
                cj = s_rcnt[jj]
                rj = s_rungs[jj]
                bj = s_rpre[jj]
                found = cutlass.Int32(0)
                if j < Rcur:
                    if cj <= cutlass.Int32(kC):
                        found = cutlass.Int32(1)
                if found != cutlass.Int32(0):
                    chosen = j
                    thr = rj
                    C = cj
                    cbase = bj
                    running = cutlass.Int32(0)
                else:
                    if j < Rcur:
                        if rj >= t_lo:
                            t_lo = rj
                    if j > cutlass.Int32(0):
                        jm = j - cutlass.Int32(1)
                        if s_rungs[jm] <= t_hi:
                            t_hi = s_rungs[jm]
                            c_hi = s_rcnt[jm]
                    descend = cutlass.Int32(0)
                    if passno >= cutlass.Int32(MAXPASS):
                        descend = cutlass.Int32(1)
                    e3 = passno * cutlass.Int32(3)
                    if e3 > cutlass.Int32(24):
                        e3 = cutlass.Int32(24)
                    step = span0 * _exp2i(e3)
                    dt = cutlass.Float32(0.0)
                    mode = cutlass.Int32(2)  # 0=up-ladder, 1=down-ladder, 2=secant
                    if t_hi == cutlass.Float32(INF):
                        mode = cutlass.Int32(0)
                        nr0 = t_lo + step * cutlass.Float32(float(1 << (AR - 1)))
                        if nr0 == cutlass.Float32(INF):
                            descend = cutlass.Int32(1)
                    else:
                        if t_lo == cutlass.Float32(-FLT_MAX):
                            mode = cutlass.Int32(1)
                        else:
                            dt = (t_hi - t_lo) * cutlass.Float32(1.0 / float(AR + 1))
                            nr_last = t_hi - dt * cutlass.Float32(float(AR))
                            nr_first = t_hi - dt
                            ok = cutlass.Int32(0)
                            if nr_last > t_lo:
                                if nr_first < t_hi:
                                    ok = cutlass.Int32(1)
                            if ok == cutlass.Int32(0):
                                descend = cutlass.Int32(1)
                    if descend != cutlass.Int32(0):
                        running = cutlass.Int32(0)
                        descend_break = cutlass.Int32(1)
                    else:
                        cute.arch.barrier()
                        if tidx == cutlass.Int32(0):
                            for r_ in cutlass.range_constexpr(AR):
                                nrv = cutlass.Float32(0.0)
                                if mode == cutlass.Int32(0):
                                    nrv = t_lo + step * cutlass.Float32(float(1 << (AR - 1 - r_)))
                                else:
                                    if mode == cutlass.Int32(1):
                                        nrv = t_hi - step * cutlass.Float32(float(1 << r_))
                                    else:
                                        nrv = t_hi - dt * cutlass.Float32(float(r_ + 1))
                                s_rungs[r_] = nrv
                        cute.arch.barrier()
                        self.count_reg(AR, a, tidx, s_rungs, s_ptcnt)
                        self.exchange_counts(
                            AR,
                            xch & cutlass.Int32(1),
                            tidx,
                            rank,
                            s_ptcnt,
                            s_rcnt,
                            s_rpre,
                            s_ipartial,
                        )
                        xch = xch + cutlass.Int32(1)
                        passno = passno + cutlass.Int32(1)
                        Rcur = cutlass.Int32(AR)

            # ---- plateau descent (exact max-below stepping) ----
            if descend_break != cutlass.Int32(0):
                pl = cutlass.Int32(1)
                while pl != cutlass.Int32(0):
                    vstar = self.max_below_reg(
                        a, t_hi, xch & cutlass.Int32(1), tidx, s_fwred, s_fpartial
                    )
                    xch = xch + cutlass.Int32(1)
                    cute.arch.barrier()
                    if tidx == cutlass.Int32(0):
                        s_rungs[0] = vstar
                    cute.arch.barrier()
                    self.count_reg(1, a, tidx, s_rungs, s_ptcnt)
                    self.exchange_counts(
                        RUNGS,
                        xch & cutlass.Int32(1),
                        tidx,
                        rank,
                        s_ptcnt,
                        s_rcnt,
                        s_rpre,
                        s_ipartial,
                    )
                    xch = xch + cutlass.Int32(1)
                    c = s_rcnt[0]
                    okc = cutlass.Int32(0)
                    if c >= cutlass.Int32(K):
                        if c <= cutlass.Int32(kC):
                            okc = cutlass.Int32(1)
                    if okc != cutlass.Int32(0):
                        chosen = cutlass.Int32(0)
                        thr = vstar
                        C = c
                        cbase = s_rpre[0]
                        pl = cutlass.Int32(0)
                    else:
                        if c < cutlass.Int32(K):
                            t_hi = vstar
                            c_hi = c
                        else:
                            thr = vstar
                            m_gt = c_hi
                            pl = cutlass.Int32(0)

            if m_gt >= cutlass.Int32(0):
                # ---- plateau direct emit from registers ----
                if rank == cutlass.Int32(0):
                    if tidx == cutlass.Int32(0):
                        s_isc[3] = cutlass.Int32(0)  # cnt_m
                        s_isc[4] = cutlass.Int32(0)  # cnt_t
                if cutlass.const_expr(CS > 1):
                    _cluster_sync_aligned()
                    a_m = _mapa_shared_cluster(s_isc.iterator + cutlass.Int32(3), cutlass.Int32(0))
                    a_t = _mapa_shared_cluster(s_isc.iterator + cutlass.Int32(4), cutlass.Int32(0))
                else:
                    cute.arch.barrier()
                nt = cutlass.Int32(K) - m_gt
                for u in cutlass.range_constexpr(MAXV):
                    i = v0 + tidx + cutlass.Int32(u * TB)
                    if i < v1:
                        gi = i << cutlass.Int32(2)
                        for q in cutlass.range_constexpr(4):
                            v = a[4 * u + q]
                            if v > thr:
                                if cutlass.const_expr(CS > 1):
                                    p = _atom_shared_cluster_add_i32(a_m, cutlass.Int32(1))
                                else:
                                    p = _atomic_add_cta(
                                        s_isc.iterator + cutlass.Int32(3), cutlass.Int32(1)
                                    )
                                out_row[p] = gi + cutlass.Int32(q)
                            else:
                                if v == thr:
                                    if cutlass.const_expr(CS > 1):
                                        p = _atom_shared_cluster_add_i32(a_t, cutlass.Int32(1))
                                    else:
                                        p = _atomic_add_cta(
                                            s_isc.iterator + cutlass.Int32(4), cutlass.Int32(1)
                                        )
                                    if p < nt:
                                        out_row[m_gt + p] = gi + cutlass.Int32(q)
            else:
                # ---- P3: rank-scatter collect from registers (NO atomics) ----
                myc = s_ptcnt[chosen * cutlass.Int32(TB) + tidx]
                lane = tidx & cutlass.Int32(31)
                wid = tidx >> cutlass.Int32(5)
                incl = myc
                for o in [1, 2, 4, 8, 16]:
                    incl = _shfl_up_add(incl, lane, o)
                if lane == cutlass.Int32(31):
                    s_iwred[wid] = incl
                cute.arch.barrier()
                if wid == cutlass.Int32(0):
                    v_ = cutlass.Int32(0)
                    if lane < cutlass.Int32(NWARP):
                        v_ = s_iwred[lane]
                    iv = v_
                    for o in [x for x in [1, 2, 4, 8, 16] if x < self.num_warps]:
                        iv = _shfl_up_add(iv, lane, o)
                    if lane < cutlass.Int32(NWARP):
                        s_iwred[lane] = iv - v_
                cute.arch.barrier()
                pos = cbase + s_iwred[wid] + (incl - myc)

                # scatter: f2u key at push + IN-FLIGHT local P4 round-0
                # histogram. thr > -FLT_MAX whenever we get here (chosen count
                # <= kC < npad), so dummy lanes never pass.
                if cutlass.const_expr(CS > 1):
                    a_cand = _mapa_shared_cluster(s_cand.iterator, cutlass.Int32(0))
                for u in cutlass.range_constexpr(MAXV):
                    gi = (v0 + tidx + cutlass.Int32(u * TB)) << cutlass.Int32(2)
                    for q in cutlass.range_constexpr(4):
                        v = a[4 * u + q]
                        if v >= thr:
                            ux = _f2u_bits(_f32_bits_u32(v))
                            _atomic_add_cta(
                                s_hist.iterator + cutlass.Int32(ux >> cutlass.Uint32(24)),
                                cutlass.Int32(1),
                            )
                            kv = (cutlass.Uint64(ux) << cutlass.Uint64(32)) | cutlass.Uint64(
                                cutlass.Uint32(gi + cutlass.Int32(q))
                            )
                            if cutlass.const_expr(CS > 1):
                                _st_shared_cluster_u64(a_cand + pos * cutlass.Int32(8), kv)
                            else:
                                s_cand[pos] = kv
                            pos = pos + cutlass.Int32(1)

                # non-rank0: merge local round-0 histogram into CTA0
                # (<=256 remote atomics)
                if cutlass.const_expr(CS > 1):
                    if rank != cutlass.Int32(0):
                        cute.arch.barrier()  # local hist final
                        a_hist = _mapa_shared_cluster(s_hist.iterator, cutlass.Int32(0))
                        b = cutlass.Int32(tidx)
                        while b < cutlass.Int32(256):
                            hv = s_hist[b]
                            if hv != cutlass.Int32(0):
                                _atom_shared_cluster_add_i32(a_hist + b * cutlass.Int32(4), hv)
                            b = b + cutlass.Int32(TB)
                    _cluster_sync_aligned()
                else:
                    cute.arch.barrier()

                # ---- P4 (CTA0 solo when CS>1) ----
                if rank == cutlass.Int32(0):
                    if C == cutlass.Int32(K):
                        ie = cutlass.Int32(tidx)
                        while ie < C:
                            out_row[ie] = cutlass.Int32(
                                cutlass.Uint32(s_cand[ie] & cutlass.Uint64(0xFFFFFFFF))
                            )
                            ie = ie + cutlass.Int32(TB)
                    else:
                        # 4x8-bit radix select; keys ALREADY f2u'd; round-0
                        # histogram already built in flight -> round 0 starts
                        # at bin-select.
                        pref = cutlass.Uint32(0)
                        want = cutlass.Int32(K)
                        m = cutlass.Int32(0)
                        final_shift = cutlass.Uint32(0)
                        active = cutlass.Int32(1)
                        for r_ in cutlass.range_constexpr(4):
                            shift = cutlass.const_expr(24 - 8 * r_)
                            if active != cutlass.Int32(0):
                                if cutlass.const_expr(r_ > 0):
                                    ih = cutlass.Int32(tidx)
                                    while ih < C:
                                        u = cutlass.Uint32(s_cand[ih] >> cutlass.Uint64(32))
                                        if (u >> cutlass.Uint32(shift + 8)) == pref:
                                            bb = cutlass.Int32(
                                                (u >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                                            )
                                            _atomic_add_cta(s_hist.iterator + bb, cutlass.Int32(1))
                                        ih = ih + cutlass.Int32(TB)
                                cute.arch.barrier()
                                if tidx < cutlass.Int32(32):
                                    b8 = tidx * cutlass.Int32(8)
                                    h = cute.make_rmem_tensor((8,), cutlass.Int32)
                                    Ssum = cutlass.Int32(0)
                                    for q in cutlass.range_constexpr(8):
                                        h[q] = s_hist[b8 + cutlass.Int32(q)]
                                        Ssum = Ssum + h[q]
                                        s_hist[b8 + cutlass.Int32(q)] = cutlass.Int32(0)
                                    x = Ssum
                                    for o in [1, 2, 4, 8, 16]:
                                        x = _shfl_down_add(x, tidx, o)
                                    A = x - Ssum
                                    if A < want:
                                        if want <= A + Ssum:
                                            run = A
                                            for q in range(7, -1, -1):
                                                if run < want:
                                                    if want <= run + h[q]:
                                                        s_isc[0] = b8 + cutlass.Int32(q)
                                                        s_isc[1] = run
                                                        s_isc[2] = h[q]
                                                run = run + h[q]
                                cute.arch.barrier()
                                sel = s_isc[0]
                                above = s_isc[1]
                                m = m + above
                                want = want - above
                                pref = (pref << cutlass.Uint32(8)) | cutlass.Uint32(sel)
                                if s_isc[2] == want:
                                    final_shift = cutlass.Uint32(shift)
                                    active = cutlass.Int32(0)
                        kth = pref
                        if tidx == cutlass.Int32(0):
                            s_isc[3] = cutlass.Int32(0)
                            s_isc[4] = cutlass.Int32(0)
                        cute.arch.barrier()
                        nt = cutlass.Int32(K) - m
                        ie = cutlass.Int32(tidx)
                        while ie < C:
                            kv = s_cand[ie]
                            u = cutlass.Uint32(kv >> cutlass.Uint64(32)) >> final_shift
                            idx = cutlass.Int32(cutlass.Uint32(kv & cutlass.Uint64(0xFFFFFFFF)))
                            if u > kth:
                                p = _atomic_add_cta(
                                    s_isc.iterator + cutlass.Int32(3), cutlass.Int32(1)
                                )
                                out_row[p] = idx
                            else:
                                if u == kth:
                                    p = _atomic_add_cta(
                                        s_isc.iterator + cutlass.Int32(4), cutlass.Int32(1)
                                    )
                                    if p < nt:
                                        out_row[m + p] = idx
                            ie = ie + cutlass.Int32(TB)

            # Exit rendezvous — PLATEAU PATH ONLY (wave-overlap fix).
            # On the P3/P4 path the post-merge _cluster_sync_aligned() above is
            # the LAST remote-op rendezvous (P4 is CTA0-local smem + global
            # stores only), so non-rank0 CTAs exit right after it — mirroring
            # CUDA's `if (rank != 0) return;` and freeing 15/16 SMs one
            # P4-duration earlier (~+5us/wave at BS>=8 otherwise). The
            # plateau-emit path still has remote atomics in flight up to its
            # end (timing-dependent CUDA 719 territory without this barrier —
            # the DSL emits NO implicit pre-ret cluster barrier) and therefore
            # KEEPS the trailing rendezvous. m_gt is cluster-uniform
            # (redundantly computed driver), so this is a uniform branch
            # around the cluster barrier. Degenerate rows (outer branch)
            # issue no remote ops, so skipping it there is safe too.
            if cutlass.const_expr(CS > 1):
                if m_gt >= cutlass.Int32(0):
                    _cluster_sync_aligned()

    # ------------------------------------------------------------------
    # host launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        seq_lens: cute.Tensor,
        out_idx: cute.Tensor,
        stream,
    ):
        num_rows = logits.shape[0]
        CS = cutlass.const_expr(self.cluster_size)
        self.gvr_reg_kernel(logits, pre_idx, seq_lens, out_idx).launch(
            grid=(num_rows * CS, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(CS, 1, 1) if cutlass.const_expr(CS > 1) else None,
            stream=stream,
            min_blocks_per_mp=1,
        )


# ---------------------------------------------------------------------------
# compile cache + public entry (explicit variant params; dispatcher selects)
# ---------------------------------------------------------------------------
_LAUNCH_CACHE = {}


def _get_compiled(K: int, CS: int, TB: int, MAXV: int, AR: int, next_n: int = 1, cr: int = 4):
    key = (K, CS, TB, MAXV, AR, next_n, cr)
    compiled = _LAUNCH_CACHE.get(key)
    if compiled is None:
        kC = 8192 if K >= 2048 else 6144
        kern = GvrRegKernel(
            top_k=K,
            kC=kC,
            cluster_size=CS,
            ar=AR,
            maxv=MAXV,
            num_threads=TB,
            next_n=next_n,
            compress_ratio=cr,
        )
        n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
        # pre_idx is request-level (num_rows // next_n rows); keep the shared
        # n_rows sym at next_n == 1 (identical compiled artifact to v1).
        n_pre = n_rows if next_n == 1 else cute.sym_int()
        logits_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (n_rows, n_cols), stride_order=(1, 0), assumed_align=16
        )
        pre_idx_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (n_pre, K), stride_order=(1, 0), assumed_align=16
        )
        seq_lens_fake = _crt.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, K), stride_order=(1, 0), assumed_align=16
        )
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            kern,
            logits_fake,
            pre_idx_fake,
            seq_lens_fake,
            out_fake,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )
        _LAUNCH_CACHE[key] = compiled
    return compiled


_FAST = {}  # (K, cs, tb, maxv, ar, npad) -> compiled (contract checked once)


def reg_topk(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    K: int,
    cs: int,
    tb: int,
    maxv: int,
    ar: int,
    next_n: int = 1,
    cr: int = 4,
) -> None:
    """CuTe DSL gvr_topk_reg<cs, tb, maxv, ar>. logits [BS, npad] fp32 (npad
    mult of 64; tail beyond each row's N_eff may be garbage — masked at the
    register load), pre_idx [BS // next_n, K] int32 hint (request-level),
    seq_lens [BS // next_n] int32 (uncompressed-token space), out [BS, K]
    int32. Variant params are explicit — the dispatcher selects; compile
    cache keyed (K, cs, tb, maxv, ar, next_n, cr)."""
    npad = logits.shape[1]
    key = (K, cs, tb, maxv, ar, npad, next_n, cr)
    fn = _FAST.get(key)
    if fn is None:
        assert npad % 64 == 0, f"npad {npad} not a multiple of 64"
        vpc = (npad // 4 + cs - 1) // cs
        assert vpc <= maxv * tb, (
            f"slice {vpc} float4s exceeds MAXV*TB={maxv * tb} (npad={npad}, cs={cs})"
        )
        kc = 8192 if K >= 2048 else 6144
        assert npad > kc, f"npad {npad} <= kC {kc}: reg path has no trivial branch"
        fn = _get_compiled(K, cs, tb, maxv, ar, next_n, cr)
        _FAST[key] = fn
    fn(logits, pre_idx, seq_lens, out)
