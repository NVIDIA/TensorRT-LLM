"""Decode hyper-connection site for 2..16 rows on warp tensor cores: two PDL launches, the
first emitting the bottleneck `low`, the second the up projection, gate, stream mean and NVFP4.

Tier 5 (CuTe DSL written for this target). A cluster spans the 4 streams x 2 K halves, so the
first launch finishes the site's first half itself:

first (`_First2`): grid (2 K halves, 4 streams, 14 N chunks of 24 columns), cluster (2, 4, 1),
8 warps. A CTA forms its rows' injection over its 1280-element K half (rounded once; stored by the
chunk-0 CTAs), the slice's sums of squares and `y = x * (1 + w)` in bf16; each warp multiplies a
160-wide K slice of `y` by the CTA's 24 weight rows (staged before the PDL wait) on mma.sync;
the warps' partials are summed in shared memory. Every CTA pushes rows (2o, 2o+1) of its partial
tile and of its sums of squares to cluster rank o; rank o then forms, per row,
`rstd_s = (ss_s / H + eps)^-1/2`, the logits `z = sum_s rstd_s (P_s,0 + P_s,1) / C`, and writes
`low = bf16(silu(z))` (columns < 320), the injection weights `2 sigmoid(z)` (columns 320..323)
and the rows' rstd.

mix (`_Mix2`): 160 CTAs of 8 warps (16 hidden columns x 4 streams). Before the wait: the 64
up-weight rows. After: `low` (640 bytes a row), rstd and the CTA's residual elements; the
[16 x 64] up projection; gate x normed; the stream mean and its NVFP4.

Accuracy as `hc_tc.py` (bf16 operands into both projections; `hc_out` bitwise).
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_inject_norm import (
    _asm,
    _mapa,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

#: This checkpoint's hyper-connection bottleneck width.
RANK = 320
#: Rows a launch serves (one m16 MMA tile); the forward sends 2..MAX_ROWS-row sites here.
MAX_ROWS = 16
M = 16
VEC = 8
PAD = 8
KH = 2  # K halves
KHW = 1280
NC = 24  # columns a CTA
NCH = 14  # 14 x 24 = 336 >= 324
WARPS = 8
THREADS = WARPS * 32
KW = KHW // WARPS  # K a warp
CL = KH * 4  # cluster size
RPO = M // CL
#: Item iterations a thread whose activation loads issue together (all of them at <= 16 rows).
GROUP_IT = 10
#: Rows from which the finalize fold is split over a K half's 4 CTAs (each forms a quarter of
#: the half's rows and pushes it to all four); below it each CTA folds its whole K half itself
#: in its items (4x the gathers, no exchange: 1.4-1.7 us a layer faster at 2 and 3 rows behind
#: the runner, 1.7-2.3 us slower from 4, in the chained [MoE + site + MoE site] bench).
FOLD_SPLIT_ROWS = 4
#: Fold vectors a thread whose gathers issue together, split (all of them at <= 16 rows) and not
#: (ten 16-byte gathers a vector: register budget).
GROUP_FOLD = 3
GROUP_FOLD_ITEMS = 2
#: Rows of First2's A stage allocated at least: its shared-memory footprint (~90 KB at 2..8
#: rows, ~118 KB at 16) leaves room beside it for the mix kernel's CTAs of the same site.
A_MIN_ROWS = 8
BLOCK_H = 16


@dsl_user_op
def _st_dsmem_f32(addr, val, *, loc=None, ip=None):
    """4 bytes into a cluster CTA's shared memory."""
    _asm(
        T.i32(),
        [cutlass.Int32(addr), cutlass.Float32(val)],
        "{ st.shared::cluster.f32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _st_async_f32(addr, val, mbar, *, loc=None, ip=None):
    """4 bytes into a cluster CTA's shared memory, completing on that CTA's mbarrier."""
    _asm(
        T.i32(),
        [cutlass.Int32(addr), cutlass.Float32(val), cutlass.Int32(mbar)],
        "{ st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$1], $2, [$3]; "
        "mov.u32 $0, 0; }",
        "=r,r,f,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _mbar_init_tx(mbar, nbytes, *, loc=None, ip=None):
    """Init an mbarrier (one arrival), make it visible to the cluster, arrive expecting
    `nbytes` of transactions."""
    _asm(
        T.i32(),
        [cutlass.Int32(mbar), cutlass.Int32(nbytes)],
        "{ mbarrier.init.shared::cta.b64 [$1], 1; fence.mbarrier_init.release.cluster; "
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _mbar_wait0(mbar, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int32(mbar)],
        "{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$1], 0; "
        "@!p bra WAIT; mov.u32 $0, 0; }",
        "=r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _st_async_v4(addr, a, b, c, d, mbar, *, loc=None, ip=None):
    """16 bytes into a cluster CTA's shared memory, completing on that CTA's mbarrier."""
    _asm(
        T.i32(),
        [cutlass.Int32(addr)] + [cutlass.Int32(v) for v in (a, b, c, d)] + [cutlass.Int32(mbar)],
        "{ st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$1], {$2, $3, $4, $5}, "
        "[$6]; mov.u32 $0, 0; }",
        "=r,r,r,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _bf16x2(lo, hi, *, loc=None, ip=None):
    """Two fp32 values rounded (RN) into one bf16x2 word, `lo` low (exact for bf16 values)."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
        )
    )


class _First2:
    def __init__(self, *, hidden, rank, n_cols, rows, inject, combine, pdl, topk=0):
        assert hidden == KH * KHW and n_cols <= NC * NCH and rows <= M
        assert topk == 0 or combine, "the finalize fold needs the combine"
        self.hidden, self.rank, self.n_cols, self.rows = hidden, rank, n_cols, rows
        self.inject, self.combine, self.pdl, self.topk = inject, combine, pdl, topk
        self.ld = KHW + PAD
        self.rld = NC + 2  # pushed slot: NC partials + the sum of squares (+pad)

    def smem_bytes(self):
        def r(n):
            return -(-n // 128) * 128

        assert WARPS * M * NC * 4 + M * NC * 4 <= NC * self.ld * 2
        b = r(max(self.rows, A_MIN_ROWS) * self.ld * 2) + r(NC * self.ld * 2)
        b += r(self.rows * (KHW // VEC) * 4)
        b += r(CL * RPO * self.rld * 4) + r(M * 4)
        b += r(KHW * 2)  # the norm-weight slice
        if self.topk:
            b += 2 * r(self.rows * self.topk * 4)  # the fold's slot rows and weights
        if self.topk and self.rows >= FOLD_SPLIT_ROWS:
            b += r(self.rows * KHW * 2) + 128  # the block rows of the K half, and an mbarrier
        return b + 256 + 128

    @cute.jit
    def __call__(
        self,
        mHc,
        mBlock,
        mInj,
        mShared,
        mGate,
        mTopW,
        mTopRows,
        mNorm,
        mW,
        mHcOut,
        mLow,
        mInjOut,
        mRstd,
        eps: cutlass.Float32,
        stream: cuda_drv.CUstream,
    ):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            cute.make_layout((1, 1, 1)),
        )
        g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        self.kernel(
            mHc,
            mBlock,
            mInj,
            mShared,
            mGate,
            mTopW,
            mTopRows,
            mNorm,
            mW,
            mHcOut,
            mLow,
            mInjOut,
            mRstd,
            eps,
            mma,
            g2s,
        ).launch(
            grid=[KH, 4, NCH],
            block=[THREADS, 1, 1],
            cluster=[KH, 4, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mHc,
        mBlock,
        mInj,
        mShared,
        mGate,
        mTopW,
        mTopRows,
        mNorm,
        mW,
        mHcOut,
        mLow,
        mInjOut,
        mRstd,
        eps: cutlass.Float32,
        mma: cute.TiledMma,
        g2s: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kh, s, nc = cute.arch.block_idx()
        warp_i = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        rows: cutlass.Constexpr = self.rows
        hidden: cutlass.Constexpr = self.hidden
        rank: cutlass.Constexpr = self.rank
        n_cols: cutlass.Constexpr = self.n_cols
        ld: cutlass.Constexpr = self.ld
        rld: cutlass.Constexpr = self.rld
        nvec: cutlass.Constexpr = KHW // VEC
        k0 = kh * KHW
        n0 = nc * NC
        me = kh + KH * s  # cluster rank

        smem = cutlass.utils.SmemAllocator()
        # sA holds the R live rows only; the MMA's 16-row A view runs on into sW (rows past R
        # are garbage, their outputs discarded: the MMA's rows are independent).
        sA = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((max(rows, A_MIN_ROWS), KHW), stride=(ld, 1)),
            byte_alignment=128,
        )
        sW = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((NC, KHW), stride=(ld, 1)), byte_alignment=128
        )
        sSs = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((rows, nvec), stride=(nvec, 1)), byte_alignment=128
        )
        # The warps' partials and their sum reuse sW once every warp holds its B fragment.
        wp_ptr = cute.recast_ptr(sW.iterator, dtype=cutlass.Float32)
        sWp = cute.make_tensor(wp_ptr, cute.make_layout((WARPS, M, NC), stride=(M * NC, NC, 1)))
        sAcc = cute.make_tensor(wp_ptr + WARPS * M * NC, cute.make_layout((M, NC), stride=(NC, 1)))
        sRed = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((CL, RPO, rld), stride=(RPO * rld, rld, 1)),
            byte_alignment=128,
        )
        sSsr = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M,), stride=(1,)), byte_alignment=128
        )
        sNw = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((1, KHW), stride=(KHW, 1)), byte_alignment=128
        )
        topk: cutlass.Constexpr = self.topk
        split: cutlass.Constexpr = topk > 0 and rows >= FOLD_SPLIT_ROWS
        if const_expr(topk > 0):
            sTopR = smem.allocate_tensor(
                cutlass.Int32, cute.make_layout((rows * topk,), stride=(1,)), byte_alignment=128
            )
            sTopW = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((rows * topk,), stride=(1,)), byte_alignment=128
            )
        sBar = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((2 if split else 1,)), byte_alignment=8
        )
        rbar = cutlass.Int32(sBar.iterator.toint())
        bbar = rbar + 8
        if const_expr(split):
            # The K half's block rows (finalize + combine), a quarter formed by each of the 4
            # stream CTAs of this K half and pushed (st.async) into all four.
            sBlk = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((rows, KHW), stride=(KHW, 1)), byte_alignment=128
            )
            blk_a = cutlass.Int32(sBlk.iterator.toint())
            sblk_vec = cute.logical_divide(sBlk, (None, VEC))
        # This CTA receives rows [me * RPO, me * RPO + RPO) of the cluster's partials from all CL
        # CTAs, by st.async on `rbar` (no cluster barrier, no fence, after the pushes).
        rows_me = cutlass.max(cutlass.min(cutlass.Int32(RPO), rows - me * RPO), cutlass.Int32(0))
        if tidx == 0:
            _mbar_init_tx(rbar, rows_me * (CL * (NC + 1) * 4))
            if const_expr(split):
                _mbar_init_tx(bbar, rows * KHW * 2)
        cute.arch.cluster_arrive()

        # 1. Before the wait: the norm weight's slice (its own cp.async group, waited on right
        # after the PDL wait), then the CTA's 24 weight rows over its K half.
        nw_vec = cute.logical_divide(mNorm, (None, VEC))
        snw_vec = cute.logical_divide(sNw, (None, VEC))
        if tidx < KHW // VEC:
            cute.copy(
                g2s,
                nw_vec[0, (None, (s * hidden + k0) // VEC + tidx)],
                snw_vec[0, (None, tidx)],
            )
        cute.arch.cp_async_commit_group()
        w_vec = cute.logical_divide(mW, (None, VEC))
        sw_vec = cute.logical_divide(sW, (None, VEC))
        pieces: cutlass.Constexpr = NC * nvec
        col0 = (s * hidden + k0) // VEC
        for i in cutlass.range_constexpr(-(-pieces // THREADS)):
            c = tidx + i * THREADS
            if c < pieces:
                n = c // nvec
                p = c % nvec
                cute.copy(g2s, w_vec[n0 + n, (None, col0 + p)], sw_vec[n, (None, p)])
        cute.arch.cp_async_commit_group()
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. The K half's injection, sums of squares, y (bf16) into sA. Every activation load of
        # a group of up to GROUP_IT items a thread (the row's injection weight and gate logit
        # included) issues before the first is used: one L2 round trip a group.
        hc_vec = cute.logical_divide(mHc, (None, VEC))
        b_vec = cute.logical_divide(mBlock, (None, VEC))
        sh_vec = cute.logical_divide(mShared, (None, VEC))
        o_vec = cute.logical_divide(mHcOut, (None, VEC))
        sa_vec = cute.logical_divide(sA, (None, VEC))
        items: cutlass.Constexpr = rows * nvec
        n_it: cutlass.Constexpr = -(-items // THREADS)
        if const_expr(split):
            # With the split fold, every load that does not need the slot map issues with it: the
            # items' hc rows and injection weights (one group: <= 10 items a thread at <= 16
            # rows), and the fold quarter's shared-expert rows and gate logits. The gathers
            # are then the fold's one added L2 round trip.
            assert n_it <= GROUP_IT
            fvq: cutlass.Constexpr = KHW // 4 // VEC
            f_items: cutlass.Constexpr = rows * fvq
            n_f: cutlass.Constexpr = -(-f_items // THREADS)
            assert n_f <= GROUP_FOLD
            fcol0 = (kh * KHW + s * (KHW // 4)) // VEC
            xg0 = cute.make_rmem_tensor(
                cute.make_layout((n_it, VEC), stride=(VEC, 1)), cutlass.BFloat16
            )
            ig0 = cute.make_rmem_tensor(cute.make_layout((n_it,), stride=(1,)), cutlass.Float32)
            for k in cutlass.range_constexpr(n_it):
                c = cutlass.min(tidx + k * THREADS, cutlass.Int32(items - 1))
                r = c // nvec
                p = c % nvec
                cute.autovec_copy(hc_vec[r, (None, col0 + p)], xg0[k, None])
                ig0[k] = mInj[r, s].to(cutlass.Float32)
            shq = cute.make_rmem_tensor(
                cute.make_layout((n_f, VEC), stride=(VEC, 1)), cutlass.BFloat16
            )
            zq = cute.make_rmem_tensor(cute.make_layout((n_f,), stride=(1,)), cutlass.Float32)
            for k in cutlass.range_constexpr(n_f):
                c = cutlass.min(tidx + k * THREADS, cutlass.Int32(f_items - 1))
                cute.autovec_copy(sh_vec[c // fvq, (None, fcol0 + c % fvq)], shq[k, None])
                zq[k] = mGate[c // fvq].to(cutlass.Float32)
        if const_expr(topk > 0):
            # The rows' expanded->permuted slot map and routing weights (the gathers below
            # depend on it).
            for i in cutlass.range_constexpr(-(-(rows * topk) // THREADS)):
                cs = tidx + i * THREADS
                if cs < rows * topk:
                    sTopR[cs] = mTopRows[cs // topk, cs % topk]
                    sTopW[cs] = mTopW[cs // topk, cs % topk].to(cutlass.Float32)
        cute.arch.cp_async_wait_group(1)  # the norm slice (the weights may still be landing)
        cute.arch.sync_threads()
        if const_expr(split):
            # The finalize and the combine of this CTA's quarter of its K half (40 vectors a
            # row): the runner's fp32 FMAs in slot order from zero (a -1 slot skipped, a -1 slot
            # reads row 0, unused), rounded once; then fma(sigmoid(gate), shared, routed) rounded
            # once; pushed to the K half's 4 CTAs.
            cute.arch.cluster_wait()  # every peer has started (its mbarriers initialized)
            for f0 in cutlass.range_constexpr(0, n_f, GROUP_FOLD):
                f_n: cutlass.Constexpr = min(GROUP_FOLD, n_f - f0)
                fgq = cute.make_rmem_tensor(
                    cute.make_layout((f_n, topk, VEC), stride=(topk * VEC, VEC, 1)),
                    cutlass.BFloat16,
                )
                for k in cutlass.range_constexpr(f_n):
                    c = cutlass.min(tidx + (f0 + k) * THREADS, cutlass.Int32(f_items - 1))
                    r = c // fvq
                    q = c % fvq
                    for j in cutlass.range_constexpr(topk):
                        cute.autovec_copy(
                            b_vec[
                                cutlass.max(sTopR[r * topk + j], cutlass.Int32(0)),
                                (None, fcol0 + q),
                            ],
                            fgq[k, j, None],
                        )
                for k in cutlass.range_constexpr(f_n):
                    c = tidx + (f0 + k) * THREADS
                    if c < f_items:
                        r = c // fvq
                        q = c % fvq
                        sg = 1.0 / (1.0 + cute.math.exp(-zq[f0 + k]))
                        bq = [cutlass.Float32(0.0)] * VEC
                        for v in cutlass.range_constexpr(VEC):
                            acc = cutlass.Float32(0.0)
                            for j in cutlass.range_constexpr(topk):
                                if sTopR[r * topk + j] >= 0:
                                    acc = cute.math.fma(
                                        fgq[k, j, v].to(cutlass.Float32), sTopW[r * topk + j], acc
                                    )
                            bv = acc.to(cutlass.BFloat16).to(cutlass.Float32)
                            bq[v] = (
                                cute.math.fma(sg, shq[f0 + k, v].to(cutlass.Float32), bv)
                                .to(cutlass.BFloat16)
                                .to(cutlass.Float32)
                            )
                        words = [_bf16x2(bq[2 * w], bq[2 * w + 1]) for w in range(4)]
                        dst = blk_a + (r * KHW + s * (KHW // 4) + q * VEC) * 2
                        for sp in cutlass.range_constexpr(4):
                            peer = kh + KH * sp
                            _st_async_v4(_mapa(dst, peer), *words, _mapa(bbar, peer))
        nwr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        outr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        yr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        group: cutlass.Constexpr = GROUP_FOLD_ITEMS if topk > 0 and not split else GROUP_IT
        for g0 in cutlass.range_constexpr(0, n_it, group):
            g_n: cutlass.Constexpr = min(group, n_it - g0)
            if const_expr(split):
                xg = xg0
            else:
                xg = cute.make_rmem_tensor(
                    cute.make_layout((g_n, VEC), stride=(VEC, 1)), cutlass.BFloat16
                )
            bg = cute.make_rmem_tensor(
                cute.make_layout((g_n, VEC), stride=(VEC, 1)), cutlass.BFloat16
            )
            shg = cute.make_rmem_tensor(
                cute.make_layout((g_n, VEC), stride=(VEC, 1)), cutlass.BFloat16
            )
            if const_expr(split):
                ig = ig0
            else:
                ig = cute.make_rmem_tensor(cute.make_layout((g_n,), stride=(1,)), cutlass.Float32)
            zg = cute.make_rmem_tensor(cute.make_layout((g_n,), stride=(1,)), cutlass.Float32)
            if const_expr(topk > 0 and not split):
                fg = cute.make_rmem_tensor(
                    cute.make_layout((g_n, topk, VEC), stride=(topk * VEC, VEC, 1)),
                    cutlass.BFloat16,
                )
            for k in cutlass.range_constexpr(0 if split else g_n):
                # Past the last item a thread reads the last item's operands (unused).
                c = cutlass.min(tidx + (g0 + k) * THREADS, cutlass.Int32(items - 1))
                r = c // nvec
                p = c % nvec
                cute.autovec_copy(hc_vec[r, (None, col0 + p)], xg[k, None])
                if const_expr(self.inject):
                    if const_expr(topk > 0):
                        # The finalize's slot rows of the permuted FC2 output (a -1 slot reads
                        # row 0, unused).
                        for j in cutlass.range_constexpr(topk):
                            cute.autovec_copy(
                                b_vec[
                                    cutlass.max(sTopR[r * topk + j], cutlass.Int32(0)),
                                    (None, k0 // VEC + p),
                                ],
                                fg[k, j, None],
                            )
                    else:
                        cute.autovec_copy(b_vec[r, (None, k0 // VEC + p)], bg[k, None])
                    ig[k] = mInj[r, s].to(cutlass.Float32)
                    if const_expr(self.combine):
                        cute.autovec_copy(sh_vec[r, (None, k0 // VEC + p)], shg[k, None])
                        zg[k] = mGate[r].to(cutlass.Float32)
            if const_expr(split and g0 == 0):
                _mbar_wait0(bbar)  # the K half's block rows have landed
            for k in cutlass.range_constexpr(g_n):
                c = tidx + (g0 + k) * THREADS
                if c < items:
                    r = c // nvec
                    p = c % nvec
                    cute.autovec_copy(snw_vec[0, (None, p)], nwr)
                    ssum = cutlass.Float32(0.0)
                    if const_expr(self.inject):
                        g = ig[k]
                        if const_expr(split):
                            cute.autovec_copy(sblk_vec[r, (None, p)], bg[k, None])
                        if const_expr(self.combine and not split):
                            sg = 1.0 / (1.0 + cute.math.exp(-zg[k]))
                        for v in cutlass.range_constexpr(VEC):
                            if const_expr(topk > 0 and not split):
                                # The runner's finalize: fp32 FMAs in slot order from zero, a
                                # -1 slot skipped, one rounding.
                                acc = cutlass.Float32(0.0)
                                for j in cutlass.range_constexpr(topk):
                                    if sTopR[r * topk + j] >= 0:
                                        acc = cute.math.fma(
                                            fg[k, j, v].to(cutlass.Float32),
                                            sTopW[r * topk + j],
                                            acc,
                                        )
                                bv = acc.to(cutlass.BFloat16).to(cutlass.Float32)
                            else:
                                bv = bg[k, v].to(cutlass.Float32)
                            if const_expr(self.combine and not split):
                                bv = (
                                    cute.math.fma(sg, shg[k, v].to(cutlass.Float32), bv)
                                    .to(cutlass.BFloat16)
                                    .to(cutlass.Float32)
                                )
                            outr[v] = cute.math.fma(bv, g, xg[k, v].to(cutlass.Float32)).to(
                                cutlass.BFloat16
                            )
                        if nc == 0:
                            cute.autovec_copy(outr, o_vec[r, (None, col0 + p)])
                    else:
                        for v in cutlass.range_constexpr(VEC):
                            outr[v] = xg[k, v]
                    for v in cutlass.range_constexpr(VEC):
                        xv = outr[v].to(cutlass.Float32)
                        ssum = xv * xv + ssum
                        yr[v] = (xv * (1.0 + nwr[v].to(cutlass.Float32))).to(cutlass.BFloat16)
                    cute.autovec_copy(yr, sa_vec[r, (None, p)])
                    sSs[r, p] = ssum
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # 3. Warp w: rows x its 160-wide K slice x the 24 columns; the warps' partials summed in
        # warp order; the K half's sums of squares per row.
        # The warp's K slice: 320-byte offsets, so 16-byte aligned (ldmatrix rows).
        a_w = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16,
                sA.iterator.toint() + warp_i * (KW * 2),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((M, KW), stride=(ld, 1)),
        )
        b_w = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16,
                sW.iterator.toint() + warp_i * (KW * 2),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((NC, KW), stride=(ld, 1)),
        )
        thr = mma.get_slice(lane)
        tCsA = thr.partition_A(a_w)
        tCrA = thr.make_fragment_A(tCsA)
        tCsB = thr.partition_B(b_w)
        tCrB = thr.make_fragment_B(tCsB)
        # Fragments by ldmatrix (4 / 2 8x8 matrices an instruction).
        ldsm_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16), mma
        )
        ldsm_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16), mma
        )
        cpa = ldsm_a.get_slice(lane)
        cpb = ldsm_b.get_slice(lane)
        cute.copy(ldsm_a, cpa.partition_S(a_w), cpa.retile(tCrA))
        cute.copy(ldsm_b, cpb.partition_S(b_w), cpb.retile(tCrB))
        acc = cute.make_rmem_tensor(thr.partition_shape_C((M, NC)), cutlass.Float32)
        acc.fill(0.0)
        cute.gemm(mma, acc, tCrA, tCrB, acc)
        cute.arch.sync_threads()  # every warp's fragments are loaded: sW may be reused
        co = thr.partition_C(cute.make_identity_tensor((M, NC)))
        for j in cutlass.range_constexpr(cute.size(acc)):
            rr, nn = co[j]
            sWp[warp_i, rr, nn] = acc[j]
        # Sums of squares: 16 threads a row (10 vectors each), then a 16-lane butterfly.
        segs: cutlass.Constexpr = THREADS // M
        per_seg: cutlass.Constexpr = nvec // segs
        rq = tidx // segs
        sq = tidx % segs
        t = cutlass.Float32(0.0)
        if rq < rows:
            for p in cutlass.range_constexpr(per_seg):
                t = t + sSs[rq, sq * per_seg + p]
        t = cute.arch.warp_reduction_sum(t, threads_in_group=segs)
        if sq == 0:
            if rq < rows:
                sSsr[rq] = t
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(-(-(M * NC) // THREADS)):
            e = tidx + i * THREADS
            if e < M * NC:
                rr = e // NC
                nn = e % NC
                t = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(WARPS):
                    t = t + sWp[w, rr, nn]
                sAcc[rr, nn] = t
        cute.arch.sync_threads()

        # 4. Push rows (2o, 2o+1) (and their sums of squares) to cluster rank o, slot `me`.
        if const_expr(not split):
            cute.arch.cluster_wait()
        red_addr = cutlass.Int32(sRed.iterator.toint())
        per: cutlass.Constexpr = RPO * (NC + 1)
        for it in cutlass.range_constexpr(-(-(CL * per) // THREADS)):
            c = tidx + it * THREADS
            if c < CL * per:
                o = c // per
                rem = c % per
                rl = rem // (NC + 1)
                nn = rem % (NC + 1)
                rr = o * RPO + rl
                if rr < rows:
                    v = cutlass.Float32(0.0)
                    if nn < NC:
                        v = sAcc[rr, nn]
                    else:
                        v = sSsr[rr]
                    dst = red_addr + ((me * RPO + rl) * rld + nn) * 4
                    _st_async_f32(_mapa(dst, o), v, _mapa(rbar, o))
        _mbar_wait0(rbar)

        # 5. Rank `me` finishes rows (2 me, 2 me + 1): rstd per stream, the logits, low / injection.
        for it in cutlass.range_constexpr(-(-(RPO * NC) // THREADS)):
            c = tidx + it * THREADS
            if c < RPO * NC:
                rl = c // NC
                nn = c % NC
                rr = me * RPO + rl
                col = n0 + nn
                if rr < rows:
                    if col < n_cols:
                        z = cutlass.Float32(0.0)
                        for st in cutlass.range_constexpr(4):
                            ssv = sRed[KH * st, rl, NC] + sRed[KH * st + 1, rl, NC]
                            rs = cute.math.rsqrt(ssv / hidden + eps)
                            z = (sRed[KH * st, rl, nn] + sRed[KH * st + 1, rl, nn]) * rs + z
                        z = z * 0.25
                        sig = 1.0 / (1.0 + cute.math.exp(-z))
                        if col < rank:
                            mLow[rr, col] = (z * sig).to(cutlass.BFloat16)
                        else:
                            mInjOut[rr, col - rank] = (2.0 * sig).to(cutlass.BFloat16)
        if nc == 0:
            if tidx < RPO * 4:
                rl = tidx // 4
                st = tidx % 4
                rr = me * RPO + rl
                if rr < rows:
                    ssv = sRed[KH * st, rl, NC] + sRed[KH * st + 1, rl, NC]
                    mRstd[rr, st] = cute.math.rsqrt(ssv / hidden + eps)


def _dl(t: torch.Tensor, align: int = 16):
    return from_dlpack(t, assumed_align=align)


@functools.lru_cache(maxsize=None)
def _compiled_first(
    hidden,
    hc_count,
    rank,
    n_cols,
    n_wrows,
    rows,
    inject,
    combine,
    pdl,
    hc_stride,
    block_stride,
    topk=0,
    n_perm=0,
):
    kern = _First2(
        hidden=hidden,
        rank=rank,
        n_cols=n_cols,
        rows=rows,
        inject=inject,
        combine=combine,
        pdl=pdl,
        topk=topk,
    )
    dev, bf = "cuda", torch.bfloat16
    width = hidden * hc_count
    hc = torch.empty(rows, hc_stride, dtype=bf, device=dev)[:, :width]
    block = torch.empty(n_perm if topk else rows, block_stride, dtype=bf, device=dev)[:, :hidden]
    topw = torch.empty(rows, max(topk, 1), dtype=bf, device=dev)
    toprows = torch.empty(rows, max(topk, 1), dtype=torch.int32, device=dev)
    inj = torch.empty(rows, hc_count, dtype=bf, device=dev)
    shared = torch.empty(rows, hidden, dtype=bf, device=dev)
    gate = torch.empty(rows, dtype=bf, device=dev)
    norm = torch.empty(1, width, dtype=bf, device=dev)
    w = torch.empty(n_wrows, width, dtype=bf, device=dev)
    hc_out = torch.empty(rows, width, dtype=bf, device=dev)
    low = torch.empty(rows, rank, dtype=bf, device=dev)
    inj_out = torch.empty(rows, hc_count, dtype=bf, device=dev)
    rstd = torch.empty(rows, hc_count, dtype=torch.float32, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        _dl(hc),
        _dl(block),
        _dl(inj, 2),
        _dl(shared),
        _dl(gate, 2),
        _dl(topw, 2),
        _dl(toprows, 4),
        _dl(norm),
        _dl(w),
        _dl(hc_out),
        _dl(low),
        _dl(inj_out, 2),
        _dl(rstd, 4),
        cutlass.Float32(1e-6),
        stream,
    )


def first_rows(
    hyper_input,
    block_out,
    inject,
    norm_weight,
    first,
    hc_count,
    n_cols,
    eps,
    shared=None,
    gate_logits=None,
    topk_weights=None,
    topk_rows=None,
    rank=320,
):
    """Returns `(hc_out, low [R, rank] bf16, inject_out [R, C] bf16 or None, rstd [R, C] fp32)`;
    `n_cols` = rank + C with the injection, rank without (the stack-final mixer). With
    `topk_weights` / `topk_rows` (the runner left unfinalized), `block_out` is its permuted FC2
    output `[P, H]` and the routed rows are its finalize, formed here."""
    rows, width = hyper_input.shape
    hidden = width // hc_count
    dev = hyper_input.device
    has_inject = block_out is not None
    has_shared = shared is not None
    hc_out = (
        torch.empty((rows, width), dtype=hyper_input.dtype, device=dev)
        if has_inject
        else hyper_input
    )
    low = torch.empty((rows, rank), dtype=hyper_input.dtype, device=dev)
    inj_out = torch.empty((rows, hc_count), dtype=hyper_input.dtype, device=dev)
    rstd = torch.empty((rows, hc_count), dtype=torch.float32, device=dev)
    pdl = get_env_enable_pdl()
    weight = first.t()
    assert weight.shape[0] >= NC * NCH
    top_k = topk_rows.shape[1] if topk_rows is not None else 0
    fn = _compiled_first(
        hidden,
        hc_count,
        rank,
        n_cols,
        weight.shape[0],
        rows,
        has_inject,
        has_shared,
        bool(pdl),
        hyper_input.stride(0),
        block_out.stride(0) if has_inject else hidden,
        top_k,
        block_out.shape[0] if top_k else 0,
    )
    blk = block_out if has_inject else hyper_input.new_empty(rows, hidden)
    inj = inject if has_inject else hyper_input.new_empty(rows, hc_count)
    sh = shared if has_shared else hyper_input.new_empty(rows, hidden)
    gate = gate_logits.reshape(rows) if has_shared else hyper_input.new_empty(rows)
    topw = topk_weights if top_k else hyper_input.new_empty(rows, 1)
    toprows = topk_rows if top_k else torch.empty(rows, 1, dtype=torch.int32, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(
        _dl(hyper_input),
        _dl(blk),
        _dl(inj, 2),
        _dl(sh),
        _dl(gate, 2),
        _dl(topw, 2),
        _dl(toprows, 4),
        _dl(norm_weight.reshape(1, -1)),
        _dl(weight),
        _dl(hc_out),
        _dl(low),
        _dl(inj_out, 2),
        _dl(rstd, 4),
        cutlass.Float32(eps),
        stream,
    )
    return hc_out, low, (inj_out if n_cols > rank else None), rstd


class _Mix2:
    def __init__(self, *, hidden, streams, rank, rows, fp4, pdl):
        assert streams * BLOCK_H == WARPS * 8 and rows <= M and rank % 16 == 0
        self.hidden, self.streams, self.rank, self.rows, self.fp4, self.pdl = (
            hidden,
            streams,
            rank,
            rows,
            fp4,
            pdl,
        )
        self.ld = rank + PAD

    def smem_bytes(self):
        def r(n):
            return -(-n // 128) * 128

        nrow = self.streams * BLOCK_H
        b = r(nrow * self.ld * 2) + r(M * self.ld * 2)
        b += r(M * nrow * 4) + r(M * nrow * 2) + r(M * self.streams * 4) + r(nrow * 4)
        return b + 256

    @cute.jit
    def __call__(
        self, mLow, mRstd, mUp, mHc, mNorm, mGs, mMixed, mXq, mXsf, stream: cuda_drv.CUstream
    ):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            cute.make_layout((1, WARPS, 1)),
        )
        g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        self.kernel(mLow, mRstd, mUp, mHc, mNorm, mGs, mMixed, mXq, mXsf, mma, g2s).launch(
            grid=[self.hidden // BLOCK_H, 1, 1],
            block=[THREADS, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=2,
        )

    @cute.kernel
    def kernel(
        self,
        mLow,
        mRstd,
        mUp,
        mHc,
        mNorm,
        mGs,
        mMixed,
        mXq,
        mXsf,
        mma: cute.TiledMma,
        g2s: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_i = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        hidden: cutlass.Constexpr = self.hidden
        streams: cutlass.Constexpr = self.streams
        rank: cutlass.Constexpr = self.rank
        rows: cutlass.Constexpr = self.rows
        ld: cutlass.Constexpr = self.ld
        nrow: cutlass.Constexpr = streams * BLOCK_H
        i0 = bidx * BLOCK_H

        smem = cutlass.utils.SmemAllocator()
        sU = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((nrow, rank), stride=(ld, 1)), byte_alignment=128
        )
        sL = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, rank), stride=(ld, 1)), byte_alignment=128
        )
        sV = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M, nrow), stride=(nrow, 1)), byte_alignment=128
        )
        sX = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, nrow), stride=(nrow, 1)), byte_alignment=128
        )
        sR = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M, streams), stride=(streams, 1)), byte_alignment=128
        )
        sG = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((nrow,), stride=(1,)), byte_alignment=128
        )

        u_vec = cute.logical_divide(mUp, (None, VEC))
        su_vec = cute.logical_divide(sU, (None, VEC))
        pieces: cutlass.Constexpr = nrow * (rank // VEC)
        for i in cutlass.range_constexpr(-(-pieces // THREADS)):
            c = tidx + i * THREADS
            if c < pieces:
                n = c // (rank // VEC)
                p = c % (rank // VEC)
                jrow = (n // BLOCK_H) * hidden + i0 + n % BLOCK_H
                cute.copy(g2s, u_vec[jrow, (None, p)], su_vec[n, (None, p)])
        cute.arch.cp_async_commit_group()
        if tidx < nrow:
            sG[tidx] = 1.0 + mNorm[0, (tidx // BLOCK_H) * hidden + i0 + tidx % BLOCK_H].to(
                cutlass.Float32
            )
        # Rows of sL past R are never written (MMA rows are independent; discarded).
        gs = cutlass.Float32(0.0)
        if const_expr(self.fp4):
            gs = mGs[0]
        # Before the wait as well: the up rows land, and every thread's B fragments go to
        # registers (the post-wait MMA then reads only its A fragments from shared memory).
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        thr = mma.get_slice(tidx)
        tCsB = thr.partition_B(sU)
        tCrB = thr.make_fragment_B(tCsB)
        cute.autovec_copy(tCsB, tCrB)
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()

        # low rows (16-byte pieces), rstd, this CTA's residual elements.
        l_vec = cute.logical_divide(mLow, (None, VEC))
        sl_vec = cute.logical_divide(sL, (None, VEC))
        lp: cutlass.Constexpr = rows * (rank // VEC)
        for i in cutlass.range_constexpr(-(-lp // THREADS)):
            c = tidx + i * THREADS
            if c < lp:
                rr = c // (rank // VEC)
                p = c % (rank // VEC)
                cute.copy(g2s, l_vec[rr, (None, p)], sl_vec[rr, (None, p)])
        cute.arch.cp_async_commit_group()
        if tidx < rows * streams:
            sR[tidx // streams, tidx % streams] = mRstd[tidx // streams, tidx % streams]
        h_vec = cute.logical_divide(mHc, (None, VEC))
        sx_vec = cute.logical_divide(sX, (None, VEC))
        xp: cutlass.Constexpr = rows * (nrow // VEC)
        if tidx < xp:
            rr = tidx // (nrow // VEC)
            q = tidx % (nrow // VEC)
            cute.copy(
                g2s,
                h_vec[
                    rr,
                    (
                        None,
                        ((q * VEC) // BLOCK_H) * (hidden // VEC)
                        + (i0 + (q * VEC) % BLOCK_H) // VEC,
                    ),
                ],
                sx_vec[rr, (None, q)],
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        tCsA = thr.partition_A(sL)
        tCrA = thr.make_fragment_A(tCsA)
        ldsm_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16), mma
        )
        cpa = ldsm_a.get_slice(tidx)
        cute.copy(ldsm_a, cpa.partition_S(sL), cpa.retile(tCrA))
        acc = cute.make_rmem_tensor(thr.partition_shape_C((M, nrow)), cutlass.Float32)
        acc.fill(0.0)
        cute.gemm(mma, acc, tCrA, tCrB, acc)
        co = thr.partition_C(cute.make_identity_tensor((M, nrow)))
        for j in cutlass.range_constexpr(cute.size(acc)):
            rr, n = co[j]
            if rr < rows:
                gate = 1.0 / (1.0 + cute.math.exp(-acc[j]))
                x = sX[rr, n].to(cutlass.Float32)
                sV[rr, n] = gate * (x * sR[rr, n // BLOCK_H] * sG[n])
        cute.arch.sync_threads()

        for rb in cutlass.range_constexpr(0, rows, WARPS):
            r = rb + warp_i
            if r < rows:
                m = cutlass.Float32(0.0)
                if lane < BLOCK_H:
                    tot = cutlass.Float32(0.0)
                    for t in cutlass.range_constexpr(streams):
                        tot = tot + sV[r, t * BLOCK_H + lane]
                    mb = (tot * (1.0 / streams)).to(cutlass.BFloat16)
                    mMixed[r, i0 + lane] = mb
                    m = mb.to(cutlass.Float32)
                if const_expr(self.fp4):
                    amax = cute.arch.warp_reduction_max(cute.math.abs(m), threads_in_group=BLOCK_H)
                    sf8 = (gs * (amax * cute.arch.rcp_approx(6.0))).to(cutlass.Float8E4M3FN)
                    osc = cute.arch.rcp_approx(sf8.to(cutlass.Float32) * cute.arch.rcp_approx(gs))
                    if amax == 0.0:
                        osc = cutlass.Float32(0.0)
                    sv = m * osc
                    a = cute.math.abs(sv)
                    code = cutlass.Int32(0)
                    if a > 0.25:
                        code = code + 1
                    if a >= 0.75:
                        code = code + 1
                    if a > 1.25:
                        code = code + 1
                    if a >= 1.75:
                        code = code + 1
                    if a > 2.5:
                        code = code + 1
                    if a >= 3.5:
                        code = code + 1
                    if a > 5.0:
                        code = code + 1
                    if cute.math.copysign(1.0, sv) < 0.0:
                        code = code + 8
                    hi = cute.arch.shuffle_sync_down(code, offset=1)
                    if lane < BLOCK_H:
                        if lane % 2 == 0:
                            mXq[r, (i0 + lane) // 2] = (code + hi * 16).to(cutlass.Uint8)
                        if lane == 0:
                            mXsf[r * (hidden // BLOCK_H) + bidx] = sf8


@functools.lru_cache(maxsize=None)
def _compiled_mix(hidden, streams, rank, rows, hc_stride, fp4, pdl):
    kern = _Mix2(hidden=hidden, streams=streams, rank=rank, rows=rows, fp4=fp4, pdl=pdl)
    dev, bf = "cuda", torch.bfloat16
    width = hidden * streams
    low = torch.empty(rows, rank, dtype=bf, device=dev)
    rstd = torch.empty(rows, streams, dtype=torch.float32, device=dev)
    up = torch.empty(width, rank, dtype=bf, device=dev)
    hc = torch.empty(rows, hc_stride, dtype=bf, device=dev)[:, :width]
    norm = torch.empty(1, width, dtype=bf, device=dev)
    gs = torch.empty(1, dtype=torch.float32, device=dev)
    mixed = torch.empty(rows, hidden, dtype=bf, device=dev)
    xq = torch.empty(rows, hidden // 2, dtype=torch.uint8, device=dev)
    xsf = torch.empty(rows * (hidden // 16), dtype=torch.float8_e4m3fn, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        _dl(low),
        _dl(rstd, 4),
        _dl(up),
        _dl(hc, 16),
        _dl(norm, 2),
        _dl(gs, 4),
        _dl(mixed, 2),
        _dl(xq, 1),
        _dl(xsf, 1),
        stream,
    )


def mix_rows(low, rstd, up, hc_out, norm_weight, hc_count, global_scale=None):
    """Returns `(mixed, xq, xsf)` (xq / xsf None without `global_scale`)."""
    rows, width = hc_out.shape
    hidden = width // hc_count
    rank = up.shape[0]
    dev, dt = hc_out.device, hc_out.dtype
    fp4 = global_scale is not None
    mixed = torch.empty((rows, hidden), dtype=dt, device=dev)
    xq = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=dev) if fp4 else None
    xsf = torch.empty((rows * (hidden // 16),), dtype=torch.uint8, device=dev) if fp4 else None
    pdl = get_env_enable_pdl()
    fn = _compiled_mix(hidden, hc_count, rank, rows, hc_out.stride(0), fp4, bool(pdl))
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    gs = global_scale.reshape(1) if fp4 else mixed.new_empty(1, dtype=torch.float32)
    q = xq if fp4 else mixed.new_empty(rows, hidden // 2, dtype=torch.uint8)
    sf = (
        xsf.view(torch.float8_e4m3fn)
        if fp4
        else mixed.new_empty(rows * (hidden // 16), dtype=torch.float8_e4m3fn)
    )
    fn(
        _dl(low),
        _dl(rstd, 4),
        _dl(up.t()),
        _dl(hc_out, 16),
        _dl(norm_weight.reshape(1, -1), 2),
        _dl(gs, 4),
        _dl(mixed, 2),
        _dl(q, 1),
        _dl(sf, 1),
        stream,
    )
    return mixed, xq, xsf


def _reference_first(
    hyper_input,
    block_out,
    inject,
    norm_weight,
    first,
    hc_count,
    n_cols,
    eps,
    shared=None,
    gate_logits=None,
    topk_weights=None,
    topk_rows=None,
):
    """float64: the finalize (with `topk_rows`), the combine and the injection each rounded once
    as the 1-row kernel does, the grouped RMS, the first projection over `normed`, `low =
    silu(z)` and `2 sigmoid(z)` (z = logits / C) each rounded once, rstd in fp32."""
    dt = hyper_input.dtype
    rows, width = hyper_input.shape
    hidden = width // hc_count
    rank = RANK
    if block_out is None:
        hc_out = hyper_input
    else:
        if topk_rows is not None:
            valid = (topk_rows >= 0).unsqueeze(-1)
            gathered = block_out.double()[topk_rows.clamp(min=0).long()]
            w = topk_weights.double().unsqueeze(-1)
            block_out = torch.where(valid, gathered * w, 0.0).sum(1).to(block_out.dtype)
        blk = block_out.double()
        if shared is not None:
            sg = torch.sigmoid(gate_logits.double().reshape(rows, 1))
            blk = (blk + sg * shared.double()).to(dt).double()
        streams = hyper_input.double().unflatten(-1, (hc_count, hidden))
        hc_out = (streams + blk.unsqueeze(-2) * inject.double().unsqueeze(-1)).flatten(-2).to(dt)
    x = hc_out.double().unflatten(-1, (hc_count, hidden))
    rstd = torch.rsqrt(x.pow(2).mean(-1) + eps)
    normed = (x * rstd.unsqueeze(-1)).flatten(-2) * (1.0 + norm_weight.double())
    z = normed @ first[:, :n_cols].double() / hc_count
    low = torch.nn.functional.silu(z[:, :rank]).to(dt)
    inj = (2.0 * torch.sigmoid(z[:, rank:])).to(dt) if n_cols > rank else None
    return hc_out, low, inj, rstd.float()


class Qwen4ExpHcDecodeRowsFirst(OpWrapper):
    """First launch of a 2..16-row decode hyper-connection site: the previous block's injection
    (and, with `shared` / `gate_logits`, the MoE block's combine `routed + sigmoid(gate) * shared`
    rounded once), the grouped RMS, the first projection over `[down; block_inject]`, the SiLU
    bottleneck and the injection weights. Returns `(hc_out, low [R, 320] bf16, inject_out [R, C]
    or None, rstd [R, C] fp32)`; `hc_out` *is* `hyper_input` without `block_out`.

    `hc_out` is bitwise the 1-row kernels'; `low` and the injection weights come from bf16
    operands on tensor cores (`y = bf16(x * (1 + w))` against the bf16 weight) with fp32
    accumulation, inside 1-2 bf16 ulp of the float64 composition.

    With `topk_weights` / `topk_rows` as well (the MoE runner left unfinalized), `block_out` is
    its permuted FC2 output `[P, H]` and the routed row is the runner's finalize, formed here:
    `routed[t, i] = sum_k topk_weights[t, k] * block_out[topk_rows[t, k], i]` as fp32 FMAs in
    slot order from zero (a -1 slot skipped), rounded once -- bitwise what the runner's finalize
    stores (as `qwen4_exp_hc_decode_first_cute`'s fold). The fold needs the combine.

    Fusion boundary: the routed experts' finalize (optional), the combine, injection, grouped
    RMS statistics, first projection and activations.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="site after a MoE block: combine + injection, 2 rows",
            spec=dict(rows=2, site="combine"),
        ),
        Cell(
            why="site after a MoE block, runner finalize folded in, 4 rows (MTP3 verify)",
            spec=dict(rows=4, site="fold"),
        ),
        Cell(why="MoE site: injection, 8 rows", spec=dict(rows=8, site="inject")),
        Cell(
            why="norm-only site (stack start, MTP head), 16 rows, 320 cols",
            spec=dict(rows=16, site="norm"),
        ),
    )
    note = """
    Metadata consumed: none. Call sites: `DecodeTarget._hc_site` / `_hc_moe_site` at 2..16 rows
    (decode batches, MTP verify and draft-extend rows), followed by `qwen4_exp_hc_decode_rows_mix`;
    after a TRTLLM-backend MoE block (`DecodeTarget._moe_block` on the permuted runner) with the
    fold. Grid (2 K halves, 4 streams, 14 N chunks of 24 columns), cluster (2, 4, 1), 8 warps;
    weights staged before the PDL wait. With the fold from 4 rows each of a K half's 4 CTAs
    forms a quarter of the half's routed + shared rows and pushes it to all four (DSMEM); at 2
    and 3 rows each CTA folds its K half in its items. One compile per row count and variant.
    """

    def __call__(
        self,
        hyper_input,
        block_out,
        inject,
        norm_weight,
        first,
        hc_count,
        n_cols,
        eps,
        shared=None,
        gate_logits=None,
        topk_weights=None,
        topk_rows=None,
    ):
        return first_rows(
            hyper_input,
            block_out,
            inject,
            norm_weight,
            first,
            hc_count,
            n_cols,
            eps,
            shared,
            gate_logits,
            topk_weights,
            topk_rows,
        )

    def reference(
        self,
        hyper_input,
        block_out,
        inject,
        norm_weight,
        first,
        hc_count,
        n_cols,
        eps,
        shared=None,
        gate_logits=None,
        topk_weights=None,
        topk_rows=None,
    ):
        return _reference_first(
            hyper_input,
            block_out,
            inject,
            norm_weight,
            first,
            hc_count,
            n_cols,
            eps,
            shared,
            gate_logits,
            topk_weights,
            topk_rows,
        )

    def is_valid(
        self,
        hyper_input,
        block_out,
        inject,
        norm_weight,
        first,
        hc_count,
        n_cols,
        eps,
        shared=None,
        gate_logits=None,
        topk_weights=None,
        topk_rows=None,
    ) -> None:
        rows, width = hyper_input.shape
        assert 1 <= rows <= MAX_ROWS, f"1..{MAX_ROWS} rows"
        assert hyper_input.dtype == torch.bfloat16 and hyper_input.stride(1) == 1
        assert width // hc_count == KH * KHW, "hidden 2560"
        assert first.stride(0) == 1 and first.shape[1] >= NC * NCH, (
            "the padded [width, 384] .t() view"
        )
        assert (block_out is None) == (inject is None)
        assert (shared is None) == (gate_logits is None)
        assert (topk_weights is None) == (topk_rows is None), "topk_weights and topk_rows together"
        if topk_rows is not None:
            assert shared is not None, "the finalize fold needs the combine"
            assert topk_rows.dtype == torch.int32 and topk_rows.shape[0] == rows
            assert topk_rows.is_contiguous() and topk_weights.is_contiguous()
            assert topk_weights.shape == topk_rows.shape and topk_weights.dtype == torch.bfloat16
            assert block_out.dim() == 2 and block_out.shape[1] == width // hc_count
            assert block_out.stride(1) == 1, "permuted rows [P, hidden]"


class Qwen4ExpHcDecodeRowsMix(OpWrapper):
    """Second launch of a 2..16-row decode hyper-connection site: the up projection of `low`, the
    sigmoid gate on the normed stream (`hc_out * rstd * (1 + w)`), the stream mean, and with
    `global_scale` the NVFP4 of the mean for the routed experts. Returns `(mixed, xq, xsf)`
    (`xq` / `xsf` None without `global_scale`); one rounding per output.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MoE site with NVFP4 output, 8 rows", spec=dict(rows=8, site="inject")),
        Cell(why="plain mix, 2 rows", spec=dict(rows=2, site="combine")),
        Cell(why="plain mix, 16 rows", spec=dict(rows=16, site="norm")),
    )
    note = """
    Metadata consumed: none. Call sites: after `qwen4_exp_hc_decode_rows_first`. 160 CTAs of 8
    warps (16 hidden columns x 4 streams); the up weight's 64 rows staged before the PDL wait.
    """

    def __call__(self, low, rstd, up, hc_out, norm_weight, hc_count, global_scale=None):
        return mix_rows(low, rstd, up, hc_out, norm_weight, hc_count, global_scale)

    def reference(self, low, rstd, up, hc_out, norm_weight, hc_count, global_scale=None):
        """float64 mixed, rounded once (the NVFP4 is checked against the kernel's own mixed)."""
        rows, width = hc_out.shape
        hidden = width // hc_count
        gate = torch.sigmoid(low.double() @ up.double())
        x = hc_out.double().unflatten(-1, (hc_count, hidden)) * rstd.double().unsqueeze(-1)
        normed = x.flatten(-2) * (1.0 + norm_weight.double())
        mixed = (gate * normed).unflatten(-1, (hc_count, hidden)).mean(-2).to(hc_out.dtype)
        return mixed, None, None

    def is_valid(self, low, rstd, up, hc_out, norm_weight, hc_count, global_scale=None) -> None:
        rows = hc_out.shape[0]
        assert 1 <= rows <= MAX_ROWS and low.shape[0] == rows and rstd.shape == (rows, hc_count)
        assert up.t().is_contiguous(), "up the [R, width] view of a row-major [width, R] weight"
        assert global_scale is None or global_scale.numel() == 1


qwen4_exp_hc_decode_rows_first = Qwen4ExpHcDecodeRowsFirst()
qwen4_exp_hc_decode_rows_mix = Qwen4ExpHcDecodeRowsMix()
