# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode hyper-connection site for 1..16 rows on warp tensor cores: two PDL launches, the
first emitting the bottleneck `low`, the second the up projection, gate and stream mean.
The NVFP4 tp1 target's `qwen4_exp_hc_decode_rows_tc` kernels, bf16 variants only (the routed
experts' output arrives finalized and summed; the mix feeds bf16 blocks).

first (`_First2`): a (2 K halves x 4 streams) cluster per 24-column chunk forms the injection,
the per-stream sums of squares and `y = bf16(x * (1 + w))` and multiplies `y` by the weight
rows on mma.sync; cluster rank o reduces rows (2o, 2o+1) into `rstd_s = (ss_s / H + eps)^-1/2`,
the logits `z = sum_s rstd_s (P_s,0 + P_s,1) / C`, `low = bf16(silu(z))` (columns < 320) and
the injection weights `2 sigmoid(z)` (columns 320..323). mix (`_Mix2`): the up projection of
`low`, gate x normed, the stream mean.

`hc_out` is bitwise `qwen4_exp_hc_inject_norm`'s; the rest is bf16 operands on tensor cores
with fp32 accumulation (1-2 bf16 ulp of the float64 composition).
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
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (
    _asm,
    _mapa,
    _sigmoid,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

#: This checkpoint's hyper-connection bottleneck width.
RANK = 320
#: Rows a launch serves (one m16 MMA tile).
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
#: Rows of First2's A stage allocated at least; the footprint leaves an SM room for the mix CTAs.
A_MIN_ROWS = 8
BLOCK_H = 16


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
def _bulk_g2s(dst, src, nbytes: int, mbar, *, loc=None, ip=None):
    """One non-tensor bulk copy global -> this CTA's shared memory, completing on `mbar`."""
    _asm(
        T.i32(),
        [cutlass.Int32(dst), cutlass.Int64(src), cutlass.Int32(mbar)],
        "{ cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2], "
        + str(nbytes)
        + ", [$3]; mov.u32 $0, 0; }",
        "=r,r,l,r",
        loc,
        ip,
        side_effects=True,
    )


class _First2:
    def __init__(self, *, hidden, rank, n_cols, rows, inject, combine, pdl):
        assert hidden == KH * KHW and n_cols <= NC * NCH and rows <= M
        assert inject or not combine, "the combine forms the injected block output"
        self.hidden, self.rank, self.n_cols, self.rows = hidden, rank, n_cols, rows
        self.inject, self.combine, self.pdl = inject, combine, pdl
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
        return b + 256 + 128

    @cute.jit
    def __call__(
        self,
        mHc,
        mBlock,
        mInj,
        mShared,
        mGate,
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
        sBar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((2,)), byte_alignment=8)
        rbar = cutlass.Int32(sBar.iterator.toint())
        wbar = rbar + 8  # the weight rows' bulk copies
        # This CTA receives rows [me * RPO, me * RPO + RPO) of the cluster's partials from all CL
        # CTAs, by st.async on `rbar` (no cluster barrier, no fence, after the pushes).
        rows_me = cutlass.max(cutlass.min(cutlass.Int32(RPO), rows - me * RPO), cutlass.Int32(0))
        if tidx == 0:
            _mbar_init_tx(rbar, rows_me * (CL * (NC + 1) * 4))
            _mbar_init_tx(wbar, NC * KHW * 2)
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
        # One bulk copy a weight row (2560 B) into its padded shared row, completing on `wbar`.
        col0 = (s * hidden + k0) // VEC
        if tidx == 0:
            w_base = mW.iterator.toint() + cutlass.Int64(col0 * VEC * 2)
            sw_base = cutlass.Int32(sW.iterator.toint())
            for n in cutlass.range_constexpr(NC):
                _bulk_g2s(
                    sw_base + n * (ld * 2),
                    w_base + cutlass.Int64(n0 + n) * (4 * hidden * 2),
                    KHW * 2,
                    wbar,
                )
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. The K half's injection, sums of squares, y (bf16) into sA. A thread issues all
        # loads of a group of GROUP_IT items before using the first: one L2 round trip a group.
        hc_vec = cute.logical_divide(mHc, (None, VEC))
        b_vec = cute.logical_divide(mBlock, (None, VEC))
        sh_vec = cute.logical_divide(mShared, (None, VEC))
        o_vec = cute.logical_divide(mHcOut, (None, VEC))
        sa_vec = cute.logical_divide(sA, (None, VEC))
        items: cutlass.Constexpr = rows * nvec
        n_it: cutlass.Constexpr = -(-items // THREADS)
        cute.arch.cp_async_wait_group(0)  # the norm slice (the weights land on `wbar`)
        cute.arch.sync_threads()
        nwr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        outr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        yr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        for g0 in cutlass.range_constexpr(0, n_it, GROUP_IT):
            g_n: cutlass.Constexpr = min(GROUP_IT, n_it - g0)
            xg = cute.make_rmem_tensor(
                cute.make_layout((g_n, VEC), stride=(VEC, 1)), cutlass.BFloat16
            )
            bg = cute.make_rmem_tensor(
                cute.make_layout((g_n, VEC), stride=(VEC, 1)), cutlass.BFloat16
            )
            shg = cute.make_rmem_tensor(
                cute.make_layout((g_n, VEC), stride=(VEC, 1)), cutlass.BFloat16
            )
            ig = cute.make_rmem_tensor(cute.make_layout((g_n,), stride=(1,)), cutlass.Float32)
            zg = cute.make_rmem_tensor(cute.make_layout((g_n,), stride=(1,)), cutlass.Float32)
            for k in cutlass.range_constexpr(g_n):
                # Past the last item a thread reads the last item's operands (unused).
                c = cutlass.min(tidx + (g0 + k) * THREADS, cutlass.Int32(items - 1))
                r = c // nvec
                p = c % nvec
                cute.autovec_copy(hc_vec[r, (None, col0 + p)], xg[k, None])
                if const_expr(self.inject):
                    cute.autovec_copy(b_vec[r, (None, k0 // VEC + p)], bg[k, None])
                    ig[k] = mInj[r, s].to(cutlass.Float32)
                    if const_expr(self.combine):
                        cute.autovec_copy(sh_vec[r, (None, k0 // VEC + p)], shg[k, None])
                        zg[k] = mGate[r, 0].to(cutlass.Float32)
            for k in cutlass.range_constexpr(g_n):
                c = tidx + (g0 + k) * THREADS
                if c < items:
                    r = c // nvec
                    p = c % nvec
                    cute.autovec_copy(snw_vec[0, (None, p)], nwr)
                    ssum = cutlass.Float32(0.0)
                    if const_expr(self.inject):
                        g = ig[k]
                        if const_expr(self.combine):
                            # `tl.sigmoid`'s instructions, so the combine (and so `hc_out`)
                            # is bitwise `qwen4_exp_hc_inject_norm`'s.
                            sg = _sigmoid(zg[k])
                        for v in cutlass.range_constexpr(VEC):
                            bv = bg[k, v].to(cutlass.Float32)
                            if const_expr(self.combine):
                                # The MoE block output, rounded once.
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
        _mbar_wait0(wbar)
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
    gate_stride,
):
    kern = _First2(
        hidden=hidden,
        rank=rank,
        n_cols=n_cols,
        rows=rows,
        inject=inject,
        combine=combine,
        pdl=pdl,
    )
    dev, bf = "cuda", torch.bfloat16
    width = hidden * hc_count
    hc = torch.empty(rows, hc_stride, dtype=bf, device=dev)[:, :width]
    block = torch.empty(rows, block_stride, dtype=bf, device=dev)[:, :hidden]
    inj = torch.empty(rows, hc_count, dtype=bf, device=dev)
    shared = torch.empty(rows, hidden, dtype=bf, device=dev)
    gate = torch.empty(rows, gate_stride, dtype=bf, device=dev)[:, :1]
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
    rank=RANK,
):
    """Returns `(hc_out, low [R, rank] bf16, inject_out [R, C] bf16 or None, rstd [R, C] fp32)`;
    `n_cols` = rank + C with the injection, rank without (the stack-final mixer)."""
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
    weight = first.t()
    assert weight.shape[0] >= NC * NCH
    blk = block_out if has_inject else hyper_input.new_empty(rows, hidden)
    inj = inject if has_inject else hyper_input.new_empty(rows, hc_count)
    sh = shared if has_shared else hyper_input.new_empty(rows, hidden)
    # The gate logit is a [R, 1] column (the shared expert's gate|up output carries it).
    gate = gate_logits.reshape(rows, 1) if has_shared else hyper_input.new_empty(rows, 1)
    fn = _compiled_first(
        hidden,
        hc_count,
        rank,
        n_cols,
        weight.shape[0],
        rows,
        has_inject,
        has_shared,
        bool(get_env_enable_pdl()),
        hyper_input.stride(0),
        blk.stride(0),
        gate.stride(0),
    )
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(
        _dl(hyper_input),
        _dl(blk),
        _dl(inj, 2),
        _dl(sh),
        _dl(gate, 2),
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
    def __init__(self, *, hidden, streams, rank, rows, pdl):
        assert streams * BLOCK_H == WARPS * 8 and rows <= M and rank % 16 == 0
        self.hidden, self.streams, self.rank, self.rows, self.pdl = hidden, streams, rank, rows, pdl
        self.ld = rank + PAD

    def smem_bytes(self):
        def r(n):
            return -(-n // 128) * 128

        nrow = self.streams * BLOCK_H
        b = r(nrow * self.ld * 2) + r(M * self.ld * 2)
        b += r(M * nrow * 4) + r(M * nrow * 2) + r(M * self.streams * 4) + r(nrow * 4)
        return b + 256

    @cute.jit
    def __call__(self, mLow, mRstd, mUp, mHc, mNorm, mMixed, stream: cuda_drv.CUstream):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            cute.make_layout((1, WARPS, 1)),
        )
        g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        self.kernel(mLow, mRstd, mUp, mHc, mNorm, mMixed, mma, g2s).launch(
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
        mMixed,
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
        # Rows of sL past R are never written (MMA rows are independent; discarded). Before the
        # wait the up rows land and every thread loads its B fragments into registers.
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
                if lane < BLOCK_H:
                    tot = cutlass.Float32(0.0)
                    for t in cutlass.range_constexpr(streams):
                        tot = tot + sV[r, t * BLOCK_H + lane]
                    mMixed[r, i0 + lane] = (tot * (1.0 / streams)).to(cutlass.BFloat16)


@functools.lru_cache(maxsize=None)
def _compiled_mix(hidden, streams, rank, rows, hc_stride, pdl):
    kern = _Mix2(hidden=hidden, streams=streams, rank=rank, rows=rows, pdl=pdl)
    dev, bf = "cuda", torch.bfloat16
    width = hidden * streams
    low = torch.empty(rows, rank, dtype=bf, device=dev)
    rstd = torch.empty(rows, streams, dtype=torch.float32, device=dev)
    up = torch.empty(width, rank, dtype=bf, device=dev)
    hc = torch.empty(rows, hc_stride, dtype=bf, device=dev)[:, :width]
    norm = torch.empty(1, width, dtype=bf, device=dev)
    mixed = torch.empty(rows, hidden, dtype=bf, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        _dl(low),
        _dl(rstd, 4),
        _dl(up),
        _dl(hc, 16),
        _dl(norm, 2),
        _dl(mixed, 2),
        stream,
    )


def mix_rows(low, rstd, up, hc_out, norm_weight, hc_count):
    rows, width = hc_out.shape
    hidden = width // hc_count
    rank = up.shape[0]
    dev, dt = hc_out.device, hc_out.dtype
    mixed = torch.empty((rows, hidden), dtype=dt, device=dev)
    fn = _compiled_mix(hidden, hc_count, rank, rows, hc_out.stride(0), bool(get_env_enable_pdl()))
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(
        _dl(low),
        _dl(rstd, 4),
        _dl(up.t()),
        _dl(hc_out, 16),
        _dl(norm_weight.reshape(1, -1), 2),
        _dl(mixed, 2),
        stream,
    )
    return mixed


def _reference_first(
    hyper_input, block_out, inject, norm_weight, first, hc_count, n_cols, eps, shared, gate_logits
):
    """float64: the combine and the injection each rounded once as `qwen4_exp_hc_inject_norm`
    does, the grouped RMS, the projection operand `y = x * (1 + w)` rounded once to bf16 (the
    MMA's A operand), the logits `z = sum_s rstd_s (y_s @ W_s) / C`, `low = silu(z)` and
    `2 sigmoid(z)` each rounded once, rstd in fp32."""
    dt = hyper_input.dtype
    rows, width = hyper_input.shape
    hidden = width // hc_count
    if block_out is None:
        hc_out = hyper_input
    else:
        blk = block_out.double()
        if shared is not None:
            sg = torch.sigmoid(gate_logits.double().reshape(rows, 1))
            blk = (blk + sg * shared.double()).to(dt).double()
        streams = hyper_input.double().unflatten(-1, (hc_count, hidden))
        hc_out = (streams + blk.unsqueeze(-2) * inject.double().unsqueeze(-1)).flatten(-2).to(dt)
    x = hc_out.double().unflatten(-1, (hc_count, hidden))
    rstd = torch.rsqrt(x.pow(2).mean(-1) + eps)
    y = (x * (1.0 + norm_weight.double().unflatten(-1, (hc_count, hidden)))).to(dt).double()
    z = (y * rstd.unsqueeze(-1)).flatten(-2) @ first[:, :n_cols].double() / hc_count
    low = torch.nn.functional.silu(z[:, :RANK]).to(dt)
    inj = (2.0 * torch.sigmoid(z[:, RANK:])).to(dt) if n_cols > RANK else None
    return hc_out, low, inj, rstd.float()


class Qwen4ExpHcDecodeRowsFirst(OpWrapper):
    """First launch of a 1..16-row decode hyper-connection site: the previous block's injection
    (and, with `shared` / `gate_logits`, the MoE block's combine `routed + sigmoid(gate) * shared`
    rounded once), the grouped RMS, the first projection over `[down; block_inject]`, the SiLU
    bottleneck and the injection weights. Returns `(hc_out, low [R, 320] bf16, inject_out [R, C]
    or None, rstd [R, C] fp32)`; `hc_out` *is* `hyper_input` without `block_out`.

    `hc_out` is bitwise `qwen4_exp_hc_inject_norm`'s; `low` and the injection weights are within
    one bf16 step of the float64 composition over `y = bf16(x * (1 + w))` (`reference`).

    Fusion boundary: `qwen4_exp_hc_inject_norm` + `qwen4_exp_hc_decode_tc_first`, with the rstd
    handed on instead of the `[R, C * H]` normed stream.
    """

    ARCHS = frozenset({Arch.SM_103})
    # spec: rows, site (combine: after a MoE block; inject: after an attention / GDN block or
    # the draft front's embedding add; norm: no pending block), cols (324, or 320 for the
    # stack-final / draft-head mixer); the checkpoint's geometry, bf16.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="1 row after a MoE block: a one-request decode step / draft pass",
            spec=dict(rows=1, site="combine", cols=324),
        ),
        Cell(
            why="1 row after an attention / GDN block: a draft pass's MoE site",
            spec=dict(rows=1, site="inject", cols=324),
        ),
        Cell(
            why="4 rows, the stack-final mixer after the last MoE block (320 cols)",
            spec=dict(rows=4, site="combine", cols=320),
        ),
        Cell(
            why="4 rows, norm only: the stack's first site (324 cols)",
            spec=dict(rows=4, site="norm", cols=324),
        ),
        Cell(
            why="16 rows after a MoE block: a 4-request MTP-3 verify step (MAX_ROWS)",
            spec=dict(rows=16, site="combine", cols=324),
        ),
        Cell(
            why="16 rows after an attention / GDN block: a 16-request rank graph's draft MoE site",
            spec=dict(rows=16, site="inject", cols=324),
        ),
        Cell(
            why="16 rows, norm only, 320 cols (MAX_ROWS)",
            spec=dict(rows=16, site="norm", cols=320),
        ),
    )
    note = """
    Source: `_First2` above (CuTe DSL). Grid (2 K halves, 4 streams, 14 N chunks), cluster
    (2, 4, 1); the norm slice and the CTA's 24 weight rows staged before the PDL wait,
    dependents released there. The gate logit is read through its row stride (a column of the
    shared expert's gate|up output). One compile per row count, variant and operand row
    strides (lru_cache, first use: compile before capture).
    Not certified: hc_count other than 4, hidden other than 2560, rank other than 320.
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
    ):
        """float64 (`_reference_first`)."""
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
    ) -> None:
        rows, width = hyper_input.shape
        assert 1 <= rows <= MAX_ROWS, f"1..{MAX_ROWS} rows"
        assert hyper_input.dtype == torch.bfloat16 and hyper_input.stride(1) == 1
        assert hc_count == 4 and width // hc_count == KH * KHW, "hc_count 4, hidden 2560"
        assert n_cols in (RANK, RANK + hc_count), "the bottleneck (+ the injection) columns"
        assert first.stride(0) == 1 and first.shape[1] >= NC * NCH, (
            "the padded [width, 384] .t() view"
        )
        assert (block_out is None) == (inject is None)
        assert (shared is None) == (gate_logits is None)
        assert shared is None or block_out is not None, "the combine needs the routed rows"
        if block_out is not None:
            assert block_out.shape == (rows, width // hc_count) and block_out.stride(1) == 1
            assert inject.is_contiguous() and inject.shape == (rows, hc_count)
        if shared is not None:
            assert shared.is_contiguous() and shared.shape == block_out.shape
            assert gate_logits.numel() == rows and gate_logits.stride(-1) == 1
        for t in (hyper_input, block_out, shared, first):
            assert t is None or t.data_ptr() % 16 == 0, "16-byte aligned rows"
        assert hyper_input.stride(0) % VEC == 0, "16-byte aligned rows"
        assert block_out is None or block_out.stride(0) % VEC == 0, "16-byte aligned rows"


class Qwen4ExpHcDecodeRowsMix(OpWrapper):
    """Second launch of a 1..16-row decode hyper-connection site: the up projection of `low`,
    the sigmoid gate on the normed stream (`hc_out * rstd * (1 + w)`, fp32) and the stream
    mean, rounded once. Returns `mixed` `[R, hidden]`. Fusion boundary:
    `qwen4_exp_hc_up_gate_mean`, from `hc_out` and its rstd instead of the normed stream.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="1 row: a one-request decode step / draft pass", spec=dict(rows=1)),
        Cell(why="16 rows: a 4-request MTP-3 verify step (MAX_ROWS)", spec=dict(rows=16)),
    )
    note = """
    Source: `_Mix2` above (CuTe DSL), after `qwen4_exp_hc_decode_rows_first`. The up weight's
    64 rows staged and their B fragments loaded before the PDL wait; dependents released at
    the start.
    """

    def __call__(self, low, rstd, up, hc_out, norm_weight, hc_count):
        return mix_rows(low, rstd, up, hc_out, norm_weight, hc_count)

    def reference(self, low, rstd, up, hc_out, norm_weight, hc_count):
        """float64 mixed, rounded once."""
        hidden = hc_out.shape[1] // hc_count
        gate = torch.sigmoid(low.double() @ up.double())
        x = hc_out.double().unflatten(-1, (hc_count, hidden)) * rstd.double().unsqueeze(-1)
        normed = x.flatten(-2) * (1.0 + norm_weight.double())
        return (gate * normed).unflatten(-1, (hc_count, hidden)).mean(-2).to(hc_out.dtype)

    def is_valid(self, low, rstd, up, hc_out, norm_weight, hc_count) -> None:
        rows = hc_out.shape[0]
        assert 1 <= rows <= MAX_ROWS and low.shape == (rows, RANK) and low.is_contiguous()
        assert rstd.shape == (rows, hc_count) and rstd.is_contiguous()
        assert hc_count == 4 and hc_out.shape[1] == hc_count * KH * KHW, "hc_count 4, hidden 2560"
        assert up.shape == (RANK, hc_out.shape[1]) and up.t().is_contiguous(), (
            "up the [R, width] view of a row-major [width, R] weight"
        )
        assert hc_out.stride(1) == 1 and hc_out.stride(0) % VEC == 0


qwen4_exp_hc_decode_rows_first = Qwen4ExpHcDecodeRowsFirst()
qwen4_exp_hc_decode_rows_mix = Qwen4ExpHcDecodeRowsMix()
