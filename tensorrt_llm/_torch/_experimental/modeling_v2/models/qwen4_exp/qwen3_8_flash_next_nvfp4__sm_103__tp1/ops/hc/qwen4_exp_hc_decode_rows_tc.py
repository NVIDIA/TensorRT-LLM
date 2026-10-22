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
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from .qwen4_exp_hc_inject_norm import _asm, _mapa

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


class _First2:
    def __init__(self, *, hidden, rank, n_cols, rows, inject, combine, pdl):
        assert hidden == KH * KHW and n_cols <= NC * NCH and rows <= M
        self.hidden, self.rank, self.n_cols, self.rows = hidden, rank, n_cols, rows
        self.inject, self.combine, self.pdl = inject, combine, pdl
        self.ld = KHW + PAD
        self.rld = NC + 2  # pushed slot: NC partials + the sum of squares (+pad)

    def smem_bytes(self):
        def r(n):
            return -(-n // 128) * 128

        b = r(NC * self.ld * 2) + r(M * self.ld * 2)
        b += r(M * (KHW // VEC) * 4)
        b += r(WARPS * M * NC * 4) + r(M * NC * 4)
        b += r(CL * RPO * self.rld * 4) + r(M * 4) * 3
        return b + 256

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
        cute.arch.cluster_arrive_relaxed()

        smem = cutlass.utils.SmemAllocator()
        sW = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((NC, KHW), stride=(ld, 1)), byte_alignment=128
        )
        sA = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((M, KHW), stride=(ld, 1)), byte_alignment=128
        )
        sSs = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M, nvec), stride=(nvec, 1)), byte_alignment=128
        )
        sWp = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((WARPS, M, NC), stride=(M * NC, NC, 1)),
            byte_alignment=128,
        )
        sAcc = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M, NC), stride=(NC, 1)), byte_alignment=128
        )
        sRed = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((CL, RPO, rld), stride=(RPO * rld, rld, 1)),
            byte_alignment=128,
        )
        sInj = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M,), stride=(1,)), byte_alignment=128
        )
        sSg = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M,), stride=(1,)), byte_alignment=128
        )
        sSsr = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((M,), stride=(1,)), byte_alignment=128
        )

        # 1. Before the wait: the CTA's 24 weight rows over its K half; zero A rows past R.
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
        for i in cutlass.range_constexpr(-(-((M - rows) * KHW) // THREADS)):
            e = tidx + i * THREADS
            if e < (M - rows) * KHW:
                sA[rows + e // KHW, e % KHW] = cutlass.BFloat16(0.0)
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        # 2. Scalars; the K half's injection, sums of squares, y (bf16) into sA.
        if tidx < rows:
            if const_expr(self.inject):
                sInj[tidx] = mInj[tidx, s].to(cutlass.Float32)
                if const_expr(self.combine):
                    z = mGate[tidx].to(cutlass.Float32)
                    sSg[tidx] = 1.0 / (1.0 + cute.math.exp(-z))
        cute.arch.sync_threads()
        hc_vec = cute.logical_divide(mHc, (None, VEC))
        b_vec = cute.logical_divide(mBlock, (None, VEC))
        sh_vec = cute.logical_divide(mShared, (None, VEC))
        o_vec = cute.logical_divide(mHcOut, (None, VEC))
        nw_vec = cute.logical_divide(mNorm, (None, VEC))
        sa_vec = cute.logical_divide(sA, (None, VEC))
        items: cutlass.Constexpr = rows * nvec
        xr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        br = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        shr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        nwr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        outr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        yr = cute.make_rmem_tensor(cute.make_layout((VEC,), stride=(1,)), cutlass.BFloat16)
        for it in cutlass.range_constexpr(-(-items // THREADS)):
            c = tidx + it * THREADS
            if c < items:
                r = c // nvec
                p = c % nvec
                cute.autovec_copy(hc_vec[r, (None, col0 + p)], xr)
                cute.autovec_copy(nw_vec[0, (None, col0 + p)], nwr)
                ssum = cutlass.Float32(0.0)
                if const_expr(self.inject):
                    cute.autovec_copy(b_vec[r, (None, k0 // VEC + p)], br)
                    g = sInj[r]
                    if const_expr(self.combine):
                        cute.autovec_copy(sh_vec[r, (None, k0 // VEC + p)], shr)
                        sg = sSg[r]
                    for v in cutlass.range_constexpr(VEC):
                        bv = br[v].to(cutlass.Float32)
                        if const_expr(self.combine):
                            bv = (
                                cute.math.fma(sg, shr[v].to(cutlass.Float32), bv)
                                .to(cutlass.BFloat16)
                                .to(cutlass.Float32)
                            )
                        outr[v] = cute.math.fma(bv, g, xr[v].to(cutlass.Float32)).to(
                            cutlass.BFloat16
                        )
                    if nc == 0:
                        cute.autovec_copy(outr, o_vec[r, (None, col0 + p)])
                else:
                    for v in cutlass.range_constexpr(VEC):
                        outr[v] = xr[v]
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
        a_w = cute.make_tensor(sA.iterator + warp_i * KW, cute.make_layout((M, KW), stride=(ld, 1)))
        b_w = cute.make_tensor(
            sW.iterator + warp_i * KW, cute.make_layout((NC, KW), stride=(ld, 1))
        )
        thr = mma.get_slice(lane)
        tCsA = thr.partition_A(a_w)
        tCrA = thr.make_fragment_A(tCsA)
        cute.autovec_copy(tCsA, tCrA)
        tCsB = thr.partition_B(b_w)
        tCrB = thr.make_fragment_B(tCsB)
        cute.autovec_copy(tCsB, tCrB)
        acc = cute.make_rmem_tensor(thr.partition_shape_C((M, NC)), cutlass.Float32)
        acc.fill(0.0)
        cute.gemm(mma, acc, tCrA, tCrB, acc)
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
                    _st_dsmem_f32(_mapa(dst, o), v)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

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
    hidden, hc_count, rank, n_cols, n_wrows, rows, inject, combine, pdl, hc_stride, block_stride
):
    kern = _First2(
        hidden=hidden, rank=rank, n_cols=n_cols, rows=rows, inject=inject, combine=combine, pdl=pdl
    )
    dev, bf = "cuda", torch.bfloat16
    width = hidden * hc_count
    hc = torch.empty(rows, hc_stride, dtype=bf, device=dev)[:, :width]
    block = torch.empty(rows, block_stride, dtype=bf, device=dev)[:, :hidden]
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
    rank=320,
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
    pdl = get_env_enable_pdl()
    weight = first.t()
    assert weight.shape[0] >= NC * NCH
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
    )
    blk = block_out if has_inject else hyper_input.new_empty(rows, hidden)
    inj = inject if has_inject else hyper_input.new_empty(rows, hc_count)
    sh = shared if has_shared else hyper_input.new_empty(rows, hidden)
    gate = gate_logits.reshape(rows) if has_shared else hyper_input.new_empty(rows)
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
        for i in cutlass.range_constexpr(-(-((M - rows) * rank) // THREADS)):
            e = tidx + i * THREADS
            if e < (M - rows) * rank:
                sL[rows + e // rank, e % rank] = cutlass.BFloat16(0.0)
        gs = cutlass.Float32(0.0)
        if const_expr(self.fp4):
            gs = mGs[0]
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
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

        thr = mma.get_slice(tidx)
        tCsA = thr.partition_A(sL)
        tCrA = thr.make_fragment_A(tCsA)
        cute.autovec_copy(tCsA, tCrA)
        tCsB = thr.partition_B(sU)
        tCrB = thr.make_fragment_B(tCsB)
        cute.autovec_copy(tCsB, tCrB)
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
):
    """float64: the injection (and combine) rounded once as the 1-row kernel does, the grouped RMS,
    the first projection over `normed`, `low = silu(z)` and `2 sigmoid(z)` (z = logits / C) each
    rounded once, rstd in fp32."""
    dt = hyper_input.dtype
    rows, width = hyper_input.shape
    hidden = width // hc_count
    rank = RANK
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

    Fusion boundary: the combine, injection, grouped RMS statistics, first projection and
    activations. The routed experts' finalize is the runner's (the forward does not fold it at
    these row counts).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="site after a MoE block: combine + injection, 2 rows",
            spec=dict(rows=2, site="combine"),
        ),
        Cell(why="MoE site: injection, 8 rows", spec=dict(rows=8, site="inject")),
        Cell(
            why="norm-only site (stack start, MTP head), 16 rows, 320 cols",
            spec=dict(rows=16, site="norm"),
        ),
    )
    note = """
    Metadata consumed: none. Call sites: `DecodeTarget._hc_site` / `_hc_moe_site` at 2..16 rows
    (decode batches, MTP verify and draft-extend rows), followed by `qwen4_exp_hc_decode_rows_mix`.
    Grid (2 K halves, 4 streams, 14 N chunks of 24 columns), cluster (2, 4, 1), 8 warps; weights
    staged before the PDL wait. One compile per row count and variant.
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
        assert width // hc_count == KH * KHW, "hidden 2560"
        assert first.stride(0) == 1 and first.shape[1] >= NC * NCH, (
            "the padded [width, 384] .t() view"
        )
        assert (block_out is None) == (inject is None)
        assert (shared is None) == (gate_logits is None)


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
