# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2 large-message all-reduce + residual + RMSNorm (+ FP8) in one launch, one warp per row.

Persistent grid; warp ``gw`` of ``G`` owns rows gw, gw + G, ... of each half (rank r owns half r):
  A. push my partial of its peer-half rows to the peer's LA (lane l writes vectors l + 32 i: 512 B per warp
     store), then __syncwarp + system fence + the peer's flag A[gw] = epoch;
  B. wait for my flag A[gw]; for its own-half rows: sum = RN_bf16(FP32 mine + FP32 peer's) (NCCL's 2-rank sum),
     push the sums to the peer's LB, finalize; then the peer's flag B[gw] = epoch;
  C. wait for my flag B[gw]; finalize its peer-half rows from LB.
finalize reproduces ``reduce_fusion::rms_norm_kernel<bf16, false, true, true, false>`` (H / 8 threads x one vector)
bit for bit: BF16 residual add; per reference thread (= one vector here) an FP32 FMA chain over its 8 elements;
the reference warp k's butterfly (16..1) is this warp's butterfly over its k-th vectors; the reference's final
reduction (lanes below its warp count hold the warp sums, butterfly) is replayed; ``rsqrt(sum / H + eps)``;
``bf16((x * rstd) * w)``; FP8 = satfinite(norm_bf16 * (1 / scale)) as ``scaleMatrixPerTensorVec``.
Flags carry an epoch (the call count): nothing is cleared between calls; the last warp to finish advances it.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .mnnvl_allreduce_norm import (
    _add,
    _div_rn,
    _e4m3x2,
    _fma,
    _hi,
    _ld_s64,
    _ld_v4,
    _lo,
    _mul,
    _pack_bf16,
    _rsqrt,
    _shfl_bfly,
    _st_v2,
    _st_v4,
)
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

VEC = 8
REF_MAX_THREADS = 1024  # the reference launch's largest block (one vector per thread up to it)


def _asm(res, operands, text, constraints, side_effects=True, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc=None, ip=None):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def _ld_cg_v4(addr: Int64, *, loc=None, ip=None):
    """Four words at a 16-byte aligned global address, cached in L2 only (data another GPU wrote)."""
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [_v(Int64(addr), loc, ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_acquire_sys(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "ld.acquire.sys.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_release_sys(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(value), loc, ip)],
        "st.release.sys.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_sys(*, loc=None, ip=None) -> None:
    _asm(None, [], "fence.acq_rel.sys;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _atom_add_acq_rel_gpu(addr: Int64, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip), _v(Int32(value), loc, ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_gpu(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(addr), loc, ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_relaxed_gpu(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(Int64(addr), loc, ip), _v(Int32(value), loc, ip)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _add_bf16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """BF16x2 add (``__hadd2``: exact sum, one RN rounding)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(a), loc, ip), _v(Int32(b), loc, ip)],
            "{\n\t.reg .b32 one;\n\tmov.b32 one, 0x3f803f80;\n\tfma.rn.bf16x2 $0, $1, one, $2;\n\t}",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


class Tp2WarpAllReduceNormKernel:
    """Tensors: x [M, H] BF16, residual [M, H], weight [H] BF16, scale [1] FP32, residual_out / norm_out [M, H],
    quant_out [M, H] uint8, table int64 [8] (``Tp2AllReduceNorm.table``). ``ctas`` x ``warps`` warps."""

    @staticmethod
    def supports_hidden(hidden: int) -> bool:
        """The reference block spans the row in whole warps, one vector per thread."""
        return hidden % (VEC * 32) == 0 and 0 < hidden // VEC <= REF_MAX_THREADS

    def __init__(
        self,
        *,
        hidden: int,
        norm_eps: float,
        ctas: int,
        warps: int,
        norm_out: bool,
        quant: bool,
        use_pdl: bool,
    ):
        if not self.supports_hidden(hidden):
            raise ValueError(f"H {hidden} is not one reference block of {VEC}-element vectors")
        self.hidden = hidden
        self.vpl = hidden // VEC // 32  # vectors per lane = reference warps
        self.norm_eps = float(norm_eps)
        self.ctas = ctas
        self.warps = warps
        self.norm_out = norm_out
        self.quant = quant
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        gX: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        stream,
    ):
        self.kernel(gX, gRes, gW, gScale, gResOut, gOut, gQuant, gWs, rank).launch(
            grid=[self.ctas, 1, 1],
            block=[32 * self.warps, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gRes: cute.Tensor,
        gW: cute.Tensor,
        gScale: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gQuant: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        H: cutlass.Constexpr = self.hidden
        VPL: cutlass.Constexpr = self.vpl
        WPC: cutlass.Constexpr = self.warps
        G: cutlass.Constexpr = self.ctas * self.warps
        warp = tidx // 32
        lane = tidx % 32
        gw = cta * Int32(WPC) + warp
        m = cute.size(gRes, mode=[0])
        h0 = (m + 1) // 2
        own_base = rank * h0
        own_n = h0 + rank * (m - Int32(2) * h0)
        peer_base = h0 - own_base
        peer_n = m - own_n
        row_bytes = Int64(H * 2)
        lv = Int64(lane) * Int64(16)  # this lane's first vector, bytes

        tbl = Int64(gWs.iterator.toint())
        la_own = _ld_s64(tbl)
        lb_own = _ld_s64(tbl + Int64(8))
        fl_own = _ld_s64(tbl + Int64(16))
        la_peer = _ld_s64(tbl + Int64(24))
        lb_peer = _ld_s64(tbl + Int64(32))
        fl_peer = _ld_s64(tbl + Int64(40))
        epoch_p = _ld_s64(tbl + Int64(48))
        arrive_p = _ld_s64(tbl + Int64(56))
        xb = Int64(gX.iterator.toint())
        rb = Int64(gRes.iterator.toint())

        if const_expr(self.use_pdl):
            griddepcontrol_wait()
        ep = _ld_relaxed_gpu(epoch_p) + Int32(1)

        # ---- A: my partial of the peer half's rows -> peer LA ----
        r = gw
        while r < peer_n:
            src = xb + Int64(peer_base + r) * row_bytes + lv
            dst = la_peer + Int64(r) * row_bytes + lv
            buf = cute.make_rmem_tensor((4 * VPL,), Int32)
            for i in cutlass.range_constexpr(VPL):
                a0, a1, a2, a3 = _ld_v4(src + Int64(512 * i))
                buf[4 * i + 0] = a0
                buf[4 * i + 1] = a1
                buf[4 * i + 2] = a2
                buf[4 * i + 3] = a3
            for i in cutlass.range_constexpr(VPL):
                _st_v4(
                    dst + Int64(512 * i),
                    buf[4 * i + 0],
                    buf[4 * i + 1],
                    buf[4 * i + 2],
                    buf[4 * i + 3],
                )
            r = r + Int32(G)
        cute.arch.sync_warp()
        if lane == Int32(0):
            _fence_sys()
            _st_release_sys(fl_peer + Int64(gw) * Int64(4), ep)

        # ---- B: own half: wait for the peer's partials, sum, push the sums, finalize ----
        if lane == Int32(0):
            while _ld_acquire_sys(fl_own + Int64(gw) * Int64(4)) != ep:
                pass
        cute.arch.sync_warp()
        r = gw
        while r < own_n:
            row = own_base + r
            mine = cute.make_rmem_tensor((4 * VPL,), Int32)
            theirs = cute.make_rmem_tensor((4 * VPL,), Int32)
            for i in cutlass.range_constexpr(VPL):
                a0, a1, a2, a3 = _ld_v4(xb + Int64(row) * row_bytes + lv + Int64(512 * i))
                mine[4 * i + 0] = a0
                mine[4 * i + 1] = a1
                mine[4 * i + 2] = a2
                mine[4 * i + 3] = a3
                b0, b1, b2, b3 = _ld_cg_v4(la_own + Int64(r) * row_bytes + lv + Int64(512 * i))
                theirs[4 * i + 0] = b0
                theirs[4 * i + 1] = b1
                theirs[4 * i + 2] = b2
                theirs[4 * i + 3] = b3
            for w in cutlass.range_constexpr(4 * VPL):
                mine[w] = _pack_bf16(
                    _add(_lo(mine[w]), _lo(theirs[w])), _add(_hi(mine[w]), _hi(theirs[w]))
                )
            for i in cutlass.range_constexpr(VPL):
                _st_v4(
                    lb_peer + Int64(r) * row_bytes + lv + Int64(512 * i),
                    mine[4 * i + 0],
                    mine[4 * i + 1],
                    mine[4 * i + 2],
                    mine[4 * i + 3],
                )
            self._finalize(mine, row, lane, rb, gW, gScale, gResOut, gOut, gQuant)
            r = r + Int32(G)
        cute.arch.sync_warp()
        if lane == Int32(0):
            _fence_sys()
            _st_release_sys(fl_peer + Int64(G + gw) * Int64(4), ep)

        # ---- C: peer half: wait for the peer's sums, finalize ----
        if lane == Int32(0):
            while _ld_acquire_sys(fl_own + Int64(G + gw) * Int64(4)) != ep:
                pass
        cute.arch.sync_warp()
        r = gw
        while r < peer_n:
            sums = cute.make_rmem_tensor((4 * VPL,), Int32)
            for i in cutlass.range_constexpr(VPL):
                q0, q1, q2, q3 = _ld_cg_v4(lb_own + Int64(r) * row_bytes + lv + Int64(512 * i))
                sums[4 * i + 0] = q0
                sums[4 * i + 1] = q1
                sums[4 * i + 2] = q2
                sums[4 * i + 3] = q3
            self._finalize(sums, peer_base + r, lane, rb, gW, gScale, gResOut, gOut, gQuant)
            r = r + Int32(G)

        # ---- epoch: the last warp to finish advances it ----
        if const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        if lane == Int32(0):
            old = _atom_add_acq_rel_gpu(arrive_p, Int32(1))
            if old == Int32(G - 1):
                _st_relaxed_gpu(arrive_p, Int32(0))
                _st_relaxed_gpu(epoch_p, ep)

    @cute.jit
    def _finalize(self, sums, row, lane, rb, gW, gScale, gResOut, gOut, gQuant):
        """Residual add + RMSNorm (+ FP8) of ``row`` from its BF16 sums (lane holds vectors lane + 32 i)."""
        H: cutlass.Constexpr = self.hidden
        VPL: cutlass.Constexpr = self.vpl
        row_bytes = Int64(H * 2)
        base = Int64(row) * row_bytes + Int64(lane) * Int64(16)
        res = cute.make_rmem_tensor((4 * VPL,), Int32)
        for i in cutlass.range_constexpr(VPL):
            c0, c1, c2, c3 = _ld_v4(rb + base + Int64(512 * i))
            res[4 * i + 0] = _add_bf16x2(sums[4 * i + 0], c0)
            res[4 * i + 1] = _add_bf16x2(sums[4 * i + 1], c1)
            res[4 * i + 2] = _add_bf16x2(sums[4 * i + 2], c2)
            res[4 * i + 3] = _add_bf16x2(sums[4 * i + 3], c3)
        for i in cutlass.range_constexpr(VPL):
            _st_v4(
                Int64(gResOut.iterator.toint()) + base + Int64(512 * i),
                res[4 * i + 0],
                res[4 * i + 1],
                res[4 * i + 2],
                res[4 * i + 3],
            )
        # reference warp k = this warp's k-th vectors: per-thread FMA chain, butterfly 16..1
        wsum = cute.make_rmem_tensor((VPL,), Float32)
        for k in cutlass.range_constexpr(VPL):
            a = Float32(0.0)
            for j in cutlass.range_constexpr(4):
                a = _fma(_lo(res[4 * k + j]), _lo(res[4 * k + j]), a)
                a = _fma(_hi(res[4 * k + j]), _hi(res[4 * k + j]), a)
            for s in cutlass.range_constexpr(5):
                a = _add(a, _shfl_bfly(a, Int32(16 >> s)))
            wsum[k] = a
        # reference block reduction: lanes < VPL hold the warp sums (VPL warps), the rest 0, butterfly
        v = Float32(0.0)
        for k in cutlass.range_constexpr(VPL):
            if lane == Int32(k):
                v = wsum[k]
        for s in cutlass.range_constexpr(5):
            v = _add(v, _shfl_bfly(v, Int32(16 >> s)))
        rstd = _rsqrt(_add(_div_rn(v, Float32(float(H))), Float32(self.norm_eps)))
        inv = Float32(1.0)
        if const_expr(self.quant):
            inv = _div_rn(Float32(1.0), gScale[0])
        wb = Int64(gW.iterator.toint()) + Int64(lane) * Int64(16)
        for i in cutlass.range_constexpr(VPL):
            w0, w1, w2, w3 = _ld_v4(wb + Int64(512 * i))
            ww = (w0, w1, w2, w3)
            o = cute.make_rmem_tensor((4,), Int32)
            for j in cutlass.range_constexpr(4):
                o[j] = _pack_bf16(
                    _mul(_mul(_lo(res[4 * i + j]), rstd), _lo(ww[j])),
                    _mul(_mul(_hi(res[4 * i + j]), rstd), _hi(ww[j])),
                )
            if const_expr(self.norm_out):
                _st_v4(Int64(gOut.iterator.toint()) + base + Int64(512 * i), o[0], o[1], o[2], o[3])
            if const_expr(self.quant):
                q01 = _e4m3x2(_mul(_lo(o[0]), inv), _mul(_hi(o[0]), inv))
                q23 = _e4m3x2(_mul(_lo(o[1]), inv), _mul(_hi(o[1]), inv))
                q45 = _e4m3x2(_mul(_lo(o[2]), inv), _mul(_hi(o[2]), inv))
                q67 = _e4m3x2(_mul(_lo(o[3]), inv), _mul(_hi(o[3]), inv))
                _st_v2(
                    Int64(gQuant.iterator.toint())
                    + Int64(row) * Int64(H)
                    + Int64(lane) * Int64(8)
                    + Int64(256 * i),
                    q01 | (q23 << 16),
                    q45 | (q67 << 16),
                )
