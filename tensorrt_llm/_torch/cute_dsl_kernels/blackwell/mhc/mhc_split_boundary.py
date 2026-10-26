# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-sized (small M) mHC boundary kernels for DeepSeek-V4.1's lagged hyper-connections.

A V4.1 boundary does two independent things for every token:

* the *critical* half -- the previous sublayer's post update
  ``res[j, h] = post[j] * x[h] + sum_k comb[k, j] * res_prev[k, h]`` (rounded to BF16), the
  collapse with the previous sublayer's ``pre`` (``li[h] = sum_j pre[j] * res[j, h]``, BF16) and
  the next RMSNorm of ``li``. The sublayer that follows only needs this.
* the *coefficients* for the next boundary -- the 24-wide mixer GEMM over the 4 x hidden residual,
  its RMS scale, sigmoid gates and the Sinkhorn-normalized 4x4 ``comb``. Nothing reads them
  until the next boundary, one attention or MoE sublayer later.

:class:`MhcPostCollapseNormKernel` is the critical half as one launch, one CTA per token and one
8-wide column vector per thread: the residual and the norm weight are loaded before
``griddepcontrol.wait`` (they are not written by the kernel this one depends on -- the
all-reduce producing ``x``), everything else after it. The arithmetic and its rounding points
are those of ``fused_pmap_gemm_fma_ksplit`` (post update, BF16 residual) followed by
``mhcBigFuseKernel`` phase 2 (lagged collapse of the BF16 residual, BF16 ``li``, sum of
squares of the BF16 values, ``bf16(li * rsqrt * w)``).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from ..utils import griddepcontrol_launch_dependents, griddepcontrol_wait
from .mhc_boundary import (
    bf16x2_absmax4,
    bf16x2_hi,
    bf16x2_lo,
    ldg_f32,
    mxfp8_block,
    nvfp4_block,
    shfl_xor_u32,
    stg_u8,
    stg_v2_u32,
)

VEC = 8  # BF16 elements per thread (one 16-byte vector)
HC = 4  # hyper-connection multiplicity


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """``fma.rn.f32`` (no reliance on contraction of a separate mul/add)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``mul.rn.f32`` (kept separate from a following add)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _asm(res, operands, text: str, constraints: str, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_volatile_v4(addr: Int64, *, loc=None, ip=None):
    """Four 32-bit words at a 16-byte aligned global address, re-read from memory every time."""
    words = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_volatile_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_volatile_s64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.b64 $0, [$1];",
            "=l,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_s64(addr: Int64, value: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.b64 [$0], $1;",
        "l,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atomic_add_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_async_release_add_u32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """Release add that does not stall the issuing warp (TensorRT-LLM's ``LamportFlags`` arrival)."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "red.async.release.global.gpu.add.u32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _arrive_barrier_1(threads: Int32, *, wait: bool, loc=None, ip=None) -> None:
    """Named barrier 1 over ``threads`` threads: wait for every arrival, or only arrive."""
    _asm(
        None,
        [Int32(threads).ir_value(loc=loc, ip=ip)],
        "barrier.cta.sync 1, $0;" if wait else "barrier.cta.arrive 1, $0;",
        "r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16x2_neg_zero_to_zero(word: Int32, *, loc=None, ip=None) -> Int32:
    """The two BF16 values of a 32-bit word with each -0.0 replaced by +0.0."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 t, r;\n\t.reg .pred p;\n\tmov.b32 r, $1;\n\t"
            "and.b32 t, r, 0xffff0000;\n\tsetp.eq.b32 p, t, 0x80000000;\n\t@p and.b32 r, r, 0x0000ffff;\n\t"
            "and.b32 t, r, 0x0000ffff;\n\tsetp.eq.b32 p, t, 0x00008000;\n\t@p and.b32 r, r, 0xffff0000;\n\t"
            "mov.b32 $0, r;\n\t}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_dsmem_f32(smem_addr: Int32, cta_rank: Int32, value: Float32, *, loc=None, ip=None) -> None:
    """Store FP32 ``value`` at this CTA's shared-memory address ``smem_addr`` in cluster CTA ``cta_rank``."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int32(cta_rank).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 ra;\n\tmapa.shared::cluster.u32 ra, $0, $1;\n\t"
        "st.shared::cluster.f32 [ra], $2;\n\t}",
        "r,r,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# A 32-bit Lamport slot holding -0.0 has not been written yet.
_NEG_ZERO = -2147483648  # 0x80000000 as int32


class MhcPostCollapseNormKernel:
    """Post update + lagged collapse + RMSNorm for ``M`` tokens (one CTA per token).

    Tensors (all contiguous):
      x       [M, H]      bf16  output of the previous sublayer (after its all-reduce)
      res     [M, 4H]     bf16  previous residual stream (4 copies of H)
      post    [M, 4]      fp32  previous post gates
      comb    [M, 16]     fp32  previous comb, row-major [k][j]
      pre     [M, 4]      fp32  previous sublayer's pre (lagged collapse)
      weight  [1, H]      bf16  RMSNorm weight
      res_out [M, 4H]     bf16  new residual stream (written only with ``post_update``)
      out     [M, H]      bf16  RMSNorm(collapse) -- the next sublayer's input

    ``post_update=False``: ``res`` already is the boundary's residual (a resolved state);
    only the collapse and the norm run and ``x`` / ``post`` / ``comb`` / ``res_out`` are unused.
    """

    def __init__(
        self,
        *,
        hidden: int,
        norm_eps: float,
        use_pdl: bool,
        post_update: bool = True,
        quant: str | None = None,
    ):
        if hidden % (32 * VEC) or hidden // VEC > 1024:
            raise ValueError(f"unsupported hidden size {hidden}")
        if quant not in (None, "mxfp8", "nvfp4"):
            raise ValueError(f"unsupported output quantization {quant}")
        self.hidden = hidden
        # Also write ``out`` quantized for its consumer, byte for byte what the native quantizer
        # returns for the M rows:
        # "mxfp8": torch.ops.trtllm.mxfp8_quantize(out, True), E4M3 [M, H] at ``q_addr`` and UE8M0
        #     scales in the 128x4 swizzled layout at ``sf_addr`` (padding rows' scales not written);
        # "nvfp4": fp4_quantize(out, global scale at ``scale_addr``, 16, False, False), E2M1
        #     [M, H / 2] at ``q_addr`` and row-major E4M3 scales [M, H / 16] at ``sf_addr``.
        self.quant = quant
        self.threads = hidden // VEC
        self.num_warps = self.threads // 32
        self.norm_eps = float(norm_eps)
        self.use_pdl = use_pdl
        self.post_update = post_update

    @cute.jit
    def __call__(
        self,
        gX: cute.Tensor,
        gRes: cute.Tensor,
        gPost: cute.Tensor,
        gComb: cute.Tensor,
        gPre: cute.Tensor,
        gW: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        q_addr: Int64,
        sf_addr: Int64,
        scale_addr: Int64,
        stream,
    ) -> None:
        m = cute.size(gRes, mode=[0])
        copy_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_f32 = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Float32,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        store_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        self.kernel(
            gX,
            gRes,
            gPost,
            gComb,
            gPre,
            gW,
            gResOut,
            gOut,
            q_addr,
            sf_addr,
            scale_addr,
            copy_bf16,
            copy_f32,
            store_bf16,
        ).launch(
            grid=[m, 1, 1],
            block=[self.threads, 1, 1],
            smem=self.num_warps * 4,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gRes: cute.Tensor,
        gPost: cute.Tensor,
        gComb: cute.Tensor,
        gPre: cute.Tensor,
        gW: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        q_addr: Int64,
        sf_addr: Int64,
        scale_addr: Int64,
        copy_bf16: cute.CopyAtom,
        copy_f32: cute.CopyAtom,
        store_bf16: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        tok, _, _ = cute.arch.block_idx()
        vecs: cutlass.Constexpr = self.hidden // VEC

        gRes_v = cute.logical_divide(gRes, (None, VEC))
        gX_v = cute.logical_divide(gX, (None, VEC))
        gResOut_v = cute.logical_divide(gResOut, (None, VEC))
        gOut_v = cute.logical_divide(gOut, (None, VEC))
        gW_v = cute.logical_divide(gW, (None, VEC))
        gPost_v = cute.logical_divide(gPost, (None, 4))
        gComb_v = cute.logical_divide(gComb, (None, 4))
        gPre_v = cute.logical_divide(gPre, (None, 4))

        # ---- inputs not produced by the kernel this one depends on ----
        # With the post update the residual is the previous boundary's output, two or more launches
        # back. Without it the residual is a resolved state that the launch just before may have
        # written (Engram gate, post mapping): it is read after the dependency wait.
        r_bf = cute.make_rmem_tensor(cute.make_layout((HC, VEC), stride=(VEC, 1)), BFloat16)
        if const_expr(self.post_update):
            for j in cutlass.range_constexpr(HC):
                cute.copy(copy_bf16, gRes_v[tok, (None, j * vecs + tidx)], r_bf[j, None])
        w_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        cute.copy(copy_bf16, gW_v[0, (None, tidx)], w_bf)
        gscale = Float32(0.0)
        if const_expr(self.quant == "nvfp4"):
            gscale = ldg_f32(scale_addr)

        if const_expr(self.use_pdl):
            griddepcontrol_wait()
        if const_expr(not self.post_update):
            for j in cutlass.range_constexpr(HC):
                cute.copy(copy_bf16, gRes_v[tok, (None, j * vecs + tidx)], r_bf[j, None])

        x_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        if const_expr(self.post_update):
            cute.copy(copy_bf16, gX_v[tok, (None, tidx)], x_bf)
        pre, post, comb = self._load_coefficients(tok, gPost_v, gComb_v, gPre_v, copy_f32)
        self._boundary(
            tok,
            tidx,
            tidx,
            1,
            x_bf,
            r_bf,
            w_bf,
            pre,
            post,
            comb,
            gResOut_v,
            gOut_v,
            store_bf16,
            self.use_pdl,
            q_addr,
            sf_addr,
            gscale,
        )

    @cute.jit
    def _load_coefficients(
        self,
        tok,
        gPost_v: cute.Tensor,
        gComb_v: cute.Tensor,
        gPre_v: cute.Tensor,
        copy_f32: cute.CopyAtom,
    ):
        """The token's lagged ``pre`` [4] and, with ``post_update``, ``post`` [4] / ``comb`` [4, 4] (FP32)."""
        pre = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        cute.copy(copy_f32, gPre_v[tok, (None, 0)], pre)
        post = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        comb = cute.make_rmem_tensor(cute.make_layout((HC, 4), stride=(4, 1)), Float32)
        if const_expr(self.post_update):
            cute.copy(copy_f32, gPost_v[tok, (None, 0)], post)
            for k in cutlass.range_constexpr(HC):
                cute.copy(copy_f32, gComb_v[tok, (None, k)], comb[k, None])
        return pre, post, comb

    @cute.jit
    def _boundary(
        self,
        tok,
        tidx,
        vi,
        ctas: cutlass.Constexpr,
        x_bf: cute.Tensor,
        r_bf: cute.Tensor,
        w_bf: cute.Tensor,
        pre: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        gResOut_v: cute.Tensor,
        gOut_v: cute.Tensor,
        store_bf16: cute.CopyAtom,
        trigger_dependents: cutlass.Constexpr,
        q_addr: Int64,
        sf_addr: Int64,
        gscale: Float32,
    ) -> None:
        """Everything after ``x`` is known: post update, residual store, collapse, RMSNorm.

        ``vi`` is the thread's vector within the token; with ``ctas`` > 1 a cluster of that many
        CTAs shares the token and the sum of squares is reduced through distributed shared
        memory, in the same warp order as one CTA.
        """
        warp = tidx // 32
        lane = tidx % 32
        vecs: cutlass.Constexpr = self.hidden // VEC
        num_warps: cutlass.Constexpr = self.num_warps // ctas

        res_bf = cute.make_rmem_tensor(cute.make_layout((HC, VEC), stride=(VEC, 1)), BFloat16)
        if const_expr(self.post_update):
            # new_r[j] = post[j] * x + sum_k comb[k][j] * r[k]   (FP32, then BF16)
            for v in cutlass.range_constexpr(VEC):
                xf = x_bf[v].to(Float32)
                acc0 = _mul(post[0], xf)
                acc1 = _mul(post[1], xf)
                acc2 = _mul(post[2], xf)
                acc3 = _mul(post[3], xf)
                for k in cutlass.range_constexpr(HC):
                    rf = r_bf[k, v].to(Float32)
                    acc0 = _fma(comb[k, 0], rf, acc0)
                    acc1 = _fma(comb[k, 1], rf, acc1)
                    acc2 = _fma(comb[k, 2], rf, acc2)
                    acc3 = _fma(comb[k, 3], rf, acc3)
                res_bf[0, v] = acc0.to(BFloat16)
                res_bf[1, v] = acc1.to(BFloat16)
                res_bf[2, v] = acc2.to(BFloat16)
                res_bf[3, v] = acc3.to(BFloat16)
            for j in cutlass.range_constexpr(HC):
                cute.copy(store_bf16, res_bf[j, None], gResOut_v[tok, (None, j * vecs + vi)])
        else:
            for j in cutlass.range_constexpr(HC):
                for v in cutlass.range_constexpr(VEC):
                    res_bf[j, v] = r_bf[j, v]

        if const_expr(trigger_dependents):
            griddepcontrol_launch_dependents()

        # ---- lagged collapse: li = sum_j pre[j] * res[j] (BF16), sum of squares of BF16 li ----
        li_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        sum_sq = Float32(0.0)
        for v in cutlass.range_constexpr(VEC):
            acc = _mul(pre[0], res_bf[0, v].to(Float32))
            for j in cutlass.range_constexpr(1, HC):
                acc = _fma(pre[j], res_bf[j, v].to(Float32), acc)
            li_bf[v] = acc.to(BFloat16)
            lf = li_bf[v].to(Float32)
            sum_sq = _fma(lf, lf, sum_sq)

        sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        smem = SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((num_warps,)), byte_alignment=16)
        if lane == 0:
            red[warp] = sum_sq
        total = Float32(0.0)
        if const_expr(ctas == 1):
            cute.arch.sync_threads()
            for w in cutlass.range_constexpr(num_warps):
                total = total + red[w]
        else:
            # Every CTA holds all ctas x num_warps partials: lane 0 of each warp pushes its partial
            # into slot (this CTA, warp) of every CTA, one cluster barrier publishes them, and the
            # sum runs over local shared memory in the same order.
            all_red = smem.allocate_tensor(
                Float32, cute.make_layout((ctas * num_warps,)), byte_alignment=16
            )
            me = (vi - tidx) // (num_warps * 32)
            if lane == 0:
                slot_addr = Int32(all_red.iterator.toint()) + Int32(4) * (me * num_warps + warp)
                for p in cutlass.range_constexpr(ctas):
                    _st_dsmem_f32(slot_addr, Int32(p), sum_sq)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            for p in cutlass.range_constexpr(ctas):
                for w in cutlass.range_constexpr(num_warps):
                    total = total + all_red[p * num_warps + w]
        rstd = cute.math.rsqrt(total / Float32(self.hidden) + Float32(self.norm_eps))

        out_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        for v in cutlass.range_constexpr(VEC):
            o = _mul(_mul(li_bf[v].to(Float32), rstd), w_bf[v].to(Float32))
            out_bf[v] = o.to(BFloat16)
        cute.copy(store_bf16, out_bf, gOut_v[tok, (None, vi)])
        if const_expr(self.quant == "mxfp8"):
            self._store_mxfp8(tok, vi, out_bf, q_addr, sf_addr)
        elif const_expr(self.quant == "nvfp4"):
            self._store_nvfp4(tok, vi, out_bf, q_addr, sf_addr, gscale)

    @cute.jit
    def _store_mxfp8(self, tok, vi, out_bf: cute.Tensor, q_addr: Int64, sf_addr: Int64) -> None:
        """MXFP8 of the token's 8 output values: the 4 threads of a 32-element block share its
        absolute maximum, its first thread stores the scale."""
        vals = [out_bf[v].to(Float32) for v in range(VEC)]
        amax = Float32(0.0)
        for v in cutlass.range_constexpr(VEC):
            amax = cute.arch.fmax(amax, cute.arch.fmax(vals[v], -vals[v]))
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
        w0, w1, sf = mxfp8_block(amax, vals)
        stg_v2_u32(q_addr + Int64(tok) * Int64(self.hidden) + Int64(vi) * Int64(VEC), w0, w1)
        k_tiles: cutlass.Constexpr = self.hidden // 128
        if vi % 4 == 0:
            k = vi // 4
            off = (
                (tok // 128) * (k_tiles * 512)
                + (k // 4) * 512
                + (tok % 32) * 16
                + ((tok % 128) // 32) * 4
                + k % 4
            )
            stg_u8(sf_addr + Int64(off), sf)

    @cute.jit
    def _store_nvfp4(
        self, tok, vi, out_bf: cute.Tensor, q_addr: Int64, sf_addr: Int64, gscale: Float32
    ) -> None:
        """NVFP4 of the token's 8 output values: the 2 threads of a 16-element block share its
        absolute maximum, the first stores the scale."""
        vals = [out_bf[v].to(Float32) for v in range(VEC)]
        words = cute.recast_tensor(out_bf, Int32)
        amax = bf16x2_absmax4([words[p] for p in range(VEC // 2)])
        word, sf = nvfp4_block(amax, shfl_xor_u32(amax, 1), gscale, vals)
        _st_s32(q_addr + Int64(tok) * Int64(self.hidden // 2) + Int64(vi) * Int64(VEC // 2), word)
        if vi % 2 == 0:
            stg_u8(sf_addr + Int64(tok) * Int64(self.hidden // 16) + Int64(vi // 2), sf)


class MhcAllReduceBoundaryKernel(MhcPostCollapseNormKernel):
    """One-shot all-reduce of the sublayer's TP partial output fused with the boundary's critical half.

    The all-reduce is TensorRT-LLM's one-shot Lamport protocol on the IPC workspace of
    ``get_allreduce_workspace`` (``allreduce_fusion_kernel_oneshot_lamport``, FP32 accumulation in
    rank order), so launches of this kernel and of the library's all-reduce kernels share and
    advance the same triple buffer. ``gWs`` is the workspace slice ``[2N, 3N + 2)``: the Lamport
    buffer of every rank, the flag words and the layout words.

    Each CTA (one token) pushes its 16-byte slices to every rank, clears the slot the previous
    launch used, polls its own buffer until the ``N`` slices of the token have arrived, sums them
    and continues as :class:`MhcPostCollapseNormKernel` with that ``x``.

    With ``mnnvl`` the group spans trays and the all-reduce is TensorRT-LLM's MNNVL one-shot
    Lamport protocol on the fabric-memory workspace of ``MNNVLAllReduce``
    (``oneshotAllreduceFusionKernel``), sharing its triple buffer and control flags. ``gWs`` is then
    ``[local buffer, multicast buffer, flags]``: one store to the multicast buffer reaches every
    rank, the dirty buffer is cleared as the flags describe, BF16 ``-0.0`` inputs are pushed as
    ``+0.0`` and the sum is the FP32 sum in rank order rounded to BF16 once -- the library
    kernel's -- so the attention boundary equals the MNNVL all-reduce followed by
    :class:`MhcPostCollapseNormKernel` bit for bit.

    With ``moe_topk`` the partial is not read from ``gX`` but finalized from an un-finalized MoE
    output (as TensorRT-LLM's MoE-finalize all-reduce): ``sum_k scale[t, k] * fc2[perm[t, k]]`` plus
    the shared-expert output, accumulated in FP32 and rounded to BF16 once (``perm < 0`` rows are
    skipped). ``gX`` is then unused; without it ``gFc2`` / ``gScale`` / ``gPerm`` / ``gShared`` are.
    """

    def __init__(
        self,
        *,
        hidden: int,
        norm_eps: float,
        nranks: int,
        use_pdl: bool,
        moe_topk: int = 0,
        ctas_per_token: int = 1,
        mnnvl: bool = False,
        quant: str | None = None,
    ):
        super().__init__(
            hidden=hidden, norm_eps=norm_eps, use_pdl=use_pdl, post_update=True, quant=quant
        )
        self.mnnvl = mnnvl
        if nranks not in (2, 4, 8):
            raise ValueError(f"unsupported all-reduce group size {nranks}")
        if ctas_per_token not in (1, 2, 4) or self.threads % (32 * ctas_per_token):
            raise ValueError(f"unsupported CTAs per token {ctas_per_token}")
        self.nranks = nranks
        self.moe_topk = moe_topk
        # A cluster of CTAs per token spreads the token's pushes, polls and math over SMs.
        self.ctas = ctas_per_token
        self.threads_cta = self.threads // ctas_per_token

    @cute.jit
    def __call__(
        self,
        gX: cute.Tensor,
        gRes: cute.Tensor,
        gPost: cute.Tensor,
        gComb: cute.Tensor,
        gPre: cute.Tensor,
        gW: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        gFc2: cute.Tensor,
        gScale: cute.Tensor,
        gPerm: cute.Tensor,
        gShared: cute.Tensor,
        q_addr: Int64,
        sf_addr: Int64,
        scale_addr: Int64,
        stream,
    ) -> None:
        m = cute.size(gRes, mode=[0])
        copy_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_f32 = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Float32,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        store_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        self.ar_kernel(
            gX,
            gRes,
            gPost,
            gComb,
            gPre,
            gW,
            gResOut,
            gOut,
            gWs,
            rank,
            gFc2,
            gScale,
            gPerm,
            gShared,
            q_addr,
            sf_addr,
            scale_addr,
            copy_bf16,
            copy_f32,
            store_bf16,
        ).launch(
            grid=[m * self.ctas, 1, 1],
            block=[self.threads_cta, 1, 1],
            cluster=[self.ctas, 1, 1],
            smem=self.num_warps * 4 * (2 if self.ctas > 1 else 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def ar_kernel(
        self,
        gX: cute.Tensor,
        gRes: cute.Tensor,
        gPost: cute.Tensor,
        gComb: cute.Tensor,
        gPre: cute.Tensor,
        gW: cute.Tensor,
        gResOut: cute.Tensor,
        gOut: cute.Tensor,
        gWs: cute.Tensor,
        rank: Int32,
        gFc2: cute.Tensor,
        gScale: cute.Tensor,
        gPerm: cute.Tensor,
        gShared: cute.Tensor,
        q_addr: Int64,
        sf_addr: Int64,
        scale_addr: Int64,
        copy_bf16: cute.CopyAtom,
        copy_f32: cute.CopyAtom,
        store_bf16: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        ctas: cutlass.Constexpr = self.ctas
        tok = bid // ctas
        m = grid // ctas
        # The thread's 8-wide vector within the token.
        vi = (bid % ctas) * self.threads_cta + tidx
        vecs: cutlass.Constexpr = self.hidden // VEC
        nranks: cutlass.Constexpr = self.nranks
        neg0 = Int32(_NEG_ZERO)

        gRes_v = cute.logical_divide(gRes, (None, VEC))
        gX_v = cute.logical_divide(gX, (None, VEC))
        gResOut_v = cute.logical_divide(gResOut, (None, VEC))
        gOut_v = cute.logical_divide(gOut, (None, VEC))
        gW_v = cute.logical_divide(gW, (None, VEC))
        gPost_v = cute.logical_divide(gPost, (None, 4))
        gComb_v = cute.logical_divide(gComb, (None, 4))
        gPre_v = cute.logical_divide(gPre, (None, 4))

        # ---- inputs not produced by the kernel this one depends on ----
        r_bf = cute.make_rmem_tensor(cute.make_layout((HC, VEC), stride=(VEC, 1)), BFloat16)
        for j in cutlass.range_constexpr(HC):
            cute.copy(copy_bf16, gRes_v[tok, (None, j * vecs + vi)], r_bf[j, None])
        w_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        cute.copy(copy_bf16, gW_v[0, (None, vi)], w_bf)
        gscale = Float32(0.0)
        if const_expr(self.quant == "nvfp4"):
            gscale = ldg_f32(scale_addr)

        if const_expr(self.use_pdl):
            griddepcontrol_wait()
        # The coefficients come from the coefficient stream, which this launch may depend on
        # programmatically: only the wait makes their values visible.
        pre, post, comb = self._load_coefficients(tok, gPost_v, gComb_v, gPre_v, copy_f32)

        # ---- this rank's slice of the token, or the routing of its MoE rows: loaded with the
        # Lamport state below, in one round trip ----
        x_in = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        if const_expr(self.moe_topk > 0):
            perm = cute.make_rmem_tensor(cute.make_layout((self.moe_topk,)), Int32)
            weight = cute.make_rmem_tensor(cute.make_layout((self.moe_topk,)), Float32)
            for k in cutlass.range_constexpr(self.moe_topk):
                perm[k] = Int32(gPerm[tok, k])
                weight[k] = gScale[tok, k].to(Float32)
        else:
            cute.copy(copy_bf16, gX_v[tok, (None, vi)], x_in)

        # ---- Lamport state of this launch (read before any CTA bumps the arrival counter) ----
        if const_expr(self.mnnvl):
            # Flags (uint32): current, dirty, bytes per buffer, dirty stages, 4 x dirty bytes per
            # stage, arrival counter.
            own = Int64(gWs[0])
            mcast = Int64(gWs[1])
            flags = Int64(gWs[2])
            cur = _ld_volatile_s32(flags)
            dirty = _ld_volatile_s32(flags + Int64(4))
            buf_bytes = Int64(_ld_volatile_s32(flags + Int64(8)))
            dirty_stages = _ld_volatile_s32(flags + Int64(12))
            data_off = Int64(cur) * buf_bytes
            # Warp 0 counts the CTA once all its threads have read the flags; the other warps go on.
            if tidx < 32:
                _arrive_barrier_1(Int32(self.threads_cta), wait=True)
                if tidx == 0:
                    _red_async_release_add_u32(flags + Int64(32), Int32(1))
            else:
                _arrive_barrier_1(Int32(self.threads_cta), wait=False)
        else:
            flag_addr = Int64(gWs[nranks])
            layout_addr = Int64(gWs[nranks + 1])
            flag = _ld_volatile_s32(flag_addr + Int64(8))
            clear_size = _ld_volatile_s64(layout_addr)
            comm_size = _ld_volatile_s64(layout_addr + Int64(8))
            data_off = Int64(flag % 3) * comm_size
            clear_off = Int64((flag + 2) % 3) * comm_size
            own = Int64(gWs[rank])
            if tidx < 32:
                _arrive_barrier_1(Int32(self.threads_cta), wait=True)
                if tidx == 0:
                    _atomic_add_s32(flag_addr, Int32(1))
            else:
                _arrive_barrier_1(Int32(self.threads_cta), wait=False)
        if const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()

        # ---- push this rank's slice of the token to every rank ----
        tot = Int64(m) * Int64(vecs)
        idx = Int64(tok) * Int64(vecs) + Int64(vi)
        if const_expr(self.moe_topk > 0):
            # MoE finalize: FP32 sum of the routed expert rows and the shared-expert output. All
            # rows are loaded before the sum (a skipped row, ``perm < 0``, reads row 0 unused).
            gFc2_v = cute.logical_divide(gFc2, (None, VEC))
            gShared_v = cute.logical_divide(gShared, (None, VEC))
            rows = cute.make_rmem_tensor(
                cute.make_layout((self.moe_topk, VEC), stride=(VEC, 1)), BFloat16
            )
            for k in cutlass.range_constexpr(self.moe_topk):
                p = perm[k]
                if p < 0:
                    p = Int32(0)
                cute.copy(copy_bf16, gFc2_v[p, (None, vi)], rows[k, None])
            shared = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
            cute.copy(copy_bf16, gShared_v[tok, (None, vi)], shared)
            acc = cute.make_rmem_tensor(cute.make_layout((VEC,)), Float32)
            for v in cutlass.range_constexpr(VEC):
                acc[v] = Float32(0.0)
            for k in cutlass.range_constexpr(self.moe_topk):
                if perm[k] >= 0:
                    for v in cutlass.range_constexpr(VEC):
                        acc[v] = _fma(rows[k, v].to(Float32), weight[k], acc[v])
            for v in cutlass.range_constexpr(VEC):
                x_in[v] = (acc[v] + shared[v].to(Float32)).to(BFloat16)
        x_words = cute.recast_tensor(x_in, Int32)
        w0 = x_words[0]
        w1 = x_words[1]
        w2 = x_words[2]
        w3 = x_words[3]
        if const_expr(self.mnnvl):
            # Every BF16 -0.0 becomes +0.0 (the library kernel's sanitizing), then one multicast
            # store reaches every rank.
            w0 = _bf16x2_neg_zero_to_zero(w0)
            w1 = _bf16x2_neg_zero_to_zero(w1)
            w2 = _bf16x2_neg_zero_to_zero(w2)
            w3 = _bf16x2_neg_zero_to_zero(w3)
            slot = (Int64(rank) * tot + idx) * Int64(16)
            _st_v4(mcast + data_off + slot, w0, w1, w2, w3)
        else:
            if w0 == neg0:
                w0 = Int32(0)
            if w1 == neg0:
                w1 = Int32(0)
            if w2 == neg0:
                w2 = Int32(0)
            if w3 == neg0:
                w3 = Int32(0)
            slot = (Int64(rank) * tot + idx) * Int64(16)
            for r in cutlass.range_constexpr(nranks):
                _st_v4(Int64(gWs[r]) + data_off + slot, w0, w1, w2, w3)

        # ---- clear the slot the previous launch used ----
        if const_expr(self.mnnvl):
            # Stage s of the dirty buffer holds the previous launch's bytes (flags[4 + s]).
            for s in cutlass.range_constexpr(4):
                if Int32(s) < dirty_stages:
                    stage_bytes = buf_bytes // Int64(dirty_stages)
                    stage = own + (Int64(dirty) * Int64(dirty_stages) + Int64(s)) * stage_bytes
                    clear_vecs = Int64(_ld_volatile_s32(flags + Int64(16 + 4 * s))) // Int64(16)
                    c = idx
                    while c < clear_vecs:
                        _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                        c = c + tot
        else:
            # Its size in BF16 elements.
            clear_vecs = clear_size // Int64(VEC)
            c = idx
            while c < clear_vecs:
                _st_v4(own + clear_off + c * Int64(16), neg0, neg0, neg0, neg0)
                c = c + tot

        # ---- wait for the token's slice of every rank ----
        got = cute.make_rmem_tensor(cute.make_layout((nranks, 4), stride=(4, 1)), Int32)
        pending = Int32(1)
        while pending != 0:
            pending = Int32(0)
            for r in cutlass.range_constexpr(nranks):
                a, b, cc, d = _ld_volatile_v4(own + data_off + (Int64(r) * tot + idx) * Int64(16))
                got[r, 0] = a
                got[r, 1] = b
                got[r, 2] = cc
                got[r, 3] = d
                if a == neg0:
                    pending = Int32(1)
                if b == neg0:
                    pending = Int32(1)
                if cc == neg0:
                    pending = Int32(1)
                if d == neg0:
                    pending = Int32(1)

        # ---- x = BF16(FP32 sum in rank order), as the library kernel's FP32 accumulation ----
        x_bf = cute.make_rmem_tensor(cute.make_layout((VEC,)), BFloat16)
        for i in cutlass.range_constexpr(4):
            lo = bf16x2_lo(got[0, i])
            hi = bf16x2_hi(got[0, i])
            for r in cutlass.range_constexpr(1, nranks):
                lo = lo + bf16x2_lo(got[r, i])
                hi = hi + bf16x2_hi(got[r, i])
            x_bf[2 * i] = lo.to(BFloat16)
            x_bf[2 * i + 1] = hi.to(BFloat16)

        self._boundary(
            tok,
            tidx,
            vi,
            ctas,
            x_bf,
            r_bf,
            w_bf,
            pre,
            post,
            comb,
            gResOut_v,
            gOut_v,
            store_bf16,
            False,
            q_addr,
            sf_addr,
            gscale,
        )

        # ---- hand the triple buffer to the next launch once every CTA has read the flag ----
        if bid == 0:
            if tidx == 0:
                if const_expr(self.mnnvl):
                    arrived = _ld_acquire_s32(flags + Int64(32))
                    while arrived != grid:
                        arrived = _ld_acquire_s32(flags + Int64(32))
                    # This launch's buffer becomes the dirty one (one stage, all ranks' slices).
                    _st_s32(flags, (cur + 1) % 3)
                    _st_s32(flags + Int64(4), cur)
                    _st_s32(flags + Int64(12), Int32(1))
                    _st_s32(flags + Int64(16), Int32(m) * Int32(self.hidden * nranks * 2))
                    _st_s32(flags + Int64(20), Int32(0))
                    _st_s32(flags + Int64(24), Int32(0))
                    _st_s32(flags + Int64(28), Int32(0))
                    _st_s32(flags + Int64(32), Int32(0))
                else:
                    arrived = _ld_volatile_s32(flag_addr)
                    while arrived != grid:
                        arrived = _ld_volatile_s32(flag_addr)
                    _st_s32(flag_addr + Int64(8), (flag + 1) % 3)
                    _st_s64(layout_addr, Int64(m) * Int64(self.hidden * nranks))
                    _st_s32(flag_addr, Int32(0))
