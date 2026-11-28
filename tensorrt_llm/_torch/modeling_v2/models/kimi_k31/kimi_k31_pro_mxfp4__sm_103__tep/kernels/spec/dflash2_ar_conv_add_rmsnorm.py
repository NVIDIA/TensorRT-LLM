# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0): flashinfer/norm/kernels/fused_add_rmsnorm.py
"""The DFlash2 drafter's sublayer epilogue finishing the row-parallel projection's all-reduce (CuTe DSL): the
two-shot MNNVL all-reduce whose input every rank pushed
(``kimi_k31_pro_mxfp4__sm_103__tep/kernels/comm/bf16_ar_push_gemm``), the block convolution of its rows, the
residual add and the RMSNorm, in one launch.

``kimi_k31_pro_mxfp4__sm_103__tep/kernels/spec/dflash2_conv_add_rmsnorm`` reads the all-reduced rows from a tensor that
the Lamport finish kernel (``kernels/comm/lamport_ar``) wrote. Here the kernel finishes the all-reduce itself, on
the same workspace and with the same arithmetic as the finish kernel:

* the rows this rank owns in the two-shot call (``t % W == rank``) are reduced by every thread of the grid, one
  16-byte vector at a time: the ``W`` scatter slots polled, summed in FP32 in rank order (``chunk`` ranks per partial
  sum), rounded to BF16, ``-0`` sent as ``+0`` and multicast into every rank's broadcast stage;
* each CTA then polls the broadcast rows of its row ``t`` and, for the convolution's tap, ``t - 1`` (only inside the
  block), and runs the convolution, add and norm of ``dflash2_conv_add_rmsnorm`` on them;
* the previous call's buffer is re-armed and the rotation state advanced by the grid's last CTA, as the finish
  kernel does, so fused and separate calls can alternate on one workspace.

A row's own reduction never waits for another row, so no CTA waits on a CTA that waits on it. Every arithmetic step
is the finish kernel's followed by the epilogue's: the outputs are bitwise the push GEMM + finish + epilogue chain.

With PDL the grid does all of it before its ``griddepcontrol.wait`` (the push GEMM releases its
dependents after its own wait, so the rotation state is final and every input older than the GEMM is complete),
and only stores and advances the state after it: its stores may land in memory a grid between the GEMM and this one
still reads.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel as _FiFusedAddRMSNorm
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel as _FiRMSNorm
from flashinfer.norm.utils import get_cutlass_dtype, get_sm_version, row_reduce_sum_multirow

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.spec.dflash2_grouped_conv import (
    _add_mul_select_rn,
    _add_rn,
    _mul_rn,
)


@dsl_user_op
def _unpack_bf16x2(word, *, loc=None, ip=None):
    """``(float(lo), float(hi))`` of a packed BF16 pair (exact)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\nand.b32 t, $2, 0xffff0000;\nmov.b32 $1, t;\n}",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


class DFlash2ArConvAddRMSNormKernel(_FiFusedAddRMSNorm):
    """``(out, residual) = fused_add_rmsnorm(conv(all_reduce(pushed); base, delta), residual)`` per row: one CTA per
    row (FlashInfer's tiling at ``H``, no cluster split), every thread one share of the owned rows' reduction."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        taps: int,
        group_size: int,
        sm_version: int,
        world: int,
        chunk: int,
    ):
        super().__init__(dtype, H, 0.0, sm_version=sm_version)
        if dtype.width != 16 or self.vec_size != 8:
            raise ValueError("16-bit rows in 16-byte vectors only")
        if group_size % self.vec_size or H % group_size:
            raise ValueError(
                f"group_size ({group_size}) must be a multiple of {self.vec_size} dividing {H}"
            )
        if taps not in (1, 2):
            raise ValueError(
                f"taps 1 or 2 (the finished rows are this row and the one before); got {taps}"
            )
        if self.cluster_n != 1 or self.rows_per_block != 1 or self.cols_per_tile != H:
            raise ValueError(f"H={H}: one CTA per row without a cluster split only")
        if not 2 <= world <= 64:
            raise ValueError(f"world {world} not served")
        self.taps = taps
        self.group_size = group_size
        self.world = world
        self.chunk = chunk
        self.use_async_copy = False

    @cute.jit
    def __call__(
        self,
        mD: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        M: Int32,
        block_size: Int32,
        rank: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = _FiRMSNorm._make_tv_layout(
            self.threads_per_row, self.rows_per_block, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(
            mD,
            mB,
            mR,
            mW,
            mO,
            uc_local,
            mc_base,
            flags,
            M,
            block_size,
            rank,
            eps,
            enable_pdl,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mD: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        M: Int32,
        block_size: Int32,
        rank: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H = self.H
        W = self.world
        vec = self.vec_size
        nvec = cutlass.const_expr(self.num_vec_blocks)
        threads = self.num_threads

        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((rows_per_block, warps_per_row)), byte_alignment=4
        )

        # ===== The all-reduce: state, the previous buffer re-armed, this rank's rows reduced and multicast =====
        cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
        cute.arch.barrier()
        if cutlass.const_expr(not enable_pdl):
            if tidx == 0:
                _lam.arrive_finish(flags, M, cur, nbytes, M, W, H, False, 0)
        gtid = bidx * threads + tidx
        nthreads = M * threads
        _lam.clear_dirty_stages(uc_local, dirty, nbytes, dstages, c0, c1, c2, c3, gtid, nthreads)
        scatter = _lam.stage_addr(uc_local, cur, Int32(0), 2, nbytes)
        bcast_mc = _lam.stage_addr(mc_base, cur, Int32(1), 2, nbytes)
        bcast = _lam.stage_addr(uc_local, cur, Int32(1), 2, nbytes)
        V = H // 8
        n_own = (M + (W - 1) - rank) // W
        for i in cutlass.range(gtid, n_own * V, nthreads, unroll=1):
            k = i // V
            v = i - k * V
            acc = _lam.poll_reduce_v4(scatter + Int64(k * W * H + v * 8) * 2, H * 2, W, self.chunk)
            o = [
                _lam.sanitize_bf16x2(_lam.pack_bf16x2(acc[2 * q], acc[2 * q + 1])) for q in range(4)
            ]
            _lam.multimem_st_v4(bcast_mc + Int64((rank + k * W) * H + v * 8) * 2, *o)

        # ===== This row's tile: coordinates, residual, weight =====
        idX = cute.make_identity_tensor(mR.shape)
        gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
        gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))
        bcast_l = cute.make_layout((tiler_mn[0],), stride=(0,))
        mW_2d = cute.make_tensor(mW.iterator, cute.prepend(mW.layout, bcast_l))
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mR.element_type, num_bits_per_copy=self.copy_bits
        )
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thr_copy = tiled_copy.get_slice(tidx)
        tXcX = thr_copy.partition_S(cX)
        tRgR = thr_copy.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        tWgW = thr_copy.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXgO = thr_copy.partition_D(gO)
        tRgO = thr_copy.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)
        tCrC = cute.make_fragment_like(tRgR)
        cute.copy(copy_atom, tRgR, tRrR)
        cute.copy(copy_atom, tWgW, tWrW)

        row = bidx
        pos = row % block_size
        prev = Int32(0)
        if cutlass.const_expr(self.taps > 1):
            if pos >= 1:
                prev = Int32(1)

        # ===== The finished rows t (and t - 1 inside the block) from the broadcast stage, all vectors in flight =====
        words = cute.make_rmem_tensor((nvec * 4,), Int32)
        pwords = cute.make_rmem_tensor((nvec * 4,), Int32)
        for j in cutlass.range_constexpr(nvec):
            for e in cutlass.range_constexpr(4):
                pwords[j * 4 + e] = Int32(0)
        row_addr = bcast + Int64(row) * (H * 2)
        ready = Int32(0)
        while ready == 0:
            ready = Int32(1)
            for j in cutlass.range_constexpr(nvec):
                col = tXcX[j * vec][1]
                w = _lam.ld_volatile_v4(row_addr + Int64(col) * 2)
                for e in cutlass.range_constexpr(4):
                    words[j * 4 + e] = w[e]
                ready = ready & _lam.clean4(*w)
            if cutlass.const_expr(self.taps > 1):
                if prev != 0:
                    for j in cutlass.range_constexpr(nvec):
                        col = tXcX[j * vec][1]
                        w = _lam.ld_volatile_v4(row_addr - Int64(H * 2) + Int64(col) * 2)
                        for e in cutlass.range_constexpr(4):
                            pwords[j * 4 + e] = w[e]
                        ready = ready & _lam.clean4(*w)
        arr_old = Int32(0)
        if cutlass.const_expr(enable_pdl):
            # Every vector of this row arrived: every rank's producer read the rotation state.
            cute.arch.barrier()
            if tidx == 0:
                arr_old = _lam.arrive_count(flags)

        # ===== Block convolution (FP32, every op rounded on its own), as dflash2_conv_add_rmsnorm =====
        acc = cute.make_fragment_like(tRrR, Float32)
        tBr = cute.make_fragment_like(tWgW)
        for tap in cutlass.range_constexpr(self.taps):
            mBt = cute.make_tensor(mB.iterator + tap * H, cute.prepend(mW.layout, bcast_l))
            cute.copy(copy_atom, thr_copy.partition_S(cute.local_tile(mBt, tiler_mn, (0, 0))), tBr)
            d = Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(acc)):
                if cutlass.const_expr(i % vec == 0):
                    d = Float32(mD[row, tap, tXcX[i][1] // self.group_size])
                coef = _add_rn(Float32(tBr[i]), d)
                j, e = i // vec, i % vec
                if cutlass.const_expr(tap == 0):
                    lo, hi = _unpack_bf16x2(words[j * 4 + e // 2])
                    acc[i] = _mul_rn(coef, lo if e % 2 == 0 else hi)
                else:
                    lo, hi = _unpack_bf16x2(pwords[j * 4 + e // 2])
                    acc[i] = _add_mul_select_rn(acc[i], coef, lo if e % 2 == 0 else hi, prev)
        for i in cutlass.range_constexpr(cute.size(acc)):
            tCrC[i] = acc[i].to(mR.element_type)

        # ===== FlashInfer's fused add + RMSNorm on the convolved row =====
        x_in = tCrC.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in
        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(h_sq, threads_per_row, reduction_buffer, None, 1)
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
        cute.arch.barrier()
        w_f = tWrW.load().to(Float32)
        y = h * rstd * (w_f + Float32(self.weight_bias))

        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
        tXrO.store(h.to(mR.element_type))
        cute.copy(copy_atom, tXrO, tRgO)
        tXrO.store(y.to(mO.element_type))
        cute.copy(copy_atom, tXrO, tXgO)
        if cutlass.const_expr(enable_pdl):
            if tidx == 0:
                _lam.arrive_rotate(flags, arr_old, M, cur, nbytes, M, W, H, False, 0)

        if enable_pdl:
            cute.arch.barrier()
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(
    H: int,
    taps: int,
    group_size: int,
    dtype: torch.dtype,
    world: int,
    chunk: int,
    enable_pdl: bool,
    sm: int,
):
    from flashinfer.norm.utils import _torch_dtype_to_str

    cdt = get_cutlass_dtype(_torch_dtype_to_str(dtype))
    kernel = DFlash2ArConvAddRMSNormKernel(cdt, H, taps, group_size, sm, world, chunk)
    sym_m = cute.sym_int(64)
    align = math.gcd(128, H * (cdt.width // 8))
    r_fake = cute.runtime.make_fake_compact_tensor(
        cdt, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    o_fake = cute.runtime.make_fake_compact_tensor(
        cdt, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    d_fake = cute.runtime.make_fake_tensor(
        cdt, (sym_m, taps, H // group_size), stride=(cute.sym_int64(), cute.sym_int64(), 1)
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cdt, (taps, H), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(cdt, (H,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        d_fake,
        b_fake,
        r_fake,
        w_fake,
        o_fake,
        a,
        a,
        a,
        Int32(1),
        Int32(1),
        Int32(0),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def dflash2_ar_conv_add_rmsnorm(
    ws: "_lam.LamportWorkspace",
    rows: int,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    chunk: int,
    enable_pdl: bool,
) -> torch.Tensor:
    """Finish the two-shot all-reduce of ``rows`` x ``H`` BF16 rows every rank pushed into ``ws`` (the next kernel on
    the workspace after the push), then ``out`` = the RMSNorm of ``conv(rows) + residual`` and ``residual`` updated in
    place, as ``dflash2_conv_add_rmsnorm``. ``delta`` ``[rows, taps, H // group_size]`` with unit last stride;
    ``base_kernel`` ``[taps, H]``, ``weight`` ``[H]``, ``residual`` / ``out`` ``[rows, H]`` contiguous of one 16-bit
    dtype. Compiles once per shape class outside graph capture."""
    if rows == 0:
        return out
    H = residual.shape[1]
    compiled = _compiled(
        H,
        base_kernel.shape[0],
        group_size,
        residual.dtype,
        ws.world,
        chunk,
        bool(enable_pdl),
        get_sm_version(residual.device),
    )
    compiled(
        delta.detach(),
        base_kernel.detach(),
        residual,
        weight.detach(),
        out,
        ws.uc_local,
        ws.mc_base,
        ws.flags.data_ptr(),
        rows,
        int(block_size),
        ws.rank,
        float(eps),
    )
    return out


__all__ = ["DFlash2ArConvAddRMSNormKernel", "dflash2_ar_conv_add_rmsnorm"]
