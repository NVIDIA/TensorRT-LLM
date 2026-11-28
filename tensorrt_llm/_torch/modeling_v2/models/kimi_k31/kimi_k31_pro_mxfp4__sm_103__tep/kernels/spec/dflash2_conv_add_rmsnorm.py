# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0): flashinfer/norm/kernels/fused_add_rmsnorm.py
"""The DFlash2 drafter's sublayer epilogue in one launch (CuTe DSL): block convolution of the
all-reduced sublayer output, residual add and RMSNorm.

The drafter's attention and MLP sublayers end with ``finish`` (the dynamic block convolution of
the row-parallel projection's all-reduced output) followed by the fused residual add + RMSNorm:

* ``c[t] = bf16(sum_i (base[i] + delta[t, i, group]) * x[t - i])`` (taps past the start of row
  ``t``'s block multiply ``0.0``), the FP32 arithmetic of ``ops/spec/dflash2_grouped_conv``:
  ``base + delta`` per tap, the tap-0 product, then ``acc + coef * x`` per tap, every product and
  sum rounded on its own;
* ``h = c + residual`` (FP32), ``residual = bf16(h)``, ``out = bf16(h * rsqrt(mean(h^2) + eps) *
  weight)``, FlashInfer's CuTe DSL ``fused_add_rmsnorm`` (the catalog ``flashinfer_fused_add_rmsnorm``
  on SM 10.x).

This kernel is FlashInfer's ``FusedAddRMSNormKernel`` (its tiling and cluster split are inherited; the kernel
prologue, row reduction and normalization follow FlashInfer's code) whose input load is replaced by the
convolution, so both
outputs are bitwise those of the two kernels it replaces. ``x`` is read-only (a row's
convolution reads the previous rows of its block), and the normalized output goes to its own
tensor.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel as _FiFusedAddRMSNorm
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel as _FiRMSNorm
from flashinfer.norm.utils import (
    get_cutlass_dtype,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.spec.dflash2_grouped_conv import (
    _add_mul_select_rn,
    _add_rn,
    _mul_rn,
)


class DFlash2ConvAddRMSNormKernel(_FiFusedAddRMSNorm):
    """``(out, residual) = fused_add_rmsnorm(conv(x; base, delta), residual)`` per row."""

    def __init__(self, dtype: cutlass.Numeric, H: int, taps: int, group_size: int, sm_version: int):
        super().__init__(dtype, H, 0.0, sm_version=sm_version)
        if group_size % self.vec_size or H % group_size:
            raise ValueError(
                f"group_size ({group_size}) must be a multiple of {self.vec_size} dividing {H}"
            )
        if taps < 1:
            raise ValueError(f"taps must be positive; got {taps}")
        if self.cols_per_tile * self.cluster_n != H:
            # Every tile column is a channel: the per-vector delta loads are unpredicated.
            raise ValueError(
                f"H={H} does not tile exactly ({self.cols_per_tile} x {self.cluster_n})"
            )
        self.taps = taps
        self.group_size = group_size
        # The input tile is computed, not copied: no shared-memory staging.
        self.use_async_copy = False

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mD: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        M: Int64,
        block_size: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = _FiRMSNorm._make_tv_layout(
            self.threads_per_row, self.rows_per_block, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        cluster_n = self.cluster_n
        self.kernel(
            mA, mD, mB, mR, mW, mO, M, block_size, eps, enable_pdl, tv_layout, tiler_mn
        ).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mD: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        M: Int64,
        block_size: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32, cute.make_layout((rows_per_block, warps_per_row)), byte_alignment=4
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        idX = cute.make_identity_tensor(mA.shape)
        gA = cute.local_tile(mA, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        # Row-broadcast views (stride 0 over the tile's rows) of the weight and of each base tap.
        bcast = cute.make_layout((tiler_mn[0],), stride=(0,))
        mW_2d = cute.make_tensor(mW.iterator, cute.prepend(mW.layout, bcast))
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mA.element_type, num_bits_per_copy=copy_bits
        )
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thr_copy = tiled_copy.get_slice(tidx)

        tAgA = thr_copy.partition_S(gA)
        tXcX = thr_copy.partition_S(cX)
        tArA = cute.make_fragment_like(tAgA)
        tRgR = thr_copy.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        tWgW = thr_copy.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXgO = thr_copy.partition_D(gO)
        tRgO = thr_copy.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)
        tCrC = cute.make_fragment_like(tAgA)

        tXpX = predicate_k(tXcX, limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row = row_coord[0]
        row_in_bounds = row < M
        pos = Int32(row % Int64(block_size))

        tArA.store(cute.zeros_like(tArA, dtype=mA.element_type))
        tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom, tAgA, tArA, pred=tXpX)
            cute.copy(copy_atom, tRgR, tRrR, pred=tXpX)
        cute.copy(copy_atom, tWgW, tWrW, pred=tXpX)

        # ===== Block convolution of the all-reduced rows (FP32, every op rounded on its own) =====
        # The fragment holds 16-byte vectors of consecutive channels (vector-fastest), each inside
        # one coefficient group: one delta load per vector and tap. Rows past M read row 0's delta
        # (their results are never stored).
        vec = self.vec_size
        row_safe = row
        if row >= M:
            row_safe = row - row
        acc = cute.make_fragment_like(tArA, Float32)
        tBr = cute.make_fragment_like(tWgW)
        tPrP = cute.make_fragment_like(tAgA)
        for tap in cutlass.range_constexpr(self.taps):
            mBt = cute.make_tensor(mB.iterator + tap * H, cute.prepend(mW.layout, bcast))
            cute.copy(
                copy_atom,
                thr_copy.partition_S(cute.local_tile(mBt, tiler_mn, (0, cluster_y))),
                tBr,
                pred=tXpX,
            )
            keep = Int32(1)
            if cutlass.const_expr(tap > 0):
                keep = Int32(0)
                if pos >= tap:
                    keep = Int32(1)
                # The previous rows of the block: the same tile one tap up (read only where kept).
                tPrP.store(cute.zeros_like(tPrP, dtype=mA.element_type))
                if row_in_bounds:
                    if keep != 0:
                        mAp = cute.make_tensor(mA.iterator + (-tap * H), mA.layout)
                        cute.copy(
                            copy_atom,
                            thr_copy.partition_S(cute.local_tile(mAp, tiler_mn, (bidx, cluster_y))),
                            tPrP,
                            pred=tXpX,
                        )
            d = Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(acc)):
                if cutlass.const_expr(i % vec == 0):
                    d = Float32(mD[row_safe, tap, tXcX[i][1] // self.group_size])
                coef = _add_rn(Float32(tBr[i]), d)
                if cutlass.const_expr(tap == 0):
                    acc[i] = _mul_rn(coef, Float32(tArA[i]))
                else:
                    acc[i] = _add_mul_select_rn(acc[i], coef, Float32(tPrP[i]), keep)
        for i in cutlass.range_constexpr(cute.size(acc)):
            tCrC[i] = acc[i].to(mA.element_type)

        # ===== FlashInfer's fused add + RMSNorm on the convolved rows =====
        x_in = tCrC.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        w = tWrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))
        tXrO.store(y.to(mO.element_type))
        if row_in_bounds:
            cute.copy(copy_atom, tXrO, tXgO, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(H: int, taps: int, group_size: int, dtype: torch.dtype, enable_pdl: bool, sm: int):
    from flashinfer.norm.utils import _torch_dtype_to_str

    cdt = get_cutlass_dtype(_torch_dtype_to_str(dtype))
    kernel = DFlash2ConvAddRMSNormKernel(cdt, H, taps, group_size, sm)
    sym_m = cute.sym_int(64)
    align = math.gcd(128, H * (cdt.width // 8))
    a_fake = cute.runtime.make_fake_compact_tensor(
        cdt, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
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
    return cute.compile(
        kernel,
        a_fake,
        d_fake,
        b_fake,
        r_fake,
        w_fake,
        o_fake,
        Int64(1),
        Int32(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def dflash2_conv_add_rmsnorm(
    x: torch.Tensor,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """``out`` = the RMSNorm of ``conv(x) + residual`` and ``residual`` updated in place (one launch).

    ``x``, ``residual``, ``out`` ``[rows, H]`` contiguous of one 16-bit dtype (``out`` must not alias
    ``x``); ``delta`` ``[rows, taps, H // group_size]`` with unit last stride; ``base_kernel``
    ``[taps, H]`` and ``weight`` ``[H]`` contiguous. Compiles once per shape class outside graph capture.
    """
    rows, H = x.shape
    if rows == 0:
        return out
    compiled = _compiled(
        H, base_kernel.shape[0], group_size, x.dtype, bool(enable_pdl), get_sm_version(x.device)
    )
    compiled(
        x.detach(),
        delta.detach(),
        base_kernel.detach(),
        residual,
        weight.detach(),
        out,
        rows,
        int(block_size),
        float(eps),
    )
    return out


__all__ = ["DFlash2ConvAddRMSNormKernel", "dflash2_conv_add_rmsnorm"]
