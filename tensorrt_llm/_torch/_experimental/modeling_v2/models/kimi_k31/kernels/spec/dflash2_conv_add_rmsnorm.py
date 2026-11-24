# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer 0.6.18 (Apache-2.0): flashinfer/norm/kernels/rmsnorm.py (RMSNormKernel) and
# flashinfer/norm/kernels/fused_add_rmsnorm.py (FusedAddRMSNormKernel); modified by NVIDIA: dependents released
# right after griddepcontrol.wait, and the DFlash2 block convolution fused into the add-RMSNorm input.
"""The DFlash2 drafter's norms with an early PDL release (CuTe DSL), on FlashInfer's norm kernels.

Two kernels, both bitwise FlashInfer's (``flashinfer.norm.kernels`` of the served FlashInfer):

- :class:`DFlash2RmsNormKernel`: FlashInfer's ``RMSNormKernel`` with the dependents released right
  after ``griddepcontrol.wait`` instead of after the last store.
- :class:`DFlash2ConvAddRmsNormKernel`: the output side of a DFlash2 block convolution
  (``dflash2_grouped_conv``) fused into FlashInfer's ``FusedAddRMSNormKernel``:
  ``c = Conv(a; base, delta)`` rounded to the activation dtype, ``residual = c + residual``, and
  ``out = RMSNorm(residual)``, with the release right after the wait (or, with ``early=False``,
  after the last store as FlashInfer does). ``ctas`` > 0 splits each row over a cluster of that
  many CTAs (one vector per thread, the row's sum of squares reduced over the cluster in DSMEM by
  FlashInfer's own cluster reduction): the residual stays bitwise, the norm's sum is taken in
  another order.

Both subclasses take FlashInfer's configuration (cluster, threads per row, vector width) from its
constructor and its row reduction (``row_reduce_sum_multirow``) as is, and repeat its kernel body:
the same thread-value layout, the same FP32 products and sums in the same order. So each output is
bitwise FlashInfer's on the same input; the convolution's values are bitwise the
``dflash2_grouped_conv`` kernel's (its round-to-nearest FP32 helpers, one round to BF16).

The release right after the wait lets a dependent that waits before reading this kernel's output
(the drafter's split-K GEMMs, which stream their constant weights before their wait) start while
the norm runs. A dependent that reads without waiting must not follow these kernels.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import get_sm_version, predicate_k, row_reduce_sum_multirow

from .dflash2_grouped_conv import VEC, _add_mul_select_rn, _add_rn, _mul_rn

_CUTE_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


class DFlash2RmsNormKernel(RMSNormKernel):
    """FlashInfer's ``RMSNormKernel``; with PDL it releases its dependents right after its wait."""

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

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
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
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

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mY.element_type, num_bits_per_copy=copy_bits
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=copy_bits
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))
        tXrO.store(y.to(mY.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


class DFlash2ConvAddRmsNormKernel(FusedAddRMSNormKernel):
    """``residual += Conv(a)``; ``out = RMSNorm(residual)``: FlashInfer's ``FusedAddRMSNormKernel``
    with the convolution in place of its input load. One row per CTA tile (``rows_per_block`` 1),
    so a tap's earlier row is the tile ``taps`` rows up. ``ctas`` 0 keeps FlashInfer's layout
    (hidden 7168: one CTA of 128 threads x 7 vectors per row); ``ctas`` > 0 spreads the row over a
    cluster of that many CTAs, each thread one vector (hidden 7168, ``ctas`` 7: 7 x 128 threads)."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        taps: int,
        group_size: int,
        early: bool,
        sm_version: int | None = None,
        ctas: int = 0,
    ):
        self._ctas = ctas
        super().__init__(dtype, H, 0.0, sm_version=sm_version)
        if ctas:
            threads = self.H_per_cta // self.vec_size
            if threads % 32 or threads > 1024 or self.H_per_cta % self.vec_size:
                raise ValueError(f"hidden {H} over {ctas} CTAs: {self.H_per_cta} columns per CTA")
            self.threads_per_row = self.num_threads = threads
            self.rows_per_block = 1
            self.warps_per_row = threads // 32
            self.num_vec_blocks = 1
            self.cols_per_tile = self.vec_size * threads
        if self.rows_per_block != 1 or self.cols_per_tile * self.cluster_n != H:
            raise ValueError(
                f"hidden {H}: FlashInfer's fused-add layout is {self.rows_per_block} rows x "
                f"{self.cols_per_tile} columns x {self.cluster_n} CTAs; the fused convolution needs "
                "one row per CTA and no partial tile"
            )
        if group_size % VEC or H % group_size or taps < 1:
            raise ValueError(f"group_size {group_size} / taps {taps} for hidden {H}")
        self.taps = taps
        self.group_size = group_size
        self.early = early

    def _compute_cluster_n(self, H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        if self._ctas:
            if H % self._ctas:
                raise ValueError(f"hidden {H} does not split over {self._ctas} CTAs")
            return self._ctas
        return FusedAddRMSNormKernel._compute_cluster_n(H, dtype, sm_version)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mD: cute.Tensor,
        mB: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        block_size: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row, self.rows_per_block, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        cluster_n = self.cluster_n
        self.kernel(
            mA, mD, mB, mR, mW, mY, M, eps, block_size, enable_pdl, tv_layout, tiler_mn
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
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        block_size: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.early):
                cute.arch.griddepcontrol_launch_dependents()

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
        if cutlass.const_expr(self.use_async_copy):
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
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

        idX = cute.make_identity_tensor(mR.shape)
        gA = cute.local_tile(mA, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mR.element_type, num_bits_per_copy=copy_bits
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mR.element_type, num_bits_per_copy=copy_bits
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), mR.element_type, num_bits_per_copy=copy_bits
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXcX = thr_copy_X.partition_S(cX)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        if cutlass.const_expr(self.use_async_copy):
            tRsR = thr_copy_X.partition_D(sR)
        # The convolution's operands in the weight's (synchronous) partition, retiled to the input's.
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        tAgA = thr_copy_W.partition_S(gA)
        tArA = cute.make_fragment_like(tAgA)
        tXrA = thr_copy_X.retile(tArA)
        tPrP = cute.make_fragment_like(tAgA)
        tXrP = thr_copy_X.retile(tPrP)
        tBrB = cute.make_fragment_like(tWgW)
        tXrB = thr_copy_X.retile(tBrB)
        tXrC = cute.make_fragment_like(tRrR)

        tXgO = thr_copy_O.partition_D(gY)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row = row_coord[0]
        row_in_bounds = row < M
        pos = row % block_size

        # ===== Pass 1: residual (async as FlashInfer), the convolution, h = conv + residual =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()
        else:
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)
        cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        # Tap 0: (base[0] + delta[row, 0]) * a[row].
        tArA.store(cute.zeros_like(tArA, dtype=mA.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_sync, tAgA, tArA, pred=tWpW)
        gB = cute.local_tile(mB, tiler_mn, (0, cluster_y))
        cute.copy(copy_atom_sync, thr_copy_W.partition_S(gB), tBrB, pred=tWpW)
        acc = cute.make_rmem_tensor((cute.size(tXrC),), Float32)
        for k in cutlass.range_constexpr(cute.size(tXrC)):
            group = tXcX[k][1] // self.group_size
            d = Float32(mD[row, 0, group])
            acc[k] = _mul_rn(_add_rn(Float32(tXrB[k]), d), Float32(tXrA[k]))
        # Later taps: + (base[i] + delta[row, i]) * a[row - i], or * 0.0 past the block start
        # (the row's own values loaded and not used, as dflash2_grouped_conv).
        for tap in cutlass.range_constexpr(1, self.taps):
            keep = Int32(0)
            src = row
            if pos >= tap:
                keep = Int32(1)
                src = row - tap
            gP = cute.local_tile(mA, tiler_mn, (src, cluster_y))
            tPrP.store(cute.zeros_like(tPrP, dtype=mA.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, thr_copy_W.partition_S(gP), tPrP, pred=tWpW)
            gB = cute.local_tile(mB, tiler_mn, (tap, cluster_y))
            cute.copy(copy_atom_sync, thr_copy_W.partition_S(gB), tBrB, pred=tWpW)
            for k in cutlass.range_constexpr(cute.size(tXrC)):
                group = tXcX[k][1] // self.group_size
                d = Float32(mD[row, tap, group])
                acc[k] = _add_mul_select_rn(
                    acc[k], _add_rn(Float32(tXrB[k]), d), Float32(tXrP[k]), keep
                )
        for k in cutlass.range_constexpr(cute.size(tXrC)):
            tXrC[k] = acc[k].to(mR.element_type)

        if cutlass.const_expr(self.use_async_copy):
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tRsR, tRrR)

        x_in = tXrC.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

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

        # ===== Pass 2: normalize and store =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))
        tXrO.store(y.to(mY.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if cutlass.const_expr(not self.early):
            if enable_pdl:
                cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compile_rmsnorm(dtype: torch.dtype, hidden: int, pdl: bool, contiguous: bool, sm: int):
    """As FlashInfer's ``_get_compiled_rmsnorm_kernel`` (same fake tensors and options)."""
    cdt = _CUTE_DTYPES[dtype]
    kernel = DFlash2RmsNormKernel(cdt, hidden, 0.0, sm_version=sm)
    rows = cute.sym_int(64)
    if contiguous:
        align = math.gcd(128, hidden * (cdt.width // 8))
        x_fake = cute.runtime.make_fake_compact_tensor(
            cdt, (rows, hidden), stride_order=(1, 0), assumed_align=align
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            cdt, (rows, hidden), stride_order=(1, 0), assumed_align=align
        )
    else:
        x_fake = cute.runtime.make_fake_tensor(
            cdt,
            (rows, hidden),
            (cute.sym_int64(divisibility=kernel.vec_size), 1),
            assumed_align=16,
        )
        y_fake = cute.runtime.make_fake_tensor(
            cdt,
            (rows, hidden),
            (cute.sym_int64(divisibility=kernel.vec_size), 1),
            assumed_align=16,
        )
    w_fake = cute.runtime.make_fake_compact_tensor(cdt, (hidden,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x_fake,
        w_fake,
        y_fake,
        Int64(1),
        Float32(1e-6),
        pdl,
        stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_conv_add_rmsnorm(
    dtype: torch.dtype,
    hidden: int,
    taps: int,
    group_size: int,
    pdl: bool,
    early: bool,
    sm: int,
    ctas: int = 0,
):
    cdt = _CUTE_DTYPES[dtype]
    kernel = DFlash2ConvAddRmsNormKernel(
        cdt, hidden, taps, group_size, early, sm_version=sm, ctas=ctas
    )
    rows = cute.sym_int(64)
    align = math.gcd(128, hidden * (cdt.width // 8))

    def rows_fake():
        return cute.runtime.make_fake_compact_tensor(
            cdt, (rows, hidden), stride_order=(1, 0), assumed_align=align
        )

    delta_fake = cute.runtime.make_fake_tensor(
        cdt,
        (rows, taps, hidden // group_size),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    base_fake = cute.runtime.make_fake_compact_tensor(
        cdt, (taps, hidden), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(cdt, (hidden,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        rows_fake(),
        delta_fake,
        base_fake,
        rows_fake(),
        w_fake,
        rows_fake(),
        Int64(1),
        Float32(1e-6),
        Int32(1),
        pdl,
        stream,
        options="--enable-tvm-ffi",
    )


def dflash2_rmsnorm_into(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor, pdl: bool
) -> torch.Tensor:
    """``out = RMSNorm(x) * weight``, bitwise FlashInfer's ``rmsnorm``; with ``pdl`` the launch
    waits for the previous kernel and releases the next one right after. ``x`` / ``out``:
    ``[rows, hidden]`` with a unit last stride (row-strided views take the strided compile, as in
    FlashInfer). Compiles on the first call per shape class (outside graph capture)."""
    rows, hidden = x.shape
    if rows == 0:
        return out
    contiguous = x.is_contiguous() and out.is_contiguous()
    compiled = _compile_rmsnorm(x.dtype, hidden, pdl, contiguous, get_sm_version(x.device))
    compiled(x, weight, out, rows, eps)
    return out


def dflash2_conv_add_rmsnorm_into(
    a: torch.Tensor,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    block_size: int,
    group_size: int,
    out: torch.Tensor,
    pdl: bool,
    early: bool,
    ctas: int = 0,
) -> torch.Tensor:
    """``residual += Conv(a; base_kernel, delta)`` (in place) and ``out = RMSNorm(residual) *
    weight``: bitwise ``dflash2_grouped_conv`` followed by FlashInfer's ``fused_add_rmsnorm``
    (``ctas`` 0), or with the row over a ``ctas``-CTA cluster (the residual bitwise, the norm's sum
    in another order). With ``pdl`` the launch waits for the previous kernel; ``early`` releases
    the next one right after the wait, else after the last store. Contiguous ``a`` / ``residual`` /
    ``out``; ``delta`` with a unit last stride (one side's view of the kernel projection's
    output)."""
    rows, hidden = a.shape
    if rows == 0:
        return out
    compiled = _compile_conv_add_rmsnorm(
        a.dtype,
        hidden,
        base_kernel.shape[0],
        group_size,
        pdl,
        early,
        get_sm_version(a.device),
        ctas,
    )
    compiled(
        a.detach(),
        delta.detach(),
        base_kernel.detach(),
        residual,
        weight.detach(),
        out,
        rows,
        eps,
        int(block_size),
    )
    return out


__all__ = [
    "DFlash2ConvAddRmsNormKernel",
    "DFlash2RmsNormKernel",
    "dflash2_conv_add_rmsnorm_into",
    "dflash2_rmsnorm_into",
]
