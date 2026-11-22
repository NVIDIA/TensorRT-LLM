# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) under the Apache-2.0
# license.  Original file: flashinfer/norm/kernels/fused_add_rmsnorm.py
# Original copyright: Copyright (c) 2025 by FlashInfer team.
"""Fused Add + RMSNorm (CuTe DSL) with an early PDL trigger.

Subclass of flashinfer's ``FusedAddRMSNormKernel``, imported from the installed
flashinfer and left unmodified. Only the placement of ``griddepcontrol.launch_dependents``
differs: it is issued right after the pass-1 global loads (input, residual, weight) have
landed in registers, instead of after the final store. The arithmetic, the loads and the
stores are the parent kernel's, so the outputs are bitwise identical. Dependents still
order every read of this kernel's outputs behind their own ``griddepcontrol.wait`` (which
waits for this grid's completion and memory flush); the early trigger only lets them
launch and run their pre-wait prologue (e.g. Marlin's weight L2 prefetch) while this
kernel computes and stores.

Only the compact (contiguous) layout is compiled; the compiled kernel has its own cache,
separate from the parent's.
"""

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.utils import (
    _torch_dtype_to_str,
    get_cutlass_dtype,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)


class FusedAddRMSNormEarlyTriggerKernel(FusedAddRMSNormKernel):
    """FusedAddRMSNormKernel with the PDL trigger moved to just after the loads."""

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
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

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output destinations
        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)

        # PDL: early trigger. The global loads of this CTA have landed (their
        # values are in registers); dependents may launch now. They read this
        # kernel's outputs only after their own griddepcontrol.wait.
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()

        h = x_in + r_in

        # Write h to residual (global)
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

        # ===== Pass 2: Normalize and store output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)


@functools.cache
def _get_compiled_fused_add_rmsnorm_early_trigger_kernel(
    dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
):
    """flashinfer's compact-layout ``_get_compiled_fused_add_rmsnorm_kernel``, building
    FusedAddRMSNormEarlyTriggerKernel (own cache)."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = FusedAddRMSNormEarlyTriggerKernel(dtype, H, weight_bias, sm_version=sm_version)

    sym_m = cute.sym_int(64)

    elem_bytes = dtype.width // 8
    tensor_align = math.gcd(128, H * elem_bytes)
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )
    r_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )

    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        r_fake,
        w_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


def gemma_fused_add_rmsnorm_early_trigger(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool,
) -> None:
    """Gemma-style (weight + 1) fused add RMSNorm with the early PDL trigger, in place:
    residual = input + residual; input = rmsnorm(residual) * (weight + 1)."""
    assert input.dim() == 2 and input.shape == residual.shape
    assert input.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    M, H = input.shape
    # The compact layout indexes rows in 32-bit offsets below this bound (flashinfer
    # switches to its strided kernel above it).
    assert M * H <= 2**31 - 1, "more than 2^31 - 1 elements needs the strided kernel"
    if M == 0:
        return
    kernel = _get_compiled_fused_add_rmsnorm_early_trigger_kernel(
        _torch_dtype_to_str(input.dtype),
        H,
        1.0,
        bool(enable_pdl),
        get_sm_version(input.device),
    )
    kernel(input, residual, weight, M, eps)
