# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) under the Apache-2.0
# license.  Original file: flashinfer/norm/kernels/fused_add_rmsnorm.py
# Original copyright: Copyright (c) 2025 by FlashInfer team.
"""One-row-per-CTA geometry for flashinfer's CuTe-DSL gemma fused add + RMSNorm.

Subclasses flashinfer's ``FusedAddRMSNormKernel`` (imported from the installed
flashinfer, never modified) with one row per CTA and ``TPR`` = 320 threads per row, for
decode row counts where the stock two-rows-a-CTA grid leaves most SMs idle:

``TPR`` threads load X/R, store the residual and write h = f32(x) + f32(r) to shared
memory; the first 64 threads then replay the stock 64-threads-a-row sum (sequential over
columns 512*j + 8*t + v, butterfly warp reduce, warp 0 + warp 1), and every thread
normalizes its own columns. y and the residual are bitwise equal to the stock kernel.

``TPR`` must divide H / 8 (= 640 at H = 5120), be a multiple of 32 and at least 64, and is
never 256 (cp.async has no zero-fill).
"""

import functools
import math
import operator

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
    warp_reduce,
)

#: Threads a row in the stock geometry at hidden 3072 < H <= 6144: the sum order replayed.
STOCK_TPR = 64
#: Threads a row (and a CTA) here.
TPR = 320


class FusedAddRMSNorm1RowKernel(FusedAddRMSNormKernel):
    """TPR-thread loads/stores, one row a CTA, the stock tpr64 sum order (bitwise)."""

    def __init__(
        self,
        dtype,
        H: int,
        weight_bias: float = 0.0,
        tpr: int = TPR,
        sm_version=None,
    ):
        super().__init__(dtype, H, weight_bias, sm_version=sm_version)
        assert self.cluster_n == 1, "1-row geometry needs cluster_n == 1"
        n_vec = self.H_per_cta // self.vec_size
        assert n_vec % tpr == 0 and tpr % 32 == 0 and tpr != 256, tpr
        assert n_vec % STOCK_TPR == 0 and tpr >= STOCK_TPR
        self.tpr = tpr
        self.threads_per_row = tpr
        self.num_threads = tpr
        self.rows_per_block = 1
        self.warps_per_row = tpr // 32
        self.num_vec_blocks = n_vec // tpr
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * tpr
        self.stock_vec_blocks = n_vec // STOCK_TPR
        elem_bytes = dtype.width // 8
        if self.copy_bits >= 32:
            tile_bytes_2 = 2 * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = tile_bytes_2 <= props.shared_memory_per_block_optin // 2
        else:
            self.use_async_copy = False

    def _smem_size_in_bytes(self) -> int:
        base = super()._smem_size_in_bytes()
        # + sH (f32 row) + 2 warp partials, with alignment slack.
        return (base + 15) // 16 * 16 + self.H * 4 + 32

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

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        vec_size = self.vec_size
        n_blk0 = self.stock_vec_blocks

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
        sH = smem.allocate_tensor(Float32, cute.make_layout((H,)), byte_alignment=16)
        sP = smem.allocate_tensor(Float32, cute.make_layout((2,)), byte_alignment=16)

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
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

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

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
        h = x_in + r_in

        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        # h -> shared memory at its column (f32, exact).
        tHr = cute.make_fragment_like(tXrX, Float32)
        tHr.store(h)
        for i in cutlass.range_constexpr(cute.size(tHr)):
            sH[tXcX[i][1]] = tHr[i]
        cute.arch.barrier()

        # Replay the stock tpr64 sum: thread t, columns 512*j + 8*t + v,
        # sequential over j then v; butterfly warp reduce; warp0 + warp1.
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        if warp_idx < 2:
            acc = Float32(0.0)
            for j in cutlass.range_constexpr(n_blk0):
                for v in cutlass.range_constexpr(vec_size):
                    hv = sH[j * (STOCK_TPR * vec_size) + tidx * vec_size + v]
                    acc = acc + hv * hv
            acc = warp_reduce(acc, operator.add, width=32)
            if lane_idx == 0:
                sP[warp_idx] = acc
        cute.arch.barrier()
        sum_sq = sP[0] + sP[1]

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))
        tXrO.store(y.to(mX.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _get_compiled_1row_kernel(
    dtype_str: str, H: int, weight_bias: float, enable_pdl: bool, sm_version: int
):
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = FusedAddRMSNorm1RowKernel(dtype, H, weight_bias, tpr=TPR, sm_version=sm_version)
    sym_m = cute.sym_int(64)
    tensor_align = math.gcd(128, H * (dtype.width // 8))
    x_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )
    r_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
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


def gemma_fused_add_rmsnorm_1row(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool,
) -> None:
    """In place: residual += input; input = rmsnorm(residual) * (weight + 1)."""
    assert input.dim() == 2 and input.shape == residual.shape
    assert input.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    if input.shape[0] == 0:
        return
    kernel = _get_compiled_1row_kernel(
        _torch_dtype_to_str(input.dtype),
        input.shape[-1],
        1.0,
        bool(enable_pdl),
        get_sm_version(input.device),
    )
    kernel(input, residual, weight, input.shape[0], eps)
