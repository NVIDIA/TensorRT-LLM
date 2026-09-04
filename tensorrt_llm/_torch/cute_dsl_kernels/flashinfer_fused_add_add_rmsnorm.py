# SPDX-FileCopyrightText: Copyright (c) 2025 FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer-derived three-input fused add + add + RMSNorm kernel.

The kernel preserves FlashInfer's fused-add/RMSNorm tiling, reduction, stores,
and two shared-memory tiles.  The additional MoE output is loaded directly to
registers while the original input and residual use FlashInfer's asynchronous
global-to-shared pipeline.
"""

import functools
import math
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import (
    _torch_dtype_to_str,
    get_cutlass_dtype,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)


class FusedAddAddRMSNormKernel(FusedAddRMSNormKernel):
    """Compute a BF16-rounded MoE sum followed by fused add + RMSNorm.

    The in-place semantics exactly match::

        input.add_(additional)
        flashinfer.fused_add_rmsnorm(input, residual, weight, eps)

    In particular, ``input + additional`` is rounded to the input dtype before
    the residual is accumulated in FP32.
    """

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mA: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel(mX, mA, mR, mW, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), self.cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if cutlass.const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mA: cute.Tensor,
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

        hidden_size = self.H
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

        # Keep the exact two-tile FlashInfer shared-memory pipeline.  A third
        # tile would increase the H=7168 specialization beyond its measured
        # 28,688-byte footprint and could reduce block residency.
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

        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gA = cute.local_tile(mA, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

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

        tiled_copy_A = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        tAgA = thr_copy_A.partition_S(gA)
        tArA = cute.make_fragment_like(tAgA)

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

        tXpX = predicate_k(tXcX, limit=hidden_size)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=hidden_size)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # Stage input and residual exactly as FlashInfer does.  The additional
        # MoE output uses a synchronous vector load into registers while those
        # two cp.async transactions are in flight.
        tArA.store(cute.zeros_like(tArA, dtype=mA.element_type))
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            if row_in_bounds:
                cute.copy(copy_atom_sync, tAgA, tArA, pred=tXpX)
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tAgA, tArA, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        # This explicit narrowing is required for exact compatibility with the
        # existing two-op sequence: BF16 add_(additional), then FlashInfer's
        # FP32 input + residual accumulation.
        moe_sum = (tXrX.load().to(Float32) + tArA.load().to(Float32)).to(mX.element_type)
        h = moe_sum.to(Float32) + tRrR.load().to(Float32)

        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        sum_sq = row_reduce_sum_multirow(
            h * h, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )
        rstd = cute.math.rsqrt(sum_sq / Float32(hidden_size) + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        y = h * rstd * (tXrW.load().to(Float32) + Float32(weight_bias))
        tXrO.store(y.to(mX.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.lru_cache(maxsize=32)
def _get_compiled_fused_add_add_rmsnorm_kernel(
    dtype_str: str,
    hidden_size: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
    contiguous: bool = True,
) -> Callable[..., None]:
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = FusedAddAddRMSNormKernel(dtype, hidden_size, weight_bias, sm_version=sm_version)
    sym_m = cute.sym_int(64)

    if contiguous:
        elem_bytes = dtype.width // 8
        tensor_align = math.gcd(128, hidden_size * elem_bytes)
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=tensor_align
        )
        a_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=tensor_align
        )
        r_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=tensor_align
        )
    else:
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_a = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_r = cute.sym_int64(divisibility=kernel_obj.vec_size)
        x_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, hidden_size), (sym_row_stride_x, 1), assumed_align=16
        )
        a_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, hidden_size), (sym_row_stride_a, 1), assumed_align=16
        )
        r_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, hidden_size), (sym_row_stride_r, 1), assumed_align=16
        )

    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (hidden_size,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_obj,
        x_fake,
        a_fake,
        r_fake,
        w_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def fused_add_add_rmsnorm_cute(
    input: torch.Tensor,  # noqa: A002 - stable custom-op forwarding name
    additional: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """Run three-input BF16 fused add + add + RMSNorm in place."""
    if input.ndim != 2 or additional.ndim != 2 or residual.ndim != 2:
        raise ValueError(
            "input, additional, and residual must all be 2D (M, H); got "
            f"{tuple(input.shape)}, {tuple(additional.shape)}, and {tuple(residual.shape)}"
        )
    if additional.shape != input.shape or residual.shape != input.shape:
        raise ValueError(
            "additional and residual must match input shape; got "
            f"{tuple(additional.shape)} and {tuple(residual.shape)} versus {tuple(input.shape)}"
        )
    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must use torch.bfloat16, got {input.dtype}")
    if additional.dtype != input.dtype or residual.dtype != input.dtype:
        raise ValueError(
            "additional and residual must match input dtype; got "
            f"{additional.dtype} and {residual.dtype} versus {input.dtype}"
        )
    if not input.is_cuda:
        raise ValueError("input, additional, residual, and weight must be CUDA tensors")
    if additional.device != input.device or residual.device != input.device:
        raise ValueError(
            "additional and residual must be on the same device as input; got "
            f"{additional.device} and {residual.device} versus {input.device}"
        )

    shape = input.shape
    hidden_size = shape[-1]
    num_rows = shape[0]
    if weight.ndim != 1 or weight.shape[0] != hidden_size:
        raise ValueError(f"weight must have shape ({hidden_size},), got {tuple(weight.shape)}")
    if weight.dtype != input.dtype or weight.device != input.device:
        raise ValueError(
            "weight must match input dtype and device; got "
            f"{weight.dtype} on {weight.device} versus {input.dtype} on {input.device}"
        )
    if any(tensor.stride(-1) != 1 for tensor in (input, additional, residual)):
        raise ValueError(
            "input, additional, and residual must be contiguous in the hidden dimension"
        )
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")

    is_contiguous = (
        input.is_contiguous() and additional.is_contiguous() and residual.is_contiguous()
    )
    if is_contiguous and num_rows * hidden_size > 2**31 - 1:
        is_contiguous = False

    kernel = _get_compiled_fused_add_add_rmsnorm_kernel(
        _torch_dtype_to_str(input.dtype),
        hidden_size,
        weight_bias,
        enable_pdl,
        get_sm_version(input.device),
        contiguous=is_contiguous,
    )
    kernel(input, additional, residual, weight, num_rows, eps)


__all__ = [
    "FusedAddAddRMSNormKernel",
    "fused_add_add_rmsnorm_cute",
]
