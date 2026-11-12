# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm of the MLA q_a latent that releases its dependent grid at launch (Kimi K3 decode).

FlashInfer's CuTe DSL ``RMSNormKernel`` (what ``RMSNorm`` runs for this BF16 norm) with one
change: ``griddepcontrol.launch_dependents`` is issued when a CTA starts instead of after its
store. The arithmetic is the kernel's own code, so the output is bitwise the same.

Why: the dependent is the q_b projection (``k3_fp8w_lowm_gemm``), a weight-streaming GEMM. With
the trigger at the end it launches only when this norm finishes, after the q_a GEMM. Released
at launch, its CTAs become resident on SMs the q_a GEMM leaves free and stream q_b's weights
while q_a and this norm still run.

PDL argument for triggering before ``griddepcontrol.wait``: the trigger only lets the dependent
grid launch; what it reads before its own wait must not depend on this grid or anything
upstream that is not complete. The q_b GEMM reads only its weights (written at load time)
before its wait; its activation load waits for this grid's completion, and this grid's
completion implies the q_a GEMM's (this grid waits on it before reading). Grids further down
the chain (q absorb, MLA decode) are released by q_b's own trigger and read only weights
before their waits, as before; the decode additionally has a full graph dependency on the
KV append. So no read moves ahead of its writer.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64

try:
    from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
    from flashinfer.norm.utils import (
        get_cutlass_dtype,
        get_sm_version,
        predicate_k,
        row_reduce_sum_multirow,
    )

    IS_AVAILABLE = True
except ImportError:  # pragma: no cover - FlashInfer without the CuTe DSL norm kernels
    RMSNormKernel = object
    IS_AVAILABLE = False


class _LaunchReleaseRMSNormKernel(RMSNormKernel):
    """``RMSNormKernel`` whose PDL trigger comes first; everything else is unchanged."""

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
            # See the module docstring for why the trigger may precede the wait.
            cute.arch.griddepcontrol_launch_dependents()
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

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
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
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
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
            mY.element_type,
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


@functools.cache
def _compiled(dtype_str: str, H: int, sm_version: int, contiguous: bool):
    """Compile like FlashInfer's ``_get_compiled_rmsnorm_kernel`` (weight_bias 0, PDL on)."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = _LaunchReleaseRMSNormKernel(dtype, H, 0.0, sm_version=sm_version)
    sym_m = cute.sym_int(64)
    if contiguous:
        elem_bytes = dtype.width // 8
        tensor_align = math.gcd(128, H * elem_bytes)
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
    else:
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
        x_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
        )
        y_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
        )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int64(1),
        Float32(1e-6),
        True,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def can_use(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`rmsnorm_release_at_launch` serves ``x`` (2D BF16, unit column stride)."""
    return (
        IS_AVAILABLE
        and x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.shape[-1] == weight.numel()
        and x.data_ptr() % 16 == 0
        and x.stride(0) % 8 == 0
        and x.shape[0] * x.stride(0) < 2**31
    )


def rmsnorm_release_at_launch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``flashinfer.norm.rmsnorm(x, weight, eps, enable_pdl=True)``, bit for bit, with the PDL
    trigger at launch (see the module docstring). Returns a contiguous BF16 tensor."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    contiguous = x.is_contiguous()
    kernel = _compiled("bfloat16", x.shape[-1], get_sm_version(x.device), contiguous)
    kernel(x, weight, out, x.shape[0], eps)
    return out


__all__ = ["IS_AVAILABLE", "can_use", "rmsnorm_release_at_launch"]
