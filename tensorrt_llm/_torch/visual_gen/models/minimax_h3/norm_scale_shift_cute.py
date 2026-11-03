# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from SGLang python/sglang/kernels/ops/diffusion/norm/scale_residual_norm_cutedsl.py,
# cutedsl_common/norm_fusion.py, cutedsl_common/reduce.py and cutedsl_utils.py
# (https://github.com/sgl-project/sglang, commit 09b7383624c11a60e50af75ab40f65d4d4a99939),
# licensed under the Apache License, Version 2.0.
"""Fused RMSNorm + AdaLN modulation with per-row table lookup (CuTe DSL).

For every row ``r`` of ``x [S, D]``::

    y[r] = rmsnorm(x[r]) * weight * (1 + scale[indices[r]]) + shift[indices[r]]

One CTA per row; each thread moves its share of the row in 128-bit pieces. ``D`` must be a
multiple of 256 and at most 8192.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

_WARP_SIZE = 32
_VEC = 8
_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int64: cutlass.Int64,
    torch.int32: cutlass.Int32,
}
_COMPILE_CACHE = {}


@cute.jit
def _warp_reduce_sum(val: cute.Numeric) -> cute.Numeric:
    for i in range(5):
        val = val + cute.arch.shuffle_sync_down(val, offset=1 << (4 - i))
    return val


@cute.jit
def _cta_reduce_sum(
    val: cute.Numeric, num_warps: cutlass.Constexpr, tidx: cutlass.Int32
) -> cute.Numeric:
    smem = cutlass.utils.SmemAllocator()
    acc = smem.allocate_tensor(cutlass.Float32, num_warps + 1)
    warp_id = tidx >> 5
    lane_id = tidx & 31
    if lane_id == 0:
        acc[warp_id] = val
    cute.arch.sync_threads()
    if warp_id == 0:
        val = acc[lane_id] if lane_id < num_warps else cutlass.Float32(0)
        val = _warp_reduce_sum(val)
        if lane_id == 0:
            acc[num_warps] = val
    cute.arch.sync_threads()
    return acc[num_warps]


def _default_num_warps(D: int) -> int:
    # the largest warp count <= 4 that splits the row into whole 128-bit pieces per thread
    pieces = D // (_WARP_SIZE * _VEC)
    return max(w for w in range(1, 5) if pieces % w == 0)


class RMSNormIndexedScaleShift:
    """One CTA per row. x stays in registers across the row reduction; the weight and table rows are then
    loaded, combined and stored one 128-bit piece at a time, which keeps few registers live per row and
    lets more rows be in flight per SM. fp32 math with one rounding."""

    def __init__(self, D: int):
        self.D = D
        self.num_warps = _default_num_warps(D)
        assert D % (self.num_warps * _WARP_SIZE * _VEC) == 0
        self.num_threads = self.num_warps * _WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mY,
        mX,
        mW,
        mShift,
        mScale,
        mIdx,
        eps: cutlass.Float32 = cutlass.Float32(1e-5),
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        tiled_copy = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout(self.num_threads), cute.make_layout(_VEC)
        )
        self.kernel(mY, mX, mW, mShift, mScale, mIdx, tiled_copy, eps).launch(
            grid=[mX.shape[0], 1, 1], block=[self.num_threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self, mY, mX, mW, mShift, mScale, mIdx, tiled_copy: cute.TiledCopy, eps: cutlass.Float32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        thr_copy = tiled_copy.get_slice(tidx)
        mod_row = cutlass.Int32(mIdx[bid])

        @cute.jit
        def row_slice(mV, r):
            # local_tile keeps the gmem base alignment that the 128-bit copies need
            return thr_copy.partition_S(
                cute.local_tile(mV, tiler=(1, self.D), coord=(r, 0))[0, None]
            )

        tXgX = row_slice(mX, bid)
        tYgY = row_slice(mY, bid)
        tSCgSC = row_slice(mScale, mod_row)
        tSHgSH = row_slice(mShift, mod_row)
        tWgW = thr_copy.partition_S(mW)
        tXrX = cute.make_fragment_like(tXgX, tXgX.element_type)
        cute.autovec_copy(tXgX, tXrX)

        val = cute.Float32(0.0)
        for i in range(cute.size(tXrX)):
            x_fp32 = tXrX[i].to(cutlass.Float32)
            val += x_fp32 * x_fp32
        val = _warp_reduce_sum(val)
        acc_sq = _cta_reduce_sum(val, self.num_warps, tidx)
        factor = cute.rsqrt(acc_sq / self.D + eps)

        for j in cutlass.range_constexpr(cute.size(tXgX, mode=[1])):
            pW = cute.make_fragment_like(tWgW[None, j], tWgW.element_type)
            pSC = cute.make_fragment_like(tSCgSC[None, j], tSCgSC.element_type)
            pSH = cute.make_fragment_like(tSHgSH[None, j], tSHgSH.element_type)
            cute.autovec_copy(tWgW[None, j], pW)
            cute.autovec_copy(tSCgSC[None, j], pSC)
            cute.autovec_copy(tSHgSH[None, j], pSH)
            x = tXrX[None, j].load().to(cutlass.Float32)
            y = (x * factor * pW.load().to(cutlass.Float32)) * (
                1.0 + pSC.load().to(cutlass.Float32)
            ) + pSH.load().to(cutlass.Float32)
            pY = cute.make_fragment_like(tYgY[None, j], tYgY.element_type)
            pY.store(y.to(pY.element_type))
            cute.autovec_copy(pY, tYgY[None, j])


def _fake(t: torch.Tensor, static_last: bool):
    # Only the hidden dim is compile-time, so one build serves every S and table height.
    if static_last:
        D = t.shape[-1]
        shape = (*(cute.sym_int() for _ in range(t.ndim - 1)), D)
        stride = (*(cute.sym_int(divisibility=D) for _ in range(t.ndim - 1)), 1)
    else:
        shape, stride = (cute.sym_int(),), (1,)
    return cute.runtime.make_fake_tensor(
        _CUTE_DTYPE[t.dtype],
        shape,
        stride,
        memspace=cute.AddressSpace.gmem,
        assumed_align=32 if static_last else 8,
    )


def can_use_rmsnorm_indexed_scale_shift_cute(x, weight, shift, scale, indices) -> bool:
    D = x.shape[-1] if x.dim() == 2 else 0
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and D % (_WARP_SIZE * _VEC) == 0
        and 0 < D <= 8192
        and weight.dtype == x.dtype
        and weight.shape == (D,)
        and weight.stride(0) == 1
        and shift.dtype == scale.dtype == x.dtype
        and shift.dim() == scale.dim() == 2
        and shift.shape == scale.shape
        and shift.shape[1] == D
        and shift.stride(1) == scale.stride(1) == 1
        and shift.stride(0) % D == 0
        and scale.stride(0) % D == 0
        and shift.data_ptr() % 32 == 0
        and scale.data_ptr() % 32 == 0
        and indices.dim() == 1
        and indices.numel() == x.shape[0]
        and indices.stride(0) == 1
        and indices.dtype in (torch.int64, torch.int32)
    )


def rmsnorm_indexed_scale_shift_cute(x, weight, shift, scale, indices, eps: float):
    if not can_use_rmsnorm_indexed_scale_shift_cute(x, weight, shift, scale, indices):
        raise ValueError(
            "rmsnorm_indexed_scale_shift_cute: unsupported inputs "
            f"x {tuple(x.shape)}/{x.stride()}/{x.dtype}, shift {tuple(shift.shape)}/{shift.stride()}, "
            f"scale {tuple(scale.shape)}/{scale.stride()}, indices {tuple(indices.shape)}/{indices.dtype}"
        )
    x = x.contiguous()
    y = torch.empty_like(x)
    if x.shape[0] == 0:
        return y
    tensors = [y, x, weight, shift, scale, indices]
    key = (x.dtype, x.shape[-1], indices.dtype)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        kernel = RMSNormIndexedScaleShift(x.shape[-1])
        fakes = [_fake(t, t is not indices) for t in tensors]
        fakes[2] = cute.runtime.make_fake_tensor(
            _CUTE_DTYPE[weight.dtype],
            (x.shape[-1],),
            (1,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=32,
        )
        compiled = cute.compile(kernel, *fakes, options="--enable-tvm-ffi")
        _COMPILE_CACHE[key] = compiled
    compiled(*tensors, eps, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return y


@torch.library.custom_op(
    "trtllm::minimax_h3_rmsnorm_indexed_scale_shift_cute",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def rmsnorm_indexed_scale_shift_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``rmsnorm(x) * weight * (1 + scale[indices]) + shift[indices]`` over ``[S, D]`` rows."""
    return rmsnorm_indexed_scale_shift_cute(x, weight, shift, scale, indices, eps)


@rmsnorm_indexed_scale_shift_op.register_fake
def _(x, weight, shift, scale, indices, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
