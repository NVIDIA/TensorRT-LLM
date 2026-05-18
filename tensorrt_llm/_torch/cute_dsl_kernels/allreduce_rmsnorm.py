# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental CuTe bf16 allreduce + residual + RMSNorm kernel.

This module is intentionally narrow and supports eager experimentation only.
The prototype still uses host-side symmetric-memory barriers, so CUDA graph
capture is outside this path's supported envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith, llvm, vector
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import dsl_user_op

_DEFAULT_TPC_VALUES = (64, 128, 256, 512, 1024)


@dataclass
class _SymmTensor:
    tensor: torch.Tensor
    handle: object

    @property
    def multicast_ptr(self) -> int | None:
        try:
            ptr = self.handle.multicast_ptr
        except RuntimeError:
            return None
        return int(ptr) if ptr else None


@dataclass
class _Runtime:
    symm_input: _SymmTensor
    launch: Callable
    threads_per_cta: int


_RUNTIME_CACHE: dict[tuple[int, int, int, torch.dtype], _Runtime] = {}


def _require_cute() -> None:
    if not IS_CUTLASS_DSL_AVAILABLE:
        raise NotImplementedError("CuTe DSL is not available")


def _current_cu_stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _valid_threads_per_cta(hidden_size: int) -> int:
    valid = tuple(tpc for tpc in _DEFAULT_TPC_VALUES if hidden_size % (tpc * 8) == 0)
    if not valid:
        raise NotImplementedError(f"hidden_size={hidden_size} is not supported")
    return valid[-1]


def _world_size_supports_multicast(world_size: int) -> bool:
    return world_size <= 4


def _enable_symmetric_memory() -> None:
    enable_fn = getattr(symm_mem, "enable_symm_mem_for_group", None)
    if enable_fn is not None:
        try:
            enable_fn(dist.group.WORLD.group_name)
        except RuntimeError:
            pass


def _alloc_symm_tensor(numel: int, dtype: torch.dtype) -> _SymmTensor:
    tensor = symm_mem.empty(numel, dtype=dtype, device="cuda")
    handle = symm_mem.rendezvous(tensor, dist.group.WORLD)
    return _SymmTensor(tensor=tensor, handle=handle)


if IS_CUTLASS_DSL_AVAILABLE:

    @cute.jit
    def _raw_tensor_1d(addr: Int64, dtype, size):
        ptr = cute.make_ptr(dtype, addr, cute.AddressSpace.gmem)
        return cute.make_tensor(ptr, cute.make_layout((size,), stride=(1,)))

    @cute.jit
    def _raw_tensor_2d(addr: Int64, dtype, rows, cols):
        ptr = cute.make_ptr(dtype, addr, cute.AddressSpace.gmem)
        return cute.make_tensor(ptr, cute.make_layout((rows, cols), stride=(cols, 1)))

    @dsl_user_op
    def _i32_to_bf16_pair(x, *, loc=None, ip=None):
        x_ir = x.ir_value(loc=loc, ip=ip) if hasattr(x, "ir_value") else x
        vec_ty = ir.VectorType.get([2], BFloat16.mlir_type, loc=loc)
        vec = llvm.bitcast(vec_ty, x_ir, loc=loc, ip=ip)
        lo = vector.extractelement(
            vec,
            position=arith.constant(Int32.mlir_type, 0, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        hi = vector.extractelement(
            vec,
            position=arith.constant(Int32.mlir_type, 1, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        return BFloat16(lo), BFloat16(hi)

    @cute.kernel
    def _fused_ar_rmsnorm_bf16_kernel(
        mc_in: cute.Tensor,
        uc_residual: cute.Tensor,
        uc_weight: cute.Tensor,
        uc_h_out: cute.Tensor,
        uc_y_out: cute.Tensor,
        eps: Float32,
        H: cutlass.Constexpr,
        vec_chunks: cutlass.Constexpr,
        threads_per_cta: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        frag_h = cute.make_rmem_tensor((vec_chunks, 8), Float32)
        sum_sq = Float32(0.0)

        for c in cutlass.range_constexpr(vec_chunks):
            col = c * threads_per_cta * 8 + tidx * 8
            offset = row * H + col

            r0, r1, r2, r3 = utils.distributed.multimem_ld_reduce_8xbf16(mc_in.iterator + offset)
            b0, b1 = _i32_to_bf16_pair(r0)
            b2, b3 = _i32_to_bf16_pair(r1)
            b4, b5 = _i32_to_bf16_pair(r2)
            b6, b7 = _i32_to_bf16_pair(r3)

            h0 = Float32(b0) + Float32(uc_residual[row, col + 0])
            h1 = Float32(b1) + Float32(uc_residual[row, col + 1])
            h2 = Float32(b2) + Float32(uc_residual[row, col + 2])
            h3 = Float32(b3) + Float32(uc_residual[row, col + 3])
            h4 = Float32(b4) + Float32(uc_residual[row, col + 4])
            h5 = Float32(b5) + Float32(uc_residual[row, col + 5])
            h6 = Float32(b6) + Float32(uc_residual[row, col + 6])
            h7 = Float32(b7) + Float32(uc_residual[row, col + 7])

            uc_h_out[row, col + 0] = BFloat16(h0)
            uc_h_out[row, col + 1] = BFloat16(h1)
            uc_h_out[row, col + 2] = BFloat16(h2)
            uc_h_out[row, col + 3] = BFloat16(h3)
            uc_h_out[row, col + 4] = BFloat16(h4)
            uc_h_out[row, col + 5] = BFloat16(h5)
            uc_h_out[row, col + 6] = BFloat16(h6)
            uc_h_out[row, col + 7] = BFloat16(h7)

            sum_sq = (
                sum_sq
                + h0 * h0
                + h1 * h1
                + h2 * h2
                + h3 * h3
                + h4 * h4
                + h5 * h5
                + h6 * h6
                + h7 * h7
            )

            frag_h[c, 0] = h0
            frag_h[c, 1] = h1
            frag_h[c, 2] = h2
            frag_h[c, 3] = h3
            frag_h[c, 4] = h4
            frag_h[c, 5] = h5
            frag_h[c, 6] = h6
            frag_h[c, 7] = h7

        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = threads_per_cta // 32

        sum_sq = cute.arch.warp_reduction_sum(sum_sq)
        smem_ptr = cute.arch.alloc_smem(Float32, num_warps + 1)
        smem = cute.make_tensor(smem_ptr, cute.make_layout((num_warps + 1,)))

        if lane_idx == 0:
            smem[warp_idx] = sum_sq
        cute.arch.barrier()

        if warp_idx == 0:
            part = Float32(0.0)
            if lane_idx < num_warps:
                part = smem[lane_idx]
            part = cute.arch.warp_reduction_sum(part)
            if lane_idx == 0:
                smem[num_warps] = cute.math.rsqrt(part / Float32(H) + eps, fastmath=True)
        cute.arch.barrier()

        inv_rms = smem[num_warps]

        for c in cutlass.range_constexpr(vec_chunks):
            col = c * threads_per_cta * 8 + tidx * 8

            uc_y_out[row, col + 0] = BFloat16(frag_h[c, 0] * inv_rms * Float32(uc_weight[col + 0]))
            uc_y_out[row, col + 1] = BFloat16(frag_h[c, 1] * inv_rms * Float32(uc_weight[col + 1]))
            uc_y_out[row, col + 2] = BFloat16(frag_h[c, 2] * inv_rms * Float32(uc_weight[col + 2]))
            uc_y_out[row, col + 3] = BFloat16(frag_h[c, 3] * inv_rms * Float32(uc_weight[col + 3]))
            uc_y_out[row, col + 4] = BFloat16(frag_h[c, 4] * inv_rms * Float32(uc_weight[col + 4]))
            uc_y_out[row, col + 5] = BFloat16(frag_h[c, 5] * inv_rms * Float32(uc_weight[col + 5]))
            uc_y_out[row, col + 6] = BFloat16(frag_h[c, 6] * inv_rms * Float32(uc_weight[col + 6]))
            uc_y_out[row, col + 7] = BFloat16(frag_h[c, 7] * inv_rms * Float32(uc_weight[col + 7]))

    @cute.jit
    def _launch_fused_ar_rmsnorm_bf16_stream(
        mc_addr_in: Int64,
        uc_addr_residual: Int64,
        uc_addr_weight: Int64,
        uc_addr_h_out: Int64,
        uc_addr_y_out: Int64,
        eps: Float32,
        stream: cuda.CUstream,
        M: cutlass.Constexpr,
        H: cutlass.Constexpr,
        threads_per_cta: cutlass.Constexpr = 512,
    ):
        mc_in = _raw_tensor_2d(mc_addr_in, BFloat16, M, H)
        uc_res = _raw_tensor_2d(uc_addr_residual, BFloat16, M, H)
        uc_w = _raw_tensor_1d(uc_addr_weight, BFloat16, H)
        uc_h = _raw_tensor_2d(uc_addr_h_out, BFloat16, M, H)
        uc_y = _raw_tensor_2d(uc_addr_y_out, BFloat16, M, H)
        vec_chunks = H // (threads_per_cta * 8)

        _fused_ar_rmsnorm_bf16_kernel(
            mc_in,
            uc_res,
            uc_w,
            uc_h,
            uc_y,
            eps,
            H,
            vec_chunks,
            threads_per_cta,
        ).launch(grid=[M, 1, 1], block=[threads_per_cta, 1, 1], stream=stream)


def _make_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
) -> _Runtime:
    _require_cute()
    rows, hidden_size = input_2d.shape
    symm_input = _alloc_symm_tensor(input_2d.numel(), input_2d.dtype)
    mc_addr = symm_input.multicast_ptr
    if mc_addr is None:
        raise NotImplementedError("symmetric memory multicast pointer is unavailable")

    h_out = torch.empty_like(input_2d)
    y_out = torch.empty_like(input_2d)
    compile_args = (
        Int64(mc_addr),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(h_out.data_ptr()),
        Int64(y_out.data_ptr()),
        Float32(eps),
        make_fake_stream(),
        rows,
        hidden_size,
    )
    launch = cute.compile(_launch_fused_ar_rmsnorm_bf16_stream, *compile_args, threads_per_cta)
    return _Runtime(symm_input=symm_input, launch=launch, threads_per_cta=threads_per_cta)


def _get_runtime(
    input_2d: torch.Tensor,
    residual_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    threads_per_cta: int,
) -> _Runtime:
    key = (
        torch.cuda.current_device(),
        input_2d.shape[0],
        input_2d.shape[1],
        input_2d.dtype,
    )
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None or runtime.threads_per_cta != threads_per_cta:
        runtime = _make_runtime(input_2d, residual_2d, weight, eps, threads_per_cta)
        _RUNTIME_CACHE[key] = runtime
    return runtime


def fused_allreduce_residual_rmsnorm_bf16(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the eager-only CuTe POC.

    Raises:
        NotImplementedError: If this call is outside the narrow supported POC
            envelope.
    """
    _require_cute()
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("CuTe allreduce RMSNorm POC is not CUDA-graph safe")
    if not dist.is_initialized():
        raise NotImplementedError("torch.distributed must be initialized")
    if input.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise NotImplementedError("only bf16 activations are supported")
    if weight.dtype != torch.bfloat16:
        raise NotImplementedError("only bf16 RMSNorm weight is supported")
    if input.shape != residual.shape:
        raise NotImplementedError("input and residual shapes must match")
    if weight.ndim != 1:
        raise NotImplementedError("RMSNorm weight must be 1D")

    world_size = dist.get_world_size()
    if not _world_size_supports_multicast(world_size):
        raise NotImplementedError(f"world_size={world_size} is outside the POC envelope")

    hidden_size = weight.numel()
    if input.numel() % hidden_size != 0:
        raise NotImplementedError("activation size is not divisible by hidden size")

    input_2d = input.contiguous().view(-1, hidden_size)
    residual_2d = residual.contiguous().view(-1, hidden_size)
    threads_per_cta = _valid_threads_per_cta(hidden_size)

    _enable_symmetric_memory()
    cutlass.cuda.initialize_cuda_context(torch.cuda.current_device())
    runtime = _get_runtime(input_2d, residual_2d, weight, eps, threads_per_cta)
    runtime.symm_input.tensor.copy_(input_2d.reshape(-1))
    mc_addr = runtime.symm_input.multicast_ptr
    if mc_addr is None:
        raise NotImplementedError("symmetric memory multicast pointer is unavailable")

    h_out = torch.empty_like(input_2d)
    y_out = torch.empty_like(input_2d)

    runtime.symm_input.handle.barrier()
    runtime.launch(
        Int64(mc_addr),
        Int64(residual_2d.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(h_out.data_ptr()),
        Int64(y_out.data_ptr()),
        Float32(eps),
        _current_cu_stream(),
    )
    torch.cuda.synchronize()
    runtime.symm_input.handle.barrier()
    return y_out.view_as(input), h_out.view_as(input)
