# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/cute_utils/__init__.py
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/cute_utils/cvt.py
"""Warp-level PTX intrinsics for CuTe DSL kernels that use ``mma.sync``.

Most Blackwell kernels in this tree drive the tensor cores through
``cute.gemm`` / tcgen05, which the CuTe DSL exposes directly. A kernel whose
GEMM has a tiny N dimension does better with warp-level ``mma.sync`` and high
CTA occupancy than with a deep single-CTA tcgen05 pipeline, and the DSL has no
wrapper for that instruction, so it is spelled out as inline PTX here.

Vendored from the vLLM sources linked in the file header
(v0.26.1rc0-77-g6f91edf96), reduced to the symbols
:mod:`minimax_m3_index_decode_score` needs.
"""

import torch
from cutlass import BFloat16, Float8E4M3FN, Float16, Float32, Int32, Int64, Uint32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "EVICT_FIRST",
    "TORCH_TO_CUTE_DTYPE",
    "fp8x4_to_fp16x4",
    "mma_sync",
    "simple_tma_copy",
]

TORCH_TO_CUTE_DTYPE = {
    torch.bfloat16: BFloat16,
    torch.float8_e4m3fn: Float8E4M3FN,
}

_CUTE_TO_PTX_DTYPE = {
    BFloat16: "bf16",
    Float16: "f16",
    Float8E4M3FN: "e4m3",
    Float32: "f32",
}

# L2 cache-eviction policy descriptor; see CUTLASS
# include/cute/arch/copy_sm90_desc.hpp (v4.3.2, L193-L197).
EVICT_FIRST = Int64(0x12F0000000000000)


def simple_tma_copy(atom, src, dst, mbar=None, cache_policy=None):
    """Wrap ``group_modes()`` + ``tma_partition()`` for a whole-tile TMA copy.

    Call this WITHOUT ``cute.elect_one()``: ``tma_partition`` already reduces
    the copy to a single issuing lane.
    """
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem = src
        smem = dst
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        smem = src
        gmem = dst
    else:
        raise ValueError(f"simple_tma_copy expects a bulk-tensor TMA atom, got {atom.op!r}.")

    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )

    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=cache_policy)
    else:
        cute.copy(atom, s_part, g_part, cache_policy=cache_policy)


@dsl_user_op
def mma_sync(a, b, c: cute.Tensor, *, loc=None, ip=None):
    """Warp-level ``mma.sync.aligned.m16n8kK`` accumulating into ``c``.

    ``K`` follows from the operand width (32B of A per lane), so this covers
    m16n8k16 for 16-bit operands and m16n8k32 for 8-bit ones.
    """
    a_ty = _CUTE_TO_PTX_DTYPE[a.element_type]
    b_ty = _CUTE_TO_PTX_DTYPE[b.element_type]
    c_ty = _CUTE_TO_PTX_DTYPE[c.element_type]
    mlir_ty = c.element_type.mlir_type
    K = 256 // a.element_type.width  # 32B

    # recast_tensor needs tensor-backed fragments, so materialize SSA values
    # here and let callers pass converted FP8 fragments straight through.
    if isinstance(a, cute.TensorSSA):
        a_ = cute.make_rmem_tensor_like(a)
        a_.store(a, loc=loc, ip=ip)
        a = a_
    if isinstance(b, cute.TensorSSA):
        b_ = cute.make_rmem_tensor_like(b)
        b_.store(b, loc=loc, ip=ip)
        b = b_

    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        f"mma.sync.aligned.m16n8k{K}.row.col.{c_ty}.{a_ty}.{b_ty}.{c_ty} "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, c.element_type)


@dsl_user_op
def fp8x4_to_fp16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    """Convert four packed E4M3 values to four FP16 values, as two ``Uint32``.

    SM100 has no native ``mma.sync.f8``; ptxas lowers it to F2FP.F16.E4M3 plus
    HMMA anyway, so converting explicitly gives better codegen and keeps the
    two FP16 k-fragments visible to the caller.
    """
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, hi;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)
