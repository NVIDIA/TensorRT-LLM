# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 dynamic block convolution in one launch (CuTe DSL).

``out[t, c] = sum_i (base[i, c] + delta[t, i, c // group_size]) * x[t - i, c]``, where a tap
``i`` that reaches past the start of row ``t``'s drafted block (``t % block_size < i``)
contributes ``coefficient * 0.0``. Every operation is the FP32 one the torch reference performs,
in the same order: ``base + delta`` per tap, the tap-0 product, then per tap ``acc + coef * x``,
each product and sum rounded to nearest on its own (``mul.rn`` / ``add.rn``, never contracted to
an FMA), and one round to the activation dtype at the end. The output is therefore bitwise the
reference's, signed zeros and NaN propagation included: a tap past the block start multiplies
an explicit ``0.0`` (selected, not loaded), so a NaN in the previous request's row never leaks.

One thread owns one row and eight consecutive channels (one 16-byte vector; eight channels never
straddle a coefficient group because ``group_size`` is a multiple of eight).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

#: Channels per thread: one 16-byte vector of a 2-byte activation.
VEC = 8

_CUTE_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a + b`` rounded to nearest, never contracted."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b`` rounded to nearest, never contracted."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _add_mul_select_rn(
    acc: Float32, coef: Float32, x: Float32, keep: Int32, *, loc=None, ip=None
) -> Float32:
    """``acc + coef * (keep ? x : 0.0)``, the product and the sum each rounded to nearest."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                acc.ir_value(loc=loc, ip=ip),
                coef.ir_value(loc=loc, ip=ip),
                x.ir_value(loc=loc, ip=ip),
                keep.ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred cvkp;\n\t.reg .f32 cvks, cvkm;\n\t"
            "setp.ne.s32 cvkp, $4, 0;\n\t"
            "selp.f32 cvks, $3, 0f00000000, cvkp;\n\t"
            "mul.rn.f32 cvkm, $2, cvks;\n\t"
            "add.rn.f32 $0, $1, cvkm;\n\t}",
            "=f,f,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class DFlash2GroupedConvKernel:
    """``out = Conv(x; base, delta)`` over request-major rows, one thread per row and 8 channels."""

    num_threads = 256

    def __init__(self, hidden_size: int, taps: int, group_size: int):
        if group_size % VEC or hidden_size % group_size:
            raise ValueError(
                f"group_size ({group_size}) must be a multiple of {VEC} dividing hidden_size "
                f"({hidden_size})"
            )
        if taps < 1:
            raise ValueError(f"taps must be positive; got {taps}")
        self.hidden_size = hidden_size
        self.taps = taps
        self.vectors = hidden_size // VEC
        self.vectors_per_group = group_size // VEC

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        delta: cute.Tensor,
        base: cute.Tensor,
        out: cute.Tensor,
        block_size: Int32,
        stream: cuda.CUstream,
    ):
        gx = cute.zipped_divide(x, (1, VEC))
        gb = cute.zipped_divide(base, (1, VEC))
        go = cute.zipped_divide(out, (1, VEC))
        work = x.shape[0] * self.vectors
        self.kernel(gx, delta, gb, go, block_size).launch(
            grid=[cute.ceil_div(work, self.num_threads), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gx: cute.Tensor,
        delta: cute.Tensor,
        gb: cute.Tensor,
        go: cute.Tensor,
        block_size: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_rows = delta.shape[0]
        idx = bidx * self.num_threads + tidx
        if idx < num_rows * self.vectors:
            row = idx // self.vectors
            vec = idx - row * self.vectors
            pos = row % block_size
            group = vec // self.vectors_per_group

            acc = cute.make_rmem_tensor((VEC,), Float32)
            x_tile = gx[(None, (row, vec))]
            xs = cute.make_fragment_like(x_tile)
            cute.autovec_copy(x_tile, xs)
            b_tile = gb[(None, (0, vec))]
            bs = cute.make_fragment_like(b_tile)
            cute.autovec_copy(b_tile, bs)
            d = Float32(delta[row, 0, group])
            for k in cutlass.range_constexpr(VEC):
                acc[k] = _mul_rn(_add_rn(Float32(bs[k]), d), Float32(xs[k]))

            for tap in cutlass.range_constexpr(1, self.taps):
                # Past the block start the tap reads its own row and multiplies 0.0 instead.
                keep = Int32(0)
                src = row
                if pos >= tap:
                    keep = Int32(1)
                    src = row - tap
                p_tile = gx[(None, (src, vec))]
                ps = cute.make_fragment_like(p_tile)
                cute.autovec_copy(p_tile, ps)
                b_tile = gb[(None, (tap, vec))]
                bs = cute.make_fragment_like(b_tile)
                cute.autovec_copy(b_tile, bs)
                d = Float32(delta[row, tap, group])
                for k in cutlass.range_constexpr(VEC):
                    acc[k] = _add_mul_select_rn(
                        acc[k], _add_rn(Float32(bs[k]), d), Float32(ps[k]), keep
                    )

            o_tile = go[(None, (row, vec))]
            os_ = cute.make_fragment_like(o_tile)
            for k in cutlass.range_constexpr(VEC):
                os_[k] = acc[k].to(go.element_type)
            cute.autovec_copy(os_, o_tile)


@functools.cache
def _compile(hidden_size: int, taps: int, group_size: int, dtype: torch.dtype):
    cute_dtype = _CUTE_DTYPES[dtype]
    rows = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (rows, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    delta_fake = cute.runtime.make_fake_tensor(
        cute_dtype,
        (rows, taps, hidden_size // group_size),
        stride=(cute.sym_int64(), cute.sym_int64(), 1),
    )
    base_fake = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (taps, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (rows, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DFlash2GroupedConvKernel(hidden_size, taps, group_size)
    return cute.compile(
        kernel,
        x_fake,
        delta_fake,
        base_fake,
        out_fake,
        Int32(1),
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dflash2_grouped_conv_into(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write the convolution of ``hidden_states`` into ``out`` with one launch; returns ``out``.

    Args:
        hidden_states: ``[rows, hidden]`` BF16/FP16, contiguous, 16-byte aligned; request-major
            rows, ``block_size`` per request.
        delta: ``[rows, taps, hidden // group_size]`` of the same dtype, unit last stride (any
            other strides: the side's view of the kernel projection's output).
        base_kernel: ``[taps, hidden]`` of the same dtype, contiguous, 16-byte aligned.
        out: ``[rows, hidden]`` like ``hidden_states``.

    The kernel compiles once per ``(hidden, taps, group_size, dtype)`` on the first call, which
    must run outside CUDA-graph capture; later calls only launch.
    """
    rows, hidden_size = hidden_states.shape
    if rows == 0:
        return out
    compiled = _compile(hidden_size, base_kernel.shape[0], group_size, hidden_states.dtype)
    # detach(): DLPack export refuses tensors that require grad (parameters).
    compiled(hidden_states.detach(), delta.detach(), base_kernel.detach(), out, int(block_size))
    return out


__all__ = ["DFlash2GroupedConvKernel", "dflash2_grouped_conv_into"]
