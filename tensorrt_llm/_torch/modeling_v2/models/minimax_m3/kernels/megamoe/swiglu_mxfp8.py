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
"""The SwiGLU of BF16 rows and the MXFP8 form of its result in one kernel (``swiglu_mxfp8``).

Per row t of x = [gate | up], each half I wide, and per element, in FP32:
    a    = min(gate, limit)
    b    = min(max(up, -limit), limit)
    h[t] = bf16(a * (1 / (1 + 2^((alpha * a) * -log2(e)))) * (b + beta))
as the Triton ``silu_and_mul_kernel`` behind ``torch.ops.trtllm.silu_and_mul`` computes it (without a limit neither
clamp runs): every product and sum rounded to nearest on its own, the exponential ``ex2.approx.f32`` and the
reciprocal ``RECIPROCALS[reciprocal]``. The MXFP8 form of h is ``mxfp8_rows.store_mxfp8_vector``'s, bit for bit
``trtllm::mxfp8_quantize(h, True)``. Each CTA owns ``num_threads`` vectors of one row.
"""

import functools
import struct
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .megamoe_topk_reduce_tail import _aligned
from .mxfp8_rows import (
    VEC,
    check_mxfp8_rows,
    mxfp8_fakes,
    rcp_approx_ftz,
    store_mxfp8_vector,
    zero_mxfp8_pad_rows,
)

#: The PTX of ``1 / d`` per form, ``{dst}`` and ``{src}`` the registers.
RECIPROCALS = {
    "div.full": "div.full.f32 {dst}, 0f3F800000, {src};",
    "div.rn": "div.rn.f32 {dst}, 0f3F800000, {src};",
    "div.approx": "div.approx.f32 {dst}, 0f3F800000, {src};",
    "rcp.approx": "rcp.approx.f32 {dst}, {src};",
}
#: The form ``swiglu_mxfp8`` runs.
RECIPROCAL = "div.full"

_LOG2E = 1.4426950408889634


def _f32_literal(value: float) -> str:
    """``value`` rounded to FP32, as a PTX hexadecimal literal."""
    return "0f" + struct.pack(">f", value).hex().upper()


def swiglu_ptx(alpha: float, beta: float, limit: Optional[float], reciprocal: str) -> str:
    """The PTX of one element: ``$1`` gate, ``$2`` up, ``$0`` the FP32 result."""
    if reciprocal not in RECIPROCALS:
        raise ValueError(f"reciprocal must be one of {sorted(RECIPROCALS)}, got {reciprocal!r}")
    if limit is None:
        clamps = ["mov.f32 sw_a, $1;", "mov.f32 sw_b, $2;"]
    else:
        clamps = [
            f"min.f32 sw_a, $1, {_f32_literal(limit)};",
            f"max.f32 sw_b, $2, {_f32_literal(-limit)};",
            f"min.f32 sw_b, sw_b, {_f32_literal(limit)};",
        ]
    body = clamps + [
        f"mul.rn.f32 sw_t, sw_a, {_f32_literal(alpha)};",
        f"mul.rn.f32 sw_t, sw_t, {_f32_literal(-_LOG2E)};",
        "ex2.approx.f32 sw_t, sw_t;",
        "add.rn.f32 sw_t, sw_t, 0f3F800000;",
        RECIPROCALS[reciprocal].format(dst="sw_t", src="sw_t"),
        "mul.rn.f32 sw_a, sw_a, sw_t;",
        f"add.rn.f32 sw_b, sw_b, {_f32_literal(beta)};",
        "mul.rn.f32 $0, sw_a, sw_b;",
    ]
    return "{\n\t.reg .f32 sw_a, sw_b, sw_t;\n\t" + "\n\t".join(body) + "\n\t}\n"


def _swiglu_op(ptx: str):
    @dsl_user_op
    def swiglu(gate: Float32, up: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(gate).ir_value(loc=loc, ip=ip), Float32(up).ir_value(loc=loc, ip=ip)],
                ptx,
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                loc=loc,
                ip=ip,
            )
        )

    return swiglu


class SwigluMxfp8Kernel:
    """The SwiGLU of BF16 rows [M, 2 * I] into BF16 rows [M, I], plus their MXFP8 form.

    Args:
        width: The output width I, a multiple of 256 (whole warps of 8-element vectors).
        num_threads: Threads per CTA, whole warps; I / 8 must be a multiple of it.
        alpha: The gain inside the sigmoid.
        beta: The offset added to up.
        limit: The clamp of gate (from above) and up (both sides), or None for none.
        reciprocal: The key of ``RECIPROCALS`` that computes the sigmoid's reciprocal.
    """

    def __init__(
        self,
        width: int,
        num_threads: int,
        alpha: float,
        beta: float,
        limit: Optional[float],
        reciprocal: str,
    ):
        num_vectors = width // VEC
        if width % 256 != 0 or num_threads % 32 != 0 or num_vectors % num_threads != 0:
            raise ValueError(
                f"{num_threads} threads do not tile the {num_vectors} vectors of a {width}-wide row"
            )
        self.width = width
        self.num_threads = num_threads
        self.num_segments = num_vectors // num_threads
        self._swiglu = _swiglu_op(swiglu_ptx(alpha, beta, limit, reciprocal))

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mH: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mX, mH, mQ, mSF, Int32(M), enable_pdl).launch(
            grid=[M, self.num_segments, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mH: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        num_rows: Int32,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, segment, _ = cute.arch.block_idx()
        num_threads: cutlass.Constexpr[int] = self.num_threads
        num_vectors: cutlass.Constexpr[int] = self.width // VEC
        num_k_tiles: cutlass.Constexpr[int] = (self.width // 32 + 3) // 4

        bf16_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        vec_bytes = VEC * cutlass.BFloat16.width // 8
        gX = cute.zipped_divide(mX[row, None], (VEC,))
        gH = cute.zipped_divide(mH[row, None], (VEC,))

        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        v = segment * num_threads + tidx
        gate = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
        up = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
        cute.copy(bf16_atom, _aligned(gX[(None,), v], vec_bytes), gate)
        cute.copy(bf16_atom, _aligned(gX[(None,), v + num_vectors], vec_bytes), up)
        a = cute.make_rmem_tensor((VEC,), Float32)
        b = cute.make_rmem_tensor((VEC,), Float32)
        a.store(gate.load().to(Float32))
        b.store(up.load().to(Float32))
        h = cute.make_rmem_tensor((VEC,), Float32)
        for j in cutlass.range_constexpr(VEC):
            h[j] = self._swiglu(a[j], b[j])
        out = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
        out.store(h.load().to(cutlass.BFloat16))
        cute.copy(bf16_atom, out, _aligned(gH[(None,), v], vec_bytes))
        store_mxfp8_vector(out, mQ, mSF, row, v, num_k_tiles, rcp_approx_ftz(Float32(448.0)))
        if segment == 0:
            zero_mxfp8_pad_rows(mSF, row, num_rows, tidx, num_threads, num_k_tiles)


def _num_threads(width: int) -> int:
    """128 threads where they tile the row's vectors, else the largest whole-warp count that does."""
    num_vectors = width // VEC
    for threads in (128, 96, 64, 32):
        if num_vectors % threads == 0:
            return threads
    raise ValueError(f"no whole-warp CTA tiles the {num_vectors} vectors of a {width}-wide row")


@functools.cache
def _compile_swiglu_mxfp8(
    width: int,
    alpha: float,
    beta: float,
    limit: Optional[float],
    reciprocal: str,
    enable_pdl: bool,
):
    kernel_obj = SwigluMxfp8Kernel(width, _num_threads(width), alpha, beta, limit, reciprocal)
    sym_m = cute.sym_int(64)
    x = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, 2 * width), stride_order=(1, 0), assumed_align=16
    )
    h = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, width), stride_order=(1, 0), assumed_align=16
    )
    q, sf = mxfp8_fakes(width, cute.sym_int())
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj, x, h, q, sf, Int64(1), enable_pdl, stream, options="--enable-tvm-ffi"
    )


def swiglu_mxfp8(
    x: torch.Tensor,
    h: torch.Tensor,
    q: torch.Tensor,
    sf: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: Optional[float],
    enable_pdl: bool,
    reciprocal: str = RECIPROCAL,
) -> None:
    """``h = torch.ops.trtllm.silu_and_mul(x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta)`` and
    ``trtllm::mxfp8_quantize(h, True)`` into ``q`` / ``sf``, in one launch.

    Args:
        x: [M, 2 * I] BF16, contiguous, 16-byte aligned: gate in the first half, up in the second; I a multiple
            of 256.
        h: [M, I] BF16, contiguous, 16-byte aligned. Receives the SwiGLU.
        q: E4M3 [M, I], contiguous. Receives the MXFP8 rows.
        sf: UINT8, at least ``mxfp8_rows.scale_bytes(M, I)``. Receives the swizzled UE8M0 scales.
        alpha: The gain inside the sigmoid.
        beta: The offset added to up.
        limit: The clamp of gate (from above) and up (both sides), or None for none.
        enable_pdl: Launch with programmatic dependent launch.
        reciprocal: The key of ``RECIPROCALS`` for the sigmoid's reciprocal.
    """
    if x.dim() != 2 or h.dim() != 2 or x.shape[0] != h.shape[0] or x.shape[1] != 2 * h.shape[1]:
        raise ValueError(f"x {tuple(x.shape)} is not [M, 2 * I] for h {tuple(h.shape)}")
    for name, tensor in (("x", x), ("h", h)):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16")
        if not tensor.is_contiguous() or tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} must be contiguous and 16-byte aligned")
    num_rows, width = h.shape
    if width % 256 != 0:
        raise ValueError(f"I {width} is not a multiple of 256 (whole warps of 8-element vectors)")
    check_mxfp8_rows(q, sf, num_rows, width)
    if num_rows == 0:
        return
    kernel = _compile_swiglu_mxfp8(
        width,
        float(alpha),
        float(beta),
        None if limit is None else float(limit),
        reciprocal,
        enable_pdl,
    )
    kernel(x, h, q.view(torch.int64), sf, num_rows)


__all__ = ["RECIPROCAL", "RECIPROCALS", "SwigluMxfp8Kernel", "swiglu_mxfp8", "swiglu_ptx"]
