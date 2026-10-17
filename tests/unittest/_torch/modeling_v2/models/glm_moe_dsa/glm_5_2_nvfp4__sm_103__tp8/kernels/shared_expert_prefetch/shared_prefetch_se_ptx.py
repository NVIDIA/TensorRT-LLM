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
"""Inline-PTX primitives of the shared_expert_prefetch shared-expert kernels
(``shared_prefetch_se_kernel.py``), in addition to routed_moe's."""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int64, T, Uint32, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.ptx import (  # noqa: E501
    _asm,
)


@dsl_user_op
def mma_bf16_16816(
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    *,
    loc=None,
    ip=None,
):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``: D = A x B + C.

    A fragment (row-major 16 x 16): ``a0`` = (row g, k 2t..2t+1), ``a1`` =
    (row g+8, k 2t..), ``a2`` = (row g, k 2t+8..), ``a3`` = (row g+8, k
    2t+8..); B fragment (16 x 8, column n = g): ``b0`` = (k 2t..2t+1),
    ``b1`` = (k 2t+8..); C/D: (row g, n 2t / 2t+1), (row g+8, n 2t / 2t+1),
    with g = lane / 4 and t = lane % 4.
    """
    ops = [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
    ops += [Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)]
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        ops,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def st_u16(addr: Int64, value: Uint32, *, loc=None, ip=None):
    """16-bit global store of the low half of ``value``."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,r",
        loc=loc,
        ip=ip,
    )


# Candidate instruction sequences of the native Triton SiLU-and-mul for an FP32
# gate ``$1`` and up ``$2`` (``a * sigmoid(a) * (b + 0.0)`` with ``sigmoid(a) =
# 1 / (1 + exp(-a))``): the exponential as ``ex2.approx`` of ``-a * log2(e)``
# (with or without flush-to-zero) and the reciprocal as an IEEE or an
# approximate division. ``SILU_DEFAULT`` is Triton's lowering of FP32 ``exp``
# (``ex2.approx.f32``) and FP32 ``/`` (``div.full.f32``); the shared_expert_prefetch tests check
# every candidate against the Triton kernel over every BF16 gate value.
_SILU_EX2 = ("ex2.approx.f32", "ex2.approx.ftz.f32")
_SILU_DIV = ("div.rn.f32", "div.full.f32", "div.approx.f32")
SILU_VARIANTS = tuple((e, d) for d in _SILU_DIV for e in _SILU_EX2)
SILU_DEFAULT = 2


def _silu_text(variant: int) -> str:
    ex2, div = SILU_VARIANTS[variant]
    return (
        "{\n"
        ".reg .f32 n, e, d, s, gs, ub;\n"
        "mul.rn.f32 n, $1, 0fBFB8AA3B;\n"
        f"{ex2} e, n;\n"
        "add.rn.f32 d, e, 0f3F800000;\n"
        f"{div} s, 0f3F800000, d;\n"
        "mul.rn.f32 gs, $1, s;\n"
        "add.rn.f32 ub, $2, 0f00000000;\n"
        "mul.rn.f32 $0, gs, ub;\n"
        "}"
    )


@dsl_user_op
def silu_mul_triton(
    g: Float32, u: Float32, variant: int = SILU_DEFAULT, *, loc=None, ip=None
) -> Float32:
    """``g * sigmoid(g) * (u + 0.0)`` in FP32 with one of ``SILU_VARIANTS``."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(g).ir_value(loc=loc, ip=ip), Float32(u).ir_value(loc=loc, ip=ip)],
            _silu_text(variant),
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )
