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
from functools import partial
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute import typing as cutlass_typing
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def warp_reduction_fmax(
    val: cutlass.Float32,
    mask: cutlass.Int32 = 0xFFFFFFFF,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass_typing.Float32.mlir_type,
            [
                cutlass_typing.Float32(val).ir_value(loc=loc, ip=ip),
                cutlass_typing.Int32(mask).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            redux.sync.max.f32 $0, $1, $2;\n\t
            \n\t}""",
            "=f,f,r",
        )
    )


@dsl_user_op
def __cvta_generic_to_shared(
    ptr: cutlass.Pointer,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    # NOTE: assume the SMEM pointer fits in a 32-bit register
    return cutlass.Uint32(
        llvm.inline_asm(
            cutlass_typing.Uint32.mlir_type,
            [
                cutlass_typing.Int32(ptr.toint()).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            mov.u32 $0, $1;
            \n\t}""",
            "=r, r",
        )
    )


@dsl_user_op
def atomicAdd_f32(
    val: cutlass.Float32,
    ptr: cutlass.Pointer,
    *,
    loc=None,
    ip=None,
):
    if cutlass.const_expr(ptr.memspace == cutlass_typing.AddressSpace.smem):
        ptr = __cvta_generic_to_shared(ptr, loc=loc, ip=ip)
        llvm.inline_asm(
            None,
            [
                cutlass_typing.Uint32(ptr).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(val).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            atom.relaxed.shared::cta.cta.add.f32 _, [$0], $1;\n\t
            \n\t}""",
            "r, f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    else:
        llvm.inline_asm(
            None,
            [
                cutlass_typing.Int64(ptr.toint()).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(val).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            atom.relaxed.shared::cta.cta.add.f32 _, [$0], $1;\n\t
            \n\t}""",
            "l, f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def atomicMax_f32(
    val: cutlass.Float32,
    ptr: cutlass.Pointer,
    *,
    loc=None,
    ip=None,
):
    val_i32 = llvm.bitcast(
        cutlass_typing.Int32.mlir_type, val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    if cutlass.const_expr(ptr.memspace == cutlass_typing.AddressSpace.smem):
        ptr = __cvta_generic_to_shared(ptr, loc=loc, ip=ip)
        llvm.inline_asm(
            None,
            [
                cutlass_typing.Uint32(ptr).ir_value(loc=loc, ip=ip),
                cutlass_typing.Int32(val_i32).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            atom.relaxed.shared::cta.cta.max.s32 _, [$0], $1;\n\t
            \n\t}""",
            "r, r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    else:
        llvm.inline_asm(
            None,
            [
                cutlass_typing.Int64(ptr.toint()).ir_value(loc=loc, ip=ip),
                cutlass_typing.Int32(val_i32).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            atom.relaxed.shared::cta.cta.max.s32 _, [$0], $1;\n\t
            \n\t}""",
            "l, r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def exp2f(
    val: cutlass.Float32,
    *,
    loc=None,
    ip=None,
):
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass_typing.Float32.mlir_type,
            [
                cutlass_typing.Float32(val).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            .reg .f32 f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11;\n\t
            .reg .s32 r1, r2, r3;\n\t
            max.ftz.f32 f1, $1, 0fC2FE0000;\n\t
            mov.f32 f3, 0f4B400000;\n\t
            add.rm.ftz.f32 f4, f1, f3;\n\t
            sub.rn.ftz.f32 f5, f4, f3;\n\t
            sub.rn.ftz.f32 f6, f1, f5;\n\t
            mov.f32 f7, 0f3D9DF09D;\n\t
            mov.f32 f8, 0f3E6906A4;\n\t
            mov.f32 f9, 0f3F31F519;\n\t
            mov.f32 f10, 0f3F800000;\n\t
            fma.rn.ftz.f32 f11, f6, f7, f8;\n\t
            fma.rn.ftz.f32 f11, f11, f6, f9;\n\t
            fma.rn.ftz.f32 f11, f11, f6, f10;\n\t
            mov.b32 r3, f11;\n\t
            shl.b32 r1, f4, 23;\n\t
            add.s32 r2, r1, r3;\n\t
            mov.b32 $0, r2;\n\t
            \n\t}""",
            "=f, f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma(
    a: cutlass.Float32,
    b: cutlass.Float32,
    c: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass_typing.Float32.mlir_type,
            [
                cutlass_typing.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            fma.rn.ftz.f32 $0, $1, $2, $3;\n\t
            \n\t}""",
            "=f, f, f, f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max3f(
    a: cutlass.Float32,
    b: cutlass.Float32,
    c: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass_typing.Float32.mlir_type,
            [
                cutlass_typing.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            max.f32 $0, $1, $2, $3;\n\t
            \n\t}""",
            "=f, f, f, f",
            loc=loc,
            ip=ip,
        )
    )


sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
    rnd=nvvm.RoundingModeKind.RN,
)


@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: cutlass.Float32,
    y: cutlass.Float32,
    poly: Tuple[cutlass.Float32, ...],
    *,
    loc=None,
    ip=None,
) -> Tuple[cutlass.Float32, cutlass.Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def combine_int_frac_ex2(
    x_rounded: cutlass.Float32,
    frac_ex2: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass_typing.Float32.mlir_type,
            [
                cutlass_typing.Float32(x_rounded).ir_value(loc=loc, ip=ip),
                cutlass_typing.Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            """{\n\t
            .reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i; \n\t
            mov.b32 x_rounded_i, $1; \n\t
            mov.b32 frac_ex_i, $2; \n\t
            shl.b32 x_rounded_e, x_rounded_i, 23; \n\t
            add.s32 out_i, x_rounded_e, frac_ex_i; \n\t
            mov.b32 $0, out_i; \n\t
            \n\t}""",
            "=f, f, f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def exp2_emulation_2(
    x: cutlass.Float32,
    y: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[cutlass.Float32, cutlass.Float32]:
    # assume x <= 127.0 and y <= 127.0
    poly_ex2_deg3 = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd=nvvm.RoundingModeKind.RM
    )
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, poly_ex2_deg3, loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def exp2f_packed_f32x2(
    x: cutlass.Float32,
    y: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[cutlass.Float32, cutlass.Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal(
            [
                cutlass_typing.Float32.mlir_type,
                cutlass_typing.Float32.mlir_type,
            ]
        ),
        [
            cutlass_typing.Float32(x).ir_value(loc=loc, ip=ip),
            cutlass_typing.Float32(y).ir_value(loc=loc, ip=ip),
        ],
        """{\n\t
        ex2.approx.f32 $0, $2;\n\t
        ex2.approx.f32 $1, $3;\n\t
        \n\t}""",
        # Keep constraints compact (no spaces) and in the correct order
        "=f,=f,f,f",
        loc=loc,
        ip=ip,
    )

    # Extract struct fields
    out0_val = llvm.extractvalue(cutlass_typing.Float32.mlir_type, result, [0], loc=loc, ip=ip)
    out1_val = llvm.extractvalue(cutlass_typing.Float32.mlir_type, result, [1], loc=loc, ip=ip)

    # Wrap back into cutlass.Float32
    return cutlass.Float32(out0_val), cutlass.Float32(out1_val)
