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
import cutlass
from cutlass._mlir.dialects import llvm
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
