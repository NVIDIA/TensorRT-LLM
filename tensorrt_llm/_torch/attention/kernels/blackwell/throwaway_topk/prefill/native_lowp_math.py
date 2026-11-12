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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Packed FP16/BF16 operations for explicit native low-precision head math.

FP32 dot products are rounded at ReLU, each multiply-add, and the balanced
four-chain reduction. FP8 FP16 helpers also apply the fixed QK/weight
rebalance. These are separate numerical contracts from FP32 head math.
"""
import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def native_pack(dtype,a,b,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Float32(a).ir_value(loc=loc,ip=ip),cutlass.Float32(b).ir_value(loc=loc,ip=ip)],
        ("cvt.rn.f16x2.f32 $0, $2, $1;" if dtype == cutlass.Float16 else "cvt.rn.bf16x2.f32 $0, $2, $1;"), "=r,f,f",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_pack_relu(dtype,a,b,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Float32(a).ir_value(loc=loc,ip=ip),cutlass.Float32(b).ir_value(loc=loc,ip=ip)],
        ("cvt.rn.relu.f16x2.f32 $0, $2, $1;" if dtype == cutlass.Float16 else "cvt.rn.relu.bf16x2.f32 $0, $2, $1;"), "=r,f,f",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_fma(dtype,a,b,c,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Uint32(v).ir_value(loc=loc,ip=ip) for v in [a,b,c]],
        ("fma.rn.f16x2 $0, $1, $2, $3;" if dtype == cutlass.Float16 else "fma.rn.bf16x2 $0, $1, $2, $3;"), "=r,r,r,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_add(dtype,a,b,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Uint32(v).ir_value(loc=loc,ip=ip) for v in [a,b]],
        ("add.rn.f16x2 $0, $1, $2;" if dtype == cutlass.Float16 else "add.rn.bf16x2 $0, $1, $2;"), "=r,r,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_lo(dtype,a,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Uint32(a).ir_value(loc=loc,ip=ip)],
        ("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.f16 $0, lo; }" if dtype == cutlass.Float16 else "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }"), "=f,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)

@dsl_user_op
def native_hi(dtype,a,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Uint32(a).ir_value(loc=loc,ip=ip)],
        ("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.f16 $0, hi; }" if dtype == cutlass.Float16 else "{ .reg .b32 t; and.b32 t, $1, 0xffff0000; mov.b32 $0, t; }"), "=f,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)


@dsl_user_op
def native_sum_fp32(dtype,packed,*,loc=None,ip=None):
    # The input holds the two already rounded pairs of the four-chain sum.
    # Add native halves and expand the result exactly to FP32.
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Uint32(packed).ir_value(loc=loc,ip=ip)],
        ("{ .reg .b32 swapped, result; .reg .b16 lo, hi; prmt.b32 swapped,$1,$1,0x1032; add.rn.f16x2 result,$1,swapped; mov.b32 {lo, hi},result; cvt.f32.f16 $0,lo; }" if dtype == cutlass.Float16 else "{ .reg .b32 swapped, result; prmt.b32 swapped,$1,$1,0x1032; add.rn.bf16x2 result,$1,swapped; shl.b32 $0,result,16; }"),
        "=f,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)


@dsl_user_op
def native_fp8_pack(dtype,a,b,*,loc=None,ip=None):
    # FP8 E4M3FN, D=128: max |QK| is 128*448**2. The smallest
    # power-of-two divisor fitting that bound in FP16 is 512. Rebalance
    # weights inversely; this is a dtype/dimension rule, never data-dependent.
    if dtype == cutlass.Float16:
        r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Float32(a).ir_value(loc=loc,ip=ip),cutlass.Float32(b).ir_value(loc=loc,ip=ip)],
            "{ .reg .b64 pair, scale, result; .reg .f32 lo, hi; mov.b64 pair, {$1,$2}; mov.b64 scale,0x4400000044000000; mul.rn.f32x2 result,pair,scale; mov.b64 {lo,hi},result; cvt.rn.f16x2.f32 $0,hi,lo; }",
            "=r,f,f",has_side_effects=False,loc=loc,ip=ip)
        return cutlass.Uint32(r)
    return native_pack(dtype,a,b,loc=loc,ip=ip)

@dsl_user_op
def native_fp8_pack_relu(dtype,a,b,*,loc=None,ip=None):
    if dtype == cutlass.Float16:
        r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Float32(a).ir_value(loc=loc,ip=ip),cutlass.Float32(b).ir_value(loc=loc,ip=ip)],
            "{ .reg .b64 pair, scale, result; .reg .f32 lo, hi; mov.b64 pair, {$1,$2}; mov.b64 scale,0x3b0000003b000000; mul.rn.f32x2 result,pair,scale; mov.b64 {lo,hi},result; cvt.rn.relu.f16x2.f32 $0,hi,lo; }",
            "=r,f,f",has_side_effects=False,loc=loc,ip=ip)
        return cutlass.Uint32(r)
    return native_pack_relu(dtype,a,b,loc=loc,ip=ip)
