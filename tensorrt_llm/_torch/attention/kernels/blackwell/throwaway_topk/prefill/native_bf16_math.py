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
"""Packed BF16 operations for explicit native_bf16 head accumulation.

FP32 dot products are rounded at ReLU, each multiply-add, and the balanced
four-chain reduction. This is a separate numerical contract from FP32 math.
"""
import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def native_bfpack(a,b,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Float32(a).ir_value(loc=loc,ip=ip),cutlass.Float32(b).ir_value(loc=loc,ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_bfpack_relu(a,b,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Float32(a).ir_value(loc=loc,ip=ip),cutlass.Float32(b).ir_value(loc=loc,ip=ip)],
        "cvt.rn.relu.bf16x2.f32 $0, $2, $1;", "=r,f,f",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_bffma(a,b,c,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Uint32(v).ir_value(loc=loc,ip=ip) for v in [a,b,c]],
        "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_bfadd(a,b,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Uint32.mlir_type,[cutlass.Uint32(v).ir_value(loc=loc,ip=ip) for v in [a,b]],
        "add.rn.bf16x2 $0, $1, $2;", "=r,r,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def native_bflo(a,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Uint32(a).ir_value(loc=loc,ip=ip)],
        "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }", "=f,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)

@dsl_user_op
def native_bfhi(a,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Uint32(a).ir_value(loc=loc,ip=ip)],
        "{ .reg .b32 t; and.b32 t, $1, 0xffff0000; mov.b32 $0, t; }", "=f,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)


@dsl_user_op
def native_bfsum_fp32(packed,*,loc=None,ip=None):
    # The input holds the two already rounded pairs of the four-chain sum.
    # Add BF16 halves and expand the result exactly to FP32.
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Uint32(packed).ir_value(loc=loc,ip=ip)],
        "{ .reg .b32 swapped, result; prmt.b32 swapped,$1,$1,0x1032; add.rn.bf16x2 result,$1,swapped; shl.b32 $0,result,16; }",
        "=f,r",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)
