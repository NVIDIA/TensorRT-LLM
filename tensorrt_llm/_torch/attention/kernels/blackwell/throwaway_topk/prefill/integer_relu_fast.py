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

"""Integer ReLU for bounded dot products and either-sign NaNs.
The caller must preserve its existing safe-path contract: E4M3FN D128, or
MXFP4 D128 with every E8M0 scale byte <=159. The latter bounds the raw dot
by 128*36*2**64, so the doubled-value ReLU trick cannot overflow either.
"""
from ..packed import add_packed_f32x2 as indexer_add_packed_f32x2
import cutlass,cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
@dsl_user_op
def integer_relu(x,*,loc=None,ip=None):
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Float32(x).ir_value(loc=loc,ip=ip)],
        "{ .reg .b32 bits; .reg .pred keep; mov.b32 bits,$1; setp.gt.s32 keep,bits,-8388608; selp.b32 $0,bits,0,keep; }",
        "=f,f",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)
@cute.jit
def integer_relu4(x):
    result=cute.make_rmem_tensor((4,),cutlass.Float32)
    for j in cutlass.range_constexpr(4):result[j]=integer_relu(x[j])
    return result.load()

@dsl_user_op
def finite_integer_relu(x,*,loc=None,ip=None):
    # Only for the safe MXFP4 path: E2M1 has no nonfinite values,
    # E8M0 bytes <=159 bound D128 dot products below 2**77. Weights are
    # applied AFTER this operation. FP8 can encode NaNs and must not use it.
    r=llvm.inline_asm(cutlass.Float32.mlir_type,[cutlass.Float32(x).ir_value(loc=loc,ip=ip)],
        "{ .reg .b32 bits,out; mov.b32 bits,$1; max.s32 out,bits,0; mov.b32 $0,out; }",
        "=f,f",has_side_effects=False,loc=loc,ip=ip)
    return cutlass.Float32(r)
@cute.jit
def finite_integer_relu4(x):
    result=cute.make_rmem_tensor((4,),cutlass.Float32)
    for j in cutlass.range_constexpr(4):result[j]=finite_integer_relu(x[j])
    return result.load()

@cute.jit
def mixed_integer_relu4(x,finite:cutlass.Constexpr=False):
    pair=cute.make_rmem_tensor((2,),cutlass.Float32)
    pair[0]=x[2];pair[1]=x[3]
    tail=pair.load()
    tail=indexer_add_packed_f32x2(tail,abs(tail),rnd="rn")
    result=cute.make_rmem_tensor((4,),cutlass.Float32)
    result[0]=finite_integer_relu(x[0]) if cutlass.const_expr(finite) else integer_relu(x[0])
    result[1]=finite_integer_relu(x[1]) if cutlass.const_expr(finite) else integer_relu(x[1])
    result[2]=tail[0];result[3]=tail[1]
    return result.load()
