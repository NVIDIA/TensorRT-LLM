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

"""Prove a non-clipped histogram bin contains at most one finite 16-bit value."""
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from .half_filter_boundary import half_value

@cute.jit
def mapped_bucket(score,lower,inverse_span,bins:cutlass.Constexpr):
    # Must match numeric_bucket for finite scores and a positive inverse span.
    normalized=(score-lower)*inverse_span
    normalized=min(cutlass.Float32(bins-1),max(cutlass.Float32(0),normalized))
    return (bins-1)-cutlass.Int32(normalized)

@cute.jit
def uniform_half_bucket(crossing,lower,inverse_span,bins:cutlass.Constexpr,dtype:cutlass.Constexpr=cutlass.Float16):
    uniform=False
    if (crossing>0)&(crossing<bins-1)&(inverse_span>0.0)&(inverse_span<cutlass.Float32(float('inf'))):
        center=lower+(cutlass.Float32(bins-1-crossing)+0.5)/inverse_span
        rounded=dtype(center)
        bits=cutlass.Uint32(cutlass.Uint16(llvm.bitcast(cutlass.Uint16.mlir_type,rounded.ir_value())))
        if ((bits&0x7fff)>0)&((bits&0x7fff)<(0x7f80 if cutlass.const_expr(dtype==cutlass.BFloat16) else 0x7c00)):
            prev=bits-1;next_=bits+1
            if (bits&0x8000)!=0:prev=bits+1;next_=bits-1
            value=cutlass.Float32(rounded)
            uniform=(mapped_bucket(value,lower,inverse_span,bins)==crossing)&(mapped_bucket(half_value(prev,dtype),lower,inverse_span,bins)>crossing)&(mapped_bucket(half_value(next_,dtype),lower,inverse_span,bins)<crossing)
    return uniform
