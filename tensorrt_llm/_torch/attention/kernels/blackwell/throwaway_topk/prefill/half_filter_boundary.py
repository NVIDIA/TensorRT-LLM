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

"""Exact FP32 preimage of a round-to-nearest-even 16-bit threshold."""
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from .selection import bits32, float32

@cute.jit
def half_value(bits,dtype:cutlass.Constexpr=cutlass.Float16):
    h=dtype(llvm.bitcast(dtype.mlir_type,cutlass.Uint16(bits).ir_value()))
    return cutlass.Float32(h)

@cute.jit
def filter_boundary(tau,dtype:cutlass.Constexpr):
    result=tau
    if cutlass.const_expr(dtype==cutlass.Float16):
        rounded=cutlass.Float16(tau)
        b=cutlass.Uint32(cutlass.Uint16(llvm.bitcast(cutlass.Uint16.mlir_type,rounded.ir_value())))
        # Find the least representable half >= tau, including overflow.
        if cutlass.Float32(rounded)<tau:
            if (b&0x8000)!=0 and (b&0x7fff)!=0:b=b-1
            else:b=cutlass.Uint32(1) if (b&0x7fff)==0 else b+1
        previous=b-1
        if (b&0x8000)!=0:previous=b+1
        if (b&0x7fff)==0:previous=cutlass.Uint32(0x8001)
        midpoint=(half_value(b)+half_value(previous))*0.5
        if b==0x7c00:midpoint=cutlass.Float32(65520.0)
        if b==0xfbff:midpoint=cutlass.Float32(-65520.0)
        if b==0xfc00:midpoint=cutlass.Float32(float('-inf'))
        # An odd upper significand loses the exact halfway tie. Advance by
        # one FP32 ULP so the final >= comparison expresses strict >.
        if (b&1)!=0:
            m=bits32(midpoint)
            if (m&0x80000000)!=0:m=m-1
            else:m=m+1
            midpoint=float32(m)
        result=midpoint
        if tau!=tau:result=tau
    elif cutlass.const_expr(dtype==cutlass.BFloat16):
        # BF16 values occupy the upper 16 bits of FP32. Round the threshold
        # upward to BF16, then locate its lower RNE boundary in FP32 bits.
        raw=bits32(tau)
        upper=raw>>16
        if (raw&0x80000000)==0:upper=(raw+cutlass.Uint32(0xffff))>>16
        boundary=(upper<<16)-cutlass.Uint32(0x8000)
        if (upper&0x8000)!=0:boundary=(upper<<16)+cutlass.Uint32(0x8000)
        if (upper&0x7fff)==0:boundary=cutlass.Uint32(0x80008000)
        if upper==0xff80:boundary=cutlass.Uint32(0xff800000)
        # Odd BF16 significands lose halfway ties to the preceding value.
        if (upper&1)!=0:
            if (boundary&0x80000000)!=0:boundary=boundary-1
            else:boundary=boundary+1
        result=float32(boundary)
        if tau!=tau:result=tau
    return result
