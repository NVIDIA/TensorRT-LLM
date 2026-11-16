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
"""TensorSSA packed FP32 FMA adapter for the indexer's four-element epilogue.

The installed CuTe arch wrapper accepts a Python two-tuple. Iterating a
TensorSSA in that wrapper does not terminate. Explicit pairs preserve the source
epilogue's elementwise round-to-nearest FMA semantics.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op

_native_fma = cute.arch.fma_packed_f32x2


@dsl_user_op
def fma_packed_f32x2(a,b,c,*,rnd='rn',ftz=None,loc=None,ip=None):
    if isinstance(a,cute.TensorSSA):
        if rnd!='rn' or ftz not in (None,False):
            raise ValueError('indexer vector adapter requires RN without FTZ')
        if any(not isinstance(x,cute.TensorSSA) or x.dtype!=cutlass.Float32 or x.type!=a.type for x in (b,c)):
            raise TypeError('matching Float32 TensorSSA operands required')
        result=cute.make_rmem_tensor(a.shape,cutlass.Float32)
        for offset in range(0,cute.size(a.shape),2):
            x,y=_native_fma((a[offset],a[offset+1]),(b[offset],b[offset+1]),
                            (c[offset],c[offset+1]),rnd=rnd,ftz=ftz,loc=loc,ip=ip)
            result[offset]=x
            result[offset+1]=y
        return result.load()
    return _native_fma(a,b,c,rnd=rnd,ftz=ftz,loc=loc,ip=ip)


_native_add = cute.arch.add_packed_f32x2


@dsl_user_op
def add_packed_f32x2(a,b,*,rnd='rn',ftz=None,loc=None,ip=None):
    if isinstance(a,cute.TensorSSA):
        result=cute.make_rmem_tensor(a.shape,cutlass.Float32)
        for offset in range(0,cute.size(a.shape),2):
            x,y=_native_add((a[offset],a[offset+1]),(b[offset],b[offset+1]),
                            rnd=rnd,ftz=ftz,loc=loc,ip=ip)
            result[offset]=x
            result[offset+1]=y
        return result.load()
    return _native_add(a,b,rnd=rnd,ftz=ftz,loc=loc,ip=ip)
