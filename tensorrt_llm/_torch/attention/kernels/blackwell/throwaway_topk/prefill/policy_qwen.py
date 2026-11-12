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

"""Qwen short-domain sampling; all other models keep the established policy."""
import cutlass
import cutlass.cute as cute
from .policy_q4_128 import runtime_samples as original_runtime_samples

from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def sample_hash_u32(pos,qb,*,loc=None,ip=None):
    # All shifts and wrapping arithmetic must be unsigned in both producer
    # and math specializations; an inferred signed first shift changes masks.
    return cutlass.Uint32(llvm.inline_asm(cutlass.Uint32.mlir_type,
        [cutlass.Uint32(pos).ir_value(loc=loc,ip=ip),cutlass.Uint32(qb).ir_value(loc=loc,ip=ip)],
        "{ .reg .b32 h,t; mad.lo.u32 h,$2,0x9e3779b9,$1; add.u32 h,h,0x243f6a88; "
        "shr.u32 t,h,16; xor.b32 h,h,t; mul.lo.u32 h,h,0x85ebca6b; "
        "shr.u32 t,h,13; xor.b32 h,h,t; mul.lo.u32 h,h,0xc2b2ae35; "
        "shr.u32 t,h,16; xor.b32 $0,h,t; }", "=r,r,r",has_side_effects=False,loc=loc,ip=ip))

@cute.jit
def runtime_samples(length,k:cutlass.Constexpr=2048,heads:cutlass.Constexpr=64,coverage:cutlass.Constexpr=8):
    if cutlass.const_expr(heads in (4,32)):
        samples=min(cutlass.Int32(1024),original_runtime_samples(length,k))
        # Keep an expected >=8 top-k hits until the existing 1024 cap.
        # This is based only on the visible domain and top-k, never input IDs.
        if length<=(128 if coverage==4 else 64)*k:
            samples=cutlass.Int32(512)
            if length<=(64 if coverage==4 else 32)*k:samples=cutlass.Int32(256)
        return samples
    else:
        return original_runtime_samples(length,k)

@cute.jit
def block_samples(qb,ends,nq,bq:cutlass.Constexpr,k:cutlass.Constexpr=2048,heads:cutlass.Constexpr=64,coverage:cutlass.Constexpr=8):
    return runtime_samples(ends[min(nq-1,qb*bq+bq-1)],k,heads,coverage)

@cute.jit
def qwen_cached_sample_key(pos,qb,lower,upper,k:cutlass.Constexpr=512,coverage:cutlass.Constexpr=8):
 """Identical stratified sample positions with the Q-tile bounds hoisted."""
 width=max(1,upper-lower)
 samples=runtime_samples(upper,k,heads=4,coverage=coverage);shift=31-cutlass.Int32(cute.arch.clz(cutlass.Uint32(samples)))
 # One reproducible pseudorandom offset per stratum and query tile avoids
 # systematic midpoint aliasing. The seed is fixed, never fitted to inputs.
 h=sample_hash_u32(pos,qb)
 begin=cutlass.Int32((cutlass.Int64(pos)*cutlass.Int64(width))>>shift)
 finish=cutlass.Int32((cutlass.Int64(pos+1)*cutlass.Int64(width))>>shift)
 span=max(1,finish-begin)
 offset=begin+cutlass.Int32((cutlass.Uint64(h)*cutlass.Uint64(span))>>32)
 return max(cutlass.Int32(0),min(upper-1,lower+offset))
