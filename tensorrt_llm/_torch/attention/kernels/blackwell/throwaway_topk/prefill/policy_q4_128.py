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

"""Input-independent stratified single-key sampling, shared within each Q4 tile."""
import cutlass,cutlass.cute as cute
@cute.jit
def runtime_samples(length,k:cutlass.Constexpr=2048):
 target=max(cutlass.Int32(256),min(cutlass.Int32(2048),(length*4+k-1)//k))
 if cutlass.const_expr(k==2048):
  if length<=8*k:target=cutlass.Int32(128)
 return cutlass.Int32(cutlass.Uint32(1)<<(32-cutlass.Int32(cute.arch.clz(cutlass.Uint32(target-1)))))*2
@cute.jit
def block_samples(qb,ends,nq,bq:cutlass.Constexpr,k:cutlass.Constexpr=2048):
 return runtime_samples(ends[min(nq-1,qb*bq+bq-1)],k)

@cute.jit
def individual_sample_key(pos,qb,starts,ends,nq,bq:cutlass.Constexpr,k:cutlass.Constexpr=2048):
 first=qb*bq;lower=starts[first]
 for token in cutlass.range_constexpr(1,bq):lower=min(lower,starts[min(nq-1,first+token)])
 upper=ends[min(nq-1,first+bq-1)];width=max(1,upper-lower)
 samples=runtime_samples(upper,k);shift=31-cutlass.Int32(cute.arch.clz(cutlass.Uint32(samples)))
 # One reproducible pseudorandom offset per stratum and query tile avoids
 # systematic midpoint aliasing. The seed is fixed, never fitted to inputs.
 sample_qb=qb
 if cutlass.const_expr(k==2048):
  if upper<=8*k:sample_qb=cutlass.Int32(0)
 h=cutlass.Uint32(pos)+cutlass.Uint32(sample_qb)*cutlass.Uint32(0x9e3779b9)+cutlass.Uint32(0x243f6a88)
 h=(h^(h>>16))*cutlass.Uint32(0x85ebca6b)
 h=(h^(h>>13))*cutlass.Uint32(0xc2b2ae35)
 h=h^(h>>16)
 begin=cutlass.Int32((cutlass.Int64(pos)*cutlass.Int64(width))>>shift)
 finish=cutlass.Int32((cutlass.Int64(pos+1)*cutlass.Int64(width))>>shift)
 span=max(1,finish-begin)
 offset=begin+cutlass.Int32((cutlass.Uint64(h)*cutlass.Uint64(span))>>32)
 return max(cutlass.Int32(0),min(upper-1,lower+offset))


@cute.jit
def qwen_cached_sample_key(pos,qb,lower,upper,k:cutlass.Constexpr=512):
 """Identical stratified sample positions with the Q-tile bounds hoisted."""
 width=max(1,upper-lower)
 samples=runtime_samples(upper,k);shift=31-cutlass.Int32(cute.arch.clz(cutlass.Uint32(samples)))
 # One reproducible pseudorandom offset per stratum and query tile avoids
 # systematic midpoint aliasing. The seed is fixed, never fitted to inputs.
 sample_qb=qb
 if cutlass.const_expr(k==2048):
  if upper<=8*k:sample_qb=cutlass.Int32(0)
 h=cutlass.Uint32(pos)+cutlass.Uint32(sample_qb)*cutlass.Uint32(0x9e3779b9)+cutlass.Uint32(0x243f6a88)
 h=(h^(h>>16))*cutlass.Uint32(0x85ebca6b)
 h=(h^(h>>13))*cutlass.Uint32(0xc2b2ae35)
 h=h^(h>>16)
 begin=cutlass.Int32((cutlass.Int64(pos)*cutlass.Int64(width))>>shift)
 finish=cutlass.Int32((cutlass.Int64(pos+1)*cutlass.Int64(width))>>shift)
 span=max(1,finish-begin)
 offset=begin+cutlass.Int32((cutlass.Uint64(h)*cutlass.Uint64(span))>>32)
 return max(cutlass.Int32(0),min(upper-1,lower+offset))
