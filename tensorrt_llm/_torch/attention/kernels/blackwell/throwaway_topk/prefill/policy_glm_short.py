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

"""GLM MXFP4 short-domain sampling; other models keep their policy."""
import cutlass
import cutlass.cute as cute
from .policy_qwen import runtime_samples as original_samples, sample_hash_u32, qwen_cached_sample_key
from .sample_policy import proposal_rank as original_rank

@cute.jit
def runtime_samples(length,k:cutlass.Constexpr=2048,heads:cutlass.Constexpr=64,coverage:cutlass.Constexpr=8):
    samples=original_samples(length,k,heads,coverage)
    if cutlass.const_expr(heads==32):
        if length<=64*k:samples=cutlass.Int32(256 if k==2048 else 128)
        if cutlass.const_expr(k==2048):
            if length>64*k:samples=cutlass.Int32(1024)
    return samples

@cute.jit
def block_samples(qb,ends,nq,bq:cutlass.Constexpr,k:cutlass.Constexpr=2048,heads:cutlass.Constexpr=64,coverage:cutlass.Constexpr=8):
    return runtime_samples(ends[min(nq-1,qb*bq+bq-1)],k,heads,coverage)

@cute.jit
def model_sample_key(pos,qb,lower,upper,k:cutlass.Constexpr=512,heads:cutlass.Constexpr=64,coverage:cutlass.Constexpr=8,common:cutlass.Constexpr=False):
    if cutlass.const_expr(heads==32):
        width=max(1,upper-lower)
        samples=runtime_samples(upper,k,heads,coverage)
        shift=31-cutlass.Int32(cute.arch.clz(cutlass.Uint32(samples)))
        h=sample_hash_u32(pos,cutlass.Int32(0) if cutlass.const_expr(common and k==2048) else qb)
        begin=cutlass.Int32((cutlass.Int64(pos)*cutlass.Int64(width))>>shift)
        finish=cutlass.Int32((cutlass.Int64(pos+1)*cutlass.Int64(width))>>shift)
        offset=begin+cutlass.Int32((cutlass.Uint64(h)*cutlass.Uint64(max(1,finish-begin)))>>32)
        return max(cutlass.Int32(0),min(upper-1,lower+offset))
    else:
        return qwen_cached_sample_key(pos,qb,lower,upper,k,coverage)

@cute.jit
def model_proposal_rank(width,samples,k:cutlass.Constexpr,sigma:cutlass.Constexpr,heads:cutlass.Constexpr):
    rank=original_rank(width,samples,k,sigma)
    if cutlass.const_expr(heads==32):
        if samples==128:rank=original_rank(width,samples,k,5)
        if cutlass.const_expr(k==2048):
            if samples<=256:rank=original_rank(width,samples,k,4.5)
            else:rank=original_rank(width,samples,k,sigma)
    return rank
