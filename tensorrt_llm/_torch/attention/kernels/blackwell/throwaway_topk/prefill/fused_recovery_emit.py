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

import cutlass,cutlass.cute as cute
from .selection import encode
from .selection import emit as dense_emit
from .recovery_selection import finish_recovery as sampled_fallback
from .recovery_streaming import select_stream
@cute.jit
def emit(score,row,col,sample_col,valid,nq,dense,thresholds,counts,candidates,mode:cutlass.Constexpr,dtype:cutlass.Constexpr,packed:cutlass.Constexpr,cap:cutlass.Constexpr,partition,info,k:cutlass.Constexpr,recovery_threshold):
    dense_emit(score,row,col,sample_col,valid,nq,dense,thresholds,counts,candidates,mode,dtype,packed,cap)
    if row<nq:
        if counts[row]<k:
            value=cutlass.Float32(dtype(score))
            lower=recovery_threshold
            hit=valid&((value>=lower)|(value!=value));mask=cute.arch.vote_ballot_sync(hit)
            lane=cute.arch.lane_idx();amount=cutlass.Int32(cute.arch.popc(mask));base=cutlass.Int32(0)
            if amount>0:
                if lane==0:base=cutlass.Int32(cute.arch.atomic_add((candidates.iterator+cutlass.Int64(row)*candidates.stride[0]+partition).llvm_ptr,cutlass.Int64(amount),sem='relaxed',scope='gpu'))
                base=cute.arch.shuffle_sync(base,0)
                pos=base+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<lane)-1)))
                span=(candidates.shape[1]-8)//8
                if hit&(pos<span):candidates[row,8+partition*span+pos]=cutlass.Int64(encode(value,col,dtype,False))
@cute.jit
def finish_recovery(row,tid,candidates,counts,starts,ends,dense,hist,scalars,out,thresholds,k:cutlass.Constexpr,cap:cutlass.Constexpr,dtype:cutlass.Constexpr,packed:cutlass.Constexpr,sample_storage,threads:cutlass.Constexpr=256):
    lane=cute.arch.lane_idx();total=cutlass.Int32(0)
    if lane<8:total=cutlass.Int32(candidates[row,lane])
    maximum=total
    for step in cutlass.range_constexpr(5):
        total+=cute.arch.shuffle_sync_bfly(total,1<<step)
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    span=(candidates.shape[1]-8)//8;required=min(k,ends[row]-starts[row])
    if (total>=required)&(maximum<=span):
        select_stream(row,tid,candidates,total,required,hist,out,k,dtype)
    else:
        sampled_fallback(row,tid,candidates,counts,starts,ends,dense,hist,scalars,out,thresholds,k,cap,dtype,packed,sample_storage,threads)

@cute.jit
def prepare_threshold(row,info,counts,k:cutlass.Constexpr):
    original=cutlass.Float32(info[row,info.shape[1]-32+1])
    high=cutlass.Float32(info[row,info.shape[1]-32])
    gap=max(cutlass.Float32(0),high-original)
    tau=original-cutlass.Float32(0.25)*gap
    count=counts[row]
    if count<0:
        actual=count&cutlass.Int32(0xFFFFFF)
        tau=original
        if actual>8*k:tau=original+cutlass.Float32(0.25)*gap
    if not ((tau<float('inf'))&(tau>float('-inf'))):tau=cutlass.Float32(float('-inf'))
    return tau
