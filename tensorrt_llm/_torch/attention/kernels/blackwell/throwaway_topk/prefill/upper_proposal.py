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
from .subset_barrier import subset_barrier
from .selection import score_key, key_score

@cute.jit
def propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,items:cutlass.Constexpr):
    """One adaptive 256-bin proposal, rather than an exact sample sort (K1)."""
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    cached=cute.make_rmem_tensor((items,),cutlass.Float32)
    for j in cutlass.range_constexpr(items):
        i=tid+j*256
        cached[j]=cutlass.Float32(0)
        if i<samples:
            value=cutlass.Float32(scores[row,i])
            cached[j]=value
            key=score_key(value)
            minimum=min(minimum,key);maximum=max(maximum,key)
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    if cute.arch.lane_idx()==0:
        hist[2*(tid//32)]=cutlass.Int32(minimum)
        hist[2*(tid//32)+1]=cutlass.Int32(maximum)
    subset_barrier(barrier_id=13,number_of_threads=256)
    lane=cute.arch.lane_idx()
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    if lane < 8:
        minimum=cutlass.Uint32(hist[2*lane])
        maximum=cutlass.Uint32(hist[2*lane+1])
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    span=maximum-minimum
    low=key_score(maximum);high=key_score(minimum)
    scale=cutlass.Float32(0)
    if (high>low)&(high<float('inf'))&(low>float('-inf')):
        scale=cutlass.Float32(255.0)/(high-low)
    shift=cutlass.Int32(0)
    if span>0:
        shift=max(0,24-cutlass.Int32(cute.arch.clz(span)))
    subset_barrier(barrier_id=13,number_of_threads=256)
    hist[tid]=0
    subset_barrier(barrier_id=13,number_of_threads=256)
    for j in cutlass.range_constexpr(items):
        i=j*256+tid
        bucket=cutlass.Int32(256)
        if i<samples:
            value=cached[j]
            key=score_key(value)
            bucket=min(255,cutlass.Int32((key-minimum)>>shift))
            if scale>0:
                bucket=cutlass.Int32(min(cutlass.Float32(255),max(cutlass.Float32(0),(high-value)*scale)))
        peers=cute.arch.match_sync(-1,bucket)
        lower=(cutlass.Uint32(1)<<cute.arch.lane_idx())-1
        if (bucket<256)&((peers&lower)==0):
            cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(cute.arch.popc(peers)),sem='relaxed',scope='cta')
    subset_barrier(barrier_id=13,number_of_threads=256)
    n=hist[tid];prefix=n;lane=cute.arch.lane_idx()
    for step in cutlass.range_constexpr(5):
        delta=1<<step;other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
        if lane>=delta:prefix+=other
    hist[tid]=prefix
    subset_barrier(barrier_id=13,number_of_threads=256)
    warp_total=cutlass.Int32(0)
    if lane < 8:warp_total=hist[lane*32+31]
    for step in cutlass.range_constexpr(5):
        delta=1<<step;other=cute.arch.shuffle_sync_up(warp_total,delta,mask=-1,mask_and_clamp=0)
        if lane>=delta:warp_total+=other
    prior=cute.arch.shuffle_sync(warp_total,max(0,tid//32-1))
    if tid//32==0:prior=cutlass.Int32(0)
    cumulative=prior+prefix
    if (cumulative>=rank)&(cumulative-n<rank):scalars[0]=tid
    subset_barrier(barrier_id=13,number_of_threads=256)
    if tid==0:
        edge=cutlass.Uint64(minimum)+((cutlass.Uint64(scalars[0])+1)<<shift)-1
        upper=cutlass.Uint32(min(edge,cutlass.Uint64(maximum)))
        threshold=key_score(upper)
        if scale>0:
            threshold=max(low,high-(cutlass.Float32(scalars[0])+1.0)/scale)
        thresholds[row]=threshold
        scores[row,scores.shape[1]-32+0]=scores.element_type(high)
        scores[row,scores.shape[1]-32+1]=scores.element_type(threshold)


@cute.jit
def propose(row,tid,scores,hist,scalars,thresholds,samples,rank):
    # Runtime extent selects cache work; the compiled function is shared by N.
    if samples<=512:
        propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,2)
    elif samples<=1024:
        propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,4)
    elif samples<=2048:
        propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,8)
    elif samples<=4096:
        propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,16)
    else:
        propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,32)
