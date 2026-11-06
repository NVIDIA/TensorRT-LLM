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

from .sample_bucket import unsigned_bucket_index
import cutlass,cutlass.cute as cute
from .subset_barrier import subset_barrier
from .selection import score_key, key_score

@cute.jit
def propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,items:cutlass.Constexpr,barrier,high_rank,sample_info,score_row=None,unsigned_bucket:cutlass.Constexpr=False):
    """One adaptive 256-bin proposal, rather than an exact sample sort (K1)."""
    if cutlass.const_expr(score_row is None):score_row=row
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    for i in cutlass.range(tid,samples,128):
        value=cutlass.Float32(scores[score_row,i])
        key=score_key(value)
        # Causal Q4 sampling can produce -inf padding for earlier rows.
        # Padding participates in ranks, but must not widen the numeric range.
        if value != float('-inf'):
            minimum=min(minimum,key);maximum=max(maximum,key)
    minimum=cute.arch.warp_redux_sync(minimum,'umin')
    maximum=cute.arch.warp_redux_sync(maximum,'umax')
    if cute.arch.lane_idx()==0:
        hist[2*(tid//32)]=cutlass.Int32(minimum)
        hist[2*(tid//32)+1]=cutlass.Int32(maximum)
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    lane=cute.arch.lane_idx()
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    if lane < 4:
        minimum=cutlass.Uint32(hist[2*lane])
        maximum=cutlass.Uint32(hist[2*lane+1])
    minimum=cute.arch.warp_redux_sync(minimum,'umin')
    maximum=cute.arch.warp_redux_sync(maximum,'umax')
    if minimum>maximum:
        minimum=score_key(cutlass.Float32(float('-inf')))
        maximum=minimum
    span=maximum-minimum
    low=key_score(maximum);high=key_score(minimum)
    scale=cutlass.Float32(0)
    if (high>low)&(high<float('inf'))&(low>float('-inf')):
        scale=cutlass.Float32(255.0)/(high-low)
    shift=cutlass.Int32(0)
    if span>0:
        shift=max(0,24-cutlass.Int32(cute.arch.clz(span)))
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    hist[tid]=0;hist[tid+128]=0
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    for i in cutlass.range(tid,samples,128):
        bucket=cutlass.Int32(256)
        if i<samples:
            value=cutlass.Float32(scores[score_row,i])
            key=score_key(value)
            if cutlass.const_expr(unsigned_bucket):
                bucket=unsigned_bucket_index(key-minimum,shift)
            else:
                bucket=min(255,cutlass.Int32((key-minimum)>>shift))
            if scale>0:
                bucket=cutlass.Int32(min(cutlass.Float32(255),max(cutlass.Float32(0),(high-value)*scale)))
            # Padding participates in ranks after every finite score. When
            # the finite range is constant, its key distance can exceed INT_MAX
            # before the signed cast and otherwise index before shared memory.
            if (value == float('-inf')) | (value != value):
                bucket=cutlass.Int32(255)
        if bucket<256:
            cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='cta')
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    n0=hist[tid];n1=hist[tid+128];prefix0=n0;prefix1=n1;lane=cute.arch.lane_idx()
    for step in cutlass.range_constexpr(5):
        delta=1<<step
        other0=cute.arch.shuffle_sync_up(prefix0,delta,mask=-1,mask_and_clamp=0)
        other1=cute.arch.shuffle_sync_up(prefix1,delta,mask=-1,mask_and_clamp=0)
        if lane>=delta:prefix0+=other0;prefix1+=other1
    if lane==31:
        hist[256+tid//32]=prefix0;hist[260+tid//32]=prefix1
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    warp_total=cutlass.Int32(0)
    if lane<8:warp_total=hist[256+lane]
    for step in cutlass.range_constexpr(5):
        delta=1<<step;other=cute.arch.shuffle_sync_up(warp_total,delta,mask=-1,mask_and_clamp=0)
        if lane>=delta:warp_total+=other
    for half in cutlass.range_constexpr(2):
        number=tid+128*half;warp=tid//32+4*half
        prior=cute.arch.shuffle_sync(warp_total,max(0,warp-1))
        if warp==0:prior=cutlass.Int32(0)
        prefix=prefix0 if cutlass.const_expr(half==0) else prefix1
        n=n0 if cutlass.const_expr(half==0) else n1
        cumulative=prior+prefix
        if (cumulative>=rank)&(cumulative-n<rank):scalars[0]=number
        if (cumulative>=high_rank)&(cumulative-n<high_rank):scalars[1]=number
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    if tid==0:
        edge=cutlass.Uint64(minimum)+((cutlass.Uint64(scalars[0])+1)<<shift)-1
        upper=cutlass.Uint32(min(edge,cutlass.Uint64(maximum)))
        threshold=key_score(upper)
        if scale>0:
            threshold=max(low,high-(cutlass.Float32(scalars[0])+1.0)/scale)
        thresholds[row]=threshold
        sample_info[row,256+0]=sample_info.element_type(high)
        sample_info[row,256+1]=sample_info.element_type(threshold)
        # Concentrate K3 bins around the top-k boundary; keep the true sample
        # maximum in 512 for recovery. Clipping changes no candidate score.
        hist_high=high
        if (high_rank>1)&(scale>0):
            hist_high=max(low,high-cutlass.Float32(scalars[1])/scale)
        sample_info[row,256+2]=sample_info.element_type(hist_high)


@cute.jit
def propose(row,tid,scores,hist,scalars,thresholds,samples,rank,barrier,high_rank,sample_info,ignore_padding:cutlass.Constexpr=False,extend_high:cutlass.Constexpr=False,short_top16:cutlass.Constexpr=False,score_row=None,narrow_range=False):
    if cutlass.const_expr(short_top16):
        if (samples==128)&(rank<=16)&(high_rank<=16):
            propose_top16(row,tid,scores,hist,thresholds,rank,high_rank,sample_info,barrier,score_row=score_row)
        elif (samples==256)&(rank<=16)&(high_rank<=16):
            propose_top16_256(row,tid,scores,hist,thresholds,rank,high_rank,sample_info,barrier,score_row=score_row)
        elif samples<=256:
            propose_short(row,tid,scores,hist,scalars,thresholds,samples,rank,2,barrier,high_rank,sample_info,ignore_padding=ignore_padding,score_row=score_row)
        else:
            propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,2,barrier,high_rank,sample_info,score_row=score_row,unsigned_bucket=ignore_padding)
    elif samples<=256:
        propose_short(row,tid,scores,hist,scalars,thresholds,samples,rank,2,barrier,high_rank,sample_info,ignore_padding=ignore_padding,score_row=score_row)
    else:
        propose_cached(row,tid,scores,hist,scalars,thresholds,samples,rank,2,barrier,high_rank,sample_info,score_row=score_row,unsigned_bucket=ignore_padding)

    if cutlass.const_expr(extend_high):
        if (tid==0)&(not narrow_range):
            high=cutlass.Float32(sample_info[row,256])
            threshold=thresholds[row]
            if (high<float('inf'))&(threshold>float('-inf'))&(high>threshold):
                sample_info[row,258]=sample_info.element_type(high+(high-threshold))

from .proposal_short import propose_short

from .proposal_top16 import propose_top16,propose_top16_256
