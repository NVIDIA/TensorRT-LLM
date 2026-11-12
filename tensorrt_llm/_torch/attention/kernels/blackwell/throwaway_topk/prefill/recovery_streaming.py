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

from .small_crossing import refine
"""Exact radix fallback traversing physical candidate banks directly."""
import cutlass
from .subset_barrier import subset_barrier
import cutlass.cute as cute
from .selection import score_key, candidate_score, candidate_index

@cute.jit
def select_stream(row,tid,source,n,required,hist,out,k:cutlass.Constexpr,dtype:cutlass.Constexpr):
    lane=cute.arch.lane_idx();warp=tid//32
    bank=tid//32;local=tid%32
    count=cutlass.Int32(source[row,bank]);span=(source.shape[1]-8)//8
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    for base in cutlass.range(0,count,32):
        offset=base+local
        if offset<count:
            entry=source[row,8+bank*span+offset]
            key=score_key(candidate_score(entry,dtype,False))
            minimum=min(minimum,key);maximum=max(maximum,key)
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    if lane==0:
        hist[2*warp]=cutlass.Int32(minimum);hist[2*warp+1]=cutlass.Int32(maximum)
    subset_barrier(barrier_id=13,number_of_threads=256)
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    if lane<8:
        minimum=cutlass.Uint32(hist[2*lane]);maximum=cutlass.Uint32(hist[2*lane+1])
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    subset_barrier(barrier_id=13,number_of_threads=256)
    shift=max(cutlass.Int32(0),24-cutlass.Int32(cute.arch.clz(minimum^maximum)))
    known_shift=shift+8
    pivot=cutlass.Uint32((cutlass.Uint64(minimum)>>known_shift)<<known_shift)
    above=cutlass.Int32(0);running=minimum!=maximum
    if not running:pivot=minimum
    small=cutlass.Boolean(False)
    boundary_count=cutlass.Int32(0)
    while running:
        hist[tid]=0
        subset_barrier(barrier_id=13,number_of_threads=256)
        for base in cutlass.range(0,count,32):
            offset=base+local;bucket=cutlass.Int32(256)
            if offset<count:
                entry=source[row,8+bank*span+offset]
                key=score_key(candidate_score(entry,dtype,False))
                if (cutlass.Uint64(key)>>known_shift)==(cutlass.Uint64(pivot)>>known_shift):
                    bucket=cutlass.Int32((key>>shift)&255)
            peers=cute.arch.match_sync(-1,bucket)
            lower=(cutlass.Uint32(1)<<lane)-1
            if (bucket<256)&((peers&lower)==0):
                cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(cute.arch.popc(peers)),sem='relaxed',scope='cta')
        subset_barrier(barrier_id=13,number_of_threads=256)
        bin_count=hist[tid];prefix=bin_count
        for step in cutlass.range_constexpr(5):
            delta=1<<step
            other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:prefix+=other
        hist[tid]=prefix
        subset_barrier(barrier_id=13,number_of_threads=256)
        warp_total=cutlass.Int32(0)
        if lane<8:warp_total=hist[lane*32+31]
        for step in cutlass.range_constexpr(5):
            delta=1<<step
            other=cute.arch.shuffle_sync_up(warp_total,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:warp_total+=other
        prior=cute.arch.shuffle_sync(warp_total,max(0,warp-1))
        if warp==0:prior=cutlass.Int32(0)
        cumulative=prior+prefix
        if (cumulative>=required-above)&(cumulative-bin_count<required-above):
            hist[256]=tid;hist[257]=cumulative-bin_count;hist[258]=bin_count
        subset_barrier(barrier_id=13,number_of_threads=256)
        pivot|=cutlass.Uint32(hist[256])<<shift
        above+=hist[257]
        known_shift=shift
        boundary_count=hist[258]
        small=(boundary_count<=128)&(shift>0)
        running=(shift>0)&(not small)
        shift=max(cutlass.Int32(0),shift-8)
        subset_barrier(barrier_id=13,number_of_threads=256)
    if tid==0:
        hist[0]=0;hist[1]=above;hist[2]=0
    for i in cutlass.range(required+tid,k,256):out[row,i]=-1
    subset_barrier(barrier_id=13,number_of_threads=256)
    ck=cute.make_tensor(cute.recast_ptr(hist.iterator+512,dtype=cutlass.Uint32),cute.make_layout(128))
    ci=cute.make_tensor(hist.iterator+640,cute.make_layout(128))
    for base in cutlass.range(0,count,32):
        offset=base+local;entry=cutlass.Int64(0);key=cutlass.Uint32(0xFFFFFFFF)
        if offset<count:
            entry=source[row,8+bank*span+offset]
            key=score_key(candidate_score(entry,dtype,False))
        strict=(offset<count)&(key<pivot)
        tie=(offset<count)&(key==pivot)
        if small:tie=(offset<count)&((cutlass.Uint64(key)>>known_shift)==(cutlass.Uint64(pivot)>>known_shift))
        ms=cute.arch.vote_ballot_sync(strict);mt=cute.arch.vote_ballot_sync(tie)
        lower=(cutlass.Uint32(1)<<lane)-1
        strict_base=cutlass.Int32(0);tie_base=cutlass.Int32(0)
        if (lane==0)&(ms!=0):strict_base=cute.arch.atomic_add(hist.iterator.llvm_ptr,cutlass.Int32(cute.arch.popc(ms)),sem='relaxed',scope='cta')
        if (lane==0)&(mt!=0):tie_base=cute.arch.atomic_add((hist.iterator+1).llvm_ptr,cutlass.Int32(cute.arch.popc(mt)),sem='relaxed',scope='cta')
        strict_base=cute.arch.shuffle_sync(strict_base,0);tie_base=cute.arch.shuffle_sync(tie_base,0)
        if strict:
            pos=strict_base+cutlass.Int32(cute.arch.popc(ms&lower))
            out[row,pos]=candidate_index(entry,False)
        if tie:
            pos=tie_base+cutlass.Int32(cute.arch.popc(mt&lower))
            if small:
                ck[pos-above]=key;ci[pos-above]=candidate_index(entry,False)
            elif pos<required:out[row,pos]=candidate_index(entry,False)
    subset_barrier(barrier_id=13,number_of_threads=256)
    if small:
        target=cute.make_tensor(out.iterator+cutlass.Int64(row)*out.stride[0],cute.make_layout(k))
        if boundary_count==required-above:
            if tid<boundary_count:target[above+tid]=ci[tid]
        else:
            refine(ck,ci,boundary_count,required-above,above,tid,target,256)
    subset_barrier(barrier_id=13,number_of_threads=256)
