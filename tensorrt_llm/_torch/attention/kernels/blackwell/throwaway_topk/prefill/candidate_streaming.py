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

"""Exact radix fallback traversing physical candidate banks directly."""
import cutlass
import cutlass.cute as cute
from .selection import score_key, candidate_score, candidate_index

@cute.jit
def stream_key(entry,dtype:cutlass.Constexpr,packed:cutlass.Constexpr):
    key=score_key(candidate_score(entry,dtype,packed))
    # Expanded BF16 scores have no information in the low 16 FP32 bits.
    # Preserve ordering while avoiding radix passes over constant key tails.
    if cutlass.const_expr(dtype==cutlass.BFloat16):key=key>>16
    return key

@cute.jit
def select_stream(row,tid,source,n,required,hist,out,k:cutlass.Constexpr,dtype:cutlass.Constexpr,threads:cutlass.Constexpr=512,barrier_id:cutlass.Constexpr=13,banks:cutlass.Constexpr=8):
    bank_threads=threads//banks
    # The radix width must match log2(threads); 128-thread callers use four banks.
    radix_bits=7 if cutlass.const_expr(threads==128) else 8 if cutlass.const_expr(threads==256) else 9
    lane=cute.arch.lane_idx();warp=tid//32
    bank=tid//bank_threads;local=tid%bank_threads
    # Both bank layouts reserve eight count slots before candidate payloads.
    count=cutlass.Int32(source[row,bank]);span=(source.shape[1]-8)//banks
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    for base in cutlass.range(0,count,bank_threads):
        offset=base+local
        if offset<count:
            entry=source[row,8+bank*span+offset]
            key=stream_key(entry,dtype,source.element_type==cutlass.Int32)
            minimum=min(minimum,key);maximum=max(maximum,key)
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    if lane==0:
        hist[2*warp]=cutlass.Int32(minimum);hist[2*warp+1]=cutlass.Int32(maximum)
    cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    if lane<threads//32:
        minimum=cutlass.Uint32(hist[2*lane]);maximum=cutlass.Uint32(hist[2*lane+1])
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
    shift=max(cutlass.Int32(0),(32-radix_bits)-cutlass.Int32(cute.arch.clz(minimum^maximum)))
    known_shift=shift+radix_bits
    pivot=cutlass.Uint32((cutlass.Uint64(minimum)>>known_shift)<<known_shift)
    above=cutlass.Int32(0);running=minimum!=maximum
    if not running:pivot=minimum
    while running:
        hist[tid]=0
        cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
        for base in cutlass.range(0,count,bank_threads):
            offset=base+local;bucket=cutlass.Int32(threads)
            if offset<count:
                entry=source[row,8+bank*span+offset]
                key=stream_key(entry,dtype,source.element_type==cutlass.Int32)
                if (cutlass.Uint64(key)>>known_shift)==(cutlass.Uint64(pivot)>>known_shift):
                    bucket=cutlass.Int32((key>>shift)&(threads-1))
            peers=cute.arch.match_sync(-1,bucket)
            lower=(cutlass.Uint32(1)<<lane)-1
            if (bucket<threads)&((peers&lower)==0):
                cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(cute.arch.popc(peers)),sem='relaxed',scope='cta')
        cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
        bin_count=hist[tid];prefix=bin_count
        for step in cutlass.range_constexpr(5):
            delta=1<<step
            other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:prefix+=other
        hist[tid]=prefix
        cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
        warp_total=cutlass.Int32(0)
        if lane<threads//32:warp_total=hist[lane*32+31]
        for step in cutlass.range_constexpr(5):
            delta=1<<step
            other=cute.arch.shuffle_sync_up(warp_total,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:warp_total+=other
        prior=cute.arch.shuffle_sync(warp_total,max(0,warp-1))
        if warp==0:prior=cutlass.Int32(0)
        cumulative=prior+prefix
        if (cumulative>=required-above)&(cumulative-bin_count<required-above):
            hist[threads]=tid;hist[threads+1]=cumulative-bin_count
        cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
        pivot|=cutlass.Uint32(hist[threads])<<shift
        above+=hist[threads+1]
        known_shift=shift;running=shift>0;shift=max(cutlass.Int32(0),shift-radix_bits)
        cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
    if tid==0:
        hist[0]=0;hist[1]=above
    for i in cutlass.range(required+tid,k,threads):out[row,i]=-1
    cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
    for base in cutlass.range(0,count,bank_threads):
        offset=base+local;entry=source.element_type(0);key=cutlass.Uint32(0xFFFFFFFF)
        if offset<count:
            entry=source[row,8+bank*span+offset]
            key=stream_key(entry,dtype,source.element_type==cutlass.Int32)
        strict=(offset<count)&(key<pivot);tie=(offset<count)&(key==pivot)
        ms=cute.arch.vote_ballot_sync(strict);mt=cute.arch.vote_ballot_sync(tie)
        lower=(cutlass.Uint32(1)<<lane)-1
        strict_base=cutlass.Int32(0);tie_base=cutlass.Int32(0)
        if (lane==0)&(ms!=0):strict_base=cute.arch.atomic_add(hist.iterator.llvm_ptr,cutlass.Int32(cute.arch.popc(ms)),sem='relaxed',scope='cta')
        if (lane==0)&(mt!=0):tie_base=cute.arch.atomic_add((hist.iterator+1).llvm_ptr,cutlass.Int32(cute.arch.popc(mt)),sem='relaxed',scope='cta')
        strict_base=cute.arch.shuffle_sync(strict_base,0);tie_base=cute.arch.shuffle_sync(tie_base,0)
        if strict:
            pos=strict_base+cutlass.Int32(cute.arch.popc(ms&lower))
            out[row,pos]=candidate_index(entry,source.element_type==cutlass.Int32)
        if tie:
            pos=tie_base+cutlass.Int32(cute.arch.popc(mt&lower))
            if pos<required:out[row,pos]=candidate_index(entry,source.element_type==cutlass.Int32)
    cute.arch.barrier(barrier_id=barrier_id,number_of_threads=threads)
