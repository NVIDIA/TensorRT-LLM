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
"""Exact sampled recovery with shared score encoding and candidate helpers."""
import cutlass
import cutlass.cute as cute
from .subset_barrier import subset_barrier
from .recovery_streaming import select_stream
from .selection import (bits32, candidate_index, candidate_score, emit_selected, encode, float32, histogram_add, histogram_window, key_score, load_banked, load_spilled, read_key, score_key)

@cute.jit
def select(row, tid, source, start, end, rank, hist, scalars, out, thresholds,
           k: cutlass.Constexpr, kind: cutlass.Constexpr, dtype: cutlass.Constexpr,
           packed: cutlass.Constexpr, threshold_only: cutlass.Constexpr = False,
           cache_size: cutlass.Constexpr = 0, threads:cutlass.Constexpr=256):
    """Exact four-byte radix threshold and tie-aware unsorted selection, threads threads."""
    i = cutlass.Int32(0)
    key = cutlass.Uint32(0xFFFFFFFF)
    if cutlass.const_expr(cache_size > 0):
        cached_keys = cute.make_rmem_tensor((cache_size,), cutlass.Uint32)
        for j in cutlass.range_constexpr(cache_size):
            i = start + tid + j * threads
            cached_keys[j] = cutlass.Uint32(0xFFFFFFFF)
            if i < end:
                cached_keys[j] = read_key(source, row, i, kind, dtype, packed)
    pivot = cutlass.Uint32(0)
    above = cutlass.Int32(0)
    if tid == 0:
        scalars[0] = 0
        scalars[1] = 0
    subset_barrier(barrier_id=13, number_of_threads=threads)
    for level in cutlass.range_constexpr(4 if dtype == cutlass.Float32 else 2):
        hist[tid] = 0
        subset_barrier(barrier_id=13, number_of_threads=threads)
        shift = cutlass.const_expr(24 - 8 * level)
        if cutlass.const_expr(cache_size > 0):
            for j in cutlass.range_constexpr(cache_size):
                i = start + tid + j * threads
                histogram_add(cached_keys[j], i < end, pivot, shift, hist)
        else:
            for base in cutlass.range(start, end, threads):
                i = base + tid
                key = cutlass.Uint32(0xFFFFFFFF)
                if i < end:
                    key = read_key(source, row, i, kind, dtype, packed)
                histogram_add(key, i < end, pivot, shift, hist)
        subset_barrier(barrier_id=13, number_of_threads=threads)
        n = hist[tid]
        prefix = n
        lane = cute.arch.lane_idx()
        for step in cutlass.range_constexpr(5):
            delta = 1 << step
            other = cute.arch.shuffle_sync_up(prefix, delta, mask=-1, mask_and_clamp=0)
            if lane >= delta:
                prefix += other
        hist[tid] = prefix
        subset_barrier(barrier_id=13, number_of_threads=threads)
        prior = cutlass.Int32(0)
        for warp in cutlass.range_constexpr(threads//32):
            if warp < tid // 32:
                prior += hist[warp * 32 + 31]
        cumulative = prior + prefix
        if (cumulative >= rank - above) & (cumulative - n < rank - above):
            scalars[0] = tid
            scalars[1] = cumulative - n
        subset_barrier(barrier_id=13, number_of_threads=threads)
        pivot |= cutlass.Uint32(scalars[0]) << shift
        above += scalars[1]
        subset_barrier(barrier_id=13, number_of_threads=threads)
    if cutlass.const_expr(threshold_only):
        if tid == 0:
            thresholds[row] = key_score(pivot)
    else:
        for i in cutlass.range(tid, k, threads):
            out[row, i] = -1
        if tid == 0:
            scalars[0] = 0
        subset_barrier(barrier_id=13, number_of_threads=threads)
        for tie in cutlass.range_constexpr(2):
            if cutlass.const_expr(cache_size > 0):
                for j in cutlass.range_constexpr(cache_size):
                    i = start + tid + j * threads
                    emit_selected(cached_keys[j], i < end, pivot, tie, i, row, source, kind, packed, k, scalars, out)
            else:
                for base in cutlass.range(start, end, threads):
                    i = base + tid
                    key = cutlass.Uint32(0xFFFFFFFF)
                    if i < end:
                        key = read_key(source, row, i, kind, dtype, packed)
                    emit_selected(key, i < end, pivot, tie, i, row, source, kind, packed, k, scalars, out)
            subset_barrier(barrier_id=13, number_of_threads=threads)

@cute.jit
def select_candidates(row,tid,candidates,counts,starts,ends,hist,scalars,out,thresholds,
                      k:cutlass.Constexpr,cap:cutlass.Constexpr,dtype:cutlass.Constexpr,packed:cutlass.Constexpr,threads:cutlass.Constexpr=256):
    n=counts[row];required=min(k,ends[row]-starts[row])
    if required==0:
        for i in cutlass.range(tid,k,threads):out[row,i]=-1
    elif n>=required:
        if n<=cap:
            select_narrow(row,tid,candidates,cutlass.Int32(0),n,required,hist,scalars,out,thresholds,k,dtype,packed,(cap+threads-1)//threads,threads)
        else:
            select(row,tid,candidates,cutlass.Int32(0),n,required,hist,scalars,out,thresholds,k,'candidate',dtype,packed,threads=threads)

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

@cute.jit
def select_narrow(row,tid,source,start,end,rank,hist,scalars,out,thresholds,
                  k:cutlass.Constexpr,dtype:cutlass.Constexpr,
                  packed:cutlass.Constexpr,cache_size:cutlass.Constexpr,threads:cutlass.Constexpr=256,threshold_only:cutlass.Constexpr=False):
    """Skip common key bits and refine only the first crossing bucket."""
    assert cute.size(hist)>=1540+cache_size*threads, "selection scratch must include hist, refinement keys and state"
    bits=cutlass.const_expr(9 if threads==512 else 8)
    keys=cute.make_tensor(cute.recast_ptr(hist.iterator+1540,dtype=cutlass.Uint32),cute.make_layout(cache_size*threads))
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    for i in cutlass.range(start+tid,end,threads):
        key=read_key(source,row,i,'candidate',dtype,packed)
        keys[i-start]=key;minimum=min(minimum,key);maximum=max(maximum,key)
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    if cute.arch.lane_idx()==0:
        hist[2*(tid//32)]=cutlass.Int32(minimum)
        hist[2*(tid//32)+1]=cutlass.Int32(maximum)
    subset_barrier(barrier_id=13,number_of_threads=threads)
    lane=cute.arch.lane_idx()
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    if lane < threads//32:
        minimum=cutlass.Uint32(hist[2*lane])
        maximum=cutlass.Uint32(hist[2*lane+1])
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    subset_barrier(barrier_id=13,number_of_threads=threads)
    # One caller-owned allocation gives histogram and refinement scratch
    # explicit disjoint offsets.
    narrow=cute.make_tensor(cute.recast_ptr(hist.iterator+512,dtype=cutlass.Uint32),cute.make_layout(1024))
    state=cute.make_tensor(hist.iterator+1536,cute.make_layout(2))
    small=cute.make_rmem_tensor(((1024+threads-1)//threads,),cutlass.Uint32)
    end_bit=cutlass.const_expr(0 if dtype==cutlass.Float32 else 16)
    shift=max(cutlass.Int32(end_bit),32-bits-cutlass.Int32(cute.arch.clz(minimum^maximum)))
    known_shift=shift+bits
    pivot=cutlass.Uint32((cutlass.Uint64(minimum)>>known_shift)<<known_shift)
    above=cutlass.Int32(0);narrow_size=cutlass.Int32(0)
    use_narrow=cutlass.Boolean(False);first=cutlass.Boolean(True)
    running=minimum!=maximum
    if not running:pivot=minimum
    while running:
        hist[tid]=0
        subset_barrier(barrier_id=13,number_of_threads=threads)
        if use_narrow:
            for j in cutlass.range_constexpr((1024+threads-1)//threads):
                histogram_window(small[j],tid+j*threads<narrow_size,pivot,shift,known_shift,hist,bits)
        else:
            for chunk in cutlass.range(0,end-start,threads*4):
                for j in cutlass.range_constexpr(4):
                    offset=chunk+tid+j*threads;key=cutlass.Uint32(0xFFFFFFFF)
                    if start+offset<end:key=cutlass.Uint32(keys[offset])
                    histogram_window(key,start+chunk+tid+j*threads<end,pivot,shift,known_shift,hist,bits)
        subset_barrier(barrier_id=13,number_of_threads=threads)
        n=hist[tid];prefix=n;lane=cute.arch.lane_idx()
        for step in cutlass.range_constexpr(5):
            delta=1<<step;other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:prefix+=other
        hist[tid]=prefix
        subset_barrier(barrier_id=13,number_of_threads=threads)
        warp_total=cutlass.Int32(0)
        if lane < threads//32:warp_total=hist[lane*32+31]
        for step in cutlass.range_constexpr(5):
            delta=1<<step
            other=cute.arch.shuffle_sync_up(warp_total,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:warp_total+=other
        prior=cute.arch.shuffle_sync(warp_total,max(0,tid//32-1))
        if tid//32==0:prior=cutlass.Int32(0)
        cumulative=prior+prefix
        if (cumulative>=rank-above)&(cumulative-n<rank-above):
            scalars[0]=tid;scalars[1]=cumulative-n;state[0]=n
        subset_barrier(barrier_id=13,number_of_threads=threads)
        pivot|=cutlass.Uint32(scalars[0])<<shift
        above+=scalars[1]
        if first:narrow_size=state[0]
        if first & (narrow_size<=1024):
            if tid==0:state[1]=0
            subset_barrier(barrier_id=13,number_of_threads=threads)
            for chunk in cutlass.range(0,end-start,threads*4):
                for j in cutlass.range_constexpr(4):
                    offset=chunk+tid+j*threads;key=cutlass.Uint32(0xFFFFFFFF)
                    if start+offset<end:key=cutlass.Uint32(keys[offset])
                    hit=(start+chunk+tid+j*threads<end)&((key>>shift)==(pivot>>shift))
                    mask=cute.arch.vote_ballot_sync(hit);total=cutlass.Int32(cute.arch.popc(mask))
                    base=cutlass.Int32(0)
                    if total>0:
                        if lane==0:base=cute.arch.atomic_add((state.iterator+1).llvm_ptr,total,sem='relaxed',scope='cta')
                        base=cute.arch.shuffle_sync(base,0)
                        pos=base+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<lane)-1)))
                        if hit:narrow[pos]=key
            subset_barrier(barrier_id=13,number_of_threads=threads)
            for j in cutlass.range_constexpr((1024+threads-1)//threads):
                small[j]=cutlass.Uint32(0xFFFFFFFF)
                if tid+j*threads<narrow_size:small[j]=narrow[tid+j*threads]
            use_narrow=True
        first=False
        subset_barrier(barrier_id=13,number_of_threads=threads)
        known_shift=shift
        running=shift>end_bit
        shift=max(cutlass.Int32(end_bit),shift-bits)
    if cutlass.const_expr(threshold_only):
        if tid==0:
            value=key_score(pivot)
            if cutlass.const_expr(dtype!=cutlass.Float32):
                ordered=cutlass.Uint32(pivot>>16)
                payload=cutlass.Uint16((~ordered)&0x7FFF)
                if (ordered&0x8000)!=0:payload=cutlass.Uint16(ordered)
                value=cutlass.Float32(dtype(llvm.bitcast(dtype.mlir_type,payload.ir_value())))
            thresholds[row]=value
    else:
        for i in cutlass.range(tid,k,threads):out[row,i]=-1
        strict_local=cutlass.Int32(0);tie_local=cutlass.Int32(0)
        for chunk in cutlass.range(0,end-start,threads*4):
            for j in cutlass.range_constexpr(4):
                offset=chunk+tid+j*threads;key=cutlass.Uint32(0xFFFFFFFF)
                if start+offset<end:key=cutlass.Uint32(keys[offset])
                active=start+chunk+tid+j*threads<end
                strict_local+=cutlass.Int32(active&(key<pivot))
                tie_local+=cutlass.Int32(active&(key==pivot))
        for step in cutlass.range_constexpr(5):
            strict_local+=cute.arch.shuffle_sync_bfly(strict_local,1<<step)
            tie_local+=cute.arch.shuffle_sync_bfly(tie_local,1<<step)
        if lane==0:
            hist[2*(tid//32)]=strict_local;hist[2*(tid//32)+1]=tie_local
        subset_barrier(barrier_id=13,number_of_threads=threads)
        strict_prefix=cutlass.Int32(0);tie_prefix=cutlass.Int32(0)
        if lane<threads//32:
            strict_prefix=hist[2*lane];tie_prefix=hist[2*lane+1]
        for step in cutlass.range_constexpr(5):
            delta=1<<step
            ps=cute.arch.shuffle_sync_up(strict_prefix,delta,mask=-1,mask_and_clamp=0)
            pt=cute.arch.shuffle_sync_up(tie_prefix,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:strict_prefix+=ps;tie_prefix+=pt
        strict_cursor=cute.arch.shuffle_sync(strict_prefix,max(0,tid//32-1))
        tie_cursor=above+cute.arch.shuffle_sync(tie_prefix,max(0,tid//32-1))
        if tid//32==0:strict_cursor=cutlass.Int32(0);tie_cursor=above
        for chunk in cutlass.range(0,end-start,threads*4):
            for j in cutlass.range_constexpr(4):
                offset=chunk+tid+j*threads;key=cutlass.Uint32(0xFFFFFFFF)
                if start+offset<end:key=cutlass.Uint32(keys[offset])
                i=start+chunk+tid+j*threads;active=i<end
                strict=active&(key<pivot);tie=active&(key==pivot)
                ms=cute.arch.vote_ballot_sync(strict);mt=cute.arch.vote_ballot_sync(tie)
                lower=(cutlass.Uint32(1)<<lane)-1
                pos=cutlass.Int32(0)
                if strict:pos=strict_cursor+cutlass.Int32(cute.arch.popc(ms&lower))
                else:pos=tie_cursor+cutlass.Int32(cute.arch.popc(mt&lower))
                if (strict|tie)&(pos<k):out[row,pos]=candidate_index(load_banked(source,row,i),packed)
                strict_cursor+=cutlass.Int32(cute.arch.popc(ms));tie_cursor+=cutlass.Int32(cute.arch.popc(mt))
        subset_barrier(barrier_id=13,number_of_threads=threads)

@cute.jit
def select_or_prepare_retry(row,tid,candidates,counts,starts,ends,hist,scalars,out,thresholds,
                            k:cutlass.Constexpr,cap:cutlass.Constexpr,dtype:cutlass.Constexpr,
                            packed:cutlass.Constexpr,threads:cutlass.Constexpr=256):
    # K4 retains overflow candidates losslessly; this selector does not rerun GEMM.
    select_candidates(row,tid,candidates,counts,starts,ends,hist,scalars,out,thresholds,k,cap,dtype,packed,threads)

@cute.jit
def finish_recovery(row,tid,candidates,counts,starts,ends,dense,hist,scalars,out,thresholds,
                    k:cutlass.Constexpr,cap:cutlass.Constexpr,dtype:cutlass.Constexpr,packed:cutlass.Constexpr,sample_storage,threads:cutlass.Constexpr=256):
    # Reference recovery_sample16: sample -> compact -> reduced exact selection.
    # Wide global banks replace its bounded shared packed16 array.
    length=ends[row]-starts[row];required=min(k,length)
    for j in cutlass.range_constexpr(3):
        slot=tid+j*256
        col=starts[row]+cutlass.Int32((cutlass.Int64(2*slot+1)*cutlass.Int64(length))//1536)
        sample_storage[slot]=cutlass.Float32(dense[row,col])
    subset_barrier(barrier_id=13,number_of_threads=256)
    samples=cute.make_tensor(sample_storage.iterator,cute.make_layout((dense.shape[0],768),stride=(0,1)))
    rank=min(cutlass.Int32(768),max(cutlass.Int32(1),(6*k*768+length-1)//max(1,length)))
    propose(row,tid,samples,hist,scalars,thresholds,cutlass.Int32(768),rank)
    subset_barrier(barrier_id=13,number_of_threads=256)
    tau=thresholds[row];lane=cute.arch.lane_idx();bank=tid//32
    span=(candidates.shape[1]-8)//8;source_span=(length+7)//8;cursor=cutlass.Int32(0)
    for base in cutlass.range(0,source_span,32):
        offset=base+lane;col=starts[row]+bank*source_span+offset
        active=(offset<source_span)&(col<ends[row]);value=cutlass.Float32(0);hit=cutlass.Boolean(False)
        if active:
            value=cutlass.Float32(dense[row,col]);hit=(value>=tau)|(value!=value)
        mask=cute.arch.vote_ballot_sync(hit)
        if hit:
            position=cursor+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<lane)-1)))
            if position<span:candidates[row,8+bank*span+position]=cutlass.Int64(encode(value,col,dtype,False))
        cursor+=cutlass.Int32(cute.arch.popc(mask))
    if lane==0:candidates[row,bank]=cutlass.Int64(cursor)
    subset_barrier(barrier_id=13,number_of_threads=256)
    total=cutlass.Int32(0)
    if lane<8:total=cutlass.Int32(candidates[row,lane])
    maximum=total
    for step in cutlass.range_constexpr(5):
        total+=cute.arch.shuffle_sync_bfly(total,1<<step)
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    if (total>=required)&(maximum<=span):
        select_stream(row,tid,candidates,total,required,hist,out,k,dtype)
    else:
        select(row,tid,dense,starts[row],ends[row],required,hist,scalars,out,thresholds,k,'dense',dtype,packed,threads=threads)
