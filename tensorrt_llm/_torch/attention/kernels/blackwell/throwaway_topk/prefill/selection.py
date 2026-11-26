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
"""CuTe DSL sampling, filtering and exact radix selection.

Selection uses rounded scores and INT32 output indices. Helpers support
wide candidates and packed (score16, index16) records."""
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm


@cute.jit
def bits32(x):
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, cutlass.Float32(x).ir_value()))


@cute.jit
def float32(x):
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, cutlass.Uint32(x).ir_value()))


@cute.jit
def score_key(x):
    # Descending numeric order; canonical zero and NaN. Padding is UINT_MAX.
    v = cutlass.Float32(x)
    if v == 0.0:
        v = cutlass.Float32(0.0)
    b = bits32(v)
    key = (~b) & cutlass.Uint32(0x7FFFFFFF)
    if (b & 0x80000000) != 0:
        key = b
    if v != v:
        key = cutlass.Uint32(0xFF800000)
    return cutlass.Uint32(key)


@cute.jit
def key_score(key):
    b = (~key) & cutlass.Uint32(0x7FFFFFFF)
    if (key & 0x80000000) != 0:
        b = key
    return float32(b)


@cute.jit
def encode(score, index, dtype: cutlass.Constexpr, packed: cutlass.Constexpr):
    # Keep the rounded payload; key/bucket conversion ranks NaNs as -inf.
    # Avoid redundant canonicalization in the candidate publication path.
    value = cutlass.Uint64(0)
    if cutlass.const_expr(packed):
        s = dtype(score)
        payload = cutlass.Uint16(llvm.bitcast(cutlass.Uint16.mlir_type, s.ir_value()))
        value = (cutlass.Uint64(payload) << 16) | cutlass.Uint64(cutlass.Uint32(index))
    else:
        value = (cutlass.Uint64(bits32(cutlass.Float32(dtype(score)))) << 32) | cutlass.Uint64(cutlass.Uint32(index))
    return value


@cute.jit
def candidate_score(value, dtype: cutlass.Constexpr, packed: cutlass.Constexpr):
    result = cutlass.Float32(0)
    if cutlass.const_expr(packed):
        b = cutlass.Uint16(cutlass.Uint32(value) >> 16)
        result = cutlass.Float32(dtype(llvm.bitcast(dtype.mlir_type, b.ir_value())))
    else:
        result = float32(cutlass.Uint32(cutlass.Uint64(value) >> 32))
    return result


@cute.jit
def candidate_index(value, packed: cutlass.Constexpr):
    index = cutlass.Int32(cutlass.Uint64(value) & 0xFFFFFFFF)
    if cutlass.const_expr(packed):
        index = cutlass.Int32(cutlass.Uint32(value) & 0xFFFF)
    return index


@cute.jit
def append(score, row, index, counts, candidates, cap: cutlass.Constexpr,
           dtype: cutlass.Constexpr, packed: cutlass.Constexpr, thresholds):
    value = cutlass.Float32(dtype(score))
    hit = (value >= thresholds[row]) | (value != value)
    mask = cute.arch.vote_ballot_sync(hit)
    total = cutlass.Int32(cute.arch.popc(mask))
    lane = cute.arch.lane_idx()
    base = cutlass.Int32(0)
    if total > 0:
        if lane == 0:
            base = cute.arch.atomic_add((counts.iterator + row).llvm_ptr, total, sem='relaxed', scope='gpu')
        base = cute.arch.shuffle_sync(base, 0)
        rank = cutlass.Int32(cute.arch.popc(mask & ((cutlass.Uint32(1) << lane) - 1)))
        if hit & (base + rank < cap):
            candidates[row, base + rank] = encode(value, index, dtype, packed)


@cute.jit
def read_key(source, row, i, kind: cutlass.Constexpr, dtype: cutlass.Constexpr,
             packed: cutlass.Constexpr):
    v = cutlass.Float32(0)
    if cutlass.const_expr(kind == 'spill'):
        v = candidate_score(load_spilled(source,row,i),dtype,False)
    elif cutlass.const_expr(kind == 'candidate'):
        v = candidate_score(load_banked(source,row,i), dtype, packed)
    else:
        v = cutlass.Float32(source[row, i])
    key = cutlass.Uint32(0)
    if cutlass.const_expr(dtype == cutlass.Float32):
        key = score_key(v)
    else:
        value = dtype(v)
        bits = cutlass.Uint16(llvm.bitcast(cutlass.Uint16.mlir_type,value.ir_value()))
        if v == 0.0:bits = cutlass.Uint16(0)
        ordered = (~cutlass.Uint32(bits)) & 0x7FFF
        if (bits & 0x8000)!=0:ordered = cutlass.Uint32(bits)
        if v != v:
            padding = dtype(float('-inf'))
            ordered = cutlass.Uint32(llvm.bitcast(cutlass.Uint16.mlir_type, padding.ir_value()))
        key = ordered << 16
    return key


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
    cute.arch.barrier(barrier_id=13, number_of_threads=threads)
    for level in cutlass.range_constexpr(4 if dtype == cutlass.Float32 else 2):
        hist[tid] = 0
        cute.arch.barrier(barrier_id=13, number_of_threads=threads)
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
        cute.arch.barrier(barrier_id=13, number_of_threads=threads)
        n = hist[tid]
        prefix = n
        lane = cute.arch.lane_idx()
        for step in cutlass.range_constexpr(5):
            delta = 1 << step
            other = cute.arch.shuffle_sync_up(prefix, delta, mask=-1, mask_and_clamp=0)
            if lane >= delta:
                prefix += other
        hist[tid] = prefix
        cute.arch.barrier(barrier_id=13, number_of_threads=threads)
        prior = cutlass.Int32(0)
        for warp in cutlass.range_constexpr(threads//32):
            if warp < tid // 32:
                prior += hist[warp * 32 + 31]
        cumulative = prior + prefix
        if (cumulative >= rank - above) & (cumulative - n < rank - above):
            scalars[0] = tid
            scalars[1] = cumulative - n
        cute.arch.barrier(barrier_id=13, number_of_threads=threads)
        pivot |= cutlass.Uint32(scalars[0]) << shift
        above += scalars[1]
        cute.arch.barrier(barrier_id=13, number_of_threads=threads)
    if cutlass.const_expr(threshold_only):
        if tid == 0:
            thresholds[row] = key_score(pivot)
    else:
        for i in cutlass.range(tid, k, threads):
            out[row, i] = -1
        if tid == 0:
            scalars[0] = 0
        cute.arch.barrier(barrier_id=13, number_of_threads=threads)
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
            cute.arch.barrier(barrier_id=13, number_of_threads=threads)


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
def emit(score, row, col, sample_col, valid, nq, dense, thresholds, counts,
         candidates, mode: cutlass.Constexpr, dtype: cutlass.Constexpr,
         packed: cutlass.Constexpr, cap: cutlass.Constexpr):
    if row < nq:
        if cutlass.const_expr(mode == 'filter'):
            # Uniform warp call even when the causal mask cuts a warp in half.
            value = cutlass.Float32(dtype(score))
            threshold = thresholds[row]
            hit = valid & ((value >= threshold) | (value != value))
            mask = cute.arch.vote_ballot_sync(hit)
            total = cutlass.Int32(cute.arch.popc(mask))
            lane = cute.arch.lane_idx()
            base = cutlass.Int32(0)
            if total > 0:
                if lane == 0:
                    base = cute.arch.atomic_add((counts.iterator + row).llvm_ptr, total, sem='relaxed', scope='gpu')
                base = cute.arch.shuffle_sync(base, 0)
                pos = base + cutlass.Int32(cute.arch.popc(mask & ((cutlass.Uint32(1) << lane) - 1)))
                if hit & (pos < cap):
                    entry = encode(value, col, dtype, packed)
                    if cutlass.const_expr(packed):
                        candidates[row, pos] = cutlass.Int32(entry)
                    else:
                        candidates[row, pos] = cutlass.Int64(entry)
                if hit & (pos >= cap):
                    spill=overflow_view(dense,dtype)
                    spill[row,pos]=cutlass.Int64(encode(value,col,dtype,False))
        elif valid:
            if score != score:
                score = cutlass.Float32(float('-inf'))
            index = col
            if cutlass.const_expr(mode == 'sample'):
                index = sample_col
            if cutlass.const_expr(mode=='sample'):
                dense[row,index]=dtype(score)
            elif counts[row]<=cap:
                dense[row,index]=dtype(score)


@cute.jit
def runtime_samples(length,k:cutlass.Constexpr=2048,sample_factor:cutlass.Constexpr=32):
    target=max(512,min(8192,(max(1,length)*sample_factor+k-1)//k))
    return cutlass.Int32(cutlass.Uint32(1)<<(32-cutlass.Int32(cute.arch.clz(cutlass.Uint32(target-1)))))

@cute.jit
def block_samples(qb,ends,nq,bq:cutlass.Constexpr,k:cutlass.Constexpr=2048,sample_factor:cutlass.Constexpr=32):
    return runtime_samples(ends[min(nq-1,qb*bq+bq-1)],k,sample_factor=sample_factor)


@cute.jit
def sample_tile(tile, qb, ends, nq, bq: cutlass.Constexpr, maximum_samples:cutlass.Constexpr=8192,k:cutlass.Constexpr=2048,sample_factor:cutlass.Constexpr=32):
    """Stratified 128-key TMA tiles; exact recovery covers proposal errors."""
    n = ends[min(qb * bq + bq - 1, nq - 1)]
    tiles = (n + 127) // 128
    samples=block_samples(qb,ends,nq,bq,k,sample_factor=sample_factor)
    # Midpoint of each stratum. This samples clustered TMA tiles, rather than
    # the reference's individual-key gather. It affects proposal efficiency only.
    return min(tiles - 1, ((2 * tile + 1) * tiles) // (2*(samples//128)))


@cute.jit
def histogram_add(key, active, pivot, shift: cutlass.Constexpr, hist):
    if cutlass.const_expr(shift < 24):
        active &= (key >> (shift+8)) == (pivot >> (shift+8))
    bucket = cutlass.Int32(256)
    if active:
        bucket = cutlass.Int32((key >> shift) & 255)
    peers = cute.arch.match_sync(-1, bucket)
    lower = (cutlass.Uint32(1) << cute.arch.lane_idx())-1
    if (bucket < 256) & ((peers & lower)==0):
        cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr, cutlass.Int32(cute.arch.popc(peers)), sem='relaxed', scope='cta')


@cute.jit
def emit_selected(key, active, pivot, tie: cutlass.Constexpr, i, row, source,
                  kind: cutlass.Constexpr, packed: cutlass.Constexpr,
                  k: cutlass.Constexpr, scalars, out):
    hit = active & (key < pivot)
    if cutlass.const_expr(tie == 1):
        hit = active & (key == pivot)
    mask = cute.arch.vote_ballot_sync(hit)
    total = cutlass.Int32(cute.arch.popc(mask))
    base = cutlass.Int32(0)
    if total > 0:
        if cute.arch.lane_idx() == 0:
            base = cute.arch.atomic_add(scalars.iterator.llvm_ptr, total, sem='relaxed', scope='cta')
        base = cute.arch.shuffle_sync(base, 0)
        pos = base + cutlass.Int32(cute.arch.popc(mask & ((cutlass.Uint32(1) << cute.arch.lane_idx())-1)))
        if hit & (pos < k):
            index = i
            if cutlass.const_expr(kind == 'spill'):
                index=candidate_index(load_spilled(source,row,i),False)
            elif cutlass.const_expr(kind == 'candidate'):
                index = candidate_index(load_banked(source,row,i), packed)
            out[row, pos] = index


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
    cute.arch.barrier(barrier_id=13,number_of_threads=256)
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
    cute.arch.barrier(barrier_id=13,number_of_threads=256)
    hist[tid]=0
    cute.arch.barrier(barrier_id=13,number_of_threads=256)
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
    cute.arch.barrier(barrier_id=13,number_of_threads=256)
    n=hist[tid];prefix=n;lane=cute.arch.lane_idx()
    for step in cutlass.range_constexpr(5):
        delta=1<<step;other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
        if lane>=delta:prefix+=other
    hist[tid]=prefix
    cute.arch.barrier(barrier_id=13,number_of_threads=256)
    warp_total=cutlass.Int32(0)
    if lane < 8:warp_total=hist[lane*32+31]
    for step in cutlass.range_constexpr(5):
        delta=1<<step;other=cute.arch.shuffle_sync_up(warp_total,delta,mask=-1,mask_and_clamp=0)
        if lane>=delta:warp_total+=other
    prior=cute.arch.shuffle_sync(warp_total,max(0,tid//32-1))
    if tid//32==0:prior=cutlass.Int32(0)
    cumulative=prior+prefix
    if (cumulative>=rank)&(cumulative-n<rank):scalars[0]=tid
    cute.arch.barrier(barrier_id=13,number_of_threads=256)
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
def publish_private(score, active, threshold, row, index, cursor, counts, candidates,
                    cap: cutlass.Constexpr, dtype: cutlass.Constexpr,
                    packed: cutlass.Constexpr, splits: cutlass.Constexpr, dense):
    """A row-private publishing warp avoids global reservations in prefill."""
    value=cutlass.Float32(dtype(score))
    hit=active & ((value>=threshold)|(value!=value))
    mask=cute.arch.vote_ballot_sync(hit)
    total=cutlass.Int32(cute.arch.popc(mask))
    base=cursor
    if cutlass.const_expr(splits > 1):
        base=cutlass.Int32(0)
        if total>0:
            if cute.arch.lane_idx()==0:
                base=cute.arch.atomic_add((counts.iterator+row).llvm_ptr,total,sem='relaxed',scope='gpu')
            base=cute.arch.shuffle_sync(base,0)
    rank=cutlass.Int32(cute.arch.popc(mask & ((cutlass.Uint32(1)<<cute.arch.lane_idx())-1)))
    if hit & (base+rank<cap):
        entry=encode(value,index,dtype,packed)
        if cutlass.const_expr(packed):candidates[row,base+rank]=cutlass.Int32(entry)
        else:candidates[row,base+rank]=cutlass.Int64(entry)
    if hit & (base+rank>=cap):
        spill=overflow_view(dense,dtype)
        spill[row,base+rank]=cutlass.Int64(encode(value,index,dtype,False))
    return cursor+total


@cute.jit
def histogram_window(key,active,pivot,shift,known_shift,hist,bits:cutlass.Constexpr):
    active &= (cutlass.Uint64(key)>>known_shift)==(cutlass.Uint64(pivot)>>known_shift)
    bucket=cutlass.Int32(1<<bits)
    if active:bucket=cutlass.Int32((key>>shift)&((1<<bits)-1))
    peers=cute.arch.match_sync(-1,bucket)
    lower=(cutlass.Uint32(1)<<cute.arch.lane_idx())-1
    if (bucket<(1<<bits))&((peers&lower)==0):
        cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(cute.arch.popc(peers)),sem='relaxed',scope='cta')


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
    cute.arch.barrier(barrier_id=13,number_of_threads=threads)
    lane=cute.arch.lane_idx()
    minimum=cutlass.Uint32(0xFFFFFFFF);maximum=cutlass.Uint32(0)
    if lane < threads//32:
        minimum=cutlass.Uint32(hist[2*lane])
        maximum=cutlass.Uint32(hist[2*lane+1])
    for step in cutlass.range_constexpr(5):
        minimum=min(minimum,cute.arch.shuffle_sync_bfly(minimum,1<<step))
        maximum=max(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<step))
    cute.arch.barrier(barrier_id=13,number_of_threads=threads)
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
        cute.arch.barrier(barrier_id=13,number_of_threads=threads)
        if use_narrow:
            for j in cutlass.range_constexpr((1024+threads-1)//threads):
                histogram_window(small[j],tid+j*threads<narrow_size,pivot,shift,known_shift,hist,bits)
        else:
            for chunk in cutlass.range(0,end-start,threads*4):
                for j in cutlass.range_constexpr(4):
                    offset=chunk+tid+j*threads;key=cutlass.Uint32(0xFFFFFFFF)
                    if start+offset<end:key=cutlass.Uint32(keys[offset])
                    histogram_window(key,start+chunk+tid+j*threads<end,pivot,shift,known_shift,hist,bits)
        cute.arch.barrier(barrier_id=13,number_of_threads=threads)
        n=hist[tid];prefix=n;lane=cute.arch.lane_idx()
        for step in cutlass.range_constexpr(5):
            delta=1<<step;other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
            if lane>=delta:prefix+=other
        hist[tid]=prefix
        cute.arch.barrier(barrier_id=13,number_of_threads=threads)
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
        cute.arch.barrier(barrier_id=13,number_of_threads=threads)
        pivot|=cutlass.Uint32(scalars[0])<<shift
        above+=scalars[1]
        if first:narrow_size=state[0]
        if first & (narrow_size<=1024):
            if tid==0:state[1]=0
            cute.arch.barrier(barrier_id=13,number_of_threads=threads)
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
            cute.arch.barrier(barrier_id=13,number_of_threads=threads)
            for j in cutlass.range_constexpr((1024+threads-1)//threads):
                small[j]=cutlass.Uint32(0xFFFFFFFF)
                if tid+j*threads<narrow_size:small[j]=narrow[tid+j*threads]
            use_narrow=True
        first=False
        cute.arch.barrier(barrier_id=13,number_of_threads=threads)
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
        cute.arch.barrier(barrier_id=13,number_of_threads=threads)
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
        cute.arch.barrier(barrier_id=13,number_of_threads=threads)


@cute.jit
def overflow_view(dense,dtype:cutlass.Constexpr):
    stride=cutlass.Int64(dense.stride[0])*(dtype.width//8)//8
    return cute.make_tensor(cute.recast_ptr(dense.iterator,dtype=cutlass.Int64),
        cute.make_layout((dense.shape[0],stride),stride=(stride,1)))

@cute.jit
def load_spilled(source,row,i):
    value=cutlass.Int64(0)
    if i<source[2]:value=source[0][row,i]
    else:value=source[1][row,i]
    return value

@cute.jit
def select_or_prepare_retry(row,tid,candidates,counts,starts,ends,hist,scalars,out,thresholds,
                            k:cutlass.Constexpr,cap:cutlass.Constexpr,dtype:cutlass.Constexpr,
                            packed:cutlass.Constexpr,threads:cutlass.Constexpr=256):
    # K4 retains overflow candidates losslessly; this selector does not rerun GEMM.
    select_candidates(row,tid,candidates,counts,starts,ends,hist,scalars,out,thresholds,k,cap,dtype,packed,threads)

@cute.jit
def finish_recovery(row,tid,candidates,counts,starts,ends,dense,hist,scalars,out,thresholds,
                    k:cutlass.Constexpr,cap:cutlass.Constexpr,dtype:cutlass.Constexpr,packed:cutlass.Constexpr,threads:cutlass.Constexpr=256):
    """Select dense logits after the caller has detected candidate underflow."""
    required=min(k,ends[row]-starts[row])
    select(row,tid,dense,starts[row],ends[row],required,hist,scalars,out,thresholds,k,'dense',dtype,packed,threads=threads)

@cute.jit
def append_cached(score,active,threshold,row,index,counts,candidates,dense,cap:cutlass.Constexpr,dtype:cutlass.Constexpr,packed:cutlass.Constexpr):
    value=cutlass.Float32(dtype(score))
    hit=active & ((value>=threshold)|(value!=value))
    mask=cute.arch.vote_ballot_sync(hit)
    total=cutlass.Int32(cute.arch.popc(mask))
    base=cutlass.Int32(0)
    if total>0:
        if cute.arch.lane_idx()==0:
            base=cute.arch.atomic_add((counts.iterator+row).llvm_ptr,total,sem='relaxed',scope='gpu')
        base=cute.arch.shuffle_sync(base,0)
        pos=base+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<cute.arch.lane_idx())-1)))
        if hit:
            entry=encode(value,index,dtype,packed)
            if pos<cap:
                if cutlass.const_expr(packed):candidates[row,pos]=cutlass.Int32(entry)
                else:candidates[row,pos]=cutlass.Int64(entry)
            else:
                spill=overflow_view(dense,dtype)
                spill[row,pos]=cutlass.Int64(encode(value,index,dtype,False))

@cute.jit
def load_banked(source,row,i):
    span=(source.shape[1]-8)//8
    physical=cutlass.Int32(8)+i
    prefix=cutlass.Int32(0)
    for bank in cutlass.range_constexpr(8):
        count=cutlass.Int32(source[row,bank])
        if (i>=prefix)&(i<prefix+count):physical=8+bank*span+i-prefix
        prefix+=count
    return source[row,physical]

@cute.jit
def publish_bank(score,active,tau,row,index,cursor,candidates,bank,dtype:cutlass.Constexpr):
    value=cutlass.Float32(dtype(score));hit=active&((value>=tau)|(value!=value))
    mask=cute.arch.vote_ballot_sync(hit);total=cutlass.Int32(cute.arch.popc(mask))
    lane=cute.arch.lane_idx();pos=cursor+cutlass.Int32(cute.arch.popc(mask&((cutlass.Uint32(1)<<lane)-1)))
    if hit:
        span=(candidates.shape[1]-8)//8
        candidates[row,8+bank*span+pos]=cutlass.Int64(encode(value,index,dtype,False))
    return cursor+total
