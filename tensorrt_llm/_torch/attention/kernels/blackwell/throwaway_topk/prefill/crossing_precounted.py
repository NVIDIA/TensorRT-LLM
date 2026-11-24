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

"""Reference-style coarse histogram, exact crossing-bin refinement, radix fallback."""
from .. import memory as indexer_memory
from ..pdl import pdl_prologue
from .short_domain import candidate_bounds
import functools
from .candidate_streaming import select_stream
from .small_crossing import refine_quantized
from .half_bucket_uniform import uniform_half_bucket
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
import cuda.bindings.driver as cuda
from ..api import DTYPES
from .selection import score_key, candidate_score, candidate_index, select, bits32, float32


@cute.jit
def bucket_for(key, tau_key, shift: cutlass.Constexpr):
    delta=cutlass.Uint32(0)
    if key<tau_key:delta=tau_key-key
    return cutlass.Int32(1023-min(cutlass.Uint32(1023),delta>>shift))


@cute.jit
def numeric_bucket(score,lower,inverse_span,bins:cutlass.Constexpr=1024):
    # FP32 subtraction and scaling are monotone; clipping only merges buckets.
    # Exact original keys are retained for crossing-bin selection/recovery.
    bucket=cutlass.Int32(0)
    # Callers use -1 for invalid spans; a finite positive span implies finite
    # lower and a positive reciprocal (bins >= 512 prevents underflow to zero).
    if inverse_span>=0.0:
        normalized=(score-lower)*inverse_span
        normalized=min(cutlass.Float32(bins-1),max(cutlass.Float32(0.0),normalized))
        bucket=(bins-1)-cutlass.Int32(normalized)
    if score!=score:bucket=cutlass.Int32(bins-1)
    return bucket


class Crossing:
    def __init__(self,k,dtype,shift=15,qblock=2,precounted=False,wide_cache=False,compact=False,packed=False,pro_short=False,exact_ties=False,glm_short=False,glm_mid_bins=False, pdl=False, exhaustive=False, candidate_cols=0):
        self.pdl = bool(pdl)
        self.exhaustive = exhaustive
        self.k,self.dtype,self.shift=k,dtype,shift
        self.packed=packed
        self.exact_ties=exact_ties
        self.entry_type=cutlass.Int32 if packed else cutlass.Int64
        self.qblock=qblock
        self.barrier_id=0 if qblock>4 or compact else 13
        self.precounted=precounted
        self.threads=256 if (qblock>4 or compact) and (k==512 or pro_short or glm_short) and not wide_cache else 512
        # One warp per candidate bank; keep 512 bins and the same per-bank cache.
        self.half_threads = compact and packed and qblock == 4 and k == 512 and not wide_cache
        if self.half_threads: self.threads=128
        self.bins_per_thread=4 if glm_mid_bins and glm_short and self.threads==256 else 2
        if self.half_threads: self.bins_per_thread=4
        self.bins=self.threads*self.bins_per_thread
        self.bank_threads=self.threads//4
        self.cache_items=16 if glm_short else 8
        if self.half_threads: self.cache_items=16
        self.bank_cache=self.bank_threads*self.cache_items
        # Use this geometry for arenas up to twice the shared cache size;
        # larger arenas need more threads to limit global reloads.
        self.ds_half = packed and qblock == 2 and k == 2048 and dtype == cutlass.BFloat16 and 0 < candidate_cols - 8 <= 2 * self.threads * self.cache_items
        if self.ds_half:
            self.barrier_id=0
            self.threads=256
            self.half_threads=True
            self.bins_per_thread=2
            self.bins=512
            self.bank_threads=64
            self.cache_items=16
            self.bank_cache=1024

    @cute.jit
    def __call__(self,candidates,counts,starts,ends,out,thresholds,samples,recovery_queue,stream:cuda.CUstream):
        self.kernel(candidates,counts,starts,ends,out,thresholds,samples,recovery_queue).launch(
            grid=(out.shape[0],1,1),block=(self.threads,1,1),stream=stream,min_blocks_per_mp=(6 if self.ds_half else 12 if self.half_threads else 6 if self.exact_ties and self.threads==256 else 2048//self.threads), use_pdl=self.pdl)

    @cute.kernel
    def kernel(self,candidates,counts,starts,ends,out,thresholds,samples,recovery_queue):
        pdl_prologue(self.pdl)
        row=cute.arch.block_idx()[0];tid=cute.arch.thread_idx()[0]
        lane=cute.arch.lane_idx();warp=tid//32
        if cutlass.const_expr(self.qblock > 4):
            if row%self.qblock==0 and tid<32:
                task_row=row+lane
                needs=False
                if task_row<out.shape[0] and lane<self.qblock:
                    needs=(ends[task_row]-starts[task_row]>self.k)&(counts[task_row]<self.k)
                mask=cute.arch.vote_ballot_sync(needs)
                if lane==0 and mask!=0:
                    position=cute.arch.atomic_add(recovery_queue.iterator.llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='gpu')
                    recovery_queue[position+1]=row//self.qblock
        else:
            if row%self.qblock==0 and tid<32:
                task_row=row+lane//8
                valid=(lane//8<self.qblock)&(task_row<out.shape[0])
                amount=cutlass.Int32(0);required_row=cutlass.Int32(0);visible_row=cutlass.Int32(0)
                if cutlass.const_expr(self.precounted):
                    if valid and lane%8==0:
                        amount=counts[task_row]
                        visible_row=ends[task_row]-starts[task_row]
                        required_row=min(self.k,visible_row)
                else:
                    if valid:
                        amount=cutlass.Int32(candidates[task_row,lane%8])
                        visible_row=ends[task_row]-starts[task_row]
                        required_row=min(self.k,visible_row)
                    for step in cutlass.range_constexpr(3):amount+=cute.arch.shuffle_sync_bfly(amount,1<<step)
                needs=valid&(lane%8==0)&(visible_row>self.k)&(amount<required_row)
                mask=cute.arch.vote_ballot_sync(needs)
                if lane==0 and mask!=0:
                    position=cute.arch.atomic_add(recovery_queue.iterator.llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='gpu')
                    recovery_queue[position+1]=row//self.qblock
        mem=indexer_memory.SmemAllocator()
        cached=mem.allocate_tensor(self.entry_type,cute.make_layout(self.threads*self.cache_items),byte_alignment=128)
        hist=mem.allocate_tensor(cutlass.Int32,cute.make_layout(2048),byte_alignment=128)
        ck=cute.make_tensor(cute.recast_ptr(hist.iterator+1024,dtype=cutlass.Uint32),cute.make_layout(128))
        ci=cute.make_tensor(hist.iterator+1152,cute.make_layout(128))
        state=cute.make_tensor(hist.iterator+1280,cute.make_layout(8))
        scalars=cute.make_tensor(hist.iterator+1290,cute.make_layout(2))
        staged=mem.allocate_tensor(cutlass.Int32,cute.make_layout(self.k),byte_alignment=128)
        if cutlass.const_expr(not self.precounted):
            if tid==0:
                total=cutlass.Int32(0)
                for bank in cutlass.range_constexpr(8):total+=cutlass.Int32(candidates[row,bank])
                counts[row]=total
            cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
        n=counts[row];required=min(self.k,ends[row]-starts[row])
        if ends[row]-starts[row]<=self.k:
            # Every visible key belongs to the result; scores and sampling are
            # irrelevant. Output order is unspecified by the TopK contract.
            for i in cutlass.range(tid,self.k,self.threads):
                value=cutlass.Int32(-1)
                if i<required:value=starts[row]+i
                out[row,i]=value
        elif n>=required:
            bank=tid//self.bank_threads;local=tid%self.bank_threads
            bank_count=cutlass.Int32(candidates[row,bank])
            threshold_value=thresholds[row]
            span=(candidates.shape[1]-8)//4
            # Each bank has four warps; each lane copies two adjacent entries.
            # Entries beyond the fixed cache are read from global memory.
            # Copy an odd final entry separately; padding is never read.
            if cutlass.const_expr(self.packed):
                copy_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),self.entry_type,num_bits_per_copy=128)
                tail_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Int64,num_bits_per_copy=64)
                copy_entries=4 if cutlass.const_expr(self.packed) else 2
                for group in cutlass.range_constexpr(self.cache_items//copy_entries):
                    offset=group*(self.bank_threads*copy_entries)+local*copy_entries
                    if offset<bank_count:
                        src=cute.make_tensor((candidates.iterator+cutlass.Int64(row)*candidates.stride[0]+8+bank*span+offset).align(16),cute.make_layout(copy_entries))
                        dst=cute.make_tensor((cached.iterator+bank*self.bank_cache+offset).align(16),cute.make_layout(copy_entries))
                        if cutlass.const_expr(self.k==1024):
                            cute.copy(copy_atom,src,dst)
                        else:
                            if offset+1<bank_count:
                                cute.copy(copy_atom,src,dst)
                            else:
                                tail_src=cute.make_tensor(src.iterator,cute.make_layout(1))
                                tail_dst=cute.make_tensor(dst.iterator,cute.make_layout(1))
                                cute.copy(tail_atom,tail_src,tail_dst)
            else:
                copy_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Int64,num_bits_per_copy=128)
                tail_atom=cute.make_copy_atom(cpasync.CopyG2SOp(),cutlass.Int64,num_bits_per_copy=64)
                for group in cutlass.range_constexpr(self.cache_items//2):
                    offset=group*(self.bank_threads*2)+local*2
                    if offset<bank_count:
                        src=cute.make_tensor((candidates.iterator+cutlass.Int64(row)*candidates.stride[0]+8+bank*span+offset).align(16),cute.make_layout(2))
                        dst=cute.make_tensor((cached.iterator+bank*self.bank_cache+offset).align(16),cute.make_layout(2))
                        if offset+1<bank_count:
                            cute.copy(copy_atom,src,dst)
                        else:
                            tail_src=cute.make_tensor(src.iterator,cute.make_layout(1))
                            tail_dst=cute.make_tensor(dst.iterator,cute.make_layout(1))
                            cute.copy(tail_atom,tail_src,tail_dst)
            cute.arch.cp_async_commit_group()

            # K1 also proposes a quantile upper bound for this histogram only.
            # NaN keys sort before numerics in K1; preserve K3's ignore-NaN max.
            upper=cutlass.Float32(samples[row,samples.shape[1]-32+2])
            if upper!=upper:
                upper=cutlass.Float32(float('-inf'))
                for col in cutlass.range(tid,256,self.threads):
                    score=cutlass.Float32(samples[row,col])
                    if score==score:upper=max(upper,score)
                for step in cutlass.range_constexpr(5):
                    other=cute.arch.shuffle_sync_bfly(upper,1<<step)
                    upper=max(upper,other)
                if lane==0:hist[warp]=cutlass.Int32(bits32(upper))
                cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
                upper=cutlass.Float32(float('-inf'))
                if lane<self.threads//32:upper=float32(cutlass.Uint32(hist[lane]))
                for step in cutlass.range_constexpr(5):
                    other=cute.arch.shuffle_sync_bfly(upper,1<<step)
                    upper=max(upper,other)
                cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
            if cutlass.const_expr(self.exhaustive):
                threshold_value, upper = candidate_bounds(
                    candidates, row, bank, local, bank_count, span, hist,
                    self.dtype, self.packed, self.threads, self.bank_threads, self.barrier_id)
            span_value=upper-threshold_value;inverse_span=cutlass.Float32(-1.0)
            if (span_value>0.0)&(span_value<float('inf')):
                # Bucket boundaries need only be monotone. Full score keys
                # still determine the exact crossing-bin and fallback result.
                # Keep normal reciprocal/multiply operands away from FTZ and
                # overflow; exceptional spans use division.
                if (span_value>=1.2037062152420224e-35)&(span_value<=8.307674973655724e34):
                    inverse_span=cutlass.Float32(self.bins-1)*cute.arch.rcp_approx(span_value)
                else:
                    inverse_span=cutlass.Float32(self.bins-1)/span_value
            cute.arch.cp_async_wait_group(0)
            if cutlass.const_expr(self.bins_per_thread==4):
                for j in cutlass.range_constexpr(4):hist[tid*4+j]=0
            else:
                hist[tid*2]=0;hist[tid*2+1]=0
            cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
            # Keep two 16-bit bucket IDs per register; indices stay in the shared candidate cache.
            prefix_meta=cute.make_rmem_tensor((self.cache_items//2 if self.ds_half else self.cache_items,),cutlass.Uint32)
            for j in cutlass.range_constexpr(self.cache_items):
                offset=j*self.bank_threads+local
                if cutlass.const_expr(self.ds_half):
                    if cutlass.const_expr(j%2==0):prefix_meta[j//2]=cutlass.Uint32(0)
                else:prefix_meta[j]=cutlass.Uint32(0)
                if offset<bank_count:
                    entry=cached[bank*self.bank_cache+offset]
                    bucket=numeric_bucket(candidate_score(entry,self.dtype,self.packed),threshold_value,inverse_span,self.bins)
                    if cutlass.const_expr(self.ds_half):
                        prefix_meta[j//2]=prefix_meta[j//2]|(cutlass.Uint32(bucket)<<((j%2)*16))
                    else:prefix_meta[j]=(cutlass.Uint32(bucket)<<20)|cutlass.Uint32(candidate_index(entry,self.packed))
                    cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='cta')
            entry_buffer=cute.make_rmem_tensor((4,),self.entry_type)
            for base in cutlass.range(self.bank_cache,bank_count,self.bank_threads*4):
                for j in cutlass.range_constexpr(4):
                    offset=base+j*self.bank_threads+local
                    entry_buffer[j]=self.entry_type(0)
                    if offset<bank_count:
                        if offset<self.bank_cache:entry_buffer[j]=cached[bank*self.bank_cache+offset]
                        else:entry_buffer[j]=candidates[row,8+bank*span+offset]
                for j in cutlass.range_constexpr(4):
                    offset=base+j*self.bank_threads+local
                    if offset<bank_count:
                        entry=entry_buffer[j]
                        key=score_key(candidate_score(entry,self.dtype,self.packed))
                        bucket=numeric_bucket(candidate_score(entry,self.dtype,self.packed),threshold_value,inverse_span,self.bins)
                        cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='cta')
            cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
            if cutlass.const_expr(self.bins_per_thread==4):
                if cutlass.const_expr(self.half_threads):
                    count=cutlass.Int32(0)
                    for j in cutlass.range_constexpr(self.bins_per_thread):
                        count+=hist[tid*self.bins_per_thread+j]
                    prefix=count
                    for step in cutlass.range_constexpr(5):
                        delta=1<<step
                        other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
                        if lane>=delta:prefix+=other
                    # Slots1312..1327 are disjoint from bucket counts, ck/ci and state.
                    # One CTA barrier publishes all sixteen warp totals and completes
                    # every original histogram read before in-place prefix writes.
                    if lane==31:hist[1312+warp]=prefix
                    cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
                    prior=cutlass.Int32(0)
                    if lane<warp:prior=hist[1312+lane]
                    prior=cute.arch.warp_redux_sync(prior,'add')
                    cumulative=prior+prefix
                    bin_prefix=cumulative-count
                    for j in cutlass.range_constexpr(self.bins_per_thread):
                        # Re-read after the scan barrier to avoid keeping four counts live.
                        # Each thread exclusively owns these four histogram slots.
                        bin_count=hist[tid*self.bins_per_thread+j]
                        hist[tid*self.bins_per_thread+j]=bin_prefix
                        next_prefix=bin_prefix+bin_count
                        if (next_prefix>=required)&(bin_prefix<required):
                            state[0]=tid*self.bins_per_thread+j;state[1]=bin_prefix;state[2]=bin_count
                            if cutlass.const_expr(self.qblock==2 and self.dtype==cutlass.BFloat16 and self.k==2048 and not self.exact_ties):
                                proof=False
                                if (bin_count>32)&(bin_count!=required-(bin_prefix)):
                                    proof=uniform_half_bucket(tid*self.bins_per_thread+j,threshold_value,inverse_span,self.bins,self.dtype)
                                state[4]=cutlass.Int32(proof)
                        bin_prefix=next_prefix
                else:
                    bin_counts=cute.make_rmem_tensor((self.bins_per_thread,),cutlass.Int32)
                    count=cutlass.Int32(0)
                    for j in cutlass.range_constexpr(self.bins_per_thread):
                        bin_counts[j]=hist[tid*self.bins_per_thread+j]
                        count+=bin_counts[j]
                    prefix=count
                    for step in cutlass.range_constexpr(5):
                        delta=1<<step
                        other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
                        if lane>=delta:prefix+=other
                    # Slots1312..1327 are disjoint from bucket counts, ck/ci and state.
                    # One CTA barrier publishes all sixteen warp totals and completes
                    # every original histogram read before in-place prefix writes.
                    if lane==31:hist[1312+warp]=prefix
                    cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
                    prior=cutlass.Int32(0)
                    if lane<warp:prior=hist[1312+lane]
                    prior=cute.arch.warp_redux_sync(prior,'add')
                    cumulative=prior+prefix
                    bin_prefix=cumulative-count
                    for j in cutlass.range_constexpr(self.bins_per_thread):
                        hist[tid*self.bins_per_thread+j]=bin_prefix
                        next_prefix=bin_prefix+bin_counts[j]
                        if (next_prefix>=required)&(bin_prefix<required):
                            state[0]=tid*self.bins_per_thread+j;state[1]=bin_prefix;state[2]=bin_counts[j]
                            if cutlass.const_expr(self.qblock==2 and self.dtype==cutlass.BFloat16 and self.k==2048 and not self.exact_ties):
                                proof=False
                                if (bin_counts[j]>32)&(bin_counts[j]!=required-(bin_prefix)):
                                    proof=uniform_half_bucket(tid*self.bins_per_thread+j,threshold_value,inverse_span,self.bins,self.dtype)
                                state[4]=cutlass.Int32(proof)
                        bin_prefix=next_prefix
            else:
                count0=hist[tid*2];count1=hist[tid*2+1]
                count=count0+count1;prefix=count
                for step in cutlass.range_constexpr(5):
                    delta=1<<step
                    other=cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
                    if lane>=delta:prefix+=other
                # Slots1312..1327 are disjoint from bucket counts, ck/ci and state.
                # One CTA barrier publishes all sixteen warp totals and completes
                # every original histogram read before in-place prefix writes.
                if lane==31:hist[1312+warp]=prefix
                cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
                prior=cutlass.Int32(0)
                if lane<warp:prior=hist[1312+lane]
                prior=cute.arch.warp_redux_sync(prior,'add')
                cumulative=prior+prefix
                hist[tid*2]=cumulative-count
                hist[tid*2+1]=cumulative-count1
                if (cumulative-count1>=required)&(cumulative-count<required):
                    state[0]=tid*2;state[1]=cumulative-count;state[2]=count0
                    if cutlass.const_expr(self.qblock==2 and self.dtype==cutlass.BFloat16 and self.k==2048 and not self.exact_ties):
                        proof=False
                        if (count0>32)&(count0!=required-(cumulative-count)):
                            proof=uniform_half_bucket(tid*2,threshold_value,inverse_span,self.bins,self.dtype)
                        state[4]=cutlass.Int32(proof)
                if (cumulative>=required)&(cumulative-count1<required):
                    state[0]=tid*2+1;state[1]=cumulative-count1;state[2]=count1
                    if cutlass.const_expr(self.qblock==2 and self.dtype==cutlass.BFloat16 and self.k==2048 and not self.exact_ties):
                        proof=False
                        if (count1>32)&(count1!=required-(cumulative-count1)):
                            proof=uniform_half_bucket(tid*2+1,threshold_value,inverse_span,self.bins,self.dtype)
                        state[4]=cutlass.Int32(proof)
            if tid==0:state[3]=0
            cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
            crossing=state[0];above=state[1];m=state[2]
            need=required-above;whole=m==need
            uniform_boundary=False
            if cutlass.const_expr(self.dtype==cutlass.BFloat16 and self.k==2048 and not self.exact_ties):
                if cutlass.const_expr(self.qblock==2):
                    uniform_boundary=state[4]!=0
                else:
                    if (m>32)&(not whole):
                        uniform_boundary=uniform_half_bucket(crossing,threshold_value,inverse_span,self.bins,self.dtype)
            if cutlass.const_expr(self.exact_ties):
                if (m>32)&(not whole):
                    uniform_boundary=uniform_half_bucket(crossing,threshold_value,inverse_span,self.bins,self.dtype)
                if (m>128)&(not whole)&(not uniform_boundary):
                    lo=cutlass.Uint32(0xFFFFFFFF);hi=cutlass.Uint32(0)
                    for j in cutlass.range_constexpr(self.cache_items):
                        offset=j*self.bank_threads+local
                        if offset<bank_count:
                            if cutlass.const_expr(self.ds_half):bucket=cutlass.Int32((prefix_meta[j//2]>>((j%2)*16))&cutlass.Uint32(65535))
                            else:bucket=cutlass.Int32(prefix_meta[j]>>20)
                            if bucket==crossing:
                                entry=cached[bank*self.bank_cache+offset]
                                key=score_key(candidate_score(entry,self.dtype,self.packed))
                                lo=min(lo,key);hi=max(hi,key)
                    for offset in cutlass.range(self.bank_cache+local,bank_count,self.bank_threads):
                        entry=candidates[row,8+bank*span+offset]
                        score=candidate_score(entry,self.dtype,self.packed)
                        if numeric_bucket(score,threshold_value,inverse_span,self.bins)==crossing:
                            key=score_key(score);lo=min(lo,key);hi=max(hi,key)
                    lo=cute.arch.warp_redux_sync(lo,'umin');hi=cute.arch.warp_redux_sync(hi,'umax')
                    if lane==0:
                        hist[1344+warp]=cutlass.Int32(lo);hist[1376+warp]=cutlass.Int32(hi)
                    cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
                    lo=cutlass.Uint32(0xFFFFFFFF);hi=cutlass.Uint32(0)
                    if lane<self.threads//32:
                        lo=cutlass.Uint32(hist[1344+lane]);hi=cutlass.Uint32(hist[1376+lane])
                    lo=cute.arch.warp_redux_sync(lo,'umin');hi=cute.arch.warp_redux_sync(hi,'umax')
                    uniform_boundary=lo==hi
            if (m<=128)|whole|uniform_boundary:
                for i in cutlass.range(required+tid,self.k,self.threads):out[row,i]=-1
                for j in cutlass.range_constexpr(self.cache_items):
                    offset=j*self.bank_threads+local
                    if offset<bank_count:
                        if cutlass.const_expr(self.ds_half):
                            meta_entry=cached[bank*self.bank_cache+offset]
                            index=candidate_index(meta_entry,self.packed)
                            bucket=cutlass.Int32((prefix_meta[j//2]>>((j%2)*16))&cutlass.Uint32(65535))
                        else:
                            meta=prefix_meta[j]
                            index=cutlass.Int32(meta&cutlass.Uint32(0xFFFFF))
                            bucket=cutlass.Int32(meta>>20)
                        if (bucket<crossing)|(whole&(bucket==crossing)):
                            pos=cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='cta')
                            staged[pos]=index
                        elif bucket==crossing:
                            entry=cached[bank*self.bank_cache+offset]
                            key=score_key(candidate_score(entry,self.dtype,self.packed))
                            pos=cute.arch.atomic_add((state.iterator+3).llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='cta')
                            if uniform_boundary:
                                if pos<need:staged[above+pos]=index
                            else:
                                ck[pos]=key;ci[pos]=index
                entry_buffer=cute.make_rmem_tensor((4,),self.entry_type)
                for base in cutlass.range(self.bank_cache,bank_count,self.bank_threads*4):
                    for j in cutlass.range_constexpr(4):
                        offset=base+j*self.bank_threads+local
                        entry_buffer[j]=self.entry_type(0)
                        if offset<bank_count:
                            if offset<self.bank_cache:entry_buffer[j]=cached[bank*self.bank_cache+offset]
                            else:entry_buffer[j]=candidates[row,8+bank*span+offset]
                    for j in cutlass.range_constexpr(4):
                        offset=base+j*self.bank_threads+local
                        if offset<bank_count:
                            entry=entry_buffer[j]
                            key=score_key(candidate_score(entry,self.dtype,self.packed))
                            bucket=numeric_bucket(candidate_score(entry,self.dtype,self.packed),threshold_value,inverse_span,self.bins)
                            if (bucket<crossing)|(whole&(bucket==crossing)):
                                pos=cute.arch.atomic_add((hist.iterator+bucket).llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='cta')
                                staged[pos]=candidate_index(entry,self.packed)
                            elif bucket==crossing:
                                pos=cute.arch.atomic_add((state.iterator+3).llvm_ptr,cutlass.Int32(1),sem='relaxed',scope='cta')
                                if uniform_boundary:
                                    if pos<need:staged[above+pos]=candidate_index(entry,self.packed)
                                else:
                                    ck[pos]=key;ci[pos]=candidate_index(entry,self.packed)
                cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
                if (not whole)&(not uniform_boundary):
                    if cutlass.const_expr(self.dtype==cutlass.BFloat16):
                        refine_quantized(ck,ci,m,need,above,tid,staged,self.threads)
                    else:
                        # One warp ranks one candidate using parallel comparisons.
                        # Original ck order breaks ties, so ranks remain unique.
                        for item in cutlass.range(warp,m,self.threads//32):
                            key=cutlass.Uint32(ck[item])
                            rank=cutlass.Int32(0)
                            for base in cutlass.range(0,m,32):
                                j=base+lane
                                other_key=cutlass.Uint32(0xFFFFFFFF)
                                if j<m:other_key=cutlass.Uint32(ck[j])
                                less=(j<m)&((other_key<key)|((other_key==key)&(j<item)))
                                rank+=cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(less)))
                            if lane==0 and rank<need:staged[above+rank]=ci[item]
                cute.arch.barrier(barrier_id=self.barrier_id,number_of_threads=self.threads)
                write_atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),cutlass.Int32,num_bits_per_copy=128)
                for col in cutlass.range(tid*4,required,self.threads*4):
                    if col+3<required:
                        src=cute.make_tensor((staged.iterator+col).align(16),cute.make_layout(4))
                        dst=cute.make_tensor((out.iterator+cutlass.Int64(row)*out.stride[0]+col).align(16),cute.make_layout(4))
                        cute.copy(write_atom,src,dst)
                    else:
                        for j in cutlass.range_constexpr(4):
                            if col+j<required:out[row,col+j]=staged[col+j]
            else:
                select_stream(row,tid,candidates,n,required,hist,out,self.k,self.dtype,self.threads,self.barrier_id, banks=4)


@functools.cache
def compile_selector(k,dtype,shift,signature,qblock=2,precounted=False,wide_cache=False,compact=False,pro_short=False,exact_ties=False,glm_short=False,glm_mid_bins=False, pdl=False, exhaustive=False, candidate_cols=0):
    tensors=[]
    for i,(d,ndim) in enumerate(signature):
        dims=tuple(cute.sym_int() for _ in range(ndim))
        if i==0:
            t=cute.runtime.make_fake_tensor(DTYPES[d],dims,stride=(cute.sym_int64(),1),assumed_align=16)
        else:
            t=cute.runtime.make_fake_compact_tensor(DTYPES[d],dims,stride_order=tuple(reversed(range(ndim))),assumed_align=16)
        tensors.append(t)
    return cute.compile(Crossing(k,dtype,shift,qblock,precounted,wide_cache,compact,DTYPES[signature[0][0]]==cutlass.Int32,pro_short,exact_ties,glm_short,glm_mid_bins, pdl=pdl, exhaustive=exhaustive, candidate_cols=candidate_cols),*tensors,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi')
