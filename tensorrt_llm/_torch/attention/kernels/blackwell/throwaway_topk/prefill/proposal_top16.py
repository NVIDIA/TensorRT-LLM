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

"""Exact top16 of 128 samples; disjoint merge scratch avoids SMEM races."""
import cutlass
import cutlass.cute as cute
from .subset_barrier import subset_barrier
from .selection import score_key, key_score

@cute.jit
def propose_top16(row,tid,scores,hist,thresholds,rank,high_rank,info,barrier,score_row=None):
    if cutlass.const_expr(score_row is None):score_row=row
    lane=cute.arch.lane_idx();warp=tid//32
    value=cutlass.Int32(score_key(cutlass.Float32(scores[score_row,tid]))^cutlass.Uint32(0x80000000))
    # Each warp sorts its 32 keys. The best 16 suffice for the global top16.
    for level in cutlass.range_constexpr(1,6):
        for step in cutlass.range_constexpr(level-1,-1,-1):
            other=cute.arch.shuffle_sync_bfly(value,1<<step)
            take_min=((lane&(1<<level))==0)==((lane&(1<<step))==0)
            updated=max(value,other)
            if take_min:updated=min(value,other)
            value=updated
    if lane<16:hist[warp*16+lane]=value
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    if warp<2:
        source=lane
        if lane>=16:source=47-lane
        value=hist[warp*32+source]
        for step in cutlass.range_constexpr(4,-1,-1):
            other=cute.arch.shuffle_sync_bfly(value,1<<step)
            updated=max(value,other)
            if (lane&(1<<step))==0:updated=min(value,other)
            value=updated
        # Do not overwrite keys that the other warp may still be loading.
        if lane<16:hist[64+warp*16+lane]=value
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    if warp==0:
        source=lane
        if lane>=16:source=47-lane
        value=hist[64+source]
        for step in cutlass.range_constexpr(4,-1,-1):
            other=cute.arch.shuffle_sync_bfly(value,1<<step)
            updated=max(value,other)
            if (lane&(1<<step))==0:updated=min(value,other)
            value=updated
        kth=cute.arch.shuffle_sync(value,rank-1)
        next_kth=cute.arch.shuffle_sync(value,min(rank,15))
        hith=cute.arch.shuffle_sync(value,high_rank-1)
        if lane==0:
            high=key_score(cutlass.Uint32(value)^cutlass.Uint32(0x80000000))
            threshold=key_score(cutlass.Uint32(kth)^cutlass.Uint32(0x80000000))
            neighbor=key_score(cutlass.Uint32(next_kth)^cutlass.Uint32(0x80000000))
            # Two half-products avoid overflow on finite same-sign values.
            # Nonfinite results retain the existing conservative K2/K4 rules.
            threshold=threshold*0.5+neighbor*0.5
            hist_high=key_score(cutlass.Uint32(hith)^cutlass.Uint32(0x80000000))
            thresholds[row]=threshold
            info[row,256]=info.element_type(high)
            info[row,257]=info.element_type(threshold)
            info[row,258]=info.element_type(hist_high)


@cute.jit
def propose_top16_256(row,tid,scores,hist,thresholds,rank,high_rank,info,barrier,score_row=None):
    if cutlass.const_expr(score_row is None):score_row=row
    lane=cute.arch.lane_idx();warp=tid//32
    initial=cute.make_rmem_tensor((2,),cutlass.Int32)
    for j in cutlass.range_constexpr(2):
        initial[j]=cutlass.Int32(score_key(cutlass.Float32(scores[score_row,tid+128*j]))^cutlass.Uint32(0x80000000))
    values=initial.load()
    for level in cutlass.range_constexpr(1,7):
        for step in cutlass.range_constexpr(level-1,-1,-1):
            sort_next=cute.make_rmem_tensor((2,),cutlass.Int32)
            for j in cutlass.range_constexpr(2):
                value=values[j]
                if cutlass.const_expr(step<5):other=cute.arch.shuffle_sync_bfly(value,1<<step)
                else:other=values[1-j]
                index=lane+32*j
                take_min=((index&(1<<level))==0)==((index&(1<<step))==0)
                result=max(value,other)
                if take_min:result=min(value,other)
                sort_next[j]=result
            values=sort_next.load()
    value=values[0]
    if lane<16:hist[warp*16+lane]=value
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    if warp<2:
        source=lane
        if lane>=16:source=47-lane
        value=hist[warp*32+source]
        for step in cutlass.range_constexpr(4,-1,-1):
            other=cute.arch.shuffle_sync_bfly(value,1<<step)
            updated=max(value,other)
            if (lane&(1<<step))==0:updated=min(value,other)
            value=updated
        # Do not overwrite keys that the other warp may still be loading.
        if lane<16:hist[64+warp*16+lane]=value
    subset_barrier(barrier_id=barrier,number_of_threads=128)
    if warp==0:
        source=lane
        if lane>=16:source=47-lane
        value=hist[64+source]
        for step in cutlass.range_constexpr(4,-1,-1):
            other=cute.arch.shuffle_sync_bfly(value,1<<step)
            updated=max(value,other)
            if (lane&(1<<step))==0:updated=min(value,other)
            value=updated
        kth=cute.arch.shuffle_sync(value,rank-1)
        next_kth=cute.arch.shuffle_sync(value,min(rank,15))
        hith=cute.arch.shuffle_sync(value,high_rank-1)
        if lane==0:
            high=key_score(cutlass.Uint32(value)^cutlass.Uint32(0x80000000))
            threshold=key_score(cutlass.Uint32(kth)^cutlass.Uint32(0x80000000))
            neighbor=key_score(cutlass.Uint32(next_kth)^cutlass.Uint32(0x80000000))
            # Two half-products avoid overflow on finite same-sign values.
            # Nonfinite results retain the existing conservative K2/K4 rules.
            threshold=threshold*0.5+neighbor*0.5
            hist_high=key_score(cutlass.Uint32(hith)^cutlass.Uint32(0x80000000))
            thresholds[row]=threshold
            info[row,256]=info.element_type(high)
            info[row,257]=info.element_type(threshold)
            info[row,258]=info.element_type(hist_high)
