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

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Exact proposal bins for a compile-time sample count of 128.

Use a 256-bin mapping with the model rank policy. Warp reductions find the
same quantile bins without shared histogram initialization and atomics.
The caller must specialize this helper to exactly 128 scores per row.
"""
import cutlass
import cutlass.cute as cute
from .selection import score_key, key_score


@cute.jit
def propose_warp_128(row, lane, scores, hist, thresholds, samples, rank, high_rank, info, extend_short:cutlass.Constexpr=False,score_row=None):
    if cutlass.const_expr(score_row is None):score_row=row
    minimum = cutlass.Uint32(0xFFFFFFFF)
    maximum = cutlass.Uint32(0)
    for base in cutlass.range(0,samples,128):
        values=cute.make_tensor((scores.iterator+cutlass.Int64(score_row)*scores.stride[0]+base+lane*4).align(4*scores.element_type.width//8),cute.make_layout(4)).load()
        for j in cutlass.range_constexpr(4):
            value = cutlass.Float32(values[j])
            key = score_key(value)
            if value != float('-inf'):
                minimum = min(minimum, key)
                maximum = max(maximum, key)
    minimum = cute.arch.warp_redux_sync(minimum, 'umin')
    maximum = cute.arch.warp_redux_sync(maximum, 'umax')
    if minimum > maximum:
        minimum = score_key(cutlass.Float32(float('-inf')))
        maximum = minimum
    span = maximum - minimum
    low = key_score(maximum)
    high = key_score(minimum)
    scale = cutlass.Float32(0)
    if (high > low) & (high < float('inf')) & (low > float('-inf')):
        scale = cutlass.Float32(255.0) / (high - low)
    shift = cutlass.Int32(0)
    if span > 0:
        shift = max(0, 24 - cutlass.Int32(cute.arch.clz(span)))
    chosen=cutlass.Int32(0);chosen_high=cutlass.Int32(0)
    # Same 8-bit bins; exact rank selection without shared histogram.
    buckets=cute.make_rmem_tensor((4,),cutlass.Int32)
    values=cute.make_tensor((scores.iterator+cutlass.Int64(score_row)*scores.stride[0]+lane*4).align(4*scores.element_type.width//8),cute.make_layout(4)).load()
    for j in cutlass.range_constexpr(4):
        value=cutlass.Float32(values[j]);key=score_key(value)
        bucket=min(255,cutlass.Int32((key-minimum)>>shift))
        if scale>0:
            bucket=cutlass.Int32(min(cutlass.Float32(255),max(cutlass.Float32(0),(high-value)*scale)))
        if (value==float('-inf')) | (value!=value):bucket=cutlass.Int32(255)
        buckets[j]=bucket
    lower_bin=cutlass.Int32(0);upper_bin=cutlass.Int32(255)
    for step in cutlass.range_constexpr(8):
        middle=(lower_bin+upper_bin)>>1
        count=cutlass.Int32(0)
        for j in cutlass.range_constexpr(4):count+=cutlass.Int32(buckets[j]<=middle)
        count=cutlass.Int32(cute.arch.warp_redux_sync(cutlass.Uint32(count),'add'))
        if count>=rank:upper_bin=middle
        else:lower_bin=middle+1
    chosen=lower_bin
    chosen_high=cutlass.Int32(0)
    if high_rank>1:
        lower_high=cutlass.Int32(0);upper_high=cutlass.Int32(255)
        for step in cutlass.range_constexpr(8):
            middle=(lower_high+upper_high)>>1
            count=cutlass.Int32(0)
            for j in cutlass.range_constexpr(4):count+=cutlass.Int32(buckets[j]<=middle)
            count=cutlass.Int32(cute.arch.warp_redux_sync(cutlass.Uint32(count),'add'))
            if count>=high_rank:upper_high=middle
            else:lower_high=middle+1
        chosen_high=lower_high
    if lane == 0:
        edge = cutlass.Uint64(minimum) + ((cutlass.Uint64(chosen) + 1) << shift) - 1
        threshold = key_score(cutlass.Uint32(min(edge, cutlass.Uint64(maximum))))
        if scale > 0:
            threshold = max(low, high - (cutlass.Float32(chosen) + 1.0) / scale)
        thresholds[row] = threshold
        info[row, 256] = info.element_type(high)
        info[row, 257] = info.element_type(threshold)
        hist_high = high
        if (high_rank > 1) & (scale > 0):
            hist_high = max(low, high - cutlass.Float32(chosen_high) / scale)
        # A smaller sample maximum need not exceed the true kth score.
        # Extend the histogram domain only; filtering threshold stays unchanged.
        if (samples<=1024)&((samples>256)|extend_short)&(high<float('inf'))&(threshold>float('-inf'))&(high>threshold):
            hist_high=high+(high-threshold)
        info[row, 258] = info.element_type(hist_high)
