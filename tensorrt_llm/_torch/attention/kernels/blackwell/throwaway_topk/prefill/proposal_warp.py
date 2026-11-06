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

"""H4 proposal: independent warp histograms, preserving the K1 quantile policy."""
import cutlass
import cutlass.cute as cute
from .selection import score_key, key_score


@cute.jit
def propose_warp(row, lane, scores, hist, thresholds, samples, rank, high_rank, info, extend_short:cutlass.Constexpr=False,score_row=None):
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
    for part in cutlass.range_constexpr(8):
        hist[lane + part * 32] = 0
    cute.arch.sync_warp()
    for base in cutlass.range(0,samples,128):
        values=cute.make_tensor((scores.iterator+cutlass.Int64(score_row)*scores.stride[0]+base+lane*4).align(4*scores.element_type.width//8),cute.make_layout(4)).load()
        for j in cutlass.range_constexpr(4):
            value = cutlass.Float32(values[j])
            key = score_key(value)
            bucket = min(255, cutlass.Int32((key - minimum) >> shift))
            if scale > 0:
                bucket = cutlass.Int32(min(cutlass.Float32(255), max(cutlass.Float32(0), (high - value) * scale)))
            if (value == float('-inf')) | (value != value):
                bucket = cutlass.Int32(255)
            if bucket < 256:
                cute.arch.atomic_add((hist.iterator + bucket).llvm_ptr, cutlass.Int32(1), sem='relaxed', scope='cta')
    cute.arch.sync_warp()
    # Each lane owns eight adjacent bins. One warp scan supplies the
    # prefix of these groups; each lane then scans its eight bins locally.
    local_counts = cute.make_rmem_tensor((8,),cutlass.Int32)
    total = cutlass.Int32(0)
    for part in cutlass.range_constexpr(8):
        local_counts[part] = hist[lane*8+part]
        total += local_counts[part]
    prefix = total
    for step in cutlass.range_constexpr(5):
        delta = 1 << step
        other = cute.arch.shuffle_sync_up(prefix,delta,mask=-1,mask_and_clamp=0)
        if lane >= delta:
            prefix += other
    cumulative = prefix-total
    chosen = cutlass.Int32(0)
    chosen_high = cutlass.Int32(0)
    for part in cutlass.range_constexpr(8):
        count = local_counts[part]
        cumulative += count
        if (cumulative >= rank) & (cumulative-count < rank):
            chosen = lane*8+part
        if (cumulative >= high_rank) & (cumulative-count < high_rank):
            chosen_high = lane*8+part
    chosen = cutlass.Int32(cute.arch.warp_redux_sync(cutlass.Uint32(chosen), 'umax'))
    chosen_high = cutlass.Int32(cute.arch.warp_redux_sync(cutlass.Uint32(chosen_high), 'umax'))
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
