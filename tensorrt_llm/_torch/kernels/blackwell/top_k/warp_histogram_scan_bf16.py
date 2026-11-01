# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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
"""Single-warp crossing-bin scans for the small (256/512-bin) BF16 register-kernel histograms."""

import cutlass
import cutlass.cute as cute

RES_B = 0
RES_M = 1
RES_ABOVE = 2
RES_TOT = 3


@cute.jit
def scan_cross_small_warp_u32(
    histogram,
    target,
    thread_index,
    result,
    nb: cutlass.Constexpr,
    vector: cutlass.Constexpr = True,
):
    """Find the bin that holds the target-th largest element and turn the bins into emit cursors.

    Warp 0 handles `nb // 32` bins per lane. Each bin is replaced in place by the count of the higher bins;
    `result` receives the crossing bin (`RES_B`), its count (`RES_M`), the count above it (`RES_ABOVE`) and
    the total (`RES_TOT`).
    """
    assert nb in (256, 512)
    if thread_index < cutlass.Int32(32):
        lane = cutlass.Int32(thread_index)
        bins_per_lane = nb // 32
        bin_base = lane * cutlass.Int32(bins_per_lane)
        target_count = cutlass.Uint32(target)
        counts = cute.make_rmem_tensor((bins_per_lane,), cutlass.Uint32)
        if cutlass.const_expr(vector):
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
            )
            histogram_span = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    histogram.iterator.toint() + bin_base * cutlass.Int32(4),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout((bins_per_lane,)),
            )
            cute.copy(copy_atom, histogram_span, counts)
        else:
            for local_bin in cutlass.range_constexpr(bins_per_lane):
                counts[local_bin] = cutlass.Uint32(histogram[bin_base + cutlass.Int32(local_bin)])
        lane_mass = cutlass.Uint32(0)
        for local_bin in cutlass.range_constexpr(bins_per_lane):
            lane_mass = lane_mass + cutlass.Uint32(counts[local_bin])
        lane_suffix = lane_mass
        for scan_step in cutlass.range_constexpr(5):
            scan_offset = 1 << scan_step
            higher_mass = cutlass.Uint32(
                cute.arch.shuffle_sync_down(lane_suffix, scan_offset, mask_and_clamp=31)
            )
            if lane + cutlass.Int32(scan_offset) < cutlass.Int32(32):
                lane_suffix = lane_suffix + higher_mass
        total_mass = cutlass.Uint32(cute.arch.shuffle_sync(lane_suffix, 0))
        if lane == cutlass.Int32(0):
            result[RES_TOT] = total_mass
        above = lane_suffix - lane_mass
        for local_bin in cutlass.range_constexpr(bins_per_lane - 1, -1, -1):
            bin_index = bin_base + cutlass.Int32(local_bin)
            mass = cutlass.Uint32(counts[local_bin])
            through = above + mass
            if cutlass.const_expr(vector):
                counts[local_bin] = above
            else:
                histogram[bin_index] = above
            if above < target_count:
                if through >= target_count or bin_index == cutlass.Int32(0):
                    result[RES_B] = cutlass.Uint32(bin_index)
                    result[RES_M] = mass
                    result[RES_ABOVE] = above
            above = through
        if cutlass.const_expr(vector):
            cute.copy(copy_atom, counts, histogram_span)


@cute.jit
def scan_cross_small_warp_parts_u32(
    histogram,
    target,
    thread,
    result,
    part,
    nb: cutlass.Constexpr,
    parts: cutlass.Constexpr,
    vector: cutlass.Constexpr = True,
):
    """`scan_cross_small_warp_u32` over `parts` per-partition copies of the histogram (partitioned emit).

    The crossing search uses the summed counts. The first copy receives the cursors of partition `part`:
    the count above the bin plus the bin's count in the earlier partitions.
    """
    assert nb in (256, 512) and parts in (1, 2, 4)
    if thread < cutlass.Int32(32):
        lane = cutlass.Int32(thread)
        lane_bins = nb // 32
        bin_base = lane * cutlass.Int32(lane_bins)
        target_count = cutlass.Uint32(target)
        counts = cute.make_rmem_tensor((lane_bins,), cutlass.Uint32)
        prefix = cute.make_rmem_tensor((lane_bins,), cutlass.Uint32)
        values = cute.make_rmem_tensor((lane_bins,), cutlass.Uint32)
        for local in cutlass.range_constexpr(lane_bins):
            counts[local] = cutlass.Uint32(0)
            prefix[local] = cutlass.Uint32(0)
        if cutlass.const_expr(vector):
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
            )
        for source in cutlass.range_constexpr(parts):
            offset = bin_base + cutlass.Int32(source * nb)
            if cutlass.const_expr(vector):
                span = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        histogram.iterator.toint() + offset * cutlass.Int32(4),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((lane_bins,)),
                )
                cute.copy(copy_atom, span, values)
            else:
                for local in cutlass.range_constexpr(lane_bins):
                    values[local] = cutlass.Uint32(histogram[offset + cutlass.Int32(local)])
            for local in cutlass.range_constexpr(lane_bins):
                mass = cutlass.Uint32(values[local])
                counts[local] = cutlass.Uint32(counts[local]) + mass
                if cutlass.Int32(source) < cutlass.Int32(part):
                    prefix[local] = cutlass.Uint32(prefix[local]) + mass
        lane_mass = cutlass.Uint32(0)
        for local in cutlass.range_constexpr(lane_bins):
            lane_mass = lane_mass + cutlass.Uint32(counts[local])
        suffix = lane_mass
        for scan_step in cutlass.range_constexpr(5):
            scan_offset = 1 << scan_step
            higher = cutlass.Uint32(
                cute.arch.shuffle_sync_down(suffix, scan_offset, mask_and_clamp=31)
            )
            if lane + cutlass.Int32(scan_offset) < cutlass.Int32(32):
                suffix = suffix + higher
        total_mass = cutlass.Uint32(cute.arch.shuffle_sync(suffix, 0))
        if lane == cutlass.Int32(0):
            result[RES_TOT] = total_mass
        above = suffix - lane_mass
        for local in cutlass.range_constexpr(lane_bins - 1, -1, -1):
            bin_index = bin_base + cutlass.Int32(local)
            mass = cutlass.Uint32(counts[local])
            through = above + mass
            cursor = above + cutlass.Uint32(prefix[local])
            if cutlass.const_expr(vector):
                counts[local] = cursor
            else:
                histogram[bin_index] = cursor
            if above < target_count:
                if through >= target_count or bin_index == cutlass.Int32(0):
                    result[RES_B] = cutlass.Uint32(bin_index)
                    result[RES_M] = mass
                    result[RES_ABOVE] = above
            above = through
        if cutlass.const_expr(vector):
            span = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    histogram.iterator.toint() + bin_base * cutlass.Int32(4),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout((lane_bins,)),
            )
            cute.copy(copy_atom, counts, span)
