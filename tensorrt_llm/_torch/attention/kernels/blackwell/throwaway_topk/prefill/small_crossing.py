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

"""Exact small-boundary selection; ascending score keys, slot-order ties.

Use argmax for one selected entry and complement selection for small exclusions.
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def refine(ck, ci, m, need, above, tid, out, threads: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    warp = tid // 32
    if need == 1:
        if warp == 0:
            best = cutlass.Uint32(0xFFFFFFFF)
            slot = cutlass.Int32(0x7FFFFFFF)
            for j in cutlass.range(lane, m, 32):
                key = cutlass.Uint32(ck[j])
                if (key < best) | ((key == best) & (j < slot)):
                    best = key
                    slot = j
            for step in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_bfly(best, 1 << step)
                other_slot = cute.arch.shuffle_sync_bfly(slot, 1 << step)
                if (other < best) | ((other == best) & (other_slot < slot)):
                    best = other
                    slot = other_slot
            if lane == 0:
                out[above] = ci[slot]
    elif (m <= 64) & (m - need <= 4):
        if warp == 0:
            live0 = lane < m
            live1 = lane + 32 < m
            key0 = cutlass.Uint32(0)
            key1 = cutlass.Uint32(0)
            if live0:
                key0 = cutlass.Uint32(ck[lane])
            if live1:
                key1 = cutlass.Uint32(ck[lane + 32])
            for removed in cutlass.range(m - need):
                worst = cutlass.Uint32(0)
                slot = cutlass.Int32(-1)
                if live0:
                    worst = key0
                    slot = lane
                if live1 & ((key1 > worst) | ((key1 == worst) & (lane + 32 > slot))):
                    worst = key1
                    slot = lane + 32
                for step in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_bfly(worst, 1 << step)
                    other_slot = cute.arch.shuffle_sync_bfly(slot, 1 << step)
                    if (other > worst) | ((other == worst) & (other_slot > slot)):
                        worst = other
                        slot = other_slot
                live0 = live0 & (lane != slot)
                live1 = live1 & (lane + 32 != slot)
            mask0 = cute.arch.vote_ballot_sync(live0)
            mask1 = cute.arch.vote_ballot_sync(live1)
            lower = (cutlass.Uint32(1) << lane) - 1
            if live0:
                out[above + cutlass.Int32(cute.arch.popc(mask0 & lower))] = ci[lane]
            if live1:
                out[
                    above
                    + cutlass.Int32(cute.arch.popc(mask0))
                    + cutlass.Int32(cute.arch.popc(mask1 & lower))
                ] = ci[lane + 32]
    elif m <= 32:
        voter = cutlass.Uint32(0xFFFFFFFF)
        if lane < m:
            voter = cutlass.Uint32(ck[lane])
        for item in cutlass.range(warp, m, threads // 32):
            target = cute.arch.shuffle_sync(voter, item)
            less = (lane < m) & ((voter < target) | ((voter == target) & (lane < item)))
            rank = cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(less)))
            if (lane == 0) & (rank < need):
                out[above + rank] = ci[item]
    else:
        for item in cutlass.range(warp, m, threads // 32):
            key = cutlass.Uint32(ck[item])
            rank = cutlass.Int32(0)
            for base in cutlass.range(0, m, 32):
                j = base + lane
                other = cutlass.Uint32(0xFFFFFFFF)
                if j < m:
                    other = cutlass.Uint32(ck[j])
                less = (j < m) & ((other < key) | ((other == key) & (j < item)))
                rank += cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(less)))
            if (lane == 0) & (rank < need):
                out[above + rank] = ci[item]


@cute.jit
def refine_quantized(ck, ci, m, need, above, tid, out, threads: cutlass.Constexpr):
    # Coarsely quantized logits often fill a boundary bin with one score.
    # Each warp proves equality independently; no additional CTA barrier is
    # needed after the caller's boundary-candidate publication barrier.
    lane = cute.arch.lane_idx()
    first = cutlass.Uint32(ck[0])
    same = True
    for base in cutlass.range(0, m, 32):
        j = base + lane
        if j < m:
            same = same & (cutlass.Uint32(ck[j]) == first)
    uniform = cute.arch.vote_ballot_sync(same) == cutlass.Uint32(0xFFFFFFFF)
    if uniform:
        for j in cutlass.range(tid, need, threads):
            out[above + j] = ci[j]
    else:
        refine(ck, ci, m, need, above, tid, out, threads)
