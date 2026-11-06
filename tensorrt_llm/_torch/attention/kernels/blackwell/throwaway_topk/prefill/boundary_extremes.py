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

"""Exact single-item or single-exclusion selection in a small crossing bin."""
import cutlass
import cutlass.cute as cute

@cute.jit
def select_extreme(ck, ci, staged, m, need, above):
    # One complete warp participates; arbitrary equal-score indices are valid.
    lane=cute.arch.lane_idx()
    keep_one=need==1
    selected=cutlass.Uint32(0xFFFFFFFF) if keep_one else cutlass.Uint32(0)
    for j in cutlass.range_constexpr(4):
        i=lane+32*j
        if i<m:
            key=cutlass.Uint32(ck[i])
            selected=min(selected,key) if keep_one else max(selected,key)
    if keep_one:selected=cute.arch.warp_redux_sync(selected,'umin')
    else:selected=cute.arch.warp_redux_sync(selected,'umax')
    ordinal=cutlass.Uint32(0xFFFFFFFF)
    for j in cutlass.range_constexpr(4):
        i=lane+32*j
        if i<m:
            if cutlass.Uint32(ck[i])==selected:ordinal=min(ordinal,cutlass.Uint32(i))
    ordinal=cute.arch.warp_redux_sync(ordinal,'umin')
    if keep_one:
        if lane==0:staged[above]=ci[cutlass.Int32(ordinal)]
    else:
        for j in cutlass.range_constexpr(4):
            i=lane+32*j
            if (i<m)&(i!=cutlass.Int32(ordinal)):
                staged[above+i-cutlass.Int32(i>cutlass.Int32(ordinal))]=ci[i]
