# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm

"""
block prefix sum kernel (input is loading from shared memory) in CuTe DSL.
The parallel strategy is one thread process one element from shared memory.
"""


@cute.jit
def fence_acq_rel_cta(*, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="membar.cta;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def warp_scan(val: cutlass.Int32, tidx, lane_id, num_threads_per_warp: cutlass.Constexpr):
    """Warp scan kernel"""
    mask_val = cutlass.const_expr(((1 << num_threads_per_warp) - 1) & 0xFFFFFFFF)
    mask_and_clamp_val = 0
    iteration = cute.arch.log2_of_pow2_int(cutlass.Int32(num_threads_per_warp))
    for i in cutlass.range(iteration, unroll_full=True):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(
            val, offset, mask=mask_val, mask_and_clamp=mask_and_clamp_val
        )
        if lane_id >= offset:
            val = val + other
    return val


@cute.jit
def block_prefix_sum_kernel(
    val: cutlass.Int32,
    warp_sums: cute.Tensor,
    tidx,
    num_threads,
    num_warps,
    barrier_id=1,
    need_total_sum=False,
):
    """Block prefix sum kernel in CuTe DSL"""
    # Thread and warp id
    warp_id = tidx // 32
    lane_id = tidx % 32

    # Currently, we only support num_warps > 1, will support num_warps <= 1 logic later.
    assert num_threads % 32 == 0, "num_threads must be divisible by 32, but got {}".format(
        num_threads
    )
    assert num_warps > 1, "num_warps must be > 1, but got {}".format(num_warps)
    assert num_warps == 2 ** int(math.log2(num_warps)), "num_warps must be a power of 2"

    # Step 1: Warp-level prefix sum using shuffle
    val = warp_scan(val, tidx, lane_id, num_threads_per_warp=32)

    # Step 2: Store warp prefix sums
    if lane_id == 31:  # Last thread in warp stores warp sum
        warp_sums[warp_id] = val
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 3: Prefix sum across warps
    if warp_id == 0:
        if lane_id < num_warps:
            warp_val = warp_sums[lane_id]
            # call warp-level prefix sum
            warp_val = warp_scan(warp_val, tidx, lane_id, num_threads_per_warp=num_warps)
            warp_sums[lane_id] = warp_val
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # Step 4: Add warp-level prefix
    if warp_id > 0:
        val = val + warp_sums[warp_id - 1]

    # Step 5: Get total sum if need_total_sum is True
    total_sum = 0
    if need_total_sum:
        total_sum = warp_sums[num_warps - 1]

    return val, total_sum
