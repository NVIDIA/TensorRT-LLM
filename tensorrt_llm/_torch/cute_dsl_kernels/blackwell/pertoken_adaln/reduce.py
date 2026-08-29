# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
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
#
# Vendored from SGLang (Apache-2.0):
#   upstream repo: https://github.com/sgl-project/sglang
#   upstream path: python/sglang/kernels/ops/diffusion/cutedsl/common/reduce.py
#   pinned SHA:    e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7
#   upstream file sha256: 90b8a0ea9a857849799ae8c17e3306271b68156082fcc4c257b28a1d051e7e2e
# LOCAL MODIFICATIONS (NVIDIA, 2026): this header only. Code byte-identical.

import math

import cutlass
import cutlass.cute as cute


@cute.jit
def warp_reduce_sum(val: cute.Numeric, reduce_size: int = 32) -> cute.Numeric:
    iters = int(math.log2(reduce_size))
    for i in range(iters):
        val = val + cute.arch.shuffle_sync_down(val, offset=1 << (iters - i - 1))
    return val


@cute.jit
def cta_reduce_sum(
    val: cute.Numeric, num_warps: cutlass.Constexpr, tidx: cutlass.Int32
) -> cute.Numeric:
    smem = cutlass.utils.SmemAllocator()
    acc = smem.allocate_tensor(cutlass.Float32, num_warps + 1)
    warp_id = tidx >> 5
    lane_id = tidx & 31
    if lane_id == 0:
        acc[warp_id] = val
    cute.arch.sync_threads()
    if warp_id == 0:
        val = acc[lane_id] if lane_id < num_warps else cutlass.Float32(0)
        val = warp_reduce_sum(val)
        if lane_id == 0:
            acc[num_warps] = val
    cute.arch.sync_threads()
    val = acc[num_warps]
    return val
