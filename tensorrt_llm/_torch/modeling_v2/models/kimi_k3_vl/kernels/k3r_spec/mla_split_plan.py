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
"""Split-KV planner of the CuTe DSL MLA decode: its limits and the Triton planning kernel."""

import triton
import triton.language as tl

# Per-request split-KV plan for the CuTe DSL MLA decode kernels. The kernel grid
# carries the tactic's split_kv slots per request; the plan decides how many a
# request uses this step from the batch's KV lengths (the kernel clamps it to
# the grid and the slots beyond exit before their mainloop).
_SPLIT_PLAN_MAX_REQUESTS = 1024
# CuteDSLNVMlaDecodeBlackwellRunner._MAX_SPLIT_KV: the largest grid any tactic uses.
_SPLIT_PLAN_MAX_SPLIT = 32
# mma_qk_tiler N of every tactic: the KV tile a split is made of.
_SPLIT_PLAN_KV_TILE = 128
# Shortest piece worth its partial-output write and reduction read.
_SPLIT_PLAN_MIN_TILES = 4


@triton.jit(
    do_not_specialize=["num_requests", "rows_per_request", "wave_clusters", "max_split"],
    do_not_specialize_on_alignment=[
        "num_requests",
        "rows_per_request",
        "wave_clusters",
        "max_split",
    ],
)
def _mla_split_kv_plan_kernel(
    cache_seqs_ptr,
    plan_ptr,
    num_requests,
    rows_per_request,
    wave_clusters,
    max_split,
    KV_TILE: tl.constexpr,
    MIN_TILES: tl.constexpr,
    PIECE_COST: tl.constexpr,
    MAX_PIECES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_requests
    kv_len = tl.load(cache_seqs_ptr + offsets, mask=mask, other=0)
    tiles = tl.maximum((kv_len + KV_TILE - 1) // KV_TILE, 1)
    tiles = tl.where(mask, tiles, 0)
    longest = tl.max(tiles, axis=0)
    # Cut the longest request into k equal pieces (every request then takes
    # ceil(tiles / piece) pieces) and keep the k whose wave-quantized makespan,
    # waves x (piece + fixed piece cost), is smallest; ties go to fewer pieces.
    best_piece = longest
    best_cost = tl.full([], 2**31 - 1, tl.int32)
    for pieces in tl.static_range(1, MAX_PIECES + 1):
        piece = tl.maximum((longest + pieces - 1) // pieces, MIN_TILES)
        splits = tl.minimum(tl.maximum((tiles + piece - 1) // piece, 1), max_split)
        clusters = tl.sum(tl.where(mask, splits, 0), axis=0) * rows_per_request
        waves = (clusters + wave_clusters - 1) // wave_clusters
        cost = waves * (piece + PIECE_COST)
        better = cost < best_cost
        best_cost = tl.where(better, cost, best_cost)
        best_piece = tl.where(better, piece, best_piece)
    plan = tl.minimum(tl.maximum((tiles + best_piece - 1) // best_piece, 1), max_split)
    tl.store(plan_ptr + offsets, plan, mask=mask)
