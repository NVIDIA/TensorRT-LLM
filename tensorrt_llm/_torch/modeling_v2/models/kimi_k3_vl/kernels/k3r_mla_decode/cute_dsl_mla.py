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
"""The per-request split-KV plan of the CuTe DSL MLA decode FMHA library."""

import math

import torch
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
# Fixed cost of one piece (cluster launch, Q load, pipeline fill, partial store
# and its reduction) in KV tiles, fitted on SM107 forced-split sweeps (H96,
# 14K-256K KV, B1-B16): with it the wave-quantized makespan picks the measured
# optimum within 5%. Callers whose decode geometry was fitted separately pass
# their own ``piece_cost``.
_SPLIT_PLAN_PIECE_COST_TILES = 60
# The plan kernel is one program over the batch, padded to a power of two of at
# least one warp: its candidate cuts are each a reduction over the batch, so a
# block sized for the maximum request count would make every launch pay for
# _SPLIT_PLAN_MAX_REQUESTS lanes.
_SPLIT_PLAN_MIN_BLOCK = 32
_SPLIT_PLAN_LANES_PER_WARP = 256


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


def plan_mla_split_kv(
    kv_lens: list[int],
    *,
    rows_per_request: int,
    wave_clusters: int,
    max_split: int = _SPLIT_PLAN_MAX_SPLIT,
    kv_tile: int = _SPLIT_PLAN_KV_TILE,
    min_tiles: int = _SPLIT_PLAN_MIN_TILES,
    piece_cost: int = _SPLIT_PLAN_PIECE_COST_TILES,
) -> list[int]:
    """Host mirror of ``_mla_split_kv_plan_kernel``: per-request split counts.

    ``rows_per_request`` is the kernel's packed 128-row query tiles per
    request, each a 2-CTA cluster per split; ``wave_clusters`` the clusters one
    wave holds.
    The longest request is cut into ``k`` equal pieces (every request takes
    ``ceil(tiles / piece)`` of them) for the ``k`` with the smallest
    wave-quantized makespan ``waves * (piece + piece_cost)``.
    """
    num_requests = len(kv_lens)
    if num_requests <= 0 or rows_per_request <= 0 or wave_clusters <= 0 or max_split <= 0:
        raise ValueError("split plan needs requests, rows, wave clusters and a split cap")
    tiles = [max(math.ceil(max(kv, 0) / kv_tile), 1) for kv in kv_lens]
    longest = max(tiles)
    best_piece, best_cost = longest, None
    for pieces in range(1, max_split + 1):
        piece = max(math.ceil(longest / pieces), min_tiles)
        splits = [min(max(math.ceil(t / piece), 1), max_split) for t in tiles]
        waves = math.ceil(sum(splits) * rows_per_request / wave_clusters)
        cost = waves * (piece + piece_cost)
        if best_cost is None or cost < best_cost:
            best_piece, best_cost = piece, cost
    return [min(max(math.ceil(t / best_piece), 1), max_split) for t in tiles]


def launch_mla_split_kv_plan(
    cache_seqs: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_requests: int,
    rows_per_request: int,
    wave_clusters: int,
    max_split: int = _SPLIT_PLAN_MAX_SPLIT,
    piece_cost: int = _SPLIT_PLAN_PIECE_COST_TILES,
) -> None:
    """Write the per-request split counts for ``cache_seqs`` into ``plan`` (one launch).

    The same plan as ``plan_mla_split_kv`` with the same arguments.
    """
    if not 0 < num_requests <= _SPLIT_PLAN_MAX_REQUESTS:
        raise ValueError(f"split plan supports 1..{_SPLIT_PLAN_MAX_REQUESTS} requests")
    if piece_cost < 0:
        raise ValueError(f"split plan piece cost must be >= 0, got {piece_cost}")
    for name, tensor in (("cache_seqs", cache_seqs), ("plan", plan)):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.stride(0) != 1
            or tensor.numel() < num_requests
            or not tensor.is_cuda
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int32 vector of >= {num_requests}")
    block = max(_SPLIT_PLAN_MIN_BLOCK, triton.next_power_of_2(num_requests))
    _mla_split_kv_plan_kernel[(1,)](
        cache_seqs,
        plan,
        num_requests,
        max(rows_per_request, 1),
        max(wave_clusters, 1),
        max(max_split, 1),
        KV_TILE=_SPLIT_PLAN_KV_TILE,
        MIN_TILES=_SPLIT_PLAN_MIN_TILES,
        PIECE_COST=int(piece_cost),
        MAX_PIECES=_SPLIT_PLAN_MAX_SPLIT,
        BLOCK=block,
        num_warps=max(1, min(4, block // _SPLIT_PLAN_LANES_PER_WARP)),
    )
