# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request split-KV plan of the CuTe DSL MLA decode kernel (``trtllm::cute_dsl_mla_decode_fp16_blackwell``).

The same kernel and constants as the stock CuTe DSL MLA FMHA library (``attention/backends/fmha/cute_dsl_mla.py``),
which the built-in MLA runs Kimi K3's decode-only generation on; kept here so the K3 MLA module runs the same plan
without importing the attention backends. The kernel grid carries the tactic's split_kv slots per request; the plan
decides how many a request uses this step from the batch's KV lengths (the kernel clamps it to the grid and the
slots beyond exit before their mainloop).
"""

import math

import torch
import triton
import triton.language as tl

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
# optimum within 5%.
_SPLIT_PLAN_PIECE_COST_TILES = 60
# ``assumed_align`` the decode runner compiles the page table and the KV
# lengths with; their runtime pointers must keep it.
_KERNEL_INPUT_ALIGN_BYTES = 16


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
) -> None:
    """Write the per-request split counts for ``cache_seqs`` into ``plan`` (one launch)."""
    if not 0 < num_requests <= _SPLIT_PLAN_MAX_REQUESTS:
        raise ValueError(f"split plan supports 1..{_SPLIT_PLAN_MAX_REQUESTS} requests")
    for name, tensor in (("cache_seqs", cache_seqs), ("plan", plan)):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.stride(0) != 1
            or tensor.numel() < num_requests
            or not tensor.is_cuda
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int32 vector of >= {num_requests}")
    _mla_split_kv_plan_kernel[(1,)](
        cache_seqs,
        plan,
        num_requests,
        max(rows_per_request, 1),
        max(wave_clusters, 1),
        max(max_split, 1),
        KV_TILE=_SPLIT_PLAN_KV_TILE,
        MIN_TILES=_SPLIT_PLAN_MIN_TILES,
        PIECE_COST=_SPLIT_PLAN_PIECE_COST_TILES,
        MAX_PIECES=_SPLIT_PLAN_MAX_SPLIT,
        BLOCK=_SPLIT_PLAN_MAX_REQUESTS,
        num_warps=4,
    )
