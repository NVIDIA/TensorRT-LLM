# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device task plan of the 4x FP4 MLA decode split mode ("mixed" split plans).

The plan (int32) is ``[0, used_tasks, begins[0..R], request[cap], split[cap], count[cap], tile_begin[cap],
tile_count[cap]]`` = ``3 + R + 5 * cap`` entries. Its first ``3 + R + 3 * cap`` entries are the
``fp4_mla_packed_reduce`` layout with ``tiles_per_task := 0``, so ``reduce_packed_splits`` merges the slots of
multi-task requests unchanged (single-task requests are written by the decode kernel itself); the 4x kernel
(``_s34_task_identity``) reads the explicit KV tile range of every slot. Slot ``begins[r] + split`` holds the
partial of request ``r``'s task ``split``.

Policy (one CTA, one warp per candidate): the ``n_split`` longest requests get ``k`` tasks each (balanced
contiguous KV chunks), every other request one task. Whole requests fill the full waves of ``lanes`` four-CTA
cluster slots and the tail is cut so that its chunks fill the leftover slots in as few waves as possible, which
is where a uniform task length loses (every request then pays the per-task fixed cost k times and the last wave
spills). Candidates: unsplit, and ``n_split in {T, min(T + ceil(T / 2), R)}`` x ``k in 2..MAX_K`` with ``T``
the requests left after whole requests fill the full waves. A candidate's cost is the simulated grid makespan
(list scheduling of the longest-task-first grid order on ``lanes`` slots; task = max chunk tiles +
``F_TASK_X16`` in 1/16-tile fixed point) plus the reducer's cost for the split slots; the cheapest wins (ties:
fewer tasks). Calibration: Rubin, B32/Q4/H128 real fixtures (per-task fixed cost ~9.5 us = 16.4 tiles of
0.58 us, reducer ~3 us + 0.2 us per MB of partials). Grid order = longest task first (whole requests lead, the
chunks follow), which the hardware's in-order cluster dispatch turns into the simulated schedule.

Host mirror (same integer arithmetic): ``plan_4x_tasks``; sizes: ``plan_4x_task_capacity`` / ``plan_4x_length``.
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .cute_dsl_utils import _compile_cutedsl, _current_cu_stream
from .fp4_mla_cutedsl_mufu16 import KV_TILE

PLAN_4X_THREADS = 1024
PLAN_4X_MAX_REQUESTS = 1024
PLAN_4X_MAX_LANES = 64
PLAN_4X_CANDIDATE_WARPS = PLAN_4X_THREADS // 32
MAX_K = 6
LANES = (
    46  # fp4_mla_cutedsl_4x_host.FULL_ONLY_MAX_PAIRS: four-CTA cluster slots of one wave on SM107
)
UNUSED_SLOT = 1 << 15
# cost model, 1/16 KV tile units (0.58 us per 256-key tile at B32/Q4 KV30K)
F_TASK_X16 = (
    262  # 16.4 tiles = 9.5 us per task (cluster start, Q load, pipeline fill, partial write)
)
REDUCER_BASE_X16 = 83  # 5.2 tiles = 3 us launch + plan scan
REDUCER_SLOT_ROW_NUM = (
    11  # per (slot, row): 2 KiB of partial at 0.2 us/MB = 0.0107 x16-units = 11 / 1024
)
REDUCER_SLOT_ROW_SHIFT = 10

_PLAN_4X_COMPILE_CACHE: dict[tuple[int], Callable] = {}


def plan_4x_length(num_requests: int, task_capacity: int) -> int:
    """Int32 entries of the 4x plan buffer."""
    return 3 + num_requests + 5 * task_capacity


def plan_4x_tail(num_requests: int, pairs_per_request: int, lanes: int = LANES) -> tuple[int, int]:
    """(T, T2): the tail requests after whole requests fill the full waves, and the wider second candidate."""
    if num_requests <= 0 or pairs_per_request <= 0 or lanes <= 0:
        raise ValueError("plan needs positive request, pair and lane counts")
    pairs = num_requests * pairs_per_request
    whole_fit = (lanes * (pairs // lanes)) // pairs_per_request
    tail = num_requests - whole_fit
    more = min(tail + (tail + 1) // 2, num_requests)
    return tail, more


def plan_4x_task_capacity(
    num_requests: int, pairs_per_request: int, lanes: int = LANES, max_k: int = MAX_K
) -> int:
    """Slots the policy can use at most: one per request plus the extra tasks of the widest split candidate.
    The decode grid is ``capacity * pairs_per_request`` clusters (static for CUDA graphs); the slots past the
    plan's ``used_tasks`` exit at kernel entry. ``reduce_packed_splits`` takes it as ``task_capacity``."""
    _, more = plan_4x_tail(num_requests, pairs_per_request, lanes)
    return num_requests + more * (max_k - 1)


def _warp_candidate(
    warp: int, split_tail: int, split_more: int, max_k: int
) -> tuple[bool, int, int]:
    """Host mirror of the warp -> (valid, n_split, k) assignment."""
    if warp == 0:
        return True, 0, 1
    if warp <= max_k - 1:
        return split_tail > 0, split_tail, warp + 1
    if warp <= 2 * (max_k - 1):
        return split_more > split_tail, split_more, warp - (max_k - 1) + 1
    return False, 0, 1


def _reducer_cost_x16(slots: int, rows_per_request: int) -> int:
    if slots <= 0:
        return 0
    return REDUCER_BASE_X16 + (
        (slots * rows_per_request * REDUCER_SLOT_ROW_NUM) >> REDUCER_SLOT_ROW_SHIFT
    )


def plan_4x_tasks(
    kv_lens: list[int],
    *,
    task_capacity: int,
    pairs_per_request: int,
    rows_per_request: int,
    lanes: int = LANES,
    max_k: int = MAX_K,
) -> list[int]:
    """Host mirror of ``plan_4x_kernel`` (same integer arithmetic and tie rules): the int32 plan list."""
    R = len(kv_lens)
    if R <= 0 or R > PLAN_4X_MAX_REQUESTS or lanes <= 0 or lanes > PLAN_4X_MAX_LANES:
        raise ValueError("plan supports 1..1024 requests and 1..64 lanes")
    tiles = [max((int(kv) + KV_TILE - 1) // KV_TILE, 1) for kv in kv_lens]
    rank = [
        sum(1 for o in range(R) if tiles[o] > tiles[r] or (tiles[o] == tiles[r] and o < r))
        for r in range(R)
    ]
    split_tail, split_more = plan_4x_tail(R, pairs_per_request, lanes)

    def count_of(r: int, n_split: int, k: int) -> int:
        return min(k, tiles[r]) if rank[r] < n_split else 1

    def simulate(n_split: int, k: int) -> int:
        slots = [0] * lanes
        makespan = 0
        taken = [False] * R
        for _ in range(R):
            key = -1
            for r in range(R):
                if not taken[r]:
                    cnt = count_of(r, n_split, k)
                    tl = (tiles[r] + cnt - 1) // cnt
                    key = max(key, tl * PLAN_4X_MAX_REQUESTS + (PLAN_4X_MAX_REQUESTS - 1 - r))
            r = PLAN_4X_MAX_REQUESTS - 1 - key % PLAN_4X_MAX_REQUESTS
            tl = key // PLAN_4X_MAX_REQUESTS
            taken[r] = True
            cnt = count_of(r, n_split, k)
            cost = tl * 16 + F_TASK_X16
            for _ in range(cnt * pairs_per_request):
                j = min(range(lanes), key=lambda i: (slots[i], i))
                start = slots[j]
                slots[j] = start + cost
                makespan = max(makespan, start + cost)
        split_slots = sum(count_of(r, n_split, k) for r in range(R) if rank[r] < n_split)
        return makespan + _reducer_cost_x16(split_slots, rows_per_request)

    best = None
    for warp in range(PLAN_4X_CANDIDATE_WARPS):
        valid, n_split, k = _warp_candidate(warp, split_tail, split_more, max_k)
        if not valid:
            continue
        total = simulate(n_split, k)
        tasks = sum(count_of(r, n_split, k) for r in range(R))
        if best is None or total < best[0] or (total == best[0] and tasks < best[1]):
            best = (total, tasks, n_split, k)
    _, used, n_split, k = best
    if used > task_capacity:
        raise ValueError(f"plan needs {used} slots > task_capacity {task_capacity}")
    counts = [count_of(r, n_split, k) for r in range(R)]
    begins = [0]
    for c in counts:
        begins.append(begins[-1] + c)
    plan = [0, used] + begins
    req = [R - 1] * task_capacity
    spl = [UNUSED_SLOT] * task_capacity
    cnt_t = [UNUSED_SLOT] * task_capacity
    tb = [0] * task_capacity
    tn = [0] * task_capacity
    task_len = [(tiles[r] + counts[r] - 1) // counts[r] for r in range(R)]
    for r in range(R):
        pos = sum(
            counts[o]
            for o in range(R)
            if task_len[o] > task_len[r] or (task_len[o] == task_len[r] and o < r)
        )
        for s in range(counts[r]):
            idx = pos + s
            req[idx] = r
            spl[idx] = s
            cnt_t[idx] = counts[r]
            tb[idx] = tiles[r] * s // counts[r]
            tn[idx] = tiles[r] * (s + 1) // counts[r] - tb[idx]
    plan += req + spl + cnt_t + tb + tn
    assert len(plan) == plan_4x_length(R, task_capacity)
    return plan


@cute.jit
def _plan_4x_simulate(
    sTiles: cute.Tensor,
    sRank: cute.Tensor,
    num_requests: cutlass.Int32,
    n_split: cutlass.Int32,
    k: cutlass.Int32,
    pairs_per_request: cutlass.Int32,
    rows_per_request: cutlass.Int32,
    lanes: cutlass.Int32,
    f_task_x16: cutlass.Int32,
    reducer_base_x16: cutlass.Int32,
    lane: cutlass.Int32,
) -> cutlass.Int32:
    """One warp: makespan (1/16 tiles) of the candidate's grid on ``lanes`` slots plus its reducer cost.

    Slot j lives in lane j % 32, register j // 32 (lanes <= 64). Requests are scheduled longest task first
    (ties: lower request), each one placing count * pairs_per_request equal tasks on the slots that free first.
    """
    idle = cutlass.Int32(1 << 30)
    slot0 = idle
    slot1 = idle
    if lane < lanes:
        slot0 = cutlass.Int32(0)
    if lane + cutlass.Int32(32) < lanes:
        slot1 = cutlass.Int32(0)
    makespan = cutlass.Int32(0)
    taken = cutlass.Int32(0)
    split_slots = cutlass.Int32(0)
    for request_idx in cutlass.range(lane, num_requests, 32, unroll=1):
        if sRank[request_idx] < n_split:
            split_slots = split_slots + cutlass.min(k, sTiles[request_idx])
    split_slots = cute.arch.warp_redux_sync(split_slots, "add")
    for _ in cutlass.range(0, num_requests, 1, unroll=1):
        local_key = cutlass.Int32(-1)
        for request_idx in cutlass.range(lane, num_requests, 32, unroll=1):
            taken_bit = (taken >> (request_idx // cutlass.Int32(32))) & cutlass.Int32(1)
            if taken_bit == cutlass.Int32(0):
                tiles = sTiles[request_idx]
                cnt = cutlass.Int32(1)
                if sRank[request_idx] < n_split:
                    cnt = cutlass.min(k, tiles)
                task_len = (tiles + cnt - cutlass.Int32(1)) // cnt
                local_key = cutlass.max(
                    local_key,
                    task_len * cutlass.Int32(PLAN_4X_MAX_REQUESTS)
                    + cutlass.Int32(PLAN_4X_MAX_REQUESTS - 1)
                    - request_idx,
                )
        key = cute.arch.warp_redux_sync(local_key, "max")
        request = cutlass.Int32(PLAN_4X_MAX_REQUESTS - 1) - key % cutlass.Int32(
            PLAN_4X_MAX_REQUESTS
        )
        task_len = key // cutlass.Int32(PLAN_4X_MAX_REQUESTS)
        if lane == request % cutlass.Int32(32):
            taken = taken | (cutlass.Int32(1) << (request // cutlass.Int32(32)))
        tiles = sTiles[request]
        cnt = cutlass.Int32(1)
        if sRank[request] < n_split:
            cnt = cutlass.min(k, tiles)
        cost = task_len * cutlass.Int32(16) + f_task_x16
        for _ in cutlass.range(0, cnt * pairs_per_request, 1, unroll=1):
            local_free = cutlass.min(slot0, slot1)
            start = cute.arch.warp_redux_sync(local_free, "min")
            owner_key = cutlass.Int32(32)
            if local_free == start:
                owner_key = lane
            owner = cute.arch.warp_redux_sync(owner_key, "min")
            if lane == owner:
                if slot0 == start:
                    slot0 = start + cost
                else:
                    slot1 = start + cost
            makespan = cutlass.max(makespan, start + cost)
    total = makespan
    if split_slots > cutlass.Int32(0):
        total = (
            total
            + reducer_base_x16
            + (
                (split_slots * rows_per_request * cutlass.Int32(REDUCER_SLOT_ROW_NUM))
                >> cutlass.Int32(REDUCER_SLOT_ROW_SHIFT)
            )
        )
    return total


@cute.kernel
def plan_4x_kernel(
    mValidK_l: cute.Tensor,
    mPlan_i: cute.Tensor,
    num_requests: cutlass.Int32,
    task_capacity: cutlass.Int32,
    lanes: cutlass.Int32,
    pairs_per_request: cutlass.Int32,
    rows_per_request: cutlass.Int32,
    split_tail: cutlass.Int32,
    split_more: cutlass.Int32,
    f_task_x16: cutlass.Int32,
    reducer_base_x16: cutlass.Int32,
    max_k: cutlass.Constexpr,
) -> None:
    """One CTA of PLAN_4X_THREADS: the mixed split plan (module docstring); host mirror ``plan_4x_tasks``."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // cutlass.Int32(32)
    lane = tidx % cutlass.Int32(32)
    sTiles = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_4X_MAX_REQUESTS),
        cute.make_layout(PLAN_4X_MAX_REQUESTS),
    )
    sRank = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_4X_MAX_REQUESTS),
        cute.make_layout(PLAN_4X_MAX_REQUESTS),
    )
    sBegins = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_4X_MAX_REQUESTS + 1),
        cute.make_layout(PLAN_4X_MAX_REQUESTS + 1),
    )
    # per candidate warp: total cost, n_split, k; then the winner's (n_split, k)
    sCandidates = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, 3 * PLAN_4X_CANDIDATE_WARPS + 2),
        cute.make_layout(3 * PLAN_4X_CANDIDATE_WARPS + 2),
    )
    for request_idx in cutlass.range(tidx, num_requests, PLAN_4X_THREADS, unroll=1):
        sTiles[request_idx] = cutlass.max(
            (cutlass.Int32(mValidK_l[request_idx]) + cutlass.Int32(KV_TILE - 1))
            // cutlass.Int32(KV_TILE),
            cutlass.Int32(1),
        )
    cute.arch.sync_threads()
    # rank by KV tiles, longest first (ties: lower request)
    for request_idx in cutlass.range(tidx, num_requests, PLAN_4X_THREADS, unroll=1):
        tiles = sTiles[request_idx]
        rank = cutlass.Int32(0)
        for other_idx in cutlass.range(0, num_requests, 1, unroll=1):
            other = sTiles[other_idx]
            if (other > tiles) | ((other == tiles) & (other_idx < request_idx)):
                rank = rank + cutlass.Int32(1)
        sRank[request_idx] = rank
    cute.arch.sync_threads()
    # one candidate per warp
    valid = cutlass.Boolean(False)
    n_split = cutlass.Int32(0)
    k = cutlass.Int32(1)
    if warp_idx == cutlass.Int32(0):
        valid = cutlass.Boolean(True)
    elif warp_idx <= cutlass.Int32(max_k - 1):
        valid = ctm_bool(split_tail > cutlass.Int32(0))
        n_split = split_tail
        k = warp_idx + cutlass.Int32(1)
    elif warp_idx <= cutlass.Int32(2 * (max_k - 1)):
        valid = ctm_bool(split_more > split_tail)
        n_split = split_more
        k = warp_idx - cutlass.Int32(max_k - 1) + cutlass.Int32(1)
    total = cutlass.Int32(1 << 30)
    if valid:
        total = _plan_4x_simulate(
            sTiles,
            sRank,
            num_requests,
            n_split,
            k,
            pairs_per_request,
            rows_per_request,
            lanes,
            f_task_x16,
            reducer_base_x16,
            lane,
        )
    if lane == cutlass.Int32(0):
        sCandidates[warp_idx] = total
        sCandidates[PLAN_4X_CANDIDATE_WARPS + warp_idx] = n_split
        sCandidates[2 * PLAN_4X_CANDIDATE_WARPS + warp_idx] = k
    cute.arch.sync_threads()
    if tidx == cutlass.Int32(0):
        best_total = cutlass.Int32(1 << 30)
        best_tasks = cutlass.Int32(1 << 30)
        best_split = cutlass.Int32(0)
        best_k = cutlass.Int32(1)
        for candidate_idx in cutlass.range(0, PLAN_4X_CANDIDATE_WARPS, 1, unroll=1):
            cand_total = sCandidates[candidate_idx]
            if cand_total < cutlass.Int32(1 << 30):
                cand_split = sCandidates[PLAN_4X_CANDIDATE_WARPS + candidate_idx]
                cand_k = sCandidates[2 * PLAN_4X_CANDIDATE_WARPS + candidate_idx]
                cand_tasks = cutlass.Int32(0)
                for request_idx in cutlass.range(0, num_requests, 1, unroll=1):
                    if sRank[request_idx] < cand_split:
                        cand_tasks = cand_tasks + cutlass.min(cand_k, sTiles[request_idx])
                    else:
                        cand_tasks = cand_tasks + cutlass.Int32(1)
                if (cand_total < best_total) | (
                    (cand_total == best_total) & (cand_tasks < best_tasks)
                ):
                    best_total = cand_total
                    best_tasks = cand_tasks
                    best_split = cand_split
                    best_k = cand_k
        sCandidates[3 * PLAN_4X_CANDIDATE_WARPS] = best_split
        sCandidates[3 * PLAN_4X_CANDIDATE_WARPS + 1] = best_k
        # begins (request order) and the header
        task_begin = cutlass.Int32(0)
        for request_idx in cutlass.range(0, num_requests, 1, unroll=1):
            sBegins[request_idx] = task_begin
            mPlan_i[cutlass.Int32(2) + request_idx] = task_begin
            cnt = cutlass.Int32(1)
            if sRank[request_idx] < best_split:
                cnt = cutlass.min(best_k, sTiles[request_idx])
            task_begin = task_begin + cnt
        sBegins[num_requests] = task_begin
        mPlan_i[cutlass.Int32(2) + num_requests] = task_begin
        mPlan_i[0] = cutlass.Int32(0)  # ranges plan: tile_begin / tile_count tables follow count[]
        mPlan_i[1] = task_begin
    cute.arch.sync_threads()
    best_split = sCandidates[3 * PLAN_4X_CANDIDATE_WARPS]
    best_k = sCandidates[3 * PLAN_4X_CANDIDATE_WARPS + 1]
    used_tasks = sBegins[num_requests]
    tables = cutlass.Int32(3) + num_requests
    # grid order: longest task first (ties: lower request); a request's tasks are consecutive, split-ordered
    for request_idx in cutlass.range(tidx, num_requests, PLAN_4X_THREADS, unroll=1):
        tiles = sTiles[request_idx]
        cnt = cutlass.Int32(1)
        if sRank[request_idx] < best_split:
            cnt = cutlass.min(best_k, tiles)
        task_len = (tiles + cnt - cutlass.Int32(1)) // cnt
        position = cutlass.Int32(0)
        for other_idx in cutlass.range(0, num_requests, 1, unroll=1):
            other_tiles = sTiles[other_idx]
            other_cnt = cutlass.Int32(1)
            if sRank[other_idx] < best_split:
                other_cnt = cutlass.min(best_k, other_tiles)
            other_len = (other_tiles + other_cnt - cutlass.Int32(1)) // other_cnt
            if (other_len > task_len) | ((other_len == task_len) & (other_idx < request_idx)):
                position = position + other_cnt
        for split_idx in cutlass.range(0, cnt, 1, unroll=1):
            task_idx = position + split_idx
            tile_begin = tiles * split_idx // cnt
            tile_end = tiles * (split_idx + cutlass.Int32(1)) // cnt
            mPlan_i[tables + task_idx] = request_idx
            mPlan_i[tables + task_capacity + task_idx] = split_idx
            mPlan_i[tables + cutlass.Int32(2) * task_capacity + task_idx] = cnt
            mPlan_i[tables + cutlass.Int32(3) * task_capacity + task_idx] = tile_begin
            mPlan_i[tables + cutlass.Int32(4) * task_capacity + task_idx] = tile_end - tile_begin
    for task_idx in cutlass.range(used_tasks + tidx, task_capacity, PLAN_4X_THREADS, unroll=1):
        mPlan_i[tables + task_idx] = num_requests - cutlass.Int32(1)
        mPlan_i[tables + task_capacity + task_idx] = cutlass.Int32(UNUSED_SLOT)
        mPlan_i[tables + cutlass.Int32(2) * task_capacity + task_idx] = cutlass.Int32(UNUSED_SLOT)
        mPlan_i[tables + cutlass.Int32(3) * task_capacity + task_idx] = cutlass.Int32(0)
        mPlan_i[tables + cutlass.Int32(4) * task_capacity + task_idx] = cutlass.Int32(0)


def ctm_bool(value):
    """cutlass.Boolean from a dynamic comparison (keeps the kernel body readable)."""
    return cutlass.Boolean(value)


@cute.jit
def plan_4x_host(
    kv_lens_ptr: cute.Pointer,
    plan_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_requests: cutlass.Int32,
    task_capacity: cutlass.Int32,
    lanes: cutlass.Int32,
    pairs_per_request: cutlass.Int32,
    rows_per_request: cutlass.Int32,
    split_tail: cutlass.Int32,
    split_more: cutlass.Int32,
    f_task_x16: cutlass.Int32,
    reducer_base_x16: cutlass.Int32,
    max_k: cutlass.Constexpr,
) -> None:
    mValidK_l = cute.make_tensor(kv_lens_ptr, cute.make_layout((num_requests,), stride=(1,)))
    mPlan_i = cute.make_tensor(
        plan_ptr,
        cute.make_layout((cutlass.Int32(3) + num_requests + 5 * task_capacity,), stride=(1,)),
    )
    plan_4x_kernel(
        mValidK_l,
        mPlan_i,
        num_requests,
        task_capacity,
        lanes,
        pairs_per_request,
        rows_per_request,
        split_tail,
        split_more,
        f_task_x16,
        reducer_base_x16,
        max_k,
    ).launch(grid=(1, 1, 1), block=(PLAN_4X_THREADS, 1, 1), stream=stream)


def launch_4x_plan(
    kv_lens: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_requests: int,
    task_capacity: int,
    pairs_per_request: int,
    rows_per_request: int,
    lanes: int = LANES,
    max_k: int = MAX_K,
) -> None:
    """Write the mixed split plan for ``num_requests`` requests into ``plan`` (``plan_4x_length`` int32 entries).

    ``kv_lens`` are the per-request int32 KV lengths the decode kernel reads; ``task_capacity`` must come from
    ``plan_4x_task_capacity`` with the same ``pairs_per_request`` / ``lanes`` / ``max_k`` (the policy never needs
    more slots). Runs on the current stream; CUDA-graph safe (no host reads).
    """
    if num_requests <= 0 or num_requests > PLAN_4X_MAX_REQUESTS:
        raise ValueError(f"plan supports 1..{PLAN_4X_MAX_REQUESTS} requests, got {num_requests}")
    if lanes <= 0 or lanes > PLAN_4X_MAX_LANES or pairs_per_request <= 0 or rows_per_request <= 0:
        raise ValueError("plan needs 1..64 lanes and positive pairs / rows per request")
    if max_k < 2 or 2 * (max_k - 1) >= PLAN_4X_CANDIDATE_WARPS:
        raise ValueError(f"max_k must be in 2..{PLAN_4X_CANDIDATE_WARPS // 2}")
    if task_capacity < plan_4x_task_capacity(num_requests, pairs_per_request, lanes, max_k):
        raise ValueError(
            f"task_capacity {task_capacity} < plan_4x_task_capacity "
            f"{plan_4x_task_capacity(num_requests, pairs_per_request, lanes, max_k)}"
        )
    if kv_lens.dtype != torch.int32 or kv_lens.numel() < num_requests or kv_lens.stride(0) != 1:
        raise ValueError(f"kv_lens must be contiguous int32 with >= {num_requests} entries")
    plan_len = plan_4x_length(num_requests, task_capacity)
    if plan.dtype != torch.int32 or plan.numel() < plan_len or plan.stride(0) != 1:
        raise ValueError(f"plan must be contiguous int32 with >= {plan_len} entries")
    split_tail, split_more = plan_4x_tail(num_requests, pairs_per_request, lanes)
    gmem = cute.AddressSpace.gmem
    ptrs = (
        make_ptr(cutlass.Int32, kv_lens.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, plan.data_ptr(), gmem, assumed_align=4),
    )
    stream = _current_cu_stream()
    cache_key = (max_k,)
    plan_fn = _PLAN_4X_COMPILE_CACHE.get(cache_key)
    if plan_fn is None:
        plan_fn = _compile_cutedsl(
            plan_4x_host,
            *ptrs,
            stream,
            *(cutlass.Int32(1) for _ in range(9)),
            max_k,
        )
        _PLAN_4X_COMPILE_CACHE[cache_key] = plan_fn
    plan_fn(
        *ptrs,
        stream,
        cutlass.Int32(num_requests),
        cutlass.Int32(task_capacity),
        cutlass.Int32(lanes),
        cutlass.Int32(pairs_per_request),
        cutlass.Int32(rows_per_request),
        cutlass.Int32(split_tail),
        cutlass.Int32(split_more),
        cutlass.Int32(F_TASK_X16),
        cutlass.Int32(REDUCER_BASE_X16),
    )
