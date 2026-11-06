# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL split reduction for packed FP4 MLA decode tasks.

The decode kernel writes one FP32 partial per task covering the request's
128-padded [query token, head] rows together with the raw row maximum and the
row sum relative to it. Tasks follow the balanced plan of the decode kernel
(consecutive tasks per request, more for longer KV), which this kernel rederives
from the same per-request bounds. It merges the tasks of every valid row into the
BF16/FP16 output and, for Helix, the per-row (max, sum) softmax stats. Global FP4
scales and KV lengths are read on the device, so CUDA graph replays follow them.
"""

import math
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .cute_dsl_utils import _compile_cutedsl, _current_cu_stream
from .fp4_mla_cutedsl_mufu16 import (
    KV_TILE,
    LOG2_E,
    PACKED_MIN_TILES_PER_TASK,
    _cutlass_output_dtype,
    _packed_query_bound,
    _validate_output_dtype,
    _validate_tensor_pointer_alignment,
)

_PACKED_REDUCE_COMPILE_CACHE: dict[tuple[int, bool, torch.dtype], Callable] = {}
_PACKED_PLAN_COMPILE_CACHE: dict[tuple[int, bool, bool], Callable] = {}
_REDUCE_THREADS = 128
_PLAN_THREADS = 128
# Requests one plan CTA stages in shared memory (4 KiB of tiles + begins).
PLAN_MAX_REQUESTS = 1024
_VALUE_DIM = 512
# Tasks whose partial loads a warp issues back-to-back before any is consumed
# with a full 512-column band (16 vector loads per lane; narrower bands take
# proportionally more tasks per chunk). The reduction is latency-bound, so
# this sets its memory-level parallelism; 32 loads in flight measured slower
# (register pressure), as did issuing all loads ahead of the weight math.
_TASKS_PER_CHUNK = 4
# Persistent reducer grid: up to this many CTAs per SM, one warp per output row.
_REDUCE_CTAS_PER_SM = 4


@cute.jit
def _reduce_packed_row(
    mPartial_trc: cute.Tensor,
    mRowMax_tr: cute.Tensor,
    mRowSum_tr: cute.Tensor,
    mOut_rc: cute.Tensor,
    mStatsMax_r: cute.Tensor,
    mStatsSum_r: cute.Tensor,
    lane: cutlass.Int32,
    row: cutlass.Int32,
    local_row: cutlass.Int32,
    band: cutlass.Int32,
    task_begin: cutlass.Int32,
    request_tasks: cutlass.Int32,
    physical_m: cutlass.Int32,
    scale: cutlass.Float32,
    output_scale: cutlass.Float32,
    band_columns: cutlass.Constexpr,
    write_stats: cutlass.Constexpr,
) -> None:
    """One warp merges the planned tasks of one (output row, column band).

    Lane ``l`` owns columns ``band * band_columns + i * 128 + 4 * l .. + 4`` for
    ``i < band_columns // 128``, so every vector load and store of the warp
    covers 512 contiguous bytes. Tasks are consumed in chunks sized to keep 16
    vector loads in flight per lane; the per-task weights are recomputed by
    every lane from the row state (L1-resident), so no shuffles or shared
    memory are needed. Empty tasks (row sum 0) carry zero weight and a row
    with no visible key anywhere yields O=0 and stats (-inf, 0), the neutral
    Helix contribution.
    """
    columns_per_lane: cutlass.Constexpr = 4
    column_groups: cutlass.Constexpr = band_columns // (32 * columns_per_lane)
    tasks_per_chunk: cutlass.Constexpr = _TASKS_PER_CHUNK * (_VALUE_DIM // band_columns)
    band_base = band * cutlass.Int32(band_columns) + lane * cutlass.Int32(4)
    gPartial_arr = cutlass.make_array_view(mPartial_trc)
    gOut_arr = cutlass.make_array_view(mOut_rc)
    global_row_max = cutlass.Float32(-cutlass.Float32.inf)
    for task_idx in cutlass.range(lane, request_tasks, 32, unroll=1):
        task = task_begin + task_idx
        if mRowSum_tr[task, local_row] != cutlass.Float32(0.0):
            global_row_max = cute.arch.fmax(global_row_max, mRowMax_tr[task, local_row] * scale)
    global_row_max = cute.arch.warp_redux_sync(global_row_max, "fmax")

    # Four independent accumulators (the DSL tracks rebinding per object).
    numerator0 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    numerator1 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    numerator2 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    numerator3 = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_lane)), dtype=cutlass.Float32
    )
    denominator = cutlass.Float32(0.0)
    for chunk_base in cutlass.range(0, request_tasks, tasks_per_chunk, unroll=1):
        chunk_vecs = []
        chunk_weights = []
        for slot in cutlass.range_constexpr(tasks_per_chunk):
            # Out-of-range slots re-read the chunk's last valid task with zero weight.
            task_idx = cutlass.min(chunk_base + slot, request_tasks - cutlass.Int32(1))
            task = task_begin + task_idx
            task_weight = cutlass.Float32(0.0)
            if chunk_base + slot < request_tasks:
                task_row_sum = mRowSum_tr[task, local_row]
                if task_row_sum != cutlass.Float32(0.0):
                    task_weight = cute.exp2(
                        mRowMax_tr[task, local_row] * scale - global_row_max, fastmath=True
                    )
                    denominator += task_row_sum * task_weight
            chunk_weights.append(task_weight)
            partial_row = (task * physical_m + local_row) * cutlass.Int32(_VALUE_DIM) + band_base
            for group in cutlass.range_constexpr(column_groups):
                chunk_vecs.append(
                    gPartial_arr.subview(partial_row + cutlass.Int32(group * 128))
                    .data_ptr()
                    .load(count=columns_per_lane, alignment=4 * columns_per_lane)
                )
        for slot in cutlass.range_constexpr(tasks_per_chunk):
            # Empty tasks keep zero weight; their payload is never trusted.
            if chunk_weights[slot] != cutlass.Float32(0.0):
                numerator0 = numerator0 + chunk_vecs[slot * column_groups] * chunk_weights[slot]
                if cutlass.const_expr(column_groups > 1):
                    numerator1 = (
                        numerator1 + chunk_vecs[slot * column_groups + 1] * chunk_weights[slot]
                    )
                if cutlass.const_expr(column_groups > 2):
                    numerator2 = (
                        numerator2 + chunk_vecs[slot * column_groups + 2] * chunk_weights[slot]
                    )
                    numerator3 = (
                        numerator3 + chunk_vecs[slot * column_groups + 3] * chunk_weights[slot]
                    )

    normalizer = cutlass.Float32(0.0)
    if denominator != cutlass.Float32(0.0):
        normalizer = output_scale / denominator
    out_row = row * cutlass.Int32(_VALUE_DIM) + band_base
    gOut_arr.store(
        (numerator0 * normalizer).to(mOut_rc.element_type),
        idx=out_row,
        vector_size=columns_per_lane,
        alignment=2 * columns_per_lane,
    )
    if cutlass.const_expr(column_groups > 1):
        gOut_arr.store(
            (numerator1 * normalizer).to(mOut_rc.element_type),
            idx=out_row + cutlass.Int32(128),
            vector_size=columns_per_lane,
            alignment=2 * columns_per_lane,
        )
    if cutlass.const_expr(column_groups > 2):
        gOut_arr.store(
            (numerator2 * normalizer).to(mOut_rc.element_type),
            idx=out_row + cutlass.Int32(256),
            vector_size=columns_per_lane,
            alignment=2 * columns_per_lane,
        )
        gOut_arr.store(
            (numerator3 * normalizer).to(mOut_rc.element_type),
            idx=out_row + cutlass.Int32(384),
            vector_size=columns_per_lane,
            alignment=2 * columns_per_lane,
        )
    if cutlass.const_expr(write_stats):
        if band == cutlass.Int32(0) and lane == cutlass.Int32(0):
            # Helix combine convention: natural-log-domain scaled max and the
            # sum relative to it; an empty row contributes (-inf, 0).
            stats_max = cutlass.Float32(-cutlass.Float32.inf)
            if denominator != cutlass.Float32(0.0):
                stats_max = global_row_max * cutlass.Float32(1.0 / LOG2_E)
            mStatsMax_r[row] = stats_max
            mStatsSum_r[row] = denominator


@cute.kernel
def packed_split_reduce_kernel(
    mPartial_trc: cute.Tensor,
    mRowMax_tr: cute.Tensor,
    mRowSum_tr: cute.Tensor,
    mOut_rc: cute.Tensor,
    mStatsMax_r: cute.Tensor,
    mStatsSum_r: cute.Tensor,
    mQScale: cute.Tensor,
    mKvScale: cute.Tensor,
    mPlan_i: cute.Tensor,
    softmax_scale_log2: cutlass.Float32,
    output_scale: cutlass.Float32,
    num_requests: cutlass.Int32,
    valid_m: cutlass.Int32,
    physical_m: cutlass.Int32,
    band_columns: cutlass.Constexpr,
    write_stats: cutlass.Constexpr,
) -> None:
    """Persistent grid: each warp walks (row, column band) items of multi-task requests.

    Every CTA stages the plan's task prefix (plan[2 .. 2 + num_requests]) and
    scans the requests. The (row, band) pairs of requests with more than one
    task form one compact item space that the grid's warps stride through, one
    item per warp at a time (narrower bands when few rows would leave SMs
    idle); a single-task request was stored by the decode kernel itself and,
    for Helix, only needs its (max, sum) stats derived from its row state.
    With no multi-task request the kernel exits after the plan scan, so an
    even batch pays only the launch.
    """
    bands: cutlass.Constexpr = _VALUE_DIM // band_columns
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // cutlass.Int32(32)
    lane = tidx % cutlass.Int32(32)
    warps_per_cta: cutlass.Constexpr = _REDUCE_THREADS // 32
    sBegins = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_MAX_REQUESTS + 1),
        cute.make_layout(PLAN_MAX_REQUESTS + 1),
    )
    for entry in cutlass.range(tidx, num_requests + cutlass.Int32(1), _REDUCE_THREADS, unroll=1):
        sBegins[entry] = mPlan_i[cutlass.Int32(2) + entry]
    cute.arch.sync_threads()
    scale = softmax_scale_log2 / (mQScale[0] * mKvScale[0])
    # Compact item c of the multi-task (row, band) pairs belongs to warp slot
    # c % item_stride.
    item_slot = bidx * cutlass.Int32(warps_per_cta) + warp_idx
    item_stride = grid_x * cutlass.Int32(warps_per_cta)
    compact_offset = cutlass.Int32(0)
    for request in cutlass.range(0, num_requests, 1, unroll=1):
        task_begin = cutlass.Int32(sBegins[request])
        request_tasks = cutlass.Int32(sBegins[request + cutlass.Int32(1)]) - task_begin
        if request_tasks > cutlass.Int32(1):
            request_items = valid_m * cutlass.Int32(bands)
            first_item = compact_offset + (
                (item_slot - compact_offset % item_stride + item_stride) % item_stride
            )
            for compact_item in cutlass.range(
                first_item, compact_offset + request_items, item_stride, unroll=1
            ):
                local_item = compact_item - compact_offset
                local_row = local_item // cutlass.Int32(bands)
                band = local_item - local_row * cutlass.Int32(bands)
                _reduce_packed_row(
                    mPartial_trc,
                    mRowMax_tr,
                    mRowSum_tr,
                    mOut_rc,
                    mStatsMax_r,
                    mStatsSum_r,
                    lane,
                    request * valid_m + local_row,
                    local_row,
                    band,
                    task_begin,
                    request_tasks,
                    physical_m,
                    scale,
                    output_scale,
                    band_columns,
                    write_stats,
                )
            compact_offset += request_items
        else:
            if cutlass.const_expr(write_stats):
                # Thread-parallel over the request's rows: chunk k of 128 rows
                # is owned by CTA k % grid.
                row_chunks = (valid_m + cutlass.Int32(127)) // cutlass.Int32(128)
                for chunk in cutlass.range(bidx, row_chunks, grid_x, unroll=1):
                    local_row = chunk * cutlass.Int32(128) + tidx
                    if local_row < valid_m:
                        row = request * valid_m + local_row
                        row_sum = mRowSum_tr[task_begin, local_row]
                        stats_max = cutlass.Float32(-cutlass.Float32.inf)
                        if row_sum != cutlass.Float32(0.0):
                            stats_max = (
                                mRowMax_tr[task_begin, local_row]
                                * scale
                                * cutlass.Float32(1.0 / LOG2_E)
                            )
                        mStatsMax_r[row] = stats_max
                        mStatsSum_r[row] = row_sum


@cute.jit
def packed_split_reduce_host(
    partial_ptr: cute.Pointer,
    row_max_ptr: cute.Pointer,
    row_sum_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    stats_max_ptr: cute.Pointer,
    stats_sum_ptr: cute.Pointer,
    q_scale_ptr: cute.Pointer,
    kv_scale_ptr: cute.Pointer,
    plan_ptr: cute.Pointer,
    softmax_scale_log2: cutlass.Float32,
    output_scale: cutlass.Float32,
    stream: cuda.CUstream,
    num_requests: cutlass.Int32,
    num_splits: cutlass.Int32,
    valid_m: cutlass.Int32,
    physical_m: cutlass.Int32,
    total_rows: cutlass.Int32,
    stats_stride: cutlass.Int32,
    grid_ctas: cutlass.Int32,
    band_columns: cutlass.Constexpr,
    write_stats: cutlass.Constexpr,
) -> None:
    tasks = num_requests * num_splits
    partial_layout = cute.make_layout(
        (tasks, physical_m, _VALUE_DIM),
        stride=(cutlass.Int64(physical_m) * _VALUE_DIM, _VALUE_DIM, 1),
    )
    row_state_layout = cute.make_layout((tasks, physical_m), stride=(physical_m, 1))
    out_layout = cute.make_layout((total_rows, _VALUE_DIM), stride=(_VALUE_DIM, 1))
    stats_layout = cute.make_layout((total_rows,), stride=(stats_stride,))
    mPartial_trc = cute.make_tensor(partial_ptr, partial_layout)
    mRowMax_tr = cute.make_tensor(row_max_ptr, row_state_layout)
    mRowSum_tr = cute.make_tensor(row_sum_ptr, row_state_layout)
    mOut_rc = cute.make_tensor(out_ptr, out_layout)
    mStatsMax_r = cute.make_tensor(stats_max_ptr, stats_layout)
    mStatsSum_r = cute.make_tensor(stats_sum_ptr, stats_layout)
    mQScale = cute.make_tensor(q_scale_ptr, cute.make_layout((1,), stride=(1,)))
    mKvScale = cute.make_tensor(kv_scale_ptr, cute.make_layout((1,), stride=(1,)))
    mPlan_i = cute.make_tensor(
        plan_ptr, cute.make_layout((cutlass.Int32(3) + num_requests + 3 * tasks,), stride=(1,))
    )
    packed_split_reduce_kernel(
        mPartial_trc,
        mRowMax_tr,
        mRowSum_tr,
        mOut_rc,
        mStatsMax_r,
        mStatsSum_r,
        mQScale,
        mKvScale,
        mPlan_i,
        softmax_scale_log2,
        output_scale,
        num_requests,
        valid_m,
        physical_m,
        band_columns,
        write_stats,
    ).launch(
        grid=(grid_ctas, 1, 1),
        block=(_REDUCE_THREADS, 1, 1),
        stream=stream,
    )


@cute.kernel
def packed_plan_kernel(
    mValidK_l: cute.Tensor,
    mHelixKvBounds_l: cute.Tensor,
    mPromptLens_l: cute.Tensor,
    mCorrectedKvLens_l: cute.Tensor,
    mGenerationLens_l: cute.Tensor,
    mPlan_i: cute.Tensor,
    num_requests: cutlass.Int32,
    task_capacity: cutlass.Int32,
    wave_tasks: cutlass.Int32,
    max_tiles_per_task: cutlass.Int32,
    generation_len: cutlass.Int32,
    query_len_per_seq: cutlass.Constexpr,
    use_helix_kv_bounds: cutlass.Constexpr,
    write_generation_lengths: cutlass.Constexpr,
) -> None:
    """One CTA: the balanced task plan both decode and reduction kernels read.

    With ``write_generation_lengths`` the CTA first derives the per-step
    generation lengths every FP4 MLA kernel of the step reads (``kv_lens -
    prompt_lens + generation_len`` and ``generation_len`` per request) and
    plans on those, so the first FP4 MLA kernel of a step pays one launch for
    both; otherwise ``mValidK_l`` already holds the corrected lengths.

    ``plan = [tiles_per_task, used_tasks, begins[0..num_requests],
    request[task_capacity], split[task_capacity], count[task_capacity]]``.
    Request ``r`` owns ``ceil(tiles_r / tiles_per_task)`` consecutive tasks;
    slots past ``used_tasks`` map to the last request with an out-of-range
    split and are never scheduled (the decode kernel exits them).

    Task length: a request is only split when it is at least 1.5x longer than
    the batch's work spread over one wave, ``ceil(total / wave_tasks)``, i.e.
    when its tail would outlast the machine-bound makespan (an even batch that
    fills the machine is never split). The length then targets one wave of
    tasks for an even, under-filled batch and two waves when the machine is
    full (3 * wave_tasks <= 4 * requests) or the batch is strongly skewed
    (longest > 2x average), where the extra slack lets short tasks backfill
    behind the long request's pieces; a Helix rank always targets one wave
    (it holds 1/N of every request's KV, so two-wave pieces are short and
    the per-task cost dominates); the longest request is cut into
    ``round(longest / target)`` equal pieces. If that needs more tasks than
    the capacity holds, the smaller of the guaranteed-fit share
    ``ceil(total / (capacity - requests))`` and the per-request share
    ``ceil(longest / splits)`` that fits is used. Everything is clamped to
    ``max_tiles_per_task``, the KV the compiled runtime profile covers (the
    host sizes it so the clamp never needs more than the capacity). The host
    mirror is ``plan_packed_tasks``.
    """
    tidx, _, _ = cute.arch.thread_idx()
    sTiles = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_MAX_REQUESTS), cute.make_layout(PLAN_MAX_REQUESTS)
    )
    sBegins = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_MAX_REQUESTS + 1),
        cute.make_layout(PLAN_MAX_REQUESTS + 1),
    )
    sScalars = cute.make_tensor(cute.arch.alloc_smem(cutlass.Int32, 2), cute.make_layout(2))
    for request_idx in cutlass.range(tidx, num_requests, _PLAN_THREADS, unroll=1):
        request_valid_k = cutlass.Int32(mValidK_l[request_idx])
        if cutlass.const_expr(write_generation_lengths):
            request_valid_k = (
                request_valid_k - cutlass.Int32(mPromptLens_l[request_idx]) + generation_len
            )
            mCorrectedKvLens_l[request_idx] = request_valid_k
            mGenerationLens_l[request_idx] = generation_len
        # KV tiles the request spans: its last query token's bound, at least one.
        bound = _packed_query_bound(
            mHelixKvBounds_l,
            request_idx,
            cutlass.Int32(query_len_per_seq - 1),
            request_valid_k,
            query_len_per_seq,
            use_helix_kv_bounds,
        )
        sTiles[request_idx] = cutlass.max(
            (bound + cutlass.Int32(KV_TILE - 1)) // cutlass.Int32(KV_TILE), cutlass.Int32(1)
        )
    cute.arch.sync_threads()
    if tidx < cutlass.Int32(32):
        total_tiles = cutlass.Int32(0)
        longest = cutlass.Int32(0)
        for request_idx in cutlass.range(tidx, num_requests, 32, unroll=1):
            request_tiles = sTiles[request_idx]
            total_tiles = total_tiles + request_tiles
            longest = cutlass.max(longest, request_tiles)
        total_tiles = cute.arch.warp_redux_sync(total_tiles, "add")
        longest = cute.arch.warp_redux_sync(longest, "max")
        tiles_per_task = longest
        if task_capacity > num_requests:
            share = cutlass.max(
                (total_tiles + wave_tasks - cutlass.Int32(1)) // wave_tasks,
                cutlass.Int32(PACKED_MIN_TILES_PER_TASK),
            )
            if longest > share + share // cutlass.Int32(2):
                target_tasks = wave_tasks
                if cutlass.const_expr(not use_helix_kv_bounds):
                    full_machine = wave_tasks * cutlass.Int32(3) <= num_requests * cutlass.Int32(4)
                    strong_skew = longest * num_requests > total_tiles + total_tiles
                    if full_machine or strong_skew:
                        target_tasks = wave_tasks + wave_tasks
                target = cutlass.max(
                    (total_tiles + target_tasks - cutlass.Int32(1)) // target_tasks,
                    cutlass.Int32(PACKED_MIN_TILES_PER_TASK),
                )
                pieces = (longest + longest + target) // (target + target)
                target = cutlass.max(
                    (longest + pieces - cutlass.Int32(1)) // pieces,
                    cutlass.Int32(PACKED_MIN_TILES_PER_TASK),
                )
                used_target = cutlass.Int32(0)
                for request_idx in cutlass.range(tidx, num_requests, 32, unroll=1):
                    used_target = (
                        used_target + (sTiles[request_idx] + target - cutlass.Int32(1)) // target
                    )
                used_target = cute.arch.warp_redux_sync(used_target, "add")
                if used_target <= task_capacity:
                    tiles_per_task = target
                else:
                    spare = task_capacity - num_requests
                    fallback = cutlass.max(
                        (total_tiles + spare - cutlass.Int32(1)) // spare,
                        cutlass.Int32(PACKED_MIN_TILES_PER_TASK),
                    )
                    tiles_per_task = fallback
                    splits = task_capacity // num_requests
                    even = cutlass.max(
                        (longest + splits - cutlass.Int32(1)) // splits,
                        cutlass.Int32(PACKED_MIN_TILES_PER_TASK),
                    )
                    if even < fallback:
                        used_even = cutlass.Int32(0)
                        for request_idx in cutlass.range(tidx, num_requests, 32, unroll=1):
                            used_even = (
                                used_even + (sTiles[request_idx] + even - cutlass.Int32(1)) // even
                            )
                        used_even = cute.arch.warp_redux_sync(used_even, "add")
                        if used_even <= task_capacity:
                            tiles_per_task = even
                tiles_per_task = cutlass.min(tiles_per_task, longest)
        tiles_per_task = cutlass.min(tiles_per_task, max_tiles_per_task)
        if tidx == cutlass.Int32(0):
            task_begin = cutlass.Int32(0)
            for request_idx in cutlass.range(0, num_requests, 1, unroll=1):
                sBegins[request_idx] = task_begin
                mPlan_i[cutlass.Int32(2) + request_idx] = task_begin
                task_begin = (
                    task_begin
                    + (sTiles[request_idx] + tiles_per_task - cutlass.Int32(1)) // tiles_per_task
                )
            sBegins[num_requests] = task_begin
            mPlan_i[cutlass.Int32(2) + num_requests] = task_begin
            mPlan_i[0] = tiles_per_task
            mPlan_i[1] = task_begin
            sScalars[0] = tiles_per_task
    cute.arch.sync_threads()
    tiles_per_task = sScalars[0]
    tables = cutlass.Int32(3) + num_requests
    for task_idx in cutlass.range(tidx, task_capacity, _PLAN_THREADS, unroll=1):
        request = num_requests - cutlass.Int32(1)
        split_idx = cutlass.Int32(1 << 15)
        request_tasks = cutlass.Int32(1 << 15)
        for request_idx in cutlass.range(0, num_requests, 1, unroll=1):
            begin = sBegins[request_idx]
            end = sBegins[request_idx + cutlass.Int32(1)]
            if task_idx >= begin and task_idx < end:
                request = request_idx
                split_idx = task_idx - begin
                request_tasks = end - begin
        mPlan_i[tables + task_idx] = request
        mPlan_i[tables + task_capacity + task_idx] = split_idx
        mPlan_i[tables + task_capacity + task_capacity + task_idx] = request_tasks


@cute.jit
def packed_plan_host(
    kv_lens_ptr: cute.Pointer,
    helix_bounds_ptr: cute.Pointer,
    prompt_lens_ptr: cute.Pointer,
    corrected_kv_lens_ptr: cute.Pointer,
    generation_lens_ptr: cute.Pointer,
    plan_ptr: cute.Pointer,
    stream: cuda.CUstream,
    num_requests: cutlass.Int32,
    task_capacity: cutlass.Int32,
    wave_tasks: cutlass.Int32,
    max_tiles_per_task: cutlass.Int32,
    generation_len: cutlass.Int32,
    query_len_per_seq: cutlass.Constexpr,
    use_helix_kv_bounds: cutlass.Constexpr,
    write_generation_lengths: cutlass.Constexpr,
) -> None:
    request_layout = cute.make_layout((num_requests,), stride=(1,))
    mValidK_l = cute.make_tensor(kv_lens_ptr, request_layout)
    mHelixKvBounds_l = cute.make_tensor(
        helix_bounds_ptr, cute.make_layout((num_requests * query_len_per_seq,), stride=(1,))
    )
    mPromptLens_l = cute.make_tensor(prompt_lens_ptr, request_layout)
    mCorrectedKvLens_l = cute.make_tensor(corrected_kv_lens_ptr, request_layout)
    mGenerationLens_l = cute.make_tensor(generation_lens_ptr, request_layout)
    mPlan_i = cute.make_tensor(
        plan_ptr,
        cute.make_layout((cutlass.Int32(3) + num_requests + 3 * task_capacity,), stride=(1,)),
    )
    packed_plan_kernel(
        mValidK_l,
        mHelixKvBounds_l,
        mPromptLens_l,
        mCorrectedKvLens_l,
        mGenerationLens_l,
        mPlan_i,
        num_requests,
        task_capacity,
        wave_tasks,
        max_tiles_per_task,
        generation_len,
        query_len_per_seq,
        use_helix_kv_bounds,
        write_generation_lengths,
    ).launch(grid=(1, 1, 1), block=(_PLAN_THREADS, 1, 1), stream=stream)


def packed_plan_length(num_requests: int, task_capacity: int) -> int:
    """Int32 entries of the plan buffer for ``num_requests`` and ``task_capacity``."""
    return 3 + num_requests + 3 * task_capacity


def launch_packed_plan(
    kv_lens: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_requests: int,
    num_splits: int,
    wave_tasks: int,
    max_tiles_per_task: int,
    query_len: int,
    helix_kv_bounds: torch.Tensor | None = None,
    generation_lengths: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
) -> None:
    """Write the balanced task plan for ``num_requests * num_splits`` slots into ``plan``.

    ``kv_lens`` (int32 per request) and the optional per-token int32
    ``helix_kv_bounds`` are the same bounds the decode kernel reads; see
    ``plan_packed_tasks`` for the host mirror of the plan. With
    ``generation_lengths = (prompt_lens, corrected_kv_lens, generation_lens,
    generation_len)`` the same launch first writes the per-request generation
    lengths ``kv_lens - prompt_lens + generation_len`` and ``generation_len``
    (``kv_lens`` then being the raw lengths) and plans on the corrected ones.
    """
    task_capacity = num_requests * num_splits
    if num_requests <= 0 or num_splits <= 0 or wave_tasks <= 0 or max_tiles_per_task <= 0:
        raise ValueError("plan needs positive request, split, wave and tile counts")
    if num_requests > PLAN_MAX_REQUESTS:
        raise ValueError(f"plan supports at most {PLAN_MAX_REQUESTS} requests, got {num_requests}")
    if query_len <= 0:
        raise ValueError("query_len must be positive")
    if kv_lens.dtype != torch.int32 or kv_lens.numel() < num_requests or kv_lens.stride(0) != 1:
        raise ValueError(f"kv_lens must be contiguous int32 with >= {num_requests} entries")
    if helix_kv_bounds is not None and (
        helix_kv_bounds.dtype != torch.int32
        or helix_kv_bounds.numel() < num_requests * query_len
        or helix_kv_bounds.stride(0) != 1
    ):
        raise ValueError(
            f"helix_kv_bounds must be contiguous int32 with >= {num_requests * query_len} entries"
        )
    plan_len = packed_plan_length(num_requests, task_capacity)
    if plan.dtype != torch.int32 or plan.numel() < plan_len or plan.stride(0) != 1:
        raise ValueError(f"plan must be contiguous int32 with >= {plan_len} entries")
    use_helix = helix_kv_bounds is not None
    bounds_ptr = helix_kv_bounds.data_ptr() if use_helix else kv_lens.data_ptr()
    write_lengths = generation_lengths is not None
    length_ptrs = (kv_lens.data_ptr(),) * 3
    generation_len = 0
    if write_lengths:
        prompt_lens, corrected_kv_lens, generation_lens, generation_len = generation_lengths
        for name, tensor in (
            ("prompt_lens", prompt_lens),
            ("corrected_kv_lens", corrected_kv_lens),
            ("generation_lens", generation_lens),
        ):
            if (
                tensor.dtype != torch.int32
                or tensor.numel() < num_requests
                or tensor.stride(0) != 1
            ):
                raise ValueError(f"{name} must be contiguous int32 with >= {num_requests} entries")
        if generation_len <= 0:
            raise ValueError("generation_len must be positive")
        length_ptrs = (
            prompt_lens.data_ptr(),
            corrected_kv_lens.data_ptr(),
            generation_lens.data_ptr(),
        )
    gmem = cute.AddressSpace.gmem
    ptrs = (
        make_ptr(cutlass.Int32, kv_lens.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, bounds_ptr, gmem, assumed_align=4),
        *(make_ptr(cutlass.Int32, ptr, gmem, assumed_align=4) for ptr in length_ptrs),
        make_ptr(cutlass.Int32, plan.data_ptr(), gmem, assumed_align=4),
    )
    stream = _current_cu_stream()
    cache_key = (query_len, use_helix, write_lengths)
    plan_fn = _PACKED_PLAN_COMPILE_CACHE.get(cache_key)
    if plan_fn is None:
        plan_fn = _compile_cutedsl(
            packed_plan_host,
            *ptrs,
            stream,
            *(cutlass.Int32(1) for _ in range(5)),
            query_len,
            use_helix,
            write_lengths,
        )
        _PACKED_PLAN_COMPILE_CACHE[cache_key] = plan_fn
    plan_fn(
        *ptrs,
        stream,
        cutlass.Int32(num_requests),
        cutlass.Int32(task_capacity),
        cutlass.Int32(wave_tasks),
        cutlass.Int32(max_tiles_per_task),
        cutlass.Int32(generation_len),
    )


def _make_reduce_ptrs(
    partial: torch.Tensor,
    maxima: torch.Tensor,
    sums: torch.Tensor,
    output: torch.Tensor,
    stats: torch.Tensor | None,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    plan: torch.Tensor,
) -> tuple[cute.Pointer, ...]:
    stats_max_ptr = stats.data_ptr() if stats is not None else maxima.data_ptr()
    stats_sum_ptr = (
        stats.data_ptr() + stats.element_size() if stats is not None else sums.data_ptr()
    )
    gmem = cute.AddressSpace.gmem
    return (
        make_ptr(cutlass.Float32, partial.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Float32, maxima.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Float32, sums.data_ptr(), gmem, assumed_align=32),
        make_ptr(_cutlass_output_dtype(output.dtype), output.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Float32, stats_max_ptr, gmem, assumed_align=4),
        make_ptr(cutlass.Float32, stats_sum_ptr, gmem, assumed_align=4),
        make_ptr(cutlass.Float32, q_scale.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Float32, kv_scale.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, plan.data_ptr(), gmem, assumed_align=4),
    )


def plan_packed_tasks(
    request_tiles: list[int],
    task_capacity: int,
    wave_tasks: int,
    max_tiles_per_task: int = 1 << 30,
    min_tiles_per_task: int = PACKED_MIN_TILES_PER_TASK,
    one_wave: bool = False,
) -> tuple[int, list[int]]:
    """Host mirror of the kernels' balanced task plan: (tiles per task, task begins).

    ``request_tiles`` are the per-request KV tile counts (>= 1), ``task_capacity``
    the grid.z slots (``num_requests * num_splits``) and ``wave_tasks`` the tasks
    that fill one wave of clusters; ``one_wave`` (Helix) never takes the two-wave
    target. Returns the KV tiles each task may own and the
    ``len(request_tiles) + 1`` task prefix, whose last element is the number of
    tasks actually used (<= ``task_capacity``). See ``_packed_tiles_per_task``.
    """
    num_requests = len(request_tiles)
    if num_requests <= 0 or task_capacity <= 0 or wave_tasks <= 0 or min(request_tiles) <= 0:
        raise ValueError("plan needs requests, task slots, wave tasks and positive tile counts")
    total = sum(request_tiles)
    longest = max(request_tiles)
    tiles_per_task = longest

    def used(tiles: int) -> int:
        return sum(math.ceil(t / tiles) for t in request_tiles)

    if task_capacity > num_requests:
        share = max(math.ceil(total / wave_tasks), min_tiles_per_task)
        if longest > share + share // 2:
            full_machine = 3 * wave_tasks <= 4 * num_requests
            strong_skew = longest * num_requests > 2 * total
            two_waves = not one_wave and (full_machine or strong_skew)
            target_tasks = 2 * wave_tasks if two_waves else wave_tasks
            target = max(math.ceil(total / target_tasks), min_tiles_per_task)
            pieces = (2 * longest + target) // (2 * target)
            target = max(math.ceil(longest / pieces), min_tiles_per_task)
            if used(target) <= task_capacity:
                tiles_per_task = target
            else:
                fallback = max(
                    math.ceil(total / (task_capacity - num_requests)), min_tiles_per_task
                )
                tiles_per_task = fallback
                even = max(math.ceil(longest / (task_capacity // num_requests)), min_tiles_per_task)
                if even < fallback and used(even) <= task_capacity:
                    tiles_per_task = even
            tiles_per_task = min(tiles_per_task, longest)
    tiles_per_task = min(tiles_per_task, max_tiles_per_task)
    begins = [0]
    for tiles in request_tiles:
        begins.append(begins[-1] + math.ceil(tiles / tiles_per_task))
    return tiles_per_task, begins


def request_kv_tiles(
    kv_lens: list[int], query_len: int, helix_kv_bounds: list[int] | None = None
) -> list[int]:
    """Per-request KV tile counts exactly as the kernels derive them."""
    tiles = []
    for request, kv_len in enumerate(kv_lens):
        bound = (
            helix_kv_bounds[request * query_len + query_len - 1]
            if helix_kv_bounds is not None
            else kv_len
        )
        tiles.append(max(math.ceil(max(bound, 0) / KV_TILE), 1))
    return tiles


def select_reduce_columns(num_rows: int, sm_count: int) -> int:
    """Column band per warp: narrower bands when few rows would leave SMs idle."""
    if num_rows >= 2 * sm_count:
        return 512
    if num_rows >= sm_count:
        return 256
    return 128


def select_reduce_ctas(num_items: int, sm_count: int) -> int:
    """Persistent CTAs: one warp per (row, band) item, up to four CTAs per SM."""
    warps_per_cta = _REDUCE_THREADS // 32
    return max(
        1, min((num_items + warps_per_cta - 1) // warps_per_cta, _REDUCE_CTAS_PER_SM * sm_count)
    )


def reduce_packed_splits(
    partial: torch.Tensor,
    maxima: torch.Tensor,
    sums: torch.Tensor,
    output: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    *,
    num_requests: int,
    num_splits: int,
    valid_m: int,
    physical_m: int,
    sm_scale: float,
    output_scale: float,
    plan: torch.Tensor,
    stats: torch.Tensor | None = None,
    sm_count: int = 128,
) -> None:
    """Reduce ``[splits * requests, physical_m, 512]`` FP32 partials into ``output``.

    ``output`` is a contiguous ``[requests * valid_m, 512]`` BF16/FP16 tensor and
    ``stats`` the optional contiguous ``[requests * valid_m, 2]`` FP32 Helix
    (max, sum) buffer. ``q_scale``/``kv_scale`` are the device FP4 global scales
    and ``plan`` the task plan written by ``launch_packed_plan`` for the same
    ``splits * requests`` slots the decode kernel ran.
    """
    tasks = num_splits * num_requests
    total_rows = num_requests * valid_m
    if num_requests <= 0 or num_splits <= 0 or valid_m <= 0 or physical_m < valid_m:
        raise ValueError("packed reduce needs positive requests/splits and physical_m >= valid_m")
    if partial.dtype != torch.float32 or partial.shape != (tasks, physical_m, _VALUE_DIM):
        raise ValueError(
            f"partial must be float32 {(tasks, physical_m, _VALUE_DIM)}, got {tuple(partial.shape)}"
        )
    if maxima.shape != (tasks, physical_m) or sums.shape != (tasks, physical_m):
        raise ValueError("maxima and sums must be [tasks, physical_m]")
    if maxima.dtype != torch.float32 or sums.dtype != torch.float32:
        raise TypeError("maxima and sums must be float32")
    if not (partial.is_contiguous() and maxima.is_contiguous() and sums.is_contiguous()):
        raise ValueError("partial, maxima, and sums must be contiguous")
    _validate_output_dtype(output.dtype)
    if output.shape != (total_rows, _VALUE_DIM) or not output.is_contiguous():
        raise ValueError(
            f"output must be contiguous [{total_rows}, {_VALUE_DIM}], got {tuple(output.shape)}"
        )
    if stats is not None and (
        stats.dtype != torch.float32 or stats.shape != (total_rows, 2) or not stats.is_contiguous()
    ):
        raise ValueError(f"stats must be contiguous float32 [{total_rows}, 2]")
    for name, tensor in (("q_scale", q_scale), ("kv_scale", kv_scale)):
        if tensor.dtype != torch.float32 or tensor.numel() != 1:
            raise TypeError(f"{name} must be a scalar float32 device tensor")
    plan_len = packed_plan_length(num_requests, tasks)
    if plan.dtype != torch.int32 or plan.numel() < plan_len or plan.stride(0) != 1:
        raise ValueError(f"plan must be contiguous int32 with >= {plan_len} entries")
    if not math.isfinite(sm_scale) or sm_scale <= 0.0 or not math.isfinite(output_scale):
        raise ValueError("sm_scale must be finite and positive and output_scale finite")
    for name, tensor in (
        ("partial", partial),
        ("maxima", maxima),
        ("sums", sums),
        ("output", output),
    ):
        _validate_tensor_pointer_alignment(name, tensor, alignment_bytes=32)
    if stats is not None:
        _validate_tensor_pointer_alignment("stats", stats, alignment_bytes=8)

    band_columns = select_reduce_columns(total_rows, sm_count)
    grid_ctas = select_reduce_ctas(total_rows * (_VALUE_DIM // band_columns), sm_count)
    write_stats = stats is not None
    stream = _current_cu_stream()
    ptrs = _make_reduce_ptrs(partial, maxima, sums, output, stats, q_scale, kv_scale, plan)
    cache_key = (band_columns, write_stats, output.dtype)
    reduce_fn = _PACKED_REDUCE_COMPILE_CACHE.get(cache_key)
    if reduce_fn is None:
        reduce_fn = _compile_cutedsl(
            packed_split_reduce_host,
            *ptrs,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            stream,
            *(cutlass.Int32(1) for _ in range(7)),
            band_columns,
            write_stats,
        )
        _PACKED_REDUCE_COMPILE_CACHE[cache_key] = reduce_fn
    reduce_fn(
        *ptrs,
        cutlass.Float32(sm_scale * LOG2_E),
        cutlass.Float32(output_scale),
        stream,
        cutlass.Int32(num_requests),
        cutlass.Int32(num_splits),
        cutlass.Int32(valid_m),
        cutlass.Int32(physical_m),
        cutlass.Int32(total_rows),
        cutlass.Int32(2 if stats is not None else 1),
        cutlass.Int32(grid_ctas),
    )
