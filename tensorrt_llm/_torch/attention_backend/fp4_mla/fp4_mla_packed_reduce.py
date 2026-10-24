# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL split reduction for packed FP4 MLA decode tasks.

The decode kernel writes one FP32 partial per task covering the request's
128-padded [query token, head] rows together with the raw row maximum and the
row sum relative to it. Tasks follow the balanced plan of the decode kernel
(consecutive tasks per request, more for longer KV), whose per-request task ranges
this kernel reads from the plan. It merges the tasks of every valid row into the
BF16/FP16 output and, for Helix, the per-row (max, sum) softmax stats. Global FP4
scales and KV lengths are read on the device, so CUDA graph replays follow them.
"""

import heapq
import math
import os
from collections.abc import Callable
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .fp4_mla_cutedsl_mufu16 import (
    INT32_MAX,
    KV_TILE,
    LOG2_E,
    PACKED_MIN_TILES_PER_TASK,
    PACKED_PLAN_PAGE_BEGIN_TABLE,
    PACKED_PLAN_PAGE_COUNT_TABLE,
    PACKED_PLAN_PARTIAL_SLOT_TABLE,
    PACKED_PLAN_TABLES,
    PACKED_PLAN_VALID_K_TABLE,
    _compile_cutedsl,
    _current_cu_stream,
    _cutlass_output_dtype,
    _packed_query_bound,
    _validate_output_dtype,
    _validate_tensor_pointer_alignment,
)

_PACKED_REDUCE_COMPILE_CACHE: dict[tuple[int, int, bool, torch.dtype, bool], Callable] = {}
_PACKED_PLAN_COMPILE_CACHE: dict[tuple[int, bool, bool, bool], Callable] = {}
_REDUCE_THREADS = 128
# One warp per candidate task length.
_PLAN_THREADS = 1024
# Requests one plan CTA stages in shared memory (4 KiB of tiles + begins).
PLAN_MAX_REQUESTS = 1024
# Off Helix the plan simulates the decode grid for this many task lengths, one warp each:
# the rule's (warp 0) and the longest request cut into 1..PLAN_SIM_CANDIDATES - 1 pieces.
PLAN_SIM_CANDIDATES = _PLAN_THREADS // 32
# Waves of more slots (four per lane) or batches of more requests keep the rule's length.
# 128 requests cover a DEP16 generation rank (max_batch_size 96): the rule keeps one task
# per request in an even batch that fills the machine, so 72 such requests ran on 35
# slots in three waves, the last of two tasks, as long as 96 requests did.
PLAN_SIM_MAX_SLOTS = 128
PLAN_SIM_MAX_REQUESTS = 128
# Candidates of more tasks are skipped: the simulation leaves out the reduction, whose
# cost grows with the partials.
PLAN_SIM_MAX_TASKS = 256
# Fixed cost of one task (Q load, pipeline fill, partial store) in KV tiles that the
# equal-length wave plan (TRTLLM_FP4_MLA_PACKED_VAR_PLAN=0) assumes.
PLAN_TASK_COST_TILES = 4
# Plan tables per task slot: request, split, count, first KV tile, KV tiles, partial slot,
# request KV length, request first page, request page count.
PLAN_TABLES = PACKED_PLAN_TABLES
# Simulated slot end times stay below this (KV tiles plus task costs); the earliest-free
# slot search packs (end, lane) into one int32 for a single warp reduction.
_PLAN_SLOT_END_LIMIT = (1 << 25) - 1
# Variable-length plan (TRTLLM_FP4_MLA_PACKED_VAR_PLAN, default on): warps
# 0..PLAN_V_SINGLE_CANDIDATES - 1 score equal-length plans, the others wrap-around
# plans for slot lengths ceil(total / wave_tasks) * (1 + j / 8), j < PLAN_V_WRAP_CANDIDATES.
PLAN_V_SINGLE_CANDIDATES = 16
PLAN_V_WRAP_CANDIDATES = PLAN_SIM_CANDIDATES - PLAN_V_SINGLE_CANDIDATES
# A wrap plan never cuts a piece shorter than this (4K tokens): its fixed cost dominates.
PLAN_V_MIN_PIECE_TILES = 8
# Pieces one wrap candidate may hold (the simulated task limit).
PLAN_V_MAX_PIECES = PLAN_SIM_MAX_TASKS
# Fitted task cost (KV tiles) of a tile-bound call, see packed_plan_costs.
PLAN_V_TASK_COST_TILES = 8
# Var-plan scores are in 1/64 KV tiles.
PLAN_V_COST_SCALE = 64
_VALUE_DIM = 512
# Partial loads one warp issues back-to-back before it consumes any: a chunk of
# tasks times the band's 128-column groups is 16 vector loads per lane for every
# band (4 tasks of a 512-column band, 16 of a 128-column band). The reduction is
# latency-bound, so this sets its memory-level parallelism; 32 loads in flight
# are slower (register pressure).
_TASKS_PER_CHUNK = 4
# Persistent reducer grid: up to this many CTAs per SM.
_REDUCE_CTAS_PER_SM = 4
_REDUCE_WARPS_PER_CTA = _REDUCE_THREADS // 32
# The tasks of a (row, band) item may be cut into up to _REDUCE_WARPS_PER_CTA contiguous
# groups, one warp each, whose partial sums the group-0 warp adds up through shared
# memory. That pays off only for long task lists over few output rows. Taking four groups
# up to _REDUCE_FOUR_GROUP_ROWS output rows and two up to _REDUCE_TWO_GROUP_ROWS when a
# request has at least _REDUCE_GROUP_MIN_TASK_SLOTS task slots (one or two requests), and
# one warp per item otherwise, stays close to the best choice over the head, query, request
# and context-length ranges that were swept; four groups lose at large row counts.
_REDUCE_GROUP_MIN_TASK_SLOTS = 64
_REDUCE_FOUR_GROUP_ROWS = 48
_REDUCE_TWO_GROUP_ROWS = 96


@cute.jit
def _zero_f32x4():
    """A fresh 4 x FP32 zero vector (a new object every call)."""
    return cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(4)), dtype=cutlass.Float32
    )


@cute.jit
def _load_reduce_chunk(
    gPartial_arr,
    mRowMax_tr: cute.Tensor,
    mRowSum_tr: cute.Tensor,
    local_row: cutlass.Int32,
    task_begin: cutlass.Int32,
    chunk_base: cutlass.Int32,
    group_end: cutlass.Int32,
    physical_m: cutlass.Int32,
    band_base: cutlass.Int32,
    band_columns: cutlass.Constexpr,
) -> tuple:
    """Issue the row state and partial loads of one chunk of tasks.

    The loads of the whole chunk are in flight at once; slots past the warp's
    share of the tasks (a warp-uniform predicate) load nothing and get zero
    weight.
    """
    column_groups: cutlass.Constexpr = band_columns // 128
    tasks_per_chunk: cutlass.Constexpr = _TASKS_PER_CHUNK * (_VALUE_DIM // band_columns)
    row_sums = []
    row_maxima = []
    vecs = []
    for slot in cutlass.range_constexpr(tasks_per_chunk):
        task_idx = chunk_base + cutlass.Int32(slot)
        task = task_begin + task_idx
        row_sum = cutlass.Float32(0.0)
        row_max = cutlass.Float32(0.0)
        # One object per name: the DSL tracks rebinding per object, so aliases of
        # one zero vector would not see each other's loads.
        vec0 = _zero_f32x4()
        vec1 = _zero_f32x4()
        vec2 = _zero_f32x4()
        vec3 = _zero_f32x4()
        if task_idx < group_end:
            row_sum = mRowSum_tr[task, local_row]
            row_max = mRowMax_tr[task, local_row]
            partial_row = (task * physical_m + local_row) * cutlass.Int32(_VALUE_DIM) + band_base
            vec0 = gPartial_arr.subview(partial_row).data_ptr().load(count=4, alignment=16)
            if cutlass.const_expr(column_groups > 1):
                vec1 = (
                    gPartial_arr.subview(partial_row + cutlass.Int32(128))
                    .data_ptr()
                    .load(count=4, alignment=16)
                )
            if cutlass.const_expr(column_groups > 2):
                vec2 = (
                    gPartial_arr.subview(partial_row + cutlass.Int32(256))
                    .data_ptr()
                    .load(count=4, alignment=16)
                )
                vec3 = (
                    gPartial_arr.subview(partial_row + cutlass.Int32(384))
                    .data_ptr()
                    .load(count=4, alignment=16)
                )
        row_sums.append(row_sum)
        row_maxima.append(row_max)
        vecs.append(vec0)
        if cutlass.const_expr(column_groups > 1):
            vecs.append(vec1)
        if cutlass.const_expr(column_groups > 2):
            vecs.append(vec2)
            vecs.append(vec3)
    return row_sums, row_maxima, vecs


@cute.jit
def _accumulate_reduce_chunk(
    numerator0,
    numerator1,
    numerator2,
    numerator3,
    denominator: cutlass.Float32,
    row_sums: list,
    row_maxima: list,
    vecs: list,
    chunk_base: cutlass.Int32,
    group_end: cutlass.Int32,
    scale: cutlass.Float32,
    global_row_max: cutlass.Float32,
    band_columns: cutlass.Constexpr,
) -> tuple:
    """Merge one loaded chunk in task order: weight, denominator, numerator."""
    column_groups: cutlass.Constexpr = band_columns // 128
    tasks_per_chunk: cutlass.Constexpr = _TASKS_PER_CHUNK * (_VALUE_DIM // band_columns)
    for slot in cutlass.range_constexpr(tasks_per_chunk):
        task_weight = cutlass.Float32(0.0)
        if chunk_base + cutlass.Int32(slot) < group_end:
            if row_sums[slot] != cutlass.Float32(0.0):
                task_weight = cute.exp2(row_maxima[slot] * scale - global_row_max, fastmath=True)
                denominator += row_sums[slot] * task_weight
        # Empty tasks keep zero weight; their payload is never trusted.
        if task_weight != cutlass.Float32(0.0):
            numerator0 = numerator0 + vecs[slot * column_groups] * task_weight
            if cutlass.const_expr(column_groups > 1):
                numerator1 = numerator1 + vecs[slot * column_groups + 1] * task_weight
            if cutlass.const_expr(column_groups > 2):
                numerator2 = numerator2 + vecs[slot * column_groups + 2] * task_weight
                numerator3 = numerator3 + vecs[slot * column_groups + 3] * task_weight
    return numerator0, numerator1, numerator2, numerator3, denominator


@cute.jit
def _reduce_packed_row(
    mPartial_trc: cute.Tensor,
    mRowMax_tr: cute.Tensor,
    mRowSum_tr: cute.Tensor,
    mOut_rc: cute.Tensor,
    mStatsMax_r: cute.Tensor,
    mStatsSum_r: cute.Tensor,
    sNum_arr,
    sDen,
    lane: cutlass.Int32,
    item_slot: cutlass.Int32,
    task_group: cutlass.Int32,
    row: cutlass.Int32,
    local_row: cutlass.Int32,
    band: cutlass.Int32,
    task_begin: cutlass.Int32,
    request_tasks: cutlass.Int32,
    physical_m: cutlass.Int32,
    scale: cutlass.Float32,
    output_scale: cutlass.Float32,
    band_columns: cutlass.Constexpr,
    task_groups: cutlass.Constexpr,
    write_stats: cutlass.Constexpr,
) -> None:
    """``task_groups`` warps merge the planned tasks of one (output row, column band).

    Lane ``l`` owns columns ``band * band_columns + i * 128 + 4 * l .. + 4`` for
    ``i < band_columns // 128``, so every vector load and store of the warp
    covers 512 contiguous bytes. Warp ``g`` of the item takes the contiguous
    share ``[g * tasks // groups, (g + 1) * tasks // groups)`` of the tasks. The
    first chunk's loads are issued before the scan for the row maximum, so a
    request of at most one chunk per warp costs one dependent load round. The
    per-task weights are recomputed by every lane from the row state (one
    L1-resident broadcast load each). Empty tasks (row sum 0) carry zero weight
    and a row with no visible key anywhere yields O=0 and stats (-inf, 0), the
    neutral Helix contribution.
    """
    column_groups: cutlass.Constexpr = band_columns // 128
    tasks_per_chunk: cutlass.Constexpr = _TASKS_PER_CHUNK * (_VALUE_DIM // band_columns)
    band_base = band * cutlass.Int32(band_columns) + lane * cutlass.Int32(4)
    gPartial_arr = cutlass.make_array_view(mPartial_trc)
    gOut_arr = cutlass.make_array_view(mOut_rc)
    group_begin = (task_group * request_tasks) // cutlass.Int32(task_groups)
    group_end = ((task_group + cutlass.Int32(1)) * request_tasks) // cutlass.Int32(task_groups)
    first_sums, first_maxima, first_vecs = _load_reduce_chunk(
        gPartial_arr,
        mRowMax_tr,
        mRowSum_tr,
        local_row,
        task_begin,
        group_begin,
        group_end,
        physical_m,
        band_base,
        band_columns,
    )
    global_row_max = cutlass.Float32(-cutlass.Float32.inf)
    for task_idx in cutlass.range(lane, request_tasks, 32, unroll=1):
        task = task_begin + task_idx
        task_row_sum = mRowSum_tr[task, local_row]
        task_row_max = mRowMax_tr[task, local_row]
        if task_row_sum != cutlass.Float32(0.0):
            global_row_max = cute.arch.fmax(global_row_max, task_row_max * scale)
    global_row_max = cute.arch.warp_redux_sync(global_row_max, "fmax")

    # Four independent accumulators (the DSL tracks rebinding per object).
    numerator0 = _zero_f32x4()
    numerator1 = _zero_f32x4()
    numerator2 = _zero_f32x4()
    numerator3 = _zero_f32x4()
    denominator = cutlass.Float32(0.0)
    numerator0, numerator1, numerator2, numerator3, denominator = _accumulate_reduce_chunk(
        numerator0,
        numerator1,
        numerator2,
        numerator3,
        denominator,
        first_sums,
        first_maxima,
        first_vecs,
        group_begin,
        group_end,
        scale,
        global_row_max,
        band_columns,
    )
    for chunk_base in cutlass.range(
        group_begin + cutlass.Int32(tasks_per_chunk), group_end, tasks_per_chunk, unroll=1
    ):
        chunk_sums, chunk_maxima, chunk_vecs = _load_reduce_chunk(
            gPartial_arr,
            mRowMax_tr,
            mRowSum_tr,
            local_row,
            task_begin,
            chunk_base,
            group_end,
            physical_m,
            band_base,
            band_columns,
        )
        numerator0, numerator1, numerator2, numerator3, denominator = _accumulate_reduce_chunk(
            numerator0,
            numerator1,
            numerator2,
            numerator3,
            denominator,
            chunk_sums,
            chunk_maxima,
            chunk_vecs,
            chunk_base,
            group_end,
            scale,
            global_row_max,
            band_columns,
        )

    if cutlass.const_expr(task_groups > 1):
        # Groups 1.. publish their sums; group 0 adds them in group order. The
        # second barrier keeps the buffer until group 0 has read it.
        barrier_id = item_slot + cutlass.Int32(1)
        slot_warp = item_slot * cutlass.Int32(task_groups)
        if task_group != cutlass.Int32(0):
            num_offset = (slot_warp + task_group) * cutlass.Int32(
                band_columns
            ) + lane * cutlass.Int32(4)
            sNum_arr.store(numerator0, idx=num_offset, vector_size=4, alignment=16)
            if cutlass.const_expr(column_groups > 1):
                sNum_arr.store(
                    numerator1, idx=num_offset + cutlass.Int32(128), vector_size=4, alignment=16
                )
            if cutlass.const_expr(column_groups > 2):
                sNum_arr.store(
                    numerator2, idx=num_offset + cutlass.Int32(256), vector_size=4, alignment=16
                )
                sNum_arr.store(
                    numerator3, idx=num_offset + cutlass.Int32(384), vector_size=4, alignment=16
                )
            if lane == cutlass.Int32(0):
                sDen[slot_warp + task_group] = denominator
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=task_groups * 32)
        if task_group == cutlass.Int32(0):
            for group in cutlass.range_constexpr(1, task_groups):
                num_offset = (slot_warp + cutlass.Int32(group)) * cutlass.Int32(
                    band_columns
                ) + lane * cutlass.Int32(4)
                numerator0 = numerator0 + sNum_arr.subview(num_offset).data_ptr().load(
                    count=4, alignment=16
                )
                if cutlass.const_expr(column_groups > 1):
                    numerator1 = numerator1 + sNum_arr.subview(
                        num_offset + cutlass.Int32(128)
                    ).data_ptr().load(count=4, alignment=16)
                if cutlass.const_expr(column_groups > 2):
                    numerator2 = numerator2 + sNum_arr.subview(
                        num_offset + cutlass.Int32(256)
                    ).data_ptr().load(count=4, alignment=16)
                    numerator3 = numerator3 + sNum_arr.subview(
                        num_offset + cutlass.Int32(384)
                    ).data_ptr().load(count=4, alignment=16)
                denominator += sDen[slot_warp + cutlass.Int32(group)]
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=task_groups * 32)

    if task_group == cutlass.Int32(0):
        normalizer = cutlass.Float32(0.0)
        if denominator != cutlass.Float32(0.0):
            normalizer = output_scale / denominator
        out_row = row * cutlass.Int32(_VALUE_DIM) + band_base
        gOut_arr.store(
            (numerator0 * normalizer).to(mOut_rc.element_type),
            idx=out_row,
            vector_size=4,
            alignment=8,
        )
        if cutlass.const_expr(column_groups > 1):
            gOut_arr.store(
                (numerator1 * normalizer).to(mOut_rc.element_type),
                idx=out_row + cutlass.Int32(128),
                vector_size=4,
                alignment=8,
            )
        if cutlass.const_expr(column_groups > 2):
            gOut_arr.store(
                (numerator2 * normalizer).to(mOut_rc.element_type),
                idx=out_row + cutlass.Int32(256),
                vector_size=4,
                alignment=8,
            )
            gOut_arr.store(
                (numerator3 * normalizer).to(mOut_rc.element_type),
                idx=out_row + cutlass.Int32(384),
                vector_size=4,
                alignment=8,
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


@cute.jit
def _item_task_range(
    mPlan_i: cute.Tensor, item: cutlass.Int32, valid_m: cutlass.Int32, bands: cutlass.Constexpr
) -> tuple:
    """The plan's [first task, end) of the request that owns (row, band) item ``item``."""
    request = item // cutlass.Int32(bands) // valid_m
    return mPlan_i[cutlass.Int32(2) + request], mPlan_i[cutlass.Int32(3) + request]


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
    task_groups: cutlass.Constexpr,
    write_stats: cutlass.Constexpr,
) -> None:
    """Persistent grid over the (row, column band) items of the whole batch.

    Item ``i`` is batch row ``i // bands`` and band ``i % bands``; every CTA step
    takes ``_REDUCE_WARPS_PER_CTA // task_groups`` consecutive items with
    ``task_groups`` warps each. An item reads its request's task range straight
    from the plan, so mapping it needs no scan over the batch; each warp fetches
    the next step's range before merging the current item. A single-task request
    was stored by the decode kernel itself and, for Helix, only needs its (max,
    sum) stats derived from its row state.

    The kernel is a programmatic dependent of the decode kernel. The plan and the
    global scales come from kernels that completed before the decode kernel
    started, so the first item's task range and the scales are read before
    ``griddepcontrol.wait``; partials and row state only after it.
    """
    bands: cutlass.Constexpr = _VALUE_DIM // band_columns
    items_per_step: cutlass.Constexpr = _REDUCE_WARPS_PER_CTA // task_groups
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = tidx // cutlass.Int32(32)
    lane = tidx % cutlass.Int32(32)
    item_slot = warp_idx // cutlass.Int32(task_groups)
    task_group = warp_idx - item_slot * cutlass.Int32(task_groups)
    sNum_arr = None
    sDen = None
    if cutlass.const_expr(task_groups > 1):
        sNum = cute.make_tensor(
            cute.arch.alloc_smem(
                cutlass.Float32, _REDUCE_WARPS_PER_CTA * band_columns, alignment=16
            ),
            cute.make_layout((_REDUCE_WARPS_PER_CTA * band_columns,), stride=(1,)),
        )
        sNum_arr = cutlass.make_array_view(sNum)
        sDen = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Float32, _REDUCE_WARPS_PER_CTA),
            cute.make_layout(_REDUCE_WARPS_PER_CTA),
        )
    scale = softmax_scale_log2 / (mQScale[0] * mKvScale[0])
    total_items = num_requests * valid_m * cutlass.Int32(bands)
    num_steps = (total_items + cutlass.Int32(items_per_step - 1)) // cutlass.Int32(items_per_step)
    item_stride = grid_x * cutlass.Int32(items_per_step)
    next_begin = cutlass.Int32(0)
    next_end = cutlass.Int32(0)
    first_item = bidx * cutlass.Int32(items_per_step) + item_slot
    if first_item < total_items:
        next_begin, next_end = _item_task_range(mPlan_i, first_item, valid_m, bands)
    cute.arch.griddepcontrol_wait()
    for step in cutlass.range(bidx, num_steps, grid_x, unroll=1):
        item = step * cutlass.Int32(items_per_step) + item_slot
        task_begin = next_begin
        request_tasks = next_end - next_begin
        if item + item_stride < total_items:
            next_begin, next_end = _item_task_range(mPlan_i, item + item_stride, valid_m, bands)
        if item < total_items:
            row = item // cutlass.Int32(bands)
            band = item - row * cutlass.Int32(bands)
            request = row // valid_m
            local_row = row - request * valid_m
            if request_tasks > cutlass.Int32(1):
                _reduce_packed_row(
                    mPartial_trc,
                    mRowMax_tr,
                    mRowSum_tr,
                    mOut_rc,
                    mStatsMax_r,
                    mStatsSum_r,
                    sNum_arr,
                    sDen,
                    lane,
                    item_slot,
                    task_group,
                    row,
                    local_row,
                    band,
                    task_begin,
                    request_tasks,
                    physical_m,
                    scale,
                    output_scale,
                    band_columns,
                    task_groups,
                    write_stats,
                )
            else:
                if cutlass.const_expr(write_stats):
                    if (
                        band == cutlass.Int32(0)
                        and task_group == cutlass.Int32(0)
                        and lane == cutlass.Int32(0)
                    ):
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
    task_groups: cutlass.Constexpr,
    write_stats: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
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
        plan_ptr,
        cute.make_layout((cutlass.Int32(3) + num_requests + PLAN_TABLES * tasks,), stride=(1,)),
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
        task_groups,
        write_stats,
    ).launch(
        grid=(grid_ctas, 1, 1),
        block=(_REDUCE_THREADS, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.jit
def _plan_take_earliest_slot(
    slot0: cutlass.Int32,
    slot1: cutlass.Int32,
    slot2: cutlass.Int32,
    slot3: cutlass.Int32,
    cost: cutlass.Int32,
    lane: cutlass.Int32,
) -> tuple:
    """Start a ``cost``-tile task on the slot that frees first (ties: lowest lane, then the
    lane's lowest register); returns the updated slots and the start time. One warp
    reduction finds both the earliest end and its owner lane."""
    local_free = cutlass.min(cutlass.min(slot0, slot1), cutlass.min(slot2, slot3))
    slot_key = cute.arch.warp_redux_sync(
        cutlass.min(local_free, cutlass.Int32(_PLAN_SLOT_END_LIMIT)) * cutlass.Int32(32) + lane,
        "min",
    )
    start = slot_key // cutlass.Int32(32)
    if lane == slot_key % cutlass.Int32(32):
        if slot0 == start:
            slot0 = start + cost
        elif slot1 == start:
            slot1 = start + cost
        elif slot2 == start:
            slot2 = start + cost
        else:
            slot3 = start + cost
    return slot0, slot1, slot2, slot3, start


@cute.jit
def _plan_simulated_makespan(
    sTiles: cute.Tensor,
    num_requests: cutlass.Int32,
    tiles_per_task: cutlass.Int32,
    wave_tasks: cutlass.Int32,
    task_cost_tiles: cutlass.Int32,
    lane: cutlass.Int32,
) -> cutlass.Int32:
    """Makespan (KV tiles) of the decode grid for one task length, simulated by one warp.

    The grid runs the tasks in the plan's table order (``plan_task_order``): every
    request's tasks but its last, all ``tiles_per_task`` long, then the last tasks,
    longest first. Each task takes the slot of ``wave_tasks`` that frees first and
    holds it for its KV tiles plus ``task_cost_tiles``. Slot j lives in lane
    j % 32, register j // 32; request r's last task in lane r % 32, bit r // 32.
    """
    task_cost = tiles_per_task + task_cost_tiles
    full_tasks = cutlass.Int32(0)
    for request_idx in cutlass.range(lane, num_requests, 32, unroll=1):
        full_tasks = (
            full_tasks
            + (sTiles[request_idx] + tiles_per_task - cutlass.Int32(1)) // tiles_per_task
            - cutlass.Int32(1)
        )
    full_tasks = cute.arch.warp_redux_sync(full_tasks, "add")
    # Equal tasks fill the slots round-robin.
    full_rounds = full_tasks // wave_tasks
    full_extra = full_tasks - full_rounds * wave_tasks
    idle = cutlass.Int32(1 << 30)
    slot0 = idle
    slot1 = idle
    slot2 = idle
    slot3 = idle
    if lane < wave_tasks:
        slot0 = full_rounds * task_cost
        if lane < full_extra:
            slot0 = slot0 + task_cost
    if lane + cutlass.Int32(32) < wave_tasks:
        slot1 = full_rounds * task_cost
        if lane + cutlass.Int32(32) < full_extra:
            slot1 = slot1 + task_cost
    if lane + cutlass.Int32(64) < wave_tasks:
        slot2 = full_rounds * task_cost
        if lane + cutlass.Int32(64) < full_extra:
            slot2 = slot2 + task_cost
    if lane + cutlass.Int32(96) < wave_tasks:
        slot3 = full_rounds * task_cost
        if lane + cutlass.Int32(96) < full_extra:
            slot3 = slot3 + task_cost
    makespan = full_rounds * task_cost
    if full_extra > cutlass.Int32(0):
        makespan = makespan + task_cost
    taken = cutlass.Int32(0)
    # Every last task gets a slot of its own when the full tasks leave enough slots
    # idle: each then starts at 0, so only the longest one matters.
    rounds = num_requests
    if full_tasks + num_requests <= wave_tasks:
        rounds = cutlass.Int32(0)
        local_tail = cutlass.Int32(0)
        for request_idx in cutlass.range(lane, num_requests, 32, unroll=1):
            request_tiles = sTiles[request_idx]
            request_tasks = (request_tiles + tiles_per_task - cutlass.Int32(1)) // tiles_per_task
            local_tail = cutlass.max(
                local_tail, request_tiles - (request_tasks - cutlass.Int32(1)) * tiles_per_task
            )
        makespan = cutlass.max(
            makespan, cute.arch.warp_redux_sync(local_tail, "max") + task_cost_tiles
        )
    for _ in cutlass.range(0, rounds, 1, unroll=1):
        # The longest last task left (ties: lower request), as tail * 1024 + (1023 - request).
        local_key = cutlass.Int32(-1)
        for request_idx in cutlass.range(lane, num_requests, 32, unroll=1):
            taken_bit = (taken >> (request_idx // cutlass.Int32(32))) & cutlass.Int32(1)
            if taken_bit == cutlass.Int32(0):
                request_tiles = sTiles[request_idx]
                request_tasks = (
                    request_tiles + tiles_per_task - cutlass.Int32(1)
                ) // tiles_per_task
                tail = request_tiles - (request_tasks - cutlass.Int32(1)) * tiles_per_task
                local_key = cutlass.max(
                    local_key,
                    tail * cutlass.Int32(PLAN_MAX_REQUESTS)
                    + cutlass.Int32(PLAN_MAX_REQUESTS - 1)
                    - request_idx,
                )
        key = cute.arch.warp_redux_sync(local_key, "max")
        request = cutlass.Int32(PLAN_MAX_REQUESTS - 1) - key % cutlass.Int32(PLAN_MAX_REQUESTS)
        if lane == request % cutlass.Int32(32):
            taken = taken | (cutlass.Int32(1) << (request // cutlass.Int32(32)))
        tail_cost = key // cutlass.Int32(PLAN_MAX_REQUESTS) + task_cost_tiles
        slot0, slot1, slot2, slot3, start = _plan_take_earliest_slot(
            slot0, slot1, slot2, slot3, tail_cost, lane
        )
        makespan = cutlass.max(makespan, start + tail_cost)
    return makespan


@cute.jit
def _plan_order_requests(
    sTiles: cute.Tensor,
    sOrder: cute.Tensor,
    num_requests: cutlass.Int32,
    tidx: cutlass.Int32,
) -> None:
    """``sOrder[rank] = request``, ranked by KV tiles descending, ties by lower request."""
    for request_idx in cutlass.range(tidx, num_requests, _PLAN_THREADS, unroll=1):
        request_tiles = sTiles[request_idx]
        rank = cutlass.Int32(0)
        for other_idx in cutlass.range(0, num_requests, 1, unroll=1):
            other_tiles = sTiles[other_idx]
            if (other_tiles > request_tiles) | (
                (other_tiles == request_tiles) & (other_idx < request_idx)
            ):
                rank = rank + cutlass.Int32(1)
        sOrder[rank] = request_idx


@cute.jit
def _plan_wrap_walk(
    sTiles: cute.Tensor,
    sOrder: cute.Tensor,
    num_requests: cutlass.Int32,
    chunk_tiles: cutlass.Int32,
    sPieces: cute.Tensor,
    row: cutlass.Int32,
    sCount: cute.Tensor,
    writer: cutlass.Boolean,
    materialize: cutlass.Constexpr,
) -> tuple:
    """Wrap-around (McNaughton) cut of the batch into pieces for ``chunk_tiles``-tile slots.

    The requests, longest first (``sOrder``), are laid end to end and cut every
    ``chunk_tiles`` tiles, so every slot gets about the same work and at most one
    request per slot boundary is split. A head piece shorter than
    PLAN_V_MIN_PIECE_TILES at a boundary moves the request to the next slot, and a
    shorter remainder joins its piece (which may then exceed the chunk). Every lane
    walks the same pieces; ``writer`` stores them: piece lengths in row ``row`` of
    ``sPieces``, or with ``materialize`` each piece's request, split, first tile and
    length in rows 0..3 plus each request's piece count in ``sCount``. Returns
    (pieces, tasks of split requests, longest piece); the host mirror is
    ``plan_wrap_pieces``.
    """
    min_piece = cutlass.Int32(PLAN_V_MIN_PIECE_TILES)
    max_pieces = cutlass.Int32(PLAN_V_MAX_PIECES)
    pieces = cutlass.Int32(0)
    partial_tasks = cutlass.Int32(0)
    longest_piece = cutlass.Int32(0)
    fill = cutlass.Int32(0)
    for rank in cutlass.range(0, num_requests, 1, unroll=1):
        request = sOrder[rank]
        request_tiles = sTiles[request]
        begin = cutlass.Int32(0)
        request_pieces = cutlass.Int32(0)
        # Each step either emits a piece or (once per piece) closes the current slot.
        steps = cutlass.Int32(2) * (request_tiles // chunk_tiles) + cutlass.Int32(6)
        for _ in cutlass.range(0, steps, 1, unroll=1):
            if begin < request_tiles:
                remaining = request_tiles - begin
                take = cutlass.min(chunk_tiles - fill, remaining)
                if (take < remaining) & (take < min_piece) & (fill > cutlass.Int32(0)):
                    fill = cutlass.Int32(0)
                else:
                    rest = remaining - take
                    if (rest > cutlass.Int32(0)) & (rest < min_piece):
                        take = remaining
                    if writer & (pieces < max_pieces):
                        if cutlass.const_expr(materialize):
                            sPieces[0, pieces] = request
                            sPieces[1, pieces] = request_pieces
                            sPieces[2, pieces] = begin
                            sPieces[3, pieces] = take
                        else:
                            sPieces[row, pieces] = take
                    pieces = pieces + cutlass.Int32(1)
                    request_pieces = request_pieces + cutlass.Int32(1)
                    longest_piece = cutlass.max(longest_piece, take)
                    begin = begin + take
                    fill = fill + take
                    if fill >= chunk_tiles:
                        fill = cutlass.Int32(0)
        if request_pieces > cutlass.Int32(1):
            partial_tasks = partial_tasks + request_pieces
        if cutlass.const_expr(materialize):
            if writer:
                sCount[request] = request_pieces
    return pieces, partial_tasks, longest_piece


@cute.jit
def _plan_lpt_makespan(
    sPieces: cute.Tensor,
    row: cutlass.Int32,
    num_pieces: cutlass.Int32,
    wave_tasks: cutlass.Int32,
    task_cost_tiles: cutlass.Int32,
    lane: cutlass.Int32,
) -> cutlass.Int32:
    """Makespan (KV tiles) of ``num_pieces`` (<= PLAN_V_MAX_PIECES) tasks of lengths
    ``sPieces[row, :]`` run longest first (ties: lower index) on ``wave_tasks`` (<= 128) slots,
    one warp. Piece i lives in lane i % 32, bit i // 32; slot j in lane j % 32,
    register j // 32. Host mirror: ``plan_lpt_makespan``.
    """
    idle = cutlass.Int32(1 << 30)
    slot0 = idle
    slot1 = idle
    slot2 = idle
    slot3 = idle
    if lane < wave_tasks:
        slot0 = cutlass.Int32(0)
    if lane + cutlass.Int32(32) < wave_tasks:
        slot1 = cutlass.Int32(0)
    if lane + cutlass.Int32(64) < wave_tasks:
        slot2 = cutlass.Int32(0)
    if lane + cutlass.Int32(96) < wave_tasks:
        slot3 = cutlass.Int32(0)
    makespan = cutlass.Int32(0)
    taken = cutlass.Int32(0)
    index_span = cutlass.Int32(PLAN_V_MAX_PIECES)
    # No more pieces than slots: every piece starts at 0 on a slot of its own.
    rounds = num_pieces
    if num_pieces <= wave_tasks:
        rounds = cutlass.Int32(0)
        local_piece = cutlass.Int32(0)
        for slot in cutlass.range_constexpr(PLAN_V_MAX_PIECES // 32):
            piece = lane + cutlass.Int32(32 * slot)
            if piece < num_pieces:
                local_piece = cutlass.max(local_piece, sPieces[row, piece])
        longest = cute.arch.warp_redux_sync(local_piece, "max")
        if num_pieces > cutlass.Int32(0):
            makespan = longest + task_cost_tiles
    for _ in cutlass.range(0, rounds, 1, unroll=1):
        local_key = cutlass.Int32(-1)
        for slot in cutlass.range_constexpr(PLAN_V_MAX_PIECES // 32):
            piece = lane + cutlass.Int32(32 * slot)
            if piece < num_pieces:
                if ((taken >> cutlass.Int32(slot)) & cutlass.Int32(1)) == cutlass.Int32(0):
                    local_key = cutlass.max(
                        local_key,
                        sPieces[row, piece] * index_span + index_span - cutlass.Int32(1) - piece,
                    )
        key = cute.arch.warp_redux_sync(local_key, "max")
        piece = index_span - cutlass.Int32(1) - key % index_span
        if lane == piece % cutlass.Int32(32):
            taken = taken | (cutlass.Int32(1) << (piece // cutlass.Int32(32)))
        cost = key // index_span + task_cost_tiles
        slot0, slot1, slot2, slot3, start = _plan_take_earliest_slot(
            slot0, slot1, slot2, slot3, cost, lane
        )
        makespan = cutlass.max(makespan, start + cost)
    return makespan


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
    task_cost_tiles: cutlass.Int32,
    partial_cost64: cutlass.Int32,
    page_stride: cutlass.Int32,
    query_len_per_seq: cutlass.Constexpr,
    use_helix_kv_bounds: cutlass.Constexpr,
    write_generation_lengths: cutlass.Constexpr,
    wave_plan: cutlass.Constexpr,
    var_plan: cutlass.Constexpr,
) -> None:
    """One CTA: the balanced task plan both decode and reduction kernels read.

    With ``write_generation_lengths`` the CTA first derives the per-step
    generation lengths every FP4 MLA kernel of the step reads (``kv_lens -
    prompt_lens + generation_len`` and ``generation_len`` per request) and
    plans on those, so the first FP4 MLA kernel of a step pays one launch for
    both; otherwise ``mValidK_l`` already holds the corrected lengths.

    ``plan = [tiles_per_task, used_tasks, begins[0..num_requests], request[C],
    split[C], count[C], tile_begin[C], tile_count[C], partial_slot[C], valid_k[C],
    page_begin[C], page_count[C]]`` (C = ``task_capacity``). Task t (grid order)
    owns KV tiles ``[tile_begin[t], tile_begin[t] + tile_count[t])`` of request
    ``request[t]``; it is split ``split[t]`` of the request's ``count[t]`` tasks,
    whose partials sit at ``partial_slot[t] = begins[r] + split``. ``valid_k``
    repeats the request's KV length (the corrected one when the lengths are
    written here). With a fixed-stride page table (``page_stride > 0``: request
    r pages through ``[r * page_stride, (r + 1) * page_stride)``, the window the
    device page table's decode indptr holds) ``page_begin`` / ``page_count`` are
    that window, so the decode kernel reads its whole task in one load round;
    otherwise ``page_count`` is -1 and the decode kernel reads its indptr. The
    window comes from the stride rather than the indptr because this kernel may
    run before the step's page table is materialized.
    ``tiles_per_task`` is the longest task. Slots past ``used_tasks`` map to the
    last request with an out-of-range split and no tiles and are never scheduled
    (the decode kernel exits them).

    Task length rule: a request is only split when it is at least 1.5x longer than
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
    host sizes it so the clamp never needs more than the capacity). Every
    request's tasks are then equal except its last.

    Off Helix that rule is only a candidate, scored by simulating the grid (one
    warp per candidate) and the cheapest is taken:
    - ``wave_plan`` without ``var_plan`` (``TRTLLM_FP4_MLA_PACKED_VAR_PLAN=0``):
      the rule's length and the longest request cut into 1..PLAN_SIM_CANDIDATES - 1
      equal pieces, scored by makespan alone (``_plan_simulated_makespan``; ties
      keep the longer tasks), the equal-length wave plan.
    - ``var_plan`` (default): the rule's length and the longest request cut into
      1..PLAN_V_SINGLE_CANDIDATES - 1 pieces, plus PLAN_V_WRAP_CANDIDATES
      variable-length wrap-around plans (``_plan_wrap_walk``) for slot lengths
      ``ceil(total / wave_tasks) * (1 + j / 8)``, scored by makespan plus
      ``task_cost_tiles`` per task and ``partial_cost64 / 64`` tiles per task of a
      split request (its reduction); ties keep fewer tasks,
      then the earlier candidate. Wrap tasks run longest first.
    Without ``wave_plan`` (``TRTLLM_FP4_MLA_PACKED_WAVE_PLAN=0``) the plan keeps
    the rule's length and lists every request's tasks in request order.
    The host mirror is ``plan_packed_table``.
    """
    tidx, _, _ = cute.arch.thread_idx()
    sTiles = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_MAX_REQUESTS), cute.make_layout(PLAN_MAX_REQUESTS)
    )
    sBegins = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_MAX_REQUESTS + 1),
        cute.make_layout(PLAN_MAX_REQUESTS + 1),
    )
    # Request KV lengths every task record repeats (see _plan_write_task).
    sValidK = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, PLAN_MAX_REQUESTS), cute.make_layout(PLAN_MAX_REQUESTS)
    )
    # [task length (longest task), longest request, plan kind (0 equal, 1 wrap), wrap slot length]
    sScalars = cute.make_tensor(cute.arch.alloc_smem(cutlass.Int32, 4), cute.make_layout(4))
    # Per candidate warp: score (-1 = infeasible), parameter, task count.
    sCandidates = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, 3 * PLAN_SIM_CANDIDATES),
        cute.make_layout(3 * PLAN_SIM_CANDIDATES),
    )
    for request_idx in cutlass.range(tidx, num_requests, _PLAN_THREADS, unroll=1):
        request_valid_k = cutlass.Int32(mValidK_l[request_idx])
        if cutlass.const_expr(write_generation_lengths):
            request_valid_k = (
                request_valid_k - cutlass.Int32(mPromptLens_l[request_idx]) + generation_len
            )
            mCorrectedKvLens_l[request_idx] = request_valid_k
            mGenerationLens_l[request_idx] = generation_len
        sValidK[request_idx] = request_valid_k
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
            sScalars[0] = tiles_per_task
            sScalars[1] = longest
            sScalars[2] = cutlass.Int32(0)
            sScalars[3] = total_tiles
    cute.arch.sync_threads()
    warp_idx = tidx // cutlass.Int32(32)
    lane = tidx % cutlass.Int32(32)
    fits = (wave_tasks <= cutlass.Int32(PLAN_SIM_MAX_SLOTS)) & (
        num_requests <= cutlass.Int32(PLAN_SIM_MAX_REQUESTS)
    )
    simulate = (task_capacity > num_requests) & fits
    if cutlass.const_expr(wave_plan and not use_helix_kv_bounds and not var_plan):
        if simulate:
            longest = sScalars[1]
            candidate = sScalars[0]
            if warp_idx > cutlass.Int32(0):
                candidate = cutlass.min(
                    cutlass.max(
                        (longest + warp_idx - cutlass.Int32(1)) // warp_idx,
                        cutlass.Int32(PACKED_MIN_TILES_PER_TASK),
                    ),
                    cutlass.min(longest, max_tiles_per_task),
                )
            candidate_tasks = cutlass.Int32(0)
            for request_idx in cutlass.range(lane, num_requests, 32, unroll=1):
                candidate_tasks = (
                    candidate_tasks
                    + (sTiles[request_idx] + candidate - cutlass.Int32(1)) // candidate
                )
            candidate_tasks = cute.arch.warp_redux_sync(candidate_tasks, "add")
            candidate_makespan = cutlass.Int32(-1)
            if candidate_tasks <= cutlass.min(task_capacity, cutlass.Int32(PLAN_SIM_MAX_TASKS)):
                candidate_makespan = _plan_simulated_makespan(
                    sTiles,
                    num_requests,
                    candidate,
                    wave_tasks,
                    cutlass.Int32(PLAN_TASK_COST_TILES),
                    lane,
                )
            if lane == cutlass.Int32(0):
                sCandidates[warp_idx] = candidate_makespan
                sCandidates[PLAN_SIM_CANDIDATES + warp_idx] = candidate
        cute.arch.sync_threads()
        if simulate & (tidx == cutlass.Int32(0)):
            best_makespan = cutlass.Int32(-1)
            best_tiles = sScalars[0]
            for candidate_idx in cutlass.range(0, PLAN_SIM_CANDIDATES, 1, unroll=1):
                candidate_makespan = sCandidates[candidate_idx]
                candidate = sCandidates[PLAN_SIM_CANDIDATES + candidate_idx]
                if candidate_makespan >= cutlass.Int32(0):
                    better = (
                        (best_makespan < cutlass.Int32(0))
                        | (candidate_makespan < best_makespan)
                        | ((candidate_makespan == best_makespan) & (candidate > best_tiles))
                    )
                    if better:
                        best_makespan = candidate_makespan
                        best_tiles = candidate
            sScalars[0] = best_tiles
    if cutlass.const_expr(wave_plan and not use_helix_kv_bounds and var_plan):
        sOrder = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Int32, PLAN_SIM_MAX_REQUESTS),
            cute.make_layout(PLAN_SIM_MAX_REQUESTS),
        )
        # Piece lengths of each wrap candidate; the chosen plan reuses rows 0..3 for
        # its (request, split, first tile, length) tables.
        sPieces = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Int32, PLAN_V_WRAP_CANDIDATES * PLAN_V_MAX_PIECES),
            cute.make_layout(
                (PLAN_V_WRAP_CANDIDATES, PLAN_V_MAX_PIECES), stride=(PLAN_V_MAX_PIECES, 1)
            ),
        )
        sCount = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Int32, PLAN_SIM_MAX_REQUESTS),
            cute.make_layout(PLAN_SIM_MAX_REQUESTS),
        )
        if simulate:
            _plan_order_requests(sTiles, sOrder, num_requests, tidx)
        cute.arch.sync_threads()
        # Fresh names: the DSL's scope check rejects re-binding at this level a name
        # first set inside the warp-0 block above.
        request_longest = sScalars[1]
        request_total = sScalars[3]
        candidate = cutlass.Int32(0)
        candidate_tasks = cutlass.Int32(0)
        candidate_partial = cutlass.Int32(0)
        candidate_longest = cutlass.Int32(0)
        task_limit = cutlass.min(task_capacity, cutlass.Int32(PLAN_V_MAX_PIECES))
        if simulate:
            if warp_idx < cutlass.Int32(PLAN_V_SINGLE_CANDIDATES):
                candidate = sScalars[0]
                if warp_idx > cutlass.Int32(0):
                    candidate = cutlass.min(
                        cutlass.max(
                            (request_longest + warp_idx - cutlass.Int32(1)) // warp_idx,
                            cutlass.Int32(PACKED_MIN_TILES_PER_TASK),
                        ),
                        cutlass.min(request_longest, max_tiles_per_task),
                    )
                for request_idx in cutlass.range(lane, num_requests, 32, unroll=1):
                    request_tasks = (
                        sTiles[request_idx] + candidate - cutlass.Int32(1)
                    ) // candidate
                    candidate_tasks = candidate_tasks + request_tasks
                    if request_tasks > cutlass.Int32(1):
                        candidate_partial = candidate_partial + request_tasks
                candidate_tasks = cute.arch.warp_redux_sync(candidate_tasks, "add")
                candidate_partial = cute.arch.warp_redux_sync(candidate_partial, "add")
                candidate_longest = candidate
            else:
                wrap_idx = warp_idx - cutlass.Int32(PLAN_V_SINGLE_CANDIDATES)
                base = (request_total + wave_tasks - cutlass.Int32(1)) // wave_tasks
                step = cutlass.max((base + cutlass.Int32(7)) // cutlass.Int32(8), cutlass.Int32(1))
                candidate = cutlass.max(
                    base + wrap_idx * step, cutlass.Int32(PLAN_V_MIN_PIECE_TILES)
                )
                candidate_tasks, candidate_partial, candidate_longest = _plan_wrap_walk(
                    sTiles,
                    sOrder,
                    num_requests,
                    candidate,
                    sPieces,
                    wrap_idx,
                    sCount,
                    lane == cutlass.Int32(0),
                    False,
                )
        cute.arch.sync_threads()
        if simulate:
            score = cutlass.Int32(-1)
            feasible = (candidate_tasks <= task_limit) & (candidate_longest <= max_tiles_per_task)
            if feasible:
                makespan = cutlass.Int32(0)
                if warp_idx < cutlass.Int32(PLAN_V_SINGLE_CANDIDATES):
                    makespan = _plan_simulated_makespan(
                        sTiles, num_requests, candidate, wave_tasks, task_cost_tiles, lane
                    )
                else:
                    makespan = _plan_lpt_makespan(
                        sPieces,
                        warp_idx - cutlass.Int32(PLAN_V_SINGLE_CANDIDATES),
                        candidate_tasks,
                        wave_tasks,
                        task_cost_tiles,
                        lane,
                    )
                score = (
                    makespan * cutlass.Int32(PLAN_V_COST_SCALE) + candidate_partial * partial_cost64
                )
            if lane == cutlass.Int32(0):
                sCandidates[warp_idx] = score
                sCandidates[PLAN_SIM_CANDIDATES + warp_idx] = candidate
                sCandidates[2 * PLAN_SIM_CANDIDATES + warp_idx] = candidate_tasks
        cute.arch.sync_threads()
        if simulate & (tidx == cutlass.Int32(0)):
            best_score = cutlass.Int32(-1)
            best_tasks = cutlass.Int32(0)
            best_idx = cutlass.Int32(0)
            for candidate_idx in cutlass.range(0, PLAN_SIM_CANDIDATES, 1, unroll=1):
                score = sCandidates[candidate_idx]
                tasks = sCandidates[2 * PLAN_SIM_CANDIDATES + candidate_idx]
                if score >= cutlass.Int32(0):
                    better = (
                        (best_score < cutlass.Int32(0))
                        | (score < best_score)
                        | ((score == best_score) & (tasks < best_tasks))
                    )
                    if better:
                        best_score = score
                        best_tasks = tasks
                        best_idx = candidate_idx
            if best_score >= cutlass.Int32(0):
                best_param = sCandidates[PLAN_SIM_CANDIDATES + best_idx]
                if best_idx < cutlass.Int32(PLAN_V_SINGLE_CANDIDATES):
                    sScalars[0] = best_param
                else:
                    sScalars[2] = cutlass.Int32(1)
                    sScalars[3] = best_param
        cute.arch.sync_threads()
        if sScalars[2] == cutlass.Int32(1):
            # Materialize the chosen wrap plan: pieces (request, split, first tile,
            # length) in rows 0..3 of sPieces, then begins in request order.
            if tidx == cutlass.Int32(0):
                wrap_pieces, wrap_partial, longest_piece = _plan_wrap_walk(
                    sTiles,
                    sOrder,
                    num_requests,
                    sScalars[3],
                    sPieces,
                    cutlass.Int32(0),
                    sCount,
                    True,
                    True,
                )
                task_begin = cutlass.Int32(0)
                for request_idx in cutlass.range(0, num_requests, 1, unroll=1):
                    sBegins[request_idx] = task_begin
                    mPlan_i[cutlass.Int32(2) + request_idx] = task_begin
                    task_begin = task_begin + sCount[request_idx]
                sBegins[num_requests] = task_begin
                mPlan_i[cutlass.Int32(2) + num_requests] = task_begin
                mPlan_i[0] = longest_piece
                mPlan_i[1] = task_begin
            cute.arch.sync_threads()
            used_tasks = sBegins[num_requests]
            tables = cutlass.Int32(3) + num_requests
            # Grid order: longest piece first, ties by request then split.
            for piece in cutlass.range(tidx, used_tasks, _PLAN_THREADS, unroll=1):
                length = sPieces[3, piece]
                request = sPieces[0, piece]
                split = sPieces[1, piece]
                rank = cutlass.Int32(0)
                for other in cutlass.range(0, used_tasks, 1, unroll=1):
                    other_len = sPieces[3, other]
                    other_request = sPieces[0, other]
                    ahead = (other_len > length) | (
                        (other_len == length)
                        & (
                            (other_request < request)
                            | ((other_request == request) & (sPieces[1, other] < split))
                        )
                    )
                    if ahead:
                        rank = rank + cutlass.Int32(1)
                _plan_write_task(
                    mPlan_i,
                    sBegins,
                    sValidK,
                    page_stride,
                    tables,
                    task_capacity,
                    rank,
                    request,
                    split,
                    sCount[request],
                    sPieces[2, piece],
                    length,
                )
            for task_idx in cutlass.range(
                used_tasks + tidx, task_capacity, _PLAN_THREADS, unroll=1
            ):
                _plan_write_unused_slot(mPlan_i, tables, task_capacity, task_idx, num_requests)
    cute.arch.sync_threads()
    if sScalars[2] == cutlass.Int32(0):
        _plan_write_equal_tasks(
            mPlan_i,
            sTiles,
            sBegins,
            sValidK,
            page_stride,
            sScalars[0],
            num_requests,
            task_capacity,
            tidx,
            wave_plan,
        )


@cute.jit
def _plan_write_task(
    mPlan_i: cute.Tensor,
    sBegins: cute.Tensor,
    sValidK: cute.Tensor,
    page_stride: cutlass.Int32,
    tables: cutlass.Int32,
    task_capacity: cutlass.Int32,
    task_idx: cutlass.Int32,
    request: cutlass.Int32,
    split: cutlass.Int32,
    count: cutlass.Int32,
    tile_begin: cutlass.Int32,
    tile_count: cutlass.Int32,
) -> None:
    """A used slot: the task's tables and its request's fields the decode prologue reads.

    The page window is ``[request * page_stride, +page_stride)`` with a fixed-stride
    page table and ``(0, -1)`` (read the indptr) without one.
    """
    page_begin = request * page_stride
    page_count = cutlass.Int32(-1)
    if page_stride > cutlass.Int32(0):
        page_count = page_stride
    mPlan_i[tables + task_idx] = request
    mPlan_i[tables + task_capacity + task_idx] = split
    mPlan_i[tables + cutlass.Int32(2) * task_capacity + task_idx] = count
    mPlan_i[tables + cutlass.Int32(3) * task_capacity + task_idx] = tile_begin
    mPlan_i[tables + cutlass.Int32(4) * task_capacity + task_idx] = tile_count
    mPlan_i[tables + cutlass.Int32(PACKED_PLAN_PARTIAL_SLOT_TABLE) * task_capacity + task_idx] = (
        sBegins[request] + split
    )
    mPlan_i[tables + cutlass.Int32(PACKED_PLAN_VALID_K_TABLE) * task_capacity + task_idx] = sValidK[
        request
    ]
    mPlan_i[tables + cutlass.Int32(PACKED_PLAN_PAGE_BEGIN_TABLE) * task_capacity + task_idx] = (
        page_begin
    )
    mPlan_i[tables + cutlass.Int32(PACKED_PLAN_PAGE_COUNT_TABLE) * task_capacity + task_idx] = (
        page_count
    )


@cute.jit
def _plan_write_unused_slot(
    mPlan_i: cute.Tensor,
    tables: cutlass.Int32,
    task_capacity: cutlass.Int32,
    task_idx: cutlass.Int32,
    num_requests: cutlass.Int32,
) -> None:
    """A slot past the used tasks: last request, out-of-range split, no tiles (the decode
    kernel exits it on the tile count) and zero request fields."""
    mPlan_i[tables + task_idx] = num_requests - cutlass.Int32(1)
    mPlan_i[tables + task_capacity + task_idx] = cutlass.Int32(1 << 15)
    mPlan_i[tables + cutlass.Int32(2) * task_capacity + task_idx] = cutlass.Int32(1 << 15)
    mPlan_i[tables + cutlass.Int32(3) * task_capacity + task_idx] = cutlass.Int32(1 << 30)
    for table in cutlass.range_constexpr(4, PLAN_TABLES):
        mPlan_i[tables + cutlass.Int32(table) * task_capacity + task_idx] = cutlass.Int32(0)


@cute.jit
def _plan_write_equal_tasks(
    mPlan_i: cute.Tensor,
    sTiles: cute.Tensor,
    sBegins: cute.Tensor,
    sValidK: cute.Tensor,
    page_stride: cutlass.Int32,
    task_tiles: cutlass.Int32,
    num_requests: cutlass.Int32,
    task_capacity: cutlass.Int32,
    tidx: cutlass.Int32,
    wave_plan: cutlass.Constexpr,
) -> None:
    """Tables of an equal-length plan (every request's tasks ``task_tiles`` long but its last)."""
    if tidx == cutlass.Int32(0):
        task_begin = cutlass.Int32(0)
        for request_idx in cutlass.range(0, num_requests, 1, unroll=1):
            sBegins[request_idx] = task_begin
            mPlan_i[cutlass.Int32(2) + request_idx] = task_begin
            task_begin = (
                task_begin + (sTiles[request_idx] + task_tiles - cutlass.Int32(1)) // task_tiles
            )
        sBegins[num_requests] = task_begin
        mPlan_i[cutlass.Int32(2) + num_requests] = task_begin
        mPlan_i[0] = task_tiles
        mPlan_i[1] = task_begin
    cute.arch.sync_threads()
    tables = cutlass.Int32(3) + num_requests
    used_tasks = sBegins[num_requests]
    # Grid order with wave_plan (plan_task_order): non-last tasks in request order from
    # begins[r] - r, then last tasks longest first. Partials stay at begins[r] + split.
    full_tasks = used_tasks - num_requests
    for request_idx in cutlass.range(tidx, num_requests, _PLAN_THREADS, unroll=1):
        begin = sBegins[request_idx]
        request_tiles = sTiles[request_idx]
        request_tasks = sBegins[request_idx + cutlass.Int32(1)] - begin
        first_task = begin
        last_task = begin + request_tasks - cutlass.Int32(1)
        if cutlass.const_expr(wave_plan):
            tail = request_tiles - (request_tasks - cutlass.Int32(1)) * task_tiles
            rank = cutlass.Int32(0)
            for other_idx in cutlass.range(0, num_requests, 1, unroll=1):
                other_tasks = sBegins[other_idx + cutlass.Int32(1)] - sBegins[other_idx]
                other_tail = sTiles[other_idx] - (other_tasks - cutlass.Int32(1)) * task_tiles
                if (other_tail > tail) | ((other_tail == tail) & (other_idx < request_idx)):
                    rank = rank + cutlass.Int32(1)
            first_task = begin - request_idx
            last_task = full_tasks + rank
        for split_idx in cutlass.range(0, request_tasks, 1, unroll=1):
            task_idx = last_task
            if split_idx + cutlass.Int32(1) < request_tasks:
                task_idx = first_task + split_idx
            tile_begin = split_idx * task_tiles
            _plan_write_task(
                mPlan_i,
                sBegins,
                sValidK,
                page_stride,
                tables,
                task_capacity,
                task_idx,
                request_idx,
                split_idx,
                request_tasks,
                tile_begin,
                cutlass.min(task_tiles, request_tiles - tile_begin),
            )
    for task_idx in cutlass.range(used_tasks + tidx, task_capacity, _PLAN_THREADS, unroll=1):
        _plan_write_unused_slot(mPlan_i, tables, task_capacity, task_idx, num_requests)


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
    task_cost_tiles: cutlass.Int32,
    partial_cost64: cutlass.Int32,
    page_stride: cutlass.Int32,
    query_len_per_seq: cutlass.Constexpr,
    use_helix_kv_bounds: cutlass.Constexpr,
    write_generation_lengths: cutlass.Constexpr,
    wave_plan: cutlass.Constexpr,
    var_plan: cutlass.Constexpr,
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
        cute.make_layout(
            (cutlass.Int32(3) + num_requests + PLAN_TABLES * task_capacity,), stride=(1,)
        ),
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
        task_cost_tiles,
        partial_cost64,
        page_stride,
        query_len_per_seq,
        use_helix_kv_bounds,
        write_generation_lengths,
        wave_plan,
        var_plan,
    ).launch(grid=(1, 1, 1), block=(_PLAN_THREADS, 1, 1), stream=stream)


def packed_plan_length(num_requests: int, task_capacity: int) -> int:
    """Int32 entries of the plan buffer for ``num_requests`` and ``task_capacity``."""
    return 3 + num_requests + PLAN_TABLES * task_capacity


def packed_plan_costs(m_tiles: int) -> tuple[int, int]:
    """(task cost in KV tiles, partial cost in 1/64 tiles per task) the var plan scores with.

    Fitted to the packed decode: one task costs about
    8 KV tiles of fixed work (Q load, pipeline fill, partial store) when the call is
    tile-bound (2+ M tiles per request) and about 16 when one M tile per request
    leaves it HBM-bound, and every task of a split request adds ~0.08 tiles per M
    tile (0.10 with one M tile) of reduction.
    """
    if m_tiles >= 2:
        return PLAN_V_TASK_COST_TILES, round(PLAN_V_COST_SCALE * 0.08 * m_tiles)
    return 2 * PLAN_V_TASK_COST_TILES, round(PLAN_V_COST_SCALE * 0.10)


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
    wave_plan: bool = True,
    var_plan: bool = True,
    m_tiles: int = 1,
    page_stride: int = 0,
) -> None:
    """Write the balanced task plan for ``num_requests * num_splits`` slots into ``plan``.

    ``kv_lens`` (int32 per request) and the optional per-token int32
    ``helix_kv_bounds`` are the same bounds the decode kernel reads. A positive
    ``page_stride`` declares the fixed-stride page table (request r's decode
    indptr window is ``[r * page_stride, (r + 1) * page_stride)``), which every
    task record then carries; 0 leaves the decode kernel to read its indptr. See
    ``plan_packed_table`` for the host mirror of the plan. With
    ``generation_lengths = (prompt_lens, corrected_kv_lens, generation_lens,
    generation_len)`` the same launch first writes the per-request generation
    lengths ``kv_lens - prompt_lens + generation_len`` and ``generation_len``
    (``kv_lens`` then being the raw lengths) and plans on the corrected ones.
    ``wave_plan=False`` keeps the rule's task length and request order (no
    wave simulation, no tail reordering); ``var_plan=False`` keeps the
    equal-length wave plan; ``m_tiles`` (M tiles per request) sets the var
    plan's costs (``packed_plan_costs``).
    """
    task_capacity = num_requests * num_splits
    if num_requests <= 0 or num_splits <= 0 or wave_tasks <= 0 or max_tiles_per_task <= 0:
        raise ValueError("plan needs positive request, split, wave and tile counts")
    if num_requests > PLAN_MAX_REQUESTS:
        raise ValueError(f"plan supports at most {PLAN_MAX_REQUESTS} requests, got {num_requests}")
    if query_len <= 0:
        raise ValueError("query_len must be positive")
    if m_tiles <= 0:
        raise ValueError("m_tiles must be positive")
    if kv_lens.dtype != torch.int32 or kv_lens.numel() < num_requests or kv_lens.stride(0) != 1:
        raise ValueError(f"kv_lens must be contiguous int32 with >= {num_requests} entries")
    if page_stride < 0 or num_requests * page_stride > INT32_MAX:
        raise ValueError(
            f"page_stride must be >= 0 with num_requests * page_stride <= {INT32_MAX}, "
            f"got {page_stride}"
        )
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
    task_cost_tiles, partial_cost64 = packed_plan_costs(m_tiles)
    gmem = cute.AddressSpace.gmem
    ptrs = (
        make_ptr(cutlass.Int32, kv_lens.data_ptr(), gmem, assumed_align=4),
        make_ptr(cutlass.Int32, bounds_ptr, gmem, assumed_align=4),
        *(make_ptr(cutlass.Int32, ptr, gmem, assumed_align=4) for ptr in length_ptrs),
        make_ptr(cutlass.Int32, plan.data_ptr(), gmem, assumed_align=4),
    )
    stream = _current_cu_stream()
    cache_key = (query_len, use_helix, write_lengths, wave_plan, var_plan)
    plan_fn = _PACKED_PLAN_COMPILE_CACHE.get(cache_key)
    if plan_fn is None:
        plan_fn = _compile_cutedsl(
            packed_plan_host,
            *ptrs,
            stream,
            *(cutlass.Int32(1) for _ in range(8)),
            query_len,
            use_helix,
            write_lengths,
            wave_plan,
            var_plan,
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
        cutlass.Int32(task_cost_tiles),
        cutlass.Int32(partial_cost64),
        cutlass.Int32(page_stride),
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
    wave_plan: bool = True,
) -> tuple[int, list[int]]:
    """Host mirror of the equal-length plan (``var_plan=False``): (tiles per task, task begins).

    ``request_tiles`` are the per-request KV tile counts (>= 1), ``task_capacity``
    the grid.z slots (``num_requests * num_splits``) and ``wave_tasks`` the tasks
    that fill one wave of clusters; ``one_wave`` (Helix) never takes the two-wave
    target and keeps that length. Otherwise, with ``wave_plan``, the length competes
    with the longest request cut into 1..PLAN_SIM_CANDIDATES - 1 pieces by
    ``plan_simulated_makespan``. Returns the KV tiles each task may own and the
    ``len(request_tiles) + 1`` task prefix, whose last element is the number of
    tasks actually used (<= ``task_capacity``). ``plan_packed_table`` mirrors the
    default (``var_plan``) plan.
    """
    num_requests = len(request_tiles)
    if num_requests <= 0 or task_capacity <= 0 or wave_tasks <= 0 or min(request_tiles) <= 0:
        raise ValueError("plan needs requests, task slots, wave tasks and positive tile counts")
    tiles_per_task = _plan_rule_length(
        request_tiles, task_capacity, wave_tasks, max_tiles_per_task, min_tiles_per_task, one_wave
    )
    longest = max(request_tiles)
    if (
        wave_plan
        and not one_wave
        and task_capacity > num_requests
        and wave_tasks <= PLAN_SIM_MAX_SLOTS
        and num_requests <= PLAN_SIM_MAX_REQUESTS
    ):
        candidates = [tiles_per_task] + [
            min(max(math.ceil(longest / pieces), min_tiles_per_task), longest, max_tiles_per_task)
            for pieces in range(1, PLAN_SIM_CANDIDATES)
        ]
        best = None
        for candidate in candidates:
            if _plan_used_tasks(request_tiles, candidate) > min(task_capacity, PLAN_SIM_MAX_TASKS):
                continue
            makespan = plan_simulated_makespan(request_tiles, candidate, wave_tasks)
            if best is None or (makespan, -candidate) < best:
                best = (makespan, -candidate)
        if best is not None:
            tiles_per_task = -best[1]
    begins = [0]
    for tiles in request_tiles:
        begins.append(begins[-1] + math.ceil(tiles / tiles_per_task))
    return tiles_per_task, begins


def _plan_used_tasks(request_tiles: list[int], tiles_per_task: int) -> int:
    return sum(math.ceil(t / tiles_per_task) for t in request_tiles)


def _plan_rule_length(
    request_tiles: list[int],
    task_capacity: int,
    wave_tasks: int,
    max_tiles_per_task: int,
    min_tiles_per_task: int,
    one_wave: bool,
) -> int:
    """The plan kernel's rule task length (see ``packed_plan_kernel``)."""
    num_requests = len(request_tiles)
    total = sum(request_tiles)
    longest = max(request_tiles)
    tiles_per_task = longest
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
            if _plan_used_tasks(request_tiles, target) <= task_capacity:
                tiles_per_task = target
            else:
                fallback = max(
                    math.ceil(total / (task_capacity - num_requests)), min_tiles_per_task
                )
                tiles_per_task = fallback
                even = max(math.ceil(longest / (task_capacity // num_requests)), min_tiles_per_task)
                if even < fallback and _plan_used_tasks(request_tiles, even) <= task_capacity:
                    tiles_per_task = even
            tiles_per_task = min(tiles_per_task, longest)
    return min(tiles_per_task, max_tiles_per_task)


def plan_task_order(
    request_tiles: list[int], tiles_per_task: int, wave_plan: bool = True
) -> list[tuple[int, int]]:
    """Grid order of an equal-length plan's tasks as (request, split).

    Every request's tasks but its last in request order, then the last tasks
    longest first (ties: lower request); without ``wave_plan`` all in request order.
    """
    tasks = [math.ceil(tiles / tiles_per_task) for tiles in request_tiles]
    if not wave_plan:
        return [(r, split) for r, n in enumerate(tasks) for split in range(n)]
    order = [(r, split) for r, n in enumerate(tasks) for split in range(n - 1)]
    tails = [
        (tiles - (n - 1) * tiles_per_task, r)
        for r, (tiles, n) in enumerate(zip(request_tiles, tasks))
    ]
    order += [(r, tasks[r] - 1) for _, r in sorted(tails, key=lambda tail: (-tail[0], tail[1]))]
    return order


def plan_simulated_makespan(
    request_tiles: list[int],
    tiles_per_task: int,
    wave_tasks: int,
    task_cost_tiles: int = PLAN_TASK_COST_TILES,
) -> int:
    """Host mirror of ``_plan_simulated_makespan``: grid makespan in KV tiles for one length."""
    slots = [0] * wave_tasks
    makespan = 0
    for request, split in plan_task_order(request_tiles, tiles_per_task):
        start = heapq.heappop(slots)
        tiles = min(tiles_per_task, request_tiles[request] - split * tiles_per_task)
        end = start + tiles + task_cost_tiles
        makespan = max(makespan, end)
        heapq.heappush(slots, end)
    return makespan


def plan_wrap_pieces(request_tiles: list[int], chunk_tiles: int) -> list[tuple[int, int, int, int]]:
    """Host mirror of ``_plan_wrap_walk``: (request, split, first tile, tiles) per piece.

    Requests, longest first (ties: lower request), are laid end to end and cut every
    ``chunk_tiles`` tiles; a boundary head piece shorter than PLAN_V_MIN_PIECE_TILES
    moves its request to the next slot and a shorter remainder joins its piece.
    Pieces are listed in walk order.
    """
    order = sorted(range(len(request_tiles)), key=lambda r: (-request_tiles[r], r))
    pieces = []
    fill = 0
    for request in order:
        tiles = request_tiles[request]
        begin = 0
        split = 0
        while begin < tiles:
            remaining = tiles - begin
            take = min(chunk_tiles - fill, remaining)
            if take < remaining and take < PLAN_V_MIN_PIECE_TILES and fill > 0:
                fill = 0
                continue
            if 0 < remaining - take < PLAN_V_MIN_PIECE_TILES:
                take = remaining
            pieces.append((request, split, begin, take))
            split += 1
            begin += take
            fill += take
            if fill >= chunk_tiles:
                fill = 0
    return pieces


def plan_lpt_makespan(lengths: list[int], wave_tasks: int, task_cost_tiles: int) -> int:
    """Host mirror of ``_plan_lpt_makespan``: tasks longest first on ``wave_tasks`` slots."""
    slots = [0] * wave_tasks
    makespan = 0
    for length in sorted(lengths, reverse=True):
        start = heapq.heappop(slots)
        end = start + length + task_cost_tiles
        makespan = max(makespan, end)
        heapq.heappush(slots, end)
    return makespan


class PackedPlan(NamedTuple):
    """Host mirror of the plan buffer: ``tasks`` are (request, split, count, first tile,
    tiles) in grid order; ``begins`` the per-request task prefix (partial slots)."""

    tiles_per_task: int
    begins: list[int]
    tasks: list[tuple[int, int, int, int, int]]


def _equal_length_tasks(
    request_tiles: list[int], tiles_per_task: int, wave_plan: bool
) -> list[tuple[int, int, int, int, int]]:
    counts = [math.ceil(t / tiles_per_task) for t in request_tiles]
    return [
        (
            request,
            split,
            counts[request],
            split * tiles_per_task,
            min(tiles_per_task, request_tiles[request] - split * tiles_per_task),
        )
        for request, split in plan_task_order(request_tiles, tiles_per_task, wave_plan)
    ]


def plan_packed_table(
    request_tiles: list[int],
    task_capacity: int,
    wave_tasks: int,
    max_tiles_per_task: int = 1 << 30,
    *,
    one_wave: bool = False,
    wave_plan: bool = True,
    var_plan: bool = True,
    m_tiles: int = 1,
) -> PackedPlan:
    """Host mirror of ``packed_plan_kernel``: the plan the decode and reduction kernels read.

    With ``var_plan`` (default) and the simulation in range (``task_capacity`` above
    the request count, at most PLAN_SIM_MAX_SLOTS wave slots and
    PLAN_SIM_MAX_REQUESTS requests, not Helix), PLAN_V_SINGLE_CANDIDATES
    equal-length plans (the rule's length and the longest request in 1..15 pieces)
    and PLAN_V_WRAP_CANDIDATES wrap-around plans compete on simulated makespan plus
    ``packed_plan_costs(m_tiles)``; otherwise this is ``plan_packed_tasks``.
    """
    num_requests = len(request_tiles)
    simulate = (
        task_capacity > num_requests
        and wave_tasks <= PLAN_SIM_MAX_SLOTS
        and num_requests <= PLAN_SIM_MAX_REQUESTS
    )
    if not (var_plan and wave_plan and not one_wave and simulate):
        tiles_per_task, begins = plan_packed_tasks(
            request_tiles,
            task_capacity,
            wave_tasks,
            max_tiles_per_task,
            one_wave=one_wave,
            wave_plan=wave_plan,
        )
        return PackedPlan(
            tiles_per_task, begins, _equal_length_tasks(request_tiles, tiles_per_task, wave_plan)
        )
    task_cost_tiles, partial_cost64 = packed_plan_costs(m_tiles)
    rule = _plan_rule_length(
        request_tiles,
        task_capacity,
        wave_tasks,
        max_tiles_per_task,
        PACKED_MIN_TILES_PER_TASK,
        one_wave=False,
    )
    longest = max(request_tiles)
    total = sum(request_tiles)
    limit = min(task_capacity, PLAN_V_MAX_PIECES)
    best = None  # (score, tasks, candidate index, kind, parameter)
    for idx in range(PLAN_V_SINGLE_CANDIDATES):
        length = rule
        if idx > 0:
            length = min(
                max(math.ceil(longest / idx), PACKED_MIN_TILES_PER_TASK),
                min(longest, max_tiles_per_task),
            )
        counts = [math.ceil(t / length) for t in request_tiles]
        tasks = sum(counts)
        if tasks > limit or length > max_tiles_per_task:
            continue
        partial = sum(n for n in counts if n > 1)
        makespan = plan_simulated_makespan(request_tiles, length, wave_tasks, task_cost_tiles)
        score = makespan * PLAN_V_COST_SCALE + partial * partial_cost64
        if best is None or (score, tasks) < best[:2]:
            best = (score, tasks, idx, 0, length)
    base = math.ceil(total / wave_tasks)
    step = max((base + 7) // 8, 1)
    for wrap_idx in range(PLAN_V_WRAP_CANDIDATES):
        chunk = max(base + wrap_idx * step, PLAN_V_MIN_PIECE_TILES)
        pieces = plan_wrap_pieces(request_tiles, chunk)
        lengths = [p[3] for p in pieces]
        if len(pieces) > limit or max(lengths) > max_tiles_per_task:
            continue
        counts = [0] * num_requests
        for request, _, _, _ in pieces:
            counts[request] += 1
        partial = sum(n for n in counts if n > 1)
        makespan = plan_lpt_makespan(lengths, wave_tasks, task_cost_tiles)
        score = makespan * PLAN_V_COST_SCALE + partial * partial_cost64
        if best is None or (score, len(pieces)) < best[:2]:
            best = (score, len(pieces), PLAN_V_SINGLE_CANDIDATES + wrap_idx, 1, chunk)
    if best is None or best[3] == 0:
        length = rule if best is None else best[4]
        counts = [math.ceil(t / length) for t in request_tiles]
        begins = [0]
        for n in counts:
            begins.append(begins[-1] + n)
        return PackedPlan(length, begins, _equal_length_tasks(request_tiles, length, True))
    pieces = plan_wrap_pieces(request_tiles, best[4])
    counts = [0] * num_requests
    for request, _, _, _ in pieces:
        counts[request] += 1
    begins = [0]
    for n in counts:
        begins.append(begins[-1] + n)
    ordered = sorted(pieces, key=lambda p: (-p[3], p[0], p[1]))
    tasks = [
        (request, split, counts[request], first, tiles) for request, split, first, tiles in ordered
    ]
    return PackedPlan(max(p[3] for p in pieces), begins, tasks)


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


def reduce_uses_pdl() -> bool:
    """Launch the reduction as a programmatic dependent of the decode kernel.

    Follows TensorRT-LLM's ``TRTLLM_ENABLE_PDL`` switch (default on); the decode
    kernel always signals its dependents, which only matters for a PDL launch.
    """
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def select_reduce_task_groups(num_rows: int, task_slots: int) -> int:
    """Warps per (row, band) item: groups only for long task lists over few output rows.

    ``num_rows`` is the batch's output rows (requests x valid rows) and ``task_slots`` the
    task slots of one request (``num_splits``).
    """
    if task_slots < _REDUCE_GROUP_MIN_TASK_SLOTS:
        return 1
    if num_rows <= _REDUCE_FOUR_GROUP_ROWS:
        return 4
    if num_rows <= _REDUCE_TWO_GROUP_ROWS:
        return 2
    return 1


def select_reduce_ctas(num_items: int, task_groups: int, sm_count: int) -> int:
    """Persistent CTAs: ``task_groups`` warps per (row, band) item, up to four CTAs per SM."""
    items_per_cta = _REDUCE_WARPS_PER_CTA // task_groups
    return max(
        1, min((num_items + items_per_cta - 1) // items_per_cta, _REDUCE_CTAS_PER_SM * sm_count)
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
    num_items = total_rows * (_VALUE_DIM // band_columns)
    task_groups = select_reduce_task_groups(total_rows, num_splits)
    grid_ctas = select_reduce_ctas(num_items, task_groups, sm_count)
    write_stats = stats is not None
    use_pdl = reduce_uses_pdl()
    stream = _current_cu_stream()
    ptrs = _make_reduce_ptrs(partial, maxima, sums, output, stats, q_scale, kv_scale, plan)
    cache_key = (band_columns, task_groups, write_stats, output.dtype, use_pdl)
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
            task_groups,
            write_stats,
            use_pdl,
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
