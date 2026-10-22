# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL split reduction for packed FP4 MLA decode tasks.

The decode kernel writes one FP32 partial per (split, request) task covering the
request's 128-padded [query token, head] rows together with the raw row maximum
and the row sum relative to it. This kernel merges the splits of every valid row
into the BF16/FP16 output and, for Helix, the per-row (max, sum) softmax stats.
Global FP4 scales are read on the device, so CUDA graph replays follow them.
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
    LOG2_E,
    _cutlass_output_dtype,
    _validate_output_dtype,
    _validate_tensor_pointer_alignment,
)

_PACKED_REDUCE_COMPILE_CACHE: dict[tuple[int, bool, torch.dtype], Callable] = {}
_REDUCE_THREADS = 128
_VALUE_DIM = 512
# Splits whose partial loads are issued back-to-back before any is consumed; the
# reduction is latency-bound, so this is what sets its memory-level parallelism.
_LOADS_IN_FLIGHT = 8


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
    softmax_scale_log2: cutlass.Float32,
    output_scale: cutlass.Float32,
    num_requests: cutlass.Int32,
    num_splits: cutlass.Int32,
    valid_m: cutlass.Int32,
    physical_m: cutlass.Int32,
    band_columns: cutlass.Constexpr,
    write_stats: cutlass.Constexpr,
) -> None:
    """One CTA per (output row, column band); tasks are split-major, request fastest.

    Warp zero derives the split weights once and shares them; every thread then
    accumulates ``band_columns / 128`` consecutive columns with one vector load
    per split (16 B for the 512-column band). Empty splits (row sum 0) carry
    zero weight, and a row with no visible key anywhere yields O=0 and stats
    (-inf, 0), the neutral Helix contribution.
    """
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    threads_per_block: cutlass.Constexpr = _REDUCE_THREADS
    columns_per_thread: cutlass.Constexpr = band_columns // threads_per_block
    row = bidx
    request = row // valid_m
    local_row = row - request * valid_m
    # Thread t owns columns [first_col + t * columns_per_thread, +columns_per_thread).
    thread_col = bidy * cutlass.Int32(band_columns) + tidx * cutlass.Int32(columns_per_thread)
    scale = softmax_scale_log2 / (mQScale[0] * mKvScale[0])
    # 32 split weights, followed by the row maximum and denominator.
    shared = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, 34), cute.make_layout(34))
    gPartial_arr = cutlass.make_array_view(mPartial_trc)
    gOut_arr = cutlass.make_array_view(mOut_rc)
    numerator = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(columns_per_thread)), dtype=cutlass.Float32
    )
    local_denominator = cutlass.Float32(0.0)
    global_row_max = cutlass.Float32(-cutlass.Float32.inf)

    if tidx < cutlass.Int32(32):
        for split_idx in cutlass.range(tidx, num_splits, 32, unroll=1):
            task = split_idx * num_requests + request
            if mRowSum_tr[task, local_row] != cutlass.Float32(0.0):
                global_row_max = cute.arch.fmax(global_row_max, mRowMax_tr[task, local_row] * scale)
        global_row_max = cute.arch.warp_redux_sync(global_row_max, "fmax")
        if tidx == cutlass.Int32(0):
            shared[32] = global_row_max
    cute.arch.sync_threads()
    global_row_max = shared[32]

    for split_base in cutlass.range(0, num_splits, 32, unroll=1):
        if tidx < cutlass.Int32(32):
            split_idx = split_base + tidx
            split_weight = cutlass.Float32(0.0)
            if split_idx < num_splits:
                task = split_idx * num_requests + request
                split_row_sum = mRowSum_tr[task, local_row]
                if split_row_sum != cutlass.Float32(0.0):
                    split_weight = cute.exp2(
                        mRowMax_tr[task, local_row] * scale - global_row_max, fastmath=True
                    )
                    local_denominator += split_row_sum * split_weight
            shared[tidx] = split_weight
        cute.arch.sync_threads()

        round_splits = cutlass.min(32, num_splits - split_base)
        for chunk_base in cutlass.range(0, round_splits, _LOADS_IN_FLIGHT, unroll=1):
            # Issue every load of the chunk first (out-of-range lanes re-read the
            # chunk's last valid split with zero weight), then accumulate.
            chunk_vecs = []
            chunk_weights = []
            for lane in cutlass.range_constexpr(_LOADS_IN_FLIGHT):
                split_offset = cutlass.min(chunk_base + lane, round_splits - cutlass.Int32(1))
                split_weight = shared[split_offset]
                if chunk_base + lane >= round_splits:
                    split_weight = cutlass.Float32(0.0)
                task = (split_base + split_offset) * num_requests + request
                partial_idx = (task * physical_m + local_row) * cutlass.Int32(
                    _VALUE_DIM
                ) + thread_col
                chunk_vecs.append(
                    gPartial_arr.subview(partial_idx)
                    .data_ptr()
                    .load(count=columns_per_thread, alignment=4 * columns_per_thread)
                )
                chunk_weights.append(split_weight)
            for lane in cutlass.range_constexpr(_LOADS_IN_FLIGHT):
                # Empty splits keep zero weight; their payload is never trusted.
                if chunk_weights[lane] != cutlass.Float32(0.0):
                    numerator = numerator + chunk_vecs[lane] * chunk_weights[lane]
        # Every warp must finish using the weights before warp zero replaces them.
        cute.arch.sync_threads()

    denominator = cutlass.Float32(0.0)
    if tidx < cutlass.Int32(32):
        denominator = cute.arch.warp_reduction_sum(local_denominator)
        if tidx == cutlass.Int32(0):
            shared[33] = denominator
            if cutlass.const_expr(write_stats):
                if bidy == cutlass.Int32(0):
                    # Helix combine convention: natural-log-domain scaled max and
                    # the sum relative to it; an empty row contributes (-inf, 0).
                    stats_max = cutlass.Float32(-cutlass.Float32.inf)
                    if denominator != cutlass.Float32(0.0):
                        stats_max = global_row_max * cutlass.Float32(1.0 / LOG2_E)
                    mStatsMax_r[row] = stats_max
                    mStatsSum_r[row] = denominator
    cute.arch.sync_threads()
    denominator = shared[33]

    normalizer = cutlass.Float32(0.0)
    if denominator != cutlass.Float32(0.0):
        normalizer = output_scale / denominator
    out_vec = (numerator * normalizer).to(mOut_rc.element_type)
    gOut_arr.store(
        out_vec,
        idx=row * cutlass.Int32(_VALUE_DIM) + thread_col,
        vector_size=columns_per_thread,
        alignment=2 * columns_per_thread,
    )


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
    softmax_scale_log2: cutlass.Float32,
    output_scale: cutlass.Float32,
    stream: cuda.CUstream,
    num_requests: cutlass.Int32,
    num_splits: cutlass.Int32,
    valid_m: cutlass.Int32,
    physical_m: cutlass.Int32,
    total_rows: cutlass.Int32,
    stats_stride: cutlass.Int32,
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
    packed_split_reduce_kernel(
        mPartial_trc,
        mRowMax_tr,
        mRowSum_tr,
        mOut_rc,
        mStatsMax_r,
        mStatsSum_r,
        mQScale,
        mKvScale,
        softmax_scale_log2,
        output_scale,
        num_requests,
        num_splits,
        valid_m,
        physical_m,
        band_columns,
        write_stats,
    ).launch(
        grid=(total_rows, _VALUE_DIM // band_columns, 1),
        block=(_REDUCE_THREADS, 1, 1),
        stream=stream,
    )


def _make_reduce_ptrs(
    partial: torch.Tensor,
    maxima: torch.Tensor,
    sums: torch.Tensor,
    output: torch.Tensor,
    stats: torch.Tensor | None,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
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
    )


def select_reduce_columns(num_rows: int, sm_count: int) -> int:
    """Column band per CTA: narrower bands when few rows would leave SMs idle."""
    if num_rows >= 2 * sm_count:
        return 512
    if num_rows >= sm_count:
        return 256
    return 128


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
    stats: torch.Tensor | None = None,
    sm_count: int = 128,
) -> None:
    """Reduce ``[splits * requests, physical_m, 512]`` FP32 partials into ``output``.

    ``output`` is a contiguous ``[requests * valid_m, 512]`` BF16/FP16 tensor and
    ``stats`` the optional contiguous ``[requests * valid_m, 2]`` FP32 Helix
    (max, sum) buffer. ``q_scale``/``kv_scale`` are the device FP4 global scales.
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
    write_stats = stats is not None
    stream = _current_cu_stream()
    ptrs = _make_reduce_ptrs(partial, maxima, sums, output, stats, q_scale, kv_scale)
    cache_key = (band_columns, write_stats, output.dtype)
    reduce_fn = _PACKED_REDUCE_COMPILE_CACHE.get(cache_key)
    if reduce_fn is None:
        reduce_fn = _compile_cutedsl(
            packed_split_reduce_host,
            *ptrs,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            stream,
            *(cutlass.Int32(1) for _ in range(6)),
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
    )
