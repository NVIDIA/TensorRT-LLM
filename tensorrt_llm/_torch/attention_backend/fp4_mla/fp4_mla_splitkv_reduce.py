# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared FP32 numerator/softmax-state reduction for FP4 MLA split-KV."""

import math
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from .fp4_mla_cutedsl_mufu16 import (
    _compile_cutedsl,
    _current_cu_stream,
    _cutlass_output_dtype,
    _validate_output_dtype,
    _validate_tensor_pointer_alignment,
)

_SPLIT_KV_REDUCE_COMPILE_CACHE: dict[tuple[bool, torch.dtype], Callable] = {}


@cute.kernel
def split_kv_reduce_kernel(
    mSplitO_smnl: cute.Tensor,
    mSplitRowMax_sml: cute.Tensor,
    mSplitRowSum_sml: cute.Tensor,
    mC_mnl: cute.Tensor,
    mFinalRowMax_ml: cute.Tensor,
    mFinalRowSum_ml: cute.Tensor,
    softmax_scale_log2: cutlass.Float32,
    output_scale: cutlass.Float32,
    split_kv: cutlass.Int32,
    store_final_stats: cutlass.Constexpr,
) -> None:
    """Reduce one row and up to 512 contiguous value columns per CTA.

    Warp zero computes each split weight once and shares it across the CTA.
    Each thread accumulates four columns, with neighboring lanes addressing
    neighboring values in the adapter's [split, query, head, value] storage.
    Split count, row count, column count, and strides remain runtime values.
    """
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    threads_per_block: cutlass.Constexpr = 128
    columns_per_thread: cutlass.Constexpr = 4
    tile_columns: cutlass.Constexpr = threads_per_block * columns_per_thread
    m = mC_mnl.shape[0]
    n = mC_mnl.shape[1]
    l_batch = mC_mnl.shape[2]
    row_idx = bidx % cutlass.Int32(m * l_batch)
    row = row_idx % cutlass.Int32(m)
    batch = row_idx // cutlass.Int32(m)
    first_col = bidx // cutlass.Int32(m * l_batch) * cutlass.Int32(tile_columns)
    # 32 split weights, followed by the row maximum and denominator.
    shared = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, 34), cute.make_layout(34))
    numerator = cute.make_rmem_tensor((columns_per_thread,), cutlass.Float32)
    numerator.fill(0.0)
    local_denominator = cutlass.Float32(0.0)
    global_row_max = cutlass.Float32(-cutlass.Float32.inf)

    if tidx < cutlass.Int32(32):
        for split_idx in cutlass.range(tidx, split_kv, 32, unroll=1):
            split_row_sum = mSplitRowSum_sml[split_idx, row, batch]
            if split_row_sum != cutlass.Float32(0.0):
                global_row_max = cute.arch.fmax(
                    global_row_max, mSplitRowMax_sml[split_idx, row, batch]
                )
        global_row_max = cute.arch.warp_redux_sync(global_row_max, "fmax")
        if tidx == cutlass.Int32(0):
            shared[32] = global_row_max
    cute.arch.sync_threads()
    global_row_max = shared[32]

    for split_base in cutlass.range(0, split_kv, 32, unroll=1):
        if tidx < cutlass.Int32(32):
            split_idx = split_base + tidx
            split_scale = cutlass.Float32(0.0)
            if split_idx < split_kv:
                split_row_sum = mSplitRowSum_sml[split_idx, row, batch]
                if split_row_sum != cutlass.Float32(0.0):
                    split_scale = cute.exp2(
                        (mSplitRowMax_sml[split_idx, row, batch] - global_row_max)
                        * softmax_scale_log2,
                        fastmath=True,
                    )
                    local_denominator += split_row_sum * split_scale
            shared[tidx] = split_scale
        cute.arch.sync_threads()

        for split_offset in cutlass.range(0, cutlass.min(32, split_kv - split_base), 1, unroll=1):
            split_scale = shared[split_offset]
            if split_scale != cutlass.Float32(0.0):
                for column_idx in cutlass.range_constexpr(columns_per_thread):
                    col = first_col + tidx + column_idx * threads_per_block
                    if col < n:
                        numerator[column_idx] += (
                            mSplitO_smnl[split_base + split_offset, row, col, batch] * split_scale
                        )
        # Every warp must finish using the weights before warp zero replaces them.
        cute.arch.sync_threads()

    denominator = cutlass.Float32(0.0)
    if tidx < cutlass.Int32(32):
        denominator = cute.arch.warp_reduction_sum(local_denominator)
        if tidx == cutlass.Int32(0):
            shared[33] = denominator
            if cutlass.const_expr(store_final_stats):
                if first_col == cutlass.Int32(0):
                    mFinalRowMax_ml[row, batch] = global_row_max
                    mFinalRowSum_ml[row, batch] = denominator
    cute.arch.sync_threads()
    denominator = shared[33]

    for column_idx in cutlass.range_constexpr(columns_per_thread):
        col = first_col + tidx + column_idx * threads_per_block
        if col < n:
            out = cutlass.Float32(0.0)
            if denominator != cutlass.Float32(0.0):
                out = numerator[column_idx] / denominator * output_scale
            mC_mnl[row, col, batch] = out.to(mC_mnl.element_type)


@cute.jit
def split_kv_reduce_host(
    split_o_ptr: cute.Pointer,
    split_row_max_ptr: cute.Pointer,
    split_row_sum_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    final_row_max_ptr: cute.Pointer,
    final_row_sum_ptr: cute.Pointer,
    softmax_scale_log2: cutlass.Float32,
    output_scale: cutlass.Float32,
    stream: cuda.CUstream,
    split_kv: cutlass.Int32,
    m: cutlass.Int32,
    n: cutlass.Int32,
    l_batch: cutlass.Int32,
    split_o_s0: cutlass.Int64,
    split_o_s1: cutlass.Int64,
    split_o_s2: cutlass.Int64,
    split_o_s3: cutlass.Int64,
    split_row_max_s0: cutlass.Int64,
    split_row_max_s1: cutlass.Int64,
    split_row_max_s2: cutlass.Int64,
    split_row_sum_s0: cutlass.Int64,
    split_row_sum_s1: cutlass.Int64,
    split_row_sum_s2: cutlass.Int64,
    c_s0: cutlass.Int64,
    c_s1: cutlass.Int64,
    c_s2: cutlass.Int64,
    final_row_max_s0: cutlass.Int64,
    final_row_max_s1: cutlass.Int64,
    final_row_sum_s0: cutlass.Int64,
    final_row_sum_s1: cutlass.Int64,
    store_final_stats: cutlass.Constexpr,
) -> None:
    """Launch split reduction with runtime extents and concrete strides."""
    split_o_layout = cute.make_layout(
        (split_kv, m, n, l_batch),
        stride=(split_o_s0, split_o_s1, split_o_s2, split_o_s3),
    )
    split_row_max_layout = cute.make_layout(
        (split_kv, m, l_batch),
        stride=(split_row_max_s0, split_row_max_s1, split_row_max_s2),
    )
    split_row_sum_layout = cute.make_layout(
        (split_kv, m, l_batch),
        stride=(split_row_sum_s0, split_row_sum_s1, split_row_sum_s2),
    )
    c_layout = cute.make_layout((m, n, l_batch), stride=(c_s0, c_s1, c_s2))
    final_row_max_layout = cute.make_layout(
        (m, l_batch), stride=(final_row_max_s0, final_row_max_s1)
    )
    final_row_sum_layout = cute.make_layout(
        (m, l_batch), stride=(final_row_sum_s0, final_row_sum_s1)
    )
    mSplitO_smnl = cute.make_tensor(split_o_ptr, split_o_layout)
    mSplitRowMax_sml = cute.make_tensor(split_row_max_ptr, split_row_max_layout)
    mSplitRowSum_sml = cute.make_tensor(split_row_sum_ptr, split_row_sum_layout)
    mC_mnl = cute.make_tensor(c_ptr, c_layout)
    mFinalRowMax_ml = cute.make_tensor(final_row_max_ptr, final_row_max_layout)
    mFinalRowSum_ml = cute.make_tensor(final_row_sum_ptr, final_row_sum_layout)
    split_kv_reduce_kernel(
        mSplitO_smnl,
        mSplitRowMax_sml,
        mSplitRowSum_sml,
        mC_mnl,
        mFinalRowMax_ml,
        mFinalRowSum_ml,
        softmax_scale_log2,
        output_scale,
        split_kv,
        store_final_stats,
    ).launch(
        grid=(m * l_batch * cute.ceil_div(n, 512), 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )


def _make_split_reduce_ptrs(
    split_o_data_ptr: int,
    split_row_max_data_ptr: int,
    split_row_sum_data_ptr: int,
    c_data_ptr: int,
    final_row_max_data_ptr: int,
    final_row_sum_data_ptr: int,
    output_dtype: torch.dtype = torch.float16,
) -> tuple[cute.Pointer, ...]:
    return (
        make_ptr(
            cutlass.Float32,
            split_o_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            cutlass.Float32,
            split_row_max_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            cutlass.Float32,
            split_row_sum_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            _cutlass_output_dtype(output_dtype),
            c_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            cutlass.Float32,
            final_row_max_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            cutlass.Float32,
            final_row_sum_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
    )


def _compile_split_kv_reduce(
    split_o_data_ptr: int,
    split_row_max_data_ptr: int,
    split_row_sum_data_ptr: int,
    c_data_ptr: int,
    final_row_max_data_ptr: int,
    final_row_sum_data_ptr: int,
    store_final_stats: bool,
    stream: cuda.CUstream,
    output_dtype: torch.dtype = torch.float16,
) -> Callable:
    cache_key = (store_final_stats, output_dtype)
    cached = _SPLIT_KV_REDUCE_COMPILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ptrs = _make_split_reduce_ptrs(
        split_o_data_ptr,
        split_row_max_data_ptr,
        split_row_sum_data_ptr,
        c_data_ptr,
        final_row_max_data_ptr,
        final_row_sum_data_ptr,
        output_dtype,
    )
    compiled = _compile_cutedsl(
        split_kv_reduce_host,
        *ptrs,
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        stream,
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        *(cutlass.Int64(1) for _ in range(17)),
        store_final_stats,
    )
    _SPLIT_KV_REDUCE_COMPILE_CACHE[cache_key] = compiled
    return compiled


def fp4_mla_split_kv_reduce(
    split_o: torch.Tensor,
    split_row_max: torch.Tensor,
    split_row_sum: torch.Tensor,
    out: torch.Tensor,
    final_row_max: torch.Tensor | None = None,
    final_row_sum: torch.Tensor | None = None,
    *,
    softmax_scale_log2: float,
    output_scale: float = 1.0,
) -> None:
    """Reduce FP32 split numerators/row state into final FP16 or BF16 output."""
    if split_o.dtype != torch.float32:
        raise TypeError(f"split_o must be torch.float32, got {split_o.dtype}")
    if split_row_max.dtype != torch.float32 or split_row_sum.dtype != torch.float32:
        raise TypeError("split_row_max and split_row_sum must be torch.float32")
    _validate_output_dtype(out.dtype)
    if not math.isfinite(softmax_scale_log2) or softmax_scale_log2 <= 0.0:
        raise ValueError(
            f"softmax_scale_log2 must be finite and positive, got {softmax_scale_log2}"
        )
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError(f"output_scale must be finite and positive, got {output_scale}")
    if split_o.dim() != 4:
        raise ValueError(f"split_o must be [S, M, N, L], got {tuple(split_o.shape)}")
    if split_row_max.shape != split_row_sum.shape or split_row_max.dim() != 3:
        raise ValueError(
            "split_row_max and split_row_sum must both be [S, M, L], got "
            f"{tuple(split_row_max.shape)} and {tuple(split_row_sum.shape)}"
        )
    split_kv, m, n, l_batch = (int(dim) for dim in split_o.shape)
    if split_kv <= 0:
        raise ValueError(f"split_kv must be positive, got {split_kv}")
    if tuple(split_row_max.shape) != (split_kv, m, l_batch):
        raise ValueError(
            f"split row stats shape mismatch: expected {(split_kv, m, l_batch)}, "
            f"got {tuple(split_row_max.shape)}"
        )
    if tuple(out.shape) != (m, n, l_batch):
        raise ValueError(f"out shape mismatch: expected {(m, n, l_batch)}, got {tuple(out.shape)}")
    tensors = (split_o, split_row_max, split_row_sum, out)
    if any(tensor.device != out.device for tensor in tensors):
        raise ValueError("split_o, split row stats, and out must be on the same device")
    store_final_stats = final_row_max is not None or final_row_sum is not None
    if store_final_stats:
        if final_row_max is None or final_row_sum is None:
            raise ValueError("final_row_max and final_row_sum must be provided together")
        if final_row_max.dtype != torch.float32 or final_row_sum.dtype != torch.float32:
            raise TypeError("final row stats must be torch.float32")
        if tuple(final_row_max.shape) != (m, l_batch) or tuple(final_row_sum.shape) != (
            m,
            l_batch,
        ):
            raise ValueError(
                f"final row stats must both have shape {(m, l_batch)}, got "
                f"{tuple(final_row_max.shape)} and {tuple(final_row_sum.shape)}"
            )
        if final_row_max.device != out.device or final_row_sum.device != out.device:
            raise ValueError("final row stats and out must be on the same device")
        final_row_max_tensor = final_row_max
        final_row_sum_tensor = final_row_sum
    else:
        final_row_max_tensor = split_row_max[0]
        final_row_sum_tensor = split_row_sum[0]

    launch_tensors = (
        ("split_o", split_o),
        ("split_row_max", split_row_max),
        ("split_row_sum", split_row_sum),
        ("out", out),
        ("final_row_max", final_row_max_tensor),
        ("final_row_sum", final_row_sum_tensor),
    )
    for name, tensor in launch_tensors:
        strides = tuple(int(stride) for stride in tensor.stride())
        if any(stride <= 0 for stride in strides):
            raise ValueError(f"{name} strides must be positive, got {strides}")
        _validate_tensor_pointer_alignment(name, tensor)

    stream = _current_cu_stream()
    reduce_fn = _compile_split_kv_reduce(
        split_o.data_ptr(),
        split_row_max.data_ptr(),
        split_row_sum.data_ptr(),
        out.data_ptr(),
        final_row_max_tensor.data_ptr(),
        final_row_sum_tensor.data_ptr(),
        store_final_stats,
        stream,
        out.dtype,
    )
    ptrs = _make_split_reduce_ptrs(
        split_o.data_ptr(),
        split_row_max.data_ptr(),
        split_row_sum.data_ptr(),
        out.data_ptr(),
        final_row_max_tensor.data_ptr(),
        final_row_sum_tensor.data_ptr(),
        out.dtype,
    )
    reduce_fn(
        *ptrs,
        cutlass.Float32(softmax_scale_log2),
        cutlass.Float32(output_scale),
        stream,
        cutlass.Int32(split_kv),
        cutlass.Int32(m),
        cutlass.Int32(n),
        cutlass.Int32(l_batch),
        *(cutlass.Int64(stride) for stride in split_o.stride()),
        *(cutlass.Int64(stride) for stride in split_row_max.stride()),
        *(cutlass.Int64(stride) for stride in split_row_sum.stride()),
        *(cutlass.Int64(stride) for stride in out.stride()),
        *(cutlass.Int64(stride) for stride in final_row_max_tensor.stride()),
        *(cutlass.Int64(stride) for stride in final_row_sum_tensor.stride()),
    )
