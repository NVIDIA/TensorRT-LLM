# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-updated split-KV adapter for the CuTeDSL FP4 MLA backends."""

import math
import os
from types import ModuleType
from typing import Any

import torch
import triton
import triton.language as tl

_SPLIT_KV_ENV = "TRTLLM_FP4_MLA_SPLIT_KV"
_MAX_PARTITION_TOKENS = 160 * 1024


def select_split_count(max_kv_len: int) -> int:
    """Use one partition by default, or bounded partitions for long contexts.

    The environment override is a testing/tuning control. The default
    long-context size is conservative and has not been performance tuned.
    """
    value = os.getenv(_SPLIT_KV_ENV, "auto")
    if value == "auto":
        if max_kv_len <= _MAX_PARTITION_TOKENS:
            return 1
        return triton.next_power_of_2(triton.cdiv(max_kv_len, 16 * 1024))
    splits = int(value)
    if splits < 1:
        raise ValueError(f"{_SPLIT_KV_ENV} must be 'auto' or a positive integer")
    if splits == 1:
        return 1  # The native kernel validates its geometric runtime profile.
    if triton.cdiv(triton.cdiv(max_kv_len, 256), splits) * 256 > _MAX_PARTITION_TOKENS:
        raise ValueError(f"{splits} splits do not cover max_kv_len={max_kv_len}")
    return splits


@triton.jit
def _repeat_q_kernel(
    q_ptr,
    sf_ptr,
    repeated_q_ptr,
    repeated_sf_ptr,
    Q_BYTES: tl.constexpr,
    SF_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.program_id(1)
    q = tl.load(q_ptr + offsets, offsets < Q_BYTES, other=0)
    tl.store(repeated_q_ptr + split * Q_BYTES + offsets, q, offsets < Q_BYTES)
    sf = tl.load(sf_ptr + offsets, offsets < SF_BYTES, other=0)
    tl.store(repeated_sf_ptr + split * SF_BYTES + offsets, sf, offsets < SF_BYTES)


@triton.jit
def _partition_pages_kernel(
    src_pages,
    src_indptr,
    lengths,
    dst_pages,
    dst_indptr,
    dst_lengths,
    NUM_QUERIES: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
    PARTITION_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    split = task // NUM_QUERIES
    query = task % NUM_QUERIES
    sequence = query // QUERY_LENGTH
    query_offset = query % QUERY_LENGTH
    length = tl.maximum(tl.load(lengths + sequence), 0)
    tiles = tl.cdiv(tl.cdiv(length, 256), SPLITS)
    causal_end = tl.maximum(length - (QUERY_LENGTH - 1 - query_offset), 0)
    begin = tl.minimum(split * tiles * 256, causal_end)
    end = tl.minimum((split + 1) * tiles * 256, causal_end)
    valid = tl.maximum(end - begin, 0)
    pages = tl.cdiv(valid, 128)
    source_begin = tl.load(src_indptr + sequence)
    source_end = tl.load(src_indptr + sequence + 1)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = source_begin + begin // 128 + offsets
    page = tl.load(src_pages + source, (offsets < pages) & (source < source_end), other=0)
    tl.store(dst_pages + task * PARTITION_PAGES + offsets, page, offsets < PARTITION_PAGES)
    if tl.program_id(1) == 0:
        tl.store(dst_lengths + task, valid)
        tl.store(dst_indptr + task, task * PARTITION_PAGES)
        if task == NUM_QUERIES * SPLITS - 1:
            tl.store(dst_indptr + task + 1, (task + 1) * PARTITION_PAGES)


@triton.jit
def _scale_row_max_kernel(
    maxima,
    q_scale,
    kv_scale,
    COUNT: tl.constexpr,
    SOFTMAX_SCALE_LOG2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scale = SOFTMAX_SCALE_LOG2 / (tl.load(q_scale) * tl.load(kv_scale))
    values = tl.load(maxima + offsets, offsets < COUNT, other=0)
    tl.store(maxima + offsets, values * scale, offsets < COUNT)


def run_fp4_mla_splitkv(
    module: ModuleType,
    metadata: Any,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor | None,
    v_sf: torch.Tensor,
    kv_global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    src_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    splits: int,
    max_kv_len: int,
    sm_scale: float,
    q_global_scale: torch.Tensor,
    query_len_per_seq: int,
    v_page_offset: int,
    q_batch_capacity: int,
) -> None:
    """Launch split MLA using live raw TRTLLM Q, scales, and physical KV pages.

    Q contains contiguous packed bytes for [queries, 128, 320]. QSF uses the
    TRTLLM swizzle for the same rows. Output is [queries, 1-128, 512], FP16/BF16.
    KV lengths include the final query of each uniform linear-MTP request.
    Workspace is owned by metadata and must be warmed before graph capture.
    Each graph bucket reserves only its fixed query count; eager metadata
    reserves the configured query capacity for reuse across batch sizes.
    All input replication and partition planning execute on the current stream
    on every invocation; graph replay therefore consumes the current inputs.
    """
    from . import _ensure_workspace_tensor
    from .fp4_mla_splitkv_reduce import fp4_mla_split_kv_reduce

    queries, output_heads, value_dim = output.shape
    heads = 128
    if not 0 < output_heads <= heads or value_dim != 512:
        raise ValueError("split-KV requires 1-128 output heads and 512 value features")
    if query_len_per_seq <= 0 or queries % query_len_per_seq:
        raise ValueError("query_len_per_seq must divide the number of queries")
    if q_batch_capacity < queries:
        raise ValueError("Q capacity must cover the active query count")
    tasks = splits * queries
    workspace_queries = queries if getattr(metadata, "is_cuda_graph", False) else q_batch_capacity
    task_capacity = splits * workspace_queries
    if splits < 2 or tasks > 65535:
        raise ValueError(f"split-KV requires splits >= 2 and <= 65535 tasks, got {tasks}")
    partition_pages = triton.cdiv(triton.cdiv(max_kv_len, 256), splits) * 2
    if not 0 < partition_pages * 128 <= module.SMEM_P4_RUNTIME_MAX_KV:
        raise ValueError("split-KV partition exceeds the kernel's runtime length limit")
    if q_fp4.dtype != torch.uint8 or q_fp4.numel() != queries * heads * 320:
        raise ValueError("split-KV Q must contain packed [queries, 128, 320] bytes")
    if q_sf.element_size() != 1 or q_sf.numel() != queries * heads * 40:
        raise ValueError("split-KV QSF must contain the swizzled scales for all query rows")
    if not q_fp4.is_contiguous() or not q_sf.is_contiguous():
        raise ValueError("split-KV Q and QSF must be contiguous")
    sequences = queries // query_len_per_seq
    if (
        kv_lens.shape != (sequences,)
        or src_indptr.shape != (sequences + 1,)
        or kv_lens.dtype != torch.int32
        or src_indptr.dtype != torch.int32
        or src_page_ids.dtype != torch.int32
        or src_page_ids.ndim != 1
    ):
        raise ValueError("split-KV requires int32 lengths, CSR offsets, and page IDs")
    inputs = (q_fp4, q_sf, kv_lens, src_indptr, src_page_ids)
    if any(t.device != output.device or not t.is_contiguous() for t in inputs):
        raise ValueError("split-KV inputs must be contiguous on the output device")

    def workspace(name: str, count: int, capacity: int, dtype: torch.dtype) -> torch.Tensor:
        # Graph buckets own separate metadata and have fixed query counts.
        # Avoid reserving the full engine capacity independently in each bucket.
        # Eager metadata still reserves capacity to reuse storage across batches.
        return _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_splitkv_" + name,
            (capacity,),
            dtype=dtype,
            device=output.device,
        )[:count]

    repeated_q = workspace("q", tasks * heads * 320, task_capacity * heads * 320, torch.uint8)
    repeated_sf = workspace("q_sf", tasks * heads * 40, task_capacity * heads * 40, torch.uint8)
    page_ids = workspace(
        "pages", tasks * partition_pages, task_capacity * partition_pages, torch.int32
    )
    indptr = workspace("indptr", tasks + 1, task_capacity + 1, torch.int32)
    lengths = workspace("lengths", tasks, task_capacity, torch.int32)
    partial = workspace(
        "partial", tasks * heads * value_dim, task_capacity * heads * value_dim, torch.float32
    ).view(tasks, heads, value_dim)
    maxima = workspace("maxima", tasks * heads, task_capacity * heads, torch.float32).view(
        tasks, heads
    )
    sums = workspace("sums", tasks * heads, task_capacity * heads, torch.float32).view(tasks, heads)

    _repeat_q_kernel[(triton.cdiv(q_fp4.numel(), 1024), splits)](
        q_fp4,
        q_sf.view(torch.uint8),
        repeated_q,
        repeated_sf,
        q_fp4.numel(),
        q_sf.numel(),
        1024,
    )
    _partition_pages_kernel[(tasks, triton.cdiv(partition_pages, 128))](
        src_page_ids,
        src_indptr,
        kv_lens,
        page_ids,
        indptr,
        lengths,
        queries,
        query_len_per_seq,
        splits,
        partition_pages,
        128,
    )
    # The partial-only main launch never writes its final-output pointer. Alias
    # the numerator allocation to satisfy its output ABI without another buffer.
    unused_output = (
        partial.view(output.dtype).flatten()[: partial.numel()].view(tasks, heads, value_dim)
    )
    module.run_trtllm_fp4_mla_decode_page_native_from_raw(
        repeated_q,
        repeated_sf,
        kv_cache,
        sf_cache,
        v_packed,
        v_sf,
        kv_global_scale,
        page_ids,
        indptr,
        lengths,
        unused_output,
        max_kv_len=partition_pages * 128,
        sm_scale=sm_scale,
        num_heads=heads,
        q_global_scale=q_global_scale,
        page_size=128,
        query_len_per_seq=1,
        v_page_offset=v_page_offset,
        q_batch_capacity=tasks,
        split_kv_buffers=(partial, maxima, sums),
    )
    # These tiles cover the small-head TEP and full-head DEP decode shapes.
    default_columns = 512 if output_heads >= 96 else (128 if queries >= 24 else 256)
    vector_columns = getattr(metadata, "_fp4_mla_splitkv_reduce_tile", default_columns)
    if vector_columns:
        from .fp4_mla_splitkv_vector import reduce_splitkv_vector

        reduce_splitkv_vector(
            partial,
            maxima,
            sums,
            output,
            q_global_scale,
            kv_global_scale,
            splits=splits,
            queries=queries,
            softmax_scale_log2=sm_scale * math.log2(math.e),
            output_scale=1.0 / module.FP4_MLA_P_GLOBAL_SCALE,
            tile_columns=vector_columns,
        )
        return

    # Read global scales on the device so their values need no host sync and
    # may change between graph replays. The existing reducer then uses scale=1.
    _scale_row_max_kernel[(triton.cdiv(maxima.numel(), 256),)](
        maxima,
        q_global_scale,
        kv_global_scale,
        maxima.numel(),
        sm_scale * math.log2(math.e),
        256,
    )
    fp4_mla_split_kv_reduce(
        partial.view(splits, queries, heads, value_dim)[:, :, :output_heads].permute(0, 2, 3, 1),
        maxima.view(splits, queries, heads)[:, :, :output_heads].permute(0, 2, 1),
        sums.view(splits, queries, heads)[:, :, :output_heads].permute(0, 2, 1),
        output.permute(1, 2, 0),
        softmax_scale_log2=1.0,
        output_scale=1.0 / module.FP4_MLA_P_GLOBAL_SCALE,
    )
