# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Split-KV for H6/Q6, with all queries of a request sharing one CTA group."""

import math
from types import ModuleType
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ..trtllm import TrtllmAttentionMetadata


@triton.jit
def _sf_offset(row, col):
    return col % 4 + col // 4 * 512 + row % 32 * 16 + row % 128 // 32 * 4 + row // 128 * 5120


@triton.jit
def _fold_q_kernel(
    q,
    sf,
    dst_q,
    dst_sf,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    task = tl.program_id(1)
    sequence = task % BATCH
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = x // 320, x % 320
    source_row = sequence * WIDTH * HEADS + row
    value = tl.load(q + source_row * 320 + col, (x < 128 * 320) & (row < WIDTH * HEADS), other=0)
    tl.store(dst_q + task * 128 * 320 + x, value, x < 128 * 320)
    sf_row, sf_col = x // 40, x % 40
    sf_value = tl.load(
        sf + _sf_offset(sequence * WIDTH * HEADS + sf_row, sf_col),
        (x < 128 * 40) & (sf_row < WIDTH * HEADS),
        other=0,
    )
    tl.store(dst_sf + task * 5120 + _sf_offset(sf_row, sf_col), sf_value, x < 128 * 40)


@triton.jit
def _fold_pages_kernel(
    src_pages,
    src_indptr,
    lengths,
    dst_pages,
    dst_indptr,
    dst_lengths,
    first_query_bounds,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLITS: tl.constexpr,
    PARTITION_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    split, sequence = task // BATCH, task % BATCH
    length = tl.maximum(tl.load(lengths + sequence), 0)
    span = tl.cdiv(tl.cdiv(length, 256), SPLITS) * 256
    begin = split * span
    valid = tl.minimum(tl.maximum(length - begin, 0), span)
    pages = tl.cdiv(valid, 128)
    source_begin = tl.load(src_indptr + sequence)
    source_end = tl.load(src_indptr + sequence + 1)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = source_begin + begin // 128 + offsets
    page = tl.load(src_pages + source, (offsets < pages) & (source < source_end), other=0)
    tl.store(dst_pages + task * PARTITION_PAGES + offsets, page, offsets < PARTITION_PAGES)
    if tl.program_id(1) == 0:
        tl.store(dst_lengths + task, valid)
        # Keep this signed and unclamped: early query rows may have zero
        # valid tokens even when later rows intersect the last split.
        tl.store(first_query_bounds + task, length - begin - (WIDTH - 1))
        tl.store(dst_indptr + task, task * PARTITION_PAGES)
        if task == BATCH * SPLITS - 1:
            tl.store(dst_indptr + task + 1, (task + 1) * PARTITION_PAGES)


@triton.jit
def _fold_reduce_kernel(
    partial,
    maxima,
    sums,
    output,
    q_scale,
    kv_scale,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_Q: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_D: tl.constexpr,
    SCALE: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    sequence = row // (WIDTH * HEADS)
    local_row = row % (WIDTH * HEADS)
    query, head = row // HEADS, row % HEADS
    split = tl.arange(0, BLOCK_S)
    col = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    state = (split * BATCH + sequence) * 128 + local_row
    row_sum = tl.load(sums + state, split < SPLITS, other=0)
    valid = (split < SPLITS) & (row_sum != 0)
    scale = SCALE / (tl.load(q_scale) * tl.load(kv_scale))
    raw_max = tl.load(maxima + state, valid, other=0)
    scaled_max = tl.where(valid, raw_max * scale, -float("inf"))
    global_max = tl.max(scaled_max, 0)
    weights = tl.exp2(tl.where(valid, scaled_max - global_max, -float("inf")))
    denominator = tl.sum(row_sum * weights, 0)
    values = tl.load(
        partial + state[:, None] * 512 + col[None, :],
        (weights[:, None] != 0) & (col[None, :] < 512),
        other=0,
    )
    numerator = tl.sum(values * weights[:, None], 0)
    result = tl.where(denominator != 0, numerator / denominator * OUTPUT_SCALE, 0)
    tl.store(output + query * OUT_Q + head * OUT_H + col * OUT_D, result, col < 512)


def run_fp4_mla_foldq(
    module: ModuleType,
    metadata: "TrtllmAttentionMetadata",
    q: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor,
    v_sf: torch.Tensor,
    kv_scale: torch.Tensor,
    src_pages: torch.Tensor,
    src_indptr: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    *,
    splits: int,
    max_kv_len: int,
    sm_scale: float,
    q_global_scale: torch.Tensor,
    query_len_per_seq: int,
    v_page_offset: int,
) -> None:
    """Fold six queries and six heads into shared-KV split tasks and reduce to output."""
    from .layout import _ensure_workspace_tensor

    queries, heads, value_dim = output.shape
    width = query_len_per_seq
    if heads != 6 or width != 6 or queries % width or value_dim != 512:
        raise ValueError("folded-query specialization requires H6/Q6/D512")
    batch = queries // width
    tasks = batch * splits
    if not 1 < splits or tasks > 65535:
        raise ValueError("folded-query split count exceeds the launch bounds")
    partition_pages = triton.cdiv(triton.cdiv(max_kv_len, 256), splits) * 2
    if partition_pages * 128 > module.SMEM_P4_RUNTIME_MAX_KV:
        raise ValueError("folded-query partition exceeds the core KV bound")

    def workspace(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return _ensure_workspace_tensor(
            metadata, "_fp4_mla_foldq_" + name, shape, dtype=dtype, device=output.device
        )

    folded_q = workspace("q", (tasks, 128, 320), torch.uint8)
    folded_sf = workspace("sf", (tasks * 5120,), torch.uint8)
    pages = workspace("pages", (tasks * partition_pages,), torch.int32)
    indptr = workspace("indptr", (tasks + 1,), torch.int32)
    valid = workspace("lengths", (tasks,), torch.int32)
    bounds = workspace("bounds", (tasks,), torch.int32)
    partial = workspace("partial", (tasks, 128, 512), torch.float32)
    maxima = workspace("maxima", (tasks, 128), torch.float32)
    sums = workspace("sums", (tasks, 128), torch.float32)
    _fold_q_kernel[(triton.cdiv(128 * 320, 1024), tasks)](
        q, q_sf.view(torch.uint8), folded_q, folded_sf, batch, heads, width, 1024
    )
    _fold_pages_kernel[(tasks, triton.cdiv(partition_pages, 128))](
        src_pages,
        src_indptr,
        lengths,
        pages,
        indptr,
        valid,
        bounds,
        batch,
        width,
        splits,
        partition_pages,
        128,
    )
    unused_output = partial.view(output.dtype).flatten()[: partial.numel()].view(tasks, 128, 512)
    module.run_trtllm_fp4_mla_decode_page_native_from_raw(
        folded_q,
        folded_sf,
        kv_cache,
        sf_cache,
        v_packed,
        v_sf,
        kv_scale,
        pages,
        indptr,
        valid,
        unused_output,
        max_kv_len=partition_pages * 128,
        sm_scale=sm_scale,
        num_heads=128,
        q_global_scale=q_global_scale,
        page_size=128,
        query_len_per_seq=1,
        v_page_offset=v_page_offset,
        q_batch_capacity=tasks,
        helix_kv_bounds=bounds,
        split_kv_buffers=(partial, maxima, sums),
        folded_query_heads=heads,
    )
    columns = getattr(metadata, "_fp4_mla_splitkv_reduce_tile", 256) or 256
    _fold_reduce_kernel[(queries * heads, triton.cdiv(512, columns))](
        partial,
        maxima,
        sums,
        output,
        q_global_scale,
        kv_scale,
        batch,
        heads,
        width,
        splits,
        *output.stride(),
        sm_scale * math.log2(math.e),
        1.0 / module.FP4_MLA_P_GLOBAL_SCALE,
        triton.next_power_of_2(splits),
        columns,
        num_warps=4,
    )
