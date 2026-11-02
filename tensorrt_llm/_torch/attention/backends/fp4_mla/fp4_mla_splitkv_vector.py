# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel split reduction with live device-scale conversion."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_splitkv_vector_kernel(
    partial,
    maxima,
    sums,
    output,
    q_scale,
    kv_scale,
    QUERIES: tl.constexpr,
    OUTPUT_HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_QUERY_STRIDE: tl.constexpr,
    OUT_HEAD_STRIDE: tl.constexpr,
    OUT_COLUMN_STRIDE: tl.constexpr,
    SOFTMAX_SCALE_LOG2: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    row = tl.program_id(0)
    query = row // OUTPUT_HEADS
    head = row % OUTPUT_HEADS
    split = tl.arange(0, BLOCK_SPLITS)
    col = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
    state_index = split * QUERIES * 128 + query * 128 + head
    row_sum = tl.load(sums + state_index, split < SPLITS, other=0)
    valid = (split < SPLITS) & (row_sum != 0)
    scale = SOFTMAX_SCALE_LOG2 / (tl.load(q_scale) * tl.load(kv_scale))
    raw_max = tl.load(maxima + state_index, valid, other=0)
    scaled_max = tl.where(valid, raw_max * scale, -float("inf"))
    global_max = tl.max(scaled_max, 0)
    weights = tl.exp2(tl.where(valid, scaled_max - global_max, -float("inf")))
    denominator = tl.sum(row_sum * weights, 0)
    values = tl.load(
        partial + state_index[:, None] * 512 + col[None, :],
        (weights[:, None] != 0) & (col[None, :] < 512),
        other=0,
    )
    numerator = tl.sum(values * weights[:, None], 0)
    result = tl.where(denominator != 0, numerator / denominator * OUTPUT_SCALE, 0)
    tl.store(
        output + query * OUT_QUERY_STRIDE + head * OUT_HEAD_STRIDE + col * OUT_COLUMN_STRIDE,
        result,
        col < 512,
    )


def reduce_splitkv_vector(
    partial: torch.Tensor,
    maxima: torch.Tensor,
    sums: torch.Tensor,
    output: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    *,
    splits: int,
    queries: int,
    softmax_scale_log2: float,
    output_scale: float,
    tile_columns: int = 128,
) -> None:
    """Reduce [split * query, 128, 512] partials to FP16/BF16 logical-head output."""
    assert partial.shape == (splits * queries, 128, 512)
    assert maxima.shape == sums.shape == (splits * queries, 128)
    assert output.shape[0] == queries and output.shape[2] == 512
    assert tile_columns in (64, 128, 256, 512)
    _reduce_splitkv_vector_kernel[(queries * output.shape[1], triton.cdiv(512, tile_columns))](
        partial,
        maxima,
        sums,
        output,
        q_scale,
        kv_scale,
        queries,
        output.shape[1],
        splits,
        *output.stride(),
        softmax_scale_log2,
        output_scale,
        triton.next_power_of_2(splits),
        tile_columns,
        num_warps=4,
    )
