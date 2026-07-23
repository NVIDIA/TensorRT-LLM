# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels, launch helpers, and mean-phase table builders for TriAttention
(fp32 math; int64 past-2^31 flat offsets; masked ragged tails; scoring in the CuTe pack)."""

from __future__ import annotations

from typing import Dict

import torch
import triton
import triton.language as tl

# ---- Mean-phase table: RoPE-style position table of mean trig phases ----


# Positions past this row count are not exactly representable in fp32.
_MEAN_PHASE_MAX_ROWS = 1 << 24

# Score z-normalization epsilon; must stay a plain float (the CuTe DSL traces it).
STD_EPSILON = 1e-6


@triton.jit
def _gather_mean_phase_kernel(
    round_starts,
    table_cos,
    table_sin,
    mean_cos,
    mean_sin,
    valid_seq_lens,
    token_starts,
    valid_widths,
    swa_destination_bases,
    table_rows,
    rebase_delta,
    NUM_FREQS: tl.constexpr,
    F_BLOCK: tl.constexpr,
    HAS_SWA: tl.constexpr,
):
    """Copy each request's phase-table row; derive valid widths and SWA landing bases."""
    request = tl.program_id(0)
    frequency = tl.arange(0, F_BLOCK)
    frequency_mask = frequency < NUM_FREQS
    table_row = tl.load(round_starts + request).to(tl.int64)
    # Clamp stale or padded round starts instead of faulting.
    table_row = tl.minimum(tl.maximum(table_row, 0), table_rows - 1)
    source_offset = table_row * NUM_FREQS + frequency
    output_offset = request * NUM_FREQS + frequency
    row_cos = tl.load(table_cos + source_offset, mask=frequency_mask, other=0.0)
    row_sin = tl.load(table_sin + source_offset, mask=frequency_mask, other=0.0)
    tl.store(mean_cos + output_offset, row_cos, mask=frequency_mask)
    tl.store(mean_sin + output_offset, row_sin, mask=frequency_mask)
    token_start = tl.load(token_starts + request)
    tl.store(valid_widths + request, tl.load(valid_seq_lens + request) - token_start)
    if HAS_SWA:
        tl.store(swa_destination_bases + request, token_start + rebase_delta)


def grow_mean_phase_table(phase: Dict[str, object], rows: int) -> None:
    """Cover positions ``[0, rows)``, rebuilding the table if it must grow."""
    rows = int(rows)
    if rows <= phase["rows"]:
        return
    if rows > _MEAN_PHASE_MAX_ROWS:
        raise ValueError(f"a {rows}-row mean-phase table exceeds the exact-FP32 position range")
    target = 1
    while target < rows:
        target *= 2
    target = min(max(target, 2 * phase["rows"]), _MEAN_PHASE_MAX_ROWS)
    omega = phase["omega"]
    positions = torch.arange(target, device=omega.device, dtype=torch.float32)
    cos_table = torch.zeros((target, omega.numel()), dtype=torch.float32, device=omega.device)
    sin_table = torch.zeros_like(cos_table)
    # Fixed summation order keeps the table bit-stable across rebuilds.
    for offset in phase["offset_values"]:
        angle = torch.outer(positions + offset, omega)
        cos_table += torch.cos(angle)
        sin_table += torch.sin(angle)
    scale = 1.0 / len(phase["offset_values"])
    phase["cos"] = cos_table.mul_(scale)
    phase["sin"] = sin_table.mul_(scale)
    phase["rows"] = target


# ---- Selection: combine scores per mode, then finalize the top-k set ----


@triton.jit
def _score_row_stats_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
):
    """Compute one valid-prefix mean and inverse standard deviation per score row."""
    flat_row = tl.program_id(0)
    request = flat_row // ROWS
    valid_width = tl.load(valid_widths + request)
    score_row = scores + flat_row * WIDTH
    lane = tl.arange(0, BLOCK)
    score_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < valid_width
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        score_sum += tl.sum(value, axis=0)
    mean = score_sum / valid_width
    square_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < valid_width
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        centered = tl.where(valid, value - mean, 0.0)
        square_sum += tl.sum(centered * centered, axis=0)
    std = tl.sqrt(square_sum / valid_width)
    tl.store(row_mean + flat_row, mean)
    tl.store(row_inv_std + flat_row, 1.0 / tl.maximum(std, EPSILON))


@triton.jit
def _score_per_head_reduce_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    selection_scores,
    selection_seq_lens,
    NUM_LAYERS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PER_LAYER: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reduce query-head score rows into one selector row per KV-head domain."""
    request = tl.program_id(0)
    selection_row = tl.program_id(1)
    token_block = tl.program_id(2)
    token = token_block * BLOCK + tl.arange(0, BLOCK)
    valid_width = tl.load(valid_widths + request)
    valid_token = token < valid_width

    if token_block == 0:
        tl.store(
            selection_seq_lens + request * SELECTION_ROWS + selection_row,
            valid_width,
        )

    kv_head = selection_row % NUM_KV_HEADS
    if PER_LAYER:
        layer = selection_row // NUM_KV_HEADS
        reduced = tl.full((BLOCK,), -float("inf"), tl.float32)
        for query_in_group in tl.static_range(0, QUERY_GROUP_SIZE):
            query_head = kv_head * QUERY_GROUP_SIZE + query_in_group
            flat_row = (request * NUM_LAYERS + layer) * NUM_QUERY_HEADS + query_head
            value = tl.load(
                scores + flat_row * WIDTH + token,
                mask=valid_token,
                other=-float("inf"),
            ).to(tl.float32)
            if NORMALIZE:
                mean = tl.load(row_mean + flat_row)
                inv_std = tl.load(row_inv_std + flat_row)
                value = tl.where(valid_token, (value - mean) * inv_std, -float("inf"))
            reduced = tl.maximum(reduced, value)
    else:
        reduced = tl.zeros((BLOCK,), tl.float32)
        for layer in tl.static_range(0, NUM_LAYERS):
            layer_max = tl.full((BLOCK,), -float("inf"), tl.float32)
            for query_in_group in tl.static_range(0, QUERY_GROUP_SIZE):
                query_head = kv_head * QUERY_GROUP_SIZE + query_in_group
                flat_row = (request * NUM_LAYERS + layer) * NUM_QUERY_HEADS + query_head
                value = tl.load(
                    scores + flat_row * WIDTH + token,
                    mask=valid_token,
                    other=-float("inf"),
                ).to(tl.float32)
                if NORMALIZE:
                    mean = tl.load(row_mean + flat_row)
                    inv_std = tl.load(row_inv_std + flat_row)
                    value = tl.where(valid_token, (value - mean) * inv_std, -float("inf"))
                layer_max = tl.maximum(layer_max, value)
            reduced += layer_max
        reduced /= NUM_LAYERS

    output = (request * SELECTION_ROWS + selection_row) * WIDTH + token
    tl.store(selection_scores + output, reduced, mask=token < WIDTH)


def prepare_per_head_scores(
    scores: torch.Tensor,
    valid_widths: torch.Tensor,
    row_mean: torch.Tensor,
    row_inv_std: torch.Tensor,
    selection_scores: torch.Tensor,
    selection_seq_lens: torch.Tensor,
    request_count: int,
    *,
    num_kv_heads: int,
    per_layer: bool,
    normalize_scores: bool,
) -> None:
    """Normalize and reduce score rows for either per-head eviction mode."""
    request_count = int(request_count)
    num_kv_heads = int(num_kv_heads)
    _, num_layers, num_query_heads, width = scores.shape
    selection_rows = num_layers * num_kv_heads if per_layer else num_kv_heads
    # 256 lanes / 4 warps, matching the settle shape.
    stats_block = 256
    rows = num_layers * num_query_heads
    if normalize_scores:
        _score_row_stats_kernel[(request_count * rows,)](
            scores,
            valid_widths,
            row_mean,
            row_inv_std,
            ROWS=rows,
            WIDTH=width,
            BLOCK=stats_block,
            EPSILON=STD_EPSILON,
            num_warps=4,
        )
    reduction_block = 256
    _score_per_head_reduce_kernel[
        (request_count, selection_rows, triton.cdiv(width, reduction_block))
    ](
        scores,
        valid_widths,
        row_mean,
        row_inv_std,
        selection_scores,
        selection_seq_lens,
        NUM_LAYERS=num_layers,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        QUERY_GROUP_SIZE=num_query_heads // num_kv_heads,
        SELECTION_ROWS=selection_rows,
        WIDTH=width,
        PER_LAYER=per_layer,
        NORMALIZE=normalize_scores,
        BLOCK=reduction_block,
        num_warps=4,
    )


# ---- Selection finalize: settle threshold ties into the kept-ordinal rows ----


# Settle launch shape, shared by every launch site.
SETTLE_BLOCK = 256
SETTLE_NUM_WARPS = 4


@triton.jit
def _settle_ties_kernel(
    scores,
    seq_lens,
    prompt_offsets,
    provisional_indices,
    output_indices,
    WIDTH: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Settle one selection row's ties into its kept-ordinal output row
    (threshold recovery with sentinel-skip, strictly-greater count,
    lowest-index tie quota, ascending prompt-rebased emission; entries past
    the emitted count keep their previous value)."""
    request = tl.program_id(0)
    selection_domain = tl.program_id(1)
    row = request * SELECTION_ROWS + selection_domain
    row_output = output_indices + row * KEEP_COUNT
    row_scores = scores + row * WIDTH
    row_selected = provisional_indices + row * KEEP_COUNT
    # Rebases the decode-relative ordinals to absolute positions (per request:
    # every selection row of a request shares its pinned prompt length).
    prompt_len = tl.load(prompt_offsets + request)

    threshold = float("inf")
    for start in tl.static_range(0, KEEP_COUNT, BLOCK):
        selected_offset = start + tl.arange(0, BLOCK)
        selected_mask = selected_offset < KEEP_COUNT
        token_index = tl.load(
            row_selected + selected_offset,
            mask=selected_mask,
            other=0,
        )
        # Mask the top-k's -1 pad sentinels so no lane dereferences ``row_scores - 1``.
        selected_valid = selected_mask & (token_index >= 0)
        selected_score = tl.load(
            row_scores + token_index,
            mask=selected_valid,
            other=float("inf"),
        ).to(tl.float32)
        threshold = tl.minimum(threshold, tl.min(selected_score, axis=0))

    seq_len = tl.load(seq_lens + row)
    greater_count = 0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        valid = (token_index < WIDTH) & (token_index < seq_len)
        score = tl.load(
            row_scores + token_index,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        greater_count += tl.sum((valid & (score > threshold)).to(tl.int32))

    tie_quota = KEEP_COUNT - greater_count
    output_count = 0
    ties_seen = 0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        valid = (token_index < WIDTH) & (token_index < seq_len)
        score = tl.load(
            row_scores + token_index,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        greater = valid & (score > threshold)
        tied = valid & (score == threshold)
        tied_i32 = tied.to(tl.int32)
        tie_rank = ties_seen + tl.cumsum(tied_i32, axis=0) - tied_i32
        selected = greater | (tied & (tie_rank < tie_quota))
        selected_i32 = selected.to(tl.int32)
        write_offset = output_count + tl.cumsum(selected_i32, axis=0) - selected_i32
        tl.store(
            row_output + write_offset,
            token_index + prompt_len,
            mask=selected,
        )
        output_count += tl.sum(selected_i32)
        ties_seen += tl.sum(tied_i32)
