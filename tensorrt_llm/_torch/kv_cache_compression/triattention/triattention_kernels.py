# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton reduction, TP-fold, and selection kernels for TriAttention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---- Mean-phase gather: per-request phase-row fetch + width derivation ----

# Score z-normalization epsilon; must stay a plain float (the CuTe DSL traces it).
STD_EPSILON = 1e-6


@triton.jit
def _gather_mean_phase_kernel(
    logical_source_lengths,
    phase_cos,
    phase_sin,
    source_lengths,
    prompt_lengths,
    mean_cos,
    mean_sin,
    decode_lengths,
    swa_destination_bases,
    swa_rebase_delta,
    NUM_FREQS: tl.constexpr,
    F_BLOCK: tl.constexpr,
    HAS_SWA: tl.constexpr,
):
    """Copy each request's phase-table row; derive decode lengths and SWA landing bases."""
    request = tl.program_id(0)
    frequency = tl.arange(0, F_BLOCK)
    frequency_mask = frequency < NUM_FREQS
    table_row = tl.load(logical_source_lengths + request).to(tl.int64)
    source_offset = table_row * NUM_FREQS + frequency
    output_offset = request * NUM_FREQS + frequency
    row_cos = tl.load(phase_cos + source_offset, mask=frequency_mask, other=0.0)
    row_sin = tl.load(phase_sin + source_offset, mask=frequency_mask, other=0.0)
    tl.store(mean_cos + output_offset, row_cos, mask=frequency_mask)
    tl.store(mean_sin + output_offset, row_sin, mask=frequency_mask)
    prompt_length = tl.load(prompt_lengths + request)
    tl.store(decode_lengths + request, tl.load(source_lengths + request) - prompt_length)
    if HAS_SWA:
        tl.store(swa_destination_bases + request, prompt_length + swa_rebase_delta)


def gather_mean_phase(
    logical_source_lengths: torch.Tensor,
    phase_cos: torch.Tensor,
    phase_sin: torch.Tensor,
    source_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    mean_cos: torch.Tensor,
    mean_sin: torch.Tensor,
    decode_lengths: torch.Tensor,
    swa_destination_bases: torch.Tensor | None,
    *,
    request_count: int,
    swa_rebase_delta: int,
) -> None:
    """Gather mean-phase rows and derive per-request decode metadata."""
    num_freqs = int(phase_cos.shape[1])
    _gather_mean_phase_kernel[(request_count,)](
        logical_source_lengths,
        phase_cos,
        phase_sin,
        source_lengths,
        prompt_lengths,
        mean_cos,
        mean_sin,
        decode_lengths,
        swa_destination_bases,
        swa_rebase_delta,
        NUM_FREQS=num_freqs,
        F_BLOCK=triton.next_power_of_2(num_freqs),
        HAS_SWA=swa_destination_bases is not None,
        num_warps=1,
    )


# ---- Selection: combine scores per mode, then finalize the top-k set ----


@triton.jit
def _score_row_stats_kernel(
    score_scratch,
    decode_lengths,
    prompt_lengths,
    row_mean,
    row_inv_std,
    segment_tokens,
    ROWS: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PADDED_COLUMNS: tl.constexpr,
    BUCKET: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr = 256,
    # Triton rejects plain-global capture; the default binds the module float
    # at def time (STD_EPSILON itself must stay a plain float for the CuTe import).
    EPSILON: tl.constexpr = STD_EPSILON,
):
    """Compute decode-window mean and inverse standard deviation for each score row."""
    QUERY_GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    flat_row = tl.program_id(0)
    request = flat_row // ROWS
    row_in_request = flat_row % ROWS
    layer = row_in_request // NUM_Q_HEADS
    query_head = row_in_request % NUM_Q_HEADS
    kv_head = query_head // QUERY_GROUP_SIZE
    plane = kv_head * PADDED_COLUMNS + query_head % QUERY_GROUP_SIZE
    decode_length = tl.load(decode_lengths + request)
    prompt_start = tl.load(prompt_lengths + request)
    score_row = (
        score_scratch
        + plane.to(tl.int64) * segment_tokens
        + ((request * NUM_LAYERS + layer) * BUCKET + prompt_start).to(tl.int64)
    )
    lane = tl.arange(0, BLOCK)
    score_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < decode_length
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        score_sum += tl.sum(value, axis=0)
    mean = score_sum / decode_length
    square_sum = 0.0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token = start + lane
        valid = token < decode_length
        value = tl.load(score_row + token, mask=valid, other=0.0).to(tl.float32)
        centered = tl.where(valid, value - mean, 0.0)
        square_sum += tl.sum(centered * centered, axis=0)
    std = tl.sqrt(square_sum / decode_length)
    tl.store(row_mean + flat_row, mean)
    tl.store(row_inv_std + flat_row, 1.0 / tl.maximum(std, EPSILON))


@triton.jit
def _score_per_head_reduce_kernel(
    score_scratch,
    decode_lengths,
    prompt_lengths,
    row_mean,
    row_inv_std,
    selection_scores,
    selection_row_lengths,
    segment_tokens,
    NUM_LAYERS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PADDED_COLUMNS: tl.constexpr,
    BUCKET: tl.constexpr,
    WIDTH: tl.constexpr,
    PER_LAYER: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    """Reduce each KV-head decode window from score scratch into a selector row."""
    QUERY_GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    SELECTION_ROWS: tl.constexpr = NUM_LAYERS * NUM_KV_HEADS if PER_LAYER else NUM_KV_HEADS
    request = tl.program_id(0)
    selection_row = tl.program_id(1)
    token_block = tl.program_id(2)
    token = token_block * BLOCK + tl.arange(0, BLOCK)
    decode_length = tl.load(decode_lengths + request)
    prompt_start = tl.load(prompt_lengths + request)
    valid_token = token < decode_length

    if token_block == 0:
        tl.store(
            selection_row_lengths + request * SELECTION_ROWS + selection_row,
            decode_length,
        )

    kv_head = selection_row % NUM_KV_HEADS
    if PER_LAYER:
        layer = selection_row // NUM_KV_HEADS
        reduced = tl.full((BLOCK,), -float("inf"), tl.float32)
        for query_in_group in tl.static_range(0, QUERY_GROUP_SIZE):
            query_head = kv_head * QUERY_GROUP_SIZE + query_in_group
            flat_row = (request * NUM_LAYERS + layer) * NUM_Q_HEADS + query_head
            plane = kv_head * PADDED_COLUMNS + query_in_group
            value = tl.load(
                score_scratch
                + plane.to(tl.int64) * segment_tokens
                + ((request * NUM_LAYERS + layer) * BUCKET + prompt_start).to(tl.int64)
                + token,
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
                flat_row = (request * NUM_LAYERS + layer) * NUM_Q_HEADS + query_head
                plane = kv_head * PADDED_COLUMNS + query_in_group
                value = tl.load(
                    score_scratch
                    + plane.to(tl.int64) * segment_tokens
                    + ((request * NUM_LAYERS + layer) * BUCKET + prompt_start).to(tl.int64)
                    + token,
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


def reduce_per_head_scores(
    score_scratch: torch.Tensor,
    decode_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    row_mean: torch.Tensor,
    row_inv_std: torch.Tensor,
    selection_scores_rows: torch.Tensor,
    selection_row_lengths: torch.Tensor,
    *,
    request_count: int,
    padded_head_columns: int,
    score_token_capacity: int,
    per_layer: bool,
    normalize_scores: bool,
) -> None:
    """Reduce score-scratch decode windows into per-head selection rows."""
    request_capacity = int(decode_lengths.numel())
    num_layers = int(row_mean.shape[1])
    num_q_heads = int(row_mean.shape[2])
    selection_rows = int(selection_scores_rows.shape[0]) // request_capacity
    num_kv_heads = selection_rows // num_layers if per_layer else selection_rows
    selection_width = int(selection_scores_rows.shape[1])
    rows = num_layers * num_q_heads
    segment_tokens = request_count * num_layers * score_token_capacity
    if normalize_scores:
        _score_row_stats_kernel[(request_count * rows,)](
            score_scratch,
            decode_lengths,
            prompt_lengths,
            row_mean,
            row_inv_std,
            segment_tokens,
            ROWS=rows,
            NUM_LAYERS=num_layers,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            PADDED_COLUMNS=padded_head_columns,
            BUCKET=score_token_capacity,
            WIDTH=selection_width,
        )
    # 256-token tiles match the reduce kernel's BLOCK default.
    _score_per_head_reduce_kernel[
        (request_count, selection_rows, triton.cdiv(selection_width, 256))
    ](
        score_scratch,
        decode_lengths,
        prompt_lengths,
        row_mean,
        row_inv_std,
        selection_scores_rows,
        selection_row_lengths,
        segment_tokens,
        NUM_LAYERS=num_layers,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        PADDED_COLUMNS=padded_head_columns,
        BUCKET=score_token_capacity,
        WIDTH=selection_width,
        PER_LAYER=per_layer,
        NORMALIZE=normalize_scores,
    )


@triton.jit
def _fold_union_ranks_kernel(
    gathered_rows,
    selection_scores_rows,
    request_count,
    TP_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
):
    """Max-fold TP-gathered rank-local rows into each global union row."""
    request = tl.program_id(0)
    token_block = tl.program_id(1)
    token = token_block * BLOCK + tl.arange(0, BLOCK)
    mask = token < WIDTH
    folded = tl.full((BLOCK,), -float("inf"), tl.float32)
    for rank in tl.static_range(0, TP_SIZE):
        value = tl.load(
            gathered_rows + (rank * request_count + request) * WIDTH + token,
            mask=mask,
            other=-float("inf"),
        )
        folded = tl.maximum(folded, value)
    tl.store(selection_scores_rows + request * WIDTH + token, folded, mask=mask)


def fold_union_ranks(
    gathered_rows: torch.Tensor,
    selection_scores_rows: torch.Tensor,
    *,
    request_count: int,
) -> None:
    """Max-fold TP rank-local score rows into the global union rows."""
    width = int(selection_scores_rows.shape[1])
    tp_size = int(gathered_rows.shape[0]) // request_count
    block = 1024
    _fold_union_ranks_kernel[(request_count, triton.cdiv(width, block))](
        gathered_rows,
        selection_scores_rows,
        request_count,
        TP_SIZE=tp_size,
        WIDTH=width,
        BLOCK=block,
    )


@triton.jit
def _settle_ties_kernel(
    selection_scores_rows,
    selection_row_lengths,
    prompt_lengths,
    provisional_rows,
    kept_ordinal_rows,
    WIDTH: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    BLOCK: tl.constexpr = 256,
):
    """Settle score ties by lowest index and sort the kept-token indices."""
    request = tl.program_id(0)
    selection_domain = tl.program_id(1)
    row = request * SELECTION_ROWS + selection_domain
    row_output = kept_ordinal_rows + row * KEEP_COUNT
    row_scores = selection_scores_rows + row * WIDTH
    row_selected = provisional_rows + row * KEEP_COUNT
    # Rebases the decode-relative ordinals to absolute positions (per request:
    # every selection row of a request shares its pinned prompt length).
    prompt_length = tl.load(prompt_lengths + request)

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

    row_length = tl.load(selection_row_lengths + row)
    greater_count = 0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        valid = (token_index < WIDTH) & (token_index < row_length)
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
        valid = (token_index < WIDTH) & (token_index < row_length)
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
            token_index + prompt_length,
            mask=selected,
        )
        output_count += tl.sum(selected_i32)
        ties_seen += tl.sum(tied_i32)


def settle_ties(
    selection_scores_rows: torch.Tensor,
    selection_row_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    provisional_rows: torch.Tensor,
    kept_ordinal_rows: torch.Tensor,
    *,
    request_count: int,
    selection_rows_per_request: int,
) -> None:
    """Settle TopK score ties into ascending absolute token ordinals."""
    _settle_ties_kernel[(request_count, selection_rows_per_request)](
        selection_scores_rows,
        selection_row_lengths,
        prompt_lengths,
        provisional_rows,
        kept_ordinal_rows,
        WIDTH=int(selection_scores_rows.shape[1]),
        KEEP_COUNT=int(kept_ordinal_rows.shape[1]),
        SELECTION_ROWS=selection_rows_per_request,
    )
