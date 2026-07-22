# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU kernels for the TriAttention KV-eviction pipeline.

Scoring runs EXCLUSIVELY through the SM100 CuTe-DSL fused score pack
(``triattention_cute_score_fused.py``): mean aggregation, BF16 KV pools,
head size 64/128, 32/128-token pages, GQA group 4 or 8, per-request score
window starts. Any geometry outside that contract raises loudly at
workspace construction. The per-head modes use the pack's score-only entry
plus the row-stats kernel; union eviction runs the fused
score+stats+union pipeline. One-time buffer staging and runner compilation
live in ``triattention.prepare_eviction_workspace``; this module keeps the
Triton kernels, their launch helpers, and the mean-phase table builders.

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every flat buffer offset that can exceed 2^31.
  * mask ragged valid-width tails (and frequency tails) in every load and store.
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

from typing import Dict

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Mean-phase table: RoPE-style position table of mean trig phases.            #
# --------------------------------------------------------------------------- #


# Positions past this row count are no longer exactly representable in fp32,
# so a larger table would silently degrade every downstream phase.
_MEAN_PHASE_MAX_ROWS = 1 << 24


@triton.jit
def _gather_mean_phase_kernel(
    round_starts,
    table_cos,
    table_sin,
    mean_cos,
    mean_sin,
    table_rows,
    NUM_FREQS: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    """Copy each request's precomputed phase-table row into the fixed buffers."""
    request = tl.program_id(0)
    frequency = tl.arange(0, F_BLOCK)
    frequency_mask = frequency < NUM_FREQS
    table_row = tl.load(round_starts + request).to(tl.int64)
    # Clamp stale or padded round starts into the table instead of faulting;
    # staged cohorts are host-validated, so live rows are never clamped.
    table_row = tl.minimum(tl.maximum(table_row, 0), table_rows - 1)
    source_offset = table_row * NUM_FREQS + frequency
    output_offset = request * NUM_FREQS + frequency
    row_cos = tl.load(table_cos + source_offset, mask=frequency_mask, other=0.0)
    row_sin = tl.load(table_sin + source_offset, mask=frequency_mask, other=0.0)
    tl.store(mean_cos + output_offset, row_cos, mask=frequency_mask)
    tl.store(mean_sin + output_offset, row_sin, mask=frequency_mask)


def build_mean_phase_table(
    offsets: torch.Tensor, omega: torch.Tensor, initial_rows: int
) -> Dict[str, object]:
    """Build the plain-dict mean-phase table shared by every workspace.

    Row ``p`` holds ``mean_o(trig((p + offset_o) * omega_f))`` over the
    calibration offsets for every frequency, so refreshing a round's
    ``mean_cos``/``mean_sin`` is one pure-gather launch over the staged round
    starts. The dict is shared BY REFERENCE between the manager and its
    workspace, so ``grow_mean_phase_table`` reaches both. Grow the table while
    the round starts are still host integers; the gather kernel clamps stale
    rows instead of faulting.
    """
    phase: Dict[str, object] = {
        "offsets": offsets.contiguous(),
        "omega": omega.contiguous(),
        "offset_values": offsets.tolist(),
        "cos": None,
        "sin": None,
        "rows": 0,
    }
    grow_mean_phase_table(phase, max(int(initial_rows), 1))
    return phase


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
    # Accumulate offset-by-offset in fp32 (fixed summation order keeps the
    # table bit-stable across rebuilds).
    for offset in phase["offset_values"]:
        angle = torch.outer(positions + offset, omega)
        cos_table += torch.cos(angle)
        sin_table += torch.sin(angle)
    scale = 1.0 / len(phase["offset_values"])
    phase["cos"] = cos_table.mul_(scale)
    phase["sin"] = sin_table.mul_(scale)
    phase["rows"] = target


def gather_mean_phases(
    phase: Dict[str, object],
    round_starts: torch.Tensor,
    mean_cos: torch.Tensor,
    mean_sin: torch.Tensor,
    request_count: int,
) -> None:
    """Refresh the fixed mean buffers in place from staged round starts.

    Writes in place because the compiled CuTe score launch captured the
    destination buffers' device pointers. CUDA-only; eviction never runs
    under CUDA graph capture.
    """
    num_freqs = phase["omega"].numel()
    _gather_mean_phase_kernel[(request_count,)](
        round_starts,
        phase["cos"],
        phase["sin"],
        mean_cos,
        mean_sin,
        phase["rows"],
        NUM_FREQS=num_freqs,
        F_BLOCK=triton.next_power_of_2(num_freqs),
        num_warps=1,
    )


# --------------------------------------------------------------------------- #
# Selection: combine scores per mode, then finalize the top-k set.            #
# --------------------------------------------------------------------------- #


@triton.jit
def _score_row_stats_kernel(
    scores,
    valid_widths,
    row_mean,
    row_inv_std,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
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
    tl.store(row_inv_std + flat_row, 1.0 / tl.maximum(std, 1e-6))


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
    """Reduce query-head score rows into one selector row per KV-head domain.

    per_layer: row (layer, kv_head) = max over the KV head's query group.
    Otherwise: row kv_head = mean over layers of that per-layer group max.
    Optionally z-normalizes each query-head row with the precomputed
    mean/inv-std before reducing.
    """
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


# --------------------------------------------------------------------------- #
# Compaction: pack the kept ordinals into per-request move indices.           #
# --------------------------------------------------------------------------- #


@triton.jit
def _settle_ties_and_pack_compaction_sources_kernel(
    scores,
    seq_lens,
    prompt_offsets,
    provisional_indices,
    output_indices,
    valid_seq_lens,
    dense_offsets,
    dense_indices,
    swa_offsets,
    swa_indices,
    WIDTH: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    DENSE_TOTAL: tl.constexpr,
    SWA_TOTAL: tl.constexpr,
    MOVE_CAPACITY: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    UNION: tl.constexpr,
    PER_LAYER: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_SETTLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Settle one selection row's ties, then pack its compaction move sources.

    One program per (request, selection row). The settle half recovers the
    threshold from the provisional top-k, counts the strictly greater
    scores, then emits the kept ordinals in increasing order
    (lowest-index-wins ties), rebased by the row's pinned prompt length.
    The pack half then writes the move sources for the packed rows this
    selection row feeds: the kept ordinals it just wrote, the request's
    protected tail, plus the SWA rows (latest window). Union selection has
    one row per request feeding every KV head's packed row, so that single
    program writes all of them. ``HAS_SETTLE=False`` compiles the settle
    half away, packing pre-settled ordinals read from ``output_indices`` --
    the draft co-compaction flow, whose keep set is the target's and needs
    no settling.
    """
    request = tl.program_id(0)
    selection_domain = tl.program_id(1)
    row = request * SELECTION_ROWS + selection_domain
    row_scores = scores + row * WIDTH
    row_selected = provisional_indices + row * KEEP_COUNT
    row_output = output_indices + row * KEEP_COUNT
    if HAS_SETTLE:
        # Scores are decode-relative; this row's pinned prompt length rebases
        # the emitted ordinals to absolute positions (per row, so one launch
        # may mix prompt lengths).
        prompt_len = tl.load(prompt_offsets + row)

        threshold = float("inf")
        for start in tl.static_range(0, KEEP_COUNT, BLOCK):
            selected_offset = start + tl.arange(0, BLOCK)
            selected_mask = selected_offset < KEEP_COUNT
            token_index = tl.load(
                row_selected + selected_offset,
                mask=selected_mask,
                other=0,
            )
            # Rows shorter than KEEP_COUNT arrive padded with -1 sentinels
            # from the top-k's short-row path (zero-width padded rows are all
            # sentinels). Mask those lanes out of the gather so no lane
            # dereferences ``row_scores - 1`` and a sentinel never joins the
            # threshold; rows without sentinels load exactly as before.
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

    if HAS_SETTLE:
        # The emission above scatters through other lanes of this program;
        # make those global stores visible to every lane before the pack
        # half reads the row back.
        tl.debug_barrier()
    dense_begin = tl.load(dense_offsets + request)
    dense_end = tl.load(dense_offsets + request + 1)
    dense_count = dense_end - dense_begin
    valid_len = tl.load(valid_seq_lens + request)
    if HAS_SWA:
        swa_begin = tl.load(swa_offsets + request)
        swa_end = tl.load(swa_offsets + request + 1)
        swa_count = swa_end - swa_begin
    for move_start in tl.static_range(0, MOVE_CAPACITY, BLOCK):
        move = move_start + tl.arange(0, BLOCK)
        selected = tl.load(
            row_output + move,
            mask=move < KEEP_COUNT,
            other=0,
        )
        dense_source = tl.where(move < KEEP_COUNT, selected, valid_len + move - KEEP_COUNT)
        if UNION:
            # The one union row per request feeds every KV head's packed
            # row with the same move sources.
            for head in tl.static_range(0, NUM_KV_HEADS):
                tl.store(
                    dense_indices + head * DENSE_TOTAL + dense_begin.to(tl.int64) + move,
                    dense_source,
                    mask=move < dense_count,
                )
        else:
            domain = tl.program_id(1)
            dense_output = domain.to(tl.int64) * DENSE_TOTAL + dense_begin.to(tl.int64) + move
            tl.store(dense_indices + dense_output, dense_source, mask=move < dense_count)
        if HAS_SWA:
            swa_source = valid_len - SWA_WINDOW + move
            if UNION:
                for head in tl.static_range(0, NUM_KV_HEADS):
                    tl.store(
                        swa_indices + head * SWA_TOTAL + swa_begin.to(tl.int64) + move,
                        swa_source,
                        mask=move < swa_count,
                    )
            else:
                domain = tl.program_id(1)
                # Per-layer selection has one dense domain per (layer,
                # head). SWA uses one shared source row per head, so only
                # the first layer writes it.
                if PER_LAYER:
                    write_swa = domain < NUM_KV_HEADS
                else:
                    write_swa = move >= 0
                head = domain % NUM_KV_HEADS
                swa_output = head.to(tl.int64) * SWA_TOTAL + swa_begin.to(tl.int64) + move
                tl.store(
                    swa_indices + swa_output,
                    swa_source,
                    mask=write_swa & (move < swa_count),
                )
