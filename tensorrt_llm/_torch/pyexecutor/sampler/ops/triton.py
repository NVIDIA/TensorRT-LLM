# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal Triton kernels for finish reasons and selected sampler logprobs."""

import functools

import torch
import triton
import triton.language as tl

__all__ = [
    "MAX_FUSED_ELEMENTS_PER_REQUEST",
    "fused_write_finish_reasons",
    "selected_raw_logprobs",
    "store_selected_logprobs",
]

# One Triton program owns all token/beam elements for one request.
MAX_FUSED_ELEMENTS_PER_REQUEST = 1024


@triton.jit
def _finish_reasons_kernel(
    finish_reasons_ptr,
    new_tokens_ptr,
    seq_slots_ptr,
    seq_lens_ptr,
    max_lengths_ptr,
    end_ids_ptr,
    stride_reasons_token,
    stride_reasons_slot,
    stride_reasons_beam,
    stride_tokens_token,
    stride_tokens_slot,
    stride_tokens_beam,
    beam_width: tl.constexpr,
    num_elements: tl.constexpr,
    not_finished: tl.constexpr,
    length: tl.constexpr,
    end_id: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    slot = tl.load(seq_slots_ptr + request).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + request).to(tl.int32)
    max_length = tl.load(max_lengths_ptr + slot).to(tl.int32)
    request_end_id = tl.load(end_ids_ptr + slot)

    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_elements
    # Flatten in token-major, beam-minor order, matching the backing tensors.
    token_index = offsets // beam_width
    beam_index = offsets % beam_width
    tokens = tl.load(
        new_tokens_ptr
        + token_index * stride_tokens_token
        + slot * stride_tokens_slot
        + beam_index * stride_tokens_beam,
        mask=mask,
        other=0,
    )

    at_max_length = (seq_len + token_index + 1) >= max_length
    is_end_id = tokens == request_end_id
    # End ID has the same final-write precedence as the tensor implementation.
    reasons = tl.where(is_end_id, end_id, tl.where(at_max_length, length, not_finished))
    tl.store(
        finish_reasons_ptr
        + token_index * stride_reasons_token
        + slot * stride_reasons_slot
        + beam_index * stride_reasons_beam,
        reasons.to(finish_reasons_ptr.dtype.element_ty),
        mask=mask,
    )


@functools.lru_cache(maxsize=None)
def _launch_shape(max_tokens: int, beam_width: int) -> tuple[int, int, int]:
    """Elements per request, Triton block, and warp count for that block."""
    num_elements = max_tokens * beam_width
    block = max(32, triton.next_power_of_2(num_elements))
    return num_elements, block, min(8, block // 32)


def fused_write_finish_reasons(
    *,
    finish_reasons: torch.Tensor,
    new_tokens: torch.Tensor,
    seq_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    max_lengths: torch.Tensor,
    end_ids: torch.Tensor,
    max_tokens: int,
    beam_width: int,
    not_finished_value: int,
    length_value: int,
    end_id_value: int,
) -> None:
    """Write end-ID and maximum-length reasons without intermediate tensors."""
    num_requests = seq_slots.shape[0]
    num_elements, block, num_warps = _launch_shape(max_tokens, beam_width)
    _finish_reasons_kernel[(num_requests,)](
        finish_reasons,
        new_tokens,
        seq_slots,
        seq_lens,
        max_lengths,
        end_ids,
        finish_reasons.stride(0),
        finish_reasons.stride(1),
        finish_reasons.stride(2),
        new_tokens.stride(0),
        new_tokens.stride(1),
        new_tokens.stride(2),
        beam_width=beam_width,
        num_elements=num_elements,
        not_finished=not_finished_value,
        length=length_value,
        end_id=end_id_value,
        BLOCK=block,
        num_warps=num_warps,
    )


@triton.jit
def _selected_raw_logprobs_kernel(
    logits,
    rows,
    sampled_tokens,
    sampled_values,
    sampled_ranks,
    top_values,
    top_tokens,
    row_stride,
    VOCAB: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(rows + row).to(tl.int64)
    token = tl.load(sampled_tokens + row)
    columns = tl.arange(0, BLOCK)
    values = tl.load(logits + source * row_stride + columns, columns < VOCAB, other=-float("inf"))
    maximum = tl.max(values, 0)
    shifted = values - maximum
    log_sum = tl.log(tl.sum(tl.exp(shifted), 0))
    logprobs = shifted - log_sum
    sampled = (tl.load(logits + source * row_stride + token) - maximum) - log_sum
    rank = tl.sum(((columns < VOCAB) & (logprobs > sampled)).to(tl.int32), 0)
    tl.store(sampled_values + row, sampled)
    tl.store(sampled_ranks + row, rank)
    available = columns < VOCAB
    for k in range(TOP_K):
        value = tl.max(tl.where(available, logprobs, -float("inf")), 0)
        index = tl.min(
            tl.where(available & ((logprobs == value) | (value != value)), columns, 2147483647), 0
        )
        tl.store(top_values + row * TOP_K + k, value)
        tl.store(top_tokens + row * TOP_K + k, index)
        available = available & (columns != index)


@triton.jit
def _raw_logprobs_partials_kernel(
    logits,
    rows,
    partial_max,
    partial_sum,
    candidates,
    candidate_ids,
    row_stride,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    source = tl.load(rows + row).to(tl.int64)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + source * row_stride + columns, columns < VOCAB, other=-float("inf"))
    maximum = tl.max(values, 0)
    total = tl.sum(tl.exp(values - maximum), 0)
    tl.store(partial_max + row * TILES + tile, maximum)
    tl.store(partial_sum + row * TILES + tile, total)
    if TOP_K > 8:
        # Monotonic float bits form the primary key; inverse token IDs break ties.
        bits = values.to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - columns.to(tl.uint64))
        SORT_K: tl.constexpr = BLOCK if BLOCK == 1024 else triton.next_power_of_2(TOP_K)
        if BLOCK == 1024:
            keys = tl.sort(keys, descending=True)
        else:
            keys = tl.topk(keys, SORT_K)
        ordered = (keys >> 32).to(tl.uint32)
        bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
        sorted_values = bits.to(tl.float32, bitcast=True)
        sorted_ids = 0xFFFFFFFF - (keys & 0xFFFFFFFF)
        offsets = tl.arange(0, SORT_K)
        destination = (row * TILES + tile) * TOP_K + offsets
        tl.store(candidates + destination, sorted_values, offsets < TOP_K)
        tl.store(candidate_ids + destination, sorted_ids, offsets < TOP_K)
    else:
        available = columns < VOCAB
        for k in range(TOP_K):
            value = tl.max(tl.where(available, values, -float("inf")), 0)
            index = tl.min(
                tl.where(available & ((values == value) | (value != value)), columns, 2147483647), 0
            )
            offset = (row * TILES + tile) * TOP_K + k
            tl.store(candidates + offset, value)
            tl.store(candidate_ids + offset, index)
            available = available & (columns != index)


@triton.jit
def _raw_logprobs_normalize_kernel(
    logits,
    rows,
    tokens,
    partial_max,
    partial_sum,
    maxima,
    log_sums,
    sampled,
    candidates,
    row_stride,
    TILES: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0)
    tiles = tl.arange(0, BLOCK)
    local_max = tl.load(partial_max + row * TILES + tiles, tiles < TILES, other=-float("inf"))
    local_sum = tl.load(partial_sum + row * TILES + tiles, tiles < TILES, other=0.0)
    maximum = tl.max(local_max, 0)
    total = tl.sum(
        tl.where(local_max == -float("inf"), 0.0, local_sum * tl.exp(local_max - maximum)), 0
    )
    log_sum = tl.log(total)
    tl.store(maxima + row, maximum)
    tl.store(log_sums + row, log_sum)
    source = tl.load(rows + row).to(tl.int64)
    token = tl.load(tokens + row)
    value = tl.load(logits + source * row_stride + token)
    tl.store(sampled + row, (value - maximum) - log_sum)
    if TOP_K:
        offsets = tl.arange(0, CANDIDATES)
        values = tl.load(
            candidates + row * TILES * TOP_K + offsets, offsets < TILES * TOP_K, other=-float("inf")
        )
        tl.store(
            candidates + row * TILES * TOP_K + offsets,
            (values - maximum) - log_sum,
            offsets < TILES * TOP_K,
        )


@triton.jit
def _raw_logprobs_rank_kernel(
    logits,
    rows,
    maxima,
    log_sums,
    sampled,
    ranks,
    row_stride,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    source = tl.load(rows + row).to(tl.int64)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + source * row_stride + columns, columns < VOCAB, other=-float("inf"))
    # Compare normalized float32 values so rounding ties have the same rank semantics.
    normalized = (values - tl.load(maxima + row)) - tl.load(log_sums + row)
    rank = tl.sum(((columns < VOCAB) & (normalized > tl.load(sampled + row))).to(tl.int32), 0)
    tl.store(ranks + row * TILES + tile, rank)


@triton.jit
def _raw_logprobs_finish_kernel(
    candidates,
    candidate_ids,
    partial_ranks,
    ranks,
    top_values,
    top_tokens,
    TILES: tl.constexpr,
    TOP_K: tl.constexpr,
    VALID: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tiles = tl.arange(0, TILE_BLOCK)
    rank = tl.sum(tl.load(partial_ranks + row * TILES + tiles, tiles < TILES, other=0), 0)
    tl.store(ranks + row, rank)
    columns = tl.arange(0, BLOCK)
    values = tl.load(
        candidates + row * TILES * TOP_K + columns, columns < VALID, other=-float("inf")
    )
    available = columns < VALID
    for k in range(TOP_K):
        value = tl.max(tl.where(available, values, -float("inf")), 0)
        index = tl.min(
            tl.where(available & ((values == value) | (value != value)), columns, 2147483647), 0
        )
        token = tl.load(candidate_ids + row * TILES * TOP_K + index)
        tl.store(top_values + row * TOP_K + k, value)
        tl.store(top_tokens + row * TOP_K + k, token)
        available = available & (columns != index)


@triton.jit
def _raw_logprobs_direct_topk_kernel(
    top_values,
    top_tokens,
    rows,
    maxima,
    log_sums,
    partial_ranks,
    ranks,
    out_values,
    out_tokens,
    TOP_K: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(rows + row).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    values = tl.load(top_values + source * TOP_K + columns, columns < TOP_K, other=0.0)
    tokens = tl.load(top_tokens + source * TOP_K + columns, columns < TOP_K, other=0)
    values = (values - tl.load(maxima + row)) - tl.load(log_sums + row)
    tl.store(out_values + row * TOP_K + columns, values, columns < TOP_K)
    tl.store(out_tokens + row * TOP_K + columns, tokens, columns < TOP_K)
    tiles = tl.arange(0, TILE_BLOCK)
    rank = tl.sum(tl.load(partial_ranks + row * TILES + tiles, tiles < TILES, other=0), 0)
    tl.store(ranks + row, rank)


def _selected_raw_logprobs_tiled(
    logits: torch.Tensor,
    rows: torch.Tensor,
    sampled_tokens: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = rows.numel()
    vocab = logits.shape[1]
    # Retain only tile statistics and candidates, not a [rows, vocab] tensor.
    # Dense row coverage can use topk on the existing logits without a gather copy.
    # Small source batches benefit from filling the GPU; otherwise cap extra scans at 2x.
    direct_topk = top_k > 8 and (logits.shape[0] <= 32 or count * 2 >= logits.shape[0])
    candidate_k = 0 if direct_topk else top_k
    # Wider tiles limit candidate staging for small row sets.
    block = (1024 if count <= 16 else 512) if candidate_k > 8 else 4096
    tiles = triton.cdiv(vocab, block)
    options = dict(device=logits.device, dtype=torch.float32)
    partial_max = torch.empty((count, tiles), **options)
    partial_sum = torch.empty_like(partial_max)
    candidates = torch.empty((count, tiles * candidate_k), **options)
    candidate_ids = torch.empty_like(candidates, dtype=torch.int32)
    maxima = torch.empty((count,), **options)
    log_sums = torch.empty_like(maxima)
    sampled = torch.empty_like(maxima)
    ranks = torch.empty_like(partial_max, dtype=torch.int32)
    _raw_logprobs_partials_kernel[(count, tiles)](
        logits,
        rows,
        partial_max,
        partial_sum,
        candidates,
        candidate_ids,
        logits.stride(0),
        vocab,
        tiles,
        candidate_k,
        block,
    )
    _raw_logprobs_normalize_kernel[(count,)](
        logits,
        rows,
        sampled_tokens,
        partial_max,
        partial_sum,
        maxima,
        log_sums,
        sampled,
        candidates,
        logits.stride(0),
        tiles,
        candidate_k,
        triton.next_power_of_2(tiles),
        triton.next_power_of_2(max(1, tiles * candidate_k)),
    )
    _raw_logprobs_rank_kernel[(count, tiles)](
        logits,
        rows,
        maxima,
        log_sums,
        sampled,
        ranks,
        logits.stride(0),
        vocab,
        tiles,
        block,
    )
    if direct_topk:
        all_values, all_tokens = torch.topk(logits, top_k, dim=-1)
        sampled_ranks = torch.empty_like(sampled, dtype=torch.int32)
        top_values = torch.empty((count, top_k), **options)
        top_tokens = torch.empty_like(top_values, dtype=torch.int32)
        _raw_logprobs_direct_topk_kernel[(count,)](
            all_values,
            all_tokens,
            rows,
            maxima,
            log_sums,
            ranks,
            sampled_ranks,
            top_values,
            top_tokens,
            top_k,
            tiles,
            triton.next_power_of_2(top_k),
            triton.next_power_of_2(tiles),
        )
        return sampled, sampled_ranks, top_values, top_tokens
    valid_candidates = (tiles - 1) * top_k + min(top_k, vocab - (tiles - 1) * block)
    if top_k <= 8:
        sampled_ranks = torch.empty_like(sampled, dtype=torch.int32)
        top_values = torch.empty((count, top_k), **options)
        top_tokens = torch.empty_like(top_values, dtype=torch.int32)
        _raw_logprobs_finish_kernel[(count,)](
            candidates,
            candidate_ids,
            ranks,
            sampled_ranks,
            top_values,
            top_tokens,
            tiles,
            top_k,
            valid_candidates,
            triton.next_power_of_2(tiles),
            triton.next_power_of_2(max(1, valid_candidates)),
        )
        return sampled, sampled_ranks, top_values, top_tokens
    candidates = candidates[:, :valid_candidates]
    candidate_ids = candidate_ids[:, :valid_candidates]
    top_values, indices = torch.topk(candidates, top_k, dim=-1)
    top_tokens = candidate_ids.gather(1, indices)
    return sampled, ranks.sum(-1, dtype=torch.int32), top_values, top_tokens


def selected_raw_logprobs(
    logits: torch.Tensor,
    rows: torch.Tensor,
    sampled_tokens: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute selected raw logprobs without a full-vocabulary output tensor.

    Args:
        logits: CUDA float32 scores of shape [input_rows, vocab], with unit last stride.
        rows: Contiguous CUDA int32 or int64 source indices of shape [rows].
        sampled_tokens: Contiguous CUDA int32 or int64 token IDs of shape [rows].
        top_k: Number of candidates to return, between zero and the vocabulary size.

    Returns:
        Sampled float32 logprobs and int32 zero-based ranks, both [rows], followed
        by float32 top logprobs and int32 token IDs, both [rows, top_k]. Ranks count
        strictly greater logprobs; ties among top candidates are unordered.
    """
    if not 0 <= top_k <= logits.shape[1]:
        raise ValueError("top_k must be between zero and the vocabulary size")
    count = rows.numel()
    if count and logits.shape[1] > 4096:
        return _selected_raw_logprobs_tiled(logits, rows, sampled_tokens, top_k)
    values = torch.empty((count,), dtype=torch.float32, device=logits.device)
    ranks = torch.empty((count,), dtype=torch.int32, device=logits.device)
    top_values = torch.empty((count, top_k), dtype=torch.float32, device=logits.device)
    top_tokens = torch.empty((count, top_k), dtype=torch.int32, device=logits.device)
    if count:
        block = triton.next_power_of_2(logits.shape[1])
        _selected_raw_logprobs_kernel[(count,)](
            logits=logits,
            rows=rows,
            sampled_tokens=sampled_tokens,
            sampled_values=values,
            sampled_ranks=ranks,
            top_values=top_values,
            top_tokens=top_tokens,
            row_stride=logits.stride(0),
            VOCAB=logits.shape[1],
            TOP_K=top_k,
            BLOCK=block,
            num_warps=min(32, max(4, block // 4096)),
        )
    return values, ranks, top_values, top_tokens


@triton.jit
def _store_selected_logprobs_kernel(
    processed_values,
    processed_ranks,
    processed_top_values,
    processed_top_tokens,
    raw_values,
    raw_ranks,
    raw_top_values,
    raw_top_tokens,
    sampled_tokens,
    destinations,
    out_values,
    out_ranks,
    out_tokens,
    out_top_values,
    out_top_tokens,
    value_stride,
    rank_stride,
    token_stride,
    top_value_stride,
    top_token_stride,
    processed_rows,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    destination = tl.load(destinations + row)
    columns = tl.arange(0, BLOCK)
    if row < processed_rows:
        value = tl.load(processed_values + row)
        rank = tl.load(processed_ranks + row).to(tl.int32)
        tops = tl.load(processed_top_values + row * TOP_K + columns, columns < TOP_K, other=0.0)
        ids = tl.load(processed_top_tokens + row * TOP_K + columns, columns < TOP_K, other=0).to(
            tl.int32
        )
    else:
        raw_row = row - processed_rows
        value = tl.load(raw_values + raw_row)
        rank = tl.load(raw_ranks + raw_row).to(tl.int32)
        tops = tl.load(raw_top_values + raw_row * TOP_K + columns, columns < TOP_K, other=0.0)
        ids = tl.load(raw_top_tokens + raw_row * TOP_K + columns, columns < TOP_K, other=0).to(
            tl.int32
        )
    tl.store(out_values + destination * value_stride, value)
    tl.store(out_ranks + destination * rank_stride, rank)
    tl.store(out_tokens + destination * token_stride, tl.load(sampled_tokens + row))
    tl.store(out_top_values + destination * top_value_stride + columns, tops, columns < TOP_K)
    tl.store(out_top_tokens + destination * top_token_stride + columns, ids, columns < TOP_K)


def store_selected_logprobs(
    processed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    processed_rows: int,
    sampled_tokens: torch.Tensor,
    destinations: torch.Tensor,
    out_values: torch.Tensor,
    out_ranks: torch.Tensor,
    out_tokens: torch.Tensor,
    out_top_values: torch.Tensor,
    out_top_tokens: torch.Tensor,
    top_k: int,
) -> None:
    """Scatter selected results in processed-then-raw order into slot-major storage.

    Each input tuple contains values/ranks [rows] and top values/IDs [rows, top_k].
    An unused partition may alias the other tuple. destinations and sampled_tokens
    are contiguous integer vectors [total_rows].
    Output vectors and matrices may have strided rows; their last dimensions are
    respectively flattened slot/step and contiguous candidate positions.
    """
    _store_selected_logprobs_kernel[(destinations.numel(),)](
        *processed,
        *raw,
        sampled_tokens,
        destinations,
        out_values,
        out_ranks,
        out_tokens,
        out_top_values,
        out_top_tokens,
        out_values.stride(0),
        out_ranks.stride(0),
        out_tokens.stride(0),
        out_top_values.stride(0),
        out_top_tokens.stride(0),
        processed_rows,
        top_k,
        triton.next_power_of_2(max(32, top_k)),
    )
