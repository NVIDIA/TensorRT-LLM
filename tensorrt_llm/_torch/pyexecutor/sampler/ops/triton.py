# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton sampling kernels.

Finish-reason and compact stop-word operations for the sampler.
"""

import functools

import torch
import triton
import triton.language as tl

__all__ = [
    "MAX_FUSED_ELEMENTS_PER_REQUEST",
    "fused_write_finish_reasons",
    "compact_stop_words",
    "reorder_compact_stop_word_history",
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
def _compact_stop_history_kernel(
    metadata,
    slots,
    tokens,
    accepted,
    predecessors,
    token_stride_step,
    token_stride_request,
    token_stride_beam,
    predecessor_stride,
    MAX_TOKENS: tl.constexpr,
    ACCEPTED_TENSOR: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    beam = tl.program_id(1)
    slot = tl.load(slots + request)
    row = metadata + slot * 5
    window = tl.load(row + 3)
    if window > 0:
        base = tl.load(row).to(tl.pointer_type(tl.int32))
        history = base + tl.load(row + 2)
        width = tl.load(row + 4)
        scratch = history + window * width
        if MODE == 0:
            if ACCEPTED_TENSOR:
                shift = tl.load(accepted + request)
            else:
                shift = accepted
        if MODE == 2:
            parent = tl.load(predecessors + request * predecessor_stride + beam)
        for start in range(tl.cdiv(window, BLOCK)):
            position = start * BLOCK + tl.arange(0, BLOCK)
            valid = position < window
            if MODE == 0:
                retained = window - MAX_TOKENS
                old = tl.load(
                    history + (position + shift) * width + beam,
                    valid & (position < retained),
                    other=0,
                )
                new = tl.load(
                    tokens
                    + (position - retained) * token_stride_step
                    + request * token_stride_request
                    + beam * token_stride_beam,
                    valid & (position >= retained),
                    other=0,
                )
                value = tl.where(position < retained, old, new)
            elif MODE == 1:
                value = tl.load(scratch + position * width + beam, valid, other=0)
            else:
                value = tl.load(history + position * width + parent, valid, other=0)
            if MODE == 1:
                tl.store(history + position * width + beam, value, valid)
            else:
                tl.store(scratch + position * width + beam, value, valid)


@triton.jit
def _compact_stop_match_kernel(
    metadata,
    slots,
    tokens,
    output,
    token_stride_step,
    token_stride_request,
    token_stride_beam,
    REQUESTS,
    MAX_TOKENS: tl.constexpr,
    BEAMS: tl.constexpr,
    BLOCK_RULES: tl.constexpr,
):
    request = tl.program_id(0)
    step_beam = tl.program_id(1)
    step = step_beam // BEAMS
    beam = step_beam % BEAMS
    slot = tl.load(slots + request)
    row = metadata + slot * 5
    base = tl.load(row).to(tl.pointer_type(tl.int32))
    rule_count = tl.load(row + 1)
    window = tl.load(row + 3)
    width = tl.load(row + 4)
    scratch = base + tl.load(row + 2) + window * width
    new_token = tl.load(
        tokens
        + step * token_stride_step
        + request * token_stride_request
        + beam * token_stride_beam
    )
    found = tl.full((), False, tl.int1)
    for block in range(tl.cdiv(rule_count, BLOCK_RULES)):
        rule = block * BLOCK_RULES + tl.arange(0, BLOCK_RULES)
        valid = rule < rule_count
        begin = tl.load(base + rule, valid, other=0)
        end = tl.load(base + rule + 1, valid, other=0)
        length = end - begin
        matches = valid & (length > 0)
        for offset in range(tl.max(length, 0)):
            active = valid & (offset < length)
            expected = tl.load(base + rule_count + 1 + begin + offset, active, other=0)
            if window > 0:
                position = window - MAX_TOKENS + step - length + 1 + offset
                actual = tl.load(scratch + position * width + beam, active, other=0)
            else:
                actual = tl.full((BLOCK_RULES,), 0, tl.int32) + new_token
            matches = matches & (~active | (expected == actual))
        found = found | (tl.sum(matches.to(tl.int32), 0) > 0)
    tl.store(output + (step * REQUESTS + request) * BEAMS + beam, found)


def compact_stop_words(
    metadata: torch.Tensor,
    slots: torch.Tensor,
    tokens: torch.Tensor,
    accepted: torch.Tensor | int,
    *,
    single_token_only: bool = False,
) -> torch.Tensor:
    """Match packed per-request rules without a rule-by-token comparison tensor.

    Metadata points to int32 allocations containing rule offsets, rule tokens,
    and two rolling-history windows. The caller retains these allocations and
    records their use on the current stream because they are indirect inputs.

    Args:
        metadata: Contiguous CUDA int64 tensor of shape [max_num_sequences, 5].
            Columns are allocation address, rule count, history offset in int32
            elements, history window length, and allocated beam width.
        slots: Contiguous CUDA int32 or int64 tensor of shape [requests], mapping
            batch requests to metadata rows.
        tokens: CUDA int32 or int64 tensor of shape [steps, requests, beams].
            All three strides are passed to the kernels; contiguity is not required.
        accepted: Contiguous CUDA int32 or int64 tensor of shape [requests], or
            a Python int shared by all requests. Counts accepted draft tokens plus
            one, used to advance the rolling history. Unused for single-token rules.
        single_token_only: Whether all rules contain exactly one token, allowing
            history updates to be skipped.

    Returns:
        CUDA bool tensor of shape [steps, requests, beams], true where a sequence
        ends with a stop word.
    """
    steps, requests, beams = tokens.shape
    result = torch.empty(tokens.shape, device=tokens.device, dtype=torch.bool)
    if requests == 0:
        return result
    if not single_token_only:
        _compact_stop_history_kernel[(requests, beams)](
            metadata,
            slots,
            tokens,
            accepted,
            slots,
            *tokens.stride(),
            0,
            steps,
            isinstance(accepted, torch.Tensor),
            0,
            128,
        )
    _compact_stop_match_kernel[(requests, steps * beams)](
        metadata,
        slots,
        tokens,
        result,
        *tokens.stride(),
        requests,
        steps,
        beams,
        32,
    )
    if not single_token_only:
        _compact_stop_history_kernel[(requests, beams)](
            metadata,
            slots,
            tokens,
            accepted,
            slots,
            *tokens.stride(),
            0,
            steps,
            isinstance(accepted, torch.Tensor),
            1,
            128,
        )
    return result


def reorder_compact_stop_word_history(
    metadata: torch.Tensor, slots: torch.Tensor, predecessors: torch.Tensor
) -> None:
    """Reorder packed histories through scratch storage, including duplicate parents.

    Args:
        metadata: Contiguous CUDA int64 tensor of shape [max_num_sequences, 5],
            with the column layout described in :func:`compact_stop_words`.
            The caller retains the pointed-to allocations and records their use
            on the current stream.
        slots: Contiguous CUDA int32 or int64 tensor of shape [requests], mapping
            batch requests to metadata rows.
        predecessors: CUDA int32 or int64 tensor of shape [requests, beams],
            containing each beam's parent index. The last dimension must have
            unit stride because only stride(0) is passed to the kernel.
    """
    requests, beams = predecessors.shape
    if requests == 0:
        return
    for mode in (2, 1):
        _compact_stop_history_kernel[(requests, beams)](
            metadata,
            slots,
            predecessors,
            1,
            predecessors,
            0,
            0,
            0,
            predecessors.stride(0),
            1,
            False,
            mode,
            128,
        )
