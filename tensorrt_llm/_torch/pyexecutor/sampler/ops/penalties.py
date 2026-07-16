# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repetition / presence / frequency penalty kernels (PyTorch-native).

Ops for the occurrence penalties (repetition / presence / frequency), the torch/Triton
counterpart of the C++ ``batchApplyPenalty`` kernel
(``cpp/tensorrt_llm/kernels/penaltyKernels.cu``), with no dependency on the
sampling_utils interface or other backend modules:

* :func:`update_occurrence_workspace` -- initializes prompt occurrence state.
* :func:`apply_batched_occurrence_penalties_triton` -- increments regular state and
  applies penalties in one pass over packed logits.
"""

import torch
import triton
import triton.language as tl


def update_occurrence_workspace(
    counts: torch.Tensor,
    presence_prefix: torch.Tensor | None,
    counted_slots: torch.Tensor,
    counted_tokens: torch.Tensor,
    prefix_slots: torch.Tensor,
    prefix_tokens: torch.Tensor,
) -> None:
    """Fold newly-committed tokens into the persistent occurrence workspace.

    Mirrors the logical accumulation ``batchApplyPenalty`` performs in its
    ``penaltyWorkspace`` while packing the prefix-presence half: tokens in the ignored
    prompt prefix ``[0, prompt_ignore_length)`` set the presence bitmap only (they count
    for repetition but not for presence/frequency), while all other tokens (the rest of
    the prompt plus every generated token) increment the occurrence counts.

    Arguments (all index tensors are 1-D and pre-split on the host):
      counts: ``int32[num_slots, vocab_size]`` occurrence counts, updated in place.
      presence_prefix: packed int32 ``[num_slots, ceil(vocab_size / 32)]`` prefix
        presence bitmap, or ``None`` when no active request uses
        ``prompt_ignore_length``.
      counted_slots / counted_tokens: (slot, token) pairs to increment in ``counts``.
      prefix_slots / prefix_tokens: (slot, token) pairs to mark in ``presence_prefix``.
    """
    if counted_slots.numel() > 0:
        ones = torch.ones(counted_slots.shape[0], dtype=counts.dtype, device=counts.device)
        # accumulate=True sums repeated (slot, token) pairs -> occurrence count.
        counts.index_put_((counted_slots, counted_tokens), ones, accumulate=True)
    if presence_prefix is not None and prefix_slots.numel() > 0:
        block_size = 256
        _mark_presence_prefix_kernel[(triton.cdiv(prefix_slots.numel(), block_size),)](
            presence_prefix,
            prefix_slots,
            prefix_tokens,
            presence_prefix.stride(0),
            prefix_slots.numel(),
            BLOCK_SIZE=block_size,
        )


@triton.jit
def _mark_presence_prefix_kernel(
    presence_prefix_ptr,
    prefix_slots_ptr,
    prefix_tokens_ptr,
    presence_prefix_row_stride,
    num_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomically mark ignored-prefix tokens in the packed presence bitmap."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    slots = tl.load(prefix_slots_ptr + offsets, mask=mask, other=0)
    tokens = tl.load(prefix_tokens_ptr + offsets, mask=mask, other=0).to(tl.int32)
    word_offsets = tokens // 32
    bit_offsets = tokens % 32
    bits = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_offsets
    tl.atomic_or(
        presence_prefix_ptr + slots * presence_prefix_row_stride + word_offsets,
        bits,
        mask=mask,
    )


@triton.jit
def _apply_batched_occurrence_penalties_kernel(
    logits_ptr,
    counts_ptr,
    presence_prefix_ptr,
    active_ptr,
    has_previous_token_ptr,
    new_tokens_ptr,
    seq_slots_ptr,
    request_offsets_ptr,
    request_num_steps_ptr,
    request_history_synced_ptr,
    repetition_ptr,
    presence_ptr,
    frequency_ptr,
    vocab,
    logits_row_stride,
    workspace_row_stride,
    presence_prefix_row_stride,
    new_tokens_step_stride,
    new_tokens_slot_stride,
    new_tokens_beam_stride,
    LOGIT_LIMIT: tl.constexpr,
    HAS_PRESENCE_PREFIX: tl.constexpr,
    HISTORY_SYNCED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Update the regular workspace and apply penalties in one pass."""
    request_idx = tl.program_id(0)
    step_idx = tl.program_id(1)
    vocab_block = tl.program_id(2)

    slot = tl.load(seq_slots_ptr + request_idx)
    active = tl.load(active_ptr + slot) != 0
    num_steps = tl.load(request_num_steps_ptr + request_idx)
    row = tl.load(request_offsets_ptr + request_idx) + step_idx
    valid_row = active & (step_idx < num_steps)

    offsets = vocab_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vocab_mask = offsets < vocab
    mask = vocab_mask & valid_row

    count_offset = slot * workspace_row_stride + offsets
    count = tl.load(counts_ptr + count_offset, mask=mask, other=0)
    history_synced = False
    if HISTORY_SYNCED:
        history_synced = tl.load(request_history_synced_ptr + request_idx) != 0
    has_previous_token = tl.load(has_previous_token_ptr + slot, mask=valid_row, other=0) != 0
    accumulate_previous = has_previous_token
    if HISTORY_SYNCED:
        accumulate_previous &= ~history_synced
    previous_token = tl.load(
        new_tokens_ptr
        + slot * new_tokens_slot_stride
        + 0 * new_tokens_step_stride
        + 0 * new_tokens_beam_stride,
        mask=valid_row & accumulate_previous,
        other=-1,
    )
    count += tl.where(accumulate_previous & (offsets == previous_token), 1, 0)

    logit_offset = row * logits_row_stride + offsets
    logit = tl.load(logits_ptr + logit_offset, mask=mask, other=0.0).to(tl.float32)
    seen = count > 0
    if HAS_PRESENCE_PREFIX:
        prefix_word_offsets = vocab_block * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        prefix_words = tl.load(
            presence_prefix_ptr + slot * presence_prefix_row_stride + prefix_word_offsets,
            mask=valid_row & (prefix_word_offsets < tl.cdiv(vocab, 32)),
            other=0,
        )
        prefix_seen = (prefix_words[:, None] >> tl.arange(0, 32)[None, :]) & 1
        seen |= prefix_seen.to(tl.int1).reshape(BLOCK_SIZE)

    repetition = tl.load(repetition_ptr + slot, mask=valid_row, other=1.0)
    presence = tl.load(presence_ptr + slot, mask=valid_row, other=0.0)
    frequency = tl.load(frequency_ptr + slot, mask=valid_row, other=0.0)

    repeated = tl.where(logit < 0.0, logit * repetition, logit / repetition)
    logit = tl.where(seen, repeated, logit)
    logit -= tl.where(
        count > 0,
        presence + frequency * count.to(tl.float32),
        0.0,
    )
    logit = tl.maximum(-LOGIT_LIMIT, tl.minimum(logit, LOGIT_LIMIT))

    tl.store(logits_ptr + logit_offset, logit.to(logits_ptr.dtype.element_ty), mask=mask)
    # Regular non-speculative requests have one logits row. Store the incremented
    # count from that row so the next sampling step observes it.
    if HISTORY_SYNCED:
        tl.store(
            counts_ptr + count_offset,
            count,
            mask=mask & (step_idx == 0) & ~history_synced,
        )
    else:
        tl.store(counts_ptr + count_offset, count, mask=mask & (step_idx == 0))
    tl.store(
        has_previous_token_ptr + slot,
        1,
        mask=valid_row & (step_idx == 0) & (vocab_block == 0),
    )


def apply_batched_occurrence_penalties_triton(
    logits: torch.Tensor,
    counts: torch.Tensor,
    presence_prefix: torch.Tensor | None,
    active: torch.Tensor,
    has_previous_token: torch.Tensor,
    new_tokens: torch.Tensor,
    seq_slots: torch.Tensor,
    request_offsets: torch.Tensor,
    request_num_steps: torch.Tensor,
    request_history_synced: torch.Tensor | None,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
    *,
    max_num_steps: int,
    history_synced: bool = False,
) -> torch.Tensor:
    """Apply regular occurrence penalties before downstream temperature handling."""
    num_requests = seq_slots.numel()
    if num_requests == 0:
        return logits

    vocab = logits.size(-1)
    block_size = 1024
    grid = (num_requests, max_num_steps, triton.cdiv(vocab, block_size))
    has_presence_prefix = presence_prefix is not None
    if history_synced:
        assert request_history_synced is not None
    _apply_batched_occurrence_penalties_kernel[grid](
        logits,
        counts,
        presence_prefix if has_presence_prefix else counts,
        active,
        has_previous_token,
        new_tokens,
        seq_slots,
        request_offsets,
        request_num_steps,
        request_history_synced if request_history_synced is not None else request_num_steps,
        repetition,
        presence,
        frequency,
        vocab,
        logits.stride(0),
        counts.stride(0),
        presence_prefix.stride(0) if has_presence_prefix else counts.stride(0),
        new_tokens.stride(0),
        new_tokens.stride(1),
        new_tokens.stride(2),
        LOGIT_LIMIT=torch.finfo(logits.dtype).max,
        HAS_PRESENCE_PREFIX=has_presence_prefix,
        HISTORY_SYNCED=history_synced,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return logits
