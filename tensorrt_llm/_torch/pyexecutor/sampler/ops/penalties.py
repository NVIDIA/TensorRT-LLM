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

* :func:`update_occurrence_workspace` -- folds newly-committed tokens into the
  persistent per-slot occurrence workspace (a small sparse scatter).
* :func:`apply_occurrence_penalties_triton` -- a fused Triton kernel that applies the
  penalties in place on selected logits rows.
* :func:`apply_beam_occurrence_penalties_triton` -- the parent-aware beam-search variant
  operating on normalized log probabilities and double-buffered occurrence state.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(kw_only=True)
class BeamPenaltyMetadata:
    """Persistent state consumed by the beam-search occurrence-penalty kernel."""

    workspace_a: torch.Tensor
    workspace_b: torch.Tensor
    active: torch.Tensor
    buffer_indices: torch.Tensor
    prompt_lengths: torch.Tensor
    new_tokens: torch.Tensor
    predecessor_beams: torch.Tensor
    repetition: torch.Tensor
    presence: torch.Tensor
    frequency: torch.Tensor


def update_occurrence_workspace(
    counts: torch.Tensor,
    presence_prefix: torch.Tensor | None,
    counted_slots: torch.Tensor,
    counted_tokens: torch.Tensor,
    prefix_slots: torch.Tensor,
    prefix_tokens: torch.Tensor,
) -> None:
    """Fold newly-committed tokens into the persistent occurrence workspace.

    Mirrors the accumulation ``batchApplyPenalty`` performs in its ``penaltyWorkspace``
    (``[..., 2 * vocabSize]`` = ``[presencePrefix, counts]``): tokens in the ignored
    prompt prefix ``[0, prompt_ignore_length)`` set the presence bitmap only (they count
    for repetition but not for presence/frequency), while all other tokens (the rest of
    the prompt plus every generated token) increment the occurrence counts.

    Arguments (all index tensors are 1-D and pre-split on the host):
      counts: ``int32[num_slots, vocab_size]`` occurrence counts, updated in place.
      presence_prefix: ``int32[num_slots, vocab_size]`` prefix presence bitmap, or
        ``None`` when no active request uses ``prompt_ignore_length``.
      counted_slots / counted_tokens: (slot, token) pairs to increment in ``counts``.
      prefix_slots / prefix_tokens: (slot, token) pairs to mark in ``presence_prefix``.
    """
    if counted_slots.numel() > 0:
        ones = torch.ones(counted_slots.shape[0], dtype=counts.dtype, device=counts.device)
        # accumulate=True sums repeated (slot, token) pairs -> occurrence count.
        counts.index_put_((counted_slots, counted_tokens), ones, accumulate=True)
    if presence_prefix is not None and prefix_slots.numel() > 0:
        presence_prefix[prefix_slots, prefix_tokens] = 1


@triton.jit
def _occurrence_penalty_kernel(
    logits_ptr,  # [num_logits_rows, vocab] (in/out)
    counts_ptr,  # [num_slots, vocab] int32
    presence_ptr,  # [num_slots, vocab] int32 (placeholder when HAS_PRESENCE=False)
    active_rows_ptr,  # [num_active] int64 -> logits row
    row_slots_ptr,  # [num_active] int64 -> workspace/param row (slot)
    repetition_ptr,  # [num_slots] float32
    presence_pen_ptr,  # [num_slots] float32
    frequency_ptr,  # [num_slots] float32
    temperature_ptr,  # [num_slots] float32
    vocab,
    logits_row_stride,
    workspace_row_stride,
    LOGIT_LIMIT: tl.constexpr,
    HAS_PRESENCE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (active row, vocab block). Fully indirect: logits are read/
    # written at row ``active_rows[a]`` while counts/presence/params are read at
    # slot ``row_slots[a]`` -- no [num_active, vocab] gather/scatter materialized.
    a = tl.program_id(0)
    r = tl.load(active_rows_ptr + a)  # logits row
    s = tl.load(row_slots_ptr + a)  # workspace / param row (slot)

    rep = tl.load(repetition_ptr + s)
    pre = tl.load(presence_pen_ptr + s)
    freq = tl.load(frequency_ptr + s)
    temp = tl.load(temperature_ptr + s)

    offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab

    logit = tl.load(logits_ptr + r * logits_row_stride + offs, mask=mask, other=0.0).to(tl.float32)
    cnt = tl.load(counts_ptr + s * workspace_row_stride + offs, mask=mask, other=0)

    present = cnt > 0
    if HAS_PRESENCE:
        pp = tl.load(presence_ptr + s * workspace_row_stride + offs, mask=mask, other=0)
        present = present | (pp > 0)

    # Repetition (sign-based multiply/divide) where present anywhere in the sequence.
    repeated = tl.where(logit < 0.0, logit * rep, logit / rep)
    logit = tl.where(present, repeated, logit)
    # Presence + frequency where counted; the additive term is scaled by temperature so
    # the later ``logits / temperature`` reproduces the C++ ordering.
    sub = tl.where(cnt > 0, pre + freq * cnt.to(tl.float32), 0.0) * temp
    logit = logit - sub
    # Match batchApplyPenalty's finite dtype clamp before writing the result.
    logit = tl.maximum(-LOGIT_LIMIT, tl.minimum(logit, LOGIT_LIMIT))

    tl.store(
        logits_ptr + r * logits_row_stride + offs,
        logit.to(logits_ptr.dtype.element_ty),
        mask=mask,
    )


def apply_occurrence_penalties_triton(
    logits: torch.Tensor,
    active_rows: torch.Tensor,
    row_slots: torch.Tensor,
    counts: torch.Tensor,
    presence_prefix: torch.Tensor | None,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Fused CUDA penalty application; modifies ``logits`` in place and returns it.

    Applies ``batchApplyPenalty`` steps 3-4 to each row ``logits[active_rows[a]]``,
    reading counts / presence bitmap / per-slot penalty scalars at slot ``row_slots[a]``:
    repetition (sign-based multiply/divide, where the token is present anywhere), then
    presence + frequency (where counted), with the additive term scaled by temperature so
    the later ``logits / temperature`` reproduces the C++ ordering.
    ``repetition/presence/frequency/temperature`` are per-slot ``[num_slots]`` float32
    tensors; ``counts`` / ``presence_prefix`` are ``[num_slots, vocab]`` int32.
    """
    num_active = active_rows.shape[0]
    if num_active == 0:
        return logits
    vocab = logits.shape[1]
    has_presence = presence_prefix is not None
    # Fixed block size: the kernel is in-place, so autotune (which relaunches on the
    # same logits) is unsafe; 1024 is a good default for this element-wise pass.
    block_size = 1024
    grid = (num_active, triton.cdiv(vocab, block_size))
    _occurrence_penalty_kernel[grid](
        logits,
        counts,
        presence_prefix if has_presence else counts,
        active_rows,
        row_slots,
        repetition,
        presence,
        frequency,
        temperature,
        vocab,
        logits.stride(0),
        counts.stride(0),
        LOGIT_LIMIT=torch.finfo(logits.dtype).max,
        HAS_PRESENCE=has_presence,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return logits


@triton.jit
def _beam_occurrence_penalty_kernel(
    log_probs_ptr,
    workspace_a_ptr,
    workspace_b_ptr,
    active_ptr,
    buffer_indices_ptr,
    prompt_lengths_ptr,
    new_tokens_ptr,
    predecessor_beams_ptr,
    seq_slots_ptr,
    seq_lens_ptr,
    repetition_ptr,
    presence_ptr,
    frequency_ptr,
    vocab,
    beam_width: tl.constexpr,
    logits_row_stride,
    workspace_slot_stride,
    workspace_beam_stride,
    workspace_plane_stride,
    new_tokens_slot_stride,
    new_tokens_beam_stride,
    predecessor_slot_stride,
    predecessor_beam_stride,
    LOGIT_LIMIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy the selected parent state, add its token, and penalize one beam row."""
    row = tl.program_id(0)
    batch_idx = row // beam_width
    beam_idx = row % beam_width
    slot = tl.load(seq_slots_ptr + batch_idx)
    active = tl.load(active_ptr + slot) != 0
    current_buffer = tl.load(buffer_indices_ptr + slot)
    is_context = tl.load(seq_lens_ptr + batch_idx) <= tl.load(prompt_lengths_ptr + slot)

    parent_beam = tl.load(
        predecessor_beams_ptr + slot * predecessor_slot_stride + beam_idx * predecessor_beam_stride
    )
    source_beam = tl.where(is_context, beam_idx, parent_beam)

    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vocab_mask = offsets < vocab
    source_offset = slot * workspace_slot_stride + source_beam * workspace_beam_stride + offsets
    prefix_offset = source_offset
    counts_offset = source_offset + workspace_plane_stride

    read_a_mask = vocab_mask & active & (current_buffer == 0)
    read_b_mask = vocab_mask & active & (current_buffer != 0)
    prefix_a = tl.load(workspace_a_ptr + prefix_offset, mask=read_a_mask, other=0)
    prefix_b = tl.load(workspace_b_ptr + prefix_offset, mask=read_b_mask, other=0)
    counts_a = tl.load(workspace_a_ptr + counts_offset, mask=read_a_mask, other=0)
    counts_b = tl.load(workspace_b_ptr + counts_offset, mask=read_b_mask, other=0)
    prefix = prefix_a + prefix_b
    counts = counts_a + counts_b

    last_token = tl.load(
        new_tokens_ptr + slot * new_tokens_slot_stride + beam_idx * new_tokens_beam_stride
    )
    counts += tl.where((~is_context) & (offsets == last_token), 1, 0)

    destination_offset = slot * workspace_slot_stride + beam_idx * workspace_beam_stride + offsets
    generation_mask = vocab_mask & active & (~is_context)
    tl.store(
        workspace_b_ptr + destination_offset,
        prefix,
        mask=generation_mask & (current_buffer == 0),
    )
    tl.store(
        workspace_b_ptr + destination_offset + workspace_plane_stride,
        counts,
        mask=generation_mask & (current_buffer == 0),
    )
    tl.store(
        workspace_a_ptr + destination_offset,
        prefix,
        mask=generation_mask & (current_buffer != 0),
    )
    tl.store(
        workspace_a_ptr + destination_offset + workspace_plane_stride,
        counts,
        mask=generation_mask & (current_buffer != 0),
    )

    log_prob = tl.load(
        log_probs_ptr + row * logits_row_stride + offsets,
        mask=vocab_mask & active,
        other=0.0,
    ).to(tl.float32)
    repetition = tl.load(repetition_ptr + slot)
    presence = tl.load(presence_ptr + slot)
    frequency = tl.load(frequency_ptr + slot)
    present = (prefix > 0) | (counts > 0)
    repeated = tl.where(log_prob < 0.0, log_prob * repetition, log_prob / repetition)
    penalized = tl.where(present, repeated, log_prob)
    penalized -= tl.where(
        counts > 0,
        presence + frequency * counts.to(tl.float32),
        0.0,
    )
    # Beam scores are already post-temperature, so this is the exact C++ clamp point.
    penalized = tl.maximum(-LOGIT_LIMIT, tl.minimum(penalized, LOGIT_LIMIT))
    tl.store(
        log_probs_ptr + row * logits_row_stride + offsets,
        penalized.to(log_probs_ptr.dtype.element_ty),
        mask=vocab_mask & active,
    )


@triton.jit
def _toggle_beam_workspace_kernel(
    active_ptr,
    buffer_indices_ptr,
    prompt_lengths_ptr,
    seq_slots_ptr,
    seq_lens_ptr,
):
    batch_idx = tl.program_id(0)
    slot = tl.load(seq_slots_ptr + batch_idx)
    active = tl.load(active_ptr + slot) != 0
    is_generation = tl.load(seq_lens_ptr + batch_idx) > tl.load(prompt_lengths_ptr + slot)
    current_buffer = tl.load(buffer_indices_ptr + slot)
    next_buffer = tl.where(active & is_generation, 1 - current_buffer, current_buffer)
    tl.store(buffer_indices_ptr + slot, next_buffer)


def apply_beam_occurrence_penalties_triton(
    log_probs: torch.Tensor,
    metadata: BeamPenaltyMetadata,
    seq_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    beam_width: int,
) -> torch.Tensor:
    """Apply parent-aware beam penalties to normalized log probabilities in place.

    ``workspace_a`` and ``workspace_b`` use the C++ layout with its decoding-step
    dimension collapsed: ``[max_slots, max_beam_width, 2, vocab_size]``. TorchSampler
    rejects speculative decoding with beam search, so that dimension is always one.
    During generation, each output beam copies the workspace selected by its predecessor,
    adds the token selected in the preceding sampling step, and writes the result to the
    other buffer. Context rows use the prompt-initialized current buffer directly.
    """
    if seq_slots.numel() == 0:
        return log_probs
    vocab = log_probs.size(-1)
    flat_log_probs = log_probs.view(-1, vocab)
    block_size = 1024
    num_rows = seq_slots.numel() * beam_width
    assert flat_log_probs.size(0) == num_rows
    grid = (num_rows, triton.cdiv(vocab, block_size))
    workspace = metadata.workspace_a
    _beam_occurrence_penalty_kernel[grid](
        flat_log_probs,
        metadata.workspace_a,
        metadata.workspace_b,
        metadata.active,
        metadata.buffer_indices,
        metadata.prompt_lengths,
        metadata.new_tokens,
        metadata.predecessor_beams,
        seq_slots,
        seq_lens,
        metadata.repetition,
        metadata.presence,
        metadata.frequency,
        vocab,
        beam_width,
        flat_log_probs.stride(0),
        workspace.stride(0),
        workspace.stride(1),
        workspace.stride(2),
        metadata.new_tokens.stride(1),
        metadata.new_tokens.stride(2),
        metadata.predecessor_beams.stride(0),
        metadata.predecessor_beams.stride(1),
        LOGIT_LIMIT=torch.finfo(log_probs.dtype).max,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    _toggle_beam_workspace_kernel[(seq_slots.numel(),)](
        metadata.active,
        metadata.buffer_indices,
        metadata.prompt_lengths,
        seq_slots,
        seq_lens,
    )
    return log_probs
