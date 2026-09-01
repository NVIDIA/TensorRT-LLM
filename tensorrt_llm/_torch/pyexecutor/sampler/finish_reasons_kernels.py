# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused stop-criteria kernel for :class:`FinishReasonsHandler`.

The end-id and max-length criteria are per-element integer comparisons over a
handful of values, but expressing them with tensor ops costs eleven kernel
launches per step (three gathers, a fill, two comparisons, two masked fills and
a scatter). That is invisible while the batch is large, and dominant at decode
batch sizes where every launch is a round trip through the host between two
one-CTA kernels. This computes the same reasons in one launch.

Stop words and beam search keep the tensor-op path; only the criteria that
depend on nothing but the step's own tokens are fused here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["fused_write_finish_reasons", "MAX_FUSED_ELEMENTS_PER_REQUEST"]

#: Upper bound on ``max_tokens * beam_width`` the fused path accepts. One
#: program handles a whole request, so this is the widest block a single
#: program may load; wider shapes fall back to the tensor-op path.
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
    num_requests,
    beam_width: tl.constexpr,
    num_elements: tl.constexpr,
    not_finished: tl.constexpr,
    length: tl.constexpr,
    end_id: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    if request >= num_requests:
        return

    slot = tl.load(seq_slots_ptr + request).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + request).to(tl.int32)
    max_length = tl.load(max_lengths_ptr + slot).to(tl.int32)
    request_end_id = tl.load(end_ids_ptr + slot)

    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_elements
    # Row-major over (token, beam), matching the [max_tokens, slot, beam] view
    # both buffers are indexed through.
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

    # The step's i-th token would leave the sequence at seq_len + i + 1 tokens.
    at_max_length = (seq_len + token_index + 1) >= max_length
    is_end_id = tokens == request_end_id
    # End-id outranks max-length: the tensor-op path writes the max-length
    # reason first and lets the end-id fill overwrite it.
    reasons = tl.where(is_end_id, end_id, tl.where(at_max_length, length, not_finished))

    tl.store(
        finish_reasons_ptr
        + token_index * stride_reasons_token
        + slot * stride_reasons_slot
        + beam_index * stride_reasons_beam,
        reasons.to(finish_reasons_ptr.dtype.element_ty),
        mask=mask,
    )


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
    """Write the end-id and max-length finish reasons for one sampler step.

    Overwrites ``finish_reasons[:, seq_slots, :]`` in full, so it subsumes the
    reset to ``not_finished_value`` the tensor-op path needs before it starts
    masking reasons in.

    Args:
        finish_reasons: Destination. Shape ``[max_tokens, num_slots, beam_width]``.
        new_tokens: The step's sampled tokens, indexed by slot.
          Shape ``[max_tokens, num_slots, beam_width]``.
        seq_slots: Slot of each processed request. Shape ``[num_requests]``.
        seq_lens: Sequence length of each processed request, before this step's
          tokens. Shape ``[num_requests]``.
        max_lengths: Per-slot maximum sequence length.
        end_ids: Per-slot end id.
        max_tokens: Tokens produced per request this step.
        beam_width: Beams per request.
        not_finished_value: ``FinishReason.NOT_FINISHED`` as an int.
        length_value: ``FinishReason.LENGTH`` as an int.
        end_id_value: ``FinishReason.END_ID`` as an int.
    """
    num_requests = seq_slots.shape[0]
    num_elements = max_tokens * beam_width
    # A program covers one request's whole (token, beam) block, which is a
    # single element at decode -- size the block to the work rather than to
    # the buffer, so the common case launches one warp.
    block = max(32, triton.next_power_of_2(num_elements))
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
        num_requests,
        beam_width=beam_width,
        num_elements=num_elements,
        not_finished=not_finished_value,
        length=length_value,
        end_id=end_id_value,
        BLOCK=block,
        num_warps=min(8, block // 32),
    )
