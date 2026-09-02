# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused end-ID and maximum-length checks for the sampler fast path."""

import torch
import triton
import triton.language as tl

__all__ = ["MAX_FUSED_ELEMENTS_PER_REQUEST", "fused_write_finish_reasons"]

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
    num_elements = max_tokens * beam_width
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
        beam_width=beam_width,
        num_elements=num_elements,
        not_finished=not_finished_value,
        length=length_value,
        end_id=end_id_value,
        BLOCK=block,
        num_warps=min(8, block // 32),
    )
