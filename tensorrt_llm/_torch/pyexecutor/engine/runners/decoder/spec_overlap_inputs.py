# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch staging of an overlap-scheduled speculative step's inputs.

The step's host inputs arrive as one int32 slab; the rows that verify the
previous step's drafts take their tokens, drafts and acceptance corrections
from that step's device outputs. One kernel writes all of it.
"""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_BLOCK = 1024
# Values that change from step to step; specializing on them would recompile mid-run.
_PER_STEP_ARGS = [
    "ids_end",
    "drafts_end",
    "accepted_end",
    "slots_end",
    "positions_end",
    "total",
    "batch",
    "start",
    "tokens_per_step",
]


@triton.jit(do_not_specialize=_PER_STEP_ARGS, do_not_specialize_on_alignment=_PER_STEP_ARGS)
def _stage_kernel(
    slab,
    ids_end,
    drafts_end,
    accepted_end,
    slots_end,
    positions_end,
    total,
    input_ids,
    draft_tokens,
    num_accepted_draft_tokens,
    previous_slots,
    position_ids,
    gather_ids,
    new_tokens,
    new_tokens_step,
    new_tokens_slot,
    next_drafts,
    next_drafts_slot,
    next_drafts_col,
    new_tokens_lens,
    new_tokens_lens_slot,
    position_offsets,
    position_offsets_len,
    kv_lens_offsets,
    kv_lens_offsets_len,
    batch,
    start,
    tokens_per_step,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # Each slab segment goes to the head of its buffer.
    mask = j < total
    value = tl.load(slab + j, mask=mask)
    tl.store(input_ids + j, value, mask=mask & (j < ids_end))
    segment = mask & (j >= ids_end) & (j < drafts_end)
    tl.store(draft_tokens + (j - ids_end), value, mask=segment)
    segment = mask & (j >= drafts_end) & (j < accepted_end)
    tl.store(num_accepted_draft_tokens + (j - drafts_end), value, mask=segment)
    segment = mask & (j >= accepted_end) & (j < slots_end)
    tl.store(previous_slots + (j - accepted_end), value, mask=segment)
    segment = mask & (j >= slots_end) & (j < positions_end)
    tl.store(position_ids + (j - slots_end), value, mask=segment)
    tl.store(gather_ids + (j - positions_end), value, mask=mask & (j >= positions_end))
    # The rows with a previous step append its tokens and drafts. Their slots are
    # read from the slab, not from the buffer this launch writes.
    mask = j < batch * tokens_per_step
    slot = tl.load(slab + accepted_end + j // tokens_per_step, mask=mask, other=0).to(tl.int64)
    token = tl.load(
        new_tokens + (j % tokens_per_step) * new_tokens_step + slot * new_tokens_slot, mask=mask
    )
    tl.store(input_ids + ids_end + j, token, mask=mask)
    draft_width = tokens_per_step - 1
    mask = j < batch * draft_width
    divisor = tl.maximum(draft_width, 1)
    slot = tl.load(slab + accepted_end + j // divisor, mask=mask, other=0).to(tl.int64)
    draft = tl.load(
        next_drafts + slot * next_drafts_slot + (j % divisor) * next_drafts_col, mask=mask
    )
    tl.store(draft_tokens + (drafts_end - ids_end) + j, draft, mask=mask)
    # Their acceptance corrections, zero for every other row.
    mask = j < position_offsets_len
    row = j - start * tokens_per_step
    rows = (row >= 0) & (row < batch * tokens_per_step)
    slot = tl.load(slab + accepted_end + row // tokens_per_step, mask=mask & rows, other=0).to(
        tl.int64
    )
    value = tl.load(new_tokens_lens + slot * new_tokens_lens_slot, mask=mask & rows, other=0)
    tl.store(position_offsets + j, tl.where(rows, value, 0), mask=mask)
    mask = j < kv_lens_offsets_len
    rows = (j >= start) & (j < start + batch)
    slot = tl.load(slab + accepted_end + (j - start), mask=mask & rows, other=0).to(tl.int64)
    value = tl.load(new_tokens_lens + slot * new_tokens_lens_slot, mask=mask & rows, other=0)
    tl.store(kv_lens_offsets + j, tl.where(rows, value - tokens_per_step, 0), mask=mask)


def stage_spec_overlap_inputs(
    slab: torch.Tensor,
    segment_ends: Sequence[int],
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_accepted_draft_tokens: torch.Tensor,
    previous_slots: torch.Tensor,
    position_ids: torch.Tensor,
    gather_ids: torch.Tensor,
    position_offsets: torch.Tensor,
    kv_lens_offsets: torch.Tensor,
    new_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    first_previous_row: int,
    tokens_per_step: int,
) -> None:
    """Write one step's inputs from ``slab`` and the previous step's outputs.

    ``slab`` holds the host values of ``input_ids`` through ``gather_ids`` back to
    back, segment ``k`` ending at ``segment_ends[k]``; each goes to the head of its
    buffer. Every previous slot then appends ``new_tokens[:tokens_per_step, slot, 0]``
    to ``input_ids`` and ``next_draft_tokens[slot, :tokens_per_step - 1]`` to
    ``draft_tokens``. ``position_offsets`` and ``kv_lens_offsets`` get its accepted
    length (minus ``tokens_per_step`` for the kv lengths) in its rows, counted from
    ``first_previous_row``, and zero everywhere else.
    """
    total = segment_ends[-1]
    batch = segment_ends[3] - segment_ends[2]
    count = max(total, batch * tokens_per_step, position_offsets.numel(), kv_lens_offsets.numel())
    _stage_kernel[(triton.cdiv(count, _BLOCK),)](
        slab,
        *segment_ends,
        input_ids,
        draft_tokens,
        num_accepted_draft_tokens,
        previous_slots,
        position_ids,
        gather_ids,
        new_tokens,
        new_tokens.stride(0),
        new_tokens.stride(1),
        next_draft_tokens,
        next_draft_tokens.stride(0),
        next_draft_tokens.stride(1),
        new_tokens_lens,
        new_tokens_lens.stride(0),
        position_offsets,
        position_offsets.numel(),
        kv_lens_offsets,
        kv_lens_offsets.numel(),
        batch,
        first_previous_row,
        tokens_per_step,
        BLOCK=_BLOCK,
    )
