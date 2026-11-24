# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The flat chain's step begin in one launch: the target's embedding rows and the KDA
checkpoint indices.

Unfolded, the begin is the embedding gather (``flat_glue.embedding``) and the ten integer
ops of ``KDAReplayState.checkpoint_indices``, each its own launch at the head of the decode
graph. :func:`begin_fold` writes the same three tensors from one kernel: row copies and
integer index arithmetic only, so the results are bitwise the unfolded ones (the gate is
``test_flat_begin.py``).
"""

import torch
import triton
import triton.language as tl

_BLOCK_H = 1024


def folds_embedding(ids: torch.Tensor, table: torch.Tensor) -> bool:
    """Whether ``flat_glue.embedding(ids, table)`` would take the vectorized row copy that the
    fold reproduces (``F.embedding`` otherwise)."""
    vec = 16 // table.element_size()
    return (
        ids.dim() == 1
        and ids.is_cuda
        and table.dim() == 2
        and table.is_contiguous()
        and table.shape[1] % vec == 0
        and ids.shape[0] > 0
    )


@triton.jit
def _flat_begin_kernel(
    ids_ptr,
    table_ptr,
    embed_ptr,
    vocab,
    hidden,
    state_ptr,
    state_stride,
    count_ptr,
    count_stride,
    slots_ptr,
    accepted_ptr,
    num_requests,
    header_units,
    slot_units,
    state_units,
    TOKENS: tl.constexpr,
    TOKENS_POW2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VEC: tl.constexpr,
):
    row = tl.program_id(0)
    # Embedding row: spec_head_kernels._embedding_gather_kernel (ids outside the table copy a
    # zero row).
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    offs = tl.max_contiguous(tl.multiple_of(offs, VEC), VEC)
    mask = offs < hidden
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= 0) & (token < vocab)
    src = table_ptr + tl.where(valid, token, 0) * hidden
    vals = tl.load(src + offs, mask=mask & valid, other=0.0)
    tl.store(embed_ptr + row.to(tl.int64) * hidden + offs, vals, mask=mask)
    if (tl.program_id(1) == 0) & (row < num_requests):
        # checkpoint_indices: header + base * slot_units + position * state_units in int64,
        # truncated to int32, and prev_num_accepted_tokens[base] + 1 in its own dtype,
        # truncated to int32.
        base = tl.load(state_ptr + row.to(tl.int64) * state_stride).to(tl.int64)
        positions = tl.arange(0, TOKENS_POW2).to(tl.int64)
        rows = header_units + base * slot_units + positions * state_units
        tl.store(
            slots_ptr + row * TOKENS + tl.arange(0, TOKENS_POW2),
            rows.to(tl.int32),
            mask=tl.arange(0, TOKENS_POW2) < TOKENS,
        )
        count = tl.load(count_ptr + base * count_stride) + 1
        tl.store(accepted_ptr + row, count.to(tl.int32))


def begin_fold(
    replay, state_indices: torch.Tensor, ids: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(*replay.checkpoint_indices(state_indices), flat_glue.embedding(ids, table))`` in one
    launch, for ids and a table :func:`folds_embedding` accepts.

    Args:
        replay: The manager's ``KDAReplayState`` in checkpoint mode.
        state_indices: [B] KDA state slots of the step's requests (any integer dtype, stride).
        ids: [rows] token ids, rows >= B.
        table: [vocab, hidden] contiguous embedding table.

    Returns:
        ``slots`` [B, 1 + num_speculative_tokens] int32, ``accepted`` [B] int32 and the
        [rows, hidden] embedding rows.
    """
    if not replay.checkpoints or not replay.checkpoint_slot_units:
        raise RuntimeError("KDA checkpoint indices need a bound checkpoint-mode state")
    counts = replay.prev_num_accepted_tokens
    if state_indices.dim() != 1 or counts.dim() != 1:
        raise ValueError("The step begin takes 1-D state indices and accepted counts")
    num_requests = state_indices.shape[0]
    rows = ids.shape[0]
    if num_requests > rows:
        raise ValueError("Every request needs at least one embedding row")
    tokens = 1 + replay.num_speculative_tokens
    device = table.device
    slots = torch.empty((num_requests, tokens), dtype=torch.int32, device=device)
    accepted = torch.empty((num_requests,), dtype=torch.int32, device=device)
    vocab, hidden = table.shape
    embed = torch.empty((rows, hidden), dtype=table.dtype, device=device)
    _flat_begin_kernel[(rows, triton.cdiv(hidden, _BLOCK_H))](
        ids,
        table,
        embed,
        vocab,
        hidden,
        state_indices,
        state_indices.stride(0),
        counts,
        counts.stride(0),
        slots,
        accepted,
        num_requests,
        replay.checkpoint_header_units,
        replay.checkpoint_slot_units,
        replay.checkpoint_state_units,
        TOKENS=tokens,
        TOKENS_POW2=triton.next_power_of_2(tokens),
        BLOCK_H=_BLOCK_H,
        VEC=16 // table.element_size(),
        num_warps=4,
    )
    return slots, accepted, embed
