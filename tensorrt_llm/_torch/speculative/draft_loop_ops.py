# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch versions of the one-model draft loop's row bookkeeping, for a generation-only
step that a CUDA graph captures.

* `draft_row_select`: after the first draft step over all `p` rows of every request, each
  request continues from its last accepted row `g = batch_indices[i] * p + accepted[i] - 1 +
  num_ctx_tokens` (a negative `g` wraps, as torch indexing does). One launch writes the row
  ids, the selected hidden rows (the next step's input and the LM-head rows) and the next
  step's positions `position_ids[..., g] + 1`; given the attention lengths, it also moves
  them to one row per request (`seq_lens[i] = 1`, `kv_lens[i] -= draft_len - accepted[i]`).
* `next_new_tokens`: the next verify step's input, `[accepted_tokens[batch_indices[i],
  accepted[i] - 1], next_draft_tokens[i, :]]` per request, in one launch.

Both equal the composed torch ops they replace element for element (integer arithmetic and
copies only).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Most requests one launch covers.
MAX_REQUESTS = 256
#: Hidden-state columns a program copies.
_BLOCK_H = 1024


@triton.jit(do_not_specialize=["num_rows", "num_ctx_tokens"])
def _row_select_kernel(
    batch_indices,
    accepted,
    hidden,
    hidden_stride,
    positions,
    pos_stride,
    out_ids,
    out_hidden,
    out_pos,
    kv_lens,
    seq_lens,
    num_rows,
    num_ctx_tokens,
    width,
    B,
    P: tl.constexpr,
    DRAFT_LEN: tl.constexpr,
    HAS_KV: tl.constexpr,
    HAS_SEQ: tl.constexpr,
    POS_ROWS: tl.constexpr,
    POS_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    i = tl.program_id(0)
    hb = tl.program_id(1)
    acc = tl.load(accepted + i)
    g = tl.load(batch_indices + i).to(tl.int64) * P + acc.to(tl.int64) - 1
    ids = g + num_ctx_tokens
    g = tl.where(ids < 0, ids + num_rows, ids)
    c = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    m = c < width
    vals = tl.load(hidden + g * hidden_stride + c, mask=m)
    tl.store(out_hidden + i.to(tl.int64) * width + c, vals, mask=m)
    if hb == 0:
        tl.store(out_ids + i, ids)
        r = tl.arange(0, POS_BLOCK)
        rm = r < POS_ROWS
        pos = tl.load(positions + r * pos_stride + g, mask=rm)
        tl.store(out_pos + r * B + i, pos + 1, mask=rm)
        if HAS_KV:
            kv = tl.load(kv_lens + i)
            tl.store(kv_lens + i, kv - (DRAFT_LEN - acc))
        if HAS_SEQ:
            tl.store(seq_lens + i, 1)


def draft_row_select(
    batch_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    rows_per_request: int,
    num_ctx_tokens: int = 0,
    kv_lens: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    draft_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(gather_ids, hidden_states[gather_ids], position_ids[..., gather_ids] + 1)` with
    `gather_ids = batch_indices * rows_per_request + num_accepted_tokens - 1 + num_ctx_tokens`
    (int64), for `num_accepted_tokens.shape[0]` requests. `hidden_states` is `[rows, width]`
    with unit column stride; `position_ids` has the rows along its last dim. In place, when
    given: `kv_lens[:n] -= draft_len - num_accepted_tokens` and `seq_lens[:n] = 1`."""
    n = num_accepted_tokens.shape[0]
    rows, width = hidden_states.shape
    assert 0 < n <= MAX_REQUESTS and hidden_states.stride(1) == 1
    assert position_ids.shape[-1] == rows
    pos2 = position_ids.reshape(-1, rows)
    assert pos2.stride(1) == 1
    dev = hidden_states.device
    out_ids = torch.empty(n, dtype=torch.int64, device=dev)
    out_hidden = torch.empty((n, width), dtype=hidden_states.dtype, device=dev)
    out_pos = torch.empty(position_ids.shape[:-1] + (n,), dtype=position_ids.dtype, device=dev)
    _row_select_kernel[(n, triton.cdiv(width, _BLOCK_H))](
        batch_indices,
        num_accepted_tokens,
        hidden_states,
        hidden_states.stride(0),
        pos2,
        pos2.stride(0),
        out_ids,
        out_hidden,
        out_pos,
        kv_lens if kv_lens is not None else out_ids,
        seq_lens if seq_lens is not None else out_ids,
        rows,
        num_ctx_tokens,
        width,
        n,
        P=rows_per_request,
        DRAFT_LEN=draft_len,
        HAS_KV=kv_lens is not None,
        HAS_SEQ=seq_lens is not None,
        POS_ROWS=pos2.shape[0],
        POS_BLOCK=triton.next_power_of_2(pos2.shape[0]),
        BLOCK_H=_BLOCK_H,
        num_warps=4,
    )
    return out_ids, out_hidden, out_pos


@triton.jit(do_not_specialize=["num_gens"])
def _next_new_tokens_kernel(
    accepted_tokens,
    accepted_stride,
    accepted_cols,
    drafts,
    draft_stride,
    batch_indices,
    num_accepted,
    out,
    num_gens,
    D: tl.constexpr,
    W_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    rows = i < num_gens
    b = tl.load(batch_indices + i, mask=rows, other=0).to(tl.int64)
    col = tl.load(num_accepted + i, mask=rows, other=1).to(tl.int64) - 1
    col = tl.where(col < 0, col + accepted_cols, col)
    golden = tl.load(accepted_tokens + b * accepted_stride + col, mask=rows)
    j = tl.arange(0, W_BLOCK)
    is_draft = (j >= 1) & (j < D + 1)
    dr = tl.load(
        drafts + i[:, None].to(tl.int64) * draft_stride + (j - 1)[None, :],
        mask=rows[:, None] & is_draft[None, :],
    )
    vals = tl.where((j == 0)[None, :], golden[:, None], dr)
    tl.store(
        out + i[:, None].to(tl.int64) * (D + 1) + j[None, :],
        vals,
        mask=rows[:, None] & (j < D + 1)[None, :],
    )


def next_new_tokens(
    accepted_tokens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    batch_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """`torch.concat([accepted_tokens[batch_indices, num_accepted_tokens - 1, None],
    next_draft_tokens], 1)` for `num_accepted_tokens.shape[0]` requests, in one launch."""
    n = num_accepted_tokens.shape[0]
    d = next_draft_tokens.shape[1]
    assert 0 < n <= MAX_REQUESTS
    assert accepted_tokens.stride(1) == 1 and next_draft_tokens.stride(1) == 1
    assert accepted_tokens.dtype == next_draft_tokens.dtype
    out = torch.empty((n, d + 1), dtype=accepted_tokens.dtype, device=accepted_tokens.device)
    _next_new_tokens_kernel[(1,)](
        accepted_tokens,
        accepted_tokens.stride(0),
        accepted_tokens.shape[1],
        next_draft_tokens,
        next_draft_tokens.stride(0),
        batch_indices,
        num_accepted_tokens,
        out,
        n,
        D=d,
        W_BLOCK=triton.next_power_of_2(d + 1),
        BLOCK=MAX_REQUESTS,
        num_warps=4,
    )
    return out
