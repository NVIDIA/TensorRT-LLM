# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BOUNDARY_PACK BP-1's graph tail for a batch-1 decode step, in one launch: the spec sampler's four slot stores and
the engine's next-step gathers, both gated by a device-side live word.

``ctl`` (int64 [2]) holds (slot, live), written by the step fold's ``begin_step`` before the replay. With ``live`` 0
the launch writes nothing (a step the fold did not take: a dummy-only replay, a step it could not identify). With
``live`` 1, for the one row:

* the stores, as the sampler's ``_store_outputs`` (zero padding / truncation to the store widths, then index_copy_
  at ``slot``)::

      new_tokens[slot][c]      = src_new[c]   if c < src width else 0      (c < store width), likewise
      next_new_tokens[slot][c] , next_draft_tokens[slot][c];  new_tokens_lens[slot] = src_lens

* with ``GATHER``, the engine's gathers for the next step of the same row (P = 1, no rows before it), as
  ``step_gather`` reads them back from the store just written::

      ids[t]   = next_new_tokens[slot][t]   (t < T)        draft[d] = next_draft_tokens[slot][d]   (d < T - 1)
      pos_off[i] = lens[slot] if i < T else 0              kv_off[i] = lens[slot] - T if i < 1 else 0
      (every i of pos_off / kv_off: the engine zeroes both buffers before its gathers)

Integer copies only. Program 0 does the row; every program zero-fills its ``ZBLOCK`` chunk of the offsets. The int
arguments are not specialized and every pointer is 16-byte aligned (the caller checks), so the two ``GATHER``
variants compiled by ``warm_up`` outside capture serve every graph.
"""

import torch
import triton
import triton.language as tl

ZBLOCK = 1024
_INTS = (
    "s_new_s1",
    "s_new_w",
    "d_new_s0",
    "d_new_s1",
    "d_new_w",
    "s_nnew_s1",
    "s_nnew_w",
    "d_nnew_s0",
    "d_nnew_s1",
    "d_nnew_w",
    "d_lens_s0",
    "s_draft_s1",
    "s_draft_w",
    "d_draft_s0",
    "d_draft_s1",
    "d_draft_w",
    "n_pos",
    "n_kv",
    "T",
)


@triton.jit(do_not_specialize=list(_INTS))
def _bp_tail_kernel(
    ctl,
    s_new,
    s_new_s1,
    s_new_w,
    d_new,
    d_new_s0,
    d_new_s1,
    d_new_w,
    s_nnew,
    s_nnew_s1,
    s_nnew_w,
    d_nnew,
    d_nnew_s0,
    d_nnew_s1,
    d_nnew_w,
    s_lens,
    d_lens,
    d_lens_s0,
    s_draft,
    s_draft_s1,
    s_draft_w,
    d_draft,
    d_draft_s0,
    d_draft_s1,
    d_draft_w,
    ids,
    draft,
    pos_off,
    n_pos,
    kv_off,
    n_kv,
    T,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
    ZBLOCK_: tl.constexpr,
):
    slot = tl.load(ctl)
    live = tl.load(ctl + 1)
    if live != 0:
        pid = tl.program_id(0)
        lens = tl.load(s_lens)
        if pid == 0:
            c = tl.arange(0, BLOCK)
            new = tl.load(s_new + c * s_new_s1, mask=c < s_new_w, other=0)
            tl.store(d_new + slot * d_new_s0 + c * d_new_s1, new, mask=c < d_new_w)
            nnew = tl.load(s_nnew + c * s_nnew_s1, mask=c < s_nnew_w, other=0)
            tl.store(d_nnew + slot * d_nnew_s0 + c * d_nnew_s1, nnew, mask=c < d_nnew_w)
            dr = tl.load(s_draft + c * s_draft_s1, mask=c < s_draft_w, other=0)
            tl.store(d_draft + slot * d_draft_s0 + c * d_draft_s1, dr, mask=c < d_draft_w)
            tl.store(d_lens + slot * d_lens_s0, lens)
            if GATHER:
                tl.store(ids + c, nnew, mask=c < T)
                tl.store(draft + c, dr, mask=c < T - 1)
        if GATHER:
            z = pid * ZBLOCK_ + tl.arange(0, ZBLOCK_)
            tl.store(pos_off + z, tl.where(z < T, lens, 0), mask=z < n_pos)
            tl.store(kv_off + z, tl.where(z < 1, lens - T, 0), mask=z < n_kv)


def block_for(store) -> int:
    """The row block of a sampler store: every store width fits it (fixed per store, so per fold)."""
    widths = (
        store.new_tokens.shape[0],
        store.next_new_tokens.shape[0],
        store.next_draft_tokens.shape[1],
    )
    return triton.next_power_of_2(max(*widths, 1))


def aligned(*tensors: torch.Tensor) -> bool:
    """Every pointer the launch takes is 16-byte aligned (the compiled variants' assumption)."""
    return all(t.data_ptr() % 16 == 0 for t in tensors)


def bp_tail(ctl, src, store, engine, t: int, gather: bool, block: int) -> None:
    """One launch: ``src`` = (new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens) row 0 of the worker's
    int32 outputs ([1, w] / [1]); ``store`` the sampler's store; ``engine`` = (input_ids, draft_tokens,
    position_id_offsets, kv_lens_offsets) int32 contiguous; ``t`` tokens per request (gather only)."""
    s_new, s_nnew, s_lens, s_draft = src
    d_new = store.new_tokens.squeeze(-1).T
    d_nnew = store.next_new_tokens.squeeze(-1).T
    d_lens, d_draft = store.new_tokens_lens, store.next_draft_tokens
    ids, draft, pos_off, kv_off = engine
    n_pos, n_kv = pos_off.numel(), kv_off.numel()
    grid = (max(1, triton.cdiv(max(n_pos, n_kv), ZBLOCK)) if gather else 1,)
    _bp_tail_kernel[grid](
        ctl,
        s_new,
        s_new.stride(1),
        s_new.shape[1],
        d_new,
        d_new.stride(0),
        d_new.stride(1),
        d_new.shape[1],
        s_nnew,
        s_nnew.stride(1),
        s_nnew.shape[1],
        d_nnew,
        d_nnew.stride(0),
        d_nnew.stride(1),
        d_nnew.shape[1],
        s_lens,
        d_lens,
        d_lens.stride(0),
        s_draft,
        s_draft.stride(1),
        s_draft.shape[1],
        d_draft,
        d_draft.stride(0),
        d_draft.stride(1),
        d_draft.shape[1],
        ids,
        draft,
        pos_off,
        n_pos,
        kv_off,
        n_kv,
        t,
        GATHER=gather,
        BLOCK=block,
        ZBLOCK_=ZBLOCK,
        num_warps=4,
    )


def warm_up(ctl, store, engine, block: int) -> None:
    """Compile both variants outside capture, with ``ctl`` live 0 (the launch writes nothing)."""
    src = (
        store.new_tokens.squeeze(-1).T[:1],
        store.next_new_tokens.squeeze(-1).T[:1],
        store.new_tokens_lens[:1],
        store.next_draft_tokens[:1],
    )
    for gather in (False, True):
        bp_tail(ctl, src, store, engine, 2, gather, block)
