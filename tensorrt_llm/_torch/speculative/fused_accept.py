# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict (token-equality) acceptance of a generation-only verify step in one launch.

`SpecWorkerBase._sample_and_accept_draft_tokens_base` builds the accepted tokens and counts
of a uniform verify step from the target's sampled tokens with a chain of small kernels: a
compare, a cumulative product, a row sum, the int32 copies and, under
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, the counter bump, pool hash and threshold of
the synthetic acceptance. `strict_accept` writes the same two outputs from one program:

* `accepted_tokens[i, :] = target_tokens[i * (D + 1) : (i + 1) * (D + 1)]` (int32);
* `num_accepted_tokens[i] = 1 + ` the number of leading draft tokens equal to the target's
  token at the same position, or, with a forced count, `base` (mode 1) or
  `min(base + (pool[(c * CS + i * SS) & (len(pool) - 1)] < frac), D + 1)` with the counter
  `c` advanced by one first (mode 2), all in the arithmetic of the composed ops (int64 hash,
  float32 compare).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Most generation requests one launch covers (one program).
MAX_REQUESTS = 256

FORCE_NONE = 0
FORCE_BASE = 1
FORCE_RANDOM = 2


# Runtime scalars are not specialized: one compiled variant per force mode and draft length,
# whichever batch size a graph first records it at.
@triton.jit(do_not_specialize=["num_gens", "frac", "base", "pool_mask"])
def _strict_accept_kernel(
    target,
    draft,
    accepted,
    num_accepted,
    counter,
    pool,
    num_gens,
    frac,
    base,
    pool_mask,
    FORCE: tl.constexpr,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, D_BLOCK)
    rows = i < num_gens
    cols = j < D + 1
    tgt = tl.load(target + i[:, None] * (D + 1) + j[None, :], mask=rows[:, None] & cols[None, :])
    tl.store(
        accepted + i[:, None] * (D + 1) + j[None, :],
        tgt.to(tl.int32),
        mask=rows[:, None] & cols[None, :],
    )
    if FORCE == 0:
        is_draft = j < D
        dr = tl.load(
            draft + i[:, None] * D + j[None, :], mask=rows[:, None] & is_draft[None, :], other=0
        )
        miss = (dr.to(tl.int64) != tgt.to(tl.int64)) & is_draft[None, :]
        first_miss = tl.min(tl.where(miss, j[None, :], D), axis=1)
        count = 1 + first_miss
    elif FORCE == 1:
        count = tl.zeros([BLOCK], dtype=tl.int32) + base
    else:
        c = tl.load(counter) + 1
        tl.store(counter, c)
        idx = (c * COUNTER_STRIDE + i.to(tl.int64) * SLOT_STRIDE) & pool_mask
        rand = tl.load(pool + idx, mask=rows, other=1.0)
        extra = (rand < frac).to(tl.int32)
        count = tl.minimum(base + extra, D + 1)
    tl.store(num_accepted + i, count.to(tl.int32), mask=rows)


_WARMED: set = set()


def _launch(
    target,
    draft,
    accepted,
    num_accepted,
    counter,
    pool,
    num_gens,
    frac,
    base,
    pool_mask,
    force_mode,
    draft_len,
    counter_stride,
    slot_stride,
) -> None:
    _strict_accept_kernel[(1,)](
        target,
        draft,
        accepted,
        num_accepted,
        counter,
        pool,
        num_gens,
        frac,
        base,
        pool_mask,
        FORCE=force_mode,
        D=draft_len,
        D_BLOCK=triton.next_power_of_2(draft_len + 1),
        BLOCK=MAX_REQUESTS,
        COUNTER_STRIDE=counter_stride,
        SLOT_STRIDE=slot_stride,
        num_warps=4,
    )


def _warm(
    device, draft_len: int, target_dtype, draft_dtype, counter_stride: int, slot_stride: int
) -> None:
    """Launch every force mode once on scratch buffers, so that a mode a CUDA graph first
    records (the synthetic acceptance is off in the eager graph warmup) is compiled and
    loaded already."""
    key = (device, draft_len, target_dtype, draft_dtype, counter_stride, slot_stride)
    if key in _WARMED or torch.cuda.is_current_stream_capturing():
        return
    target = torch.zeros(draft_len + 1, dtype=target_dtype, device=device)
    draft = torch.zeros(max(draft_len, 1), dtype=draft_dtype, device=device)
    accepted = torch.empty(draft_len + 1, dtype=torch.int32, device=device)
    num_accepted = torch.empty(1, dtype=torch.int32, device=device)
    counter = torch.zeros(1, dtype=torch.int64, device=device)
    pool = torch.zeros(1, dtype=torch.float32, device=device)
    for mode in (FORCE_NONE, FORCE_BASE, FORCE_RANDOM):
        # strict_accept passes num_accepted as both pointers outside FORCE_RANDOM.
        mode_counter, mode_pool = (
            (counter, pool) if mode == FORCE_RANDOM else (num_accepted, num_accepted)
        )
        _launch(
            target,
            draft,
            accepted,
            num_accepted,
            mode_counter,
            mode_pool,
            1,
            0.5,
            1,
            0,
            mode,
            draft_len,
            counter_stride,
            slot_stride,
        )
    _WARMED.add(key)


def strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_gens: int,
    draft_len: int,
    force_mode: int = FORCE_NONE,
    force_base: int = 0,
    force_frac: float = 0.0,
    counter: torch.Tensor | None = None,
    pool: torch.Tensor | None = None,
    counter_stride: int = 0,
    slot_stride: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(accepted_tokens [num_gens, draft_len + 1] int32, num_accepted_tokens [num_gens]
    int32)` of a generation-only uniform verify step (see the module docstring).
    `target_tokens` holds `num_gens * (draft_len + 1)` sampled tokens, `draft_tokens` the
    `num_gens * draft_len` proposals (any integer dtype, contiguous). Mode `FORCE_RANDOM`
    advances `counter` (int64 `[1]`) in place and reads `pool` (float32, power-of-two
    length). Graph-capturable; `num_gens` at most `MAX_REQUESTS`."""
    assert 0 < num_gens <= MAX_REQUESTS
    assert target_tokens.is_contiguous() and draft_tokens.is_contiguous()
    dev = target_tokens.device
    accepted = torch.empty((num_gens, draft_len + 1), dtype=torch.int32, device=dev)
    num_accepted = torch.empty(num_gens, dtype=torch.int32, device=dev)
    if force_mode == FORCE_RANDOM:
        assert counter is not None and pool is not None
        pool_size = pool.numel()
        assert pool_size & (pool_size - 1) == 0
        pool_mask = pool_size - 1
    else:
        counter = pool = num_accepted
        pool_mask = 0
    _warm(dev, draft_len, target_tokens.dtype, draft_tokens.dtype, counter_stride, slot_stride)
    _launch(
        target_tokens,
        draft_tokens,
        accepted,
        num_accepted,
        counter,
        pool,
        num_gens,
        float(force_frac),
        int(force_base),
        pool_mask,
        force_mode,
        draft_len,
        counter_stride,
        slot_stride,
    )
    return accepted, num_accepted
