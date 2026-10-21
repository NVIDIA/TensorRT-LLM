# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash verify: strict acceptance and the forced-acceptance override in one launch.

The built-in ``SpecWorkerBase._sample_and_accept_draft_tokens_base``
(``_torch/speculative/interface.py``) turns the sampled target tokens into the
step's accepted tokens and counts with a chain of torch ops -- a ``ones``
fill, the token comparison, two dtype copies, a cub scan for ``cumprod``, a
sum, an add, the accepted-token copy -- and its forced-acceptance override
(``_apply_force_accepted_tokens``) adds a counter bump, an ``arange``, two
multiplies, an add, a mask, a pool gather, a compare, a cast, an add, a clamp
and two copies: about twenty launches of one to five microseconds for a
handful of integers. This entry does all of it in one single-program kernel: for up to
``LOOP_REQUESTS`` requests a loop over the requests, above that (the decode
steps of a generation worker of ``max_batch_size`` up to ``MAX_REQUESTS``) the
same arithmetic on one ``[requests, K + 1]`` tile.

Everything it computes is integer arithmetic plus one fp32 comparison of a
pool value against the fractional rate, so its outputs are bitwise the
built-in ones.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Forced-acceptance modes, as the built-in override branches.
FORCE_NONE, FORCE_FIXED, FORCE_FRACTIONAL = 0, 1, 2

#: The built-in override's hash constants (``interface.py`` module constants
#: ``_FORCE_ACCEPT_RNG_COUNTER_STRIDE`` / ``_SLOT_STRIDE``); the pool size is
#: read off the pool, which must be a power of two.
COUNTER_STRIDE, SLOT_STRIDE = 6007, 1009

#: Requests the single-program loop serves; above this the loop stops being
#: cheaper than the built-in vectorized chain, and the tile kernel takes over.
LOOP_REQUESTS = 64
#: Requests one call takes (the tile kernel's request block); callers send larger
#: batches to the built-in.
MAX_REQUESTS = 128


@triton.jit
def _dflash_accept_kernel(
    target_ptr,
    draft_ptr,
    draft_row_stride,
    accepted_ptr,
    num_accepted_ptr,
    pool_ptr,
    counter_ptr,
    num_contexts,
    num_gens,
    base_total,
    max_total,
    frac,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    row_width = K + 1
    # Context rows take their one sampled token; the rest of their row is
    # left as allocated, as in the built-in.
    for b in range(num_contexts):
        token = tl.load(target_ptr + b)
        tl.store(accepted_ptr + b * row_width, token.to(tl.int32))
        tl.store(num_accepted_ptr + b, 1)
    counter = tl.zeros([], dtype=tl.int64)
    if FORCE == 2:
        # One bump per call, before any request reads it.
        counter = tl.load(counter_ptr).to(tl.int64) + 1
        tl.store(counter_ptr, counter)
    for g in range(num_gens):
        b = num_contexts + g
        tokens = tl.load(
            target_ptr + num_contexts + g * row_width + cols, mask=cols < row_width, other=0
        )
        tl.store(accepted_ptr + b * row_width + cols, tokens.to(tl.int32), mask=cols < row_width)
        if FORCE == 0:
            drafts = tl.load(draft_ptr + g * draft_row_stride + cols, mask=cols < K, other=0)
            mismatch = (cols < K) & (drafts.to(tl.int64) != tokens.to(tl.int64))
            # Leading matches = index of the first mismatch, or K.
            accepted = tl.min(tl.where(mismatch, cols, K), axis=0) + 1
        elif FORCE == 1:
            accepted = base_total
        else:
            index = (counter * COUNTER_STRIDE + g * SLOT_STRIDE) & POOL_MASK
            draw = tl.load(pool_ptr + index)
            accepted = tl.minimum(base_total + (draw < frac).to(tl.int32), max_total)
        tl.store(num_accepted_ptr + b, accepted.to(tl.int32))


@triton.jit
def _dflash_accept_tile_kernel(
    target_ptr,
    draft_ptr,
    draft_row_stride,
    accepted_ptr,
    num_accepted_ptr,
    pool_ptr,
    counter_ptr,
    num_contexts,
    num_gens,
    base_total,
    max_total,
    frac,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    GENS: tl.constexpr,
    FORCE: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
):
    """``_dflash_accept_kernel`` with the generation requests as one tile's rows."""
    cols = tl.arange(0, BLOCK)
    gens = tl.arange(0, GENS)
    row_width = K + 1
    for b in range(num_contexts):
        token = tl.load(target_ptr + b)
        tl.store(accepted_ptr + b * row_width, token.to(tl.int32))
        tl.store(num_accepted_ptr + b, 1)
    in_gen = gens < num_gens
    in_row = cols < row_width
    tokens = tl.load(
        target_ptr + num_contexts + gens[:, None] * row_width + cols[None, :],
        mask=in_gen[:, None] & in_row[None, :],
        other=0,
    )
    tl.store(
        accepted_ptr + (num_contexts + gens[:, None]) * row_width + cols[None, :],
        tokens.to(tl.int32),
        mask=in_gen[:, None] & in_row[None, :],
    )
    if FORCE == 0:
        drafts = tl.load(
            draft_ptr + gens[:, None] * draft_row_stride + cols[None, :],
            mask=in_gen[:, None] & (cols < K)[None, :],
            other=0,
        )
        mismatch = (cols < K)[None, :] & (drafts.to(tl.int64) != tokens.to(tl.int64))
        # Leading matches = index of the first mismatch, or K.
        accepted = tl.min(tl.where(mismatch, cols[None, :], K), axis=1) + 1
    elif FORCE == 1:
        accepted = tl.zeros([GENS], dtype=tl.int32) + base_total
    else:
        # One bump per call, before any request reads it. The program has several warps:
        # every warp reads the counter before any warp writes it back.
        counter = tl.load(counter_ptr).to(tl.int64) + 1
        tl.debug_barrier()
        tl.store(counter_ptr, counter)
        index = (counter * COUNTER_STRIDE + gens.to(tl.int64) * SLOT_STRIDE) & POOL_MASK
        draw = tl.load(pool_ptr + index, mask=in_gen, other=1.0)
        accepted = tl.minimum(base_total + (draw < frac).to(tl.int32), max_total)
    tl.store(num_accepted_ptr + num_contexts + gens, accepted.to(tl.int32), mask=in_gen)


class DflashAccept(OpWrapper):
    """Strict acceptance of a uniform verify block, with the forced-acceptance override.

    ``target_tokens`` are the sampled target tokens: one per context request,
    then ``K + 1`` per generation request. ``draft_tokens`` are the
    generation requests' ``K`` drafts. Returns ``(accepted_tokens,
    num_accepted_tokens)``, int32 ``[batch, K + 1]`` and ``[batch]``:

    * context row ``b``: ``accepted_tokens[b, 0] = target_tokens[b]``, count 1
      (the rest of the row is unspecified, as in the built-in);
    * generation row: the request's ``K + 1`` target tokens, and a count of 1 +
      the number of leading drafts equal to the target token at the same
      position -- unless ``force_mode`` overrides the count:
      ``FORCE_FIXED`` sets it to ``base_total``; ``FORCE_FRACTIONAL`` first
      bumps ``counter`` by one and then sets it to ``min(base_total +
      (pool[(counter * 6007 + g * 1009) & (len(pool) - 1)] < frac),
      max_total)`` for the ``g``-th generation request, with ``frac`` compared
      in fp32.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the target's decode step: one request, K = 7, forced 3.05 (fractional)",
            spec=dict(contexts=0, gens=1, k=7, force=FORCE_FRACTIONAL, rate=3.05),
        ),
        Cell(
            why="strict acceptance, no override: leading-match count",
            spec=dict(contexts=0, gens=1, k=7, force=FORCE_NONE, rate=0.0),
        ),
        Cell(
            why="integer override: every request gets base_total",
            spec=dict(contexts=0, gens=3, k=5, force=FORCE_FIXED, rate=2.0),
        ),
        Cell(
            why="fractional override, several requests: per-request pool slots",
            spec=dict(contexts=0, gens=4, k=3, force=FORCE_FRACTIONAL, rate=1.6),
        ),
        Cell(
            why="mixed batch: context rows take one token, gens are strict",
            spec=dict(contexts=2, gens=2, k=7, force=FORCE_NONE, rate=0.0),
        ),
        Cell(
            why="mixed batch under the fractional override",
            spec=dict(contexts=1, gens=2, k=7, force=FORCE_FRACTIONAL, rate=3.05),
        ),
        Cell(
            why="K = 1: one draft per request",
            spec=dict(contexts=0, gens=2, k=1, force=FORCE_NONE, rate=0.0),
        ),
        Cell(
            why="int64 target tokens (the greedy argmax path)",
            spec=dict(contexts=0, gens=2, k=7, force=FORCE_NONE, rate=0.0, int64=True),
        ),
        Cell(
            why="64 requests: the loop kernel's largest batch, forced 3.05",
            spec=dict(contexts=0, gens=64, k=7, force=FORCE_FRACTIONAL, rate=3.05),
        ),
        Cell(
            why="65 requests: the tile kernel's smallest batch, strict acceptance",
            spec=dict(contexts=0, gens=65, k=7, force=FORCE_NONE, rate=0.0),
        ),
        Cell(
            why="96 requests on the tile kernel, forced 3.05 (fractional)",
            spec=dict(contexts=0, gens=96, k=7, force=FORCE_FRACTIONAL, rate=3.05),
        ),
        Cell(
            why="128 requests on the tile kernel, integer override",
            spec=dict(contexts=0, gens=128, k=7, force=FORCE_FIXED, rate=2.0),
        ),
        Cell(
            why="128 requests on the tile kernel, int64 tokens, strict",
            spec=dict(contexts=0, gens=128, k=7, force=FORCE_NONE, rate=0.0, int64=True),
        ),
    )

    note = """
    Metadata consumed: none. One Triton program (grid 1), JIT-compiled per (K,
    force mode): up to ``LOOP_REQUESTS`` requests a loop over them (1 warp),
    above that one ``[MAX_REQUESTS, K + 1]`` tile (4 warps; the same integer
    arithmetic per request). Writes ``counter`` in place under
    ``FORCE_FRACTIONAL``.

    Call site: the DFlash worker's ``_sample_and_accept_draft_tokens_base``
    when the ``fused_accept`` fast path is selected (``spec_worker.py``), after the
    built-in target-token sampler, on uniform (non-ragged) verify blocks.

    Numerics: integer results, bitwise the built-in chain; the one float
    operation is ``pool[i] < frac`` in fp32, as torch compares an fp32 tensor
    with a Python float.

    Hard limits
      K <= 63 (one block of K + 1 columns); the pool length is a power of two.

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        force_mode: int,
        base_total: int,
        max_total: int,
        frac: float,
        pool: Optional[torch.Tensor],
        counter: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_gens, k = draft_tokens.shape
        batch = num_contexts + num_gens
        device = target_tokens.device
        accepted = torch.empty((batch, k + 1), dtype=torch.int32, device=device)
        num_accepted = torch.empty((batch,), dtype=torch.int32, device=device)
        if batch == 0:
            return accepted, num_accepted
        fractional = force_mode == FORCE_FRACTIONAL
        if batch > LOOP_REQUESTS:
            _dflash_accept_tile_kernel[(1,)](
                target_tokens,
                draft_tokens,
                draft_tokens.stride(0),
                accepted,
                num_accepted,
                pool if fractional else num_accepted,
                counter if fractional else num_accepted,
                num_contexts,
                num_gens,
                base_total,
                max_total,
                frac,
                K=k,
                BLOCK=triton.next_power_of_2(k + 1),
                GENS=MAX_REQUESTS,
                FORCE=force_mode,
                COUNTER_STRIDE=COUNTER_STRIDE,
                SLOT_STRIDE=SLOT_STRIDE,
                POOL_MASK=(pool.numel() - 1) if fractional else 0,
                num_warps=4,
            )
            return accepted, num_accepted
        _dflash_accept_kernel[(1,)](
            target_tokens,
            draft_tokens,
            draft_tokens.stride(0),
            accepted,
            num_accepted,
            pool if fractional else num_accepted,
            counter if fractional else num_accepted,
            num_contexts,
            num_gens,
            base_total,
            max_total,
            frac,
            K=k,
            BLOCK=triton.next_power_of_2(k + 1),
            FORCE=force_mode,
            COUNTER_STRIDE=COUNTER_STRIDE,
            SLOT_STRIDE=SLOT_STRIDE,
            POOL_MASK=(pool.numel() - 1) if fractional else 0,
            num_warps=1,
        )
        return accepted, num_accepted

    def reference(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        force_mode: int,
        base_total: int,
        max_total: int,
        frac: float,
        pool: Optional[torch.Tensor],
        counter: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The built-in torch chain, unrolled; bumps ``counter`` in place like the op.

        Context rows' columns past 0 are zero here (unspecified in the op).
        """
        num_gens, k = draft_tokens.shape
        batch = num_contexts + num_gens
        device = target_tokens.device
        accepted = torch.zeros((batch, k + 1), dtype=torch.int32, device=device)
        num_accepted = torch.ones(batch, dtype=torch.int32, device=device)
        accepted[:num_contexts, 0] = target_tokens[:num_contexts]
        gen_targets = target_tokens[num_contexts:].reshape(num_gens, k + 1)
        num_accepted[num_contexts:] += torch.cumprod(
            (draft_tokens == gen_targets[:, :k]).int(), dim=-1
        ).sum(1)
        accepted[num_contexts:, : k + 1] = gen_targets
        if force_mode == FORCE_FIXED:
            num_accepted[num_contexts:] = base_total
        elif force_mode == FORCE_FRACTIONAL:
            counter += 1
            slots = torch.arange(num_gens, device=device, dtype=torch.int64)
            index = (counter * COUNTER_STRIDE + slots * SLOT_STRIDE) & (pool.numel() - 1)
            extra = (pool[index] < frac).to(num_accepted.dtype)
            num_accepted[num_contexts:] = (base_total + extra).clamp_(max=max_total)
        return accepted, num_accepted

    def is_valid(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        force_mode: int,
        base_total: int,
        max_total: int,
        frac: float,
        pool: Optional[torch.Tensor],
        counter: Optional[torch.Tensor],
    ) -> None:
        assert draft_tokens.dim() == 2 and draft_tokens.stride(-1) == 1, (
            f"draft_tokens must be [gens, K] with unit column stride, got "
            f"{tuple(draft_tokens.shape)} stride {draft_tokens.stride()}"
        )
        num_gens, k = draft_tokens.shape
        assert 1 <= k <= 63, f"K must be in [1, 63], got {k}"
        assert target_tokens.dim() == 1 and target_tokens.is_contiguous(), (
            "target_tokens must be a contiguous 1-D tensor"
        )
        expected = num_contexts + num_gens * (k + 1)
        assert target_tokens.numel() == expected, (
            f"target_tokens must hold {expected} tokens (contexts + gens * (K + 1)), "
            f"got {target_tokens.numel()}"
        )
        assert num_contexts + num_gens <= MAX_REQUESTS, (
            f"at most {MAX_REQUESTS} requests, got {num_contexts + num_gens}"
        )
        assert not target_tokens.dtype.is_floating_point, "target_tokens must be integers"
        assert not draft_tokens.dtype.is_floating_point, "draft_tokens must be integers"
        assert force_mode in (FORCE_NONE, FORCE_FIXED, FORCE_FRACTIONAL), (
            f"unknown force_mode {force_mode}"
        )
        if force_mode == FORCE_FRACTIONAL:
            assert pool is not None and pool.dtype == torch.float32 and pool.dim() == 1, (
                "the fractional override needs the fp32 pool"
            )
            assert pool.numel() & (pool.numel() - 1) == 0, "the pool length must be a power of two"
            assert counter is not None and counter.dtype == torch.int64 and counter.numel() == 1, (
                "the fractional override needs the int64 [1] counter"
            )


dflash_accept = DflashAccept()
