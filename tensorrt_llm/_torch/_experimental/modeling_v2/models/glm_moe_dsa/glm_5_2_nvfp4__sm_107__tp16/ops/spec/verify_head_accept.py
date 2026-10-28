# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict draft acceptance with the forced accepted-token count in one kernel, via the glm_verify_head_accept op."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class GlmVerifyHeadAccept(OpWrapper):
    """The accepted-token counts of one verify step, and optionally its accepted-token rows.

    A context request accepts its one sampled token (count 1). A generation request with sampled target tokens
    `t[0..K]` and draft tokens `d[0..K-1]` accepts `1 + (length of the leading run of d[j] == t[j])`, then the
    forced count replaces that as `force_mode` says:
    - 0: no force;
    - 1: `force_base`;
    - 2: `min(force_base + (force_pool[i] < force_frac), force_max)` with
      `i = (c * counter_stride + g * slot_stride) & (pool_size - 1)`, `g` the generation index and `c` the value
      of `force_counter` after this call adds one to it (the comparison in FP32).

    That is `SpecWorkerBase._sample_and_accept_draft_tokens_base` after sampling with
    `_apply_force_accepted_tokens`, for the padded layout (`target_tokens` holds one token per context request,
    then `K + 1` per generation request). Returns the counts `[num_contexts + num_gens]` int32. With
    `accepted_tokens` the op also writes the accepted-token rows: column 0 of each context row, the whole row of
    each generation request (its target tokens); without it the caller reads the generation rows from
    `target_tokens` itself.
    """

    ARCHS = frozenset({Arch.SM_107})

    # spec: num_contexts, num_gens, draft_len (K = 7 at c1), target dtype, matches (the leading draft tokens equal
    # to the targets, per generation request), force (mode, base, max, frac), rows (accepted_tokens given). The
    # hash constants are the spec worker's: counter stride 6007, slot stride 1009, pool 2^16.
    # At c1 the gen server verifies one request with K = 7 and the cell config forces 2.91 accepted draft tokens:
    # mode 2, base 3, max 8, frac 0.91; the accepted rows alias the sampled tokens.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 verify step: one request, K 7, forced 2.91 (base 3, frac 0.91)",
            spec=dict(
                num_contexts=0,
                num_gens=1,
                draft_len=7,
                target_dtype=torch.int32,
                matches=(2,),
                force=(2, 3, 8, 0.91),
                rows=False,
            ),
        ),
        Cell(
            why="c1 verify step without a forced count: the strict count",
            spec=dict(
                num_contexts=0,
                num_gens=1,
                draft_len=7,
                target_dtype=torch.int32,
                matches=(5,),
                force=(0, 0, 0, 0.0),
                rows=False,
            ),
        ),
        Cell(
            why="every draft token accepted: the count reaches K + 1",
            spec=dict(
                num_contexts=0,
                num_gens=1,
                draft_len=7,
                target_dtype=torch.int32,
                matches=(7,),
                force=(0, 0, 0, 0.0),
                rows=True,
            ),
        ),
        Cell(
            why="an integral forced count (3.0): mode 1, every request gets base 4",
            spec=dict(
                num_contexts=0,
                num_gens=1,
                draft_len=7,
                target_dtype=torch.int32,
                matches=(0,),
                force=(1, 4, 8, 0.0),
                rows=False,
            ),
        ),
        Cell(
            why="warm-up batch with a context request, int64 targets, rows written",
            spec=dict(
                num_contexts=1,
                num_gens=1,
                draft_len=7,
                target_dtype=torch.int64,
                matches=(3,),
                force=(2, 3, 8, 0.91),
                rows=True,
            ),
        ),
        Cell(
            why="several generation requests: the pool index follows the slot",
            spec=dict(
                num_contexts=0,
                num_gens=4,
                draft_len=7,
                target_dtype=torch.int32,
                matches=(0, 7, 2, 4),
                force=(2, 3, 8, 0.5),
                rows=True,
            ),
        ),
    )

    note = """
    Not for ragged verification: a request that verified fewer than K + 1 positions has no padded row, and the op
    would read its neighbour's tokens. The caller keeps the ATen chain there.

    The kernel waits for its producer (the sampler) before it reads anything and lets its dependents launch at its
    end. Mode 2 advances `force_counter` once per call, also inside a captured graph's replays, exactly as the ATen
    chain's in-place add does, so the op and the chain draw the same pool entries given the same counter.
    """

    def __call__(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        accepted_tokens: Optional[torch.Tensor],
        force_pool: Optional[torch.Tensor],
        force_counter: Optional[torch.Tensor],
        force_mode: int,
        force_base: int,
        force_max: int,
        force_frac: float,
        counter_stride: int,
        slot_stride: int,
        pool_size: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.glm_verify_head_accept(
            target_tokens,
            draft_tokens,
            num_contexts,
            accepted_tokens,
            force_pool,
            force_counter,
            force_mode,
            force_base,
            force_max,
            force_frac,
            counter_stride,
            slot_stride,
            pool_size,
        )

    def reference(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        accepted_tokens: Optional[torch.Tensor],
        force_pool: Optional[torch.Tensor],
        force_counter: Optional[torch.Tensor],
        force_mode: int,
        force_base: int,
        force_max: int,
        force_frac: float,
        counter_stride: int,
        slot_stride: int,
        pool_size: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """The ATen chain of the one-model spec worker: `(counts, rows, counter)`.

        `rows` is what `accepted_tokens` holds after the call (its unwritten entries keep the values it was handed)
        and `counter` what `force_counter` holds after it; each is None when the op is not handed that buffer
        (or, for the counter, does not advance it). Both buffers are read as they were before the call.
        """
        num_gens, draft_len = draft_tokens.shape
        batch_size = num_contexts + num_gens
        target = target_tokens.reshape(-1)
        counts = torch.ones(batch_size, dtype=torch.int32, device=draft_tokens.device)
        gen_target = target[num_contexts : num_contexts + num_gens * (draft_len + 1)].reshape(
            num_gens, draft_len + 1
        )
        counts[num_contexts:] += (
            torch.cumprod((draft_tokens.long() == gen_target[:, :draft_len].long()).int(), dim=-1)
            .sum(1)
            .int()
        )
        counter = None
        if force_mode == 1:
            counts[num_contexts:] = force_base
        elif force_mode == 2:
            counter = force_counter + 1
            slots = torch.arange(num_gens, dtype=torch.int64, device=draft_tokens.device)
            index = (counter * counter_stride + slots * slot_stride) & (pool_size - 1)
            extra = (force_pool[index] < force_frac).int()
            counts[num_contexts:] = (force_base + extra).clamp(max=force_max).int()
        rows = None
        if accepted_tokens is not None:
            rows = accepted_tokens.clone()
            rows[:num_contexts, 0] = target[:num_contexts].int()
            rows[num_contexts:batch_size] = gen_target.int()
        return counts, rows, counter

    def is_valid(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        accepted_tokens: Optional[torch.Tensor],
        force_pool: Optional[torch.Tensor],
        force_counter: Optional[torch.Tensor],
        force_mode: int,
        force_base: int,
        force_max: int,
        force_frac: float,
        counter_stride: int,
        slot_stride: int,
        pool_size: int,
    ) -> None:
        num_gens, draft_len = draft_tokens.shape
        # The kernel writes row r at offset r * (K + 1): a wider buffer (rows of the maximum draft length while
        # this step verifies fewer) is accepted by the op's size check and filled with the wrong row stride.
        assert accepted_tokens is None or tuple(accepted_tokens.shape) == (
            num_contexts + num_gens,
            draft_len + 1,
        ), "accepted_tokens must be exactly [num_contexts + num_gens, draft_len + 1]"
        # A forced count outside [1, K + 1] is written as given and sends the caller past its draft window.
        if force_mode != 0:
            assert 1 <= force_base <= force_max == draft_len + 1, (
                "a forced count needs 1 <= force_base <= force_max == draft_len + 1"
            )


glm_verify_head_accept = GlmVerifyHeadAccept()
