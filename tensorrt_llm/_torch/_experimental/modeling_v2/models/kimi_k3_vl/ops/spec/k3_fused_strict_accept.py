# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict acceptance of a uniform-width draft against the sampled target tokens, with the
forced-acceptance override, in one Triton launch (one program)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import fused_strict_accept
from ._reference import strict_accept


class K3FusedStrictAccept(OpWrapper):
    """``(accepted [batch, K + 1], num_accepted [batch])``, both int32 and new.

    ``target_tokens`` is ``[num_contexts + num_gens * (K + 1)]`` of any integer dtype,
    ``draft_tokens`` ``[num_gens, K]`` with a unit inner stride. A context row accepts its one
    target token (column 0 only is defined); a generation row takes its ``K + 1`` target tokens
    and accepts ``1 +`` the number of leading draft tokens equal to them. ``force``: None,
    ``(base_total,)`` (every generation row accepts ``base_total``) or ``(base_total, frac,
    max_total, pool, counter, counter_stride, slot_stride)``, which advances ``counter`` by one and
    adds 1 where the pool draw is below ``frac``, capped at ``max_total``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="dep16 (c383): 16 generation rows, DSpark 5",
            spec=dict(num_contexts=0, num_gens=16, draft_len=5, force=0.0),
        ),
        Cell(
            why="helix8 c240: 20 generation rows of DSpark 5, past the 16 the selection kernel takes",
            spec=dict(num_contexts=0, num_gens=20, draft_len=5, force=0.0),
        ),
        Cell(
            why="dep16 shape with 3 context rows ahead of 13 generation rows: column 0 only",
            spec=dict(num_contexts=3, num_gens=13, draft_len=5, force=0.0),
        ),
        Cell(
            why="dep16 under a whole forced acceptance (3.0): every row accepts 4",
            spec=dict(num_contexts=0, num_gens=16, draft_len=5, force=3.0),
        ),
        Cell(
            why="helix8 c240 under a fractional forced acceptance (2.5): the pool draw and the "
            "device counter",
            spec=dict(num_contexts=0, num_gens=20, draft_len=5, force=2.5),
        ),
    )

    note = """
    One program of 4 warps covers the whole batch (block size the next power of two of the larger
    of the context and generation counts). The values are those of
    ``SpecWorkerBase._sample_and_accept_draft_tokens_base`` with the same force draws. No host
    synchronization; the launch shape depends on host batch sizes only.
    """

    def __call__(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_strict_accept(target_tokens, draft_tokens, num_contexts, batch_size, force)

    def reference(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The op's outputs; context rows hold 0 past column 0, which the op leaves undefined.
        The fractional ``force`` reads ``counter + 1`` without writing ``counter``."""
        return strict_accept(target_tokens, draft_tokens, num_contexts, batch_size, force)

    def is_valid(
        self,
        target_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
    ) -> None:
        num_gens = batch_size - num_contexts
        k = draft_tokens.shape[-1]
        # The kernel reads target rows at (K + 1) strides past the contexts: a shorter vector is
        # read out of bounds instead of rejected.
        assert target_tokens.numel() >= num_contexts + num_gens * (k + 1), "target too short"
        assert draft_tokens.shape[0] >= num_gens, "one draft row per generation request"
        if force is not None and len(force) > 1:
            pool = force[3]
            # The draw masks the pool index with numel - 1.
            assert pool.numel() & (pool.numel() - 1) == 0, "the force pool must be a power of two"


k3_fused_strict_accept = K3FusedStrictAccept()
