# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the helix CP ranks' local Top-K candidates into this rank's share of the global Top-K.

The helix8 target's fallback Top-K, for a decode step the key-sharded merge cannot serve (a ragged
or bound-less verify window): each rank takes its exact local Top-K, the candidates' scores are
all-gathered over the CP group (``allgather_module``), and this entry keeps the rank's entries that
make the global Top-K. The engine's ``merge_helix_topk`` without its all-gather, in plain torch.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class RubinMergeHelixTopk(OpWrapper):
    """Keep in ``topk_rows`` (int32 ``[T, k]``, this rank's local Top-K rows, -1 for none) only the
    entries that make each row's global Top-K, kept entries first in their original order, the
    rest -1; returns ``topk_rows``.

    ``gathered`` FP32 ``[cp, T, k]``: every rank's candidate scores in rank order (-inf where its
    row holds -1). The global Top-K is ``torch.topk`` over the ``cp * k`` candidates of a row,
    ordered by (rank, slot); a candidate is this rank's when it lies in segment ``cp_rank``.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 GEN b16 k5, fallback: 96 rows, 8 ranks, Top-2048 each",
            spec=dict(cp=8, rows=96, k=2048, fill=1.0),
        ),
        Cell(
            why="sparse candidates: most local rows padded with -1",
            spec=dict(cp=8, rows=12, k=2048, fill=0.1),
        ),
        Cell(why="two ranks, one row", spec=dict(cp=2, rows=1, k=2048, fill=0.5)),
    )
    note = """
    The ops of the engine's ``merge_helix_topk`` after its CP all-gather, in its order: the same
    values. Eager steps only (the targets reach it when the key-sharded merge cannot serve a step,
    which a CUDA graph never captures).
    """

    def __call__(
        self, gathered: torch.Tensor, topk_rows: torch.Tensor, cp_rank: int
    ) -> torch.Tensor:
        num_tokens, k = topk_rows.shape
        if num_tokens == 0:
            return topk_rows
        all_scores = gathered.transpose(0, 1).reshape(num_tokens, -1)
        top_scores, top_pos = torch.topk(all_scores, k, dim=1)
        owner = torch.div(top_pos, k, rounding_mode="floor")
        mine = (owner == cp_rank) & torch.isfinite(top_scores)
        hits = torch.zeros_like(topk_rows, dtype=torch.int32)
        hits.scatter_add_(1, torch.where(mine, top_pos - owner * k, 0), mine.to(torch.int32))
        keep = hits > 0
        # Stable compaction: kept entries first, in their original order.
        order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices
        compacted = torch.gather(topk_rows.masked_fill(~keep, -1), 1, order)
        topk_rows.copy_(compacted)
        return topk_rows

    def reference(
        self, gathered: torch.Tensor, topk_rows: torch.Tensor, cp_rank: int
    ) -> torch.Tensor:
        """A new tensor: per row, the set of this rank's candidates among the row's ``k`` best by
        (score desc, rank, slot), kept first in their original order, then -1; ``topk_rows`` is
        left untouched."""
        cp, rows, k = gathered.shape
        out = torch.full_like(topk_rows, -1)
        for r in range(rows):
            scores = gathered[:, r, :].reshape(-1).double()
            order = torch.sort(-scores, stable=True).indices[:k]
            chosen = order[torch.isfinite(scores[order])]
            slots = torch.sort(chosen[chosen // k == cp_rank] - cp_rank * k).values
            kept = topk_rows[r, slots]
            out[r, : kept.numel()] = kept
        return out

    def is_valid(self, gathered: torch.Tensor, topk_rows: torch.Tensor, cp_rank: int) -> None:
        # A row's candidates are read as cp segments of k: another width splits them wrongly.
        assert gathered.shape[1:] == topk_rows.shape, "gathered must be [cp, T, k] over topk_rows"
        assert 0 <= cp_rank < gathered.shape[0]
        # torch.topk orders equal scores by position on CPU, not on CUDA: a tie at the boundary
        # may keep either candidate (the engine's merge has the same freedom).


rubin_merge_helix_topk = RubinMergeHelixTopk()
