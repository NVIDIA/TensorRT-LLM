# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 proposals a verification step reads back from the store."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_proposal_load import TOP_K, proposal_load
from .dflash2_sampling import DFlash2ProposalStore


def _spec(n, slots, *, walked="mixed", ids=torch.int64, rows=None) -> dict:
    return dict(n=n, slots=slots, walked=walked, ids=ids, rows=rows or 2 * n + 1)


class K31DFlash2ProposalLoad(OpWrapper):
    """``DFlash2ProposalStore.load``: each verified request's candidate ids and proposal
    distribution, or the point mass on its verified tokens where the store row did not draft them.

    Args:
        store_candidate_ids, store_q, store_tokens: the store's ``[rows, K_max, 32]`` int64,
            ``[rows, K_max, 32]`` FP32 and ``[rows, K_max]`` int64 tensors.
        slot_ids: ``[n]`` store rows of the requests (int32 or int64).
        draft_tokens: ``[n, K]`` the tokens being verified (int32 or int64), ``K <= K_max``.
    Returns:
        ``(candidate_ids [n, K, 32] int64, q [n, K, 32] FP32)``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, K = 7, walked", spec=_spec(1, 7, walked="all")),
        Cell(why="one request, K = 4, not walked (point mass)", spec=_spec(1, 4, walked="none")),
        Cell(why="8 requests, K = 7, walked and not", spec=_spec(8, 7)),
        Cell(why="64 requests, K = 7, int32 slots and drafts", spec=_spec(64, 7, ids=torch.int32)),
        Cell(why="64 requests, K = 4 of a K_max = 7 store", spec=_spec(64, 4)),
        Cell(why="17 requests, K = 1", spec=_spec(17, 1)),
        Cell(why="sm_107 DEP16 generation: 12 requests a rank, K = 7", spec=_spec(12, 7)),
        Cell(why="sm_107 Helix CP 8 generation: 16 requests, K = 5", spec=_spec(16, 5)),
    )

    note = """
    Kernel: ``kernels/spec/dflash2_proposal_load.py``, one launch (the torch method is about ten),
    one warp per request. Pure copies and comparisons: bitwise. ``store`` is the other half and
    lives in ``k31_dflash2_selector_draft``'s walk epilogue.
    """

    def __call__(
        self,
        store_candidate_ids: torch.Tensor,
        store_q: torch.Tensor,
        store_tokens: torch.Tensor,
        slot_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return proposal_load(store_candidate_ids, store_q, store_tokens, slot_ids, draft_tokens)

    def reference(
        self,
        store_candidate_ids: torch.Tensor,
        store_q: torch.Tensor,
        store_tokens: torch.Tensor,
        slot_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``DFlash2ProposalStore.load`` on a store holding these tensors."""
        store = SimpleNamespace(
            top_k=store_candidate_ids.shape[-1],
            candidate_ids=store_candidate_ids,
            q=store_q,
            tokens=store_tokens,
        )
        return DFlash2ProposalStore.load(store, slot_ids, draft_tokens)

    def is_valid(
        self,
        store_candidate_ids: torch.Tensor,
        store_q: torch.Tensor,
        store_tokens: torch.Tensor,
        slot_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        assert store_candidate_ids.shape[-1] == TOP_K, f"the store holds {TOP_K} candidates a slot"
        assert draft_tokens.shape[1] <= min(32, store_tokens.shape[1]), "K exceeds the store"
        for t in (store_candidate_ids, store_q, store_tokens, slot_ids, draft_tokens):
            assert t.is_contiguous(), "the kernel reads contiguous rows"


k31_dflash2_proposal_load = K31DFlash2ProposalLoad()

__all__ = ["K31DFlash2ProposalLoad", "k31_dflash2_proposal_load"]
