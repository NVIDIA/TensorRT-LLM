# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 selector's greedy path into the block logits (``select_candidate_path``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_selector_walk import TOP_K, selector_greedy_path, selector_operands
from .dflash2_sampling import dflash2_score_edges, dflash2_walk_candidate_paths


def _spec(n, slots, *, unary=torch.bfloat16, block=torch.bfloat16, scores="randn") -> dict:
    return dict(n=n, slots=slots, unary=unary, block=block, scores=scores)


class K31DFlash2SelectorGreedyPath(OpWrapper):
    """The drafter's ``select_candidate_path``: scores every adjacent candidate pair with the
    low-rank selector, walks the lattice greedily (``dflash2_walk_candidate_paths``) and rewrites
    ``block_logits`` to ``-inf`` except at each slot's candidates, which get the slot's score row
    under the predecessor the walk took, in ``block_logits``' dtype.

    Args:
        predecessor_codebook, successor_codebook: ``[vocab, rank]``.
        hidden_projection_weight: ``[rank, hidden]`` (``candidate_selector.hidden_projection``).
        candidate_ids: ``[n, K, 32]`` int64 global candidate ids (distinct within a slot).
        unary_logits: ``[n, K, 32]`` BF16 / FP32.
        block_hidden_states: ``[n, K, hidden]``.
        anchor_token_ids: ``[n]`` int32 / int64.
        block_logits: ``[n, K, vocab]`` BF16 / FP32 destination, rewritten in place.
    Returns:
        ``block_logits``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, K = 7", spec=_spec(1, 7)),
        Cell(why="8 requests, K = 4, FP32 unary logits", spec=_spec(8, 4, unary=torch.float32)),
        Cell(why="64 requests, K = 7", spec=_spec(64, 7)),
        Cell(why="8 requests, K = 7, FP32 block logits", spec=_spec(8, 7, block=torch.float32)),
        Cell(
            why="5 requests, K = 7, tied scores: the walk takes the first maximum",
            spec=_spec(5, 7, scores="ties"),
        ),
        Cell(
            why="sm_107 TEP8 generation: one request, K = 7, FP32 unary and block logits",
            spec=_spec(1, 7, unary=torch.float32, block=torch.float32),
        ),
        Cell(
            why="sm_107 DEP16 generation: 12 requests a rank, K = 7, FP32 logits",
            spec=_spec(12, 7, unary=torch.float32, block=torch.float32),
        ),
        Cell(
            why="sm_107 Helix CP 8 generation: 16 requests, K = 5, FP32 logits",
            spec=_spec(16, 5, unary=torch.float32, block=torch.float32),
        ),
    )

    note = """
    Kernels: ``kernels/spec/dflash2_selector_walk.py`` (the operands kernel, torch's einsum, a
    ``fill_`` and the walk in its ``realized`` mode, which scatters the rows itself): four
    launches instead of about 45. Bitwise: the einsum is torch's own call, the walk's argmax is
    torch's (first maximum, NaN above everything), and the scattered values are the same FP32 rows
    rounded once to the destination dtype.
    """

    def __call__(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        hidden_projection_weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        block_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        block_logits: torch.Tensor,
    ) -> torch.Tensor:
        weight = hidden_projection_weight
        gate = F.linear(block_hidden_states.to(weight.dtype), weight)
        pred, succ = selector_operands(
            predecessor_codebook, successor_codebook, candidate_ids, gate, anchor_token_ids
        )
        bilinear = torch.einsum("blpr,blcr->blpc", pred, succ)
        return selector_greedy_path(
            bilinear, candidate_ids, block_logits, unary_logits=unary_logits
        )

    def reference(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        hidden_projection_weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        block_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        block_logits: torch.Tensor,
    ) -> torch.Tensor:
        """The selector's edge scores (its hidden projection of ``block_hidden_states``), the greedy walk, and
        ``block_logits`` set to ``-inf`` except at the candidates, which take the realized scores."""
        gate = F.linear(
            block_hidden_states.to(hidden_projection_weight.dtype), hidden_projection_weight
        )
        edge_scores = dflash2_score_edges(
            predecessor_codebook,
            successor_codebook,
            candidate_ids,
            unary_logits,
            gate,
            anchor_token_ids.long(),
        )
        realized_scores = dflash2_walk_candidate_paths(candidate_ids, edge_scores)
        block_logits.fill_(float("-inf"))
        block_logits.scatter_(-1, candidate_ids, realized_scores.to(block_logits.dtype))
        return block_logits

    def is_valid(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        hidden_projection_weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        block_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        block_logits: torch.Tensor,
    ) -> None:
        assert candidate_ids.shape[-1] == TOP_K, f"top_k must be {TOP_K}"
        assert predecessor_codebook.shape[1] % 256 == 0, (
            "the selector rank must be a multiple of 256"
        )
        assert block_logits.dtype in (torch.bfloat16, torch.float32), "BF16 or FP32 block logits"
        for t in (candidate_ids, unary_logits, block_logits):
            assert t.is_contiguous(), "the kernels read and write contiguous rows"
        vocab = block_logits.shape[-1]
        assert int(candidate_ids.min()) >= 0 and int(candidate_ids.max()) < vocab, "id out of range"
        # A repeated id would be scattered twice; torch's scatter_ keeps an unspecified one.
        sorted_ids = candidate_ids.sort(dim=-1).values
        assert bool((sorted_ids[..., 1:] != sorted_ids[..., :-1]).all()), "repeated candidate id"


k31_dflash2_selector_greedy_path = K31DFlash2SelectorGreedyPath()


class K31DFlash2CandidateSelector(nn.Module):
    """Parameter holder + op call; not a layer.

    The DFlash2 pairwise path selector over a drafted block's top-k candidates: the predecessor / successor
    codebooks and the projection of each slot's hidden state that gates their low-rank match
    (``dflash2_score_edges``)."""

    def __init__(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        hidden_projection_weight: torch.Tensor,
        top_k: int,
        vocab_size: int,
        rank: int,
    ):
        super().__init__()
        for name, table in (
            ("predecessor_codebook", predecessor_codebook),
            ("successor_codebook", successor_codebook),
        ):
            if tuple(table.shape) != (vocab_size, rank):
                raise ValueError(
                    f"DFlash2 {name} has shape {tuple(table.shape)}, expected [vocab_size, selector_rank] = "
                    f"({vocab_size}, {rank})."
                )
        if hidden_projection_weight.shape[0] != rank:
            raise ValueError(
                f"DFlash2 candidate_selector.hidden_projection has shape {tuple(hidden_projection_weight.shape)}, "
                f"expected selector_rank={rank} output features."
            )
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(predecessor_codebook, requires_grad=False)
        self.successor_codebook = nn.Parameter(successor_codebook, requires_grad=False)
        self.hidden_projection = nn.Linear(
            hidden_projection_weight.shape[1],
            rank,
            bias=False,
            device=hidden_projection_weight.device,
            dtype=hidden_projection_weight.dtype,
        )
        self.hidden_projection.weight.data.copy_(hidden_projection_weight)

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, K, top_k, top_k]`` FP32 scores of every adjacent candidate pair (``dflash2_score_edges``)."""
        gate = self.hidden_projection(hidden_states.to(self.hidden_projection.weight.dtype))
        return dflash2_score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            gate,
            anchor_token_ids,
        )


__all__ = [
    "K31DFlash2CandidateSelector",
    "K31DFlash2SelectorGreedyPath",
    "k31_dflash2_selector_greedy_path",
]
