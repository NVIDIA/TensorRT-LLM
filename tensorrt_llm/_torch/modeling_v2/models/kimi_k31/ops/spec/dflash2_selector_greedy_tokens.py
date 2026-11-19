# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 selector's greedy draft tokens: the argmax of the greedy path's block logits,
without the block."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_selector_walk import TOP_K, selector_greedy_tokens, selector_operands
from .dflash2_selector_greedy_path import k31_dflash2_selector_greedy_path


def _spec(n, slots, *, unary=torch.float32, block=torch.float32, scores="randn") -> dict:
    return dict(n=n, slots=slots, unary=unary, block=block, scores=scores)


class K31DFlash2SelectorGreedyTokens(OpWrapper):
    """``torch.argmax(select_candidate_path(block, ...), -1)`` for a ``block`` of ``block_dtype``
    filled with ``-inf``: the greedy (all-greedy draft) tokens of the DFlash2 selector, int32
    ``[n, K]``, without writing or reading the ``[n, K, vocab]`` block.

    Slot ``e``'s token is the candidate whose score under the predecessor the greedy walk took is
    largest once rounded to ``block_dtype``, the lowest id among equal ones (NaN above everything),
    or 0 when every score is ``-inf`` -- torch's argmax of that block row.

    Args:
        predecessor_codebook, successor_codebook: ``[vocab, rank]``.
        hidden_projection_weight: ``[rank, hidden]`` (``candidate_selector.hidden_projection``).
        candidate_ids: ``[n, K, 32]`` int64 global candidate ids (distinct within a slot).
        unary_logits: ``[n, K, 32]`` BF16 / FP32.
        block_hidden_states: ``[n, K, hidden]``.
        anchor_token_ids: ``[n]`` int32 / int64.
        block_dtype: the dtype of the block logits the tokens are read from (BF16 / FP32).
    Returns:
        ``[n, K]`` int32 draft tokens.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, K = 7, FP32 block: the ll block", spec=_spec(1, 7)),
        Cell(why="2 requests, K = 4, FP32 block: the mid block", spec=_spec(2, 4)),
        Cell(why="12 requests, K = 4, FP32 block: the M2 block", spec=_spec(12, 4)),
        Cell(why="24 requests, K = 4, FP32 block", spec=_spec(24, 4)),
        Cell(
            why="8 requests, K = 7, BF16 unary and block",
            spec=_spec(8, 7, unary=torch.bfloat16, block=torch.bfloat16),
        ),
        Cell(
            why="5 requests, K = 7, tied scores: the lowest id among equal maxima",
            spec=_spec(5, 7, scores="ties"),
        ),
        Cell(
            why="4 requests, K = 7, -inf and NaN scores: an all -inf row reads token 0",
            spec=_spec(4, 7, scores="special"),
        ),
    )

    note = """
    Kernels: ``kernels/spec/dflash2_selector_walk.py``: the operands kernel, torch's einsum, and the
    walk in its ``realized_tokens`` mode (the greedy walk of ``select_candidate_path``, each slot's
    token written in place of the scatter). Three launches where the block path takes four plus a
    ``[n, K, vocab]`` argmax. Bitwise: the walk is the block path's, and the token is torch's argmax
    of the block row it would have written.
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
        block_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = hidden_projection_weight
        gate = F.linear(block_hidden_states.to(weight.dtype), weight)
        pred, succ = selector_operands(
            predecessor_codebook, successor_codebook, candidate_ids, gate, anchor_token_ids
        )
        bilinear = torch.einsum("blpr,blcr->blpc", pred, succ)
        return selector_greedy_tokens(
            bilinear, candidate_ids, block_dtype, unary_logits=unary_logits
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
        block_dtype: torch.dtype,
    ) -> torch.Tensor:
        """The argmax of ``k31_dflash2_selector_greedy_path``'s reference block."""
        n, slots, _ = candidate_ids.shape
        block = torch.full(
            (n, slots, predecessor_codebook.shape[0]),
            float("-inf"),
            dtype=block_dtype,
            device=candidate_ids.device,
        )
        block = k31_dflash2_selector_greedy_path.reference(
            predecessor_codebook,
            successor_codebook,
            hidden_projection_weight,
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_token_ids,
            block,
        )
        return torch.argmax(block, dim=-1).to(torch.int32)

    def is_valid(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        hidden_projection_weight: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        block_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        block_dtype: torch.dtype,
    ) -> None:
        assert candidate_ids.shape[-1] == TOP_K, f"top_k must be {TOP_K}"
        assert predecessor_codebook.shape[1] % 256 == 0, (
            "the selector rank must be a multiple of 256"
        )
        assert block_dtype in (torch.bfloat16, torch.float32), "BF16 or FP32 block logits"
        for t in (candidate_ids, unary_logits):
            assert t.is_contiguous(), "the kernels read contiguous rows"
        vocab = predecessor_codebook.shape[0]
        assert int(candidate_ids.min()) >= 0 and int(candidate_ids.max()) < vocab, "id out of range"
        assert vocab < (1 << 31), "ids compare as int32"
        # The block path scatters a repeated id twice (torch's scatter_ keeps an unspecified one).
        sorted_ids = candidate_ids.sort(dim=-1).values
        assert bool((sorted_ids[..., 1:] != sorted_ids[..., :-1]).all()), "repeated candidate id"


k31_dflash2_selector_greedy_tokens = K31DFlash2SelectorGreedyTokens()

__all__ = ["K31DFlash2SelectorGreedyTokens", "k31_dflash2_selector_greedy_tokens"]
