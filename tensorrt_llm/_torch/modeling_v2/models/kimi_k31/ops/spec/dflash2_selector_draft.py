# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 sampled selector draft after the global top-k: scores, uniforms, walk and store."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_selector_walk import TOP_K, selector_operands, selector_walk
from .dflash2_sampling import (
    DFlash2ProposalStore,
    dflash2_score_edges,
    dflash2_selector_walk,
    dflash2_uniforms,
)


def _spec(
    n,
    slots,
    *,
    unary=torch.bfloat16,
    codebook=torch.bfloat16,
    temps="mixed",
    scores="randn",
    rows=None,
) -> dict:
    return dict(
        n=n,
        slots=slots,
        unary=unary,
        codebook=codebook,
        temps=temps,
        scores=scores,
        rows=rows or n + 1,
    )


class K31DFlash2SelectorDraft(OpWrapper):
    """Draft tokens of a DFlash2 sampling batch from each slot's global top-k candidates.

    Scores every adjacent candidate pair with the low-rank selector (``dflash2_score_edges``),
    draws ``K`` SplitMix64 uniforms per request from its ``(seed, offset)``
    (``dflash2_uniforms``), walks the lattice once (``dflash2_selector_walk``: a sampled slot by
    inverse CDF of ``softmax(row / max(T, 1e-5))``, a request with ``temperature <=
    greedy_threshold`` by argmax) and writes the proposals into the store rows ``slot_ids``
    (``DFlash2ProposalStore.store``).

    Args:
        predecessor_codebook, successor_codebook: ``[vocab, rank]``.
        candidate_ids: ``[n, K, 32]`` int64 global candidate ids.
        unary_logits: ``[n, K, 32]`` the drafter's logits of the candidates (BF16 or FP32).
        gate: ``[n, K, rank]`` ``hidden_projection`` of the slots' hidden states.
        anchor_token_ids: ``[n]`` the last accepted token per request (int32 or int64).
        seed, offset: ``[n]`` int64, the requests' Philox pair of the draft slot.
        temperatures: ``[n]`` FP32 request temperatures.
        greedy_threshold: temperatures at or below it decode greedily.
        store_candidate_ids, store_q, store_tokens: the ``DFlash2ProposalStore`` tensors
            ``[rows, K_max, 32]`` int64, ``[rows, K_max, 32]`` FP32, ``[rows, K_max]`` int64.
        slot_ids: ``[n]`` int64 store rows of the requests.
    Returns:
        ``[n, K]`` int32 draft tokens.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, K = 7: the sequential (CUB) cumsum row", spec=_spec(1, 7)),
        Cell(why="one request, K = 4", spec=_spec(1, 4)),
        Cell(why="8 requests, K = 7, mixed greedy and sampled rows", spec=_spec(8, 7)),
        Cell(
            why="8 requests, K = 4, FP32 unary logits (vocab-sharded head)",
            spec=_spec(8, 4, unary=torch.float32),
        ),
        Cell(why="64 requests, K = 7: the largest batch", spec=_spec(64, 7, rows=129)),
        Cell(
            why="64 requests, K = 4, FP32 unary logits",
            spec=_spec(64, 4, unary=torch.float32),
        ),
        Cell(
            why="17 requests, K = 7, all sampled, hot temperatures",
            spec=_spec(17, 7, temps="hot"),
        ),
        Cell(
            why="17 requests, K = 7, temperatures at the greedy and floor edges",
            spec=_spec(17, 7, temps="edges"),
        ),
        Cell(
            why="5 requests, K = 7, tied scores: argmax takes the first maximum",
            spec=_spec(5, 7, scores="ties"),
        ),
        Cell(why="3 requests, K = 1: a runtime draft length of one", spec=_spec(3, 1)),
        Cell(
            why="4 requests, K = 7, FP32 codebooks",
            spec=_spec(4, 7, codebook=torch.float32),
        ),
        Cell(
            why="sm_107 TEP8 generation: one request, K = 7, FP32 unary logits (TP8 merge)",
            spec=_spec(1, 7, unary=torch.float32),
        ),
        Cell(
            why="sm_107 DEP16 generation: 12 requests a rank, K = 7, FP32 unary logits",
            spec=_spec(12, 7, unary=torch.float32),
        ),
        Cell(
            why="sm_107 Helix CP 8 generation: 16 requests, K = 5, FP32 unary logits",
            spec=_spec(16, 5, unary=torch.float32),
        ),
    )

    note = """
    Kernels: ``kernels/spec/dflash2_selector_walk.py``. ``SelectorOperandsKernel`` writes the two
    FP32 einsum operands ``[n, K, 32, rank]``; the einsum is torch's (TF32 by default on this
    torch, cuBLAS rounding, so it is kept: the draws are bitwise only when the scores are);
    ``SelectorWalkKernel`` adds the unary logits, derives the uniforms, walks and writes the
    tokens and the store rows. Three launches for any batch.

    Bitwise: the walk reproduces torch's softmax (warp softmax), cumsum (Sklansky scan; a single
    request's row is scanned sequentially, as CUB does), inverse CDF, argmax and the SplitMix64
    stream, so tokens, q and the store rows equal the torch path's bit for bit on these cells
    (run under torch 2.14; a torch whose softmax or cumsum reduce in another order would show
    up as a failing cell). Duplicate ``slot_ids`` (the CUDA-graph padding row) are written by
    several requests; the store keeps one of them, as torch's index_put does.

    Shapes: top_k 32 (one lane per candidate), rank a multiple of 256, any K and n; fixed
    shapes per call, no host synchronization, compiled per (K, n == 1, dtypes) on first use,
    which must be an eager call before CUDA-graph capture.
    """

    def __call__(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        gate: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_threshold: float,
        store_candidate_ids: torch.Tensor,
        store_q: torch.Tensor,
        store_tokens: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        pred, succ = selector_operands(
            predecessor_codebook,
            successor_codebook,
            candidate_ids,
            gate,
            anchor_token_ids,
        )
        bilinear = torch.einsum("blpr,blcr->blpc", pred, succ)
        return selector_walk(
            bilinear,
            candidate_ids,
            seed,
            offset,
            temperatures,
            unary_logits=unary_logits,
            greedy_threshold=greedy_threshold,
            store=(store_candidate_ids, store_q, store_tokens),
            slot_ids=slot_ids,
        )

    def reference(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        gate: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_threshold: float,
        store_candidate_ids: torch.Tensor,
        store_q: torch.Tensor,
        store_tokens: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        """The torch path of ``dflash2_sample_draft`` after ``_dflash2_global_top_k``."""
        edge_scores = dflash2_score_edges(
            predecessor_codebook,
            successor_codebook,
            candidate_ids,
            unary_logits,
            gate,
            anchor_token_ids.long(),
        )
        uniforms = dflash2_uniforms(seed, offset, candidate_ids.shape[1])
        tokens, q, _ = dflash2_selector_walk(
            candidate_ids,
            edge_scores,
            uniforms,
            temperatures,
            temperatures <= greedy_threshold,
        )
        store = SimpleNamespace(candidate_ids=store_candidate_ids, q=store_q, tokens=store_tokens)
        DFlash2ProposalStore.store(store, slot_ids, candidate_ids, q, tokens)
        return tokens.to(torch.int32)

    def is_valid(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        gate: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_threshold: float,
        store_candidate_ids: torch.Tensor,
        store_q: torch.Tensor,
        store_tokens: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> None:
        n, slots, top_k = candidate_ids.shape
        assert top_k == TOP_K, f"the kernels hold one candidate per lane: top_k must be {TOP_K}"
        assert predecessor_codebook.shape[1] % 256 == 0, (
            "the selector rank must be a multiple of 256"
        )
        for t in (
            predecessor_codebook,
            successor_codebook,
            candidate_ids,
            unary_logits,
            gate,
        ):
            assert t.is_contiguous(), "the kernels read contiguous rows"
        assert temperatures.dtype == torch.float32, "temperatures are read as FP32"
        assert seed.dtype == offset.dtype == torch.int64, "seed and offset are int64"
        assert store_candidate_ids.shape[1] >= slots, "the store holds fewer slots than K"
        # The kernels read ids without a bounds check (torch's indexing asserts on the device).
        vocab = predecessor_codebook.shape[0]
        assert int(candidate_ids.min()) >= 0 and int(candidate_ids.max()) < vocab, "id out of range"
        assert int(anchor_token_ids.min()) >= 0 and int(anchor_token_ids.max()) < vocab, (
            "anchor out of range"
        )


k31_dflash2_selector_draft = K31DFlash2SelectorDraft()

__all__ = ["K31DFlash2SelectorDraft", "k31_dflash2_selector_draft"]
