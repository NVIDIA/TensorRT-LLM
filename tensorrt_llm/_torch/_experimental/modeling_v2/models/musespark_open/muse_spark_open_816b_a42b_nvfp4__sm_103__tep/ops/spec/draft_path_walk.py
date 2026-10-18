# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash 2 path walk and the greedy draft tokens it implies, in one launch.

The built-in selector adds the unary logits to the einsum (``edge_scores =
unary[:, :, None] + bilinear``), walks the lattice one position at a time
(``dflash2_walk_candidate_paths``: a gather, a copy and an argmax per
position), writes each position's realized score row into a full-vocab
``-inf`` tensor (``fill_`` + ``scatter_`` over ``[B, K, vocab]``) and the
greedy sampler takes ``argmax`` over the vocabulary. Only the argmaxes reach
the draft tokens, so this entry keeps the walk and the two argmax rules and
drops the vocabulary-wide tensor:

* the walk's predecessor is ``argmax`` over the 16 candidate positions;
* the draft token is ``argmax`` over the vocabulary of a row that is ``-inf``
  except at the candidates -- the best candidate, ties to the *lowest vocab
  id* (not the lowest position), and vocab id 0 if every score is ``-inf``.

Both follow torch's ``ArgMaxOps`` (``GreaterOrNan``): NaN wins, then the larger
value, equal values (+0.0 == -0.0) go to the lower index. The score is the
same single fp32 add as the built-in one, so the walk takes the same path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _draft_path_walk_kernel(
    edge_ptr,
    unary_ptr,
    ids_ptr,
    tokens_ptr,
    num_positions,
    TOP_K: tl.constexpr,
):
    b = tl.program_id(0)
    cand = tl.arange(0, TOP_K)
    predecessor = b * 0
    for t in range(num_positions):
        row = (b * num_positions + t).to(tl.int64)
        edge = tl.load(edge_ptr + (row * TOP_K + predecessor) * TOP_K + cand)
        unary = tl.load(unary_ptr + row * TOP_K + cand)
        ids = tl.load(ids_ptr + row * TOP_K + cand)
        score = unary + edge
        is_nan = score != score
        nan_count = tl.sum(is_nan.to(tl.int32), axis=0)
        best = tl.max(tl.where(is_nan, float("-inf"), score), axis=0)
        hit = tl.where(nan_count > 0, is_nan, score == best)
        predecessor = tl.min(tl.where(hit, cand, TOP_K), axis=0)
        token = tl.min(tl.where(hit, ids, 9223372036854775807), axis=0)
        token = tl.where((nan_count == 0) & (best == float("-inf")), 0, token)
        tl.store(tokens_ptr + row, token.to(tl.int32))


class DraftPathWalk(OpWrapper):
    """Greedy DFlash 2 draft tokens from the selector's lattice.

        edges          [B, K, top_k, top_k] fp32: the einsum ``bilinear[b, t, pred, cand]``
                       (without the unary term), contiguous
        unary          [B, K, top_k] fp32, contiguous
        candidate_ids  [B, K, top_k] int64 vocab ids (distinct per row), contiguous

    Returns ``[B, K]`` int32: what ``argmax(select_candidate_path(...))`` gives,
    i.e. the built-in walk then the built-in greedy sampler, without the
    ``[B, K, vocab]`` tensor. Exact.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            "this target's draft head: one request, K = 7, top 16",
            dict(requests=1, positions=7, scores="random"),
        ),
        Cell("one drafted position", dict(requests=1, positions=1, scores="random")),
        Cell("two requests walk independently", dict(requests=2, positions=7, scores="random")),
        Cell(
            "tied scores: the walk picks the lowest position, the token the lowest id",
            dict(requests=1, positions=7, scores="ties"),
        ),
        Cell(
            "NaN and infinite scores follow torch's argmax",
            dict(requests=1, positions=7, scores="special"),
        ),
    )

    def __call__(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        requests, positions, top_k = unary.shape
        tokens = torch.empty((requests, positions), dtype=torch.int32, device=unary.device)
        _draft_path_walk_kernel[(requests,)](
            edges, unary, candidate_ids, tokens, positions, TOP_K=top_k, num_warps=1
        )
        return tokens

    def reference(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """The built-in walk, scatter into a ``-inf`` vocabulary row, and argmax."""
        requests, positions, top_k = unary.shape
        edge_scores = unary.float()[:, :, None] + edges
        realized = edge_scores.new_empty(requests, positions, top_k)
        predecessor = torch.zeros(requests, 1, 1, dtype=torch.long, device=edges.device)
        for step in range(positions):
            scores = edge_scores[:, step].gather(1, predecessor.expand(-1, 1, top_k)).squeeze(1)
            realized[:, step] = scores
            predecessor = scores.argmax(-1).view(requests, 1, 1)
        vocab = int(candidate_ids.max().item()) + 1
        block = torch.full((requests, positions, vocab), float("-inf"), device=edges.device)
        block.scatter_(-1, candidate_ids, realized)
        return torch.argmax(block, dim=-1).to(torch.int32)

    def is_valid(
        self, edges: torch.Tensor, unary: torch.Tensor, candidate_ids: torch.Tensor, **unused
    ) -> None:
        requests, positions, top_k = unary.shape
        if tuple(edges.shape) != (requests, positions, top_k, top_k):
            raise ValueError(f"edges {tuple(edges.shape)} != [B, K, top_k, top_k]")
        if tuple(candidate_ids.shape) != tuple(unary.shape) or candidate_ids.dtype != torch.int64:
            raise ValueError("candidate_ids must be int64 [B, K, top_k]")
        if edges.dtype != torch.float32 or unary.dtype != torch.float32:
            raise ValueError("edges and unary must be fp32")
        if not (edges.is_contiguous() and unary.is_contiguous() and candidate_ids.is_contiguous()):
            raise ValueError("edges, unary and candidate_ids must be contiguous")
        if top_k & (top_k - 1):
            raise ValueError(f"top_k {top_k} must be a power of two")
