# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One step of the multi-launch vocab-parallel DSpark Markov chain: resolve the previous token
from the exchanged pairs and gather its ``markov_w1`` row (Triton)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.vocab_parallel_kernels import markov_reduce_embed
from ._reference import argmax_pairs, interleaved_pairs


class K3MarkovReduceEmbed(OpWrapper):
    """For each request ``b < emb.shape[0]``: ``prev = anchors[b]`` at ``step == 0``, else the
    winner of the exchanged pairs of row ``b`` (as ``k3_vocab_reduce_pairs``), also stored as
    ``tokens[b, step - 1]`` (int32); then ``emb[b] = markov_w1[prev]``, or zeros for a ``prev``
    outside ``[0, vocab)``. Returns None."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The helix8 batch-20 recipe runs 20 requests, more than the one-launch chains take: its draft
    # block (6 positions, Markov rank 256) goes through the multi-launch chain over 8 ranks x 10
    # splits.
    CELLS = (
        Cell(
            why="helix8 c240, step 0: the anchors, two of them outside the vocab (zero rows)",
            spec=dict(tp=8, reqs=20, block=6, step=0),
        ),
        Cell(
            why="helix8 c240, step 3: the winner of 80 exchanged slots, stored as token 2",
            spec=dict(tp=8, reqs=20, block=6, step=3),
        ),
        Cell(
            why="helix8 c240, step 6 = block: the last token only",
            spec=dict(tp=8, reqs=20, block=6, step=6),
        ),
    )

    note = """
    The multi-launch chain (``K3VocabParallelHead.markov_tokens``) that runs where the one-launch
    chains do not apply; its bias GEMV (``F.linear``) and local argmax run between two calls.
    One program of 4 warps per request.
    """

    def __call__(
        self,
        pairs: torch.Tensor,
        stride_row: int,
        stride_slot: int,
        n_slots: int,
        anchors: torch.Tensor,
        markov_w1: torch.Tensor,
        emb: torch.Tensor,
        tokens: torch.Tensor,
        vocab: int,
        step: int,
    ) -> None:
        markov_reduce_embed(
            pairs,
            stride_row,
            stride_slot,
            n_slots,
            anchors,
            markov_w1,
            emb,
            tokens,
            vocab,
            step,
        )

    def reference(
        self,
        pairs: torch.Tensor,
        stride_row: int,
        stride_slot: int,
        n_slots: int,
        anchors: torch.Tensor,
        markov_w1: torch.Tensor,
        emb: torch.Tensor,
        tokens: torch.Tensor,
        vocab: int,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(emb, tokens)`` after the call (clones)."""
        num_reqs = emb.shape[0]
        tokens_out = tokens.clone()
        if step == 0:
            prev = anchors[:num_reqs].to(torch.int64)
        else:
            vals, idx = interleaved_pairs(pairs, stride_row, stride_slot, n_slots, num_reqs)
            _, prev = argmax_pairs(vals, idx)
            tokens_out[:num_reqs, step - 1] = prev.to(tokens.dtype)
        valid = (prev >= 0) & (prev < vocab)
        rows = markov_w1[prev.clamp(0, markov_w1.shape[0] - 1)]
        emb_out = torch.where(valid[:, None], rows, torch.zeros_like(rows)).to(emb.dtype)
        return emb_out, tokens_out

    def is_valid(
        self,
        pairs: torch.Tensor,
        stride_row: int,
        stride_slot: int,
        n_slots: int,
        anchors: torch.Tensor,
        markov_w1: torch.Tensor,
        emb: torch.Tensor,
        tokens: torch.Tensor,
        vocab: int,
        step: int,
    ) -> None:
        # Rows are addressed as w1 + prev * RANK and emb + b * RANK.
        assert markov_w1.is_contiguous() and emb.is_contiguous(), "w1 and emb must be contiguous"
        assert emb.shape[1] == markov_w1.shape[1] and emb.dtype == markov_w1.dtype
        assert tokens.dtype == torch.int32 and tokens.stride(-1) == 1, "tokens: int32 rows"
        assert vocab <= markov_w1.shape[0], "vocab exceeds markov_w1's rows"


k3_markov_reduce_embed = K3MarkovReduceEmbed()
