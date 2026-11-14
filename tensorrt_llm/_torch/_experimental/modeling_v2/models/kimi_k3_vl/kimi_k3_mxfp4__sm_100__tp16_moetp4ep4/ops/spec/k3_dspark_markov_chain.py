# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark vanilla-Markov greedy block chain over the drafter's block logits (full vocabulary, or this rank's
vocabulary shard with the TP group's pair exchange): the biased logits the speculative worker samples the draft
tokens from."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.kernels.k3_dspark_markov_chain import (  # noqa: E501
    k3_dspark_markov_chain_kernel as _kernel,
)


class K3DSparkMarkovChain(OpWrapper):
    """Return ``biased`` [B, K, n] fp32 for the block logits ``base_logits`` [B, K, n] fp32 (unit vocabulary stride):
    position k adds ``F.linear(markov_w1[prev_k], markov_w2)`` (this shard's rows of ``markov_w2``) to
    ``base_logits[:, k]``, where prev_0 is ``first_prev_tokens[b]`` and prev_k (k > 0) is the first maximum of
    ``biased[b, k - 1]`` over the whole vocabulary (every rank's shard with ``tp_exchange``; NaN beats every number,
    ties go to the lowest index). An out-of-vocabulary prev gives a zero bias. ``markov_w1`` / ``markov_w2`` are
    [vocab, rank] of one dtype, ``markov_w1`` contiguous."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{why}: B={b}, K={k}, vocabulary {vocab}, Markov rank {rank}, {dtype}",
            spec=dict(batch=b, block=k, vocab=vocab, rank=rank, dtype=dtype),
        )
        for b, k, vocab, rank, dtype, why in (
            (1, 7, 163840, 256, torch.bfloat16, "one decode request on the full K3 vocabulary"),
            (8, 7, 163840, 256, torch.bfloat16, "eight requests (the target's decode maximum)"),
            (4, 3, 163840, 256, torch.bfloat16, "a shorter block"),
            (3, 7, 5000, 4, torch.float32, "a small fp32 head, anchors outside the vocabulary"),
        )
    )

    note = """
    The chain the speculative worker runs through the drafter's
    ``apply_markov_chain_logits`` when the block logits arrive full-vocabulary
    (the drafter shares the target's gathered LM head) or as a plain TP
    column shard. Each block position is three launches: resolve prev and
    gather its markov_w1 row, the bias GEMV (F.linear, cuBLAS), add and
    argmax per column split; with ``tp_exchange`` one float32 all-reduce
    exchanges the splits' (value, index) pairs. The test drives the
    full-vocabulary form on one GPU; the shard form shares every kernel.
    """

    def __call__(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        tp_exchange=None,
    ) -> torch.Tensor:
        return _kernel.markov_chain_logits(
            base_logits, first_prev_tokens, markov_w1, markov_w2, tp_exchange
        )

    def reference(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        tp_exchange=None,
    ) -> torch.Tensor:
        """The full-vocabulary chain, eager (``tp_exchange`` must be None)."""
        assert tp_exchange is None, "the reference runs the full-vocabulary chain"
        vocab = markov_w1.shape[0]
        biased = torch.empty_like(base_logits)
        prev = first_prev_tokens.long()
        for k in range(base_logits.shape[1]):
            valid = (prev >= 0) & (prev < vocab)
            emb = markov_w1[prev.clamp(0, vocab - 1)] * valid.unsqueeze(-1).to(markov_w1.dtype)
            biased[:, k] = base_logits[:, k] + torch.nn.functional.linear(emb, markov_w2).float()
            prev = biased[:, k].argmax(-1)
        return biased

    def is_valid(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        tp_exchange=None,
    ) -> None:
        assert base_logits.dtype == torch.float32 and base_logits.stride(-1) == 1
        assert markov_w1.is_contiguous() and markov_w1.dtype == markov_w2.dtype
        assert markov_w1.shape == markov_w2.shape
        assert base_logits.shape[-1] < 2**24, "indices must be exact in float32"
        if tp_exchange is None:
            assert base_logits.shape[-1] == markov_w2.shape[0]
        else:
            assert base_logits.shape[-1] * tp_exchange.tp_size == markov_w2.shape[0]


k3_dspark_markov_chain = K3DSparkMarkovChain()
