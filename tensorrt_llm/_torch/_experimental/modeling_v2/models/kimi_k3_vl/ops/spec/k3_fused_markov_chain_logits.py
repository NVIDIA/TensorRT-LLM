# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark vanilla-Markov greedy chain on full-vocab block logits, three launches per block
position (two Triton kernels around the bias GEMV)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.dspark_markov_fused import fused_markov_chain_logits
from ._reference import argmax_pairs


class K3FusedMarkovChainLogits(OpWrapper):
    """``[B, K, vocab]`` FP32 biased logits (new): position ``i`` is
    ``base_logits[:, i] + markov_w2 @ markov_w1[prev_i]`` with the bias in the weights' dtype
    (``F.linear``) and the sum in FP32; ``prev_0 = first_prev_tokens`` (int32 / int64) and
    ``prev_{i+1}`` the ``torch.argmax`` of position ``i``'s biased logits. A ``prev`` outside
    ``[0, vocab)`` adds a zero bias.

    ``base_logits`` is FP32 ``[B, K, vocab]`` with a unit vocab stride; ``markov_w1`` and
    ``markov_w2`` are contiguous ``[vocab, R]`` of one dtype.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # Full-vocab drafter logits (163840 columns, Markov rank 256): the deployments whose LM head
    # is not vocab-parallel, with DSpark 5 (6 block positions).
    CELLS = (
        Cell(
            why="dep16 c383: 16 requests per attention-DP rank, int64 anchors",
            spec=dict(reqs=16, block=6, anchor="int64"),
        ),
        Cell(
            why="tp1 context server: 4 requests, int32 anchors, one outside the vocab",
            spec=dict(reqs=4, block=6, anchor="int32"),
        ),
    )

    note = """
    Bit-identical to ``dspark_markov_chain_logits`` driven by a plain full-vocab argmax: the bias
    GEMV is that ``F.linear``, the add is FP32 and the argmax keeps NaN-first, lowest-index
    order. Reached where the drafter's logits are full-vocab (the shared LM head gathers them)
    and the vocab-parallel chains do not apply. 2048 columns per program; launch shapes depend
    on (B, vocab) only.
    """

    def __call__(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        return fused_markov_chain_logits(base_logits, first_prev_tokens, markov_w1, markov_w2)

    def reference(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """The biased logits; the bias is an FP64 product rounded to the weights' dtype."""
        num_reqs, block_size, vocab = base_logits.shape
        out = torch.empty_like(base_logits)
        prev = first_prev_tokens.to(torch.int64)
        cols = torch.arange(vocab, device=base_logits.device).expand(num_reqs, -1)
        w2 = markov_w2.double()
        for step in range(block_size):
            valid = (prev >= 0) & (prev < vocab)
            row = markov_w1[prev.clamp(0, vocab - 1)].double() * valid[:, None]
            bias = (row @ w2.t()).to(markov_w2.dtype).float()
            out[:, step] = base_logits[:, step] + bias
            _, prev = argmax_pairs(out[:, step], cols)
        return out

    def is_valid(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> None:
        # fused_markov_supported, which the caller checks: anything else is read with the wrong
        # strides or dtype rather than rejected.
        assert base_logits.dim() == 3 and base_logits.dtype == torch.float32
        assert base_logits.stride(-1) == 1, "unit vocab stride"
        assert markov_w1.dtype == markov_w2.dtype and markov_w1.shape == markov_w2.shape
        assert markov_w1.is_contiguous() and markov_w2.is_contiguous()
        assert markov_w2.shape[0] == base_logits.shape[-1], "full-vocab logits"


k3_fused_markov_chain_logits = K3FusedMarkovChainLogits()
