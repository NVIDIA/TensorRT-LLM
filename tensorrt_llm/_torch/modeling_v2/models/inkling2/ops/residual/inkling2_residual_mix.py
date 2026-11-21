# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2's block attention residual: a layer's input is a learned mix of the completed
residual blocks and the current partial block, followed by that input's RMSNorm.

Every 6 layers the residual stream closes a block: the input of layers 0, 6, 12, ... (the
embedding for layer 0) becomes a block state, and the stream restarts from the next sublayer's
output. A sublayer reads, instead of the stream, a softmax mix over the unit-RMS-normalized
block states and partial block, with a learned fp32 query per input (3 for a global layer's q, k
and v inputs, 1 otherwise; the final norm has its own):

    n_s     = s / rms(s)                   (eps 1e-6), for each block s and the partial block
    p_q     = softmax_s(q . n_s / sqrt(H))
    mix_q   = sum_s p_q[s] * n_s
    out_q   = weight_q * mix_q / rms(mix_q)   (eps), rounded to bf16

all in fp32. `blocks` holds the completed blocks already normalized, `[S, T, H]` fp32; a block's
normalization is the same in every later mix, so the caller computes it once
(`normalize_block`).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: The unweighted normalization of block states (Inkling2UnweightedRMSNorm's default).
BLOCK_EPS = 1e-6


def normalize_block(state: torch.Tensor) -> torch.Tensor:
    """A block state `[T, H]` divided by its RMS (eps 1e-6), fp32."""
    x = state.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + BLOCK_EPS)


def rmsnorm_to_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`weight * x / rms(x)` in fp32, rounded once to bf16 (Inkling2RMSNorm on an fp32 input)."""
    x = x.float()
    return (weight.float() * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))).to(
        torch.bfloat16
    )


class Inkling2ResidualMix(OpWrapper):
    """`blocks` `[S, T, H]` fp32 (normalized), `partial` `[T, H]` (bf16 or fp32), `queries` `[Q, H]`
    fp32, `norm_weights` Q weights `[H]` -> `[Q, T, H]` bf16, each mix through its RMSNorm."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="layer 1's attention input: the embedding block, one query",
            spec=dict(blocks=1, queries=1, tokens=17),
        ),
        Cell(
            why="a global layer's q/k/v inputs: 3 queries over 5 blocks",
            spec=dict(blocks=5, queries=3, tokens=33),
        ),
        Cell(
            why="the final norm's input: 8 blocks, one decode token",
            spec=dict(blocks=8, queries=1, tokens=1),
        ),
    )

    note = """
    Values are the normalized states, not the raw ones: a mix of raw states is plausible but
    scaled by the states' RMS. The partial block is normalized here, the completed blocks by
    the caller, once.
    """

    def __call__(
        self,
        blocks: torch.Tensor,
        partial: torch.Tensor,
        queries: torch.Tensor,
        norm_weights: Sequence[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        hidden = partial.shape[-1]
        n_p = normalize_block(partial)
        with true_fp32_matmul():
            logits_b = torch.einsum("sth,qh->tqs", blocks, queries)
            logit_p = (n_p @ queries.t()).unsqueeze(-1)
        probs = torch.softmax(torch.cat([logits_b, logit_p], dim=-1) / math.sqrt(hidden), dim=-1)
        with true_fp32_matmul():
            mix = torch.einsum("tqs,sth->qth", probs[..., :-1].contiguous(), blocks)
        mix = mix + probs[..., -1].t().unsqueeze(-1) * n_p.unsqueeze(0)
        return torch.stack([rmsnorm_to_bf16(mix[i], w, eps) for i, w in enumerate(norm_weights)])

    def reference(
        self,
        blocks: torch.Tensor,
        partial: torch.Tensor,
        queries: torch.Tensor,
        norm_weights: Sequence[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """fp64 throughout, the partial block appended to the normalized blocks as one more state."""
        hidden = partial.shape[-1]
        x = partial.double()
        n_p = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + BLOCK_EPS)
        stacked = torch.cat([blocks.double(), n_p.unsqueeze(0)])  # [S + 1, T, H]
        probs = torch.softmax(
            torch.einsum("sth,qh->tqs", stacked, queries.double()) / math.sqrt(hidden), dim=-1
        )
        mix = torch.einsum("tqs,sth->qth", probs, stacked)
        out = []
        for i, w in enumerate(norm_weights):
            m = mix[i]
            out.append(
                (w.double() * (m * torch.rsqrt(m.square().mean(-1, keepdim=True) + eps))).to(
                    torch.bfloat16
                )
            )
        return torch.stack(out)

    def is_valid(
        self,
        blocks: torch.Tensor,
        partial: torch.Tensor,
        queries: torch.Tensor,
        norm_weights: Sequence[torch.Tensor],
        eps: float,
    ) -> None:
        assert blocks.dtype == torch.float32 and blocks.dim() == 3, (
            "blocks are normalized fp32 [S, T, H]"
        )
        assert blocks.shape[1:] == partial.shape, "blocks and the partial block share [T, H]"
        assert queries.dtype == torch.float32 and queries.shape[1] == partial.shape[1], (
            "queries are fp32 [Q, H]"
        )
        assert len(norm_weights) == queries.shape[0], "one norm weight per query"


inkling2_residual_mix = Inkling2ResidualMix()
