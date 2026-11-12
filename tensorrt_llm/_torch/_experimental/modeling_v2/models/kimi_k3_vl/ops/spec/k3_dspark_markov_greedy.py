# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark greedy Markov chain over a vocab-sharded LM head in one CuTe DSL launch
(SM100 / SM103 / SM107), per-CTA keys exchanged through multicast stores."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.dspark_markov_greedy import run_dspark_markov_greedy
from ._reference import markov_chain_tokens

Shards = Union[torch.Tensor, Sequence[torch.Tensor]]


class K3DsparkMarkovGreedy(OpWrapper):
    """Writes ``tokens`` (int32 ``[B, K]``): the chain of ``k3_markov_chain`` on BF16 block
    logits ``[B, K, shard]`` (unit vocab stride), ``anchors`` ``[B]`` int32 / int64, the full
    BF16 ``markov_w1`` ``[vocab, R]`` and this rank's ``w2_shard`` ``[shard, R]`` (contiguous).
    Returns None.

    ``state`` is the rank's int32 ``[STATE_WORDS]`` launch state (zero-initialized), ``box_uc`` /
    ``box_mc`` the local and multicast addresses of the ``NUM_SETS`` key-exchange sets
    (zero-initialized; equal on one rank, ``multicast`` False).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # One rank of the helix group with the others' keys in the exchange (no multicast):
    # 163840 columns over 16 ranks (40 CTAs) or 8 ranks (80 CTAs), Markov rank 256, 8 block
    # positions (DSpark 7); the request widths 1, 2, 4 and 8 of the recipes.
    CELLS = (
        Cell(
            why="helix16 ULL c1, rank 0 of 16: one request, int32 anchor",
            spec=dict(tp=16, rank=0, reqs=1, block=8, anchor="int32"),
        ),
        Cell(
            why="helix16 ULL c4, last rank: 4 requests, int64 anchors, one outside the vocab",
            spec=dict(tp=16, rank=15, reqs=4, block=8, anchor="int64"),
        ),
        Cell(
            why="helix8 c24/c42, rank 3 of 8: 2 requests over a 20480-column shard",
            spec=dict(tp=8, rank=3, reqs=2, block=8, anchor="int32"),
        ),
        Cell(
            why="helix8 c75/c174, last rank: 8 requests, the 5120 keys of a position",
            spec=dict(tp=8, rank=7, reqs=8, block=8, anchor="int64"),
        ),
    )

    note = """
    One CTA of 512 threads per 256 shard rows, all resident (the caller checks SM count and
    shared memory: R a multiple of 16 up to 256, at most 16 requests and positions). The bias is
    ``mma.sync`` BF16 x BF16 accumulated in FP32 over R in order and rounded to BF16, then added to
    the BF16 logit in FP32; keys order by value, then lower global index, NaN first. A
    process-wide compile cache holds one kernel per (request width, positions, tp, CTAs, R,
    multicast, anchor dtype, PDL, staged, gate); compile before capture. The launch ordinal in
    ``state`` rotates three exchange sets; every rank makes the same sequence of calls.
    """

    def __call__(
        self,
        local_block: torch.Tensor,
        anchors: torch.Tensor,
        markov_w1: torch.Tensor,
        w2_shard: torch.Tensor,
        tokens: torch.Tensor,
        *,
        vocab: int,
        tp: int,
        rank: int,
        state: torch.Tensor,
        box_uc: int,
        box_mc: int,
        multicast: bool,
        use_pdl: bool = True,
        staged: Optional[bool] = None,
        gate: Optional[bool] = None,
    ) -> None:
        run_dspark_markov_greedy(
            local_block,
            anchors,
            markov_w1,
            w2_shard,
            tokens,
            vocab=vocab,
            tp=tp,
            rank=rank,
            state=state,
            box_uc=box_uc,
            box_mc=box_mc,
            multicast=multicast,
            use_pdl=use_pdl,
            staged=staged,
            gate=gate,
        )

    def reference(
        self,
        local_block: Shards,
        anchors: torch.Tensor,
        markov_w1: torch.Tensor,
        w2_shard: Shards,
        tokens: torch.Tensor,
        *,
        vocab: int,
        tp: int,
        rank: int,
        state: torch.Tensor,
        box_uc: int,
        box_mc: int,
        multicast: bool,
        use_pdl: bool = True,
        staged: Optional[bool] = None,
        gate: Optional[bool] = None,
    ) -> torch.Tensor:
        """The tokens (new). A collective: with ``tp > 1`` ``local_block`` and ``w2_shard`` are
        the group's, in rank order. The launch state and the exchange are not read."""
        if not isinstance(local_block, torch.Tensor):
            local_block = torch.cat(list(local_block), dim=-1)
            w2_shard = torch.cat(list(w2_shard), dim=0)
        return markov_chain_tokens(local_block, anchors, markov_w1, w2_shard, vocab)

    def is_valid(
        self,
        local_block: torch.Tensor,
        anchors: torch.Tensor,
        markov_w1: torch.Tensor,
        w2_shard: torch.Tensor,
        tokens: torch.Tensor,
        **kwargs,
    ) -> None:
        num_reqs, steps, shard = local_block.shape
        assert local_block.dtype == markov_w1.dtype == w2_shard.dtype == torch.bfloat16
        assert 1 <= num_reqs <= 16 and 1 <= steps <= 16, "1..16 requests and positions"
        assert w2_shard.shape == (shard, markov_w1.shape[1]) and w2_shard.is_contiguous()
        assert markov_w1.shape[1] % 16 == 0 and markov_w1.shape[1] <= 256, "R: 16 x n <= 256"
        assert local_block.stride(-1) == 1 and markov_w1.is_contiguous()
        assert tokens.dtype == torch.int32 and tokens.stride(-1) == 1


k3_dspark_markov_greedy = K3DsparkMarkovGreedy()
