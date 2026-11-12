# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark greedy Markov chain of one draft block over a vocab-sharded LM head in one Triton
launch, the per-rank pairs exchanged through a multicast buffer every program polls."""

from __future__ import annotations

from typing import Sequence, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.vocab_parallel_kernels import markov_chain
from ._reference import markov_chain_tokens

Shards = Union[torch.Tensor, Sequence[torch.Tensor]]


class K3MarkovChain(OpWrapper):
    """``tokens`` ``[B, K]`` int32 (written and returned): for each block position ``i``, the
    global argmax over the vocab of ``local_block[:, i] + bf16(markov_w2 @ markov_w1[prev_i])``
    (added in FP32), which is draft token ``i`` and ``prev_{i+1}``; ``prev_0`` is
    ``first_prev_tokens`` and a ``prev`` outside ``[0, vocab)`` adds nothing.

    Each rank passes its ``[B, K, shard]`` block logits (unit inner stride), its ``[shard, R]``
    rows of ``markov_w2`` and the full ``[vocab, R]`` ``markov_w1`` (contiguous). ``partial``,
    ``tickets`` and ``launch`` are the rank's scratch (``_CHAIN_MAX_STEPS * programs * 16 * 2``
    float32, 16 int32 zeros, 1 int32 launch ordinal); ``mcast`` / ``local`` the multicast and
    local views of the three-set exchange of ``set_floats`` floats each (one buffer on one rank),
    initialized to -0.0.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # One rank of the helix16 ULL group (10240-column shards, 80 programs; Markov rank 256,
    # 8 block positions); the other ranks' pairs are in the exchange. The chain where the CuTe DSL
    # chain (``k3_dspark_markov_greedy``) does not apply. A helix8 shard (160 programs) needs
    # more SMs than one GB300 has, so its cells belong to the archs that hold them.
    CELLS = (
        Cell(
            why="helix16 ULL c1, rank 0 of 16: one request, int32 anchor",
            spec=dict(tp=16, rank=0, reqs=1, block=8, anchor="int32"),
        ),
        Cell(
            why="helix16 ULL c4, last rank: 4 requests, int64 anchors, one outside the vocab",
            spec=dict(tp=16, rank=15, reqs=4, block=8, anchor="int64"),
        ),
    )

    note = """
    One program per 128 shard columns; every program must be resident (the caller checks against
    the SM count) because each waits for all ranks' pairs every step. Up to 16 requests and 16
    positions. The bias is an FP32 dot rounded to the weights' dtype, summed in a different order
    than ``F.linear``: a token can differ where two vocab entries tie within that rounding.
    Launched with PDL; the program waits before reading the logits. Every rank makes the same
    sequence of calls; the launch ordinal rotates the exchange sets, so no fence is needed.
    """

    def __call__(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        w2_shard: torch.Tensor,
        tokens: torch.Tensor,
        partial: torch.Tensor,
        tickets: torch.Tensor,
        launch: torch.Tensor,
        mcast: torch.Tensor,
        local: torch.Tensor,
        set_floats: int,
        vocab: int,
        col_base: int,
        tp: int,
        tp_rank: int,
    ) -> torch.Tensor:
        return markov_chain(
            local_block,
            first_prev_tokens,
            markov_w1,
            w2_shard,
            tokens,
            partial,
            tickets,
            launch,
            mcast,
            local,
            set_floats,
            vocab,
            col_base,
            tp,
            tp_rank,
        )

    def reference(
        self,
        local_block: Shards,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        w2_shard: Shards,
        tokens: torch.Tensor,
        partial: torch.Tensor,
        tickets: torch.Tensor,
        launch: torch.Tensor,
        mcast: torch.Tensor,
        local: torch.Tensor,
        set_floats: int,
        vocab: int,
        col_base: int,
        tp: int,
        tp_rank: int,
    ) -> torch.Tensor:
        """The tokens (new). A collective: with ``tp > 1`` ``local_block`` and ``w2_shard`` are
        the group's, in rank order (a tensor is taken as the whole vocab); ``col_base`` is then
        not read. The scratch and exchange buffers are not read."""
        if not isinstance(local_block, torch.Tensor):
            local_block = torch.cat(list(local_block), dim=-1)
            w2_shard = torch.cat(list(w2_shard), dim=0)
        return markov_chain_tokens(local_block, first_prev_tokens, markov_w1, w2_shard, vocab)

    def is_valid(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        w2_shard: torch.Tensor,
        tokens: torch.Tensor,
        partial: torch.Tensor,
        tickets: torch.Tensor,
        launch: torch.Tensor,
        mcast: torch.Tensor,
        local: torch.Tensor,
        set_floats: int,
        vocab: int,
        col_base: int,
        tp: int,
        tp_rank: int,
    ) -> None:
        num_reqs, block_size, shard = local_block.shape
        assert num_reqs <= 16 and block_size <= 16, "the exchange sets hold 16 x 16"
        assert w2_shard.shape[0] == shard and w2_shard.stride(-1) == 1, "w2 rows of the shard"
        assert markov_w1.is_contiguous() and local_block.stride(-1) == 1
        assert set_floats == 16 * 16 * max(tp, 1) * 2, "set_floats of another exchange"
        assert tokens.dtype == torch.int32 and tuple(tokens.shape) == (num_reqs, block_size)


k3_markov_chain = K3MarkovChain()
