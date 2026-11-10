# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE round trip as an all-to-all (`NVLinkOneSided`), run when
`TRTLLM_MODELING_V2_MOE_ALLTOALL=1` (`modeling.MOE_ALLTOALL_ENV`) in place of the default
all-gather / reduce-scatter. `moe_a2a_dispatch` sends every rank the rows with a top-k expert in
its whole-expert window, with their top-k ids and weights; `moe_a2a_combine` returns each
partial to the row's rank and sums them there. The strategy (`build_strategy`) holds the MNNVL
workspace and the state between a dispatch and its combine: build it once, pass it to every call."""

from __future__ import annotations

from typing import List, NamedTuple, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HIDDEN = 2560
NUM_EXPERTS = 512
TOPK = 10

#: The id the strategy writes over a received slot that holds no token: outside every window.
NO_EXPERT = -1


def build_strategy(mapping, max_num_tokens_per_rank: int):
    """The stock `NVLinkOneSided` strategy over the attention-DP group (`mapping`'s TP group,
    every rank one EP rank), or None where the platform has no MNNVL. Collective: every rank of
    the group builds it at the same point."""
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    if not NVLinkOneSided.is_platform_supported():
        return None
    return NVLinkOneSided(
        mapping,
        num_slots=NUM_EXPERTS,
        top_k=TOPK,
        max_num_tokens_per_rank=max_num_tokens_per_rank,
        hidden_size=HIDDEN,
        dtype=torch.bfloat16,
    )


def _cell(world: int, counts: tuple, why: str) -> Cell:
    """One round trip over an attention-DP group of `world` ranks holding `counts` rows."""
    return Cell(
        why=why,
        spec=dict(world=world, counts=counts, hidden=HIDDEN, topk=TOPK, dtype=torch.bfloat16),
    )


_CELLS = (
    _cell(2, (1, 1), "dep2: a decode step at batch 1"),
    _cell(2, (5, 300), "dep2: a ragged eager step, one rank holding a context chunk"),
    _cell(4, (0, 17, 3, 0), "dep4: a ragged step where two ranks hold no rows"),
    _cell(4, (2048,) * 4, "dep4: a prefill chunk of 2048 rows a rank"),
)


class Dispatched(NamedTuple):
    """What `moe_a2a_dispatch` hands this rank, `[world * max(counts), ...]` each: the received
    `rows`, their `topk_ids` (global expert ids; `NO_EXPERT` on a slot that holds no token) and
    `topk_weights`. Block `s` holds source rank `s`'s tokens in an order the kernel picks."""

    rows: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


class MoeA2aDispatch(OpWrapper):
    """Send every rank the rows routed into its window; see the module docstring."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _CELLS

    note = """
    Source: `NVLinkOneSided.dispatch` (`torch.ops.trtllm.moe_a2a_dispatch`): each token written
    once to every rank one of its top-k experts lives on, into that rank's block of the sender
    (slot order inside a block is not fixed), the slots past each block's count stamped
    `NO_EXPERT`. Measured on GB300; received rows compared with `reference` as a set.
    Not certified: world sizes other than 2 and 4, groups spanning nodes, sm_100, post-quant
    dispatch, low-precision combine, EPLB statistics.
    """

    def __call__(
        self,
        strategy,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        all_rank_num_tokens: List[int],
    ) -> Dispatched:
        # The weights travel as fp32 (bf16 -> fp32 -> bf16 is exact).
        rows, _, ids, weights = strategy.dispatch(
            x, None, topk_ids, topk_weights.float(), list(all_rank_num_tokens)
        )
        return Dispatched(rows, ids, weights.to(topk_weights.dtype))

    def reference(
        self,
        per_rank_inputs: Sequence[Sequence[torch.Tensor]],
        all_rank_num_tokens: List[int],
        experts_per_rank: int,
        rank: int = 0,
    ) -> List[tuple]:
        """What `rank` receives from every rank's `(x, topk_ids, topk_weights)` (ascending rank
        order): per source, its tokens with a top-k id in `rank`'s window, in token order.
        Compare as a set: the kernel's order inside a block is its own."""
        lo, hi = rank * experts_per_rank, (rank + 1) * experts_per_rank
        out = []
        for x, ids, w in per_rank_inputs:
            hit = ((ids >= lo) & (ids < hi)).any(1)
            out.append((x[hit], ids[hit], w[hit]))
        return out

    def is_valid(
        self,
        strategy,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        all_rank_num_tokens: List[int],
    ) -> None:
        assert x.shape[0] == topk_ids.shape[0] == topk_weights.shape[0], (
            x.shape,
            topk_ids.shape,
            topk_weights.shape,
        )
        assert x.shape[0] <= strategy.max_num_tokens_per_rank, (
            f"{x.shape[0]} rows past the workspace's {strategy.max_num_tokens_per_rank}"
        )
        assert topk_ids.dtype == torch.int32, topk_ids.dtype


class MoeA2aCombine(OpWrapper):
    """Return each partial to the rank its row came from and sum a row's partials there.
    `partials` is `[world * max(counts), hidden]` in the order `moe_a2a_dispatch` handed the rows
    (a slot that holds no token may hold anything); returns this rank's `[rows, hidden]`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _CELLS

    note = """
    Source: `NVLinkOneSided.combine` (`torch.ops.trtllm.moe_a2a_combine`). Must follow the
    strategy's `dispatch` (the strategy keeps the slot map between them). Not certified: as
    `MoeA2aDispatch`.
    """

    def __call__(self, strategy, partials: torch.Tensor) -> torch.Tensor:
        return strategy.combine(partials)

    def reference(self, per_rank_token_partials: Sequence[torch.Tensor]) -> torch.Tensor:
        """This rank's rows: the partial every rank computed for each of them (zero where the
        row was not sent there), `[rows, hidden]` each in ascending rank order, summed in fp32
        and rounded once."""
        out = torch.zeros_like(per_rank_token_partials[0], dtype=torch.float32)
        for partial in per_rank_token_partials:
            out += partial.float()
        return out.to(per_rank_token_partials[0].dtype)

    def is_valid(self, strategy, partials: torch.Tensor) -> None:
        assert partials.dim() == 2 and partials.shape[1] == HIDDEN, partials.shape


moe_a2a_dispatch = MoeA2aDispatch()
moe_a2a_combine = MoeA2aCombine()
