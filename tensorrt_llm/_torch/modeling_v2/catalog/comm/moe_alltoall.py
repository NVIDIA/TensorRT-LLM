# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exchange MoE tokens between expert-parallel ranks over NVLink: send each token to the ranks
that hold its slots, then sum what they computed back onto it (the engine's one-sided alltoall).

A routed MoE layer under wide expert parallelism runs one round trip per round of tokens:

1. `moe_alltoall_dispatch`: every rank sends its rows, with their slot ids and routing
   weights, to the ranks whose slot windows hold any of the rows' slots. Each rank receives
   `[ep_size * max_tokens, ...]` rows, `max_tokens` the largest count any rank sent this
   round; a received row's slot ids stay global, and a row that holds no token for this rank
   carries `INVALID_SLOT` in every slot.
2. The rank's expert kernel runs over the received rows, its own slot window only, and
   finalizes each row: the row's weighted sum over the slots this rank holds. A kernel that
   finalizes into `moe_alltoall_combine_rows`, the rows the combine reads in the workspace,
   saves the combine a copy of every row.
3. `moe_alltoall_combine`: every rank sends each finalized row back to the rank it came from,
   which sums the rows of its tokens, `[local_tokens, hidden]`.

The communicator (`moe_alltoall_workspace`) is the engine's `NVLinkOneSided`: an NVLink
workspace every EP rank maps, plus the state of the round in flight. One round is in flight
at a time: dispatch and combine alternate, and every rank runs the same rounds
(`alltoall_rounds`), also a rank with no tokens left to send.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: The slot id of a row that goes to no rank. A received row that holds no token for this
#: rank carries it in every slot; a caller gives it to a row it does not want dispatched, such
#: as padding, and combine returns zeros for that row.
INVALID_SLOT = -1

#: The dispatch payloads a workspace is laid out for: rows of NVFP4 codes with their linear
#: block scales, or BF16 rows.
PAYLOAD_FORMATS = frozenset({"NVFP4", "BF16"})


def alltoall_rounds(
    all_rank_num_tokens: Sequence[int], max_tokens_per_rank: int
) -> List[List[int]]:
    """Every rank's row count in each round of a MoE alltoall that sends at most
    `max_tokens_per_rank` rows a rank a round.

    Round `r` sends a rank's rows `[r * max_tokens_per_rank, (r + 1) * max_tokens_per_rank)`.
    Every rank takes part in every round, a rank whose rows ran out with none, because the
    dispatch and the combine are collective: a rank that skipped a round would leave every
    other rank waiting on it. There is at least one round, also when no rank holds a row.

    Args:
        all_rank_num_tokens: Every EP rank's row count this forward, in rank order. Every rank
            holds the same list, so every rank plans the same rounds without communicating.
        max_tokens_per_rank: The most rows a rank sends a round, the workspace's capacity
            (`moe_alltoall_workspace`).
    """
    assert max_tokens_per_rank > 0, max_tokens_per_rank
    counts = [int(n) for n in all_rank_num_tokens]
    assert all(n >= 0 for n in counts), counts
    rounds = max(1, -(-max(counts) // max_tokens_per_rank))
    return [
        [min(max(n - r * max_tokens_per_rank, 0), max_tokens_per_rank) for n in counts]
        for r in range(rounds)
    ]


def moe_alltoall_workspace(
    mapping,
    *,
    num_slots: int,
    top_k: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    payload: str,
    eplb_num_experts: Optional[int] = None,
):
    """Build the communicator of one dispatch payload: the NVLink workspace every EP rank maps,
    and the state of the round in flight.

    Collective over the EP group: every rank calls it with the same arguments, after the
    weights load, never under a CUDA graph capture. The engine keeps one workspace per layout
    for the process, so a target builds one communicator per payload it dispatches and shares
    it between its layers; a round's dispatch and combine alternate on it.

    Args:
        mapping: The deployment's mapping. Pure expert parallelism: every rank is an EP rank
            (`world_size == moe_ep_size`).
        num_slots: Slots over all EP ranks, `num_experts` without EPLB.
        top_k: Slots each token selects.
        max_tokens_per_rank: The most rows a rank sends a round.
        hidden_size: Elements of one token's hidden state; combine returns BF16 rows of it.
        payload: What a dispatch carries, one of `PAYLOAD_FORMATS`: NVFP4 codes with their
            linear block scales, or BF16 rows. It selects the transport the ranks agree on for
            small rounds.
        eplb_num_experts: Under online EPLB, the experts a histogram counts when a dispatch
            carries one (`EplbLayer.local_statistic`); None otherwise.
    """
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    assert payload in PAYLOAD_FORMATS, (
        f"payload {payload!r} is not one of {sorted(PAYLOAD_FORMATS)}"
    )
    assert mapping.world_size == mapping.moe_ep_size, (
        f"the alltoall exchanges tokens between EP ranks only; world_size {mapping.world_size} "
        f"is not moe_ep_size {mapping.moe_ep_size}"
    )
    return NVLinkOneSided(
        mapping,
        num_slots,
        top_k,
        max_tokens_per_rank,
        payload_in_workspace=False,
        hidden_size=hidden_size,
        dtype=torch.bfloat16,
        num_experts=eplb_num_experts,
        sentinel_payload_format=payload,
    )


class MoeAlltoallDispatch(OpWrapper):
    """Send this rank's rows to the EP ranks that hold their slots; return the rows this rank
    received, `[ep_size * max_tokens, ...]` each, `max_tokens = max(all_rank_num_tokens)`.

    The rows from rank `src` sit at `[src * max_tokens, (src + 1) * max_tokens)`; which of
    them hold tokens, and in which order, is the transport's business. A received row's slot
    ids are global, so the expert kernel picks out this rank's slots by its own window. A row
    that holds no token for this rank has `INVALID_SLOT` in every slot. The hidden payload
    is copied byte for byte: NVFP4 codes and their block scales arrive as they were sent.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The DeepSeek-R1 default target at wide EP is the caller: NVFP4 rows of the trunk's
    # 7168-wide hidden state with their 448 linear scale bytes, BF16 rows of the MTP layer,
    # top-8 over 256 experts, or 288 slots under EPLB. Round sizes span one decode token to a
    # full round; a rank with no rows in a round still takes part.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one NVFP4 row a rank, every rank sending",
            spec=dict(payload="NVFP4", rows=(1, 1, 1, 1), slots=256, top_k=8),
        ),
        Cell(
            why="uneven rounds: ranks send different counts, one sends none",
            spec=dict(payload="NVFP4", rows=(64, 3, 0, 17), slots=256, top_k=8),
        ),
        Cell(
            why="288 slots: replicas of an expert on several ranks",
            spec=dict(payload="NVFP4", rows=(32, 32, 32, 32), slots=288, top_k=8),
        ),
        Cell(
            why="BF16 rows of the MTP layer",
            spec=dict(payload="BF16", rows=(16, 0, 5, 16), slots=256, top_k=8),
        ),
    )

    note = """
    Sending each row once to every rank that holds any of its slots, whatever how many of
    its slots that rank holds, and the transport the ranks agree on for small rounds, are
    the engine's `NVLinkOneSided`; this entry certifies the rows each rank receives, not the
    route they took. CUDA graph capture and replay of a round, and a histogram carried by
    the dispatch under online EPLB, are checked by name in the entry's test.
    """

    def __call__(
        self,
        workspace,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_selected_slots: torch.Tensor,
        token_final_scales: torch.Tensor,
        all_rank_num_tokens: Sequence[int],
        eplb_local_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Dispatch one round.

        Args:
            workspace: The communicator `moe_alltoall_workspace` built for this payload.
            hidden_states: This rank's rows of the round, `[tokens, hidden]` BF16, or
                `[tokens, hidden // 2]` uint8 NVFP4 codes.
            hidden_states_sf: The codes' linear block scales, `[tokens, hidden // 16]`
                uint8; None for BF16 rows.
            token_selected_slots: `[tokens, top_k]` int32 global slot ids; `INVALID_SLOT`
                in every slot of a row that goes to no rank.
            token_final_scales: `[tokens, top_k]` routing weights, sent as they are.
            all_rank_num_tokens: Every EP rank's row count this round, in rank order.
            eplb_local_stats: This rank's histogram, `[eplb_num_experts]` int32, on the
                forward's last dispatch under online EPLB; every rank passes one on the same
                dispatch, or none does. The communicator then holds every rank's
                (`moe_alltoall_gathered_statistic`).

        Returns:
            `(hidden_states, hidden_states_sf, token_selected_slots, token_final_scales)`
            received, each `[ep_size * max(all_rank_num_tokens), ...]`.
        """
        kwargs = {} if eplb_local_stats is None else {"eplb_local_stats": eplb_local_stats}
        return workspace.dispatch(
            hidden_states,
            hidden_states_sf,
            token_selected_slots,
            token_final_scales,
            list(all_rank_num_tokens),
            **kwargs,
        )

    def reference(
        self,
        rank: int,
        per_rank_slots: Sequence[torch.Tensor],
        local_slots: int,
    ) -> List[List[int]]:
        """Which rows rank `rank` receives: for every source rank, the indices of its rows
        that select a slot in `rank`'s window `[rank * local_slots, (rank + 1) *
        local_slots)`, in the source's order.

        The documented exception to mirroring `__call__`, as for every collective: what a rank
        receives is a function of every rank's rows, so this takes the group's slot ids in
        rank order. The payloads are compared row for row through these indices, since where
        a received row lands within its source's block is not specified.
        """
        lo, hi = rank * local_slots, (rank + 1) * local_slots
        return [
            [i for i, row in enumerate(slots.tolist()) if any(lo <= s < hi for s in row)]
            for slots in per_rank_slots
        ]

    def is_valid(
        self,
        workspace,
        hidden_states: torch.Tensor,
        hidden_states_sf: Optional[torch.Tensor],
        token_selected_slots: torch.Tensor,
        token_final_scales: torch.Tensor,
        all_rank_num_tokens: Sequence[int],
        eplb_local_stats: Optional[torch.Tensor] = None,
    ) -> None:
        # Each guard is for an input the kernels take and answer wrongly.
        rows = hidden_states.shape[0]
        assert (
            token_selected_slots.dtype == torch.int32 and token_selected_slots.shape[0] == rows
        ), (
            "slot ids must be int32 with a row per token; the kernels read them as int32 and "
            "route a short tensor's missing rows from whatever follows it"
        )
        assert all_rank_num_tokens[workspace.ep_rank] == rows, (
            f"all_rank_num_tokens says this rank sends {all_rank_num_tokens[workspace.ep_rank]} "
            f"rows, it holds {rows}: the receiving ranks size their blocks from the list"
        )
        assert max(all_rank_num_tokens) <= workspace.max_num_tokens_per_rank, (
            f"a round of {max(all_rank_num_tokens)} rows a rank overruns the workspace's "
            f"{workspace.max_num_tokens_per_rank}; plan the rounds with alltoall_rounds"
        )


class MoeAlltoallCombine(OpWrapper):
    """Send the rows this rank finalized back to the ranks they came from, and sum the rows of
    this rank's tokens: `[local_tokens, hidden]`, `local_tokens` this rank's rows of the round.

    A token's result is the sum, over the ranks it was sent to, of each rank's finalized row --
    its weighted sum over the slots that rank holds -- so the token gets each of its slots
    once. A row that went to no rank comes back as zeros.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = MoeAlltoallDispatch.CELLS

    note = """
    Its cells are the dispatch's: a combine sums the rows of the dispatch before it, whose
    shapes it inherits. The sum's order across ranks is the transport's, so the entry's test
    holds it to an fp32 reference within a band.
    """

    def __call__(self, workspace, final_hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine the round the last dispatch on `workspace` sent.

        Args:
            workspace: The communicator of that dispatch.
            final_hidden_states: The finalized rows, `[ep_size * max_tokens, hidden]` BF16,
                in the order the dispatch received them. Rows in `moe_alltoall_combine_rows`
                are read where they are; others are first copied there.
        """
        return workspace.combine(final_hidden_states)

    def reference(
        self,
        rank: int,
        per_rank_partials: Sequence[torch.Tensor],
        per_rank_sources: Sequence[Sequence[Sequence[int]]],
        local_tokens: int,
    ) -> torch.Tensor:
        """Rank `rank`'s combined rows, in fp32: each of its token's rows summed over the ranks
        that received it.

        Collective, so it takes the group's view: `per_rank_partials[d][i]` is the row rank `d`
        finalized for `per_rank_sources[d][rank][i]`, the index of the row of rank `rank` it
        received.
        """
        hidden = per_rank_partials[0].shape[-1]
        out = torch.zeros(local_tokens, hidden, dtype=torch.float32)
        for partials, sources in zip(per_rank_partials, per_rank_sources):
            for i, token in enumerate(sources[rank]):
                out[token] += partials[i].float()
        return out


def moe_alltoall_combine_rows(workspace, max_tokens: int, hidden_size: int) -> torch.Tensor:
    """The rows the next combine on `workspace` reads, `[ep_size * max_tokens, hidden_size]`
    BF16 in the communicator's workspace, `max_tokens` the largest count any rank sent in the
    round the last dispatch on `workspace` opened. An expert kernel that writes its finalized
    rows here, in the order the dispatch received them, saves `moe_alltoall_combine` a copy of
    every row. The rows hold nothing until the kernel writes them, and only until the combine
    reads them."""
    return workspace.get_combine_payload_tensor_in_workspace(
        max_tokens, hidden_size, torch.bfloat16
    )


def moe_alltoall_gathered_statistic(workspace) -> Optional[torch.Tensor]:
    """Every rank's histogram, `[ep_size, eplb_num_experts]`, that the last dispatch on
    `workspace` carried; None when it carried none. `EplbLayer.gathered` takes it."""
    return workspace.get_eplb_gathered_statistics()


moe_alltoall_dispatch = MoeAlltoallDispatch()
moe_alltoall_combine = MoeAlltoallCombine()
