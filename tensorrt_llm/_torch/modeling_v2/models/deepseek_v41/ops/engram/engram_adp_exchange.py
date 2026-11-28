# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram lookups under attention DP through peer memory: the engine's `EngramAdpExchange`
(`tensorrt_llm/_torch/modules/engram/adp_exchange.py`), which stock DeepSeek-V4.1 runs on decode-sized engines.

Each rank of an exchange group holds contiguous heads of every Engram table and needs all heads of its own tokens. A
forward does three launches per group rank:

1. `engram_adp_exchange_hashes` (model stream, once per forward, every Engram layer at once): every rank pushes its
   row count and hash rows into every group rank's peer buffer and gathers theirs.
2. `engram_adp_push` (one launch per layer, may run on a side stream; never waits on another rank): this rank looks
   up its heads of every group token, dequantizing exactly as the table's lookup kernel, and stores each row straight
   into the requesting rank's receive buffer.
3. `engram_adp_receive` (model stream, one launch per layer, at the layer): waits for this rank's rows and copies them
   out as `[tokens, heads * dim]` bf16.

Each head of a token has one owner, so the result is the table's own lookup of all heads, bit for bit. No zero-fill,
no all-reduce, no rows that are thrown away (what an all-gather + local lookup + all-reduce + slice pays for).

The exchange object is created once at load time (`engram_adp_exchange_create`, collective over the tensor-parallel
group) and owns the peer buffers and the launch epoch; every rank of a group must issue the same exchanges, pushes and
receives in the same order (the decision to take the exchange, `engram_adp_takes`, reads only the token counts every
rank holds)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.engram.adp_exchange import ExchangeRound  # noqa: F401


def engram_adp_exchange_create(
    mapping, group_start: int, group_size: int, capacity: int, heads: int, dim: int, num_layers: int
):
    """The engine's `EngramAdpExchange` of the tensor-parallel ranks `[group_start, group_start + group_size)` that
    hold `mapping`'s rank, sized for `capacity` tokens per rank, or None where the group has no peer memory. A setup
    helper (not an entry): collective over the tensor-parallel
    group, once per rank at load time, never under capture."""
    from tensorrt_llm._torch.modules.engram.adp_exchange import EngramAdpExchange

    return EngramAdpExchange.create(
        mapping, group_start, group_size, capacity, heads, dim, num_layers
    )


def engram_adp_takes(exchange, counts: Optional[Sequence[int]]) -> bool:
    """Whether a forward with these per-rank token counts (every tensor-parallel rank's, in rank order) takes the
    exchange; every rank answers alike."""
    return exchange is not None and isinstance(counts, (list, tuple)) and exchange.takes(counts)


class EngramAdpExchangeHashes(OpWrapper):
    """`(exchange, hashes, counts) -> ExchangeRound`: every group rank's int64 `[count, heads]` hash rows of every
    Engram layer. `round.gathered` is `[group, layers, max(counts), heads]`; rows past a rank's count are unset.

    `hashes`: one `[counts[rank], heads]` int64 tensor per layer (unit column stride; any row stride). `counts`: the
    group's token counts in group-rank order."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 c7000 decode: 160 rows per rank (40 requests x 1 + 3 drafts), 2 layers of 24 heads",
            spec=dict(world=4, counts=(160, 160, 160, 160), heads=24, dim=256, layers=2),
        ),
        Cell(
            why="uneven ranks, one of them empty (attention-DP padding of an idle rank)",
            spec=dict(world=4, counts=(5, 0, 37, 150), heads=24, dim=256, layers=2),
        ),
        Cell(
            why="dep32 decode: 25 requests x (1 + 5 drafts) = 150 rows, at the capacity",
            spec=dict(world=4, counts=(150, 150, 150, 150), heads=24, dim=256, layers=2),
        ),
    )
    note = """
    A collective: the reference needs the other ranks' hashes; the test attaches the group's hash tensors to the
    exchange as `exchange.reference_hashes` (rank order, one tuple per rank), else the group is this rank alone.
    """

    def __call__(
        self, exchange, hashes: Sequence[torch.Tensor], counts: Sequence[int]
    ) -> "ExchangeRound":
        return exchange.exchange_hashes(hashes, counts)

    def reference(
        self, exchange, hashes: Sequence[torch.Tensor], counts: Sequence[int]
    ) -> torch.Tensor:
        """`[group, layers, max(counts), heads]` with the rows past each rank's count zero."""
        group = getattr(exchange, "reference_hashes", None) or [tuple(hashes)]
        out = torch.zeros(
            len(group),
            len(hashes),
            max(counts),
            hashes[0].shape[1],
            dtype=torch.int64,
            device=hashes[0].device,
        )
        for r, per_layer in enumerate(group):
            for layer, h in enumerate(per_layer):
                out[r, layer, : h.shape[0]] = h
        return out

    def is_valid(self, exchange, hashes: Sequence[torch.Tensor], counts: Sequence[int]) -> None:
        assert exchange.takes(counts), (
            "counts past the exchange's capacity (or all empty): the caller keeps NCCL"
        )
        assert len(counts) == exchange.nranks and len(hashes) == exchange.num_layers
        assert all(
            h.dtype == torch.int64
            and h.shape == (counts[exchange.rank], exchange.heads)
            and h.stride(1) == 1
            for h in hashes
        ), "one int64 [own count, heads] tensor per layer, unit column stride"


class EngramAdpPush(OpWrapper):
    """`(exchange, round_, layer_index, table) -> None`: looks up this rank's heads (`table`, the engine-layout head
    shard: `storage()`, `offsets`, `head_start`, `num_local_heads`, `block_size`) of every group token of layer
    `layer_index` and stores them into the requesters' receive buffers. Never waits on another rank, so it may run on
    a side stream; the effect is read by `engram_adp_receive` (whose reference states the values)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = EngramAdpExchangeHashes.CELLS
    note = "No output of its own: the test gates it through `engram_adp_receive`, bit for bit."

    def __call__(self, exchange, round_, layer_index: int, table) -> None:
        exchange.push(round_, layer_index, table)

    def reference(self, exchange, round_, layer_index: int, table) -> None:
        """Nothing visible on this rank (the rows land in the requesters' buffers)."""
        return None

    def is_valid(self, exchange, round_, layer_index: int, table) -> None:
        assert 0 <= layer_index < exchange.num_layers
        assert table.shard_heads and table.num_local_heads * exchange.nranks == exchange.heads, (
            "the table must be this rank's head shard of the exchange group"
        )
        assert (
            table.head_start == exchange.rank * table.num_local_heads
            and table.dtype == torch.bfloat16
        )
        assert table.embedding_dim == exchange.dim


class EngramAdpReceive(OpWrapper):
    """`(exchange, round_, layer_index) -> [own count, heads * dim]` bf16: this rank's rows of layer `layer_index`,
    every head, waiting until each owner has pushed them. The caller's stream must already be ordered after this
    rank's own push of the layer (an event), so no rank sits in a later collective while its push is pending."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = EngramAdpExchangeHashes.CELLS
    note = """
    The reference is the full (unsharded) table's lookup of this rank's own hashes; the test attaches one full table
    per layer to the exchange as `exchange.reference_tables`.
    """

    def __call__(self, exchange, round_, layer_index: int) -> torch.Tensor:
        return exchange.receive(round_, layer_index)

    def reference(self, exchange, round_, layer_index: int) -> torch.Tensor:
        table = exchange.reference_tables[layer_index]
        ids = round_.hashes[layer_index]
        return table(ids).reshape(ids.shape[0], table.num_heads * table.embedding_dim)

    def is_valid(self, exchange, round_, layer_index: int) -> None:
        assert (
            0 <= layer_index < exchange.num_layers
            and round_.local_count == round_.hashes[layer_index].shape[0]
        )


engram_adp_exchange_hashes = EngramAdpExchangeHashes()
engram_adp_push = EngramAdpPush()
engram_adp_receive = EngramAdpReceive()
