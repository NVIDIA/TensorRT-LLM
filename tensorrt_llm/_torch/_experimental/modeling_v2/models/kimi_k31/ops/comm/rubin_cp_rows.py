# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row split of a context-parallel (cep) step, the gather of its rows and the all-to-all its
KDA layers exchange.

A cep context server runs Ulysses context parallelism over N ranks: every rank sees the whole
batch (replicated metadata, ``num_tokens`` rows), and CP rank ``r`` runs rows
``[r * rows, (r + 1) * rows)`` of it through every layer, ``rows = ceil(num_tokens / world)``. The
split is padded to ``rows * world`` rows: the padding rows of the last ranks start at zero and stay
zero through every layer. :func:`cp_rows` computes the split (a runtime helper, host only).

``rubin_cp_allgather`` brings every rank's rows back together over the CP group, in row order.

``rubin_cp_alltoall`` is the exchange of the KDA layers' Ulysses step: each rank hands rank ``j``
one equal-shape chunk (the KDA inputs of rank ``j``'s heads for this rank's rows, or the KDA output
of this rank's heads for rank ``j``'s rows) and gets back one chunk from every rank, in rank
order.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch.distributed.ops import alltoall_helix, cp_allgather


class CpRows(NamedTuple):
    """One rank's rows of a context-parallel step.

    ``rows`` is the per-rank row count of the padded split, the same on every rank; this rank's
    live rows are ``[begin, end)`` of the step (``end - begin`` may be smaller than ``rows``, or 0
    on the last ranks of a short step).
    """

    world: int
    rank: int
    num_tokens: int
    rows: int
    begin: int
    end: int

    @property
    def live(self) -> int:
        """This rank's live rows: ``end - begin``."""
        return self.end - self.begin

    @property
    def padded_tokens(self) -> int:
        """Rows of the padded split: ``rows * world``."""
        return self.rows * self.world


def cp_rows(num_tokens: int, world: int, rank: int) -> CpRows:
    """The row split of a step of ``num_tokens`` rows over ``world`` ranks, seen from ``rank``."""
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"rank {rank} of a group of {world}")
    if num_tokens < 0:
        raise ValueError(f"{num_tokens} rows")
    rows = -(-num_tokens // world)
    begin = min(rank * rows, num_tokens)
    end = min(begin + rows, num_tokens)
    return CpRows(world, rank, num_tokens, rows, begin, end)


def cp_row_counts(num_tokens: int, world: int) -> tuple[int, ...]:
    """Every rank's live rows of the split, in rank order."""
    return tuple(cp_rows(num_tokens, world, r).live for r in range(world))


# A cep rank's Ulysses chunk widths, by N: its 96 / N KDA heads x head dim, and the projections'
# row [q | k | v | g] of those heads with their beta logits, zero columns to 16 bytes.
_KDA_P = {n: 96 // n * 128 for n in (8, 16)}
_KDA_PACK = {n: -(-(4 * p + 96 // n) // 8) * 8 for n, p in _KDA_P.items()}


class RubinCpAllToAll(OpWrapper):
    """Hand rank ``j`` the chunk ``chunks[j]`` and return ``[world, *chunk]``: row ``i`` is what
    rank ``i`` handed this rank.

    ``chunks`` holds one contiguous chunk per rank of ``group`` (global ranks, in group order), all
    of one shape and dtype. A group of one returns ``chunks[0]`` with a leading axis of 1.
    """

    ARCHS = frozenset()
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"cep{n} KDA Ulysses, {what}: {rows} rows per rank, BF16",
            spec=dict(world=n, shape=(rows, width[n]), dtype=torch.bfloat16),
        )
        for n in (8, 16)
        for what, width in (
            (f"the projections of one rank's {96 // n} heads to their owner", _KDA_PACK),
            (
                f"the recurrence output of one rank's {96 // n} heads back to the rows' owner",
                _KDA_P,
            ),
        )
        for rows in (1, 513, 8192)
    )
    note = """
    Engine-function-internal: this entry calls
    `tensorrt_llm._torch.distributed.ops.alltoall_helix` with one list of `world`
    chunks, the trtllm alltoall_helix op (an NCCL group of send / recv pairs). An
    exchange is a copy, so the result is bit-exact against the reference.

    Not driven yet: ARCHS is empty. The cells need 8 or 16 ranks; the CPU test drives the
    reference only. A run on an 8- or 16-rank VR200 allocation (2 or 4 hosts) certifies them.
    """

    def __call__(self, chunks: Sequence[torch.Tensor], group: Sequence[int]) -> torch.Tensor:
        if len(group) == 1:
            return chunks[0].unsqueeze(0)
        return alltoall_helix([c.contiguous() for c in chunks], list(group))[0]

    def reference(self, chunks_in_rank_order: Sequence[Sequence[torch.Tensor]], rank: int):
        """What rank ``rank`` should get back.

        The documented collective exception to mirroring ``__call__``: the output is not a function
        of what the calling rank holds, so this takes every rank's ``chunks`` in rank order.
        """
        return torch.stack([list(chunks)[rank] for chunks in chunks_in_rank_order])


rubin_cp_alltoall = RubinCpAllToAll()


# The rows a cep rank gathers, by what they are: (width, dtype).
_GATHERED = {
    "the final hidden states": (7168, torch.bfloat16),
    "the new MLA latent rows": (576, torch.bfloat16),
    "the new FP8 index keys (as bytes)": (128, torch.uint8),
    "the new index keys' FP32 scales": (1, torch.float32),
}


class RubinCpAllGather(OpWrapper):
    """Concatenate every CP rank's ``input`` along dim 0, in CP rank order: ``[world * rows, ...]``.

    ``mapping`` names the CP group (``mapping.cp_group``, this rank ``mapping.cp_rank``). Every rank
    holds the same shape (the padded row split). A one-rank group returns ``input`` itself.
    """

    ARCHS = frozenset()
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"cep{n}, {what}: {rows} rows per rank",
            spec=dict(world=n, shape=(rows, width), dtype=dtype),
        )
        for n in (8, 16)
        for what, (width, dtype) in _GATHERED.items()
        for rows in (1, 513, 8192)
    )
    note = """
    Engine-function-internal: this entry calls
    `tensorrt_llm._torch.distributed.ops.cp_allgather` on dim 0 with equal sizes
    (`sizes=None`), the trtllm allgather op over the CP group. A gather is a copy,
    so the result is bit-exact against the concatenation.

    Not driven yet: ARCHS is empty. The cells need 8 or 16 ranks; the CPU test drives the
    reference only. A run on an 8- or 16-rank VR200 allocation (2 or 4 hosts) certifies them.
    """

    def __call__(self, input: torch.Tensor, mapping) -> torch.Tensor:
        if mapping.cp_size == 1:
            return input
        return cp_allgather(input, mapping, 0, None)

    def reference(self, inputs_in_rank_order: Sequence[torch.Tensor], mapping) -> torch.Tensor:
        """What every rank should get back; ``mapping`` is ignored.

        The documented collective exception to mirroring ``__call__``: the output is not a function
        of what the calling rank holds, so this takes every CP rank's ``input`` in rank order.
        """
        return torch.cat(list(inputs_in_rank_order), dim=0)


rubin_cp_allgather = RubinCpAllGather()
