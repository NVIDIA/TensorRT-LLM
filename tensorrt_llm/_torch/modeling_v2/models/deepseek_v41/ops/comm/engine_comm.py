# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine's tensor-parallel collectives as the DeepSeek-V4.1 targets call them (`tensorrt_llm/_torch/distributed`):
the `AllReduce` module (its AUTO strategy: MNNVL / one-shot / two-shot / NCCL by size and topology, as stock) and the
`allgather` function (rows of every rank, ragged by `sizes`). The module is built once at load time
(`engine_allreduce_create`) and called through `engine_allreduce`; nothing about the call changes."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def engine_allreduce_create(mapping, strategy=None, dtype: Optional[torch.dtype] = None):
    """The engine's `AllReduce` module of `mapping`'s tensor-parallel group (`strategy` None: the module's default),
    a setup helper: built once per layer at construction, it owns its workspaces."""
    from tensorrt_llm._torch.distributed import AllReduce

    kwargs = {}
    if strategy is not None:
        kwargs["strategy"] = strategy
    if dtype is not None:
        kwargs["dtype"] = dtype
    return AllReduce(mapping=mapping, **kwargs)


class EngineAllreduce(OpWrapper):
    """`(allreduce, x)` -> the sum of `x` over the tensor-parallel group, `x`'s shape and dtype, by the engine's
    `AllReduce` module (`engine_allreduce_create`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 / dep2 decode: a [6, 5120] bf16 partial sum",
            spec=dict(world=4, rows=6, width=5120),
        ),
        Cell(
            why="an engram lookup of 160 rows, 24 heads x 256",
            spec=dict(world=4, rows=160, width=6144),
        ),
        Cell(why="a prefill chunk", spec=dict(world=4, rows=1280, width=5120)),
    )
    note = """
    A collective: the reference needs the other ranks' inputs; the test attaches them as `allreduce.reference_inputs`
    (rank order), else the group is this rank alone. Band: the engine sums in fp32 and rounds once to bf16.
    """

    def __call__(self, allreduce, x: torch.Tensor) -> torch.Tensor:
        return allreduce(x)

    def reference(self, allreduce, x: torch.Tensor) -> torch.Tensor:
        parts = getattr(allreduce, "reference_inputs", None) or [x]
        return sum(p.double() for p in parts).to(x.dtype)


class EngineAllgather(OpWrapper):
    """`(x, mapping, dim, sizes)` -> every tensor-parallel rank's `x` concatenated along `dim` in rank order (rank `r`
    contributes `sizes[r]` rows when `sizes` is given), by the engine's `allgather`. Moves bits: exact."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="engram hash rows of four ranks, ragged (attention DP)",
            spec=dict(world=4, sizes=(5, 0, 37, 160), width=24),
        ),
        Cell(why="equal rows, no sizes", spec=dict(world=4, sizes=None, rows=6, width=3200)),
    )
    note = "A collective: the test attaches the group's inputs as `mapping.reference_inputs` for the reference."

    def __call__(
        self, x: torch.Tensor, mapping, dim: int = 0, sizes: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        from tensorrt_llm._torch.distributed import allgather

        return allgather(x, mapping, dim=dim, sizes=sizes)

    def reference(
        self, x: torch.Tensor, mapping, dim: int = 0, sizes: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        parts = getattr(mapping, "reference_inputs", None) or [x]
        return torch.cat(list(parts), dim=dim)


class EngineAlltoallvTokens(OpWrapper):
    """`(x, counts, group, rank)` -> `[len(group), counts[rank], width]`: every member of `group` (global ranks) holds
    the rows `x [sum(counts), width]` of all the group's tokens in member order (member `j` owns `counts[j]` of them);
    member `rank` (its index in `group`) receives, from each member in order, the slice of its own tokens. The engine's
    `trtllm::alltoallv_tokens` (NCCL), as stock's attention-DP Engram sends each rank its tokens' rows of every table
    shard (`DeepseekV41Engram._local_token_embeddings`). Moves bits: exact."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="engram rows of a 4-rank exchange group, ragged (attention DP, one rank idle)",
            spec=dict(world=4, counts=(5, 0, 37, 160), width=1536),
        ),
        Cell(
            why="a prefill-sized step: 4096 rows per rank",
            spec=dict(world=4, counts=(4096, 4096, 4096, 4096), width=1536),
        ),
    )
    note = (
        "A collective over `group`: the test attaches the members' "
        "inputs to `x` as `x.reference_inputs` for the reference."
    )

    def __call__(
        self, x: torch.Tensor, counts: Sequence[int], group: Sequence[int], rank: int
    ) -> torch.Tensor:
        return torch.ops.trtllm.alltoallv_tokens(
            x, [int(c) for c in counts], [int(g) for g in group], int(rank)
        )

    def reference(
        self, x: torch.Tensor, counts: Sequence[int], group: Sequence[int], rank: int
    ) -> torch.Tensor:
        """Member `rank`'s rows of every member's input (`x.reference_inputs`,
        else `x` for all), stacked in member order."""
        parts = list(getattr(x, "reference_inputs", None) or [x] * len(group))
        start = sum(int(c) for c in counts[:rank])
        return torch.stack([p[start : start + int(counts[rank])] for p in parts])

    def is_valid(
        self, x: torch.Tensor, counts: Sequence[int], group: Sequence[int], rank: int
    ) -> None:
        assert x.dim() == 2 and x.is_contiguous() and x.shape[0] == sum(int(c) for c in counts)
        assert (
            len(counts) == len(group)
            and 0 <= rank < len(group)
            and all(int(c) >= 0 for c in counts)
        )


engine_allreduce = EngineAllreduce()
engine_allgather = EngineAllgather()
engine_alltoallv_tokens = EngineAlltoallvTokens()
