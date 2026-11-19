# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 tensor-parallel all-gather along any dim, through the engine `allgather` function."""

from typing import List, Optional, Sequence

import torch

from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

#: Kimi K3.1's DSA indexer top-k (index_topk): the width of each gathered row of selected indices.
_INDEX_TOPK = 2048
#: Query-token count from which the indexer splits a prefill's queries across the TP group
#: (the default q_split_threshold).
_Q_SPLIT_THRESHOLD = 8192


def _q_split_sizes(count: int, world: int) -> tuple[int, ...]:
    """The per-rank query counts the indexer's prefill split hands each rank (rank order)."""
    return tuple((r + 1) * count // world - r * count // world for r in range(world))


class AllgatherModule(OpWrapper):
    """Concatenate every TP rank's `input` along `dim`, in rank order, as the engine `allgather` does.

    `mapping` names the TP group (`mapping.tp_group`, this rank `mapping.tp_rank`). `sizes` is
    `None` when every rank holds the same shape, or each rank's `input.shape[dim]` in rank order;
    ranks may differ only along `dim`. Returns a new tensor on every rank, identical across the
    group; `input` is not written. A one-rank group returns `input` itself.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 indexer prefill top-k rows, {count} queries split over TP4 "
                f"({'even' if len(set(_q_split_sizes(count, 4))) == 1 else 'uneven'} sizes)",
                spec=dict(
                    world=4,
                    shape=(count, _INDEX_TOPK),
                    dim=0,
                    dtype=torch.int32,
                    sizes=_q_split_sizes(count, 4),
                ),
            )
            for count in (_Q_SPLIT_THRESHOLD, _Q_SPLIT_THRESHOLD + 3, 16387, 32769)
        ),
        *(
            Cell(
                why=f"K3.1 DFlash2 {what}, TP4 vocab shards: [{gens}, {block}, {width}] fp32 on dim -1",
                spec=dict(
                    world=4, shape=(gens, block, width), dim=-1, dtype=torch.float32, sizes=None
                ),
            )
            for what, gens, block, width in (
                ("log-normalizer", 1, 5, 1),
                ("log-normalizer", 16, 8, 1),
                ("(id, value) candidate pairs", 1, 5, 16),
                ("(id, value) candidate pairs", 16, 5, 32),
                ("(id, value) candidate pairs", 64, 8, 32),
                ("log-normalizer", 64, 8, 1),
                ("top-32 (id, value) candidate pairs", 1, 5, 64),
                ("top-32 (id, value) candidate pairs", 16, 8, 64),
            )
        ),
    )
    note = """
    Engine-function-internal: this entry calls
    `tensorrt_llm._torch.distributed.ops.allgather` exactly as the target does.
    That function views `input` as rows of everything but `dim`, all-gathers
    the rows with the trtllm allgather op (NCCL all-gather for `sizes=None`,
    a group of broadcasts for ragged `sizes`; the `_pg` symbol when MPI is
    disabled), then splits by rank and concatenates along `dim`. A gather is
    a copy, so the result is bit-exact against the concatenation.

    Kimi K3.1 calls it in two places. The DSA indexer gathers its int32
    top-k index rows ([rows, 2048], dim 0, ragged `sizes`) when a prefill of
    at least q_split_threshold (8192) query tokens is split across the
    attention-TP group (never under attention DP); the per-rank counts differ
    by at most one. The DFlash2 drafter, when its LM head is vocab-sharded,
    gathers fp32 per-position log-normalizers ([gens, block, 1]) and
    interleaved (id, value) top-k candidate pairs ([gens, block, 2 * top_k])
    on dim -1 with `sizes=None`. The candidate width is twice the drafter's
    selector top-k: the shipped DFlash2 selector keeps the top 32 (width 64);
    the cells also take 8 and 16.

    Not driven: groups wider than one GB300 tray (attention TP8 and wider),
    the MPI-disabled `_pg` path, and CUDA-graph capture. The trtllm
    allgather op allocates its output; under a captured graph the gathered
    tensor is the graph's buffer.

    ARCHS lists only what was driven: the SM_107 cells ran on four VR200 GPUs.
    """

    def __call__(
        self,
        input: torch.Tensor,
        mapping: Mapping,
        dim: int,
        sizes: Optional[List[int]],
    ) -> torch.Tensor:
        return allgather(input, mapping, dim=dim, sizes=sizes)

    def reference(self, inputs_in_rank_order: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
        """What every rank should get back.

        The documented collective exception to mirroring `__call__`: the output
        is not a function of what the calling rank holds, so this takes every
        rank's `input`, in rank order, and no mapping or sizes (the inputs'
        own extents along `dim` are the sizes).
        """
        return torch.cat([x.contiguous() for x in inputs_in_rank_order], dim=dim)


allgather_module = AllgatherModule()
