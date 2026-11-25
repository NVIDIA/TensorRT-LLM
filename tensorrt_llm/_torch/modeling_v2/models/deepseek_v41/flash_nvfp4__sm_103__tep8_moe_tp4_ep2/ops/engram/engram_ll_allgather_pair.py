# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-gather of two equally shaped `[rows, width]` shards in one launch (peer memory, no NCCL), optionally on a capped
grid.

The engine's `LowLatencyAllGather.gather_pair` (`tensorrt_llm/_torch/distributed/ll_allgather.py`), which stock
DeepSeek-V4.1 uses at TP4 / TP8 on sm_103 for both Engram layers at once: the two head-shard lookups on the model
stream, then the two WKV column shards on the Engram stream with the grid capped (`_EngramKvPair.launch_on`,
`_ENGRAM_WKV_GATHER_SIDE_MAX_PROGRAMS = 16`)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Grid cap of the pair gather that spins on the Engram stream beside the layers (stock's
# `_ENGRAM_WKV_GATHER_SIDE_MAX_PROGRAMS`): the layers' all-reduce boundaries (and their resident PDL dependents) need
# free SMs on every rank while it waits for its peers, or the ranks deadlock. Uncapped, the gather takes one program per
# 128 words up to every SM.
ENGRAM_SIDE_MAX_PROGRAMS = 16


def ll_allgather_pair_takes(gather, rows: int, width: int, dtype: torch.dtype) -> bool:
    """Whether `engram_ll_allgather_pair` takes two contiguous `[rows, width]` shards of `dtype` (the engine's
    `supports_pair` on shapes: both together fit one shard slot). A setup-free predicate; it reads only shapes, so every
    rank of the group decides alike."""
    if gather is None or rows <= 0:
        return False
    size = dtype.itemsize
    return (width * size) % 4 == 0 and 2 * (rows * width * size // 4) <= gather.max_shard_words


class EngramLlAllgatherPair(OpWrapper):
    """`(gather, x0, x1 [rows, width], max_programs=None)` -> `(gather(x0), gather(x1))`, each `[rows, nranks * width]`
    with every rank's shard side by side in rank order, bit for bit, from one launch. `max_programs` caps the grid as
    stock's Engram-stream WKV pair gather does (`pair_programs(..., max_programs=cap)`, then `gather_pair` on it); the
    programs then each move more blocks, the result is the same.

    `gather` is the engine's `LowLatencyAllGather` of the tensor-parallel group, created once at load time (collective)
    and kept by the target; it owns the peer buffers and the launch epoch."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: both Engram layers' head shards (3 heads of 256) of six rows, model stream",
            spec=dict(rows=6, width=768, nranks=1),
        ),
        Cell(
            why="tep8 decode: both WKV column shards (25600 / 8) of six rows, Engram stream, grid capped at 16",
            spec=dict(rows=6, width=3200, nranks=1, max_programs=ENGRAM_SIDE_MAX_PROGRAMS),
        ),
        Cell(
            why="tep8 verify of three requests: both WKV shards of 18 rows under the cap (the widest pair)",
            spec=dict(rows=18, width=3200, nranks=1, max_programs=ENGRAM_SIDE_MAX_PROGRAMS),
        ),
        Cell(
            why="tep8 verify of four requests: both head shards of 24 rows, model stream",
            spec=dict(rows=24, width=768, nranks=1),
        ),
    )
    note = """
    A collective: every rank of the group must issue the same gathers in the same order (the epoch is shared by all of
    them, single and pair alike). The single-GPU test drives a one-rank group (its own buffer as the only peer), which
    runs the same kernel and protocol; the multi-rank exchange is the engine's and is exercised by the model run.
    """

    def __call__(
        self, gather, x0: torch.Tensor, x1: torch.Tensor, max_programs: Optional[int] = None
    ):
        if max_programs is not None:  # as stock: the launch's grid for these shards under the cap
            max_programs = gather.pair_programs(x0, x1, max_programs=max_programs)
        out = gather.gather_pair(x0, x1, max_programs=max_programs)
        assert out is not None, (
            "the gather refused a shard pair that ll_allgather_pair_takes admitted"
        )
        return out

    def reference(
        self, gather, x0: torch.Tensor, x1: torch.Tensor, max_programs: Optional[int] = None
    ):
        """One rank's view: the reference needs the other shards; the test attaches them as `gather.reference_shards`
        (a list in rank order of `(x0, x1)` pairs), else the group is this rank alone."""
        pairs = getattr(gather, "reference_shards", None) or [(x0, x1)]
        return tuple(torch.cat([p[i].reshape(x0.shape) for p in pairs], dim=-1) for i in range(2))

    def is_valid(
        self, gather, x0: torch.Tensor, x1: torch.Tensor, max_programs: Optional[int] = None
    ) -> None:
        for x in (x0, x1):
            assert x.is_cuda and x.dim() == 2 and x.is_contiguous(), (
                "contiguous [rows, width] CUDA shards"
            )
        assert x0.shape == x1.shape and x0.dtype == x1.dtype, "two equally shaped shards"
        assert ll_allgather_pair_takes(gather, x0.shape[0], x0.shape[1], x0.dtype), (
            "the pair is larger than one shard slot of the peer buffers: the caller gathers the shards one by one"
        )
        assert max_programs is None or max_programs >= 1, "a grid cap is at least one program"


engram_ll_allgather_pair = EngramLlAllgatherPair()
