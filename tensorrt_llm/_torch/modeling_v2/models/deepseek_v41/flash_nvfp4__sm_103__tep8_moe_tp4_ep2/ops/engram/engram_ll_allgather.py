# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-gather of a small `[rows, width]` shard along the last dimension in one launch (peer memory, no NCCL).

The engine's `LowLatencyAllGather` (`tensorrt_llm/_torch/distributed/ll_allgather.py`), which stock DeepSeek-V4.1 uses
for the Engram head shards and the WKV column shards at decode sizes."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def ll_allgather_workspace(mapping):
    """The engine's `LowLatencyAllGather` of `mapping`'s tensor-parallel group, or None where peer memory is
    unavailable. A setup helper (not an entry): collective over the group, called once by every rank at load time, never
    under graph capture."""
    from tensorrt_llm._torch.distributed.ll_allgather import LowLatencyAllGather

    return LowLatencyAllGather.create(mapping)


def ll_allgather_takes(gather, rows: int, width: int, dtype: torch.dtype) -> bool:
    """Whether `engram_ll_allgather` takes a contiguous `[rows, width]` shard of `dtype` (a setup-free predicate the
    target uses to pick its path; it reads only shapes, so every rank of the group decides alike)."""
    if gather is None or rows <= 0:
        return False
    size = dtype.itemsize
    return (width * size) % 4 == 0 and rows * width * size // 4 <= gather.max_shard_words


class EngramLlAllgather(OpWrapper):
    """`(gather, x [rows, width])` -> `[rows, nranks * width]`: every rank's shard side by side in rank order, bit for
    bit (what `torch.distributed` all-gather along the last dim returns).

    `gather` is the engine's `LowLatencyAllGather` of the tensor-parallel group, created once at load time (collective)
    and kept by the target; it owns the peer buffers and the launch epoch."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: the Engram head shard (3 heads of 256) of six rows",
            spec=dict(rows=6, width=768, nranks=1),
        ),
        Cell(
            why="tep8 decode: the WKV column shard (25600 / 8) of six rows",
            spec=dict(rows=6, width=3200, nranks=1),
        ),
        Cell(
            why="tep8 verify of four requests: 24 rows of the WKV shard",
            spec=dict(rows=24, width=3200, nranks=1),
        ),
    )
    note = """
    A collective: every rank of the group must issue the same gathers in the same order on one stream (the epoch is
    shared by all of them). The single-GPU test drives a one-rank group (its own buffer as the only peer), which runs
    the same kernel and protocol; the multi-rank exchange is the engine's and is exercised by the model run.
    """

    def __call__(self, gather, x: torch.Tensor) -> torch.Tensor:
        out = gather.gather(x)
        assert out is not None, "the gather refused a shard that ll_allgather_takes admitted"
        return out

    def reference(self, gather, x: torch.Tensor) -> torch.Tensor:
        """One rank's view: the reference needs the other shards; the test attaches them as `gather.reference_shards`
        (a list in rank order), else the group is this rank alone."""
        shards = getattr(gather, "reference_shards", None) or [x]
        return torch.cat([s.reshape(x.shape) for s in shards], dim=-1)

    def is_valid(self, gather, x: torch.Tensor) -> None:
        assert x.is_cuda and x.dim() == 2 and x.is_contiguous(), (
            "a contiguous [rows, width] CUDA shard"
        )
        assert ll_allgather_takes(gather, x.shape[0], x.shape[1], x.dtype), (
            "shard larger than the peer buffers (or rows not 4-byte aligned): the caller keeps NCCL for it"
        )


engram_ll_allgather = EngramLlAllgather()
