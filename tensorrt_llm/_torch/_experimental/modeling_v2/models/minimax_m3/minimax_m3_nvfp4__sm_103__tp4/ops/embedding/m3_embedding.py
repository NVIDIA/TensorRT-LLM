# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generic token-embedding gather over the replicated table (an id outside the table gives a zero row)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.embedding import pre_comm_embedding_ops


class M3Embedding(OpWrapper):
    """``table[ids]`` with a zero row for an id outside ``[0, V)`` (the speculative sampler pads with -1).

    The table is the full (replicated) bf16 embedding [V, 6144]; ids int32 / int64 of any count. The fused decode
    embedding (``m3_embedding_firstnorm``) takes decode steps of up to 8 rows; this entry takes every other step.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a prefill chunk, int32 ids", spec=dict(rows=2048, id_dtype=torch.int32, pad=False)
        ),
        Cell(
            why="a 48-row decode verify, int64 ids",
            spec=dict(rows=48, id_dtype=torch.int64, pad=False),
        ),
        Cell(
            why="draft rows padded with -1 ids", spec=dict(rows=32, id_dtype=torch.int32, pad=True)
        ),
    )

    note = """
    An exact gather: `reference` must match bit for bit.
    """

    def __call__(self, ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        return pre_comm_embedding_ops(ids, table, 1, 0, None, 0, table.shape[0], False, 0)

    def reference(self, ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        valid = (ids >= 0) & (ids < table.shape[0])
        rows = table[torch.where(valid, ids, torch.zeros_like(ids)).long()]
        return rows * valid.unsqueeze(-1).to(table.dtype)


m3_embedding = M3Embedding()
