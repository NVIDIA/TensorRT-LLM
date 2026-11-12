# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-embedding lookup of a whole table as a vectorized row copy (Triton,
``kernels/k3r_spec/embedding_gather.py``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.embedding_gather import embedding_gather


class K3EmbeddingGather(OpWrapper):
    """Return the new ``[len(ids), hidden]`` tensor of rows ``ids`` of ``weight [vocab, hidden]``
    (1-D ``ids``, contiguous ``weight``); an id outside ``[0, vocab)`` gives a zero row."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The replicated table of Kimi K3 (vocab 163840, hidden 7168), rows: tokens of the decode step.
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{why}: {rows} rows of the replicated [163840, 7168] table",
                spec=dict(rows=rows, vocab=163840, hidden=7168, padded=False),
            )
            for why, rows in (
                ("helix16 ULL 1-request step", 8),
                ("helix8 / helix16 ULL 2-request step", 16),
                ("helix16 ULL 4-request step", 32),
                ("helix8 b16 mtp5 verify step", 96),
            )
        ),
        Cell(
            why="ids padded with -1 (rejection sampler): zero rows, 8 rows",
            spec=dict(rows=8, vocab=163840, hidden=7168, padded=True),
        ),
    )

    note = """
    A copy: bitwise F.embedding on in-range ids. One program per (token, 1024
    columns) with 16-byte accesses instead of the scalar index-select kernel;
    the launch shape depends only on the token count, so it is capturable after
    a warm-up per Triton specialization (dtype, divisibility). No metadata or
    state; ids and weight are read-only. The decode step's embedding where the
    target runs the speculative-decode head (helix8, helix16 ULL); the context
    worker uses F.embedding.
    """

    def __call__(self, ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return embedding_gather(ids, weight)

    def reference(self, ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``F.embedding`` with out-of-range ids mapped to a zero row."""
        valid = (ids >= 0) & (ids < weight.shape[0])
        rows = F.embedding(torch.where(valid, ids, torch.zeros_like(ids)), weight)
        return rows.masked_fill(~valid.unsqueeze(-1), 0)


k3_embedding_gather = K3EmbeddingGather()
