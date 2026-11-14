# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding on a K3-owned table.

:class:`EmbeddingTable` holds ``weight`` ``[rows, hidden]``: the whole vocabulary, or with a
vocab-parallel mapping (``tp_size > 1``, no attention DP) rank ``r``'s rows
``[r * ceil(V / tp), min((r + 1) * ceil(V / tp), V))``, zero-padded to ``ceil(V / tp)`` rows. Its
forward is

    hit = (ids >= start) & (ids < end)                  # start, end: the rows this rank holds
    out = F.embedding(hit * (ids - start * hit), weight)
    out.masked_fill_(~hit, 0)
    out = all_reduce(out)                               # vocab-parallel, unless the caller reduces

so an id outside the table (the rejection sampler pads with -1) embeds to zeros and every vocab
row comes from exactly one rank. :func:`gather_embedding` is the one-launch lookup of a whole table
(``ops/embedding/k3_embedding_gather.py``, bitwise the masked lookup).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from ..comm.sm107_allreduce import TpAllReduce


def masked_embedding(ids: torch.Tensor, weight: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Rows ``ids - start`` of ``weight`` where ``start <= ids < end``, zeros elsewhere."""
    hit = (ids >= start) & (ids < end)
    out = F.embedding(hit * (ids - start * hit), weight)
    out.masked_fill_(~hit.unsqueeze(-1), 0)
    return out


def gather_embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """:func:`masked_embedding` of a whole table in one Triton launch (PDL where enabled)."""
    from .k3_embedding_gather import k3_embedding_gather

    return k3_embedding_gather(ids, weight)


class EmbeddingTable(nn.Module):
    """``weight`` of one rank and the lookup above."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        vocab_parallel: bool = False,
        reduce_output: bool = True,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        mapping = mapping or Mapping()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.vocab_parallel = (
            vocab_parallel and mapping.tp_size > 1 and not mapping.enable_attention_dp
        )
        if self.vocab_parallel:
            rows = math.ceil(num_embeddings / mapping.tp_size)
            self.vocab_start = mapping.tp_rank * rows
            self.vocab_end = min(self.vocab_start + rows, num_embeddings)
            self.all_reduce = (
                TpAllReduce(mapping, dtype or torch.bfloat16, allreduce_strategy)
                if reduce_output
                else None
            )
        else:
            rows = num_embeddings
            self.vocab_start, self.vocab_end = 0, num_embeddings
            self.all_reduce = None
        self.weight = nn.Parameter(
            torch.empty(rows, embedding_dim, dtype=dtype, device=device), requires_grad=False
        )

    def load(self, weight: torch.Tensor) -> None:
        """Copy this rank's rows of the full checkpoint table (zero padding rows past the vocab)."""
        rows = self.vocab_end - self.vocab_start
        self.weight.data[:rows].copy_(weight[self.vocab_start : self.vocab_end])
        self.weight.data[rows:].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = masked_embedding(
            ids,
            self.weight,
            self.vocab_start,
            self.vocab_end if self.vocab_parallel else self.weight.shape[0],
        )
        if self.all_reduce is not None:
            out = self.all_reduce(out)
        return out
