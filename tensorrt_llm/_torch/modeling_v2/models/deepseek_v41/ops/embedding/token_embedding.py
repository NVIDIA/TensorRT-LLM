# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The token embedding lookup of the DeepSeek-V4.1 targets: the engine's `Embedding` module (replicated table, one row
per token id), called unchanged. `token_embedding_create(..., masked_kernel=True)` builds it as stock's
`DeepseekV4Model` does (`use_masked_embedding_kernel=True`: the lookup is one CuTe DSL launch on SM100/103/107)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class TokenEmbedding(OpWrapper):
    """`(embedding, input_ids)` -> `[n, hidden]`: row `input_ids[i]` of the table of the engine's `Embedding` module
    `embedding` (not tensor-parallel: the whole vocabulary on every rank). Moves bits: exact."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="dep8 c7000 decode: 160 tokens", spec=dict(n=160, vocab=129280, hidden=5120)),
        Cell(why="tep8 decode: six tokens", spec=dict(n=6, vocab=129280, hidden=5120)),
        Cell(why="a prefill chunk", spec=dict(n=1280, vocab=129280, hidden=5120)),
        Cell(
            why="tep8 decode, stock's masked-embedding kernel (rejected drafts as -1)",
            spec=dict(n=6, vocab=129280, hidden=5120, masked_kernel=True, rejected=2),
        ),
        Cell(
            why="dep8 c7000 decode, stock's masked-embedding kernel",
            spec=dict(n=160, vocab=129280, hidden=5120, masked_kernel=True),
        ),
    )

    def __call__(self, embedding, input_ids: torch.Tensor) -> torch.Tensor:
        return embedding(input_ids)

    def reference(self, embedding, input_ids: torch.Tensor) -> torch.Tensor:
        """Row `input_ids[i]`; ids outside the table (the -1 a rejected draft token is padded with) read zeros."""
        ids = input_ids.long()
        inside = (ids >= 0) & (ids < embedding.weight.shape[0])
        rows = embedding.weight[torch.where(inside, ids, 0)]
        return torch.where(inside[:, None], rows, torch.zeros_like(rows))

    def is_valid(self, embedding, input_ids: torch.Tensor) -> None:
        assert getattr(embedding, "tp_size", 1) == 1, (
            "a replicated table (the targets' token embedding)"
        )
        assert input_ids.dtype in (torch.int32, torch.int64)


def token_embedding_create(
    vocab: int, hidden: int, dtype: torch.dtype = torch.bfloat16, masked_kernel: bool = False
):
    """The engine's replicated `Embedding(vocab, hidden)` module (the table the target loads), a setup helper.
    `masked_kernel`: the lookup runs stock's masked-embedding CuTe DSL kernel (`use_masked_embedding_kernel`, as stock's
    `DeepseekV4Model` builds its `embed_tokens`) instead of the module's torch ops."""
    from tensorrt_llm._torch.modules.embedding import Embedding

    return Embedding(vocab, hidden, dtype=dtype, use_masked_embedding_kernel=masked_kernel)


token_embedding = TokenEmbedding()
