# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-step embedding gather and the first layer's input RMSNorm in one CuTe DSL launch, into caller buffers."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..embedding_firstnorm.op import DecodeEmbedding, m3_embed_norm

__all__ = ["DecodeEmbedding", "m3_embedding_firstnorm"]

VOCAB = 200064  # MiniMax-M3's vocabulary; the decode table is the full (replicated) embedding


class M3EmbeddingFirstnorm(OpWrapper):
    """Write ``table[ids]`` into ``raw`` (bf16 [M, 6144], a zero row for an id outside ``[0, V)``) and
    ``raw * rsqrt(mean(raw^2) + eps) * weight`` into ``out``; returns ``(out, raw)`` -- the first layer's normed input
    and its residual. ``ids`` int32 or int64 [M], M = 1..8; ``table`` bf16 [V, 6144] contiguous; ``weight`` bf16
    [6144]. The first call per (M, id dtype, table shape, PDL) compiles (``DecodeEmbedding`` does it at build)."""

    ARCHS = frozenset({Arch.SM_103})

    # The target builds every (rows 1..8, id dtype) owner; the kernel is one CTA per row, so the rows differ only in
    # grid size: the ends of the range and both id dtypes.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, int32 ids", spec=dict(rows=1, id_dtype=torch.int32)),
        Cell(why="eight requests, int32 ids", spec=dict(rows=8, id_dtype=torch.int32)),
        Cell(why="one request, int64 ids", spec=dict(rows=1, id_dtype=torch.int64)),
        Cell(
            why="six rows (one request verifying a draft), int64 ids",
            spec=dict(rows=6, id_dtype=torch.int64),
        ),
        # Every other row count the decode dispatch takes (dispatch.EMBED_FIRSTNORM), both id dtypes.
        *(
            Cell(
                why=f"{rows} rows: a decode row count between the ones above",
                spec=dict(rows=rows, id_dtype=torch.int32 if rows % 2 else torch.int64),
            )
            for rows in (2, 3, 4, 5, 7)
        ),
    )

    note = """
    The gathered rows are exact. The norm is flashinfer's RMSNorm arithmetic in its reduction order with a fast
    rsqrt, where `reference` sums in fp64 and takes torch's rsqrt: within the band the test derives, not bit for bit.
    An id outside [0, V) gives a zero row and a zero norm (torch's embedding would raise). No accepted input is known
    to compute wrongly: the host side rejects any other dtype, shape or a strided buffer.
    """

    def __call__(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        raw: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return m3_embed_norm(ids, table, weight, eps, raw, out, use_pdl)

    def reference(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        raw: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """New tensors (``raw`` / ``out`` are not written); `use_pdl` is ignored."""
        valid = (ids >= 0) & (ids < table.shape[0])
        rows = table[torch.where(valid, ids, torch.zeros_like(ids)).long()] * valid.unsqueeze(1).to(
            table.dtype
        )
        x = rows.double()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x * rstd * weight.double()).to(torch.bfloat16), rows


m3_embedding_firstnorm = M3EmbeddingFirstnorm()
