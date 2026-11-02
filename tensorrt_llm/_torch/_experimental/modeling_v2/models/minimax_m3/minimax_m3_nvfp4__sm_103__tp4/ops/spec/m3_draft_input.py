# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 draft pass's input in one launch: embedding lookup, both RMSNorms and the concat
(``m3_spec_tail.draft_input``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Largest hidden size the kernel's register tile holds (4 x 256 threads x 8 bf16).
MAX_HIDDEN = 8192


def rmsnorm_rows(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """bf16(x * rsqrt(mean(x^2) + eps) * w) per row, fp32 in between ((x * r) * w, as the kernels multiply)."""
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * r * weight.float()).to(torch.bfloat16)


class M3DraftInput(OpWrapper):
    """``out [M, 2H] = [rmsnorm(table[ids]) * w_embed, rmsnorm(hidden[rows]) * w_hidden]`` and ``residual [M, H] =
    hidden[rows]`` (rows default to 0..M-1) in one launch; ids outside the table embed to zero (the masked lookup of
    a vocab-local embedding). Returns ``(out, residual)``."""

    ARCHS = frozenset({Arch.SM_103})

    # Every pass of the fused draft loop (dispatch.DRAFT_INPUT) at draft_len 3, drafter hidden 6144, eps 1e-6: the
    # first pass of 1..8 requests (4..32 rows, the carried hidden read in place) and the later passes of 1..8 requests
    # (one row each; the second pass gathers the last accepted rows from the first pass's 4 rows per request).
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"first pass of {n} request(s): {4 * n} rows, hidden read in place",
                spec=dict(m=4 * n, h=6144, gather=False, invalid_id=False),
            )
            for n in range(1, 9)
        ),
        *(
            Cell(
                why=f"second pass of {n} request(s): the last accepted rows gathered from {4 * n} rows",
                spec=dict(m=n, h=6144, gather=True, invalid_id=False),
            )
            for n in range(1, 9)
        ),
        Cell(
            why="third pass of 3 requests: one row each, hidden read in place",
            spec=dict(m=3, h=6144, gather=False, invalid_id=False),
        ),
        Cell(
            why="one row with an out-of-table id: the embedding half is zero",
            spec=dict(m=1, h=6144, gather=False, invalid_id=True),
        ),
    )

    note = """
    The arithmetic is the in-tree RMSNorm's (fp32 statistic, (x * rsqrt) * w, one bf16 rounding); the statistic's
    summation order differs from FlashInfer's, which moves an output by one bf16 ulp now and then (the test's band,
    also held against the in-tree modules). The out-of-table id is the masked lookup of a vocab-local embedding; the
    drafter's embedding is the whole table, so its ids are always in range.
    """

    def __call__(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        hidden: torch.Tensor,
        rows: Optional[torch.Tensor],
        w_embed: torch.Tensor,
        w_hidden: torch.Tensor,
        eps_embed: float,
        eps_hidden: float,
        out: torch.Tensor,
        residual: Optional[torch.Tensor],
        pdl: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        torch.ops.trtllm.minimax_m3_draft_input(
            ids, table, hidden, rows, w_embed, w_hidden, eps_embed, eps_hidden, out, residual, pdl
        )
        return out, residual

    def reference(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        hidden: torch.Tensor,
        rows: Optional[torch.Tensor],
        w_embed: torch.Tensor,
        w_hidden: torch.Tensor,
        eps_embed: float,
        eps_hidden: float,
        out: torch.Tensor,
        residual: Optional[torch.Tensor],
        pdl: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The two normalized halves (fp32 statistic, one bf16 rounding) and the gathered hidden rows. ``out`` /
        ``residual`` only say where the op writes; ``pdl`` only shapes the launch."""
        valid = (ids >= 0) & (ids < table.shape[0])
        emb = rmsnorm_rows(table[torch.where(valid, ids, 0).long()], w_embed, eps_embed)
        emb = torch.where(valid[:, None], emb, torch.zeros_like(emb))
        hid = hidden[rows.long()] if rows is not None else hidden[: ids.shape[0]]
        ref = torch.cat([emb, rmsnorm_rows(hid, w_hidden, eps_hidden)], dim=-1)
        return ref, (hid.clone() if residual is not None else None)

    def is_valid(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        hidden: torch.Tensor,
        rows: Optional[torch.Tensor],
        w_embed: torch.Tensor,
        w_hidden: torch.Tensor,
        eps_embed: float,
        eps_hidden: float,
        out: torch.Tensor,
        residual: Optional[torch.Tensor],
        pdl: bool = True,
    ) -> None:
        h = table.shape[1]
        # Rows are read as hidden + rows[r] * H and table + id * H: both must be dense H-wide rows.
        assert table.is_contiguous() and hidden.is_contiguous(), (
            "table / hidden must be dense [*, H]"
        )
        if rows is not None:
            assert bool(((rows >= 0) & (rows < hidden.shape[0])).all()), "rows must index hidden"
        else:
            assert hidden.shape[0] >= ids.shape[0], "hidden must hold a row per id"
        # Vector loads of 16 bytes: H must be a multiple of 8 within the register tile, and the weights aligned.
        assert h % 8 == 0 and h <= MAX_HIDDEN, "H must be a multiple of 8, at most 8192"
        assert residual is None or (
            residual.is_contiguous() and residual.shape == (ids.shape[0], h)
        ), "residual must be contiguous [M, H]"


m3_draft_input = M3DraftInput()
