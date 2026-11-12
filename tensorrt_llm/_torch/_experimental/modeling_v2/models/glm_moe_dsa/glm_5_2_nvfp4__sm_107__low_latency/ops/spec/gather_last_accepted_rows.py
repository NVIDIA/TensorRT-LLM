# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP draft step 0's row selection -- each request's last accepted row, and its index -- in one Triton launch."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._gather_last_accepted_rows import gather_last_accepted_rows as _gather_last_accepted_rows


class GlmGatherLastAcceptedRows(OpWrapper):
    """Each request's last accepted row of `rows` (and of `second`), and the row's index.

    Request `r` selects `g[r] = batch_indices[r] * rows_per_request + num_accepted[r] - 1 + row_offset`. Draft
    step 0 of the MTP loop runs the layer on every verified row of a request (`K + 1` of them), but only the row of
    the request's last accepted token feeds the LM head and the next step. Returns `(g, rows[g])`, or
    `(g, rows[g], second[g])`: `g` int64 `[R]`, the rows fresh `[R, n]` tensors.
    """

    ARCHS = frozenset({Arch.SM_107})

    # At c1 one request verifies K + 1 = 8 rows and step 0 selects one: after the post-attention all-reduce, the
    # normalized hidden rows and the residual together (the MoE, its all-reduce and the LM head then run on that
    # row), or the target's hidden rows alone where no row select applies. Generation-only batches only, so the
    # row offset (the context tokens ahead of them) is 0.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 step-0 row select: hidden and residual rows [8, 6144], one request",
            spec=dict(requests=1, rows_per_request=8, cols=6144, second=True, accepted=(3,)),
        ),
        Cell(
            why="c1 step-0 gather of the target hidden rows alone, every draft accepted",
            spec=dict(requests=1, rows_per_request=8, cols=6144, second=False, accepted=(8,)),
        ),
        Cell(
            why="four requests in a permuted batch order, first rows accepted",
            spec=dict(
                requests=4,
                rows_per_request=8,
                cols=6144,
                second=True,
                accepted=(1, 8, 4, 2),
                batch_order=(2, 0, 3, 1),
            ),
        ),
    )

    note = """
    The kernel reads `batch_indices` and `num_accepted` as dense vectors and does not bound `g`: a count outside
    [1, rows_per_request] reads another request's row or memory outside `rows`.
    """

    def __call__(
        self,
        rows: torch.Tensor,
        batch_indices: torch.Tensor,
        num_accepted: torch.Tensor,
        rows_per_request: int,
        row_offset: int,
        second: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return _gather_last_accepted_rows(
            rows, batch_indices, num_accepted, rows_per_request, row_offset, second
        )

    def reference(
        self,
        rows: torch.Tensor,
        batch_indices: torch.Tensor,
        num_accepted: torch.Tensor,
        rows_per_request: int,
        row_offset: int,
        second: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """The draft loop's index chain and its row gathers."""
        gather_ids = batch_indices.long() * rows_per_request + num_accepted.long() - 1 + row_offset
        if second is None:
            return gather_ids, rows[gather_ids]
        return gather_ids, rows[gather_ids], second[gather_ids]

    def is_valid(
        self,
        rows: torch.Tensor,
        batch_indices: torch.Tensor,
        num_accepted: torch.Tensor,
        rows_per_request: int,
        row_offset: int,
        second: Optional[torch.Tensor] = None,
    ) -> None:
        # The kernel addresses rows through stride(0) with unit column stride, and the index vectors as dense
        # arrays; any other layout is read from the wrong elements.
        assert rows.dim() == 2 and rows.stride(1) == 1, "rows must be 2D with unit column stride"
        assert second is None or (second.dim() == 2 and second.stride(1) == 1), (
            "second must be 2D with unit column stride"
        )
        # The kernel copies rows.shape[1] columns of both tensors: a wider second keeps garbage in its tail.
        assert second is None or second.shape[1] == rows.shape[1], (
            "second must have the columns of rows"
        )
        assert batch_indices.dim() == 1 and batch_indices.is_contiguous(), (
            "batch_indices must be a contiguous vector"
        )
        assert num_accepted.dim() == 1 and num_accepted.is_contiguous(), (
            "num_accepted must be a contiguous vector"
        )
        assert num_accepted.shape[0] >= batch_indices.shape[0], "one accepted count per request"
        # The selected row is not bounded on the device.
        count = num_accepted[: batch_indices.shape[0]]
        assert bool(((count >= 1) & (count <= rows_per_request)).all()), (
            "num_accepted must lie in [1, rows_per_request]"
        )
        gather_ids = batch_indices.long() * rows_per_request + count.long() - 1 + row_offset
        assert bool(((gather_ids >= 0) & (gather_ids < rows.shape[0])).all()), (
            "the selected rows must lie inside rows"
        )


glm_gather_last_accepted_rows = GlmGatherLastAcceptedRows()
