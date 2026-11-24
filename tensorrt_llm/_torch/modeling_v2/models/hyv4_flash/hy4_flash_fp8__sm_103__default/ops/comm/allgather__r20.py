# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor via the trtllm allgather op.

Target copy of ``catalog/comm/allgather.py`` with cells at this checkpoint's width.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is `None`
    when every rank holds the same number of rows. The result is a fresh tensor.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank, trailing 3072",
            spec=dict(rows=1, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="64-row end: a padded batch, trailing 3072",
            spec=dict(rows=64, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk: 5120 rows, trailing 3072",
            spec=dict(rows=5120, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
    )

    note = """
    The cells hold the even split (`sizes=None`, an equalized decode step); the ragged
    split an eager attention-DP step passes is covered by the op matrix, not a cell.
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> torch.Tensor:
        """`process_group` None runs the MPI symbol; a boxed c10d ProcessGroup runs `allgather_pg`."""
        if process_group is None:
            return torch.ops.trtllm.allgather(input, sizes, group)
        return torch.ops.trtllm.allgather_pg(input, sizes, group, process_group)

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        sizes: Optional[List[int]] = None,
        group: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The concatenation, from the group's inputs in ascending rank order."""
        return torch.cat(list(per_rank_inputs), dim=0)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        assert input.dim() >= 1, "a 0-d input segfaults inside AllgatherOp::run_list"
        assert input.is_contiguous(), "input must be contiguous; a strided view is gathered wrong"
        assert sizes is None or len(sizes) == len(group), (
            f"len(sizes)={len(sizes) if sizes is not None else None} must equal "
            f"len(group)={len(group)}; the op sizes its output from `sizes` alone"
        )


allgather = Allgather()
