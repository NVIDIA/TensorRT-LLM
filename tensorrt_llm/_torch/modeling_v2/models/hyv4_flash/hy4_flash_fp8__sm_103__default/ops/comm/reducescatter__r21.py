# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks and keep this rank's slice via the trtllm reducescatter op.

Target copy of ``catalog/comm/reducescatter.py`` with cells at this checkpoint's width.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Dtypes `getDtypeMap` (`cpp/tensorrt_llm/common/opUtils.cpp`) maps to a native NCCL
#: float or int type, so `ncclSum` is a real sum of the values.
_SUMMABLE_DTYPES = frozenset(
    {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.uint8,
    }
)


class Reducescatter(OpWrapper):
    """Sum `input` across every rank in `group`, then keep this rank's slice.

    Every rank passes the same full shape; the sum is split along dim 0 in ascending
    rank order. `sizes` gives each slice's height, or is `None` for an even split.
    The result is a fresh tensor.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank after the split, hidden 3072",
            spec=dict(rows_per_rank=1, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="64-row end: a decode batch's per-rank rows, hidden 3072",
            spec=dict(rows_per_rank=64, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk: 5120 rows per rank, hidden 3072",
            spec=dict(rows_per_rank=5120, trailing=(3072,), dtype=torch.bfloat16, sizes=None),
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
        """`process_group` None runs the MPI symbol; a boxed c10d ProcessGroup runs `reducescatter_pg`."""
        if process_group is None:
            return torch.ops.trtllm.reducescatter(input, sizes, group)
        return torch.ops.trtllm.reducescatter_pg(input, sizes, group, process_group)

    def reference(
        self,
        contributions_to_this_slice: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """The sum of the block each rank sent to this rank's slice, in ascending rank order.

        Accumulated in fp32 (int64 for integer dtypes) and rounded once.
        """
        first = contributions_to_this_slice[0]
        acc_dtype = torch.float32 if first.dtype.is_floating_point else torch.int64
        acc = torch.zeros(first.shape, dtype=acc_dtype, device=first.device)
        for block in contributions_to_this_slice:
            acc += block.to(acc_dtype)
        return acc.to(first.dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        assert input.dim() >= 1, "a 0-d input segfaults inside ReducescatterOp::run_list"
        assert input.is_contiguous(), "input must be contiguous; a strided view is reduced wrong"
        # fp8 and bool map to ncclInt8 and are summed as raw bytes; unmapped dtypes throw
        # inside an open NCCL group, which corrupts every later collective.
        assert input.dtype in _SUMMABLE_DTYPES, (
            f"{input.dtype} is not summable by this op; reduce in one of "
            f"{sorted(str(d) for d in _SUMMABLE_DTYPES)} and cast afterwards"
        )
        if sizes is None:
            assert input.shape[0] % len(group) == 0, (
                f"input.shape[0]={input.shape[0]} must be divisible by len(group)={len(group)} "
                "when sizes is None; the remainder is silently dropped"
            )
        else:
            assert len(sizes) == len(group), (
                f"len(sizes)={len(sizes)} must equal len(group)={len(group)}; a mismatch wedges "
                "or returns garbage"
            )
            assert sum(sizes) == input.shape[0], (
                f"sum(sizes)={sum(sizes)} must equal input.shape[0]={input.shape[0]}"
            )


reducescatter = Reducescatter()
