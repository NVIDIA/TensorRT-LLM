# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum across ranks and keep this rank's slice, via the trtllm reducescatter op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class _Reducescatter(OpWrapper):
    """Sum `input` across every rank in `group`, then keep this rank's slice.

    Every rank passes a tensor of the same full shape; the elementwise sum is
    split along dim 0 in ascending rank order and rank `i` of the group gets
    slice `i`. `sizes` gives each rank's slice height in that order, or is
    `None` when the split is even. The result is a freshly allocated tensor on
    every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 at dep4 is the only caller and always passes `sizes=None`:
    # it pads to a common height first. The row counts are the full heights an
    # attention-DP step reduces, which is the rank count times a slice.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank after the split, hidden 7168",
            spec=dict(rows_per_rank=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch",
            spec=dict(rows_per_rank=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows_per_rank=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
    )

    note = """
    The ragged form and dtypes other than bf16 work and are exercised by the op
    matrix, but no shipped target reaches them, so they are not cells. CUDA
    graph behaviour is carried by the matrix's own named checks rather than as
    a cell axis.
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
    ) -> torch.Tensor:
        return torch.ops.trtllm.reducescatter(input, sizes, group)

    def reference(
        self,
        contributions_to_this_slice: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """The sum of what every rank sent to this rank's slice.

        The documented exception to mirroring `__call__`, for the same reason as
        allgather: a collective's output is not a function of what the calling
        rank holds. It takes the one block each rank contributed to *this*
        position, in ascending rank order -- not their whole inputs, because a
        rank can regenerate exactly those WORLD blocks from the shared seed
        convention and never has to materialize anybody else's full tensor.

        Accumulated in fp32 (int64 for integer dtypes) and rounded once. The op
        sums in the input dtype in whatever order NCCL picks; on a cell's payload
        that costs nothing, because every partial sum there is exact in the input
        dtype either way -- which is what lets `compare` be bitwise.
        """
        first = contributions_to_this_slice[0]
        acc_dtype = torch.float32 if first.dtype.is_floating_point else torch.int64
        acc = torch.zeros(first.shape, dtype=acc_dtype, device=first.device)
        for block in contributions_to_this_slice:
            acc += block.to(acc_dtype)
        return acc.to(first.dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """A gate of exactly zero, which the cells earn rather than assume.

        This op computes, so NCCL's summation order and the reference's rank
        order could in general differ by rounding. They do not here, because a
        cell's payload is drawn from a value set -- multiples of 1/8, bounded --
        on which every partial sum is exactly representable in the input dtype.
        That makes the strong gate available, and taking it is the point: any
        difference at all is then a wrong reduction or a wrong slice, never
        rounding, and a band would have hidden both.

        A cell built from unrestricted random values would need a band instead.
        None is; see the matrix's payload builder.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption, a
        # wedge, or a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "ReducescatterOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently reduced over the wrong elements"
        )
        assert input.dtype is not torch.float8_e4m3fn, (
            "float8_e4m3fn is accepted by the op but summed as raw unsigned bytes, "
            "not as floats, so the result is meaningless; reduce in bf16/fp16/fp32 "
            "and quantize afterwards"
        )
        if sizes is None:
            assert input.shape[0] % len(group) == 0, (
                f"input.shape[0]={input.shape[0]} must be divisible by "
                f"len(group)={len(group)} when sizes is None; the op takes "
                "shape[0] // len(group) rows per rank and silently drops the remainder"
            )
        else:
            assert len(sizes) == len(group), (
                f"len(sizes)={len(sizes)} must equal len(group)={len(group)}; a short "
                "list raises on the highest rank and wedges the others, a long one "
                "reduces more rows than the input holds and returns garbage"
            )
            assert sum(sizes) == input.shape[0], (
                f"sum(sizes)={sum(sizes)} must equal input.shape[0]={input.shape[0]}; "
                "the op sizes the reduction from `sizes` alone, so a short split "
                "silently drops the trailing rows and a long one reads past the input"
            )


reducescatter = _Reducescatter()
