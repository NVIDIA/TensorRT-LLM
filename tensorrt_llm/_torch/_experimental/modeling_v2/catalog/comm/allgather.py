# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class _Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 at dep4 is the only caller and always passes `sizes=None`:
    # it pads every rank to a common height first, so the ragged form the op
    # also supports is not something a shipped target reaches. Row counts span
    # what an attention-DP rank feeds its expert-parallel MoE call, from one
    # decode token to a full prefill chunk.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank, hidden 7168",
            spec=dict(rows=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch, the common case between decode and prefill",
            spec=dict(rows=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
    )

    note = """
    The ragged form (`sizes` as an explicit per-rank vector) and dtypes other
    than bf16 work and are exercised by the op matrix, but no shipped target
    reaches them, so they are not cells.

    CUDA graph behaviour -- capture at every engine batch size, replays
    surviving eager calls of other shapes, the captured sizes vector being held
    -- is not a cell axis either: it is about the op across capture and replay
    rather than about an input configuration, and the matrix carries it as its
    own named checks.
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
    ) -> torch.Tensor:
        return torch.ops.trtllm.allgather(input, sizes, group)

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        sizes: Optional[List[int]] = None,
        group: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The concatenation the collective is supposed to make.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds, so a reference given
        only `input` could not state anything at all. It takes the group's
        inputs in ascending rank order instead. The matrix builds that list on
        one rank from the seed convention every rank shares, which is what keeps
        the reference arithmetic rather than a second collective.
        """
        return torch.cat(list(per_rank_inputs), dim=0)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """A gate of exactly zero: the op moves bytes and computes nothing.

        Tightened from the base default rather than loosened. Any difference at
        all is a wrong gather.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        if out.dtype is torch.float8_e4m3fn:
            # torch.testing cannot compare float8; for a pure data move the byte
            # pattern is the honest gate anyway.
            out, ref = out.view(torch.uint8), ref.view(torch.uint8)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption or
        # a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "AllgatherOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently gathered from the wrong elements"
        )
        assert sizes is None or len(sizes) == len(group), (
            f"len(sizes)={len(sizes) if sizes is not None else None} must equal "
            f"len(group)={len(group)}; the op sizes its output from `sizes` alone, "
            "so a short list silently drops the trailing ranks and a long one "
            "appends uninitialized rows"
        )


allgather = _Allgather()
