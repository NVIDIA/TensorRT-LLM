# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`hidden[rows]`: a row lookup in the hidden-width (6144) bf16 states, one CuTe DSL launch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import hidden_rmsnorm_add_h6144_bf16 as _kernel

HIDDEN = _kernel.HIDDEN


@dataclass(frozen=True)
class CompiledHiddenRowGather:
    """The kernel compiled for each index dtype the op takes."""

    int32: Callable
    int64: Callable


def compile() -> CompiledHiddenRowGather:  # noqa: A001 -- the kernel module's own name
    """Compile both index dtypes; return the handle the op takes as `compiled`. Call once
    at load."""
    return CompiledHiddenRowGather(
        int32=_kernel.compile_gather(torch.int32), int64=_kernel.compile_gather(torch.int64)
    )


def takes(hidden: torch.Tensor, rows: torch.Tensor) -> bool:
    """Whether the op takes these operands: `hidden` CUDA bf16 `[N, 6144]` with a unit
    inner stride and 16-byte aligned rows, `rows` a contiguous 1-D int32 or int64 index.
    Anything else is the caller's to send to `torch.nn.functional.embedding`."""
    return (
        _kernel.supports(hidden)
        and rows.dtype in (torch.int32, torch.int64)
        and rows.ndim == 1
        and rows.is_contiguous()
    )


class CuteDslHiddenRowGather(OpWrapper):
    """`hidden[rows]` as a new compact `[len(rows), 6144]` bf16, bitwise.

    What the target's spec path hands the logits head: the hidden states of the
    rows whose logits the step needs (`spec_metadata.gather_ids`), the lookup
    upstream spells `nn.functional.embedding(gather_ids, hidden)`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # A decode step gathers every row (MTP 2 per sequence); a context step the last
    # row of each prompt plus the generation rows. `gather_ids` is int32 upstream;
    # int64 is the other index type the kernel is built for.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row", spec=dict(n=1, rows=1, dtype=torch.int32)),
        Cell(
            why="a full decode batch, every row once", spec=dict(n=64, rows=64, dtype=torch.int32)
        ),
        Cell(
            why="a context step: a few rows out of thousands, repeats allowed",
            spec=dict(n=7168, rows=40, dtype=torch.int32),
        ),
        Cell(why="int64 indices", spec=dict(n=300, rows=17, dtype=torch.int64)),
    )

    note = """
    The target compiles both index dtypes at load and takes this op wherever
    `takes` holds; every other operand goes to `nn.functional.embedding`. The
    kernel checks dtypes and layout on the host and raises; it does not read the
    index values there, see `is_valid`.
    """

    def __call__(
        self, hidden: torch.Tensor, rows: torch.Tensor, compiled: CompiledHiddenRowGather
    ) -> torch.Tensor:
        fn = compiled.int64 if rows.dtype is torch.int64 else compiled.int32
        return _kernel.gather_rows(fn, hidden, rows)

    def reference(
        self, hidden: torch.Tensor, rows: torch.Tensor, compiled: CompiledHiddenRowGather
    ) -> torch.Tensor:
        """`hidden[rows]`. `compiled` is ignored."""
        return hidden[rows.long()]

    def is_valid(
        self, hidden: torch.Tensor, rows: torch.Tensor, compiled: CompiledHiddenRowGather
    ) -> None:
        """Reads the index back to the host: never on the serving path."""
        # The kernel reads row `rows[i]` with no bound: an index past the states
        # copies whatever memory follows them.
        if rows.numel():
            lo, hi = int(rows.min()), int(rows.max())
            assert 0 <= lo and hi < hidden.shape[0], (
                f"rows must index the {hidden.shape[0]} hidden rows, got [{lo}, {hi}]"
            )


cute_dsl_hidden_row_gather = CuteDslHiddenRowGather()
