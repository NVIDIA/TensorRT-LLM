# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row gather / scatter of 2-D row blocks of any dtype, moved as raw bytes, in one CuTe DSL
launch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.elementwise import mtp_rows_select as _kernel

as_words = _kernel.as_words


@dataclass(frozen=True)
class CompiledRowsSelect:
    """The gather and scatter kernels `compile` built."""

    gather: Callable
    scatter: Callable


def compile() -> CompiledRowsSelect:  # noqa: A001 -- the kernel module's own name
    """Compile both directions once (row count, width and pitches are runtime values) and
    launch each once, so no step meets a first module load."""
    handle = CompiledRowsSelect(_kernel.compile_select(False), _kernel.compile_select(True))
    x = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
    rows = torch.tensor([3, 0], dtype=torch.int32, device="cuda")
    y = torch.empty(2, 64, dtype=torch.bfloat16, device="cuda")
    _kernel.gather(handle.gather, x, rows, y)
    _kernel.scatter(handle.scatter, y, rows, x)
    torch.cuda.synchronize()
    return handle


def takes(t: torch.Tensor) -> bool:
    """Whether `t` is a row block the kernel moves: CUDA, 2-D, unit inner stride, row width
    and pitch multiples of 16 bytes, 16-byte aligned."""
    return as_words(t) is not None


class CuteDslRowsSelect(OpWrapper):
    """`scatter=False`: `y[i] = x[rows[i]]`; `scatter=True`: `y[rows[i]] = x[i]` (rows of `y`
    not named are not touched). Returns `y`.

    `x` and `y` are 2-D row blocks of one width in bytes (`takes`); `rows` an int32 device
    vector of in-range row indices. Rows move as 16-byte vectors, so every dtype is copied
    bitwise. Indices are not range-checked on the device; `is_valid` checks them.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The MTP module's context step: the selected rows' e4m3 queries `[T, 16 x 576]`, the
    # bf16 residual rows `[T, 6144]` and the output scattered back; one row, and a 64-row set
    # (up to 256 at MTP3 / MTP5, whose mixed steps carry 4 / 6 generation rows per request).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gather 2 of 7300 fp8 query rows",
            spec=dict(src_rows=7300, sel=2, width=9216, dtype="float8", scatter=False),
        ),
        Cell(
            why="gather 64 bf16 hidden rows, repeated indices included",
            spec=dict(src_rows=900, sel=64, width=6144, dtype="bfloat16", scatter=False),
        ),
        Cell(
            why="scatter 32 bf16 hidden rows back into 7300",
            spec=dict(src_rows=7300, sel=32, width=6144, dtype="bfloat16", scatter=True),
        ),
        Cell(
            why="MTP3 / MTP5: gather 128 of 7300 fp8 query rows",
            spec=dict(src_rows=7300, sel=128, width=9216, dtype="float8", scatter=False),
        ),
        Cell(
            why="MTP5: gather 256 bf16 hidden rows, repeated indices included",
            spec=dict(src_rows=900, sel=256, width=6144, dtype="bfloat16", scatter=False),
        ),
        Cell(
            why="MTP5: scatter 256 bf16 hidden rows back into 7300",
            spec=dict(src_rows=7300, sel=256, width=6144, dtype="bfloat16", scatter=True),
        ),
    )

    note = """
    The target takes this op on the MTP module's context steps: the attention and everything
    after it run on the rows the draft head reads (each context request's last row and the
    generation rows), which are gathered from the full-row query and residual and scattered
    back into the full output. The kernel raises on operands it does not take.
    """

    def __call__(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        y: torch.Tensor,
        compiled: CompiledRowsSelect,
        scatter: bool = False,
    ) -> torch.Tensor:
        if scatter:
            _kernel.scatter(compiled.scatter, x, rows, y)
        else:
            _kernel.gather(compiled.gather, x, rows, y)
        return y

    def reference(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        y: torch.Tensor,
        compiled: CompiledRowsSelect,
        scatter: bool = False,
    ) -> torch.Tensor:
        """`y` as the op leaves it, as a new tensor (byte views, so any dtype is exact)."""
        out = y.clone()
        idx = rows.long()
        if scatter:
            out.view(torch.uint8)[idx] = x.view(torch.uint8)[: idx.numel()]
        else:
            out.view(torch.uint8)[: idx.numel()] = x.view(torch.uint8)[idx]
        return out

    def is_valid(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        y: torch.Tensor,
        compiled: CompiledRowsSelect,
        scatter: bool = False,
    ) -> None:
        """Reads the indices back to the host: never on the serving path."""
        assert takes(x) and takes(y), "x and y must be 2-D row blocks of 16-byte rows"
        assert x.shape[1] * x.element_size() == y.shape[1] * y.element_size(), "row widths differ"
        assert rows.dtype == torch.int32 and rows.dim() == 1, "rows must be an int32 vector"
        bound = int(y.shape[0]) if scatter else int(x.shape[0])
        if rows.numel():
            r = rows.cpu()
            assert int(r.min()) >= 0 and int(r.max()) < bound, f"a row index is outside 0..{bound}"
        n = int(x.shape[0]) if scatter else int(y.shape[0])
        assert rows.numel() <= n, f"{rows.numel()} rows do not fit {n}"


cute_dsl_rows_select = CuteDslRowsSelect()
