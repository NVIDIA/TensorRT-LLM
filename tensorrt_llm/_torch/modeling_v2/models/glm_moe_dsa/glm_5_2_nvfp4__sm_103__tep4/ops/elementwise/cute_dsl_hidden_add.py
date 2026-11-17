# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The residual add of hidden-width (6144) bf16 rows, one CuTe DSL launch."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import hidden_rmsnorm_add_h6144_bf16 as _kernel

HIDDEN = _kernel.HIDDEN


def compile() -> Callable:  # noqa: A001 -- the kernel module's own name for this step
    """Compile the kernel; return the handle the op takes as `compiled`. Call once at load."""
    return _kernel.compile_add()


def takes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the op takes these operands: both CUDA bf16 `[M >= 1, 6144]` of one shape,
    unit inner stride, 16-byte aligned rows. Anything else is the caller's to send to
    `torch.add`."""
    return _kernel.supports(a, b) and a.shape == b.shape


class CuteDslHiddenAdd(OpWrapper):
    """`bf16(float(a) + float(b))`, a new compact `[M, 6144]`: bitwise `torch.add` on bf16.

    Either operand may be a column slice of a wider buffer (a runtime row pitch per
    operand).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The MTP module's two output adds (routed + shared, then the residual) at its
    # draft rows: decode 1..64, context steps up to 8192.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one row", spec=dict(rows=1, pitch_a=HIDDEN, pitch_b=HIDDEN)),
        Cell(why="the full decode batch", spec=dict(rows=64, pitch_a=HIDDEN, pitch_b=HIDDEN)),
        Cell(
            why="each operand a column slice with its own row pitch",
            spec=dict(rows=37, pitch_a=HIDDEN + 64, pitch_b=HIDDEN + 512),
        ),
        Cell(why="a context step", spec=dict(rows=4100, pitch_a=HIDDEN, pitch_b=HIDDEN)),
    )

    note = """
    The target compiles one handle at load and takes this op wherever `takes`
    holds; every other pair goes to `torch.add` at the call site. The kernel
    checks the operands on the host on every call and raises.
    """

    def __call__(self, a: torch.Tensor, b: torch.Tensor, compiled: Callable) -> torch.Tensor:
        return _kernel.add(compiled, a, b)

    def reference(self, a: torch.Tensor, b: torch.Tensor, compiled: Callable) -> torch.Tensor:
        """`torch.add`. `compiled` is ignored."""
        return torch.add(a, b)


cute_dsl_hidden_add = CuteDslHiddenAdd()
