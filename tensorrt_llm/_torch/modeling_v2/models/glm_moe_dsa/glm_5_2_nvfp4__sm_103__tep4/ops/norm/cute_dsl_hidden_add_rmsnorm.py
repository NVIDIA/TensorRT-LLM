# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Residual add then RMSNorm of hidden-width (6144) bf16 rows, one CuTe DSL launch: the
MTP module's output on an MTP3 / MTP5 decode step, bit for bit the two launches it fuses."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import hidden_rmsnorm_add_h6144_bf16 as _kernel

HIDDEN = _kernel.HIDDEN


def compile() -> Callable:  # noqa: A001 -- the kernel module's own name for this step
    """Compile the kernel; return the handle the op takes as `compiled`. Call once at load.

    `eps` is a runtime argument. Cached by the kernel module, so a second call is free; the
    first costs one CuTe DSL compile.
    """
    return _kernel.compile_add_rmsnorm()


def takes(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the op takes these operands: what `cute_dsl_hidden_add` takes for `(a, b)`
    (both CUDA bf16 `[M >= 1, 6144]` of one shape, unit inner stride, 16-byte aligned rows)
    and a contiguous bf16 `[6144]` `weight` on their device, which `cute_dsl_hidden_rmsnorm`
    takes for the sum. Anything else is the caller's to send to the two launches."""
    return (
        _kernel.supports(a, b)
        and a.shape == b.shape
        and a.device == b.device
        and weight.device == a.device
        and weight.dtype is torch.bfloat16
        and weight.is_contiguous()
        and tuple(weight.shape) == (HIDDEN,)
    )


class CuteDslHiddenAddRmsnorm(OpWrapper):
    """`cute_dsl_hidden_rmsnorm(cute_dsl_hidden_add(a, b), weight, eps)` in one launch, as a
    new compact `[M, 6144]`; the sum is not stored.

    Bit for bit the two kernels in sequence: each thread rounds its elements of `a + b` to
    bf16 exactly as the add kernel stores them and runs the norm kernel's operations on them
    in its order (fp32 accumulation, the same sum-of-squares order, `rsqrt.approx`). Against
    an fp64 reference of the norm of `torch.add(a, b)` that is the norm kernel's own bound.
    Either operand may be a column slice of a wider buffer (a runtime row pitch per operand).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The MTP module's residual add and head norm on an MTP3 / MTP5 decode step: 32 x 1
    # rows on its 1-token draft steps, 32 x 4 / 6 on its first, and any batch below 32
    # sequences; the odd counts and a column-sliced operand keep the row loop honest.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row", spec=dict(rows=1, pitch_a=HIDDEN, pitch_b=HIDDEN)),
        Cell(
            why="a 1-token draft step at 32 sequences",
            spec=dict(rows=32, pitch_a=HIDDEN, pitch_b=HIDDEN),
        ),
        Cell(why="65 rows, odd", spec=dict(rows=65, pitch_a=HIDDEN, pitch_b=HIDDEN)),
        Cell(
            why="MTP3's first draft step at 32 sequences, 32 x 4",
            spec=dict(rows=128, pitch_a=HIDDEN, pitch_b=HIDDEN),
        ),
        Cell(why="127 rows, odd", spec=dict(rows=127, pitch_a=HIDDEN, pitch_b=HIDDEN)),
        Cell(why="129 rows, odd", spec=dict(rows=129, pitch_a=HIDDEN, pitch_b=HIDDEN)),
        Cell(why="191 rows, odd", spec=dict(rows=191, pitch_a=HIDDEN, pitch_b=HIDDEN)),
        Cell(
            why="MTP5's first draft step at 32 sequences, 32 x 6",
            spec=dict(rows=192, pitch_a=HIDDEN, pitch_b=HIDDEN),
        ),
        Cell(
            why="each operand a column slice with its own row pitch",
            spec=dict(rows=37, pitch_a=HIDDEN + 64, pitch_b=HIDDEN + 512),
        ),
    )

    note = """
    The target compiles one handle at load on an engine drafting more than one token, and
    takes this op for the MTP module's output on its decode steps wherever `takes` holds and
    the add and norm kernels built (the pair it is bitwise); every other call keeps
    `cute_dsl_hidden_add` then `cute_dsl_hidden_rmsnorm`. The kernel checks the operands
    on the host on every call and raises.
    """

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        compiled: Callable,
    ) -> torch.Tensor:
        return _kernel.add_rmsnorm(compiled, a, b, weight, eps)

    def reference(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        compiled: Callable,
    ) -> torch.Tensor:
        """`torch.add` (the add kernel's bits), then the norm in fp64, rounded once to bf16.
        `compiled` is ignored."""
        xd = torch.add(a, b).double()
        inv = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
        return (xd * inv * weight.double()).to(torch.bfloat16)


cute_dsl_hidden_add_rmsnorm = CuteDslHiddenAddRmsnorm()
