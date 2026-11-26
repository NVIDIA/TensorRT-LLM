# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Residual add then RMSNorm of hidden-width (6144) bf16 rows, one CuTe DSL launch: the
MTP module's output on an MTP3 / MTP5 decode step, bit for bit the two launches it fuses; and
`hidden_add_rmsnorm`, that launch with the two launches as its fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import hidden_rmsnorm_add_h6144_bf16 as _kernel
from .._build import Count, build_or_reason
from ..elementwise.cute_dsl_hidden_add import HiddenAddHandle, hidden_add
from .cute_dsl_hidden_rmsnorm import HiddenRmsnormHandle, hidden_rmsnorm

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


@dataclass(frozen=True)
class HiddenAddRmsnormHandle:
    """What `hidden_add_rmsnorm` launches: the fused kernel (None: not built, `missing` says
    why), the handles of the two launches it fuses (its fallback), and the route counter
    (`hidden_add_norm`)."""

    compiled: Callable | None
    add: HiddenAddHandle
    norm: HiddenRmsnormHandle
    count: Count
    missing: str = ""


def build(
    add: HiddenAddHandle, norm: HiddenRmsnormHandle, count: Count, enabled: bool = True
) -> HiddenAddRmsnormHandle:
    """Compile the fused kernel into `hidden_add_rmsnorm`'s handle. It serves only where both
    launches it fuses built their kernels (the pair it is bit for bit); otherwise, or with
    `enabled=False`, nothing is compiled and every call runs the two launches."""
    if not enabled:
        return HiddenAddRmsnormHandle(None, add, norm, count, "not built")
    if add.compiled is None or norm.compiled is None:
        return HiddenAddRmsnormHandle(None, add, norm, count, "the add or the norm did not build")
    compiled, missing = build_or_reason("hidden add + RMSNorm", compile)
    return HiddenAddRmsnormHandle(compiled, add, norm, count, missing)


class HiddenAddRmsnorm(OpWrapper):
    """`rmsnorm(add(a, b), weight)`; the sum is not kept. One `cute_dsl_hidden_add_rmsnorm`
    launch where the handle's kernel built and takes the operands, else `hidden_add` then
    `hidden_rmsnorm`. Counted as `hidden_add_norm`."""

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        handle: HiddenAddRmsnormHandle,
    ) -> torch.Tensor:
        hit = handle.compiled is not None and takes(a, b, weight)
        handle.count("hidden_add_norm", hit)
        if hit:
            return cute_dsl_hidden_add_rmsnorm(a, b, weight, eps, handle.compiled)
        return hidden_rmsnorm(hidden_add(a, b, handle.add), weight, eps, handle.norm)

    def reference(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        handle: HiddenAddRmsnormHandle,
    ) -> torch.Tensor:
        """`cute_dsl_hidden_add_rmsnorm`'s reference. `handle` is ignored."""
        return cute_dsl_hidden_add_rmsnorm.reference(a, b, weight, eps, None)


hidden_add_rmsnorm = HiddenAddRmsnorm()
