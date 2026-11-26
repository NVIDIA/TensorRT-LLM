# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm of hidden-width (6144) bf16 rows outside the fused all-reduce, one CuTe DSL launch;
and `hidden_rmsnorm`, that launch with FlashInfer's RMSNorm as its fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm

from ...kernels.common import hidden_rmsnorm_add_h6144_bf16 as _kernel
from .._build import Count, build_or_reason

HIDDEN = _kernel.HIDDEN


def compile() -> Callable:  # noqa: A001 -- the kernel module's own name for this step
    """Compile the kernel; return the handle the op takes as `compiled`. Call once at load.

    `eps` is a runtime argument, so one handle serves every norm. Cached by the
    kernel module, so a second call is free; the first costs one CuTe DSL compile.
    """
    return _kernel.compile_rmsnorm()


def takes(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the op takes these operands: `x` CUDA bf16 `[M >= 1, 6144]` with a unit
    inner stride and 16-byte aligned rows, `weight` contiguous bf16 `[6144]`. Anything
    else is the caller's to send to the stock norm."""
    return (
        _kernel.supports(x)
        and weight.dtype is torch.bfloat16
        and weight.is_contiguous()
        and tuple(weight.shape) == (HIDDEN,)
    )


class CuteDslHiddenRmsnorm(OpWrapper):
    """`bf16((x * rsqrt(mean(x^2) + eps)) * weight)` per row, as a new compact `[M, 6144]`.

    `x` is `[M, 6144]` bf16 with any row pitch that keeps rows 16-byte aligned (a
    column slice of a wider buffer included). fp32 accumulation and an approximate
    rsqrt, in FlashInfer's RMSNorm's operation order; only the order of the
    sum of squares differs, which can move a result by one bf16 ulp.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The norms the target runs outside the all-reduce: layer 0's input norm at
    # every step's row count, and the MTP module's enorm / hnorm / norm1 / head
    # norm at its draft rows. Decode is 1..64 rows; context steps up to 8192.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one row", spec=dict(rows=1, pitch=HIDDEN)),
        Cell(why="the full decode batch, 32 sequences x MTP 2", spec=dict(rows=64, pitch=HIDDEN)),
        Cell(
            why="a column slice of a wider row: the row pitch is a runtime value",
            spec=dict(rows=33, pitch=HIDDEN + 512),
        ),
        Cell(why="a ragged context step", spec=dict(rows=777, pitch=HIDDEN)),
        Cell(why="max_num_tokens", spec=dict(rows=8192, pitch=HIDDEN)),
    )

    note = """
    The target compiles one handle at load and takes this op wherever `takes`
    holds; every other operand goes to FlashInfer's RMSNorm at the call site.
    The kernel checks the operands on the host on every call and raises.
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, compiled: Callable
    ) -> torch.Tensor:
        return _kernel.rmsnorm(compiled, x, weight, eps)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, compiled: Callable
    ) -> torch.Tensor:
        """In fp64, rounded once to bf16. `compiled` is ignored."""
        xd = x.double()
        inv = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
        return (xd * inv * weight.double()).to(torch.bfloat16)


cute_dsl_hidden_rmsnorm = CuteDslHiddenRmsnorm()


@dataclass(frozen=True)
class HiddenRmsnormHandle:
    """What `hidden_rmsnorm` launches: the compiled kernel (None: not built, `missing` says
    why) and the route counter (`hidden_norm`)."""

    compiled: Callable | None
    count: Count
    missing: str = ""


def build(count: Count, enabled: bool = True) -> HiddenRmsnormHandle:
    """Compile the kernel into `hidden_rmsnorm`'s handle; one that fails keeps FlashInfer's
    norm. `enabled=False` builds nothing (every call then takes the fallback)."""
    if not enabled:
        return HiddenRmsnormHandle(None, count, "not built")
    compiled, missing = build_or_reason("hidden RMSNorm", compile)
    return HiddenRmsnormHandle(compiled, count, missing)


class HiddenRmsnorm(OpWrapper):
    """RMSNorm of hidden-width rows: `cute_dsl_hidden_rmsnorm` where the handle's kernel built
    and takes the operands (`takes`), else FlashInfer's RMSNorm. Counted as `hidden_norm`."""

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, handle: HiddenRmsnormHandle
    ) -> torch.Tensor:
        hit = handle.compiled is not None and takes(x, weight)
        handle.count("hidden_norm", hit)
        if hit:
            return cute_dsl_hidden_rmsnorm(x, weight, eps, handle.compiled)
        return flashinfer_rmsnorm(x, weight, eps)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, handle: HiddenRmsnormHandle
    ) -> torch.Tensor:
        """`cute_dsl_hidden_rmsnorm`'s reference. `handle` is ignored."""
        return cute_dsl_hidden_rmsnorm.reference(x, weight, eps, None)


hidden_rmsnorm = HiddenRmsnorm()
