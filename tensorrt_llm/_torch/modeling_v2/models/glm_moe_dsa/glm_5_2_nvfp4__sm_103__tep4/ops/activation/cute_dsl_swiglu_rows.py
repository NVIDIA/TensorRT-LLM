# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`silu(gate) * up` over `[gate | up]` bf16 rows at any row count and width, one CuTe DSL launch;
and `swiglu_rows`, that launch with FlashInfer's `silu_and_mul` as its fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)

from ...kernels.common import moe_sigmoid_topk_swiglu_rows as _kernel
from .._build import Count, build_or_reason


def compile() -> Callable:  # noqa: A001 -- the kernel module's own name for this step
    """Compile the kernel; return the handle the op takes as `compiled`. Call once at load.
    Rows and width are symbolic: one handle serves every call."""
    return _kernel.compile_swiglu()


def takes(src: torch.Tensor) -> bool:
    """Whether the op takes `src`: contiguous bf16 `[T >= 1, 2W]`. Anything else is
    `flashinfer_silu_and_mul`'s."""
    return (
        src.dtype is torch.bfloat16
        and src.ndim == 2
        and src.shape[0] >= 1
        and src.shape[1] % 2 == 0
        and src.is_contiguous()
    )


class CuteDslSwigluRows(OpWrapper):
    """`out[r, c] = silu(src[r, c]) * src[r, W + c]` for `c < W`, a new bf16 `[T, W]`.

    The gate half first, the layout `flashinfer_silu_and_mul` reads. fp32 math,
    one rounding to bf16.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The shared expert's gate/up output at a row count its fused gate/up+SwiGLU
    # kernel has no runner for: W = 512 per rank on the trunk and the MTP layer.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row", spec=dict(rows=1, width=512)),
        Cell(why="a row count with no fused runner", spec=dict(rows=65, width=512)),
        Cell(why="a context chunk", spec=dict(rows=4100, width=512)),
        Cell(
            why="a width past one 2048-column CTA tile, not a multiple of it",
            spec=dict(rows=9, width=3072),
        ),
    )

    note = """
    The target compiles one handle at load and takes this op wherever `takes`
    holds; every other operand goes to `flashinfer_silu_and_mul` at the call site.
    The kernel checks the operand on the host and raises.
    """

    def __call__(self, src: torch.Tensor, compiled: Callable) -> torch.Tensor:
        return _kernel.swiglu(compiled, src)

    def reference(self, src: torch.Tensor, compiled: Callable) -> torch.Tensor:
        """In fp64, rounded once. `compiled` is ignored."""
        width = src.shape[1] // 2
        gate, up = src[:, :width].double(), src[:, width:].double()
        return (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)


cute_dsl_swiglu_rows = CuteDslSwigluRows()


@dataclass(frozen=True)
class SwigluRowsHandle:
    """What `swiglu_rows` launches: the compiled kernel (None: not built, `missing` says why)
    and the route counter (`swiglu_rows`)."""

    compiled: Callable | None
    count: Count
    missing: str = ""


def build(count: Count, enabled: bool = True) -> SwigluRowsHandle:
    """Compile the kernel into `swiglu_rows`'s handle; one that fails keeps FlashInfer's.
    `enabled=False` builds nothing."""
    if not enabled:
        return SwigluRowsHandle(None, count, "not built")
    compiled, missing = build_or_reason("SwiGLU rows", compile)
    return SwigluRowsHandle(compiled, count, missing)


class SwigluRows(OpWrapper):
    """`silu(gate) * up` over `[gate | up]` rows: `cute_dsl_swiglu_rows` where the handle's
    kernel built and takes `src`, else `flashinfer_silu_and_mul`. Counted as
    `swiglu_rows`."""

    def __call__(self, src: torch.Tensor, handle: SwigluRowsHandle) -> torch.Tensor:
        hit = handle.compiled is not None and takes(src)
        handle.count("swiglu_rows", hit)
        if hit:
            return cute_dsl_swiglu_rows(src, handle.compiled)
        return flashinfer_silu_and_mul(src)

    def reference(self, src: torch.Tensor, handle: SwigluRowsHandle) -> torch.Tensor:
        """`cute_dsl_swiglu_rows`'s reference. `handle` is ignored."""
        return cute_dsl_swiglu_rows.reference(src, None)


swiglu_rows = SwigluRows()
