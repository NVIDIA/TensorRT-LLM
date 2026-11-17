# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`silu(gate) * up` over `[gate | up]` bf16 rows at any row count and width, one CuTe DSL launch."""

from __future__ import annotations

from typing import Callable

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import moe_sigmoid_topk_swiglu_rows as _kernel


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
