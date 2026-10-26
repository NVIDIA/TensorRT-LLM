# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`x * alpha + beta`: the native-code fixture's op, an entry like any other.

The kernel is ``cpp/tensorrt_llm/modeling_v2/fixture/scaleAddSm10xDefault/``; the
op is reached through the target's ``native_ops`` namespace, which loading the
target package filled. Nothing here is specific to native code beyond that one name.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ... import native_ops


class ScaleAdd(OpWrapper):
    """`(x.float() * alpha + beta).to(x.dtype)`, elementwise, new tensor out.

    The product and the sum are each rounded to fp32 and the result once to
    `x`'s dtype; the kernel does the same with no fused multiply-add, so the two
    agree bitwise.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="fp32, one element", spec=dict(numel=1, dtype=torch.float32)),
        Cell(
            why="fp32, not a multiple of the block size",
            spec=dict(numel=1000, dtype=torch.float32),
        ),
        Cell(
            why="bf16, more elements than one grid covers",
            spec=dict(numel=65535 * 256 + 3, dtype=torch.bfloat16),
        ),
        Cell(why="empty", spec=dict(numel=0, dtype=torch.bfloat16)),
    )

    note = """
    A fixture: it proves the native path works, not anything about a model.
    """

    def __call__(self, x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        return native_ops.scale_add(x, alpha, beta)

    def reference(self, x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        return (x.float() * alpha + beta).to(x.dtype)

    def is_valid(self, x: torch.Tensor, alpha: float, beta: float) -> None:
        assert x.is_contiguous(), "the kernel indexes x as one contiguous run"


scale_add = ScaleAdd()
