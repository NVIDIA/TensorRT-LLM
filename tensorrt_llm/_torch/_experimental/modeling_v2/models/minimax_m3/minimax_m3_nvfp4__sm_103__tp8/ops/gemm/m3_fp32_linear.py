# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The router's fp32 projection beyond the decode routers' rows: ``float(x) @ w^T`` through torch's GEMM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class M3Fp32Linear(OpWrapper):
    """fp32 router logits ``F.linear(float(x[M, 6144]), w[128, 6144])`` for prefill and steps above 48 rows (the
    fp32 GEMV and the tensor-core router take 1..48). Whatever fp32 matmul precision the process runs with applies
    (the engine leaves torch's default)."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="a prefill chunk", spec=dict(m=2048, e=128, k=6144)),
        Cell(
            why="a generation step above the tensor-core router's 48 rows",
            spec=dict(m=64, e=128, k=6144),
        ),
    )

    note = """
    Under torch's default fp32 matmul precision the product may be TF32: the test's band covers that rounding against
    the true fp32 reference.
    """

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(torch.float32), w)

    def reference(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        with true_fp32_matmul():
            return x.float() @ w.t()


m3_fp32_linear = M3Fp32Linear()
