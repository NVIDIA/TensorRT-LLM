# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 Eagle3 drafter's projections: ``x @ w^T`` through torch's GEMM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class M3Bf16Linear(OpWrapper):
    """``F.linear(x, w)`` for bf16 ``x`` [M, K] and a bf16 weight [N, K] (cuBLAS, fp32 accumulation), bf16 out.

    The drafter's fc over the captured target layers (K = 3 x 6144), its fused QKV (K = 2 x 6144, this rank's heads),
    o_proj (this rank's input columns), gate_up and down projections run it.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="drafter fc, 32 verify rows", spec=dict(m=32, n=6144, k=18432)),
        Cell(why="drafter QKV, eight draft rows", spec=dict(m=8, n=2304, k=12288)),
        Cell(why="drafter o_proj, a prefill chunk", spec=dict(m=1024, n=6144, k=2048)),
        Cell(why="drafter gate_up, one row", spec=dict(m=1, n=9216, k=6144)),
        Cell(why="drafter down, eight rows", spec=dict(m=8, n=6144, k=4608)),
    )

    note = """
    cuBLAS's split and accumulation order: `reference` multiplies in fp32 (one bf16 rounding), within the test's band.
    """

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.linear(x, w)

    def reference(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (x.float() @ w.float().t()).to(x.dtype)


m3_bf16_linear = M3Bf16Linear()
