# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter MLP's SiLU-and-mul on the trtllm Triton ``silu_and_mul`` kernel."""

from __future__ import annotations

import torch
from torch.nn import functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.silu_and_mul
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class K3DraftSiluAndMul(OpWrapper):
    """``silu(x[:, :d]) * x[:, d:]`` with ``d = x.shape[1] // 2``: a new ``[M, d]`` tensor in ``x``'s dtype."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"drafter MLP {name}, M={rows}: a decode block to a prefill chunk",
            spec=dict(rows=rows, width=width),
        )
        for name, width in (("TP16 rank (2 x 896)", 1792), ("one rank (2 x 14336)", 28672))
        for rows in (1, 7, 56, 64, 1000)
    )

    note = """
    The kernel the built-in drafter's GatedMLP runs (`swiglu` with no limit,
    alpha or beta, no output scale), so the drafter's generic block keeps its
    bits. 2-D bf16 input with a dense last dimension; any row stride. The kernel
    computes in fp32 and rounds once; the reference does the same, so the band is
    one bf16 ulp from the different exp. No metadata or state is consumed.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.silu_and_mul(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """fp32, rounded once."""
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)

    def is_valid(self, x: torch.Tensor) -> None:
        assert x.dim() == 2, f"x must be 2-D [M, 2 d]; got {tuple(x.shape)}"
        assert x.shape[1] % 2 == 0, f"x.shape[1] must be even; got {x.shape[1]}"
        assert x.stride(1) == 1, "x's last dimension must be dense"


k3_draft_silu_and_mul = K3DraftSiluAndMul()
