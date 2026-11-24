# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply on the fused Triton kernel ``torch.ops.trtllm.situ_and_mul``."""

from __future__ import annotations

from typing import Optional

import torch

# Registers torch.ops.trtllm.situ_and_mul. A target that imports only this entry (the ULL helix16
# generation target under its routing keys) has no other import of it.
import tensorrt_llm._torch.modules.situ  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Kimi K3.1's activation_situ_beta and activation_situ_linear_beta.
BETA, LINEAR_BETA = 4.0, 25.0


def _cell(why: str, rows: int, d: int, linear_beta: Optional[float] = LINEAR_BETA) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, d=d, beta=BETA, linear_beta=linear_beta))


class SituAndMul(OpWrapper):
    """``beta tanh(g / beta) sigmoid(g) * linear_beta tanh(u / linear_beta)`` for ``[g | u] = x``
    (``g, u = x[:, :d], x[:, d:]``, ``d = x.shape[-1] // 2``; ``u`` as is when ``linear_beta`` is
    None). ``x`` 2-D with unit last stride; returns a new [M, d] tensor in ``x.dtype``."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell(
            "L0 dense fallback, TP16 slice (I = 2112), past the split-K kernels' 32 rows", 64, 2112
        ),
        _cell("L0 dense fallback, TP16 slice, 33 rows", 33, 2112),
        _cell("shared expert (MXFP8 projections), TP16 (2 x 3072 / 16 = 384), b1 verify", 8, 384),
        _cell("shared expert, TP16, b4 verify", 32, 384),
        _cell("without the up soft cap (linear_beta None)", 8, 384, None),
    )

    note = """
    The L0 dense MLP's fallback (rows past the fused kernels). The op is registered by
    ``tensorrt_llm._torch.modules.situ``, which this entry imports. Triton specializes on dtype,
    on whether linear_beta is None and on the divisibility of widths, strides and pointers: warm
    each specialization before CUDA-graph capture. libdevice tanh differs from the exact one in the
    last FP32 bits.
    """

    def __call__(
        self, x: torch.Tensor, beta: float, linear_beta: Optional[float] = None
    ) -> torch.Tensor:
        return torch.ops.trtllm.situ_and_mul(x, beta, linear_beta)

    def reference(
        self, x: torch.Tensor, beta: float, linear_beta: Optional[float] = None
    ) -> torch.Tensor:
        """FP64 activation, rounded once to ``x``'s dtype."""
        d = x.shape[-1] // 2
        g, u = x[:, :d].double(), x[:, d:].double()
        situ = beta * torch.tanh(g / beta) * torch.sigmoid(g)
        if linear_beta is not None:
            u = linear_beta * torch.tanh(u / linear_beta)
        return (situ * u).to(x.dtype)


situ_and_mul = SituAndMul()
