# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense MLP's swigluoai activation as one C++ launch with PDL and an early release (DENSE_SWIGLU_CPP)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# The dense layers' swigluoai parameters, which the kernel hard-codes (modeling_minimaxm3._build_swiglu_oai_dense_mlp).
LIMIT, ALPHA, BETA = 7.0, 1.702, 1.0


class M3DenseSwiglu(OpWrapper):
    """``min(g, 7) * sigmoid(1.702 * min(g, 7)) * (clamp(u, -7, 7) + 1)`` of a bf16 [rows, 2d] gate | up tensor,
    rounded once to bf16: trtllm::minimax_m3_sm107_tp8_swiglu_oai (minimaxM3Sm107Tp8/swigluOai.cu), bit for bit the
    Triton ``torch.ops.trtllm.silu_and_mul`` at swiglu_limit 7, swiglu_alpha 1.702, swiglu_beta 1. With ``use_pdl`` the
    grid waits on its predecessor and releases its dependents before it computes, so the down GEMV stages its weights
    during the activation (the GEMV reads the activation only after its own grid wait). Contiguous rows with an even
    half width; ``__call__`` returns None otherwise or for other activation parameters (the caller runs the Triton
    op)."""

    ARCHS = frozenset({Arch.SM_107})

    # The TP8 dense MLP shard: gate | up 3072 wide (d = 1536), at the decode rows.
    CELLS: tuple[Cell, ...] = (
        Cell(why="the TP8 dense MLP at 1 row", spec=dict(rows=1, d=1536)),
        Cell(why="the TP8 dense MLP at 4 rows (1 request x 4 tokens)", spec=dict(rows=4, d=1536)),
        Cell(why="the TP8 dense MLP at 8 rows", spec=dict(rows=8, d=1536)),
    )

    note = """
    Elementwise, one rounding: bit for bit the Triton kernel (its instruction sequence: the clamps, (u + 1), the
    alpha-scaled ex2.approx sigmoid, div.full), checked over every bf16 gate and every bf16 up pattern.
    """

    def __call__(
        self,
        x: torch.Tensor,
        limit: Optional[float] = LIMIT,
        alpha: Optional[float] = ALPHA,
        beta: Optional[float] = BETA,
        use_pdl: bool = True,
    ) -> Optional[torch.Tensor]:
        if (
            (limit, alpha, beta) != (LIMIT, ALPHA, BETA)
            or x.dtype != torch.bfloat16
            or x.dim() != 2
            or not x.is_cuda
            or not x.is_contiguous()
            or x.shape[1] % 4 != 0
            or x.data_ptr() % 4 != 0
        ):
            return None
        out = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
        torch.ops.trtllm.minimax_m3_sm107_tp8_swiglu_oai(x, out, bool(use_pdl))
        return out

    def reference(self, x, limit=LIMIT, alpha=ALPHA, beta=BETA, use_pdl=True) -> torch.Tensor:
        return torch.ops.trtllm.silu_and_mul(
            x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta
        )

    def is_valid(self, x, limit=LIMIT, alpha=ALPHA, beta=BETA, use_pdl=True) -> None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2, "x: bf16 [rows, 2d]"


m3_dense_swiglu = M3DenseSwiglu()
