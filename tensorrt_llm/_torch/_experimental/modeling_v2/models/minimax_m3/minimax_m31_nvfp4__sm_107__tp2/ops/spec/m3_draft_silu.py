# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter MLP's plain SwiGLU as one C++ launch (``modeling.DRAFT_SILU_CPP``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class M3DraftSilu(OpWrapper):
    """``silu(g) * u`` of a bf16 [rows, 2d] gate | up tensor, rounded once to bf16:
    trtllm::minimax_m3_sm107_tp4_silu_and_mul (minimaxM3Sm107Tp4/siluAndMul.cu), bit for bit the Triton
    ``torch.ops.trtllm.silu_and_mul`` at its defaults (no limit, alpha 1, beta 0), the activation the drafter's GatedMLP
    runs. Contiguous, 16-byte aligned rows and d a multiple of 8; ``__call__`` returns None otherwise (the caller runs
    the Triton op)."""

    ARCHS = frozenset({Arch.SM_107})

    # The TP4 drafter MLP shard: intermediate 18432 / 4 = 4608, gate | up 9216 wide, at the drafter's decode rows.
    CELLS: tuple[Cell, ...] = (
        Cell(why="the TP4 drafter MLP at 1 row", spec=dict(rows=1, d=4608)),
        Cell(why="the TP4 drafter MLP at 4 rows (1 request x 4 tokens)", spec=dict(rows=4, d=4608)),
        Cell(why="the TP4 drafter MLP at 8 rows", spec=dict(rows=8, d=4608)),
    )

    note = """
    Elementwise, one rounding: bit for bit the Triton kernel (the same PTX instruction sequence), checked over every
    bf16 gate and every bf16 up pattern.
    """

    def __call__(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            x.dtype != torch.bfloat16
            or x.dim() != 2
            or not x.is_cuda
            or not x.is_contiguous()
            or x.shape[1] % 16 != 0
            or x.data_ptr() % 16 != 0
            or x.shape[0] > 65535
        ):
            return None
        out = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
        torch.ops.trtllm.minimax_m3_sm107_tp4_silu_and_mul(x, out)
        return out

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.silu_and_mul(x)

    def is_valid(self, x: torch.Tensor) -> None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2, "x: bf16 [rows, 2d]"


m3_draft_silu = M3DraftSilu()
