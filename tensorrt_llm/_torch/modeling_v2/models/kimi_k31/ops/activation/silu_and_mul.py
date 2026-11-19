# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 DFlash2 drafter MLP activation ``silu(gate) * up`` (``trtllm::silu_and_mul``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The drafter MLP's per-rank intermediate width at each tensor-parallel width (14336 / tp).
_DRAFT_INTER = 14336


def _cell(rows: int, tp: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, width=_DRAFT_INTER // tp))


class K31SiluAndMul(OpWrapper):
    """``silu(x[..., :I]) * x[..., I:]`` for BF16 ``[..., 2I]`` (``[gate | up]``), as one ``trtllm::silu_and_mul``
    launch without a limit, gain or offset: the activation of the drafter's SwiGLU MLP, between ``gate_up_proj``
    and ``down_proj``. Returns a new ``[..., I]`` tensor.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell(5, 8, "TEP8 drafter block at B1 (5 rows), 1792 per rank"),
        _cell(40, 8, "TEP8 drafter block at B8 (40 rows)"),
        _cell(50, 16, "TEP16 drafter block at B10 (50 rows), 896 per rank"),
        _cell(40, 32, "TEP32 drafter block at B8 (40 rows), 448 per rank"),
        _cell(5, 1, "DEP16 drafter block at B1 (replicated MLP), 14336"),
        _cell(320, 1, "DEP16 drafter block at B64 (320 rows)"),
    )

    note = """
    Bitwise the BF16 SwiGLU call of the engine GatedMLP (swiglu -> trtllm::silu_and_mul, no limit,
    gain or offset) on the gate_up projection's output; close to the FP32 reference.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.silu_and_mul(
            x, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None
        )

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """``silu(gate) * up`` in FP32, rounded once."""
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)


silu_and_mul = K31SiluAndMul()
