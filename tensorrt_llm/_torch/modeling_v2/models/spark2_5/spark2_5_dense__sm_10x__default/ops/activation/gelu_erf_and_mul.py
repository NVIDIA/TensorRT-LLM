# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spark's gated MLP activation: exact (erf) GELU on the gate half, multiplied
by the up half."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The published checkpoints' MLP widths (intermediate_size).
_INTER_1_7B = 6656
_INTER_4B = 10240


class GeluErfAndMul(OpWrapper):
    """`gelu_erf(x[..., :I]) * x[..., I:]` on a fused `[T, 2I]` gate_up product.

    First half is the gate (the weight table concatenates `gate_proj` ahead of
    `up_proj`). Exact erf GELU (`F.gelu`, `approximate="none"`), not tanh --
    the reference runs `ACT2FN["gelu"]`. Returns a new tensor.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="1.7B decode: one token",
            spec=dict(tokens=1, inter=_INTER_1_7B, dtype=torch.bfloat16),
        ),
        Cell(
            why="1.7B prefill-shaped",
            spec=dict(tokens=2048, inter=_INTER_1_7B, dtype=torch.bfloat16),
        ),
        Cell(
            why="a token count that is not a multiple of any tile",
            spec=dict(tokens=17, inter=_INTER_1_7B, dtype=torch.bfloat16),
        ),
        Cell(
            why="4B decode: one token",
            spec=dict(tokens=1, inter=_INTER_4B, dtype=torch.bfloat16),
        ),
        Cell(
            why="4B prefill-shaped",
            spec=dict(tokens=2048, inter=_INTER_4B, dtype=torch.bfloat16),
        ),
    )

    note = """
    The catalog carries only `flashinfer_silu_and_mul` and FlashInfer's gated
    gelu is tanh-only, so the gated erf form is native torch: one `F.gelu`
    kernel and one multiply. Metadata consumed: none.
    """

    def __call__(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return F.gelu(gate) * up

    def reference(self, gate_up: torch.Tensor) -> torch.Tensor:
        """fp32 throughout, rounded once at the end."""
        gate, up = gate_up.float().chunk(2, dim=-1)
        return (F.gelu(gate) * up).to(gate_up.dtype)

    def is_valid(self, gate_up: torch.Tensor) -> None:
        # An odd last dim makes chunk() silently produce halves of different
        # widths and the multiply broadcasts or raises far from the cause.
        assert gate_up.dim() >= 1 and gate_up.shape[-1] % 2 == 0, (
            f"the fused gate_up width must be even, got {gate_up.shape[-1]}"
        )
        assert gate_up.stride(-1) == 1, "the fused width must be contiguous"


gelu_erf_and_mul = GeluErfAndMul()
