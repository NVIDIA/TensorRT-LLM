# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU with the checkpoint's training clamps (plain torch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class ClampedSwiglu(OpWrapper):
    """fp32 `silu(gate) * up`; with `limit > 0` the up branch is clamped to `[-limit, limit]`, the gate to `<= limit`.

    The result is fp32 whatever the input dtype: the shared expert casts it back itself.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="shared expert, decode token: [1, 2304] bf16, limit 10",
            spec=dict(shape=(1, 2304), dtype=torch.bfloat16, limit=10.0),
        ),
        Cell(
            why="shared expert, prefill chunk: [4096, 2304] bf16",
            spec=dict(shape=(4096, 2304), dtype=torch.bfloat16, limit=10.0),
        ),
        Cell(
            why="limit 0 means unclamped",
            spec=dict(shape=(8, 2304), dtype=torch.bfloat16, limit=0.0),
        ),
        Cell(why="fp32 inputs", spec=dict(shape=(8, 256), dtype=torch.float32, limit=3.0)),
    )
    note = "Not the routed experts' clamp: that one lives in the MoE kernel, in accumulator units."

    def __call__(self, gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        gate, up = gate.float(), up.float()
        if limit > 0:
            up = up.clamp(-limit, limit)
            gate = gate.clamp(max=limit)
        return F.silu(gate) * up

    def reference(self, gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """float64 `g * sigmoid(g) * u` with explicit min / max clamps."""
        g, u = gate.double(), up.double()
        if limit > 0:
            u = torch.minimum(
                torch.maximum(u, torch.full_like(u, -limit)), torch.full_like(u, limit)
            )
            g = torch.minimum(g, torch.full_like(g, limit))
        return (g * torch.sigmoid(g) * u).float()

    def is_valid(self, gate: torch.Tensor, up: torch.Tensor, limit: float) -> None:
        assert gate.shape == up.shape
        assert limit >= 0, "a negative limit is treated as no clamp"


clamped_swiglu = ClampedSwiglu()
