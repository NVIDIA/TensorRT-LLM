# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gated MLPs' fused SwiGLU: ``silu_and_mul`` with MiniMax-M3's SwiGLU-OAI gain, offset and clamp."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class M3SiluAndMul(OpWrapper):
    """``out = g * sigmoid(alpha * g) * (u + beta)`` over ``x = [g | u]`` (bf16 [M, 2I] -> [M, I]), with
    ``g = min(gate, limit)`` and ``u = clamp(up, -limit, limit)``; ``limit``, ``alpha`` and ``beta`` None give plain
    SwiGLU ``silu(gate) * up``.

    The MXFP8 dense MLPs (layers 0-2, intermediate 3072 per rank) and the MoE's shared expert (intermediate 768 per
    rank) run it at MiniMax-M3's (limit 7, alpha 1.702, beta 1); the BF16 Eagle3 drafter's MLP (intermediate 4608 per
    rank) runs it plain.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="shared expert, one decode token", spec=dict(m=1, inter=768, oai=True)),
        Cell(why="shared expert, a 48-row decode verify", spec=dict(m=48, inter=768, oai=True)),
        Cell(why="dense MLP, a prefill chunk", spec=dict(m=2048, inter=3072, oai=True)),
        Cell(why="dense MLP, one decode token", spec=dict(m=1, inter=3072, oai=True)),
        Cell(
            why="drafter MLP (plain SwiGLU), eight decode rows",
            spec=dict(m=8, inter=4608, oai=False),
        ),
        Cell(
            why="drafter MLP (plain SwiGLU), a prefill chunk",
            spec=dict(m=1024, inter=4608, oai=False),
        ),
    )

    note = """
    The kernel computes in fp32 and rounds once to bf16; `reference` does the same arithmetic in fp32, so the two
    differ only by the fast sigmoid's last bits (the test's band).
    """

    def __call__(
        self,
        x: torch.Tensor,
        limit: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> torch.Tensor:
        return torch.ops.trtllm.silu_and_mul(
            x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta
        )

    def reference(
        self,
        x: torch.Tensor,
        limit: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> torch.Tensor:
        gate, up = x.float().chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        a = 1.0 if alpha is None else alpha
        b = 0.0 if beta is None else beta
        return (gate * torch.sigmoid(a * gate) * (up + b)).to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        limit: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
    ) -> None:
        assert x.stride(-1) == 1, "the gate/up halves are read from contiguous rows"


m3_silu_and_mul = M3SiluAndMul()
