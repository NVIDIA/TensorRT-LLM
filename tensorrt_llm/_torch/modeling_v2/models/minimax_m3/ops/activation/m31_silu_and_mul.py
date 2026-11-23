# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gated MLPs' fused SwiGLU: ``silu_and_mul`` with MiniMax-M3's SwiGLU-OAI gain, offset and clamp.

``native_vec`` (the tp4 and tp8 targets pass their SILU_VEC switch; the tp2 target passes False): at MiniMax-M3's
SwiGLU-OAI constants (7, 1.702, 1) the activation runs the family NATIVE_SWIGLU op
(mv2_minimax_m3::minimax_m31_swiglu_oai_vec, row grid + PDL), bit for bit the Triton silu_and_mul, at any row count, so
a dense / shared-expert call past the rows the target's m3_dense_swiglu is certified for does not launch the Triton
kernel inside the decode graph. Plain SwiGLU (no constants) keeps the stock op. Outside the native contract (not a
contiguous 2-D bf16 CUDA tensor, under torch.compile, op not built) the stock op runs.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import native_swiglu


def _cells(shared: int, dense: int, draft: int) -> tuple[Cell, ...]:
    return (
        Cell(why="shared expert, one decode token", spec=dict(m=1, inter=shared, oai=True)),
        Cell(why="shared expert, a 48-row decode verify", spec=dict(m=48, inter=shared, oai=True)),
        Cell(why="dense MLP, a prefill chunk", spec=dict(m=2048, inter=dense, oai=True)),
        Cell(why="dense MLP, one decode token", spec=dict(m=1, inter=dense, oai=True)),
        Cell(
            why="drafter MLP (plain SwiGLU), eight decode rows",
            spec=dict(m=8, inter=draft, oai=False),
        ),
        Cell(
            why="drafter MLP (plain SwiGLU), a prefill chunk",
            spec=dict(m=1024, inter=draft, oai=False),
        ),
    )


class M31SiluAndMul(OpWrapper):
    """``out = g * sigmoid(alpha * g) * (u + beta)`` over ``x = [g | u]`` (bf16 [M, 2I] -> [M, I]), with
    ``g = min(gate, limit)`` and ``u = clamp(up, -limit, limit)``; ``limit``, ``alpha`` and ``beta`` None give plain
    SwiGLU ``silu(gate) * up``. ``native_vec`` selects the family native op at the SwiGLU-OAI constants (see the
    module docstring).

    The MXFP8 dense MLPs (layers 0-2) and the MoE's shared expert run it at MiniMax-M3's (limit 7, alpha 1.702,
    beta 1); the BF16 Eagle3 drafter's MLP runs it plain. Intermediate per rank (dense / shared / drafter): 3072 / 768
    / 4608 on the tp2 and tp4 targets, 1536 / 384 / 2304 on the tp8 target.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The cells each Rubin target runs, by tensor-parallel size; ``CELLS`` is their union.
    CELLS_BY_TP: dict[int, tuple[Cell, ...]] = {
        2: _cells(768, 3072, 4608),
        4: _cells(768, 3072, 4608),
        8: _cells(384, 1536, 2304),
    }
    CELLS: tuple[Cell, ...] = tuple(cell for cells in CELLS_BY_TP.values() for cell in cells)

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
        native_vec: bool,
    ) -> torch.Tensor:
        if (
            native_vec
            and (limit, alpha, beta)
            == (native_swiglu.LIMIT, native_swiglu.ALPHA, native_swiglu.BETA)
            and native_swiglu.outside_contract(x) is None
        ):
            return native_swiglu.swiglu_oai(x)
        return torch.ops.trtllm.silu_and_mul(
            x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta
        )

    def reference(
        self,
        x: torch.Tensor,
        limit: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
        native_vec: bool,
    ) -> torch.Tensor:
        """``native_vec`` is ignored: both paths compute this activation."""
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
        native_vec: bool,
    ) -> None:
        assert x.stride(-1) == 1, "the gate/up halves are read from contiguous rows"


m31_silu_and_mul = M31SiluAndMul()
