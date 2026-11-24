# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-tower LayerNorm (width 1152, eps 1e-5) via the FLA Triton kernel behind `layernorm_fn`."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.layernorm_gated import layernorm_fn as _layernorm_fn

#: The vision tower's width: `vision_config.hidden_size` 1152.
WIDTH = 1152


class LayerNorm(OpWrapper):
    """`(x - mean) / sqrt(var + eps) * weight + bias` over the last dim, biased variance, fp32 math, new tensor."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="vision norm, one patch row x 1152 bf16",
            spec=dict(shape=(1, WIDTH), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="vision norm, a 512-pixel frame: 32 x 32 patches",
            spec=dict(shape=(1024, WIDTH), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="vision norm at the tower ceiling, 2048 px: 128 x 128 patches",
            spec=dict(shape=(16384, WIDTH), dtype=torch.bfloat16, eps=1e-5),
        ),
    )

    note = """
    No `is_valid`: a weight/bias of the wrong length fails the callable's own assert. Row offsets are i32
    products, so `x` past ~1.86M rows at width 1152 would wrap; the tower calls at most 16384 rows.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        return _layernorm_fn(x, weight, bias, z=None, eps=eps)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """fp64 from primitive ops, rounded once to `x.dtype`."""
        xd = x.double()
        mean = xd.mean(dim=-1, keepdim=True)
        var = (xd - mean).square().mean(dim=-1, keepdim=True)
        out = (xd - mean) / torch.sqrt(var + eps) * weight.double()
        if bias is not None:
            out = out + bias.double()
        return out.to(x.dtype)


layer_norm = LayerNorm()
