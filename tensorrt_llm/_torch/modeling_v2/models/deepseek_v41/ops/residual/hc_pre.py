# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collapse the hc residual copies with the pre-mix (plain torch)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_TILE = 4096


class HcPre(OpWrapper):
    """`[n, hc, d] x [n, hc] -> [n, d]`: sum over copies of `pre_mix * x`, fp32 accumulate, cast to `x.dtype`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="decode token, bf16 stream at the model's width",
            spec=dict(n=1, hc=4, d=5120, dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill tile edge, 4097 rows",
            spec=dict(n=4097, hc=4, d=1024, dtype=torch.bfloat16),
        ),
        Cell(
            why="fp32 stream: output stays fp32", spec=dict(n=9, hc=4, d=512, dtype=torch.float32)
        ),
    )
    note = "Elementwise multiply-add, no matmul: precision flags do not apply."

    def __call__(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        out = x.new_empty(x.shape[0], x.shape[2])
        for a in range(0, x.shape[0], _TILE):
            t = slice(a, min(a + _TILE, x.shape[0]))
            out[t] = torch.sum(pre_mix[t].unsqueeze(-1) * x[t].float(), dim=1).to(x.dtype)
        return out

    def reference(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Untiled, float64 accumulation, one rounding at the end."""
        return sum(pre_mix[:, i : i + 1].double() * x[:, i].double() for i in range(x.shape[1])).to(
            x.dtype
        )

    def is_valid(self, x: torch.Tensor, pre_mix: torch.Tensor) -> None:
        assert x.dim() == 3 and pre_mix.shape == x.shape[:2], "x [n, hc, d], pre_mix [n, hc]"
        assert pre_mix.dtype == torch.float32


hc_pre = HcPre()
