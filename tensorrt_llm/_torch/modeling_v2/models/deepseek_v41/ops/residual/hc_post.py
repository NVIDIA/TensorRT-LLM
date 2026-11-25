# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expand a sub-layer output back to hc copies and mix the residual in (plain torch)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_TILE = 4096


class HcPost(OpWrapper):
    """`-> [n, hc, d]`: `post[t, i] * x[t] + sum_j comb[t, j, i] * residual[t, j]` (fp32, cast to `x.dtype`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="decode token at the model's width",
            spec=dict(n=1, hc=4, d=5120, dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill tile edge, 4097 rows",
            spec=dict(n=4097, hc=4, d=1024, dtype=torch.bfloat16),
        ),
        Cell(why="fp32 stream", spec=dict(n=9, hc=4, d=512, dtype=torch.float32)),
    )
    note = "Elementwise; `comb` is indexed [token, source copy, destination copy]."

    def __call__(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(residual)
        for a in range(0, x.shape[0], _TILE):
            t = slice(a, min(a + _TILE, x.shape[0]))
            y = post[t].unsqueeze(-1) * x[t].unsqueeze(-2) + torch.sum(
                comb[t].unsqueeze(-1) * residual[t].unsqueeze(-2), dim=1
            )
            out[t] = y.type_as(x)
        return out

    def reference(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        """The loops over destination and source copy written out, float64."""
        n, hc, d = residual.shape
        out = torch.empty_like(residual)
        for j in range(hc):
            v = post[:, j : j + 1].double() * x.double()
            for i in range(hc):
                v = v + comb[:, i, j : j + 1].double() * residual[:, i].double()
            out[:, j] = v.to(x.dtype)
        return out

    def is_valid(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> None:
        n, hc, d = residual.shape
        assert x.shape == (n, d) and post.shape == (n, hc) and comb.shape == (n, hc, hc)
        assert post.dtype == torch.float32 and comb.dtype == torch.float32


hc_post = HcPost()
