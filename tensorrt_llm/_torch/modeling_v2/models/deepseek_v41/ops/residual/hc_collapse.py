# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collapse the hc residual copies with the pre-mix in one launch: the engine's `mHC.collapse`
(`tensorrt_llm/_torch/modules/mhc/hyper_connection.py`), which stock DeepSeek-V4.1 closes its stream with when the
fused close does not apply (attention DP at decode sizes)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class HcCollapse(OpWrapper):
    """`[n, hc, d] x [n, hc] -> [n, d]`: `sum_i pre_mix[:, i] * x[:, i]` in fp32, one term at a time in copy order,
    cast to `x.dtype` (the family's `hc_pre` in one launch instead of a cast, a product, a reduction and a cast)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 c7000 decode close: 160 rows at the model's width",
            spec=dict(n=160, hc=4, d=5120),
        ),
        Cell(why="dep32 decode close: 150 rows", spec=dict(n=150, hc=4, d=5120)),
        Cell(
            why="a single row; a width that is not a multiple of the 1024-column block",
            spec=dict(n=1, hc=4, d=1500),
        ),
    )
    note = "Same per-term fp32 products and summation order as the plain-torch loop; one rounding to bf16."

    def __call__(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

        return mHC.collapse(x, pre_mix.unsqueeze(-1))

    def reference(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Float64 accumulation, one rounding at the end."""
        return sum(pre_mix[:, i : i + 1].double() * x[:, i].double() for i in range(x.shape[1])).to(
            x.dtype
        )

    def is_valid(self, x: torch.Tensor, pre_mix: torch.Tensor) -> None:
        assert x.is_cuda and x.dim() == 3 and x.is_contiguous(), (
            "a contiguous [n, hc, d] CUDA tensor (else torch)"
        )
        assert pre_mix.shape == x.shape[:2] and pre_mix.dtype == torch.float32


hc_collapse = HcCollapse()
