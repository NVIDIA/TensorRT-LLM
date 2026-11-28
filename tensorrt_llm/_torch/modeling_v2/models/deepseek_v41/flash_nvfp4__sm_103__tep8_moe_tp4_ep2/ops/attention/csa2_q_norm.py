# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query latent's RMSNorm as its own launch, bf16 out: what stock's CSA2 module runs for a FULL / REINDEX layer
whose Q-B takes the small-M kernel (`_project_query_latent`: `rms_norm_mxfp8(latent, q_norm, eps, quantize=False)`), so
the attention Q-B and the index Q-B both read the same normalized bf16 rows. (Other layers fold the norm into the Q-B
kernel, as this target does.)"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class Csa2QNorm(OpWrapper):
    """`rms_norm(x) * weight` of bf16 rows `x` `[n, K]` (K a multiple of 128, at most 8192), bf16 out."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: one request with five drafts, the 1280-wide query latent",
            spec=dict(n=6, k=1280),
        ),
        Cell(why="a single decode row", spec=dict(n=1, k=1280)),
    )
    note = "fp32 sum of squares, one bf16 rounding of the normalized row (the reference states the band)."

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8

        qr, _, _ = rms_norm_mxfp8(x, weight, eps, quantize=False)
        return qr

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        a = x.float()
        return (a * torch.rsqrt(a.square().mean(-1, keepdim=True) + eps) * weight.float()).to(
            torch.bfloat16
        )

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_cuda and x.stride(1) == 1
        assert (
            x.shape[1] % 128 == 0
            and x.shape[1] <= 8192
            and weight.shape == (x.shape[1],)
            and weight.dtype == x.dtype
        )


csa2_q_norm = Csa2QNorm()
