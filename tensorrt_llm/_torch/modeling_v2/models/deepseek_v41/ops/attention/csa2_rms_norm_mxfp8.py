# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The q_norm of the q_a latent and the MXFP8 quantization of its output in one launch (the engine's
`csa2/projection.py::rms_norm_mxfp8`, as stock's decode query path runs it once for the Q-B and index Q-B projections).

`csa2_mxfp8_linear(..., rms_norm=...)` runs the same launch inside each projection it serves; a layer with two
projections of the same latent (an index-source layer: q_b and the index q_b) runs it once with this entry and hands
the result to both (`csa2_mxfp8_linear(..., x_mxfp8=...)`), bit for bit what each would have computed itself.
"""

from __future__ import annotations

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class Csa2RmsNormMxfp8(OpWrapper):
    """`x [M, K] bf16` (unit column stride: a column slice of the fused q_a | kv output is read in place), `weight` bf16
    `[K]`, `eps` -> `(data E4M3 [M, K], scales)`: `mxfp8_quantize(rms_norm(x) * weight, True)` as one launch (UE8M0
    1x32 block scales in the CUTLASS-swizzled layout the quantized GEMM reads; K a multiple of 128, at most 8192)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep32 decode, 24 rows: the q_a latent (K 1280) of an index-source layer",
            spec=dict(m=24, k=1280),
        ),
        Cell(why="dep8 decode, 160 rows", spec=dict(m=160, k=1280)),
        Cell(why="dep8 decode, 1280 rows (the largest step)", spec=dict(m=1280, k=1280)),
    )
    note = "Bit for bit the norm + quantize launch `csa2_mxfp8_linear(rms_norm=...)` runs inside the projection."

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8

        _, data, scales = rms_norm_mxfp8(x, weight, eps, normalized=False)
        return data, scales

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """The normalized rows `rms_norm(x) * weight` in float64 (what
        the MXFP8 values approximate, 3 mantissa bits)."""
        a = x.double()
        return a * torch.rsqrt(a.square().mean(-1, keepdim=True) + eps) * weight.double()

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_cuda and x.stride(1) == 1
        assert x.shape[1] % 128 == 0 and x.shape[1] <= 8192, "K a multiple of 128, at most 8192"
        assert (
            weight.shape == (x.shape[1],)
            and weight.dtype == torch.bfloat16
            and weight.is_contiguous()
        )
        assert weight.device == x.device, "the norm weight is read by the kernel on x's device"


csa2_rms_norm_mxfp8 = Csa2RmsNormMxfp8()
