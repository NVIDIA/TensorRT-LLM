# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp32 router GEMV plus the routed experts' NVFP4 activation, one token."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_router_gemv import fp32_logits, nvfp4_quantize


class M3RouterGemvFp4(OpWrapper):
    """``[logits fp32 [M, E], x_fp4 [M, K / 2] uint8, x_sf [M * K / 16] uint8]`` from one kernel.

    The logits are ``m3_router_gemv``'s; the NVFP4 activation is ``fp4_quantize(x, sf_scale, 16, False, False)``
    (16-element blocks, E4M3 scales in the linear layout, per-tensor ``sf_scale``) for the routed experts' FC1.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The fused-router decode step at one token: 128 experts, hidden 6144, the experts' input scale.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode token: logits for 128 experts and the NVFP4 activation of hidden 6144",
            spec=dict(m=1, e=128, k=6144, sf_scale=448.0 * 6.0 / 8.0),
        ),
        Cell(
            why="a global scale that puts block scales near the E4M3 top, where the scale cast saturates",
            spec=dict(m=1, e=128, k=6144, sf_scale=448.0 * 6.0 / 2.0),
        ),
    )

    note = """
    The e2m1 codes are exact except within `TIE_WINDOW` of a rounding midpoint, where the kernel's approximate
    reciprocals decide the direction. Up to 16 rows are accepted; the target sends one.
    """

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor, sf_scale: torch.Tensor, pdl: bool
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.minimax_m3_router_gemv_fp4(x, w, sf_scale, pdl)

    def reference(
        self, x: torch.Tensor, w: torch.Tensor, sf_scale: torch.Tensor, pdl: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """True fp32 logits and the NVFP4 recipe; ``pdl`` does not enter the arithmetic."""
        packed, sf, _ = nvfp4_quantize(x, sf_scale)
        return fp32_logits(x, w), packed, sf

    def is_valid(self, x: torch.Tensor, w: torch.Tensor, sf_scale: torch.Tensor, pdl: bool) -> None:
        assert x.data_ptr() % 16 == 0, "x must start 16-byte aligned (cp.async.bulk source)"
        # A zero or non-finite global scale makes every block scale 0 or NaN and the data codes garbage; the op
        # reads the scalar on the device and cannot refuse it.
        value = float(sf_scale.reshape(()))
        assert value > 0.0 and value != float("inf"), (
            f"sf_scale must be positive and finite, got {value}"
        )


m3_router_gemv_fp4 = M3RouterGemvFp4()
