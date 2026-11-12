# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eagle3 hidden-state capture with the drafter's fc_norm folded in (``m3_spec_tail.capture_norm``)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_chunk_norm import rmsnorm_rows


class M3CaptureNorm(OpWrapper):
    """``dst [M, H] = rmsnorm(bf16(h + r)) * w`` (``r`` optional), the rounding chunk_norm applies, into a capture
    buffer slice whose row stride may exceed H. Returns ``dst``."""

    ARCHS = frozenset({Arch.SM_103})

    # The target's captured layers (hidden 6144) into the 3-layer capture buffer (row stride 3 x 6144), over the
    # verify rows of draft_len 3 (4 .. 32 rows); the residual is present for every captured decoder layer.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="4 verify rows of one request, layer slot 0, residual added",
            spec=dict(m=4, slot=0, residual=True),
        ),
        Cell(
            why="cap 8 (32 rows) into the last layer slot", spec=dict(m=32, slot=2, residual=True)
        ),
        Cell(
            why="no residual (a capture point without one): plain RMSNorm",
            spec=dict(m=8, slot=1, residual=False),
        ),
    )

    note = """
    fp32 statistic (block sum, rsqrtf), one bf16 rounding after the bf16 sum h + r; the band states the occasional
    one-ulp move against torch's statistic.
    """

    def __call__(
        self,
        h: torch.Tensor,
        r: Optional[torch.Tensor],
        w: torch.Tensor,
        eps: float,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.trtllm.minimax_m3_capture_norm(h, r, w, eps, dst)
        return dst

    def reference(
        self,
        h: torch.Tensor,
        r: Optional[torch.Tensor],
        w: torch.Tensor,
        eps: float,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """bf16 sum, then the normalized row; ``dst`` only says where the op writes."""
        x = (h.float() + r.float()).to(torch.bfloat16) if r is not None else h
        return rmsnorm_rows(x, w, eps)

    def is_valid(
        self,
        h: torch.Tensor,
        r: Optional[torch.Tensor],
        w: torch.Tensor,
        eps: float,
        dst: torch.Tensor,
    ) -> None:
        # The norm weight is read as H dense bf16 values; a strided view (a column of a stacked weight) is read
        # past its elements without an error.
        assert w.is_contiguous(), "w must be dense [H]"


m3_capture_norm = M3CaptureNorm()
