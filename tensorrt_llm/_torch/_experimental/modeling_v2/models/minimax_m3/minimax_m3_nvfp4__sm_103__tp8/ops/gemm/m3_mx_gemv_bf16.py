# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode MXFP8 GEMV (CuTe DSL) on a bf16 activation, quantized to MXFP8 in the kernel's prologue."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..moe.m3_moe_engine import mxfp8_quant_dequant
from ..mx_gemv import op as _gemv
from .m3_mx_gemv import (
    DENSE_DOWN,
    DENSE_GATE_UP,
    O_PROJ,
    QKV,
    QKV_SPARSE,
    gemv_is_valid,
    gemv_reference,
    served_cells,
)

__all__ = ["m3_mx_gemv_bf16"]


class M3MxGemvBf16(OpWrapper):
    """bf16 [M, N] = mxfp8_quantize(x) x weight [N, K]^T for a contiguous, 16-byte aligned bf16 ``x`` [M, K]; the
    quantization is mxfp8_quantize's (UE8M0 scale per 32 = the smallest power of two >= amax / 448). None when the
    shape is not served. The other arguments as ``m3_mx_gemv``'s.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The prologue quantizes one 32-element block per thread and round: the projections at the token counts the
    # target sends here (no producer-quantized activation), up to the widest each serves.
    CELLS: tuple[Cell, ...] = (
        Cell(why="sparse-layer QKV, one token", spec=dict(nk=QKV_SPARSE, m=1)),
        Cell(why="sparse-layer QKV, 2 tokens: its split-4 config", spec=dict(nk=QKV_SPARSE, m=2)),
        Cell(
            why="sparse-layer QKV, 16 rows: the most its split-4 prologue serves",
            spec=dict(nk=QKV_SPARSE, m=16),
        ),
        Cell(why="dense-attention QKV, 8 tokens", spec=dict(nk=QKV, m=8)),
        Cell(why="o_proj, one token", spec=dict(nk=O_PROJ, m=1)),
        Cell(why="o_proj, 48 rows: the 48-column MMA", spec=dict(nk=O_PROJ, m=48)),
        Cell(
            why="dense gate_up, 16 rows: the most split 4 at K 6144 serves",
            spec=dict(nk=DENSE_GATE_UP, m=16),
        ),
        Cell(why="dense down, one token", spec=dict(nk=DENSE_DOWN, m=1)),
        Cell(
            why="dense down, 32 rows: the most split 2 at K 1536 serves",
            spec=dict(nk=DENSE_DOWN, m=32),
        ),
    )
    # Every other (projection, rows) the decode dispatch can send here (dispatch.MX_GEMV_BF16_MAX_TOKENS).
    CELLS = CELLS + served_cells(CELLS, x_bf16=True, max_m=48)

    note = """
    As `m3_mx_gemv`. The in-kernel quantization is mxfp8_quantize's bit for bit except where the kernel's
    rcp.approx(448) and an exact 1/448 put a block's amax on different sides of a power of two (not observed on the
    cells). Not covered: `ar`, `cfg` overrides. Returns None for rows beyond 8 rounds of the prologue per CTA.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
        ar: Optional[tuple] = None,
    ) -> Optional[torch.Tensor]:
        return _gemv.m3_mx_gemv_bf16(x, weight, weight_scale, trigger_early, cfg, use_pdl, ar)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
        ar: Optional[tuple] = None,
    ) -> torch.Tensor:
        """The MXFP8 round trip of x times the dequantized weight; the launch options are ignored."""
        return gemv_reference(mxfp8_quant_dequant(x), weight, weight_scale)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        trigger_early: bool = True,
        cfg: Optional[tuple] = None,
        use_pdl: Optional[bool] = None,
        ar: Optional[tuple] = None,
    ) -> None:
        gemv_is_valid(weight, weight_scale)


m3_mx_gemv_bf16 = M3MxGemvBf16()
