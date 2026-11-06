# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TF32 tensor-core router of the decode MoE for 2..48 tokens, optionally with the NVFP4 activation."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.m3_router_gemv import (
    nvfp4_quantize,
)


def tf32_rna(w: torch.Tensor) -> torch.Tensor:
    """fp32 -> TF32 as ``cvt.rna.tf32.f32``: keep 10 mantissa bits, round to nearest with ties away from zero."""
    bits = w.contiguous().view(torch.int32)
    rounded = (bits + 0x1000) & ~0x1FFF
    finite = (bits & 0x7F800000) != 0x7F800000
    return torch.where(finite, rounded, bits).view(torch.float32)


CELLS_BY_TP = {
    2: (
        Cell(
            why="one token, logits only (the wide target's router_path(1) with no fp32 GEMV rows)",
            spec=dict(m=1, k=6144, fp4=False),
        ),
        Cell(
            why="one token with the NVFP4 activation (the fused-router path at one row)",
            spec=dict(m=1, k=6144, fp4=True),
        ),
        Cell(
            why="two tokens, logits only (the smallest call the target makes here)",
            spec=dict(m=2, k=6144, fp4=False),
        ),
        Cell(
            why="8 tokens with the NVFP4 activation: one 8-token MMA tile, every expert tile quantizes one row",
            spec=dict(m=8, k=6144, fp4=True),
        ),
        Cell(
            why="13 tokens: a partial second token tile",
            spec=dict(m=13, k=6144, fp4=True),
        ),
        Cell(
            why="the 48-token maximum, six token tiles, logits only",
            spec=dict(m=48, k=6144, fp4=False),
        ),
        Cell(
            why="the 48-token maximum with the NVFP4 activation",
            spec=dict(m=48, k=6144, fp4=True),
        ),
        # Every other row count the router takes (modeling's ROUTER_TC), with and without the NVFP4 activation.
        *(
            Cell(
                why=f"{m} rows: a router row count between the ones above",
                spec=dict(m=m, k=6144, fp4=m % 2 == 1),
            )
            for m in range(2, 49)
            if m not in (2, 8, 13, 48)
        ),
    ),
    4: (
        Cell(
            why="9 rows: a 3-request verify past the MoE engine", spec=dict(m=9, k=6144, fp4=False)
        ),
        Cell(why="10 rows", spec=dict(m=10, k=6144, fp4=False)),
        Cell(why="11 rows", spec=dict(m=11, k=6144, fp4=False)),
        Cell(why="12 rows: 3 requests x 4 tokens", spec=dict(m=12, k=6144, fp4=False)),
        Cell(why="13 rows: a partial second token tile", spec=dict(m=13, k=6144, fp4=False)),
        Cell(why="14 rows", spec=dict(m=14, k=6144, fp4=False)),
        Cell(why="15 rows", spec=dict(m=15, k=6144, fp4=False)),
        Cell(
            why="16 rows: 4 requests x 4 tokens, two full token tiles",
            spec=dict(m=16, k=6144, fp4=False),
        ),
    ),
    8: (
        Cell(
            why="two tokens, logits only (the smallest call the target makes here)",
            spec=dict(m=2, k=6144, fp4=False),
        ),
        Cell(
            why="8 tokens with the NVFP4 activation: one 8-token MMA tile, every expert tile quantizes one row",
            spec=dict(m=8, k=6144, fp4=True),
        ),
        Cell(
            why="13 tokens: a partial second token tile",
            spec=dict(m=13, k=6144, fp4=True),
        ),
        Cell(
            why="the 48-token maximum, six token tiles, logits only",
            spec=dict(m=48, k=6144, fp4=False),
        ),
        Cell(
            why="the 48-token maximum with the NVFP4 activation",
            spec=dict(m=48, k=6144, fp4=True),
        ),
        # Every other row count the router takes (modeling's ROUTER_TC), with and without the NVFP4 activation.
        *(
            Cell(
                why=f"{m} rows: a router row count between the ones above",
                spec=dict(m=m, k=6144, fp4=m % 2 == 1),
            )
            for m in range(2, 49)
            if m not in (2, 8, 13, 48)
        ),
    ),
}


class M3RouterTc(OpWrapper):
    """Router logits on TF32 tensor cores for 128 experts and 1..48 tokens: ``x @ tf32(w)^T``, fp32 accumulate.

    ``x`` is BF16 (exact in TF32), ``w`` the fp32 router weight, rounded to TF32 in the kernel. The 16 K slices'
    partials are summed in slice order whatever cluster size the device co-schedules (16 or 8 CTAs), so the result
    does not depend on it. With ``sf_scale`` also the NVFP4 activation as ``m3_router_gemv_fp4``: returns
    ``[logits]`` or ``[logits, x_fp4, x_sf]``.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS_BY_TP = CELLS_BY_TP

    # The target routes 1..48 tokens here: plain logits on the unfused path, logits + NVFP4 on the fused-router path.
    CELLS: tuple[Cell, ...] = tuple(cell for cells in CELLS_BY_TP.values() for cell in cells)

    note = """
    The kernel uses 16-CTA clusters when the device co-schedules eight of them and 8-CTA clusters otherwise; both
    sum the same 16 slice partials in the same order. Only the one the test device picks is driven.
    """

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor, sf_scale: Optional[torch.Tensor], pdl: bool
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.minimax_m3_router_tc(x, w, sf_scale, pdl)

    def reference(
        self, x: torch.Tensor, w: torch.Tensor, sf_scale: Optional[torch.Tensor], pdl: bool
    ) -> tuple[torch.Tensor, ...]:
        """fp32 product of ``x`` and the TF32-rounded weight, plus the NVFP4 recipe with ``sf_scale``."""
        with true_fp32_matmul():
            logits = x.float() @ tf32_rna(w).t()
        if sf_scale is None:
            return (logits,)
        packed, sf, _ = nvfp4_quantize(x, sf_scale)
        return logits, packed, sf

    def is_valid(
        self, x: torch.Tensor, w: torch.Tensor, sf_scale: Optional[torch.Tensor], pdl: bool
    ) -> None:
        assert x.data_ptr() % 16 == 0, "x must start 16-byte aligned (cp.async.bulk source)"
        if sf_scale is not None:
            value = float(sf_scale.reshape(()))
            assert value > 0.0 and value != float("inf"), (
                f"sf_scale must be positive and finite, got {value}"
            )


m3_router_tc = M3RouterTc()
