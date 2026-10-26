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

# ROUTER48 at TP2: the sm_107 TP8 wrapper (cells 2..48) with the TP4 wrapper's inline NVFP4 reference quantize
# (this target has no m3_router_gemv module).
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
#: Midpoints between consecutive e2m1 magnitudes; ``_TIE_UP[i]``: round-to-nearest-even takes the upper code.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)


def nvfp4_quantize(
    x: torch.Tensor, sf_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp4_quantize(x, sf_scale, 16, False, False)``: (packed e2m1 [M, K / 2], linear E4M3 scales [M * K / 16],
    the pre-rounding scaled values [M, K]).

    Per 16 columns: scale = e4m3(sf_scale * amax / 6), data = x * sf_scale / scale rounded to nearest-even e2m1.
    """
    gs = sf_scale.float().reshape(())
    m, k = x.shape
    blocks = x.float().view(m, k // 16, 16)
    vecmax = blocks.abs().amax(-1)
    sf = (gs * (vecmax / _E2M1_MAX)).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
    out_scale = torch.where(vecmax != 0, gs / sf.float(), torch.zeros_like(vecmax))
    scaled = (blocks * out_scale.unsqueeze(-1)).view(m, k)
    mid = torch.tensor(_E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    tie_up = torch.tensor(_TIE_UP, device=x.device, dtype=torch.bool)
    a = scaled.abs().unsqueeze(-1)
    code = ((a > mid).sum(-1) + ((a == mid) & tie_up).any(-1)).to(torch.uint8)
    negative = (scaled < 0) | ((scaled == 0) & torch.signbit(scaled))
    code = code | (negative.to(torch.uint8) << 3)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, sf.view(torch.uint8).reshape(-1), scaled


def tf32_rna(w: torch.Tensor) -> torch.Tensor:
    """fp32 -> TF32 as ``cvt.rna.tf32.f32``: keep 10 mantissa bits, round to nearest with ties away from zero."""
    bits = w.contiguous().view(torch.int32)
    rounded = (bits + 0x1000) & ~0x1FFF
    finite = (bits & 0x7F800000) != 0x7F800000
    return torch.where(finite, rounded, bits).view(torch.float32)


class M3RouterTc(OpWrapper):
    """Router logits on TF32 tensor cores for 128 experts and 1..48 tokens: ``x @ tf32(w)^T``, fp32 accumulate.

    ``x`` is BF16 (exact in TF32), ``w`` the fp32 router weight, rounded to TF32 in the kernel. The 16 K slices'
    partials are summed in slice order whatever cluster size the device co-schedules (16 or 8 CTAs), so the result
    does not depend on it. With ``sf_scale`` also the NVFP4 activation (``fp4_quantize``'s bytes): returns
    ``[logits]`` or ``[logits, x_fp4, x_sf]``.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The target routes 2..48 tokens here: plain logits on the unfused path, logits + NVFP4 on the fused-router path.
    CELLS: tuple[Cell, ...] = (
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
    )

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
