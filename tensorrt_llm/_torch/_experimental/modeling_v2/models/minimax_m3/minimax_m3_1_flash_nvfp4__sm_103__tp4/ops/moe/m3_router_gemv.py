# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp32 router GEMV of the decode MoE: ``logits[M, E] = float(x) @ W^T`` at one token."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
#: Midpoints between consecutive e2m1 magnitudes; ``_TIE_UP[i]``: round-to-nearest-even takes the upper code.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)
#: The kernel builds each block's output scale through two ``rcp.approx.ftz.f32`` reciprocals (~2^-23 relative):
#: a value this close to an e2m1 midpoint may round to either neighbour, everything further out must match.
TIE_WINDOW = 2.0**-20


def nvfp4_quantize(
    x: torch.Tensor, sf_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp4_quantize(x, sf_scale, 16, False, False)``: (packed e2m1 [M, K / 2], linear E4M3 scales [M * K / 16],
    the pre-rounding scaled values [M, K] that ``near_tie`` reads).

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


def near_tie(scaled: torch.Tensor) -> torch.Tensor:
    """Elements whose e2m1 rounding the kernel's approximate reciprocals decide (see ``TIE_WINDOW``)."""
    mid = torch.tensor(_E2M1_MIDPOINTS, device=scaled.device, dtype=torch.float32)
    a = scaled.abs().unsqueeze(-1)
    return ((a - mid).abs() <= TIE_WINDOW * a).any(-1)


def fp32_logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``float(x) @ w^T`` as a true fp32 product (TF32 off)."""
    with true_fp32_matmul():
        return x.float() @ w.t()


class M3RouterGemv(OpWrapper):
    """fp32 router logits ``float(x[M, K]) @ w[E, K]^T`` with exact fp32 FMA, one CTA per expert.

    ``x`` is BF16 with 1..16 rows, unit column stride and 16-byte rows; ``w`` is the fp32 router weight; K a
    multiple of 1024. ``rows`` experts per CTA; only 1 is built.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target takes the GEMV at exactly one token (2..48 go to the tensor-core router): 128 experts, hidden 6144.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode token, 128 experts x hidden 6144",
            spec=dict(m=1, e=128, k=6144),
        ),
    )

    note = """
    The op accepts up to 16 rows; the target sends one, so wider calls are not cells. The logits differ from a true
    fp32 product only in accumulation order (per-thread FMA chains over K / 1024 strided column groups, a warp tree
    and an 8-warp sum).
    """

    def __call__(self, x: torch.Tensor, w: torch.Tensor, rows: int, pdl: bool) -> torch.Tensor:
        return torch.ops.trtllm.minimax_m3_router_gemv(x, w, rows, pdl)

    def reference(self, x: torch.Tensor, w: torch.Tensor, rows: int, pdl: bool) -> torch.Tensor:
        """True fp32 product; ``rows`` and ``pdl`` do not enter the arithmetic."""
        return fp32_logits(x, w)

    def is_valid(self, x: torch.Tensor, w: torch.Tensor, rows: int, pdl: bool) -> None:
        # The activation rows are bulk-copied from x's base address; the op checks the row stride but not the base,
        # and a misaligned bulk copy faults the context rather than raising.
        assert x.data_ptr() % 16 == 0, "x must start 16-byte aligned (cp.async.bulk source)"


m3_router_gemv = M3RouterGemv()
