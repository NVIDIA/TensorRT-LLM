# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One NVFP4 global weight scale for an expert's fused FC1 (gate `w1` + up `w3`); plain torch, load time only.

The checkpoint gives `w1` and `w3` each their own `weight_scale_2`, but every FC1 runner of this family applies ONE
weight scalar to the fused `[gate; up]` GEMM. As stock's single-alpha (CUTLASS) path does
(`_resolve_gate_up_weight_scale_2` in `fused_moe/quantization.py`), the shared scale is the LARGER of the two. The
half whose own scale is smaller has its e4m3 block scales multiplied by `its weight_scale_2 / the larger`, a factor
<= 1, so a folded block scale never exceeds the original and cannot overflow e4m3 (`.to(torch.float8_e4m3fn)` does
not saturate: past 464 it gives NaN). The dequantized weight `code * block_scale * weight_scale_2` is unchanged up to
one e4m3 rounding of the folded block scales. An expert whose two scales are equal is left byte for byte as it is.
"""

from __future__ import annotations

from typing import Tuple

import torch

_E4M3_MAX = 448.0


def fc1_global_scale(
    ws1: torch.Tensor, ws3: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`ws1`, `ws3`: `weight_scale_2` of `w1` (gate) and `w3` (up), any matching shape (one per expert).

    -> `(ws13, w1_fold, w3_fold)`, float32, same shape: `ws13 = max(ws1, ws3)`, the FC1 weight scalar every alpha
    of the fused FC1 is built from, and the factors `ws1 / ws13`, `ws3 / ws13` (both in `(0, 1]`, exactly 1.0 for
    the half that holds the maximum) that `fold_block_scales` applies to that half's block scales."""
    ws1, ws3 = ws1.float(), ws3.float()
    ws13 = torch.maximum(ws1, ws3)
    return ws13.contiguous(), (ws1 / ws13).contiguous(), (ws3 / ws13).contiguous()


def fold_block_scales(sf: torch.Tensor, fold: torch.Tensor) -> torch.Tensor:
    """`sf` e4m3 block scales (any 1-byte dtype) times `fold`, rounded to e4m3 -> float8_e4m3fn, `sf`'s shape.

    `fold` is a 0-d factor for the whole tensor, or `[S]` with one factor per leading index of `sf`. Factors are
    the `<= 1` ones of `fc1_global_scale`; the product is clamped to the e4m3 range anyway, so no input gives NaN.
    """
    f = fold.float()
    if f.dim() == 1:
        f = f.reshape(-1, *([1] * (sf.dim() - 1)))
    x = sf.view(torch.float8_e4m3fn).float() * f
    return x.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
