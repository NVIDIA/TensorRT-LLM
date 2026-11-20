# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 form of a BF16 weight that is exactly an MXFP8 dequantization, shared by the M3.1 targets that run the
DSpark drafter's projections as MXFP8 (the TP8 target's DRAFTER_MXFP8 GEMV, the DEP target's DEP_DRAFTER_MXFP8 GEMMs).

The drafter checkpoint ships its decoder-layer linears as MXFP8 (E4M3 codes, one UE8M0 scale per 32 columns) and the
stock drafter dequantizes them to BF16 at load. ``mxfp8_from_bf16`` recovers codes and scales from that BF16 weight and
checks that they dequantize back to the same BF16 bits; ``swizzle_sf`` lays the scales out in the 128x4-swizzled order
``trtllm::mxfp8_quantize(x, True)`` writes for an activation, which the MXFP8 GEMMs take for the weight too.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

E4M3_MAX = 448.0
BLOCK = 32


def mxfp8_from_bf16(weight: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The MXFP8 form (E4M3 codes [N, K], UE8M0 scale bytes [N, K / 32]) of a BF16 weight that is exactly an MXFP8
    dequantization, or None. Each block takes the smallest power-of-two scale that fits its amax in E4M3, which is at
    most the shipped scale, so the codes are exact whenever the weight came from MXFP8."""
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or weight.shape[1] % BLOCK != 0:
        return None
    n, k = weight.shape
    blocks = weight.float().view(n, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(-1)
    exp = torch.ceil(torch.log2(amax.clamp_min(2.0**-126) / E4M3_MAX)).clamp(-127, 127)
    scale = torch.exp2(exp).unsqueeze(-1)
    codes = (blocks / scale).to(torch.float8_e4m3fn)
    back = (codes.float() * scale).view(n, k).to(torch.bfloat16)
    if not torch.equal(back, weight):
        return None
    return codes.view(n, k).contiguous(), (exp + 127).to(torch.uint8).contiguous()


def swizzle_sf(byte: torch.Tensor) -> torch.Tensor:
    """Scatter [rows, K / 32] scale bytes (K / 32 a multiple of 4) into the 128x4-swizzled buffer, zero padding to whole
    128-row tiles."""
    rows, cols = byte.shape
    if cols % 4 != 0:
        raise ValueError(f"{cols} scale columns are not a multiple of 4")
    padded_rows = (rows + 127) // 128 * 128
    r = torch.arange(rows, device=byte.device).view(-1, 1)
    c = torch.arange(cols, device=byte.device).view(1, -1)
    index = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * cols)
    )
    out = torch.zeros(padded_rows * cols, dtype=torch.uint8, device=byte.device)
    out[index.reshape(-1)] = byte.reshape(-1)
    return out


def unswizzle_sf(swizzled: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The [rows, cols] scale bytes of ``swizzle_sf``'s layout (the inverse of its scatter, padding dropped)."""
    r = torch.arange(rows, device=swizzled.device).view(-1, 1)
    c = torch.arange(cols, device=swizzled.device).view(1, -1)
    index = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * cols)
    )
    return swizzled.view(torch.uint8).reshape(-1)[index.reshape(-1)].view(rows, cols)
