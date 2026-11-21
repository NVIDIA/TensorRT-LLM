# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Independent payload and scale decoders for paired low-precision tests."""

import math

import torch


def decode_x(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Independent 1x128 decode: s[k//128, m] * float(q[m,k])."""
    m, k = payload.shape
    if scale.shape != (math.ceil(k / 128), m) or scale.dtype != torch.float32:
        raise ValueError((payload.shape, scale.shape, scale.dtype))
    return payload.float() * scale.t().repeat_interleave(128, dim=1)[:, :k]


def decode_w(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Independent 128x128 decode; no production dequantization function is used."""
    n, k = payload.shape
    if scale.dtype != torch.float32 or scale.shape != (math.ceil(n / 128), math.ceil(k / 128)):
        raise ValueError((payload.shape, scale.shape, scale.dtype))
    expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return payload.float() * expanded[:n, :k]


def decode_ue8m0(scale: torch.Tensor, groups: int) -> torch.Tensor:
    """Decode four little-endian exponent bytes per int32, exponent bias 127."""
    if scale.dtype != torch.int32:
        raise ValueError("Expected DeepGEMM packed int32 scales")
    words = scale.long()
    exponents = torch.stack([(words >> (8 * i)) & 255 for i in range(4)], dim=-1)
    exponents = exponents.reshape(scale.shape[0], -1)[:, :groups]
    if bool((exponents == 255).any()):
        raise ValueError("Nonfinite UE8M0 scale")
    return torch.exp2(exponents.float() - 127)


def decode_e4m3(payload: torch.Tensor) -> torch.Tensor:
    """Decode scale bytes arithmetically, without a production float8 decoder."""
    bits = payload.view(torch.uint8).long()
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    normal = (1 + mantissa.float() / 8) * torch.exp2(exponent.float() - 7)
    value = torch.where(exponent == 0, mantissa.float() * (2.0**-9), normal)
    value = torch.where((bits & 128) != 0, -value, value)
    return torch.where((exponent == 15) & (mantissa == 7), torch.nan, value)


def unswizzle_scales(payload: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Gather the native R128xC4 SF layout by independently computed byte offsets."""
    groups = math.ceil(columns / 16)
    row = torch.arange(rows, device=payload.device)[:, None]
    group = torch.arange(groups, device=payload.device)[None, :]
    ktiles = math.ceil(groups / 4)
    offsets = (
        (row // 128) * ktiles * 512
        + (group // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + group % 4
    )
    return payload.view(torch.uint8).flatten()[offsets]


def decode_operand(payload: torch.Tensor, scales: torch.Tensor, *, swizzled: bool) -> torch.Tensor:
    """Decode low-nibble-even/high-nibble-odd FP4; result excludes global divisor."""
    if payload.device.type != "cuda" or payload.dtype != torch.uint8:
        raise ValueError("Expected CUDA packed uint8 E2M1")
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=payload.device)
    values = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
    sf = unswizzle_scales(scales, *values.shape) if swizzled else scales
    expanded = decode_e4m3(sf).repeat_interleave(16, dim=1)
    return values * expanded[:, : values.shape[1]]
