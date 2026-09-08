# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only MXFP4 (E2M1 + E8M0) quantization utilities.

Kimi K3's routed expert Linear weights are stored in the
``mxfp4-pack-quantized`` format with ``group_size=32`` (see
``config.json::quantization_config``). Each 32-element group along the
input axis has a single ``uint8`` E8M0 scale (biased exponent, value
``2 ** (scale_u8 - 127)``) and each fp4 element is one of the eight
representable E2M1 magnitudes ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` signed.
Two fp4 values pack into a single ``uint8``: low nibble at even element
index, high nibble at odd element index.

The pack/unpack path is pure PyTorch so random-weight tests can
construct MXFP4 tensors on-the-fly without a full quantization
pipeline. Correctness properties tests depend on:

* ``quantize_last_dim_mxfp4`` picks the E8M0 scale so the largest fp4
  magnitude ``6.0`` covers the group's peak absolute value, then
  quantizes each element to the nearest representable magnitude.
* ``dequantize_last_dim_mxfp4`` returns exactly what was stored — no
  rounding, no dtype loss beyond the E2M1 grid the values were
  quantized onto. So ``dequantize(quantize(x))`` is idempotent.
"""

from __future__ import annotations

from typing import Tuple

import torch

# E2M1 representable magnitudes at fp4 codes 0..7. Signed via the top
# bit: codes 8..15 are the negatives of codes 0..7.
_FP4_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
_FP4_VALUES = torch.cat([_FP4_MAGNITUDES, -_FP4_MAGNITUDES], dim=0)

FP4_MAX_MAGNITUDE = 6.0
E8M0_BIAS = 127
E8M0_MIN = 0
E8M0_MAX = 255
DEFAULT_GROUP_SIZE = 32


def _pack_two_nibbles(fp4_codes: torch.Tensor) -> torch.Tensor:
    assert fp4_codes.dtype == torch.uint8, fp4_codes.dtype
    assert fp4_codes.shape[-1] % 2 == 0, fp4_codes.shape
    lo = fp4_codes[..., 0::2] & 0x0F
    hi = fp4_codes[..., 1::2] & 0x0F
    return (hi << 4) | lo


def _unpack_two_nibbles(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8, packed.dtype
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _quantize_group_e2m1(
    group: torch.Tensor, atol_e8m0: float = 1e-30
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_abs = group.abs().amax(dim=-1)
    zero_mask = max_abs < atol_e8m0
    ratio = max_abs / FP4_MAX_MAGNITUDE
    ratio = torch.where(zero_mask, torch.ones_like(ratio), ratio)
    log2r = torch.log2(ratio)
    e_signed = torch.ceil(log2r).to(torch.int64) + E8M0_BIAS
    e_signed = e_signed.clamp(min=E8M0_MIN, max=E8M0_MAX)
    e_signed = torch.where(zero_mask, torch.zeros_like(e_signed), e_signed)
    scale_u8 = e_signed.to(torch.uint8)

    scale_val = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=group.device),
        (scale_u8.to(torch.float32) - E8M0_BIAS),
    )
    scaled = group / scale_val.unsqueeze(-1)

    values = _FP4_VALUES.to(device=group.device)
    diffs = (scaled.unsqueeze(-1) - values.view(*(1,) * scaled.ndim, 16)).abs()
    codes = diffs.argmin(dim=-1).to(torch.uint8)
    return codes, scale_u8


def _dequantize_group_e2m1(codes: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    values = _FP4_VALUES.to(device=codes.device)
    magnitudes = values[codes.to(torch.long)]
    scale_val = torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=codes.device),
        (scale_u8.to(torch.float32) - E8M0_BIAS),
    )
    return magnitudes * scale_val.unsqueeze(-1)


def quantize_last_dim_mxfp4(
    x: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an arbitrary-shape fp32 tensor along the last dim to MXFP4.

    Returns ``(packed_u8, scales_u8)`` where:

    * ``packed_u8`` has shape ``x.shape[:-1] + (x.shape[-1] // 2,)``,
      dtype uint8; two fp4 codes packed per byte.
    * ``scales_u8`` has shape ``x.shape[:-1] + (x.shape[-1] // group_size,)``,
      dtype uint8; one E8M0 scale byte per group of ``group_size`` elements
      along the last dim.

    Constraints: ``x.shape[-1] % group_size == 0`` and
    ``group_size % 2 == 0``.
    """
    assert x.dtype == torch.float32, f"expected fp32 input, got {x.dtype}"
    assert x.shape[-1] % group_size == 0, (
        f"last dim {x.shape[-1]} not divisible by group_size {group_size}"
    )
    assert group_size % 2 == 0, group_size

    last = x.shape[-1]
    num_groups = last // group_size
    lead = x.shape[:-1]

    grouped = x.reshape(*lead, num_groups, group_size)
    codes, scales = _quantize_group_e2m1(grouped)

    codes_flat = codes.reshape(*lead, num_groups * group_size)
    packed = _pack_two_nibbles(codes_flat)
    return packed, scales


def dequantize_last_dim_mxfp4(
    packed_u8: torch.Tensor,
    scales_u8: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Inverse of :func:`quantize_last_dim_mxfp4`. Returns fp32."""
    assert packed_u8.dtype == torch.uint8, packed_u8.dtype
    assert scales_u8.dtype == torch.uint8, scales_u8.dtype
    lead = packed_u8.shape[:-1]
    n_over_2 = packed_u8.shape[-1]
    n = n_over_2 * 2
    num_groups = scales_u8.shape[-1]
    assert num_groups * group_size == n, (
        f"num_groups {num_groups} * group_size {group_size} != last_dim {n}"
    )

    codes_flat = _unpack_two_nibbles(packed_u8)
    codes_grouped = codes_flat.reshape(*lead, num_groups, group_size)
    deq_grouped = _dequantize_group_e2m1(codes_grouped, scales_u8)
    return deq_grouped.reshape(*lead, n)
