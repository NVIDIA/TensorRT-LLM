# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch readers of the NVFP4 layouts this target's GEMMs take.

The entries' ``reference`` methods take the packed operands their kernels take,
not the values those were built from, so the reference covers the layout as
well as the arithmetic: a scale order the kernel and the reference disagreed
about would otherwise never show. Three layouts:

* packed E2M1 rows ``[R, K/2]`` uint8, element ``2i`` in the low nibble of byte ``i``;
* E4M3 block scales, one per 16 elements of K, either linear ``[R, K/16]`` or in the
  128x4-swizzled order ``fp4_quantize`` and ``nvfp4_gemm`` use (rows padded to 128,
  scale columns to 4, padding zero);
* the Marlin W4A16 layout ``gptq_marlin_repack`` + ``marlin_permute_scales`` +
  ``nvfp4_marlin_process_scales`` / ``_global_scale`` produce: int32 tiles of 16 (K) x
  64 (N) nibbles under Marlin's in-tile permutation, S0E5M3 scale bytes under the
  scale permutation, and a BF16 global scale carrying a 2^119 exponent bias.

Everything here is exact (integer and power-of-two arithmetic); nothing rounds.
Private to the target.
"""

from __future__ import annotations

import functools

import torch

SF_VEC = 16

#: e2m1 code (bit 3 = sign) -> value.
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: ``nvfp4_marlin_process_global_scale`` multiplies the BF16 global scale by this.
MARLIN_GLOBAL_SCALE_BIAS = 2.0**119

#: Smallest normal E4M3 value. Marlin's S0E5M3 dequantization yields zero for a block
#: whose E4M3 scale is below it (subnormal); see `dequantize_marlin`.
E4M3_MIN_NORMAL = 2.0**-6


def pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """``[R, K/2]`` packed bytes -> ``[R, K]`` 4-bit codes (element ``2i`` in the low nibble)."""
    packed = packed.view(torch.uint8)
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)


def e2m1_to_float(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return table[codes.long()]


def e4m3_to_float(sf: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes (uint8 or float8_e4m3fn) -> fp32, exact."""
    return sf.view(torch.uint8).view(torch.float8_e4m3fn).float()


def swizzled_sf_index(rows: int, cols: int, device) -> torch.Tensor:
    """``[rows, cols]`` scale coordinates -> flat offsets in the 128x4-swizzled buffer."""
    padded_cols = pad_up(cols, 4)
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(cols, device=device).view(1, -1)
    return (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )


def swizzled_sf_numel(rows: int, cols: int) -> int:
    return pad_up(rows, 128) * pad_up(cols, 4)


def unswizzle_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Swizzled scale buffer -> linear ``[rows, cols]`` uint8."""
    index = swizzled_sf_index(rows, cols, sf.device).reshape(-1)
    return sf.reshape(-1).view(torch.uint8)[index].reshape(rows, cols)


def swizzle_sf(sf: torch.Tensor) -> torch.Tensor:
    """Linear ``[rows, cols]`` scale bytes -> the swizzled 1-D buffer, padding zero."""
    rows, cols = sf.shape
    out = torch.zeros(swizzled_sf_numel(rows, cols), dtype=torch.uint8, device=sf.device)
    out[swizzled_sf_index(rows, cols, sf.device).reshape(-1)] = sf.reshape(-1).view(torch.uint8)
    return out


def dequantize(
    packed: torch.Tensor, sf_linear: torch.Tensor, global_scale: float = 1.0
) -> torch.Tensor:
    """``[R, K/2]`` codes and linear ``[R, K/16]`` scales -> ``[R, K]`` fp32 values times
    ``global_scale`` (exact for a power-of-two global scale)."""
    values = e2m1_to_float(unpack_codes(packed))
    rows, k = values.shape
    scale = e4m3_to_float(sf_linear.reshape(rows, k // SF_VEC))
    out = (values.view(rows, k // SF_VEC, SF_VEC) * scale.unsqueeze(-1)).view(rows, k)
    return out * global_scale


def dequantize_swizzled(packed: torch.Tensor, sf_swizzled: torch.Tensor) -> torch.Tensor:
    """As `dequantize`, the scales read from the swizzled buffer, no global scale."""
    rows, half = packed.shape
    return dequantize(packed, unswizzle_sf(sf_swizzled, rows, half * 2 // SF_VEC))


# ── Marlin W4A16 layout ──────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _marlin_weight_perm_list() -> tuple[int, ...]:
    """Marlin's nibble order inside one 1024-nibble chunk (four 16x16 tiles) for 4-bit
    weights: entry ``j`` is the source position of packed position ``j``."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend(p + 256 * j for p in perm1)
    interleave = (0, 2, 4, 6, 1, 3, 5, 7)
    return tuple(perm[g * 8 + i] for g in range(len(perm) // 8) for i in interleave)


@functools.lru_cache(maxsize=None)
def _marlin_scale_perm_list() -> tuple[int, ...]:
    """``marlin_permute_scales``'s grouped order (64 entries)."""
    return tuple(i + 8 * j for i in range(8) for j in range(8))


def _inverse(perm: tuple[int, ...], device) -> torch.Tensor:
    return torch.argsort(torch.tensor(perm, device=device))


def unpack_marlin_codes(
    weight: torch.Tensor, size_k: int, size_n: int, n0: int = 0, n1: int | None = None
) -> torch.Tensor:
    """Marlin int32 ``[K/16, 2N]`` -> E2M1 codes ``[n1-n0, K]`` for output columns
    ``[n0, n1)``; both bounds multiples of 64 (one permutation chunk)."""
    n1 = size_n if n1 is None else n1
    assert n0 % 64 == 0 and n1 % 64 == 0 and size_k % 16 == 0
    nn = n1 - n0
    words = weight.view(torch.int32).reshape(size_k // 16, 2 * size_n)[:, 2 * n0 : 2 * n1]
    packed = words.contiguous().view(torch.uint8)  # [K/16, 8 nn], little-endian bytes
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(size_k // 16, 16 * nn)
    inv = _inverse(_marlin_weight_perm_list(), weight.device)
    tiles = nibbles.reshape(-1, inv.numel())[:, inv].reshape(size_k // 16, 16 * nn)
    codes_kn = tiles.reshape(size_k // 16, nn // 16, 16, 16).permute(0, 2, 1, 3).reshape(size_k, nn)
    return codes_kn.t().contiguous()


def unpack_marlin_scales(scale: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    """Marlin S0E5M3 scale bytes ``[K/16, N]`` -> the E4M3 block scales as fp32 ``[N, K/16]``.

    ``nvfp4_marlin_process_scales`` keeps bits 14..7 of ``half(sf * 2^7)``; an E4M3 value
    times 2^7 is a normal half whose low 7 bits are zero, so shifting back is exact.
    """
    groups = size_k // SF_VEC
    raw = scale.view(torch.uint8).reshape(groups, size_n)
    half = (raw.to(torch.int16) << 7).view(torch.float16)
    value = half.float() * 2.0**-7
    value = value.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(groups, size_n)
    inv = _inverse(_marlin_scale_perm_list(), scale.device)
    value = value.reshape(-1, inv.numel())[:, inv].reshape(groups, size_n)
    return value.t().contiguous()


def marlin_global_scale(weight_global_scale: torch.Tensor) -> float:
    """The dequantization global scale the Marlin BF16 global-scale operand carries."""
    return float(weight_global_scale.float().reshape(-1)[0].item()) / MARLIN_GLOBAL_SCALE_BIAS


def dequantize_marlin(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    size_k: int,
    size_n: int,
    n0: int = 0,
    n1: int | None = None,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Output columns ``[n0, n1)`` of a Marlin NVFP4 weight as fp32 ``[n1-n0, K]``: E2M1
    times its E4M3 block scale times the global scale, with every block whose E4M3 scale
    is subnormal (``0 < sf < 2^-6``) zeroed, as Marlin's S0E5M3 dequantization computes
    it. ``scales`` may pass a precomputed `unpack_marlin_scales` result."""
    n1 = size_n if n1 is None else n1
    codes = unpack_marlin_codes(weight, size_k, size_n, n0, n1)
    if scales is None:
        scales = unpack_marlin_scales(weight_scale, size_k, size_n)
    sf = scales[n0:n1]
    sf = torch.where(sf < E4M3_MIN_NORMAL, torch.zeros_like(sf), sf)
    values = e2m1_to_float(codes).view(n1 - n0, size_k // SF_VEC, SF_VEC) * sf.unsqueeze(-1)
    return values.view(n1 - n0, size_k) * marlin_global_scale(weight_global_scale)
