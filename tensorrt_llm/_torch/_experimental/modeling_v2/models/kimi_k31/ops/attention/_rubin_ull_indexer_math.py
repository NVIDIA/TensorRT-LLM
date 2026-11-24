# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch arithmetic of the K3.1 indexer prologue, shared by the references of
``dsa_index_front`` and ``dsa_indexer_prologue`` (not an entry).

The query half is exact: given the BF16 query rows, the FP8 codes and the scaled head weights are
a function of them that FP32 arithmetic reproduces bit for bit (the scale is a power of two, so
the kernel's ``rcp.approx`` of it is exact). The key half is the FP64 LayerNorm the kernel
approximates (FP32 sums, ``rsqrt.approx``, a BF16 rounding before the FP8 one).
"""

from __future__ import annotations

import torch

HEADS, DIM = 32, 128
#: The indexer K cache page: 64 tokens of 128 FP8 codes, then their 64 FP32 scales.
PAGE, PAGE_BYTES = 64, 64 * (DIM + 4)


def _f32_bits(bits: int) -> torch.Tensor:
    return torch.tensor([bits], dtype=torch.int32).view(torch.float32)


#: ``0f2B8CBCCC`` (1e-12) and ``0f3B124925`` (1 / 448): the kernels' FP32 constants.
_TINY, _INV448 = _f32_bits(0x2B8CBCCC), _f32_bits(0x3B124925)


def ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """FP32 ``max(amax, 1e-12) * (1 / 448)`` rounded up to a power of two (UE8M0), as the kernels
    compute it (``fused_cat_fp8``'s scale)."""
    a = torch.maximum(amax.float(), _TINY.to(amax.device))
    bits = (a * _INV448.to(amax.device)).view(torch.int32)
    exp = bits & 0x7F800000
    exp = torch.where((bits & 0x007FFFFF) != 0, exp + 0x00800000, exp)
    return exp.view(torch.float32)


def quantize_queries(
    q_rows: torch.Tensor, iw: torch.Tensor, weight_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 queries [M, 32, 128] and FP32 head weights [M, 32] of BF16 query rows [M, 4096] and BF16
    head weights ``iw`` [M, 32]: per head ``scale = ue8m0(amax / 448)``, ``q / scale`` rounded to
    E4M3, ``(iw * scale) * weight_scale`` in FP32 (each product rounded)."""
    q = q_rows.float().view(-1, HEADS, DIM)
    scale = ue8m0_scale(q.abs().amax(-1))
    codes = (q * (1.0 / scale).unsqueeze(-1)).to(torch.float8_e4m3fn)
    ws = torch.tensor(weight_scale, dtype=torch.float32, device=q.device)
    return codes, (iw.float() * scale) * ws


def key_layer_norm(
    ik: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, eps: float
) -> torch.Tensor:
    """FP64 LayerNorm of the BF16 keys [M, 128] with the FP32 [128] weight and bias."""
    x = ik.double()
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).pow(2).mean(-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * ln_w.double() + ln_b.double()


def cache_rows(
    cache: torch.Tensor, slot_fp8: torch.Tensor, slot_scale: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The FP8 codes [n, 128] and FP32 scales [n] the first ``n`` slot mappings address in the
    page-major indexer K cache ``cache`` [blocks, 64, 1, 132] uint8 (flat byte offsets, as
    ``indexer_k_cache_scatter_op`` takes them)."""
    flat = cache.reshape(-1)
    col = torch.arange(DIM, device=cache.device)
    codes = flat[slot_fp8[:n].long().unsqueeze(1) + col].view(torch.float8_e4m3fn)
    four = torch.arange(4, device=cache.device)
    scales = flat[slot_scale[:n].long().unsqueeze(1) + four].contiguous().view(torch.float32)
    return codes, scales.view(-1)
