# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch references of the indexer prologue's arithmetic, shared by its two entries.

fp32 math rounded where the kernel rounds: the LayerNorm output and the RoPE output to bf16, one
rounding for each fused multiply-add, ``fused_cat_fp8``'s per-row scale and E4M3 conversion, and
the indexer K-cache scatter's byte placement.
"""

from __future__ import annotations

from typing import Tuple

import torch


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 ``a * b + c`` with one rounding: the fp64 product of two fp32 values is exact."""
    return (a.double() * b.double() + c.double()).float()


def rope_rows(
    x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, rope_dim: int, interleave: bool
) -> torch.Tensor:
    """RoPE on the first ``rope_dim`` values of the fp32 rows ``x`` [..., D] at ``positions``
    (one per leading row), rounded to bf16 and returned as fp32.

    ``cos_sin`` rows are ``[cos | sin]`` of ``rope_dim / 2`` values each. Interleaved pairs
    (2i, 2i + 1) share column i and the even value subtracts; otherwise the two halves pair up and
    the first half subtracts.
    """
    d = torch.arange(x.shape[-1], device=x.device)
    half = rope_dim // 2
    is_rope = d < rope_dim
    if interleave:
        partner = torch.where(is_rope, d ^ 1, d)
        cs_col = d // 2
        neg = d % 2 == 0
    else:
        partner = torch.where(d < half, d + half, torch.where(is_rope, d - half, d))
        cs_col = d % half
        neg = d < half
    cs_col = torch.where(is_rope, cs_col, torch.zeros_like(cs_col))
    table = cos_sin[positions.long()]
    while table.dim() < x.dim():
        table = table.unsqueeze(-2)
    cos = torch.where(is_rope, table[..., cs_col], torch.ones_like(table[..., cs_col]))
    sin = torch.where(is_rope, table[..., half + cs_col], torch.zeros_like(table[..., cs_col]))
    xp = x[..., partner]
    rotated = fma(x, cos, torch.where(neg, -xp, xp) * sin)
    return torch.where(is_rope, rotated, x).to(torch.bfloat16).float()


def quantize_rows(v: torch.Tensor, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """``fused_cat_fp8``'s rule over the last dim of fp32 ``v``: (E4M3 values, fp32 scales with a
    trailing dim of 1). scale = max(amax, 1e-12) / 448, rounded up to a power of two with
    ``use_ue8m0``; values = e4m3(v * (1 / scale))."""
    amax = v.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
    ratio = amax * 0.002232142857142857
    if use_ue8m0:
        bits = ratio.view(torch.int32)
        exp_bits = bits & 0x7F800000
        exp_bits = torch.where((bits & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
        scale = exp_bits.view(torch.float32)
    else:
        scale = ratio
    # The kernel multiplies by the MUFU reciprocal, exact for a power-of-two scale.
    return (v * (1.0 / scale)).to(torch.float8_e4m3fn), scale


def reference_k_rows(
    indexer_k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    rope_dim: int,
    interleave: bool,
    use_ue8m0: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(k_fp8 [N, D], k_scale [N, 1])``: LayerNorm of bf16(indexer_k) rounded to bf16, RoPE
    rounded to bf16, then the E4M3 rule."""
    rows = indexer_k.shape[0]
    kx = indexer_k.to(torch.bfloat16).float()
    mean = kx.mean(dim=1, keepdim=True)
    kc = kx - mean
    rstd = torch.rsqrt((kc * kc).mean(dim=1, keepdim=True) + eps)
    kn = fma(kc * rstd, ln_weight.float()[None, :], ln_bias.float()[None, :]).to(torch.bfloat16)
    kv = rope_rows(kn.float(), positions[:rows], cos_sin, rope_dim, interleave)
    return quantize_rows(kv, use_ue8m0)


def reference_q_rows(
    q: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    weight_scale: float,
    rope_dim: int,
    interleave: bool,
    use_ue8m0: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(q_fp8 [N, H, D], q_scale [N, H, 1], weights [N, H] fp32)``: RoPE of the bf16 Q rounded
    to bf16, the E4M3 rule per head, and ``(weights * q_scale) * weight_scale`` in fp32."""
    rows = q.shape[0]
    qv = rope_rows(q.float(), positions[:rows], cos_sin, rope_dim, interleave)
    q8, q_scale = quantize_rows(qv, use_ue8m0)
    w = (weights.float() * q_scale[..., 0]) * torch.tensor(weight_scale, dtype=torch.float32)
    return q8, q_scale, w


def unravel(flat: torch.Tensor, shape: torch.Size) -> Tuple[torch.Tensor, ...]:
    """The 4-D index of each logical flat index of a [d0, d1, d2, d3] view."""
    d1, d2, d3 = shape[1:]
    i3 = flat % d3
    flat = flat // d3
    i2 = flat % d2
    flat = flat // d2
    i1 = flat % d1
    return flat // d1, i1, i2, i3


def write_cache_rows(
    cache: torch.Tensor,
    k8: torch.Tensor,
    k_scale: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    num_cache_tokens: int,
) -> None:
    """The indexer K-cache scatter, in place: for each of the first ``num_cache_tokens`` rows with
    both slots >= 0, its 128 FP8 bytes at the logical flat byte index ``slot_mapping_fp8[row]`` of
    the 4-D byte view ``cache`` and its fp32 scale's 4 little-endian bytes at
    ``slot_mapping_scale[row]``."""
    view = cache.view(torch.uint8)
    fp8_slots = slot_mapping_fp8[:num_cache_tokens].long()
    scale_slots = slot_mapping_scale[:num_cache_tokens].long()
    keep = (fp8_slots >= 0) & (scale_slots >= 0)
    rows = torch.nonzero(keep).flatten()
    if rows.numel() == 0:
        return
    data = k8.view(torch.uint8)[rows]
    scale_bytes = k_scale.reshape(-1, 1).contiguous().view(torch.uint8)[rows]
    width = data.shape[1]
    fp8_index = fp8_slots[rows, None] + torch.arange(width, device=view.device)
    scale_index = scale_slots[rows, None] + torch.arange(4, device=view.device)
    view[unravel(fp8_index, view.shape)] = data
    view[unravel(scale_index, view.shape)] = scale_bytes
