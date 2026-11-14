# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch dequantize-then-matmul reference shared by the FP8-weight GEMM entries.

Every GEMM of this family multiplies an E4M3 operand pair whose scales are powers of two (UE8M0
bytes, ``2^(e - 127)``): the weight has one exponent per 128 x 128 block, the activation one per
1 x 32 (MXFP8) or 1 x 128 group. The products of the decoded operands are exact in FP64, so the
reference is the FP64 matmul of the decoded operands rounded once to BF16; the kernels accumulate
the same products in FP32. The scale layouts decoded here:

* DeepGEMM-packed weight scale: ``int32 [N, ceil(K / 512)]`` (strides ``(1, N)``), four K-block
  exponents per word, little-endian.
* K32 R128c4 (the CuTe MXFP8 GEMM's): byte ``((r // 128) * (K / 128) + c // 4) * 512 +
  (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4`` for row ``r``, K32 column ``c``.
* The mid-M GEMM's activation layout: the same 512-byte atoms ordered ``[K / 128][row blocks]``.
* Row-major ``uint8 [M, K / 32]`` (the MXFP8 low-M GEMM's activation scales).
"""

from __future__ import annotations

import torch

#: Smallest magnitude (BF16 bits) the kernels' activation quantization takes as a group maximum.
AMAX_FLOOR_BF16 = 0x38D2


def exp2_ue8m0(e: torch.Tensor) -> torch.Tensor:
    """``2^(e - 127)`` in FP64 for UE8M0 bytes ``e``."""
    return torch.pow(2.0, e.to(torch.float64) - 127.0)


def packed_block_exponents(w_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """``uint8 [N, K / 128]``: each weight row's K-block exponents from the DeepGEMM-packed scale."""
    kb = k // 128
    words = w_scale.to(torch.int64) & 0xFFFFFFFF
    parts = [(words >> (8 * j)) & 0xFF for j in range(4)]
    return torch.stack(parts, dim=-1).reshape(n, -1)[:, :kb].to(torch.uint8)


def _atom_offsets(rows: int, k: int, blocks: int, k_major: bool, device) -> torch.Tensor:
    r = torch.arange(rows, device=device).unsqueeze(1)
    c = torch.arange(k // 32, device=device).unsqueeze(0)
    if k_major:
        atom = (c // 4) * blocks + r // 128
    else:
        atom = (r // 128) * (k // 128) + c // 4
    return atom * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4


def r128c4_exponents(sf: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """``uint8 [rows, K / 32]`` from a K32 R128c4 scale (the CuTe MXFP8 GEMM's layout)."""
    blocks = -(-rows // 128)
    flat = sf.reshape(-1).view(torch.uint8)
    return flat[_atom_offsets(rows, k, blocks, False, sf.device)]


def mid_exponents(xs: torch.Tensor, rows: int, k: int, sf_blocks: int) -> torch.Tensor:
    """``uint8 [rows, K / 32]`` from the mid-M GEMM's ``[K / 128][sf_blocks][512 B]`` layout."""
    flat = xs.reshape(-1).view(torch.uint8)
    return flat[_atom_offsets(rows, k, sf_blocks, True, xs.device)]


def dequant(codes: torch.Tensor, exponents: torch.Tensor, group: int) -> torch.Tensor:
    """FP64 ``codes * 2^(e - 127)`` with one exponent per ``group`` consecutive columns."""
    scale = exp2_ue8m0(exponents).repeat_interleave(group, dim=1)[:, : codes.shape[1]]
    return codes.view(torch.float8_e4m3fn).to(torch.float64) * scale


def fp8_block_weight(w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """FP64 ``[N, K]`` weight from its E4M3 codes and either 128 x 128 block-scale layout the
    projections hold (DeepGEMM-packed ``int32`` or K32 R128c4 ``uint8``)."""
    n, k = w.shape
    if w_scale.dtype == torch.int32:
        return dequant(w, packed_block_exponents(w_scale, n, k), 128)
    return dequant(w, r128c4_exponents(w_scale, n, k), 32)


def mxfp8_quantize(x: torch.Tensor, group: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes E4M3 [M, K], exponents uint8 [M, K / group])`` of BF16 ``x`` as the kernels
    quantize their activation: per group ``e`` = the biased exponent of ``2^ceil(log2(amax /
    448))`` (``amax`` floored at BF16 ``0x38D2``), codes ``satfinite_rn_e4m3(x * 2^(127 - e))``
    (an exact BF16 multiply)."""
    m, k = x.shape
    g = x.reshape(m, k // group, group)
    bits = g.contiguous().view(torch.int16).to(torch.int32) & 0x7FFF
    amax = bits.amax(dim=-1).clamp_min(AMAX_FLOOR_BF16)
    e = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).to(torch.int32)
    inv = torch.pow(2.0, (127 - e).float()).to(torch.bfloat16)
    scaled = (g * inv.unsqueeze(-1)).float()
    codes = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(m, k)
    return codes, e.to(torch.uint8)


def matmul_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BF16 ``a @ b.T`` of FP64 operands, rounded once."""
    return (a.to(torch.float64) @ b.to(torch.float64).T).to(torch.bfloat16)


def lowm_gemm(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, group: int = 32):
    """BF16 ``x @ W^T`` as the FP8-weight GEMMs compute it on BF16 rows: ``x`` quantized to E4M3
    with one exponent per ``group`` columns, times the dequantized weight."""
    codes, e = mxfp8_quantize(x, group)
    return matmul_bf16(dequant(codes, e, group), fp8_block_weight(w, w_scale))


def all_reduce_bf16(parts) -> torch.Tensor:
    """The MNNVL all-reduce of BF16 per-rank tensors given in rank order: FP32 sum in rank order,
    rounded to BF16 (the one-shot and two-shot kernels' order up to eight ranks)."""
    total = parts[0].float()
    for part in parts[1:]:
        total = total + part.float()
    return total.to(torch.bfloat16)
