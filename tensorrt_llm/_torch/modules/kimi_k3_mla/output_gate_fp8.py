# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA output gate fused with the o_proj FP8 1x128 input quantization (Rubin layout).

``gate_mul_fp8_r128c4(attn_out, gate)`` returns, byte for byte, what
``torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(attn_out * gate.sigmoid())`` returns -- the
operand ``_Fp8BlockScaleWeightReadLinear.forward_prequantized`` takes on Rubin -- in one kernel
instead of the sigmoid, the multiply and the quantization:

* ``sigmoid`` as PyTorch computes it for BF16: ``bf16(1 / (1 + expf(-float(g))))`` with the
  libdevice ``expf`` and an IEEE division, then ``bf16(float(attn_out) * float(sig))``;
* per row and 128-column block the quantizer's own instruction sequence: ``amax`` floored at
  ``1e-10``, times ``rcp.approx.ftz(448)``, rounded up to a UE8M0 power of two (saturating at
  254), values multiplied by ``2^(127 - e)`` and rounded to E4M3 (nearest even);
* scales as uint8 ``[roundup(M, 128) * K / 32]`` in the CuTe 128x4 layout, each 1x128 scale
  replicated over its four 32-column groups, the padding rows zeroed.

The scale layout is the one the Rubin CuTe DSL MXFP8 GEMM reads.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from tensorrt_llm._utils import get_sm_version

from ...flashinfer_utils import get_env_enable_pdl

_BLOCK_M = 16


@triton.jit
def _gate_mul_fp8_kernel(
    a_ptr,
    g_ptr,
    q_ptr,
    sf_ptr,
    M,
    stride_a,
    stride_g,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = kb * 128 + tl.arange(0, 128)
    in_rows = rows < M
    mask = in_rows[:, None]
    rows64 = rows.to(tl.int64)
    a = tl.load(a_ptr + rows64[:, None] * stride_a + cols[None, :], mask=mask, other=0.0)
    g = tl.load(g_ptr + rows64[:, None] * stride_g + cols[None, :], mask=mask, other=0.0)
    e = libdevice.exp(-g.to(tl.float32))
    sig = tl.math.div_rn(tl.full(e.shape, 1.0, tl.float32), 1.0 + e).to(tl.bfloat16)
    y = (a.to(tl.float32) * sig.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(y), axis=1)
    # fp8_quantize_1x128_cutedsl_ue8m0: max(amax, 1e-10) * rcp.approx.ftz(448).
    raw = tl.inline_asm_elementwise(
        "{\n.reg .f32 a, d;\nmov.b32 a, $1;\nmax.f32 a, a, 0f2EDBE6FF;\n"
        "rcp.approx.ftz.f32 d, 0f43E00000;\nmul.f32 d, a, d;\nmov.b32 $0, d;\n}",
        "=r,r",
        [amax.to(tl.int32, bitcast=True)],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    # UE8M0, rounded toward +inf (raw is a positive normal number), saturating at 254.
    e8 = ((raw >> 23) & 0xFF) + ((raw & 0x7FFFFF) != 0).to(tl.int32)
    e8 = tl.minimum(e8, 254)
    qscale = ((254 - e8) << 23).to(tl.float32, bitcast=True)  # 2^(127 - e8)
    scaled = y * qscale[:, None]
    # cvt.rn.satfinite: +-inf (an inf amax) saturates to +-448 and NaN stays NaN, as in the C++
    # quantizer; the plain Triton conversion would turn inf into NaN.
    scaled = tl.where(scaled != scaled, scaled, tl.minimum(tl.maximum(scaled, -448.0), 448.0))
    codes = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(q_ptr + rows64[:, None] * K + cols[None, :], codes, mask=mask)
    # One 32-bit word per (row, 128-column block): the scale byte for each of its four K32 groups,
    # at byte (row // 128) * (K // 128) * 512 + kb * 512 + (row % 32) * 16 + (row % 128) // 32 * 4.
    word = tl.where(in_rows, e8 * 0x01010101, 0)
    w_off = (rows // 128) * (K // 128) * 128 + kb * 128 + (rows % 32) * 4 + (rows % 128) // 32
    tl.store(sf_ptr + w_off, word)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def gate_mul_fp8_r128c4(
    attn_out: torch.Tensor, gate: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """E4M3 ``[M, K]`` codes and R128c4 UE8M0 scales (uint8) of ``attn_out * gate.sigmoid()``.

    ``attn_out`` and ``gate``: BF16 ``[M, K]`` with unit column stride, ``K % 128 == 0``.
    """
    if attn_out.dim() != 2 or gate.shape != attn_out.shape:
        raise ValueError("attn_out and gate must be [M, K] tensors of the same shape")
    if attn_out.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("attn_out and gate must be BF16")
    m, k = attn_out.shape
    if k % 128 or attn_out.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("K must be a multiple of 128 and the columns contiguous")
    padded_m = (m + 127) // 128 * 128
    codes = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=attn_out.device)
    scales = torch.empty((padded_m * (k // 32),), dtype=torch.uint8, device=attn_out.device)
    if m == 0:
        return codes, scales
    launch_with_pdl = get_env_enable_pdl() and get_sm_version() >= 90
    _gate_mul_fp8_kernel[(padded_m // _BLOCK_M, k // 128)](
        attn_out,
        gate,
        codes,
        scales.view(torch.int32),
        m,
        attn_out.stride(0),
        gate.stride(0),
        K=k,
        BLOCK_M=_BLOCK_M,
        LAUNCH_WITH_PDL=launch_with_pdl,
        num_warps=4,
        launch_pdl=launch_with_pdl,
    )
    return codes, scales


__all__ = ["gate_mul_fp8_r128c4"]
