# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused RMSNorm + NVFP4 quantize for the Gemma4 pre-feedforward norm.

Replaces the per-layer pair in front of the NVFP4 gate_up GEMM:

    trtllm::flashinfer_rmsnorm (bf16 out, full [M, H] HBM round-trip)
    -> trtllm::fp4_quantize (reads it back, emits FP4 + swizzled scales)

with one Triton kernel that reads the residual-stream input once and emits
the packed E2M1 payload plus the 128x4-swizzled E4M3 block scales directly -
eliminating the intermediate bf16 activation write+read.

Numerics replicate the unfused chain op-for-op:
- the norm mirrors flashinfer's RMSNormKernel (fp32 sum of squares,
  `rsqrt.approx.ftz.f32(ssq/d + eps)`, `(x * rcp) * w`, bf16 round - the
  write the unfused path performs) before any quant math;
- the scale/quant math mirrors cvt_warp_fp16_to_fp4 in
  cpp/tensorrt_llm/kernels/quantization.cuh: rcp.approx.ftz.f32 reciprocals,
  cvt.rn.satfinite e4m3 scale rounding, and the cvt.rn.satfinite.e2m1x2.f32
  payload conversion, with scales stored at the 128x4 swizzled offsets of
  get_sf_out_offset_128x4 (rows padded to 128; the padding region is left
  uninitialized, exactly like the unfused op).

The only residual difference vs the unfused chain is the fp32 reduction
order inside the norm's sum of squares (same class as the other fused
Gemma4 kernels).

The unfused path is kept as the reference / fallback; set
TRTLLM_GEMMA4_DISABLE_FUSED_NORM_QUANT=1 to force it (rollback switch).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _gemma4_norm_fp4_kernel(
    x_ptr,  # [M, H] bf16 residual-stream input, row stride SX
    w_ptr,  # [H] bf16 RMSNorm weight (plain multiplier convention)
    out_ptr,  # [M, H//2] uint8, two e2m1 per byte (element 0 = low nibble)
    sf_ptr,  # [pad128(M) * 4 * NKT] uint8, swizzled 128x4 layout
    gs_ptr,  # [1] fp32 global scale (gate_up_proj.input_scale)
    M,
    SX,  # x row stride (elements)
    NKT,  # numKTiles = ceil((H/16) / 4)
    EPS,
    H: tl.constexpr,  # hidden size (elements, multiple of 32)
    BM: tl.constexpr,
    BK: tl.constexpr,  # column tile (multiple of 16)
):
    # Two passes over the row in BK-column tiles (statically unrolled): the
    # norm needs the full-row sum of squares before any block can be
    # quantized, and tiling keeps the register footprint small enough for
    # the quant math. The second pass re-reads the row while it is still
    # L2-hot, so effective HBM traffic stays ~one read + the FP4 writes.
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M

    # Pass 1: fp32 sum of squares (flashinfer RMSNormKernel accumulation).
    ssq = tl.zeros((BM,), tl.float32)
    for k0 in range(0, H, BK):
        cols = k0 + tl.arange(0, BK)
        mask = rmask[:, None] & (cols[None, :] < H)
        x = tl.load(x_ptr + rows[:, None] * SX + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        ssq += tl.sum(x * x, axis=1)
    rcp = tl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        [ssq / H + EPS],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    gs = tl.load(gs_ptr)

    # Pass 2: apply the norm (with the bf16 round the unfused path performs
    # when writing the normed tensor the quantizer then reads), then the
    # NVFP4 block quantize, one BK tile at a time.
    for k0 in range(0, H, BK):
        cols = k0 + tl.arange(0, BK)
        mask = rmask[:, None] & (cols[None, :] < H)
        x = tl.load(x_ptr + rows[:, None] * SX + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)
        v = ((x * rcp[:, None]) * w[None, :]).to(tl.bfloat16).to(tl.float32)

        # Per-16-element block amax (exact; max is order-insensitive).
        # Masked columns hold zeros, so out-of-range blocks get vmax == 0.
        v16 = tl.reshape(v, (BM, BK // 16, 16))
        vmax = tl.max(tl.abs(v16), axis=2)  # [BM, BK/16]

        # Scale math replicating cvt_warp_fp16_to_fp4 (e4m3 branch):
        #   SFValue = gs * (vecMax * rcp.approx(6));  sf8 = e4m3(SFValue)
        #   outputScale = vecMax != 0 ? rcp.approx(f32(sf8)*rcp.approx(gs)) : 0
        rcp6 = _rcp_approx(tl.full((BM, BK // 16), 6.0, tl.float32))
        sf8 = (gs * (vmax * rcp6)).to(tl.float8e4nv)
        rcpgs = _rcp_approx(tl.full((BM, BK // 16), 0.0, tl.float32) + gs)
        oscale = _rcp_approx(sf8.to(tl.float32) * rcpgs)
        oscale = tl.where(vmax != 0.0, oscale, 0.0)

        # Scale store at the swizzled 128x4 offsets (get_sf_out_offset_128x4).
        kvec = (k0 // 16 + tl.arange(0, BK // 16)).to(tl.int64)
        kvmask = rmask[:, None] & (kvec[None, :] * 16 < H)
        m2 = rows[:, None]
        k2 = kvec[None, :]
        sfoff = (
            (m2 // 128) * (NKT * 512)
            + (k2 // 4) * 512
            + (m2 % 32) * 16
            + ((m2 % 128) // 32) * 4
            + (k2 % 4)
        )
        tl.store(sf_ptr + sfoff, sf8.to(tl.uint8, bitcast=True), mask=kvmask)

        # E2M1 conversion + pairwise packing via the same PTX instruction the
        # CUDA quantize kernel uses (first source operand -> high nibble).
        y = tl.reshape(v16 * oscale[:, :, None], (BM, BK))
        lo, hi = tl.split(tl.reshape(y, (BM, BK // 2, 2)))
        byte = tl.inline_asm_elementwise(
            "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
            "=h,f,f",
            [lo, hi],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)

        ocols = k0 // 2 + tl.arange(0, BK // 2)
        omask = rmask[:, None] & (ocols[None, :] * 2 < H)
        tl.store(out_ptr + rows[:, None] * (H // 2) + ocols[None, :], byte, mask=omask)


def gemma4_fused_norm_fp4(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + NVFP4 block-scale quantize.

    Args:
        x: [M, H] bf16 residual-stream input (a row stride larger than the
            width is allowed; the innermost dim must be contiguous).
        norm_weight: [H] bf16 - pre_feedforward_layernorm weight (plain
            multiplier convention, ``use_gemma=False``).
        eps: RMSNorm epsilon.
        global_scale: [1] fp32 tensor - the consumer Linear's static
            ``input_scale`` (448*6/amax convention).

    Returns:
        (fp4, sf): the packed E2M1 payload [M, H//2] (uint8, element 2j in
        the low nibble of byte j) and the E4M3 block scales (uint8, 1D,
        swizzled 128x4 layout padded to 128 rows, padding uninitialized) -
        byte-compatible with ``trtllm::fp4_quantize``'s outputs and
        consumable as ``Fp4QuantizedTensor(fp4, sf)``.
    """
    assert x.dim() == 2 and x.stride(-1) == 1
    assert x.dtype == torch.bfloat16
    assert global_scale.dtype == torch.float32
    m, h = x.shape
    assert h % 32 == 0, "hidden size must be a multiple of 32"
    assert norm_weight.shape == (h,)

    nkt = triton.cdiv(h // 16, 4)
    out = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    # Like the unfused fp4_quantize, the 128-row padding region of the scale
    # buffer is left uninitialized (nothing reads it) - torch.empty avoids a
    # per-call memset kernel.
    sf = torch.empty((((m + 127) // 128) * 128 * nkt * 4,), dtype=torch.uint8, device=x.device)
    if m == 0:
        return out, sf

    # B200-tuned (BK=2048/num_warps=4 runs 35.2 us at [7500, 5376] vs 44.5
    # at BK=1024 and 63.2 unfused; decode shape is config-insensitive at
    # ~10.7 us). Fixed (no runtime autotune) so
    # launches stay deterministic under CUDA-graph capture. One program per
    # BM rows, two statically unrolled BK-tile passes per row (sum of
    # squares, then quantize).
    bm, bk = 1, 2048
    grid = (triton.cdiv(m, bm),)
    _gemma4_norm_fp4_kernel[grid](
        x,
        norm_weight,
        out,
        sf,
        global_scale,
        m,
        x.stride(0),
        nkt,
        eps,
        H=h,
        BM=bm,
        BK=bk,
        num_warps=4,
    )
    return out, sf
