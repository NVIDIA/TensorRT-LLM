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
"""Fused gelu_tanh+mul+NVFP4-quantize (single Triton kernel).

Replaces the unfused pair between a packed [gate | up] GEMM output and an
NVFP4 GEMM consumer (e.g. a gated-MLP down_proj):

    trtllm::flashinfer_gelu_tanh_and_mul (bf16 out, full HBM round-trip)
    -> trtllm::fp4_quantize (reads it back, emits FP4 + swizzled scales)

with one Triton kernel that reads the packed [gate | up] GEMM output once and
emits the packed E2M1 payload plus the 128x4-swizzled E4M3 block scales
directly - eliminating the intermediate activation write+read.

Numerics replicate the unfused chain byte-for-byte on SM100:
- gelu_tanh uses flashinfer's exact expression order and its `tanh.approx.f32`
  hardware instruction (flashinfer JIT builds with -use_fast_math);
- the product is rounded to bf16 (the write the unfused path performs) before
  quantization;
- the scale/quant math mirrors cvt_warp_fp16_to_fp4 in
  cpp/tensorrt_llm/kernels/quantization.cuh: rcp.approx.ftz.f32 reciprocals,
  cvt.rn.satfinite e4m3 scale rounding, and the cvt.rn.satfinite.e2m1x2.f32
  payload conversion, with scales stored at the 128x4 swizzled offsets of
  get_sf_out_offset_128x4 (rows padded to 128; padding is zero-filled here,
  uninitialized in the unfused op).

Callers own enablement and keep the unfused pair as the fallback for
configurations this kernel does not support (non-NVFP4 consumer, LoRA,
torch.compile, ...); see the gelu_tanh + down_proj quantize fusion in
modeling_gemma4.py.
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
def _gelu_tanh_mul_fp4_kernel(
    x_ptr,  # [M, 2I] bf16 packed [gate | up], row stride SX
    out_ptr,  # [M, I//2] uint8, two e2m1 per byte (element 0 = low nibble)
    sf_ptr,  # [pad128(M) * 4 * NKT] uint8, swizzled 128x4 layout
    gs_ptr,  # [1] fp32 global scale (down_proj.input_scale)
    M,
    SX,  # x row stride (elements)
    NKT,  # numKTiles = ceil((IDIM/16) / 4)
    IDIM: tl.constexpr,  # intermediate size (elements, multiple of 16)
    BM: tl.constexpr,
    BK: tl.constexpr,  # multiple of 16
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    rmask = rows < M
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = rmask[:, None] & (cols[None, :] < IDIM)

    gate = tl.load(x_ptr + rows[:, None] * SX + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + rows[:, None] * SX + IDIM + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    # gelu_tanh exactly as flashinfer's JIT kernel computes it, then the bf16
    # round the unfused path performs when writing its output tensor.
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [inner], dtype=tl.float32, is_pure=True, pack=1
    )
    act = gate * (0.5 * (1.0 + t))
    v = (act * up).to(tl.bfloat16).to(tl.float32)

    # Per-16-element block amax (exact; max is order-insensitive).
    v16 = tl.reshape(v, (BM, BK // 16, 16))
    vmax = tl.max(tl.abs(v16), axis=2)  # [BM, BK/16]

    # Scale math replicating cvt_warp_fp16_to_fp4 (e4m3 branch):
    #   SFValue = gs * (vecMax * rcp.approx(6));  sf8 = e4m3(SFValue)
    #   outputScale = vecMax != 0 ? rcp.approx(f32(sf8) * rcp.approx(gs)) : 0
    gs = tl.load(gs_ptr)
    rcp6 = _rcp_approx(tl.full((BM, BK // 16), 6.0, tl.float32))
    sf8 = (gs * (vmax * rcp6)).to(tl.float8e4nv)
    rcpgs = _rcp_approx(tl.full((BM, BK // 16), 0.0, tl.float32) + gs)
    oscale = _rcp_approx(sf8.to(tl.float32) * rcpgs)
    oscale = tl.where(vmax != 0.0, oscale, 0.0)

    # Scale store at the swizzled 128x4 offsets (get_sf_out_offset_128x4).
    kvec = (tl.program_id(1) * (BK // 16) + tl.arange(0, BK // 16)).to(tl.int64)
    kvmask = rmask[:, None] & (kvec[None, :] * 16 < IDIM)
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

    ocols = tl.program_id(1) * (BK // 2) + tl.arange(0, BK // 2)
    omask = rmask[:, None] & (ocols[None, :] * 2 < IDIM)
    tl.store(out_ptr + rows[:, None] * (IDIM // 2) + ocols[None, :], byte, mask=omask)


def sf_swizzled_offsets(m: int, nvec: int, device: torch.device) -> torch.Tensor:
    """Flat swizzled offsets of the valid (row, kvec) scale region.

    Mirrors get_sf_out_offset_128x4; used by the parity tests to compare only
    the valid region (the unfused op leaves the 128-row padding
    uninitialized).
    """
    nkt = (nvec + 3) // 4
    mm = torch.arange(m, device=device, dtype=torch.int64)[:, None]
    kk = torch.arange(nvec, device=device, dtype=torch.int64)[None, :]
    off = (
        (mm // 128) * (nkt * 512)
        + (kk // 4) * 512
        + (mm % 32) * 16
        + ((mm % 128) // 32) * 4
        + (kk % 4)
    )
    return off.reshape(-1)


def gelu_tanh_mul_fp4_quant(
    x: torch.Tensor, global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused gelu_tanh(gate) * up + NVFP4 block-scale quantize.

    Args:
        x: [M, 2*I] bf16, packed [gate | up] (a row stride larger than the
            width is allowed; the innermost dim must be contiguous).
        global_scale: [1] fp32 tensor - the consumer Linear's static
            ``input_scale`` (448*6/amax convention).

    Returns:
        (fp4, sf): the packed E2M1 payload [M, I//2] (uint8, element 2j in
        the low nibble of byte j) and the E4M3 block scales (uint8, 1D,
        swizzled 128x4 layout padded to 128 rows, padding zero-filled) -
        byte-compatible with ``trtllm::fp4_quantize``'s outputs and
        consumable as ``Fp4QuantizedTensor(fp4, sf)``.
    """
    assert x.dim() == 2 and x.stride(-1) == 1
    assert x.dtype == torch.bfloat16
    assert global_scale.dtype == torch.float32
    m, two_i = x.shape
    i = two_i // 2
    assert two_i % 32 == 0, "intermediate size must be a multiple of 16"

    nkt = triton.cdiv(i // 16, 4)
    out = torch.empty((m, i // 2), dtype=torch.uint8, device=x.device)
    sf = torch.zeros((((m + 127) // 128) * 128 * nkt * 4,), dtype=torch.uint8, device=x.device)
    if m == 0:
        return out, sf

    # B200-tuned; fixed (no runtime autotune) so launches stay deterministic
    # under CUDA-graph capture. Measured 234 -> 145 us vs the unfused pair
    # at [6455, 2x21504]; 13.6 -> 7.6 us graph-replayed at the 228-token
    # decode shape.
    bm, bk = 8, 512
    grid = (triton.cdiv(m, bm), triton.cdiv(i, bk))
    _gelu_tanh_mul_fp4_kernel[grid](
        x, out, sf, global_scale, m, x.stride(0), nkt, IDIM=i, BM=bm, BK=bk, num_warps=8
    )
    return out, sf
