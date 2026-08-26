# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused SiLU + NVFP4 activation quantization for the FP4 Wan VAE.

The NVFP4 conv kernel consumes a pre-quantized activation (packed FP4 values +
un-swizzled FP8-E4M3 block scale factors). Un-fused, the residual block writes a
BF16 ``SiLU(RMSNorm(x))`` tensor and ``_fp4_conv_run`` then reads it back through a
separate ``torch.ops.trtllm.fp4_quantize`` pass. This module fuses the SiLU into
the quantization so the preceding activation is produced directly in FP4 (mirrors
the fused LayerNorm+quant idea of the DiT transformer path), removing one full
read+write of the large activation tensor per quantized conv.

The kernel emits the same packed layout and rounding semantics as
``fp4_quantize(silu(x), gs, 16, False, isSfSwizzledLayout=False)``: ``xq`` uint8
``[M, C//2]`` (low nibble = even channel) and ``sf`` E4M3 ``[M, C//16]``
un-swizzled. Triton import is deferred so the BF16 VAE path carries no Triton
dependency.
"""

from __future__ import annotations

from typing import Any

import torch


def _lazy_triton() -> tuple[Any, Any]:
    import triton
    import triton.language as tl

    return triton, tl


_KERNEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def _build_kernel() -> tuple[Any, Any, Any]:
    if "fn" in _KERNEL_CACHE:
        return _KERNEL_CACHE["fn"]
    triton, tl = _lazy_triton()

    @triton.jit
    def _e2m1_code(v):
        # Round-to-nearest-even among {0,.5,1,1.5,2,3,4,6}; comparison
        # strictness alternates so exact midpoints select the even code.
        # Inspect the fp32 sign bit so negative zero matches the hardware cvt.
        s = v.to(tl.uint32, bitcast=True) >> 31
        a = tl.abs(v)
        m = tl.where(
            a <= 0.25,
            0,
            tl.where(
                a < 0.75,
                1,
                tl.where(
                    a <= 1.25,
                    2,
                    tl.where(
                        a < 1.75,
                        3,
                        tl.where(a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7))),
                    ),
                ),
            ),
        )
        return (m | (s << 3)).to(tl.uint8)

    # Row-tiled: BM rows/program, 3D tiles [BM, SF_K, 8] (even/odd channel pairs).
    # Autotuned per (C, M): optimal BM/num_warps shift with channel count and row count.
    _CONFIGS = [triton.Config({"BM": bm}, num_warps=nw) for bm in (2, 4, 8, 16) for nw in (2, 4, 8)]

    @triton.autotune(configs=_CONFIGS, key=["C", "M"])
    @triton.jit
    def _silu_nvfp4_quant_kernel(
        x_ptr,
        gs_ptr,
        xq_ptr,
        sf_ptr,
        M,
        C,
        stride_xm,
        stride_qm,
        stride_sm,
        BM: tl.constexpr,
        SF_K: tl.constexpr,
        BLOCK_SF_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < M
        kk = tl.arange(0, BLOCK_SF_K)
        kmask = kk < SF_K
        jc = tl.arange(0, 8)
        gs = tl.load(gs_ptr).to(tl.float32)
        r3 = rows[:, None, None]
        k3 = kk[None, :, None]
        j3 = jc[None, None, :]
        c_even = k3 * 16 + 2 * j3
        c_odd = c_even + 1
        base = r3 * stride_xm
        rm = rmask[:, None, None] & kmask[None, :, None]
        xe = tl.load(x_ptr + base + c_even, mask=rm, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + base + c_odd, mask=rm, other=0.0).to(tl.float32)
        xe = xe * (1.0 / (1.0 + tl.exp(-xe)))
        xo = xo * (1.0 / (1.0 + tl.exp(-xo)))
        # Match torch SiLU on a BF16 VAE activation. This is a register-only
        # rounding point; it does not reintroduce the eliminated BF16 tensor.
        xe = xe.to(tl.bfloat16).to(tl.float32)
        xo = xo.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xe), axis=2), tl.max(tl.abs(xo), axis=2))
        # Match the saturating E4M3 conversion used by fp4_quantize. Static
        # calibration can be exceeded by a later activation block.
        sf = tl.minimum(amax * gs / 6.0, 448.0)
        sf_e4 = sf.to(tl.float8e4nv)
        sf_dec = sf_e4.to(tl.float32)
        inv = tl.where(sf_dec > 0, gs / sf_dec, 0.0)[:, :, None]
        lo = _e2m1_code(xe * inv)
        hi = _e2m1_code(xo * inv)
        packed = (lo | (hi << 4)).to(tl.uint8)
        qidx = r3 * stride_qm + (k3 * 8 + j3)
        tl.store(xq_ptr + qidx, packed, mask=rm)
        sidx = rows[:, None] * stride_sm + kk[None, :]
        tl.store(
            sf_ptr + sidx,
            sf_e4.to(tl.uint8, bitcast=True),
            mask=rmask[:, None] & kmask[None, :],
        )

    # Tier B additionally folds WanRMSNorm (channel-wise L2 normalize * sqrt(C) * gamma) into
    # the same pass. The L2 reduction is per-row over all C; the FP4 block scale is per-16.
    # gamma is padded to Cp with 0 and pad channels of x are 0, so both reductions ignore pad.
    @triton.autotune(configs=_CONFIGS, key=["C", "M"])
    @triton.jit
    def _rmsnorm_silu_nvfp4_quant_kernel(
        x_ptr,
        gs_ptr,
        g_ptr,
        scale,
        xq_ptr,
        sf_ptr,
        M,
        C,
        stride_xm,
        stride_qm,
        stride_sm,
        BM: tl.constexpr,
        SF_K: tl.constexpr,
        BLOCK_SF_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < M
        kk = tl.arange(0, BLOCK_SF_K)
        kmask = kk < SF_K
        jc = tl.arange(0, 8)
        gs = tl.load(gs_ptr).to(tl.float32)
        r3 = rows[:, None, None]
        k3 = kk[None, :, None]
        j3 = jc[None, None, :]
        c_even = k3 * 16 + 2 * j3
        c_odd = c_even + 1
        base = r3 * stride_xm
        rm = rmask[:, None, None] & kmask[None, :, None]
        xe = tl.load(x_ptr + base + c_even, mask=rm, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + base + c_odd, mask=rm, other=0.0).to(tl.float32)
        # per-row L2 norm over all C (pad channels are 0)
        ss = tl.sum(tl.sum(xe * xe, axis=2), axis=1) + tl.sum(tl.sum(xo * xo, axis=2), axis=1)
        inv_norm = (1.0 / tl.maximum(tl.sqrt(ss), 1e-12))[:, None, None]  # [BM,1,1]
        gc_even = kk[:, None] * 16 + 2 * jc[None, :]  # [SF_K,8]
        gc_odd = gc_even + 1
        gamma_mask = kmask[:, None]
        ge = tl.load(g_ptr + gc_even, mask=gamma_mask, other=0.0).to(tl.float32)[None, :, :]
        go = tl.load(g_ptr + gc_odd, mask=gamma_mask, other=0.0).to(tl.float32)[None, :, :]
        # WanRMSNorm rounds F.normalize back to BF16 before applying its
        # sqrt(C) scale and BF16 gamma. Preserve both BF16 rounding points in
        # registers so fusion remains numerically equivalent without a spill.
        xe = (xe * inv_norm).to(tl.bfloat16).to(tl.float32)
        xo = (xo * inv_norm).to(tl.bfloat16).to(tl.float32)
        xe = (xe * scale).to(tl.bfloat16).to(tl.float32)
        xo = (xo * scale).to(tl.bfloat16).to(tl.float32)
        xe = (xe * ge).to(tl.bfloat16).to(tl.float32)
        xo = (xo * go).to(tl.bfloat16).to(tl.float32)
        xe = xe * (1.0 / (1.0 + tl.exp(-xe)))
        xo = xo * (1.0 / (1.0 + tl.exp(-xo)))
        xe = xe.to(tl.bfloat16).to(tl.float32)
        xo = xo.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xe), axis=2), tl.max(tl.abs(xo), axis=2))
        sf = tl.minimum(amax * gs / 6.0, 448.0)
        sf_e4 = sf.to(tl.float8e4nv)
        sf_dec = sf_e4.to(tl.float32)
        inv = tl.where(sf_dec > 0, gs / sf_dec, 0.0)[:, :, None]
        lo = _e2m1_code(xe * inv)
        hi = _e2m1_code(xo * inv)
        packed = (lo | (hi << 4)).to(tl.uint8)
        qidx = r3 * stride_qm + (k3 * 8 + j3)
        tl.store(xq_ptr + qidx, packed, mask=rm)
        sidx = rows[:, None] * stride_sm + kk[None, :]
        tl.store(
            sf_ptr + sidx,
            sf_e4.to(tl.uint8, bitcast=True),
            mask=rmask[:, None] & kmask[None, :],
        )

    _KERNEL_CACHE["fn"] = (
        triton,
        _silu_nvfp4_quant_kernel,
        _rmsnorm_silu_nvfp4_quant_kernel,
    )
    return _KERNEL_CACHE["fn"]


def _validate_inputs(x2d: torch.Tensor, gs: torch.Tensor) -> None:
    if x2d.ndim != 2 or x2d.shape[1] % 16 != 0:
        raise ValueError(f"Expected a 2D input with 16-aligned channels, got {tuple(x2d.shape)}")
    if not x2d.is_cuda or x2d.dtype is not torch.bfloat16:
        raise ValueError("Fused NVFP4 quantization requires a CUDA bfloat16 input")
    if gs.numel() != 1 or gs.dtype is not torch.float32 or gs.device != x2d.device:
        raise ValueError("NVFP4 global scale must be one float32 value on the input device")


def silu_nvfp4_quant(
    x2d: torch.Tensor,
    gs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``SiLU(x)`` then NVFP4 quantize.

    Args:
        x2d: BF16 ``[M, C]`` *pre-SiLU* activation, C a multiple of 16.
        gs: FP32 scalar divisor-form global scale ``(448*6)/amax``.
    Returns:
        (xq uint8 ``[M, C//2]``, sf uint8 ``[M, C//16]``) matching
        ``fp4_quantize(silu(x), gs, 16, False, isSfSwizzledLayout=False)``.
    """
    _validate_inputs(x2d, gs)
    triton, kernel, _ = _build_kernel()
    M, C = x2d.shape
    SF_K = C // 16
    block_sf_k = triton.next_power_of_2(SF_K)
    x2d = x2d.contiguous()
    xq = torch.empty((M, C // 2), dtype=torch.uint8, device=x2d.device)
    sf = torch.empty((M, SF_K), dtype=torch.uint8, device=x2d.device)

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(M, meta["BM"]),)

    kernel[grid](
        x2d,
        gs,
        xq,
        sf,
        M,
        C,
        x2d.stride(0),
        xq.stride(0),
        sf.stride(0),
        SF_K=SF_K,
        BLOCK_SF_K=block_sf_k,
    )
    return xq, sf


def rmsnorm_silu_nvfp4_quant(
    x2d: torch.Tensor,
    gs: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused WanRMSNorm(channel-wise) -> SiLU -> NVFP4 quantize (tier B).

    Args:
        x2d: BF16 ``[M, C]`` *pre-norm* activation (C == padded Cp; pad channels 0).
        gs: FP32 scalar divisor-form global scale ``(448*6)/amax`` (calibrated on the
            SiLU(norm) output).
        gamma: FP32/BF16 ``[C]`` RMSNorm weight, padded to Cp with 0 for pad channels.
        scale: ``sqrt(real_C)`` (WanRMSNorm ``self.scale``; uses the *unpadded* channel count).
    Returns:
        (xq uint8 ``[M, C//2]``, sf uint8 ``[M, C//16]``) matching
        ``fp4_quantize(silu(rmsnorm(x)), gs, 16, False, isSfSwizzledLayout=False)``.
    """
    _validate_inputs(x2d, gs)
    if gamma.ndim != 1 or gamma.numel() != x2d.shape[1]:
        raise ValueError(
            f"RMSNorm gamma must have shape ({x2d.shape[1]},), got {tuple(gamma.shape)}"
        )
    if gamma.device != x2d.device or gamma.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("RMSNorm gamma must be bfloat16/float32 on the input device")
    if scale <= 0:
        raise ValueError(f"RMSNorm scale must be positive, got {scale}")
    triton, _, kernel = _build_kernel()
    M, C = x2d.shape
    SF_K = C // 16
    block_sf_k = triton.next_power_of_2(SF_K)
    x2d = x2d.contiguous()
    gamma = gamma.contiguous()
    xq = torch.empty((M, C // 2), dtype=torch.uint8, device=x2d.device)
    sf = torch.empty((M, SF_K), dtype=torch.uint8, device=x2d.device)

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(M, meta["BM"]),)

    kernel[grid](
        x2d,
        gs,
        gamma,
        float(scale),
        xq,
        sf,
        M,
        C,
        x2d.stride(0),
        xq.stride(0),
        sf.stride(0),
        SF_K=SF_K,
        BLOCK_SF_K=block_sf_k,
    )
    return xq, sf
