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
"""Fused NVFP4 activation quantization for the FP4 Wan VAE.

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
    def _rcp_approx(v):
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [v],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _pack_e2m1(even, odd):
        # Match the CUDA quantize kernel's hardware conversion. The first PTX
        # source becomes the high nibble, hence the reversed operand order.
        return tl.inline_asm_elementwise(
            "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
            "=h,f,f",
            [even, odd],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)

    # Row-tiled: BM rows/program, 3D tiles [BM, SF_K, 8] (even/odd channel pairs).
    # Autotuned per (C, M): optimal BM/num_warps shift with channel count and row count.
    _CONFIGS = [triton.Config({"BM": bm}, num_warps=nw) for bm in (2, 4, 8, 16) for nw in (2, 4, 8)]

    @triton.autotune(configs=_CONFIGS, key=["C", "M", "APPLY_SILU"])
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
        SF_STORAGE_K: tl.constexpr,
        BLOCK_SF_K: tl.constexpr,
        APPLY_SILU: tl.constexpr,
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
        xe = tl.load(x_ptr + base + c_even, mask=rm & (c_even < C), other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + base + c_odd, mask=rm & (c_odd < C), other=0.0).to(tl.float32)
        if APPLY_SILU:
            xe = xe * (1.0 / (1.0 + tl.exp(-xe)))
            xo = xo * (1.0 / (1.0 + tl.exp(-xo)))
            # Match torch SiLU on a BF16 VAE activation at its BF16 rounding point.
            xe = xe.to(tl.bfloat16).to(tl.float32)
            xo = xo.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xe), axis=2), tl.max(tl.abs(xo), axis=2))
        # Match cvt_warp_fp16_to_fp4 in quantization.cuh, including its
        # approximate reciprocal instructions and hardware E2M1 conversion.
        rcp6 = _rcp_approx(tl.full((BM, BLOCK_SF_K), 6.0, tl.float32))
        sf_e4 = (gs * (amax * rcp6)).to(tl.float8e4nv)
        rcpgs = _rcp_approx(tl.full((BM, BLOCK_SF_K), 0.0, tl.float32) + gs)
        inv = _rcp_approx(sf_e4.to(tl.float32) * rcpgs)
        inv = tl.where(amax != 0.0, inv, 0.0)[:, :, None]
        packed = _pack_e2m1(xe * inv, xo * inv)
        qidx = r3 * stride_qm + (k3 * 8 + j3)
        tl.store(xq_ptr + qidx, packed, mask=rm)
        sidx = rows[:, None] * stride_sm + kk[None, :]
        tl.store(
            sf_ptr + sidx,
            tl.where(kmask[None, :], sf_e4.to(tl.uint8, bitcast=True), 0),
            mask=rmask[:, None] & (kk < SF_STORAGE_K)[None, :],
        )

    # Fold WanRMSNorm (channel-wise L2 normalize * sqrt(C) * gamma) into the
    # same pass. The L2 reduction is per-row over all C; the FP4 block scale is
    # per-16. Gamma and input padding are zero, so both reductions ignore it.
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
        SF_STORAGE_K: tl.constexpr,
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
        xe = tl.load(x_ptr + base + c_even, mask=rm & (c_even < C), other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + base + c_odd, mask=rm & (c_odd < C), other=0.0).to(tl.float32)
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
        rcp6 = _rcp_approx(tl.full((BM, BLOCK_SF_K), 6.0, tl.float32))
        sf_e4 = (gs * (amax * rcp6)).to(tl.float8e4nv)
        rcpgs = _rcp_approx(tl.full((BM, BLOCK_SF_K), 0.0, tl.float32) + gs)
        inv = _rcp_approx(sf_e4.to(tl.float32) * rcpgs)
        inv = tl.where(amax != 0.0, inv, 0.0)[:, :, None]
        packed = _pack_e2m1(xe * inv, xo * inv)
        qidx = r3 * stride_qm + (k3 * 8 + j3)
        tl.store(xq_ptr + qidx, packed, mask=rm)
        sidx = rows[:, None] * stride_sm + kk[None, :]
        tl.store(
            sf_ptr + sidx,
            tl.where(kmask[None, :], sf_e4.to(tl.uint8, bitcast=True), 0),
            mask=rmask[:, None] & (kk < SF_STORAGE_K)[None, :],
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


def _validate_padded_channels(channels: int, padded_channels: int | None) -> int:
    output_channels = channels if padded_channels is None else padded_channels
    if output_channels < channels or output_channels % 16 != 0:
        raise ValueError(
            "NVFP4 padded channels must be a 16-aligned value no smaller than the "
            f"input channels, got input={channels}, padded={output_channels}"
        )
    return output_channels


def _scale_storage_blocks(output_channels: int) -> int:
    """Pad the SFA row stride to the four-byte TMA alignment."""
    scale_blocks = output_channels // 16
    return ((scale_blocks + 3) // 4) * 4


def silu_nvfp4_quant(
    x2d: torch.Tensor,
    gs: torch.Tensor,
    padded_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``SiLU(x)`` then NVFP4 quantize.

    Args:
        x2d: BF16 ``[M, C]`` *pre-SiLU* activation, C a multiple of 16.
        gs: FP32 scalar divisor-form global scale ``(448*6)/amax``.
        padded_channels: Optional output channel count. Channels in ``[C, padded_channels)``
            are quantized directly as zero without materializing a BF16 padded tensor.
    Returns:
        (xq uint8 ``[M, Cp//2]``, sf uint8 ``[M, round_up(Cp//16, 4)]``), where
        ``Cp`` is ``padded_channels`` or ``C``. The scale tail is storage-only.
    """
    _validate_inputs(x2d, gs)
    triton, kernel, _ = _build_kernel()
    M, C = x2d.shape
    output_channels = _validate_padded_channels(C, padded_channels)
    SF_K = output_channels // 16
    sf_storage_k = _scale_storage_blocks(output_channels)
    block_sf_k = triton.next_power_of_2(sf_storage_k)
    x2d = x2d.contiguous()
    xq = torch.empty((M, output_channels // 2), dtype=torch.uint8, device=x2d.device)
    sf = torch.empty((M, sf_storage_k), dtype=torch.uint8, device=x2d.device)

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
        SF_STORAGE_K=sf_storage_k,
        BLOCK_SF_K=block_sf_k,
        APPLY_SILU=True,
    )
    return xq, sf


def nvfp4_quant(
    x2d: torch.Tensor,
    gs: torch.Tensor,
    padded_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 activation and directly emit zero-padded NVFP4 blocks.

    This is the dynamic-scale counterpart of :func:`silu_nvfp4_quant`: the
    caller derives ``gs`` from the real input channels, while this kernel emits
    the kernel-required padded channel extent without materializing a padded
    BF16 tensor.
    """
    _validate_inputs(x2d, gs)
    triton, kernel, _ = _build_kernel()
    M, C = x2d.shape
    output_channels = _validate_padded_channels(C, padded_channels)
    SF_K = output_channels // 16
    sf_storage_k = _scale_storage_blocks(output_channels)
    block_sf_k = triton.next_power_of_2(sf_storage_k)
    x2d = x2d.contiguous()
    xq = torch.empty((M, output_channels // 2), dtype=torch.uint8, device=x2d.device)
    sf = torch.empty((M, sf_storage_k), dtype=torch.uint8, device=x2d.device)

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
        SF_STORAGE_K=sf_storage_k,
        BLOCK_SF_K=block_sf_k,
        APPLY_SILU=False,
    )
    return xq, sf


def rmsnorm_silu_nvfp4_quant(
    x2d: torch.Tensor,
    gs: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    padded_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused WanRMSNorm(channel-wise) -> SiLU -> NVFP4 quantize.

    Args:
        x2d: BF16 ``[M, C]`` *pre-norm* activation before channel padding.
        gs: FP32 scalar divisor-form global scale ``(448*6)/amax`` (calibrated on the
            SiLU(norm) output).
        gamma: FP32/BF16 RMSNorm weight, padded to ``padded_channels`` with zero when needed.
        scale: ``sqrt(real_C)`` (WanRMSNorm ``self.scale``; uses the *unpadded* channel count).
        padded_channels: Optional output channel count. Channels in ``[C, padded_channels)``
            are quantized directly as zero without materializing a BF16 padded tensor.
    Returns:
        (xq uint8 ``[M, Cp//2]``, sf uint8 ``[M, round_up(Cp//16, 4)]``), where
        ``Cp`` is ``padded_channels`` or ``C``. The scale tail is storage-only.
    """
    _validate_inputs(x2d, gs)
    output_channels = _validate_padded_channels(x2d.shape[1], padded_channels)
    if gamma.ndim != 1 or gamma.numel() != output_channels:
        raise ValueError(
            f"RMSNorm gamma must have shape ({output_channels},), got {tuple(gamma.shape)}"
        )
    if gamma.device != x2d.device or gamma.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("RMSNorm gamma must be bfloat16/float32 on the input device")
    if scale <= 0:
        raise ValueError(f"RMSNorm scale must be positive, got {scale}")
    triton, _, kernel = _build_kernel()
    M, C = x2d.shape
    SF_K = output_channels // 16
    sf_storage_k = _scale_storage_blocks(output_channels)
    block_sf_k = triton.next_power_of_2(sf_storage_k)
    x2d = x2d.contiguous()
    gamma = gamma.contiguous()
    xq = torch.empty((M, output_channels // 2), dtype=torch.uint8, device=x2d.device)
    sf = torch.empty((M, sf_storage_k), dtype=torch.uint8, device=x2d.device)

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
        SF_STORAGE_K=sf_storage_k,
        BLOCK_SF_K=block_sf_k,
    )
    return xq, sf
