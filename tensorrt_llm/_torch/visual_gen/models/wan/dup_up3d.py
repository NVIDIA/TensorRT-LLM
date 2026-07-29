# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused channel-repeat and pixel-shuffle mapping for Wan ``DupUp3D``."""

from __future__ import annotations

import torch

_KERNEL_CACHE: dict = {}


def _build_kernel():
    if "fn" in _KERNEL_CACHE:
        return _KERNEL_CACHE["fn"]

    import triton
    import triton.language as tl

    @triton.jit
    def _dup_up3d_kernel(
        x_ptr,
        output_ptr,
        output_elements,
        output_channels: tl.constexpr,
        output_frames: tl.constexpr,
        output_height: tl.constexpr,
        output_width: tl.constexpr,
        repeats: tl.constexpr,
        factor_t: tl.constexpr,
        factor_s: tl.constexpr,
        temporal_crop: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        stride_xt: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_xw: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < output_elements

        output_channel = offsets % output_channels
        logical = offsets // output_channels
        output_w = logical % output_width
        logical = logical // output_width
        output_h = logical % output_height
        logical = logical // output_height
        output_t = logical % output_frames
        batch = logical // output_frames

        full_output_t = output_t + temporal_crop
        input_t = full_output_t // factor_t
        factor_t_index = full_output_t % factor_t
        input_h = output_h // factor_s
        factor_h_index = output_h % factor_s
        input_w = output_w // factor_s
        factor_w_index = output_w % factor_s

        repeat_channel = output_channel * factor_t * factor_s * factor_s
        repeat_channel += (factor_t_index * factor_s + factor_h_index) * factor_s
        repeat_channel += factor_w_index
        input_channel = repeat_channel // repeats

        input_offset = (
            batch * stride_xn
            + input_channel * stride_xc
            + input_t * stride_xt
            + input_h * stride_xh
            + input_w * stride_xw
        )
        value = tl.load(x_ptr + input_offset, mask=mask)
        tl.store(output_ptr + offsets, value, mask=mask)

    _KERNEL_CACHE["fn"] = (triton, _dup_up3d_kernel)
    return _KERNEL_CACHE["fn"]


def dup_up3d(
    x: torch.Tensor,
    output_channels: int,
    repeats: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> torch.Tensor:
    """Write repeated-and-shuffled ``x`` directly in final channels-last layout."""
    if x.dim() != 5:
        raise ValueError(f"DupUp3D expects a five-dimensional tensor, got shape {x.shape}")

    batch, _, input_frames, input_height, input_width = x.shape
    temporal_crop = factor_t - 1 if first_chunk else 0
    output_frames = input_frames * factor_t - temporal_crop
    output_height = input_height * factor_s
    output_width = input_width * factor_s
    output = torch.empty(
        (batch, output_channels, output_frames, output_height, output_width),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )

    triton, kernel = _build_kernel()
    output_elements = output.numel()
    block_size = 256
    kernel[(triton.cdiv(output_elements, block_size),)](
        x,
        output,
        output_elements,
        output_channels=output_channels,
        output_frames=output_frames,
        output_height=output_height,
        output_width=output_width,
        repeats=repeats,
        factor_t=factor_t,
        factor_s=factor_s,
        temporal_crop=temporal_crop,
        stride_xn=x.stride(0),
        stride_xc=x.stride(1),
        stride_xt=x.stride(2),
        stride_xh=x.stride(3),
        stride_xw=x.stride(4),
        block_size=block_size,
        num_warps=8,
    )
    return output
