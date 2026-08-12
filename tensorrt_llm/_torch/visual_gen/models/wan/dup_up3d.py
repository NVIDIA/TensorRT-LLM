# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused channel-repeat and pixel-shuffle mapping for Wan ``DupUp3D``."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

_BLOCK_SIZE = 256
_NUM_WARPS = 8
_MAX_TRITON_INDEXED_ELEMENTS = 1 << 31


def _supports_triton_indexing(output_elements: int) -> bool:
    """Return whether the output fits the kernel's signed 32-bit offsets."""
    return output_elements <= _MAX_TRITON_INDEXED_ELEMENTS


def _validate_dup_up3d_contract(
    x: torch.Tensor,
    output_channels: int,
    repeats: int,
    factor_t: int,
    factor_s: int,
) -> None:
    """Assert invariants supplied by the internal ``DupUp3D`` caller."""
    assert x.dim() == 5, f"DupUp3D expects NCTHW input, got shape {x.shape}"
    assert min(output_channels, repeats, factor_t, factor_s) >= 1
    input_channels = x.shape[1]
    factor = factor_t * factor_s * factor_s
    assert input_channels * repeats == output_channels * factor


def _dup_up3d_output_shape(
    x: torch.Tensor,
    output_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> tuple[int, int, int, int, int]:
    batch, _, input_frames, input_height, input_width = x.shape
    temporal_crop = factor_t - 1 if first_chunk else 0
    return (
        batch,
        output_channels,
        input_frames * factor_t - temporal_crop,
        input_height * factor_s,
        input_width * factor_s,
    )


def can_implement_dup_up3d(
    x: torch.Tensor,
    *,
    output_channels: int,
    repeats: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> bool:
    """Return whether the fused kernel supports this internal invocation.

    Args:
        x: Logical NCTHW input tensor.
        output_channels: Number of logical output channels.
        repeats: Number of channel copies before pixel shuffle.
        factor_t: Temporal pixel-shuffle factor.
        factor_s: Height and width pixel-shuffle factor.
        first_chunk: Whether the cache-initializing temporal crop is required.

    Returns:
        ``True`` for a non-empty CUDA invocation whose input span and output
        fit the kernel's signed 32-bit indexing range; otherwise ``False``.
    """
    _validate_dup_up3d_contract(x, output_channels, repeats, factor_t, factor_s)
    if not x.is_cuda or 0 in x.shape:
        return False

    output_shape = _dup_up3d_output_shape(
        x,
        output_channels,
        factor_t,
        factor_s,
        first_chunk,
    )
    output_elements = math.prod(output_shape)
    input_span_elements = 1 + sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
    supported = _supports_triton_indexing(output_elements) and _supports_triton_indexing(
        input_span_elements
    )
    if not supported:
        logger.warning_once(
            f"Fused DupUp3D input span ({input_span_elements} elements) or output size "
            f"({output_elements} elements) exceeds the Triton "
            "signed 32-bit indexing limit; falling back to the eager implementation.",
            key="wan_dup_up3d_int32_index_fallback",
        )
    return supported


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


def dup_up3d(
    x: torch.Tensor,
    *,
    output_channels: int,
    repeats: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool,
) -> torch.Tensor:
    """Write the shuffled logical NCTHW output in physical NTHWC order.

    Args:
        x: Logical NCTHW CUDA tensor with shape ``[N, C, T, H, W]``. It may
            have arbitrary non-overlapping strides, and its dtype is preserved.
        output_channels: Number of channels in the shuffled output.
        repeats: Number of copies of each input channel before shuffling.
        factor_t: Temporal pixel-shuffle factor.
        factor_s: Spatial pixel-shuffle factor applied to height and width.
        first_chunk: Whether to crop the leading ``factor_t - 1`` frames
            produced by the cache-initializing temporal chunk.

    Returns:
        A logical NCTHW tensor stored in PyTorch ``channels_last_3d`` physical
        order, which corresponds to NTHWC.
    """
    _validate_dup_up3d_contract(x, output_channels, repeats, factor_t, factor_s)

    temporal_crop = factor_t - 1 if first_chunk else 0
    output_shape = _dup_up3d_output_shape(
        x,
        output_channels,
        factor_t,
        factor_s,
        first_chunk,
    )
    _, _, output_frames, output_height, output_width = output_shape
    output_elements = math.prod(output_shape)

    output = torch.empty(
        output_shape,
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )

    _dup_up3d_kernel[(triton.cdiv(output_elements, _BLOCK_SIZE),)](
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
        block_size=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return output
