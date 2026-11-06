# Copyright 2026 The VDN Authors.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-tiled temporal stencil for VDN's separable short convolution."""

import torch
import triton
import triton.language as tl


@triton.jit
def _temporal_conv_kernel(
    X,
    W,
    Y,
    T,
    S,
    C,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ACTIVATE: tl.constexpr,
    L2: tl.constexpr,
):
    frames = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = tl.program_id(2) * D + tl.arange(0, D)
    spatial = tl.program_id(1)
    acc = tl.zeros((BLOCK_T, D), tl.float32)
    for tap in tl.static_range(5):
        source = frames + tap - 2
        value = tl.load(
            X + (source[:, None] * S + spatial) * C + channels[None, :],
            mask=((frames < T) & (source >= 0) & (source < T))[:, None],
            other=0,
        ).to(tl.float32)
        weight = tl.load(W + channels * 5 + tap).to(tl.float32)
        acc += value * weight[None, :]
    if ACTIVATE:
        acc = acc * tl.sigmoid(acc)
        if L2:
            acc *= tl.rsqrt(tl.maximum(tl.sum(acc * acc, axis=1), 1e-12))[:, None]
    tl.store(
        Y + (frames[:, None] * S + spatial) * C + channels[None, :], acc, mask=(frames < T)[:, None]
    )


def temporal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    head_dim: int,
    *,
    activate: bool = False,
    l2norm: bool = False,
) -> torch.Tensor:
    """Five symmetric temporal taps, accumulated in FP32 like upstream Inductor.

    Inputs are contiguous [frames, spatial, channels] and [channels, 5].
    Neighboring frames share stencil loads through the GPU cache.
    """
    if not x.is_cuda or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("VDN temporal convolution requires contiguous CUDA tensors.")
    if head_dim < 1 or head_dim & (head_dim - 1):
        raise ValueError("VDN temporal convolution requires a power-of-two head dimension.")
    frames, spatial, channels = x.shape
    out = torch.empty_like(x)
    _temporal_conv_kernel[(triton.cdiv(frames, 16), spatial, channels // head_dim)](
        x,
        weight,
        out,
        frames,
        spatial,
        channels,
        D=head_dim,
        BLOCK_T=16,
        ACTIVATE=activate,
        L2=l2norm,
        num_warps=4,
        num_stages=2,
    )
    return out
