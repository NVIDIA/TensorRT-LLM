# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime input classification for FP8 block-scaling GEMM dispatch."""

import torch

from .fp8_block_scaling_dispatch import (
    ActivationScaleLayout,
    DispatchKey,
    MatrixLayout,
    WeightScaleLayout,
)


def classify_activation_scale_layout(
    scale: torch.Tensor,
    m: int,
    k_blocks: int,
) -> ActivationScaleLayout:
    if scale.dim() == 2 and scale.shape == (m, k_blocks):
        return ActivationScaleLayout.LOGICAL_M_K_BLOCKS
    if scale.dim() == 2 and scale.shape[0] == k_blocks and scale.shape[1] >= m:
        return ActivationScaleLayout.TRT_TRANSPOSED_K_M
    m_padded = ((m + 3) // 4) * 4
    if scale.dim() == 1 and scale.numel() >= k_blocks * m_padded:
        return ActivationScaleLayout.TRT_PADDED_1D
    return ActivationScaleLayout.UNSUPPORTED


def make_dispatch_key(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> DispatchKey:
    """Classify a runtime call into an exact cache key."""
    m, k = a.shape
    n = b.shape[0]
    k_blocks = k // 128
    activation_layout = classify_activation_scale_layout(a_scale, m, k_blocks)
    weight_layout = (
        WeightScaleLayout.LOGICAL_N_K_BLOCKS
        if b_scale.dim() == 2 and b_scale.shape == ((n + 127) // 128, k_blocks)
        else WeightScaleLayout.UNSUPPORTED
    )
    same_device = a.device == b.device == a_scale.device == b_scale.device
    matrix_layout = (
        MatrixLayout.K_MAJOR_CONTIGUOUS
        if same_device and a.stride(1) == 1 and b.stride(1) == 1
        else MatrixLayout.UNSUPPORTED
    )
    return DispatchKey(
        m=m,
        n=n,
        k=k,
        activation_scale_layout=activation_layout,
        weight_scale_layout=weight_layout,
        matrix_layout=matrix_layout,
    )


def is_deep_gemm_compatible(
    key: DispatchKey,
    tensors: tuple[torch.Tensor, ...],
) -> bool:
    a, b, a_scale, b_scale = tensors
    return (
        key.k % 128 == 0
        and key.activation_scale_layout is not ActivationScaleLayout.UNSUPPORTED
        and key.weight_scale_layout is WeightScaleLayout.LOGICAL_N_K_BLOCKS
        and key.matrix_layout is MatrixLayout.K_MAJOR_CONTIGUOUS
        and a.is_cuda
        and a.dtype is torch.float8_e4m3fn
        and b.dtype is torch.float8_e4m3fn
        and a_scale.dtype is torch.float32
        and b_scale.dtype is torch.float32
        and a_scale.is_contiguous()
        and b_scale.is_contiguous()
    )


def is_deep_gemm_tensor_metadata_compatible(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k_blocks: int,
) -> bool:
    """Validate metadata not represented by a cached activation layout."""
    same_device = a.device == b.device == a_scale.device == b_scale.device
    return (
        b_scale.dim() == 2
        and b_scale.shape == ((n + 127) // 128, k_blocks)
        and same_device
        and a.stride(1) == 1
        and b.stride(1) == 1
        and a.is_cuda
        and a.dtype is torch.float8_e4m3fn
        and b.dtype is torch.float8_e4m3fn
        and a_scale.dtype is torch.float32
        and b_scale.dtype is torch.float32
        and a_scale.is_contiguous()
        and b_scale.is_contiguous()
    )
