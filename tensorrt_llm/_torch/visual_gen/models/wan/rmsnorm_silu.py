# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Triton kernel is from SGLang (Apache-2.0), commit
# cc171fbad0266e8eaabb031f4d3858557a23d7e8:
# python/sglang/kernels/ops/diffusion/norm/wan_rmsnorm_silu_triton.py
# Kernel arithmetic and launch configuration are unchanged. The surrounding
# registration and decoder-only automatic dispatch are adapted for TensorRT-LLM.

"""BF16 Wan decoder RMSNorm + SiLU for eligible SM100 channels-last input."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _wan_rmsnorm_silu_kernel(
    x_ptr,
    gamma_ptr,
    bias_ptr,
    out_ptr,
    channels: tl.constexpr,
    rms_scale,
    eps,
    has_bias: tl.constexpr,
    block_c: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_c)
    mask = offsets < channels
    # Dense channels-last-3d stores each pixel as one contiguous channel row.
    # Address it directly instead of recovering b/t/h/w with integer div/mod.
    row_offsets = row * channels + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    inv_norm = 1.0 / tl.maximum(norm, eps)

    # Eager op boundaries: normalize/*scale in x.dtype; *gamma/+bias in the
    # promoted output dtype; SiLU in fp32, stored in the output dtype.
    y = (x * inv_norm).to(x_ptr.dtype.element_ty)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    y = (y * rms_scale).to(x_ptr.dtype.element_ty)
    y = (y.to(tl.float32) * gamma.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if has_bias:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        y = (y.to(tl.float32) + bias.to(tl.float32)).to(out_ptr.dtype.element_ty)
    y = y.to(tl.float32)
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + row_offsets, y, mask=mask)


def _rmsnorm_silu_cuda(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    rms_scale: float,
    eps: float,
    has_bias: bool,
) -> torch.Tensor:
    bsz, channels, t_size, h_size, w_size = x.shape
    # Preserve strides, including singleton dimensions, through the decoder.
    dtype = torch.promote_types(x.dtype, gamma.dtype)
    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=dtype)
    block_c = triton.next_power_of_2(channels)
    num_warps = 1 if block_c <= 64 else 4 if block_c <= 512 else 8

    with torch.get_device_module().device(x.device):
        _wan_rmsnorm_silu_kernel[(bsz * t_size * h_size * w_size,)](
            x,
            gamma,
            bias,
            out,
            channels,
            rms_scale,
            eps,
            has_bias,
            block_c,
            num_warps=num_warps,
        )
    return out


def _fake_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    rms_scale: float,
    eps: float,
    has_bias: bool,
) -> torch.Tensor:
    dtype = torch.promote_types(x.dtype, gamma.dtype)
    return torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=dtype)


_LIBRARY = torch.library.Library("trtllm", "FRAGMENT")
_LIBRARY.define(
    "wan_rmsnorm_silu(Tensor x, Tensor gamma, Tensor bias, float rms_scale, "
    "float eps, bool has_bias) -> Tensor",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
_LIBRARY.impl("wan_rmsnorm_silu", _rmsnorm_silu_cuda, "CUDA")
torch.library.register_fake("trtllm::wan_rmsnorm_silu", _fake_rmsnorm_silu)


def rmsnorm_silu(
    x: torch.Tensor, gamma: torch.Tensor, zero_bias: torch.Tensor, scale: float
) -> torch.Tensor:
    """Fuse an eligible NCTHW BF16 invocation with a preallocated zero bias.

    The internal caller checks dense channels-last strides and inference-only
    ownership. ``gamma`` is contiguous [C, 1, 1, 1] and ``zero_bias`` is [C],
    both BF16 on the input device. Retain the bias addition and its rounding
    boundary even though the native module has a scalar zero bias.
    """
    return torch.ops.trtllm.wan_rmsnorm_silu(x, gamma, zero_bias, scale, 1e-12, True)
