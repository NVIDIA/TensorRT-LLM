# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark RMSNorm and RoPE."""

import functools

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version, is_sm_100f
from ..cute_dsl_kernels.blackwell.dspark_rmsnorm_rope import DSparkRMSNormRoPEKernel


def is_fused_dspark_rmsnorm_rope_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
) -> bool:
    """Return whether tensors satisfy the production fused-op contract."""
    if not is_sm_100f() or not all(t.is_cuda for t in (x, weight, freqs)):
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if freqs.dtype != torch.float32:
        return False
    if x.ndim < 2 or x.shape[-1] % 32 != 0:
        return False
    if weight.shape != (x.shape[-1],):
        return False
    if rope_dim < 0 or rope_dim > x.shape[-1] or rope_dim % 2 != 0:
        return False
    if (x.shape[-1] - rope_dim) % 32 != 0 or (rope_dim // 2) % 32 != 0:
        return False
    rows = x.numel() // x.shape[-1]
    if num_heads <= 0 or rows % num_heads != 0:
        return False
    return (
        freqs.ndim == 3
        and freqs.shape[0] == rows // num_heads
        and freqs.shape[1] >= max(1, rope_dim // 2)
        and freqs.shape[2] == 2
        and x.is_contiguous()
        and weight.is_contiguous()
        and freqs.is_contiguous()
    )


@functools.cache
def _compile_fused_dspark_rmsnorm_rope(
    hidden_dim: int,
    rope_dim: int,
    num_heads: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
):
    rows = cute.sym_int()
    freq_rows = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, hidden_dim), stride_order=(1, 0)
    )
    weight_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_dim,), stride_order=(0,)
    )
    freqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (freq_rows, cute.sym_int(), 2),
        stride_order=(2, 1, 0),
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (rows, hidden_dim), stride_order=(1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkRMSNormRoPEKernel(
        hidden_dim,
        rope_dim,
        num_heads,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
    )
    return cute.compile(
        kernel,
        x_fake,
        weight_fake,
        freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_rmsnorm_rope",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_dspark_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
) -> torch.Tensor:
    """Apply fused RMSNorm and adjacent-pair RoPE to contiguous BF16 rows."""
    if not is_fused_dspark_rmsnorm_rope_supported(x, weight, freqs, num_heads, rope_dim):
        raise ValueError(
            "cute_dsl_dspark_rmsnorm_rope requires contiguous BF16 tensors on "
            f"SM100/SM103 with a valid FP32 frequency view; got SM {get_sm_version()}"
        )

    original_shape = x.shape
    x_flat = x.view(-1, x.shape[-1])
    output = torch.empty_like(x_flat)
    compiled = _compile_fused_dspark_rmsnorm_rope(
        x.shape[-1],
        rope_dim,
        num_heads,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
    )
    compiled(x_flat, weight, freqs, output)
    return output.view(original_shape)


@torch.library.register_fake("trtllm::cute_dsl_dspark_rmsnorm_rope")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
) -> torch.Tensor:
    return torch.empty_like(x)
