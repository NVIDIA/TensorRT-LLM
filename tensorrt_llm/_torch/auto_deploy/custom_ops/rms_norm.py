"""Custom operator for FlashInfer and Triton RMSNorm implementation."""

import flashinfer
import torch

from ...modules.mamba.layernorm_gated import _layer_norm_fwd
from .triton_kernels.rms_norm import rms_norm


@torch.library.custom_op("auto_deploy::flashinfer_rms_norm", mutates_args=())
def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Custom operator for FlashInfer RMSNorm implementation.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor using FlashInfer implementation.
    """
    # Flashinfer rmsnorm expects a 2D input
    input_flat = input.reshape(-1, input.shape[-1])
    rmsnorm_flat = flashinfer.norm.rmsnorm(input_flat, weight, eps)
    return rmsnorm_flat.reshape(input.shape)


@flashinfer_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Empty tensor with same shape as input.
    """
    return torch.empty_like(input)


@torch.library.custom_op("auto_deploy::triton_rms_norm", mutates_args=())
def triton_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Custom operator for Triton RMSNorm implementation.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor using Triton implementation.
    """
    return rms_norm(input, weight, eps)


@triton_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing."""
    return torch.empty_like(input)


@torch.library.custom_op("auto_deploy::torch_rmsnorm", mutates_args=())
def torch_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Custom operator for Torch RMSNorm implementation.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.
    """
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


@torch_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing."""
    return torch.empty_like(input)


@torch.library.custom_op("auto_deploy::torch_rmsnorm_gated", mutates_args=())
def torch_rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool = False,
) -> torch.Tensor:
    """
    Group RMSNorm with optional SiLU gating, using Triton kernel `_layer_norm_fwd`.

    Shapes:
      x:     [..., H]
      gate:  same as x or None
      weight:[H]
      H % group_size == 0

    Returns:
      fp32 tensor of shape like x.
    """
    assert x.dim() >= 2, "x must be at least 2D"
    H = weight.numel()
    assert x.shape[-1] == H, "weight must match last dim (hidden size)"
    assert (H % group_size) == 0, f"H={H} must be divisible by group_size={group_size}"
    if gate is not None:
        assert gate.shape == x.shape, "gate must match x shape"

    # Flatten to (M, H), ensure last-dim contiguous, and run in fp32
    x_shape = x.shape
    x2 = x.to(torch.float32).reshape(-1, H)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()

    z2 = None
    if gate is not None:
        z2 = gate.to(torch.float32).reshape(-1, H)
        if z2.stride(-1) != 1:
            z2 = z2.contiguous()

    w = weight.to(torch.float32).contiguous()

    out2, _, _ = _layer_norm_fwd(
        x2,
        w,
        None,  # bias
        eps,
        z=z2,
        out=None,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )

    return out2.reshape(x_shape)


@torch_rmsnorm_gated.register_fake
def _torch_rmsnorm_gated_meta(
    x,
    weight,
    gate,
    eps: float,
    group_size: int,
    norm_before_gate: bool = False,
):
    assert x.dim() >= 2, "x must be at least 2D"
    H = x.shape[-1]
    assert weight.numel() == H, "weight must match last dim (hidden size)"
    assert (H % group_size) == 0, f"H={H} must be divisible by group_size={group_size}"
    if gate is not None:
        assert gate.shape == x.shape, "gate must match x shape"

    return x.new_empty(x.shape, dtype=torch.float32)
