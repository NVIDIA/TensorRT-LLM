"""Custom operator for FlashInfer and Triton RMSNorm implementation."""

import flashinfer
import torch

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
