"""Custom operator for FlashInfer and Triton RMSNorm implementation."""

import flashinfer
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from ...flashinfer_utils import get_env_enable_pdl
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
    rmsnorm_flat = flashinfer.norm.rmsnorm(input_flat, weight, eps, enable_pdl=get_env_enable_pdl())
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
    # pre-allocate output to ensure same dtype+stride as input
    out = torch.empty_like(input)
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    out.copy_((weight * input.to(out.dtype)))
    return out


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
    """Custom operator for Torch gated RMSNorm implementation.

    Group RMSNorm with optional SiLU gating, using pure PyTorch operations.

    Args:
        x: Input tensor of shape [..., H].
        weight: Scaling weights of shape [H].
        gate: Optional gate tensor with same shape as x, or None.
        eps: Small constant for numerical stability.
        group_size: Size of groups for grouped normalization. H must be divisible by group_size.
        norm_before_gate: If True, apply gating after normalization. If False, apply before.

    Returns:
        Normalized and optionally gated tensor of shape like x.
    """
    dtype = x.dtype
    weight = weight.float()
    x = x.float()
    z = gate.float() if gate is not None else gate

    if z is not None and not norm_before_gate:
        x = x * F.silu(z)

    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = x * rstd * weight
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight

    if z is not None and norm_before_gate:
        out *= F.silu(z)

    return out.to(dtype)


@torch_rmsnorm_gated.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool = False,
) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing."""
    return x.new_empty(x.shape, dtype=x.dtype)


@torch.library.custom_op("auto_deploy::triton_rmsnorm_gated", mutates_args=())
def triton_rmsnorm_gated(
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
    x2 = x.reshape(-1, H)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()

    z2 = None
    if gate is not None:
        z2 = gate.reshape(-1, H)
        if z2.stride(-1) != 1:
            z2 = z2.contiguous()
    assert weight.is_contiguous(), "weight must be contiguous"

    out2, _, _ = _layer_norm_fwd(
        x2,
        weight,
        None,  # bias
        eps,
        z=z2,
        out=None,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )

    return out2.reshape(x_shape)


@triton_rmsnorm_gated.register_fake
def _triton_rmsnorm_gated_meta(
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


# Forked from:
# https://github.com/state-spaces/mamba/blob/6b32be06d026e170b3fdaf3ae6282c5a6ff57b06/mamba_ssm/ops/triton/layernorm_gated.py
# NOTES:
# 1. At time of writing (09/25/2025), the nano nemotron v2 modeling code expects `mamba_ssm`
#    to be installed so as to be able to make use of its grouped gated RMS norm operation.
#    We therefore replace it with one that uses einops + pytorch.
def gated_rms_norm_ref(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True
):
    dtype = x.dtype
    # N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


# =============================================================================
# Sharded RMSNorm (for sharded activations)
# =============================================================================


@torch.library.custom_op("auto_deploy::sharded_rmsnorm", mutates_args=())
def sharded_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float, world_size: int
) -> torch.Tensor:
    """RMSNorm for sharded activations that need global reduction.

    When activations are sharded (split along the last dimension across devices),
    standard RMSNorm computes an incorrect local mean. This op uses all_reduce to compute
    the global mean of squared values across all shards, ensuring correct normalization.

    The computation is:
        1. Compute local sum of squares: sum(input^2) over local features
        2. All-reduce to get global sum of squares across all shards
        3. Compute global mean: global_sum / (local_dim * world_size)
        4. Normalize: input * rsqrt(global_mean + eps)
        5. Scale with local weight (weight is also column-sharded)

    Args:
        input: Input tensor, shape [..., local_hidden_size] where local_hidden_size
               is the shard of the full hidden dimension on this device.
        weight: Scaling weights, shape [local_hidden_size] (column-sharded).
        eps: Small constant for numerical stability.
        world_size: Number of devices across which the activation is sharded.

    Returns:
        Normalized and scaled tensor with same shape as input.
    """
    local_dim = input.shape[-1]

    # Cast to float32 for precision
    input_fp32 = input.to(torch.float32)

    # Compute local sum of squares (NOT mean - we need sum for all_reduce)
    local_sum_sq = input_fp32.pow(2).sum(-1, keepdim=True)

    # All-reduce to get global sum of squares
    global_sum_sq = local_sum_sq.clone()
    dist.all_reduce(global_sum_sq, op=dist.ReduceOp.SUM)

    # Compute global mean: global_sum / total_elements
    global_count = local_dim * world_size
    global_mean_sq = global_sum_sq / global_count

    # Normalize
    input_normalized = input_fp32 * torch.rsqrt(global_mean_sq + eps)

    # Apply weight (local weight since it's also column-sharded)
    out = weight * input_normalized.to(input.dtype)
    return out


@sharded_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float, world_size: int) -> torch.Tensor:
    """Fake implementation for tracing."""
    return torch.empty_like(input)
