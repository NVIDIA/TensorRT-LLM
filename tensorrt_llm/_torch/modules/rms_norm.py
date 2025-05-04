import enum
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..custom_ops import IS_FLASHINFER_AVAILABLE


class RMSNorm(nn.Module):

    def __init__(self,
                 *,
                 hidden_size: int,
                 eps: float,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 has_weights: bool = True):
        super().__init__()
        if has_weights:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer('weight',
                                 torch.ones(hidden_size,
                                            dtype=dtype,
                                            device=device),
                                 persistent=False)
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if IS_FLASHINFER_AVAILABLE:
            from ..custom_ops import (flashinfer_fused_add_rmsnorm,
                                      flashinfer_rmsnorm)
            if isinstance(residual, torch.Tensor):
                flashinfer_fused_add_rmsnorm(hidden_states, residual,
                                             self.weight, self.variance_epsilon)
            else:
                hidden_states = flashinfer_rmsnorm(hidden_states, self.weight,
                                                   self.variance_epsilon)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            if isinstance(residual, torch.Tensor):
                hidden_states = hidden_states + residual.to(torch.float32)
                residual = hidden_states.to(input_dtype)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance +
                                                        self.variance_epsilon)
            hidden_states = self.weight * hidden_states.to(input_dtype)

        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual


class GroupRMSNormKernelSelection(enum.Enum):
    heuristic = 0
    base = 1
    large_batch = 2


def group_rms_norm(
        inputs: list[torch.Tensor],
        weights: Optional[list[torch.Tensor]] = [],
        eps: Optional[float] = 1e-5,
        weight_bias: Optional[float] = 0.0,
        kernel: GroupRMSNormKernelSelection = GroupRMSNormKernelSelection.
    heuristic,
        outputs: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
    '''Group RMS Normalization for up to 2 inputs.

    This function applies RMS normalization to up to 2 inputs simultaneously,
    which can be more efficient than normalizing each tensor using multi-stream.

    Available kernel implementations:
    - Base kernel: Allocates warps proportional to the sum of last dimension of inputs,
      providing better SM occupancy for most cases.
    - Large batch kernel: Allocates warps proportional to the maximum last dimension of inputs,
      which can be more efficient for large batch sizes with 2 inputs.

    The heuristic mode uses a logistic regression model to dynamically select the optimal
    kernel based on batch size, input dimensions, and GPU architecture.
    The selection model was trained on benchmark data from various configurations
    and is optimized for Compute Capabilities 9.x and 10.x.

    Args:
        inputs: List of input tensors to normalize
        weights: Optional list of weight tensors for each input
        eps: Small constant added to variance for numerical stability
        weight_bias: Optional bias to apply to weights
        heuristic: Kernel selection strategy:
            - heuristic: Automatically select between base and large batch kernels
            - base: Always use the base kernel
            - large_batch: Always use the large batch kernel
        outputs: Optional pre-allocated output tensors

    Returns:
        List of normalized tensors with the same shapes as inputs
    '''
    out = outputs
    if out is None:
        out = [torch.empty_like(input) for input in inputs]
    match kernel:
        case GroupRMSNormKernelSelection.heuristic:
            torch.ops.trtllm.group_rms_norm_heuristic(inputs, out, weights, eps,
                                                      weight_bias)
        case GroupRMSNormKernelSelection.base:
            torch.ops.trtllm.group_rms_norm_base(inputs, out, weights, eps,
                                                 weight_bias)
        case GroupRMSNormKernelSelection.large_batch:
            torch.ops.trtllm.group_rms_norm_large_batch(inputs, out, weights,
                                                        eps, weight_bias)
    return out
