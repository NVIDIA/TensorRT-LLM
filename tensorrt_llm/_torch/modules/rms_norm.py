from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..custom_ops import IS_FLASHINFER_AVAIABLE


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
        if IS_FLASHINFER_AVAIABLE:
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


class GroupRMSNorm(nn.Module):

    def __init__(self,
                 *,
                 eps: float = 1e-5,
                 weight_bias: float = 0.0,
                 enable_weights: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):
        """
        Group RMS Normalization for multiple tensors.

        Args:
            hidden_sizes: List of hidden dimensions for each input tensor
            eps: Epsilon for numerical stability
            enable_weights: Whether to use learnable weights
            dtype: Data type for weights
            device: Device for weights
        """
        super().__init__()
        self.variance_epsilon = eps
        self.weight_bias = weight_bias
        self.enable_weights = enable_weights

    def forward(
            self,
            inputs: list[torch.Tensor],
            weights: Optional[list[torch.Tensor]] = None,
            outputs: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
        """
        Apply RMS normalization to a group of inputs.

        Args:
            inputs: List of tensors to normalize [batch_size, hidden_dim]

        Returns:
            List of normalized tensors with same shape as inputs
        """

        if len(inputs) == 0:
            return []
        if outputs is None:
            outputs = [torch.empty_like(input) for input in inputs]
        for input, output in zip(inputs, outputs):
            assert input.device == output.device, "inputs and outputs must have the same device"
            assert input.shape == output.shape, "inputs and outputs must have the same shape"
            assert input.dtype == output.dtype, "inputs and outputs must have the same dtype"

        if self.enable_weights:
            assert weights is not None, "weights must be provided if enable_weights is True"

        w = weights if weights is not None else [
            nn.Parameter(
                torch.empty(
                    input.shape[-1], dtype=input.dtype, device=input.device))
            for input in inputs
        ]

        torch.ops.trtllm.group_rms_norm(inputs, outputs, w,
                                        self.variance_epsilon, self.weight_bias,
                                        self.enable_weights)
        return outputs
