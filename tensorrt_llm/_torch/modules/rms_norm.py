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
                 device: Optional[torch.device] = None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if IS_FLASHINFER_AVAIABLE:
            from ..custom_ops import (flashinfer_fused_add_rmsnorm,
                                      flashinfer_rmsnorm)
            if residual is not None:
                flashinfer_fused_add_rmsnorm(hidden_states, residual,
                                             self.weight, self.variance_epsilon)
                return hidden_states, residual
            return flashinfer_rmsnorm(hidden_states, self.weight,
                                      self.variance_epsilon)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            if residual is not None:
                hidden_states = hidden_states + residual.to(torch.float32)
                residual = hidden_states.to(input_dtype)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance +
                                                        self.variance_epsilon)
            hidden_states = self.weight * hidden_states.to(input_dtype)

            if residual is not None:
                return hidden_states, residual
            return hidden_states


class GroupRMSNorm(nn.Module):

    def __init__(self,
                 *,
                 hidden_sizes: list[int],
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
        # Create dummy weights is enable_weights is False
        # The weights will be ignored by the CUDA kernel
        if not enable_weights:
            self.dummy_weights = [
                nn.Parameter(
                    torch.empty(hidden_size, dtype=dtype, device=device))
                for hidden_size in hidden_sizes
            ]

    def forward(
            self,
            inputs: list[torch.Tensor],
            weights: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
        """
        Apply RMS normalization to a group of inputs.

        Args:
            inputs: List of tensors to normalize [batch_size, hidden_dim]

        Returns:
            List of normalized tensors with same shape as inputs
        """

        if len(inputs) == 0:
            return []
        if self.enable_weights:
            assert weights is not None, "weights must be provided if enable_weights is True"
            outputs = torch.ops.trtllm.group_rms_norm(inputs, weights,
                                                      self.variance_epsilon,
                                                      self.weight_bias,
                                                      self.enable_weights)
            return outputs

        outputs = torch.ops.trtllm.group_rms_norm(inputs, self.dummy_weights,
                                                  self.variance_epsilon,
                                                  self.weight_bias,
                                                  self.enable_weights)
        return outputs
