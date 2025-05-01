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


def group_rms_norm(
        inputs: list[torch.Tensor],
        weights: Optional[list[torch.Tensor]] = [],
        eps: Optional[float] = 1e-5,
        weight_bias: Optional[float] = 0.0,
        outputs: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
    '''
    Group RMS Normalization for multiple inputs.
    '''
    out = outputs
    if out is None:
        out = [torch.empty_like(input) for input in inputs]
    torch.ops.trtllm.group_rms_norm(inputs, out, weights, eps, weight_bias)
    return out


def group_rms_norm_large_batch(
        inputs: list[torch.Tensor],
        weights: Optional[list[torch.Tensor]] = [],
        eps: Optional[float] = 1e-5,
        weight_bias: Optional[float] = 0.0,
        outputs: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
    '''
    Group RMS Normalization for 2 inputs with large batch size.
    '''
    out = outputs
    if out is None:
        out = [torch.empty_like(input) for input in inputs]
    torch.ops.trtllm.group_rms_norm_large_batch(inputs, out, weights, eps,
                                                weight_bias)
    return out
