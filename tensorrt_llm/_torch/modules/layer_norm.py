from typing import Optional, Tuple, Union

import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        has_weights: bool = True,
        has_bias: bool = True,
    ):
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
        if has_bias:
            self.bias = nn.Parameter(
                torch.zeros(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer('bias',
                                 torch.zeros(hidden_size,
                                             dtype=dtype,
                                             device=device),
                                 persistent=False)
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if isinstance(residual, torch.Tensor):
            hidden_states = hidden_states + residual.to(torch.float32)
            residual = hidden_states.to(input_dtype)

        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.var(-1, keepdim=True, unbiased=False)
        hidden_states = (hidden_states -
                         mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype) + self.bias

        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual
