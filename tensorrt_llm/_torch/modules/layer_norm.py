# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..utils import maybe_compile


class LayerNorm(nn.Module):
    """Layer normalization module with configurable weight and bias parameters.

    This implementation provides standard layer normalization with optional
    learnable parameters and residual connection support.

    Args:
        hidden_size: The size of the hidden dimension to normalize.
        eps: Small constant for numerical stability.
        dtype: Optional data type for parameters.
        device: Optional device for parameters.
        has_weights: Whether to include learnable weight parameters.
        has_bias: Whether to include learnable bias parameters.
    """

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

    @maybe_compile(dynamic=True)
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply layer normalization to input tensor.

        Args:
            hidden_states: Input tensor to normalize.
            residual: Optional residual tensor to add before normalization.

        Returns:
            Normalized tensor, or tuple of (normalized_tensor, residual) if residual provided.
        """

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if isinstance(residual, torch.Tensor):
            hidden_states = hidden_states + residual.to(torch.float32)
            residual = hidden_states.to(input_dtype)

        hidden_states = nn.functional.layer_norm(
            hidden_states,
            (hidden_states.shape[-1], ),
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        ).to(input_dtype)

        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Skip normalization and return inputs unchanged.

        Args:
            hidden_states: Input tensor to pass through.
            residual: Optional residual tensor to pass through.

        Returns:
            Input tensors unchanged, maintaining same signature as forward.
        """

        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual
