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

import enum
from types import EllipsisType  # https://stackoverflow.com/a/66636313
from typing import Optional, Tuple, TypeAlias, Union, cast

import torch
from torch import nn

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE


class RMSNorm(nn.Module):

    _ARGUMENT_NOT_SPECIFIED_SENTINEL = ...
    _ArgumentNotSpecifiedSentinelType: TypeAlias = EllipsisType

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        has_weights: bool = True,
        use_gemma: bool = False,
    ):
        super().__init__()

        if use_gemma and not has_weights:
            raise ValueError("has_weights must be True if use_gemma is True")

        if has_weights:
            if not use_gemma:
                self.weight = nn.Parameter(
                    torch.ones(hidden_size, dtype=dtype, device=device))
            else:
                self.weight = nn.Parameter(
                    torch.zeros(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer('weight',
                                 torch.ones(hidden_size,
                                            dtype=dtype,
                                            device=device),
                                 persistent=False)
        self.variance_epsilon = eps
        self.use_gemma = use_gemma

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Union[
            Optional[torch.Tensor],
            _ArgumentNotSpecifiedSentinelType] = _ARGUMENT_NOT_SPECIFIED_SENTINEL,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        return_residual = True
        if residual is self._ARGUMENT_NOT_SPECIFIED_SENTINEL:
            return_residual = False
            residual = None

        if IS_FLASHINFER_AVAILABLE:
            from ..custom_ops import (flashinfer_fused_add_rmsnorm,
                                      flashinfer_gemma_fused_add_rmsnorm,
                                      flashinfer_gemma_rmsnorm,
                                      flashinfer_rmsnorm)
            if residual is not None:
                if not self.use_gemma:
                    flashinfer_fused_add_rmsnorm(hidden_states, residual,
                                                 self.weight,
                                                 self.variance_epsilon)
                else:
                    flashinfer_gemma_fused_add_rmsnorm(hidden_states, residual,
                                                       self.weight,
                                                       self.variance_epsilon)
            else:
                if not self.use_gemma:
                    hidden_states = flashinfer_rmsnorm(hidden_states,
                                                       self.weight,
                                                       self.variance_epsilon)
                else:
                    hidden_states = flashinfer_gemma_rmsnorm(
                        hidden_states, self.weight, self.variance_epsilon)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            if residual is not None:
                hidden_states = hidden_states + residual.to(torch.float32)
                residual = hidden_states.to(input_dtype)

            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance +
                                                        self.variance_epsilon)
            if not self.use_gemma:
                hidden_states = self.weight * hidden_states.to(input_dtype)
            else:
                hidden_states = (self.weight +
                                 1) * hidden_states.to(input_dtype)

        if return_residual:
            return hidden_states, cast(Optional[torch.Tensor], residual)
        else:
            return hidden_states

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        residual: Union[
            Optional[torch.Tensor],
            _ArgumentNotSpecifiedSentinelType] = _ARGUMENT_NOT_SPECIFIED_SENTINEL,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if residual is self._ARGUMENT_NOT_SPECIFIED_SENTINEL:
            return hidden_states
        else:
            return hidden_states, cast(Optional[torch.Tensor], residual)


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
    '''Group RMS Normalization optimized for up to 2 inputs.

    This function applies RMS normalization to multiple inputs simultaneously,
    achieving better performance than normalizing each tensor separately with multi-stream.

    Args:
        inputs: List of input tensors to normalize
        weights: Optional list of weight tensors corresponding to each input
        eps: Small constant added to variance for numerical stability
        weight_bias: Optional bias added to weights during normalization
        kernel: Kernel selection strategy:
            - heuristic: Automatically selects optimal kernel based on inputs and hardware
            - base: Uses base kernel (optimal for most cases)
            - large_batch: Uses large batch kernel (may be better for large batches)
        outputs: Optional pre-allocated output tensors (created if None)

    Returns:
        List of normalized tensors with the same shapes as inputs

    Technical Details:
        Available kernel implementations:
        - Base kernel: Allocates warps proportional to the sum of last dimensions,
          providing better SM occupancy for most workloads.
        - Large batch kernel: Allocates warps proportional to the maximum last dimension,
          which can be more efficient for large batch sizes with 2 inputs.

        The heuristic mode uses a logistic regression model trained on benchmark data
        to dynamically select the optimal kernel based on batch size, input dimensions,
        and GPU architecture. This selection is optimized for compute capabilities 9.x and 10.x.
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
