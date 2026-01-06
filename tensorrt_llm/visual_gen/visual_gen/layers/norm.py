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

import numbers
from typing import Optional

import torch
from torch.nn import Parameter

try:
    import flashinfer
except ImportError:
    flashinfer = None


class ditRMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

    @torch.compiler.disable
    def forward(self, hidden_states):
        origin_dtype = hidden_states.dtype
        # currently FlashInfer don't support float32
        if self.weight.dtype == torch.float32:
            self.weight.data = self.weight.data.to(torch.bfloat16)
        if origin_dtype != self.weight.dtype:
            hidden_states = hidden_states.to(self.weight.dtype)
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]
        reshaped = hidden_states.reshape(-1, hidden_size)
        if flashinfer is None:
            raise ImportError("flashinfer is not installed")
        out = flashinfer.norm.rmsnorm(reshaped, self.weight, self.eps)
        if out.dtype != origin_dtype:
            out = out.to(origin_dtype)
        return out.view(original_shape)

    @classmethod
    def from_torch(cls, module, load_parameters: bool = False):
        if isinstance(module, torch.nn.RMSNorm):
            normalized_shape = module.normalized_shape
            eps = module.eps
            elementwise_affine = module.elementwise_affine
            weight = module.weight
            if weight is not None:
                device = weight.device
                dtype = weight.dtype
            else:
                device = None
                dtype = None
        else:
            # try to load from customized rmsnorm, such as in HunyuanImage-2.1
            assert module.weight is not None
            weight = module.weight
            normalized_shape = weight.shape
            eps = module.eps
            elementwise_affine = True
            device = weight.device
            dtype = weight.dtype

        new_norm = cls(normalized_shape, eps, elementwise_affine, device, dtype)
        if load_parameters and module.weight is not None:
            new_norm.weight.data = module.weight.data
        return new_norm


class ditLayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        origin_dtype = input.dtype
        output = super().forward(input)
        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output

    @classmethod
    def from_torch(cls, module, load_parameters: bool = False):
        if isinstance(module, torch.nn.LayerNorm):
            normalized_shape = module.normalized_shape
            eps = module.eps
            elementwise_affine = module.elementwise_affine
            weight = module.weight
            bias = module.bias is not None
            if weight is not None:
                device = weight.device
                dtype = weight.dtype
            else:
                device = None
                dtype = None
        else:
            raise ValueError(f"Unsupported norm module: {type(module)}")
        new_norm = cls(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        if load_parameters and weight is not None:
            new_norm.weight.data = weight.data
            if bias:
                new_norm.bias.data = module.bias.data
        return new_norm
