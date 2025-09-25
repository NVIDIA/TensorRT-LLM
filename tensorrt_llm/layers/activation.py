# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from ..functional import Tensor, gelu, relu, silu, softplus, tanh
from ..module import Module


class SiLU(Module):

    def forward(self, input: Tensor) -> Tensor:
        return silu(input)


class FP32SiLU(Module):
    r"""
    SiLU activation function with input upcasted to float32.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return silu(inputs.cast('float32')).cast(inputs.dtype)


class Mish(Module):

    def forward(self, input: Tensor) -> Tensor:
        return input * tanh(softplus(input, beta=1.0, threshold=20.0))


class GELU(Module):

    def __init__(self, approximate: str = 'tanh') -> None:
        super().__init__()
        self.approximate = approximate
        if approximate != 'tanh':
            raise NotImplementedError('GELU only support tanh now.')

    def forward(self, input: Tensor) -> Tensor:
        return gelu(input)


class ReLU(Module):

    def forward(self, input):
        return relu(input)


ACTIVATION_FUNCTIONS = {
    "swish": SiLU(),
    "silu": SiLU(),
    "mish": Mish(),
    "gelu": GELU(),
    "relu": ReLU(),
    "silu_fp32": FP32SiLU(),
}


def get_activation(act_fn: str) -> Module:
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
