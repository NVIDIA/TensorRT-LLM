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
from ..functional import group_norm, layer_norm, rms_norm
from ..module import Module
from ..parameter import Parameter


class LayerNorm(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        return layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class RmsNorm(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-06,
                 elementwise_affine=True,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        return rms_norm(x, self.normalized_shape, weight, self.eps)


class GroupNorm(Module):

    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-05,
                 affine=True,
                 dtype=None):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.affine = affine

        if self.affine:
            self.weight = Parameter(shape=(self.num_channels, ), dtype=dtype)
            self.bias = Parameter(shape=(self.num_channels, ), dtype=dtype)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        return group_norm(x, self.num_groups, weight, bias, self.eps)
