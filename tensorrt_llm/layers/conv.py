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
from typing import Tuple

from ..functional import conv1d, conv2d, conv3d, conv_transpose2d
from ..module import Module
from ..parameter import Parameter


class Conv2d(Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            dilation: Tuple[int, int] = (1, 1),
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            dtype=None) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Parameter(shape=(out_channels, in_channels // groups,
                                       *kernel_size),
                                dtype=dtype)
        if bias:
            self.bias = Parameter(shape=(out_channels, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return conv2d(input, self.weight.value,
                      None if self.bias is None else self.bias.value,
                      self.stride, self.padding, self.dilation, self.groups)


class ConvTranspose2d(Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            output_padding: Tuple[int, int] = (0, 0),
            dilation: Tuple[int, int] = (1, 1),
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            dtype=None) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Parameter(shape=(in_channels, out_channels // groups,
                                       *kernel_size),
                                dtype=dtype)

        if bias:
            self.bias = Parameter(shape=(out_channels, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def _output_padding(self,
                        input,
                        output_size,
                        stride,
                        padding,
                        kernel_size,
                        num_spatial_dims: int,
                        dilation=None):
        if output_size is None:
            ret = self.output_padding
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
                    .format(num_spatial_dims, input.dim(), num_spatial_dims,
                            num_non_spatial_dims + num_spatial_dims,
                            len(output_size)))

            min_sizes = []
            max_sizes = []
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d] -
                    2 * padding[d] +
                    (dilation[d] if dilation is not None else 1) *
                    (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes,
                            input.size()[2:]))

            res = []
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def forward(self, input, output_size=None):
        num_spatial_dims = 2
        output_padding = self._output_padding(input, output_size, self.stride,
                                              self.padding, self.kernel_size,
                                              num_spatial_dims, self.dilation)

        return conv_transpose2d(input, self.weight.value,
                                None if self.bias is None else self.bias.value,
                                self.stride, self.padding, output_padding,
                                self.dilation, self.groups)


class Conv1d(Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            dtype=None) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Parameter(shape=(out_channels, in_channels // groups,
                                       kernel_size, 1),
                                dtype=dtype)
        if bias:
            self.bias = Parameter(shape=(out_channels, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return conv1d(input, self.weight.value,
                      None if self.bias is None else self.bias.value,
                      self.stride, self.padding, self.dilation, self.groups)


class Conv3d(Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int, int],
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (0, 0, 0),
            dilation: Tuple[int, int, int] = (1, 1, 1),
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            dtype=None) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Parameter(shape=(out_channels, in_channels // groups,
                                       kernel_size[0], kernel_size[1],
                                       kernel_size[2]),
                                dtype=dtype)
        if bias:
            self.bias = Parameter(shape=(out_channels, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return conv3d(input, self.weight.value,
                      None if self.bias is None else self.bias.value,
                      self.stride, self.padding, self.dilation, self.groups)
