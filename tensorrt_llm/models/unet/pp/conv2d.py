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
import tensorrt as trt

from ....functional import allgather, concat, conv2d, slice, stack, unsqueeze
from ....layers import Conv2d
from ....mapping import Mapping
from ....module import Module


def pad(input, pad):
    assert input.ndim() == 4
    n, c, h, w = input.shape
    padded_input = slice(input,
                         starts=[0, 0, -pad[2], -pad[0]],
                         sizes=[n, c, pad[2] + h + pad[3], pad[0] + w + pad[1]],
                         mode=trt.SampleMode.FILL,
                         fill_value=0.0)
    return padded_input


class DistriConv2dPP(Module):

    def __init__(self,
                 conv: Conv2d,
                 mapping: Mapping = Mapping(),
                 is_first_layer: bool = False):
        super().__init__()
        self.mapping = mapping
        self.conv = conv
        self.is_first_layer = is_first_layer

    def sliced_forward(self, x):
        mapping = self.mapping
        b, c, h, w = x.shape
        assert h % mapping.tp_size == 0

        stride = self.conv.stride[0]
        padding = self.conv.padding[0]

        output_h = x.shape[2] // stride // mapping.tp_size
        idx = mapping.tp_rank
        h_begin = output_h * idx * stride - padding
        h_end = output_h * (idx + 1) * stride + padding
        final_padding = [padding, padding, 0, 0]
        if h_begin < 0:
            h_begin = 0
            final_padding[2] = padding
        if h_end > h:
            h_end = h
            final_padding[3] = padding
        sliced_input = slice(x, [0, 0, h_begin, 0], [b, c, h_end - h_begin, w])
        padded_input = pad(sliced_input, final_padding)
        return conv2d(padded_input,
                      self.conv.weight.value,
                      None if self.conv.bias is None else self.conv.bias.value,
                      stride=self.conv.stride,
                      padding=(0, 0))

    def forward(self, x, *args, **kwargs):
        mapping = self.mapping
        if self.is_first_layer:
            full_x = x
            output = self.sliced_forward(full_x)
        else:
            boundary_size = self.conv.padding[0]

            def create_padded_x(x, boundaries):
                if mapping.tp_rank == 0:
                    b = boundaries.select(0, mapping.tp_rank + 1).select(0, 0)
                    concat_x = concat([x, b], dim=2)
                    padded_x = pad(concat_x, [0, 0, boundary_size, 0])
                elif mapping.tp_rank == mapping.tp_size - 1:
                    b = boundaries.select(0, mapping.tp_rank - 1).select(0, 1)
                    concat_x = concat([b, x], dim=2)
                    padded_x = pad(concat_x, [0, 0, 0, boundary_size])
                else:
                    b0 = boundaries.select(0, mapping.tp_rank - 1).select(0, 1)
                    b1 = boundaries.select(0, mapping.tp_rank + 1).select(0, 0)
                    padded_x = concat(
                        [
                            b0,
                            x,
                            b1,
                        ],
                        dim=2,
                    )
                return padded_x

            n, c, h, w = x.shape
            b0 = slice(x, [0, 0, 0, 0], [n, c, boundary_size, w])
            b1 = slice(x, [0, 0, h - boundary_size, 0],
                       [n, c, boundary_size, w])
            boundary = stack([b0, b1], dim=0)

            boundaries = allgather(unsqueeze(boundary, 0),
                                   group=mapping.tp_group)
            padded_x = create_padded_x(x, boundaries)
            output = conv2d(
                padded_x,
                self.conv.weight.value,
                self.conv.bias.value,
                stride=self.conv.stride,
                padding=(0, self.conv.padding[1]),
            )

        return output
