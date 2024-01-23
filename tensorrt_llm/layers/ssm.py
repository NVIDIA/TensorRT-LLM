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

import math
from dataclasses import dataclass

from ..functional import (ACT2FN, Tensor, concat, selective_scan, shape, slice,
                          split)
from ..module import Module
from ..parameter import Parameter
from .conv import Conv2d
from .linear import Linear


@dataclass
class MambaParameters:
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: str = "auto"
    conv_bias: bool = True
    bias: bool = False


class Mamba(Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        conv_bias=True,
        bias=False,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model /
                                 16) if dt_rank == "auto" else dt_rank
        self.dtype = dtype

        self.A = Parameter(shape=(self.d_inner, self.d_state), dtype="float32")
        self.D = Parameter(shape=(self.d_inner, ), dtype="float32")
        self.dt_bias = Parameter(shape=(self.d_inner, ), dtype="float32")

        self.in_proj = Linear(self.d_model,
                              self.d_inner * 2,
                              bias=bias,
                              dtype=dtype,
                              gather_output=False)

        self.conv1d = Conv2d(self.d_inner,
                             self.d_inner,
                             kernel_size=(d_conv, 1),
                             groups=self.d_inner,
                             bias=conv_bias,
                             dtype=dtype)

        self.x_proj = Linear(self.d_inner,
                             self.dt_rank + self.d_state * 2,
                             bias=False,
                             dtype=dtype,
                             gather_output=False)

        self.dt_proj = Linear(self.dt_rank,
                              self.d_inner,
                              bias=False,
                              dtype=dtype,
                              gather_output=False)

        self.out_proj = Linear(self.d_inner,
                               self.d_model,
                               bias=bias,
                               dtype=dtype,
                               gather_output=False)

    def forward(self, hidden_states: Tensor, conv_state: Tensor,
                ssm_state: Tensor, host_request_types: Tensor):
        '''
        Parameters:
            hidden_states: [B, L, D]
            conv_state: [B, D, W]
            ssm_state: [B, D, N]
            host_request_types: [B]
        '''
        # in_proj
        xz = self.in_proj(hidden_states)
        xz = xz.permute([0, 2, 1])
        x, z = split(xz, [self.d_inner, self.d_inner], dim=1)

        # In context phase, conv_state is a zero tensor, and it is used for padding
        # In generation phase, conv_state is a tensor of the past x
        x_pad = concat([conv_state, x], dim=2)

        # Update conv_state
        slice_shape = concat([shape(x, 0), self.d_inner, self.d_conv - 1])
        conv_state = slice(x_pad, concat([0, 0, shape(x, 2)]), slice_shape)

        # Convolution
        x_pad = x_pad.view(
            concat([shape(x_pad, 0),
                    shape(x_pad, 1),
                    shape(x_pad, 2), 1]))
        x_conv = ACT2FN['silu'](self.conv1d(x_pad))
        x_conv = x_conv.view(
            concat([shape(x_conv, 0),
                    shape(x_conv, 1),
                    shape(x_conv, 2)]))

        # Get dt, B and C
        x_dbl = self.x_proj(x_conv.permute([0, 2, 1]))
        dt, B, C = split(x_dbl, [self.dt_rank, self.d_state, self.d_state],
                         dim=2)
        dt = self.dt_proj(dt).permute([0, 2, 1])
        B = B.permute([0, 2, 1])
        C = C.permute([0, 2, 1])

        # selective scan
        y, ssm_state = selective_scan(x_conv,
                                      ssm_state,
                                      dt,
                                      self.dt_bias.value,
                                      self.A.value,
                                      B,
                                      C,
                                      self.D.value,
                                      z,
                                      host_request_types,
                                      self.d_inner,
                                      self.d_state,
                                      is_variable_B=True,
                                      is_variable_C=True,
                                      delta_softplus=True)

        # out_proj
        out = self.out_proj(y.permute([0, 2, 1]))
        return out, conv_state, ssm_state
