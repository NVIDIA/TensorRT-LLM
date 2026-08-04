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
from typing import Optional

from .._common import default_net
from ..functional import (ACT2FN, Tensor, concat, conv2d, gather, mamba_conv1d,
                          permute, selective_scan, shape, split, view)
from ..module import Module
from ..parameter import Parameter
from .linear import ColumnLinear, Linear, RowLinear
from .normalization import RmsNorm


class MambaConv1d(Module):

    def __init__(self,
                 d_inner,
                 d_conv=4,
                 pre_stride=0,
                 post_stride=0,
                 dtype=None,
                 apply_silu=True):
        super().__init__()
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.pre_stride = pre_stride
        self.post_stride = post_stride
        self.dtype = dtype
        self.weight = Parameter(shape=(self.d_inner, 1, self.d_conv, 1),
                                dtype=dtype)
        self.bias = Parameter(shape=(self.d_inner, ), dtype=dtype)
        self.apply_silu = apply_silu

    def forward(self,
                x: Tensor,
                conv_state: Tensor,
                host_request_types: Tensor,
                last_token_ids: Tensor,
                host_context_lengths: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                conv_indices: Optional[Tensor] = None):
        '''
        Parameters:
            x: [B, L, D] or [T, D]
            conv_state: [B, W, D] or [1] of type int64 for paged state
            host_request_types: [B]
            last_token_ids: [B]
            host_context_lengths: [B]
            slot_mapping: [B]
            conv_indices: [B]
        '''
        if default_net().plugin_config.mamba_conv1d_plugin:
            transposed_weight = permute(
                view(self.weight.value, shape=[self.d_inner, 1, self.d_conv]),
                (1, 2, 0))
            x_conv, conv_state = mamba_conv1d(
                x, conv_state, transposed_weight, self.bias.value,
                host_request_types, last_token_ids, self.d_inner, self.d_conv,
                self.dtype, self.pre_stride, self.post_stride,
                host_context_lengths, slot_mapping, self.apply_silu)
        else:
            assert not default_net().plugin_config.paged_state
            assert len(
                x.shape
            ) == 3, "remove_input_padding is not supported by OOTB for Mamba."
            if self.pre_stride > 0:
                _, x = split(x,
                             [self.pre_stride, self.d_inner + self.post_stride],
                             dim=-1)
            if self.post_stride > 0:
                x, _ = split(x, [self.d_inner, self.post_stride], dim=-1)
            x = x.permute([0, 2, 1])

            # In context phase, conv_state is a zero tensor, and it is used for padding
            # In generation phase, conv_state is a tensor of the past x
            x_pad = concat([conv_state, x], dim=2)

            # Update conv_state
            conv_state = gather(x_pad, 2, conv_indices)

            # Convolution
            x_pad = x_pad.view(
                concat([shape(x_pad, 0),
                        shape(x_pad, 1),
                        shape(x_pad, 2), 1]))
            x_conv = conv2d(x_pad,
                            self.weight.value,
                            self.bias.value,
                            groups=self.d_inner)
            if self.apply_silu:
                x_conv = ACT2FN['silu'](x_conv)
            x_conv = x_conv.view(
                concat([shape(x_conv, 0),
                        shape(x_conv, 1),
                        shape(x_conv, 2)]))

            # Get dt, B and C
            x_conv = x_conv.permute([0, 2, 1])
        return x_conv, conv_state


class Mamba(Module):

    def __init__(self,
                 d_model,
                 d_inner,
                 d_state=16,
                 d_conv=4,
                 dt_rank="auto",
                 bias=False,
                 dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_inner
        self.dt_rank = math.ceil(self.d_model /
                                 16) if dt_rank == "auto" else dt_rank
        self.dtype = dtype

        self.A = Parameter(shape=(self.d_state, self.d_inner), dtype="float32")

        self.D = Parameter(shape=(self.d_inner, ), dtype="float32")
        self.dt_bias = Parameter(shape=(self.d_inner, ), dtype="float32")

        self.in_proj_x = Linear(self.d_model,
                                self.d_inner,
                                bias=bias,
                                dtype=dtype,
                                gather_output=False)
        self.in_proj_z = Linear(self.d_model,
                                self.d_inner,
                                bias=bias,
                                dtype=dtype,
                                gather_output=False)

        self.conv1d = MambaConv1d(self.d_inner, self.d_conv, dtype=self.dtype)

        self.x_proj = Linear(self.d_inner,
                             self.dt_rank + self.d_state * 2,
                             bias=False,
                             dtype=dtype,
                             gather_output=False)

        self.dt_proj = Linear(self.dt_rank,
                              self.d_inner,
                              bias=False,
                              dtype=dtype,
                              gather_output=False,
                              pad_lda=self.d_state * 2)

        self.out_proj = Linear(self.d_inner,
                               self.d_model,
                               bias=bias,
                               dtype=dtype,
                               gather_output=False)

    def forward(self,
                hidden_states: Tensor,
                conv_state: Tensor,
                ssm_state: Tensor,
                host_request_types: Tensor,
                last_token_ids: Tensor,
                host_context_lengths: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                conv_indices: Optional[Tensor] = None):
        '''
        Parameters:
            hidden_states: [B, L, D] or [T, D]
            conv_state: [B, W, D] or [1] of type int64 for paged state
            ssm_state: [B, N, D] or [1] of type int64 for paged state
            host_request_types: [B]
            last_token_ids: [B]
            host_context_lengths: [B]
            slot_mapping: [B]
            conv_indices: [B]
        '''
        # in_proj
        x = self.in_proj_x(hidden_states)
        z = self.in_proj_z(hidden_states)

        x_conv, conv_state = self.conv1d(x, conv_state, host_request_types,
                                         last_token_ids, host_context_lengths,
                                         slot_mapping, conv_indices)

        # Get dt, B and C
        x_dbl = self.x_proj(x_conv)
        if default_net().plugin_config.gemm_plugin:
            dt = self.dt_proj(x_dbl)
        else:
            dt, _ = split(x_dbl, [self.dt_rank, self.d_state * 2], dim=-1)
            dt = self.dt_proj(dt)

        # selective scan
        y, ssm_state = selective_scan(x_conv,
                                      ssm_state,
                                      dt,
                                      self.dt_bias.value,
                                      self.A.value,
                                      x_dbl,
                                      self.D.value,
                                      host_request_types,
                                      last_token_ids,
                                      self.d_inner,
                                      self.d_state,
                                      self.dt_rank,
                                      delta_softplus=True,
                                      dtype=self.dtype,
                                      z=z,
                                      host_context_lengths=host_context_lengths,
                                      slot_mapping=slot_mapping)
        # out_proj
        out = self.out_proj(y)
        return out, conv_state, ssm_state


class Mamba2(Module):

    def __init__(self,
                 d_model,
                 d_inner,
                 d_state=16,
                 d_conv=4,
                 headdim=64,
                 ngroups=1,
                 chunk_size=256,
                 bias=False,
                 rmsnorm=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        assert d_inner % tp_size == 0
        self.d_inner = d_inner // tp_size
        self.headdim = headdim
        assert ngroups % tp_size == 0
        self.ngroups = ngroups // tp_size
        self.chunk_size = chunk_size
        self.rmsnorm = rmsnorm
        self.dtype = dtype
        assert d_inner % headdim == 0
        nheads = d_inner // headdim
        assert nheads % tp_size == 0
        self.nheads = nheads // tp_size
        # conv1d needs alignment to 8 fp16s
        self.pad_ldc = (self.nheads + 7) // 8 * 8 - self.nheads
        pad_ldc = self.pad_ldc * tp_size

        self.A = Parameter(shape=(self.nheads, ), dtype="float32")
        self.D = Parameter(shape=(self.nheads, ), dtype="float32")
        self.dt_bias = Parameter(shape=(self.nheads, ), dtype="float32")

        d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
        self.in_proj = ColumnLinear(d_model,
                                    d_in_proj,
                                    bias=bias,
                                    dtype=dtype,
                                    tp_group=tp_group,
                                    tp_size=tp_size,
                                    gather_output=False,
                                    pad_ldc=pad_ldc)

        self.conv_dim = (d_inner + 2 * ngroups * d_state) // tp_size
        self.conv1d = MambaConv1d(self.conv_dim,
                                  self.d_conv,
                                  pre_stride=self.d_inner,
                                  post_stride=self.nheads + self.pad_ldc,
                                  dtype=self.dtype)

        if rmsnorm:
            self.norm = RmsNorm(normalized_shape=self.d_inner,
                                num_groups=self.ngroups,
                                eps=1e-5,
                                dtype=dtype)

        self.out_proj = RowLinear(d_inner,
                                  d_model,
                                  bias=bias,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                conv_state: Tensor,
                ssm_state: Tensor,
                host_request_types: Tensor,
                last_token_ids: Tensor,
                host_context_lengths: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                conv_indices: Optional[Tensor] = None):
        '''
        Parameters:
            hidden_states: [B, L, D] or [T, D]
            conv_state: [B, W, D_conv] or [1] of type int64 for paged state
            ssm_state: [B, H, N, D] or [1] of type int64 for paged state
            host_request_types: [B]
            last_token_ids: [B]
            host_context_lengths: [B]
            slot_mapping: [B]
            conv_indices: [B]
        '''
        # in_proj
        zxbcdt = self.in_proj(hidden_states)

        # conv1d
        xbc_conv, conv_state = self.conv1d(zxbcdt, conv_state,
                                           host_request_types, last_token_ids,
                                           host_context_lengths, slot_mapping,
                                           conv_indices)

        # mamba scan
        y, ssm_state = selective_scan(xbc_conv,
                                      ssm_state,
                                      zxbcdt,
                                      self.dt_bias.value,
                                      self.A.value,
                                      xbc_conv,
                                      self.D.value,
                                      host_request_types,
                                      last_token_ids,
                                      self.d_inner,
                                      self.d_state,
                                      dt_rank=0,
                                      delta_softplus=True,
                                      dtype=self.dtype,
                                      z=zxbcdt,
                                      host_context_lengths=host_context_lengths,
                                      slot_mapping=slot_mapping,
                                      nheads=self.nheads,
                                      ngroups=self.ngroups,
                                      chunk_size=self.chunk_size,
                                      mamba_version='Mamba2')

        # norm
        if self.rmsnorm:
            y = self.norm(y)

        # out_proj
        out = self.out_proj(y)
        return out, conv_state, ssm_state
