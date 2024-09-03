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

from typing import Optional

import torch

from .._utils import set_obj_attrs
from ..functional import Tensor, allgather, cast, concat, matmul, rg_lru, shape
from ..mapping import Mapping
from ..module import Module
from ..parameter import Parameter
from .linear import ColumnLinear, RowLinear
from .ssm import MambaConv1d


class GroupedLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 num_blocks,
                 bias=True,
                 dtype=None,
                 use_fp8=False,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 strict_dtype=False,
                 fuse_bias=False):
        super().__init__()
        assert in_features % num_blocks == 0 and out_features % num_blocks == 0
        assert num_blocks % tp_size == 0
        assert not (gather_output and fuse_bias)
        self.in_features = in_features // tp_size
        self.out_features = out_features // tp_size
        self.num_blocks = num_blocks // tp_size
        self.dtype = dtype
        self.use_fp8 = use_fp8
        self.fuse_bias = fuse_bias

        self.weight = Parameter(shape=(self.num_blocks,
                                       self.in_features // self.num_blocks,
                                       self.out_features // self.num_blocks),
                                dtype=('fp8' if use_fp8 else dtype))
        set_obj_attrs(self.weight, {
            "weight_loader": self.weight_loader,
        })

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.strict_dtype = self.dtype if strict_dtype else None

        if bias:
            self.bias = Parameter(shape=(self.num_blocks,
                                         self.out_features // self.num_blocks),
                                  dtype=dtype)
            set_obj_attrs(self.bias, {
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter('bias', None)

    def multiply_gather(self, x, weight):
        grouped_shape = []
        out_shape = []
        ndim = x.ndim()
        for i in range(x.ndim() - 1):
            grouped_shape.append(shape(x, i))
            out_shape.append(shape(x, i))
        grouped_shape.extend(
            [self.num_blocks, self.in_features // self.num_blocks])
        out_shape.append(self.out_features)
        x = x.view(concat(grouped_shape)).permute([i for i in range(ndim - 2)] +
                                                  [-2, -3, -1])
        x = matmul(x, weight)
        x = x.permute([i for i in range(ndim - 2)] + [-2, -3, -1])

        if self.bias is not None and not self.fuse_bias:
            bias = cast(self.bias.value, x.dtype)
            x = x + bias
        x = x.view(concat(out_shape))

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=-1)

        return x

    def forward(self, x):
        return self.multiply_gather(x, self.weight.value)

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor):
        tp_rank = mapping.tp_rank
        output_dim = 0
        shard_size = param._shape[output_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param.value = loaded_weight


class RgLru(Module):

    def __init__(self,
                 lru_width,
                 num_heads=1,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.lru_width = lru_width
        self.dtype = dtype
        self.num_heads = num_heads
        self.tp_group = tp_group
        self.tp_size = tp_size

        self.recurrent_param = Parameter(shape=(self.lru_width //
                                                self.tp_size, ),
                                         dtype=self.dtype)
        self.input_gate = GroupedLinear(self.lru_width,
                                        self.lru_width,
                                        self.num_heads,
                                        dtype=self.dtype,
                                        tp_group=self.tp_group,
                                        tp_size=self.tp_size,
                                        gather_output=False,
                                        fuse_bias=True)
        self.recurrent_gate = GroupedLinear(self.lru_width,
                                            self.lru_width,
                                            self.num_heads,
                                            dtype=self.dtype,
                                            tp_group=self.tp_group,
                                            tp_size=self.tp_size,
                                            gather_output=False,
                                            fuse_bias=True)

    def forward(self,
                x: Tensor,
                y: Tensor,
                y_bias: Tensor,
                lru_state: Tensor,
                host_request_types: Tensor,
                last_token_ids: Tensor,
                slot_mapping: Optional[Tensor] = None):
        gate_x = self.input_gate(x)
        gate_a = self.recurrent_gate(x)
        out, lru_state = rg_lru(input=x,
                                gate_x=gate_x,
                                gate_x_bias=self.input_gate.bias.value,
                                gate_a=gate_a,
                                gate_a_bias=self.recurrent_gate.bias.value,
                                y=y,
                                y_bias=y_bias,
                                state_or_ptr=lru_state,
                                A=self.recurrent_param.value,
                                host_request_types=host_request_types,
                                last_token_ids=last_token_ids,
                                dim=self.lru_width // self.tp_size,
                                dtype=self.dtype,
                                slot_mapping=slot_mapping)
        return out, lru_state


class FusedRgLru(Module):

    def __init__(self,
                 lru_width,
                 num_heads=1,
                 dtype=None,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.lru_width = lru_width
        self.tp_size = tp_size
        self.dtype = dtype
        self.dim = self.lru_width // self.tp_size
        self.block_size = self.lru_width // num_heads

        self.recurrent_param = Parameter(shape=(self.lru_width // tp_size, ),
                                         dtype=dtype)
        self.gate = GroupedLinear(self.lru_width,
                                  self.lru_width * 2,
                                  num_heads,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  gather_output=False,
                                  fuse_bias=True)

    def forward(self,
                x: Tensor,
                y: Tensor,
                y_bias: Tensor,
                lru_state: Tensor,
                host_request_types: Tensor,
                last_token_ids: Tensor,
                slot_mapping: Optional[Tensor] = None):
        gate = self.gate(x)
        out, lru_state = rg_lru(input=x,
                                gate=gate,
                                gate_bias=self.gate.bias.value,
                                block_size=self.block_size,
                                y=y,
                                y_bias=y_bias,
                                state_or_ptr=lru_state,
                                A=self.recurrent_param.value,
                                host_request_types=host_request_types,
                                last_token_ids=last_token_ids,
                                dim=self.dim,
                                dtype=self.dtype,
                                slot_mapping=slot_mapping)
        return out, lru_state


class Recurrent(Module):

    def __init__(
        self,
        width,
        lru_width,
        d_conv=4,
        num_heads=1,
        dtype=None,
        tp_group=None,
        tp_size=1,
    ):
        super().__init__()
        self.width = width
        self.lru_width = lru_width
        self.d_conv = d_conv
        self.dtype = dtype

        self.linear_x = ColumnLinear(self.width,
                                     self.lru_width,
                                     dtype=dtype,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     gather_output=False)
        self.linear_y = ColumnLinear(self.width,
                                     self.lru_width,
                                     bias=False,
                                     dtype=dtype,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     gather_output=False)
        self.y_bias = Parameter(shape=(self.lru_width // tp_size, ),
                                dtype=dtype)

        self.conv1d = MambaConv1d(self.lru_width // tp_size,
                                  self.d_conv,
                                  dtype=self.dtype,
                                  apply_silu=False)

        self.rg_lru = RgLru(self.lru_width,
                            num_heads=num_heads,
                            dtype=dtype,
                            tp_group=tp_group,
                            tp_size=tp_size)

        self.linear_out = RowLinear(self.lru_width,
                                    self.width,
                                    dtype=dtype,
                                    tp_group=tp_group,
                                    tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                conv_state: Tensor,
                lru_state: Tensor,
                host_request_types: Tensor,
                last_token_ids: Tensor,
                host_context_lengths: Optional[Tensor] = None,
                slot_mapping: Optional[Tensor] = None,
                conv_indices: Optional[Tensor] = None):
        '''
        Parameters:
            hidden_states: [B, L, D] or [T, D]
            conv_state: [B, W, D] or [1] of type int64 for paged state
            lru_state: [B, N] or [1] of type int64 for paged state
            host_request_types: [B]
            last_token_ids: [B]
            host_context_lengths: [B]
            slot_mapping: [B]
            conv_indices: [B]
        '''
        # y branch
        y = self.linear_y(hidden_states)

        # x branch
        x = self.linear_x(hidden_states)
        x_conv, conv_state = self.conv1d(x, conv_state, host_request_types,
                                         last_token_ids, host_context_lengths,
                                         slot_mapping, conv_indices)

        # rg-lru
        out, lru_state = self.rg_lru(x_conv, y, self.y_bias.value, lru_state,
                                     host_request_types, last_token_ids,
                                     slot_mapping)

        # linear out
        out = self.linear_out(out)
        return out, conv_state, lru_state
