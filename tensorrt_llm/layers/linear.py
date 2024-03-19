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

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

from .._common import default_net, default_trtnet
from .._utils import pad_vocab_size, set_obj_attrs, str_dtype_to_trt
from ..functional import (Tensor, _add_plugin_info, _create_tensor, allgather,
                          allreduce, cast, matmul)
from ..mapping import Mapping
from ..module import Module
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from .lora import LoraRuntimeParams


def _gemm_plugin(input: Tensor,
                 mat2: Tensor,
                 transa: bool = False,
                 transb: bool = False,
                 use_fp8: bool = False,
                 strict_dtype: Optional[trt.DataType] = None) -> Tensor:
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Gemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    transa = 1 if transa else 0
    transa = trt.PluginField("transa", np.array(transa, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    transb = 1 if transb else 0
    transb = trt.PluginField("transb", np.array(transb, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    use_fp8 = 1 if use_fp8 else 0
    use_fp8 = trt.PluginField("use_fp8", np.array(use_fp8, dtype=np.int32),
                              trt.PluginFieldType.INT32)

    if strict_dtype is not None:
        assert isinstance(strict_dtype, trt.DataType)
        p_dtype = strict_dtype
    else:
        p_dtype = str_dtype_to_trt(default_net().plugin_config.gemm_plugin)
    pf_type = trt.PluginField("type_id", np.array([int(p_dtype)], np.int32),
                              trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([transa, transb, pf_type, use_fp8])
    gemm_plug = plg_creator.create_plugin("gemm", pfc)
    plug_inputs = [input.trt_tensor, mat2.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
    _add_plugin_info(layer, plg_creator, "gemm", pfc)
    return _create_tensor(layer.get_output(0), layer)


class Linear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 use_fp8=False,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 share_weight=None,
                 strict_dtype=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features // tp_size
        self.dtype = dtype
        self.use_fp8 = use_fp8

        if not share_weight:
            self.weight = Parameter(shape=(self.out_features, self.in_features),
                                    dtype=('fp8' if use_fp8 else dtype))
            set_obj_attrs(self.weight, {
                "weight_loader": self.weight_loader,
            })
        else:
            self.weight = share_weight

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.strict_dtype = self.dtype if strict_dtype else None

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
            set_obj_attrs(self.bias, {
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter('bias', None)

    def multiply_gather(self,
                        x,
                        weight,
                        gemm_plugin,
                        lora_runtime_params: LoraRuntimeParams = None):
        hidden_state = x
        if gemm_plugin:
            x = _gemm_plugin(x,
                             weight,
                             transb=True,
                             use_fp8=self.use_fp8,
                             strict_dtype=self.strict_dtype)
        else:
            x = matmul(x, weight, transb=True)

        if default_net(
        ).plugin_config.lora_plugin and lora_runtime_params is not None:
            x = x + self.lora(hidden_state,
                              lora_runtime_params=lora_runtime_params)

        if self.bias is not None:
            bias = cast(self.bias.value, x.dtype)
            x = x + bias

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=-1)

        return x

    def forward(self, x, lora_runtime_params: LoraRuntimeParams = None):
        return self.multiply_gather(x,
                                    self.weight.value,
                                    default_net().plugin_config.gemm_plugin,
                                    lora_runtime_params=lora_runtime_params)

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor):
        tp_rank = mapping.tp_rank
        output_dim = 0
        shard_size = param._shape[output_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        param.value = loaded_weight


ColumnLinear = Linear


class QKVColumnLinear(ColumnLinear):

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor):
        tp_rank = mapping.tp_rank
        output_dim = 0
        shard_size = param._shape[output_dim] // 3
        start_idx = tp_rank * shard_size
        # reshape for qkv_weights
        assert loaded_weight.shape[output_dim] % 3 == 0
        loaded_weight = loaded_weight.reshape(
            3, loaded_weight.shape[output_dim] // 3, -1)
        loaded_weight = loaded_weight.narrow(output_dim + 1, start_idx,
                                             shard_size)
        loaded_weight = loaded_weight.reshape(
            loaded_weight.shape[output_dim + 1] * 3, -1)
        # for bias
        if len(param._shape) == 1:
            loaded_weight.squeeze_(-1)
        param.value = loaded_weight


class ParallelLMHead(ColumnLinear):

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor):
        tp_rank = mapping.tp_rank
        output_dim = 0
        shard_size = param._shape[output_dim]
        start_idx = tp_rank * shard_size
        # vocab padding for TP
        vocab_size = loaded_weight.shape[output_dim]
        pad_width = pad_vocab_size(vocab_size, self.tp_size) - vocab_size
        if pad_width > 0:
            loaded_weight = F.pad(loaded_weight, (0, 0, 0, pad_width),
                                  mode="constant",
                                  value=0)
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param.value = loaded_weight


class RowLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 use_fp8=False,
                 tp_group=None,
                 tp_size=1,
                 strict_dtype: bool = False):
        super().__init__()
        self.in_features = in_features // tp_size
        self.out_features = out_features
        self.dtype = dtype
        self.use_fp8 = use_fp8

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=('fp8' if use_fp8 else dtype))
        set_obj_attrs(self.weight, {
            "weight_loader": self.weight_loader,
        })

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_group = tp_group
        self.tp_size = tp_size
        self.strict_dtype = self.dtype if strict_dtype else None

    def multiply_reduce(self,
                        x,
                        weight,
                        gemm_plugin,
                        use_fp8=False,
                        lora_runtime_params: LoraRuntimeParams = None):
        hidden_state = x
        if gemm_plugin:
            x = _gemm_plugin(x,
                             weight,
                             transb=True,
                             use_fp8=self.use_fp8,
                             strict_dtype=self.strict_dtype)
        else:
            x = matmul(x, weight, transb=True)

        if default_net(
        ).plugin_config.lora_plugin and lora_runtime_params is not None:
            x = x + self.lora(hidden_state,
                              lora_runtime_params=lora_runtime_params)

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            bias = cast(self.bias.value, x.dtype)
            x = x + bias

        return x

    def forward(self, x, lora_runtime_params: LoraRuntimeParams = None):
        return self.multiply_reduce(x,
                                    self.weight.value,
                                    default_net().plugin_config.gemm_plugin,
                                    lora_runtime_params=lora_runtime_params)

    def weight_loader(self, mapping: Mapping, param: Parameter,
                      loaded_weight: torch.Tensor):
        tp_rank = mapping.tp_rank
        input_dim = 1
        shard_size = param._shape[input_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        param.value = loaded_weight
