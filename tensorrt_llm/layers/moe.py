# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from enum import IntEnum
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._utils import str_dtype_to_trt

from .._common import default_trtnet
from ..functional import _create_tensor, allreduce
from ..module import Module
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from ..quantization import QuantMode
from .linear import ColumnLinear

activation_str_to_int_map = {
    # [WARNING] Keep the below in sync with cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h
    "gelu": 0,
    "gelu_new": 0,
    "relu": 1,
    "silu": 2,
    "swiglu": 3,
    "geglu": 4,
    "identity": 5,
}


# [WARNING] Keep the below in sync with cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h
class MOEParallelismMode(IntEnum):
    NONE = 0
    EXPERT_PARALLEL = 1
    TENSOR_PARALLEL = 2


class MOEExpertScaleNormalizationMode(IntEnum):
    NONE = 0
    RENORMALIZE = 1


def is_gated_activation(activation_str):
    return activation_str in ("swiglu", "geglu")


def _moe_plugin(
        hidden_states,
        routing,
        finished,
        expert_weight_1,
        expert_weight_2,
        expert_bias_1,
        expert_bias_2,
        expert_scale_1,
        expert_scale_2,
        num_experts,
        top_k,
        hidden_size,
        ffn_hidden_size,
        act_fn,
        dtype,
        weight_dtype,  # TODO Is this the right way to do this API?
        tp_size=1,
        tp_group=None,
        tp_rank=0,
        parallelism_mode=MOEParallelismMode.TENSOR_PARALLEL,
        normalization_mode=MOEExpertScaleNormalizationMode.NONE):
    if isinstance(dtype, str):
        dtype = str_dtype_to_trt(dtype)

    # Create the plugin with our required state
    p_num_experts = trt.PluginField("number_of_experts",
                                    np.array(num_experts, dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    p_top_k = trt.PluginField("top_k", np.array(top_k, dtype=np.int32),
                              trt.PluginFieldType.INT32)
    p_expert_hidden_size = trt.PluginField(
        "expert_hidden_size", np.array(hidden_size, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_expert_inter_size = trt.PluginField(
        "expert_inter_size", np.array(ffn_hidden_size, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_activation_type = trt.PluginField(
        "activation_type",
        np.array(activation_str_to_int_map[act_fn], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_type_id = trt.PluginField("type_id", np.array([int(dtype)],
                                                    dtype=np.int32),
                                trt.PluginFieldType.INT32)
    p_weight_type_id = trt.PluginField(
        "weight_type_id", np.array([int(weight_dtype)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_use_finished = trt.PluginField(
        "use_finished", np.array([int(finished is not None)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_use_bias = trt.PluginField(
        "use_bias", np.array([int(expert_bias_1 is not None)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_tp_size = trt.PluginField("tp_size", np.array(tp_size, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    p_tp_rank = trt.PluginField("tp_rank", np.array(tp_rank, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    p_parallelism_mode = trt.PluginField(
        "parallelism_mode", np.array(parallelism_mode, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_normalization_mode = trt.PluginField(
        "normalization_mode", np.array(normalization_mode, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([
        p_num_experts, p_top_k, p_expert_hidden_size, p_expert_inter_size,
        p_activation_type, p_type_id, p_weight_type_id, p_use_finished,
        p_use_bias, p_tp_size, p_tp_rank, p_parallelism_mode,
        p_normalization_mode
    ])

    # Create the plugin with our constant inputs to the constructor
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'MixtureOfExperts', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None
    moe_plugin = plugin_creator.create_plugin("mixture_of_experts", pfc)

    # Instantiate the plugin with our specific inputs
    plugin_inputs = [
        hidden_states, routing, expert_weight_1.value, expert_weight_2.value
    ]

    if expert_bias_1:
        assert expert_bias_2
        plugin_inputs += [expert_bias_1.value, expert_bias_2.value]

    if finished is not None:
        plugin_inputs += [finished]

    # Add conditional inputs
    if expert_scale_1 is not None:
        assert expert_scale_2
        plugin_inputs += [expert_scale_1.value, expert_scale_2.value]

    plugin_inputs = [i.trt_tensor for i in plugin_inputs]
    layer = default_trtnet().add_plugin_v2(plugin_inputs, moe_plugin)
    for ii in range(layer.num_inputs):
        if layer.get_input(ii).dtype == str_dtype_to_trt("int8"):
            layer.get_input(ii).set_dynamic_range(-127, 127)
    output = _create_tensor(layer.get_output(0), layer)
    return output


class MixtureOfExperts(Module):

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 hidden_act: int,
                 top_k: int,
                 bias: bool = True,
                 dtype=None,
                 tp_group: List[int] = None,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 instance_id: int = 0,
                 parallelism_mode=MOEParallelismMode.TENSOR_PARALLEL,
                 normalization_mode=MOEExpertScaleNormalizationMode.NONE,
                 quant_mode=QuantMode(0)):
        super().__init__()

        self.num_experts = num_experts

        self.top_k = top_k
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.dtype = dtype
        self.weight_dtype = dtype
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.instance_id = instance_id
        self.parallelism_mode = parallelism_mode
        self.normalization_mode = normalization_mode

        if self.num_experts % self.tp_size != 0:
            raise ValueError(
                "MixtureOfExperts - Number of experts {} is not a multiple of TP size {}"
                .format(self.num_experts, self.tp_size))
        experts_per_node = num_experts // tp_size

        if quant_mode.is_int8_weight_only():
            self.weight_dtype = trt.int8
        elif quant_mode.is_int4_weight_only():
            raise ValueError(
                "MixtureOfExperts - int4 weight quantization is not supported")

        # TODO We do the routing in parallel and gather afterwards since the softmax needs the results from all threads
        #   we need to determine if its worthwhile, or if there is some more intelligent way we can do the routing
        self.router = ColumnLinear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,  # TODO Quantization here or not?
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=True)

        # Note we use horizontal fusion for gated activation to do the operation in one GEMM invocation
        #  The left matrix is a linear projection (no activation applied)
        #  The right matrix is the gating value (activation applied)
        # The naming convention is the inverse of GatedMLP, but the same as `tensorrt_llm/functional.py`
        expert_1_out_size = ffn_hidden_size * 2 if is_gated_activation(
            hidden_act) else ffn_hidden_size
        # Note that the in/out features is transposed compared to an MLP
        #  This is the order the plugin expects, we should revisit to determine if this is the most efficient choice
        self.experts_weight_1 = Parameter(shape=(experts_per_node, hidden_size,
                                                 expert_1_out_size),
                                          dtype=self.weight_dtype)
        self.experts_weight_2 = Parameter(shape=(experts_per_node,
                                                 ffn_hidden_size, hidden_size),
                                          dtype=self.weight_dtype)

        if quant_mode.is_weight_only():
            self.experts_scale_1 = Parameter(shape=(experts_per_node,
                                                    expert_1_out_size),
                                             dtype=dtype)
            self.experts_scale_2 = Parameter(shape=(experts_per_node,
                                                    hidden_size),
                                             dtype=dtype)
        else:
            self.register_parameter('experts_scale_1', None)
            self.register_parameter('experts_scale_2', None)

        # Note: the bias uses dtype NOT weight_dtype, i.e. it is not quantized
        if bias:
            self.experts_bias_1 = Parameter(shape=(experts_per_node,
                                                   expert_1_out_size),
                                            dtype=dtype)
            self.experts_bias_2 = Parameter(shape=(experts_per_node,
                                                   hidden_size),
                                            dtype=dtype)
        else:
            self.register_parameter('experts_bias_1', None)
            self.register_parameter('experts_bias_2', None)

    def forward(self, hidden_states, finished=None, workspace=None):
        routing = self.router(hidden_states)
        output = _moe_plugin(hidden_states,
                             routing,
                             expert_weight_1=self.experts_weight_1,
                             expert_weight_2=self.experts_weight_2,
                             expert_bias_1=self.experts_bias_1,
                             expert_bias_2=self.experts_bias_2,
                             expert_scale_1=self.experts_scale_1,
                             expert_scale_2=self.experts_scale_2,
                             finished=finished,
                             num_experts=self.num_experts,
                             top_k=self.top_k,
                             hidden_size=self.hidden_size,
                             ffn_hidden_size=self.ffn_hidden_size,
                             act_fn=self.hidden_act,
                             dtype=self.dtype,
                             weight_dtype=self.weight_dtype,
                             tp_size=self.tp_size,
                             tp_rank=self.tp_rank,
                             parallelism_mode=self.parallelism_mode,
                             normalization_mode=self.normalization_mode)

        if self.tp_size > 1 and self.tp_group is not None:
            output = allreduce(output,
                               self.tp_group,
                               workspace=workspace,
                               instance_id=self.instance_id)

        return output


MOE = MixtureOfExperts
