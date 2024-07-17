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
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import List, Type, Union

import numpy as np
import tensorrt as trt

from tensorrt_llm._utils import get_init_params, str_dtype_to_trt
from tensorrt_llm.layers.lora import LoraParams

from .._common import default_net, default_trtnet
from ..functional import (AllReduceStrategy, _add_plugin_info, _create_tensor,
                          allreduce, cast, div, is_gated_activation,
                          non_gated_version, softmax, sum, topk)
from ..layers import MLP, GatedMLP
from ..mapping import Mapping
from ..module import Module, ModuleList
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from ..quantization import QuantMode
from ..quantization.functional import quantize
from .linear import RowLinear

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


@dataclass
class MoeConfig:

    class ExpertScaleNormalizationMode(IntEnum):
        NONE = 0
        RENORMALIZE = 1

    num_experts: int = 0
    top_k: int = 0
    normalization_mode: ExpertScaleNormalizationMode = ExpertScaleNormalizationMode.RENORMALIZE

    def validate(self) -> "MoeConfig":
        if (self.num_experts == 0) != (self.top_k == 0):
            raise ValueError(
                "Both or neither MoeConfig's num_experts and top_k must be set to 0"
            )
        return self

    def has_moe(self) -> bool:
        return self.num_experts > 1

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_dict(self):
        return asdict(self)


def _moe_plugin(moe_config,
                hidden_states,
                routing,
                finished,
                expert_weight_1,
                expert_weight_2,
                expert_bias_1,
                expert_bias_2,
                expert_scale_1,
                expert_scale_2,
                expert_scale_3,
                expert_scale_4,
                hidden_size,
                ffn_hidden_size,
                act_fn,
                dtype,
                weight_dtype,
                output_dtype,
                quant_mode=QuantMode(0),
                tp_size=1,
                ep_size=1,
                tp_rank=0,
                ep_rank=0):
    if isinstance(dtype, str):
        dtype = str_dtype_to_trt(dtype)

    if isinstance(weight_dtype, str):
        weight_dtype = str_dtype_to_trt(weight_dtype)

    if isinstance(output_dtype, str):
        output_dtype = str_dtype_to_trt(output_dtype)

    def from_parameter(x):
        if isinstance(x, Parameter):
            return x.value
        return x

    expert_weight_1 = from_parameter(expert_weight_1)
    expert_weight_2 = from_parameter(expert_weight_2)
    expert_bias_1 = from_parameter(expert_bias_1)
    expert_bias_2 = from_parameter(expert_bias_2)
    expert_scale_1 = from_parameter(expert_scale_1)
    expert_scale_2 = from_parameter(expert_scale_2)
    expert_scale_3 = from_parameter(expert_scale_3)
    expert_scale_4 = from_parameter(expert_scale_4)

    # Create the plugin with our required state
    num_experts = moe_config.num_experts
    # We pass the full number of experts (not divided by ep_size) even for EP mode
    p_num_experts = trt.PluginField("number_of_experts",
                                    np.array(num_experts, dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    p_top_k = trt.PluginField("top_k", np.array(moe_config.top_k,
                                                dtype=np.int32),
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
    p_output_type_id = trt.PluginField(
        "output_type_id", np.array([int(output_dtype)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_quant_mode = trt.PluginField("quant_mode",
                                   np.array([int(quant_mode)], dtype=np.int32),
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
    p_ep_size = trt.PluginField("ep_size", np.array(ep_size, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    p_ep_rank = trt.PluginField("ep_rank", np.array(ep_rank, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    p_normalization_mode = trt.PluginField(
        "normalization_mode",
        np.array(moe_config.normalization_mode, dtype=np.int32),
        trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([
        p_num_experts, p_top_k, p_expert_hidden_size, p_expert_inter_size,
        p_activation_type, p_type_id, p_weight_type_id, p_output_type_id,
        p_quant_mode, p_use_finished, p_use_bias, p_tp_size, p_tp_rank,
        p_ep_size, p_ep_rank, p_normalization_mode
    ])

    # Create the plugin with our constant inputs to the constructor
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'MixtureOfExperts', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None
    moe_plugin = plugin_creator.create_plugin("mixture_of_experts", pfc)

    # Instantiate the plugin with our specific inputs
    plugin_inputs = [hidden_states, routing, expert_weight_1, expert_weight_2]

    if expert_bias_1:
        assert expert_bias_2
        plugin_inputs += [expert_bias_1, expert_bias_2]

    if finished is not None:
        plugin_inputs += [finished]

    # Add conditional inputs
    if quant_mode.is_weight_only() or quant_mode.has_fp8_qdq():
        assert expert_scale_1
        assert expert_scale_2
        plugin_inputs += [expert_scale_1, expert_scale_2]

    # Add conditional inputs
    if quant_mode.has_fp8_qdq():
        assert expert_scale_3
        plugin_inputs += [expert_scale_3]

    if expert_scale_4 is not None:
        assert quant_mode.has_fp8_qdq()
        assert output_dtype == trt.fp8
        plugin_inputs += [expert_scale_4]

    plugin_inputs = [i.trt_tensor for i in plugin_inputs]
    layer = default_trtnet().add_plugin_v2(plugin_inputs, moe_plugin)
    _add_plugin_info(layer, plugin_creator, "mixture_of_experts", pfc)
    if not default_net().strongly_typed:
        for ii in range(layer.num_inputs):
            if layer.get_input(ii).dtype == str_dtype_to_trt("int8"):
                layer.get_input(ii).set_dynamic_range(-127, 127)
    output = _create_tensor(layer.get_output(0), layer)
    return output


# This exists so that MOE can have the same name format as a regular MLP, just with different shaped weight tensors
class MOEWeightWrapper(Module):

    def __init__(self, in_features: int, out_features: int,
                 experts_per_node: int, quant_mode: QuantMode,
                 dtype: Union[str, trt.DataType],
                 weight_dtype: Union[str, trt.DataType], has_bias: bool):
        super().__init__()
        self.quant_mode = quant_mode
        self.expert_shape = (experts_per_node, out_features, in_features)
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.has_bias = has_bias

        if quant_mode.is_weight_only():
            bytes_per_col_scale = 2 if quant_mode.is_int4_weight_only() else 1
            # We use a different shape here because the quantized weights have their own layout
            self.expert_shape = (experts_per_node, in_features,
                                 out_features // bytes_per_col_scale)
            self.per_channel_scale = Parameter(shape=(experts_per_node,
                                                      out_features),
                                               dtype=dtype)
        else:
            self.register_parameter('per_channel_scale', None)

        self.weight = Parameter(shape=self.expert_shape, dtype=weight_dtype)

        if has_bias:
            self.bias = Parameter(shape=(experts_per_node, out_features),
                                  dtype=dtype)
        else:
            self.register_parameter('bias', None)

        if quant_mode.has_fp8_qdq():
            self.activation_scaling_factor = Parameter(shape=(1, ),
                                                       dtype=trt.float32)
            self.weights_scaling_factor = Parameter(shape=(experts_per_node, 1),
                                                    dtype=trt.float32)
        else:
            self.register_parameter('activation_scaling_factor', None)
            self.register_parameter('weights_scaling_factor', None)


class MixtureOfExperts(Module):

    def __init__(self,
                 moe_config: MoeConfig,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 hidden_act: str,
                 mapping: Mapping = Mapping(),
                 bias: bool = True,
                 dtype=None,
                 tp_group: List[int] = None,
                 tp_size: int = 1,
                 quant_mode=QuantMode(0)):
        super().__init__()

        self.moe_config = moe_config
        self.num_experts = moe_config.num_experts
        self.top_k = moe_config.top_k

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.expert_inter_size = ffn_hidden_size
        self.dtype = dtype
        self.weight_dtype = dtype
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.mapping = mapping
        self.quant_mode = quant_mode
        self.bias = bias

        self.experts_per_node = self.num_experts
        if self.mapping.has_moe_ep():
            if self.num_experts % self.mapping.moe_ep_size != 0:
                raise ValueError(
                    f"MixtureOfExperts - Number of experts {self.num_experts} is not a multiple of EP size {self.mapping.moe_ep_size}"
                )
            self.experts_per_node = self.experts_per_node // self.mapping.moe_ep_size

        if self.mapping.has_moe_tp():
            if self.ffn_hidden_size % self.mapping.moe_tp_size != 0:
                raise ValueError(
                    f"MixtureOfExperts - FFN Hidden Size {self.ffn_hidden_size} is not a multiple of TP size {self.mapping.moe_tp_size}"
                )
            self.expert_inter_size = self.ffn_hidden_size // self.mapping.moe_tp_size

        if quant_mode.has_fp8_qdq() and self.bias:
            # TODO (dastokes) We will need to revisit this if we have a use case for it
            raise ValueError(
                f"MixtureOfExperts - Bias is not supported with FP8")

        if quant_mode.is_weight_only():
            self.weight_dtype = trt.int8
        elif quant_mode.has_fp8_qdq():
            self.weight_dtype = trt.fp8

        # Since output dimension is usually low (in the order of 10s), no TP at
        # all is more efficient as no allreduce required in the end.
        # Note that if we see models that have large number of experts, we may
        # need to consider add TP back here.
        # TODO: Arctic has large # experts, we may need to add TP back here.
        self.router = RowLinear(
            hidden_size,
            self.num_experts,
            bias=False,
            dtype=trt.
            float32,  # Routing is sensitive since it conditions what experts are used
            tp_group=None,
            tp_size=1,
            strict_dtype=True)

        self.init_experts()

    def init_experts(self):
        # Note we use horizontal fusion for gated activation to do the operation in one GEMM invocation
        #  The left matrix is a linear projection (no activation applied)
        #  The right matrix is the gating value (activation applied)
        # The naming convention is the inverse of GatedMLP, but the same as `tensorrt_llm/functional.py`
        fc_out_size = self.expert_inter_size * 2 if is_gated_activation(
            self.hidden_act) else self.expert_inter_size

        self.fc = MOEWeightWrapper(self.hidden_size, fc_out_size,
                                   self.experts_per_node, self.quant_mode,
                                   self.dtype, self.weight_dtype, self.bias)
        self.proj = MOEWeightWrapper(self.expert_inter_size, self.hidden_size,
                                     self.experts_per_node, self.quant_mode,
                                     self.dtype, self.weight_dtype, self.bias)

    def forward(self, hidden_states, finished=None, lora_layer_params=None):
        moe_router_lora_params = None
        if lora_layer_params is not None:
            moe_router_lora_params = lora_layer_params.get_runtime_params(
                0, "moe_router")
        routing_input = cast(hidden_states, trt.float32)
        routing = self.router(routing_input, moe_router_lora_params)
        return self.forward_experts(hidden_states, routing, finished,
                                    lora_layer_params)

    def forward_experts(self, hidden_states, routing, finished,
                        lora_layer_params):
        if lora_layer_params is not None:
            for module in ["mlp_h_to_4h", "mlp_4h_to_h", "mlp_gate"]:
                if lora_layer_params.get_runtime_params(0, module) is not None:
                    raise RuntimeError(
                        f"MoE plugin does not support {module} LoRA module, please disable MoE plugin"
                    )
        if self.quant_mode.has_fp8_qdq():
            assert self.fc.weight.value.dtype == trt.fp8, (
                "mlp fc weight dtype should be fp8 in the fp8 quantization mode."
            )
            assert self.proj.weight.value.dtype == trt.fp8, (
                "mlp proj weight dtype should be fp8 in the fp8 quantization mode."
            )
            hidden_states_quant = hidden_states
            if hidden_states_quant.dtype != trt.fp8:
                hidden_states_quant = quantize(
                    hidden_states, self.fc.activation_scaling_factor.value,
                    'fp8')

            dtype_quant = trt.fp8
            weight_dtype_quant = trt.fp8

            fc1_dequant = self.fc.weights_scaling_factor.value * self.fc.activation_scaling_factor.value
            fc2_quant = div(1.0, self.proj.activation_scaling_factor.value)
            fc2_dequant = self.proj.weights_scaling_factor.value * self.proj.activation_scaling_factor.value

            scale_1 = fc1_dequant
            scale_2 = fc2_quant
            scale_3 = fc2_dequant
            scale_4 = None

            output_dtype_quant = self.dtype

            if output_dtype_quant == trt.fp8 and scale_4 is None:
                raise RuntimeError(
                    "Cannot output FP8 value without knowing quantization parameter"
                )

        else:
            hidden_states_quant = hidden_states
            dtype_quant = self.dtype
            weight_dtype_quant = self.weight_dtype
            output_dtype_quant = self.dtype

            scale_1 = self.fc.per_channel_scale
            scale_2 = self.proj.per_channel_scale
            scale_3 = None
            scale_4 = None
        output = _moe_plugin(self.moe_config,
                             hidden_states_quant,
                             routing,
                             expert_weight_1=self.fc.weight.value,
                             expert_weight_2=self.proj.weight.value,
                             expert_bias_1=self.fc.bias,
                             expert_bias_2=self.proj.bias,
                             expert_scale_1=scale_1,
                             expert_scale_2=scale_2,
                             expert_scale_3=scale_3,
                             expert_scale_4=scale_4,
                             finished=finished,
                             hidden_size=self.hidden_size,
                             ffn_hidden_size=self.expert_inter_size,
                             act_fn=self.hidden_act,
                             dtype=dtype_quant,
                             weight_dtype=weight_dtype_quant,
                             output_dtype=output_dtype_quant,
                             quant_mode=self.quant_mode,
                             tp_size=self.mapping.moe_tp_size,
                             tp_rank=self.mapping.moe_tp_rank,
                             ep_size=self.mapping.moe_ep_size,
                             ep_rank=self.mapping.moe_ep_rank)

        if self.tp_size > 1 and self.tp_group is not None:
            output = allreduce(output, self.tp_group)

        return output

    def load_weights(self, moe: "MixtureOfExperts"):
        '''
        Load weights from base MOE layer
        '''
        raise NotImplementedError("Subclass shall override this")

    def to(self,
           moe_cls: Type["MixtureOfExperts"],
           config=None) -> "MixtureOfExperts":
        from ..quantization.quantize import quantize

        new_moe = moe_cls(**get_init_params(self))
        if config is not None:
            quantize(new_moe, config.quantization)
        new_moe.load_weights(self)
        new_moe.router = self.router
        return new_moe


MOE = MixtureOfExperts


class MoeOOTB(MOE):

    def init_experts(self):
        if self.quant_mode.is_weight_only():
            raise ValueError(
                f"OOTB MOE does not support weight only quantization now, current quant mode: {self.quant_mode}"
            )
        ClsMLP = GatedMLP if is_gated_activation(self.hidden_act) else MLP

        # In OOTB mode, when TP is enabled, using MLP class to do TP settings
        # pass self.ffn_hidden_size to original size,
        if self.mapping.has_moe_tp():
            tp_size = self.mapping.moe_tp_size
            tp_group = self.mapping.moe_tp_group
        else:
            tp_size = 1
            tp_group = None
        self.experts = ModuleList([
            ClsMLP(self.hidden_size, self.ffn_hidden_size,
                   non_gated_version(self.hidden_act), self.bias, self.dtype,
                   tp_group, tp_size, self.quant_mode)
            for _ in range(self.experts_per_node)
        ])

    def moe_to_expert_lora_params(self, lora_layer_params, expert_idx):

        def get_params(module):
            ranks = lora_layer_params.get_runtime_params(0,
                                                         module).lora_ranks[0]
            weights_pointers = lora_layer_params.get_runtime_params(
                0, module).lora_weights_pointers[0]
            return ranks, weights_pointers

        if lora_layer_params is None:
            return None
        fc_lora_ranks, fc_lora_weights_pointers = get_params("moe_h_to_4h")
        proj_lora_ranks, proj_lora_weights_pointers = get_params("moe_4h_to_h")
        gate_lora_ranks = None
        gate_lora_weights_pointers = None
        if is_gated_activation(self.hidden_act):
            gate_lora_ranks, gate_lora_weights_pointers = get_params("moe_gate")
        return LoraParams(
            lora_ranks=[{
                "mlp_h_to_4h_lora_ranks": fc_lora_ranks,
                "mlp_4h_to_h_lora_ranks": proj_lora_ranks,
                "mlp_gate_lora_ranks": gate_lora_ranks,
            }],
            lora_weights_pointers=[{
                "mlp_h_to_4h_lora_weights_pointers":
                fc_lora_weights_pointers,
                "mlp_4h_to_h_lora_weights_pointers":
                proj_lora_weights_pointers,
                "mlp_gate_lora_weights_pointers":
                gate_lora_weights_pointers,
            }],
            host_context_lengths=lora_layer_params.host_context_lengths,
            max_context_length=lora_layer_params.max_context_length,
            max_encoder_context_length=lora_layer_params.
            max_encoder_context_length,
            host_request_types=lora_layer_params.host_request_types,
            host_encoder_input_lengths=lora_layer_params.
            host_encoder_input_lengths,
            weight_index=expert_idx,
        )

    def forward_experts(self, hidden_states, routing, finished,
                        lora_layer_params):
        if self.moe_config.normalization_mode == MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE:
            topk_values, topk_indices = topk(routing, self.top_k, dim=-1)
            topk_values = softmax(topk_values, -1)
        else:
            router_probs = softmax(routing, -1)
            topk_values, topk_indices = topk(router_probs, self.top_k, dim=-1)

        output = hidden_states * 0.0  # Create output space
        # Experts inference
        for i, expert in enumerate(self.experts):
            if self.mapping.has_moe_ep():
                index = i + self.experts_per_node * self.mapping.moe_ep_rank
            else:
                index = i
            # inference expert
            out = expert(hidden_states,
                         lora_layer_params=self.moe_to_expert_lora_params(
                             lora_layer_params, index))

            expert_mask = topk_indices == index
            expert_weights = cast(
                sum(topk_values * cast(expert_mask, topk_values.dtype),
                    dim=-1,
                    keepdim=True), self.dtype)

            output += out * expert_weights
        if self.mapping.has_moe_ep() and self.mapping.moe_ep_group is not None:
            output = allreduce(output,
                               self.mapping.moe_ep_group,
                               strategy=AllReduceStrategy.NCCL)

        return output

    def load_weights(self, moe: MOE):
        for i, expert in enumerate(self.experts):
            is_gated_act = is_gated_activation(self.hidden_act)
            # Gated weight pack in expert1 weights
            # expert_weight_1
            experts_weight_1_raw = moe.fc.weight.raw_value
            if self.quant_mode.is_weight_only():
                expert.fc.weight.value = experts_weight_1_raw[
                    i, :, -self.expert_inter_size:]
                if is_gated_act:
                    expert.gate.weight.value = experts_weight_1_raw[
                        i, :, :self.expert_inter_size]
            else:
                expert.fc.weight.value = experts_weight_1_raw[
                    i, -self.expert_inter_size:, :]
                if is_gated_act:
                    expert.gate.weight.value = experts_weight_1_raw[
                        i, :self.expert_inter_size, :]

            # expert_weight_2
            experts_weight_2_raw = moe.proj.weight.raw_value
            expert.proj.weight.value = experts_weight_2_raw[i, :, :]

            has_bias = self.bias
            if has_bias:
                experts_bias_1_raw = moe.fc.bias.raw_value
                expert.fc.bias.value = experts_bias_1_raw[
                    i, -self.expert_inter_size:]
                experts_bias_2_raw = moe.proj.bias.raw_value
                expert.proj.bias.value = experts_bias_2_raw[i, :]
                if is_gated_act:
                    expert.gate.bias.value = experts_bias_1_raw[
                        i, :self.expert_inter_size]
