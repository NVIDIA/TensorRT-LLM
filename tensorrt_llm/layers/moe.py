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
from typing import List, Optional, Type, Union

import numpy as np
import tensorrt as trt
import torch

from tensorrt_llm._utils import (get_init_params, str_dtype_to_torch,
                                 str_dtype_to_trt)
from tensorrt_llm.layers.lora import LoraParams

from .._common import default_net, default_trtnet
from .._utils import int32_array
from ..functional import (AllReduceFusionParams, _add_plugin_info,
                          _create_tensor, allreduce, cast, concat, constant,
                          div, expand, gather_nd, is_gated_activation,
                          non_gated_version, nonzero, repeat_interleave,
                          scatter_nd, shape, softmax, split, sum, topk)
from ..layers import MLP, GatedMLP
from ..mapping import Mapping
from ..module import Module, ModuleList
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from ..quantization import QuantMode
from ..quantization.functional import postprocess_weight_only, quantize
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
        SPARSE_MIXER = 2

    num_experts: int = 0
    moe_intermediate_size: int = 0  # Add moe inter size (shanshan)
    num_shared_experts: int = 0  # Add number of shared experts (shanshan)

    top_k: int = 0
    normalization_mode: ExpertScaleNormalizationMode = ExpertScaleNormalizationMode.RENORMALIZE
    sparse_mixer_epsilon: float = 0.01
    tp_mode: int = 0

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
                expert_weights_1,
                expert_weights_2,
                expert_bias_1,
                expert_bias_2,
                expert_scale_1,
                expert_scale_2,
                expert_scale_3,
                expert_scale_4,
                act_scale,
                hidden_size,
                ffn_hidden_size,
                act_fn,
                dtype,
                weight_dtype,
                output_dtype,
                lora_params: LoraParams,
                lora_max_low_rank,
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

    expert_weights_1 = from_parameter(expert_weights_1)
    expert_weights_2 = from_parameter(expert_weights_2)
    expert_bias_1 = from_parameter(expert_bias_1)
    expert_bias_2 = from_parameter(expert_bias_2)
    expert_scale_1 = from_parameter(expert_scale_1)
    expert_scale_2 = from_parameter(expert_scale_2)
    expert_scale_3 = from_parameter(expert_scale_3)
    expert_scale_4 = from_parameter(expert_scale_4)
    act_scale = from_parameter(act_scale)

    # Create the plugin with our required state
    num_experts = moe_config.num_experts
    p_remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int32(default_net().plugin_config.remove_input_padding),
                 dtype=np.int32), trt.PluginFieldType.INT32)
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

    p_sparse_mixer_epsilon = trt.PluginField(
        "sparse_mixer_epsilon",
        np.array(moe_config.sparse_mixer_epsilon, dtype=np.float32),
        trt.PluginFieldType.FLOAT32)

    p_force_determinism = trt.PluginField(
        "force_determinism", np.array([int(False)], dtype=np.int32),
        trt.PluginFieldType.INT32)

    use_lora = default_net().plugin_config.lora_plugin is not None
    p_use_lora = trt.PluginField("use_lora", np.array([int(use_lora)],
                                                      np.int32),
                                 trt.PluginFieldType.INT32)
    if use_lora:
        p_lora_type_id = trt.PluginField(
            "lora_type_id",
            np.array([
                int(str_dtype_to_trt(default_net().plugin_config.lora_plugin))
            ], np.int32), trt.PluginFieldType.INT32)
        p_max_low_rank = trt.PluginField(
            "max_low_rank", np.array(lora_max_low_rank, dtype=np.int32),
            trt.PluginFieldType.INT32)

    pfc_inputs = [
        p_remove_input_padding, p_num_experts, p_top_k, p_expert_hidden_size,
        p_expert_inter_size, p_activation_type, p_type_id, p_weight_type_id,
        p_output_type_id, p_quant_mode, p_use_finished, p_use_bias, p_tp_size,
        p_tp_rank, p_ep_size, p_ep_rank, p_normalization_mode,
        p_sparse_mixer_epsilon, p_force_determinism, p_use_lora
    ]

    if use_lora:
        pfc_inputs += [p_lora_type_id, p_max_low_rank]

    pfc = trt.PluginFieldCollection(pfc_inputs)

    # Create the plugin with our constant inputs to the constructor
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'MixtureOfExperts', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None
    moe_plugin = plugin_creator.create_plugin("mixture_of_experts", pfc)

    # Instantiate the plugin with our specific inputs
    plugin_inputs = [hidden_states, routing, expert_weights_1, expert_weights_2]

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

    if use_lora:
        if quant_mode.has_fp8_qdq():
            assert act_scale
            plugin_inputs += [act_scale]

        moe_h_4h_weight_ptrs = lora_params.get_runtime_params(
            0, "moe_h_to_4h").lora_weights_pointers
        moe_h_4h_lora_ranks = lora_params.get_runtime_params(
            0, "moe_h_to_4h").lora_ranks
        plugin_inputs += (moe_h_4h_weight_ptrs + moe_h_4h_lora_ranks)

        moe_4h_h_weight_ptrs = lora_params.get_runtime_params(
            0, "moe_4h_to_h").lora_weights_pointers
        moe_4h_h_lora_ranks = lora_params.get_runtime_params(
            0, "moe_4h_to_h").lora_ranks
        plugin_inputs += (moe_4h_h_weight_ptrs + moe_4h_h_lora_ranks)

        moe_gate_weight_ptrs = None
        moe_gate_lora_ranks = None
        if is_gated_activation(act_fn):
            moe_gate_weight_ptrs = lora_params.get_runtime_params(
                0, "moe_gate").lora_weights_pointers
            moe_gate_lora_ranks = lora_params.get_runtime_params(
                0, "moe_gate").lora_ranks
            plugin_inputs += (moe_gate_weight_ptrs + moe_gate_lora_ranks)

        host_request_types = lora_params.host_request_types
        plugin_inputs += [host_request_types]

        if default_net().plugin_config.remove_input_padding:
            plugin_inputs += [lora_params.host_context_lengths]

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
                 dtype: Union[str,
                              trt.DataType], weight_dtype: Union[str,
                                                                 trt.DataType],
                 has_bias: bool, wrapper_tllm_to_externel_key_dict: dict,
                 tp_size: int, tp_dim: int):
        super().__init__()
        self.quant_mode = quant_mode
        self.expert_shape = (experts_per_node, out_features, in_features)
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.has_bias = has_bias
        self.tllm_to_externel_key_dict = wrapper_tllm_to_externel_key_dict
        self.tp_size = tp_size
        self.tp_dim = tp_dim
        self.is_padded = False

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

        self.weight = Parameter(shape=self.expert_shape,
                                dtype=weight_dtype,
                                prefer_managed=True)

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

    def postprocess(self, tllm_key, weights, **kwargs):
        if tllm_key.endswith("weight"):
            if isinstance(weights, torch.Tensor):
                weights = [weights]
            if "fc" in tllm_key:
                weights = torch.cat([
                    torch.stack(weights[:len(weights) // 2]),
                    torch.stack(weights[len(weights) // 2:])
                ],
                                    dim=-2)
            elif "proj" in tllm_key:
                weights = torch.stack(weights)
            weights = weights.to(str_dtype_to_torch(self.dtype))

        if not self.quant_mode.has_any_quant():
            return weights
        elif self.quant_mode.is_weight_only():
            if "per_channel_scale" in tllm_key:
                return {}
            weights = weights.to(str_dtype_to_torch(self.dtype))
            return postprocess_weight_only(
                tllm_key, weights, torch.int8 if
                self.quant_mode.is_int8_weight_only() else torch.quint4x2, self)
        elif self.quant_mode.has_fp8_qdq():
            if tllm_key.endswith("activation_scaling_factor"):
                return 448.0 / weights
            elif tllm_key.endswith("weights_scaling_factor"):
                return 448.0 / weights
            else:
                return weights


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
                 quant_mode=QuantMode(0),
                 use_all_reduce=True):
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
        self.use_all_reduce = use_all_reduce

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

        rank_experts = self.mapping.ep_experts(self.num_experts)
        self.wrapper_tllm_to_externel_key_dict = {
            "mlp":
            "block_sparse_moe",
            "proj": [f"experts.{expert}.w2" for expert in rank_experts],
            "fc": [f"experts.{expert}.w3" for expert in rank_experts] +
            [f"experts.{expert}.w1" for expert in rank_experts]
        }

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
        self.router.tllm_to_externel_key_dict = {
            "mlp": "block_sparse_moe",
            "router": "gate"
        }

        self.init_experts()

        self.max_low_rank = None

    def init_experts(self):
        # Note we use horizontal fusion for gated activation to do the operation in one GEMM invocation
        #  The left matrix is a linear projection (no activation applied)
        #  The right matrix is the gating value (activation applied)
        # The naming convention is the inverse of GatedMLP, but the same as `tensorrt_llm/functional.py`
        fc_out_size = self.expert_inter_size * 2 if is_gated_activation(
            self.hidden_act) else self.expert_inter_size

        self.fc = MOEWeightWrapper(self.hidden_size, fc_out_size,
                                   self.experts_per_node, self.quant_mode,
                                   self.dtype, self.weight_dtype, self.bias,
                                   self.wrapper_tllm_to_externel_key_dict,
                                   self.mapping.moe_tp_size, 0)
        self.proj = MOEWeightWrapper(self.expert_inter_size, self.hidden_size,
                                     self.experts_per_node, self.quant_mode,
                                     self.dtype, self.weight_dtype, self.bias,
                                     self.wrapper_tllm_to_externel_key_dict,
                                     self.mapping.moe_tp_size, 1)

    def forward(self,
                hidden_states,
                finished=None,
                lora_layer_params=None,
                reduce_fusion_params: Optional[AllReduceFusionParams] = None):
        moe_router_lora_params = None
        if lora_layer_params is not None:
            moe_router_lora_params = lora_layer_params.get_runtime_params(
                0, "moe_router")
        routing_input = cast(hidden_states, trt.float32)
        routing = self.router(routing_input, moe_router_lora_params)
        output = self.forward_experts(hidden_states, routing, finished,
                                      lora_layer_params)
        if self.use_all_reduce:
            output = self.forward_allreduce(output, reduce_fusion_params)
        return output

    def forward_experts(self, hidden_states, routing, finished,
                        lora_layer_params):

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
            fc1_act_dequant = self.fc.activation_scaling_factor.value

            scale_1 = fc1_dequant
            scale_2 = fc2_quant
            scale_3 = fc2_dequant
            scale_4 = None
            scale_5 = fc1_act_dequant

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
            scale_5 = None
        output = _moe_plugin(self.moe_config,
                             hidden_states_quant,
                             routing,
                             expert_weights_1=self.fc.weight.value,
                             expert_weights_2=self.proj.weight.value,
                             expert_bias_1=self.fc.bias,
                             expert_bias_2=self.proj.bias,
                             expert_scale_1=scale_1,
                             expert_scale_2=scale_2,
                             expert_scale_3=scale_3,
                             expert_scale_4=scale_4,
                             act_scale=scale_5,
                             finished=finished,
                             hidden_size=self.hidden_size,
                             ffn_hidden_size=self.expert_inter_size,
                             act_fn=self.hidden_act,
                             dtype=dtype_quant,
                             weight_dtype=weight_dtype_quant,
                             output_dtype=output_dtype_quant,
                             lora_params=lora_layer_params,
                             lora_max_low_rank=self.max_low_rank,
                             quant_mode=self.quant_mode,
                             tp_size=self.mapping.moe_tp_size,
                             tp_rank=self.mapping.moe_tp_rank,
                             ep_size=self.mapping.moe_ep_size,
                             ep_rank=self.mapping.moe_ep_rank)

        return output

    def forward_allreduce(
            self, output,
            reduce_fusion_params: Optional[AllReduceFusionParams]):
        if self.tp_size > 1 and self.tp_group is not None:
            output = allreduce(output,
                               self.tp_group,
                               reduce_fusion_params=reduce_fusion_params)
        return output

    def load_weights(self, moe: "MixtureOfExperts"):
        '''
        Load weights from base MOE layer
        '''
        raise NotImplementedError("Subclass shall override this")

    def to(self,
           moe_cls: Type["MixtureOfExperts"],
           quant_config=None) -> "MixtureOfExperts":
        from ..quantization.quantize import quantize
        if isinstance(self, moe_cls):
            return self

        new_moe = moe_cls(**get_init_params(self))
        # If config is not None, set quantization from config
        if quant_config is not None:
            quantize(new_moe, quant_config)

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

        tp_size = 1
        tp_group = None
        self.experts = ModuleList([
            ClsMLP(self.hidden_size, self.expert_inter_size,
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
            max_encoder_context_length=lora_layer_params.
            max_encoder_context_length,
            host_request_types=lora_layer_params.host_request_types,
            host_encoder_input_lengths=lora_layer_params.
            host_encoder_input_lengths,
            weight_index=expert_idx,
        )

    def forward_experts(self, hidden_states, routing, finished,
                        lora_layer_params):
        # TODO: https://nvbugspro.nvidia.com/bug/4781396 after this nvbug is fixed, we will remove this check.
        if lora_layer_params is not None:
            for module in ["mlp_h_to_4h", "mlp_4h_to_h", "mlp_gate"]:
                if lora_layer_params.get_runtime_params(0, module) is not None:
                    raise RuntimeError(
                        f"MoE  OOTB does not support {module} LoRA module, please enable MoE plugin"
                    )

        if self.moe_config.normalization_mode == MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE:
            topk_values, topk_indices = topk(routing, self.top_k, dim=-1)
            topk_values = softmax(topk_values, -1)
        else:
            router_probs = softmax(routing, -1)
            topk_values, topk_indices = topk(router_probs, self.top_k, dim=-1)

        hidden_size = shape(hidden_states, -1)
        # [B*sq, hidden]
        inputs_merged = hidden_states.view(concat([-1, hidden_size]))
        flat_topk_indices = topk_indices.view(
            concat([-1, shape(topk_indices, -1)]))
        flat_topk_values = topk_values.view(concat([-1,
                                                    shape(topk_values, -1)]))

        # Create output space
        zero_buffer = inputs_merged * 0.0
        output = zero_buffer

        expert_indices_stack = []
        indices_stack = []
        # When topk indices are equal to expert index, the expert will inference the tokens.
        # Bundle all indices and experts index, then do mask once.
        for i, expert in enumerate(self.experts):
            if self.mapping.has_moe_ep():
                index = i + self.experts_per_node * self.mapping.moe_ep_rank
            else:
                index = i
            expert_indices_stack.append(
                flat_topk_indices.view(concat([1, shape(flat_topk_indices)])))

            indices_stack.append(constant(int32_array(index)))

        all_expert_indices = concat(expert_indices_stack, dim=0)
        indices = expand(
            concat(indices_stack).view(concat([len(self.experts), 1, 1])),
            shape(all_expert_indices))

        # Create all experts mask
        all_expert_mask = all_expert_indices == indices

        experts_weights = cast(
            sum(flat_topk_values *
                cast(all_expert_mask, flat_topk_values.dtype),
                dim=-1,
                keepdim=True), self.dtype)

        all_expert_mask = cast(
            sum(cast(all_expert_mask, flat_topk_values.dtype),
                dim=-1,
                keepdim=True), 'bool')
        all_expert_mask = repeat_interleave(all_expert_mask, shape(output, -1),
                                            2)

        # split the mask and weights for each expert
        experts_mask = split(all_expert_mask, 1, dim=0)
        expert_weights = split(experts_weights, 1, dim=0)

        for i, expert in enumerate(self.experts):
            if self.mapping.has_moe_ep():
                index = i + self.experts_per_node * self.mapping.moe_ep_rank
            else:
                index = i
            # get mask token index
            non_zero_index = nonzero(experts_mask[i].view(
                concat([-1, hidden_size])))
            non_zero_index = non_zero_index.transpose(1, 0)
            input_for_expert = gather_nd(inputs_merged, non_zero_index, 0)
            input_for_expert = input_for_expert.view(concat([-1, hidden_size]),
                                                     zero_is_placeholder=False)

            # Expert inference
            expert_output = expert(
                input_for_expert,
                lora_layer_params=self.moe_to_expert_lora_params(
                    lora_layer_params, index))

            # scatter expert output to real position
            expert_finialized_output = zero_buffer
            expert_finialized_output = scatter_nd(
                expert_finialized_output, non_zero_index,
                expert_output.view([-1])) * expert_weights[i]

            output += expert_finialized_output

        output = output.view(shape(hidden_states))

        return output

    def load_weights(self, moe: MOE):
        for i, expert in enumerate(self.experts):
            is_gated_act = is_gated_activation(self.hidden_act)
            # Gated weight pack in expert1 weights
            # expert_weights_1
            experts_weight_1_raw = moe.fc.weight.raw_value
            fc1_weight_scale = None
            fc1_activation_scale = None
            fc2_weight_scale = None
            fc2_activation_scale = None

            if self.quant_mode.has_fp8_qdq():
                fc1_weight_scale = moe.fc.weights_scaling_factor.raw_value
                fc1_activation_scale = moe.fc.activation_scaling_factor.raw_value
                fc2_weight_scale = moe.proj.weights_scaling_factor.raw_value
                fc2_activation_scale = moe.proj.activation_scaling_factor.raw_value

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

            if self.quant_mode.has_fp8_qdq():
                expert.fc.activation_scaling_factor.value = fc1_activation_scale
                expert.fc.weights_scaling_factor.value = fc1_weight_scale[i]
                expert.proj.activation_scaling_factor.value = fc2_activation_scale
                expert.proj.weights_scaling_factor.value = fc2_weight_scale[i]
                if is_gated_act:
                    expert.gate.activation_scaling_factor.value = fc1_activation_scale
                    expert.gate.weights_scaling_factor.value = fc1_weight_scale[
                        i]

            # expert_weights_2
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


# Add SharedMoE class (shanshan)
class SharedMoE(Module):

    def __init__(self,
                 moe_config: MoeConfig,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 hidden_act: str,
                 mapping: Mapping = Mapping(),
                 bias: bool = True,
                 dtype=None,
                 **kwargs):
        super().__init__()

        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.mapping = mapping
        self.bias = bias
        self.dtype = dtype

        self.moe = MOE(hidden_size=self.hidden_size,
                       moe_config=self.moe_config,
                       mapping=self.mapping,
                       ffn_hidden_size=self.moe_config.moe_intermediate_size,
                       hidden_act=self.hidden_act,
                       dtype=self.dtype,
                       bias=False,
                       tp_group=self.mapping.tp_group,
                       tp_size=self.mapping.tp_size)
        ClsMLP = GatedMLP if is_gated_activation(self.hidden_act) else MLP
        self.shared_experts = ClsMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            hidden_act=non_gated_version(self.hidden_act),  # deepseek use SiLU
            bias=False,
            dtype=self.dtype,
            tp_group=self.mapping.tp_group,
            tp_size=self.mapping.tp_size)

    def forward(self, hidden_states):
        if self.moe_config.num_shared_experts > 0:
            return self.moe(hidden_states) + self.shared_experts(hidden_states)
        else:
            return self.moe(hidden_states)
