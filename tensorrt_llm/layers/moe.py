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
from .._utils import QuantModeWrapper, get_sm_version, int32_array
from ..functional import (AllReduceParams, SideStreamIDType, Tensor,
                          _add_plugin_info, _create_tensor, abs, allreduce,
                          cast, concat, constant, cuda_stream_sync, div, expand,
                          gather_nd, gt, is_gated_activation)
from ..functional import max as trt_max
from ..functional import (maximum, non_gated_version, nonzero, reduce_scatter,
                          repeat_interleave, scatter, scatter_nd, shape,
                          sigmoid, softmax, split, sub, sum, topk, unsqueeze,
                          where)
from ..mapping import Mapping
from ..module import Module, ModuleList
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from ..quantization import GroupwiseQuantAlgo, QuantMode
from ..quantization.functional import (get_weight_scale_interleave_factor,
                                       postprocess_weight_only,
                                       preprocess_weights_for_mixed_gemm,
                                       quantize)
from .linear import RowLinear
from .mlp import MLP, GatedMLP

activation_str_to_int_map = {
    # [WARNING] Keep the below in sync with cpp/tensorrt_llm/kernels/cutlass_kernels/include/common.h
    "gelu": 0,
    "gelu_new": 0,
    "relu": 1,
    "silu": 2,
    "swiglu": 3,
    "geglu": 4,
    "swiglu_bias": 5,
    "identity": 6,
}


class MoeGroupwiseQuantParams():

    def __init__(self,
                 group_size=-1,
                 zero=False,
                 pre_quant_scale=False,
                 use_w4a8_awq=False,
                 act_scale_1=None,
                 weight_scale_1=None,
                 weight_zero_1=None,
                 alpha_1=None,
                 act_scale_2=None,
                 weight_scale_2=None,
                 weight_zero_2=None,
                 alpha_2=None) -> None:
        self.group_size = group_size
        self.quant_algo = zero * GroupwiseQuantAlgo.ZERO + pre_quant_scale * GroupwiseQuantAlgo.PRE_QUANT_SCALE + use_w4a8_awq * GroupwiseQuantAlgo.W4A8_ALPHA

        self.quant_params = []

        if group_size == -1:
            return

        assert weight_scale_1
        assert weight_scale_2
        self.quant_params += [weight_scale_1, weight_scale_2]
        if pre_quant_scale:
            assert act_scale_1
            assert act_scale_2
            self.quant_params += [act_scale_1, act_scale_2]
        if zero:
            assert weight_zero_1
            assert weight_zero_2
            self.quant_params += [weight_zero_1, weight_zero_2]
        if use_w4a8_awq:
            assert alpha_1
            assert alpha_2
            self.quant_params += [alpha_1, alpha_2]


@dataclass
class MoeConfig:

    class ExpertScaleNormalizationMode(IntEnum):
        NONE = 0
        RENORMALIZE = 1
        SPARSE_MIXER = 2
        DEVICE_LIMITED = 3
        DEVICE_LIMITED_RENORM = 4

    num_experts: int = 0
    shared_expert_intermediate_size: int = 0

    top_k: int = 0
    normalization_mode: ExpertScaleNormalizationMode = ExpertScaleNormalizationMode.RENORMALIZE
    sparse_mixer_epsilon: float = 0.01
    tp_mode: int = 0

    device_limited_n_group: int = 0
    device_limited_topk_group: int = 0
    device_limited_routed_scaling_factor: float = 1.0

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
                hidden_states_raw,
                token_selected_experts,
                token_final_scales,
                expert_weights_1,
                expert_weights_2,
                expert_bias_1,
                expert_bias_2,
                expert_scale_1,
                expert_scale_2,
                expert_scale_3,
                expert_scale_4,
                expert_scale_5,
                expert_scale_6,
                groupwise_quant_params,
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
                ep_rank=0,
                side_stream_id=SideStreamIDType.disable):
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
    expert_scale_5 = from_parameter(expert_scale_5)
    expert_scale_6 = from_parameter(expert_scale_6)

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
    p_experts_per_token = trt.PluginField(
        "experts_per_token", np.array(moe_config.top_k, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_expert_hidden_size = trt.PluginField(
        "expert_hidden_size", np.array(hidden_size, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_expert_inter_size = trt.PluginField(
        "expert_inter_size", np.array(ffn_hidden_size, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_groupwise_quant_algo = trt.PluginField(
        "groupwise_quant_algo",
        np.array(groupwise_quant_params.quant_algo, dtype=np.int32),
        trt.PluginFieldType.INT32)
    p_group_size = trt.PluginField(
        "group_size", np.array(groupwise_quant_params.group_size,
                               dtype=np.int32), trt.PluginFieldType.INT32)
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

    if isinstance(quant_mode, QuantModeWrapper):
        # We only need to get one quant mode here for specific moe layer
        quant_mode = quant_mode[0]
    p_quant_mode = trt.PluginField("quant_mode",
                                   np.array([int(quant_mode)], dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    p_use_final_scales = trt.PluginField(
        "use_final_scales",
        np.array([int(token_final_scales is not None)], dtype=np.int32),
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

    p_force_determinism = trt.PluginField(
        "force_determinism", np.array([int(False)], dtype=np.int32),
        trt.PluginFieldType.INT32)

    p_side_stream_id = trt.PluginField("side_stream_id",
                                       np.array(side_stream_id, dtype=np.int32),
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
        p_remove_input_padding, p_num_experts, p_experts_per_token,
        p_expert_hidden_size, p_expert_inter_size, p_groupwise_quant_algo,
        p_group_size, p_activation_type, p_type_id, p_weight_type_id,
        p_output_type_id, p_quant_mode, p_use_bias, p_use_final_scales,
        p_tp_size, p_tp_rank, p_ep_size, p_ep_rank, p_force_determinism,
        p_side_stream_id, p_use_lora
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
    plugin_inputs = [hidden_states, expert_weights_1, expert_weights_2]
    plugin_inputs += [token_selected_experts]

    # Add conditional inputs

    # Final scales do a final rescale of the output of the experts
    if token_final_scales is not None:
        plugin_inputs += [token_final_scales]

    # Expert biases
    if expert_bias_1:
        assert expert_bias_2
        plugin_inputs += [expert_bias_1, expert_bias_2]

    # Add conditional inputs
    if (quant_mode.is_weight_only() and not quant_mode.has_per_group_scaling()):
        assert expert_scale_1
        assert expert_scale_2
        plugin_inputs += [expert_scale_1, expert_scale_2]
    elif quant_mode.has_fp8_qdq():
        # FP8 always has scales 1-3
        assert expert_scale_1
        assert expert_scale_2
        assert expert_scale_3
        plugin_inputs += [expert_scale_1, expert_scale_2, expert_scale_3]

        if expert_scale_4 is not None:
            assert output_dtype == trt.fp8
            plugin_inputs += [expert_scale_4]

        # Lora needs an extra parameter to be able to dequant the input back to backbone type
        if use_lora:
            assert expert_scale_5
            plugin_inputs += [expert_scale_5]
    elif quant_mode.has_per_group_scaling():
        plugin_inputs += groupwise_quant_params.quant_params
        # Lora needs an extra parameter to be able to dequant the input back to backbone type
        if use_lora:
            assert expert_scale_5
            plugin_inputs += [expert_scale_5]
    elif quant_mode.has_nvfp4():
        assert expert_scale_1
        assert expert_scale_2
        assert expert_scale_3
        assert expert_scale_4
        assert expert_scale_5
        assert expert_scale_6
        plugin_inputs += [
            expert_scale_1, expert_scale_2, expert_scale_3, expert_scale_4,
            expert_scale_5, expert_scale_6
        ]

    # Lora parameters
    if use_lora:
        # Check if lora_params is not None
        moe_h_4h_params = lora_params.get_runtime_params(0, "moe_h_to_4h")
        if moe_h_4h_params is not None:
            moe_h_4h_weight_ptrs = moe_h_4h_params.lora_weights_pointers
            moe_h_4h_lora_ranks = moe_h_4h_params.lora_ranks
            plugin_inputs += (moe_h_4h_weight_ptrs + moe_h_4h_lora_ranks)

        moe_4h_h_params = lora_params.get_runtime_params(0, "moe_4h_to_h")
        if moe_4h_h_params is not None:
            moe_4h_h_weight_ptrs = moe_4h_h_params.lora_weights_pointers
            moe_4h_h_lora_ranks = moe_4h_h_params.lora_ranks
            plugin_inputs += (moe_4h_h_weight_ptrs + moe_4h_h_lora_ranks)

        if is_gated_activation(act_fn):
            moe_gate_params = lora_params.get_runtime_params(0, "moe_gate")
            if moe_gate_params is not None:
                moe_gate_weight_ptrs = moe_gate_params.lora_weights_pointers
                moe_gate_lora_ranks = moe_gate_params.lora_ranks
                plugin_inputs += (moe_gate_weight_ptrs + moe_gate_lora_ranks)

        host_request_types = lora_params.host_request_types
        plugin_inputs += [host_request_types]

        if default_net().plugin_config.remove_input_padding:
            plugin_inputs += [lora_params.host_context_lengths]

    # A control flow tensor required to synchronize the side stream
    if side_stream_id != SideStreamIDType.disable:
        plugin_inputs += [hidden_states_raw]

    # Pass the inputs to the plugin
    plugin_inputs = [i.trt_tensor for i in plugin_inputs]
    layer = default_trtnet().add_plugin_v2(plugin_inputs, moe_plugin)
    _add_plugin_info(layer, plugin_creator, "mixture_of_experts", pfc)
    if not default_net().strongly_typed:
        for ii in range(layer.num_inputs):
            if layer.get_input(ii).dtype == str_dtype_to_trt("int8"):
                layer.get_input(ii).set_dynamic_range(-127, 127)

    # Fetch the output tensor
    output = _create_tensor(layer.get_output(0), layer)

    # If the side stream is enabled, also return the synchronization tensor for the side stream
    if side_stream_id != SideStreamIDType.disable:
        output = (output, _create_tensor(layer.get_output(1), layer))
    return output


def unpack_int32_into_int8(w_packed):
    # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
    w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
    w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                             w_packed_int4x2.shape[1],
                             w_packed_int4x2.shape[2] * 2,
                             dtype=torch.int8)
    w_unpacked[:, :, ::2] = w_packed_int4x2 % 16
    w_unpacked[:, :, 1::2] = w_packed_int4x2 // 16
    w_unpacked = w_unpacked.view(-1, 8)[:, [0, 4, 1, 5, 2, 6, 3, 7]].view(
        w_unpacked.shape)
    return w_unpacked.contiguous()


# This exists so that MOE can have the same name format as a regular MLP, just with different shaped weight tensors
class MOEWeightWrapper(Module):

    def __init__(self, in_features: int, out_features: int,
                 experts_per_node: int, quant_mode: QuantMode,
                 groupwise_quant_algo: int, group_size: int,
                 dtype: Union[str,
                              trt.DataType], weight_dtype: Union[str,
                                                                 trt.DataType],
                 has_bias: bool, wrapper_tllm_to_externel_key_dict: dict,
                 tp_size: int, tp_dim: int):
        super().__init__()
        self.quant_mode = quant_mode
        self.groupwise_quant_algo = groupwise_quant_algo
        self.group_size = group_size
        self.expert_shape = (experts_per_node, out_features, in_features)
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.has_bias = has_bias
        self.tllm_to_externel_key_dict = wrapper_tllm_to_externel_key_dict
        self.tp_size = tp_size
        self.tp_dim = 1 - tp_dim if quant_mode.has_per_group_scaling(
        ) else tp_dim
        self.is_padded = False

        if quant_mode.is_weight_only(
        ) and not quant_mode.has_per_group_scaling():
            bytes_per_col_scale = 2 if quant_mode.is_int4_weight_only() else 1
            # We use a different shape here because the quantized weights have their own layout
            self.expert_shape = (experts_per_node, in_features,
                                 out_features // bytes_per_col_scale)
            self.per_channel_scale = Parameter(shape=(experts_per_node,
                                                      out_features),
                                               dtype=dtype)
        else:
            self.register_parameter('per_channel_scale', None)

        if quant_mode.has_nvfp4():
            self.expert_shape = (experts_per_node, out_features, in_features)
            weight_dtype = trt.fp4

        if not quant_mode.has_per_group_scaling():
            self.weight = Parameter(shape=self.expert_shape,
                                    dtype=weight_dtype,
                                    prefer_managed=True)

        if has_bias:
            self.bias = Parameter(shape=(experts_per_node, out_features),
                                  dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.scaling_vector_size = 16
        if quant_mode.has_fp8_qdq():
            self.activation_scaling_factor = Parameter(shape=(1, ),
                                                       dtype=trt.float32)
            self.weights_scaling_factor = Parameter(shape=(experts_per_node, 1),
                                                    dtype=trt.float32)
        elif quant_mode.has_nvfp4():
            self.weights_block_scaling_factor_interleaved = Parameter(
                shape=(experts_per_node, out_features,
                       in_features // self.scaling_vector_size),
                dtype=trt.fp8)
            self.weights_block_scaling_factor = Parameter(
                shape=(experts_per_node, out_features,
                       in_features // self.scaling_vector_size),
                dtype=trt.fp8)
            self.activation_global_scaling_factor = Parameter(shape=(1, ),
                                                              dtype=trt.float32)
            # alpha = 1.0 / (weight_global_scale * act_global_scale)
            self.alpha = Parameter(shape=(experts_per_node, ),
                                   dtype=trt.float32)
        elif quant_mode.has_per_group_scaling():
            self.weight = Parameter(
                shape=(experts_per_node, in_features,
                       out_features // 4),  # int4 <--> fp16/bf16
                dtype=dtype)
            if groupwise_quant_algo & GroupwiseQuantAlgo.W4A8_ALPHA:
                scale_interleave_factor = get_weight_scale_interleave_factor(
                    in_features, group_size)
            else:
                scale_interleave_factor = 1
            scale_shape = (experts_per_node,
                           in_features // group_size // scale_interleave_factor,
                           out_features * scale_interleave_factor)
            self.weights_scaling_factor = Parameter(shape=scale_shape,
                                                    dtype=dtype)
            if groupwise_quant_algo & GroupwiseQuantAlgo.ZERO:
                self.zero = Parameter(shape=scale_shape, dtype=dtype)
            else:
                self.register_parameter('zero', None)
            if groupwise_quant_algo & GroupwiseQuantAlgo.PRE_QUANT_SCALE:
                self.prequant_scaling_factor = Parameter(shape=(1, in_features),
                                                         dtype=dtype)
            else:
                self.register_parameter('prequant_scaling_factor', None)
            if groupwise_quant_algo & GroupwiseQuantAlgo.W4A8_ALPHA:
                self.alpha = Parameter(shape=(experts_per_node, 1),
                                       dtype=trt.float32)
            else:
                self.register_parameter('alpha', None)
            self.tllm_to_externel_key_dict.update(
                {"weight": ["qweight", "qzeros", "scales"]})
        else:
            self.register_parameter('weights_scaling_factor', None)
            self.register_parameter('weights_block_scaling_factor', None)
            self.register_parameter('weights_block_scaling_factor_interleaved',
                                    None)
            self.register_parameter('activation_scaling_factor', None)
            self.register_parameter('activation_global_scaling_factor', None)
            self.register_parameter('alpha', None)
            self.register_parameter('zero', None)
            self.register_parameter('prequant_scaling_factor', None)

    def postprocess(self, tllm_key, weights, **kwargs):

        def stack_weights(tllm_key, weights):
            if "fc" in tllm_key:
                weights = torch.cat([
                    torch.stack(weights[:len(weights) // 2]),
                    torch.stack(weights[len(weights) // 2:])
                ],
                                    dim=-2)
            elif "proj" in tllm_key:
                weights = torch.stack(weights)
            return weights

        def postprocess_awq(tllm_key, weights):
            if not tllm_key.endswith("weight"):
                return {}
            weights = [weights[i::3] for i in range(3)]
            for idx, w in enumerate(weights):
                if "fc" in tllm_key:
                    weights[idx] = torch.cat([
                        torch.stack(w[:len(w) // 2]),
                        torch.stack(w[len(w) // 2:])
                    ],
                                             dim=-1)
                elif "proj" in tllm_key:
                    weights[idx] = torch.stack(w)
            qweight_int32, qzeros_int32, scales_fp16 = weights
            qweight = unpack_int32_into_int8(qweight_int32) - 8
            qweight -= (qweight >> 4) << 4
            qweight = qweight.view(torch.uint8)
            qweight = (qweight[:, :, 1::2] * 16 + qweight[:, :, ::2]).view(
                torch.int8)
            qweight = preprocess_weights_for_mixed_gemm(
                qweight, torch.quint4x2,
                torch.float16).view(str_dtype_to_torch(self.dtype))
            # zeros = zeros * scales
            qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)
            zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8) * scales_fp16
            zeros_x_scales_fp16 = zeros_x_scales_fp16.to(
                str_dtype_to_torch(self.dtype))

            results = {
                tllm_key: qweight,
                tllm_key.replace("weight", "weights_scaling_factor"):
                scales_fp16,
                tllm_key.replace("weight", "zero"): zeros_x_scales_fp16,
            }
            return results

        if self.quant_mode.has_per_group_scaling():
            return postprocess_awq(tllm_key, weights)
        elif tllm_key.endswith("weight"):
            if isinstance(weights, torch.Tensor):
                weights = [weights]
            else:
                if self.quant_mode.has_fp8_qdq():
                    experts_per_node = self.weights_scaling_factor.shape[0]
                    if 'fc' in tllm_key:
                        # Example weights:
                        # ['model.layers.0.block_sparse_moe.experts.0.w3.weight',
                        #  'model.layers.0.block_sparse_moe.experts.0.w3.weight_scale',
                        #  ...
                        #  'model.layers.0.block_sparse_moe.experts.7.w3.weight',
                        #  'model.layers.0.block_sparse_moe.experts.7.w3.weight_scale',
                        #  ...
                        #  'model.layers.0.block_sparse_moe.experts.0.w1.weight',
                        #  'model.layers.0.block_sparse_moe.experts.0.w1.weight_scale',
                        #  ...
                        #  'model.layers.0.block_sparse_moe.experts.7.w1.weight',
                        #  'model.layers.0.block_sparse_moe.experts.7.w1.weight_scale']
                        assert experts_per_node * 4 == len(weights)

                        def w3_weight_idx(expert_id):
                            return 2 * expert_id

                        def w3_weight_scale_idx(expert_id):
                            return 2 * expert_id + 1

                        def w1_weight_idx(expert_id):
                            return 2 * (expert_id + experts_per_node)

                        def w1_weight_scale_idx(expert_id):
                            return 2 * (expert_id + experts_per_node) + 1

                        weights_requantized = [None] * (2 * experts_per_node)
                        # Since w1/w3 share weight_scale by picking the max, we need to requantize weight
                        for i in range(experts_per_node):
                            w3_weight = weights[w3_weight_idx(i)].float()
                            w3_weight_scale = weights[w3_weight_scale_idx(
                                i)].float()
                            w1_weight = weights[w1_weight_idx(i)].float()
                            w1_weight_scale = weights[w1_weight_scale_idx(
                                i)].float()

                            max_weight_scale = max(w3_weight_scale,
                                                   w1_weight_scale)

                            weights_requantized[i] = (w3_weight *
                                                      w3_weight_scale /
                                                      max_weight_scale).to(
                                                          torch.float8_e4m3fn)
                            weights_requantized[i + experts_per_node] = (
                                w1_weight * w1_weight_scale /
                                max_weight_scale).to(torch.float8_e4m3fn)

                        weights = weights_requantized
                    else:
                        assert 'proj' in tllm_key, f"tllm_key is {tllm_key}, which does not contain fc or proj"
                        # Example weights:
                        # ['model.layers.0.block_sparse_moe.experts.0.w2.weight',
                        #  'model.layers.0.block_sparse_moe.experts.0.w2.weight_scale',
                        #  ...
                        #  'model.layers.0.block_sparse_moe.experts.7.w2.weight',
                        #  'model.layers.0.block_sparse_moe.experts.7.w2.weight_scale',
                        assert 2 * experts_per_node == len(weights)

                        def w2_weight_idx(expert_id):
                            return 2 * expert_id

                        # No need to requantize, simply skip weight_scale
                        weights = [
                            weights[w2_weight_idx(i)]
                            for i in range(experts_per_node)
                        ]

                weights = stack_weights(tllm_key, weights)

            if not self.quant_mode.has_any_quant():
                # When each rank holds single expert, weights will be a list
                if isinstance(weights, list):
                    weights = stack_weights(tllm_key, weights)
                weights = weights.to(str_dtype_to_torch(self.dtype))

        # FP8 scaling factors
        if tllm_key.endswith("activation_scaling_factor"):
            # Use max input range.
            weights = max(weights).float().reshape((1, ))

        if tllm_key.endswith("weights_scaling_factor"):
            if tllm_key.split('.')[-2] == 'fc':
                # Example weights:
                # ['model.layers.0.block_sparse_moe.experts.0.w3.weight_scale',
                #   ...
                #  'model.layers.0.block_sparse_moe.experts.7.w3.weight_scale',
                #  'model.layers.0.block_sparse_moe.experts.0.w1.weight_scale',
                #  ...
                #  'model.layers.0.block_sparse_moe.experts.7.w1.weight_scale']
                experts_per_node = self.weights_scaling_factor.shape[0]
                assert experts_per_node * 2 == len(weights)

                def w3_weight_scale_idx(expert_id):
                    return expert_id

                def w1_weight_scale_idx(expert_id):
                    return expert_id + experts_per_node

                # w1 and w3 share the weight scale by picking the max
                weights = [
                    max(weights[w3_weight_scale_idx(i)],
                        weights[w1_weight_scale_idx(i)])
                    for i in range(experts_per_node)
                ]

            weights = stack_weights(tllm_key, weights)

        # FP4 scaling factors
        if tllm_key.endswith("weights_block_scaling_factor"):
            weights = stack_weights(tllm_key, weights)
        if tllm_key.endswith("weights_block_scaling_factor_interleaved"):
            weights = stack_weights(tllm_key, weights)
            weights = torch.ops.trtllm.block_scale_interleave(
                weights.to(torch.float8_e4m3fn).view(
                    torch.uint8).cpu().contiguous()).reshape(
                        weights.shape).view(torch.float8_e4m3fn)
        if tllm_key.endswith("activation_global_scaling_factor"):
            # Use max input range.
            weights = max(weights).float().reshape((1, ))
        if tllm_key.endswith("alpha"):
            # weights are: [e0_w3_weight_scale, e0_w3_input_scale, e1_w3_weight_scale, e1_w3_input_scale
            # ..., e7_w3_weight_scale, e7_w3_input_scale, e0_w1_weight_scale, e0_w1_input_scale, ...]
            weights_global_scale = weights[::2]
            activation_global_scale = weights[1::2]
            if 'fc' in tllm_key:
                weights_global_scale = torch.stack(
                    weights_global_scale[:len(weights_global_scale) // 2])
            else:
                weights_global_scale = torch.stack(weights_global_scale)
            weights = (weights_global_scale *
                       max(activation_global_scale).float()).reshape((-1, ))

        # Weight only
        if self.quant_mode.is_weight_only():
            if "per_channel_scale" in tllm_key:
                return {}
            weights = weights.to(str_dtype_to_torch(self.dtype))
            return postprocess_weight_only(
                tllm_key, weights, torch.int8 if
                self.quant_mode.is_int8_weight_only() else torch.quint4x2, self)

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
                 use_all_reduce=True,
                 pre_quant_scale=False,
                 zero=False,
                 use_w4a8_awq=False,
                 use_int8_weight=False,
                 group_size: int = -1,
                 static_routing=False):
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
        self.zero = zero
        self.pre_quant_scale = pre_quant_scale
        self.use_w4a8_awq = use_w4a8_awq
        self.use_int8_weight = use_int8_weight
        self.group_size = group_size

        if self.use_int8_weight and self.group_size > 0:
            raise NotImplementedError("INT8-GPTQ is not implemented for MoE.")

        self.static_routing = static_routing

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

        if quant_mode.has_fp8_rowwise():
            raise ValueError(
                "MixtureOfExperts - MOE Does not support FP8 rowwise quantize")

        if quant_mode.has_fp8_qdq() and self.bias:
            # TODO  We will need to revisit this if we have a use case for it
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

        if quant_mode.has_fp8_qdq():
            self.wrapper_tllm_to_externel_key_dict.update({
                "weight": [
                    "weight", "weight_scale"
                ],  # We need weight_scale to do requantization for w1/w3 fusion
                "weights_scaling_factor":
                "weight_scale",
                "activation_scaling_factor":
                "input_scale"
            })

        if quant_mode.has_nvfp4():
            self.wrapper_tllm_to_externel_key_dict.update({
                "weights_block_scaling_factor_interleaved":
                "weight_scale",
                "weights_block_scaling_factor":
                "weight_scale",
                "activation_global_scaling_factor":
                "input_scale",
                "alpha": ["weight_scale_2", "input_scale"],
            })

        # Since output dimension is usually low (in the order of 10s), no TP at
        # all is more efficient as no allreduce required in the end.
        # Note that if we see models that have large number of experts, we may
        # need to consider add TP back here.
        # TODO: Arctic has large # experts, we may need to add TP back here.
        if not self.static_routing:
            self.router = RowLinear(
                hidden_size,
                self.num_experts,
                bias=False,
                dtype=
                "float32",  # Routing is sensitive since it conditions what experts are used
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
        groupwise_quant_algo = self.zero * GroupwiseQuantAlgo.ZERO + self.pre_quant_scale * GroupwiseQuantAlgo.PRE_QUANT_SCALE + self.use_w4a8_awq * GroupwiseQuantAlgo.W4A8_ALPHA

        self.fc = MOEWeightWrapper(self.hidden_size, fc_out_size,
                                   self.experts_per_node, self.quant_mode,
                                   groupwise_quant_algo, self.group_size,
                                   self.dtype, self.weight_dtype, self.bias,
                                   self.wrapper_tllm_to_externel_key_dict,
                                   self.mapping.moe_tp_size, 0)
        self.proj = MOEWeightWrapper(self.expert_inter_size, self.hidden_size,
                                     self.experts_per_node, self.quant_mode,
                                     groupwise_quant_algo, self.group_size,
                                     self.dtype, self.weight_dtype, self.bias,
                                     self.wrapper_tllm_to_externel_key_dict,
                                     self.mapping.moe_tp_size, 1)

    def default_routing(self, logits):
        topk_values, topk_indices = topk(softmax(cast(logits, trt.float32),
                                                 dim=-1),
                                         k=self.moe_config.top_k,
                                         dim=-1)
        return topk_indices, topk_values

    def renormalize(self, logits):
        # Get top-k experts and renormalize their scores
        token_scores, token_selected_experts = topk(cast(logits, trt.float32),
                                                    k=self.moe_config.top_k,
                                                    dim=-1)
        token_final_scales = softmax(token_scores, dim=-1)
        return token_selected_experts, token_final_scales

    def group_limited_greedy(self, logits):
        n_group = self.moe_config.device_limited_n_group
        scores = softmax(cast(logits, trt.float32), -1)
        scores_shape = [shape(scores, i) for i in range(scores.ndim())]
        group_scores = scores.view(
            concat(scores_shape[:-1] +
                   [n_group, scores_shape[-1] // n_group])).max(dim=-1)
        _, group_idx = topk(group_scores,
                            k=self.moe_config.device_limited_topk_group,
                            dim=-1)
        group_mask = scatter(group_scores * 0, -1, group_idx,
                             cast(group_idx, group_scores.dtype) * 0 + 1)
        score_mask = expand(
            unsqueeze(group_mask, -1),
            concat(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]),
        ).view(concat(scores_shape))
        scores = scores * score_mask * \
            self.moe_config.device_limited_routed_scaling_factor
        return scores

    def sparse_mixer(self, logits):
        router_logits = cast(logits, trt.float32)

        topk_values = []
        topk_indices = []

        assert self.top_k == 2, "Sparse mixer only supports top_k = 2"

        def mask_and_softmax(router_logits):
            # Get max of remaining values
            max_values = trt_max(router_logits, dim=-1, keepdim=True)

            # Calculate mask for epsilon condition
            abs_values = abs(router_logits)
            max_abs = maximum(abs_values, max_values)
            diff = sub(max_values, router_logits)
            ratio = div(diff, max_abs)

            # Apply epsilon mask
            eps_mask = gt(ratio, 2 * self.moe_config.sparse_mixer_epsilon)
            router_logits = where(eps_mask, -float('inf'), router_logits)
            curr_values, curr_indices = topk(softmax(router_logits),
                                             k=1,
                                             dim=-1)
            return curr_indices, curr_values

        curr_indices, curr_values = mask_and_softmax(router_logits)
        topk_values.append(curr_values)
        topk_indices.append(curr_indices)

        # Mask the last selected expert to -inf
        router_logits = scatter(router_logits, -1, curr_indices,
                                curr_values * 0 - float('inf'))

        curr_indices, curr_values = mask_and_softmax(router_logits)
        topk_values.append(curr_values)
        topk_indices.append(curr_indices)

        # Concatenate results
        values = concat(topk_values, dim=1)
        indices = concat(topk_indices, dim=1)

        return indices, values

    def forward(self,
                hidden_states,
                lora_layer_params=None,
                all_reduce_params: Optional[AllReduceParams] = None,
                last_local_layer_residual=None,
                side_stream_id: Optional[SideStreamIDType] = SideStreamIDType.
                disable,
                static_routing_input: Optional[Tensor] = None):
        moe_router_lora_params = None
        if lora_layer_params is not None:
            moe_router_lora_params = lora_layer_params.get_runtime_params(
                0, "moe_router")

        if not self.static_routing:
            routing_input = cast(hidden_states, trt.float32)
            routing = self.router(routing_input, moe_router_lora_params)
        else:
            routing = None

        # token_selected_experts is shape (num_tokens, experts_per_token).
        #     It is a list of selected expert indices for each token
        # token_final_scales is shape (num_tokens, experts_per_token). May be None
        #     It contains a final scaling/weighting factor applied to the output of each selected expert before summing the results
        if self.static_routing:
            token_selected_experts = static_routing_input
            token_final_scales = None
        elif self.moe_config.normalization_mode == MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED:
            token_final_scales, token_selected_experts = topk(
                self.group_limited_greedy(routing),
                k=self.moe_config.top_k,
                dim=-1)
        elif self.moe_config.normalization_mode == MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED_RENORM:
            token_final_scales, token_selected_experts = topk(
                self.group_limited_greedy(routing),
                k=self.moe_config.top_k,
                dim=-1)
            token_final_scales /= sum(token_final_scales, dim=-1, keepdim=True)
        elif self.moe_config.normalization_mode == MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE:
            token_selected_experts, token_final_scales = self.renormalize(
                routing)
        elif self.moe_config.normalization_mode == MoeConfig.ExpertScaleNormalizationMode.SPARSE_MIXER:
            token_selected_experts, token_final_scales = self.sparse_mixer(
                routing)
        else:
            token_selected_experts, token_final_scales = self.default_routing(
                routing)

        output = self.forward_experts(hidden_states, token_selected_experts,
                                      token_final_scales, lora_layer_params,
                                      side_stream_id)
        if side_stream_id != SideStreamIDType.disable:
            output, side_stream_sync_tensor = output
        if self.use_all_reduce:
            output = self.forward_allreduce(output, all_reduce_params,
                                            last_local_layer_residual)
        if side_stream_id != SideStreamIDType.disable:
            # All tensors that the side channel receives as input must be synced
            # on the main stream, to prevent their memory from being released or
            # reused by the main stream before the side stream has finished.
            tensors_to_sync = (side_stream_sync_tensor, hidden_states,
                               token_selected_experts, token_final_scales,
                               lora_layer_params)
            tensors_to_sync = tuple(t for t in tensors_to_sync if t is not None)
            output = (output, tensors_to_sync)
        return output

    def forward_experts(self, hidden_states, token_selected_experts,
                        token_final_scales, lora_layer_params, side_stream_id):

        groupwise_quant_params = MoeGroupwiseQuantParams()
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
            scale_6 = None

            output_dtype_quant = self.dtype

            if output_dtype_quant == trt.fp8 and scale_4 is None:
                raise RuntimeError(
                    "Cannot output FP8 value without knowing quantization parameter"
                )
        elif self.quant_mode.has_nvfp4():
            # We pass through the weights unchanged, the quantization is done in the plugin
            hidden_states_quant = hidden_states
            dtype_quant = trt.fp4
            weight_dtype_quant = trt.fp4
            output_dtype_quant = self.dtype

            scale_1 = div(1.0, self.fc.activation_global_scaling_factor.value)
            scale_2 = self.fc.weights_block_scaling_factor_interleaved
            scale_3 = self.fc.alpha
            scale_4 = div(1.0, self.proj.activation_global_scaling_factor.value)
            scale_5 = self.proj.weights_block_scaling_factor_interleaved
            scale_6 = self.proj.alpha
        elif self.quant_mode.has_per_group_scaling():
            hidden_states_quant = hidden_states
            dtype_quant = trt.fp8 if self.use_w4a8_awq else self.dtype
            weight_dtype_quant = self.weight_dtype
            output_dtype_quant = self.dtype

            scale_1 = None
            scale_2 = None
            scale_3 = None
            scale_4 = None
            scale_5 = None
            scale_6 = None
            pre_quant_scale_1 = self.fc.prequant_scaling_factor.value if self.fc.prequant_scaling_factor else None
            zero_1 = self.fc.zero.value if self.fc.zero else None
            alpha_1 = self.fc.alpha.value if self.fc.alpha else None
            pre_quant_scale_2 = self.proj.prequant_scaling_factor.value if self.proj.prequant_scaling_factor else None
            zero_2 = self.proj.zero.value if self.proj.zero else None
            alpha_2 = self.proj.alpha.value if self.proj.alpha else None
            groupwise_quant_params = MoeGroupwiseQuantParams(
                self.group_size,
                self.zero,
                self.pre_quant_scale,
                self.use_w4a8_awq,
                pre_quant_scale_1,
                self.fc.weights_scaling_factor.value,
                zero_1,
                alpha_1,
                pre_quant_scale_2,
                self.proj.weights_scaling_factor.value,
                zero_2,
                alpha_2,
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
            scale_6 = None
        output = _moe_plugin(self.moe_config,
                             hidden_states_quant,
                             hidden_states,
                             token_selected_experts,
                             token_final_scales,
                             expert_weights_1=self.fc.weight.value,
                             expert_weights_2=self.proj.weight.value,
                             expert_bias_1=self.fc.bias,
                             expert_bias_2=self.proj.bias,
                             expert_scale_1=scale_1,
                             expert_scale_2=scale_2,
                             expert_scale_3=scale_3,
                             expert_scale_4=scale_4,
                             expert_scale_5=scale_5,
                             expert_scale_6=scale_6,
                             groupwise_quant_params=groupwise_quant_params,
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
                             ep_rank=self.mapping.moe_ep_rank,
                             side_stream_id=side_stream_id)

        return output

    def forward_allreduce(self,
                          output,
                          all_reduce_params: Optional[AllReduceParams],
                          last_local_layer_residual=None):

        if last_local_layer_residual is not None:
            if self.mapping.tp_rank == 0:
                output = output + last_local_layer_residual
            else:
                # we need to add this line here to minimize the numerical difference
                output = output + 0
            # reshape to (-1)
            output = output.view(concat([-1]))
            if self.tp_size > 1 and self.tp_group is not None:
                output = reduce_scatter(output, self.tp_group)
            # reshape to (-1, hidden_size // tp_size)
            output = output.view(concat([-1, self.hidden_size // self.tp_size]))
            return output
        if self.tp_size > 1 and self.tp_group is not None:
            output = allreduce(output,
                               self.tp_group,
                               all_reduce_params=all_reduce_params)
        return output

    def load_weights(self, moe: "MixtureOfExperts"):
        '''
        Load weights from base MOE layer
        '''
        raise NotImplementedError("Subclass shall override this")

    def to(self,
           moe_cls: Type["MixtureOfExperts"],
           quant_config=None) -> "MixtureOfExperts":

        if isinstance(moe_cls, MoeOOTB):
            if self.moe_config.normalization_mode in [
                    MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED,
                    MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED_RENORM
            ]:
                raise ValueError(
                    'MoeOOTB doesn\'t support group_limited_greedy yet.')
        from ..quantization.quantize import quantize
        if isinstance(self, moe_cls):
            return self

        new_moe = moe_cls(**get_init_params(self))
        # If config is not None, set quantization from config
        if quant_config is not None:
            quantize(new_moe, quant_config)

        new_moe.load_weights(self)
        if not self.static_routing:
            new_moe.router = self.router
        return new_moe


MOE = MixtureOfExperts


# TODO: Support `group_limited_greedy` in MoeOOTB.
class MoeOOTB(MOE):

    def init_experts(self):
        if self.quant_mode.is_weight_only():
            raise ValueError(
                f"OOTB MOE does not support weight only quantization now, current quant mode: {self.quant_mode}"
            )

        if get_sm_version() >= 100:
            raise RuntimeError(
                "MoeOOTB does not support SM version >= 100, please use SM version < 100"
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

    def forward_experts(self, hidden_states, token_selected_experts,
                        token_final_scales, lora_layer_params, side_stream_id):
        assert side_stream_id == SideStreamIDType.disable, "MoeOOTB does not support using side stream"
        # TODO: https://nvbugspro.nvidia.com/bug/4781396 after this nvbug is fixed, we will remove this check.
        if lora_layer_params is not None:
            for module in ["mlp_h_to_4h", "mlp_4h_to_h", "mlp_gate"]:
                if lora_layer_params.get_runtime_params(0, module) is not None:
                    raise RuntimeError(
                        f"MoE  OOTB does not support {module} LoRA module, please enable MoE plugin"
                    )

        topk_indices = token_selected_experts
        topk_values = token_final_scales

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

            if self.quant_mode.has_nvfp4():
                expert.fc.activation_global_scaling_factor.value = moe.fc.activation_global_scaling_factor.raw_value
                expert.fc.weights_block_scaling_factor.value = moe.fc.weights_block_scaling_factor.raw_value[
                    i, -self.expert_inter_size:, :]
                expert.fc.weights_block_scaling_factor_interleaved.value = moe.fc.weights_block_scaling_factor_interleaved.raw_value[
                    i, -self.expert_inter_size:, :]
                expert.fc.alpha.value = np.array(moe.fc.alpha.raw_value[i])
                if is_gated_act:
                    expert.gate.activation_global_scaling_factor.value = moe.fc.activation_global_scaling_factor.raw_value
                    expert.gate.weights_block_scaling_factor.value = moe.fc.weights_block_scaling_factor.raw_value[
                        i, :-self.expert_inter_size, :]
                    expert.gate.weights_block_scaling_factor_interleaved.value = moe.fc.weights_block_scaling_factor_interleaved.raw_value[
                        i, :-self.expert_inter_size, :]
                    expert.gate.alpha.value = np.array(
                        moe.fc.alpha.raw_value[i])

                expert.proj.activation_global_scaling_factor.value = moe.proj.activation_global_scaling_factor.raw_value
                expert.proj.weights_block_scaling_factor.value = moe.proj.weights_block_scaling_factor.raw_value[
                    i]
                expert.proj.weights_block_scaling_factor_interleaved.value = moe.proj.weights_block_scaling_factor_interleaved.raw_value[
                    i]
                expert.proj.alpha.value = np.array(moe.proj.alpha.raw_value[i])

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


# Add SharedMoE class
class SharedMoE(MOE):

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
                 use_shared_gate: bool = False,
                 use_side_stream: bool = False):
        super().__init__(
            moe_config=moe_config,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            mapping=mapping,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=quant_mode,
            use_all_reduce=False,
        )
        self.shared_expert = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=moe_config.shared_expert_intermediate_size,
            hidden_act=hidden_act,
            bias=False,
            dtype=self.dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=self.quant_mode,
            is_expert=True,
        )
        self.use_shared_gate = use_shared_gate
        if use_shared_gate:
            self.shared_expert_gate = RowLinear(
                hidden_size,
                1,
                bias=False,
                dtype=dtype,
                tp_group=None,
                tp_size=1,
            )
        else:
            self.shared_expert_gate = None
        self.use_side_stream = use_side_stream

    def forward(self, hidden_states, lora_layer_params=None):
        side_stream_id = SideStreamIDType.moe if self.use_side_stream else SideStreamIDType.disable
        if self.use_side_stream:
            routed_output, tensors_to_sync = super().forward(
                hidden_states,
                lora_layer_params=lora_layer_params,
                side_stream_id=side_stream_id,
            )
        else:
            routed_output = super().forward(
                hidden_states,
                lora_layer_params=lora_layer_params,
            )
        shared_output = self.shared_expert(
            hidden_states,
            lora_layer_params=lora_layer_params,
        )
        if self.shared_expert_gate is not None:
            gate_lora_params = None
            if lora_layer_params is not None:
                gate_lora_params = lora_layer_params.get_runtime_params(
                    0, "mlp_router")
            shared_output = sigmoid(
                self.shared_expert_gate(hidden_states,
                                        gate_lora_params)) * shared_output
        if self.use_side_stream:
            # tensors_to_sync are included in the inputs to ensure that their
            # memory space is not reused for other tensors on the main stream
            # until the side stream has finished
            shared_output = cuda_stream_sync([shared_output, *tensors_to_sync],
                                             side_stream_id)
        hidden_states = routed_output + shared_output
        if self.tp_size > 1 and self.tp_group is not None:
            hidden_states = allreduce(hidden_states, self.tp_group)
        return hidden_states
