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
from typing import Optional, Tuple, Union

import numpy as np
import tensorrt as trt

from .._common import default_net, default_trtnet
from .._utils import str_dtype_to_np, str_dtype_to_trt
from ..functional import (Tensor, _add_plugin_info, _create_tensor, cast, clip,
                          constant, matmul, repeat_interleave, round)
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE


def smooth_quant_gemm(input: Tensor, weights: Tensor, scales_a: Tensor,
                      scales_b: Tensor, per_token_scaling: bool,
                      per_channel_scaling: bool) -> Tensor:
    if not default_net().plugin_config.smooth_quant_gemm_plugin:
        raise TypeError("Smooth Quant GEMM is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'SmoothQuantGemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        per_channel_scaling = 1 if per_channel_scaling else 0
        per_channel_scaling = trt.PluginField(
            "has_per_channel_scaling",
            np.array(per_channel_scaling, dtype=np.int32),
            trt.PluginFieldType.INT32)

        per_token_scaling = 1 if per_token_scaling else 0
        per_token_scaling = trt.PluginField(
            "has_per_token_scaling", np.array(per_token_scaling,
                                              dtype=np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.smooth_quant_gemm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection(
            [per_channel_scaling, per_token_scaling, pf_type])
        gemm_plug = plg_creator.create_plugin("sq_gemm", pfc)
        plug_inputs = [
            input.trt_tensor, weights.trt_tensor, scales_a.trt_tensor,
            scales_b.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
        _add_plugin_info(layer, plg_creator, "sq_gemm", pfc)
        layer.get_input(0).set_dynamic_range(-127, 127)
        layer.get_input(1).set_dynamic_range(-127, 127)
        return _create_tensor(layer.get_output(0), layer)


def weight_only_quant_matmul(input: Tensor,
                             weights: Tensor,
                             scales: Tensor,
                             weightTypeId: int,
                             dtype: str = 'float16') -> Tensor:

    if not default_net().plugin_config.weight_only_quant_matmul_plugin:
        if weights.dtype != trt.int8:
            # Q->DQ
            weights = quantize(weights, scales, dtype='int8', axis=1)
            weights = dequantize(weights, scales, 1, input.dtype)
        else:
            weights = dequantize(weights, scales, 1, input.dtype)

        res = matmul(input, weights)
        return cast(res, dtype)
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'WeightOnlyQuantMatmul', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        weight_type_id = trt.PluginField("weight_type_id",
                                         np.array(weightTypeId, dtype=np.int32),
                                         trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.weight_only_quant_matmul_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([pf_type, weight_type_id])
        matmul_plug = plg_creator.create_plugin("woq_matmul", pfc)
        plug_inputs = [input.trt_tensor, weights.trt_tensor, scales.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, matmul_plug)
        _add_plugin_info(layer, plg_creator, "woq_matmul", pfc)
        layer.get_input(1).set_dynamic_range(-127, 127)
        return _create_tensor(layer.get_output(0), layer)


def weight_only_groupwise_quant_matmul(input: Tensor,
                                       pre_quant_scale: Tensor,
                                       weights: Tensor,
                                       scales: Tensor,
                                       zeros: Tensor,
                                       biases: Tensor,
                                       quant_algo: int,
                                       group_size: int,
                                       dtype: str = 'float16') -> Tensor:

    if not default_net(
    ).plugin_config.weight_only_groupwise_quant_matmul_plugin:
        scales = repeat_interleave(scales, group_size, 0)
        weights = quantize(weights, scales, dtype='int8', axis=1)
        weights = dequantize(weights, scales, 1, input.dtype)

        if quant_algo & 4:
            # pre quant
            input = input * pre_quant_scale
        elif quant_algo & 2:
            # zero
            zeros = repeat_interleave(zeros, group_size, 0)
            weights += zeros
        res = matmul(input, weights)
        if quant_algo & 1:
            # bias
            res += biases

        return cast(res, dtype)
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'WeightOnlyGroupwiseQuantMatmul', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        quant_algo_ = trt.PluginField("quant_algo",
                                      np.array(quant_algo, dtype=np.int32),
                                      trt.PluginFieldType.INT32)
        group_size_ = trt.PluginField("group_size",
                                      np.array(group_size, dtype=np.int32),
                                      trt.PluginFieldType.INT32)

        p_dtype = default_net(
        ).plugin_config.weight_only_groupwise_quant_matmul_plugin
        pf_type_ = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([pf_type_, quant_algo_, group_size_])

        matmul_plug = plg_creator.create_plugin("woq_groupwise_matmul", pfc)

        # quant_algo = pre_quant_scale * 4 + zero * 2 + bias
        plug_inputs = [input.trt_tensor]

        # Flags for indicating whether the corresponding inputs are applied in quant_algo
        # quant_algo = pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
        # Here pre_quant_scale, zero and bias are boolean type
        BIAS = 1
        ZERO = 2
        PRE_QUANT_SCALE = 4

        if quant_algo & PRE_QUANT_SCALE:
            plug_inputs += [pre_quant_scale.trt_tensor]

        plug_inputs += [weights.trt_tensor, scales.trt_tensor]

        if quant_algo & ZERO:
            plug_inputs += [zeros.trt_tensor]
        if quant_algo & BIAS:
            plug_inputs += [biases.trt_tensor]

        layer = default_trtnet().add_plugin_v2(plug_inputs, matmul_plug)
        _add_plugin_info(layer, plg_creator, "woq_groupwise_matmul", pfc)
        if quant_algo & PRE_QUANT_SCALE:
            layer.get_input(2).set_dynamic_range(-127, 127)
        else:
            layer.get_input(1).set_dynamic_range(-127, 127)

        return _create_tensor(layer.get_output(0), layer)


def smooth_quant_layer_norm(input: Tensor,
                            normalized_shape: Union[int, Tuple[int]],
                            weight: Optional[Tensor] = None,
                            bias: Optional[Tensor] = None,
                            scale: Optional[Tensor] = None,
                            eps: float = 1e-05,
                            use_diff_of_squares: bool = True,
                            dynamic_act_scaling: bool = False) -> Tensor:
    if not default_net().plugin_config.layernorm_quantization_plugin:
        raise TypeError("Smooth Quant Layer Norm is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'LayernormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)
        use_diff_of_squares = trt.PluginField(
            "use_diff_of_squares",
            np.array([int(use_diff_of_squares)], dtype=np.int32),
            trt.PluginFieldType.INT32)

        dyn_act_scaling = trt.PluginField(
            "dyn_act_scaling", np.array([int(dynamic_act_scaling)], np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.layernorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection(
            [eps, use_diff_of_squares, dyn_act_scaling, pf_type])
        layernorm_plug = plg_creator.create_plugin("layernorm_quantized", pfc)
        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if bias is None:
            bias = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        plug_inputs = [
            input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
            scale.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, layernorm_plug)
        layer.get_output(0).set_dynamic_range(-127, 127)
        _add_plugin_info(layer, plg_creator, "layernorm_quantized", pfc)
        if not dynamic_act_scaling:
            return _create_tensor(layer.get_output(0), layer)

        return _create_tensor(layer.get_output(0),
                              layer), _create_tensor(layer.get_output(1), layer)


def smooth_quant_rms_norm(input: Tensor,
                          normalized_shape: Union[int, Tuple[int]],
                          weight: Optional[Tensor] = None,
                          bias: Optional[Tensor] = None,
                          scale: Optional[Tensor] = None,
                          eps: float = 1e-05,
                          dynamic_act_scaling: bool = False) -> Tensor:
    if not default_net().plugin_config.rmsnorm_quantization_plugin:
        raise TypeError("Smooth Quant Rms Norm is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'RmsnormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)

        dyn_act_scaling = trt.PluginField(
            "dyn_act_scaling", np.array([int(dynamic_act_scaling)], np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.rmsnorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([eps, dyn_act_scaling, pf_type])
        rmsnorm_plug = plg_creator.create_plugin("rmsnorm_quantized", pfc)
        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if bias is None:
            bias = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        plug_inputs = [
            input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
            scale.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, rmsnorm_plug)
        layer.get_output(0).set_dynamic_range(-127, 127)
        _add_plugin_info(layer, plg_creator, "rmsnorm_quantized", pfc)
        if not dynamic_act_scaling:
            return _create_tensor(layer.get_output(0), layer)

        return _create_tensor(layer.get_output(0),
                              layer), _create_tensor(layer.get_output(1), layer)


def quantize(input: Tensor,
             scale_factor: Tensor,
             dtype: str,
             axis: int = -1) -> Tensor:
    layer = default_trtnet().add_quantize(input.trt_tensor,
                                          scale_factor.trt_tensor,
                                          str_dtype_to_trt(dtype))
    layer.axis = axis

    output = _create_tensor(layer.get_output(0), layer)

    return output


def dequantize(input: Tensor,
               scale_factor: Tensor,
               axis: int = -1,
               output_type: Union[str, trt.DataType] = 'float16') -> Tensor:

    if isinstance(output_type, str):
        output_type = str_dtype_to_trt(output_type)

    layer = default_trtnet().add_dequantize(input.trt_tensor,
                                            scale_factor.trt_tensor,
                                            output_type)
    layer.axis = axis

    if not default_net().strongly_typed:
        layer.precision = input.dtype

    output = _create_tensor(layer.get_output(0), layer)

    return output


def quantize_per_token(x: Tensor) -> Tuple[Tensor]:
    if not default_net().plugin_config.quantize_per_token_plugin:
        x = cast(x, 'float32')
        xmax = x.abs().max(-1, keepdim=True)
        scale = xmax / 127.0
        out = x * 127.0 / xmax
        out = round(out)
        out = clip(out, -128, 127)
        quantized_out = cast(out, 'int8')
        return quantized_out, scale
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'QuantizePerToken', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        pfc = trt.PluginFieldCollection([])
        quantize_plug = plg_creator.create_plugin("quantize_per_token_plugin",
                                                  pfc)

        plug_inputs = [x.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
        layer.get_output(0).set_dynamic_range(-127, 127)
        _add_plugin_info(layer, plg_creator, "quantize_per_token_plugin", pfc)

        quantized = _create_tensor(layer.get_output(0), layer)
        scales = _create_tensor(layer.get_output(1), layer)

        return quantized, scales


def quantize_tensor(x, scale):
    if not default_net().plugin_config.quantize_tensor_plugin:
        scaled = x * scale
        rounded = round(scaled)
        clipped = clip(rounded, -128, 127)
        quantized = cast(clipped, 'int8')
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'QuantizeTensor', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        pfc = trt.PluginFieldCollection([])
        quantize_plug = plg_creator.create_plugin("quantize_tensor_plugin", pfc)

        plug_inputs = [x.trt_tensor, scale.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
        layer.get_output(0).set_dynamic_range(-127, 127)
        _add_plugin_info(layer, plg_creator, "quantize_tensor_plugin", pfc)

        quantized = _create_tensor(layer.get_output(0), layer)
    return quantized
