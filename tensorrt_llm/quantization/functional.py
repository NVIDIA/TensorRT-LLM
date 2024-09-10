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
import torch
import torch.nn.functional as F

from .._common import default_net, default_trtnet
from .._utils import str_dtype_to_np, str_dtype_to_trt
from ..functional import (Tensor, _add_plugin_info, _create_tensor, cast, clip,
                          constant, matmul, repeat_interleave, round)
from ..layers.linear import ColumnLinear
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from .mode import QuantMode


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
        if not default_net().strongly_typed:
            layer.get_input(0).set_dynamic_range(-127, 127)
            layer.get_input(1).set_dynamic_range(-127, 127)
        return _create_tensor(layer.get_output(0), layer)


def fp8_rowwise_gemm(input: Tensor, weights: Tensor, scales_a: Tensor,
                     scales_b: Tensor, per_token_scaling: bool,
                     per_channel_scaling: bool) -> Tensor:
    if not default_net().plugin_config.fp8_rowwise_gemm_plugin:
        raise TypeError("Fp8 Rowwise GEMM is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'Fp8RowwiseGemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
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

        p_dtype = default_net().plugin_config.fp8_rowwise_gemm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection(
            [per_channel_scaling, per_token_scaling, pf_type])
        gemm_plug = plg_creator.create_plugin("fp8_rowwise_gemm", pfc)
        plug_inputs = [
            input.trt_tensor, weights.trt_tensor, scales_a.trt_tensor,
            scales_b.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
        _add_plugin_info(layer, plg_creator, "fp8_rowwise_gemm", pfc)
        if not default_net().strongly_typed:
            layer.get_input(0).set_dynamic_range(-448, 448)
            layer.get_input(1).set_dynamic_range(-448, 448)
        return _create_tensor(layer.get_output(0), layer)


def weight_only_quant_matmul(input: Tensor,
                             weights: Tensor,
                             scales: Tensor,
                             weightTypeId: int,
                             dtype: str = 'float16',
                             transa: bool = False,
                             transb: bool = False) -> Tensor:

    if not default_net(
    ).plugin_config.weight_only_quant_matmul_plugin or transa or transb:
        scale_axis = 0 if transb else 1
        if weights.dtype != trt.int8:
            # Q->DQ
            weights = quantize(weights, scales, dtype='int8', axis=1)
            weights = dequantize(weights, scales, scale_axis, input.dtype)
        else:
            weights = dequantize(weights, scales, scale_axis, input.dtype)

        res = matmul(input, weights, transa=transa, transb=transb)
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
        if not default_net().strongly_typed:
            layer.get_input(1).set_dynamic_range(-127, 127)
        return _create_tensor(layer.get_output(0), layer)


def weight_only_groupwise_quant_matmul(input: Tensor,
                                       pre_quant_scale: Tensor,
                                       weights: Tensor,
                                       scales: Tensor,
                                       zeros: Tensor,
                                       biases: Tensor,
                                       alpha: Tensor,
                                       quant_algo: int,
                                       group_size: int,
                                       dtype: str = 'float16') -> Tensor:

    if not default_net(
    ).plugin_config.weight_only_groupwise_quant_matmul_plugin:
        scales = repeat_interleave(scales, group_size, 0)
        weights = quantize(weights, scales, dtype='int8', axis=1)
        weights = dequantize(weights, scales, 1, input.dtype)

        if quant_algo & 8:
            # fp8_alpha
            input = input * alpha
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

        # quant_algo = fp8_alpha * 8 + pre_quant_scale * 4 + zero * 2 + bias
        plug_inputs = [input.trt_tensor]

        # Flags for indicating whether the corresponding inputs are applied in quant_algo
        # quant_algo = fp8_alpha * FP8_ALPHA + pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
        # Here pre_quant_scale, zero and bias are boolean type
        BIAS = 1
        ZERO = 2
        PRE_QUANT_SCALE = 4
        FP8_ALPHA = 8

        if quant_algo & PRE_QUANT_SCALE:
            plug_inputs += [pre_quant_scale.trt_tensor]

        plug_inputs += [weights.trt_tensor, scales.trt_tensor]

        if quant_algo & ZERO:
            plug_inputs += [zeros.trt_tensor]
        if quant_algo & BIAS:
            plug_inputs += [biases.trt_tensor]
        if quant_algo & FP8_ALPHA:
            plug_inputs += [alpha.trt_tensor]

        layer = default_trtnet().add_plugin_v2(plug_inputs, matmul_plug)
        _add_plugin_info(layer, plg_creator, "woq_groupwise_matmul", pfc)

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

        # LayerNorm plugin only supports float32 scale
        scale = cast(scale, "float32")
        plug_inputs = [
            input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
            scale.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, layernorm_plug)
        if not default_net().strongly_typed:
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
                          clamp_val: Optional[Tensor] = None,
                          eps: float = 1e-05,
                          dynamic_act_scaling: bool = False) -> Tensor:
    if not default_net().plugin_config.rmsnorm_quantization_plugin:
        raise TypeError("Smooth Quant Rms Norm is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'RmsnormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        output_type = trt.PluginField("out_type_id",
                                      np.array([int(trt.int8)], np.int32),
                                      trt.PluginFieldType.INT32)
        quant_mode = trt.PluginField(
            "quant_mode",
            np.array([int(QuantMode.use_smooth_quant(per_token=True))],
                     np.int32), trt.PluginFieldType.INT32)
        clamp_enabled = trt.PluginField(
            "clamp_enabled", np.array([clamp_val is not None], np.int32),
            trt.PluginFieldType.INT32)

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)

        dyn_act_scaling = trt.PluginField(
            "dyn_act_scaling", np.array([int(dynamic_act_scaling)], np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.rmsnorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([
            eps, dyn_act_scaling, clamp_enabled, quant_mode, pf_type,
            output_type
        ])
        rmsnorm_plug = plg_creator.create_plugin("rmsnorm_quantized", pfc)
        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if bias is None:
            bias = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))

        # RMS Norm Plugin only supports float32 scale
        scale = cast(scale, "float32")
        plug_inputs = [
            input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
            scale.trt_tensor
        ]
        if clamp_val:
            plug_inputs += [clamp_val.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, rmsnorm_plug)
        if not default_net().strongly_typed:
            layer.get_output(0).set_dynamic_range(-127, 127)
        _add_plugin_info(layer, plg_creator, "rmsnorm_quantized", pfc)
        if not dynamic_act_scaling:
            return _create_tensor(layer.get_output(0), layer)

        return _create_tensor(layer.get_output(0),
                              layer), _create_tensor(layer.get_output(1), layer)


def fp8_rowwise_rms_norm(input: Tensor,
                         normalized_shape: Union[int, Tuple[int]],
                         weight: Optional[Tensor] = None,
                         bias: Optional[Tensor] = None,
                         scale: Optional[Tensor] = None,
                         clamp_val: Optional[Tensor] = None,
                         eps: float = 1e-05,
                         dynamic_act_scaling: bool = True) -> Tensor:
    if not default_net().plugin_config.rmsnorm_quantization_plugin:
        raise TypeError("Fp8 Rowwise Rms Norm is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'RmsnormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        output_type = trt.PluginField("out_type_id",
                                      np.array([int(trt.fp8)], np.int32),
                                      trt.PluginFieldType.INT32)
        quant_mode = trt.PluginField(
            "quant_mode",
            np.array([int(QuantMode.from_description(use_fp8_rowwise=True))],
                     np.int32), trt.PluginFieldType.INT32)
        clamp_enabled = trt.PluginField(
            "clamp_enabled", np.array([clamp_val is not None], np.int32),
            trt.PluginFieldType.INT32)

        eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                              trt.PluginFieldType.FLOAT32)

        dyn_act_scaling = trt.PluginField(
            "dyn_act_scaling", np.array([int(dynamic_act_scaling)], np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.rmsnorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([
            eps, dyn_act_scaling, clamp_enabled, quant_mode, pf_type,
            output_type
        ])
        rmsnorm_plug = plg_creator.create_plugin("rmsnorm_quantized", pfc)
        normalized_shape = [normalized_shape] if isinstance(
            normalized_shape, int) else normalized_shape
        if weight is None:
            weight = constant(
                np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if bias is None:
            bias = constant(
                np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
        if scale is None:
            scale = constant(np.ones((1, ), dtype=str_dtype_to_np(p_dtype)))

        # RMS Norm Plugin only supports float32 scale
        scale = cast(scale, "float32")
        plug_inputs = [
            input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
            scale.trt_tensor
        ]
        if clamp_val:
            plug_inputs += [clamp_val.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, rmsnorm_plug)
        if not default_net().strongly_typed:
            layer.get_output(0).set_dynamic_range(-448, 448)
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


def quantize_per_token(x: Tensor,
                       clamp_val: Optional[Tensor] = None) -> Tuple[Tensor]:
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

        output_type = trt.PluginField("type_id",
                                      np.array([int(trt.int8)], np.int32),
                                      trt.PluginFieldType.INT32)
        quant_mode = trt.PluginField(
            "quant_mode",
            np.array([int(QuantMode.use_smooth_quant(per_token=True))],
                     np.int32), trt.PluginFieldType.INT32)
        clamp_enabled = trt.PluginField(
            "clamp_enabled", np.array([clamp_val is not None], np.int8),
            trt.PluginFieldType.INT8)
        pfc = trt.PluginFieldCollection(
            [output_type, quant_mode, clamp_enabled])
        quantize_plug = plg_creator.create_plugin("quantize_per_token_plugin",
                                                  pfc)

        plug_inputs = [x.trt_tensor]
        if clamp_val:
            plug_inputs += [clamp_val.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
        if not default_net().strongly_typed:
            layer.get_output(0).set_dynamic_range(-127, 127)
        _add_plugin_info(layer, plg_creator, "quantize_per_token_plugin", pfc)

        quantized = _create_tensor(layer.get_output(0), layer)
        scales = _create_tensor(layer.get_output(1), layer)

        return quantized, scales


def quantize_fp8_per_token(x: Tensor,
                           clamp_val: Optional[Tensor] = None) -> Tuple[Tensor]:
    if not default_net().plugin_config.quantize_per_token_plugin:
        x = cast(x, 'float32')
        xmax = x.abs().max(-1, keepdim=True)
        scale = xmax / 448.0
        out = x * 448.0 / xmax
        out = round(out)
        out = clip(out, -448, 448)
        quantized_out = cast(out, 'fp8')
        return quantized_out, scale
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'QuantizePerToken', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        output_type = trt.PluginField("type_id",
                                      np.array([int(trt.fp8)], np.int32),
                                      trt.PluginFieldType.INT32)
        quant_mode = trt.PluginField(
            "quant_mode",
            np.array([int(QuantMode.from_description(use_fp8_rowwise=True))],
                     np.int32), trt.PluginFieldType.INT32)
        clamp_enabled = trt.PluginField(
            "clamp_enabled", np.array([clamp_val is not None], np.int8),
            trt.PluginFieldType.INT8)
        pfc = trt.PluginFieldCollection(
            [output_type, quant_mode, clamp_enabled])
        quantize_plug = plg_creator.create_plugin("quantize_per_token_plugin",
                                                  pfc)

        plug_inputs = [x.trt_tensor]
        if clamp_val:
            plug_inputs += [clamp_val.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
        if not default_net().strongly_typed:
            layer.get_output(0).set_dynamic_range(-448, 448)
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
        if not default_net().strongly_typed:
            layer.get_output(0).set_dynamic_range(-127, 127)
        _add_plugin_info(layer, plg_creator, "quantize_tensor_plugin", pfc)

        quantized = _create_tensor(layer.get_output(0), layer)
    return quantized


def symmetric_quantize_last_axis_of_batched_matrix(weight, quant_mode):
    amax = weight.abs().max(dim=0)[0].to(weight.dtype)
    if quant_mode == torch.int8:
        scale = amax / 128.
        qweight = torch.clamp((weight / scale).round(), -128, 127).char()
        qweight = qweight.T.reshape(weight.shape)
    else:
        scale = amax / 8.
        qweight = torch.clamp((weight / scale).round(), -8, 7).char()
        qweight[qweight < 0] += 16
        qweight = qweight.T.view(torch.uint8)
        qweight = (qweight[:, 1::2] * 16 + qweight[:, ::2]).view(torch.int8)
        qweight = qweight.reshape(weight.shape[0], weight.shape[1] // 2)
    return qweight, scale


def preprocess_weights_for_mixed_gemm(weight, quant_mode):
    original_shape = weight.shape
    if quant_mode == torch.int8:
        return weight.T.reshape(original_shape)
    else:
        weight = weight.view(torch.uint8)
        weight_quint4x2 = torch.zeros(original_shape[0],
                                      original_shape[1] * 2).char()
        weight_quint4x2[:, ::2] = weight // 16
        weight_quint4x2[:, 1::2] = weight % 16
        weight_quint4x2 = weight_quint4x2.T
        weight_quint4x2 = weight_quint4x2[:, ::2] + weight_quint4x2[:,
                                                                    1::2] * 16
        row_idx = [
            i + (1 if i % 2 == 0 else -1)
            for i in range(weight_quint4x2.shape[0])
        ]
        weight_quint4x2 = weight_quint4x2[row_idx, :]
        return weight_quint4x2.reshape(original_shape[0],
                                       original_shape[1] // 2)


def change_qkv_leading_dim(w, num_heads):
    if w.dim() == 1:
        w = w.reshape(num_heads, 3, -1)
        w = w.transpose(0, 1).reshape(-1)
    else:
        shape = w.shape
        head_dim = shape[1] // (3 * num_heads)
        w = w.reshape(-1, num_heads, 3, head_dim)
        w = w.transpose(1, 2).reshape(shape[0], -1)
    return w


def pad_like(w, target_shape, value=0):
    if w.shape != target_shape:
        pad_dim = []
        for dim in range(len(target_shape)):
            current_dim = -1 - dim
            pad_dim.append(0)
            pad_dim.append(
                max(0, target_shape[current_dim] - w.shape[current_dim]))
        res = F.pad(w, pad_dim, value=value)
        return res
    else:
        return w


def postprocess_weight_only(tllm_key, weights, quant_mode, layer):
    if weights.dim() > 2:
        v = weights.transpose(-1, -2)
    else:
        v = weights.t()

    tp_dim = 1 if isinstance(layer, ColumnLinear) else 0
    if "weight" in tllm_key:
        if layer.is_padded:
            split_size = layer.out_features if tp_dim == 1 else layer.in_features
            v = torch.split(v, split_size, tp_dim)[layer.tp_rank]
            v = pad_like(v, (layer.in_features, layer.out_features))
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.contiguous(), quant_mode)
        return {
            tllm_key: processed_torch_weights,
            tllm_key.replace("weight", "per_channel_scale"):
            torch_weight_scales,
        }
    else:
        if layer.is_padded and tp_dim == 1:
            weights = torch.split(weights, layer.out_features,
                                  tp_dim)[layer.tp_rank]
            weights = pad_like(weights, (layer.out_features, ))
        return {tllm_key: weights}  # Bias


def postprocess_fp8_rowwise(tllm_key, weights, **kwargs):
    if tllm_key.endswith("per_channel_scale"):
        return {}

    config = kwargs.get("config", None)
    if weights[1] is not None:
        assert weights[0].dtype == torch.float8_e4m3fn
        scale = weights[1].to(torch.float32).reshape(-1)
        return {
            tllm_key: weights[0],
            tllm_key.replace("weight", "per_channel_scale"): scale
        }
    else:
        clamp_val = config.quantization.clamp_val
        # activation range bound.
        x = weights[0].to(torch.float32).clamp(clamp_val[0], clamp_val[1])
        xmax = x.abs().max(-1, keepdim=True).values
        # minimum scaling factor.
        torch_weight_scales = (xmax / 448.0).clamp(min=1.0 / (448.0 * 512.0))
        out = x / torch_weight_scales
        torch_weight_scales = torch_weight_scales.reshape(-1)
        out = torch.clamp(out, -448, 448)
        processed_torch_weights = out.to(torch.float8_e4m3fn)
        processed_torch_weights = processed_torch_weights.to(
            torch.float8_e4m3fn)
        return {
            tllm_key: processed_torch_weights,
            tllm_key.replace("weight", "per_channel_scale"): torch_weight_scales
        }
