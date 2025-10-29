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
from .._utils import (get_sm_version, str_dtype_to_np, str_dtype_to_trt,
                      trt_dtype_to_np)
from ..functional import (Tensor, _add_plugin_info, _create_tensor, cast, clip,
                          constant, flatten, layer_norm, matmul,
                          repeat_interleave, rms_norm, round, sum, view)
from ..layers.linear import ColumnLinear
from ..parameter import Parameter
from ..plugin import TRT_LLM_PLUGIN_NAMESPACE
from .mode import QuantMode


def smooth_quant_gemm(input: Tensor, weights: Tensor, scales_a: Tensor,
                      scales_b: Tensor, per_token_scaling: bool,
                      per_channel_scaling: bool, dtype: str) -> Tensor:
    if not default_net().plugin_config.smooth_quant_gemm_plugin:
        if per_token_scaling and input.size(0) == -1:
            # WAR for DQ per-token scaling doesn't support dynamic shapes

            scale_one = constant(np.array(1.0, dtype=np.float32))
            input = dequantize(input, scale_one, 0, 'float32')
            weights = dequantize(weights, scale_one, 0, 'float32')
            result = matmul(input, weights, False, True, False)
            scales = matmul(scales_a, scales_b, False, False, False)
            result = result * scales
            result = cast(result, dtype)
            return result
        else:
            if not per_token_scaling:
                scales_a = view(scales_a, [])
            else:
                scales_a = flatten(scales_a)
            if not per_channel_scaling:
                scales_b = view(scales_b, [])
            else:
                scales_b = flatten(scales_b)
            input = dequantize(input, scales_a, 0, dtype)
            weights = dequantize(weights, scales_b, 0, dtype)
            result = matmul(input, weights, False, True, False)
            return result
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


def qserve_gemm_per_group(input: Tensor,
                          act_scales: Tensor,
                          weights: Tensor,
                          s1_scales: Tensor,
                          s2_scales: Tensor,
                          s2_zeros: Tensor,
                          group_size: int = 128) -> Tensor:
    if not default_net().plugin_config.qserve_gemm_plugin:
        raise TypeError("QServe Quant GEMM is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'QServeGemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        p_dtype = default_net().plugin_config.qserve_gemm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pf_group_size = trt.PluginField("group_size",
                                        np.array([group_size], np.int32),
                                        trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([pf_type, pf_group_size])
        gemm_plug = plg_creator.create_plugin("qserve_gemm", pfc)
        plug_inputs = [
            input.trt_tensor, weights.trt_tensor, s2_zeros.trt_tensor,
            s2_scales.trt_tensor, s1_scales.trt_tensor, act_scales.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
        _add_plugin_info(layer, plg_creator, "qserve_gemm", pfc)
        if not default_net().strongly_typed:
            # Useless. But must be kept otherwise leads to the following TRT API Usage error:
            # input/output with DataType Int8 in network without Q/DQ layers must have dynamic range set when no calibrator is used
            layer.get_input(0).set_dynamic_range(-128, 127)
            layer.get_input(1).set_dynamic_range(-128, 127)
            layer.get_input(2).set_dynamic_range(-128, 127)
            layer.get_input(3).set_dynamic_range(-128, 127)
        return _create_tensor(layer.get_output(0), layer)


def qserve_gemm_per_channel(input: Tensor, act_scales: Tensor, act_sums: Tensor,
                            weights: Tensor, s1_scales: Tensor,
                            s1_szeros: Tensor) -> Tensor:
    if not default_net().plugin_config.qserve_gemm_plugin:
        raise TypeError("QServe Quant GEMM is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'QServeGemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        p_dtype = default_net().plugin_config.qserve_gemm_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pf_group_size = trt.PluginField("group_size", np.array([-1], np.int32),
                                        trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([pf_type, pf_group_size])
        gemm_plug = plg_creator.create_plugin("qserve_gemm", pfc)

        plug_inputs = [
            input.trt_tensor, weights.trt_tensor, s1_scales.trt_tensor,
            s1_szeros.trt_tensor, act_sums.trt_tensor, act_scales.trt_tensor
        ]
        layer = default_trtnet().add_plugin_v2(plug_inputs, gemm_plug)
        _add_plugin_info(layer, plg_creator, "qserve_gemm", pfc)

        if not default_net().strongly_typed:
            # Useless. But must be kept otherwise leads to the following TRT API Usage error:
            # input/output with DataType Int8 in network without Q/DQ layers must have dynamic range set when no calibrator is used
            layer.get_input(0).set_dynamic_range(-128, 127)
            layer.get_input(1).set_dynamic_range(-128, 127)

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
                                       alpha: Parameter,
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
            input = input * alpha.value
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

        if alpha:
            alpha.is_buffer = True
            alpha_value = alpha.raw_value[0]
        else:
            alpha_value = 1.0

        alpha_ = trt.PluginField("alpha", np.array(alpha_value,
                                                   dtype=np.float32),
                                 trt.PluginFieldType.FLOAT32)

        p_dtype = default_net(
        ).plugin_config.weight_only_groupwise_quant_matmul_plugin
        pf_type_ = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection(
            [pf_type_, quant_algo_, group_size_, alpha_])

        matmul_plug = plg_creator.create_plugin("woq_groupwise_matmul", pfc)

        # quant_algo = use_int8_weight * 16 + fp8_alpha * 8 + pre_quant_scale * 4 + zero * 2 + bias
        plug_inputs = [input.trt_tensor]

        # Flags for indicating whether the corresponding inputs are applied in quant_algo
        # quant_algo = use_int8_weight * INT8_WEIGHT + fp8_alpha * FP8_ALPHA + pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
        # Here use_int8_weight, pre_quant_scale, zero and bias are boolean type
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

        return _create_tensor(layer.get_output(0), layer)


# TODO: Should be renamed to layer_norm_quantize.
def smooth_quant_layer_norm(input: Tensor,
                            normalized_shape: Union[int, Tuple[int]],
                            weight: Optional[Tensor] = None,
                            bias: Optional[Tensor] = None,
                            scale: Optional[Tensor] = None,
                            eps: float = 1e-05,
                            use_diff_of_squares: bool = True,
                            dynamic_act_scaling: bool = False) -> Tensor:
    if not default_net().plugin_config.layernorm_quantization_plugin:
        dtype = trt_dtype_to_np(input.dtype)
        if weight is None:
            weight = constant(np.ones(normalized_shape, dtype=dtype))
        if bias is None:
            bias = constant(np.zeros(normalized_shape, dtype=dtype))
        result = layer_norm(input, normalized_shape, weight, bias, eps,
                            use_diff_of_squares)
        if not dynamic_act_scaling:
            return quantize_tensor(result, scale)
        else:
            return quantize_per_token(result)
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'LayernormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
        assert plg_creator is not None

        output_type = trt.PluginField("out_type_id",
                                      np.array([int(trt.int8)], np.int32),
                                      trt.PluginFieldType.INT32)
        quant_mode = trt.PluginField(
            "quant_mode",
            np.array([int(QuantMode.use_smooth_quant(per_token=True))],
                     np.int32), trt.PluginFieldType.INT32)
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
        pfc = trt.PluginFieldCollection([
            eps, use_diff_of_squares, dyn_act_scaling, pf_type, output_type,
            quant_mode
        ])
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


# TODO: Should be renamed to rms_norm_quantize. This is also used by QServe.
def smooth_quant_rms_norm(
    input: Tensor,
    normalized_shape: Union[int, Tuple[int]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    clamp_val: Optional[Tensor] = None,
    eps: float = 1e-05,
    dynamic_act_scaling: bool = False,
    scale_dtype='float32',
    sum_per_token: bool = False,
    sum_dtype='float32'
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    if sum_per_token and not dynamic_act_scaling:
        raise ValueError(
            "sum_per_token is only allowed if dynamic_act_scaling is enabled!")

    if not default_net().plugin_config.rmsnorm_quantization_plugin:
        result = rms_norm(input, normalized_shape, 1, weight, eps)
        if bias is not None:
            result += bias
        if not dynamic_act_scaling:
            return quantize_tensor(result, scale)
        else:
            return quantize_per_token(result, clamp_val, scale_dtype,
                                      sum_per_token, sum_dtype)
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

        sum_per_token_pf = trt.PluginField(
            "sum_per_token", np.array([int(sum_per_token)], np.int32),
            trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.rmsnorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([
            eps, dyn_act_scaling, sum_per_token_pf, clamp_enabled, quant_mode,
            pf_type, output_type
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

        # TODO: Why not fuse scale (which seems to be a per-tensor scaling factor of the original values) into weight?
        if scale is None:
            scale = constant(np.ones(1, dtype=str_dtype_to_np(p_dtype)))

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

        output_quantized = _create_tensor(layer.get_output(0), layer)
        output_scales = _create_tensor(layer.get_output(1), layer)

        # TODO: The plugin should be able to directly output float16 scales
        if str_dtype_to_trt(scale_dtype) != output_scales.dtype:
            output_scales = cast(output_scales, scale_dtype)

        if not sum_per_token:
            return output_quantized, output_scales

        output_sums = _create_tensor(layer.get_output(2), layer)
        # TODO: The plugin should be able to directly output float16 sums
        if str_dtype_to_trt(sum_dtype) != output_sums.dtype:
            output_sums = cast(output_sums, sum_dtype)

        return output_quantized, output_scales, output_sums


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
        sum_per_token_pf = trt.PluginField("sum_per_token",
                                           np.array([int(False)], np.int32),
                                           trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.rmsnorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([
            eps, dyn_act_scaling, sum_per_token_pf, clamp_enabled, quant_mode,
            pf_type, output_type
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


def fp8_rowwise_layer_norm(input: Tensor,
                           normalized_shape: Union[int, Tuple[int]],
                           weight: Optional[Tensor] = None,
                           bias: Optional[Tensor] = None,
                           scale: Optional[Tensor] = None,
                           clamp_val: Optional[Tensor] = None,
                           eps: float = 1e-05,
                           dynamic_act_scaling: bool = True) -> Tensor:
    if not default_net().plugin_config.layernorm_quantization_plugin:
        raise TypeError("Fp8 Rowwise Layer Norm is only supported with plugin")
    else:
        plg_creator = trt.get_plugin_registry().get_plugin_creator(
            'LayernormQuantization', '1', TRT_LLM_PLUGIN_NAMESPACE)
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
        sum_per_token_pf = trt.PluginField("sum_per_token",
                                           np.array([int(False)], np.int32),
                                           trt.PluginFieldType.INT32)

        p_dtype = default_net().plugin_config.layernorm_quantization_plugin
        pf_type = trt.PluginField(
            "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
            trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([
            eps, dyn_act_scaling, sum_per_token_pf, clamp_enabled, quant_mode,
            pf_type, output_type
        ])

        layernorm_plug = plg_creator.create_plugin("layernorm_quantized", pfc)
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

        # Layer Norm Plugin only supports float32 scale
        scale = cast(scale, "float32")
        plug_inputs = [
            input.trt_tensor, weight.trt_tensor, bias.trt_tensor,
            scale.trt_tensor
        ]
        if clamp_val:
            plug_inputs += [clamp_val.trt_tensor]
        layer = default_trtnet().add_plugin_v2(plug_inputs, layernorm_plug)
        if not default_net().strongly_typed:
            layer.get_output(0).set_dynamic_range(-448, 448)
        _add_plugin_info(layer, plg_creator, "layernorm_quantized", pfc)
        if not dynamic_act_scaling:
            return _create_tensor(layer.get_output(0), layer)

        return _create_tensor(layer.get_output(0),
                              layer), _create_tensor(layer.get_output(1), layer)


def fused_layernorm(
        input: Tensor,
        normalized_shape: Union[int, Tuple[int]],
        residual: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        # beta: Optional[Tensor] = None,
        # bias: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        eps: float = 1e-05,
        p_dtype: str = 'float16',
        need_fp32_output: bool = False) -> Tensor:
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'FusedLayernorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None
    eps = trt.PluginField("eps", np.array(eps, dtype=np.float32),
                          trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    need_fp32_output_value = need_fp32_output
    need_fp32_output = trt.PluginField(
        "need_fp32_output", np.array([int(need_fp32_output_value)], np.int32),
        trt.PluginFieldType.INT32)
    need_quantize_value = scale is not None
    need_quantize = trt.PluginField(
        "need_quantize", np.array([int(need_quantize_value)], np.int32),
        trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection(
        [eps, need_fp32_output, need_quantize, pf_type])
    fused_layernorm_plug = plg_creator.create_plugin("fused_layernorm", pfc)
    normalized_shape = [normalized_shape] if isinstance(
        normalized_shape, int) else normalized_shape
    if weight is None:
        weight = constant(
            np.ones(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
    # if beta is None:
    #     beta = constant(
    #         np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
    # if bias is None:
    #     bias = constant(
    #         np.zeros(normalized_shape, dtype=str_dtype_to_np(p_dtype)))
    if need_quantize_value:
        plug_inputs = [
            input.trt_tensor, residual.trt_tensor, weight.trt_tensor,
            scale.trt_tensor
        ]
    else:
        plug_inputs = [
            input.trt_tensor,
            residual.trt_tensor,
            weight.trt_tensor,
        ]
    layer = default_trtnet().add_plugin_v2(plug_inputs, fused_layernorm_plug)
    _add_plugin_info(layer, plg_creator, "fused_layernorm", pfc)
    if not need_quantize_value:
        return _create_tensor(layer.get_output(0),
                              layer), _create_tensor(layer.get_output(1), layer)
    return _create_tensor(layer.get_output(0), layer), _create_tensor(
        layer.get_output(1), layer), _create_tensor(layer.get_output(2), layer)


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


def quantize_per_token(
    x: Tensor,
    clamp_val: Optional[Tensor] = None,
    scale_dtype='float32',
    sum_per_token: bool = False,
    sum_dtype='float32',
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    if not default_net().plugin_config.quantize_per_token_plugin:
        x = cast(x, 'float32')
        xmax = x.abs().max(-1, keepdim=True)
        scales = xmax / 127.0
        out = x * 127.0 / xmax
        out = round(out)
        out = clip(out, -128, 127)
        quantized = cast(out, 'int8')
        if not sum_per_token:
            return quantized, scales
        sums = sum(x, -1, keepdim=True)
        if sum_dtype is not None and str_dtype_to_trt(sum_dtype) != sums.dtype:
            sums = cast(sums, sum_dtype)
        return quantized, scales, sums

    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'QuantizePerToken', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    output_type = trt.PluginField("type_id", np.array([int(trt.int8)],
                                                      np.int32),
                                  trt.PluginFieldType.INT32)
    quant_mode = trt.PluginField(
        "quant_mode",
        np.array([int(QuantMode.use_smooth_quant(per_token=True))], np.int32),
        trt.PluginFieldType.INT32)
    clamp_enabled = trt.PluginField("clamp_enabled",
                                    np.array([clamp_val is not None], np.int8),
                                    trt.PluginFieldType.INT8)

    sum_per_token_pf = trt.PluginField("sum_per_token",
                                       np.array([int(sum_per_token)], np.int32),
                                       trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection(
        [output_type, quant_mode, clamp_enabled, sum_per_token_pf])
    quantize_plug = plg_creator.create_plugin("quantize_per_token_plugin", pfc)

    plug_inputs = [x.trt_tensor]
    if clamp_val:
        plug_inputs += [clamp_val.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
    if not default_net().strongly_typed:
        layer.get_output(0).set_dynamic_range(-127, 127)
    _add_plugin_info(layer, plg_creator, "quantize_per_token_plugin", pfc)

    quantized = _create_tensor(layer.get_output(0), layer)
    scales = _create_tensor(layer.get_output(1), layer)

    # TODO: The plugin should be able to directly output float16 scales to avoid a cast
    if scale_dtype is not None and str_dtype_to_trt(
            scale_dtype) != scales.dtype:
        scales = cast(scales, scale_dtype)
    if not sum_per_token:
        return quantized, scales

    sums = _create_tensor(layer.get_output(2), layer)
    # TODO: The plugin should be able to directly output float16 sums to avoid a cast
    if sum_dtype is not None and str_dtype_to_trt(sum_dtype) != sums.dtype:
        sums = cast(sums, sum_dtype)

    return quantized, scales, sums


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
        sum_per_token_pf = trt.PluginField("sum_per_token",
                                           np.array([int(False)], np.int32),
                                           trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection(
            [output_type, quant_mode, clamp_enabled, sum_per_token_pf])
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
        if scale.dtype == str_dtype_to_trt('float32'):
            x = cast(x, 'float32')
        scaled = x * scale
        rounded = round(scaled)
        clipped = clip(rounded, -128, 127)
        quantized = cast(clipped, 'int8')
    else:
        scale = cast(scale, 'float32')

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


def preprocess_weights_for_mixed_gemm(
        tensor: torch.Tensor,
        quant_mode: torch.dtype,
        act_dtype: torch.dtype,
        sm_: int = -1,
        do_weight_interleave: bool = True) -> torch.Tensor:
    sm_ = sm_ if sm_ > 0 else get_sm_version()
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    elif sm_ >= 90:
        sm_ = 80
    if sm_ > 90:
        sm_ = 80

    permutation_map = {
        "16_8": [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        "16_4": [
            0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12,
            13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31
        ],
        "8_4": [
            0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23, 8, 9, 10,
            11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31
        ]
    }

    # permute_B_rows_for_mixed_gemm
    BITS_PER_ELT_A = 8 if act_dtype == torch.float8_e4m3fn else 16
    BITS_PER_ELT_B = 4 if quant_mode == torch.quint4x2 else 8
    MMA_SHAPE_N = 8
    B_ROWS_PER_MMA = 8 * 16 // BITS_PER_ELT_B

    num_experts = tensor.shape[0]
    num_rows = tensor.shape[1]
    num_cols = tensor.shape[2]

    assert (sm_ >= 75)
    assert (num_rows % B_ROWS_PER_MMA == 0)
    assert (num_cols % MMA_SHAPE_N == 0)

    if do_weight_interleave:
        row_idx_list = [(row_idx // B_ROWS_PER_MMA) * B_ROWS_PER_MMA +
                        permutation_map[f"{BITS_PER_ELT_A}_{BITS_PER_ELT_B}"][
                            row_idx % B_ROWS_PER_MMA]
                        for row_idx in range(num_rows)]
        tensor = tensor[:, row_idx_list, :]

    # subbyte_transpose
    original_shape = tensor.shape
    if BITS_PER_ELT_B == 4:
        tensor = tensor.view(torch.uint8)
        high_tensor = (tensor >> 4).permute(0, 2, 1).unsqueeze(2)
        low_tensor = ((tensor << 4) >> 4).permute(0, 2, 1).unsqueeze(2)
        new_tensor = torch.cat([low_tensor, high_tensor],
                               dim=2).reshape(tensor.shape[0], -1,
                                              tensor.shape[1])
        new_tensor = new_tensor[:, :, 0::2] + new_tensor[:, :, 1::2] * 16
        tensor = new_tensor.view(torch.int8).reshape(original_shape)
    else:
        tensor = tensor.permute(0, 2, 1).reshape(original_shape)

    if do_weight_interleave:
        # interleave_column_major_tensor
        interleave = BITS_PER_ELT_A // BITS_PER_ELT_B
        if interleave > 1 and sm_ < 90:
            rows_per_tile = 128 * 8 // BITS_PER_ELT_A
            elts_in_int32 = 32 // BITS_PER_ELT_B

            assert (num_rows % elts_in_int32 == 0)
            assert (num_rows % rows_per_tile == 0)

            tensor = tensor.reshape(num_experts, -1, interleave,
                                    num_rows // rows_per_tile,
                                    rows_per_tile * 4 // elts_in_int32)
            tensor = tensor.permute(0, 1, 3, 2, 4).reshape(original_shape)

        # add_bias_and_interleave_quantized_tensor_inplace
        if BITS_PER_ELT_B == 8:
            tensor += -256 * (tensor > 127).byte() + 128
            tensor = tensor.reshape(-1, 4)[:,
                                           [0, 2, 1, 3]].reshape(tensor.shape)
        elif BITS_PER_ELT_B == 4:
            tensor = tensor.view(torch.uint8)
            high_tensor = (tensor >> 4).unsqueeze(-1)
            low_tensor = ((tensor << 4) >> 4).unsqueeze(-1)
            new_tensor = torch.cat([low_tensor, high_tensor],
                                   dim=-1).reshape(tensor.shape[0],
                                                   tensor.shape[1], -1)
            new_tensor = new_tensor.reshape(
                -1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(new_tensor.shape)
            new_tensor += -16 * (new_tensor > 7).byte() + 8
            new_tensor = new_tensor[:, :, 0::2] + new_tensor[:, :, 1::2] * 16
            tensor = new_tensor.view(torch.int8)
        else:
            raise NotImplementedError

    return tensor.squeeze(0).contiguous()


def get_weight_scale_interleave_factor(interleaved_dim: int,
                                       group_size: int = 128) -> int:
    # Calculate the weight_scale interleave factor for W4A8 groupwise MoE quant
    # only Hopper w4a8 does interleave for weight scale, other arch or Hopper w4a16 default to 1
    factor = 1
    if get_sm_version() == 90:
        if interleaved_dim % (4 * group_size) == 0:
            factor = 4
        elif interleaved_dim % (2 * group_size) == 0:
            factor = 2
        elif interleaved_dim % group_size == 0:
            factor = 1
        else:
            raise NotImplementedError(
                f"Interleaved dimension must be a multiple of group_size ({group_size}), received {interleaved_dim}."
            )
    return factor


def validate_group_size(layer):
    # TODO: Remove this function and its usage after W4A8-AWQ with group_size = 64 is implemented.
    W4A8_AWQ = 8
    if layer.quant_algo & W4A8_AWQ and layer.group_size == 64:
        raise NotImplementedError(
            "W4A8_AWQ with group_size = 64 is not implemented yet!")


def unpack_int32_into_int8(w_packed, autoawq_reorder=False):
    # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
    w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
    w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                             w_packed_int4x2.shape[1] * 2,
                             dtype=torch.int8)
    w_unpacked[:, ::2] = w_packed_int4x2 % 16
    w_unpacked[:, 1::2] = w_packed_int4x2 // 16
    if autoawq_reorder:
        w_unpacked = w_unpacked.view(-1, 8)[:, [0, 4, 1, 5, 2, 6, 3, 7]].view(
            w_unpacked.shape)
    return w_unpacked.contiguous()


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


def postprocess_weight_only_groupwise(tllm_key, weights, torch_dtype, layer,
                                      **kwargs):
    using_head_as_leading_dim = kwargs.get("using_head_as_leading_dim", False)
    config = kwargs.get("config", None)
    use_autoawq = kwargs.get("use_autoawq", None)
    num_heads = config.num_attention_heads
    USE_GPTQ = layer.prequant_scaling_factor is None and use_autoawq is None
    USE_HF_AWQ = layer.prequant_scaling_factor is None and use_autoawq is not None
    USE_MODELOPT_AWQ = layer.prequant_scaling_factor is not None
    USE_INT8_WEIGHT = layer.quant_algo & 16

    tp_dim = 1 if isinstance(layer, ColumnLinear) else 0
    is_qkv = layer.is_qkv if hasattr(layer, "is_qkv") else False

    if using_head_as_leading_dim:
        assert config.num_attention_heads == config.num_key_value_heads, "using_head_as_leading_dim require head_size to be multiple of 3."
    if tllm_key.endswith("weights_scaling_factor"):
        # TODO: Remove reshaping after modelopt optimizes scale shape
        if is_qkv:
            for idx, w in enumerate(weights):
                scales = w.to(torch_dtype)
                scales = scales.reshape(-1,
                                        layer.weights_scaling_factor.shape[0]).T
                scales = scales.chunk(layer.tp_size, 1)[layer.tp_rank]
                weights[idx] = scales
            weights = torch.cat(weights, dim=1)
        else:
            scales = weights.to(torch_dtype)
            scales_shape = [
                layer.weights_scaling_factor.shape[1],
                layer.weights_scaling_factor.shape[0]
            ]
            scales_shape[1 - tp_dim] *= layer.tp_size
            scales = scales.reshape(scales_shape).T
            weights = scales.chunk(layer.tp_size, tp_dim)[layer.tp_rank]
    if is_qkv and isinstance(weights, list) and len(weights) >= 3:
        if USE_MODELOPT_AWQ:
            if tllm_key.endswith("prequant_scaling_factor"):
                weights = weights[0]
            else:
                weights = torch.cat(weights, dim=0)
        elif len(weights) > 3:
            weights = [
                torch.cat(weights[i::len(weights) // 3], dim=1)
                for i in range(len(weights) // 3)
            ]

    if tllm_key.endswith("bias"):
        if is_qkv and isinstance(weights, list):
            weights = torch.cat(weights)
        if layer.is_padded:
            weights = pad_like(weights, layer.bias.shape)
        if using_head_as_leading_dim:
            weights = change_qkv_leading_dim(weights, num_heads)
        results = {tllm_key: weights.to(torch_dtype)}
    elif tllm_key.endswith("weight"):
        if not USE_INT8_WEIGHT:
            # 4 bit quantization
            if USE_GPTQ:
                qweight = unpack_int32_into_int8(weights[0].T).T - 8
            elif USE_HF_AWQ:
                qweight = unpack_int32_into_int8(weights[0], True) - 8
            else:
                qweight = unpack_int32_into_int8(weights.T)
            qweight -= (qweight >> 4) << 4
            qweight = qweight.view(torch.uint8)
        elif USE_INT8_WEIGHT and USE_GPTQ:
            # 8 bit quantization (only consider INT8 GPTQ here)
            qweight = (
                weights[0].T.contiguous().view(torch.uint8).T.contiguous() -
                128).to(torch.int8)
        else:
            raise NotImplementedError(
                "Unsupported quantization mode for weight.")

        if using_head_as_leading_dim:
            qweight = change_qkv_leading_dim(qweight, num_heads)
        if layer.is_padded:
            qweight = torch.split(qweight, layer.out_features,
                                  tp_dim)[layer.tp_rank]
            qweight = pad_like(qweight, (layer.in_features, layer.out_features))
        # pack int8 tensor to packed int4
        if not USE_INT8_WEIGHT:
            qweight = (qweight[:, 1::2] * 16 + qweight[:, ::2]).view(torch.int8)
        weight_type = torch.int8 if USE_INT8_WEIGHT else torch.quint4x2
        qweight = preprocess_weights_for_mixed_gemm(
            qweight, weight_type, torch.float16).view(torch_dtype)
        results = {tllm_key: qweight}

        # scales and zeros for GPTQ and HF-AWQ
        if USE_GPTQ or USE_HF_AWQ:
            scales = weights[1].to(torch_dtype)
            if USE_INT8_WEIGHT:
                qzeros = weights[2].view(torch.uint8)
            else:
                qzeros = unpack_int32_into_int8(weights[2], USE_HF_AWQ)
            if using_head_as_leading_dim:
                scales = change_qkv_leading_dim(scales, num_heads)
                qzeros = change_qkv_leading_dim(qzeros, num_heads)
            if layer.is_padded:
                scales = torch.split(scales,
                                     layer.weights_scaling_factor.shape[tp_dim],
                                     tp_dim)[layer.tp_rank]
                scales = pad_like(scales, layer.weights_scaling_factor.shape, 1)
                qzeros = torch.split(qzeros,
                                     layer.weights_scaling_factor.shape[tp_dim],
                                     tp_dim)[layer.tp_rank]
                qzeros = pad_like(qzeros, layer.zero.shape, 7)
            if USE_INT8_WEIGHT:
                zeros_x_scales = (-qzeros + 128 - 1 * USE_GPTQ) * scales
            else:
                zeros_x_scales = (-qzeros + 8 - 1 * USE_GPTQ) * scales
            zeros_x_scales = zeros_x_scales.to(torch_dtype)
            results.update({
                tllm_key.replace("weight", "weights_scaling_factor"):
                scales,
                tllm_key.replace("weight", "zero"):
                zeros_x_scales,
            })
    elif tllm_key.endswith("weights_scaling_factor"):
        # TODO: Remove reshaping after modelopt optimizes scale shape
        if layer.is_padded:
            raise NotImplementedError(
                "Auto-padding is not Implemented for ModelOpt HF-AWQ.")
        results = {tllm_key: weights}
    elif tllm_key.endswith("prequant_scaling_factor"):
        prequant_scale = weights.to(torch_dtype).reshape(1, -1)
        if layer.is_padded and tp_dim == 1:
            prequant_scale = torch.split(prequant_scale,
                                         layer.prequant_scaling_factor.shape[1],
                                         1)[layer.tp_rank]
            prequant_scale = pad_like(prequant_scale,
                                      layer.prequant_scaling_factor.shape, 0)
        results = {tllm_key: prequant_scale}

    return results


def postprocess_fp8_rowwise(tllm_key, weights, **kwargs):
    if tllm_key.endswith("per_channel_scale"):
        return {}

    config = kwargs.get("config", None)
    weights, scales = weights[0::2], weights[1::2]

    if scales[0] is not None:
        assert all(w.dtype == torch.float8_e4m3fn for w in weights)
        weights = torch.cat(weights, dim=0)
        scales = torch.cat([s.to(torch.float32).flatten() for s in scales])
        return {
            tllm_key: weights,
            tllm_key.replace("weight", "per_channel_scale"): scales
        }
    else:
        x = torch.cat(weights, dim=0).to(torch.float32)
        clamp_val = config.quantization.clamp_val
        if clamp_val is not None:
            # activation range bound.
            x = x.clamp(clamp_val[0], clamp_val[1])
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


def fp4_gemm(input: Tensor,
             input_sf: Tensor,
             weight: Tensor,
             weight_sf: Tensor,
             global_sf: Tensor,
             output_dtype: str | trt.DataType,
             scaling_vector_size: int = 16):
    '''
    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, seq_len, input_dim] or [num_tokens, input_dim] for remove_input_padding, should be fp4
        input_sf : Tensor (On GPU)
            The input scaling factor tensor. Its shape is [batch_size, seq_len, input_dim / scaling_vector_size] or [num_tokens, input_dim / scaling_vector_size] for remove_input_padding, should be int32 (4 packed)
        weight : Tensor (On GPU)
            The weight tensor. Its shape is [output_dim, input_dim], should be fp4
        weight_sf : Tensor (On GPU)
            The weight scaling factor tensor. Its shape is [output_dim, input_dim / scaling_vector_size], should be fp8
        global_sf : Tensor (On GPU)
            The global scaling factor tensor. Its shape is [1,], should be float32, used as alpha of Gemm.
        output_dtype: str
            output data type
        scaling_vector_size: int
            scaling vector block size
    '''
    if isinstance(output_dtype, str):
        output_dtype = str_dtype_to_trt(output_dtype)

    fp4_gemm_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'Fp4Gemm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert fp4_gemm_plg_creator is not None
    sv_vec_size = trt.PluginField("sv_vec_size",
                                  np.array(scaling_vector_size, dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    output_dtype = trt.PluginField("output_type_id",
                                   np.array([int(output_dtype)], np.int32),
                                   trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([sv_vec_size, output_dtype])
    fp4_gemm_plug = fp4_gemm_plg_creator.create_plugin("fp4_gemm", pfc)
    plug_inputs = [input, input_sf, weight, weight_sf, global_sf]
    plug_inputs = [i.trt_tensor for i in plug_inputs]
    layer = default_trtnet().add_plugin_v2(plug_inputs, fp4_gemm_plug)
    _add_plugin_info(layer, fp4_gemm_plg_creator, "fp4_gemm", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    return output


def quantize_to_fp4_tensor(input: Tensor, sf_scale: Tensor):
    '''
    Parameters:
        input : Tensor (On GPU)
            The input tensor. Its shape is [batch_size, seq_len, input_dim] or [num_tokens, input_dim] for remove_input_padding, should be fp16
        sf_scale : Tensor (On GPU)
            The global per-tensor scaling factor. Its shape is [1,], should be float32.
            used to scale SF from input range to fp8 range (448.f / (MaxVal of input / 6.f)).
        output : Tensor (On GPU)
            The output tensor. Its shape is [batch_size, seq_len, input_dim] or [num_tokens, input_dim] for remove_input_padding, should be FP4
        output_sf : Tensor (On GPU)
            The input scaling factor tensor. Its shape is [batch_size, seq_len, input_dim / scaling_vector_size] or [num_tokens, input_dim / scaling_vector_size] for remove_input_padding, should be FP8
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'QuantizeToFP4', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pfc = trt.PluginFieldCollection([])
    quantize_plug = plg_creator.create_plugin("quantize_to_fp4_plugin", pfc)

    plug_inputs = [input.trt_tensor, sf_scale.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, quantize_plug)
    _add_plugin_info(layer, plg_creator, "quantize_to_fp4_plugin", pfc)

    quantized = _create_tensor(layer.get_output(0), layer)
    scales = _create_tensor(layer.get_output(1), layer)
    return quantized, scales


def dynamic_quantize(
        x: Tensor,
        double_scale: Tensor,
        axis: int = -1,
        block_size: int = 16,
        data_qtype: trt.DataType = trt.fp4,
        scale_qtype: trt.DataType = trt.fp8) -> Tuple[Tensor, Tensor]:
    '''
    Parameters:
        x : Tensor (On GPU)
            The input tensor.
        double_scale : Tensor (On GPU)
            The global per-tensor scaling factor. It should contain only 1 element.
        axis : int
            The axis to quantize. Default is -1 (the last axis).
        block_size : int
            The block size for quantization. Default is 16.
        data_qtype : trt.DataType
            The data type for quantized data. Default is FP4.
        scale_qtype : trt.DataType
            The data type for block scale. Default is FP8.
    Returns:
        A tuple of two tensors: quantized tensor and block scale tensor.
    '''
    if axis < 0:
        axis = len(x.shape) + axis
    dynq = default_trtnet().add_dynamic_quantize(x.trt_tensor, axis, block_size,
                                                 data_qtype, scale_qtype)
    dynq.set_input(1, double_scale.trt_tensor)
    quantized = _create_tensor(dynq.get_output(0), dynq)
    scale = _create_tensor(dynq.get_output(1), dynq)
    return quantized, scale


def block_double_dequantize(x: Tensor,
                            scale: Tensor,
                            double_scale: Tensor,
                            dtype: trt.DataType | str = 'float16') -> Tensor:
    '''
    Parameters:
        x : Tensor (On GPU)
            The input tensor.
        scale : Tensor (On GPU)
            The block scale tensor.
        double_scale : Tensor (On GPU)
            The global per-tensor scaling factor. It should contain only 1 element.
        dtype : trt.DataType | str
            The data type for dequantized data. Default is float32.
    Returns:
        The dequantized tensor.
    '''
    if isinstance(dtype, str):
        dtype = str_dtype_to_trt(dtype)
    dequantize_scale_layer = default_trtnet().add_dequantize(
        scale.trt_tensor, double_scale.trt_tensor, dtype)
    scale = _create_tensor(dequantize_scale_layer.get_output(0),
                           dequantize_scale_layer)

    dequantize_data_layer = default_trtnet().add_dequantize(
        x.trt_tensor, scale.trt_tensor, dtype)
    dequantize_data = _create_tensor(dequantize_data_layer.get_output(0),
                                     dequantize_data_layer)
    return dequantize_data
