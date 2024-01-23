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

import numpy as np
import tensorrt as trt

from .._common import default_net, precision
from .._utils import fp32_array, str_dtype_to_trt
from ..functional import (ACT2FN, Tensor, allgather, allreduce, cast, concat,
                          constant, generate_alibi_slopes, gpt_attention,
                          matmul, mul, shape, slice, softmax, split, where)
from ..layers.attention import AttentionMaskType, PositionEmbeddingType
from ..layers.linear import Linear, RowLinear
from ..module import Module
from ..parameter import Parameter
from .functional import (dequantize, quantize, quantize_per_token,
                         quantize_tensor, smooth_quant_gemm,
                         smooth_quant_layer_norm, smooth_quant_rms_norm,
                         weight_only_groupwise_quant_matmul,
                         weight_only_quant_matmul)
from .mode import QuantMode


class Quantize(Module):
    """
        Quantize Layer
        For per-tensor mode, the scaling factor is a scalar.
        For per-channel mode, the scaling factor is a vector.
        """

    def __init__(
        self,
        output_dtype: str = 'int8',
        scaling_factor_dtype: str = 'float32',
        in_channels: int = -1,
        axis=-1,
    ) -> None:
        super().__init__()
        self.scaling_factor = Parameter(shape=(in_channels, ) if axis != -1 else
                                        (),
                                        dtype=scaling_factor_dtype)
        self.output_dtype = output_dtype
        self.axis = axis

    def forward(self, x):
        return quantize(x, self.scaling_factor.value, self.output_dtype,
                        self.axis)


class QuantizePerToken(Module):
    """
        Quantize Per Token and compute dynamic scales for SmoothQuant
        """

    def forward(self, x):
        return quantize_per_token(x)


class Dequantize(Module):
    """
        Dequantize Layer.
        """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.scaling_factor = Parameter(shape=())
        self.axis = axis

    def forward(self, input):
        return dequantize(input, self.scaling_factor.value, self.axis)


class SmoothQuantLinear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 gather_output=True,
                 quant_mode=QuantMode(0),
                 max_lora_rank=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features // tp_size

        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Linear has to have act+weight quantization mode set"
            )

        weights_dtype = dtype
        if quant_mode.has_act_and_weight_quant():
            weights_dtype = "int8"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)

        if quant_mode.has_act_and_weight_quant():
            scale_shape = (1, self.out_features
                           ) if quant_mode.has_per_channel_scaling() else (1, 1)
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        if quant_mode.has_act_static_scaling():
            self.act_scale = Parameter(shape=(1, 1), dtype="float32")

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output
        self.quant_mode = quant_mode

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on SmoothQuantLinear now"
        if self.quant_mode.has_act_static_scaling():
            per_token_scale = self.act_scale.value
        else:
            # If we are in SmoothQuant with dynamic activation scaling,
            # input x has to be a tuple of int8 tensor and fp32 scaling factors
            x, per_token_scale = x
        x = smooth_quant_gemm(x, self.weight.value, per_token_scale,
                              self.per_channel_scale.value,
                              self.quant_mode.has_per_token_dynamic_scaling(),
                              self.quant_mode.has_per_channel_scaling())

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=1)

        return x


SmoothQuantColumnLinear = SmoothQuantLinear


class SmoothQuantRowLinear(Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
            max_lora_rank=None,
    ):
        super().__init__()
        self.in_features = in_features // tp_size
        self.out_features = out_features
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Linear has to have act+weight quantization mode set"
            )
        weights_dtype = dtype
        if quant_mode.has_act_and_weight_quant():
            weights_dtype = "int8"

        self.weight = Parameter(shape=(self.out_features, self.in_features),
                                dtype=weights_dtype)
        self.smoother = Parameter(shape=(1, self.in_features), dtype="float32")
        if quant_mode.has_act_and_weight_quant():
            scale_shape = (1, self.out_features
                           ) if quant_mode.has_per_channel_scaling() else (1, 1)
            self.per_channel_scale = Parameter(shape=scale_shape,
                                               dtype="float32")

        if quant_mode.has_act_static_scaling():
            self.act_scale = Parameter(shape=(1, 1), dtype="float32")

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on SmoothQuantRowLinear now"
        if self.quant_mode.has_act_static_scaling():
            per_token_scale = self.act_scale.value
        else:
            x, per_token_scale = x
        x = smooth_quant_gemm(x, self.weight.value, per_token_scale,
                              self.per_channel_scale.value,
                              self.quant_mode.has_per_token_dynamic_scaling(),
                              self.quant_mode.has_per_channel_scaling())

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            x = x + self.bias.value

        return x


class SmoothQuantLayerNorm(Module):

    def __init__(
            self,
            normalized_shape,
            eps=1e-05,
            elementwise_affine=True,
            dtype=None,
            quant_mode=QuantMode(0),
            max_lora_rank=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant layer norm has to have some quantization mode set")
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_and_weight_quant():
            self.scale_to_int = Parameter(shape=(1, ), dtype=dtype)
        else:
            self.register_parameter('scale_to_int', None)

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        scale = None if self.scale_to_int is None else self.scale_to_int.value
        return smooth_quant_layer_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            scale,
            self.eps,
            dynamic_act_scaling=self.quant_mode.has_per_token_dynamic_scaling())


class SmoothQuantRmsNorm(Module):

    def __init__(
            self,
            normalized_shape,
            eps=1e-06,
            elementwise_affine=True,
            dtype=None,
            quant_mode=QuantMode(0),
            bias=False,
            max_lora_rank=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError(
                "SmoothQuant Rms norm has to have some quantization mode set")
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.eps = eps
        self.quant_mode = quant_mode

        if self.quant_mode.has_act_and_weight_quant():
            self.scale_to_int = Parameter(shape=(1, ), dtype=dtype)
        else:
            self.register_parameter('scale_to_int', None)

    def forward(self, x):
        weight = None if self.weight is None else self.weight.value
        bias = None if self.bias is None else self.bias.value
        scale = None if self.scale_to_int is None else self.scale_to_int.value
        return smooth_quant_rms_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            scale,
            self.eps,
            dynamic_act_scaling=self.quant_mode.has_per_token_dynamic_scaling())


class WeightOnlyQuantLinear(Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            gather_output=True,
            quant_mode=QuantMode.use_weight_only(),
            max_lora_rank=None,
    ):
        super().__init__()
        if quant_mode.is_int8_weight_only():
            self.weight_only_quant_mode = 1
            quant_type_size_in_bits = 8
        elif quant_mode.is_int4_weight_only():
            self.weight_only_quant_mode = 2
            quant_type_size_in_bits = 4
        self.in_features = in_features
        self.out_features = out_features // tp_size
        # we use a fake tensor with data_type = int8
        self.weight = Parameter(shape=(self.in_features,
                                       int(self.out_features *
                                           quant_type_size_in_bits / 8)),
                                dtype="int8")

        scale_shape = (self.out_features, )
        self.per_channel_scale = Parameter(shape=scale_shape, dtype=dtype)

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on WeightOnlyQuantLinear now"
        # ootb has not supported int4 yet.
        if self.weight_only_quant_mode == 2 and not default_net(
        ).plugin_config.weight_only_quant_matmul_plugin:
            raise TypeError(
                "Int4 Weight Only Qunat MatMul is only supported with plugin")

        x = weight_only_quant_matmul(x, self.weight.value,
                                     self.per_channel_scale.value,
                                     self.weight_only_quant_mode)

        if self.bias is not None:
            x = x + self.bias.value

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=1)

        return x


WeightOnlyQuantColumnLinear = WeightOnlyQuantLinear


class WeightOnlyQuantRowLinear(Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode.use_weight_only(),
            max_lora_rank=None,
    ):
        super().__init__()
        if quant_mode.is_int8_weight_only():
            self.weight_only_quant_mode = 1
        elif quant_mode.is_int4_weight_only():
            self.weight_only_quant_mode = 2
        self.in_features = in_features // tp_size
        self.out_features = out_features
        #we use a fake tensor with data_type = int8
        self.weight = Parameter(shape=(self.in_features,
                                       int(self.out_features /
                                           self.weight_only_quant_mode)),
                                dtype="int8")
        self.per_channel_scale = Parameter(shape=(self.out_features, ),
                                           dtype=dtype)

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on WeightOnlyQuantRowLinear now"
        x = weight_only_quant_matmul(x, self.weight.value,
                                     self.per_channel_scale.value,
                                     self.weight_only_quant_mode)

        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        if self.bias is not None:
            x = x + self.bias.value

        return x


class WeightOnlyGroupwiseQuantLinear(Module):

    def __init__(
        self,
        in_features,
        out_features,
        group_size=128,
        pre_quant_scale=False,
        zero=False,
        bias=False,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
        max_lora_rank=None,
    ):

        super().__init__()

        # Flags for indicating whether the corresponding inputs are applied in quant_algo
        BIAS = 1
        ZERO = 2
        PRE_QUANT_SCALE = 4

        self.quant_algo = pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features // tp_size
        self.weight = Parameter(shape=(self.in_features,
                                       self.out_features // 4),
                                dtype="float16")

        scale_shape = (self.in_features // group_size, self.out_features)
        self.weights_scaling_factor = Parameter(shape=scale_shape, dtype=dtype)

        if pre_quant_scale:
            self.prequant_scaling_factor = Parameter(shape=(1,
                                                            self.in_features),
                                                     dtype=dtype)
        else:
            self.register_parameter('prequant_scaling_factor', None)

        if zero:
            self.zero = Parameter(shape=scale_shape, dtype=dtype)
        else:
            self.register_parameter('zero', None)

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on WeightOnlyGroupwiseQuantLinear now"
        pre_quant_scale = self.prequant_scaling_factor.value if self.prequant_scaling_factor else None
        zero = self.zero.value if self.zero else None
        bias = self.bias.value if self.bias else None

        x = weight_only_groupwise_quant_matmul(
            x, pre_quant_scale, self.weight.value,
            self.weights_scaling_factor.value, zero, bias, self.quant_algo,
            self.group_size)

        if self.gather_output and self.tp_size > 1 and self.tp_group is not None:
            # [dim0, local_dim] -> [dim0 * tp_size, local_dim] --> [dim0, local_dim * tp_size]
            x = allgather(x, self.tp_group, gather_dim=1)

        return x


WeightOnlyGroupwiseQuantColumnLinear = WeightOnlyGroupwiseQuantLinear


class WeightOnlyGroupwiseQuantRowLinear(Module):

    def __init__(
        self,
        in_features,
        out_features,
        group_size=128,
        pre_quant_scale=False,
        zero=False,
        bias=False,
        dtype=None,
        tp_group=None,
        tp_size=1,
        max_lora_rank=None,
    ):
        super().__init__()

        # Flags for indicating whether the corresponding inputs are applied in quant_algo
        BIAS = 1
        ZERO = 2
        PRE_QUANT_SCALE = 4

        self.quant_algo = pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
        self.group_size = group_size
        self.in_features = in_features // tp_size
        self.out_features = out_features
        self.weight = Parameter(shape=(self.in_features,
                                       self.out_features // 4),
                                dtype="float16")

        scale_shape = (self.in_features // group_size, self.out_features)
        self.weights_scaling_factor = Parameter(shape=scale_shape, dtype=dtype)

        if pre_quant_scale:
            self.prequant_scaling_factor = Parameter(shape=(1,
                                                            self.in_features),
                                                     dtype=dtype)
        else:
            self.register_parameter('prequant_scaling_factor', None)

        if zero:
            self.zero = Parameter(shape=scale_shape, dtype=dtype)
        else:
            self.register_parameter('zero', None)

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
        else:
            self.register_parameter('bias', None)

        self.tp_size = tp_size
        self.tp_group = tp_group

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on WeightOnlyGroupwiseQuantRowLinear now"
        pre_quant_scale = self.prequant_scaling_factor.value if self.prequant_scaling_factor else None
        zero = self.zero.value if self.zero else None
        bias = self.bias.value if self.bias else None

        x = weight_only_groupwise_quant_matmul(
            x, pre_quant_scale, self.weight.value,
            self.weights_scaling_factor.value, zero, bias, self.quant_algo,
            self.group_size)
        if self.tp_size > 1 and self.tp_group is not None:
            x = allreduce(x, self.tp_group)

        return x


class SmoothQuantMLP(Module):

    def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
            max_lora_rank=None,
    ):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        fc_output_size = 2 * ffn_hidden_size if hidden_act == 'swiglu' else ffn_hidden_size
        self.fc = SmoothQuantColumnLinear(hidden_size,
                                          fc_output_size,
                                          bias=bias,
                                          dtype=dtype,
                                          tp_group=tp_group,
                                          tp_size=tp_size,
                                          gather_output=False,
                                          quant_mode=quant_mode)

        self.proj = SmoothQuantRowLinear(ffn_hidden_size,
                                         hidden_size,
                                         bias=bias,
                                         dtype=dtype,
                                         tp_group=tp_group,
                                         tp_size=tp_size,
                                         quant_mode=quant_mode)

        self.hidden_act = hidden_act
        self.quant_mode = quant_mode
        self.dtype = dtype

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

    def forward(self, hidden_states, lora_layer_params=None):

        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        value = cast(self.proj.smoother.value, inter.dtype)
        inter = inter / value
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantiztion layers as it breaks int8 plugins
                inter = quantize_tensor(inter,
                                        self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                inter = quantize_per_token(inter)
        output = self.proj(inter)
        return output


class Int8SmoothQuantRowLinear(RowLinear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        max_lora_rank=None,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(self.out_features, ),
                                                dtype=trt.float32)
        self.prequant_scaling_factor = Parameter(shape=(self.in_features, ),
                                                 dtype=dtype)

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on Int8SmoothQuantRowLinear now"

        if default_net().strongly_typed:
            assert x.dtype == self.dtype
            assert x.dtype == self.weight.value.dtype
        x = mul(x, self.prequant_scaling_factor.value)

        activation_scaling_factor = cast(self.activation_scaling_factor.value,
                                         self.dtype)
        quantized_out = quantize(x, activation_scaling_factor, 'int8')
        dequantized_out = dequantize(quantized_out, activation_scaling_factor,
                                     -1, self.dtype)

        weights_scaling_factor = cast(self.weights_scaling_factor.value,
                                      self.dtype)
        w_quant_out = quantize(self.weight.value,
                               weights_scaling_factor,
                               'int8',
                               axis=0)
        w_deq_out = dequantize(w_quant_out, weights_scaling_factor, 0,
                               self.dtype)

        return super().multiply_reduce(dequantized_out, w_deq_out, False)


class Int8SmoothQuantLinear(Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
        max_lora_rank=None,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)

        self.weights_scaling_factor = Parameter(shape=(self.out_features, ),
                                                dtype=trt.float32)
        self.prequant_scaling_factor = Parameter(shape=(self.in_features, ),
                                                 dtype=dtype)

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on Int8SmoothQuantLinear now"
        if default_net().strongly_typed:
            assert x.dtype == self.dtype
            assert x.dtype == self.weight.value.dtype
        x = mul(x, self.prequant_scaling_factor.value)

        activation_scaling_factor = cast(self.activation_scaling_factor.value,
                                         self.dtype)
        quantized_out = quantize(x, activation_scaling_factor, 'int8')
        dequantized_out = dequantize(quantized_out, activation_scaling_factor,
                                     -1, self.dtype)

        weights_scaling_factor = cast(self.weights_scaling_factor.value,
                                      self.dtype)
        w_quant_out = quantize(self.weight.value,
                               weights_scaling_factor,
                               'int8',
                               axis=0)
        w_deq_out = dequantize(w_quant_out, weights_scaling_factor, 0,
                               self.dtype)

        return super().multiply_gather(dequantized_out, w_deq_out, False)


class FP8Linear(Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        gather_output=True,
        max_lora_rank=None,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         gather_output=gather_output)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(1, ), dtype=trt.float32)

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on FP8Linear now"
        if default_net().strongly_typed:
            if isinstance(self.dtype, str):
                assert x.dtype == str_dtype_to_trt(
                    self.dtype
                ), f"Got input type {x.dtype}, expecting {self.dtype}"
            else:
                assert x.dtype == self.dtype, f"Got input type {x.dtype}, expecting {self.dtype}"
            assert x.dtype == self.weight.value.dtype, f"Got input type {x.dtype}, got weight dtype{self.weight.value.dtype}"
        activation_scaling_factor = cast(self.activation_scaling_factor.value,
                                         self.dtype)
        quantized_out = quantize(x, activation_scaling_factor, 'fp8')
        dequantized_out = dequantize(quantized_out, activation_scaling_factor,
                                     -1, self.dtype)

        weights_scaling_factor = cast(self.weights_scaling_factor.value,
                                      self.dtype)
        w_quant_out = quantize(self.weight.value, weights_scaling_factor, 'fp8')
        w_deq_out = dequantize(w_quant_out, weights_scaling_factor, -1,
                               self.dtype)

        # TODO: allow gemm plugin default_net().plugin_config.gemm_plugin
        return self.multiply_gather(dequantized_out,
                                    w_deq_out,
                                    False,
                                    use_fp8=True)


class FP8RowLinear(RowLinear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        tp_group=None,
        tp_size=1,
        max_lora_rank=None,
    ):
        super().__init__(in_features,
                         out_features,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size)
        self.activation_scaling_factor = Parameter(shape=(1, ),
                                                   dtype=trt.float32)
        self.weights_scaling_factor = Parameter(shape=(1, ), dtype=trt.float32)

    def forward(self, x, lora_runtime_params=None):
        assert lora_runtime_params is None, "lora is not supported on FP8RowLinear now"
        if default_net().strongly_typed:
            if isinstance(self.dtype, str):
                assert x.dtype == str_dtype_to_trt(self.dtype)
            else:
                assert x.dtype == self.dtype

            assert x.dtype == self.weight.value.dtype

        activation_scaling_factor = cast(self.activation_scaling_factor.value,
                                         self.dtype)
        quantized_out = quantize(x, activation_scaling_factor, 'fp8')
        dequantized_out = dequantize(quantized_out, activation_scaling_factor,
                                     -1, self.dtype)

        weights_scaling_factor = cast(self.weights_scaling_factor.value,
                                      self.dtype)
        w_quant_out = quantize(self.weight.value, weights_scaling_factor, 'fp8')
        w_deq_out = dequantize(w_quant_out, weights_scaling_factor, -1,
                               self.dtype)

        # TODO: allow gemm plugin default_net().plugin_config.gemm_plugin
        return self.multiply_reduce(dequantized_out,
                                    w_deq_out,
                                    False,
                                    use_fp8=True)


class SmoothQuantGatedMLP(SmoothQuantMLP):

    def __init__(
            self,
            hidden_size,
            ffn_hidden_size,
            hidden_act,
            bias=True,
            dtype=None,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
            max_lora_rank=None,
    ):
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         quant_mode=quant_mode)
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        self.gate = SmoothQuantColumnLinear(hidden_size,
                                            ffn_hidden_size,
                                            bias=bias,
                                            dtype=dtype,
                                            tp_group=tp_group,
                                            tp_size=tp_size,
                                            gather_output=False,
                                            quant_mode=quant_mode)

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

    def forward(self, hidden_states, lora_layer_params=None):
        assert lora_layer_params is None, "lora is not supported on SmoothQuantGatedMLP now"
        inter = self.fc(hidden_states)
        inter = ACT2FN[self.hidden_act](inter)
        gate = self.gate(hidden_states)
        inter_x_gate = inter * gate
        smoother = cast(self.proj.smoother.value, self.dtype)
        inter_x_gate = inter_x_gate / smoother
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantiztion layers as it breaks int8 plugins
                inter_x_gate = quantize_tensor(
                    inter_x_gate, self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                inter_x_gate = quantize_per_token(inter_x_gate)

        output = self.proj(inter_x_gate)
        return output


class SmoothQuantAttention(Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_kv_heads=None,
        max_position_embeddings=1024,
        num_layers=1,
        apply_query_key_layer_scaling=False,
        attention_mask_type=AttentionMaskType.padding,
        bias=True,
        qkv_bias_only=False,
        dtype=None,
        position_embedding_type=PositionEmbeddingType.learned_absolute,
        rotary_embedding_base=10000.0,
        tp_group=None,
        tp_size=1,
        tp_rank=0,
        scale_alibi_bias=False,
        paged_kv_cache=False,
        quant_mode=QuantMode(0),
        enable_pos_shift=False,
        dense_context_fmha=False,
        max_lora_rank=None,
    ):
        super().__init__()
        self.attention_mask_type = attention_mask_type
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.num_kv_heads = (
            num_kv_heads + tp_size - 1
        ) // tp_size if num_kv_heads is not None else self.num_attention_heads
        self.hidden_size = hidden_size // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.num_layers = num_layers
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.attention_head_size)
        self.q_scaling = 1
        if self.apply_query_key_layer_scaling:
            self.norm_factor *= self.num_layers
            self.q_scaling *= self.num_layers
        # Whether to scale ALiBi bias. Mathematically, it's equivalent to
        # normalizing QK after adding bias.
        #   - False, inv_sqrt_Dh * Q*K^T + alibi_bias
        #   - True,  inv_sqrt_Dh * Q*K^T + inv_sqrt_Dh * alibi_bias
        self.scale_alibi_bias = scale_alibi_bias

        self.position_embedding_type = position_embedding_type
        self.paged_kv_cache = paged_kv_cache
        self.enable_pos_shift = enable_pos_shift
        self.dense_context_fmha = dense_context_fmha

        self.rotary_embedding_base = rotary_embedding_base
        self.rotary_embedding_dim = 0
        if self.position_embedding_type.is_rope():
            self.rotary_embedding_dim = hidden_size // num_attention_heads

        self.quant_mode = quant_mode
        self.dtype = dtype

        if self.quant_mode.has_act_static_scaling():
            self.quantization_scaling_factor = Parameter(shape=(1, ),
                                                         dtype='float32')
        else:
            self.register_parameter('quantization_scaling_factor', None)

        qkv_quant_mode = quant_mode
        if self.quant_mode.has_act_and_weight_quant():
            # We need to hijack quant_mode for QKV because QKV always uses per channel scaling
            qkv_quant_mode = QuantMode.from_description(
                True, True, quant_mode.has_per_token_dynamic_scaling(), True)

        if self.quant_mode.has_int8_kv_cache():
            self.kv_cache_scaling_factor = Parameter(shape=(1, ),
                                                     dtype='float32')
        else:
            self.register_parameter('kv_cache_scaling_factor', None)

        self.qkv = SmoothQuantColumnLinear(
            hidden_size,
            hidden_size +
            2 * self.num_kv_heads * tp_size * self.attention_head_size,
            bias=(bias or qkv_bias_only),
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False,
            quant_mode=qkv_quant_mode)

        self.dense = SmoothQuantRowLinear(hidden_size,
                                          hidden_size,
                                          bias=bias,
                                          dtype=dtype,
                                          tp_group=tp_group,
                                          tp_size=tp_size,
                                          quant_mode=quant_mode)

        self.use_lora = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        medusa_packed_mask=None,
        medusa_position_offsets=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        encoder_output=None,
        position_embedding=None,
        norm_before_bmm1=False,
        lora_layer_params=None,
    ):
        assert lora_layer_params is None, "lora is not supported on SmoothQuantAttention now"
        # TODO add in-flight batching to SmoothQuant
        if default_net().plugin_config.smooth_quant_gemm_plugin:
            qkv = self.qkv(hidden_states)
        else:
            raise ValueError("smooth_quant_gemm_plugin is not set")

        alibi_slopes = None
        if self.position_embedding_type == PositionEmbeddingType.alibi:
            dtype = trt.float32
            if default_net().plugin_config.gpt_attention_plugin or default_net(
            ).plugin_config.inflight_batching_gpt_attention_plugin:
                dtype = hidden_states.dtype if self.quant_mode.has_act_static_scaling(
                ) else hidden_states[0].dtype
                if dtype == trt.int8:
                    dtype = trt.float16
            alibi_scale = 1. / self.norm_factor if self.scale_alibi_bias else 1.
            alibi_slopes = alibi_scale * generate_alibi_slopes(
                self.num_attention_heads * self.tp_size,
                dtype=dtype,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank)

        if default_net().plugin_config.gpt_attention_plugin:

            assert attention_params.is_valid(
                default_net().plugin_config.gpt_attention_plugin,
                default_net().plugin_config.remove_input_padding)
            assert kv_cache_params.is_valid(
                default_net().plugin_config.gpt_attention_plugin)
            assert self.attention_mask_type == AttentionMaskType.causal, \
                'Plugin only support masked MHA.'
            kv_quant_scale = constant(
                fp32_array([1.0])
            ) / self.kv_cache_scaling_factor.value if self.quant_mode.has_int8_kv_cache(
            ) else None
            kv_dequant_scale = self.kv_cache_scaling_factor.value if self.quant_mode.has_int8_kv_cache(
            ) else None
            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=kv_cache_params.get_first_past_key_value(),
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base,
                position_embedding_type=self.position_embedding_type,
                kv_orig_quant_scale=kv_quant_scale,
                kv_quant_orig_scale=kv_dequant_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                alibi_slopes=alibi_slopes,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                kv_cache_block_pointers=kv_cache_params.
                get_first_kv_cache_block_pointers(),
                host_kv_cache_block_pointers=kv_cache_params.
                get_first_host_kv_cache_block_pointers(),
                host_context_lengths=attention_params.host_context_lengths,
                enable_pos_shift=self.enable_pos_shift,
                dense_context_fmha=self.dense_context_fmha,
                medusa_position_offsets=medusa_position_offsets,
                medusa_packed_mask=medusa_packed_mask)
        else:
            assert self.paged_kv_cache == False

            def transpose_for_scores(x):
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), self.num_attention_heads,
                    self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query, key, value = split(qkv, self.hidden_size, dim=2)
            query = transpose_for_scores(query)
            key = transpose_for_scores(key)
            value = transpose_for_scores(value)

            past_key_value = kv_cache_params.get_first_past_key_value()
            if past_key_value is not None:

                def dequantize_tensor(x, scale):
                    # Cast from int8 to dtype
                    casted_x = cast(x, self.dtype)
                    return casted_x * scale

                if self.quant_mode.has_int8_kv_cache():
                    past_key_value = dequantize_tensor(
                        past_key_value, self.kv_dequantization_scale.value)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)
                key = concat([past_key, key], dim=2)
                value = concat([past_value, value], dim=2)

            def merge_caches():
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)
                return past_key_value

            if self.attention_mask_type == AttentionMaskType.causal:
                query_length = shape(query, 2)
                key_length = shape(key, 2)
                starts = concat([0, 0, key_length - query_length, 0])
                sizes = concat([1, 1, query_length, key_length])
                buffer = constant(
                    np.expand_dims(
                        np.tril(
                            np.ones(
                                (self.max_position_embeddings,
                                 self.max_position_embeddings))).astype(bool),
                        (0, 1)))
                causal_mask = slice(buffer, starts, sizes)

            key = key.permute([0, 1, 3, 2])
            with precision("float32"):
                attention_scores = matmul(query, key)

                if self.attention_mask_type == AttentionMaskType.causal:
                    attention_scores = where(causal_mask, attention_scores,
                                             -10000.0)

                attention_scores = attention_scores / self.norm_factor
                attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([shape(context, 0),
                        shape(context, 1), self.hidden_size]))

            past_key_value = merge_caches()

            if use_cache and self.quant_mode.has_int8_kv_cache():
                past_key_value = quantize_tensor(
                    past_key_value, self.kv_quantization_scale.value)
        value = cast(self.dense.smoother.value, context.dtype)
        context = context / value
        if self.quant_mode.has_act_and_weight_quant():
            if self.quant_mode.has_act_static_scaling():
                # Avoid quantiztion layers as it breaks int8 plugins
                context = quantize_tensor(
                    context, self.quantization_scaling_factor.value)
            else:
                # Quantize per token outputs tuple:
                # quantized tensor and scaling factors per token
                context = quantize_per_token(context)

        context = self.dense(context)

        if use_cache:
            return (context, past_key_value)

        return context
