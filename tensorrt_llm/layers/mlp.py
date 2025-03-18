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

import tensorrt as trt

from .._common import default_net
from ..functional import (ACT2FN, AllReduceParams, cast, chunk, concat,
                          gemm_swiglu, is_gated_activation,
                          low_latency_gemm_swiglu)
from ..mapping import Mapping
from ..module import Module
from ..quantization import QuantMode
from ..quantization.functional import quantize
from ..quantization.layers import FP8Linear, FP8RowLinear
from .linear import ColumnLinear, RowLinear
from .lora import LoraRuntimeParams
from .normalization import LayerNorm


def fc_gate_lora(hidden_states, lora, fused_gate_up_lora, lora_layer_params):
    if lora_layer_params is not None:
        mlp_fc_lora_params = lora_layer_params.get_runtime_params(
            0, "mlp_h_to_4h")
        mlp_gate_lora_params = lora_layer_params.get_runtime_params(
            0, "mlp_gate")
        mlp_gate_up_lora_params = lora_layer_params.get_runtime_params(
            0, "mlp_gate_up")

        if mlp_gate_up_lora_params is not None:
            assert fused_gate_up_lora is not None
            mlp_gate_up_lora = fused_gate_up_lora(hidden_states,
                                                  mlp_gate_up_lora_params)
            return mlp_gate_up_lora

        elif mlp_fc_lora_params is not None and mlp_gate_lora_params is not None:
            mlp_in_lora_params = LoraRuntimeParams(
                lora_ranks=[
                    mlp_fc_lora_params.lora_ranks[0],
                    mlp_gate_lora_params.lora_ranks[0]
                ],
                lora_weights_pointers=[
                    mlp_fc_lora_params.lora_weights_pointers[0],
                    mlp_gate_lora_params.lora_weights_pointers[0]
                ],
                host_request_types=mlp_fc_lora_params.host_request_types,
                host_context_lengths=mlp_fc_lora_params.host_context_lengths)

            mlp_fc_lora, mlp_gate_lora = lora(hidden_states, mlp_in_lora_params)
            mlp_in_result = concat([mlp_gate_lora, mlp_fc_lora],
                                   dim=mlp_fc_lora.rank() - 1)
            return mlp_in_result
    return None


def fc_gate_dora(hidden_states, dora, fused_gate_up_dora, lora_layer_params):
    if lora_layer_params is not None:
        mlp_fc_lora_params = lora_layer_params.get_runtime_params(
            0, "mlp_h_to_4h")
        mlp_gate_lora_params = lora_layer_params.get_runtime_params(
            0, "mlp_gate")
        mlp_gate_up_lora_params = lora_layer_params.get_runtime_params(
            0, "mlp_gate_up")

        if mlp_gate_up_lora_params is not None:
            assert fused_gate_up_dora is not None
            return fused_gate_up_dora(hidden_states, mlp_gate_up_lora_params)

        if mlp_fc_lora_params is not None and mlp_gate_lora_params is not None:
            mlp_in_lora_params = LoraRuntimeParams(
                lora_ranks=[
                    mlp_fc_lora_params.lora_ranks[0],
                    mlp_gate_lora_params.lora_ranks[0]
                ],
                lora_weights_pointers=[
                    mlp_fc_lora_params.lora_weights_pointers[0],
                    mlp_gate_lora_params.lora_weights_pointers[0]
                ],
                host_request_types=mlp_fc_lora_params.host_request_types,
                host_context_lengths=mlp_fc_lora_params.host_context_lengths)

            return dora(hidden_states, mlp_in_lora_params)
    return None


class MLP(Module):

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
        inner_layernorm=False,
        eps=1e-05,
        is_expert=False,
    ):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        fc_output_size = 2 * ffn_hidden_size if hidden_act in [
            'swiglu', 'gegelu'
        ] else ffn_hidden_size
        self.inner_layernorm = LayerNorm(ffn_hidden_size, dtype=dtype,
                                         eps=eps) if inner_layernorm else None

        self.fc = ColumnLinear(hidden_size,
                               fc_output_size,
                               bias=bias,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size,
                               gather_output=False)
        self.proj = RowLinear(ffn_hidden_size,
                              hidden_size,
                              bias=bias,
                              dtype=dtype,
                              tp_group=tp_group,
                              tp_size=tp_size,
                              is_expert=is_expert)

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.dtype = dtype
        self.bias = bias
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode
        self.eps = eps
        self.is_expert = is_expert
        # see optimize_model's add_lora for LoRA initialization
        self.lora = None
        self.dora = None

    def forward(self, hidden_states, lora_layer_params=None, gegelu_limit=None):
        if lora_layer_params is not None:
            assert lora_layer_params.get_runtime_params(
                0, "mlp_gate_up"
            ) is None, f"LoRA module 'mlp_gate_up' is not supported in {self}"
        if is_gated_activation(self.hidden_act):
            inter = self.fc(hidden_states)
            lora_result = fc_gate_lora(hidden_states, self.lora, None,
                                       lora_layer_params)
            if lora_result is not None:
                inter = inter + lora_result
                if self.dora is not None:
                    inter = fc_gate_dora(inter, self.dora,
                                         self.fused_gate_up_dora,
                                         lora_layer_params)
        else:
            mlp_fc_lora_params = None
            if lora_layer_params is not None:
                mlp_fc_lora_params = lora_layer_params.get_runtime_params(
                    0, "mlp_h_to_4h")
            inter = self.fc(hidden_states, mlp_fc_lora_params)

        mlp_proj_lora_params = None
        if lora_layer_params is not None:
            mlp_proj_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_4h_to_h")

        if self.hidden_act == 'gegelu':
            inter = ACT2FN[self.hidden_act](inter, gegelu_limit)
        else:
            inter = ACT2FN[self.hidden_act](inter)
        if self.inner_layernorm is not None:
            inter = self.inner_layernorm(inter)
        output = self.proj(inter, lora_runtime_params=mlp_proj_lora_params)
        return output


class GatedMLP(MLP):

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
        inner_layernorm=False,
        eps=1e-05,
        is_expert=False,
    ):
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         quant_mode=quant_mode,
                         inner_layernorm=inner_layernorm,
                         eps=eps,
                         is_expert=is_expert)

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.tp_group = tp_group
        self.tp_size = tp_size

        self.gate = ColumnLinear(hidden_size,
                                 ffn_hidden_size,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=tp_group,
                                 tp_size=tp_size,
                                 gather_output=False)

    def forward(self,
                hidden_states,
                lora_layer_params=None,
                all_reduce_params: Optional[AllReduceParams] = None):
        if lora_layer_params is not None:
            assert lora_layer_params.get_runtime_params(
                0, "mlp_gate_up"
            ) is None, f"LoRA module 'mlp_gate_up' is not supported in {self}"

        mlp_fc_lora_params = None
        if lora_layer_params is not None:
            mlp_fc_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_h_to_4h")

        mlp_gate_lora_params = None
        if lora_layer_params is not None:
            mlp_gate_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_gate")

        mlp_proj_lora_params = None
        if lora_layer_params is not None:
            mlp_proj_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_4h_to_h")

        inter = self.fc(hidden_states, mlp_fc_lora_params)
        inter = ACT2FN[self.hidden_act](inter)
        gate = self.gate(hidden_states, mlp_gate_lora_params)
        intermediate = inter * gate
        if self.inner_layernorm is not None:
            intermediate = self.inner_layernorm(intermediate)
        output = self.proj(intermediate,
                           lora_runtime_params=mlp_proj_lora_params,
                           all_reduce_params=all_reduce_params)
        return output


class FusedGatedMLP(Module):

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
        inner_layernorm=False,
        eps=1e-05,
        is_expert=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.bias = bias
        self.dtype = dtype
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode

        self.fused_fc = ColumnLinear(
            self.hidden_size,
            self.ffn_hidden_size * 2,
            bias=self.bias,
            dtype=self.dtype,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
            gather_output=False,
        )
        self.inner_layernorm = LayerNorm(ffn_hidden_size, dtype=dtype,
                                         eps=eps) if inner_layernorm else None
        self.proj = RowLinear(ffn_hidden_size,
                              hidden_size,
                              bias=bias,
                              dtype=dtype,
                              tp_group=tp_group,
                              tp_size=tp_size,
                              is_expert=is_expert)

        # see optimize_model's add_lora for LoRA initialization
        self.lora = None  # used for split up and gate proj
        self.fused_gate_up_lora = None  # used for merged up_gate proj
        self.dora = None
        self.fused_gate_up_dora = None

    def fc_gate_plugin(self, hidden_states, lora_layer_params=None):
        # Combine the following pattern
        #
        #   SiLU(FC(x)) * Gate(x)
        #
        # into:
        #
        #   SwiGLU(FusedFC(x))
        if default_net(
        ).plugin_config.low_latency_gemm_swiglu_plugin is not None:
            p_dtype = default_net().plugin_config.low_latency_gemm_swiglu_plugin
        else:
            p_dtype = default_net().plugin_config.gemm_swiglu_plugin
        use_fp8 = p_dtype == 'fp8'
        assert use_fp8, "gemm_swiglu_plugin and low_latency_gemm_swiglu_plugin only supports fp8 now"

        if lora_layer_params is not None:
            mlp_fc_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_h_to_4h")
            mlp_gate_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_gate")

            if mlp_fc_lora_params is not None or mlp_gate_lora_params is not None:
                raise NotImplementedError(
                    f"LoRA of splitting fc and gate is not yet implemented for gemm_swiglu_plugin"
                )

        if self.hidden_act != 'silu':
            raise NotImplementedError(
                f"Activation {self.hidden_act} not yet implemented for gemm_swiglu_plugin"
            )

        if self.bias:
            raise NotImplementedError(
                f"bias not yet implemented for gemm_swiglu_plugin fp8")

        assert isinstance(
            self.fused_fc,
            FP8Linear), "fp8 gemm_swiglu only supports fp8 weights"
        assert isinstance(
            self.proj,
            FP8RowLinear), "fp8 gemm_swiglu only supports fp8 weights"
        assert self.fused_fc.weight.shape == (
            self.hidden_size, self.ffn_hidden_size * 2 //
            self.tp_size), "fp8 gemm_swiglu only supports (k, n) weights"

        scale_d0 = (self.fused_fc.weights_scaling_factor.raw_value.item() *
                    self.fused_fc.activation_scaling_factor.raw_value.item())
        scale_d1 = scale_d0
        scale_output = 1.0 / self.proj.activation_scaling_factor.raw_value.item(
        )
        activation_scaling_factor = cast(
            self.fused_fc.activation_scaling_factor.value, self.dtype)
        if hidden_states.dtype != trt.fp8:
            hidden_states = quantize(hidden_states, activation_scaling_factor,
                                     'fp8')

        if default_net(
        ).plugin_config.low_latency_gemm_swiglu_plugin is not None:
            inter = low_latency_gemm_swiglu(hidden_states,
                                            self.fused_fc.weight.value,
                                            scale_d0, scale_d1, scale_output)
        else:
            inter = gemm_swiglu(hidden_states, self.fused_fc.weight.value, None,
                                scale_d0, scale_d1, scale_output)

        lora_result = fc_gate_lora(hidden_states, self.lora,
                                   self.fused_gate_up_lora, lora_layer_params)
        if lora_result is not None:
            inter = inter + lora_result

        return inter

    def fc_gate(self, hidden_states, lora_layer_params=None):
        # Combine the following pattern
        #
        #   SiLU(FC(x)) * Gate(x)
        #
        # into:
        #
        #   SwiGLU(FusedFC(x))
        #
        # Upside is we don't need to modify 4 different weight loading paths just to concat weights

        inter = self.fused_fc(hidden_states)

        lora_result = fc_gate_lora(hidden_states, self.lora,
                                   self.fused_gate_up_lora, lora_layer_params)
        if lora_result is not None:
            inter = inter + lora_result
            if self.dora is not None:
                inter = fc_gate_dora(inter, self.dora, self.fused_gate_up_lora,
                                     lora_layer_params)

        if self.hidden_act == 'silu':
            inter = ACT2FN['swiglu'](inter)
        elif self.hidden_act == 'gelu':
            inter = ACT2FN['geglu'](inter)
        else:
            raise NotImplementedError(
                f"Activation {self.hidden_act} not yet implemented for {self.__class__.__name__}."
            )
        return inter

    def forward(self,
                hidden_states,
                lora_layer_params=None,
                all_reduce_params: Optional[AllReduceParams] = None):
        if default_net().plugin_config.gemm_swiglu_plugin or default_net(
        ).plugin_config.low_latency_gemm_swiglu_plugin:
            inter = self.fc_gate_plugin(hidden_states, lora_layer_params)
        else:
            inter = self.fc_gate(hidden_states, lora_layer_params)

        if self.inner_layernorm is not None:
            inter = self.inner_layernorm(inter)

        mlp_proj_lora_params = None
        if lora_layer_params is not None:
            mlp_proj_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_4h_to_h")
        output = self.proj(inter,
                           lora_runtime_params=mlp_proj_lora_params,
                           all_reduce_params=all_reduce_params)
        return output


class LinearGELU(Module):

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 approximate: str = 'tanh',
                 bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.proj = ColumnLinear(dim_in,
                                 dim_out,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=mapping.tp_group,
                                 tp_size=mapping.tp_size)
        if approximate != 'tanh':
            raise NotImplementedError('GELU only support tanh now.')

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = ACT2FN['gelu_pytorch_tanh'](hidden_states)
        return hidden_states


class LinearGEGLU(Module):

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 approximate: str = 'tanh',
                 bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.proj = ColumnLinear(dim_in,
                                 dim_out * 2,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=mapping.tp_group,
                                 tp_size=mapping.tp_size)
        if approximate != 'tanh':
            raise NotImplementedError('GELU only support tanh now.')

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = chunk(hidden_states,
                                    2,
                                    dim=(hidden_states.ndim() - 1))
        return hidden_states * ACT2FN['gelu_pytorch_tanh'](gate)


class LinearApproximateGELU(Module):

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.proj = ColumnLinear(dim_in,
                                 dim_out,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=mapping.tp_group,
                                 tp_size=mapping.tp_size)

    def forward(self, x):
        x = self.proj(x)
        return x * ACT2FN['sigmoid'](1.702 * x)


class LinearSwiGLU(Module):

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()

        self.proj = ColumnLinear(dim_in,
                                 dim_out * 2,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=mapping.tp_group,
                                 tp_size=mapping.tp_size)
        self.hidden_act = 'silu'

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = chunk(hidden_states,
                                    2,
                                    dim=(hidden_states.ndim() - 1))
        return hidden_states * ACT2FN[self.hidden_act](gate)


class LinearActivation(Module):

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 bias: bool = True,
                 activation: str = "silu",
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()

        self.proj = ColumnLinear(dim_in,
                                 dim_out,
                                 bias=bias,
                                 dtype=dtype,
                                 tp_group=mapping.tp_group,
                                 tp_size=mapping.tp_size)
        self.hidden_act = activation

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return ACT2FN[self.activation](hidden_states)
