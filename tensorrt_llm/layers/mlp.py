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
from ..functional import (ACT2FN, AllReduceFusionParams, cast, concat,
                          gemm_swiglu)
from ..module import Module
from ..quantization import QuantMode
from ..quantization.functional import quantize
from ..quantization.layers import FP8Linear, FP8RowLinear
from .linear import ColumnLinear, RowLinear
from .lora import LoraRuntimeParams
from .normalization import LayerNorm


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
                              tp_size=tp_size)

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.dtype = dtype
        self.bias = bias
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode
        self.eps = eps

    def forward(self, hidden_states, lora_layer_params=None, gegelu_limit=None):
        mlp_fc_lora_params = None
        if lora_layer_params is not None:
            mlp_fc_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_h_to_4h")

        mlp_proj_lora_params = None
        if lora_layer_params is not None:
            mlp_proj_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_4h_to_h")

        inter = self.fc(hidden_states, mlp_fc_lora_params)
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
                         eps=eps)

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
                reduce_fusion_params: Optional[AllReduceFusionParams] = None):

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
                           reduce_fusion_params=reduce_fusion_params)
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
                              tp_size=tp_size)

        # see optimize_model's add_lora for LoRA initialization
        self.lora = None

    def fc_gate_plugin(self, hidden_states, lora_layer_params=None):
        # Combine the following pattern
        #
        #   SiLU(FC(x)) + Gate(x)
        #
        # into:
        #
        #   SwiGLU(FusedFC(x))
        p_dtype = default_net().plugin_config.gemm_swiglu_plugin
        use_fp8 = p_dtype == 'fp8'
        assert use_fp8, "gemm_swiglu_plugin only supports fp8 now"

        if lora_layer_params is not None:
            mlp_fc_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_h_to_4h")
            mlp_gate_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_gate")

            if mlp_fc_lora_params is not None or mlp_gate_lora_params is not None:
                raise NotImplementedError(
                    f"LoRA not yet implemented for gemm_swiglu_plugin")

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

        inter = gemm_swiglu(hidden_states, self.fused_fc.weight.value, None,
                            scale_d0, scale_d1, scale_output)

        return inter

    def fc_gate(self, hidden_states, lora_layer_params=None):
        # Combine the following pattern
        #
        #   SiLU(FC(x)) + Gate(x)
        #
        # into:
        #
        #   SwiGLU(FusedFC(x))
        #
        # Upside is we don't need to modify 4 different weight loading paths just to concat weights

        inter = self.fused_fc(hidden_states)

        if lora_layer_params is not None:
            mlp_fc_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_h_to_4h")
            mlp_gate_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_gate")

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
                    host_context_lengths=mlp_fc_lora_params.
                    host_context_lengths,
                    max_context_length=mlp_fc_lora_params.max_context_length)

                mlp_fc_lora, mlp_gate_lora = self.lora(hidden_states,
                                                       mlp_in_lora_params)
                mlp_in_result = concat([mlp_gate_lora, mlp_fc_lora],
                                       dim=mlp_fc_lora.rank() - 1)
                inter = inter + mlp_in_result

        if self.hidden_act == 'silu':
            inter = ACT2FN['swiglu'](inter)
        elif self.hidden_act == 'gelu':
            inter = ACT2FN['geglu'](inter)
        else:
            raise NotImplementedError(
                f"Activation {self.hidden_act} not yet implemented for FusedGatedMLP"
            )
        return inter

    def forward(self,
                hidden_states,
                lora_layer_params=None,
                reduce_fusion_params: Optional[AllReduceFusionParams] = None):
        if default_net().plugin_config.gemm_swiglu_plugin:
            assert self.dtype == 'float16', f"Currently limited support, got {self.dtype}"
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
                           reduce_fusion_params=reduce_fusion_params)
        return output
