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

from ..functional import ACT2FN, concat
from ..module import Module
from ..quantization import QuantMode
from ..quantization.layers import FP8Linear, FP8RowLinear
from .linear import ColumnLinear, RowLinear
from .lora import Lora, LoraRuntimeParams


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
            max_lora_rank=None,
    ):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        fc_output_size = 2 * ffn_hidden_size if hidden_act == 'swiglu' else ffn_hidden_size
        self.use_fp8_qdq = quant_mode.has_fp8_qdq()

        if self.use_fp8_qdq:
            self.fc = FP8Linear(hidden_size,
                                fc_output_size,
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False,
                                max_lora_rank=max_lora_rank)
            self.proj = FP8RowLinear(ffn_hidden_size,
                                     hidden_size,
                                     bias=bias,
                                     dtype=dtype,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     max_lora_rank=max_lora_rank)
        else:
            self.fc = ColumnLinear(hidden_size,
                                   fc_output_size,
                                   bias=bias,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size,
                                   gather_output=False,
                                   max_lora_rank=max_lora_rank)
            self.proj = RowLinear(ffn_hidden_size,
                                  hidden_size,
                                  bias=bias,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  max_lora_rank=max_lora_rank)

        self.hidden_act = hidden_act
        self.dtype = dtype
        self.bias = bias

    def forward(self, hidden_states, lora_layer_params=None):
        mlp_fc_lora_params = None
        if lora_layer_params is not None:
            mlp_fc_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_h_to_4h")

        mlp_proj_lora_params = None
        if lora_layer_params is not None:
            mlp_proj_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_4h_to_h")

        inter = self.fc(hidden_states, mlp_fc_lora_params)
        inter = ACT2FN[self.hidden_act](inter)
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

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.bias = bias
        self.dtype = dtype
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode
        self.max_lora_rank = max_lora_rank

        if self.use_fp8_qdq:
            self.gate = FP8Linear(hidden_size,
                                  ffn_hidden_size,
                                  bias=bias,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  gather_output=False,
                                  max_lora_rank=max_lora_rank)
        else:
            self.gate = ColumnLinear(hidden_size,
                                     ffn_hidden_size,
                                     bias=bias,
                                     dtype=dtype,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     gather_output=False,
                                     max_lora_rank=max_lora_rank)

    def forward(self, hidden_states, lora_layer_params=None):

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
        output = self.proj(intermediate,
                           lora_runtime_params=mlp_proj_lora_params)
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
            max_lora_rank=None,
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
        self.use_fp8_qdq = quant_mode.has_fp8_qdq()

        if self.use_fp8_qdq:
            self.fused_fc = FP8Linear(
                self.hidden_size,
                self.ffn_hidden_size * 2,
                bias=self.bias,
                dtype=self.dtype,
                tp_group=self.tp_group,
                tp_size=self.tp_size,
                gather_output=False,
            )
            self.proj = FP8RowLinear(ffn_hidden_size,
                                     hidden_size,
                                     bias=bias,
                                     dtype=dtype,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     max_lora_rank=max_lora_rank)
        else:
            self.fused_fc = ColumnLinear(
                self.hidden_size,
                self.ffn_hidden_size * 2,
                bias=self.bias,
                dtype=self.dtype,
                tp_group=self.tp_group,
                tp_size=self.tp_size,
                gather_output=False,
            )
            self.proj = RowLinear(ffn_hidden_size,
                                  hidden_size,
                                  bias=bias,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  max_lora_rank=max_lora_rank)

        if max_lora_rank is None:
            max_lora_rank = min(hidden_size, ffn_hidden_size // tp_size)
        self.mlp_in_lora = Lora(
            in_hidden_size=hidden_size,
            out_hidden_sizes=[
                ffn_hidden_size // tp_size, ffn_hidden_size // tp_size
            ],
            max_low_rank=max_lora_rank,
        )

    def forward(self, hidden_states, lora_layer_params=None):
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

                mlp_fc_lora, mlp_gate_lora = self.mlp_in_lora(
                    hidden_states, mlp_in_lora_params)
                mlp_in_result = concat([mlp_gate_lora, mlp_fc_lora],
                                       dim=mlp_fc_lora.rank() - 1)
                inter = inter + mlp_in_result

        if self.hidden_act == 'silu':
            inter = ACT2FN['swiglu'](inter)
        else:
            raise NotImplementedError(
                f"Activation {self.hidden_act} not yet implemented for FusedGatedMLP"
            )

        mlp_proj_lora_params = None
        if lora_layer_params is not None:
            mlp_proj_lora_params = lora_layer_params.get_runtime_params(
                0, "mlp_4h_to_h")
        output = self.proj(inter, lora_runtime_params=mlp_proj_lora_params)
        return output
