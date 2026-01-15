# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import ABC
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import check_accuracy

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import BaseMoeRoutingMethod
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def get_test_quant_params(quant_algo, x):
    """
    Create quantization configuration and corresponding kwargs for testing.
    """
    quantize_util_cls = None
    quant_config = None
    quant_kwargs = {}
    if quant_algo is None:
        quantize_util_cls = BaseQuantizeUtil
    elif quant_algo == QuantAlgo.FP8:
        quantize_util_cls = FP8QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
        _, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
        x_scale = x_scale.float().squeeze()
        quant_kwargs["x_scale"] = x_scale
    elif quant_algo == QuantAlgo.NVFP4:
        quantize_util_cls = NVFP4QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
        x_sf_global = (448 * 6) / x.abs().max().float()
        scaling_vector_size = 16
        quant_kwargs["scaling_vector_size"] = scaling_vector_size
        quant_kwargs["x_sf_global"] = x_sf_global
    else:
        assert False, "unsupported quant_algo"

    return quantize_util_cls, quant_config, quant_kwargs


class RefGatedMLPFusedMoE(nn.Module):
    """
    RefGatedMLPFusedMoE serves as a reference implementation with Gated MLPs designed for correctness testing.
    It utilizes derived classes to provide extensible support for various quantization algorithms.
    """

    def __init__(
        self,
        num_experts: int,
        routing_method: BaseMoeRoutingMethod,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        model_config: Optional[ModelConfig] = None,
        bias=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.dtype = dtype
        if model_config is None:
            model_config = ModelConfig()
        self.quant_config = model_config.quant_config
        self.experts = nn.ModuleList(
            [
                GatedMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    bias=bias,
                    dtype=self.dtype,
                    config=model_config,
                    use_cute_dsl_blockscaling_mm=False,
                    activation=F.silu,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size
        hidden_states = hidden_states.view(-1, self.hidden_size)
        selected_experts, routing_weights = self.routing_method.apply(router_logits)
        final_hidden_states = torch.zeros(
            hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        for expert_id in range(self.num_experts):
            if not torch.any(selected_experts == expert_id):
                continue
            batch_idx, nth_expert = torch.where(selected_experts == expert_id)
            expert_inputs = hidden_states[batch_idx]
            output = self.experts[expert_id](expert_inputs)
            final_hidden_states[batch_idx] += (
                routing_weights[batch_idx, nth_expert, None] * output.float()
            )
        final_hidden_states = final_hidden_states.reshape(hidden_states.shape)
        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]
        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]
            gate_up_proj_weights[0]["weight"] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]["weight"] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]["weight"] = weights[f"{expert}.w2.weight"]
            if self.bias:
                gate_up_proj_weights[0]["bias"] = weights[f"{expert}.w1.bias"]
                gate_up_proj_weights[1]["bias"] = weights[f"{expert}.w3.bias"]
                down_proj_weights[0]["bias"] = weights[f"{expert}.w2.bias"]
            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

    def check_accuracy(self, output, ref_output):
        # Here we use same rtol and atol as test_fused_moe
        check_accuracy(output, ref_output, rtol=2e-1, atol=2e-1, percent=0.984)


class FP8RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """
    A derived class of RefGatedMLPFusedMoE serves as a reference implementation of FP8 quantization
    for correctness testing.
    """

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]
        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]
            gate_up_proj_weights[0]["weight"] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]["weight"] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]["weight"] = weights[f"{expert}.w2.weight"]
            if self.bias:
                gate_up_proj_weights[0]["bias"] = weights[f"{expert}.w1.bias"]
                gate_up_proj_weights[1]["bias"] = weights[f"{expert}.w3.bias"]
                down_proj_weights[0]["bias"] = weights[f"{expert}.w2.bias"]
            assert self.quant_config and self.quant_config.quant_algo == QuantAlgo.FP8
            gate_up_proj_weights[0]["weight_scale"] = weights[f"{expert}.w1.weight_scale"]
            gate_up_proj_weights[1]["weight_scale"] = weights[f"{expert}.w3.weight_scale"]
            down_proj_weights[0]["weight_scale"] = weights[f"{expert}.w2.weight_scale"]
            gate_up_proj_weights[0]["input_scale"] = weights[f"{expert}.w1.input_scale"]
            gate_up_proj_weights[1]["input_scale"] = weights[f"{expert}.w3.input_scale"]
            down_proj_weights[0]["input_scale"] = weights[f"{expert}.w2.input_scale"]
            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

    def check_accuracy(self, output, ref_output):
        # Here we use same rtol and atol as test_fused_moe
        check_accuracy(output, ref_output, rtol=4e-2, atol=1e-1, percent=0.99)


class NVFP4RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """
    A derived class of RefGatedMLPFusedMoE serves as a reference implementation of NVFP4 quantization
    for correctness testing.
    """

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]
        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]
            gate_up_proj_weights[0]["weight"] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]["weight"] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]["weight"] = weights[f"{expert}.w2.weight"]
            if self.bias:
                gate_up_proj_weights[0]["bias"] = weights[f"{expert}.w1.bias"]
                gate_up_proj_weights[1]["bias"] = weights[f"{expert}.w3.bias"]
                down_proj_weights[0]["bias"] = weights[f"{expert}.w2.bias"]
            assert self.quant_config and self.quant_config.quant_algo == QuantAlgo.NVFP4, (
                "expect quant_algo to be NVFP4 in load weights"
            )
            gate_up_proj_weights[0]["weight_scale"] = weights[f"{expert}.w1.weight_scale"]
            gate_up_proj_weights[1]["weight_scale"] = weights[f"{expert}.w3.weight_scale"]
            down_proj_weights[0]["weight_scale"] = weights[f"{expert}.w2.weight_scale"]
            gate_up_proj_weights[0]["input_scale"] = weights[f"{expert}.w1.input_scale"]
            gate_up_proj_weights[1]["input_scale"] = weights[f"{expert}.w3.input_scale"]
            down_proj_weights[0]["input_scale"] = weights[f"{expert}.w2.input_scale"]
            gate_up_proj_weights[0]["weight_scale_2"] = weights[f"{expert}.w1.weight_scale_2"]
            gate_up_proj_weights[1]["weight_scale_2"] = weights[f"{expert}.w3.weight_scale_2"]
            down_proj_weights[0]["weight_scale_2"] = weights[f"{expert}.w2.weight_scale_2"]
            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

    def check_accuracy(self, output, ref_output):
        # Here we use same rtol and atol as test_fused_moe
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.15)


class BaseQuantizeUtil(ABC):
    """
    BaseQuantizeUtil serves as a base class for MoE correctess testing which provides interface
    to create quantized weights and reference modules. It can be extended for different quantization algorithms.
    """

    def __init__(
        self,
        num_experts: int,
        dtype: torch.dtype,
        intermediate_size: int,
        hidden_size: int,
        quant_config: QuantConfig,
    ):
        self.num_experts = num_experts
        self.dtype = dtype
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.quant_config = quant_config

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts.
        """
        assert self.quant_config is None, "quant_config should be None for BaseQuantizeUtil"
        weights = {}
        for expert_id in range(self.num_experts):
            w1_weight = torch.randn(
                (self.intermediate_size, self.hidden_size), dtype=self.dtype, device="cuda"
            )
            w2_weight = torch.randn(
                (self.hidden_size, self.intermediate_size), dtype=self.dtype, device="cuda"
            )
            w3_weight = torch.randn(
                (self.intermediate_size, self.hidden_size), dtype=self.dtype, device="cuda"
            )

            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
        return weights

    def create_ref_module(self, routing_method, ref_cls=RefGatedMLPFusedMoE) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        ref_fused_moe = ref_cls(
            num_experts=self.num_experts,
            routing_method=routing_method,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
            model_config=ModelConfig(quant_config=self.quant_config),
        )
        return ref_fused_moe


class FP8QuantizeUtil(BaseQuantizeUtil):
    """
    FP8QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing for FP8 quantized MoE modules.
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts.
        """
        assert self.quant_config is not None and self.quant_config.quant_algo == QuantAlgo.FP8, (
            "expect quant_algo to be fp8"
        )
        weights = {}
        for expert_id in range(self.num_experts):
            w1_weight = torch.randn(
                (self.intermediate_size, self.hidden_size), dtype=self.dtype, device="cuda"
            )
            w2_weight = torch.randn(
                (self.hidden_size, self.intermediate_size), dtype=self.dtype, device="cuda"
            )
            w3_weight = torch.randn(
                (self.intermediate_size, self.hidden_size), dtype=self.dtype, device="cuda"
            )

            w1_weight_fp8, w1_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                w1_weight
            )
            w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()
            w2_weight_fp8, w2_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                w2_weight
            )
            w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()
            w3_weight_fp8, w3_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                w3_weight
            )
            w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

            assert "x_scale" in quant_kwargs, "x_scale is required for FP8 quant"
            x_scale = quant_kwargs["x_scale"]
            w1_input_scale = x_scale.cuda()
            w2_input_scale = x_scale.cuda()
            w3_input_scale = x_scale.cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
            weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
            weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
            weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale.float()
            weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale.float()
            weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale.float()
            weights[f"{expert_id}.w1.input_scale"] = w1_input_scale
            weights[f"{expert_id}.w2.input_scale"] = w2_input_scale
            weights[f"{expert_id}.w3.input_scale"] = w3_input_scale
        return weights

    def create_ref_module(self, routing_method, ref_cls=FP8RefGatedMLPFusedMoE) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class NVFP4QuantizeUtil(BaseQuantizeUtil):
    """
    NVFP4QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing for NVFP4 quantized MoE modules.
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts.
        """
        assert self.quant_config is not None and self.quant_config.quant_algo == QuantAlgo.NVFP4, (
            "expect quant_algo to be NVFP4"
        )
        weights = {}
        for expert_id in range(self.num_experts):
            w1_weight = (
                torch.randn(
                    (self.intermediate_size, self.hidden_size), dtype=self.dtype, device="cuda"
                )
                * 0.05
            )
            w2_weight = (
                torch.randn(
                    (self.hidden_size, self.intermediate_size), dtype=self.dtype, device="cuda"
                )
                * 0.05
            )
            w3_weight = (
                torch.randn(
                    (self.intermediate_size, self.hidden_size), dtype=self.dtype, device="cuda"
                )
                * 0.05
            )

            assert "scaling_vector_size" in quant_kwargs, (
                "scaling_vector_size is required for NVFP4 quant"
            )
            assert "x_sf_global" in quant_kwargs, "x_sf_global is required for NVFP4 quant"

            scaling_vector_size = quant_kwargs["scaling_vector_size"]
            x_sf_global = quant_kwargs["x_sf_global"]

            w1_sf_global = (448 * 6) / w1_weight.abs().max().float()
            w2_sf_global = (448 * 6) / w2_weight.abs().max().float()
            w3_sf_global = (448 * 6) / w3_weight.abs().max().float()

            w3_w1_global = min(
                w1_sf_global, w3_sf_global
            )  # w3 global and w1 global must be the same

            # start to quantize
            w1_weight_nvfp4, w1_sf_block_unswizzled = torch.ops.trtllm.fp4_quantize(
                w1_weight, w3_w1_global, scaling_vector_size, False, False
            )
            w1_sf_block_unswizzled = w1_sf_block_unswizzled.view(self.intermediate_size, -1)

            w2_weight_nvfp4, w2_sf_block_unswizzled = torch.ops.trtllm.fp4_quantize(
                w2_weight, w2_sf_global, scaling_vector_size, False, False
            )
            w2_sf_block_unswizzled = w2_sf_block_unswizzled.view(self.hidden_size, -1)

            w3_weight_nvfp4, w3_sf_block_unswizzled = torch.ops.trtllm.fp4_quantize(
                w3_weight, w3_w1_global, scaling_vector_size, False, False
            )
            w3_sf_block_unswizzled = w3_sf_block_unswizzled.view(self.intermediate_size, -1)

            w1_input_scale = x_sf_global.cuda()
            w2_input_scale = x_sf_global.cuda()
            w3_input_scale = x_sf_global.cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4
            weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4
            weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4
            weights[f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
                torch.float8_e4m3fn
            ).cuda()
            weights[f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
                torch.float8_e4m3fn
            ).cuda()
            weights[f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
                torch.float8_e4m3fn
            ).cuda()
            weights[f"{expert_id}.w1.input_scale"] = 1.0 / w1_input_scale
            weights[f"{expert_id}.w2.input_scale"] = 1.0 / w2_input_scale
            weights[f"{expert_id}.w3.input_scale"] = 1.0 / w3_input_scale
            weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_global
            weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf_global
            weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_global
        return weights

    def create_ref_module(
        self, routing_method, ref_cls=NVFP4RefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)
