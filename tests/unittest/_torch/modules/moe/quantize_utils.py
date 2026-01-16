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
from _torch.helpers import calc_woq_tolerence, per_block_cast_to_fp8
from utils.util import check_accuracy

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import BaseMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def round_up(x, alignment):
    return (x + alignment - 1) // alignment * alignment


def dist_to_alignment(size, alignment):
    return round_up(size, alignment) - size


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
        quant_kwargs["x_sf_global"] = x_sf_global
    elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
        quantize_util_cls = FP8BlockScalesQuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    elif quant_algo == QuantAlgo.W4A8_NVFP4_FP8:
        quantize_util_cls = W4A8NVFP4FP8QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_NVFP4_FP8)
        x_sf_global = 448 / x.abs().max().float()
        quant_kwargs["x_sf_global"] = x_sf_global
    elif quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
        quantize_util_cls = MXFP4MXFP8QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
    elif quant_algo == QuantAlgo.W4A16_MXFP4:
        quantize_util_cls = WFP4A16QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4)
    elif quant_algo == QuantAlgo.W8A16:
        quantize_util_cls = W8A16QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.W8A16)
    elif quant_algo == QuantAlgo.W4A8_AWQ:
        quantize_util_cls = W4A8AWQQuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)
    else:
        raise ValueError(f"unsupported quant_algo: {quant_algo}")

    return quantize_util_cls, quant_config, quant_kwargs


class RefGatedMLPFusedMoE(nn.Module):
    """
    RefGatedMLPFusedMoE serves as a reference implementation with Gated MLPs designed for correctness testing.
    It utilizes derived classes to provide extensible support for various quantization algorithms.

    Subclasses can override `scale_keys` to specify which scale fields to load:
        - "weight_scale": weight quantization scale
        - "input_scale": input activation scale
        - "weight_scale_2": secondary weight scale (for NVFP4-like quantization)
    """

    # Scale keys to load for this quantization method (subclasses override this)
    scale_keys: List[str] = []
    # Expected quantization algorithm (subclasses override this)
    expected_quant_algo: Optional[QuantAlgo] = None

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

    def _load_expert_weights_with_scales(self, weights: Dict, expert: int):
        """Load weights for a single expert with configured scale keys."""
        gate_up_proj_weights = [{}, {}]
        down_proj_weights = [{}]

        # Load base weights
        gate_up_proj_weights[0]["weight"] = weights[f"{expert}.w1.weight"]
        gate_up_proj_weights[1]["weight"] = weights[f"{expert}.w3.weight"]
        down_proj_weights[0]["weight"] = weights[f"{expert}.w2.weight"]

        # Load bias if enabled
        if self.bias:
            gate_up_proj_weights[0]["bias"] = weights[f"{expert}.w1.bias"]
            gate_up_proj_weights[1]["bias"] = weights[f"{expert}.w3.bias"]
            down_proj_weights[0]["bias"] = weights[f"{expert}.w2.bias"]

        # Load scale keys defined by subclass
        for scale_key in self.scale_keys:
            gate_up_proj_weights[0][scale_key] = weights[f"{expert}.w1.{scale_key}"]
            gate_up_proj_weights[1][scale_key] = weights[f"{expert}.w3.{scale_key}"]
            down_proj_weights[0][scale_key] = weights[f"{expert}.w2.{scale_key}"]

        self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
        self.experts[expert].down_proj.load_weights(down_proj_weights)

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]

        # Validate quant_algo if expected
        if self.expected_quant_algo is not None:
            assert self.quant_config and self.quant_config.quant_algo == self.expected_quant_algo, (
                f"expect quant_algo to be {self.expected_quant_algo}"
            )

        for expert in range(self.num_experts):
            self._load_expert_weights_with_scales(weights, expert)

    def check_accuracy(self, output, ref_output):
        # Here we use same rtol and atol as test_fused_moe
        check_accuracy(output, ref_output, rtol=2e-1, atol=2e-1, percent=0.984)


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


class FP8RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of FP8 quantization for correctness testing."""

    scale_keys = ["weight_scale", "input_scale"]
    expected_quant_algo = QuantAlgo.FP8

    def check_accuracy(self, output, ref_output):
        check_accuracy(output, ref_output, rtol=4e-2, atol=1e-1, percent=0.99)


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
        bias = quant_kwargs.get("bias", False)
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

            if bias:
                weights[f"{expert_id}.w1.bias"] = torch.randn(
                    (self.intermediate_size,), dtype=self.dtype, device="cuda"
                )
                weights[f"{expert_id}.w2.bias"] = torch.randn(
                    (self.hidden_size,), dtype=self.dtype, device="cuda"
                )
                weights[f"{expert_id}.w3.bias"] = torch.randn(
                    (self.intermediate_size,), dtype=self.dtype, device="cuda"
                )
        return weights

    def create_ref_module(self, routing_method, ref_cls=FP8RefGatedMLPFusedMoE) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class NVFP4RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of NVFP4 quantization for correctness testing."""

    scale_keys = ["weight_scale", "input_scale", "weight_scale_2"]
    expected_quant_algo = QuantAlgo.NVFP4

    def check_accuracy(self, output, ref_output):
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.15)


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
        bias = quant_kwargs.get("bias", False)
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

            assert "x_sf_global" in quant_kwargs, "x_sf_global is required for NVFP4 quant"

            scaling_vector_size = quant_kwargs.get("scaling_vector_size", 16)
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

            # Note: NVFP4 bias uses torch.float dtype (following test_fused_moe.py gptoss_style)
            if bias:
                weights[f"{expert_id}.w1.bias"] = torch.randn(
                    self.intermediate_size, device="cuda", dtype=torch.float
                )
                weights[f"{expert_id}.w2.bias"] = torch.randn(
                    self.hidden_size, device="cuda", dtype=torch.float
                )
                weights[f"{expert_id}.w3.bias"] = torch.randn(
                    self.intermediate_size, device="cuda", dtype=torch.float
                )
        return weights

    def create_ref_module(
        self, routing_method, ref_cls=NVFP4RefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class FP8BlockScalesRefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of FP8 block-wise quantization for correctness testing."""

    scale_keys = ["weight_scale"]
    expected_quant_algo = QuantAlgo.FP8_BLOCK_SCALES

    def check_accuracy(self, output, ref_output):
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


class FP8BlockScalesQuantizeUtil(BaseQuantizeUtil):
    """
    FP8BlockScalesQuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for FP8 block-wise quantized MoE modules.
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts using FP8 block-wise quantization.
        """
        assert (
            self.quant_config is not None
            and self.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        ), "expect quant_algo to be FP8_BLOCK_SCALES"
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

            w1_weight_fp8, w1_weight_scale = per_block_cast_to_fp8(w1_weight)
            w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w2_weight_fp8, w2_weight_scale = per_block_cast_to_fp8(w2_weight)
            w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w3_weight_fp8, w3_weight_scale = per_block_cast_to_fp8(w3_weight)
            w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
            weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
            weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
            weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale
            # Also add weight_scale_inv for compatibility with some loaders
            weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale
        return weights

    def create_ref_module(
        self, routing_method, ref_cls=FP8BlockScalesRefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class W4A8NVFP4FP8RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of W4A8_NVFP4_FP8 quantization for correctness testing."""

    scale_keys = ["weight_scale", "input_scale", "weight_scale_2"]
    expected_quant_algo = QuantAlgo.W4A8_NVFP4_FP8

    def check_accuracy(self, output, ref_output):
        torch.testing.assert_close(output, ref_output, rtol=1e-1, atol=0.5)


class W4A8NVFP4FP8QuantizeUtil(BaseQuantizeUtil):
    """
    W4A8NVFP4FP8QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for W4A8_NVFP4_FP8 quantized MoE modules.
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts using W4A8_NVFP4_FP8 quantization.
        """
        assert (
            self.quant_config is not None
            and self.quant_config.quant_algo == QuantAlgo.W4A8_NVFP4_FP8
        ), "expect quant_algo to be W4A8_NVFP4_FP8"
        assert "x_sf_global" in quant_kwargs, "x_sf_global is required for W4A8_NVFP4_FP8 quant"

        scaling_vector_size = quant_kwargs.get("scaling_vector_size", 32)
        x_sf_global = quant_kwargs["x_sf_global"]

        weights = {}
        for expert_id in range(self.num_experts):
            w1_weight = torch.randn(
                (self.intermediate_size, self.hidden_size), dtype=torch.float32, device="cpu"
            )
            w2_weight = torch.randn(
                (self.hidden_size, self.intermediate_size), dtype=torch.float32, device="cpu"
            )
            w3_weight = torch.randn(
                (self.intermediate_size, self.hidden_size), dtype=torch.float32, device="cpu"
            )

            w1_sf_global = 448 / w1_weight.abs().max().float()
            w2_sf_global = 448 / w2_weight.abs().max().float()
            w3_sf_global = 448 / w3_weight.abs().max().float()

            w3_w1_global = min(w1_sf_global, w3_sf_global)

            w1_weight_nvfp4, w1_sf_block, _ = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
                w1_weight * w3_w1_global, scaling_vector_size, 1, False
            )
            w1_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w1_sf_block.view(self.intermediate_size, -1)
            )

            w2_weight_nvfp4, w2_sf_block, _ = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
                w2_weight * w2_sf_global, scaling_vector_size, 1, False
            )
            w2_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w2_sf_block.view(self.hidden_size, -1)
            )

            w3_weight_nvfp4, w3_sf_block, _ = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
                w3_weight * w3_w1_global, scaling_vector_size, 1, False
            )
            w3_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w3_sf_block.view(self.intermediate_size, -1)
            )

            w1_input_scale = x_sf_global.cuda()
            w2_input_scale = x_sf_global.cuda()
            w3_input_scale = x_sf_global.cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4.cuda()
            weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4.cuda()
            weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4.cuda()
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
        self, routing_method, ref_cls=W4A8NVFP4FP8RefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class MXFP4MXFP8RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of W4A8_MXFP4_MXFP8 quantization for correctness testing."""

    scale_keys = ["weight_scale"]
    expected_quant_algo = QuantAlgo.W4A8_MXFP4_MXFP8

    def check_accuracy(self, output, ref_output):
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.15)


class MXFP4MXFP8QuantizeUtil(BaseQuantizeUtil):
    """
    MXFP4MXFP8QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for W4A8_MXFP4_MXFP8 quantized MoE modules.
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts using W4A8_MXFP4_MXFP8 quantization.
        """
        assert (
            self.quant_config is not None
            and self.quant_config.quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8
        ), "expect quant_algo to be W4A8_MXFP4_MXFP8"

        scaling_vector_size = quant_kwargs.get("scaling_vector_size", 32)
        hidden_size_in = quant_kwargs.get("hidden_size_in", self.hidden_size)
        hidden_size_out = quant_kwargs.get("hidden_size_out", self.hidden_size)
        intermediate_size = quant_kwargs.get("intermediate_size", self.intermediate_size)
        hidden_size_unpadded = quant_kwargs.get("hidden_size_unpadded", self.hidden_size)
        intermediate_size_unpadded = quant_kwargs.get(
            "intermediate_size_unpadded", self.intermediate_size
        )
        bias = quant_kwargs.get("bias", False)
        pad_zero_or_val = quant_kwargs.get("pad_zero_or_val", True)
        weight_alignment = quant_kwargs.get("weight_alignment", 128)
        input_hidden_alignment = quant_kwargs.get("input_hidden_alignment", 512)

        # Ensures each call gives same outcome
        torch.manual_seed(42)
        # Contamination value
        contam_val = 42

        weights = {}
        for expert_id in range(self.num_experts):
            if bias:
                w1_bias = torch.randn((intermediate_size_unpadded,), dtype=self.dtype).cuda() * 0.1
                w2_bias = torch.randn((hidden_size_unpadded,), dtype=self.dtype).cuda() * 0.1
                w3_bias = torch.randn((intermediate_size_unpadded,), dtype=self.dtype).cuda() * 0.1
                # Pad to output dimension using contamination
                w1_bias = torch.nn.functional.pad(
                    w1_bias,
                    (0, dist_to_alignment(w1_bias.shape[-1], intermediate_size)),
                    "constant",
                    0 if pad_zero_or_val else contam_val,
                )
                w2_bias = torch.nn.functional.pad(
                    w2_bias,
                    (0, dist_to_alignment(hidden_size_unpadded, hidden_size_out)),
                    "constant",
                    0 if pad_zero_or_val else contam_val,
                )
                w3_bias = torch.nn.functional.pad(
                    w3_bias,
                    (0, dist_to_alignment(w3_bias.shape[-1], intermediate_size)),
                    "constant",
                    0 if pad_zero_or_val else contam_val,
                )
                weights[f"{expert_id}.w1.bias"] = w1_bias
                weights[f"{expert_id}.w2.bias"] = w2_bias
                weights[f"{expert_id}.w3.bias"] = w3_bias

            w1_weight = (
                torch.randn(
                    (intermediate_size_unpadded, hidden_size_unpadded),
                    dtype=self.dtype,
                ).cuda()
                * 0.1
            )
            w2_weight = (
                torch.randn(
                    (hidden_size_unpadded, intermediate_size_unpadded),
                    dtype=self.dtype,
                ).cuda()
                * 0.1
            )
            w3_weight = torch.randn(
                (intermediate_size_unpadded, hidden_size_unpadded),
                dtype=self.dtype,
            ).cuda()
            # First padding step: pad weight tensors from unpadded dimensions
            # to weight-aligned dimensions using 0s
            w1_weight = torch.nn.functional.pad(
                w1_weight,
                (
                    0,
                    dist_to_alignment(hidden_size_unpadded, input_hidden_alignment),
                    0,
                    dist_to_alignment(intermediate_size_unpadded, weight_alignment),
                ),
            )
            w2_weight = torch.nn.functional.pad(
                w2_weight,
                (
                    0,
                    dist_to_alignment(intermediate_size_unpadded, weight_alignment),
                ),
            )
            w3_weight = torch.nn.functional.pad(
                w3_weight,
                (
                    0,
                    dist_to_alignment(hidden_size_unpadded, input_hidden_alignment),
                    0,
                    dist_to_alignment(intermediate_size_unpadded, weight_alignment),
                ),
            )
            # Second padding step: pad from aligned dimensions to final dimensions
            # using contamination
            w1_weight = torch.nn.functional.pad(
                w1_weight,
                (
                    0,
                    dist_to_alignment(w1_weight.shape[-1], hidden_size_in),
                    0,
                    dist_to_alignment(w1_weight.shape[-2], intermediate_size),
                ),
                "constant",
                0 if pad_zero_or_val else contam_val,
            )
            w2_weight = torch.nn.functional.pad(
                w2_weight,
                (
                    0,
                    dist_to_alignment(w2_weight.shape[-1], intermediate_size),
                    0,
                    dist_to_alignment(w2_weight.shape[-2], hidden_size_out),
                ),
                "constant",
                0 if pad_zero_or_val else contam_val,
            )
            w3_weight = torch.nn.functional.pad(
                w3_weight,
                (
                    0,
                    dist_to_alignment(w3_weight.shape[-1], hidden_size_in),
                    0,
                    dist_to_alignment(w3_weight.shape[-2], intermediate_size),
                ),
                "constant",
                0 if pad_zero_or_val else contam_val,
            )

            w1_weight_mxfp4, w1_sf_block = torch.ops.trtllm.fp4_quantize(
                w1_weight, None, scaling_vector_size, True
            )
            w1_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w1_sf_block.cpu().view(intermediate_size, -1)
            )

            w2_weight_mxfp4, w2_sf_block = torch.ops.trtllm.fp4_quantize(
                w2_weight, None, scaling_vector_size, True
            )
            w2_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w2_sf_block.cpu().view(hidden_size_out, -1)
            )

            w3_weight_mxfp4, w3_sf_block = torch.ops.trtllm.fp4_quantize(
                w3_weight, None, scaling_vector_size, True
            )
            w3_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w3_sf_block.cpu().view(intermediate_size, -1)
            )

            weights[f"{expert_id}.w1.weight"] = w1_weight_mxfp4
            weights[f"{expert_id}.w2.weight"] = w2_weight_mxfp4
            weights[f"{expert_id}.w3.weight"] = w3_weight_mxfp4
            weights[f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
                torch.uint8
            ).cuda()
            weights[f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
                torch.uint8
            ).cuda()
            weights[f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
                torch.uint8
            ).cuda()
        return weights

    def create_ref_module(
        self, routing_method, ref_cls=MXFP4MXFP8RefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class WFP4A16RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """
    A derived class of RefGatedMLPFusedMoE serves as a reference implementation of W4A16_MXFP4
    quantization for correctness testing.

    Since GatedMLP doesn't support wfp4a16 quantization, we dequantize the weights
    in load_weights and use non-quantized forward.
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
        # Store the original quant_config for assertion in load_weights
        self._original_quant_config = model_config.quant_config if model_config else None
        # Create experts without quantization config since we'll dequantize weights
        super().__init__(
            num_experts=num_experts,
            routing_method=routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            model_config=ModelConfig(),  # No quant_config
            bias=bias,
        )

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]

        assert (
            self._original_quant_config
            and self._original_quant_config.quant_algo == QuantAlgo.W4A16_MXFP4
        ), "expect quant_algo to be W4A16_MXFP4"

        unpacker = torch.ops.trtllm.mxfp4_dequantize_unswizzled

        for expert in range(self.num_experts):
            # Get quantized weights and scales
            w1 = weights[f"{expert}.w1.weight"]
            s1 = weights.get(f"{expert}.w1.weight_scale_inv", weights[f"{expert}.w1.weight_scale"])
            w3 = weights[f"{expert}.w3.weight"]
            s3 = weights.get(f"{expert}.w3.weight_scale_inv", weights[f"{expert}.w3.weight_scale"])
            w2 = weights[f"{expert}.w2.weight"]
            s2 = weights.get(f"{expert}.w2.weight_scale_inv", weights[f"{expert}.w2.weight_scale"])

            # Calculate scaling_group_size from scale shape
            # scale shape is (out_features, in_features // scaling_group_size)
            scaling_group_size = self.hidden_size // s1.shape[-1]

            # Dequantize weights
            w1_dequant = (
                unpacker(w1.cpu(), s1.cpu(), scaling_group_size)
                .to(dtype=self.dtype, device="cuda")
                .T.contiguous()
            )
            w3_dequant = (
                unpacker(w3.cpu(), s3.cpu(), scaling_group_size)
                .to(dtype=self.dtype, device="cuda")
                .T.contiguous()
            )
            w2_dequant = (
                unpacker(w2.cpu(), s2.cpu(), scaling_group_size)
                .to(dtype=self.dtype, device="cuda")
                .T.contiguous()
            )

            # Load as regular weights (no scales)
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]
            gate_up_proj_weights[0]["weight"] = w1_dequant
            gate_up_proj_weights[1]["weight"] = w3_dequant
            down_proj_weights[0]["weight"] = w2_dequant
            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

    def check_accuracy(self, output, ref_output):
        # Here we use same rtol and atol as test_fused_moe_wfp4a16
        check_accuracy(output, ref_output, rtol=1e-2, atol=0.1, percent=0.99)


class WFP4A16QuantizeUtil(BaseQuantizeUtil):
    """
    WFP4A16QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for W4A16_MXFP4 quantized MoE modules.
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts using W4A16_MXFP4 quantization.
        """
        assert (
            self.quant_config is not None and self.quant_config.quant_algo == QuantAlgo.W4A16_MXFP4
        ), "expect quant_algo to be W4A16_MXFP4"

        scaling_group_size = quant_kwargs.get("scaling_group_size", 32)

        weights = {}
        for expert_id in range(self.num_experts):
            # MXFP4 weights are stored as uint8 with half the input dimension
            w1_weight = torch.randint(
                0,
                256,
                (self.intermediate_size, self.hidden_size // 2),
                dtype=torch.uint8,
                device="cuda",
            )
            w2_weight = torch.randint(
                0,
                256,
                (self.hidden_size, self.intermediate_size // 2),
                dtype=torch.uint8,
                device="cuda",
            )
            w3_weight = torch.randint(
                0,
                256,
                (self.intermediate_size, self.hidden_size // 2),
                dtype=torch.uint8,
                device="cuda",
            )

            # Scale tensors
            w1_scale = torch.randint(
                118,
                123,
                (self.intermediate_size, self.hidden_size // scaling_group_size),
                dtype=torch.uint8,
                device="cuda",
            )
            w2_scale = torch.randint(
                118,
                123,
                (self.hidden_size, self.intermediate_size // scaling_group_size),
                dtype=torch.uint8,
                device="cuda",
            )
            w3_scale = torch.randint(
                118,
                123,
                (self.intermediate_size, self.hidden_size // scaling_group_size),
                dtype=torch.uint8,
                device="cuda",
            )

            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
            # MXFP4WeightFusedMoEMethod
            weights[f"{expert_id}.w1.weight_scale"] = w1_scale
            weights[f"{expert_id}.w2.weight_scale"] = w2_scale
            weights[f"{expert_id}.w3.weight_scale"] = w3_scale
            # WFP4A16FusedMoEMethod
            weights[f"{expert_id}.w1.weight_scale_inv"] = w1_scale
            weights[f"{expert_id}.w2.weight_scale_inv"] = w2_scale
            weights[f"{expert_id}.w3.weight_scale_inv"] = w3_scale
        return weights

    def create_ref_module(
        self, routing_method, ref_cls=WFP4A16RefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class W8A16RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """
    A derived class of RefGatedMLPFusedMoE serves as a reference implementation of W8A16
    quantization for correctness testing.

    Since GatedMLP doesn't support W8A16 quantization, we dequantize the weights
    in load_weights and use non-quantized forward.
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
        # Store the original quant_config for assertion in load_weights
        self._original_quant_config = model_config.quant_config if model_config else None
        # Create experts without quantization config since we'll dequantize weights
        super().__init__(
            num_experts=num_experts,
            routing_method=routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            model_config=ModelConfig(),  # No quant_config
            bias=bias,
        )

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]

        assert (
            self._original_quant_config
            and self._original_quant_config.quant_algo == QuantAlgo.W8A16
        ), "expect quant_algo to be W8A16"

        for expert in range(self.num_experts):
            # Get quantized weights and scales
            w1 = weights[f"{expert}.w1.weight"]
            s1 = weights[f"{expert}.w1.weight_scale"]
            w3 = weights[f"{expert}.w3.weight"]
            s3 = weights[f"{expert}.w3.weight_scale"]
            w2 = weights[f"{expert}.w2.weight"]
            s2 = weights[f"{expert}.w2.weight_scale"]

            # Dequantize weights: w_dequant = (w.float() * scale).to(dtype)
            # Note: weights are (out_features, in_features), need transpose for matmul
            w1_dequant = (w1.T.contiguous().float() * s1).to(self.dtype).T.contiguous()
            w3_dequant = (w3.T.contiguous().float() * s3).to(self.dtype).T.contiguous()
            w2_dequant = (w2.T.contiguous().float() * s2).to(self.dtype).T.contiguous()

            # Load as regular weights (no scales)
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]
            gate_up_proj_weights[0]["weight"] = w1_dequant
            gate_up_proj_weights[1]["weight"] = w3_dequant
            down_proj_weights[0]["weight"] = w2_dequant
            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

    def check_accuracy(self, output, ref_output, weight_dtype=torch.int8):
        # Align with woq_assert_near_eq function
        atol = calc_woq_tolerence(ref_output, weight_dtype)
        torch.testing.assert_close(output, ref_output, rtol=1e-7, atol=atol)


# int8_woq_per_channel
class W8A16QuantizeUtil(BaseQuantizeUtil):
    """
    W8A16QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for W8A16 quantized MoE modules (INT8 weight-only quantization).
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts using W8A16 quantization.
        """
        assert self.quant_config is not None and self.quant_config.quant_algo == QuantAlgo.W8A16, (
            "expect quant_algo to be W8A16"
        )
        weights = {}
        for expert_id in range(self.num_experts):
            w1_weight = torch.randint(
                -128, 127, (self.intermediate_size, self.hidden_size), dtype=torch.int8
            ).cuda()
            w2_weight = torch.randint(
                -128, 127, (self.hidden_size, self.intermediate_size), dtype=torch.int8
            ).cuda()
            w3_weight = torch.randint(
                -128, 127, (self.intermediate_size, self.hidden_size), dtype=torch.int8
            ).cuda()

            # Per-channel scales
            w1_scale = (
                torch.randn(self.intermediate_size, dtype=self.dtype, device="cuda")
                / self.hidden_size
            )
            w2_scale = (
                torch.randn(self.hidden_size, dtype=self.dtype, device="cuda")
                / self.intermediate_size
            )
            w3_scale = (
                torch.randn(self.intermediate_size, dtype=self.dtype, device="cuda")
                / self.hidden_size
            )

            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
            weights[f"{expert_id}.w1.weight_scale"] = w1_scale
            weights[f"{expert_id}.w2.weight_scale"] = w2_scale
            weights[f"{expert_id}.w3.weight_scale"] = w3_scale
        return weights

    def create_ref_module(
        self, routing_method, ref_cls=W8A16RefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.
        """
        return super().create_ref_module(routing_method, ref_cls)


class W4A8AWQRefGatedMLPFusedMoE(nn.Module):
    """
    A reference implementation of W4A8_AWQ quantization for MoE correctness testing.

    IMPORTANT: This class does NOT inherit from RefGatedMLPFusedMoE because W4A8_AWQ
    cannot be correctly reproduced by simply dequantizing weights and using non-quantized
    GatedMLP forward. The reasons are:

    1. W4A8_AWQ involves a complete Q/DQ (Quantize/Dequantize) process for activations:
       - Apply pre_quant_scale to activation (AWQ smoothing)
       - Quantize activation to FP8 (clamp + cast)
       - Dequantize back to original dtype
       This Q/DQ process introduces quantization noise that affects the final result.

    2. For fused gate_up_proj (w3_w1), the scales must use max() operation:
       - input_scale: p3_p1 = torch.max(p1, p3)
       - pre_quant_scale: a1_a3 = torch.max(a1, a3)
       - weight_scale_2: q3_q1 = torch.max(q1, q3)
       This ensures both w3 and w1 computations use consistent scales when fused.

    3. The output needs to be scaled by input_scale and weight_scale_2 after matmul.

    This implementation follows the reference logic in test_fused_moe.py:test_fused_moe_w4afp8.
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
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.W4A8_CUSTOM,
        scaling_group_size: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.bias = bias
        self.weight_loading_mode = weight_loading_mode
        self.scaling_group_size = scaling_group_size
        # Store raw quantized weights for forward computation
        self.weights = None

    def _unpack_weights(self, weight: torch.Tensor) -> torch.Tensor:
        """Unpack INT4 packed weights to INT8."""
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        # ModelOpt W4A8 packs pairs of 4b weights in the output dimension into one 8b element.
        if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            return unpacker(weight.cpu().T.contiguous()).cuda()
        # The custom W4A8 quantization script packs pairs of 4b weight in the input dimension.
        else:
            return unpacker(weight.cpu()).T.contiguous().cuda()

    def load_weights(self, weights: List[Dict]):
        """Store raw quantized weights for forward computation."""
        assert len(weights) == 1
        self.weights = weights[0]

    def _process_layer(
        self,
        act: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        pre_quant_scale: torch.Tensor = None,
        weight_scale_2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Process a single layer with W4A8_AWQ quantization.

        This implements the complete Q/DQ process:
        1. Apply pre_quant_scale to activation (AWQ smoothing)
        2. Quantize activation to FP8 (simulate quantization noise)
        3. Dequantize weight using weight_scale
        4. Compute matmul and scale output
        """
        # Step 1: Apply pre_quant_scale (AWQ smoothing) if present
        if pre_quant_scale is not None:
            act = act * pre_quant_scale

        # Step 2: Quantize activation to FP8 and dequantize back (Q/DQ simulation)
        # This introduces quantization noise that is part of the W4A8_AWQ computation
        act = torch.clamp((act / input_scale), -448.0, 448.0).to(torch.float8_e4m3fn).to(self.dtype)

        # Step 3: Dequantize weight
        weight = (
            weight.float() * weight_scale.repeat_interleave(self.scaling_group_size, dim=0).float()
        ).to(self.dtype)
        if weight_scale_2 is not None:
            weight = weight / weight_scale_2

        # Step 4: Compute matmul and scale output
        output = torch.matmul(act, weight) * input_scale
        if weight_scale_2 is not None:
            output = output * weight_scale_2

        return output

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the complete W4A8_AWQ reference computation.

        This follows the reference implementation in test_fused_moe.py:test_fused_moe_w4afp8.
        """
        assert hidden_states.shape[-1] == self.hidden_size
        hidden_states = hidden_states.view(-1, self.hidden_size)

        results = torch.zeros_like(hidden_states)
        selected_experts, final_scales = self.routing_method.apply(router_logits)

        # Determine weight_scale key based on weight_loading_mode
        if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            weight_scale_key = "weight_scale"
        else:
            weight_scale_key = "weight_scale_inv"

        for expert_id in range(self.num_experts):
            mask = selected_experts == expert_id
            activated_tokens = mask.sum(1).bool()
            act = hidden_states[activated_tokens, :]
            if act.shape[0] == 0:
                continue
            final_scale = (final_scales * mask).sum(1)[activated_tokens].unsqueeze(1)

            # Unpack INT4 weights to INT8
            w1 = self._unpack_weights(self.weights[f"{expert_id}.w1.weight"])
            w2 = self._unpack_weights(self.weights[f"{expert_id}.w2.weight"])
            w3 = self._unpack_weights(self.weights[f"{expert_id}.w3.weight"])
            # Fuse w3 and w1 for gate_up_proj
            w3_w1 = torch.cat([w3, w1], dim=-1)

            # Get weight scales and transpose for matmul
            s1 = self.weights[f"{expert_id}.w1.{weight_scale_key}"].T.contiguous().cuda()
            s2 = self.weights[f"{expert_id}.w2.{weight_scale_key}"].T.contiguous().cuda()
            s3 = self.weights[f"{expert_id}.w3.{weight_scale_key}"].T.contiguous().cuda()
            # Fuse scales - must cat in same order as weights
            s3_s1 = torch.cat([s3, s1], dim=-1)

            # Get input scales
            p1 = self.weights[f"{expert_id}.w1.input_scale"].cuda()
            p2 = self.weights[f"{expert_id}.w2.input_scale"].cuda()
            p3 = self.weights[f"{expert_id}.w3.input_scale"].cuda()
            # IMPORTANT: Use max for fused computation to ensure consistent quantization
            p3_p1 = torch.max(p1, p3)

            # Get pre_quant_scale (only for VANILLA mode)
            a1 = a2 = a3 = a1_a3 = None
            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                a1 = self.weights[f"{expert_id}.w1.pre_quant_scale"].T.contiguous().cuda()
                a2 = self.weights[f"{expert_id}.w2.pre_quant_scale"].T.contiguous().cuda()
                a3 = self.weights[f"{expert_id}.w3.pre_quant_scale"].T.contiguous().cuda()
                # IMPORTANT: Use max for fused computation
                a1_a3 = torch.max(a1, a3)

            # Get weight_scale_2 (only for VANILLA mode)
            q1 = q2 = q3 = q3_q1 = None
            if self.weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                q1 = self.weights[f"{expert_id}.w1.weight_scale_2"].cuda()
                q2 = self.weights[f"{expert_id}.w2.weight_scale_2"].cuda()
                q3 = self.weights[f"{expert_id}.w3.weight_scale_2"].cuda()
                # IMPORTANT: Use max for fused computation
                q3_q1 = torch.max(q3, q1)

            # Forward pass: gate_up_proj (fc13)
            fc1 = self._process_layer(
                act,
                w3_w1,
                s3_s1,
                p3_p1,
                pre_quant_scale=a1_a3,
                weight_scale_2=q3_q1,
            )
            # SwiGLU activation: first half is up (w3), second half is gate (w1)
            fc1, gate = fc1.chunk(2, dim=-1)
            fc1 = fc1 * F.silu(gate)

            # Forward pass: down_proj (fc2)
            fc2 = self._process_layer(
                fc1,
                w2,
                s2,
                p2,
                pre_quant_scale=a2,
                weight_scale_2=q2,
            )

            results[activated_tokens, :] += (fc2 * final_scale).to(results.dtype)

        return results.reshape(hidden_states.shape)

    def check_accuracy(self, output, ref_output):
        # Here we use same rtol and atol as test_fused_moe_w4afp8
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


class W4A8AWQQuantizeUtil(BaseQuantizeUtil):
    """
    W4A8AWQQuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for W4A8_AWQ quantized MoE modules.
    """

    def __init__(
        self,
        num_experts: int,
        dtype: torch.dtype,
        intermediate_size: int,
        hidden_size: int,
        quant_config: QuantConfig,
    ):
        super().__init__(num_experts, dtype, intermediate_size, hidden_size, quant_config)
        # These will be set in create_weights and used in create_ref_module
        self.weight_loading_mode = MoEWeightLoadingMode.W4A8_CUSTOM
        self.scaling_group_size = 128

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """
        Create quantized weights for MoE experts using W4A8_AWQ quantization.
        """
        assert (
            self.quant_config is not None and self.quant_config.quant_algo == QuantAlgo.W4A8_AWQ
        ), "expect quant_algo to be W4A8_AWQ"

        self.scaling_group_size = quant_kwargs.get("scaling_group_size", 128)
        self.weight_loading_mode = quant_kwargs.get(
            "weight_loading_mode", MoEWeightLoadingMode.W4A8_CUSTOM
        )
        scaling_group_size = self.scaling_group_size
        weight_loading_mode = self.weight_loading_mode
        affine_coeff = 0.005

        # Determine weight shapes based on weight_loading_mode
        # ModelOpt W4A8 packs pairs of 4b weights in the output dimension into one 8b element.
        if weight_loading_mode == MoEWeightLoadingMode.VANILLA:
            w1_shape = (self.intermediate_size // 2, self.hidden_size)
            w2_shape = (self.hidden_size // 2, self.intermediate_size)
            w3_shape = (self.intermediate_size // 2, self.hidden_size)
            weight_scale_key = "weight_scale"
        # The custom W4A8 quantization script packs pairs of 4b weight in the input dimension.
        else:  # W4A8_CUSTOM
            w1_shape = (self.intermediate_size, self.hidden_size // 2)
            w2_shape = (self.hidden_size, self.intermediate_size // 2)
            w3_shape = (self.intermediate_size, self.hidden_size // 2)
            weight_scale_key = "weight_scale_inv"

        weights = {}
        for expert_id in range(self.num_experts):
            # INT4 weights packed based on weight_loading_mode
            w1_weight = torch.randint(-128, 127, w1_shape, dtype=torch.int8).cuda()
            w2_weight = torch.randint(-128, 127, w2_shape, dtype=torch.int8).cuda()
            w3_weight = torch.randint(-128, 127, w3_shape, dtype=torch.int8).cuda()

            # Pre-quant scale
            w1_pre_quant_scale = (
                torch.rand(self.hidden_size, dtype=self.dtype, device="cuda") * 0.1 + 0.95
            )
            w2_pre_quant_scale = (
                torch.rand(self.intermediate_size, dtype=self.dtype, device="cuda") * 0.1 + 0.95
            )
            w3_pre_quant_scale = (
                torch.rand(self.hidden_size, dtype=self.dtype, device="cuda") * 0.1 + 0.95
            )

            # Weight scale
            w1_scale = (
                torch.randn(
                    (self.intermediate_size, self.hidden_size // scaling_group_size),
                    dtype=self.dtype,
                    device="cuda",
                )
                * affine_coeff
            )
            w2_scale = (
                torch.randn(
                    (self.hidden_size, self.intermediate_size // scaling_group_size),
                    dtype=self.dtype,
                    device="cuda",
                )
                * affine_coeff
            )
            w3_scale = (
                torch.randn(
                    (self.intermediate_size, self.hidden_size // scaling_group_size),
                    dtype=self.dtype,
                    device="cuda",
                )
                * affine_coeff
            )

            # Input scale
            w1_input_scale = torch.randn(1, dtype=torch.float32, device="cuda") * 0.2
            w2_input_scale = w1_input_scale
            w3_input_scale = w1_input_scale

            # Weight scale 2
            w1_weight_scale_2 = torch.ones([1], dtype=torch.float32, device="cuda")
            w2_weight_scale_2 = w1_weight_scale_2
            w3_weight_scale_2 = w1_weight_scale_2

            weights[f"{expert_id}.w1.weight"] = w1_weight
            weights[f"{expert_id}.w2.weight"] = w2_weight
            weights[f"{expert_id}.w3.weight"] = w3_weight
            weights[f"{expert_id}.w1.{weight_scale_key}"] = w1_scale
            weights[f"{expert_id}.w2.{weight_scale_key}"] = w2_scale
            weights[f"{expert_id}.w3.{weight_scale_key}"] = w3_scale
            weights[f"{expert_id}.w1.input_scale"] = w1_input_scale
            weights[f"{expert_id}.w2.input_scale"] = w2_input_scale
            weights[f"{expert_id}.w3.input_scale"] = w3_input_scale
            weights[f"{expert_id}.w1.pre_quant_scale"] = w1_pre_quant_scale
            weights[f"{expert_id}.w2.pre_quant_scale"] = w2_pre_quant_scale
            weights[f"{expert_id}.w3.pre_quant_scale"] = w3_pre_quant_scale
            weights[f"{expert_id}.w1.weight_scale_2"] = w1_weight_scale_2
            weights[f"{expert_id}.w2.weight_scale_2"] = w2_weight_scale_2
            weights[f"{expert_id}.w3.weight_scale_2"] = w3_weight_scale_2
        return weights

    def create_ref_module(
        self, routing_method, ref_cls=W4A8AWQRefGatedMLPFusedMoE
    ) -> torch.nn.Module:
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
            weight_loading_mode=self.weight_loading_mode,
            scaling_group_size=self.scaling_group_size,
        )
        return ref_fused_moe
