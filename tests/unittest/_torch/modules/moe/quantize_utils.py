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
from _torch.helpers import (
    calc_woq_tolerence,
    per_block_cast_to_fp8,
    per_block_cast_to_fp8_e8m0,
    per_token_cast_to_fp8_e8m0,
)
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


def set_tensor_value_2(x, num_row, num_cols):
    """Set tensor values using a 2x2 base pattern matrix to avoid accuracy issues."""
    pattern = torch.tensor([[0.2, -0.5], [-0.3, 0.1]], device=x.device)
    repeated = pattern.repeat((num_row + 1) // 2, (num_cols + 1) // 2)[:num_row, :num_cols]
    x.copy_(repeated)


def set_tensor_value_3(x, num_row, num_cols):
    """Set tensor values using a 3x3 base pattern matrix to avoid accuracy issues."""
    pattern = torch.tensor(
        [[0.1, 0.21, 0.31], [0.3, 0.6, 0.1], [0.11, 0.51, 0.62]], device=x.device
    )
    repeated = pattern.repeat((num_row + 2) // 3, (num_cols + 2) // 3)[:num_row, :num_cols]
    x.copy_(repeated)


def set_tensor_value_4(x, num_row, num_cols):
    """Set tensor values using a 4x4 base pattern matrix to avoid accuracy issues."""
    pattern = torch.tensor(
        [
            [0.1, 0.21, 0.31, 0.41],
            [0.3, 0.6, 0.1, 0.2],
            [0.11, 0.51, 0.61, 0.71],
            [0.11, 0.52, 0.62, 0.72],
        ],
        device=x.device,
    )
    repeated = pattern.repeat((num_row + 3) // 4, (num_cols + 3) // 4)[:num_row, :num_cols]
    x.copy_(repeated)


def _normalize_backend_name(backend_type):
    if backend_type is None:
        return None
    return backend_type.value if hasattr(backend_type, "value") else str(backend_type)


def _create_fp8_block_scale_base_weights(intermediate_size, hidden_size, dtype, device):
    w1_weight = torch.empty((intermediate_size, hidden_size), dtype=dtype, device=device)
    w2_weight = torch.empty((hidden_size, intermediate_size), dtype=dtype, device=device)
    w3_weight = torch.empty((intermediate_size, hidden_size), dtype=dtype, device=device)
    # Use deterministic patterns to avoid accuracy issues
    set_tensor_value_3(w1_weight, intermediate_size, hidden_size)
    set_tensor_value_4(w2_weight, hidden_size, intermediate_size)
    set_tensor_value_3(w3_weight, intermediate_size, hidden_size)
    return w1_weight, w2_weight, w3_weight


def _create_fp8_block_scale_input(seq_len, hidden_size, dtype, device):
    x = torch.empty((seq_len, hidden_size), dtype=dtype, device=device)
    set_tensor_value_2(x, seq_len, hidden_size)
    return x


def get_test_quant_params(quant_algo, x, backend_type=None):
    """
    Create quantization configuration and corresponding kwargs for testing.

    Args:
        quant_algo: Quantization algorithm
        x: Input tensor for deriving scales
        backend_type: Optional backend type to determine scale format.
                      DEEPGEMM requires E8M0 scale format for FP8_BLOCK_SCALES.
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
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
        # Different backends have different numerical behaviors for FP8 block scaling:
        # - DEEPGEMM: Uses E8M0 scale format with manual grouped_gemm reference
        # - TRTLLM: Uses regular float scale with relaxed accuracy thresholds
        # - Others (CUTLASS, CUTEDSL): Use FP8BlockScalesQuantizeUtil with cute_dsl_blockscaling_mm
        backend_name = _normalize_backend_name(backend_type)
        if backend_name is not None:
            if backend_name == "DEEPGEMM":
                # Use DEEPGEMM-specific util with E8M0 scales and manual grouped_gemm reference
                quantize_util_cls = DeepGemmFP8BlockScalesQuantizeUtil
            elif backend_name == "TRTLLM":
                # Use FP8BlockScalesQuantizeUtil with TRTLLMGenFP8BlockScalesRefModule as ref
                # TRTLLMGenFP8BlockScalesRefModule has relaxed accuracy thresholds
                quantize_util_cls = FP8BlockScalesQuantizeUtil
                quant_kwargs["ref_cls"] = TRTLLMGenFP8BlockScalesRefModule
            else:
                quantize_util_cls = FP8BlockScalesQuantizeUtil
        else:
            quantize_util_cls = FP8BlockScalesQuantizeUtil
    elif quant_algo == QuantAlgo.W4A8_NVFP4_FP8:
        quantize_util_cls = W4A8NVFP4FP8QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_NVFP4_FP8)
        x_sf_global = 448 / x.abs().max().float()
        quant_kwargs["x_sf_global"] = x_sf_global
    elif quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
        quantize_util_cls = MXFP4MXFP8QuantizeUtil
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
        # Different backends have different alignment requirements:
        # - CUTLASS: weight_alignment=128, input_hidden_alignment=128
        # - TRTLLM: weight_alignment=128, input_hidden_alignment=512
        backend_name = _normalize_backend_name(backend_type)
        if backend_name is not None:
            if backend_name == "TRTLLM":
                quant_kwargs["weight_alignment"] = 128
                quant_kwargs["input_hidden_alignment"] = 512
            elif backend_name == "CUTLASS":
                # CUTLASS and others use weight_alignment for both
                quant_kwargs["weight_alignment"] = 128
                quant_kwargs["input_hidden_alignment"] = 128
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
        use_cute_dsl_blockscaling_mm=False,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
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

        # Custom swiglu activation for gptoss_style
        def custom_swiglu(x):
            gate, value = x.chunk(2, dim=-1)
            if swiglu_limit is not None and swiglu_limit != float("inf"):
                gate = gate.clamp(max=swiglu_limit)
                value = value.clamp(min=-swiglu_limit, max=swiglu_limit)

            alpha = swiglu_alpha if swiglu_alpha is not None else 1.0
            gate_act = gate * torch.sigmoid(gate * alpha)

            beta = swiglu_beta if swiglu_beta is not None else 0.0

            return gate_act * (value + beta)

        self.experts = nn.ModuleList(
            [
                GatedMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    bias=bias,
                    dtype=self.dtype,
                    config=model_config,
                    use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
                    activation=custom_swiglu if swiglu_alpha is not None else F.silu,
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

    def load_weights(self, weights_list: List[Dict]):
        assert len(weights_list) == 1
        weights = weights_list[0]

        # Validate quant_algo if expected
        if self.expected_quant_algo is not None:
            assert self.quant_config and self.quant_config.quant_algo == self.expected_quant_algo, (
                f"expect quant_algo to be {self.expected_quant_algo}"
            )

        for expert in range(self.num_experts):
            self._load_expert_weights_with_scales(weights, expert)

    def check_accuracy(self, output, ref_output):
        # Relaxed percent from 0.984 to 0.96 to handle small tensor statistical variance.
        # For small outputs (e.g., h=64), a few outliers can cause high mismatch percentage.
        # Example: 2/64 mismatch = 3.125% > 1.6% (old threshold), but only 2 elements differ.
        check_accuracy(output, ref_output, rtol=2e-1, atol=2e-1, percent=0.96)


class BaseQuantizeUtil(ABC):
    """
    BaseQuantizeUtil serves as a base class for MoE correctess testing which provides interface
    to create quantized weights and reference modules. It can be extended for different quantization algorithms.
    Supports gptoss_style with custom swiglu parameters.
    """

    def __init__(
        self,
        num_experts: int,
        dtype: torch.dtype,
        intermediate_size: int,
        hidden_size: int,
        quant_config: QuantConfig,
        bias: bool = False,
        gptoss_style: bool = False,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
    ):
        self.num_experts = num_experts
        self.dtype = dtype
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.quant_config = quant_config
        self.bias = bias
        self._gptoss_style = gptoss_style
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        # Pre-create swiglu tensors if gptoss_style is enabled
        if self._gptoss_style:
            self._swiglu_tensors = self._create_swiglu_tensors()
        else:
            self._swiglu_tensors = None

    @property
    def gptoss_style(self) -> bool:
        """Check if gptoss_style is enabled."""
        return self._gptoss_style

    def _create_swiglu_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Internal method to create swiglu tensors for MoE backend.

        Returns:
            Dict with 'swiglu_alpha', 'swiglu_beta', 'swiglu_limit' tensors.
        """
        return {
            "swiglu_alpha": torch.full(
                (self.num_experts,), self.swiglu_alpha, device="cuda", dtype=torch.float
            ),
            "swiglu_beta": torch.full(
                (self.num_experts,), self.swiglu_beta, device="cuda", dtype=torch.float
            ),
            "swiglu_limit": torch.full(
                (self.num_experts,), self.swiglu_limit, device="cuda", dtype=torch.float
            ),
        }

    def get_swiglu_tensors(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get pre-created swiglu tensors.

        Returns:
            Dict with swiglu tensors if gptoss_style is enabled, None otherwise.
        """
        return self._swiglu_tensors

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
            bias=self.bias,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
        )
        return ref_fused_moe


class FP8RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of FP8 quantization for correctness testing."""

    scale_keys = ["weight_scale", "input_scale"]
    expected_quant_algo = QuantAlgo.FP8

    def check_accuracy(self, output, ref_output):
        # Relaxed percent from 0.99 to 0.97 to account for FP8 quantization error accumulation
        # in large intermediate dimensions and multi-expert routing computations.
        # Theoretical basis: FP8 (E4M3) has ~12.5% unit error, accumulated error grows as sqrt(K)
        # where K is GEMM reduction dimension. Max observed mismatch is ~2.1% < 3%.
        check_accuracy(output, ref_output, rtol=4e-2, atol=1e-1, percent=0.97)


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

    def __init__(self, *args, gptoss_style: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gptoss_style = gptoss_style

    def check_accuracy(self, output, ref_output):
        if self.gptoss_style:
            # gptoss_style uses relaxed tolerance
            check_accuracy(output, ref_output, rtol=0.1, atol=0.1, percent=0.95)
        else:
            check_accuracy(output, ref_output, rtol=1e-2, atol=0.15, percent=0.98)


class NVFP4QuantizeUtil(BaseQuantizeUtil):
    """
    NVFP4QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing for NVFP4 quantized MoE modules.
    Supports gptoss_style with custom swiglu parameters (inherited from BaseQuantizeUtil).
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

            # Note: NVFP4 bias uses torch.float dtype
            if self.bias:
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
        Create a reference module for correctness testing with gptoss_style support.
        """
        ref_fused_moe = ref_cls(
            num_experts=self.num_experts,
            routing_method=routing_method,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
            model_config=ModelConfig(quant_config=self.quant_config),
            bias=self.bias,
            gptoss_style=self.gptoss_style,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
        )
        return ref_fused_moe


class FP8BlockScalesRefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of FP8 block-wise quantization for correctness testing."""

    scale_keys = ["weight_scale"]
    expected_quant_algo = QuantAlgo.FP8_BLOCK_SCALES

    def __init__(self, *args, use_cute_dsl_blockscaling_mm=True, **kwargs):
        # Note: use deepgemm mm will cause accuracy error, so we use cute_dsl_blockscaling_mm here
        super().__init__(*args, use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm, **kwargs)

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

        Args:
            use_e8m0_scale: If True, use per_block_cast_to_fp8_e8m0 which produces E8M0
                            format scales required by DEEPGEMM and TRTLLM backends.
                            If False, use per_block_cast_to_fp8 with regular float scales.
        """
        assert (
            self.quant_config is not None
            and self.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        ), "expect quant_algo to be FP8_BLOCK_SCALES"

        # Select quantization function based on scale format requirement
        use_e8m0_scale = quant_kwargs.get("use_e8m0_scale", False)
        quant_fn = per_block_cast_to_fp8_e8m0 if use_e8m0_scale else per_block_cast_to_fp8

        weights = {}
        for expert_id in range(self.num_experts):
            w1_weight, w2_weight, w3_weight = _create_fp8_block_scale_base_weights(
                self.intermediate_size, self.hidden_size, self.dtype, "cuda"
            )

            w1_weight_fp8, w1_weight_scale = quant_fn(w1_weight)
            w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w2_weight_fp8, w2_weight_scale = quant_fn(w2_weight)
            w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w3_weight_fp8, w3_weight_scale = quant_fn(w3_weight)
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

    def create_input(self, seq_len: int) -> torch.Tensor:
        """
        Create input tensor with deterministic pattern to avoid accuracy issues.
        FP8_BLOCK_SCALES requires special input values to avoid false positive failures.
        """
        return _create_fp8_block_scale_input(seq_len, self.hidden_size, self.dtype, "cuda")


class DeepGemmFP8BlockScalesRefFusedMoE(FP8BlockScalesRefGatedMLPFusedMoE):
    """
    Reference implementation for DEEPGEMM FP8 block-wise quantization.

    Inherits from FP8BlockScalesRefGatedMLPFusedMoE but overrides forward() to use
    manual grouped_gemm computation that matches DEEPGEMM's numerical behavior.

    Key differences from base class:
    - Uses manual grouped_gemm instead of GatedMLP for computation
    - Permutes tokens by expert before GEMM (DEEPGEMM's computation pattern)
    - Uses per_token_cast_to_fp8_e8m0 for activation quantization

    """

    def __init__(self, *args, top_k: int = 2, **kwargs):
        # Initialize base class with use_cute_dsl_blockscaling_mm=False
        # (we won't use GatedMLP anyway, but need to init the base class)
        super().__init__(*args, use_cute_dsl_blockscaling_mm=False, **kwargs)
        self.top_k = top_k

        # Additional weight tensors for grouped GEMM (populated in load_weights)
        self.w3_w1_weights = None
        self.w3_w1_scales = None
        self.w2_weights_stacked = None
        self.w2_scales_stacked = None

    def load_weights(self, weights_list: List[Dict]):
        """Load weights and prepare stacked tensors for grouped GEMM."""
        # Call parent to load weights into GatedMLP experts
        super().load_weights(weights_list)

        # Also stack weights for grouped GEMM computation
        weights = weights_list[0]
        w1_list, w2_list, w3_list = [], [], []
        w1_scale_list, w2_scale_list, w3_scale_list = [], [], []

        for expert_id in range(self.num_experts):
            w1_list.append(weights[f"{expert_id}.w1.weight"])
            w2_list.append(weights[f"{expert_id}.w2.weight"])
            w3_list.append(weights[f"{expert_id}.w3.weight"])
            w1_scale_list.append(weights[f"{expert_id}.w1.weight_scale"])
            w2_scale_list.append(weights[f"{expert_id}.w2.weight_scale"])
            w3_scale_list.append(weights[f"{expert_id}.w3.weight_scale"])

        w1_weights = torch.stack(w1_list, dim=0)
        w3_weights = torch.stack(w3_list, dim=0)
        w1_scales = torch.stack(w1_scale_list, dim=0)
        w3_scales = torch.stack(w3_scale_list, dim=0)

        # Create fused w3_w1 weights and scales for gemm1 (gate_up)
        self.w3_w1_weights = torch.cat([w3_weights, w1_weights], dim=1)
        self.w3_w1_scales = torch.cat([w3_scales, w1_scales], dim=1)
        self.w2_weights_stacked = torch.stack(w2_list, dim=0)
        self.w2_scales_stacked = torch.stack(w2_scale_list, dim=0)

    def cuda(self):
        """Move all weights to CUDA."""
        super().cuda()
        if self.w3_w1_weights is not None:
            self.w3_w1_weights = self.w3_w1_weights.cuda()
            self.w3_w1_scales = self.w3_w1_scales.cuda()
            self.w2_weights_stacked = self.w2_weights_stacked.cuda()
            self.w2_scales_stacked = self.w2_scales_stacked.cuda()
        return self

    def _swiglu(self, x):
        """SwiGLU activation: silu(gate) * x"""
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x

    # Block size for FP8 block scaling (matches DEEPGEMM's block scale granularity)
    _BLOCK_SIZE = 128

    def _grouped_gemm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_sf: torch.Tensor,
        b_sf: torch.Tensor,
        offset_array: torch.Tensor,
    ) -> torch.Tensor:
        """
        Manual grouped GEMM with FP8 block scaling dequantization.

        This matches DEEPGEMM's numerical behavior by manually dequantizing
        and computing matrix multiplication.
        """
        block_size = self._BLOCK_SIZE
        num_groups = b.shape[0]
        d = torch.empty((a.shape[0], b.shape[1]), device=b.device, dtype=torch.bfloat16)

        for g in range(num_groups):
            start_idx = offset_array[g].item()
            end_idx = offset_array[g + 1].item()
            if start_idx == end_idx:
                continue

            # Get activation slice and dequantize
            aa = a[start_idx:end_idx, :].to(torch.bfloat16)
            aa_sf = a_sf[start_idx:end_idx, :]
            # Repeat scale to match activation dimensions
            aa_dq = aa * aa_sf.repeat_interleave(block_size, dim=1)[: aa.shape[0], : aa.shape[1]]

            # Get weight and dequantize
            bb = b[g, :, :].to(torch.bfloat16)
            bb_sf = b_sf[g, :, :]
            # Repeat scale to match weight dimensions (block_size x block_size)
            bb_dq = (
                bb
                * bb_sf.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[
                    : bb.shape[0], : bb.shape[1]
                ]
            )

            # Matrix multiplication
            d[start_idx:end_idx, :] = aa_dq @ bb_dq.t()

        return d

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with manual grouped GEMM computation.

        This matches DEEPGEMM's numerical behavior by using manual grouped GEMM
        instead of GatedMLP.
        """
        x = hidden_states.view(-1, self.hidden_size)
        seq_len = x.shape[0]

        # Apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)

        # Permute tokens by expert
        permuted_data = torch.empty(
            (seq_len * self.top_k, self.hidden_size),
            device=x.device,
            dtype=torch.bfloat16,
        )
        expert_first_token_offset = torch.zeros(
            self.num_experts + 1, dtype=torch.int32, device=x.device
        )
        unpermute_map = []
        scales = []

        t_idx = 0
        for e_idx in range(self.num_experts):
            for idx in range(seq_len):
                for i in range(self.top_k):
                    if token_selected_experts[idx, i] == e_idx:
                        permuted_data[t_idx, :] = x[idx]
                        unpermute_map.append(idx)
                        scales.append(token_final_scales[idx, i])
                        t_idx += 1
            expert_first_token_offset[e_idx + 1] = t_idx

        # Quantize input activation to FP8 with E8M0 scales
        act_fp8, act_sf = per_token_cast_to_fp8_e8m0(permuted_data)

        # GEMM1: gate_up projection
        h1 = self._grouped_gemm(
            a=act_fp8,
            b=self.w3_w1_weights,
            a_sf=act_sf,
            b_sf=self.w3_w1_scales,
            offset_array=expert_first_token_offset,
        )

        # Activation
        h2 = self._swiglu(h1)

        # Quantize intermediate activation
        act_fp8, act_sf = per_token_cast_to_fp8_e8m0(h2)

        # GEMM2: down projection
        h3 = self._grouped_gemm(
            a=act_fp8,
            b=self.w2_weights_stacked,
            a_sf=act_sf,
            b_sf=self.w2_scales_stacked,
            offset_array=expert_first_token_offset,
        )

        # Unpermute and apply routing weights
        output = torch.zeros_like(x)
        for token_idx, h3_token in enumerate(h3):
            original_idx = unpermute_map[token_idx]
            output[original_idx, :] += h3_token * scales[token_idx]

        return output


class DeepGemmFP8BlockScalesQuantizeUtil(BaseQuantizeUtil):
    """
    Quantization utility for DEEPGEMM + FP8_BLOCK_SCALES testing.

    Uses E8M0 scale format and DeepGemmFP8BlockScalesRefFusedMoE as reference.
    """

    def create_weights(self, **quant_kwargs) -> Dict[str, torch.Tensor]:
        """Create quantized weights using E8M0 scale format for DEEPGEMM."""
        assert (
            self.quant_config is not None
            and self.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        ), "expect quant_algo to be FP8_BLOCK_SCALES"

        weights = {}
        for expert_id in range(self.num_experts):
            w1_weight, w2_weight, w3_weight = _create_fp8_block_scale_base_weights(
                self.intermediate_size, self.hidden_size, self.dtype, "cuda"
            )

            # Use E8M0 scale format for DEEPGEMM
            w1_weight_fp8, w1_weight_scale = per_block_cast_to_fp8_e8m0(w1_weight)
            w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w2_weight_fp8, w2_weight_scale = per_block_cast_to_fp8_e8m0(w2_weight)
            w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

            w3_weight_fp8, w3_weight_scale = per_block_cast_to_fp8_e8m0(w3_weight)
            w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

            weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
            weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
            weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
            weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale
            # Also add weight_scale_inv for compatibility
            weights[f"{expert_id}.w1.weight_scale_inv"] = w1_weight_scale
            weights[f"{expert_id}.w2.weight_scale_inv"] = w2_weight_scale
            weights[f"{expert_id}.w3.weight_scale_inv"] = w3_weight_scale
        return weights

    def create_ref_module(self, routing_method) -> torch.nn.Module:
        """Create DEEPGEMM-specific reference module."""
        return DeepGemmFP8BlockScalesRefFusedMoE(
            num_experts=self.num_experts,
            routing_method=routing_method,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
            model_config=ModelConfig(quant_config=self.quant_config),
            top_k=routing_method.top_k,
        )

    def create_input(self, seq_len: int) -> torch.Tensor:
        """Create input tensor with deterministic pattern."""
        return _create_fp8_block_scale_input(seq_len, self.hidden_size, self.dtype, "cuda")


class TRTLLMGenFP8BlockScalesRefModule(FP8BlockScalesRefGatedMLPFusedMoE):
    """
    Reference module for TRTLLM FP8 block scale testing.

    Inherits FP8BlockScalesRefGatedMLPFusedMoE with cute_dsl_blockscaling_mm=True.
    """

    def check_accuracy(self, output, ref_output):
        """
        Check accuracy with relaxed tolerance for TRTLLM FP8 block scale kernel.

        The TRTLLM fp8_block_scale_moe_runner has specific numerical behavior that may
        differ from reference implementation due to kernel-specific optimizations.
        """
        check_accuracy(output, ref_output, atol=0.1, rtol=0.85, percent=0.925)


class W4A8NVFP4FP8RefGatedMLPFusedMoE(RefGatedMLPFusedMoE):
    """Reference implementation of W4A8_NVFP4_FP8 quantization for correctness testing."""

    scale_keys = ["weight_scale", "input_scale", "weight_scale_2"]
    expected_quant_algo = QuantAlgo.W4A8_NVFP4_FP8

    def check_accuracy(self, output, ref_output):
        check_accuracy(output, ref_output, rtol=0.85, atol=0.5, percent=0.925)


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
    """
    Reference implementation of W4A8_MXFP4_MXFP8 quantization for correctness testing.

    This implementation uses the same quantization method (W4A8MXFP4MXFP8LinearMethod)
    as the fused MoE kernel by passing quant_config to GatedMLP. Weights are loaded
    directly in quantized format.

    Note: When hidden_size_unpadded < hidden_size (due to different alignment requirements),
    input is padded before forward and output is truncated after forward to match
    the original hidden_size.
    """

    # Expected quantization algorithm
    expected_quant_algo = QuantAlgo.W4A8_MXFP4_MXFP8
    # Scale keys to load for this quantization method
    scale_keys: List[str] = ["weight_scale"]

    def __init__(
        self,
        num_experts: int,
        routing_method: BaseMoeRoutingMethod,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        model_config: Optional[ModelConfig] = None,
        bias=False,
        hidden_size_unpadded: Optional[int] = None,
        gptoss_style: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            routing_method=routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            model_config=model_config,
            bias=bias,
            **kwargs,
        )
        # Store hidden_size_unpadded for input padding and output truncation
        # If not specified, use hidden_size (no padding/truncation needed)
        self.hidden_size_unpadded = (
            hidden_size_unpadded if hidden_size_unpadded is not None else hidden_size
        )
        self.gptoss_style = gptoss_style

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        # Pad input if hidden_size_unpadded < hidden_size
        if self.hidden_size_unpadded < self.hidden_size:
            pad_size = self.hidden_size - self.hidden_size_unpadded
            hidden_states = torch.nn.functional.pad(hidden_states, (0, pad_size))

        output = super().forward(hidden_states, router_logits)

        # Truncate output to hidden_size_unpadded if different from hidden_size
        if self.hidden_size_unpadded < self.hidden_size:
            output = output[:, : self.hidden_size_unpadded]
        return output

    def check_accuracy(self, output, ref_output):
        if self.gptoss_style:
            check_accuracy(output, ref_output, rtol=0.1, atol=0.2, percent=0.8)
        else:
            check_accuracy(output, ref_output, rtol=0.10, atol=0.2, percent=0.85)


class MXFP4MXFP8QuantizeUtil(BaseQuantizeUtil):
    """
    MXFP4MXFP8QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for W4A8_MXFP4_MXFP8 quantized MoE modules.
    """

    def prepare_weights_from_backend(self, backend, **quant_kwargs):
        """
        Prepare weights for backend and reference module based on actual backend shapes.

        MXFP4_MXFP8 requires different weights for backend and reference module
        due to different padding/alignment requirements.

        Args:
            backend: The MoE backend instance (to get actual shapes and alignments)
            **quant_kwargs: Additional quantization parameters

        Returns:
            (backend_weights, ref_weights, ref_module_kwargs)
        """
        # Get actual shapes from backend
        num_elts_per_dtype = torch.iinfo(backend.quant_method.weight_dtype).bits // 4
        hidden_size_in = backend.w3_w1_weight.shape[-1] * num_elts_per_dtype
        # hidden_size_out_padded is used for weight creation (padded value)
        hidden_size_out_padded = backend.w2_weight.shape[-2]
        inter_size = backend.w2_weight.shape[-1] * num_elts_per_dtype
        weight_align = backend.quant_method.weight_alignment
        input_hidden_align = getattr(backend.quant_method, "input_hidden_alignment", weight_align)

        # Backend weights: contamination padding
        backend_kwargs = dict(
            quant_kwargs,
            hidden_size_in=hidden_size_in,
            hidden_size_out=hidden_size_out_padded,
            intermediate_size=inter_size,
            input_hidden_alignment=input_hidden_align,
            pad_zero_or_val=False,
            bias=self.bias,  # Pass bias from self to create bias weights
        )
        backend_weights = self.create_weights(**backend_kwargs)

        # Ref weights: zero padding, use weight_alignment for input_hidden
        ref_kwargs = dict(
            quant_kwargs,
            hidden_size_in=hidden_size_in,
            hidden_size_out=hidden_size_in,  # same as hidden_size_in
            intermediate_size=inter_size,
            input_hidden_alignment=weight_align,
            pad_zero_or_val=True,
            bias=self.bias,  # Pass bias from self to create bias weights
        )
        ref_weights = self.create_weights(**ref_kwargs)

        # Kwargs for creating ref module
        # hidden_size_unpadded is the original hidden_size for input padding and output truncation
        ref_module_kwargs = dict(
            hidden_size_in=hidden_size_in,
            intermediate_size=inter_size,
            hidden_size_unpadded=self.hidden_size,
        )

        return backend_weights, ref_weights, ref_module_kwargs

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
        self,
        routing_method,
        ref_cls=MXFP4MXFP8RefGatedMLPFusedMoE,
        hidden_size_in: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_size_unpadded: Optional[int] = None,
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing.

        Args:
            routing_method: The routing method to use
            ref_cls: The reference class to instantiate
            hidden_size_in: Padded hidden size for GatedMLP (input dimension of w1/w3)
                           If None, uses self.hidden_size
            intermediate_size: Padded intermediate size for GatedMLP
                              If None, uses self.intermediate_size
            hidden_size_unpadded: Original unpadded hidden size for input padding
                                 and output truncation. If None, uses self.hidden_size
        """
        # Use provided sizes or fall back to defaults
        hs_in = hidden_size_in if hidden_size_in is not None else self.hidden_size
        inter_size = intermediate_size if intermediate_size is not None else self.intermediate_size
        hs_unpadded = hidden_size_unpadded if hidden_size_unpadded is not None else self.hidden_size

        ref_fused_moe = ref_cls(
            num_experts=self.num_experts,
            routing_method=routing_method,
            hidden_size=hs_in,
            intermediate_size=inter_size,
            dtype=self.dtype,
            model_config=ModelConfig(quant_config=self.quant_config),
            bias=self.bias,
            hidden_size_unpadded=hs_unpadded,
            gptoss_style=self.gptoss_style,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
        )
        return ref_fused_moe


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
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )

    def load_weights(self, weights_list: List[Dict]):
        assert len(weights_list) == 1
        weights = weights_list[0]

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
            # Note: mxfp4_dequantize_unswizzled returns shape (out_features, in_features)
            # which matches F.linear weight layout (out, in). Do NOT transpose.
            w1_dequant = (
                unpacker(w1.cpu(), s1.cpu(), scaling_group_size)
                .to(dtype=self.dtype, device="cuda")
                .contiguous()
            )
            w3_dequant = (
                unpacker(w3.cpu(), s3.cpu(), scaling_group_size)
                .to(dtype=self.dtype, device="cuda")
                .contiguous()
            )
            w2_dequant = (
                unpacker(w2.cpu(), s2.cpu(), scaling_group_size)
                .to(dtype=self.dtype, device="cuda")
                .contiguous()
            )

            # Load as regular weights (no scales)
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]
            gate_up_proj_weights[0]["weight"] = w1_dequant
            gate_up_proj_weights[1]["weight"] = w3_dequant
            down_proj_weights[0]["weight"] = w2_dequant

            # Load bias if enabled
            if self.bias:
                gate_up_proj_weights[0]["bias"] = weights[f"{expert}.w1.bias"]
                gate_up_proj_weights[1]["bias"] = weights[f"{expert}.w3.bias"]
                down_proj_weights[0]["bias"] = weights[f"{expert}.w2.bias"]

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

    def check_accuracy(self, output, ref_output):
        check_accuracy(output, ref_output, rtol=0.10, atol=0.1, percent=0.85)


class WFP4A16QuantizeUtil(BaseQuantizeUtil):
    """
    WFP4A16QuantizeUtil inherits from BaseQuantizeUtil to support correctness testing
    for W4A16_MXFP4 quantized MoE modules.
    Supports gptoss_style with custom swiglu parameters (inherited from BaseQuantizeUtil).
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

            # Bias for gptoss_style
            if self.bias:
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
        self, routing_method, ref_cls=WFP4A16RefGatedMLPFusedMoE
    ) -> torch.nn.Module:
        """
        Create a reference module for correctness testing with gptoss_style support.
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
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )

    def load_weights(self, weights_list: List[Dict]):
        assert len(weights_list) == 1
        weights = weights_list[0]

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
        """Forward pass implementing the complete W4A8_AWQ reference computation."""
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
