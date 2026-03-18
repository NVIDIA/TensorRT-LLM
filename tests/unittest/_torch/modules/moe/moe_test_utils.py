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
"""
Shared utilities for MoE test files (test_moe_backend.py and test_moe_module.py).

This module contains common code extracted from both test files:
- MoeBackendType enum and get_backend_class()
- MoeModelConfig dataclass
- Skip logic functions (should_skip_trtllm, should_skip_cutedsl, should_skip_routing_method, etc.)
- get_quick_skip_reason() - unified version supporting both backend and module tests
- supports_autotuner_capture()
- replay_tactics_and_check()
- module_timer fixture
- create_test_param() helper
- Common test parameter constants
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Callable, Optional, Type

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.modules.fused_moe import (
    CuteDslFusedMoE,
    CutlassFusedMoE,
    TRTLLMGenFusedMoE,
)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
from tensorrt_llm._torch.utils import ActivationType, is_gated_activation
from tensorrt_llm.models.modeling_utils import QuantAlgo

G_LOGGER = logging.getLogger(__name__)


# ============================================================================
# MoE Backend Types
# ============================================================================
class MoeBackendType(str, Enum):
    """Enum for MoE backend types."""

    CUTLASS = "CUTLASS"
    TRTLLM = "TRTLLM"
    CUTEDSL = "CUTEDSL"
    DEEPGEMM = "DEEPGEMM"


def get_backend_class(backend_type: MoeBackendType) -> Type[MoE]:
    """Get the MoE backend class for a given backend type."""
    backend_class_map = {
        MoeBackendType.CUTLASS: CutlassFusedMoE,
        MoeBackendType.TRTLLM: TRTLLMGenFusedMoE,
        MoeBackendType.CUTEDSL: CuteDslFusedMoE,
        MoeBackendType.DEEPGEMM: DeepGemmFusedMoE,
    }
    return backend_class_map[backend_type]


# ============================================================================
# Model Configuration
# ============================================================================
@dataclass
class MoeModelConfig:
    """MoE model configuration: (num_experts, top_k, hidden_size, intermediate_size)."""

    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int

    def __str__(self) -> str:
        return f"e{self.num_experts}_k{self.top_k}_h{self.hidden_size}_i{self.intermediate_size}"


# ============================================================================
# Skip Logic Functions
# ============================================================================
def _is_fp4_fp8_standalone_gemm_available() -> bool:
    """Check if standalone fp4_fp8_gemm_trtllmgen kernel has compiled configs on this GPU.

    The W4A8_NVFP4_FP8 reference module (W4A8NVFP4FP8RefGatedMLPFusedMoE) uses
    standalone fp4_fp8_gemm_trtllmgen GEMM calls via W4A8NVFP4FP8LinearMethod.
    These standalone GEMM kernels may not have compiled configurations for all SM
    versions, even when the fused MoE kernel (TRTLLMGenFusedMoE) works fine.

    Returns True if the standalone kernel is available, False otherwise.
    Result is cached after first call.
    """
    if hasattr(_is_fp4_fp8_standalone_gemm_available, "_cached_result"):
        return _is_fp4_fp8_standalone_gemm_available._cached_result

    try:
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

        # Create minimal valid tensors for GEMM probe:
        # mat1: (m, k) FP8, mat2: (n, k/2) FP4, scale: FP8, global_scale: FP32
        m, n, k = 1, 128, 128
        fp8_input = torch.zeros((m, k), dtype=torch.float8_e4m3fn, device="cuda")
        fp4_weight = torch.zeros((n, k // 2), dtype=fp4_utils.float4_e2m1x2, device="cuda")
        weight_scale = torch.ones((n * (k // 32),), dtype=torch.float8_e4m3fn, device="cuda")
        global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        torch.ops.trtllm.fp4_fp8_gemm_trtllmgen(
            fp8_input, fp4_weight, weight_scale, global_scale, torch.float16
        )
        result = True
    except RuntimeError:
        result = False

    _is_fp4_fp8_standalone_gemm_available._cached_result = result
    return result


def should_skip_trtllm(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    model_config: "MoeModelConfig",
    routing_method_cls=None,
    swiglu_gptoss_style: bool = False,
    comm_method: Optional[str] = None,
    seq_len: Optional[int] = None,
    moe_tp_size: int = 1,
) -> Optional[str]:
    """
    Check TRTLLM Gen backend specific constraints.

    The TRTLLM Gen MoE kernels have hardware-level constraints that must be satisfied.
    These constraints are enforced in C++ layer.

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm
        model_config: The MoE model configuration
        routing_method_cls: Optional routing method class for compatibility checks
            (used by test_moe_module.py)
        swiglu_gptoss_style: Whether using swiglu gptoss style
        comm_method: Optional communication method (e.g. "DEEPEP", "DEEPEPLOWLATENCY")
            for multi-GPU EP mode checks
        seq_len: Optional sequence length for seq_len-sensitive skip checks
        moe_tp_size: MoE TP parallelism size (default: 1, no TP sharding)

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.TRTLLM:
        return None

    # Routing method compatibility check (used by test_moe_module.py)
    # TRTLLMGen C++ routing kernel (runner.cu) only implements:
    # - DeepSeekV3 (requires float32 routing_logits)
    # - Llama4 (requires top_k=1)
    # - Renormalize
    # - RenormalizeNaive
    # See: cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu:77-212
    if routing_method_cls is not None:
        from tensorrt_llm._torch.modules.fused_moe import (
            DeepSeekV3MoeRoutingMethod,
            DefaultMoeRoutingMethod,
            Llama4RenormalizeMoeRoutingMethod,
            MiniMaxM2MoeRoutingMethod,
        )

        # Routing methods NOT implemented in C++ kernel
        trtllm_unimplemented_routing = (
            DefaultMoeRoutingMethod,  # runner.cu:210 - "Unimplemented routing method"
            MiniMaxM2MoeRoutingMethod,  # runner.cu:210 - "Unimplemented routing method"
        )
        if routing_method_cls in trtllm_unimplemented_routing:
            routing_name = routing_method_cls.__name__
            return (
                f"TRTLLMGen C++ routing kernel does not implement {routing_name}. See runner.cu:210"
            )

        # Llama4 routing only supports top_k=1
        # See: runner.cu:113 - TLLM_CHECK_WITH_INFO(topK == 1, ...)
        if routing_method_cls == Llama4RenormalizeMoeRoutingMethod:
            if model_config is not None and model_config.top_k != 1:
                return (
                    f"TRTLLMGen Llama4 routing only supports top_k=1 "
                    f"(got top_k={model_config.top_k}). See runner.cu:113"
                )

        # DeepSeekV3 routing requires num_experts >= 22
        # See: RoutingDeepSeek.cu:32,664 - MaxSupportedTopExperts = 22
        if routing_method_cls == DeepSeekV3MoeRoutingMethod:
            if model_config is not None and model_config.num_experts < 22:
                return (
                    f"TRTLLMGen DeepSeekV3 routing requires num_experts >= 22 "
                    f"(got num_experts={model_config.num_experts}). See RoutingDeepSeek.cu:664"
                )

            # DeepSeekV3 routing kernel only supports topk_group <= 4.
            # topk_group is computed from num_experts in _create_routing_method:
            #   n_group = max(1, num_experts // 2)
            #   topk_group = min(n_group, max(1, n_group // 2))
            if model_config is not None:
                n_group = max(1, model_config.num_experts // 2)
                topk_group = min(n_group, max(1, n_group // 2))
                if topk_group > 4:
                    return (
                        f"TRTLLMGen DeepSeekV3 routing kernel only supports "
                        f"topk_group <= 4 (got topk_group={topk_group} from "
                        f"num_experts={model_config.num_experts})"
                    )

    if model_config is None:
        return None

    # These quantization algorithms use TRTLLM Gen kernels with the constraints
    trtllm_gen_quant_algos = {
        QuantAlgo.NVFP4,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    }

    if quant_algo not in trtllm_gen_quant_algos:
        return None

    num_experts = model_config.num_experts
    top_k = model_config.top_k
    intermediate_size = model_config.intermediate_size

    # Check: num_experts must be divisible by 4
    # Routing kernel uses vectorized operations that require this alignment
    if num_experts % 4 != 0:
        return (
            f"TRTLLMGenFusedMoE routing kernel requires num_experts divisible by 4 "
            f"(got num_experts={num_experts})"
        )

    # Check: num_experts must be greater than top_k
    # Routing logic cannot handle the case where all experts are selected
    if num_experts <= top_k:
        return (
            f"TRTLLMGenFusedMoE requires num_experts > top_k "
            f"(got num_experts={num_experts}, top_k={top_k})"
        )
    # W4A8_MXFP4_MXFP8 with non-128-aligned hidden_size or intermediate_size
    # causes block_scale_interleave_reverse to fail with
    # "rows of Interleaved block scales should be multiple of 128".
    if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
        hidden_size = model_config.hidden_size
        if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
            return (
                f"TRTLLMGenFusedMoE W4A8_MXFP4_MXFP8 with non-128-aligned "
                f"sizes (h={hidden_size}, i={intermediate_size}) causes "
                f"block_scale_interleave_reverse rows must be multiple of 128."
            )

    # -----------------Reference module constraints------------------
    # The W4A8_NVFP4_FP8 reference module (W4A8NVFP4FP8RefGatedMLPFusedMoE) uses
    # standalone fp4_fp8_gemm_trtllmgen GEMM calls via W4A8NVFP4FP8LinearMethod.
    # These standalone GEMM kernels may not have compiled configs for all SM versions,
    # even though the fused MoE kernel (TRTLLMGenFusedMoE) works fine on those SMs.
    # Skip if the standalone kernel is not available on the current GPU.
    if quant_algo == QuantAlgo.W4A8_NVFP4_FP8:
        if not _is_fp4_fp8_standalone_gemm_available():
            return (
                "W4A8_NVFP4_FP8 reference module requires standalone "
                "fp4_fp8_gemm_trtllmgen kernel which is not available on this GPU. "
                "The fused MoE kernel works but the reference GatedMLP cannot run."
            )

    # -----------------Potential issues------------------
    # These are known issues that need investigation. Skipping to avoid test failures
    # and CUDA errors that can cascade to subsequent tests.

    if quant_algo == QuantAlgo.NVFP4:
        # Issue: NVFP4 with large intermediate_size has known accuracy issues
        if intermediate_size >= 14336:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE NVFP4 with large intermediate_size "
                f"has known accuracy issues (intermediate_size={intermediate_size} >= 14336)."
            )
        # NVFP4 flaky tactic failures with large model configs at seq=8.
        # For example of observed failures:
        #   - act=Relu2-e60_k4_h2048_i1408-seq=8: tactic[28] tile [32,36],
        #     12.79% mismatch, 187/188 tactics pass.
        if (
            num_experts >= 60
            and model_config.top_k >= 4
            and model_config.hidden_size >= 2048
            and model_config.intermediate_size >= 1408
            and seq_len == 8
        ):
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE NVFP4 with large model config"
                f"(num_experts={num_experts}, top_k={model_config.top_k}, "
                f"hidden_size={model_config.hidden_size}, intermediate_size={model_config.intermediate_size})"
                f"and seq_len=8: flaky happen tactics failure with tactic[24] and tactic[28]"
            )
        # Issue: NVFP4 with large expert count + large hidden_size
        # has a single FP4BlockScaleMoERunner tactic with accuracy failure.
        # Observed: e256_k8_h7168_i2048, seq=1, bfloat16 — tactic[204] with tile
        # config [8, 83] produces 8.37% element mismatch (threshold: 3%).
        # All other 207/208 tactics pass. The swiglu_gptoss_style variant passes too
        # (uses relaxed tolerance: rtol=0.1, percent=0.95).
        # Root cause: FP4 quantization error accumulates in the large GEMM reduction
        # dimension (h=7168) and the [8, 83] tile config hits an edge case at seq=1.
        if num_experts >= 256 and model_config.hidden_size >= 7168 and not swiglu_gptoss_style:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE NVFP4 with large model "
                f"(num_experts={num_experts}, hidden_size={model_config.hidden_size}) "
                f"and seq_len=1: 207/208 tactics pass but tactic[204] "
                f"(FP4BlockScaleMoERunner tile [8, 83]) has 8.37% mismatch "
                f"(threshold 3%). seq_len=8 passes all tactics."
            )

    # Issue: W4A8_MXFP4_MXFP8 has accuracy issues on certain model configs
    if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
        if num_experts >= 60 and intermediate_size >= 1408:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE W4A8_MXFP4_MXFP8 with many experts "
                f"has accuracy issues (num_experts={num_experts} >= 60)."
            )
        # Issue: W4A8_MXFP4_MXFP8 with swiglu_gptoss_style and top_k=1 has accuracy
        # issues on TRTLLM backend. Observed mismatch ~20-22% exceeds the 20% threshold.
        # CUTLASS backend with the same configuration passes.
        if swiglu_gptoss_style and top_k == 1:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE W4A8_MXFP4_MXFP8 with "
                f"swiglu_gptoss_style and top_k={top_k} has accuracy issues "
                f"(mismatch ~20-22%). CUTLASS backend with the same config passes."
            )

    # TP per-shard alignment: when moe_tp_size > 1, intermediate_size is sharded.
    # MXFP4 variants (W4A16_MXFP4, W4A8_MXFP4_MXFP8) auto-pad to 128 alignment,
    # but other quants (FP8_BLOCK_SCALES, NVFP4, W4A8_NVFP4_FP8) crash:
    #   - FP8_BLOCK_SCALES: block scale tensor size mismatch
    #     (ceil(per_shard/128) vs floor(per_shard/128))
    #   - NVFP4: unswizzle_sf shape '[-1, w3_w1, 128]' invalid
    #   - W4A8_NVFP4_FP8: No valid config for non-aligned N dimension
    if moe_tp_size > 1 and intermediate_size % moe_tp_size == 0:
        per_shard = intermediate_size // moe_tp_size
        tp_crash_quants = {
            QuantAlgo.FP8_BLOCK_SCALES,
            QuantAlgo.NVFP4,
            QuantAlgo.W4A8_NVFP4_FP8,
        }
        if quant_algo in tp_crash_quants and per_shard % 128 != 0:
            return (
                f"TRTLLMGenFusedMoE {quant_algo}: per-shard intermediate_size="
                f"{per_shard} (= {intermediate_size} / {moe_tp_size}) is not "
                f"128-aligned."
            )

    return None


def should_skip_cutedsl(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    model_config: "MoeModelConfig" = None,
    comm_method: Optional[str] = None,
    routing_method_cls=None,
    moe_tp_size: int = 1,
) -> Optional[str]:
    """
    Check CuteDSL backend specific constraints.

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.CUTEDSL:
        return None

    if model_config is None:
        return None

    intermediate_size = model_config.intermediate_size

    # NVFP4 with large intermediate_size has known accuracy issues (8.5% mismatch
    # at i=14336, threshold 3%). Both CuteDSL and reference have FP4 intermediate
    # storage, but produce DIFFERENT FP4 values due to:
    # 1) SwiGLU precision: CuteDSL kernel uses approximate math ops for sigmoid
    #    (rcp_approx + exp2 fastmath, see utils.py:sigmoid_f32), while reference
    #    Triton kernel uses standard tl.sigmoid (see swiglu.py:42).
    # 2) Precision chain: CuteDSL computes SwiGLU in FP32 (GEMM accumulator →
    #    FP32 SwiGLU → FP4), reference goes FP32 accumulator → BF16 → SwiGLU →
    #    BF16 → fp4_quantize. Two BF16 truncation points create different values.
    # 3) FP4 quantization: CuteDSL uses rcp_approx for block scale reciprocal
    #    (blockscaled_...fusion.py:2588), fp4_quantize uses exact division.
    # These per-element FP4 value differences accumulate through FC2 GEMM dot
    # product (K=intermediate_size). CUTLASS avoids this entirely with a single
    # fused kernel keeping BF16 intermediate precision.
    if quant_algo == QuantAlgo.NVFP4 and intermediate_size >= 14336:
        return (
            f"[Design Limitation] CuteDslFusedMoE NVFP4 with large "
            f"intermediate_size has accuracy issues due to FP4 intermediate "
            f"storage between FC1+SwiGLU and FC2 kernels "
            f"(intermediate_size={intermediate_size} >= 14336, "
            f"FC2 accumulates over K={intermediate_size} with 896+ blocks)."
        )

    # NVFP4 with Llama4Renormalize routing has significant accuracy issues.
    # Same root cause as the large intermediate_size skip above: CuteDSL and
    # reference produce different FP4 intermediate values due to approximate
    # math ops (rcp_approx, exp2 fastmath) and BF16 truncation differences.
    # Llama4's sigmoid routing amplifies these differences: standard Renormalize
    # uses softmax (weights sum to 1, per-expert errors averaged), while Llama4
    # uses sigmoid (weights independent in (0,1), per-expert errors summed
    # without normalization). This amplifies FP4 value differences by ~top_k/2.
    # Mismatch correlates with hidden_size (FC1 K dimension): h=512 passes,
    # h=2048 fails 8-17%, h=7168 fails 24-35%. Observed: e60(9.4%),
    # e64(16.5%), e256(34.6%), e384(30.9%) at threshold 3%.
    if routing_method_cls is not None:
        from tensorrt_llm._torch.modules.fused_moe import Llama4RenormalizeMoeRoutingMethod

        if (
            quant_algo == QuantAlgo.NVFP4
            and routing_method_cls == Llama4RenormalizeMoeRoutingMethod
        ):
            return (
                "[Design Limitation] CuteDslFusedMoE NVFP4 with Llama4Renormalize "
                "routing: FP4 intermediate errors amplified by non-normalized "
                "sigmoid routing weights (mismatch up to 34.6%)."
            )

    # TP per-shard alignment: NVFP4 requires 128-aligned per-shard intermediate_size.
    # fp4_utils.py asserts M % 128 == 0 where M = 2 * per_shard (combined w3_w1).
    if moe_tp_size > 1 and quant_algo == QuantAlgo.NVFP4 and intermediate_size % moe_tp_size == 0:
        per_shard = intermediate_size // moe_tp_size
        if per_shard % 128 != 0:
            return (
                f"CuteDslFusedMoE NVFP4: per-shard intermediate_size="
                f"{per_shard} (= {intermediate_size} / {moe_tp_size}) is not "
                f"128-aligned. fp4_utils asserts M % 128 == 0."
            )

    return None


def should_skip_cutlass(
    backend_type: MoeBackendType,
    comm_method: Optional[str] = None,
    quant_algo: Optional[QuantAlgo] = None,
    model_config: "MoeModelConfig" = None,
    moe_tp_size: int = 1,
    dtype=None,
) -> Optional[str]:
    """
    Check CUTLASS backend specific constraints for multi-GPU tests.

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.CUTLASS:
        return None

    # TP per-shard alignment: W8A16, NVFP4, and W4A8_AWQ require 128-aligned
    # per-shard intermediate_size. W8A16 fails in preprocess_weights_for_mixed_gemm
    # (num_rows % rows_per_tile != 0). NVFP4 pads to 128-alignment
    # (NVFP4_ROW_ALIGNMENT in quantization.py:2312) but zero-padding +
    # blockwise quantization interaction causes ~6-7% mismatch.
    # W4A8_AWQ (WInt4AFP8FusedMoEMethod) requires K dimensions to be multiples
    # of 128 on SM90 for interleave factor selection (quantization.py:1310-1324).
    # W4A8_MXFP4_MXFP8 uses MXFP4 auto-padding that handles this correctly.
    if moe_tp_size > 1 and model_config is not None:
        tp_alignment_quants = {
            QuantAlgo.W8A16,
            QuantAlgo.NVFP4,
            QuantAlgo.W4A8_AWQ,
        }
        # FP8_BLOCK_SCALES has this issue only on Hopper (SM90)
        if torch.cuda.get_device_capability(0) == (9, 0):
            tp_alignment_quants.add(QuantAlgo.FP8_BLOCK_SCALES)

        if quant_algo in tp_alignment_quants:
            intermediate_size = model_config.intermediate_size
            if intermediate_size % moe_tp_size == 0:
                per_shard = intermediate_size // moe_tp_size
                if per_shard % 128 != 0:
                    return (
                        f"CutlassFusedMoE {quant_algo}: per-shard "
                        f"intermediate_size={per_shard} "
                        f"(= {intermediate_size} / {moe_tp_size}) is not "
                        f"128-aligned."
                    )

    return None


def should_skip_deepgemm(
    backend_type: MoeBackendType,
    comm_method: Optional[str] = None,
    quant_algo: Optional[QuantAlgo] = None,
    model_config: "MoeModelConfig" = None,
    moe_tp_size: int = 1,
) -> Optional[str]:
    """
    Check DeepGemm backend specific constraints.

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.DEEPGEMM:
        return None

    # TP per-shard alignment: FP8_BLOCK_SCALES requires 128-aligned per-shard
    # intermediate_size for block scale tensor operations.
    if moe_tp_size > 1 and quant_algo == QuantAlgo.FP8_BLOCK_SCALES and model_config is not None:
        intermediate_size = model_config.intermediate_size
        if intermediate_size % moe_tp_size == 0:
            per_shard = intermediate_size // moe_tp_size
            if per_shard % 128 != 0:
                return (
                    f"DeepGemmFusedMoE FP8_BLOCK_SCALES: per-shard "
                    f"intermediate_size={per_shard} "
                    f"(= {intermediate_size} / {moe_tp_size}) is not "
                    f"128-aligned."
                )

    return None


def should_skip_multi_gpu(
    parallel_mode: str,
    model_config: "MoeModelConfig",
    world_size: int = 4,
    comm_method: Optional[str] = None,
) -> Optional[str]:
    """
    Check if a multi-GPU test should be skipped due to EP partitioning constraints.

    In EP modes (DEP, TEP), num_experts must be divisible by ep_size (= world_size)
    when EPLB (Expert Load Balancing) is not enabled. Otherwise the assertion
    `num_experts % ep_size == 0` in interface.py _init_load_balancer will fail.

    Args:
        parallel_mode: Parallelism strategy ("DEP", "TEP", "DTP", "TTP")
        model_config: MoE model configuration containing num_experts
        world_size: Total number of GPUs (default: 4)
        comm_method: Optional communication method (e.g. "DEEPEP", "DEEPEPLOWLATENCY")

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    # Only EP modes have ep_size = world_size; TP modes have ep_size = 1
    if parallel_mode not in ("DEP", "TEP"):
        return None

    ep_size = world_size
    num_experts = model_config.num_experts
    if num_experts % ep_size != 0:
        return (
            f"num_experts={num_experts} is not divisible by ep_size={ep_size} "
            f"in {parallel_mode} mode. Requires EPLB to handle non-uniform "
            f"expert partitioning (tested separately in test_configurable_moe_multi_gpu_eplb)."
        )

    # DeepEP Low Latency requires NVSHMEM IBGDA transport, which needs
    # GPU-side MMIO mapping of InfiniBand UAR (User Access Region).
    # On Hopper (SM90) nodes the cudaHostRegister(IoMemory) call fails
    # (cudaErrorNotSupported), causing IBGDA init to fail.  NVSHMEM v3.2.5
    # has a double-free bug in the IBGDA cleanup path that crashes MPI
    # workers with SIGABRT, leaving the parent process hung forever.
    # Skip on Hopper until NVSHMEM ships a fix or IBRC fallback is enabled.
    if comm_method == "DEEPEPLOWLATENCY":
        if torch.cuda.get_device_capability(0) == (9, 0):
            return (
                "DEEPEPLOWLATENCY requires NVSHMEM IBGDA transport. "
                "Hopper (SM90) nodes lack GPU-side UAR mapping support "
                "(cudaHostRegister IoMemory returns cudaErrorNotSupported), "
                "and NVSHMEM v3.2.5 crashes on IBGDA init failure cleanup."
            )

    return None


def should_skip_routing_method(
    routing_method_cls,
    model_config: "MoeModelConfig",
) -> Optional[str]:
    """
    Check routing method specific constraints that are independent of backend.

    Args:
        routing_method_cls: The routing method class
        model_config: The MoE model configuration

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if routing_method_cls is None or model_config is None:
        return None

    from tensorrt_llm._torch.modules.fused_moe import DeepSeekV3MoeRoutingMethod

    # DeepSeekV3 routing: num_experts must be divisible by n_group for the
    # view operation in noaux_tc (routing.py:298). n_group = max(1, num_experts // 2),
    # so odd num_experts (e.g. 7, 13) fail because num_experts % n_group != 0.
    if routing_method_cls == DeepSeekV3MoeRoutingMethod:
        num_experts = model_config.num_experts
        experts_per_group = 2
        n_group = max(1, num_experts // experts_per_group)
        if n_group > 1 and num_experts % n_group != 0:
            return (
                f"DeepSeekV3 routing requires num_experts divisible by n_group "
                f"(num_experts={num_experts}, n_group={n_group}). "
                f"noaux_tc view([n_group, num_experts // n_group]) fails."
            )

    return None


def supports_autotuner_capture(
    backend_type: MoeBackendType,
    _quant_algo: Optional[QuantAlgo],
    use_flashinfer: bool,
) -> bool:
    """
    Determine if a backend+quant_algo combination supports AutoTuner capture/replay.

    Args:
        backend_type: The MoE backend type
        _quant_algo: The quantization algorithm (None for unquantized).
            Reserved for future per-algorithm gating; currently unused.

    Returns:
        True if autotuner capture/replay is supported, False otherwise
    """
    # DEEPGEMM does not support autotuner capture
    if backend_type == MoeBackendType.DEEPGEMM:
        return False

    if use_flashinfer:
        return False

    return True


def get_quick_skip_reason(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    dtype: torch.dtype,
    model_config: "MoeModelConfig",
    routing_method_cls=None,
    swiglu_gptoss_style: bool = False,
    seq_len: Optional[int] = None,
) -> Optional[str]:
    """
    Fast skip check that calls backend's can_implement() method.

    Unified version supporting both backend-level and module-level tests:
    - routing_method_cls: Used by test_moe_module.py for routing method compatibility checks
    - swiglu_gptoss_style: Used by test_moe_backend.py for SwiGLU parameter checks
    - seq_len: Optional sequence length for seq_len-sensitive skip checks

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    import logging as _logging

    # Suppress logger warnings during parameter generation
    trtllm_logger = _logging.getLogger("tensorrt_llm")
    original_level = trtllm_logger.level
    trtllm_logger.setLevel(_logging.ERROR)

    try:
        # Call backend's can_implement for dtype/quant_algo checks
        backend_cls = get_backend_class(backend_type)
        can_impl_kwargs = {"dtype_activation": dtype}
        if swiglu_gptoss_style:
            can_impl_kwargs["swiglu_gptoss_style"] = swiglu_gptoss_style
        can_impl, skip_reason = backend_cls.can_implement(quant_algo, **can_impl_kwargs)
        if not can_impl:
            return skip_reason

        # Chain skip checks: routing method, then per-backend constraints
        skip_checks = [
            lambda: should_skip_routing_method(routing_method_cls, model_config),
            lambda: should_skip_trtllm(
                backend_type,
                quant_algo,
                model_config,
                routing_method_cls,
                swiglu_gptoss_style,
                seq_len=seq_len,
            ),
            lambda: should_skip_cutedsl(
                backend_type, quant_algo, model_config, routing_method_cls=routing_method_cls
            ),
            lambda: should_skip_deepgemm(
                backend_type, quant_algo=quant_algo, model_config=model_config
            ),
        ]
        for check in skip_checks:
            skip_reason = check()
            if skip_reason:
                return skip_reason

        # DEEPGEMM: float16 reference module constraint
        if backend_type == MoeBackendType.DEEPGEMM and dtype == torch.float16:
            return "DeepGemmFusedMoE reference module requires bfloat16 input"

        # 128-alignment requirement for quantization
        if quant_algo is not None:
            hidden_size = model_config.hidden_size
            intermediate_size = model_config.intermediate_size
            is_hidden_128_aligned = hidden_size % 128 == 0
            is_intermediate_128_aligned = intermediate_size % 128 == 0

            if not is_hidden_128_aligned or not is_intermediate_128_aligned:
                # TRTLLM with MXFP4 variants automatically pads to 128 alignment
                is_mxfp4_variant = quant_algo in {QuantAlgo.W4A16_MXFP4, QuantAlgo.W4A8_MXFP4_MXFP8}
                is_trtllm_backend = backend_type == MoeBackendType.TRTLLM
                if not (is_trtllm_backend and is_mxfp4_variant):
                    return (
                        f"Non-128-aligned sizes (h={hidden_size}, i={intermediate_size}) "
                        f"require TRTLLM backend with MXFP4 quantization"
                    )

        return None

    finally:
        trtllm_logger.setLevel(original_level)


# ============================================================================
# GPU Memory Check
# ============================================================================
def skip_if_insufficient_gpu_memory(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype = torch.float32,
    overhead_factor: float = 4.0,
) -> None:
    """
    Skip the current test if estimated GPU memory exceeds device capacity.

    Each expert has gate_up_proj [2*I, H] + down_proj [H, I] = 3*H*I elements.
    The overhead_factor (default 4x) accounts for ref model + DUT model +
    quantization scales/activations + CUDA allocator overhead.

    Args:
        num_experts: Number of MoE experts
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate (FFN) dimension size
        dtype: Weight data type for byte-size calculation
        overhead_factor: Multiplier over single-model weight bytes
    """
    if not torch.cuda.is_available():
        return
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    single_model_bytes = num_experts * 3 * hidden_size * intermediate_size * bytes_per_elem
    estimated_total_bytes = int(single_model_bytes * overhead_factor)
    gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
    if estimated_total_bytes > gpu_total_bytes:
        pytest.skip(
            f"Estimated memory {estimated_total_bytes / (1 << 30):.1f}GB "
            f"exceeds GPU memory {gpu_total_bytes / (1 << 30):.1f}GB "
            f"(num_experts={num_experts}, hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}, dtype={dtype})"
        )


# ============================================================================
# Autotuner Tactic Replay
# ============================================================================
def replay_tactics_and_check(
    all_tactics,
    run_moe_fn: Callable[[], torch.Tensor],
    check_accuracy_fn: Callable[[torch.Tensor, torch.Tensor], None],
    ref_output: torch.Tensor,
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    fail_fast: bool = False,
) -> None:
    """
    Replay all tactics and check accuracy.

    Args:
        all_tactics: TacticsCapture object from AutoTuner.capture()
        run_moe_fn: Function to run MoE computation
        check_accuracy_fn: Function to check accuracy (output, ref_output) -> None
        ref_output: Reference output tensor
        backend_type: Backend type for error reporting
        quant_algo: Quantization algorithm for error reporting
        fail_fast: If True, fail on first error. If False, run all and report summary.
    """
    tactics_list = list(all_tactics)
    passed_tactics = []
    failed_tactics = []
    G_LOGGER.info(f"Replay tactics : {len(tactics_list)} and check accuracy")
    for idx, tactic in enumerate(tactics_list):
        with AutoTuner.get().replay(tactic), torch.inference_mode():
            output = run_moe_fn()
            try:
                check_accuracy_fn(output, ref_output)
                passed_tactics.append((idx, tactic))
            except Exception as e:
                if fail_fast:
                    pytest.fail(
                        f"Accuracy check failed for tactic[{idx}/{len(tactics_list)}]={tactic}, "
                        f"backend={backend_type}, quant_algo={quant_algo}: {e}"
                    )
                failed_tactics.append((idx, tactic, str(e)))

    # Report results (only when fail_fast=False)
    total = len(tactics_list)
    num_passed = len(passed_tactics)
    num_failed = len(failed_tactics)
    if failed_tactics:
        fail_details = "\n".join(
            f"  tactic[{idx}]={tactic}: {err}" for idx, tactic, err in failed_tactics
        )
        pytest.fail(
            f"backend={backend_type}, quant_algo={quant_algo}: "
            f"{num_passed}/{total} passed, {num_failed}/{total} failed\n"
            f"Failed tactics:\n{fail_details}"
        )


# ============================================================================
# Test Parameter Helpers
# ============================================================================
def create_test_param(param_values, test_id, skip_reason=None):
    """Create a pytest.param with optional skip mark."""
    if skip_reason:
        return pytest.param(*param_values, id=test_id, marks=pytest.mark.skip(reason=skip_reason))
    return pytest.param(*param_values, id=test_id)


# ============================================================================
# CI Mode Detection
# ============================================================================
_TRTLLM_TEST_MOE_CI_ENV = "TRTLLM_TEST_MOE_CI"
IS_CI_MODE = os.environ.get(_TRTLLM_TEST_MOE_CI_ENV, "1") == "1"

# ============================================================================
# CI Acceleration Skip Logic
# ============================================================================

# Routing methods that require full routing coverage in CI
_CI_ROUTING_METHODS = {"Renormalize", "DeepSeekV3"}


def should_skip_to_accelerate_ci(
    backend_type: "MoeBackendType",
    quant_algo: Optional[QuantAlgo],
    model_config: "MoeModelConfig",
    routing_method_cls=None,
    dtype: Optional[torch.dtype] = None,
    seq_len: Optional[int] = None,
    swiglu_gptoss_style: bool = False,
    parallel_mode: Optional[str] = None,
    activation_type: Optional[ActivationType] = ActivationType.Swiglu,
) -> Optional[str]:
    """
    Skip low-information-density test combinations to accelerate CI.

    Only active when TRTLLM_TEST_MOE_CI=1 (default). When TRTLLM_TEST_MOE_CI=0,
    all combinations run (local exhaustive testing).

    Rules applied (in order):
    0. Skip unquantized (quant=None) for most paths, but keep TRTLLM BF16
       unquantized coverage enabled.
    1. e256 model: only DeepSeekV3 routing, bfloat16, seq=1, non-gptoss
    2. Multi-GPU: only DEP and TTP parallel modes
    3. Routing: full 6 routing methods only on (CUTLASS or TRTLLM) with NVFP4;
       other backend+quant combos only run Renormalize
       and DeepSeekV3. This rule is overridden by rule 1 for e256.

    Args:
        backend_type: MoE backend type
        quant_algo: Quantization algorithm
        model_config: MoE model configuration
        routing_method_cls: Routing method class (None means no routing filter)
        dtype: Activation data type
        seq_len: Sequence length
        swiglu_gptoss_style: Whether using SwiGLU gptoss style
        parallel_mode: Multi-GPU parallel mode (None for single-GPU tests)

    Returns:
        Skip reason string if test should be skipped for CI, None otherwise
    """
    if not IS_CI_MODE:
        return None

    if model_config is None:
        return None

    # --- Rule 0: Skip gated and unquantized (quant=None) for most backends ---
    # Keep TRTLLM BF16 unquantized enabled to cover FlashInfer BF16 TRTLLM MoE.
    if (
        quant_algo is None
        and is_gated_activation(activation_type)
        and not (backend_type == MoeBackendType.TRTLLM and dtype == torch.bfloat16)
    ):
        return "[CI accel] Skip unquantized (quant=None) in CI"

    is_large_model = model_config.num_experts >= 256 and model_config.hidden_size >= 7168

    # --- Rule 1: Large model (e256_k8_h7168_i2048) restrictions ---
    if is_large_model:
        if routing_method_cls is not None:
            from tensorrt_llm._torch.modules.fused_moe import DeepSeekV3MoeRoutingMethod

            if routing_method_cls != DeepSeekV3MoeRoutingMethod:
                routing_name = routing_method_cls.__name__
                return (
                    f"[CI accel] Large model (num_experts={model_config.num_experts}) "
                    f"only tests DeepSeekV3 routing in CI (got {routing_name})"
                )

        if dtype is not None and dtype != torch.bfloat16:
            return f"[CI accel] Large model only tests bfloat16 in CI (got {dtype})"

        if seq_len is not None and seq_len != 1:
            return f"[CI accel] Large model only tests seq=1 in CI (got seq={seq_len})"

        if swiglu_gptoss_style:
            return "[CI accel] Large model only tests non-gptoss in CI"

    # --- Rule 2: Multi-GPU parallel mode restrictions ---
    if parallel_mode is not None and parallel_mode not in ("DEP", "TTP"):
        return f"[CI accel] Only DEP and TTP parallel modes in CI (got {parallel_mode})"

    # --- Rule 3: Routing method restrictions per backend+quant ---
    # Full routing coverage on: (CUTLASS, or TRTLLM) with NVFP4
    # Other combos: only Renormalize + DeepSeekV3
    # Rule 1 already handles e256 (DeepSeekV3 only), so this only applies to non-e256.
    if not is_large_model and routing_method_cls is not None:
        routing_name = routing_method_cls.__name__.replace("MoeRoutingMethod", "")
        if routing_name not in _CI_ROUTING_METHODS:
            allows_full_routing = (
                backend_type == MoeBackendType.CUTLASS or backend_type == MoeBackendType.TRTLLM
            ) and quant_algo == QuantAlgo.NVFP4
            if not allows_full_routing:
                return (
                    f"[CI accel] {backend_type.value}+{quant_algo} only tests "
                    f"Renormalize/DeepSeekV3 routing in CI (got {routing_name})"
                )

    return None


# ============================================================================
# Timing Fixture
# ============================================================================
@pytest.fixture(scope="module", autouse=True)
def module_timer(request):
    """Fixture to measure and log total module execution time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    G_LOGGER.info(
        "[TIMING] Total %s: %.3fs (%.2f min)",
        request.module.__name__,
        elapsed,
        elapsed / 60,
    )


# ============================================================================
# Base Test Config Iterator
# ============================================================================
def iter_base_test_configs(
    swiglu_combos, model_configs, seq_lens, dtypes, backend_types, quant_algos, routing_methods=None
):
    """
    Iterate over base test configurations using itertools.product.

    This is shared by test_moe_backend.py and test_moe_module.py.
    When routing_methods is None, defaults to [RenormalizeMoeRoutingMethod].

    Args:
        swiglu_combos: List of (swiglu_alpha, swiglu_beta, swiglu_limit) tuples
        model_configs: List of MoeModelConfig
        seq_lens: List of sequence lengths
        dtypes: List of data types
        backend_types: List of backend types
        quant_algos: List of quantization algorithms
        routing_methods: List of routing method classes (default: [RenormalizeMoeRoutingMethod])

    Yields:
        Tuple of (swiglu_alpha, swiglu_beta, swiglu_limit, model_config, seq_len,
                  dtype, backend_type, quant_algo, routing_method_cls, skip_reason, base_test_id)
    """
    if routing_methods is None:
        from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod

        routing_methods = [RenormalizeMoeRoutingMethod]

    for (
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
    ), model_config, seq_len, dtype, backend_type, quant_algo, routing_method_cls in product(
        swiglu_combos, model_configs, seq_lens, dtypes, backend_types, quant_algos, routing_methods
    ):
        swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")
        skip_reason = get_quick_skip_reason(
            backend_type,
            quant_algo,
            dtype,
            model_config,
            routing_method_cls,
            swiglu_gptoss_style=swiglu_gptoss_style,
            seq_len=seq_len,
        )
        routing_name = routing_method_cls.__name__.replace("MoeRoutingMethod", "")
        swiglu_id = (
            f"alpha={swiglu_alpha}_beta={swiglu_beta}_limit={swiglu_limit}-"
            if swiglu_gptoss_style
            else ""
        )
        base_test_id = (
            f"{swiglu_id}{model_config}-seq={seq_len}-dtype={dtype}-"
            f"backend={backend_type.value}-quant={quant_algo}-routing={routing_name}"
        )
        yield (
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            model_config,
            seq_len,
            dtype,
            backend_type,
            quant_algo,
            routing_method_cls,
            skip_reason,
            base_test_id,
        )
