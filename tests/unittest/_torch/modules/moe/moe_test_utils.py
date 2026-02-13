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
import time
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Callable, Optional, Type

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
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
def should_skip_trtllm(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    model_config: "MoeModelConfig",
    routing_method_cls=None,
    swiglu_gptoss_style: bool = False,
    comm_method: Optional[str] = None,
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

    # -----------------Potential issues------------------
    # These are known issues that need investigation. Skipping to avoid test failures
    # and CUDA errors that can cascade to subsequent tests.

    # Issue: W4A8_NVFP4_FP8 with top_k=1 causes CUDA illegal memory access
    if quant_algo == QuantAlgo.W4A8_NVFP4_FP8 and top_k == 1:
        return (
            "[Potential Bug] TRTLLMGenFusedMoE W4A8_NVFP4_FP8 with top_k=1 "
            "causes CUDA illegal memory access."
        )

    # Issue: NVFP4 with large intermediate_size has known accuracy issues
    if quant_algo == QuantAlgo.NVFP4 and intermediate_size >= 14336:
        return (
            f"[Potential Bug] TRTLLMGenFusedMoE NVFP4 with large intermediate_size "
            f"has known accuracy issues (intermediate_size={intermediate_size} >= 14336)."
        )

    # Issue: W4A8_MXFP4_MXFP8 has accuracy issues on certain model configs
    if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
        if intermediate_size >= 14336:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE W4A8_MXFP4_MXFP8 with large "
                f"intermediate_size has accuracy issues (intermediate_size={intermediate_size} >= 14336)."
            )
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

    # Issue: Certain TRTLLM kernel runners crash with CUDA errors in multi-GPU
    # DeepEP mode. the crash is specific to EP with DeepEP.
    # Verified on 4 GPUs with DEP + DEEPEP + TRTLLM (e60_k4_h2048_i1408):
    #   - FP8_BLOCK_SCALES:  CRASH   (fp8_block_scale_moe_runner -> CUDA_ERROR_INVALID_HANDLE)
    #   - W4A16_MXFP4:       CRASH   (bf16_mxe2m1_block_scale_moe_runner -> illegal memory access)
    #   - W4A8_MXFP4_MXFP8:  likely crash (same mxe2m1 kernel family as W4A16_MXFP4)
    if comm_method in ("DEEPEP", "DEEPEPLOWLATENCY"):
        deepep_crash_quant_algos = {
            QuantAlgo.FP8_BLOCK_SCALES,
            QuantAlgo.W4A16_MXFP4,
            QuantAlgo.W4A8_MXFP4_MXFP8,
        }
        if quant_algo in deepep_crash_quant_algos:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE {quant_algo} crashes with "
                f"CUDA error in multi-GPU DeepEP mode (comm={comm_method}). "
                f"Single-GPU tests pass; issue is in the kernel runner under EP."
            )

    return None


def should_skip_cutedsl(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    model_config: "MoeModelConfig" = None,
    comm_method: Optional[str] = None,
    routing_method_cls=None,
) -> Optional[str]:
    """
    Check CuteDSL backend specific constraints.

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.CUTEDSL:
        return None

    # DeepEPLowLatency _modify_output_to_adapt_fused_moe converts dispatch output
    # to a format where token_selected_slots has shape [num_local_experts, tokens_per_expert]
    # instead of [num_tokens, top_k]. CuteDSL moe_sort asserts
    # token_selected_experts.size(1) == top_k, which fails with this format.
    if comm_method == "DEEPEPLOWLATENCY":
        return (
            "[Potential Bug] CuteDslFusedMoE is incompatible with DeepEPLowLatency: "
            "DeepEPLowLatency _modify_output_to_adapt_fused_moe reshapes "
            "token_selected_slots to [num_local_experts, tokens_per_expert] "
            "(effectively top_k=1), but CuteDSL moe_sort requires "
            "token_selected_experts.size(1) == top_k."
        )

    if model_config is None:
        return None

    intermediate_size = model_config.intermediate_size
    num_experts = model_config.num_experts

    # NVFP4 with large intermediate_size has known accuracy issues
    if quant_algo == QuantAlgo.NVFP4 and intermediate_size >= 14336:
        return (
            f"[Potential Bug] CuteDslFusedMoE NVFP4 with large intermediate_size "
            f"has known accuracy issues (intermediate_size={intermediate_size} >= 14336)."
        )

    # NVFP4 with prime num_experts causes CUDA_ERROR_ILLEGAL_ADDRESS
    prime_experts_with_issues = {7, 13}
    if quant_algo == QuantAlgo.NVFP4 and num_experts in prime_experts_with_issues:
        return (
            f"[Potential Bug] CuteDslFusedMoE NVFP4 with prime num_experts={num_experts} "
            f"causes CUDA_ERROR_ILLEGAL_ADDRESS due to autotuner cache bucket mapping."
        )

    # NVFP4 with Llama4Renormalize routing has significant accuracy issues on bfloat16.
    # Observed mismatch up to 34.6% (threshold 2% at rtol=0.01, percent=0.98).
    if routing_method_cls is not None:
        from tensorrt_llm._torch.modules.fused_moe import Llama4RenormalizeMoeRoutingMethod

        if (
            quant_algo == QuantAlgo.NVFP4
            and routing_method_cls == Llama4RenormalizeMoeRoutingMethod
        ):
            return (
                "[Potential Bug] CuteDslFusedMoE NVFP4 with Llama4Renormalize "
                "routing has significant accuracy issues (mismatch up to 34.6%%)."
            )

    return None


def should_skip_deepgemm(
    backend_type: MoeBackendType,
    comm_method: Optional[str] = None,
    quant_algo: Optional[QuantAlgo] = None,
    model_config: "MoeModelConfig" = None,
) -> Optional[str]:
    """
    Check DeepGemm backend specific constraints.

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.DEEPGEMM:
        return None

    # DeepGemm workspace allocation in set_strides (fused_moe_deepgemm.py) uses a
    # storage size that is 4x too small when combined with DeepEPLowLatency dispatch.
    # The workspace is allocated based on assumptions that do not account for the
    # DeepEPLowLatency output format ([num_local_experts, ep_size * max_tokens, hidden_size]).
    if comm_method == "DEEPEPLOWLATENCY":
        return (
            "[Potential Bug] DeepGemmFusedMoE workspace allocation is incompatible "
            "with DeepEPLowLatency: set_strides requires storage of "
            "[num_local_experts * tokens * hidden_size] bytes but the allocated "
            "workspace is ~4x too small, causing setStorage out of bounds."
        )

    # Issue: DEEPGEMM + FP8_BLOCK_SCALES crashes with CUDA illegal memory access
    # on large expert counts (e.g. e384_k8_h7168_i2048) during post_load_weights().
    # The crash occurs in get_col_major_tma_aligned_packed_tensor (fp8_utils.py)
    # when resmoothing FP8 E8M0 scales on SM100f (Blackwell).
    # Small configs (e.g. e60_k4_h2048_i1408) pass fine.
    if quant_algo == QuantAlgo.FP8_BLOCK_SCALES and model_config is not None:
        if model_config.num_experts > 128:
            return (
                f"[Potential Bug] DeepGemmFusedMoE FP8_BLOCK_SCALES crashes with "
                f"CUDA illegal memory access on large expert count "
                f"(num_experts={model_config.num_experts}). The crash occurs in "
                f"get_col_major_tma_aligned_packed_tensor during "
                f"post_load_weights() FP8 E8M0 scale resmoothing on SM100f."
            )

    return None


def should_skip_multi_gpu(
    parallel_mode: str,
    model_config: "MoeModelConfig",
    world_size: int = 4,
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
            f"expert partitioning (tested separately in test_ConfigurableMoE_multi_gpu_eplb)."
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

    return True


def get_quick_skip_reason(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    dtype: torch.dtype,
    model_config: "MoeModelConfig",
    routing_method_cls=None,
    swiglu_gptoss_style: bool = False,
) -> Optional[str]:
    """
    Fast skip check that calls backend's can_implement() method.

    Unified version supporting both backend-level and module-level tests:
    - routing_method_cls: Used by test_moe_module.py for routing method compatibility checks
    - swiglu_gptoss_style: Used by test_moe_backend.py for SwiGLU parameter checks

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
                backend_type, quant_algo, model_config, routing_method_cls, swiglu_gptoss_style
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
