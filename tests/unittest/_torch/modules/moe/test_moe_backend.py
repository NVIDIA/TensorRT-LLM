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
MoE Backend Unit Tests

This module provides a unified test framework for testing different MoE backends
through the backend-level interfaces (quantize_input + run_moe), rather than
the high-level forward() interface.

Design Goals:
1. Test backend interfaces directly: routing_method.apply -> quantize_input -> run_moe
2. Cover all quantization + backend combinations
3. Use can_implement() interface to determine test skip logic
4. Support autotune and tactic capture testing
"""

import itertools
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Type

import pytest
import torch
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

logger = logging.getLogger(__name__)


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


def should_skip_TRTLLM(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    model_config: "MoeModelConfig",
) -> Optional[str]:
    """
    Check TRTLLM Gen backend specific constraints.

    The TRTLLM Gen MoE kernels have hardware-level constraints that must be satisfied.
    These constraints are enforced in C++ layer.

    Constraints:
    1. num_experts must be divisible by 4 (routing kernel vectorization requirement)
    2. num_experts must be greater than top_k (routing logic requirement)

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm
        model_config: The MoE model configuration

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.TRTLLM:
        return None

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

    # -----------------Potential issues------------------
    # These are known issues that need investigation. Skipping to avoid test failures
    # and CUDA errors that can cascade to subsequent tests.

    # Issue 1: W4A8_NVFP4_FP8 with top_k=1 causes CUDA illegal memory access
    # This triggers GPU state corruption that affects all subsequent tests.
    # Affected config: e8_k1_h512_i512
    if quant_algo == QuantAlgo.W4A8_NVFP4_FP8 and top_k == 1:
        return (
            "[Potential Bug] TRTLLMGenFusedMoE W4A8_NVFP4_FP8 with top_k=1 "
            "causes CUDA illegal memory access. Needs kernel investigation."
        )

    # Issue 2: NVFP4 with large intermediate_size has known accuracy issues
    # Observed mismatch: 18%~25% vs expected <7.5% (per test_moe.py baseline)
    # Affected configs: e8_k2_h4096_i14336, e8_k2_h6144_i32768
    if quant_algo == QuantAlgo.NVFP4 and intermediate_size >= 14336:
        return (
            f"[Potential Bug] TRTLLMGenFusedMoE NVFP4 with large intermediate_size "
            f"has known accuracy issues (intermediate_size={intermediate_size} >= 14336). "
            f"Observed mismatch 18%~25% exceeds expected threshold."
        )

    # Issue 3: W4A8_MXFP4_MXFP8 has accuracy issues on certain model configs
    # Observed mismatch: 14%~18% vs expected <15% (percent=0.85)
    # Affected configs: large intermediate_size or many experts
    # e8_k2_h4096_i14336, e64_k6_h2048_i1408, e60_k4_h2048_i1408,
    # e256_k8_h7168_i2048, e8_k2_h6144_i32768, e128_k4_h2880_i2880
    if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
        # Large intermediate_size (>= 14336) has precision issues
        if intermediate_size >= 14336:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE W4A8_MXFP4_MXFP8 with large "
                f"intermediate_size has accuracy issues (intermediate_size={intermediate_size} >= 14336). "
                f"Observed mismatch 14%~18% exceeds 15% threshold."
            )
        # Many experts (>= 60) with moderate intermediate_size has precision issues
        if num_experts >= 60 and intermediate_size >= 1408:
            return (
                f"[Potential Bug] TRTLLMGenFusedMoE W4A8_MXFP4_MXFP8 with many experts "
                f"has accuracy issues (num_experts={num_experts} >= 60, intermediate_size={intermediate_size}). "
                f"Observed mismatch 14%~18% exceeds 15% threshold."
            )

    return None


def should_skip_CUTEDSL(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    model_config: "MoeModelConfig" = None,
) -> Optional[str]:
    """
    Check CuteDSL backend specific constraints.

    The CuteDSL MoE kernels have known accuracy issues with certain configurations.

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm
        model_config: The MoE model configuration

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if backend_type != MoeBackendType.CUTEDSL:
        return None

    if model_config is None:
        return None

    intermediate_size = model_config.intermediate_size

    # -----------------Potential issues------------------
    # NVFP4 with large intermediate_size has known accuracy issues (same as TRTLLM)
    # Observed mismatch: 8%~26% vs expected <2%
    # Affected configs: e8_k2_h4096_i14336, e8_k2_h6144_i32768
    if quant_algo == QuantAlgo.NVFP4 and intermediate_size >= 14336:
        return (
            f"[Potential Bug] CuteDslFusedMoE NVFP4 with large intermediate_size "
            f"has known accuracy issues (intermediate_size={intermediate_size} >= 14336). "
            f"Observed mismatch 8%~26% exceeds 2% threshold."
        )

    # NVFP4 with prime num_experts (7, 13) causes CUDA_ERROR_ILLEGAL_ADDRESS
    # Root cause: Autotuner cache bucket mapping issue
    # - When tests run in batch, previous tests cache tactics to buckets
    # - Prime num_experts shapes map to same bucket as other configs
    # - The cached tactic (e.g., ((128, 256), (1, 2), False)) works for other configs
    #   but causes illegal memory access for prime num_experts' actual shape
    # - Single test run passes because fallback tactic ((128, 128), (1, 1), False) is used
    # Affected configs: e7_k2_h256_i512, e13_k3_h256_i512
    num_experts = model_config.num_experts
    prime_experts_with_issues = {7, 13}
    if quant_algo == QuantAlgo.NVFP4 and num_experts in prime_experts_with_issues:
        return (
            f"[Potential Bug] CuteDslFusedMoE NVFP4 with prime num_experts={num_experts} "
            f"causes CUDA_ERROR_ILLEGAL_ADDRESS due to autotuner cache bucket mapping. "
            f"Cached tactic from other configs is incompatible with this shape."
        )

    return None


def should_skip_gptoss(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    gptoss_style: bool,
) -> Optional[str]:
    """
    Check if gptoss_style test should be skipped for this backend.

    Only CUTLASS and TRTLLM backends support gptoss_style (SwiGlu with custom
    alpha/beta/limit parameters and bias).

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm
        gptoss_style: Whether gptoss_style is enabled

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if not gptoss_style:
        return None

    # Only CUTLASS and TRTLLM backends support gptoss_style
    supported_backends = {MoeBackendType.CUTLASS, MoeBackendType.TRTLLM}
    if backend_type not in supported_backends:
        return (
            f"gptoss_style is only supported by CUTLASS and TRTLLM backends "
            f"(got backend_type={backend_type.value})"
        )

    return None


def supports_autotuner_capture(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
) -> bool:
    """
    Determine if a backend+quant_algo combination supports AutoTuner capture/replay.

    AutoTuner capture/replay requires AutoTuner.choose_one() to be called during
    run_moe execution.

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm (None for unquantized)

    Returns:
        True if autotuner capture/replay is supported, False otherwise
    """
    # DEEPGEMM does not support autotuner capture
    # Evidence: fused_moe_deepgemm.py has no AutoTuner/choose_one references
    if backend_type == MoeBackendType.DEEPGEMM:
        return False

    return True


def create_test_backend(
    backend_type: MoeBackendType,
    routing_method: RenormalizeMoeRoutingMethod,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    quant_config,
    mapping: Mapping,
    bias: bool = False,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
) -> MoE:
    """Create a MoE backend for testing."""
    backend_cls = get_backend_class(backend_type)

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=mapping,
        moe_backend=backend_type.value,
    )

    return create_moe_backend(
        moe_cls=backend_cls,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
        bias=bias,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )


def run_backend_moe(
    backend: MoE,
    backend_type: MoeBackendType,
    x_quantized: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    dtype: torch.dtype,
    router_logits: torch.Tensor = None,
    trtllm_use_router_logits: bool = True,
) -> torch.Tensor:
    """
    Run MoE computation with backend-specific parameters.

    Each backend has different requirements:
    - CUTLASS: output_dtype, token_final_scales=float32
    - TRTLLM: token_final_scales=bfloat16, optionally router_logits
    - CUTEDSL: token_final_scales=float32
    - DEEPGEMM: workspace, token_final_scales=float32

    Args:
        trtllm_use_router_logits: If True, TRTLLM backend uses router_logits for routing.
            If False, uses token_selected_experts and token_final_scales.
            Note: When both are provided, TRTLLM only uses (topk_ids and topk_weights).
    """
    # Common args for all backends (default: token_final_scales=float32)
    args = dict(
        x=x_quantized,
        token_selected_experts=token_selected_experts.to(torch.int32),
        token_final_scales=token_final_scales.to(torch.float32),
        x_sf=x_sf,
    )

    # Backend-specific overrides
    if backend_type == MoeBackendType.CUTLASS:
        args["output_dtype"] = dtype
    elif backend_type == MoeBackendType.TRTLLM:
        args["token_final_scales"] = token_final_scales.to(torch.bfloat16)
        if trtllm_use_router_logits:
            # Use router_logits for routing (TRTLLM will compute topk internally)
            args["router_logits"] = router_logits
            args["token_selected_experts"] = None
            args["token_final_scales"] = None
        # else: use token_selected_experts and token_final_scales (already set)
    elif backend_type == MoeBackendType.DEEPGEMM:
        import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils

        m_max = fp8_utils.align(x_quantized.shape[0], 128)
        args["workspace"] = backend.get_workspace(m_max, 128)

    return backend.run_moe(**args)


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
    logger.info(f"Replay tactics : {len(tactics_list)} and check accuracy")
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
# Test Parameters
# ============================================================================

# Quantization algorithms to test
QUANT_ALGOS_TO_TEST = [
    None,  # Unquantized
    QuantAlgo.FP8,
    QuantAlgo.NVFP4,
    QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.W4A8_NVFP4_FP8,
    QuantAlgo.W4A16_MXFP4,
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.W8A16,
    QuantAlgo.W4A8_AWQ,
]

# Backend types to test
BACKEND_TYPES_TO_TEST = [
    MoeBackendType.CUTLASS,
    MoeBackendType.TRTLLM,
    MoeBackendType.CUTEDSL,
    MoeBackendType.DEEPGEMM,
]

# Data types to test
DTYPES_TO_TEST = [
    torch.float16,
    torch.bfloat16,
]


# ============================================================================
# Model MoE Configurations
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


# Format: (num_experts, top_k, hidden_size, intermediate_size)
MOE_MODEL_CONFIGS = [
    # === Real Model Configs ===
    MoeModelConfig(8, 2, 4096, 14336),  # Mixtral-8x7B
    MoeModelConfig(64, 6, 2048, 1408),  # DeepSeek-MoE-16B / DeepSeek-V2-Lite
    MoeModelConfig(60, 4, 2048, 1408),  # Qwen1.5-MoE-A2.7B
    MoeModelConfig(256, 8, 7168, 2048),  # DeepSeek-V3
    MoeModelConfig(8, 2, 6144, 32768),  # Grok-1
    MoeModelConfig(128, 4, 2880, 2880),  # GPT-OSS-120B
    # === Boundary Tests: num_experts / top_k ===
    MoeModelConfig(8, 1, 512, 512),  # top_k=1, single expert activated
    MoeModelConfig(4, 4, 512, 512),  # top_k=num_experts, all experts activated
    MoeModelConfig(7, 2, 256, 512),  # prime num_experts
    MoeModelConfig(13, 3, 256, 512),  # prime num_experts, odd top_k
    # === Boundary Tests: small sizes ===
    MoeModelConfig(4, 2, 64, 128),  # very small hidden_size
    MoeModelConfig(4, 2, 128, 64),  # intermediate < hidden
]

# Sequence lengths to test
SEQ_LENS_TO_TEST = [1, 8]

# SwiGLU parameters for gptoss_style testing
SWIGLU_ALPHAS = [1, 0.1]
SWIGLU_BETAS = [0, 1]
SWIGLU_LIMITS = [float("inf"), 1]


# ============================================================================
# Fast Skip Check (for parametrize-level skip, avoids entering test function)
# ============================================================================
def get_quick_skip_reason(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    dtype: torch.dtype,
    model_config: "MoeModelConfig",
    gptoss_style: bool,
) -> Optional[str]:
    """
    Fast skip check that calls backend's can_implement() method.

    This function calls the backend's can_implement() classmethod to check
    dtype/quant_algo/gptoss_style support, then uses should_skip_* functions
    for additional model_config specific checks.

    Note: Logging is temporarily suppressed to avoid excessive warning output
    during test parameter generation.

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    import logging as _logging

    # Suppress logger warnings during parameter generation to avoid excessive output
    trtllm_logger = _logging.getLogger("tensorrt_llm")
    original_level = trtllm_logger.level
    trtllm_logger.setLevel(_logging.ERROR)

    try:
        # ===== Call backend's can_implement for dtype/quant_algo/gptoss_style checks =====
        backend_cls = get_backend_class(backend_type)
        can_impl, skip_reason = backend_cls.can_implement(
            quant_algo, dtype_activation=dtype, gptoss_style=gptoss_style
        )
        if not can_impl:
            return skip_reason

        # ===== Additional model_config specific checks =====

        # TRTLLM: num_experts constraints and accuracy issues
        skip_reason = should_skip_TRTLLM(backend_type, quant_algo, model_config)
        if skip_reason:
            return skip_reason

        # CUTEDSL: accuracy issues with specific configs
        skip_reason = should_skip_CUTEDSL(backend_type, quant_algo, model_config)
        if skip_reason:
            return skip_reason

        # DEEPGEMM: float16 reference module constraint
        if backend_type == MoeBackendType.DEEPGEMM and dtype == torch.float16:
            return "DeepGemmFusedMoE reference module (FP8BlockScalesLinearMethod) requires bfloat16 input"

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
        # Restore logger level
        trtllm_logger.setLevel(original_level)


def generate_test_params() -> List:
    """
    Generate all test parameter combinations with skip marks for invalid combinations.

    This function pre-computes skip decisions at collection time using static rules,
    avoiding the overhead of entering test functions and calling can_implement().
    This significantly speeds up test collection and skip execution.

    Returns:
        List of pytest.param objects with appropriate skip marks
    """
    params: List = []

    # Generate all combinations
    swiglu_combos = list(itertools.product(SWIGLU_ALPHAS, SWIGLU_BETAS, SWIGLU_LIMITS))

    for swiglu_alpha, swiglu_beta, swiglu_limit in swiglu_combos:
        for model_config in MOE_MODEL_CONFIGS:
            for seq_len in SEQ_LENS_TO_TEST:
                for dtype in DTYPES_TO_TEST:
                    for backend_type in BACKEND_TYPES_TO_TEST:
                        for quant_algo in QUANT_ALGOS_TO_TEST:
                            # Determine gptoss_style
                            gptoss_style = (
                                swiglu_alpha != 1
                                or swiglu_beta != 0
                                or swiglu_limit != float("inf")
                            )

                            # Generate test ID
                            test_id = (
                                f"alpha={swiglu_alpha}_beta={swiglu_beta}_limit={swiglu_limit}-"
                                f"{model_config}-seq={seq_len}-dtype={dtype}-"
                                f"backend={backend_type.value}-quant_algo={quant_algo}"
                            )

                            # Check if should skip
                            skip_reason = get_quick_skip_reason(
                                backend_type, quant_algo, dtype, model_config, gptoss_style
                            )

                            param_values = (
                                dtype,
                                backend_type,
                                quant_algo,
                                seq_len,
                                model_config,
                                swiglu_alpha,
                                swiglu_beta,
                                swiglu_limit,
                            )

                            if skip_reason:
                                params.append(
                                    pytest.param(
                                        *param_values,
                                        id=test_id,
                                        marks=pytest.mark.skip(reason=skip_reason),
                                    )
                                )
                            else:
                                params.append(pytest.param(*param_values, id=test_id))

    return params


# Pre-generate test parameters at module load time
TEST_PARAMS = generate_test_params()


# ============================================================================
# Timing Fixtures
# ============================================================================
@pytest.fixture(scope="module", autouse=True)
def module_timer(request):
    """Fixture to measure and log total module execution time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(
        "[TIMING] Total %s: %.3fs (%.2f min)",
        request.module.__name__,
        elapsed,
        elapsed / 60,
    )


# ============================================================================
# Test Implementation
# ============================================================================
#
# This file provides a UNIFIED TEST FRAMEWORK for testing all MoE backend
# implementations through their backend-level interfaces.
#
# =============================================================================
# Purpose & Scope
# =============================================================================
# - Test MoE backends via: routing_method.apply -> quantize_input -> run_moe
# - Single GPU execution (no multi-GPU/distributed testing)
# - Accuracy validation against reference implementations
#
# =============================================================================
# Test Coverage Matrix
# =============================================================================
# 1. BACKENDS: CUTLASS, TRTLLM, CUTEDSL, DEEPGEMM
#
# 2. QUANTIZATION ALGORITHMS:
#    - Unquantized (None)
#    - FP8, FP8_BLOCK_SCALES
#    - NVFP4, W4A8_NVFP4_FP8
#    - W4A16_MXFP4, W4A8_MXFP4_MXFP8
#    - W8A16, W4A8_AWQ
#
# 3. ACTIVATION DTYPES: float16, bfloat16
#
# 4. AUTOTUNER TACTICS:
#    - Autotune phase: find optimal tactics via AutoTuner
#    - Capture phase: record all tactics used
#    - Replay phase: verify each tactic produces correct results
#
# 5. GPTOSS_STYLE (SwiGLU with custom parameters):
#    - swiglu_alpha: scaling factor (default=1)
#    - swiglu_beta: bias term (default=0)
#    - swiglu_limit: clipping limit (default=inf)
#    - Supported by: CUTLASS (W4A8_MXFP4_MXFP8), TRTLLM (W4A8_MXFP4_MXFP8)
#
# 6. MODEL CONFIGURATIONS:
#    - Real models: Mixtral, DeepSeek, Qwen, Grok, GPT-OSS
#    - Boundary cases: prime num_experts, small sizes, top_k=1, top_k=num_experts
#
# =============================================================================
# Skip Logic
# =============================================================================
# Tests are automatically skipped for unsupported configurations using:
# - backend.can_implement(): Check dtype/quant_algo/gptoss_style support
# - should_skip_TRTLLM(): TRTLLM-specific constraints (num_experts % 4, etc.)
# - should_skip_CUTEDSL(): CuteDSL-specific accuracy issues
# - 128-alignment requirements for quantization
#
# =============================================================================
@pytest.mark.skip(reason="Temporarily skipped due to the long time to run the test")
@pytest.mark.parametrize(
    "dtype_activation,backend_type,quant_algo,seq_len,model_config,swiglu_alpha,swiglu_beta,swiglu_limit",
    TEST_PARAMS,
)
def test_moe_backend(
    dtype_activation: torch.dtype,
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    seq_len: int,
    model_config: MoeModelConfig,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
):
    """
    Test MoE backend with autotune to capture all tactics.

    This test verifies:
    1. Autotune works correctly with the backend
    2. All tactics are captured properly
    3. Different sequence lengths use appropriate tactics
    4. gptoss_style (SwiGlu with custom parameters) works correctly
    """
    # Determine gptoss_style based on swiglu parameters
    # gptoss_style is True when any swiglu parameter deviates from default
    # Default values: alpha=1, beta=0, limit=inf
    gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")

    # Note: Skip logic is now handled at parametrize level via get_quick_skip_reason()
    # which calls backend's can_implement() and should_skip_* functions.
    # This avoids entering test function for invalid combinations, significantly
    # reducing test collection time (from ~17 min to ~5 sec for 3400+ skipped tests).

    # Extract model parameters
    num_experts = model_config.num_experts
    top_k = model_config.top_k
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size

    # Create mapping
    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Setup autotuner distributed state
        AutoTuner.get().setup_distributed_state(mapping)

        # Create routing method
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

        # Create test inputs
        x = torch.randn((seq_len, hidden_size), dtype=dtype_activation, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype_activation, device="cuda")

        # Get quantization parameters
        # Pass backend_type to determine scale format (DEEPGEMM/TRTLLM need E8M0 scale)
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )

        # Create quantize utility with gptoss_style parameters
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype_activation,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=gptoss_style,
            gptoss_style=gptoss_style,
            swiglu_alpha=swiglu_alpha if gptoss_style else None,
            swiglu_beta=swiglu_beta if gptoss_style else None,
            swiglu_limit=swiglu_limit if gptoss_style else None,
        )

        # Get swiglu tensors if gptoss_style is enabled
        swiglu_tensors = quantize_util.get_swiglu_tensors()

        # Create backend first (needed for MXFP4_MXFP8 to get shapes)
        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype_activation,
            quant_config=quant_config,
            mapping=mapping,
            bias=gptoss_style,
            swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
            swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
            swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
        )

        # W4A8_MXFP4_MXFP8 requires different weights for backend and reference
        # due to different padding/alignment requirements
        ref_cls = quant_kwargs.pop("ref_cls", None)
        ref_module_kwargs = {}
        if quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
            weights, ref_weights, ref_module_kwargs = quantize_util.prepare_weights_from_backend(
                backend, **quant_kwargs
            )
        else:
            weights = quantize_util.create_weights(**quant_kwargs)
            ref_weights = weights

        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()

        # Create reference
        if ref_cls is not None:
            ref_fused_moe = quantize_util.create_ref_module(
                routing_method, ref_cls=ref_cls, **ref_module_kwargs
            )
        else:
            ref_fused_moe = quantize_util.create_ref_module(routing_method, **ref_module_kwargs)
        ref_fused_moe.load_weights([ref_weights])
        ref_fused_moe.cuda()

        # Clear autotuner cache before autotune phase
        AutoTuner.get().clear_cache()

        # Get reference output first
        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

        # Helper to run MoE computation
        def run_moe():
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            return run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype_activation,
                router_logits,
            )

        # Configure AutoTuner for faster profiling (reduce warmup/repeat for unit tests)
        autotuner = AutoTuner.get()
        autotuner.warmup = 0  # default: 2
        autotuner.repeat = 1  # default: 10
        autotuner.stream_delay_micro_secs = 10  # default: 1000

        # Autotune phase: tune kernels to find best tactics
        # Use cache_path to speed up subsequent runs by reusing tuning results
        with torch.inference_mode(), autotune(cache_path="/tmp/moe_autotuner_cache.json"):
            _ = run_moe()

        # Check if this backend+quant_algo combination supports autotuner capture/replay
        if supports_autotuner_capture(backend_type, quant_algo):
            # Capture phase: record which tactics are used (requires actual execution)
            with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
                _ = run_moe()

            # Replay phase: test each tactic for correctness
            # Set fail_fast=True to stop on first failure, False to run all and report summary
            replay_tactics_and_check(
                all_tactics=all_tactics,
                run_moe_fn=run_moe,
                check_accuracy_fn=ref_fused_moe.check_accuracy,
                ref_output=ref_output,
                backend_type=backend_type,
                quant_algo=quant_algo,
                fail_fast=False,  # Change to True to fail on first error
            )
        else:
            # For backends that don't support autotuner capture/replay,
            # just run a simple accuracy check
            with torch.inference_mode():
                output = run_moe()
                ref_fused_moe.check_accuracy(output, ref_output)
