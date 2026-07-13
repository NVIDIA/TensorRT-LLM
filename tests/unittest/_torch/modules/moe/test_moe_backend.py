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
import os
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from _torch.modules.moe.moe_test_utils import (
    IS_CI_MODE,
    MoeBackendType,
    MoeModelConfig,
    create_test_param,
    get_backend_class,
    iter_base_test_configs,
    replay_tactics_and_check,
    should_skip_to_accelerate_ci,
    skip_if_insufficient_gpu_memory,
    supports_autotuner_capture,
)
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
)
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.modules.fused_moe.interface import MoE, MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoECuteDsl, MegaMoEDeepGemm
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    FusedMoEMethodBase,
    NVFP4MarlinFusedMoEMethod,
    UnquantizedFusedMoEMethod,
    W4A8MXFP4MXFP8MegaMoEDeepGemmMethod,
)
from tensorrt_llm._torch.utils import ActivationType, is_gated_activation
from tensorrt_llm._utils import get_sm_version, mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

logger = logging.getLogger(__name__)

_MEGAMOE_BACKEND_TYPES = {
    MoeBackendType.MEGAMOE_DEEPGEMM,
    MoeBackendType.MEGAMOE_CUTEDSL,
}


def _ensure_single_proc_dist_for_megamoe(backend_type: MoeBackendType, rank: int) -> None:
    """Every MegaMoE backend (DG + CuteDSL) resolves an EP ProcessGroup
    at construction time via ``_resolve_ep_pg``. Single-process tests
    must therefore initialise ``torch.distributed`` even when the test
    only exercises ``ep_size == 1`` -- otherwise the constructor raises
    ``MegaMoe*Unavailable``. Both MegaMoE backends need the same fixture
    so the dist helper must accept the full set."""
    if backend_type not in _MEGAMOE_BACKEND_TYPES:
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MegaMoE tests")
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", str(rank))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)


def should_skip_gptoss(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    swiglu_gptoss_style: bool,
) -> Optional[str]:
    """
    Check if swiglu_gptoss_style test should be skipped for this backend.

    Only CUTLASS and TRTLLM backends support swiglu_gptoss_style (SwiGlu with custom
    alpha/beta/limit parameters and bias).

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm
        swiglu_gptoss_style: Whether swiglu_gptoss_style is enabled

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if not swiglu_gptoss_style:
        return None

    # Only CUTLASS and TRTLLM backends support swiglu_gptoss_style
    supported_backends = {MoeBackendType.CUTLASS, MoeBackendType.TRTLLM}
    if backend_type not in supported_backends:
        return (
            f"swiglu_gptoss_style is only supported by CUTLASS and TRTLLM backends "
            f"(got backend_type={backend_type.value})"
        )

    return None


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
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    activation_type: ActivationType = ActivationType.Swiglu,
    n_shared_experts: int = 0,
) -> MoE:
    """Create a MoE backend for testing."""
    backend_cls = get_backend_class(backend_type)

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype
    if n_shared_experts > 0:
        # TRTLLMGenFusedMoE reads n_shared_experts off the pretrained config to
        # decide num_fused_shared_expert (shared-expert fusion, opt-in via
        # TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1).
        pretrained_config.n_shared_experts = n_shared_experts

    # CUTE_DSL_B12X is internal-only: the user-facing API selects it on the
    # CUTEDSL path when SM120/121 + NVFP4 + flashinfer is importable. Route
    # through "CUTEDSL" so the test exercises the same code path users hit.
    moe_backend_value = (
        "CUTEDSL" if backend_type == MoeBackendType.CUTE_DSL_B12X else backend_type.value
    )
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=mapping,
        moe_backend=moe_backend_value,
    )
    if n_shared_experts > 0:
        # The shared-expert-fusion gate runs after the eager create_weights()
        # in __init__, so weight creation must be deferred (as the real model
        # engine does) for the fused trailing slots to be allocated.
        model_config.skip_create_weights_in_init = True

    backend = create_moe_backend(
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
        weight_loading_mode=weight_loading_mode,
        activation_type=activation_type,
    )
    if n_shared_experts > 0:
        backend.create_weights()
    return backend


# ============================================================================
# Staged post-load hook lifecycle tests
# ============================================================================
# These tests cover staged hook contracts rather than the common backend matrix
# below. Keep them grouped so they can move to a dedicated file if the MoE test
# layout is split later.
def test_moe_post_load_weights_uses_idempotent_transform_hook():
    class HookTestMoE(MoE):
        def create_weights(self):
            raise NotImplementedError

        def load_weights(self, weights, allow_partial_loading=False):
            raise NotImplementedError

        def quantize_input(self, x, **kwargs):
            return x, None

        def run_moe(self, **kwargs):
            raise NotImplementedError

    moe = HookTestMoE.__new__(HookTestMoE)
    torch.nn.Module.__init__(moe)
    quant_method = SimpleNamespace(
        transform_weights=MagicMock(),
        cache_derived_state=MagicMock(),
    )
    moe.quant_method = quant_method

    moe.post_load_weights()
    moe.transform_weights()

    quant_method.transform_weights.assert_called_once_with(moe)
    quant_method.cache_derived_state.assert_called_once_with(moe)
    assert moe._weights_transformed is True

    moe.cache_derived_state()
    assert quant_method.cache_derived_state.call_count == 2

    moe._weights_transformed = False
    moe.transform_weights()
    assert quant_method.transform_weights.call_count == 2


def test_fused_moe_load_weights_invalidates_transform_guard():
    class GuardResetMethod(UnquantizedFusedMoEMethod):
        def load_expert_weights_to_dst(
            self,
            module,
            weights,
            weight_loading_mode,
            load_expert_ids,
            dst_w3_w1_weight,
            dst_w2_weight,
            dst_w3_w1_bias,
            dst_w2_bias,
            allow_partial_loading=False,
        ):
            module.loaded_allow_partial = allow_partial_loading

        def load_quant_scales(self, module, weights):
            module.loaded_scales = bool(weights)

        def setup_quant_scales(self, module):
            module.quant_scales = ()

    method = GuardResetMethod()
    module = SimpleNamespace(
        initial_local_expert_ids=[0],
        w3_w1_weight=torch.empty(1, 2, 2),
        w2_weight=torch.empty(1, 2, 2),
        bias=False,
        _weights_transformed=True,
    )

    method.load_weights(
        module,
        {"0.w1.weight": torch.ones(1)},
        MoEWeightLoadingMode.VANILLA,
        allow_partial_loading=True,
    )

    assert module.loaded_allow_partial is True
    assert module.loaded_scales is True
    assert module._weights_transformed is False


def test_configurable_moe_post_load_weights_uses_backend_staged_hooks():
    from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE

    class HookTestConfigurableMoE(ConfigurableMoE):
        def quantize_input(self, x, **kwargs):
            return x, None

        def run_moe(self, **kwargs):
            raise NotImplementedError

    configurable_moe = HookTestConfigurableMoE.__new__(HookTestConfigurableMoE)
    torch.nn.Module.__init__(configurable_moe)
    backend = torch.nn.Module()
    backend.transform_weights = MagicMock()
    backend.cache_derived_state = MagicMock()
    configurable_moe.backend = backend

    configurable_moe.post_load_weights()
    configurable_moe.transform_weights()

    backend.transform_weights.assert_called_once_with()
    backend.cache_derived_state.assert_called_once_with()
    assert configurable_moe._weights_transformed is True

    configurable_moe.cache_derived_state()
    assert backend.cache_derived_state.call_count == 2


def test_configurable_moe_load_weights_invalidates_wrapper_transform_guard():
    from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE

    configurable_moe = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(configurable_moe)
    backend = torch.nn.Module()
    backend.load_weights = MagicMock(return_value="loaded")
    configurable_moe.backend = backend
    configurable_moe._weights_transformed = True

    weights = [{"0.w1.weight": torch.ones(1)}]
    result = configurable_moe.load_weights(weights, allow_partial_loading=True)

    assert result == "loaded"
    backend.load_weights.assert_called_once_with(weights, True)
    assert configurable_moe._weights_transformed is False


def test_marlin_moe_repack_is_transform_stage():
    assert "transform_weights" in NVFP4MarlinFusedMoEMethod.__dict__
    assert "post_load_weights" not in NVFP4MarlinFusedMoEMethod.__dict__
    assert NVFP4MarlinFusedMoEMethod.post_load_weights is FusedMoEMethodBase.post_load_weights


def test_megamoe_cutedsl_post_load_weights_uses_staged_hooks():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    quant_method = SimpleNamespace(
        transform_weights=MagicMock(),
        cache_derived_state=MagicMock(),
    )
    moe.quant_method = quant_method

    moe.post_load_weights()
    moe.transform_weights()

    quant_method.transform_weights.assert_called_once_with(moe)
    quant_method.cache_derived_state.assert_called_once_with(moe)
    assert moe._weights_transformed is True


def test_megamoe_load_weights_invalidates_cached_deepgemm_views():
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()
    hidden_size = 128
    intermediate_size = 128
    module = SimpleNamespace(
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        initial_local_expert_ids=[0],
        w3_w1_weight=torch.empty(1, intermediate_size * 2, hidden_size // 2, dtype=torch.uint8),
        w3_w1_weight_scale=torch.empty(
            1, intermediate_size * 2, hidden_size // 32, dtype=torch.uint8
        ),
        w2_weight=torch.empty(1, hidden_size, intermediate_size // 2, dtype=torch.uint8),
        w2_weight_scale=torch.empty(1, hidden_size, intermediate_size // 32, dtype=torch.uint8),
        _t_l1=(torch.empty(1), torch.empty(1)),
        _t_l2=(torch.empty(1), torch.empty(1)),
        _t_l1_weight=torch.empty(1),
        _t_l1_scale=torch.empty(1),
        _t_l1_scale_slot=torch.empty(1),
        _t_l2_weight=torch.empty(1),
        _t_l2_scale=torch.empty(1),
        _t_l2_scale_slot=torch.empty(1),
    )
    weights = {
        "0.w1.weight": torch.full((intermediate_size, hidden_size // 2), 1, dtype=torch.uint8),
        "0.w3.weight": torch.full((intermediate_size, hidden_size // 2), 2, dtype=torch.uint8),
        "0.w2.weight": torch.full((hidden_size, intermediate_size // 2), 3, dtype=torch.uint8),
        "0.w1.weight_scale": torch.full(
            (intermediate_size, hidden_size // 32), 4, dtype=torch.uint8
        ),
        "0.w3.weight_scale": torch.full(
            (intermediate_size, hidden_size // 32), 5, dtype=torch.uint8
        ),
        "0.w2.weight_scale": torch.full(
            (hidden_size, intermediate_size // 32), 6, dtype=torch.uint8
        ),
    }

    method.load_weights(module, [weights])

    assert module.w3_w1_weight[0, 0, 0].item() == 1
    assert module.w3_w1_weight[0, intermediate_size, 0].item() == 2
    assert module._weights_loaded is True
    for attr in (
        "_t_l1",
        "_t_l2",
        "_t_l1_weight",
        "_t_l1_scale",
        "_t_l1_scale_slot",
        "_t_l2_weight",
        "_t_l2_scale",
        "_t_l2_scale_slot",
    ):
        assert getattr(module, attr) is None


def test_megamoe_cache_derived_state_sets_initial_assignments_once():
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()
    method.setup_quant_scales = MagicMock()
    load_balancer = MagicMock()
    module = SimpleNamespace(
        layer_load_balancer=load_balancer,
        initial_global_assignments=[0],
    )

    method.cache_derived_state(module)

    load_balancer.set_initial_weight_assignments.assert_called_once_with([0])
    method.setup_quant_scales.assert_called_once_with(module)


def test_megamoe_deepgemm_cache_derived_state_allocates_symm_buffer():
    moe = MegaMoEDeepGemm.__new__(MegaMoEDeepGemm)
    torch.nn.Module.__init__(moe)
    quant_method = SimpleNamespace(cache_derived_state=MagicMock())
    moe.quant_method = quant_method
    moe._alloc_symm_buffer = MagicMock()

    moe.cache_derived_state()

    moe._alloc_symm_buffer.assert_called_once_with()
    quant_method.cache_derived_state.assert_called_once_with(moe)


def test_megamoe_init_rejects_uneven_num_slots_with_value_error():
    routing_method = RenormalizeMoeRoutingMethod(top_k=1)
    model_config = ModelConfig(
        mapping=Mapping(
            world_size=4,
            rank=0,
            tp_size=4,
            moe_tp_size=1,
            moe_ep_size=4,
            enable_attention_dp=True,
        ),
        moe_backend=MoeBackendType.MEGAMOE_DEEPGEMM.value,
    )

    with pytest.raises(
        ValueError,
        match=r"MegaMoEDeepGemm requires num_slots \(10\) divisible by ep_size \(4\)",
    ):
        MegaMoEDeepGemm(
            routing_method=routing_method,
            num_experts=10,
            hidden_size=512,
            intermediate_size=512,
            dtype=torch.bfloat16,
            model_config=model_config,
            init_load_balancer=False,
        )


def test_megamoe_post_load_rejects_uneven_num_slots_with_value_error(monkeypatch):
    import tensorrt_llm._torch.modules.fused_moe.quantization as quantization_module

    class DummyModule:
        _weights_loaded = True
        num_slots = 10
        ep_size = 4

    monkeypatch.setattr(quantization_module, "_import_deep_gemm", lambda: object())
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()

    with pytest.raises(
        ValueError,
        match=r"MegaMoEDeepGemm requires num_slots \(10\) divisible by ep_size \(4\)",
    ):
        method.post_load_weights(DummyModule())


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
    - MegaMoE backends: token_selected_experts=int64, output_dtype

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
    elif backend_type in _MEGAMOE_BACKEND_TYPES:
        args["token_selected_experts"] = token_selected_experts.to(torch.int64)
        args["output_dtype"] = dtype

    return backend.run_moe(**args)


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
    QuantAlgo.W4A8_MXFP4_FP8,
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.MXFP8,
    QuantAlgo.W8A16,
    QuantAlgo.W4A8_AWQ,
]

# Backend types to test
BACKEND_TYPES_TO_TEST = [
    MoeBackendType.CUTLASS,
    MoeBackendType.TRTLLM,
    MoeBackendType.CUTEDSL,
    MoeBackendType.DEEPGEMM,
    MoeBackendType.DENSEGEMM,
    MoeBackendType.MEGAMOE_DEEPGEMM,
    MoeBackendType.MEGAMOE_CUTEDSL,
    MoeBackendType.CUTE_DSL_B12X,
    MoeBackendType.MARLIN,
]

# Data types to test
DTYPES_TO_TEST = [
    torch.float16,
    torch.bfloat16,
]

# Format: (num_experts, top_k, hidden_size, intermediate_size)
#
# Default runs the CI subset (TRTLLM_TEST_MOE_CI=1).
# Set TRTLLM_TEST_MOE_CI=0 for the full local config matrix.
CI_MOE_MODEL_CONFIGS = [
    # Real models (small/medium — tactic replay is model-size-independent,
    # e256 is covered by test_moe_module integration tests)
    MoeModelConfig(60, 4, 2048, 1408),  # Qwen1.5-MoE-A2.7B
    MoeModelConfig(128, 4, 2880, 2880),  # GPT-OSS-120B
    MoeModelConfig(8, 1, 512, 512),  # boundary: top_k=1, single expert activated
    # Boundary tests for tactic correctness
    MoeModelConfig(4, 4, 512, 512),  # top_k=num_experts, all experts activated
    MoeModelConfig(7, 2, 256, 512),  # prime num_experts
    MoeModelConfig(13, 3, 256, 512),  # prime num_experts, odd top_k
]

LOCAL_MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS + [
    MoeModelConfig(256, 8, 7168, 2048),  # DeepSeek-V3
    MoeModelConfig(256, 6, 4096, 2048),  # DeepSeek-V4-Flash
    MoeModelConfig(8, 2, 4096, 14336),  # Mixtral-8x7B
    MoeModelConfig(64, 6, 2048, 1408),  # DeepSeek-MoE-16B / DeepSeek-V2-Lite
    MoeModelConfig(8, 2, 6144, 32768),  # Grok-1
    # === Boundary Tests: small sizes ===
    MoeModelConfig(4, 2, 64, 128),  # very small hidden_size
    MoeModelConfig(4, 2, 128, 64),  # intermediate < hidden
]

MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS if IS_CI_MODE else LOCAL_MOE_MODEL_CONFIGS

# Sequence lengths to test
SEQ_LENS_TO_TEST = [1, 8]

# SwiGLU parameters for swiglu_gptoss_style testing
SWIGLU_ALPHAS = [1, 1.702]  # default, GPT-OSS (modeling_gpt_oss.py)
SWIGLU_BETAS = [0, 1.0]  # default, GPT-OSS
SWIGLU_LIMITS = [float("inf"), 7.0]  # default, GPT-OSS

# Full product of all SwiGLU combos (local exhaustive testing only)
LOCAL_SWIGLU_COMBOS = list(itertools.product(SWIGLU_ALPHAS, SWIGLU_BETAS, SWIGLU_LIMITS))

# CI: only non-gptoss (default) and one gptoss combo
# All non-default combos trigger the same swiglu_gptoss_style=True code path;
# different alpha/beta/limit values are just kernel parameters, not code branches.
CI_SWIGLU_COMBOS = [
    (1, 0, float("inf")),  # non-gptoss (default SwiGLU)
    (1.702, 1.0, 7.0),  # gptoss style (GPT-OSS real values)
]

SWIGLU_COMBOS = CI_SWIGLU_COMBOS if IS_CI_MODE else LOCAL_SWIGLU_COMBOS


def generate_test_params() -> List:
    """
    Generate test parameter combinations, filtering out unsupported configurations.

    Unsupported combinations (those with a skip_reason from get_quick_skip_reason)
    are excluded entirely so they never appear in pytest collection output.

    Returns:
        List of pytest.param objects for runnable test configurations only
    """
    params: List = []
    for (
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
        test_id,
    ) in iter_base_test_configs(
        SWIGLU_COMBOS,
        MOE_MODEL_CONFIGS,
        SEQ_LENS_TO_TEST,
        DTYPES_TO_TEST,
        BACKEND_TYPES_TO_TEST,
        QUANT_ALGOS_TO_TEST,
    ):
        if skip_reason:
            continue
        param_values = (
            dtype,
            backend_type,
            quant_algo,
            seq_len,
            model_config,
            routing_method_cls,
            ActivationType.Swiglu,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
        )
        params.append(create_test_param(param_values, test_id))

    return params


# Pre-generate test parameters at module load time
TEST_PARAMS = generate_test_params()


def generate_element_wise_test_params() -> List:
    params: List = []
    for activation_type in [ActivationType.Silu, ActivationType.Relu2]:
        for (
            _,  # swiglu_alpha  (ignored)
            _,  # swiglu_beta   (ignored)
            _,  # swiglu_limit  (ignored)
            model_config,
            seq_len,
            dtype,
            backend_type,
            quant_algo,
            routing_method_cls,
            skip_reason,
            base_test_id,
        ) in iter_base_test_configs(
            [(1, 0, float("inf"))],  # swiglu parameters are irrelevant
            MOE_MODEL_CONFIGS,
            SEQ_LENS_TO_TEST,
            DTYPES_TO_TEST,
            [MoeBackendType.CUTLASS, MoeBackendType.TRTLLM],
            [None, QuantAlgo.NVFP4],
        ):
            if skip_reason:
                continue
            if backend_type == MoeBackendType.CUTLASS and activation_type == ActivationType.Silu:
                continue
            if backend_type == MoeBackendType.TRTLLM and quant_algo is None:
                continue
            test_id = f"act={activation_type.name}-{base_test_id}"
            param_values = (
                dtype,
                backend_type,
                quant_algo,
                seq_len,
                model_config,
                routing_method_cls,
                activation_type,
                None,
                None,
                None,
            )
            params.append(create_test_param(param_values, test_id))
    return params


TEST_PARAMS += generate_element_wise_test_params()


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
#    - When using element wise activations (Relu2, Silu), only CUTLASS and TRTLLM
#      are supported
#
# 2. QUANTIZATION ALGORITHMS:
#    - When using Swiglu:
#      - Unquantized (None)
#      - FP8, FP8_BLOCK_SCALES
#      - NVFP4, W4A8_NVFP4_FP8
#      - W4A16_MXFP4, W4A8_MXFP4_MXFP8
#      - W8A16, W4A8_AWQ
#    - When using element-wise activations
#      - Unquantized (CUTLASS)
#      - NVFP4 (TRTLLM, CUTLASS)
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
# - backend.can_implement(): Check dtype/quant_algo/swiglu_gptoss_style support
# - should_skip_trtllm(): TRTLLM-specific constraints (num_experts % 4, etc.)
# - should_skip_cutedsl(): CuteDSL-specific accuracy issues
# - 128-alignment requirements for quantization
#
# =============================================================================
@pytest.mark.parametrize(
    "dtype_activation,backend_type,quant_algo,seq_len,model_config,"
    "routing_method_cls,activation_type,swiglu_alpha,swiglu_beta,swiglu_limit",
    TEST_PARAMS,
)
def test_moe_backend(
    dtype_activation: torch.dtype,
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    seq_len: int,
    model_config: MoeModelConfig,
    routing_method_cls,
    activation_type: ActivationType,
    swiglu_alpha: Optional[float],
    swiglu_beta: Optional[float],
    swiglu_limit: Optional[float],
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test MoE backend with autotune to capture all tactics.

    This test verifies:
    1. Autotune works correctly with the backend
    2. All tactics are captured properly
    3. Different sequence lengths use appropriate tactics
    4. swiglu_gptoss_style (SwiGlu with custom parameters) works correctly
    """
    # DENSEGEMM: disable fused fc2_alpha path for backend-level testing.
    if backend_type == MoeBackendType.DENSEGEMM:
        monkeypatch.setenv("TRTLLM_MOE_FUSED_FC2_ALPHA", "0")

    # MEGAMOE_CUTEDSL threads per-expert fc31_alpha / fc2_alpha /
    # fc1_norm_const through the kernel ABI, so NVFP4QuantizeUtil's non-1
    # weight_scale_2 values compute correctly without a test bypass.

    is_gated = is_gated_activation(activation_type)
    swiglu_gptoss_style = False
    if is_gated:
        # Determine swiglu_gptoss_style based on swiglu parameters
        # swiglu_gptoss_style is True when any swiglu parameter deviates from default
        # Default values: alpha=1, beta=0, limit=inf
        swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")

    ci_skip = should_skip_to_accelerate_ci(
        backend_type=backend_type,
        quant_algo=quant_algo,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
        dtype=dtype_activation,
        seq_len=seq_len,
        swiglu_gptoss_style=swiglu_gptoss_style,
        activation_type=activation_type,
    )
    if ci_skip:
        pytest.skip(ci_skip)

    # Extract model parameters
    num_experts = model_config.num_experts
    top_k = model_config.top_k
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size

    skip_if_insufficient_gpu_memory(num_experts, hidden_size, intermediate_size, dtype_activation)

    # Create mapping
    mapping = Mapping()
    mapping.rank = mpi_rank()
    _ensure_single_proc_dist_for_megamoe(backend_type, mapping.rank)

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Setup autotuner distributed state
        AutoTuner.get().setup_distributed_state(mapping)

        # Create routing method from parametrized class
        routing_method = routing_method_cls(top_k=top_k)

        # Create test inputs
        x = torch.randn((seq_len, hidden_size), dtype=dtype_activation, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype_activation, device="cuda")

        # Get quantization parameters
        # Pass backend_type to determine scale format (DEEPGEMM/TRTLLM need E8M0 scale)
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )

        # Create quantize utility with swiglu_gptoss_style parameters
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype_activation,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=swiglu_gptoss_style,
            swiglu_gptoss_style=swiglu_gptoss_style,
            swiglu_alpha=swiglu_alpha if swiglu_gptoss_style else None,
            swiglu_beta=swiglu_beta if swiglu_gptoss_style else None,
            swiglu_limit=swiglu_limit if swiglu_gptoss_style else None,
            activation_type=activation_type,
        )

        # Get swiglu tensors if swiglu_gptoss_style is enabled
        swiglu_tensors = quantize_util.get_swiglu_tensors()

        # Determine weight loading mode based on quantization algorithm
        weight_loading_mode = MoEWeightLoadingMode.VANILLA
        if hasattr(quantize_util, "weight_loading_mode"):
            weight_loading_mode = quantize_util.weight_loading_mode

        # Clear class-level permute indices cache between parametrized test cases
        # to work around a B200-specific kernel bug (tactic [32,5] illegal memory access)
        from tensorrt_llm._torch.modules.fused_moe.quantization import (
            NVFP4TRTLLMGenFusedMoEBaseMethod,
        )

        NVFP4TRTLLMGenFusedMoEBaseMethod._cache_permute_indices.clear()

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
            bias=swiglu_gptoss_style,
            swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
            swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
            swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
            weight_loading_mode=weight_loading_mode,
            activation_type=activation_type,
        )

        # W4A8_MXFP4_MXFP8 / W4A8_MXFP4_FP8 require backend-layout-aware
        # weights. CUTLASS and MegaMoE use 128 hidden alignment; TRTLLMGen
        # pads FC1 input to 512. MXFP4FP8QuantizeUtil inherits
        # prepare_weights_from_backend from MXFP4MXFP8QuantizeUtil so the
        # backend-vs-reference weight split applies to both variants.
        ref_cls = quant_kwargs.pop("ref_cls", None)
        ref_module_kwargs = {}
        if quant_algo in (QuantAlgo.W4A8_MXFP4_MXFP8, QuantAlgo.W4A8_MXFP4_FP8):
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

        # flashinfer has no capture and replay mechanisms, so we skip test_all_kernels
        use_flashinfer = getattr(backend, "use_flashinfer", False)

        # Check if this backend+quant_algo combination supports autotuner capture/replay
        if supports_autotuner_capture(backend_type, quant_algo, use_flashinfer):
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


# ============================================================================
# BF16 (unquantized) TRTLLM-Gen MoE: DeepSeekV3 / Renormalize routing
# ============================================================================
# The main test_moe_backend skips TRTLLM + quant_algo=None, so cover the BF16
# FlashInfer path here (Nemotron-H enablement): DeepSeekV3/Renormalize routing
# x Relu2/Swiglu, via both fused and separated routing.

# DeepSeekV3 trtllm-gen routing requires num_experts >= 22, multiple of 4.
_BF16_UNQUANT_NUM_EXPERTS = 72
_BF16_UNQUANT_TOP_K = 6
_BF16_UNQUANT_HIDDEN = 1024
_BF16_UNQUANT_INTERMEDIATE = 512


def _make_bf16_routing_method(routing_kind: str, top_k: int, num_experts: int, device: str):
    if routing_kind == "renormalize":
        return RenormalizeMoeRoutingMethod(top_k=top_k)
    # DeepSeekV3 (noaux_tc): sigmoid scores + correction bias, single group.
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    return DeepSeekV3MoeRoutingMethod(
        top_k=top_k,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        callable_e_score_correction_bias=lambda: bias,
    )


@pytest.mark.parametrize(
    "trtllm_use_router_logits", [True, False], ids=["fused_routing", "separated_routing"]
)
@pytest.mark.parametrize("seq_len", [8, 256])
@pytest.mark.parametrize(
    "activation_type", [ActivationType.Relu2, ActivationType.Swiglu], ids=["relu2", "swiglu"]
)
@pytest.mark.parametrize("routing_kind", ["deepseekv3", "renormalize"])
def test_trtllm_bf16_unquantized_moe(
    routing_kind, activation_type, seq_len, trtllm_use_router_logits
):
    """TRTLLM-Gen BF16 (unquantized) MoE accuracy vs the reference impl."""
    backend_type = MoeBackendType.TRTLLM
    dtype = torch.bfloat16

    can_impl, skip_reason = get_backend_class(backend_type).can_implement(
        None, dtype_activation=dtype
    )
    if not can_impl:
        pytest.skip(skip_reason)

    num_experts = _BF16_UNQUANT_NUM_EXPERTS
    top_k = _BF16_UNQUANT_TOP_K
    hidden_size = _BF16_UNQUANT_HIDDEN
    intermediate_size = _BF16_UNQUANT_INTERMEDIATE

    skip_if_insufficient_gpu_memory(num_experts, hidden_size, intermediate_size, dtype)

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)

        routing_method = _make_bf16_routing_method(routing_kind, top_k, num_experts, "cuda")

        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")

        # Unquantized path: get_test_quant_params returns BaseQuantizeUtil.
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(None, x, backend_type)
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            activation_type=activation_type,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            activation_type=activation_type,
        )
        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()

        ref_fused_moe = quantize_util.create_ref_module(routing_method)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

        AutoTuner.get().clear_cache()

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
                dtype,
                router_logits=router_logits,
                trtllm_use_router_logits=trtllm_use_router_logits,
            )

        # Autotune, then verify accuracy against the reference.
        with torch.inference_mode(), autotune(cache_path="/tmp/moe_autotuner_cache.json"):
            _ = run_moe()
        with torch.inference_mode():
            output = run_moe()
            ref_fused_moe.check_accuracy(output, ref_output)


# ============================================================================
# TRTLLM-Gen shared-expert fusion (migrated from deprecated
# tests/unittest/_torch/thop/serial/test_moe.py::TestMoeFP8 fusion coverage)
# ============================================================================
# TRTLLMGenFusedMoE can fold n_shared_experts into the routed grouped GEMM as
# always-selected experts (opt-in via TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1;
# requires FP8_BLOCK_SCALES + dp_size==1 + DeepSeekV3 routing). The fused
# experts occupy trailing weight slots [num_experts, num_experts+n_fused) and
# receive routing weight 1.0, applied after routed_scaling_factor.

# (num_experts, n_group, topk_group, top_k, n_fused) — shapes taken from the
# deprecated thop test's expert_info parametrization (n_fused > 0 variants).
FUSED_SHARED_EXPERT_INFOS = [
    (32, 8, 4, 8, 1),
    (256, 8, 4, 8, 2),
    (72, 1, 1, 6, 1),
    (72, 1, 1, 6, 2),
]


class _AppendSharedExpertsRouting:
    """Reference-side routing wrapper: appends the fused shared experts as
    always-selected entries (ids [num_experts, num_experts+n_fused), weight
    1.0 after routed scaling) to the wrapped routing method's output."""

    def __init__(self, routing_method, num_experts: int, n_fused: int):
        self._routing_method = routing_method
        self._num_experts = num_experts
        self._n_fused = n_fused

    def apply(self, router_logits):
        ids, weights = self._routing_method.apply(router_logits)
        num_tokens = ids.shape[0]
        shared_ids = torch.arange(
            self._num_experts,
            self._num_experts + self._n_fused,
            dtype=ids.dtype,
            device=ids.device,
        ).expand(num_tokens, -1)
        shared_weights = torch.ones(
            (num_tokens, self._n_fused), dtype=weights.dtype, device=weights.device
        )
        return torch.cat([ids, shared_ids], dim=1), torch.cat([weights, shared_weights], dim=1)


def _write_fused_shared_expert_slots(backend, weights, num_experts: int, n_fused: int):
    """Copy the shared experts' quantized tensors into the trailing weight
    slots, mirroring DeepSeekFP8BlockScalesFusedMoEMethod's routed-slot layout
    (w3_w1 slot = [w3; w1] along dim 0; scales likewise)."""
    for i in range(n_fused):
        expert_id = num_experts + i
        slot = num_experts + i
        dst_w3, dst_w1 = backend.w3_w1_weight.data[slot].chunk(2, dim=0)
        dst_w3.copy_(weights[f"{expert_id}.w3.weight"].view(dst_w3.dtype))
        dst_w1.copy_(weights[f"{expert_id}.w1.weight"].view(dst_w1.dtype))
        backend.w2_weight.data[slot].copy_(
            weights[f"{expert_id}.w2.weight"].view(backend.w2_weight.dtype)
        )
        dst_w3_scale, dst_w1_scale = backend.w3_w1_weight_scaling_factor.data[slot].chunk(2, dim=0)
        dst_w3_scale.copy_(weights[f"{expert_id}.w3.weight_scale"])
        dst_w1_scale.copy_(weights[f"{expert_id}.w1.weight_scale"])
        backend.w2_weight_scaling_factor.data[slot].copy_(weights[f"{expert_id}.w2.weight_scale"])


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="TRTLLM-Gen FP8 block scales requires SM100/103",
)
@pytest.mark.parametrize("seq_len", [16, 1024])
@pytest.mark.parametrize(
    "expert_info",
    FUSED_SHARED_EXPERT_INFOS,
    ids=lambda info: f"e{info[0]}g{info[1]}tg{info[2]}k{info[3]}fused{info[4]}",
)
def test_trtllm_fp8_block_scales_fused_shared_experts(
    expert_info,
    seq_len: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Fused-shared-expert accuracy for the TRTLLM backend (FP8 block scales).

    Loads routed experts through the regular path, writes the shared experts
    into the trailing fused slots, and checks the fused kernel against a
    reference that treats the shared experts as always-selected with weight
    1.0. Replays every captured tactic; also asserts the small-tile WAR
    (fused path restricted to tileN >= 32) holds in the autotuner candidates.
    """
    num_experts, n_group, topk_group, top_k, n_fused = expert_info
    hidden_size = 512
    intermediate_size = 512
    dtype = torch.bfloat16
    backend_type = MoeBackendType.TRTLLM

    monkeypatch.setenv("TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION", "1")

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)

        e_score_correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device="cuda")
        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=2.5,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )

        router_logits = torch.randn((seq_len, num_experts), dtype=torch.float32, device="cuda")

        placeholder_x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.FP8_BLOCK_SCALES, placeholder_x, backend_type
        )
        # The quantize util covers routed + fused shared experts so create_weights
        # emits per-expert tensors for all of them and the reference module gets
        # a GatedMLP per fused slot as well.
        quantize_util = quantize_util_cls(
            num_experts=num_experts + n_fused,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
        )
        weights = quantize_util.create_weights(**quant_kwargs)
        x = quantize_util.create_input(seq_len)

        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            n_shared_experts=n_fused,
        )
        assert backend.num_fused_shared_expert == n_fused, (
            "shared-expert fusion gate did not activate"
        )
        assert backend.w3_w1_weight.shape[0] == num_experts + n_fused

        # Mirror the real flow (modeling_deepseekv3): load routed experts,
        # post-process, then fill the trailing fused slots.
        routed_weights = {k: v for k, v in weights.items() if int(k.split(".")[0]) < num_experts}
        backend.load_weights([routed_weights])
        backend.post_load_weights()
        backend.cuda()
        _write_fused_shared_expert_slots(backend, weights, num_experts, n_fused)

        ref_routing = _AppendSharedExpertsRouting(routing_method, num_experts, n_fused)
        ref_fused_moe = quantize_util.create_ref_module(ref_routing)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        AutoTuner.get().clear_cache()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

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
                dtype,
                router_logits=router_logits,
            )

        autotuner = AutoTuner.get()
        autotuner.warmup = 0
        autotuner.repeat = 1
        autotuner.stream_delay_micro_secs = 10

        with (
            torch.inference_mode(),
            autotune(cache_path="/tmp/moe_autotuner_cache_fused_shared.json"),
        ):
            _ = run_moe()

        with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
            _ = run_moe()

        # WAR regression check: with num_fused_shared_experts > 0 the C++ side
        # (fp8BlockScaleMoe.cpp fusedMinTileN, default 32) must exclude the
        # small-tile (tileN 8/16) dynB cubins from the candidate tactics.
        moe_tile_ns = []
        for combo in all_tactics:
            for _, tactic in combo:
                if isinstance(tactic, (list, tuple)) and len(tactic) == 2:
                    moe_tile_ns.append(int(tactic[0]))
        assert moe_tile_ns, "expected [tileN, config] tactics from the fused MoE runner"
        assert all(tile_n >= 32 for tile_n in moe_tile_ns), (
            f"fused path produced small-tile tactics (tileN < 32): {sorted(set(moe_tile_ns))}"
        )

        replay_tactics_and_check(
            all_tactics=all_tactics,
            run_moe_fn=run_moe,
            check_accuracy_fn=ref_fused_moe.check_accuracy,
            ref_output=ref_output,
            backend_type=backend_type,
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        )


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="TRTLLM-Gen FP8 block scales requires SM100/103",
)
def test_trtllm_fp8_block_scales_fuse_shared_expert_layout(monkeypatch: pytest.MonkeyPatch):
    """Layout-only check of fuse_shared_expert (no kernel launch).

    Builds a shared GatedMLP of intermediate n_fused*I, fuses it, and asserts
    each trailing slot holds the expected per-expert sub-tensors by
    definition: slot i's [w3; w1] = gate_up rows [i*I, (i+1)*I) of the up/gate
    halves, slot i's w2 = down_proj columns [i*I, (i+1)*I) — including the
    block-scale tensors (the w2 scale split is a historically bug-prone spot).
    """
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    num_experts = 8
    top_k = 2
    hidden_size = 256
    intermediate_size = 256
    n_fused = 2
    dtype = torch.bfloat16
    scale_block = 128

    monkeypatch.setenv("TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION", "1")

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)

        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=top_k,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            callable_e_score_correction_bias=lambda: torch.zeros(
                num_experts, dtype=dtype, device="cuda"
            ),
        )
        placeholder_x = torch.randn((1, hidden_size), dtype=dtype, device="cuda")
        _, quant_config, _ = get_test_quant_params(
            QuantAlgo.FP8_BLOCK_SCALES, placeholder_x, MoeBackendType.TRTLLM
        )

        backend = create_test_backend(
            backend_type=MoeBackendType.TRTLLM,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            n_shared_experts=n_fused,
        )
        assert backend.num_fused_shared_expert == n_fused

        shared_mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=n_fused * intermediate_size,
            bias=False,
            dtype=dtype,
            config=ModelConfig(quant_config=quant_config),
        )
        shared_mlp.cuda()
        # Fill weights/scales with distinct random data (fp8 weights are filled
        # via a bf16 sample viewed as fp8 bytes; values only need to be unique).
        gate_up = shared_mlp.gate_up_proj
        down = shared_mlp.down_proj
        gate_up.weight.data.copy_(
            torch.randn(gate_up.weight.shape, dtype=torch.bfloat16, device="cuda")
            .to(torch.float8_e4m3fn)
            .view(gate_up.weight.dtype)
        )
        down.weight.data.copy_(
            torch.randn(down.weight.shape, dtype=torch.bfloat16, device="cuda")
            .to(torch.float8_e4m3fn)
            .view(down.weight.dtype)
        )
        gate_up.weight_scale.data.copy_(
            torch.rand(gate_up.weight_scale.shape, dtype=torch.float32, device="cuda")
        )
        down.weight_scale.data.copy_(
            torch.rand(down.weight_scale.shape, dtype=torch.float32, device="cuda")
        )

        backend.fuse_shared_expert(shared_mlp)
        torch.cuda.synchronize()

        # gate_up_proj rows are [gate(w1); up(w3)]; per-expert i owns rows
        # [i*I, (i+1)*I) within each half. down_proj columns [i*I, (i+1)*I).
        w1_all, w3_all = gate_up.weight.data.chunk(2, dim=0)
        w1_scale_all, w3_scale_all = gate_up.weight_scale.data.chunk(2, dim=0)
        inter_blocks = intermediate_size // scale_block
        for i in range(n_fused):
            slot = num_experts + i
            rows = slice(i * intermediate_size, (i + 1) * intermediate_size)
            scale_rows = slice(i * inter_blocks, (i + 1) * inter_blocks)

            dst_w3, dst_w1 = backend.w3_w1_weight.data[slot].chunk(2, dim=0)
            torch.testing.assert_close(dst_w3, w3_all[rows].view(dst_w3.dtype))
            torch.testing.assert_close(dst_w1, w1_all[rows].view(dst_w1.dtype))
            torch.testing.assert_close(
                backend.w2_weight.data[slot],
                down.weight.data[:, rows].view(backend.w2_weight.dtype),
            )

            dst_w3_scale, dst_w1_scale = backend.w3_w1_weight_scaling_factor.data[slot].chunk(
                2, dim=0
            )
            torch.testing.assert_close(dst_w3_scale, w3_scale_all[scale_rows])
            torch.testing.assert_close(dst_w1_scale, w1_scale_all[scale_rows])
            torch.testing.assert_close(
                backend.w2_weight_scaling_factor.data[slot],
                down.weight_scale.data[:, scale_rows],
            )
