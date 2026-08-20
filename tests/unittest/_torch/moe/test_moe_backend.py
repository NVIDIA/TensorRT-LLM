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
"""MoE backend unit tests."""

import importlib
import itertools
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from _torch.moe.moe_test_utils import (
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
from _torch.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, autotune
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import _select_explicit_fallback_tactic
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
)
from tensorrt_llm._torch.moe.fused_moe.activation import (
    DEFAULT_MOE_ACTIVATION,
    SimpleActivation,
    SiTuActivation,
    SwigluActivation,
    SwigluBiasActivation,
    materialize_activation_params,
)
from tensorrt_llm._torch.moe.fused_moe.communication.deep_ep_low_latency import DeepEPLowLatency
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import (
    CuteDslFusedMoE,
    CuteDslFusedMoENvfp4Runner,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x import CuteDslB12xFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_marlin import MarlinFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoECommPlan,
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejection,
    MoERejectReason,
    MoEResolutionReport,
    MoERunContext,
    MoEStaticCapability,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import (
    collect_moe_environment,
    override_moe_environment,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoE, MoESchedulerKind, MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl, MegaMoEDeepGemm
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import (
    build_moe_deployment,
    impl_class_for,
    resolve_moe_impl,
)
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    FusedMoEMethodBase,
    NVFP4FusedMoEMethod,
    NVFP4MarlinFusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEBaseMethod,
    NVFP4TRTLLMGenFusedMoEMethod,
    UnquantizedFusedMoEMethod,
    W4A8MXFP4MXFP8MegaMoEDeepGemmMethod,
    W4A16NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.utils import ActivationType, MxFp8QuantizedTensor, is_gated_activation
from tensorrt_llm._utils import get_sm_version, is_sm_100f, mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

logger = logging.getLogger(__name__)

_MEGAMOE_BACKEND_TYPES = {
    MoeBackendType.MEGAMOE_DEEPGEMM,
    MoeBackendType.MEGAMOE_CUTEDSL,
}


def test_import_deep_gemm_rejects_pre_situ_mega_moe_api(monkeypatch):
    import tensorrt_llm
    import tensorrt_llm._torch.moe.fused_moe.quantization as quantization_module

    def fp8_fp4_mega_moe():
        pass

    def per_token_cast_to_fp8(*, use_packed_ue8m0=False):
        pass

    deep_gemm = SimpleNamespace(
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        get_symm_buffer_for_mega_moe=lambda: None,
        transform_sf_into_required_layout=lambda: None,
        transform_weights_for_mega_moe=lambda: None,
        per_token_cast_to_fp8=per_token_cast_to_fp8,
    )
    monkeypatch.setattr(tensorrt_llm, "deep_gemm", deep_gemm)

    with pytest.raises(
        quantization_module._MegaMoEUnavailable,
        match="fp8_fp4_mega_moe does not accept.*situ_beta",
    ):
        quantization_module._import_deep_gemm()


def test_fp8_block_scale_moe_fallback_tactic_is_explicit_and_deterministic():
    valid_tactics = [
        [8, 0],
        [32, 4],
        [16, 0],
        [32, 0],
    ]

    assert _select_explicit_fallback_tactic(valid_tactics) == [32, 0]

    with pytest.raises(RuntimeError, match="no valid fallback tactic"):
        _select_explicit_fallback_tactic([])


def test_cutedsl_count_native_runner_keeps_both_tactics_and_threads_counts():
    forward_impl = MagicMock(return_value=torch.empty(1))
    runner = CuteDslFusedMoENvfp4Runner(
        forward_impl=forward_impl,
        num_experts=256,
        top_k=1,
        num_local_experts=8,
        local_expert_offset=0,
        use_direct_expert_metadata=True,
        use_count_native_expert_metadata=True,
        deep_ep_expert_capacity=32,
    )
    counts = torch.tensor([0, 1, 31, 32, 7, 0, 16, 2], dtype=torch.int32)
    inputs = [torch.empty(1) for _ in range(6)] + [counts]

    runner.forward(inputs, tactic=256)

    assert runner.unique_id()[-3:] == (
        "direct_expert_metadata",
        32,
        "count_native_expert_metadata",
    )
    assert runner.get_valid_tactics([], OptimizationProfile()) == [128, 256]
    assert runner.get_tuning_config().inputs_pre_hook is None
    assert forward_impl.call_args.kwargs["recv_expert_count"] is counts
    assert forward_impl.call_args.kwargs["use_count_native_expert_metadata"] is True


def test_cutedsl_direct_metadata_tuning_is_static_and_legacy_remains_dynamic():
    common_kwargs = {
        "forward_impl": MagicMock(),
        "num_experts": 256,
        "top_k": 1,
        "num_local_experts": 3,
        "local_expert_offset": 13,
    }
    direct_runner = CuteDslFusedMoENvfp4Runner(
        **common_kwargs,
        use_direct_expert_metadata=True,
        deep_ep_expert_capacity=5,
    )

    direct_config = direct_runner.get_tuning_config()
    assert direct_config.dynamic_tensor_specs == ()
    assert direct_config.constraint_specs == ()
    assert direct_config.inputs_pre_hook is None

    legacy_runner = CuteDslFusedMoENvfp4Runner(**common_kwargs)
    legacy_config = legacy_runner.get_tuning_config()
    assert len(legacy_config.dynamic_tensor_specs) == 1
    assert legacy_config.inputs_pre_hook is not None


@pytest.mark.parametrize(
    "disable_value,expected_disabled",
    [
        (None, False),
        ("0", False),
        ("1", True),
    ],
)
def test_cutedsl_deep_ep_direct_metadata_disable_env(
    monkeypatch, disable_value: Optional[str], expected_disabled: bool
):
    env_name = "TRTLLM_DISABLE_CUTEDSL_DEEP_EP_DIRECT_METADATA"
    if disable_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, disable_value)

    def mock_base_init(self, **kwargs):
        self.aux_stream_dict = {}
        self.event_dict = {}

    monkeypatch.setattr(CutlassFusedMoE, "__init__", mock_base_init)
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock(return_value=object()))
    monkeypatch.setattr(torch.cuda, "Event", MagicMock(return_value=object()))

    constructor_kwargs = {
        "routing_method": MagicMock(),
        "num_experts": 8,
        "hidden_size": 64,
        "intermediate_size": 128,
    }

    backend = CuteDslFusedMoE(**constructor_kwargs)
    assert backend.disable_deep_ep_direct_metadata is expected_disabled


def test_deep_ep_adapter_free_output_matches_schema_and_reuses_cache(monkeypatch):
    is_capturing = MagicMock(side_effect=AssertionError("CPU path queried CUDA capture state"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", is_capturing)
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    comm.mapping = SimpleNamespace(moe_ep_rank=1)
    comm.num_slots = 4
    comm._adapter_free_placeholder_cache = {}
    comm.deep_ep_buffer = object()

    hidden_states = torch.arange(24, dtype=torch.bfloat16).view(2, 3, 4)
    hidden_states_sf = torch.arange(12, dtype=torch.uint8).view(2, 3, 2)
    recv_expert_count = torch.tensor([2, 1], dtype=torch.int32)

    legacy = comm._modify_output_to_adapt_fused_moe(
        hidden_states,
        hidden_states_sf,
        recv_expert_count,
        torch.float32,
    )
    adapter_free = comm._modify_output_to_adapt_fused_moe(
        hidden_states,
        hidden_states_sf,
        recv_expert_count,
        torch.float32,
        remove_adapter=True,
    )
    adapter_free_reused = comm._modify_output_to_adapt_fused_moe(
        hidden_states,
        hidden_states_sf,
        recv_expert_count,
        torch.float32,
        remove_adapter=True,
    )

    torch.testing.assert_close(adapter_free[0], legacy[0])
    torch.testing.assert_close(adapter_free[1], legacy[1])
    assert adapter_free[2].shape == legacy[2].shape
    assert adapter_free[2].dtype == legacy[2].dtype
    torch.testing.assert_close(adapter_free[3], legacy[3])
    assert adapter_free_reused[2] is adapter_free[2]
    assert adapter_free_reused[3] is adapter_free[3]
    is_capturing.assert_not_called()

    comm.destroy()
    assert comm._adapter_free_placeholder_cache == {}
    assert comm.deep_ep_buffer is None


def test_deep_ep_adapter_free_cache_miss_rejected_during_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", MagicMock(return_value=True))
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    comm._adapter_free_placeholder_cache = {}
    hidden_states = MagicMock(
        shape=(2, 3, 4),
        device=torch.device("cuda:0"),
        is_cuda=True,
    )

    with pytest.raises(RuntimeError, match="must be warmed up before CUDA graph capture"):
        comm._modify_output_to_adapt_fused_moe(
            hidden_states,
            None,
            torch.tensor([2, 1], dtype=torch.int32),
            torch.float32,
            remove_adapter=True,
        )
    assert comm._adapter_free_placeholder_cache == {}


@pytest.mark.parametrize(
    "disabled,has_nvfp4,supports_post_quant,expected",
    [
        (False, True, True, True),
        (True, True, True, False),
        (False, False, True, False),
        (False, True, False, False),
    ],
)
def test_scheduler_selects_cutedsl_deep_ep_direct_metadata(
    disabled: bool, has_nvfp4: bool, supports_post_quant: bool, expected: bool
):
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    backend.disable_deep_ep_direct_metadata = disabled
    backend.use_fused_finalize = True
    backend._weights_created = True
    backend.quant_config = SimpleNamespace(
        layer_quant_mode=SimpleNamespace(
            has_nvfp4=MagicMock(return_value=has_nvfp4),
        ),
    )
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=backend,
        comm=comm,
        routing_method=SimpleNamespace(experts_per_token=1),
    )

    assert scheduler._use_cutedsl_deep_ep_direct_metadata(supports_post_quant) is expected


def test_scheduler_rejects_direct_metadata_for_other_backend_or_communication():
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=CuteDslFusedMoE.__new__(CuteDslFusedMoE),
        comm=object(),
    )
    assert scheduler._use_cutedsl_deep_ep_direct_metadata(True) is False

    scheduler.moe = SimpleNamespace(
        backend=CutlassFusedMoE.__new__(CutlassFusedMoE),
        comm=DeepEPLowLatency.__new__(DeepEPLowLatency),
    )
    assert scheduler._use_cutedsl_deep_ep_direct_metadata(True) is False


@pytest.mark.parametrize(
    "use_fused_finalize,experts_per_token",
    [(False, 1), (True, 2)],
)
def test_scheduler_rejects_direct_metadata_without_finalize_or_top1(
    use_fused_finalize: bool, experts_per_token: int
):
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    backend.disable_deep_ep_direct_metadata = False
    backend.use_fused_finalize = use_fused_finalize
    backend._weights_created = True
    backend.quant_config = SimpleNamespace(
        layer_quant_mode=SimpleNamespace(
            has_nvfp4=MagicMock(return_value=True),
        ),
    )
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=backend,
        comm=comm,
        routing_method=SimpleNamespace(experts_per_token=experts_per_token),
    )

    assert scheduler._use_cutedsl_deep_ep_direct_metadata(True) is False


@pytest.mark.parametrize("disabled,expected", [(False, True), (True, False)])
def test_scheduler_threads_deep_ep_expert_metadata_to_cutedsl(disabled: bool, expected: bool):
    recv_expert_count = torch.tensor([3, 0, 5], dtype=torch.int32)
    expert_capacity = 8
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    comm._dispatch_state = {
        "recv_expert_count": recv_expert_count,
        "expert_capacity": expert_capacity,
    }
    comm.enable_postquant_alltoall = False
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    backend.disable_deep_ep_direct_metadata = disabled
    backend.use_fused_finalize = True
    backend._weights_created = True
    backend.quant_config = SimpleNamespace(
        layer_quant_mode=SimpleNamespace(
            has_nvfp4=MagicMock(return_value=True),
        ),
    )
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=backend,
        comm=comm,
        enable_alltoall=False,
        routing_method=SimpleNamespace(experts_per_token=1),
    )

    use_direct_metadata = scheduler._use_cutedsl_deep_ep_direct_metadata(True)
    assert use_direct_metadata is expected

    plan = scheduler._build_comm_plan(
        all_rank_num_tokens=None,
        output_dtype=torch.bfloat16,
        use_deep_ep_direct_metadata=use_direct_metadata,
    )

    assert plan.recv_expert_count is recv_expert_count
    assert plan.deep_ep_expert_capacity == expert_capacity
    assert plan.use_deep_ep_direct_metadata is expected


def _ensure_single_proc_dist_for_megamoe(backend_type: MoeBackendType, rank: int) -> None:
    """Initialize the process group required by MegaMoE constructors."""
    if backend_type not in _MEGAMOE_BACKEND_TYPES:
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MegaMoE tests")
    if dist.is_initialized():
        return
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        store=dist.HashStore(),
        rank=0,
        world_size=1,
    )


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


def test_kimi_fused_route_quant_skips_prequantized_input(monkeypatch) -> None:
    """An upstream fused down projection owns quantization on this path."""
    monkeypatch.delenv("TLLM_K3_DISABLE_FUSED_ROUTE_QUANT", raising=False)
    monkeypatch.setattr(
        "tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen.get_sm_version",
        MagicMock(side_effect=AssertionError("SM probe must be short-circuited")),
    )
    backend = TRTLLMGenFusedMoE.__new__(TRTLLMGenFusedMoE)
    hidden_states = MxFp8QuantizedTensor(
        fp8_tensor=torch.empty(1, 3584, dtype=torch.float8_e4m3fn),
        scaling_factor=torch.empty(1, 112, dtype=torch.uint8),
    )

    assert (
        backend.try_fused_route_quant(hidden_states, torch.empty(1, 896, dtype=torch.float32))
        is None
    )


def test_kimi_mxfp8_quantized_tensor_handoff() -> None:
    """The NVFP4 communication path preserves an upstream MXFP8 payload."""
    fp8_tensor = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
    scaling_factor = torch.empty(2, 2, dtype=torch.uint8)
    hidden_states = MxFp8QuantizedTensor(fp8_tensor, scaling_factor)
    backend = TRTLLMGenFusedMoE.__new__(TRTLLMGenFusedMoE)
    backend._weights_created = True
    quant_mode = MagicMock()
    quant_mode.has_any_quant.return_value = True
    quant_mode.has_w4a8_mxfp4_fp8.return_value = False
    quant_mode.has_nvfp4.return_value = True
    backend.quant_config = SimpleNamespace(layer_quant_mode=quant_mode)

    quantized, scales = backend.quantize_input(hidden_states)

    assert quantized is fp8_tensor
    assert scales.data_ptr() == scaling_factor.data_ptr()
    assert torch.equal(scales, scaling_factor)


def build_test_activation(
    activation_type: ActivationType,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
) -> "SimpleActivation | SwigluActivation | SwigluBiasActivation | SiTuActivation":
    """Package the flat parameters these tests parametrize over as one activation.

    The tests still sweep alpha / beta / limit independently because that is
    what ``quantize_util.get_swiglu_tensors`` produces for the reference
    implementation. Presence of alpha or beta means the gpt-oss package, which
    is the same rule the C++ op applied when it upgraded a bare ``Swiglu`` with
    constants to ``SwigluBias``.
    """
    kind = ActivationType(activation_type)
    if kind is ActivationType.SiTu:
        return SiTuActivation(gate_softcap=swiglu_alpha, linear_softcap=swiglu_beta)
    if swiglu_alpha is not None or swiglu_beta is not None:
        return SwigluBiasActivation(
            gate_sigmoid_scale=swiglu_alpha,
            linear_offset=swiglu_beta,
            clamp=swiglu_limit,
        )
    if kind in (ActivationType.Swiglu, ActivationType.SwigluBias):
        return SwigluActivation(clamp=swiglu_limit)
    return SimpleActivation(kind=kind)


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
    locality_domain_policy: Optional[LocalityDomainPolicy] = None,
    n_shared_experts: int = 0,
) -> MoE:
    """Create a MoE backend for testing."""
    backend_cls = get_backend_class(backend_type)
    if locality_domain_policy is None:
        locality_domain_policy = LocalityDomainPolicy(enabled=False)

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
        locality_domain_policy=locality_domain_policy,
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
        weight_loading_mode=weight_loading_mode,
        activation=build_test_activation(activation_type, swiglu_alpha, swiglu_beta, swiglu_limit),
    )
    if n_shared_experts > 0:
        backend.create_weights()
    return backend


# =====================================================================
# Staged post-load hook lifecycle tests
# =====================================================================
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

        def run_moe(self, ctx, *, workspace=None):
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
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    class HookTestConfigurableMoE(ConfigurableMoE):
        def quantize_input(self, x, **kwargs):
            return x, None

        def run_moe(self, ctx, *, workspace=None):
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
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

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


def test_moe_nvfp4_activation_quantization_capability():
    assert NVFP4FusedMoEMethod.quantizes_nvfp4_activations
    assert not W4A16NVFP4CutlassFusedMoEMethod.quantizes_nvfp4_activations


def test_marlin_moe_repack_is_transform_stage():
    assert "transform_weights" in NVFP4MarlinFusedMoEMethod.__dict__
    assert "post_load_weights" not in NVFP4MarlinFusedMoEMethod.__dict__
    assert NVFP4MarlinFusedMoEMethod.post_load_weights is FusedMoEMethodBase.post_load_weights


def _marlin_model_config(quant_algo=QuantAlgo.NVFP4):
    cfg = ModelConfig()
    cfg.moe_backend = "MARLIN"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


def _marlin_environment(sm: int = 90) -> MoEEnvironment:
    """Marlin's own SM window, so quantization stays the only variable."""
    return MoEEnvironment(sm=sm)


def test_marlin_is_selected_for_nvfp4():
    with override_moe_environment(_marlin_environment()):
        report = resolve_moe_impl(_marlin_model_config())
    assert impl_class_for(report) is MarlinFusedMoE
    assert report.selected_by == "pinned"
    assert not report.degraded


@pytest.mark.parametrize(
    "quant_algo",
    [
        pytest.param(None, id="unquantized"),
        pytest.param(QuantAlgo.FP8, id="fp8"),
    ],
)
def test_marlin_degrades_to_cutlass_on_non_nvfp4(quant_algo):
    with override_moe_environment(_marlin_environment()):
        report = resolve_moe_impl(_marlin_model_config(quant_algo))
    assert impl_class_for(report) is CutlassFusedMoE
    assert report.degraded
    assert report.degraded_from.reason is MoERejectReason.QUANT_UNSUPPORTED


def test_marlin_override_quant_config_degrades_per_layer():
    cfg = _marlin_model_config()
    with override_moe_environment(_marlin_environment()):
        report = resolve_moe_impl(
            cfg,
            override_quant_config=QuantConfig(quant_algo=None),
            layer_idx=52,
        )
    assert impl_class_for(report) is CutlassFusedMoE
    assert report.degraded_from.reason is MoERejectReason.QUANT_UNSUPPORTED


# =====================================================================
# TRTLLM-Gen SiTu backend contract
# =====================================================================
# SiTu rides the generic SwiGLU geometry, so the host-side wiring is easy to
# get wrong in ways no shape check catches: it reaches the cubin through the
# same ``gemm1_alpha`` / ``gemm1_beta`` slots SwiGLU's constants use, and only
# the activation kind separates them. Kernel-level coverage -- tactic
# availability, launch evidence and SiTu-vs-SwiGLU numerics -- lives in
# test_kimi_k3_situ_moe.py (and thop/serial/test_moe.py for the runner). These
# are the contracts that hold without running a cubin.

_situ_supported = pytest.mark.skipif(
    not is_sm_100f(),
    reason="TRTLLM-Gen SiTu cubins are sm_100f: the SM100 family only",
)

# Kimi K3 defaults. The gate-side SiTu beta is the cubin's ``alpha``, the
# linear-side beta its ``beta``. See modeling_kimi_linear.py.
_SITU_GATE_ALPHA = 4.0
_SITU_LINEAR_BETA = 25.0


def _make_trtllm_gen_moe(
    *,
    quant_config: Optional[QuantConfig],
    situ: bool,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 512,
    intermediate_size: int = 256,
) -> TRTLLMGenFusedMoE:
    """A single-rank TRTLLM-Gen MoE, with or without the SiTu override."""
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        moe_backend="TRTLLM",
    )
    return TRTLLMGenFusedMoE(
        routing_method=RenormalizeMoeRoutingMethod(top_k=top_k),
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        # The soft-caps go out as scalars; the backend's declared
        # PER_EXPERT_TENSOR shape is what broadcasts them per slot.
        activation=(
            SiTuActivation(
                gate_softcap=_SITU_GATE_ALPHA,
                linear_softcap=_SITU_LINEAR_BETA,
            )
            if situ
            else DEFAULT_MOE_ACTIVATION
        ),
    )


def _make_loaded_nvfp4_trtllm_gen_moe(situ: bool) -> TRTLLMGenFusedMoE:
    """The same MoE with NVFP4 weights and quant scales actually loaded.

    The scale contracts below are written by ``load_quant_scales``, so they
    only exist after a real load; the stock quantize utils supply the weights.
    """
    num_experts, hidden_size, intermediate_size = 8, 512, 256
    torch.manual_seed(0)
    x = torch.randn((8, hidden_size), dtype=torch.bfloat16, device="cuda") * 0.5
    util_cls, quant_config, quant_kwargs = get_test_quant_params(QuantAlgo.NVFP4, x, "TRTLLM")
    quant_kwargs.pop("ref_cls", None)
    quantize_util = util_cls(
        num_experts=num_experts,
        dtype=torch.bfloat16,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        quant_config=quant_config,
        bias=False,
        activation_type=ActivationType.Swiglu,
    )
    backend = _make_trtllm_gen_moe(
        quant_config=quant_config,
        situ=situ,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    backend.load_weights([quantize_util.create_weights(**quant_kwargs)])
    backend.post_load_weights()
    return backend.cuda()


@_situ_supported
def test_trtllm_gen_nvfp4_situ_selects_padded_quant_method() -> None:
    """``_get_quant_method`` must key off ``is_situ_activation``.

    SiTu fills the act_alpha/act_beta slots from ``create_weights``, i.e.
    after ``_get_quant_method`` has already run, so keying off
    ``act_alpha is not None`` would make the selected method depend on
    *when* it is resolved. Plain SwiGLU is the control: it still gets the
    unpadded base method, so this is not asserting a constant.
    """
    situ = _make_loaded_nvfp4_trtllm_gen_moe(situ=True)
    assert situ.is_situ_activation
    assert isinstance(situ.quant_method, NVFP4TRTLLMGenFusedMoEMethod)
    assert situ.scaling_vector_size == 16

    plain = _make_loaded_nvfp4_trtllm_gen_moe(situ=False)
    assert not plain.is_situ_activation
    assert type(plain.quant_method) is NVFP4TRTLLMGenFusedMoEBaseMethod


@_situ_supported
def test_trtllm_gen_nvfp4_situ_fc31_scale_c_drops_dequant_scale() -> None:
    """SiTuGlu is nonlinear in x0, so scaleC must be quantScaleC alone.

    Pins the trtllm-gen host rule (``!isLinearInX0(mActType) && mFusedAct`` in
    ``BatchedGemm/BatchedGemmTestUtils.h``) at the level where TRT-LLM
    computes it, without needing to run a kernel. SwiGLU is the control: it
    *is* linear in x0 and keeps the combined ``quantScaleC * dequantScaleAb``.

    Getting this wrong is not a small numeric error. The extra dequantScaleAb
    factor (~6e-8) drives the FC1 output's per-block E4M3 scale factors below
    their smallest subnormal, so the NVFP4 intermediate quantizes to exactly
    zero and the whole MoE returns zeros -- an output that still passes a
    ``rtol=0.1, atol=0.15`` tolerance band, because these MoE outputs have a
    standard deviation (~0.04) well below ``atol``.
    """
    situ = _make_loaded_nvfp4_trtllm_gen_moe(situ=True)
    swiglu = _make_loaded_nvfp4_trtllm_gen_moe(situ=False)

    # fc2_input_scale is quantScaleC; fc31_alpha is dequantScaleAb (the
    # scaleGate the kernel already folds into the tanh/sigmoid arguments).
    torch.testing.assert_close(
        situ.fc31_scale_c.data.float(),
        situ.fc2_input_scale.data.float().expand_as(situ.fc31_scale_c.data),
        msg="SiTu fc31_scale_c must not carry the dequantScaleAb factor",
    )
    torch.testing.assert_close(
        swiglu.fc31_scale_c.data.float(),
        (swiglu.fc2_input_scale.data * swiglu.fc31_alpha.data).float(),
        msg="SwiGLU fc31_scale_c must keep the combined scale",
    )
    # The two conventions must actually differ, otherwise this test would pass
    # vacuously on a build where dequantScaleAb happens to be 1.
    assert not torch.allclose(situ.fc31_scale_c.data.float(), swiglu.fc31_scale_c.data.float())


@_situ_supported
@pytest.mark.parametrize(
    "quant_algo",
    [
        pytest.param(None, id="unquantized"),
        pytest.param(QuantAlgo.FP8, id="fp8"),
        pytest.param(QuantAlgo.W4A16_MXFP4, id="w4a16_mxfp4"),
    ],
)
def test_trtllm_gen_situ_rejects_quant_algos_without_fused_cubins(quant_algo) -> None:
    """There is no standalone SiTu activation kernel.

    SiTu exists only as a fused FC1 epilogue, in the NVFP4 (group-16) and
    W4A8_MXFP4_MXFP8 (group-32) cubin families. Anything else has to be
    rejected at construction rather than silently resolving to SwiGLU, which
    is structurally wrong output that no shape check would catch.
    """
    with pytest.raises(ValueError, match="requires one of .* quantization"):
        _make_trtllm_gen_moe(quant_config=QuantConfig(quant_algo=quant_algo), situ=True)


@_situ_supported
@pytest.mark.parametrize(
    "hidden_size,intermediate_size,expected",
    [
        pytest.param(512, 256, (32, 32), id="no_padding"),
        pytest.param(2880, 2880, (256, 256), id="hidden_not_256_aligned"),
        pytest.param(1536, 192, (128, 32), id="intermediate_needs_128"),
    ],
)
def test_nvfp4_trtllm_gen_resolve_alignments_matches_create_weights(
    hidden_size, intermediate_size, expected
) -> None:
    """``resolve_alignments`` must predict what ``create_weights`` selects.

    ``TRTLLMGenFusedMoE`` validates its MoE-TP shard from ``__init__``, i.e.
    before ``create_weights`` has shadowed the class attributes with
    shape-resolved instance ones, so the check has to predict them. If the two
    ever drift apart the shard contract is validated against the wrong number.
    """
    predicted = NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments(hidden_size, intermediate_size)
    assert predicted == expected

    backend = _make_trtllm_gen_moe(
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        situ=True,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    assert (
        backend.quant_method.weight_alignment,
        backend.quant_method.input_hidden_alignment,
    ) == predicted


@_situ_supported
def test_nvfp4_trtllm_gen_class_alignment_would_admit_bad_shards() -> None:
    """Why the MoE-TP check cannot read the class attribute.

    ``hidden=1536``/``intermediate=192`` resolves to a 128 weight alignment.
    192 is divisible by the class default (32) but not by 128, so validating
    against ``NVFP4TRTLLMGenFusedMoEMethod.weight_alignment`` admits a shard
    the loader cannot lay out. Pins the gap, so reverting the check to the
    class attribute fails here rather than in a multi-GPU accuracy run.
    """
    resolved, _ = NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments(1536, 192)
    assert 192 % NVFP4TRTLLMGenFusedMoEMethod.weight_alignment == 0
    assert 192 % resolved != 0


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


def test_megamoe_streaming_reload_resets_slot_claims():
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()
    method._clear_transformed_weight_cache = MagicMock()
    module = SimpleNamespace(
        rebuild_tensor_metadata={},
        _packed_mxfp4_loaded_slots={0},
        expert_size_per_partition=1,
        initial_local_expert_ids=[0],
        w3_w1_weight=torch.empty(1, 2, 1, dtype=torch.uint8),
        w3_w1_weight_scale=torch.empty(1, 2, 1, dtype=torch.uint8),
        w2_weight=torch.empty(1, 1, 1, dtype=torch.uint8),
        w2_weight_scale=torch.empty(1, 1, 1, dtype=torch.uint8),
    )

    method.pre_reload_weights(module)

    assert module._packed_mxfp4_loaded_slots == set()

    weight = torch.ones(1, 1, dtype=torch.uint8)
    method.load_packed_mxfp4_expert(
        module,
        global_expert_id=0,
        local_slot_id=0,
        w1_weight=weight,
        w1_weight_scale=weight,
        w2_weight=weight,
        w2_weight_scale=weight,
        w3_weight=weight,
        w3_weight_scale=weight,
    )

    assert module._packed_mxfp4_loaded_slots == {0}
    method._clear_transformed_weight_cache.assert_called_once_with(module)


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


def test_megamoe_bakes_situ_softcaps_as_uniform_scalars():
    # MegaMoE declares UNIFORM_SCALAR for alpha/beta because the kernels bake
    # them at codegen time, so a per-expert tensor is reduced here.
    params = materialize_activation_params(
        SiTuActivation(gate_softcap=torch.full((8,), 4.0), linear_softcap=25.0),
        MegaMoEDeepGemm.activation_support,
        num_local_experts=8,
        owner="MegaMoEDeepGemm",
    )

    assert params.activation_type is ActivationType.SiTu
    assert params.alpha == 4.0
    assert params.beta == 25.0


def test_megamoe_plain_swiglu_carries_no_constants():
    params = materialize_activation_params(
        SwigluActivation(),
        MegaMoEDeepGemm.activation_support,
        num_local_experts=8,
        owner="MegaMoEDeepGemm",
    )

    assert (params.alpha, params.beta) == (None, None)
    assert params.clamp is None


def test_create_moe_forwards_situ_activation_as_one_carrier(monkeypatch):
    create_moe_module = importlib.import_module("tensorrt_llm._torch.moe.fused_moe.create_moe")
    configurable_moe = MagicMock(return_value=object())
    monkeypatch.setattr(create_moe_module, "ConfigurableMoE", configurable_moe)
    monkeypatch.setattr(
        create_moe_module,
        "resolve_moe_cls",
        MagicMock(return_value=MegaMoEDeepGemm),
    )
    activation = SiTuActivation(gate_softcap=4.0, linear_softcap=25.0)

    result = create_moe_module.create_moe(
        routing_method=MagicMock(),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
        dtype=torch.bfloat16,
        model_config=ModelConfig(),
        activation=activation,
    )

    assert result is configurable_moe.return_value
    assert configurable_moe.call_args.kwargs["activation"] is activation


def test_create_moe_backend_rejects_apply_router_weight_on_input_by_declaration():
    """The gate runs ahead of every ``moe_cls`` branch, so a class the factory
    has never heard of still reaches it -- which is the point of reading a
    declaration instead of matching a class."""

    class _Undeclared:
        capabilities = MoEStaticCapability()

    with pytest.raises(ValueError, match="apply_router_weight_on_input"):
        create_moe_backend(
            moe_cls=_Undeclared,
            routing_method=MagicMock(),
            num_experts=8,
            hidden_size=512,
            intermediate_size=512,
            apply_router_weight_on_input=True,
        )


def test_apply_router_weight_on_input_support_is_not_inherited():
    """``CuteDslB12xFusedMoE`` is the one impl that keeps its ``CutlassFusedMoE``
    parent, and this is a field where the two disagree: only the NVFP4 prefill
    chunk reaches the parent's ``run_moe``, while the decode path hands
    ``token_final_scales`` to the flashinfer wrapper."""
    assert CutlassFusedMoE.capabilities.supports_apply_router_weight_on_input
    assert MarlinFusedMoE.capabilities.supports_apply_router_weight_on_input
    assert not CuteDslB12xFusedMoE.capabilities.supports_apply_router_weight_on_input
    assert not TRTLLMGenFusedMoE.capabilities.supports_apply_router_weight_on_input


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
    import tensorrt_llm._torch.moe.fused_moe.quantization as quantization_module

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


def _make_megamoe_cutedsl_for_ctor_test() -> MegaMoECuteDsl:
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        skip_create_weights_in_init=True,
    )
    return MegaMoECuteDsl(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
        dtype=torch.bfloat16,
        model_config=model_config,
        init_load_balancer=False,
    )


def test_megamoe_cutedsl_tuning_mode_forces_top_maxt_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Profiling scratch is sized for the largest adaptive bucket.
    monkeypatch.setenv("MEGAMOE_TACTIC_AUTOTUNE", "1")
    moe = _make_megamoe_cutedsl_for_ctor_test()
    buckets = moe._maxt_buckets
    assert len(buckets) >= 2, f"expected a multi-bucket ladder, got {buckets}"
    small_hint = buckets[0]
    assert moe._select_launch_max_tokens(small_hint) == buckets[0]
    monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)
    assert moe._select_launch_max_tokens(small_hint) == buckets[-1]


def test_megamoe_cutedsl_tactic_autotune_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Standard serving must not pay for the 36-tactic sweep by default.
    monkeypatch.delenv("MEGAMOE_TACTIC_AUTOTUNE", raising=False)
    moe = _make_megamoe_cutedsl_for_ctor_test()
    assert moe.tactic_autotune is False


def test_enumerate_megamoe_candidate_tactics_curated_space() -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    decode = megamoe_op.enumerate_megamoe_candidate_tactics(1024)
    prefill = megamoe_op.enumerate_megamoe_candidate_tactics(16384)
    assert len(decode) == len(prefill) == 36
    assert {t[-1] for t in decode} == {(1, 1)}
    assert {t[-1] for t in prefill} == {(2, 4)}
    # The deterministic fallback stays inside the curated axes.
    for num_tokens in (64, 4096, 16384):
        megamoe_op.validate_megamoe_tactic(megamoe_op.default_megamoe_tactic(num_tokens))
    invalid_tactic = list(megamoe_op.default_megamoe_tactic(64))
    invalid_tactic[2] = 511
    with pytest.raises(ValueError, match=r"group_hint must be an int >= 512"):
        megamoe_op.validate_megamoe_tactic(tuple(invalid_tactic))


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
    workspace = None

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
        workspace = backend.get_workspace(m_max, 128)
    elif backend_type in _MEGAMOE_BACKEND_TYPES:
        args["token_selected_experts"] = token_selected_experts.to(torch.int64)
        args["output_dtype"] = dtype

    # Mirror what each scheduler hands the backend: ExternalCommMoEScheduler
    # builds a plan on every path, FusedCommMoEScheduler never does because the
    # fused kernel owns the exchange. Single GPU with no comm strategy, so this
    # is the no-comm plan: quantize_input ran locally and left the scale factors
    # swizzled, no alltoall, and no workspace-backed output buffer.
    if backend.scheduler_kind == MoESchedulerKind.EXTERNAL_COMM:
        args["comm_plan"] = MoECommPlan(
            input_sf_swizzled=True,
            enable_alltoall=False,
            moe_output=None,
            payload_in_workspace=False,
        )

    return backend.run_moe(MoERunContext(**args), workspace=workspace)


# =====================================================================
# Test Parameters
# =====================================================================

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
    MoeModelConfig(60, 4, 2048, 1408),  # retained 60-expert backend shape
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


def should_skip_locality_domain_param(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    activation_type: ActivationType,
    swiglu_gptoss_style: bool,
    dtype: torch.dtype,
) -> Optional[str]:
    """Return a static skip reason for locality domain MoE backend params."""
    if backend_type != MoeBackendType.CUTEDSL:
        return "locality domain MoE backend test only supports CuteDSL"
    if quant_algo not in (QuantAlgo.NVFP4, None):
        return "locality domain MoE backend test only supports NVFP4 or BF16"
    # plan_moe only enables the unquantized path for bfloat16 activations.
    if quant_algo is None and dtype != torch.bfloat16:
        return "unquantized locality domain MoE requires bfloat16"
    if activation_type != ActivationType.Swiglu:
        return "locality domain MoE backend test only supports SwiGLU"
    if swiglu_gptoss_style:
        return "locality domain MoE backend test does not cover GPT-OSS SwiGLU style"
    return None


def should_skip_locality_domain_runtime(enable_locality_domains: bool) -> Optional[str]:
    """Return a runtime skip reason for locality domain MoE backend params."""
    if not enable_locality_domains:
        return None
    if not torch.cuda.is_available():
        return "CUDA is not available"
    sm_version = get_sm_version()
    if sm_version != 107:
        return f"Rubin (SM 107) required, got SM {sm_version}"
    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        return "public CuteDSL Rubin kernels are not available"
    is_locality_domain_enabled.cache_clear()
    if not is_locality_domain_enabled():
        return "locality domain is not enabled/supported on this system"
    return None


def test_ci_acceleration_keeps_only_locality_domain_cutedsl_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _torch.moe import moe_test_utils

    monkeypatch.setattr(moe_test_utils, "IS_CI_MODE", True)
    model_config = MoeModelConfig(60, 4, 2048, 1408)
    common_kwargs = {
        "backend_type": MoeBackendType.CUTEDSL,
        "quant_algo": None,
        "model_config": model_config,
        "routing_method_cls": RenormalizeMoeRoutingMethod,
        "activation_type": ActivationType.Swiglu,
    }

    assert should_skip_to_accelerate_ci(dtype=torch.bfloat16, **common_kwargs) is not None
    assert (
        should_skip_to_accelerate_ci(
            dtype=torch.bfloat16,
            enable_locality_domains=True,
            **common_kwargs,
        )
        is None
    )
    assert (
        should_skip_to_accelerate_ci(
            dtype=torch.float16,
            enable_locality_domains=True,
            **common_kwargs,
        )
        is not None
    )


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
            False,
        )
        params.append(create_test_param(param_values, test_id))

        if quant_algo in (QuantAlgo.NVFP4, None):
            swiglu_gptoss_style = (
                swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")
            )
            locality_domain_skip_reason = should_skip_locality_domain_param(
                backend_type,
                quant_algo,
                ActivationType.Swiglu,
                swiglu_gptoss_style,
                dtype,
            )
            locality_domain_param_values = (
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
                True,
            )
            params.append(
                create_test_param(
                    locality_domain_param_values,
                    f"locality_domain=enabled-{test_id}",
                    locality_domain_skip_reason,
                )
            )

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
            [None, QuantAlgo.NVFP4, QuantAlgo.W8A16],
        ):
            if skip_reason:
                continue
            if backend_type == MoeBackendType.CUTLASS and activation_type == ActivationType.Silu:
                continue
            if backend_type == MoeBackendType.TRTLLM and quant_algo is None:
                continue
            # INT8 weight-only per-channel non-gated MoE is CUTLASS-path only.
            if quant_algo == QuantAlgo.W8A16 and backend_type != MoeBackendType.CUTLASS:
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
                False,
            )
            params.append(create_test_param(param_values, test_id))
    return params


TEST_PARAMS += generate_element_wise_test_params()


# =====================================================================
# Test Implementation
# =====================================================================
#
# This file provides a UNIFIED TEST FRAMEWORK for testing all MoE backend
# implementations through their backend-level interfaces.
#
# ======================================================================
# Purpose & Scope
# ======================================================================
# - Test MoE backends via: routing_method.apply -> quantize_input -> run_moe
# - Single GPU execution (no multi-GPU/distributed testing)
# - Accuracy validation against reference implementations
#
# ======================================================================
# Test Coverage Matrix
# ======================================================================
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
#    - Real models: Mixtral, DeepSeek, Grok, GPT-OSS
#    - Boundary cases: prime num_experts, small sizes, top_k=1, top_k=num_experts
#
# ======================================================================
# Skip Logic
# ======================================================================
# Tests are automatically skipped for unsupported configurations using:
# - backend.can_implement(p, d): declared quant / dtype / SM / dependency support
# - should_skip_trtllm(): TRTLLM-specific constraints (num_experts % 4, etc.)
# - should_skip_cutedsl(): CuteDSL-specific accuracy issues
# - 128-alignment requirements for quantization
#
# ======================================================================
@pytest.mark.parametrize(
    "dtype_activation,backend_type,quant_algo,seq_len,model_config,"
    "routing_method_cls,activation_type,swiglu_alpha,swiglu_beta,swiglu_limit,"
    "enable_locality_domains",
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
    enable_locality_domains: bool,
    tmp_path: Path,
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

    locality_domain_runtime_skip = should_skip_locality_domain_runtime(enable_locality_domains)
    if locality_domain_runtime_skip:
        pytest.skip(locality_domain_runtime_skip)
    locality_domain_policy = LocalityDomainPolicy(enabled=enable_locality_domains)

    ci_skip = should_skip_to_accelerate_ci(
        backend_type=backend_type,
        quant_algo=quant_algo,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
        dtype=dtype_activation,
        seq_len=seq_len,
        swiglu_gptoss_style=swiglu_gptoss_style,
        activation_type=activation_type,
        enable_locality_domains=enable_locality_domains,
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
        from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEBaseMethod

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
            locality_domain_policy=locality_domain_policy,
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
        if enable_locality_domains:
            assert backend._locality_domain_runtime is not None
            assert backend._locality_domain_weight_shards is not None

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
        if enable_locality_domains:
            AutoTuner.get().reset_statistics()

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
        cache_path = (
            str(tmp_path / "moe_autotuner_cache.json")
            if enable_locality_domains
            else "/tmp/moe_autotuner_cache.json"
        )
        with torch.inference_mode(), autotune(cache_path=cache_path):
            _ = run_moe()
        if enable_locality_domains:
            quant_name = "nvfp4" if quant_algo == QuantAlgo.NVFP4 else "bf16"
            expected_tuning_ops = (
                f"CuteDslFusedMoE::run_moe_{quant_name}::locality_domain_end_to_end",
                f"trtllm::cute_dsl_{quant_name}_gather_grouped_gemm_"
                f"{'act_fusion' if quant_algo == QuantAlgo.NVFP4 else 'swiglu'}"
                "_rubin::locality_domain_concurrent",
                f"trtllm::cute_dsl_{quant_name}_grouped_gemm_finalize_"
                "inplace_rubin::locality_domain_concurrent",
            )
            for op_name in expected_tuning_ops:
                assert autotuner.stats.tuned_op_profiled_configs.get(op_name, 0) > 0
                assert not autotuner.stats.failed_profiling_count.get(op_name, set())

        # flashinfer has no capture and replay mechanisms, so we skip test_all_kernels
        use_flashinfer = getattr(backend, "use_flashinfer", False)

        # Check if this backend+quant_algo combination supports autotuner capture/replay
        if supports_autotuner_capture(backend_type, quant_algo, use_flashinfer):
            # Capture phase: record which tactics are used (requires actual execution)
            with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
                _ = run_moe()

            # Replaying every outer tile is deliberately exhaustive and would
            # multiply the inner FC tactic replay for every matrix member. One
            # representative production shape per locality domain path covers that
            # outer-tile contract; all matrix members still validate their
            # tuned/failed statistics and replay the tactics selected for their
            # own shape below.
            representative_outer_tile_replay = (
                enable_locality_domains
                and quant_algo in (QuantAlgo.NVFP4, None)
                and seq_len == 1
                and (num_experts, top_k, hidden_size, intermediate_size) == (60, 4, 2048, 1408)
            )
            if representative_outer_tile_replay:
                # The regular Cartesian replay contains inner FC tactics only
                # for the selected outer tile. Exercise every outer tile
                # directly after tuning, when all corresponding FC caches have
                # been prepared, so a non-winning tile cannot silently regress.
                outer_context = all_tactics._captured_contexts[0]
                outer_runner = outer_context["runners"][0]
                outer_tactics = outer_runner.get_valid_tactics(
                    outer_context["inputs"], OptimizationProfile()
                )
                expected_outer_tactics = (
                    {128, 256, 512} if quant_algo == QuantAlgo.NVFP4 else {64, 128, 256}
                )
                assert set(outer_tactics) == expected_outer_tactics
                for outer_tactic in outer_tactics:
                    # Direct runner replay reuses the captured inplace output;
                    # reset it to the fresh-output baseline used by run_moe().
                    with torch.inference_mode():
                        outer_context["inputs"][-1].zero_()
                        output = outer_runner(outer_context["inputs"], tactic=outer_tactic)
                    ref_fused_moe.check_accuracy(output, ref_output)

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


# =====================================================================
# BF16 (unquantized) TRTLLM-Gen MoE: DeepSeekV3 / Renormalize routing
# =====================================================================
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
    routing_kind,
    activation_type,
    seq_len,
    trtllm_use_router_logits,
    num_experts=_BF16_UNQUANT_NUM_EXPERTS,
    top_k=_BF16_UNQUANT_TOP_K,
):
    """TRTLLM-Gen BF16 (unquantized) MoE accuracy vs the reference impl."""
    backend_type = MoeBackendType.TRTLLM
    dtype = torch.bfloat16

    num_experts = _BF16_UNQUANT_NUM_EXPERTS
    top_k = _BF16_UNQUANT_TOP_K
    hidden_size = _BF16_UNQUANT_HIDDEN
    intermediate_size = _BF16_UNQUANT_INTERMEDIATE

    # This test constructs the backend directly, so query it directly.
    verdict = get_backend_class(backend_type).can_implement(
        MoEProblem(
            quant=None,
            dtype_act=dtype,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        ),
        MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=num_experts,
            env=collect_moe_environment(),
        ),
    )
    if not verdict.eligible:
        pytest.skip(verdict.detail)

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
        if trtllm_use_router_logits and backend._routes_outside_the_kernel():
            # This config only routes outside the kernel, so run_moe drops
            # router_logits and the scheduler never pairs the two. Asking for
            # fused routing here tests a combination production cannot reach.
            pytest.skip("routing happens outside the kernel; fused routing is unreachable")

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


@pytest.mark.parametrize("seq_len", [1, 8])
def test_trtllm_bf16_dsv3_routing_kimi_k3_shape(seq_len):
    """Kimi-K3 routing shape: 896 experts / top_k 16 without expert groups.

    Exercises the (1024, 32) tier of the trtllm-gen DeepSeekV3 routing
    kernels, including the cooperative small-batch kernel at num_tokens 1,
    via the fused-routing path (router logits consumed in-kernel).
    """
    test_trtllm_bf16_unquantized_moe(
        routing_kind="deepseekv3",
        activation_type=ActivationType.Swiglu,
        seq_len=seq_len,
        trtllm_use_router_logits=True,
        num_experts=896,
        top_k=16,
    )


# =====================================================================
# TRTLLM-Gen shared-expert fusion (migrated from deprecated
# tests/unittest/_torch/thop/serial/test_moe.py::TestMoeFP8 fusion coverage)
# =====================================================================
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

    def __init__(self, routing_method, num_experts: int, n_fused: int) -> None:
        self._routing_method = routing_method
        self._num_experts = num_experts
        self._n_fused = n_fused

    def apply(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _write_fused_shared_expert_slots(
    backend: MoE, weights: dict, num_experts: int, n_fused: int
) -> None:
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


@contextmanager
def fine_grained_sync_env(enabled: bool):
    """Toggle TLLM_USE_FINE_GRAINED_SYNC for the duration of a test case.

    The C++ side reads this env var fresh on each kernel-option construction
    (envUtils.cpp getEnvUseFineGrainedSync), so per-test scoping works within
    a single process.
    """
    prev = os.environ.get("TLLM_USE_FINE_GRAINED_SYNC")
    os.environ["TLLM_USE_FINE_GRAINED_SYNC"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TLLM_USE_FINE_GRAINED_SYNC", None)
        else:
            os.environ["TLLM_USE_FINE_GRAINED_SYNC"] = prev


@pytest.mark.parametrize("num_tokens", [1, 1024])
def test_moe_backend_trtllm_nvfp4_fine_grained(num_tokens: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    sm_version = get_sm_version()
    if sm_version != 107:
        pytest.skip(f"fine-grained sync requires SM107, got SM{sm_version}")

    dtype_activation = torch.bfloat16
    backend_type = MoeBackendType.TRTLLM
    quant_algo = QuantAlgo.NVFP4
    activation_type = ActivationType.Relu2

    num_experts = 2048
    top_k = 32
    hidden_size = 1024
    intermediate_size = 1024

    skip_if_insufficient_gpu_memory(num_experts, hidden_size, intermediate_size, dtype_activation)

    mapping = Mapping()
    mapping.rank = mpi_rank()

    # The C++ runner reads TLLM_USE_FINE_GRAINED_SYNC at kernel-option
    # construction time, so the env must be set for backend creation and the
    # forward pass alike.
    with fine_grained_sync_env(True), torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        AutoTuner.get().setup_distributed_state(mapping)

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        x = torch.randn((num_tokens, hidden_size), dtype=dtype_activation, device="cuda")
        router_logits = torch.randn(
            (num_tokens, num_experts), dtype=dtype_activation, device="cuda"
        )

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype_activation,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            activation_type=activation_type,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype_activation,
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

            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            output = run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype_activation,
                router_logits,
            )

            ref_fused_moe.check_accuracy(output, ref_output)


def test_build_moe_deployment_carries_finalize_and_locality_domain():
    """The two fields SM107 eligibility reads must survive ``ModelConfig``."""
    from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy

    default = build_moe_deployment(ModelConfig(), num_experts=8)
    assert default.fused_finalize_enabled is True
    assert default.locality_domain_requested is False

    disabled = build_moe_deployment(ModelConfig(moe_disable_finalize_fusion=True), num_experts=8)
    assert disabled.fused_finalize_enabled is False

    requested = build_moe_deployment(
        ModelConfig(locality_domain_policy=LocalityDomainPolicy(enabled=True)),
        num_experts=8,
    )
    assert requested.locality_domain_requested is True


def _nvfp4_problem(intermediate_size: Optional[int], activation: str) -> MoEProblem:
    return MoEProblem(
        quant=QuantAlgo.NVFP4.value,
        dtype_act=torch.bfloat16,
        hidden_size=7168,
        intermediate_size=intermediate_size,
        num_experts=256,
        top_k=8,
        activation=activation,
    )


def _deployment_at_moe_tp(moe_tp_size: int) -> MoEDeployment:
    return MoEDeployment(
        ep_size=1,
        tp_size=moe_tp_size,
        parallel_size=moe_tp_size,
        use_dp=False,
        num_slots=256,
        env=MoEEnvironment(sm=100),
    )


@pytest.mark.parametrize(
    "backend_cls", [CutlassFusedMoE, CuteDslFusedMoE], ids=["cutlass", "cutedsl"]
)
@pytest.mark.parametrize(
    "intermediate_size,activation,moe_tp_size,rejected",
    [
        # DeepSeek-R1-0528 NVFP4. moe_tp_size=64 is what tp_size=8 + cp_size=8
        # HELIX resolves to, and it shards FC1 to 64 rows: CuteDSL dies in
        # unswizzle_sf, Cutlass returns an all-zero output. NVBUG 5859751.
        (2048, "Swiglu", 32, False),
        (2048, "Swiglu", 64, True),
        # Unknown is not false: an absent shape must abstain, not reject.
        (None, "Swiglu", 64, False),
        # Nemotron-3 Nano NVFP4: 128-unaligned, but non-gated FC1 has no
        # gate/up split so the padding is a zero tail and it loads fine.
        (1856, "Relu2", 1, False),
    ],
    ids=["gated_aligned", "gated_unaligned", "unknown_shape", "non_gated"],
)
def test_nvfp4_fc1_row_alignment_gate(
    backend_cls, intermediate_size, activation, moe_tp_size, rejected
):
    """A gated NVFP4 FC1 buffer rounded up to the 128-row block-scale tile
    splits gate/up on padding, so those layers must be turned down here.
    """
    verdict = backend_cls.can_implement(
        _nvfp4_problem(intermediate_size, activation), _deployment_at_moe_tp(moe_tp_size)
    )
    if rejected:
        assert not verdict.eligible
        assert verdict.reject_reason is MoERejectReason.SHAPE_UNALIGNED
        assert "moe_expert_parallel_size" in verdict.detail
    else:
        # Other gates may still turn the layer down; this one must not.
        assert verdict.reject_reason is not MoERejectReason.SHAPE_UNALIGNED


def test_unresolvable_layer_error_carries_rejection_details():
    """describe() prints reason codes only, so impl_class_for has to add the
    details -- without them a shape rejection reaches the operator as a bare
    ``shape_unaligned`` with no shapes and no way forward.
    """
    report = MoEResolutionReport(
        problem=_nvfp4_problem(2048, "Swiglu"),
        deployment=_deployment_at_moe_tp(64),
        winner=None,
        selected_by="failed",
        rejected=(
            MoERejection(
                "CUTLASS", MoERejectReason.SHAPE_UNALIGNED, "raise moe_expert_parallel_size"
            ),
        ),
        requested="CUTLASS",
    )
    with pytest.raises(ValueError, match="raise moe_expert_parallel_size"):
        impl_class_for(report)
