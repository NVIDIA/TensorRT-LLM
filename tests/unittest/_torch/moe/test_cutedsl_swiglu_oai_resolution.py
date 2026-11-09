# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3's SwigluBias (SwiGLU-OAI) MoE must resolve THROUGH backend resolution to CuteDslFusedMoE on SM107
NVFP4 with the deterministic FC2 (CTX-7), and must be declined everywhere else: expert biases, the
non-deterministic FC2, other SMs. Gates the dispatch, not the op."""

from dataclasses import replace
from unittest.mock import Mock, create_autospec

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

import tensorrt_llm._torch.moe.fused_moe.cutedsl.grouped_gemm.nvfp4 as cute_moe
from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import _canonicalize_rubin_activation_beta
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.activation import SwigluBiasActivation
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.cutedsl.kernel_inputs import NvFp4WeightView
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    DETERMINISTIC_MOE_FINALIZE,
    MoEDeployment,
    MoERejectReason,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep, MoEEnvironment
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import (
    build_moe_deployment,
    build_moe_problem,
    impl_class_for,
    infer_swiglu_gptoss_style,
    resolve_moe_impl,
)
from tensorrt_llm._torch.utils import ActivationType, AuxStreamType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# M3's constants (modeling_minimaxm3: alpha 1.702, beta 1, limit 7).
_M3_ACT = SwigluBiasActivation(gate_sigmoid_scale=1.702, linear_offset=1.0, clamp=7.0)
_SHAPE = dict(num_experts=8, hidden_size=512, intermediate_size=256)


def _model_config():
    pretrained = PretrainedConfig()
    pretrained.torch_dtype = torch.bfloat16
    config = ModelConfig(
        pretrained_config=pretrained,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        moe_backend="CUTEDSL",
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
    )
    config.extra_attrs[DETERMINISTIC_MOE_FINALIZE] = True
    return config


def _deployment(sm):
    return MoEDeployment(
        ep_size=1,
        tp_size=1,
        use_dp=False,
        num_slots=_SHAPE["num_experts"],
        env=MoEEnvironment(sm=sm, available_deps=(MoEDep.CUTEDSL_RUBIN.value,)),
        parallel_size=1,
        deterministic_finalize_requested=True,
    )


def _resolve(sm, bias=False):
    return resolve_moe_impl(
        _model_config(),
        deployment=_deployment(sm),
        dtype=torch.bfloat16,
        swiglu_gptoss_style=infer_swiglu_gptoss_style(bias=bias, activation_type=_M3_ACT.kind),
        bias=bias,
        activation=_M3_ACT,
        allow_degradation=True,
        **_SHAPE,
    )


@pytest.mark.cpu_only
def test_m3_swiglu_bias_resolves_to_cutedsl_on_sm107(monkeypatch) -> None:
    report = _resolve(107)
    assert not report.degraded, report
    assert impl_class_for(report) is cute_moe.TrtllmCutedslGroupedGemmNvfp4Impl


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case", ["nondeterministic_fc2", "expert_bias", "sm100", "locality_domain"]
)
def test_swiglu_bias_declined_outside_the_deterministic_rubin_path(monkeypatch, case) -> None:
    sm, bias = (100, False) if case == "sm100" else (107, case == "expert_bias")
    problem = build_moe_problem(
        _model_config(),
        dtype=torch.bfloat16,
        swiglu_gptoss_style=infer_swiglu_gptoss_style(bias=bias, activation_type=_M3_ACT.kind),
        bias=bias,
        activation=_M3_ACT,
        **_SHAPE,
    )
    deployment = replace(
        _deployment(sm),
        locality_domain_requested=case == "locality_domain",
        deterministic_finalize_requested=case != "nondeterministic_fc2",
    )
    verdict = cute_moe.TrtllmCutedslGroupedGemmNvfp4Impl.can_implement(problem, deployment)
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)
def test_m3_create_moe_builds_cutedsl_on_this_gpu() -> None:
    """The real environment probe, the real create_moe path M3 takes, no degradation allowed."""
    with torch.device("cuda"):
        moe = create_moe(
            routing_method=RenormalizeMoeRoutingMethod(top_k=2),
            dtype=torch.bfloat16,
            model_config=_model_config(),
            activation=_M3_ACT,
            allow_backend_degradation=False,
            layer_idx=0,
            **_SHAPE,
        )
    assert isinstance(moe.backend, cute_moe.TrtllmCutedslGroupedGemmNvfp4Impl), type(moe.backend)


@pytest.mark.cpu_only
@pytest.mark.parametrize("linear_offset", [0.0, -1.0, -2.5, 1.0])
def test_rubin_oai_linear_offset_survives_parameter_conversion(linear_offset: float) -> None:
    assert (
        _canonicalize_rubin_activation_beta(ActivationType.SwigluBias, linear_offset)
        == linear_offset
    )
    assert _canonicalize_rubin_activation_beta(ActivationType.SiTu, linear_offset) == (
        linear_offset if linear_offset > 0 else None
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("opted_in", [False, True])
def test_deterministic_finalize_is_an_explicit_deployment_request(opted_in: bool) -> None:
    config = _model_config()
    config.extra_attrs.clear()
    if opted_in:
        config.extra_attrs[DETERMINISTIC_MOE_FINALIZE] = True
    deployment = build_moe_deployment(
        config, num_experts=_SHAPE["num_experts"], environment=_deployment(107).env
    )
    assert deployment.deterministic_finalize_requested is opted_in
    problem = build_moe_problem(
        config,
        dtype=torch.bfloat16,
        activation=_M3_ACT,
        swiglu_gptoss_style=infer_swiglu_gptoss_style(activation_type=_M3_ACT.kind),
        **_SHAPE,
    )
    verdict = cute_moe.TrtllmCutedslGroupedGemmNvfp4Impl.can_implement(problem, deployment)
    assert verdict.eligible is opted_in
    if not opted_in:
        assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.cpu_only
@pytest.mark.parametrize("opted_in", [False, True])
def test_rubin_fc2_dispatch_honors_per_backend_opt_in(
    monkeypatch: pytest.MonkeyPatch, opted_in: bool
) -> None:
    """Spy on the actual forward: ordinary models retain fused finalize and its memset."""
    op_names = [
        "moe_sort",
        "moe_output_memset_inplace",
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin",
        "cute_dsl_nvfp4_grouped_gemm_store_rubin",
        "cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin",
        "moe_unpermute_inplace",
    ]
    ops = Mock(spec_set=op_names)
    fake_torch = Mock(spec_set=torch)
    fake_torch.ops = Mock(spec_set=["trtllm"])
    fake_torch.ops.trtllm = ops
    for name in ["empty", "bfloat16", "uint8", "float4_e2m1fn_x2"]:
        setattr(fake_torch, name, getattr(torch, name))
    monkeypatch.setattr(cute_moe, "torch", fake_torch)
    monkeypatch.setattr(cute_moe, "get_sm_version", lambda: 107)
    monkeypatch.setattr(cute_moe, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)
    monkeypatch.setattr(cute_moe, "_DETERMINISTIC_FC2_LOGGED", True)
    monkeypatch.setattr(cute_moe, "_FINALIZE_RESIDUAL", [])
    backend = cute_moe.TrtllmCutedslGroupedGemmNvfp4Impl.__new__(
        cute_moe.TrtllmCutedslGroupedGemmNvfp4Impl
    )
    torch.nn.Module.__init__(backend)
    backend.deterministic_finalize = opted_in
    backend.use_fused_finalize = True
    backend.num_slots = 1
    backend.activation_type = ActivationType.Swiglu
    backend.act_clamp = float("inf")
    backend.mapping = Mapping()
    backend.fc2_input_scale = torch.ones(1)
    backend._pending_x_ready_event = None
    monkeypatch.setattr(backend, "_has_moe_output_memset_aux_stream", lambda: False)
    x = torch.zeros((2, 1), dtype=torch.uint8)
    scale = torch.ones((2, 1), dtype=torch.uint8)
    weight = torch.zeros((1, 2, 1), dtype=torch.uint8)
    global_scale = torch.ones(1)
    view = NvFp4WeightView(
        w3_w1_weight=weight,
        fc1_weight_scale=scale,
        fc1_global_scale=global_scale,
        w2_weight=weight,
        fc2_weight_scale=scale,
        fc2_global_scale=global_scale,
        expert_size_per_partition=1,
        slot_start=0,
    )
    metadata = torch.zeros(2, dtype=torch.int32)
    ops.moe_sort.return_value = (metadata,) * 6
    ops.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin.return_value = (x, scale)
    output = torch.empty((2, 2), dtype=torch.bfloat16)
    result = backend.run_moe_nvfp4_impl(
        x, torch.zeros((2, 1), dtype=torch.int32), torch.ones((2, 1)), scale, output, view
    )
    assert result is output
    assert ops.cute_dsl_nvfp4_grouped_gemm_store_rubin.call_count == int(opted_in)
    assert ops.moe_unpermute_inplace.call_count == int(opted_in)
    assert ops.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin.call_count == int(not opted_in)
    assert ops.moe_output_memset_inplace.call_count == int(not opted_in)


@pytest.mark.cpu_only
@pytest.mark.parametrize("opted_in", [False, True])
def test_cutedsl_construction_preserves_finalize_request(
    monkeypatch: pytest.MonkeyPatch, opted_in: bool
) -> None:
    config = _model_config()
    config.extra_attrs.clear()
    if opted_in:
        config.extra_attrs[DETERMINISTIC_MOE_FINALIZE] = True
    config.skip_create_weights_in_init = True
    monkeypatch.setattr(torch.cuda, "Event", create_autospec(torch.cuda.Event, spec_set=True))
    stream = create_autospec(torch.cuda.Stream, instance=True, spec_set=True)
    backend = cute_moe.TrtllmCutedslGroupedGemmNvfp4Impl(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        model_config=config,
        dtype=torch.bfloat16,
        activation=_M3_ACT,
        aux_stream_dict={AuxStreamType.MoeOutputMemset: stream},
        **_SHAPE,
    )
    deployment = build_moe_deployment(
        config, num_experts=_SHAPE["num_experts"], environment=_deployment(107).env
    )
    assert backend.deterministic_finalize is opted_in
    assert backend.deterministic_finalize is deployment.deterministic_finalize_requested
    assert not backend._weights_created
