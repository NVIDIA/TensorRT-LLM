# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3's SwigluBias (SwiGLU-OAI) MoE must resolve THROUGH backend resolution to CuteDslFusedMoE on SM107
NVFP4 with the deterministic FC2 (CTX-7), and must be declined everywhere else: expert biases, the
non-deterministic FC2, other SMs. Gates the dispatch, not the op."""

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

import tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl as cute_moe
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.activation import SwigluBiasActivation
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEDeployment, MoERejectReason
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep, MoEEnvironment
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import (
    build_moe_problem,
    impl_class_for,
    infer_swiglu_gptoss_style,
    resolve_moe_impl,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# M3's constants (modeling_minimaxm3: alpha 1.702, beta 1, limit 7).
_M3_ACT = SwigluBiasActivation(gate_sigmoid_scale=1.702, linear_offset=1.0, clamp=7.0)
_SHAPE = dict(num_experts=8, hidden_size=512, intermediate_size=256)


def _model_config():
    pretrained = PretrainedConfig()
    pretrained.torch_dtype = torch.bfloat16
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        moe_backend="CUTEDSL",
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
    )


def _deployment(sm):
    return MoEDeployment(
        ep_size=1,
        tp_size=1,
        use_dp=False,
        num_slots=_SHAPE["num_experts"],
        env=MoEEnvironment(sm=sm, available_deps=(MoEDep.CUTEDSL_RUBIN.value,)),
        parallel_size=1,
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
    monkeypatch.setattr(cute_moe, "_RUBIN_DETERMINISTIC_FC2", True)
    report = _resolve(107)
    assert not report.degraded, report
    assert impl_class_for(report) is cute_moe.CuteDslFusedMoE


@pytest.mark.cpu_only
@pytest.mark.parametrize("case", ["nondeterministic_fc2", "expert_bias", "sm100"])
def test_swiglu_bias_declined_outside_the_deterministic_rubin_path(monkeypatch, case) -> None:
    monkeypatch.setattr(cute_moe, "_RUBIN_DETERMINISTIC_FC2", case != "nondeterministic_fc2")
    sm, bias = (100, False) if case == "sm100" else (107, case == "expert_bias")
    problem = build_moe_problem(
        _model_config(),
        dtype=torch.bfloat16,
        swiglu_gptoss_style=infer_swiglu_gptoss_style(bias=bias, activation_type=_M3_ACT.kind),
        bias=bias,
        activation=_M3_ACT,
        **_SHAPE,
    )
    verdict = cute_moe.CuteDslFusedMoE.can_implement(problem, _deployment(sm))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)
def test_m3_create_moe_builds_cutedsl_on_this_gpu() -> None:
    """The real environment probe, the real create_moe path M3 takes, no degradation allowed."""
    if not cute_moe._RUBIN_DETERMINISTIC_FC2:
        pytest.skip("TLLM_CUTEDSL_MOE_DETERMINISTIC=0")
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
    assert isinstance(moe.backend, cute_moe.CuteDslFusedMoE), type(moe.backend)
