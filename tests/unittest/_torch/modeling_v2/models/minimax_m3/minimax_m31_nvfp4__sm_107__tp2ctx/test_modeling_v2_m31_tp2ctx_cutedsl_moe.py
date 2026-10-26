# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_CUTEDSL_MOE (M1): the tp2ctx routed experts run on the CuTe DSL NVFP4 MoE with the SwigluBias epilogue.

Default ON (kill switch TRTLLM_M31_DISABLE_TP2CTX_CUTEDSL_MOE=1), with the unfused, deterministic finalize. The fork
requests the CUTEDSL backend for the routed experts only (the shared expert and the role's LlmArgs are
untouched), only when the installed CuteDslFusedMoE implements SwigluBias, and with backend degradation disallowed,
so a layer CuTe DSL cannot serve raises instead of silently running another backend. CPU.
"""

from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.utils import ActivationType

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


def test_the_routed_experts_request_cutedsl_without_degradation():
    src = (_TARGET / "modeling.py").read_text()
    assert '_CUTEDSL_MOE = not m31_disabled("TP2CTX_CUTEDSL_MOE")' in src
    assert (
        'cutedsl = _CUTEDSL_MOE and getattr(CuteDslFusedMoE, "SUPPORTS_SWIGLU_BIAS", False)' in src
    )
    # The unfused finalize: the fused one's bf16 bulk scatter-add is not run-to-run deterministic.
    assert (
        'dataclasses.replace(\n                model_config, moe_backend="CUTEDSL", moe_disable_finalize_fusion=True\n'
        in src
    )
    assert "allow_backend_degradation=not cutedsl," in src
    # Only the routed experts: create_moe is called once, and the shared expert keeps the role's model_config.
    assert src.count("self.experts = create_moe(") == 1
    assert "TP2CTX_CUTEDSL_MOE routed experts on" in src


def test_the_installed_cutedsl_moe_implements_the_minimax_activation():
    # The sm_107 SwigluBias epilogue ships with the CuTe DSL MoE port (CTX's VR branch); without it the item is not
    # taken (the probe reads False), so there is nothing to check here.
    if not getattr(CuteDslFusedMoE, "SUPPORTS_SWIGLU_BIAS", False):
        pytest.skip(
            "this tree has no CuTe DSL SwigluBias epilogue (the item falls back to the role's backend)"
        )
    assert getattr(CuteDslFusedMoE, "SUPPORTS_SWIGLU_BIAS", False) is True
    assert ActivationType.SwigluBias in CuteDslFusedMoE.activation_support.kinds
