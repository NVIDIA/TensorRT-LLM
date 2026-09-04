# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# DeepSeek-R1 uses TRT-LLM's DeepSeek-V3 architecture implementation.
import tensorrt_llm._torch.models.modeling_deepseekv3 as deepseek_r1_modeling


def _make_gate_case() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    mlp = deepseek_r1_modeling.Deepseekv3MoE.__new__(deepseek_r1_modeling.Deepseekv3MoE)
    torch.nn.Module.__init__(mlp)
    mlp.allreduce = None
    hidden_states = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda"),
        shape=torch.Size((4, 7168)),
        dtype=torch.bfloat16,
        dim=lambda: 2,
        is_contiguous=lambda: True,
    )
    residual = SimpleNamespace(**vars(hidden_states))
    norm = SimpleNamespace(
        nvfp4_scale=None,
        return_hp_output=False,
        use_gemma=False,
        use_cuda_tile=False,
    )
    layer = SimpleNamespace(
        enable_wideep_flashinfer_add_add_rmsnorm=True,
        mapping=SimpleNamespace(is_multi_node=lambda: True),
        enable_attention_dp=True,
        model_config=SimpleNamespace(moe_backend="CUTEDSL"),
        mlp=mlp,
        fusion_config=SimpleNamespace(POST_MOE_FUSION=False),
        next_layer_layernorm=norm,
    )
    return layer, hidden_states, residual, norm


def _enable_gate_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deepseek_r1_modeling, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(deepseek_r1_modeling, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(deepseek_r1_modeling, "is_sm_100f", lambda: True)
    monkeypatch.setattr(
        deepseek_r1_modeling,
        "flashinfer_norm",
        SimpleNamespace(_USE_CUDA_NORM=False),
    )


def _can_use(
    layer: SimpleNamespace,
    hidden_states: SimpleNamespace,
    residual: SimpleNamespace,
) -> bool:
    return deepseek_r1_modeling.DeepseekV3DecoderLayer._can_use_wideep_flashinfer_add_add_rmsnorm(
        layer,
        hidden_states=hidden_states,
        residual=residual,
        do_finalize=True,
        spec_metadata=None,
    )


def test_wideep_flashinfer_add_add_rmsnorm_accepts_exact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate_dependencies(monkeypatch)
    layer, hidden_states, residual, _ = _make_gate_case()

    assert hidden_states is not residual
    assert _can_use(layer, hidden_states, residual)


def test_wideep_flashinfer_add_add_rmsnorm_falls_back_for_missing_shared_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlp = deepseek_r1_modeling.Deepseekv3MoE.__new__(deepseek_r1_modeling.Deepseekv3MoE)
    torch.nn.Module.__init__(mlp)
    mlp.use_dp = True
    mlp.allreduce = None
    mlp.shared_experts = None
    hidden_states = torch.zeros((2, 8), dtype=torch.bfloat16)
    routed_output = torch.ones_like(hidden_states)
    monkeypatch.setattr(
        mlp,
        "compute_routed_output",
        MagicMock(return_value=routed_output),
    )

    output = mlp(hidden_states, defer_shared_routed_add=True)

    assert output is routed_output


@pytest.mark.parametrize(
    "rejection",
    (
        "disabled",
        "unsupported_sm",
        "cuda_norm",
        "missing_cuda_norm_flag",
        "not_multi_node",
        "not_attention_dp",
        "not_cutedsl",
        "not_cuda",
        "residual_not_cuda",
        "device_mismatch",
        "shape_mismatch",
        "not_bf16",
        "residual_not_bf16",
        "not_contiguous",
        "residual_not_contiguous",
        "post_moe_fusion",
        "nvfp4_quant",
        "high_precision_output",
        "gemma",
        "cuda_tile",
    ),
)
def test_wideep_flashinfer_add_add_rmsnorm_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    rejection: str,
) -> None:
    _enable_gate_dependencies(monkeypatch)
    layer, hidden_states, residual, norm = _make_gate_case()

    if rejection == "disabled":
        layer.enable_wideep_flashinfer_add_add_rmsnorm = False
    elif rejection == "unsupported_sm":
        monkeypatch.setattr(deepseek_r1_modeling, "is_sm_100f", lambda: False)
    elif rejection == "cuda_norm":
        deepseek_r1_modeling.flashinfer_norm._USE_CUDA_NORM = True
    elif rejection == "missing_cuda_norm_flag":
        del deepseek_r1_modeling.flashinfer_norm._USE_CUDA_NORM
    elif rejection == "not_multi_node":
        layer.mapping.is_multi_node = lambda: False
    elif rejection == "not_attention_dp":
        layer.enable_attention_dp = False
    elif rejection == "not_cutedsl":
        layer.model_config.moe_backend = "CUTLASS"
    elif rejection == "not_cuda":
        hidden_states.is_cuda = False
    elif rejection == "residual_not_cuda":
        residual.is_cuda = False
    elif rejection == "device_mismatch":
        residual.device = torch.device("cuda:1")
    elif rejection == "shape_mismatch":
        residual.shape = torch.Size((3, 7168))
    elif rejection == "not_bf16":
        hidden_states.dtype = torch.float16
    elif rejection == "residual_not_bf16":
        residual.dtype = torch.float16
    elif rejection == "not_contiguous":
        hidden_states.is_contiguous = lambda: False
    elif rejection == "residual_not_contiguous":
        residual.is_contiguous = lambda: False
    elif rejection == "post_moe_fusion":
        layer.fusion_config.POST_MOE_FUSION = True
    elif rejection == "nvfp4_quant":
        norm.nvfp4_scale = object()
    elif rejection == "high_precision_output":
        norm.return_hp_output = True
    elif rejection == "gemma":
        norm.use_gemma = True
    elif rejection == "cuda_tile":
        norm.use_cuda_tile = True
    else:
        raise AssertionError(f"Unhandled rejection: {rejection}")

    assert not _can_use(layer, hidden_states, residual)
