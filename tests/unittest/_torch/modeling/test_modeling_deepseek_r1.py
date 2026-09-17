# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

# DeepSeek-R1 uses TRT-LLM's DeepSeek-V3 architecture implementation.
import tensorrt_llm._torch.models.modeling_deepseekv3 as deepseek_r1_modeling
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import EagerFusionConfig
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.mapping import Mapping

_HIDDEN_SIZE = 7168
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_gate_case() -> tuple[
    deepseek_r1_modeling.DeepseekV3DecoderLayer,
    torch.Tensor,
    torch.Tensor,
    RMSNorm,
]:
    mlp = deepseek_r1_modeling.Deepseekv3MoE.__new__(deepseek_r1_modeling.Deepseekv3MoE)
    torch.nn.Module.__init__(mlp)
    mlp.allreduce = None
    hidden_states = torch.zeros(
        (4, _HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual = torch.ones_like(hidden_states)
    norm = RMSNorm(
        hidden_size=_HIDDEN_SIZE,
        eps=1e-6,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    mapping = Mapping(
        world_size=16,
        rank=0,
        gpus_per_node=8,
        tp_size=16,
        enable_attention_dp=True,
    )
    model_config = ModelConfig(mapping=mapping, moe_backend="CUTEDSL")

    # Construct a real decoder-layer object without initializing the full
    # attention and MoE graph, which is outside the scope of this gate test.
    layer = deepseek_r1_modeling.DeepseekV3DecoderLayer.__new__(
        deepseek_r1_modeling.DeepseekV3DecoderLayer
    )
    torch.nn.Module.__init__(layer)
    layer.enable_wideep_flashinfer_add_add_rmsnorm = True
    layer.mapping = mapping
    layer.enable_attention_dp = mapping.enable_attention_dp
    layer.model_config = model_config
    layer.mlp = mlp
    layer.fusion_config = EagerFusionConfig()
    layer.next_layer_layernorm = norm
    return layer, hidden_states, residual, norm


def _enable_gate_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    flashinfer_norm = ModuleType("flashinfer.norm")
    flashinfer_norm._USE_CUDA_NORM = False
    monkeypatch.setattr(deepseek_r1_modeling, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(deepseek_r1_modeling, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(deepseek_r1_modeling, "is_sm_100f", lambda: True)
    monkeypatch.setattr(
        deepseek_r1_modeling,
        "flashinfer_norm",
        flashinfer_norm,
    )


def _can_use(
    layer: deepseek_r1_modeling.DeepseekV3DecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    return deepseek_r1_modeling.DeepseekV3DecoderLayer._can_use_wideep_flashinfer_add_add_rmsnorm(
        layer,
        hidden_states=hidden_states,
        residual=residual,
        do_finalize=True,
        spec_metadata=None,
    )


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, True), ("1", True), ("0", False)],
)
def test_wideep_flashinfer_add_add_rmsnorm_default_and_rollback(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: bool,
) -> None:
    env_name = deepseek_r1_modeling._WIDEEP_FLASHINFER_ADD_ADD_RMSNORM_ENV
    if env_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, env_value)

    assert deepseek_r1_modeling._is_wideep_flashinfer_add_add_rmsnorm_enabled() is expected


@_REQUIRES_CUDA
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


@_REQUIRES_CUDA
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
        "shape_mismatch",
        "not_bf16",
        "residual_not_bf16",
        "not_contiguous",
        "residual_not_contiguous",
        "post_moe_fusion",
        "weight_device_mismatch",
        "weight_shape_mismatch",
        "weight_not_bf16",
        "weight_not_contiguous",
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
        layer.mapping = Mapping(
            world_size=8,
            rank=0,
            gpus_per_node=8,
            tp_size=8,
            enable_attention_dp=True,
        )
    elif rejection == "not_attention_dp":
        layer.enable_attention_dp = False
    elif rejection == "not_cutedsl":
        layer.model_config.moe_backend = "CUTLASS"
    elif rejection == "not_cuda":
        hidden_states = hidden_states.cpu()
    elif rejection == "residual_not_cuda":
        residual = residual.cpu()
    elif rejection == "shape_mismatch":
        residual = torch.zeros(
            (3, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )
    elif rejection == "not_bf16":
        hidden_states = hidden_states.to(torch.float16)
    elif rejection == "residual_not_bf16":
        residual = residual.to(torch.float16)
    elif rejection == "not_contiguous":
        hidden_states = torch.zeros(
            (_HIDDEN_SIZE, 4),
            device="cuda",
            dtype=torch.bfloat16,
        ).T
    elif rejection == "residual_not_contiguous":
        residual = torch.ones(
            (_HIDDEN_SIZE, 4),
            device="cuda",
            dtype=torch.bfloat16,
        ).T
    elif rejection == "post_moe_fusion":
        layer.fusion_config.POST_MOE_FUSION = True
    elif rejection == "weight_device_mismatch":
        norm.to("cpu")
    elif rejection == "weight_shape_mismatch":
        layer.next_layer_layernorm = RMSNorm(
            hidden_size=_HIDDEN_SIZE + 1,
            eps=1e-6,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
    elif rejection == "weight_not_bf16":
        norm.to(torch.float16)
    elif rejection == "weight_not_contiguous":
        norm.weight = torch.nn.Parameter(
            torch.ones(
                (_HIDDEN_SIZE, 2),
                device="cuda",
                dtype=torch.bfloat16,
            )[:, 0]
        )
    elif rejection == "nvfp4_quant":
        norm.nvfp4_scale = torch.ones((), device="cuda")
    elif rejection == "high_precision_output":
        norm.return_hp_output = True
    elif rejection == "gemma":
        norm.use_gemma = True
    elif rejection == "cuda_tile":
        norm.use_cuda_tile = True
    else:
        raise AssertionError(f"Unhandled rejection: {rejection}")

    assert not _can_use(layer, hidden_states, residual)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_wideep_flashinfer_add_add_rmsnorm_falls_back_for_device_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_gate_dependencies(monkeypatch)
    layer, hidden_states, residual, _ = _make_gate_case()

    assert not _can_use(layer, hidden_states, residual.to("cuda:1"))
