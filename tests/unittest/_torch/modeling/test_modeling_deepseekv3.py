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
"""Per-module quant config resolution for DeepSeek MIXED_PRECISION checkpoints.

DeepSeek-R1-W4AFP8 ships an hf_quant_config with the global
quant_algo=MIXED_PRECISION, which does not map to a single QuantMode. The MoE
experts must resolve their own per-module config (W4A8_AWQ) instead of using
the global one. These tests exercise that resolution on CPU without weights.
"""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.models.modeling_deepseekv3 as deepseekv3
from tensorrt_llm._torch.models.modeling_deepseekv3 import Deepseekv3MoE
from tensorrt_llm._torch.moe.fused_moe import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

EXPERTS_KEY = "model.layers.{}.mlp.experts"


@pytest.fixture
def mixed_precision_config():
    return QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)


@pytest.fixture
def w4a8_awq_config():
    return QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ, group_size=128)


def test_w4a8_awq_config_is_int4_weight_only_per_group(w4a8_awq_config):
    # This predicate is what selects MoEWeightLoadingMode.W4A8_CUSTOM for the
    # experts in Deepseekv3MoE, so pin it down explicitly.
    assert w4a8_awq_config.layer_quant_mode.is_int4_weight_only_per_group()


def test_experts_quant_config_resolved_from_per_module_dict(
    mixed_precision_config, w4a8_awq_config
):
    model_config = SimpleNamespace(
        quant_config=mixed_precision_config,
        quant_config_dict={EXPERTS_KEY.format(0): w4a8_awq_config},
    )

    resolved = Deepseekv3MoE._get_experts_quant_config(model_config, 0)

    assert resolved is w4a8_awq_config
    assert resolved.quant_algo == QuantAlgo.W4A8_AWQ
    assert resolved.layer_quant_mode.is_int4_weight_only_per_group()
    assert not mixed_precision_config.layer_quant_mode.is_int4_weight_only_per_group()


def test_experts_quant_config_falls_back_to_global_for_unlisted_layer(
    mixed_precision_config, w4a8_awq_config
):
    model_config = SimpleNamespace(
        quant_config=mixed_precision_config,
        quant_config_dict={EXPERTS_KEY.format(0): w4a8_awq_config},
    )

    assert Deepseekv3MoE._get_experts_quant_config(model_config, 1) is mixed_precision_config


def test_experts_quant_config_falls_back_to_global_without_dict(mixed_precision_config):
    model_config = SimpleNamespace(quant_config=mixed_precision_config, quant_config_dict=None)

    assert Deepseekv3MoE._get_experts_quant_config(model_config, 0) is mixed_precision_config


def test_expert_weight_loading_mode_w4a8_custom_for_w4a8_awq(w4a8_awq_config):
    # The resolved W4A8_AWQ expert config selects the custom loading mode; this
    # is the assignment the MoE construction makes from the resolved config.
    assert (
        Deepseekv3MoE._expert_weight_loading_mode(w4a8_awq_config)
        is MoEWeightLoadingMode.W4A8_CUSTOM
    )


def test_expert_weight_loading_mode_vanilla_for_non_int4(mixed_precision_config):
    # Neither the ambiguous MIXED_PRECISION global nor a plain FP8 config is
    # int4-weight-per-group, so both fall to VANILLA.
    assert (
        Deepseekv3MoE._expert_weight_loading_mode(mixed_precision_config)
        is MoEWeightLoadingMode.VANILLA
    )
    assert (
        Deepseekv3MoE._expert_weight_loading_mode(QuantConfig(quant_algo=QuantAlgo.FP8))
        is MoEWeightLoadingMode.VANILLA
    )


def test_expert_weight_loading_mode_none_is_vanilla():
    # override_quant_config is Optional, so the resolved expert config is None on
    # an unquantized layer; the mode selection must not dereference it.
    assert Deepseekv3MoE._expert_weight_loading_mode(None) is MoEWeightLoadingMode.VANILLA


def _bare_configurable_moe(override_quant_config):
    moe = object.__new__(ConfigurableMoE)
    moe._override_quant_config = override_quant_config
    return moe


def test_quant_config_dict_prefers_override_over_mixed_precision_global(
    mixed_precision_config, w4a8_awq_config
):
    moe = _bare_configurable_moe(w4a8_awq_config)
    model_config = SimpleNamespace(quant_config=mixed_precision_config)

    result = ConfigurableMoE._get_quant_config_dict(moe, model_config)

    assert result == {
        "has_fp8_qdq": False,
        "has_nvfp4": False,
        "has_w4afp8": True,
        "has_fp8_block_scales": False,
    }
    # The global MIXED_PRECISION mode would not have flagged w4afp8.
    assert not mixed_precision_config.layer_quant_mode.is_int4_weight_only_per_group()


def test_quant_config_dict_falls_back_to_global_without_override():
    fp8_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    moe = _bare_configurable_moe(None)
    model_config = SimpleNamespace(quant_config=fp8_config)

    result = ConfigurableMoE._get_quant_config_dict(moe, model_config)

    assert result == {
        "has_fp8_qdq": True,
        "has_nvfp4": False,
        "has_w4afp8": False,
        "has_fp8_block_scales": False,
    }


def test_quant_config_dict_is_none_when_unquantized():
    moe = _bare_configurable_moe(None)
    model_config = SimpleNamespace(quant_config=None)

    assert ConfigurableMoE._get_quant_config_dict(moe, model_config) is None
def _make_gate_case() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    mlp = deepseekv3.Deepseekv3MoE.__new__(deepseekv3.Deepseekv3MoE)
    torch.nn.Module.__init__(mlp)
    mlp.allreduce = None
    tensor = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda"),
        shape=torch.Size((4, 7168)),
        dtype=torch.bfloat16,
        dim=lambda: 2,
        is_contiguous=lambda: True,
    )
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
    return layer, tensor, norm


def _enable_gate_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deepseekv3, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(deepseekv3, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(deepseekv3, "is_sm_100f", lambda: True)
    monkeypatch.setattr(
        deepseekv3,
        "flashinfer_norm",
        SimpleNamespace(_USE_CUDA_NORM=False),
    )


def _can_use(
    layer: SimpleNamespace,
    hidden_states: SimpleNamespace,
    residual: SimpleNamespace,
) -> bool:
    return deepseekv3.DeepseekV3DecoderLayer._can_use_wideep_flashinfer_add_add_rmsnorm(
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
    layer, hidden_states, _ = _make_gate_case()

    assert _can_use(layer, hidden_states, hidden_states)


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
        "shape_mismatch",
        "not_bf16",
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
    layer, hidden_states, norm = _make_gate_case()
    residual = hidden_states

    if rejection == "disabled":
        layer.enable_wideep_flashinfer_add_add_rmsnorm = False
    elif rejection == "unsupported_sm":
        monkeypatch.setattr(deepseekv3, "is_sm_100f", lambda: False)
    elif rejection == "cuda_norm":
        deepseekv3.flashinfer_norm._USE_CUDA_NORM = True
    elif rejection == "missing_cuda_norm_flag":
        del deepseekv3.flashinfer_norm._USE_CUDA_NORM
    elif rejection == "not_multi_node":
        layer.mapping.is_multi_node = lambda: False
    elif rejection == "not_attention_dp":
        layer.enable_attention_dp = False
    elif rejection == "not_cutedsl":
        layer.model_config.moe_backend = "CUTLASS"
    elif rejection == "not_cuda":
        hidden_states.is_cuda = False
    elif rejection == "shape_mismatch":
        residual = SimpleNamespace(**vars(hidden_states))
        residual.shape = torch.Size((3, 7168))
    elif rejection == "not_bf16":
        hidden_states.dtype = torch.float16
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
