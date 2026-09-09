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

from tensorrt_llm._torch.models.modeling_deepseekv3 import Deepseekv3MoE
from tensorrt_llm._torch.moe.fused_moe import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

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
