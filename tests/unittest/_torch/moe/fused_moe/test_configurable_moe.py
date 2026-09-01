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

from unittest.mock import Mock, patch

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import _BACKEND_SYNC_ATTRS, ConfigurableMoE
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _wrapper() -> ConfigurableMoE:
    wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(wrapper)
    wrapper.num_experts = 8
    wrapper.hidden_size = 16
    wrapper.intermediate_size = 32
    wrapper.dtype = torch.bfloat16
    wrapper.reduce_results = False
    wrapper.aux_stream_dict = None
    wrapper.weight_loading_mode = None
    wrapper.apply_router_weight_on_input = False
    wrapper.activation_type = None
    wrapper._override_quant_config = None
    for attr in _BACKEND_SYNC_ATTRS:
        setattr(wrapper, attr, None)
    return wrapper


def _create_backend(
    wrapper: ConfigurableMoE,
    model_config: ModelConfig,
    override_quant_config: QuantConfig | None = None,
) -> Mock:
    backend = Mock()
    with (
        patch(
            "tensorrt_llm._torch.moe.fused_moe.create_moe.resolve_moe_cls",
            return_value=Mock(),
        ),
        patch(
            "tensorrt_llm._torch.moe.fused_moe.create_moe.create_moe_backend",
            return_value=backend,
        ),
    ):
        wrapper._create_and_sync_backend(
            model_config=model_config,
            routing_method=Mock(),
            override_quant_config=override_quant_config,
        )
    return backend


def test_layerwise_quant_config_is_applied_before_weight_creation() -> None:
    global_config = QuantConfig()
    layer_config = QuantConfig()
    model_config = ModelConfig(
        quant_config=global_config,
        quant_config_dict={"model.layers.0.mlp.experts": layer_config},
    )
    wrapper = _wrapper()
    wrapper.quant_config = global_config

    backend = _create_backend(wrapper, model_config)

    backend.create_weights.assert_not_called()
    wrapper.quant_config = layer_config
    wrapper.create_weights()

    assert backend.quant_config is layer_config
    backend.create_weights.assert_called_once_with()


def test_exclusions_only_recreate_matching_moe_weights() -> None:
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8,
        exclude_modules=["*kv_b_proj*", "*k_b_proj*", "*eh_proj"],
    )
    model_config = ModelConfig(quant_config=quant_config)
    wrapper = _wrapper()
    wrapper.quant_config = quant_config

    backend = _create_backend(wrapper, model_config)

    assert backend.quant_config is quant_config
    backend.create_weights.assert_called_once_with()
    backend.create_weights.reset_mock()
    backend._weights_created = True

    root = torch.nn.Module()
    root.model_config = ModelConfig(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.FP8,
            exclude_modules=["experts"],
        )
    )
    root.experts = wrapper

    DecoderModelForCausalLM.apply_quant_config_exclude_modules(root)

    assert not backend._weights_created
    wrapper.create_weights()
    assert wrapper.quant_config.quant_algo is None
    assert backend.quant_config.quant_algo is None
    backend.create_weights.assert_called_once_with()
