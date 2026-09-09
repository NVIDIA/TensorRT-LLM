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
from tensorrt_llm._torch.moe.fused_moe.activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivationSupport,
)
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import _BACKEND_SYNC_ATTRS, ConfigurableMoE
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _wrapper() -> ConfigurableMoE:
    wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(wrapper)
    wrapper.num_experts = 8
    # Divides ``num_experts`` so ``_reject_non_divisible_ep_backend`` returns at
    # the divisibility check. It must not fall past it: the branch below reads
    # ``type(self.backend)._supports_non_divisible_ep``, and the backend here is
    # a ``Mock`` *instance*, so that lookup lands on the ``Mock`` class and
    # raises. Assigned because ``__new__`` skipped the ``MoE.__init__`` that
    # normally derives it from the mapping.
    wrapper.ep_size = 1
    wrapper.hidden_size = 16
    wrapper.intermediate_size = 32
    wrapper.dtype = torch.bfloat16
    wrapper.reduce_results = False
    wrapper.aux_stream_dict = None
    wrapper.weight_loading_mode = None
    wrapper.apply_router_weight_on_input = False
    # Both are read while the backend is being built: the carrier goes to
    # ``resolve_moe_cls`` and to ``install_activation_params``, the kind to
    # ``infer_swiglu_gptoss_style``. ``__new__`` skipped the ``__init__`` that
    # normally assigns them, so a fixture that sets neither fails before
    # reaching the quant-config behaviour these tests are about.
    wrapper.activation = DEFAULT_MOE_ACTIVATION
    wrapper.activation_type = ActivationType(DEFAULT_MOE_ACTIVATION.kind)
    wrapper.routing_method = Mock()
    wrapper._override_quant_config = None
    for attr in _BACKEND_SYNC_ATTRS:
        setattr(wrapper, attr, None)
    return wrapper


def _backend_mock() -> Mock:
    """A ``Mock`` backend that survives ``install_activation_params``.

    That call cannot be patched out here: ``ConfigurableMoE`` makes it on the
    backend after the EPLB sync *and* inside ``create_weights``, which is the
    code path under test. A bare ``Mock`` slips past
    ``resolve_activation_support`` -- every attribute of a Mock is callable, so
    the override branch is taken -- and then fails inside
    ``materialize_activation_params``, which uses the returned Mock as a real
    declaration. Declaring a real one keeps the failure surface at the quant
    config. Still GPU-free: ``DEFAULT_MOE_ACTIVATION`` carries no constants, so
    every register short-circuits to None without allocating.
    """
    backend = Mock()
    backend.activation = DEFAULT_MOE_ACTIVATION
    backend.resolve_activation_support = Mock(
        return_value=MoEActivationSupport(
            kinds=frozenset({ActivationType.Swiglu}),
            alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
            limit=ActivationParamShape.PER_EXPERT_TENSOR,
        )
    )
    return backend


def _create_backend(
    wrapper: ConfigurableMoE,
    model_config: ModelConfig,
    override_quant_config: QuantConfig | None = None,
) -> Mock:
    backend = _backend_mock()
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
