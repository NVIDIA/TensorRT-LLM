# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Factory definitions for building models related to Eagle in AutoDeploy.

This module provides EagleDrafterFactory, a specialized factory for building
Eagle speculative decoding draft models. It extends AutoModelForCausalLMFactory
to handle the mapping from base model types (e.g., "llama") to their corresponding
Eagle drafter implementations.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Type

import torch.nn as nn
from accelerate import init_empty_weights
from torch._prims_common import DeviceLikeType
from transformers import PretrainedConfig, PreTrainedModel

from ..utils._config import deep_merge_dicts
from ..utils.logger import ad_logger
from .custom.modeling_eagle import Eagle3DrafterForCausalLM
from .factory import ModelFactoryRegistry
from .hf import AutoModelForCausalLMFactory


@dataclass
class EagleConfigInfo:
    """Model-specific configuration for Eagle drafters. This class is used to store Eagle-specific information that
    we use to build the Eagle drafter model - e.g. whether to load the embedding and lm head from the target model.
    """

    config_class: Type[PreTrainedModel]
    eagle_config_defaults: dict = None

    def __post_init__(self):
        if self.eagle_config_defaults is None:
            self.eagle_config_defaults = {}


class EagleConfig(PretrainedConfig):
    """Config for Eagle3 drafter models.

    Extends PretrainedConfig with Eagle-specific parameters while preserving
    all base model config values.

    Args:
        config: Base config for the draft model from its config.json.
        draft_model_defaults: EagleConfigInfo containing model-specific defaults.
            These defaults can be overridden by values in config.json.
    """

    def __init__(self, config: PretrainedConfig, draft_model_defaults: Dict[str, Any]):
        config_dict = config.to_dict()

        # Log when config overrides a default
        for key, value in draft_model_defaults.items():
            if key in config_dict and config_dict[key] != value:
                ad_logger.info(
                    f"EagleConfig: config has '{key}={config_dict[key]}', "
                    f"overriding default '{value}'"
                )

        merged = deep_merge_dicts(draft_model_defaults, config_dict)
        super().__init__(**merged)


@ModelFactoryRegistry.register("EagleDrafter")
class EagleDrafterFactory(AutoModelForCausalLMFactory):
    """Factory for building Eagle drafter models.

    This factory handles the mapping from base model types (e.g., "llama") to
    their corresponding Eagle drafter model implementations. It overrides
    _build_model() to directly construct the appropriate drafter class based
    on the checkpoint's model_type.

    The checkpoint config is expected to have the base model's model_type
    (e.g., "llama") along with Eagle-specific fields like draft_vocab_size.
    """

    # Map config model_type -> EagleConfigInfo (drafter class + model-specific kwargs)
    # TODO: Possibly we can infer some of these kwargs from the drafter model config, e.g.
    # load_embedding_from_target can be inferred by the state_dict and/or draft vocab size.
    _drafter_mapping: Dict[str, EagleConfigInfo] = {
        "llama": EagleConfigInfo(
            config_class=Eagle3DrafterForCausalLM,
            eagle_config_defaults={
                "load_embedding_from_target": True,
                "load_lm_head_from_target": False,
                "num_capture_layers": 3,
            },
        ),
    }

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        model_config, unused_kwargs = self._get_model_config()

        # Select the appropriate drafter class and config based on the base model type
        model_type = model_config.model_type
        if model_type not in self._drafter_mapping:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for Eagle drafter. "
                f"Supported types: {list(self._drafter_mapping.keys())}"
            )
        config_info = self._drafter_mapping[model_type]
        drafter_cls = config_info.config_class
        ad_logger.info(
            f"EagleDrafterFactory: model_type='{model_type}' -> drafter_cls={drafter_cls.__name__}"
        )

        # Convert base config to Eagle3Config, preserving existing values
        # and applying model-specific kwargs from EagleConfigInfo
        model_config = EagleConfig(model_config, config_info.eagle_config_defaults)

        # Build the model (same pattern as parent's _build_model)
        with (init_empty_weights if device == "meta" else nullcontext)():
            model = drafter_cls._from_config(model_config, **unused_kwargs)

        if device == "meta":
            # post-init must be called explicitly for HF models with init_empty_weights
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        # Store checkpoint conversion mapping if present
        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)

        model.eval()

        return model

    def build_and_load_model(self, _device: DeviceLikeType) -> nn.Module:
        raise NotImplementedError(
            "EagleDrafterFactory does not support build_and_load_model(). "
            "Use build_model() + load_or_random_init() instead."
        )
