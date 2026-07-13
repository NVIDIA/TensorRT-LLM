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

# Adapted from upstream transformers Cosmos3OmniConfig:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cosmos3_omni/configuration_cosmos3_omni.py
#
# Workaround until TRT-LLM upgrades to a transformers release that registers cosmos3_omni natively.

import os

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig


@strict
class Cosmos3Config(PreTrainedConfig):
    model_type = "cosmos3"
    sub_configs = {"vision_config": AutoConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            model_type = self.vision_config.pop("model_type", "qwen3_vl_vision")
            if model_type == "qwen3_vl":
                model_type = "qwen3_vl_vision"
            self.vision_config = CONFIG_MAPPING[model_type](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["qwen3_vl_vision"]()

        if isinstance(self.text_config, dict):
            model_type = self.text_config.get("model_type", "qwen3_vl_text")
            self.text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3_vl_text"]()

        super().__post_init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config = super().from_dict(config_dict, **kwargs)
        # PreTrainedConfig.from_pretrained / from_dict hydrate declared fields from
        # config.json only. ``_name_or_path`` is runtime metadata (not stored in the
        # JSON) and is not set to the checkpoint directory automatically. Cosmos3
        # needs that path to locate ``transformer/`` and ``vision_encoder/``; TRT-LLM
        # does not pass it separately on ModelConfig, so set it when callers provide
        # one via kwargs.
        name_or_path = kwargs.get("_name_or_path") or kwargs.get("name_or_path")
        if name_or_path and (
            not getattr(config, "_name_or_path", None) or len(str(config._name_or_path)) < 2
        ):
            config._name_or_path = os.fspath(name_or_path)
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Same as from_dict above: copy the resolved checkpoint identifier onto the
        # config so Cosmos3Model can find the unified checkpoint root.
        if not getattr(config, "_name_or_path", None) or len(str(config._name_or_path)) < 2:
            config._name_or_path = os.fspath(pretrained_model_name_or_path)
        return config
