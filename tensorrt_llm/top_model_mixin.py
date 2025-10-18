# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from .lora_helper import LoraConfig
from .mapping import Mapping
from .plugin.plugin import PluginConfig


class TopModelMixin:
    """Top model mixin.

    The Module classes are reused between building blocks (like Attention, MLP) and the top level models
    (like LLaMAForCausalLM).
    While there are some functions, like the loading hf/ft weights, or build/load trt engines, that are
    only supported by the top level model, not the building blocks.

    So top level model class like: LLaMAForCausalLM shall inherit this class.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_dir: str,
        dtype: Optional[str] = "float16",
        mapping: Optional[Mapping] = None,
        **kwargs,
    ):
        """Create LLM object and load weights from hugging face.

        Parameters:
            hf_model_dir: the hugging face model directory
            dtype: str, the default weights data type when loading from the hugging face model
            mapping: Mapping, specify the multi-gpu parallel strategy, when it's None, single GPU is used
        """
        raise NotImplementedError("Subclass shall override this")

    def use_lora(self, lora_config: LoraConfig):
        """Load lora weights from the give config to the module.

        Parameters:
           lora_config: the lora config
        """
        raise NotImplementedError("Subclass shall override this")

    def use_prompt_tuning(self, max_prompt_embedding_table_size: str, prompt_table_path: str):
        """Enable p tuning when build the TRT engine, call this before to_trt."""
        raise NotImplementedError

    def default_plugin_config(self, **kwargs) -> PluginConfig:
        """Return the default plugin config for this model.

        This is used when the plugin_config value is not given in to_trt() call.
        If users need to set different plugin configs, they can start from the return object and change it.
        """
        return PluginConfig(**kwargs)
