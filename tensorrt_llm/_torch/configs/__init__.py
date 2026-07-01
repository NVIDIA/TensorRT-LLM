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

from tensorrt_llm._torch.configs.cosmos3 import Cosmos3Config
from tensorrt_llm._torch.configs.deepseek_v3 import DeepseekV3Config
from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config
from tensorrt_llm._torch.configs.gemma4 import Gemma4AssistantConfig
from tensorrt_llm._torch.configs.gemma4_unified import (
    Gemma4UnifiedAudioConfig,
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVisionConfig,
)
from tensorrt_llm._torch.configs.laguna import LagunaConfig


def _register_custom_configs_with_transformers() -> None:
    # Make AutoConfig.from_pretrained / AutoTokenizer.from_pretrained accept
    # model_types that TRT-LLM understands but upstream transformers does not
    # (DeepSeek-V3.2, Kimi K2, Laguna, and Cosmos3 omni ship config.json with
    # these model_types and rely on TRT-LLM's local config workarounds;
    # likewise the gemma4_unified family).
    #
    # Without this, transformers 5.5.x falls back to a bare PreTrainedConfig
    # that lacks attributes like `max_position_embeddings`, and
    # AutoTokenizer.from_pretrained then raises AttributeError before any
    # tokenizer can be constructed. Bypass AutoConfig.register's model_type
    # consistency check for aliases (for example, DeepseekV3Config.model_type
    # is "deepseek_v3") by writing into the underlying mapping directly.
    # Registration only fills gaps: when the installed transformers already
    # ships a model_type, the native class wins.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    custom_configs = {
        # "cosmos3" is the canonical model_type; "cosmos3_omni" is kept as a
        # backward-compat alias for checkpoints that predate the rename.
        "cosmos3": Cosmos3Config,
        "cosmos3_omni": Cosmos3Config,
        "deepseek_v32": DeepseekV3Config,
        "kimi_k2": DeepseekV3Config,
        "deepseek_v4": DeepseekV4Config,
        "gemma4_assistant": Gemma4AssistantConfig,
        "laguna": LagunaConfig,
        "gemma4_unified": Gemma4UnifiedConfig,
        "gemma4_unified_text": Gemma4UnifiedTextConfig,
        "gemma4_unified_vision": Gemma4UnifiedVisionConfig,
        "gemma4_unified_audio": Gemma4UnifiedAudioConfig,
    }
    # Cosmos3Config resolves vision sub-configs via ``qwen3_vl_vision``; that
    # alias is only present in newer transformers releases.
    if "qwen3_vl_vision" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("qwen3_vl_vision", Qwen3VLVisionConfig, exist_ok=True)
    for model_type, config_class in custom_configs.items():
        if model_type in CONFIG_MAPPING:
            continue
        CONFIG_MAPPING.register(model_type, config_class, exist_ok=True)


_register_custom_configs_with_transformers()
del _register_custom_configs_with_transformers

__all__ = [
    "Cosmos3Config",
    "DeepseekV3Config",
    "DeepseekV4Config",
    "Gemma4AssistantConfig",
    "Gemma4UnifiedAudioConfig",
    "Gemma4UnifiedConfig",
    "Gemma4UnifiedTextConfig",
    "Gemma4UnifiedVisionConfig",
    "LagunaConfig",
]
