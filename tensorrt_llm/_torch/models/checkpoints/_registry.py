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
"""Provider index for checkpoint classes registered on demand."""

import importlib
from collections.abc import Mapping

_PROVIDERS = {
    "register_checkpoint_loader": {
        "HF": "hf.checkpoint_loader",
        "MX": "mx.checkpoint_loader",
        "mistral": "mistral.checkpoint_loader",
        "mistral_large_3": "mistral.checkpoint_loader",
    },
    "register_checkpoint_weight_loader": {
        "HF": "hf.weight_loader",
        "MX": "hf.weight_loader",
        "mistral": "hf.weight_loader",
        "mistral_large_3": "hf.weight_loader",
    },
    "register_config_loader": {
        "HF": "hf.config_loader",
        "MX": "hf.config_loader",
        "mistral": "mistral.config_loader",
        "mistral_large_3": "mistral.config_loader",
    },
    "register_mapper": {
        "AfmoeForCausalLM_HF": "hf.afmoe_weight_mapper",
        "Cosmos3EdgeForConditionalGeneration_HF": "hf.cosmos3_edge_weight_mapper",
        "Cosmos3ForConditionalGeneration_HF": "hf.cosmos3_weight_mapper",
        "DeciLMForCausalLM_HF": "hf.nemotron_nas_weight_mapper",
        "Gemma3ForCausalLM_HF": "hf.gemma3_weight_mapper",
        "Gemma3ForConditionalGeneration_HF": "hf.gemma3_weight_mapper",
        "Gemma4AssistantForCausalLM_HF": "hf.gemma4_weight_mapper",
        "Gemma4ForCausalLM_HF": "hf.gemma4_weight_mapper",
        "Gemma4ForConditionalGeneration_HF": "hf.gemma4_weight_mapper",
        "Gemma4UnifiedForConditionalGeneration_HF": "hf.gemma4_weight_mapper",
        "Glm5NextForCausalLM_HF": "hf.glm5_next_weight_mapper",
        "Glm5NextForConditionalGeneration_HF": "hf.glm5_next_weight_mapper",
        "HF": "hf.weight_mapper",
        "Llama4ForConditionalGeneration_HF": "hf.llama4_weight_mapper",
        "LlavaNextForConditionalGeneration_HF": "hf.llava_next_weight_mapper",
        "MX": "hf.weight_mapper",
        "MistralForCausalLM_mistral": "mistral.weight_mapper",
        "MistralLarge3ForCausalLM_mistral": "mistral.weight_mapper",
        "MistralLarge3ForCausalLM_mistral_large_3": "mistral.weight_mapper",
        "MixtralForCausalLM_HF": "hf.mixtral_weight_mapper",
        "NemotronHForCausalLM_HF": "hf.nemotron_h_weight_mapper",
        "NemotronHPuzzleForCausalLM_HF": "hf.nemotron_h_weight_mapper",
        "PixtralForConditionalGeneration_mistral": "mistral.weight_mapper",
        "PixtralForConditionalGeneration_mistral_large_3": "mistral.weight_mapper",
        "Qwen2MoeForCausalLM_HF": "hf.qwen2_moe_weight_mapper",
        "Qwen2VLForConditionalGeneration_HF": "hf.qwen2vl_weight_mapper",
        "Qwen3MoeForCausalLM_HF": "hf.qwen3_moe_weight_mapper",
        "Qwen3NextForCausalLM_HF": "hf.qwen3_next_weight_mapper",
        "Qwen3VLForConditionalGeneration_HF": "hf.qwen3vl_weight_mapper",
        "Qwen3_5ForCausalLM_HF": "hf.qwen3_5_weight_mapper",
        "Qwen3_5ForConditionalGeneration_HF": "hf.qwen3_5_weight_mapper",
        "Qwen3_5MoeForCausalLM_HF": "hf.qwen3_5_weight_mapper",
        "Qwen3_5MoeForConditionalGeneration_HF": "hf.qwen3_5_weight_mapper",
        "Qwen4ExpForCausalLM_HF": "hf.qwen4_exp_weight_mapper",
        "Qwen4ExpForConditionalGeneration_HF": "hf.qwen4_exp_weight_mapper",
        "mistral_large_3": "mistral.weight_mapper",
    },
}


def ensure_registered(registry: Mapping[str, object], kind: str, key: str) -> None:
    """Import a built-in provider if its registry entry is absent."""
    if key not in registry:
        module = _PROVIDERS[kind].get(key)
        if module is not None:
            importlib.import_module(f".{module}", __package__)
