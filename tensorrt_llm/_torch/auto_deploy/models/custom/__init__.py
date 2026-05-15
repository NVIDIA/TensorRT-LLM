# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import importlib
import logging
import os

_logger = logging.getLogger(__name__)

# Import each custom model individually so that models with transitive TRT-LLM
# dependencies (e.g., NemotronH needing mamba layernorm_gated) don't prevent
# other models from loading in standalone mode.
_MODEL_MODULES = {
    "modeling_deepseek": ["DeepSeekV3ForCausalLM"],
    "modeling_gemma3n": ["Gemma3nForCausalLM", "Gemma3nForConditionalGeneration"],
    "modeling_gemma4": ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"],
    "modeling_glm4_moe_lite": ["Glm4MoeLiteForCausalLM"],
    "modeling_gpt_oss": ["GptOssForCausalLM"],
    "modeling_kimi_k2": ["KimiK2ForCausalLM", "KimiK25ForConditionalGeneration"],
    "modeling_llama4": ["Llama4ForCausalLM", "Llama4ForConditionalGeneration"],
    "modeling_minimax_m2": ["MiniMaxM2ForCausalLM"],
    "modeling_mistral3": ["Mistral3ForConditionalGenerationAD", "Mistral4ForCausalLM"],
    "modeling_nemotron_flash": ["NemotronFlashForCausalLM", "NemotronFlashPreTrainedTokenizerFast"],
    "modeling_nemotron_h": ["NemotronHForCausalLM"],
    "modeling_qwen3_5_moe": ["Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
    "modeling_qwen3_moe": ["Qwen3MoeForCausalLM"],
    "modeling_starcoder2": ["Starcoder2ForCausalLM"],
}

if os.environ.get("AD_USE_IR_MODELS"):
    _MODEL_MODULES["modeling_deepseek_ir"] = ["DeepSeekV3ForCausalLM"]
    _MODEL_MODULES["modeling_llama3_ir"] = ["Llama3ForCausalLM"]
    _MODEL_MODULES["modeling_nemotron_h_ir"] = ["NemotronHForCausalLM"]
    _MODEL_MODULES["modeling_qwen3_5_moe_ir"] = [
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ]
    _MODEL_MODULES["modeling_qwen3_ir"] = ["Qwen3ForCausalLM"]

__all__ = []
for _module_name, _names in _MODEL_MODULES.items():
    try:
        _mod = importlib.import_module(f".{_module_name}", __name__)
        for _name in _names:
            globals()[_name] = getattr(_mod, _name)
            if _name not in __all__:
                __all__.append(_name)
    except (ImportError, ModuleNotFoundError, ValueError) as _exc:
        _logger.debug("Skipping custom model %s: %s", _module_name, _exc)

__all__ = tuple(__all__)
