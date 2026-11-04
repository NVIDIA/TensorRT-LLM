# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._lazy_import import lazy_imports

_CHECKPOINT_CLASS_TO_MODULE = {
    "BaseCheckpointLoader": "base_checkpoint_loader",

    # Huggingface
    "HfCheckpointLoader": "hf.checkpoint_loader",
    "HfConfigLoader": "hf.config_loader",
    "HfWeightLoader": "hf.weight_loader",
    "HfWeightMapper": "hf.weight_mapper",
    "AfmoeHfWeightMapper": "hf.afmoe_weight_mapper",
    "Cosmos3EdgeHfWeightMapper": "hf.cosmos3_edge_weight_mapper",
    "Cosmos3HfWeightMapper": "hf.cosmos3_weight_mapper",
    "Gemma3HfWeightMapper": "hf.gemma3_weight_mapper",
    "Gemma4HfWeightMapper": "hf.gemma4_weight_mapper",
    "Glm5NextHfWeightMapper": "hf.glm5_next_weight_mapper",
    "Llama4HfWeightMapper": "hf.llama4_weight_mapper",
    "LlavaNextHfWeightMapper": "hf.llava_next_weight_mapper",
    "MixtralHfWeightMapper": "hf.mixtral_weight_mapper",
    "NemotronHHfWeightMapper": "hf.nemotron_h_weight_mapper",
    "NemotronNasHfWeightMapper": "hf.nemotron_nas_weight_mapper",
    "Qwen2MoeHfWeightMapper": "hf.qwen2_moe_weight_mapper",
    "Qwen2VLHfWeightMapper": "hf.qwen2vl_weight_mapper",
    "Qwen3_5MoeHfWeightMapper": "hf.qwen3_5_weight_mapper",
    "Qwen3MoeHfWeightMapper": "hf.qwen3_moe_weight_mapper",
    "Qwen3NextHfWeightMapper": "hf.qwen3_next_weight_mapper",
    "Qwen3VLHfWeightMapper": "hf.qwen3vl_weight_mapper",
    "Qwen4ExpHfWeightMapper": "hf.qwen4_exp_weight_mapper",

    # Mistral
    "MistralCheckpointLoader": "mistral.checkpoint_loader",
    "MistralLarge3CheckpointLoader": "mistral.checkpoint_loader",
    "MistralConfigLoader": "mistral.config_loader",
    "MistralLarge3WeightMapper": "mistral.weight_mapper",
    "MistralWeightMapper": "mistral.weight_mapper",

    # ModelExpress
    "MXCheckpointLoader": "mx.checkpoint_loader",
}

__all__ = [
    "AfmoeHfWeightMapper", "HfConfigLoader", "HfWeightLoader", "HfWeightMapper",
    "MistralConfigLoader", "MistralWeightMapper", "MistralCheckpointLoader",
    "BaseCheckpointLoader", "HfCheckpointLoader", "NemotronHHfWeightMapper",
    "NemotronNasHfWeightMapper", "Gemma3HfWeightMapper",
    "MixtralHfWeightMapper", "Llama4HfWeightMapper", "Qwen2MoeHfWeightMapper",
    "Qwen3MoeHfWeightMapper", "Qwen2VLHfWeightMapper",
    "Qwen3_5MoeHfWeightMapper", "Qwen3NextHfWeightMapper",
    "Gemma4HfWeightMapper", "LlavaNextHfWeightMapper",
    "MistralLarge3CheckpointLoader", "MistralLarge3WeightMapper",
    "MXCheckpointLoader", "Qwen3VLHfWeightMapper", "Cosmos3HfWeightMapper",
    "Cosmos3EdgeHfWeightMapper", "Qwen4ExpHfWeightMapper",
    "Glm5NextHfWeightMapper"
]

__getattr__, __dir__ = lazy_imports(__name__, globals(),
                                    _CHECKPOINT_CLASS_TO_MODULE)
