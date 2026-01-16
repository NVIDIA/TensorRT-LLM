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

from typing import Any, Dict

from ...utils.logger import ad_logger

EDGELLM_VERSION = "0.5.0.0"


def _export_native_llm_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export LLM configuration with required fields."""
    required_fields = [
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "rope_scaling",
    ]

    llm_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        llm_config[field] = config_dict[field]

    # Handle LongRoPE (rope_scaling already validated in required_fields)
    rope_scaling = config_dict["rope_scaling"]
    if rope_scaling and rope_scaling.get("type", None) == "longrope":
        if "original_max_position_embeddings" not in config_dict:
            raise KeyError("Required field 'original_max_position_embeddings' not found in config")
        llm_config["original_max_position_embeddings"] = config_dict[
            "original_max_position_embeddings"
        ]

    # Handle head_dim
    if "head_dim" in config_dict:
        llm_config["head_dim"] = config_dict["head_dim"]
    else:
        ad_logger.warning(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        llm_config["head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]

    if "partial_rotary_factor" in config_dict:
        llm_config["partial_rotary_factor"] = config_dict["partial_rotary_factor"]
    else:
        llm_config["partial_rotary_factor"] = 1.0

    llm_config["model_type"] = "llm"
    return llm_config


def _export_eagle_base_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export EAGLE base configuration with required fields."""
    required_fields = [
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "rope_scaling",
    ]

    eagle_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        eagle_config[field] = config_dict[field]

    # Handle head_dim
    if "head_dim" in config_dict:
        eagle_config["head_dim"] = config_dict["head_dim"]
    else:
        ad_logger.warning(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        eagle_config["head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]
    if "partial_rotary_factor" in config_dict:
        eagle_config["partial_rotary_factor"] = config_dict["partial_rotary_factor"]
    else:
        eagle_config["partial_rotary_factor"] = 1.0

    eagle_config["model_type"] = "eagle3_base"
    return eagle_config


def _export_eagle_draft_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export EAGLE draft configuration with required fields."""
    required_fields = [
        "hidden_size",
        "max_position_embeddings",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "rope_scaling",
    ]

    draft_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        draft_config[field] = config_dict[field]

    # Handle head_dim
    if "head_dim" in config_dict:
        draft_config["head_dim"] = config_dict["head_dim"]
    else:
        ad_logger.warning(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        draft_config["head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]

    # Handle draft_vocab_size based on EAGLE version
    if "draft_vocab_size" not in config_dict:
        raise KeyError("Required field 'draft_vocab_size' not found in config")
    draft_config["draft_vocab_size"] = config_dict["draft_vocab_size"]

    # Add base model configuration fields
    # The target_hidden_size from the model config represents the base model's hidden dimension
    if "target_hidden_size" in config_dict:
        # Use target_hidden_size * 3 as the base model hidden dimension (as per llm_export.py logic)
        draft_config["base_model_hidden_size"] = config_dict["target_hidden_size"] * 3
    else:
        # Fallback: assume base model hidden size is 3x draft model (Eagle3 default)
        draft_config["base_model_hidden_size"] = config_dict["hidden_size"] * 3
        ad_logger.warning(
            f"Warning: target_hidden_size not found, using default 3x draft hidden size: "
            f"{draft_config['base_model_hidden_size']}"
        )

    # Set model_type for draft
    draft_config["model_type"] = "eagle3_draft"

    return draft_config


def export_vision_config(config: Any) -> Dict[str, Any]:
    """Export vision configuration without modification."""
    config_dict = config.to_dict()

    has_vision = "vision_config" in config_dict
    has_phi4_vision = "image_embd_layer" in config_dict.get("embd_layer", {})
    if not (has_vision or has_phi4_vision):
        raise KeyError(
            "Required field 'vision_config' or 'image_embd_layer' in 'embd_layer' not found in config"
        )
    # Add EdgeLLM API version
    config_dict["edgellm_version"] = EDGELLM_VERSION

    # Return the original config_dict as-is without any modification
    # Since MRoPE needs LLM config, ViTRunner will use the LLM config.
    return config_dict


def export_llm_config(config: Any, model_type: str) -> Dict[str, Any]:
    """Export configuration based on model type and EAGLE version."""
    config_dict = config.to_dict()

    # Extract model name from config class
    config_class_name = config.__class__.__name__
    model_name = config_class_name.lower().replace("config", "")

    # For other model types, use text_config if available
    if "text_config" in config_dict:
        ad_logger.info("Detected multimodal model, using text_config")
        config_dict = config_dict["text_config"]

    if model_type == "llm":
        output_config = _export_native_llm_config(config_dict)
    elif model_type == "eagle3_base":
        output_config = _export_eagle_base_config(config_dict)
    elif model_type == "eagle_draft":
        output_config = _export_eagle_draft_config(config_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Add model name to output
    output_config["model"] = model_name

    # Add EdgeLLM API version
    output_config["edgellm_version"] = EDGELLM_VERSION

    return output_config
