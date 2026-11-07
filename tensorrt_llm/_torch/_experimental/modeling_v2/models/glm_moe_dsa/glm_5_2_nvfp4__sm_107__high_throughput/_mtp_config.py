# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and quantization scope for the high-throughput target's MTP layers."""

import copy
from typing import Optional

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.models.modeling_utils import QuantConfig


def _copy_model_config_with_moe_backend(
    model_config: ModelConfig,
    requested_moe_backend: str,
    quant_config: Optional[QuantConfig] = None,
) -> ModelConfig:
    """Copy a ModelConfig and resolve its MoE backend against its own weights."""
    architectures = getattr(model_config.pretrained_config, "architectures", None) or []
    architecture = architectures[0] if architectures else ""
    resolved_moe_backend = ModelConfig.resolve_moe_backend(
        requested_moe_backend,
        architecture,
        quant_config=(model_config.quant_config if quant_config is None else quant_config),
    )
    draft_config = copy.copy(model_config)
    # ModelConfig is frozen after loading. Temporarily thaw only the isolated
    # draft copy so the target config remains unchanged.
    was_frozen = draft_config._frozen
    draft_config._frozen = False
    try:
        draft_config.moe_backend = resolved_moe_backend
    finally:
        draft_config._frozen = was_frozen
    return draft_config


def _get_mtp_moe_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
    """Return the routed-expert quant config for an embedded MTP layer.

    The first constructed MTP layer represents all MTP layers when resolving
    AUTO because they share one backend setting.
    """
    layer_quant_configs = model_config.quant_config_dict or {}
    prefixes = [f"model.layers.{layer_idx}."]
    num_hidden_layers = getattr(model_config.pretrained_config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_idx >= num_hidden_layers:
        prefixes.append(f"mtp.layers.{layer_idx - num_hidden_layers}.")

    for prefix in prefixes:
        quant_config = layer_quant_configs.get(f"{prefix}mlp.experts")
        if quant_config is not None:
            return quant_config
    for prefix in prefixes:
        for name, quant_config in layer_quant_configs.items():
            if name.startswith(prefix) and ".experts" in name:
                return quant_config
    return model_config.quant_config


def _enable_trtllm_moe_preload_for_draft(model: nn.Module, moe_backend: str) -> None:
    """Preserve TRTLLM-Gen's serial weight-loading order for draft-only use."""
    preload_weight_modules = getattr(model, "preload_weight_modules", None)
    if moe_backend != "TRTLLM" or preload_weight_modules is None:
        return
    for module_name in ("experts", "routing_method", "all_reduce"):
        if module_name not in preload_weight_modules:
            preload_weight_modules.append(module_name)
