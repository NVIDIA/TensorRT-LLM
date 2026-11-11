# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and quantization scope for the GLM target's MTP layers."""

import copy
import os
from dataclasses import replace
from typing import Optional

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _set_draft_kv_cache_quant_algo(draft_config: ModelConfig, target_config: ModelConfig) -> None:
    """Inherit the target KV dtype unless the model requests a draft override."""
    algo = target_config.quant_config.kv_cache_quant_algo
    override = target_config.extra_attrs.get("draft_kv_cache_quant_algo_override")
    if override is not None:
        algo = QuantAlgo(override)
    draft_config.quant_config.kv_cache_quant_algo = algo
    if override is not None and draft_config.quant_config_dict is not None:
        for layer_quant_config in draft_config.quant_config_dict.values():
            layer_quant_config.kv_cache_quant_algo = algo


def _replacement_mtp_model_config(model_config: ModelConfig) -> ModelConfig:
    """Build an independent quantization scope for replacement MTP weights."""
    from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper

    checkpoint_dir = str(model_config.spec_config.speculative_model)
    quant_path = os.path.join(checkpoint_dir, "hf_quant_config.json")
    quant_config, layer_configs = QuantConfig(), None
    if os.path.isfile(quant_path):
        quant_config, layer_configs = ModelConfig.load_modelopt_quant_config(
            quant_path, checkpoint_dir, model_config.moe_backend
        )

    mapper = AutoCheckpointMapper.get("HF", model_config.pretrained_config.architectures[0])
    start_layer_idx = model_config.pretrained_config.num_hidden_layers
    if layer_configs is not None:
        normalized = {}
        for name, config in layer_configs.items():
            if name != "lm_head" and not name.startswith("mtp."):
                continue
            candidates = [name]
            if name.endswith((".q_proj", ".k_proj", ".v_proj")):
                candidates.append(name.rsplit(".", 1)[0] + ".qkv_proj")
            if any(
                quant_config.is_module_excluded_from_quantization(candidate)
                for candidate in candidates
            ):
                raise ValueError(f"Replacement quantization exclusion conflicts with {name}")
            mapped_name = mapper.map_mtp_module_name(name, start_layer_idx)
            if mapped_name.startswith("mtp."):
                raise ValueError(
                    f"{type(mapper).__name__} does not map replacement MTP quantization for {name}"
                )
            normalized[mapped_name] = config
        layer_configs = normalized
    if quant_config.exclude_modules is not None:
        quant_config.exclude_modules = [
            mapper.map_mtp_module_name(name, start_layer_idx)
            for name in quant_config.exclude_modules
        ]

    draft_config = replace(
        model_config,
        quant_config=quant_config,
        quant_config_dict=layer_configs,
        skip_create_weights_in_init=True,
    )
    draft_config.extra_attrs = model_config.extra_attrs
    _set_draft_kv_cache_quant_algo(draft_config, model_config)
    for config in (draft_config.quant_config_dict or {}).values():
        config.kv_cache_quant_algo = draft_config.quant_config.kv_cache_quant_algo
    return draft_config


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
