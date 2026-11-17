# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 checkpoint normalization and target-owned loading."""

import copy

from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig

from ._weight_loader import Glm52WeightLoader


def normalize_model_config(
    model_config: ModelConfig[PretrainedConfig],
) -> ModelConfig[PretrainedConfig]:
    """Normalize checkpoint names and the engine-facing config alias for GLM.

    - Legacy checkpoints name the fused q_a / kv_a projection ``fused_a`` in their
      per-module quant config; the modules call it ``kv_a_proj_with_mqa``.
    - Preserve the engine-facing ``deepseek_v32`` config alias for GLM's
      MLA attention with the DSA indexer.
    """
    if model_config.quant_config_dict is not None:
        model_config = copy.deepcopy(model_config)
        quant_config_dict = {}
        for key, val in model_config.quant_config_dict.items():
            key_split = key.split(".")
            if key_split[-1] == "fused_a":
                key = ".".join(key_split[:-1] + ["kv_a_proj_with_mqa"])
            quant_config_dict[key] = val
        model_config._frozen = False
        model_config.quant_config_dict = quant_config_dict
        model_config._frozen = True
    if model_config.pretrained_config.model_type == "glm_moe_dsa":
        model_config._frozen = False
        model_config.pretrained_config.model_type = "deepseek_v32"
        model_config._frozen = True
    return model_config


def load_weights(model, weights) -> None:
    Glm52WeightLoader(model).load_weights(weights)
