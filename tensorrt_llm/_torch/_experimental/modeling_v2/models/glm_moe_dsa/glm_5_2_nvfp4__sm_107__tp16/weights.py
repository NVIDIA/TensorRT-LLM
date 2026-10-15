# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 checkpoint tensors mapped onto the tp16 target's parameters.

The target's modules are the built-in GLM-5.2 modules (see ``modeling.py``),
so the checkpoint loads through the built-in loader into exactly the shapes
and layouts the context server uses.
"""

from __future__ import annotations

import copy

from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3WeightLoader


def normalize_model_config(model_config: ModelConfig[PretrainedConfig]) -> ModelConfig[PretrainedConfig]:
    """The two config rewrites the built-in implementation applies before building modules.

    - Legacy checkpoints name the fused q_a / kv_a projection ``fused_a`` in their
      per-module quant config; the modules call it ``kv_a_proj_with_mqa``.
    - GLM-5.2 publishes ``model_type = glm_moe_dsa``; its attention is DeepSeek-V3.2's
      (MLA with the DSA indexer), which the modules select by ``deepseek_v32``.
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
    """Load the checkpoint into the target's modules with the built-in loader."""
    DeepseekV3WeightLoader(model).load_weights(weights)
