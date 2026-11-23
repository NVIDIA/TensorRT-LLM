# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 checkpoint tensors mapped onto the Rubin target's parameters.

The target owns its GLM-5.2 modules and loader. The checkpoint loads into
the same shapes and layouts the context server uses. The MTP draft steps derive
operands from two of the MTP layer's checkpoint tensors (``DraftWeights``), taken before
the target loader shards them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import TensorParallelMode, load_weight_shard

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


@dataclass(frozen=True, eq=False)
class DraftWeights:
    """The MTP layer's checkpoint tensors the decode target derives draft-step operands from.

    Attributes:
        eh_proj: The whole ``[H, 2H]`` eh_proj weight, on the model's device. The built-in
            eh_proj is row-parallel (``2H / tp`` input columns per rank); the draft token table
            needs every column.
        w3_w1: ``[E, 2 * I / tp, H]``: per expert, this rank's ``I / tp`` rows of up_proj, then
            the same rows of gate_proj (the low-M BF16 MoE's layout).
        w2: ``[E, H, I / tp]``: per expert, this rank's ``I / tp`` columns of down_proj.

    ``w3_w1`` and ``w2`` hold the slices the built-in MoE loader gives this rank, in plain
    row-major layout: the built-in BF16 MoE may keep its own copy in a backend layout.
    """

    eh_proj: torch.Tensor
    w3_w1: torch.Tensor
    w2: torch.Tensor


def load_weights(model, weights, draft: bool) -> Optional[DraftWeights]:
    """Load the checkpoint into the target's modules with the target loader.

    With ``draft``, first take the MTP layer's ``DraftWeights`` (the target loader releases
    checkpoint tensors once it has copied them) and return them; None without ``draft`` or
    when the checkpoint has no MTP layer.
    """
    draft_weights = _draft_weights(model, weights) if draft else None
    Glm52WeightLoader(model).load_weights(weights)
    return draft_weights


def _draft_weights(model, weights) -> Optional[DraftWeights]:
    """The MTP layer's ``DraftWeights``: checkpoint layer ``num_hidden_layers``."""
    config = model.config
    prefix = f"model.layers.{config.num_hidden_layers}"
    eh_proj_key = f"{prefix}.eh_proj.weight"
    if eh_proj_key not in weights:
        return None
    mapping = model.model_config.mapping
    tp_size, tp_rank = mapping.tp_size, mapping.tp_rank
    hidden = config.hidden_size
    inter = config.moe_intermediate_size // tp_size
    device = model.lm_head.weight.device
    eh_proj = weights[eh_proj_key][:].to(device)
    w3_w1 = torch.empty(
        (config.n_routed_experts, 2 * inter, hidden), dtype=torch.bfloat16, device=device
    )
    w2 = torch.empty((config.n_routed_experts, hidden, inter), dtype=torch.bfloat16, device=device)

    def shard(key: str, mode: TensorParallelMode) -> torch.Tensor:
        tensor = load_weight_shard(weights[key], tp_size, tp_rank, mode, device=device)
        assert tensor.dtype == torch.bfloat16, f"{key}: the MTP layer's experts are not BF16"
        return tensor

    for expert in range(config.n_routed_experts):
        name = f"{prefix}.mlp.experts.{expert}"
        # The built-in MoE loader's slices: up_proj and gate_proj by rows, down_proj by columns.
        w3_w1[expert, :inter].copy_(shard(f"{name}.up_proj.weight", TensorParallelMode.COLUMN))
        w3_w1[expert, inter:].copy_(shard(f"{name}.gate_proj.weight", TensorParallelMode.COLUMN))
        w2[expert].copy_(shard(f"{name}.down_proj.weight", TensorParallelMode.ROW))
    return DraftWeights(eh_proj, w3_w1, w2)
