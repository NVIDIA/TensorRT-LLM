# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF adapter projections and their decoder-local layout."""

from dataclasses import dataclass

import torch
from torch import Tensor

from tensorrt_llm._torch.peft.lora.config import (
    LoraConfig,
    get_default_trtllm_modules_to_hf_modules,
)
from tensorrt_llm._torch.peft.lora.ownership import DecoderLoraOwner

HF_PROJECTIONS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_dense": "self_attn.o_proj",
    "mlp_h_to_4h": "mlp.up_proj",
    "mlp_4h_to_h": "mlp.down_proj",
}


@dataclass(frozen=True, kw_only=True)
class ShortcutLoraOwner(DecoderLoraOwner):
    strict_hf_targets = True
    head_size: int
    attention_tp: int

    def configure_hf_targets(self, config: LoraConfig) -> None:
        if any(HF_PROJECTIONS.get(k) != v for k, v in config.trtllm_modules_to_hf_modules.items()):
            raise ValueError(f"Expected decoder LoRA projection mapping {HF_PROJECTIONS}")
        requested = set(config.lora_target_modules)
        if requested == get_default_trtllm_modules_to_hf_modules().keys():
            # LlmArgs expands unspecified targets to all known projections for dynamic adapters.
            config.lora_target_modules = list(HF_PROJECTIONS)
        elif unsupported := requested - HF_PROJECTIONS.keys():
            raise ValueError(f"Unsupported decoder LoRA targets {sorted(unsupported)}")
        config.trtllm_modules_to_hf_modules = HF_PROJECTIONS.copy()
        config.lora_target_modules.extend(config.missing_qkv_modules)

    def prepare_hf_weight(self, projection: str, direction: str, weight: Tensor) -> Tensor:
        if (
            direction == "in"
            or projection not in ("self_attn.k_proj", "self_attn.v_proj")
            or self.attention_tp == 1
        ):
            return weight
        if weight.shape[0] == self.attention_tp * self.head_size:
            return weight  # The export already has one complete head per rank.
        if weight.shape[0] != self.head_size:
            raise ValueError(f"Invalid KV adapter shape {tuple(weight.shape)} for {projection}")
        # Like the base projection, each attention TP rank owns the complete single KV head.
        return torch.cat([weight] * self.attention_tp)
