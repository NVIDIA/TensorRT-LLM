# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def get_missing_qkv_modules_from_lora_modules(
        lora_target_modules: List[str]) -> List[str]:
    """Get missing QKV modules from LoRA target modules.

    In current design, q_lora_params, k_lora_params and v_lora_params should be all enabled or
    all disabled at the same time. However, some lora checkpoints (e.g. BART) only contain two of them,
    so we use zero tensor to fill the missing ones.
    """
    missing_qkv_modules = []
    if any(x in lora_target_modules for x in ["attn_q", "attn_k", "attn_v"]):
        for lora_module in ["attn_q", "attn_k", "attn_v"]:
            if lora_module not in lora_target_modules:
                missing_qkv_modules.append(lora_module)
    if any(x in lora_target_modules
           for x in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]):
        for lora_module in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]:
            if lora_module not in lora_target_modules:
                missing_qkv_modules.append(lora_module)
    return missing_qkv_modules


def get_default_trtllm_modules_to_hf_modules():
    """Get default mapping from TensorRT-LLM module names to HuggingFace module names."""
    return {
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_qkv": "qkv_proj",
        "attn_dense": "o_proj",
        "mlp_h_to_4h": "gate_proj",
        "mlp_4h_to_h": "down_proj",
        "mlp_gate": "up_proj",
        "mlp_gate_up": "gate_up_proj",
        "moe_h_to_4h": "w1",
        "moe_4h_to_h": "w2",
        "moe_gate": "w3",
        "moe_router": "gate",
    }


def use_lora(
    model,
    lora_config: "LoraConfig",
    trtllm_modules_to_hf_modules: Optional[Dict[str, str]] = None,
):
    """Use LoRA with the given model and configuration.

    This function is a wrapper that delegates to the appropriate loading function
    based on the LoRA checkpoint source.
    """
    if lora_config.lora_ckpt_source == "nemo":
        from .lora_manager import load_nemo_lora
        load_nemo_lora(model, lora_config)
    elif lora_config.lora_ckpt_source == "hf":
        from .lora_manager import load_hf_lora
        load_hf_lora(model, lora_config, trtllm_modules_to_hf_modules)
    else:
        raise ValueError(
            f"Unsupported lora_ckpt_source: {lora_config.lora_ckpt_source}")


class LoraConfig(BaseModel):
    lora_dir: List[str] = Field(default_factory=list)
    lora_ckpt_source: Literal["hf", "nemo"] = "hf"
    max_lora_rank: int = 64
    lora_target_modules: List[str] = Field(default_factory=list)
    trtllm_modules_to_hf_modules: Dict[str, str] = Field(default_factory=dict)
    max_loras: Optional[int] = None
    max_cpu_loras: Optional[int] = None
    swap_gate_up_proj_lora_b_weight: bool = True

    @property
    def missing_qkv_modules(self) -> List[str]:
        return get_missing_qkv_modules_from_lora_modules(
            self.lora_target_modules)
