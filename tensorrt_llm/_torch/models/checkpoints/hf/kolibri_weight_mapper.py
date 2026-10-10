# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Dict, Union

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_moe_weight_mapper import Qwen3MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Kolibri1ForCausalLM")
class Kolibri1HfWeightMapper(Qwen3MoeHfWeightMapper):
    """Weight mapper for Kolibri 1 HuggingFace checkpoints.

    Handles mapping and fusing:
    - QKV projection fusing: (q_proj, k_proj, v_proj) -> qkv_proj
    - Router bias: .moe.router.expert_bias -> .mlp.gate.e_score_correction_bias
    - Shared experts: .shared_experts.gate_proj + up_proj -> gate_up_proj
    - Sandwich norms: post_attn_norm and post_ffn_norm weights
    """

    def __init__(self):
        super().__init__()
        self.params_map = {
            r"(.*)moe\.router\.expert_bias(.*)": r"\1mlp.gate.e_score_correction_bias\2",
            r"(.*)mlp\.expert_bias(.*)": r"\1mlp.gate.e_score_correction_bias\2",
            r"(.*)mlp\.gate\.expert_bias(.*)": r"\1mlp.gate.e_score_correction_bias\2",
            r"^(?!.*\.mlp\.)(.*)\.shared_experts\.(.*)": r"\1.mlp.shared_experts.\2",
        }

    def init_model_and_config(self, model: nn.Module, config: ModelConfig):
        super().init_model_and_config(model, config)

    def preprocess_weights(
        self, weights: Union[Dict[str, torch.Tensor], ConsumableWeightsDict]
    ) -> Union[Dict[str, torch.Tensor], ConsumableWeightsDict]:
        remapped_weights = {}
        for k, v in weights.items():
            new_k = k
            # 1. Remap router bias keys
            if ".moe.router.expert_bias" in new_k:
                new_k = new_k.replace(
                    ".moe.router.expert_bias", ".mlp.gate.e_score_correction_bias"
                )
            elif ".mlp.gate.expert_bias" in new_k:
                new_k = new_k.replace(".mlp.gate.expert_bias", ".mlp.gate.e_score_correction_bias")
            elif ".mlp.expert_bias" in new_k:
                new_k = new_k.replace(".mlp.expert_bias", ".mlp.gate.e_score_correction_bias")

            # 2. Ensure shared experts are under .mlp.shared_experts
            if ".shared_experts." in new_k and ".mlp.shared_experts." not in new_k:
                new_k = new_k.replace(".shared_experts.", ".mlp.shared_experts.")

            remapped_weights[new_k] = v
        return ConsumableWeightsDict.take_ownership(weights, remapped_weights)

