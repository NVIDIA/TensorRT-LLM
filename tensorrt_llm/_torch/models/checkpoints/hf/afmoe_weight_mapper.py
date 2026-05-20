# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE


@register_mapper("HF", "AfmoeForCausalLM")
class AfmoeHfWeightMapper(HfWeightMapper):
    def __init__(self):
        super().__init__()

        self.params_map = {
            # MoE expert weights: gate_proj->w1, up_proj->w3, down_proj->w2
            r"(.*experts\.\d+\.)gate_proj(.*)": r"\1w1\2",
            r"(.*experts\.\d+\.)up_proj(.*)": r"\1w3\2",
            r"(.*experts\.\d+\.)down_proj(.*)": r"\1w2\2",
            # HF router weight path -> TRT-LLM gate path
            r"(.*)\.router\.gate\.(.*)": r"\1.gate.\2",
            # expert_bias -> gate.e_score_correction_bias
            r"(.*)\.mlp\.expert_bias(.*)": r"\1.mlp.gate.e_score_correction_bias\2",
        }

    def preprocess_weights(self, weights: dict) -> dict:
        weights = self.rename_by_params_map(self.params_map, weights)
        return weights

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoE)

    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if isinstance(module, MoE):
            module.load_weights(
                weights=[module_weights],
                allow_partial_loading=allow_partial_loading,
            )
