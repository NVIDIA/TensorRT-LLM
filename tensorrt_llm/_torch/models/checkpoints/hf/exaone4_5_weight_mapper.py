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

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    ConsumableWeightsDict,
    MmappedSafetensorsWeights,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Exaone4_5_ForConditionalGeneration")
class Exaone4_5HfWeightMapper(HfWeightMapper):
    def preprocess_weights(self, weights: dict):
        """Rename HF checkpoint prefixes; supports plain dict and ConsumableWeightsDict."""
        if isinstance(weights, MmappedSafetensorsWeights):
            return weights.transform_keys(
                lambda key: key.replace("model.visual.", "visual.")
                if key.startswith("model.visual.")
                else key
            )

        is_consumable = isinstance(weights, ConsumableWeightsDict)
        renamed = {}
        for key, value in weights.items():
            if key.startswith("model.visual."):
                new_key = key.replace("model.visual.", "visual.")
                renamed[new_key] = value
            elif key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "model.")
                renamed[new_key] = value
            else:
                renamed[key] = value
        if is_consumable:
            return ConsumableWeightsDict(renamed)
        return renamed
