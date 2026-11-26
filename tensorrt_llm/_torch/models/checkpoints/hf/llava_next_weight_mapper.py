# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm.checkpoint_format import CheckpointFormat


@register_mapper(CheckpointFormat.HF, "LlavaNextForConditionalGeneration")
class LlavaNextHfWeightMapper(HfWeightMapper):
    def preprocess_weights(self, weights: dict) -> dict:
        transformed_weights = {}
        for key, value in weights.items():
            if key.startswith("model."):
                new_key = key[len("model.") :]
                transformed_weights[new_key] = value
            else:
                transformed_weights[key] = value
        return transformed_weights
