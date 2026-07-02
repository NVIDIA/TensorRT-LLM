# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "MiniCPMV")
class MiniCPMVHfWeightMapper(HfWeightMapper):
    """HF weight mapper for MiniCPM-V 4.5."""

    def preprocess_weights(
        self, weights: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Normalize MiniCPM-V checkpoint prefixes for the Qwen3 LLM submodule."""
        transformed_weights = {}
        for key, value in weights.items():
            if key.startswith("llm."):
                transformed_weights[key[len("llm."):]] = value
            else:
                # Keep vpm.* and resampler.* for MiniCPMVVisionModel.load_weights().
                transformed_weights[key] = value
        return transformed_weights
