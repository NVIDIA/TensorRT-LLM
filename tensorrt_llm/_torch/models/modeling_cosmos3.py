# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, Tuple

import safetensors.torch
import torch
from transformers import PretrainedConfig

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ..model_config import ModelConfig
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.cosmos3_weight_mapper import Cosmos3HfWeightMapper
from .modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModel,
)
from .modeling_utils import register_auto_model, register_vision_encoder


def _get_cosmos3_model_paths(config: PretrainedConfig) -> Tuple[str, str, str]:
    """Resolve unified Cosmos3 checkpoint paths from the omni config.

    Unified Cosmos3 checkpoints use `transformer/` for LLM weights and
    `vision_encoder/` for the vision tower.
    """
    root_path = config._name_or_path

    root_path = os.fspath(root_path)
    llm_path = os.path.join(root_path, "transformer")
    vision_path = os.path.join(root_path, "vision_encoder")

    if not os.path.isdir(llm_path):
        raise FileNotFoundError(f"Cosmos3 transformer weights not found under {llm_path}.")

    return root_path, llm_path, vision_path


PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="",
    content_format=ContentFormat.STRING,
)


@support_multimodal_disaggregated
@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("Cosmos3ForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase, model_type="cosmos3", placeholder_metadata=PLACEHOLDER_METADATA
)
# cosmos3_omni is the backward-compat alias for cosmos3, remove it when checkpoints migrate to cosmos3
@register_input_processor(
    Qwen3VLInputProcessorBase, model_type="cosmos3_omni", placeholder_metadata=PLACEHOLDER_METADATA
)
class Cosmos3Model(Qwen3VLModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        omni_config = model_config.pretrained_config
        if omni_config is None:
            raise ValueError(
                "Cosmos3Model requires model_config.pretrained_config to resolve "
                "the LLM and vision encoder checkpoint paths, but it was None."
            )
        if not getattr(omni_config, "_name_or_path", None):
            raise ValueError(
                "Cosmos3Model requires model_config.pretrained_config._name_or_path to resolve "
                "the LLM and vision encoder checkpoint paths, but it was None or empty."
            )

        (self._checkpoint_root, self.llm_path, self._vision_encoder_path) = (
            _get_cosmos3_model_paths(omni_config)
        )

        super().__init__(model_config, *args, **kwargs)

    @property
    def llm_checkpoint_dir(self) -> str:
        """Return the directory of the LLM checkpoint (``transformer/`` subdir)."""
        return self.llm_path

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        vision_weights_file = os.path.join(self._vision_encoder_path, "model.safetensors")
        if not os.path.isfile(vision_weights_file):
            raise FileNotFoundError(
                f"Cosmos3 vision encoder weights not found at {vision_weights_file}."
            )
        weights.update(safetensors.torch.load_file(vision_weights_file))

        if not isinstance(weight_mapper, Cosmos3HfWeightMapper):
            weight_mapper = Cosmos3HfWeightMapper()
        weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(weights, weight_mapper)
