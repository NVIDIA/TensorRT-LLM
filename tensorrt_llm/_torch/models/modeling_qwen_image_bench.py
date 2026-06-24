# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Dict, List

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_disagg

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    support_multimodal_disaggregated,
)
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from .modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModelBase,
)
from .modeling_utils import ModelConfig, register_auto_model, register_vision_encoder

_QWEN_IMAGE_BENCH_PLACEHOLDERS = MultimodalPlaceholderMetadata(
    # Match Qwen3-VL/Qwen-Image-Bench chat-template vision placeholders.
    placeholder_map={
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="",
    content_format=ContentFormat.STRING,
)


class _QwenImageBenchModelMixin:
    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
        ]

    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def model(self):
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    def load_draft_weights(self, *args, **kwargs):
        return self.llm.load_draft_weights(*args, **kwargs)

    def load_weights(
        self,
        weights: Dict[str, torch.Tensor],
        _weight_mapper: BaseWeightMapper,
    ):
        if not _is_disagg():
            self.mm_encoder.load_weights(weights)

        qwen3_5_weight_mapper = Qwen3_5MoeHfWeightMapper()
        qwen3_5_weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        self.llm.load_weights(filtered_weights, qwen3_5_weight_mapper)


@support_multimodal_disaggregated
@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("QwenImageBenchForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen3_5",
    placeholder_metadata=_QWEN_IMAGE_BENCH_PLACEHOLDERS,
)
class QwenImageBenchModel(_QwenImageBenchModelMixin, Qwen3VLModelBase):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        kwargs["vision_model_class"] = Qwen3VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get("disable_fuse_rope", False)
        super().__init__(model_config, *args, **kwargs)
