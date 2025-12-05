import os
from typing import List

from transformers import PretrainedConfig

from ...inputs import (
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
)
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3vl_moe_weight_mapper import Qwen3VLMoeHfWeightMapper
from .modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModelBase,
)
from .modeling_utils import ModelConfig, register_auto_model, register_vision_encoder

DISAGG = os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("Qwen3VLMoeForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen3_vl_moe",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
class Qwen3MoeVLModel(Qwen3VLModelBase):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        # NOTE: HF implementation.
        kwargs["vision_model_class"] = Qwen3VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get(
            "disable_fuse_rope", False
        )  # TODO: Make this ModelConfig's argument
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            # "image.image_grid_thw", # TODO: Remove this once we have TRT-LLM module
            "video.pixel_values_videos",
            # "video.video_grid_thw", # TODO: Remove this once we have TRT-LLM module
            "multimodal_embedding",
        ]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        weight_mapper = Qwen3VLMoeHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)

        if not DISAGG:
            self.mm_encoder.load_weights(weights)

        transformed_weights = {}
        language_model_prefix = "model.language_model."
        vision_model_prefix = "model.visual."
        for key, value in weights.items():
            if key.startswith(language_model_prefix):
                new_key = "model." + key[len(language_model_prefix) :]
                transformed_weights[new_key] = value
            elif key.startswith(vision_model_prefix):
                continue
            else:
                transformed_weights[key] = value
        self.llm.load_weights(transformed_weights, weight_mapper)
