from typing import Dict, List

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_disagg

from ...inputs import (
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    support_multimodal_disaggregated,
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


# NOTE: this is technically not strictly necessary, since the underlying mechanism for registering
# support is tacked onto the input processor class (`Qwen3VLInputProcessorBase`). Given that
# the `Qwen3VLModel` (defined via the import of `modeling_qwen3vl.py` in this file) has that
# decorator applied to it, and uses the same input processor class, we get it "for free" here.
# However, we keep it here to explicitly signify intent that this is supported. This also shields
# it from e.g. the input processor classes becoming specialized between `Qwen3VLModel` and the
# below MoE class.
@support_multimodal_disaggregated
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
            "video.pixel_values_videos",
            "multimodal_embedding",
        ]

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if not _is_disagg():
            self.mm_encoder.load_weights(weights)

        weight_mapper = Qwen3VLMoeHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        params_map = {
            r"^model\.language_model\.(.*)$": r"model.\1",
        }
        self.llm.load_weights(filtered_weights, weight_mapper, params_map=params_map)
