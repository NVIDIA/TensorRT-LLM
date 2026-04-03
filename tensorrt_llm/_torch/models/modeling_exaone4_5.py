# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_disagg

from ...inputs import (
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from .checkpoints.hf.exaone4_5_weight_mapper import Exaone4_5HfWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_qwen2vl import (
    Qwen2_5_VisionModel,
    Qwen2VisionModelBase,
    Qwen2VLInputProcessorBase,
    Qwen2VLModelBase,
)
from .modeling_utils import ModelConfig, register_auto_model, register_vision_encoder


class Exaone4_5Config(PretrainedConfig):
    """VLM config: nested ``text_config`` / ``vision_config`` from JSON become real sub-configs."""

    model_type = "exaone4_5"

    def __init__(
        self,
        text_config: Optional[Union[PretrainedConfig, dict]] = None,
        vision_config: Optional[Union[PretrainedConfig, dict]] = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = PretrainedConfig.from_dict(copy.deepcopy(text_config))
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig.from_dict(copy.deepcopy(vision_config))
        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)


AutoConfig.register(Exaone4_5Config.model_type, Exaone4_5Config)


class Exaone4_5InputProcessor(Qwen2VLInputProcessorBase):
    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = (
            inputs.get("prompt"),
            inputs.get("multi_modal_data", {}),
            inputs.get("mm_processor_kwargs", {}),
        )
        processed_inputs = self._preprocess(text_prompt, mm_data, mm_processor_kwargs)

        multimodal_data = {}
        pixel_values = processed_inputs.get("pixel_values", None)
        if pixel_values is not None:
            multimodal_data["image"] = {
                "pixel_values": pixel_values.to(self.dtype),
                "image_grid_thw": processed_inputs.get("image_grid_thw"),
            }

        pixel_values_videos = processed_inputs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos.to(self.dtype),
                "video_grid_thw": processed_inputs.get("video_grid_thw"),
            }
        fused_input_ids = processed_inputs["input_ids"][0]
        if mm_data:
            fused_input_ids = self._postprocess(fused_input_ids)

        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


class Exaone4_5_VisionModel(Qwen2VisionModelBase):
    def __init__(
        self, model_config: ModelConfig[PretrainedConfig], model_class: type[Qwen2_5_VisionModel]
    ):
        super().__init__(model_config, model_class=model_class)
        self.config.tie_word_embeddings = False


class Exaone4_5_VLModel(Qwen2VLModelBase):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        self.original_arch = model_config.pretrained_config.architectures[0]

        config = model_config.pretrained_config

        self._supports_sdpa = True
        PreTrainedModel.__init__(self, config)

        self.model_config = model_config
        self.config = model_config.pretrained_config

        if model_config.attn_backend != "TRTLLM":
            raise ValueError("Exaone4.5 only supports TRTLLM backend")

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = llm_model_config.pretrained_config.text_config
        llm_model_config.pretrained_config.tie_word_embeddings = False
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        if not _is_disagg():
            mm_encoder_config = copy.deepcopy(model_config)
            self.mm_encoder = Exaone4_5_VisionModel(
                mm_encoder_config,
                kwargs.get("vision_model_class", Qwen2_5_VisionModel),
            )
        else:
            self.mm_encoder = None

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []

        mm_multimodal_params = self._get_requests_with_mm_data(multimodal_params)

        if len(mm_multimodal_params) > 0:
            if not _is_disagg():
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=mm_multimodal_params,
                )
            else:
                raise NotImplementedError(
                    "Exaone4.5-VL does not support disaggregated inference yet. "
                    "Unset TLLM_MULTIMODAL_DISAGGREGATED or set it to '0'."
                )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_multimodal_params)

        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            **kwargs,
        )

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
        )
        return output_prob


@register_vision_encoder(Exaone4_5_VLModel, vlm_base_model=Qwen2_5_VisionModel)
@register_auto_model("Exaone4_5_ForConditionalGeneration")
@register_input_processor(
    Exaone4_5InputProcessor,
    model_type="exaone4_5",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<vision><|image_pad|></vision>",
            "video": "<vision><|video_pad|></vision>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
class Exaone4_5_ForConditionalGeneration(Exaone4_5_VLModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        kwargs["vision_model_class"] = Qwen2_5_VisionModel
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
        ]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        assert isinstance(weight_mapper, Exaone4_5HfWeightMapper)
        weights = weight_mapper.preprocess_weights(weights)
        if not _is_disagg():
            self.mm_encoder.load_weights(weights)
        self.llm.load_weights(weights, weight_mapper)
