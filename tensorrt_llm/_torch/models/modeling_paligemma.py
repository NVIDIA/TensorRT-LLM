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

import copy
import dataclasses
import os
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PaliGemmaConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper

from ..._utils import nvtx_range
from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.linear import Linear
from .modeling_gemma2 import Gemma2ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_siglip import SiglipVisionModel
from .modeling_utils import filter_weights, register_auto_model

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


class PaliGemmaInputProcessor(
    BaseMultimodalInputProcessor,
    BaseMultimodalDummyInputsBuilder,
):
    """Input processor for PaliGemma models (v1 and v2)."""

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
        self._config = config
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=self.use_fast
        )
        self._dtype = self.config.torch_dtype

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt, mm_data = inputs.get("prompt"), inputs.get("multi_modal_data", {})
        if mm_data and "image" not in mm_data:
            raise KeyError("Expected image data in multimodal data for PaliGemma.")

        images = mm_data.get("image")
        do_rescale = self.processor.image_processor.do_rescale
        if images is not None and isinstance(images[0], torch.Tensor):
            do_rescale = False
        processor_output = self.processor(
            text=text_prompt,
            images=images,
            do_rescale=do_rescale,
            return_tensors="pt",
        )

        input_ids = processor_output["input_ids"]
        pixel_values = processor_output.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        return input_ids, pixel_values

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        input_ids, pixel_values = self._preprocess(inputs)
        multimodal_data = None
        if pixel_values is not None:
            multimodal_data = {
                "multimodal_data": {"image": {"pixel_values": pixel_values}},
            }
        return input_ids[0].to(torch.int32).tolist(), multimodal_data


class PaliGemmaMultiModalProjector(torch.nn.Module):
    """Single linear projection from SigLIP hidden size to LLM hidden size."""

    def __init__(self, model_config: ModelConfig[PaliGemmaConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.linear = Linear(
            in_features=config.vision_config.hidden_size,
            out_features=config.vision_config.projection_dim,
            bias=True,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
        )

    def load_weights(self, weights):
        self.linear.weight.data.copy_(weights["linear.weight"])
        self.linear.bias.data.copy_(weights["linear.bias"])

    @torch.inference_mode()
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


@register_auto_model("PaliGemmaForConditionalGeneration")
@register_input_processor(
    PaliGemmaInputProcessor,
    model_type="paligemma",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<image>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.STRING,
    ),
)
class PaliGemmaForConditionalGeneration(PreTrainedModel):
    """PaliGemma VLM: SigLIP vision tower + linear projector + Gemma2 LLM.

    Supports both PaliGemma1 (Gemma backbone) and PaliGemma2 (Gemma2 backbone).
    Current implementation targets the Gemma2 backbone (PaliGemma2).
    """

    def __init__(self, model_config: ModelConfig[PaliGemmaConfig]):
        if _is_disagg():
            raise NotImplementedError(
                "PaliGemmaForConditionalGeneration does not support disaggregated "
                f"inference. Unset {_MULTIMODAL_ENV_NAME} or set it to '0'."
            )

        config = model_config.pretrained_config
        if config.text_config.model_type not in ("gemma2",):
            raise NotImplementedError(
                f"PaliGemma text backbone '{config.text_config.model_type}' is not "
                "supported in the _torch path. Currently only 'gemma2' is supported."
            )

        super().__init__(config)

        self._device = "cuda"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        self.register_buffer(
            "image_token_ids",
            torch.tensor([config.image_token_index], dtype=torch.int32),
            persistent=False,
        )

        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp

        llm_model_config = self.get_sub_model_config(model_config_cp, "text_config")
        self.llm = Gemma2ForCausalLM(llm_model_config)

        vision_model_config = self.get_sub_model_config(model_config_cp, "vision_config")
        self.siglip_tower = SiglipVisionModel(vision_model_config, use_post_layernorm=True)

        self.mm_projector = PaliGemmaMultiModalProjector(model_config_cp).eval().to(self._device)

        self.post_config()
        self.is_loaded = True

    @staticmethod
    def get_sub_model_config(
        model_config: ModelConfig[PaliGemmaConfig],
        name: str,
    ) -> ModelConfig:
        assert name in ("text_config", "vision_config"), (
            f"Expected 'text_config' or 'vision_config', got {name!r}."
        )
        pretrained_config = getattr(model_config.pretrained_config, name)
        quant_config = model_config.quant_config if name == "text_config" else None
        preferred_backend = "FLASHINFER" if name == "text_config" else "TRTLLM"
        sub_model_config: ModelConfig[PaliGemmaConfig] = dataclasses.replace(
            model_config,
            pretrained_config=pretrained_config,
            attn_backend=preferred_backend,
            quant_config=quant_config,
        )
        if (
            hasattr(sub_model_config.pretrained_config, "torch_dtype")
            and sub_model_config.pretrained_config.torch_dtype is None
        ):
            sub_model_config.pretrained_config.torch_dtype = (
                model_config.pretrained_config.torch_dtype
            )
        return sub_model_config

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        llm_weights = filter_weights("language_model", weights)
        self.llm.load_weights(llm_weights, weight_mapper)

        vit_weights = filter_weights("vision_tower", weights)
        self.siglip_tower.load_weights(vit_weights)

        mm_projector_weights = filter_weights("multi_modal_projector", weights)
        self.mm_projector.load_weights(mm_projector_weights)

    def post_config(self):
        # Keep the top-level PaliGemmaConfig intact (preserves vision_config,
        # image_token_index, etc.) and only update the text sub-config so that
        # external callers can still inspect model.config.vision_config.
        self.model_config.pretrained_config.text_config = self.llm.config

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests = attn_metadata.num_contexts
        num_generation_requests = attn_metadata.num_generations
        logger.debug(
            f"[PaliGemmaModel::forward]{num_context_requests=}, {num_generation_requests=}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        pixel_values = [
            multimodal_param.multimodal_data["image"]["pixel_values"]
            for multimodal_param in multimodal_params
        ]

        mm_embeds = []
        if len(pixel_values) > 0:
            image_features = self._get_image_features(pixel_values=torch.cat(pixel_values))
            mm_embeds = [image_features.contiguous()]

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=self.image_token_ids,
            **kwargs,
        )
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "multimodal_params"}
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            **llm_kwargs,
        )
        return logits

    @nvtx_range("[Vision] process")
    def _get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        attn_metadata = self.siglip_tower.prepare_attn_metadata(pixel_values.shape[0])
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            # PaliGemma uses last_hidden_state (no post-layernorm), then projects
            image_features = self.siglip_tower(pixel_values, attn_metadata=attn_metadata)[-1]
            image_features = image_features.reshape(-1, image_features.shape[-1])
            image_features = self.mm_projector(image_features)
        return image_features

    @property
    def mm_token_ids(self):
        return self.image_token_ids
