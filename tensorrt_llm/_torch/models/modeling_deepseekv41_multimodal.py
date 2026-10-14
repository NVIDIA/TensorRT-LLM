# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 image encoder and text-model integration."""

import copy
from typing import Sequence

import torch
from torch import nn
from transformers import PreTrainedModel

from tensorrt_llm.inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    register_input_processor,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..model_config import ModelConfig
from .deepseek_v41_processor import IMAGE_PLACEHOLDER, DeepseekV41InputProcessor
from .deepseek_v41_vision import DeepseekV41Aligner, DeepseekV41VisionModel
from .modeling_deepseekv41 import DeepseekV41ForCausalLM
from .modeling_multimodal_mixin import MultimodalModelMixin
from .modeling_utils import register_auto_model


@register_auto_model("DeepseekV41ForConditionalGeneration")
@register_input_processor(
    DeepseekV41InputProcessor,
    model_type="deepseek_v41",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": IMAGE_PLACEHOLDER},
        content_format=ContentFormat.STRING,
        interleave_placeholders=True,
        placeholders_separator="",
    ),
)
class DeepseekV41ForConditionalGeneration(MultimodalModelMixin, PreTrainedModel):
    """Add vision, alignment, and image separators to the V4.1 text model.

    Attention, Engram, and DSpark remain in the text model.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        config = model_config.pretrained_config
        super().__init__(config)
        self.model_config = model_config
        self.image_token_id = config.image_token_id
        text_model_config = copy.deepcopy(model_config)
        text_config = text_model_config.pretrained_config
        text_config.architectures = ["DeepseekV41ForCausalLM"]
        text_config.text_config.use_vision_bias = True
        object.__setattr__(text_model_config, "extra_attrs", model_config.extra_attrs)
        self.llm = DeepseekV41ForCausalLM(text_model_config)
        # Re-share the attention registry after the text constructor clones configs.
        model_config.extra_attrs.update(self.llm.model_config.extra_attrs)
        self.llm.model_config.extra_attrs = model_config.extra_attrs
        self.vision = None
        self.aligner = None
        if not model_config.disable_mm_encoder:
            self.vision = self._cast_multimodal_encoder_dtype(
                DeepseekV41VisionModel(config.vision_config), self.embedding_dtype
            )
            self.aligner = self._cast_multimodal_encoder_dtype(
                DeepseekV41Aligner(config.vision_config, self.embedding_dim), self.embedding_dtype
            )
            for name in ("image_start", "image_end", "image_newline"):
                self.register_parameter(
                    name, nn.Parameter(torch.empty(self.embedding_dim, dtype=self.embedding_dtype))
                )

    @property
    def language_model(self):
        return self.llm

    @property
    def model(self):
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def text_embedding_layer(self):
        return self.llm.model.embed_tokens

    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_size

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

    @property
    def multimodal_token_ids(self):
        return torch.tensor([self.image_token_id], dtype=torch.int32, device="cpu")

    @property
    def mm_token_ids(self):
        return self.multimodal_token_ids

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        return ["image.pixel_values", "multimodal_embedding"]

    @classmethod
    def get_model_defaults(cls, llm_args):
        return DeepseekV41ForCausalLM.get_model_defaults(llm_args)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
        return DeepseekV41ForCausalLM.get_preferred_kv_cache_manager_version(pretrained_config)

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None):
        return DeepseekV41ForCausalLM.get_preferred_transceiver_runtime(pretrained_config)

    def register_cuda_graph_pre_replay_hooks(self, runner) -> None:
        self.llm.register_cuda_graph_pre_replay_hooks(runner)

    def prepare_adp_inputs(
        self, attn_metadata, *, all_token_states_required: bool, requests=None
    ) -> None:
        self.llm.prepare_adp_inputs(
            attn_metadata,
            all_token_states_required=all_token_states_required,
            requests=requests,
        )

    def prepare_request_inputs(
        self, scheduled_requests, attn_metadata, promoted_context_request_ids=frozenset()
    ) -> None:
        self.llm.prepare_request_inputs(
            scheduled_requests, attn_metadata, promoted_context_request_ids
        )

    def prepare_disagg_generation_request(self, request) -> None:
        self.llm.prepare_disagg_generation_request(request)

    def release_request_state(self, request_id: int) -> None:
        self.llm.release_request_state(request_id)

    def load_draft_weights(self, weights, weight_mapper=None) -> None:
        self.llm.load_draft_weights(weights, weight_mapper=weight_mapper)

    def post_load_weights(self) -> None:
        self.llm.post_load_weights()

    def load_weights(self, weights, weight_mapper=None) -> None:
        if self.vision is not None:
            device = self.text_embedding_layer.weight.device
            if device.type == "meta":
                device = torch.device("cuda", torch.cuda.current_device())

            def load_image_weight(
                key: str, shape: tuple[int, ...], dtype: torch.dtype
            ) -> torch.Tensor:
                if key not in weights:
                    raise ValueError(f"Missing DeepSeek-V4.1 image weight: {key}")
                value = weights[key]
                if not isinstance(value, torch.Tensor):
                    value = value[:]
                if value.shape != shape:
                    raise ValueError(f"DeepSeek-V4.1 image weight shape mismatch: {key}")
                return value.to(device=device, dtype=dtype)

            for prefix, module in (("vision", self.vision), ("aligner", self.aligner)):
                state = {
                    name: load_image_weight(f"{prefix}.{name}", target.shape, target.dtype)
                    for name, target in module.state_dict().items()
                }
                module.load_state_dict(state, strict=True, assign=True)
            for name in ("image_start", "image_end", "image_newline"):
                value = load_image_weight(name, (self.embedding_dim,), self.embedding_dtype)
                setattr(self, name, nn.Parameter(value))
        # Preserve lazy checkpoint metadata and consumable-weight ownership.
        self.llm.load_weights(weights)

    def encode_multimodal_inputs(
        self, multimodal_params: Sequence[MultimodalParams]
    ) -> torch.Tensor:
        if self.vision is None or self.aligner is None:
            raise ValueError("Raw images require a local DeepSeek-V4.1 vision encoder")
        embeddings = []
        device = self.text_embedding_layer.weight.device
        for param in multimodal_params:
            images = param.multimodal_data["image"]
            patches = images["pixel_values"]
            grids = images["image_grid_hw"]
            llm_grids = images["llm_grid_hw"]
            lengths = images["num_tokens"]
            if not (len(patches) == len(grids) == len(llm_grids) == len(lengths)):
                raise ValueError("DeepSeek-V4.1 image metadata lengths disagree")
            for pixels, (height, width), (rows, columns), length in zip(
                patches, grids, llm_grids, lengths
            ):
                pixels = pixels.to(device=device, dtype=self.embedding_dtype)
                features = self.aligner(self.vision(pixels, height, width), height, width)
                if features.shape[0] != rows * columns or length != rows * (columns + 1) + 2:
                    raise ValueError("DeepSeek-V4.1 image grid does not match its placeholder span")
                features = features.reshape(rows, columns, self.embedding_dim)
                newline = self.image_newline.to(features.dtype).expand(rows, 1, -1)
                embeddings.append(
                    torch.cat(
                        (
                            self.image_start.to(features.dtype).unsqueeze(0),
                            torch.cat((features, newline), dim=1).flatten(0, 1),
                            self.image_end.to(features.dtype).unsqueeze(0),
                        )
                    )
                )
        if not embeddings:
            raise ValueError("No images supplied to DeepSeek-V4.1 encoder")
        return torch.cat(embeddings)

    def get_language_model_extra_forward_kwargs(
        self, *, raw_input_ids, position_ids, mm_inputs, **forward_kwargs
    ) -> dict:
        image_mask = None
        if mm_inputs.inputs_embeds is not None and forward_kwargs.get("multimodal_params"):
            indices = forward_kwargs.get("mm_token_indices")
            if indices is None:
                raise ValueError("DeepSeek-V4.1 requires active mm_token_indices for image inputs")
            image_mask = torch.zeros(
                mm_inputs.inputs_embeds.shape[0],
                dtype=torch.bool,
                device=mm_inputs.inputs_embeds.device,
            )
            image_mask[indices] = True
        return {
            "orig_input_ids": raw_input_ids,
            "image_mask": image_mask,
            "spec_metadata": forward_kwargs.get("spec_metadata"),
            "resource_manager": forward_kwargs.get("resource_manager"),
            # Preserve Encoder recovery information for the text backbone.
            "context_requests": forward_kwargs.get("context_requests"),
        }
