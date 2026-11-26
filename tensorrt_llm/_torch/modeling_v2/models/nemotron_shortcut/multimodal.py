# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fuse projected visual rows into the shared text decoder."""

import torch
from torch.nn import functional as F

from tensorrt_llm._torch.configs.nemotron_shortcut import NemotronShortcutConfig
from tensorrt_llm.inputs.registry import register_input_processor

from .input_processor import PLACEHOLDER_METADATA, ShortcutInputProcessor
from .vision import ShortcutVisionFrontend


class _TokenEmbeddings:
    def __init__(self, decoder):
        self.decoder = decoder

    @property
    def num_embeddings(self):
        return self.decoder.w["embed"].shape[0]

    def __call__(self, ids):
        return F.embedding(ids.long(), self.decoder.w["embed"])


@register_input_processor(
    ShortcutInputProcessor, NemotronShortcutConfig.model_type, PLACEHOLDER_METADATA
)
class ShortcutMedia:
    supported_encoder_attention_backends = {"vision": frozenset()}
    supports_mm_encoder_item_scheduling = False
    multimodal_data_device_paths = ("image.patches", "video.patches")

    def __init__(self, model_config):
        super().__init__(model_config)
        media = model_config.pretrained_config.get_media_config()
        self.vision = ShortcutVisionFrontend(model_config.copy_with(pretrained_config=media))
        self.dtype = self.model.w["embed"].dtype
        self._token_embeddings = _TokenEmbeddings(self.model)
        self._image_token_ids = torch.tensor([media.image_token_id], dtype=torch.int, device="cpu")

    @property
    def multimodal_token_ids(self):
        return self._image_token_ids

    @property
    def text_embedding_layer(self):
        return self._token_embeddings

    @property
    def embedding_dim(self):
        return self.model.hidden

    @property
    def embedding_dtype(self):
        return self.dtype

    def encode_multimodal_inputs(self, multimodal_params):
        return self.vision(multimodal_params, self.text_embedding_layer)

    def load_weights(self, weights, *args, **kwargs):
        super().load_weights(weights, *args, **kwargs)
        self.vision.load_weights(weights)

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_context_logits=False,
        spec_metadata=None,
        lora_params=None,
        **kwargs,
    ):
        params = kwargs.pop("multimodal_params", None)
        if params:
            prepared = self.prepare_multimodal_inputs(
                input_ids=input_ids,
                positions=position_ids,
                multimodal_params=params,
                num_context_requests=attn_metadata.num_contexts,
                attn_metadata=attn_metadata,
                **kwargs,
            )
            if prepared.inputs_embeds is not None:
                if inputs_embeds is not None:
                    raise ValueError("Both visual inputs and input embeddings were provided.")
                inputs_embeds = prepared.inputs_embeds
        return super().forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            spec_metadata,
            lora_params,
            **kwargs,
        )
