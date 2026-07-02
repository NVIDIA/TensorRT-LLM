# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is based on official VILA: https://github.com/NVlabs/VILA/blob/main/llava/model/multimodal_encoder/

import os
from typing import Optional, Type

import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..attention_backend.interface import AttentionMetadata
from .modeling_multimodal_utils import multiscale_forward

# Fallback capacity for the encoder ``AttentionMetadata``'s per-segment buffers
# (``max_num_requests``). The encoder runs one attention segment per vision tile
# / window, so the real bound is the segment count, which can far exceed the
# request count (one image can expand into many tiles/windows).
#
# TODO: Once the scheduler caps an encoder forward at ``encoder_max_num_tokens``,
# derive this from the token budget instead -- ``encoder_max_num_tokens //
# min_tokens_per_segment`` (an exact segment bound). We cannot do that today:
# ``encoder_max_num_tokens`` is not yet enforced (the attention workspace grows
# past it), so a low value would under-size these fixed buffers, which -- unlike
# the token workspace -- cannot grow. Until then, fall back to the legacy
# worst-case capacity.
_ENCODER_FALLBACK_MAX_NUM_REQUESTS = 8192


class MultimodalEncoderMixin:
    """Encoder-side counterpart to ``MultimodalModelMixin``.

    Marker + default ``setup_attn_metadata`` for multimodal encoders whose
    ``AttentionMetadata`` is built by ``PyTorchModelEngine`` after model load
    using runtime sizes (``max_batch_size``, ``max_num_tokens``).

    Subclasses set ``metadata_cls`` in their own ``__init__`` (typically from
    ``get_attention_backend(model_config.attn_backend).Metadata``) and either
    use the default ``setup_attn_metadata`` below or override it for custom
    Metadata kwargs (e.g. FlashInfer ``kv_layout``, multi-metadata encoders).
    """
    metadata_cls: Type[AttentionMetadata]
    attn_metadata: Optional[AttentionMetadata] = None

    def setup_attn_metadata(self, max_num_requests: int,
                            max_num_tokens: int) -> None:
        max_num_requests = max(max_num_requests,
                               _ENCODER_FALLBACK_MAX_NUM_REQUESTS)
        self.attn_metadata = self.metadata_cls(
            max_num_requests=max_num_requests,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=None,
        )


class VisionTower(nn.Module):

    def __init__(self, model_name_or_path, config):
        super().__init__()

        assert os.path.exists(
            model_name_or_path
        ), f"Pretrained vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True)
        self.name = vision_tower_cfg.architectures[0].lower()

        if "clip" in self.name:
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path, dtype=config.model_dtype)
        elif "siglip" in self.name:
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                dtype="auto")
        else:
            raise ValueError(f"Unsupported vision tower: {self.name}")

        self.select_layer = getattr(config, "mm_vision_select_layer", -2)
        self.select_feature = getattr(config, "mm_vision_select_feature",
                                      "patch")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(
                f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(
                    image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(
                images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2


class VisionTowerS2(VisionTower):

    def __init__(self, model_name_or_path, config):
        super().__init__(model_name_or_path, config)

        self.scales = list(map(int, config.s2_scales.split(",")))
        self.scales.sort()
        self.max_split_size = config.s2_max_split_size
        self.resize_output_to_scale_idx = getattr(
            config, "s2_resize_output_to_scale_idx", 0)

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device,
                                                         dtype=self.dtype),
                                               output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(
            images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.scales,
                    max_split_size=self.max_split_size,
                    resize_output_to_idx=self.resize_output_to_scale_idx,
                )
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.scales,
                max_split_size=self.max_split_size,
                resize_output_to_idx=self.resize_output_to_scale_idx,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.scales)


class VisionTowerDynamicS2(VisionTowerS2):

    def __init__(self, model_name_or_path, config):
        super().__init__(model_name_or_path, config)

    def forward(self, images):
        assert type(images) is not list

        image_features = self.forward_feature(images)

        return image_features
