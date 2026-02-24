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

import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .modeling_multimodal_utils import multiscale_forward


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
