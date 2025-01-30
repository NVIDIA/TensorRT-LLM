# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
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
""" PyTorch Siglip model."""

from transformers import (PretrainedConfig, SiglipImageProcessor,
                          SiglipVisionModel)

from .modeling_vit import VisionTower, VisionTowerS2


class SiglipVisionTower(VisionTower):

    def __init__(self,
                 model_name_or_path: str,
                 config: PretrainedConfig,
                 state_dict=None):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(
            model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            torch_dtype=config.model_dtype[6:],
            state_dict=state_dict,
        )
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):

    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(
            model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=config.model_dtype[6:])
        self.image_processor.size["height"] = self.image_processor.size[
            "width"] = self.scales[-1]
        self.is_loaded = True
