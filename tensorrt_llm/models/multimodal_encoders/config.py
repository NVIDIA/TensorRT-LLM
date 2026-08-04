# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional, Union

import torch

from ..._utils import torch_dtype_to_str
from ...logger import logger
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class LlavaNextVisionConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 image_size: int,
                 patch_size: int,
                 text_hidden_size: int,
                 projector_hidden_act: str = 'gelu',
                 num_channels: int = 3,
                 vision_model_type: str = 'clip_vision_model',
                 **kwargs):
        self.image_size = image_size
        self.patch_size = patch_size
        self.text_hidden_size = text_hidden_size
        self.num_channels = num_channels
        self.projector_hidden_act = projector_hidden_act
        self.vision_model_type = vision_model_type

        super().__init__(**kwargs)

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True)
            if hf_config.model_type == "llava_next":
                from transformers import LlavaNextConfig
                hf_config = LlavaNextConfig.from_pretrained(hf_config_dir)
            else:
                logger.error("Provided model type is not llava_next.")

        text_hidden_size = hf_config.text_config.hidden_size
        # Extract only the vision config
        llava_next_vision_config = hf_config.vision_config

        # llava-next uses the second last layer as vision output
        num_feature_layers = llava_next_vision_config.num_hidden_layers + hf_config.vision_feature_layer + 1

        vision_model_type = getattr(llava_next_vision_config,
                                    "vision_model_type", "clip_vision_model")

        num_key_value_heads = getattr(
            llava_next_vision_config, "num_key_value_heads",
            llava_next_vision_config.num_attention_heads)

        # Default configs from HF
        hidden_act = 'quick_gelu'
        norm_epsilon = 1e-5

        head_size = llava_next_vision_config.hidden_size // llava_next_vision_config.num_attention_heads

        if dtype == 'auto':
            dtype = getattr(hf_config, 'torch_dtype', None)
            if dtype is None:
                dtype = 'float16'
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == 'float32':
                dtype = 'float16'

        return cls(
            image_size=llava_next_vision_config.image_size,
            patch_size=llava_next_vision_config.patch_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=hf_config.projector_hidden_act,
            vision_model_type=vision_model_type,
            architecture=hf_config.architectures[0],
            dtype=dtype,
            num_hidden_layers=num_feature_layers,
            num_attention_heads=llava_next_vision_config.num_attention_heads,
            hidden_size=llava_next_vision_config.hidden_size,
            intermediate_size=llava_next_vision_config.intermediate_size,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            vocab_size=llava_next_vision_config.vocab_size,
            hidden_act=hidden_act,
            norm_epsilon=norm_epsilon,
            mapping=mapping,
            quantization=quant_config,
            **kwargs)
