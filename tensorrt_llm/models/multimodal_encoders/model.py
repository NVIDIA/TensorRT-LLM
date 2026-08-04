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
import os
from collections import OrderedDict
from typing import Optional

import safetensors

from ..._utils import numpy_to_torch
from ...functional import ACT2FN, Tensor, concat, shape, slice
from ...layers import Linear
from ...logger import logger
from ...mapping import Mapping
from ...models import CLIPVisionTransformer
from ...module import Module
from ...parameter import Parameter
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import PretrainedModel, QuantConfig
from .config import LlavaNextVisionConfig


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/llava_next/modeling_llava_next.py#L149
class LlavaNextMultiModalProjector(Module):

    def __init__(self, config: LlavaNextVisionConfig):
        super().__init__()

        self.linear_1 = Linear(config.hidden_size,
                               config.text_hidden_size,
                               dtype=config.dtype)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = Linear(config.text_hidden_size,
                               config.text_hidden_size,
                               dtype=config.dtype)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaNextVisionWrapper(PretrainedModel):

    def __init__(self, config: LlavaNextVisionConfig):
        super().__init__(config)
        self.vision_tower = None
        self.config = config
        if config.vision_model_type == "clip_vision_model":
            self.vision_tower = CLIPVisionTransformer(
                image_size=config.image_size,
                num_channels=config.num_channels,
                patch_size=config.patch_size,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                norm_epsilon=config.norm_epsilon,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                num_hidden_layers=config.num_hidden_layers,
                require_ln_f=False,
                mapping=config.mapping,
                dtype=config.dtype)
        else:
            logger.error(
                "Currently TRT-LLM only supports CLIP vision transformer.")

        self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        self.image_newline = Parameter(shape=(config.text_hidden_size, ),
                                       dtype=config.dtype)

    def forward(self, pixel_values, position_ids=None):
        image_features = self.vision_tower(pixel_values)
        select_size = concat([
            shape(image_features, 0), image_features.shape[1] - 1,
            shape(image_features, 2)
        ])
        selected_image_feature = slice(image_features,
                                       starts=[0, 1, 0],
                                       sizes=select_size)  # (bs, 576, c)
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features.mark_output('image_features', self.config.dtype)
        return image_features  # (bs, 576, c)

    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir: str,
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          quant_config: Optional[QuantConfig] = None,
                          **kwargs):
        ''' Create a LlavaNextVisionWrapper object from give parameters
        '''
        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER") is not None:
            logger.error(
                "Please enable unified converter to convert llava-next checkpoints."
            )

        config = LlavaNextVisionConfig.from_hugging_face(
            hf_model_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs)

        custom_dict = {}
        if "llava" in hf_model_dir:
            custom_dict = {
                "vision_tower": "vision_tower.vision_model",
                "input_layernorm": "layer_norm1",
                "post_layernorm": "layer_norm2",
                "fc": "fc1",
                "proj": "fc2",
                "dense": "out_proj",
                "pre_layernorm": "pre_layrnorm",
                "ln_f": "post_layernorm",
            }
        loader = ModelWeightsLoader(hf_model_dir, custom_dict)
        model = cls(config)
        loader.generate_tllm_weights(model)
        return model

    def save_checkpoint(self, output_dir, save_config=True):
        rank = self.config.mapping.rank
        weights = {
            name: numpy_to_torch(param.raw_value)
            for name, param in self.named_parameters()
        }
        image_newline = {
            "image_newline": numpy_to_torch(self.image_newline.raw_value)
        }
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        safetensors.torch.save_file(
            image_newline,
            os.path.join(output_dir, f'image_newlines.safetensors'))
        if save_config:
            self.config.to_json_file(os.path.join(output_dir, 'config.json'))

    def prepare_inputs(self, max_batch_size, **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        batch_size_range = [
            1, max(1, (max_batch_size + 1) // 2), max_batch_size
        ]
        pixel_values = Tensor(
            name='pixel_values',
            dtype=self.config.dtype,
            shape=[
                -1, self.config.num_channels, self.config.image_size,
                self.config.image_size
            ],
            dim_range=OrderedDict([
                ('batch_size', [batch_size_range]),
                ('in_channels', [[self.config.num_channels] * 3]),
                ('latent_height', [[self.config.image_size] * 3]),
                ('latent_width', [[self.config.image_size] * 3]),
            ]))
        return {'pixel_values': pixel_values}
