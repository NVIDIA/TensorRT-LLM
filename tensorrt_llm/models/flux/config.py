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
from typing import Optional

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class FluxConfig(PretrainedConfig):

    def __init__(self, *, attention_head_dim: 128, guidance_embeds: True,
                 in_channels: 64, joint_attention_dim: 4096,
                 num_attention_heads: 24, num_layers: 19, num_single_layers: 38,
                 patch_size: 1, pooled_projection_dim: 768, **kwargs):

        kwargs.update({
            'hidden_size': attention_head_dim * num_attention_heads,
            'num_hidden_layers': num_layers,
            'num_attention_heads': num_attention_heads
        })
        super().__init__(**kwargs)
        self.attention_head_dim = attention_head_dim
        self.guidance_embeds = guidance_embeds
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers  # double blocks
        self.num_single_layers = num_single_layers  # single blocks
        self.patch_size = patch_size
        self.pooled_projection_dim = pooled_projection_dim

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in FluxConfig
        output['attention_head_dim'] = self.attention_head_dim
        output['guidance_embeds'] = self.guidance_embeds
        output['in_channels'] = self.in_channels
        output['joint_attention_dim'] = self.joint_attention_dim
        output['num_attention_heads'] = self.num_attention_heads
        output['num_layers'] = self.num_layers
        output['num_single_layers'] = self.num_single_layers
        output['patch_size'] = self.patch_size
        output['pooled_projection_dim'] = self.pooled_projection_dim
        return output

    @classmethod
    def from_hugging_face(cls,
                          hf_config_path: str,
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          quant_config: Optional[QuantConfig] = None,
                          **kwargs) -> "FluxConfig":
        from diffusers import FluxTransformer2DModel
        hf_config = FluxTransformer2DModel.load_config(hf_config_path)

        attention_head_dim = hf_config['attention_head_dim']
        guidance_embeds = hf_config['guidance_embeds']
        in_channels = hf_config['in_channels']
        joint_attention_dim = hf_config['joint_attention_dim']
        num_attention_heads = hf_config['num_attention_heads']
        num_layers = hf_config['num_layers']
        num_single_layers = hf_config['num_single_layers']
        patch_size = hf_config['patch_size']
        pooled_projection_dim = hf_config['pooled_projection_dim']
        dtype = infer_dtype(dtype, hf_config.get('torch_dtype'))

        return cls(architecture='Flux',
                   attention_head_dim=attention_head_dim,
                   guidance_embeds=guidance_embeds,
                   in_channels=in_channels,
                   joint_attention_dim=joint_attention_dim,
                   num_attention_heads=num_attention_heads,
                   num_layers=num_layers,
                   num_single_layers=num_single_layers,
                   patch_size=patch_size,
                   pooled_projection_dim=pooled_projection_dim,
                   dtype=dtype,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
