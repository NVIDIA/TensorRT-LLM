# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, Optional, Sequence

from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class STDiTModelConfig(PretrainedConfig):

    def __init__(self,
                 architecture: str = 'STDiT3',
                 checkpoint_path: str = 'pretrained_ckpt/model.safetensors',
                 vae_type: str = "hpcai-tech/OpenSora-VAE-v1.2",
                 text_encoder_type: str = "DeepFloyd/t5-v1_1-xxl",
                 caption_channels: int = 4096,
                 num_hidden_layers: int = 28,
                 hidden_size: int = 1152,
                 width: int = 640,
                 height: int = 360,
                 num_frames: int = 102,
                 latent_size: Sequence[int] = [30, 45, 80],
                 stdit_patch_size: Sequence[int] = [1, 2, 2],
                 spatial_patch_size: Sequence[int] = [1, 8, 8],
                 temporal_patch_size: Sequence[int] = [4, 1, 1],
                 in_channels: int = 4,
                 input_sq_size: int = 512,
                 num_attention_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 class_dropout_prob: float = 0.1,
                 model_max_length: int = 300,
                 learn_sigma: bool = True,
                 qk_norm: bool = True,
                 skip_y_embedder: bool = False,
                 dtype: Optional[str] = None,
                 mapping: Mapping = Mapping(),
                 quant_config: Optional[QuantConfig] = None,
                 **kwargs):
        kwargs.update({
            'architecture': architecture,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'hidden_size': hidden_size,
            'dtype': dtype
        })

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.vae_type = vae_type
        self.text_encoder_type = text_encoder_type
        self.caption_channels = caption_channels
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.latent_size = latent_size
        self.stdit_patch_size = stdit_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.input_sq_size = input_sq_size
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.model_max_length = model_max_length
        self.learn_sigma = learn_sigma
        self.qk_norm = qk_norm
        self.skip_y_embedder = skip_y_embedder
        self.mapping = mapping
        self.quant_config = quant_config

    @classmethod
    def from_input_config(cls,
                          input_config: Dict[str, Any],
                          dtype: str = 'auto',
                          mapping: Mapping = Mapping(),
                          quant_config: Optional[QuantConfig] = None,
                          **kwargs):
        return cls(architecture=input_config['architecture'],
                   checkpoint_path=input_config['checkpoint_path'],
                   vae_type=input_config['vae_type'],
                   text_encoder_type=input_config['text_encoder_type'],
                   caption_channels=input_config['caption_channels'],
                   num_hidden_layers=input_config['num_hidden_layers'],
                   width=input_config['width'],
                   height=input_config['height'],
                   num_frames=input_config['num_frames'],
                   latent_size=input_config['latent_size'],
                   hidden_size=input_config['hidden_size'],
                   stdit_patch_size=input_config['stdit_patch_size'],
                   spatial_patch_size=input_config['spatial_patch_size'],
                   temporal_patch_size=input_config['temporal_patch_size'],
                   in_channels=input_config['in_channels'],
                   input_sq_size=input_config['input_sq_size'],
                   num_attention_heads=input_config['num_attention_heads'],
                   mlp_ratio=input_config['mlp_ratio'],
                   class_dropout_prob=input_config['class_dropout_prob'],
                   model_max_length=input_config['model_max_length'],
                   learn_sigma=input_config['learn_sigma'],
                   qk_norm=input_config['qk_norm'],
                   skip_y_embedder=input_config['skip_y_embedder'],
                   dtype=dtype,
                   mapping=mapping,
                   quant_config=quant_config,
                   **kwargs)
