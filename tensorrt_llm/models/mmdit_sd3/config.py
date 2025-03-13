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
from typing import Any, Dict, Optional, Sequence, Tuple

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class SD3Transformer2DModelConfig(PretrainedConfig):

    def __init__(
            self,
            *,
            sample_size: int = 128,
            patch_size: int = 2,
            in_channels: int = 16,
            num_layers: int = 24,
            attention_head_dim: int = 64,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 4096,
            caption_projection_dim: int = 1536,
            pooled_projection_dim: int = 2048,
            out_channels: int = 16,
            pos_embed_max_size: int = 384,
            dual_attention_layers:
        Tuple[int] = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        ),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
            qk_norm: Optional[str] = None,
            skip_layers: Optional[Sequence[int]] = None,
            use_pretrained_pos_emb: bool = False,
            **kwargs):

        kwargs.update({
            'hidden_size': attention_head_dim * num_attention_heads,
            'num_hidden_layers': num_layers,
            'num_attention_heads': num_attention_heads
        })
        super().__init__(**kwargs)
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.caption_projection_dim = caption_projection_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.dual_attention_layers = dual_attention_layers
        self.qk_norm = qk_norm
        self.skip_layers = skip_layers
        self.use_pretrained_pos_emb = use_pretrained_pos_emb

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in SD3Transformer2DModelConfig
        output['sample_size'] = self.sample_size
        output['patch_size'] = self.patch_size
        output['in_channels'] = self.in_channels
        output['num_layers'] = self.num_layers
        output['attention_head_dim'] = self.attention_head_dim
        output['num_attention_heads'] = self.num_attention_heads
        output['joint_attention_dim'] = self.joint_attention_dim
        output['caption_projection_dim'] = self.caption_projection_dim
        output['pooled_projection_dim'] = self.pooled_projection_dim
        output['out_channels'] = self.out_channels
        output['pos_embed_max_size'] = self.pos_embed_max_size
        output['dual_attention_layers'] = self.dual_attention_layers
        output['qk_norm'] = self.qk_norm
        output['skip_layers'] = self.skip_layers
        output['use_pretrained_pos_emb'] = self.use_pretrained_pos_emb
        return output

    @classmethod
    def from_hugging_face_config(cls,
                                 hf_config: Dict[str, Any],
                                 dtype: str = 'auto',
                                 mapping: Optional[Mapping] = None,
                                 quant_config: Optional[QuantConfig] = None,
                                 **kwargs):
        sample_size = hf_config['sample_size']
        patch_size = hf_config['patch_size']
        in_channels = hf_config['in_channels']
        num_layers = hf_config['num_layers']
        attention_head_dim = hf_config['attention_head_dim']
        num_attention_heads = hf_config['num_attention_heads']
        joint_attention_dim = hf_config['joint_attention_dim']
        caption_projection_dim = hf_config['caption_projection_dim']
        pooled_projection_dim = hf_config['pooled_projection_dim']
        out_channels = hf_config['out_channels']
        pos_embed_max_size = hf_config['pos_embed_max_size']
        dual_attention_layers = hf_config['dual_attention_layers']
        qk_norm = hf_config['qk_norm']
        skip_layers = None
        use_pretrained_pos_emb = kwargs.get('use_pretrained_pos_emb', False)
        dtype = infer_dtype(dtype, hf_config.get('torch_dtype'))

        return cls(architecture='SD3Transformer2DModel',
                   sample_size=sample_size,
                   patch_size=patch_size,
                   in_channels=in_channels,
                   num_layers=num_layers,
                   attention_head_dim=attention_head_dim,
                   num_attention_heads=num_attention_heads,
                   joint_attention_dim=joint_attention_dim,
                   caption_projection_dim=caption_projection_dim,
                   pooled_projection_dim=pooled_projection_dim,
                   out_channels=out_channels,
                   pos_embed_max_size=pos_embed_max_size,
                   dual_attention_layers=dual_attention_layers,
                   qk_norm=qk_norm,
                   skip_layers=skip_layers,
                   use_pretrained_pos_emb=use_pretrained_pos_emb,
                   dtype=dtype,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
