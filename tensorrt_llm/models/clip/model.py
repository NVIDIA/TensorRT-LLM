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

from ...functional import arange, concat, expand, expand_dims, shape
from ...layers import MLP, BertAttention, Conv2d, Embedding, LayerNorm
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter


# Adapted from https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/clip/modeling_clip.py#L164
class CLIPVisionEmbeddings(Module):

    def __init__(self, image_size, num_channels, patch_size, hidden_size,
                 dtype):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = hidden_size
        self.dtype = dtype

        self.class_embedding = Parameter(shape=[
            self.embed_dim,
        ],
                                         dtype=self.dtype)

        self.patch_embedding = Conv2d(in_channels=self.num_channels,
                                      out_channels=self.embed_dim,
                                      kernel_size=(self.patch_size,
                                                   self.patch_size),
                                      stride=(self.patch_size, self.patch_size),
                                      bias=False,
                                      dtype=self.dtype)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1
        self.position_embedding = Embedding(self.num_positions,
                                            self.embed_dim,
                                            dtype=self.dtype)

    def forward(self, pixel_values):
        batch_size = shape(pixel_values, 0)
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.cast(
                dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = expand_dims(expand_dims(self.class_embedding.value, 0),
                                   0)
        expand_shape = concat(
            [batch_size,
             shape(class_embeds, -2),
             shape(class_embeds, -1)])
        class_embeds = expand(class_embeds,
                              expand_shape)  # shape = [*, 1, grid, grid]
        embeddings = concat([class_embeds, patch_embeds],
                            dim=1)  # shape = [*, width + 1, grid, grid]
        position_ids = arange(0, self.num_positions, dtype='int32')
        position_embeds = self.position_embedding(position_ids)
        position_embeds = expand_dims(position_embeds, 0)
        expand_shape = concat([
            batch_size,
            shape(position_embeds, -2),
            shape(position_embeds, -1)
        ])
        position_embeds = expand(
            position_embeds, expand_shape)  # shape = [*, width + 1, grid, grid]
        embeddings = embeddings + position_embeds
        return embeddings


class CLIPEncoderLayer(Module):

    def __init__(self, hidden_size, num_attention_heads,
                 max_position_embeddings, norm_epsilon, intermediate_size,
                 hidden_act, mapping: Mapping, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.mapping = mapping

        self.input_layernorm = LayerNorm(normalized_shape=self.hidden_size,
                                         eps=norm_epsilon,
                                         dtype=self.dtype)

        self.attention = BertAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            attention_head_size=self.hidden_size // num_attention_heads,
            num_kv_heads=num_attention_heads,
            dtype=self.dtype,
            tp_group=self.mapping.tp_group,
            tp_size=self.mapping.tp_size,
            tp_rank=self.mapping.tp_rank,
            cp_group=self.mapping.cp_group,
            cp_size=self.mapping.cp_size)

        self.post_layernorm = LayerNorm(normalized_shape=self.hidden_size,
                                        eps=norm_epsilon,
                                        dtype=self.dtype)

        self.mlp = MLP(hidden_size=self.hidden_size,
                       ffn_hidden_size=intermediate_size,
                       hidden_act=hidden_act,
                       dtype=self.dtype,
                       tp_group=self.mapping.tp_group,
                       tp_size=self.mapping.tp_size)

    def forward(self, hidden_states):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPEncoder(Module):

    def __init__(self, hidden_size, num_attention_heads,
                 max_position_embeddings, norm_epsilon, intermediate_size,
                 hidden_act, num_hidden_layers, mapping: Mapping, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.mapping = mapping

        self.layers = ModuleList([
            CLIPEncoderLayer(hidden_size=self.hidden_size,
                             num_attention_heads=num_attention_heads,
                             max_position_embeddings=max_position_embeddings,
                             norm_epsilon=norm_epsilon,
                             intermediate_size=intermediate_size,
                             hidden_act=hidden_act,
                             mapping=self.mapping,
                             dtype=self.dtype) for _ in range(num_hidden_layers)
        ])

    def forward(self, inputs_embeds):

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class CLIPVisionTransformer(Module):

    def __init__(self, image_size, num_channels, patch_size, hidden_size,
                 num_attention_heads, max_position_embeddings, norm_epsilon,
                 intermediate_size, hidden_act, num_hidden_layers, require_ln_f,
                 mapping: Mapping, dtype) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.mapping = mapping

        self.embeddings = CLIPVisionEmbeddings(image_size=image_size,
                                               num_channels=num_channels,
                                               patch_size=patch_size,
                                               hidden_size=hidden_size,
                                               dtype=self.dtype)
        self.pre_layernorm = LayerNorm(normalized_shape=self.hidden_size,
                                       eps=norm_epsilon,
                                       dtype=self.dtype)

        self.encoder = CLIPEncoder(
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            norm_epsilon=norm_epsilon,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
            mapping=self.mapping,
            dtype=self.dtype)

        self.ln_f = None

        if require_ln_f:
            self.ln_f = LayerNorm(normalized_shape=self.hidden_size,
                                  eps=norm_epsilon,
                                  dtype=self.dtype)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        hidden_states = self.encoder(inputs_embeds=hidden_states)

        if self.ln_f is None:
            return hidden_states

        return self.ln_f(hidden_states)
