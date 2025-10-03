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

import re
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, PretrainedConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.mamba.mamba2_mixer import Mamba2Mixer
from ..modules.mlp import MLP
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


def split(x: torch.Tensor,
          tp_size: int,
          idx: int,
          dim: int = 0) -> torch.Tensor:
    assert x.shape[dim] % tp_size == 0
    split_size = x.shape[dim] // tp_size
    if tp_size == 1:
        return x
    return torch.split(x, split_size, dim=dim)[idx]


def relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


class NemotronHConfig(PretrainedConfig):
    model_type = "nemotron_h"


class MLPLayer(MLP):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[layer_idx]
        else:
            intermediate_size = config.intermediate_size

        super().__init__(hidden_size=config.hidden_size,
                         intermediate_size=intermediate_size,
                         bias=False,
                         activation=relu2,
                         dtype=config.torch_dtype,
                         config=model_config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(hidden_states)


class TransformerLayer(Attention):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(position_ids=None,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata)


class NemotronHLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        # M -> MambaLayer
        # - -> MLPLayer
        # * -> TransformerLayer
        layer_type: str,
    ):
        super().__init__()

        config = model_config.pretrained_config

        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        if layer_type == "M":
            self.mixer = Mamba2Mixer(d_model=config.hidden_size,
                                     d_state=config.ssm_state_size,
                                     d_conv=config.conv_kernel,
                                     nheads=config.mamba_num_heads,
                                     n_groups=config.n_groups,
                                     head_dim=config.mamba_head_dim,
                                     chunk_size=config.chunk_size,
                                     layer_idx=layer_idx,
                                     rms_norm_eps=config.rms_norm_eps,
                                     dtype=config.torch_dtype,
                                     config=model_config)
        elif layer_type == "-":
            self.mixer = MLPLayer(model_config, layer_idx)
        elif layer_type == "*":
            self.mixer = TransformerLayer(model_config, layer_idx)
        else:
            ValueError(f"{layer_type} is not supported")

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, attn_metadata, **kwargs)
        hidden_states = torch.add(hidden_states, residual)

        return hidden_states


class NemotronHModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[NemotronHConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        # calculate embeddings
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        # create layers
        layers = []
        for layer_idx, layer_type in enumerate(config.hybrid_override_pattern):
            layers.append(NemotronHLayer(model_config, layer_idx, layer_type))
        self.layers = nn.ModuleList(layers)

        # final norm
        self.norm_f = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        self.mamba_metadata: Optional[Mamba2Metadata] = None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.mamba_metadata is None or self.mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            self.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests,
                chunk_size=self.model_config.pretrained_config.chunk_size)
        self.mamba_metadata.prepare(attn_metadata)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(position_ids,
                                  hidden_states,
                                  attn_metadata,
                                  mamba_metadata=self.mamba_metadata)

        hidden_states = self.norm_f(hidden_states)

        return hidden_states


@register_auto_model("NemotronHForCausalLM")
class NemotronHForCausalLM(DecoderModelForCausalLM[NemotronHModel,
                                                   NemotronHConfig]):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
    ):
        if not model_config.mapping.tp_size in [1, 2, 4, 8]:
            raise ValueError("TP has to be either 1, 2, 4 or 8")

        if model_config.quant_config.exclude_modules is not None:
            model_config.quant_config.exclude_modules = [
                re.sub(r'(model\.layers\.)?backbone', 'model', k)
                for k in model_config.quant_config.exclude_modules
            ]

        super().__init__(
            NemotronHModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: dict, weight_mapper: BaseWeightMapper):
        new_weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(new_weights, weight_mapper)


AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)
