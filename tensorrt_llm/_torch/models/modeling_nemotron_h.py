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
from transformers.activations import ACT2FN

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


class NemotronHMOE(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                NemotronHMLP(config, intermediate_size=config.moe_intermediate_size, layer_idx=layer_idx)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(
            config=config, intermediate_size=config.moe_shared_expert_intermediate_size, layer_idx=layer_idx
        )
        self.dtype = config.torch_dtype

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states, attn_metadata, **kwargs):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class NemotronHMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.layer_idx = layer_idx
        # if layer_idx is None:
        #     logger.warning_once(
        #         f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
        #         "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=self.dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias, dtype=self.dtype)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class NemotronHTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.dtype = config.torch_dtype

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()

        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor

        topk_weights = topk_weights.to(self.dtype)
        return topk_indices, topk_weights


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
        elif layer_type == "E":
            self.mixer = NemotronHMOE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"{layer_type} is not supported")

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
        # rms_norm_eps is with a different name in the config.
        if hasattr(model_config.pretrained_config, "layer_norm_epsilon"):
            rms_epsilon = model_config.pretrained_config.layer_norm_epsilon
        elif hasattr(model_config.pretrained_config, "rms_norm_eps"):
            rms_epsilon = model_config.pretrained_config.rms_norm_eps
        else:
            raise ValueError("layer_norm_epsilon or rms_norm_eps is not set")
        model_config.pretrained_config.rms_norm_eps = rms_epsilon

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
