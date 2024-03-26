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

from ..._utils import pad_vocab_size
from ...functional import Tensor, is_gated_activation, non_gated_version
from ...layers import (MLP, MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, LayerNorm, MoeConfig,
                       PositionEmbeddingType)
from ...lora_manager import LoraBuildConfig, use_lora
from ...module import Module
from ...quantization import QuantMode
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)


def MLPFactory(hidden_size,
               ffn_hidden_size,
               hidden_act,
               bias=True,
               dtype=None,
               moe_config: MoeConfig = MoeConfig(),
               tp_group=None,
               tp_size=1,
               tp_rank=0,
               quant_mode=QuantMode(0)):
    if moe_config.has_moe():
        return MOE(moe_config,
                   hidden_size,
                   ffn_hidden_size,
                   hidden_act,
                   bias,
                   dtype,
                   tp_group,
                   tp_size,
                   tp_rank,
                   quant_mode=quant_mode)
    MLPClass = GatedMLP if is_gated_activation(hidden_act) else MLP
    hidden_act = non_gated_version(hidden_act)
    return MLPClass(
        hidden_size,
        ffn_hidden_size,
        hidden_act,
        bias,
        dtype,
        tp_group,
        tp_size,
        quant_mode,
    )


class GPTDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank

        self.input_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                         eps=config.norm_epsilon,
                                         dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_percentage=config.rotary_pct,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            bias=config.bias,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_mode=config.quant_mode)

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        moe_config = MoeConfig()
        if config.moe_num_experts > 1:
            moe_config = MoeConfig(
                config.moe_num_experts,
                config.moe_top_k,
                config.moe_tp_mode,
                config.moe_normalization_mode,
            )
        self.mlp = MLPFactory(hidden_size=config.hidden_size,
                              ffn_hidden_size=mlp_hidden_size,
                              hidden_act=config.hidden_act,
                              dtype=config.dtype,
                              bias=config.bias,
                              moe_config=moe_config,
                              tp_group=tp_group,
                              tp_size=tp_size,
                              tp_rank=tp_rank,
                              quant_mode=config.quant_mode)

        self.post_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                        eps=config.norm_epsilon,
                                        dtype=config.dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None):

        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.mapping = config.mapping
        self.position_embedding_type = config.position_embedding_type

        self.vocab_embedding = Embedding(config.vocab_size,
                                         config.hidden_size,
                                         dtype=config.dtype)

        if config.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.position_embedding = Embedding(
                num_embeddings=config.max_position_embeddings,
                embedding_dim=config.hidden_size,
                dtype=config.dtype)
        self.layers = DecoderLayerList(GPTDecoderLayer, config)

        self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                              eps=config.norm_epsilon,
                              dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None,
                lora_params=None):
        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []
        hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            hidden_states = hidden_states + self.position_embedding(
                position_ids)

        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params,
                                    lora_params=lora_params)
        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = GPTModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        lm_head = ColumnLinear(config.hidden_size,
                               vocab_size_padded,
                               bias=False,
                               dtype=config.dtype,
                               tp_group=config.mapping.tp_group,
                               tp_size=config.mapping.tp_size,
                               gather_output=True)
        super().__init__(config, transformer, lm_head)

    def check_config(self, config: PretrainedConfig):
        config.set_if_not_exist('bias', True)
        config.set_if_not_exist('apply_query_key_layer_scaling', False)
        config.set_if_not_exist('rotary_pct', 1.0)
        config.set_if_not_exist('rotary_base', 10000.0)
        config.set_if_not_exist('rotary_scaling', None)
        config.set_if_not_exist('moe_num_experts', 0)
        config.set_if_not_exist('moe_top_k', 0)
        config.set_if_not_exist('moe_tp_mode',
                                MoeConfig.ParallelismMode.TENSOR_PARALLEL)
        config.set_if_not_exist(
            'moe_normalization_mode',
            MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE)

    def use_lora(self, lora_config: LoraBuildConfig):
        use_lora(self, lora_config)
