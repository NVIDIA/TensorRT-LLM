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

import numpy as np

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import (Tensor, concat, constant, expand, op_and, recv, send,
                           shape, slice, unsqueeze, where)
from ...layers import (AttentionMaskType, CogVLMAttention, ColumnLinear,
                       Embedding, GatedMLP, PromptTuningEmbedding, RmsNorm)
from ...mapping import Mapping
from ...module import Module
# this is to use to module global algo string with a quant_algo prefix
from ...quantization import QuantMode
from ...top_model_mixin import TopModelMixin
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import CogVLMConfig


class CogvlmDecoderLayer(Module):

    def __init__(self, config: CogVLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = CogVLMAttention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
            quant_mode=config.quant_mode)

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        self.hidden_size = config.hidden_size
        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            ffn_hidden_size=mlp_hidden_size,
                            hidden_act=config.hidden_act,
                            dtype=config.dtype,
                            bias=config.mlp_bias,
                            tp_group=config.mapping.tp_group,
                            tp_size=config.mapping.tp_size,
                            quant_mode=config.quant_mode)
        self.vis_mlp = GatedMLP(hidden_size=config.hidden_size,
                                ffn_hidden_size=mlp_hidden_size,
                                hidden_act=config.hidden_act,
                                dtype=config.dtype,
                                bias=config.mlp_bias,
                                tp_group=config.mapping.tp_group,
                                tp_size=config.mapping.tp_size,
                                quant_mode=config.quant_mode)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
        vision_token_mask=None,
        position_ids=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          vision_token_mask=vision_token_mask,
                                          position_embedding=position_ids)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        vision_mlp_out = self.vis_mlp(hidden_states)
        language_mlp_out = self.mlp(hidden_states)
        hidden_states = where(vision_token_mask, vision_mlp_out,
                              language_mlp_out)

        # hidden_states = self.mlp(hidden_states,
        #                          lora_layer_params=lora_layer_params)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class CogvlmModel(Module):

    def __init__(self, config: CogVLMConfig) -> None:
        super().__init__()

        self.mapping = config.mapping
        self.use_prompt_tuning = config.use_prompt_tuning
        self.vocab_size = config.vocab_size
        EmbeddingCls = PromptTuningEmbedding if config.use_prompt_tuning else Embedding
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = EmbeddingCls(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                dtype=config.dtype,
                tp_size=self.mapping.tp_size
                if config.use_parallel_embedding else 1,
                tp_group=self.mapping.tp_group
                if config.use_parallel_embedding else None,
                sharding_dim=config.embedding_sharding_dim,
                tp_rank=self.mapping.tp_rank,
            )

        self.layers = DecoderLayerList(CogvlmDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if self.use_prompt_tuning else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        vision_mask = input_ids > (self.vocab_size - 1)

        if default_net().plugin_config.remove_input_padding:
            seq_length = shape(vision_mask, 0)  # lvvvvvvllvvvvlll
            zero = constant(np.ascontiguousarray(np.zeros([1], dtype=bool)))
            one = constant(np.ascontiguousarray(np.ones([1], dtype=bool)))

            t1 = slice(vision_mask, [0], seq_length - 1)
            t2 = slice(vision_mask, [1], seq_length - 1)
            vision_token_mask = concat([op_and(t1 == one, t2 == one),
                                        zero])  # 0111110001110000
            vision_token_mask = unsqueeze(vision_token_mask,
                                          -1)  # [num_tokens, 1]
        else:
            seq_length = shape(vision_mask,
                               1)  # lvvvvvvllvvvvlll, lvvvvvvllvvvvlll
            batch_size = shape(vision_mask, 0)
            t1 = slice(vision_mask, [0, 0], concat([batch_size,
                                                    seq_length - 1]))
            t2 = slice(vision_mask, [0, 1], concat([batch_size,
                                                    seq_length - 1]))
            zero = expand(
                constant(np.ascontiguousarray(np.zeros([1, 1], dtype=bool))),
                concat([batch_size, 1]))
            one = constant(np.ascontiguousarray(np.ones([1, 1], dtype=bool)))

            vision_token_mask = concat([op_and(t1 == one, t2 == one), zero],
                                       dim=1)  # 0111110001110000 [bs, seqlen]
            vision_token_mask = unsqueeze(vision_token_mask,
                                          -1)  # [bs, seqlen, 1]

        hidden_states = self.layers.forward(hidden_states,
                                            use_cache=use_cache,
                                            attention_mask=attention_mask,
                                            kv_cache_params=kv_cache_params,
                                            attention_params=attention_params,
                                            lora_params=lora_params,
                                            vision_token_mask=vision_token_mask,
                                            position_ids=position_ids)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class CogVLMForCausalLM(DecoderModelForCausalLM, TopModelMixin):
    config_class = CogVLMConfig

    def __init__(self, config: CogVLMConfig):
        transformer = CogvlmModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir,
                          dtype='float16',
                          mapping: Optional[Mapping] = None,
                          quant_mode: Optional[QuantMode] = None,
                          **kwargs):
        pass

    def default_plugin_config(self, **kwargs):
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.weight_only_groupwise_quant_matmul_plugin = 'auto'
        return plugin_config

    @classmethod
    def quantize(
        cls,
        hf_model_dir,
        output_dir,
        quant_config: QuantConfig,
        *,
        dtype='float16',
        mapping: Optional[Mapping] = None,
        calib_batches=512,
        calib_batch_size=1,
        random_seed=1234,
        tokenizer_max_seq_length=2048,
        **kwargs,
    ):
        pass
