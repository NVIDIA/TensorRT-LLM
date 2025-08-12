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
import math
from collections import OrderedDict
from typing import List, Optional

import tensorrt as trt
import torch

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import (numpy_to_torch, pad_vocab_size,
                                 str_dtype_to_torch)
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     MLPType, PositionEmbeddingType, Tensor,
                                     assertion, cast, gather_last_token_logits,
                                     gelu, maximum, minimum, recv, send, shape,
                                     transpose, unsqueeze)
# yapf: disable
from tensorrt_llm.layers import (MLP, Attention, AttentionMaskParams,
                                 AttentionMaskType, AttentionParams,
                                 BertAttention, ColumnLinear, Conv1d, Embedding,
                                 FusedGatedMLP, GatedMLP, GroupNorm,
                                 KeyValueCacheParams, LanguageAdapter,
                                 LanguageAdapterConfig, LayerNorm, LoraParams,
                                 PromptTuningEmbedding, RmsNorm)
# yapf: enable
from tensorrt_llm.lora_helper import (LoraConfig,
                                      get_default_trtllm_modules_to_hf_modules,
                                      use_lora)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import PretrainedConfig, PretrainedModel
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.plugin.plugin import current_all_reduce_helper
from tensorrt_llm.quantization import QuantMode

layernorm_map = {
    LayerNormType.LayerNorm: LayerNorm,
    LayerNormType.RmsNorm: RmsNorm,
    LayerNormType.GroupNorm: GroupNorm,
}

mlp_map = {
    MLPType.MLP: MLP,
    MLPType.GatedMLP: GatedMLP,
    MLPType.FusedGatedMLP: FusedGatedMLP,
}


class EncDecTransformer(Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 has_vocab_embeddings=True,
                 max_position_embeddings=None,
                 has_position_embedding=False,
                 type_vocab_size=None,
                 has_embedding_layernorm=False,
                 has_embedding_scale=False,
                 layernorm_eps=1e-5,
                 layernorm_type=LayerNormType.LayerNorm,
                 dtype=None,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 mapping=Mapping(),
                 has_model_final_layernorm=False,
                 norm_epsilon=1e-05,
                 is_decoder=False):
        super().__init__()

        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]
        self.mapping = mapping

        if self.mapping.is_first_pp_rank():
            if has_vocab_embeddings:
                self.vocab_embedding = Embedding(
                    vocab_size,
                    hidden_size,
                    dtype=dtype,
                    tp_size=mapping.tp_size if use_parallel_embedding else 1,
                    tp_group=mapping.tp_group
                    if use_parallel_embedding else None,
                    sharding_dim=embedding_sharding_dim,
                    tp_rank=mapping.tp_rank)

            self.position_embedding = None
            self.max_position_embeddings = max_position_embeddings
            if has_position_embedding:
                self.position_embedding = Embedding(
                    max_position_embeddings,
                    hidden_size,
                    dtype=dtype,
                    tp_size=mapping.tp_size if use_parallel_embedding else 1,
                    tp_group=mapping.tp_group
                    if use_parallel_embedding else None,
                    sharding_dim=embedding_sharding_dim,
                    tp_rank=mapping.tp_rank)

            self.token_type_embedding = None
            if type_vocab_size:
                self.token_type_embedding = Embedding(
                    type_vocab_size,
                    hidden_size,
                    dtype=dtype,
                    tp_size=mapping.tp_size if use_parallel_embedding else 1,
                    tp_group=mapping.tp_group
                    if use_parallel_embedding else None,
                    sharding_dim=embedding_sharding_dim,
                    tp_rank=mapping.tp_rank)

            # e.g. BART true, T5 false
            self.ln_embed = None
            if has_embedding_layernorm:
                self.ln_embed = ln_type(normalized_shape=hidden_size,
                                        eps=layernorm_eps,
                                        dtype=dtype)

            # e.g. BART true, T5 false
            self.embedding_scale = 1.0
            if has_embedding_scale:
                self.embedding_scale = math.sqrt(hidden_size)

            # Note: embedding offset in BART is not considered as a standard. For the specific case,
            # we just need to shrink its position embedding table by [offset:] during weight loading

        self.layers = None
        self.is_decoder = is_decoder
        self.has_model_final_layernorm = has_model_final_layernorm
        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                self.ln_f = ln_type(normalized_shape=hidden_size,
                                    eps=norm_epsilon,
                                    dtype=dtype)

    def assign_module(self, module, module_name):
        setattr(self, module_name, module)

    def embedding(self,
                  input_ids,
                  position_ids=None,
                  token_type_ids=None,
                  prompt_embedding_table=None,
                  prompt_tasks=None,
                  prompt_vocab_size=None):
        # position_ids and token_type_ids are provided inputs
        # and should not be formulated deterministically

        args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size
                ] if prompt_embedding_table is not None else []

        x = self.vocab_embedding(input_ids, *args) * self.embedding_scale
        self.register_network_output('word_embeddings', x)

        if self.position_embedding:
            pos_emb = self.position_embedding(position_ids)
            self.register_network_output('position_embeddings', pos_emb)
            x = x + pos_emb
        if self.token_type_embedding:
            x = x + self.token_type_embedding(token_type_ids)

        if self.ln_embed:
            x = self.ln_embed(x)

        return x


class EncoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 num_attention_heads,
                 num_kv_heads,
                 head_size,
                 max_position_embeddings=None,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 layernorm_eps=1e-5,
                 hidden_act="relu",
                 mlp_type=MLPType.MLP,
                 mapping=Mapping(),
                 dtype=None,
                 residual_scaling=1.0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0,
                 fp16_clamping=False,
                 quant_mode=QuantMode(0),
                 language_adapter_config: LanguageAdapterConfig = None):
        super().__init__()

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART post, T5 pre
        self.layernorm_position = layernorm_position

        # e.g. BART q_scaling = 1.f, T5 q_scaling = 1.f/sqrt(head_size)
        self.attention = BertAttention(
            hidden_size,
            num_attention_heads,
            attention_head_size=head_size,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            dtype=dtype,
            relative_attention=relative_attention,
            max_distance=max_distance,
            num_buckets=num_buckets,
            quant_mode=quant_mode)

        self.attention_layernorm = ln_type(normalized_shape=hidden_size,
                                           eps=layernorm_eps,
                                           dtype=dtype)

        # T5/BART MLP, Flan-T5 GatedMLP
        self.mlp_type = mlp_type
        mlp_f = mlp_map[mlp_type]
        self.mlp = mlp_f(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            bias=has_mlp_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            dtype=dtype,
            quant_mode=quant_mode,
        )

        self.mlp_layernorm = ln_type(normalized_shape=hidden_size,
                                     eps=layernorm_eps,
                                     dtype=dtype)

        self.residual_scaling = residual_scaling

        # T5-series model(e.g. t5-large, t5-3b, flan-t5-small) has accuracy issue due to fp16 overflow
        # after residual add. We add workaround for clamping fp16 range [-64000, 64000] after every
        # residual add to avoid accuracy drop.
        self.fp16_clamping = fp16_clamping

        self.adapter = None
        self.adapter_layer_norm = None

        if language_adapter_config:
            self.adapter_layer_norm = ln_type(normalized_shape=hidden_size,
                                              eps=layernorm_eps,
                                              dtype=dtype)
            self.adapter = LanguageAdapter(
                language_adapter_config=language_adapter_config,
                hidden_size=hidden_size,
                hidden_act=hidden_act,
                mapping=mapping,
                has_mlp_bias=has_mlp_bias,
                dtype=dtype,
                quant_mode=quant_mode)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                max_input_length=None,
                lora_layer_params=None,
                language_adapter_routings: Optional[Tensor] = None):
        assert isinstance(hidden_states, Tensor)

        # self attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          input_lengths=input_lengths,
                                          max_input_length=max_input_length,
                                          lora_layer_params=lora_layer_params)

        self.register_network_output('attention_output', attention_output)

        hidden_states = residual + attention_output

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)

        # MLP
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        self.register_network_output('mlp_output', hidden_states)

        hidden_states = residual + hidden_states

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        # MT Specific: adapters
        if self.adapter:
            residual = hidden_states
            hidden_states = self.adapter_layer_norm(hidden_states)
            hidden_states = self.adapter.layers(
                hidden_states, static_routing_input=language_adapter_routings)
            hidden_states = residual + hidden_states
            if self.fp16_clamping:
                hidden_states = maximum(-64000.0, hidden_states)
                hidden_states = minimum(64000.0, hidden_states)

        return hidden_states


class DecoderLayer(Module):

    def __init__(self,
                 *,
                 local_layer_idx,
                 hidden_size,
                 ffn_hidden_size,
                 num_attention_heads,
                 num_kv_heads,
                 head_size,
                 max_position_embeddings=None,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 layernorm_eps=1e-5,
                 hidden_act="relu",
                 mlp_type=MLPType.MLP,
                 mapping=Mapping(),
                 dtype=None,
                 residual_scaling=1.0,
                 relative_attention=False,
                 max_distance=0,
                 num_buckets=0,
                 fp16_clamping=False,
                 skip_cross_kv=False,
                 use_implicit_relative_attention=False,
                 quant_mode=QuantMode(0),
                 language_adapter_config: LanguageAdapterConfig = None):
        super().__init__()

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART post, T5 pre
        self.layernorm_position = layernorm_position

        # e.g. BART q_scaling = 1.f, T5 q_scaling = 1.f/sqrt(head_size)
        self.self_attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_size=head_size,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            attention_mask_type=AttentionMaskType.causal,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            dtype=dtype,
            cross_attention=False,
            relative_attention=relative_attention,
            max_distance=max_distance if use_implicit_relative_attention else 0,
            num_buckets=num_buckets,
            position_embedding_type=PositionEmbeddingType.relative
            if relative_attention else PositionEmbeddingType.learned_absolute,
            use_implicit_relative_attention=use_implicit_relative_attention,
            quant_mode=quant_mode)

        self.self_attention_layernorm = ln_type(normalized_shape=hidden_size,
                                                eps=layernorm_eps,
                                                dtype=dtype)

        # Note: self attn uses MMHA, mask is always causal triangular
        # cross attn has two scenarios:
        # - in context phase, all ones mask, same as padding type
        # - in generation phase, same causal triangular mask as MMHA
        # - context phase special handling is done in plugin by resetting mask type
        #
        # e.g. BART q_scaling = 1.f, T5 q_scaling = 1.f/sqrt(head_size)
        self.cross_attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_size=head_size,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            q_scaling=q_scaling,
            bias=has_attention_qkvo_bias,
            attention_mask_type=AttentionMaskType.causal,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            dtype=dtype,
            cross_attention=True,
            relative_attention=
            False,  # Cross attention has no relative attention bias
            max_distance=max_distance,
            num_buckets=num_buckets,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            skip_cross_kv=skip_cross_kv,
            quant_mode=quant_mode)

        self.cross_attention_layernorm = ln_type(normalized_shape=hidden_size,
                                                 eps=layernorm_eps,
                                                 dtype=dtype)

        # T5/BART MLP, Flan-T5 GatedMLP
        self.mlp_type = mlp_type
        mlp_f = mlp_map[mlp_type]
        self.mlp = mlp_f(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            bias=has_mlp_bias,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            dtype=dtype,
            quant_mode=quant_mode,
        )

        self.mlp_layernorm = ln_type(normalized_shape=hidden_size,
                                     eps=layernorm_eps,
                                     dtype=dtype)

        self.residual_scaling = residual_scaling

        # T5-series model(e.g. t5-large, t5-3b, flan-t5-small) has accuracy issue due to fp16 overflow
        # after residual add. We add workaround for clamping fp16 range [-64000, 64000] after every
        # residual add to avoid accuracy drop.
        self.fp16_clamping = fp16_clamping

        self.adapter = None
        self.adapter_layer_norm = None

        if language_adapter_config:
            self.adapter_layer_norm = ln_type(normalized_shape=hidden_size,
                                              eps=layernorm_eps,
                                              dtype=dtype)
            self.adapter = LanguageAdapter(
                language_adapter_config=language_adapter_config,
                hidden_size=hidden_size,
                hidden_act=hidden_act,
                mapping=mapping,
                has_mlp_bias=has_mlp_bias,
                dtype=dtype,
                quant_mode=quant_mode)

    def forward(self,
                hidden_states: Tensor,
                encoder_output: Optional[Tensor] = None,
                attention_mask_params=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None,
                cross_kv_cache_gen: Optional[Tensor] = None,
                cross_kv_reuse: Optional[Tensor] = None,
                language_adapter_routings: Optional[Tensor] = None):
        assert isinstance(hidden_states, Tensor)

        if encoder_output:
            assert isinstance(encoder_output, Tensor)

        # self-attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        attention_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask_params.self_attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents_self = attention_output

        self.register_network_output('self_attention_output', attention_output)

        hidden_states = residual + attention_output

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        # cross attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.cross_attention_layernorm(hidden_states)

        attention_output = self.cross_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask_params.cross_attention_mask,
            attention_packed_mask=attention_mask_params.
            cross_attention_packed_mask,
            encoder_output=encoder_output,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
            cross_kv_cache_gen=cross_kv_cache_gen,
            cross_kv_reuse=cross_kv_reuse)

        if use_cache:
            attention_output, presents_cross = attention_output

        self.register_network_output('cross_attention_output', attention_output)

        hidden_states = residual + attention_output

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.cross_attention_layernorm(hidden_states)

        # MLP
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)
        self.register_network_output('mlp_output', hidden_states)

        hidden_states = residual + hidden_states

        if self.fp16_clamping:
            hidden_states = maximum(-64000.0, hidden_states)
            hidden_states = minimum(64000.0, hidden_states)

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        # MT Specific: adapters
        if self.adapter:
            residual = hidden_states
            hidden_states = self.adapter_layer_norm(hidden_states)
            hidden_states = self.adapter.layers(
                hidden_states, static_routing_input=language_adapter_routings)
            hidden_states = residual + hidden_states
            if self.fp16_clamping:
                hidden_states = maximum(-64000.0, hidden_states)
                hidden_states = minimum(64000.0, hidden_states)

        if use_cache:
            return (hidden_states, presents_self, presents_cross)
        return hidden_states


class EncoderModel(PretrainedModel):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        super().__init__(config)
        self.mapping = self.config.mapping

        self.has_position_embedding = self.config.has_position_embedding
        type_vocab_size = self.config.type_vocab_size
        self.has_token_type_embedding = False if type_vocab_size is None else True

        # e.g. BART regular, T5 RMS
        self.layernorm_type = self.config.layernorm_type

        # e.g. BART true, T5 false
        self.has_attention_qkvo_bias = self.config.has_attention_qkvo_bias
        self.has_mlp_bias = self.config.has_mlp_bias

        # e.g. BART false, T5 true
        self.has_model_final_layernorm = self.config.has_model_final_layernorm

        self._dtype = self.config.dtype

        self.total_num_layers = self.config.num_hidden_layers
        self.num_layers = self.config.num_hidden_layers // self.mapping.pp_size

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        num_kv_heads = self.num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = self.config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = self.hidden_size // self.num_heads if self.config.head_size is None else self.config.head_size

        self.fp16_clamping = (self.config.dtype
                              == 'float16') and (self.config.model_type == 't5')
        self.mlp_type = MLPType.MLP if not hasattr(
            self.config, "mlp_type") else self.config.mlp_type
        self.language_adapter_config = None if not hasattr(
            self.config,
            'language_adapter_config') else LanguageAdapterConfig.from_dict(
                self.config.language_adapter_config)

        self.transformer = EncDecTransformer(
            self.config.vocab_size,
            self.config.hidden_size,
            max_position_embeddings=self.config.max_position_embeddings,
            has_position_embedding=self.has_position_embedding,
            type_vocab_size=type_vocab_size,
            has_embedding_layernorm=self.config.has_embedding_layernorm,
            has_embedding_scale=self.config.has_embedding_scale,
            layernorm_eps=self.config.norm_epsilon,
            layernorm_type=self.layernorm_type,
            dtype=self.config.dtype,
            use_parallel_embedding=self.config.use_parallel_embedding,
            embedding_sharding_dim=self.config.embedding_sharding_dim,
            mapping=self.mapping,
            has_model_final_layernorm=self.has_model_final_layernorm,
            norm_epsilon=self.config.norm_epsilon,
            is_decoder=False)

        encoder_layers = ModuleList([
            EncoderLayer(
                hidden_size=self.hidden_size,
                ffn_hidden_size=self.config.intermediate_size,
                num_attention_heads=self.num_heads,
                num_kv_heads=num_kv_heads,
                head_size=self.head_size,
                max_position_embeddings=self.config.max_position_embeddings,
                q_scaling=self.config.q_scaling,
                has_attention_qkvo_bias=self.has_attention_qkvo_bias,
                has_mlp_bias=self.has_mlp_bias,
                layernorm_position=self.config.layernorm_position,
                layernorm_eps=self.config.norm_epsilon,
                layernorm_type=self.layernorm_type,
                hidden_act=self.config.hidden_act,
                mlp_type=self.mlp_type,
                mapping=self.mapping,
                dtype=self.config.dtype,
                residual_scaling=1.0
                if not hasattr(self.config, "residual_scaling") else
                self.config.residual_scaling,
                relative_attention=self.config.relative_attention,
                max_distance=self.config.max_distance,
                num_buckets=self.config.num_buckets,
                fp16_clamping=self.fp16_clamping,
                quant_mode=self.config.quant_mode,
                language_adapter_config=self.language_adapter_config)
            for _ in self.mapping.pp_layers(self.total_num_layers)
        ])
        self.transformer.assign_module(encoder_layers, "layers")

    def check_config(self, config: PretrainedConfig):
        config.set_if_not_exist('has_position_embedding', False)
        config.set_if_not_exist('type_vocab_size', None)
        config.set_if_not_exist('rescale_before_lm_head', False)
        config.set_if_not_exist('layernorm_type', LayerNormType.LayerNorm)
        config.set_if_not_exist('layernorm_position',
                                LayerNormPositionType.pre_layernorm)
        config.set_if_not_exist('has_attention_qkvo_bias', False)
        config.set_if_not_exist('has_mlp_bias', False)
        config.set_if_not_exist('has_model_final_layernorm', False)
        config.set_if_not_exist('encoder_hidden_size', None)
        config.set_if_not_exist('encoder_num_heads', None)
        config.set_if_not_exist('encoder_num_kv_heads', None)
        config.set_if_not_exist('encoder_head_size', None)
        config.set_if_not_exist('model_type', 't5')
        config.set_if_not_exist('skip_cross_kv', False)
        config.set_if_not_exist('mlp_type', MLPType.MLP)
        config.set_if_not_exist('has_embedding_scale', False)
        config.set_if_not_exist('residual_scaling', 1.0)
        config.set_if_not_exist('has_lm_head_bias', False)
        config.set_if_not_exist('num_buckets', None)
        config.set_if_not_exist('max_distance', None)
        config.set_if_not_exist('relative_attention', False)
        config.set_if_not_exist('residual_scaling', 1.0)

    def forward(self,
                input_ids: Tensor,
                input_lengths=None,
                position_ids=None,
                token_type_ids=None,
                hidden_states=None,
                max_input_length=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None,
                attention_mask=None,
                lora_params: LoraParams = None,
                language_adapter_routings: Optional[Tensor] = None):
        # In PP, layer 0 has ids as inputs, all other layers have hidden_states as inputs
        if self.mapping.is_first_pp_rank():
            ptuning_args = [
                prompt_embedding_table, prompt_tasks, prompt_vocab_size
            ] if prompt_embedding_table is not None else []

            hidden_states = self.transformer.embedding(input_ids, position_ids,
                                                       token_type_ids,
                                                       *ptuning_args)
            self.register_network_output('embedding_layer_output',
                                         hidden_states)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        for layer_idx, encoder_layer in enumerate(self.transformer.layers):
            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)
            hidden_states = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                max_input_length=max_input_length,
                lora_layer_params=lora_layer_params,
                language_adapter_routings=language_adapter_routings)

        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                hidden_states = self.transformer.ln_f(hidden_states)
            hidden_states.mark_output('encoder_output', self._dtype)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())
            hidden_states.mark_output('hidden_states_output', self._dtype)

        return hidden_states

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       prompt_embedding_table_size: int = 0,
                       lora_target_modules: List[str] = None,
                       *args,
                       **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        hidden_size = self.hidden_size

        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        inlen_range = [1, (max_input_len + 1) // 2, max_input_len]
        num_tokens_range = [
            1,
            (max_input_len * max_batch_size + 1) // 2,
            max_input_len * max_batch_size,
        ]

        input_ids, position_ids, token_type_ids, hidden_states = None, None, None, None
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_lora_plugin = default_net().plugin_config.lora_plugin

        attention_mask = None
        if remove_input_padding:
            if self.mapping.is_first_pp_rank():
                input_ids = Tensor(
                    name="input_ids",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([("num_tokens", [num_tokens_range])]),
                )
                if self.has_position_embedding:
                    position_ids = Tensor(
                        name='position_ids',
                        dtype=trt.int32,
                        shape=[-1],
                        dim_range=OrderedDict([('num_tokens',
                                                [num_tokens_range])]),
                    )
                if self.has_token_type_embedding:
                    token_type_ids = Tensor(
                        name='token_type_ids',
                        dtype=trt.int32,
                        shape=[-1],
                        dim_range=OrderedDict([('num_tokens',
                                                [num_tokens_range])]),
                    )
            else:
                hidden_states = Tensor(name='hidden_states_input',
                                       dtype=self._dtype,
                                       shape=[-1, hidden_size],
                                       dim_range=OrderedDict([
                                           ('num_tokens', [num_tokens_range]),
                                           ('hidden_size', [hidden_size]),
                                       ]))
        else:
            if self.mapping.is_first_pp_rank():
                input_ids = Tensor(
                    name="input_ids",
                    dtype=trt.int32,
                    shape=[-1, -1],
                    dim_range=OrderedDict([("batch_size", [bs_range]),
                                           ("input_len", [inlen_range])]),
                )
                if self.has_position_embedding:
                    position_ids = Tensor(
                        name='position_ids',
                        dtype=trt.int32,
                        shape=[-1, -1],
                        dim_range=OrderedDict([('batch_size', [bs_range]),
                                               ('input_len', [inlen_range])]),
                    )
                if self.has_token_type_embedding:
                    token_type_ids = Tensor(
                        name='token_type_ids',
                        dtype=trt.int32,
                        shape=[-1, -1],
                        dim_range=OrderedDict([('batch_size', [bs_range]),
                                               ('input_len', [inlen_range])]),
                    )
            else:
                hidden_states = Tensor(name='hidden_states_input',
                                       dtype=self._dtype,
                                       shape=[-1, -1, hidden_size],
                                       dim_range=OrderedDict([
                                           ('batch_size', [bs_range]),
                                           ('input_len', [inlen_range]),
                                           ('hidden_size', [hidden_size]),
                                       ]))

            if not default_net().plugin_config.bert_attention_plugin:
                attention_mask = Tensor(
                    name='attention_mask',
                    dtype=trt.int32,
                    shape=[-1, -1],
                    dim_range=OrderedDict([
                        ('batch_size', [bs_range]),
                        ('input_len', [inlen_range]),
                    ]),
                )

        # if self.mapping.tp_size > 1:
        #     current_all_reduce_helper().set_workspace_tensor(self.mapping, 1)
        # FIXME(TRTLLM-996): Support custom allreduce for encoder models on C++ runtime

        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [bs_range])]),
        )
        max_input_length = Tensor(
            name="max_input_length",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("max_input_length", [inlen_range])]),
        )

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None

        if self.mapping.is_first_pp_rank() and prompt_embedding_table_size > 0:
            p_embedding_range = [[
                1, prompt_embedding_table_size // 2, prompt_embedding_table_size
            ]]

            prompt_embedding_table = Tensor(name='prompt_embedding_table',
                                            dtype=self._dtype,
                                            shape=[-1, hidden_size],
                                            dim_range=OrderedDict([
                                                ('prompt_embedding_table_size',
                                                 p_embedding_range),
                                                ('hidden_size', [hidden_size]),
                                            ]))
            if remove_input_padding:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([('input_len_task',
                                                       [num_tokens_range])]))
            else:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1, 1],
                               dim_range=OrderedDict([
                                   ('batch_size', bs_range),
                                   ('broadcast_dim', [1]),
                               ]))
            prompt_vocab_size = Tensor(name='prompt_vocab_size',
                                       dtype=trt.int32,
                                       shape=[1],
                                       dim_range=OrderedDict([('size', [1])]))
        '''
        LoRA plugin related inputs:
        lora_target_modules for BART-encoder:
            ['attn_q', 'attn_v']
        For BART-decoder, the lora_target_modules is different.
        See comments in the DecoderModel.prepare_inputs() for more details.
        '''
        lora_weights_pointers = None
        lora_ranks = None
        lora_params = None
        if use_lora_plugin:
            lora_weights_pointers = []
            lora_ranks = []
            # In current design, q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time.
            # However, BART lora modules only contain two of them, so we use zero tensor to fill the missing ones.
            missing_qkv_modules = []
            if any(x in lora_target_modules
                   for x in ["attn_q", "attn_k", "attn_v"]):
                for lora_module in ["attn_q", "attn_k", "attn_v"]:
                    if lora_module not in lora_target_modules:
                        missing_qkv_modules.append(lora_module)

            layers_range = self.mapping.pp_layers(self.total_num_layers)
            for i in layers_range:
                lora_weight_pointer_dict = {}
                lora_rank_dict = {}
                for lora_module in (lora_target_modules + missing_qkv_modules):
                    lora_weight_pointer = Tensor(
                        name=f'{lora_module}_lora_weights_pointers_{i}',
                        dtype=trt.int64,
                        shape=[-1, 3],
                        dim_range=OrderedDict([('batch_size', [bs_range]),
                                               ('in_out_scales', [3])]))
                    lora_weight_pointer_dict.update({
                        f'{lora_module}_lora_weights_pointers':
                        lora_weight_pointer
                    })

                    lora_rank = Tensor(name=f'{lora_module}_lora_ranks_{i}',
                                       dtype=trt.int32,
                                       shape=[-1],
                                       dim_range=OrderedDict([('batch_size',
                                                               [bs_range])]))
                    lora_rank_dict.update(
                        {f'{lora_module}_lora_ranks': lora_rank})

                lora_weights_pointers.append(lora_weight_pointer_dict)
                lora_ranks.append(lora_rank_dict)

            host_request_types = Tensor(name='host_request_types',
                                        dtype=trt.int32,
                                        shape=[-1],
                                        dim_range=OrderedDict([('batch_size',
                                                                [bs_range])]))

            host_context_lengths = None
            if remove_input_padding:
                host_context_lengths = Tensor(name='host_context_lengths',
                                              dtype=trt.int32,
                                              shape=[-1],
                                              dim_range=OrderedDict([
                                                  ('batch_size', [bs_range])
                                              ]))

            lora_params = LoraParams(
                lora_ranks=lora_ranks,
                lora_weights_pointers=lora_weights_pointers,
                host_request_types=host_request_types,
                host_context_lengths=host_context_lengths,
            )

        language_adapter_routings = None
        if self.language_adapter_config:
            language_adapter_routings = Tensor(
                name="language_adapter_routings",
                dtype=trt.int32,
                shape=[-1, 1],
                dim_range=OrderedDict([('num_tokens', [num_tokens_range]),
                                       ('routing_dim', [1])]),
            )

        result = {
            'input_ids': input_ids,
            'input_lengths': input_lengths,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
            'hidden_states': hidden_states,
            'max_input_length': max_input_length,
            'prompt_embedding_table': prompt_embedding_table,
            'prompt_tasks': tasks,
            'prompt_vocab_size': prompt_vocab_size,
            'attention_mask': attention_mask,
            'lora_params': lora_params,
            'language_adapter_routings': language_adapter_routings,
        }

        return result

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config)

    def use_prompt_tuning(self):
        embedding = self.transformer.vocab_embedding
        self.transformer.vocab_embedding = PromptTuningEmbedding(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            dtype=embedding.dtype,
            tp_size=embedding.tp_size,
            tp_group=embedding.tp_group,
            sharding_dim=embedding.sharding_dim,
            tp_rank=embedding.tp_rank)

        self.transformer.vocab_embedding.weight.value = embedding.weight.raw_value

    def precompute_relative_attention_bias(self, build_config):
        pass


class DecoderModel(PretrainedModel):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        super().__init__(config)

        self.mapping = self.config.mapping

        self.has_position_embedding = self.config.has_position_embedding
        type_vocab_size = self.config.type_vocab_size
        self.has_token_type_embedding = (type_vocab_size is not None)
        self.rescale_before_lm_head = self.config.rescale_before_lm_head

        # e.g. BART regular, T5 RMS
        self.layernorm_type = self.config.layernorm_type

        # e.g. BART true, T5 false
        self.has_attention_qkvo_bias = self.config.has_attention_qkvo_bias
        self.has_mlp_bias = self.config.has_mlp_bias

        # e.g. BART false, T5 true
        self.has_model_final_layernorm = self.config.has_model_final_layernorm
        self._dtype = self.config.dtype
        # no quantization considered for now
        self._kv_dtype = self._dtype
        self._logits_dtype = self.config.logits_dtype

        self.total_num_layers = self.config.num_hidden_layers
        self.num_layers = self.config.num_hidden_layers // self.mapping.pp_size

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        num_kv_heads = self.num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = self.num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = self.hidden_size // self.num_heads if self.config.head_size is None else self.config.head_size

        self.encoder_hidden_size = self.config.encoder_hidden_size
        self.encoder_num_heads = self.config.encoder_num_heads
        encoder_num_kv_heads = None if not hasattr(
            self.config,
            "encoder_num_kv_heads") else self.config.encoder_num_kv_heads
        if encoder_num_kv_heads is None or encoder_num_kv_heads <= 0:
            encoder_num_kv_heads = self.encoder_num_heads
        self.encoder_num_kv_heads = encoder_num_kv_heads
        self.encoder_head_size = self.encoder_hidden_size // self.num_heads if self.config.encoder_head_size is None else self.config.encoder_head_size

        self.has_position_embedding = self.config.has_position_embedding
        self.has_token_type_embedding = type_vocab_size is not None

        self.fp16_clamping = (self.config.dtype
                              == 'float16') and (self.config.model_type
                                                 in ['t5', 'pix2struct'])

        self.skip_cross_kv = self.config.skip_cross_kv
        self.mlp_type = MLPType.MLP if not hasattr(
            self.config, "mlp_type") else self.config.mlp_type
        self.use_implicit_relative_attention = self.config.use_implicit_relative_attention if hasattr(
            self.config, "use_implicit_relative_attention") else False

        self.language_adapter_config = None if not hasattr(
            self.config,
            'language_adapter_config') else LanguageAdapterConfig.from_dict(
                self.config.language_adapter_config)

        self.transformer = EncDecTransformer(
            self.config.vocab_size,
            self.config.hidden_size,
            max_position_embeddings=self.config.max_position_embeddings,
            has_position_embedding=self.config.has_position_embedding,
            type_vocab_size=type_vocab_size,
            has_embedding_layernorm=self.config.has_embedding_layernorm,
            has_embedding_scale=self.config.has_embedding_scale,
            layernorm_eps=self.config.norm_epsilon,
            layernorm_type=self.config.layernorm_type,
            dtype=self._dtype,
            use_parallel_embedding=self.config.use_parallel_embedding,
            embedding_sharding_dim=self.config.embedding_sharding_dim,
            mapping=self.mapping,
            has_model_final_layernorm=self.has_model_final_layernorm,
            norm_epsilon=self.config.norm_epsilon,
            is_decoder=True)

        layers_range = self.mapping.pp_layers(self.total_num_layers)
        decoder_layers = ModuleList([
            DecoderLayer(
                local_layer_idx=layer_idx - layers_range[0],
                hidden_size=self.config.hidden_size,
                ffn_hidden_size=self.config.intermediate_size,
                num_attention_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                max_position_embeddings=self.config.max_position_embeddings,
                q_scaling=self.config.q_scaling,
                has_attention_qkvo_bias=self.config.has_attention_qkvo_bias,
                has_mlp_bias=self.config.has_mlp_bias,
                layernorm_position=self.config.layernorm_position,
                layernorm_eps=self.config.norm_epsilon,
                layernorm_type=self.config.layernorm_type,
                hidden_act=self.config.hidden_act,
                mlp_type=self.mlp_type,
                mapping=self.mapping,
                dtype=self._dtype,
                residual_scaling=self.config.residual_scaling,
                relative_attention=self.config.relative_attention,
                max_distance=self.config.max_distance,
                num_buckets=self.config.num_buckets,
                fp16_clamping=self.fp16_clamping,
                skip_cross_kv=self.skip_cross_kv,
                use_implicit_relative_attention=self.
                use_implicit_relative_attention,
                quant_mode=self.config.quant_mode,
                language_adapter_config=self.language_adapter_config)
            for layer_idx in layers_range
        ])
        self.transformer.assign_module(decoder_layers, "layers")

        if self.mapping.is_last_pp_rank():
            vocab_size_padded = pad_vocab_size(self.config.vocab_size,
                                               self.config.mapping.tp_size)
            self.lm_head = ColumnLinear(
                self.config.hidden_size,
                vocab_size_padded,
                bias=False if not hasattr(self.config, "has_lm_head_bias") else
                self.config.has_lm_head_bias,
                dtype=self.config.dtype,
                tp_group=self.config.mapping.tp_group,
                tp_size=self.config.mapping.tp_size,
                gather_output=True,
            )

        self.trtllm_modules_to_hf_modules = {
            **get_default_trtllm_modules_to_hf_modules(),
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_dense": "self_attn.o_proj",
            "cross_attn_q": "encoder_attn.q_proj",
            "cross_attn_k": "encoder_attn.k_proj",
            "cross_attn_v": "encoder_attn.v_proj",
            "cross_attn_dense": "encoder_attn.o_proj",
        }

        if self.config.relative_attention and not self.use_implicit_relative_attention:
            self.rel_attn_table = Parameter(
                shape=(self.config.num_attention_heads // self.mapping.tp_size,
                       self.config.num_buckets),
                dtype=self._dtype)

    def check_config(self, config: PretrainedConfig):
        config.set_if_not_exist('has_position_embedding', False)
        config.set_if_not_exist('type_vocab_size', None)
        config.set_if_not_exist('rescale_before_lm_head', False)
        config.set_if_not_exist('layernorm_type', LayerNormType.LayerNorm)
        config.set_if_not_exist('layernorm_position',
                                LayerNormPositionType.pre_layernorm)
        config.set_if_not_exist('has_attention_qkvo_bias', False)
        config.set_if_not_exist('has_mlp_bias', False)
        config.set_if_not_exist('has_model_final_layernorm', False)
        config.set_if_not_exist('encoder_hidden_size', None)
        config.set_if_not_exist('encoder_num_heads', None)
        config.set_if_not_exist('encoder_num_kv_heads', None)
        config.set_if_not_exist('encoder_head_size', None)
        config.set_if_not_exist('model_type', 't5')
        config.set_if_not_exist('skip_cross_kv', False)
        config.set_if_not_exist('mlp_type', MLPType.MLP)
        config.set_if_not_exist('has_embedding_scale', False)
        config.set_if_not_exist('residual_scaling', 1.0)
        config.set_if_not_exist('has_lm_head_bias', False)
        config.set_if_not_exist('num_buckets', None)
        config.set_if_not_exist('max_distance', None)
        config.set_if_not_exist('relative_attention', False)

    def forward(self,
                decoder_input_ids: Tensor,
                encoder_output: Tensor,
                position_ids=None,
                token_type_ids=None,
                use_cache=False,
                attention_mask_params=None,
                last_token_ids=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                lora_params: LoraParams = None,
                cross_kv_cache_gen: Optional[Tensor] = None,
                cross_kv_reuse: Optional[Tensor] = None,
                language_adapter_routings: Optional[Tensor] = None):
        if self.mapping.is_first_pp_rank():
            assert isinstance(decoder_input_ids, Tensor)
        else:
            assert isinstance(hidden_states, Tensor)

        # In PP, layer 0 has ids as inputs, all other layers have hidden_states as inputs
        if self.mapping.is_first_pp_rank():
            hidden_states = self.transformer.embedding(decoder_input_ids,
                                                       position_ids,
                                                       token_type_ids)
            self.register_network_output('embedding_layer_output',
                                         hidden_states)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        kv_cache_params.fill_none_tensor_list(len(self.transformer.layers))

        if use_cache:
            presents = []

        for i, (decoder_layer, past) in enumerate(
                zip(self.transformer.layers, kv_cache_params.past_key_value)):

            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(i)

            hidden_states = decoder_layer(
                hidden_states,
                encoder_output=encoder_output,
                attention_mask_params=attention_mask_params,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=past,
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.
                    host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    cache_indirection=kv_cache_params.cache_indirection,
                    kv_cache_block_offsets=kv_cache_params.
                    kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.
                    host_cross_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.
                    host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.
                    host_kv_cache_pool_mapping,
                    cross_kv_cache_block_offsets=kv_cache_params.
                    cross_kv_cache_block_offsets,
                    host_cross_kv_cache_block_offsets=kv_cache_params.
                    host_cross_kv_cache_block_offsets,
                    host_cross_kv_cache_pool_pointers=kv_cache_params.
                    host_cross_kv_cache_pool_pointers,
                    host_cross_kv_cache_pool_mapping=kv_cache_params.
                    host_cross_kv_cache_pool_mapping),
                attention_params=attention_params,
                lora_layer_params=lora_layer_params,
                cross_kv_cache_gen=cross_kv_cache_gen,
                cross_kv_reuse=cross_kv_reuse,
                language_adapter_routings=language_adapter_routings)

            if use_cache:
                presents_self, presents_cross = hidden_states[1], hidden_states[
                    2]
                presents.append((presents_self, presents_cross))
                hidden_states = hidden_states[0]
            self.register_network_output(f'decoder_layer_{i}_output',
                                         hidden_states)

        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                hidden_states = self.transformer.ln_f(hidden_states)

            # [bs, seq, hidden_size] or [num_tokens, hidden_size] -> [bs, hidden_size]
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)
            self.register_network_output('logits_before_lmhead', hidden_states)

            # Rescale output before projecting on vocab (for T5)
            # See https://github.com/huggingface/transformers/blob/0b192de1f353b0e04dad4813e02e2c672de077be/src/transformers/models/t5/modeling_t5.py#L1769-L1772
            # Note: this is specific for T5, to make it more generic, one can pass in a config:
            #   self.config.tie_word_embeddings - default to be True for T5
            # openai whisper model didn't use this rescale
            if self.rescale_before_lm_head:
                hidden_states = hidden_states * (self.hidden_size**-0.5)

            # [bs, hidden_size] -> [bs, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output('logits', self._logits_dtype)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())
            hidden_states.mark_output('hidden_states_output', self._dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(self.mapping.pp_layers(self.total_num_layers),
                                  presents):
                present[0].mark_output(f'present_key_value_{i}', self._kv_dtype)
                if default_net().plugin_config.gpt_attention_plugin:
                    present[1].mark_output(f'cross_present_key_value_{i}',
                                           self._kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, tuple(presents))
            return (hidden_states, tuple(presents))
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(self,
                       max_batch_size,
                       max_beam_width,
                       max_decoder_input_len,
                       max_seq_len,
                       max_encoder_input_len,
                       gather_context_logits: bool = False,
                       lora_target_modules: List[str] = None,
                       use_cache=True,
                       *args,
                       **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        max_output_len = max_decoder_input_len + max_seq_len

        head_size = self.head_size
        num_kv_heads = (self.num_kv_heads + self.mapping.tp_size -
                        1) // self.mapping.tp_size

        encoder_head_size = self.encoder_head_size
        encoder_num_kv_heads = (self.encoder_num_kv_heads + self.mapping.tp_size
                                - 1) // self.mapping.tp_size

        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [
            1, 1, max_decoder_input_len
        ]  # context phase >= 1 (if forced_input_ids), generation phase = 1
        encoder_inlen_range = [
            1, (max_encoder_input_len + 1) // 2, max_encoder_input_len
        ]
        mask_len_range = [1, (max_output_len + 1) // 2 + 1, max_output_len + 1]
        max_output_len_range = [0, (max_output_len + 1) // 2, max_output_len]

        encoder_num_tokens_range = [
            0,  # 0 for generation phase, >0 for context phase
            (max_encoder_input_len * max_batch_size + 1) // 2,
            max_encoder_input_len * max_batch_size,
        ]
        decoder_num_tokens_range = [
            1,
            max_batch_size * max_beam_width,
            max(max_decoder_input_len * max_batch_size,
                max_beam_width * max_batch_size),
        ]

        # No enable_two_optimization_profiles support yet

        encoder_input_len_range = [
            0,  # 0 for generation phase, >0 for context phase
            (max_encoder_input_len + 1) // 2,
            max_encoder_input_len
        ]
        max_cross_packed_mask_dim0 = max_batch_size * (
            (max_decoder_input_len + 128 - 1) // 128) * 128
        max_cross_packed_mask_dim1 = (
            (max_encoder_input_len + 256 - 1) // 256) * 256 // 32
        cross_packed_mask_dim0_range = [
            1, (max_cross_packed_mask_dim0 + 1) // 2, max_cross_packed_mask_dim0
        ]
        cross_packed_mask_dim1_range = [
            0,  # 0 for generation phase, >0 for context phase
            (max_cross_packed_mask_dim1 + 1) // 2,
            max_cross_packed_mask_dim1
        ]

        past_key_value = []
        sequence_length = None
        host_past_key_value_lengths = None
        runtime_perf_knobs = None
        context_progress = None
        attention_mask = None
        cross_attention_mask = None
        cross_attention_packed_mask = None
        attention_mask_params = AttentionMaskParams()
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_lora_plugin = default_net().plugin_config.lora_plugin

        input_ids, position_ids, token_type_ids, hidden_states = None, None, None, None
        if remove_input_padding:
            if self.mapping.is_first_pp_rank():
                input_ids = Tensor(name='input_ids',
                                   dtype=trt.int32,
                                   shape=[-1],
                                   dim_range=OrderedDict([
                                       ('decoder_num_tokens',
                                        [decoder_num_tokens_range]),
                                   ]))
                if self.has_position_embedding:
                    position_ids = Tensor(name='position_ids',
                                          dtype=trt.int32,
                                          shape=[-1],
                                          dim_range=OrderedDict([
                                              ('decoder_num_tokens',
                                               [decoder_num_tokens_range]),
                                          ]))
                if self.has_token_type_embedding:
                    token_type_ids = Tensor(
                        name='token_type_ids',
                        dtype=trt.int32,
                        shape=[-1],
                        dim_range=OrderedDict([('decoder_num_tokens',
                                                [decoder_num_tokens_range])]),
                    )
            else:
                hidden_states = Tensor(name='hidden_states_input',
                                       dtype=self._dtype,
                                       shape=[-1, self.hidden_size],
                                       dim_range=OrderedDict([
                                           ('decoder_num_tokens',
                                            [decoder_num_tokens_range]),
                                           ('hidden_size', [self.hidden_size]),
                                       ]))
        else:
            if self.mapping.is_first_pp_rank():
                input_ids = Tensor(name='input_ids',
                                   dtype=trt.int32,
                                   shape=[-1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size_beam_width', [bb_range]),
                                       ('input_len', [inlen_range]),
                                   ]))
                if self.has_position_embedding:
                    position_ids = Tensor(name='position_ids',
                                          dtype=trt.int32,
                                          shape=[-1, -1],
                                          dim_range=OrderedDict([
                                              ('batch_size_beam_width',
                                               [bb_range]),
                                              ('input_len', [inlen_range]),
                                          ]))
                if self.has_token_type_embedding:
                    token_type_ids = Tensor(
                        name='token_type_ids',
                        dtype=trt.int32,
                        shape=[-1, -1],
                        dim_range=OrderedDict([('batch_size_beam_width',
                                                [bb_range]),
                                               ('input_len', [inlen_range])]),
                    )
            else:
                hidden_states = Tensor(name='hidden_states_input',
                                       dtype=self._dtype,
                                       shape=[-1, -1, self.hidden_size],
                                       dim_range=OrderedDict([
                                           ('batch_size_beam_width', [bb_range
                                                                      ]),
                                           ('input_len', [inlen_range]),
                                           ('hidden_size', [self.hidden_size]),
                                       ]))

        encoder_input_lengths = Tensor(
            name="encoder_input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size_beam_width", [bb_range])]),
        )
        encoder_max_input_length = Tensor(
            name="encoder_max_input_length",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("encoder_max_input_length",
                                    [encoder_inlen_range])]),
        )
        encoder_output = None
        if remove_input_padding:
            encoder_output = Tensor(
                name="encoder_output",
                dtype=self._dtype,
                shape=[-1, self.encoder_hidden_size],
                dim_range=OrderedDict([
                    ("encoder_num_tokens", [encoder_num_tokens_range]),
                    ("encoder_hidden_size", [self.encoder_hidden_size]),
                ]),
            )
        else:
            encoder_output = Tensor(
                name="encoder_output",
                dtype=self._dtype,
                shape=[-1, -1, self.encoder_hidden_size],
                dim_range=OrderedDict([
                    ("batch_size_beam_width_encoder", [bb_range]),
                    ("encoder_input_len", [encoder_input_len_range]),
                    ("encoder_hidden_size", [self.encoder_hidden_size]),
                ]),
            )

        if use_gpt_attention_plugin:
            host_past_key_value_lengths = Tensor(
                name='host_past_key_value_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )

        context_lengths = None
        host_context_lengths = None
        host_request_types = None
        if use_gpt_attention_plugin and remove_input_padding:
            host_context_lengths = Tensor(name='host_context_lengths',
                                          dtype=trt.int32,
                                          shape=[-1],
                                          dim_range=OrderedDict([
                                              ('batch_size_beam_width',
                                               [bb_range])
                                          ]))

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width', [bb_range])]),
            )

            context_lengths = Tensor(name='context_lengths',
                                     dtype=trt.int32,
                                     shape=[-1],
                                     dim_range=OrderedDict([
                                         ('batch_size_beam_width', [bb_range])
                                     ]))
            host_request_types = Tensor(name='host_request_types',
                                        dtype=trt.int32,
                                        shape=[-1],
                                        dim_range=OrderedDict([
                                            ('batch_size_beam_width',
                                             [bb_range])
                                        ]))
            runtime_perf_knobs = Tensor(name='host_runtime_perf_knobs',
                                        dtype=trt.int64,
                                        shape=[16],
                                        dim_range=OrderedDict([
                                            ('perf_knob_size', [16])
                                        ]))
            context_progress = Tensor(name='host_context_progress',
                                      dtype=trt.int64,
                                      shape=[1],
                                      dim_range=OrderedDict([
                                          ('context_progress_size', [1])
                                      ]))

        last_token_ids = None
        if self.mapping.is_last_pp_rank() and not gather_context_logits:
            last_token_ids = Tensor(
                name="last_token_ids",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size_last_token_ids", [bb_range])
                                       ]),
            )

        if not use_gpt_attention_plugin:
            attention_mask = Tensor(
                name='attention_mask',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('batch_size_beam_width', [bb_range]),
                    ('mask_len', [mask_len_range]),
                ]),
            )

            cross_attention_mask = Tensor(
                name='cross_attention_mask',
                dtype=trt.int32,
                shape=[-1, -1, -1],
                dim_range=OrderedDict([
                    ('batch_size_beam_width', [bb_range]),
                    ('query_len', [inlen_range]),
                    ('encoder_input_len_2', [encoder_input_len_range]),
                ]),
            )
        else:
            cross_attention_mask = Tensor(
                name='cross_attention_mask',
                dtype=trt.bool,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('decoder_num_tokens_2',
                     [decoder_num_tokens_range
                      ]),  # TODO should use same name as input_ids
                    ('encoder_input_len_2', [encoder_input_len_range]),
                ]),
            )

            cross_attention_packed_mask = Tensor(
                name='cross_attention_packed_mask',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([
                    ('cross_packed_mask_dim0', [cross_packed_mask_dim0_range]),
                    ('cross_packed_mask_dim1', [cross_packed_mask_dim1_range]),
                ]),
            )

        # create the attention_mask_params.
        attention_mask_params = AttentionMaskParams(
            attention_mask, None, cross_attention_mask,
            cross_attention_packed_mask)

        cache_indirection = Tensor(
            name='cache_indirection',
            dtype=trt.int32,
            shape=[-1, -1, -1],
            dim_range=OrderedDict([
                ('batch_size_cache', [bs_range]),
                ('beam_width', [beam_width_range]),
                ('max_seq_len', [max_output_len_range]),
            ]),
        )

        if self.mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(self.mapping, 1)

        layers_range = self.mapping.pp_layers(self.total_num_layers)
        num_pp_layers = len(layers_range)

        host_max_attention_window_sizes = None
        host_sink_token_length = None
        if use_gpt_attention_plugin:
            host_max_attention_window_sizes = Tensor(
                name=f'host_max_attention_window_sizes',
                dtype=trt.int32,
                shape=[num_pp_layers],
                dim_range=OrderedDict([('num_layers', [num_pp_layers])]))
            host_sink_token_length = Tensor(name='host_sink_token_length',
                                            dtype=trt.int32,
                                            shape=[1],
                                            dim_range=OrderedDict([('scalar',
                                                                    [1])]))
        '''
        LoRA plugin related inputs:
        lora_target_modules for BART-decoder:
            ['attn_q', 'cross_attn_q',
             'attn_v', 'cross_attn_v']
        This is NOT directly loaded from the adapter-config file
        We make it this way because BART has LoRA weights for both self-attention and cross-attention in decoder
        '''
        lora_weights_pointers = None
        lora_ranks = None
        lora_params = None
        if use_lora_plugin:
            lora_weights_pointers = []
            lora_ranks = []
            # In current design, q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time.
            # However, BART lora modules only contain two of them, so we use zero tensor to fill the missing ones.
            missing_qkv_modules = []
            if any(x in lora_target_modules
                   for x in ["attn_q", "attn_k", "attn_v"]):
                for lora_module in [
                        "attn_q",
                        "attn_k",
                        "attn_v",
                ]:
                    if lora_module not in lora_target_modules:
                        missing_qkv_modules.append(lora_module)
            if any(x in lora_target_modules
                   for x in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]):
                for lora_module in [
                        "cross_attn_q", "cross_attn_k", "cross_attn_v"
                ]:
                    if lora_module not in lora_target_modules:
                        missing_qkv_modules.append(lora_module)

            for i in layers_range:
                lora_weight_pointer_dict = {}
                lora_rank_dict = {}
                for lora_module in (lora_target_modules + missing_qkv_modules):
                    lora_weight_pointer = Tensor(
                        name=f'{lora_module}_lora_weights_pointers_{i}',
                        dtype=trt.int64,
                        shape=[-1, 3],
                        dim_range=OrderedDict([('batch_size_beam_width',
                                                [bb_range]),
                                               ('in_out_scales', [3])]))
                    lora_weight_pointer_dict.update({
                        f'{lora_module}_lora_weights_pointers':
                        lora_weight_pointer
                    })

                    lora_rank = Tensor(name=f'{lora_module}_lora_ranks_{i}',
                                       dtype=trt.int32,
                                       shape=[-1],
                                       dim_range=OrderedDict([
                                           ('batch_size_beam_width', [bb_range])
                                       ]))
                    lora_rank_dict.update(
                        {f'{lora_module}_lora_ranks': lora_rank})

                lora_weights_pointers.append(lora_weight_pointer_dict)
                lora_ranks.append(lora_rank_dict)

            # For cross attention, we need to use encoder_input_lengths (in CPU) to pass
            # as the host_context_lengths to the lora_plugin. But for self attention, we
            # should keep using the original host_context_lengths. Therefore, we keep both
            # of them in the lora_params.
            host_encoder_input_lengths = None
            if remove_input_padding:
                host_encoder_input_lengths = Tensor(
                    name="host_encoder_input_lengths",
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict([("batch_size_beam_width", [bb_range])
                                           ]),
                )

            lora_params = LoraParams(
                lora_ranks=lora_ranks,
                lora_weights_pointers=lora_weights_pointers,
                host_context_lengths=host_context_lengths,
                max_encoder_context_length=max_encoder_input_len,
                host_request_types=host_request_types,
                host_encoder_input_lengths=host_encoder_input_lengths,
            )

        kv_cache_block_offsets = None
        host_kv_cache_block_offsets = None
        host_kv_cache_pool_pointers = None
        host_kv_cache_pool_mapping = None

        cross_kv_cache_block_offsets = None
        host_cross_kv_cache_block_offsets = None
        host_cross_kv_cache_pool_pointers = None
        host_cross_kv_cache_pool_mapping = None

        if use_cache:
            if not paged_kv_cache:
                for i in layers_range:
                    kv_dim_range = OrderedDict([
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('num_heads', [num_kv_heads]),
                        ('past_key_len', [max_output_len_range]),
                        ('head_size', [head_size]),
                    ])
                    kv = Tensor(name=f'past_key_value_{i}',
                                dtype=self._kv_dtype,
                                shape=[-1, 2, num_kv_heads, -1, head_size],
                                dim_range=kv_dim_range)

                    if use_gpt_attention_plugin:
                        cross_kv_dim_range = OrderedDict([
                            ('batch_size_beam_width', [bb_range]),
                            ('kv', [2]),
                            ('cross_num_heads', [encoder_num_kv_heads]),
                            ('cross_past_key_len', [encoder_input_len_range]),
                            ('cross_head_size', [encoder_head_size]),
                        ])
                        cross_kv = Tensor(name=f'cross_past_key_value_{i}',
                                          dtype=self._kv_dtype,
                                          shape=[
                                              -1, 2, encoder_num_kv_heads, -1,
                                              encoder_head_size
                                          ],
                                          dim_range=cross_kv_dim_range)
                        past_key_value.append((kv, cross_kv))
                    else:
                        # use encoder_output directly, no need to save cross_past_key_value
                        past_key_value.append((kv, ))

                # TODO: Remove this when TRT fix the named dimension
                if not remove_input_padding:
                    assertion(
                        shape(
                            input_ids if self.mapping.is_first_pp_rank() else
                            hidden_states, 0) == shape(kv, 0), 'batch size')

            else:  # paged_kv_cache == True
                # PagedKV setup for KV cache of self-attention
                max_blocks_per_seq_range = [[
                    math.ceil(max_output_len_range[0] / tokens_per_block),
                    math.ceil(max_output_len_range[1] / tokens_per_block),
                    math.ceil(max_output_len_range[2] / tokens_per_block)
                ]]
                max_blocks_per_seq_range = [[
                    x for x in max_blocks_per_seq_range[0]
                ]]

                # PagedKV setup for KV cache of cross-attention
                max_cross_blocks_per_seq_range = [[
                    math.ceil(encoder_input_len_range[0] / tokens_per_block),
                    math.ceil(encoder_input_len_range[1] / tokens_per_block),
                    math.ceil(encoder_input_len_range[2] / tokens_per_block)
                ]]
                max_cross_blocks_per_seq_range = [[
                    x for x in max_cross_blocks_per_seq_range[0]
                ]]

                # TODO(oargov): add support for vgqa + vwindow, meanwhile assume a single kv cache pool
                num_kv_cache_pools = 1

                kv_cache_block_offsets = Tensor(
                    name=f'kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_blocks_per_seq', max_blocks_per_seq_range),
                    ]))
                host_kv_cache_block_offsets = Tensor(
                    name=f'host_kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_blocks_per_seq', max_blocks_per_seq_range),
                    ]))
                host_kv_cache_pool_pointers = Tensor(
                    name=f'host_kv_cache_pool_pointers',
                    dtype=trt.int64,
                    shape=[num_kv_cache_pools, 2],
                    dim_range=OrderedDict([
                        ('num_pools_layers', [num_kv_cache_pools]),
                        ('num_pools_kv', [2]),
                    ]))
                host_kv_cache_pool_mapping = Tensor(
                    name=f"host_kv_cache_pool_mapping",
                    dtype=trt.int32,
                    # 2: (Index of pool, Index of layer within pool)
                    shape=[num_pp_layers, 2],
                    dim_range=OrderedDict([
                        ('pools_mapping', [num_pp_layers]),
                        ('layer_cache_pool_locator', [2]),
                    ]))

                # paged blocks for cross kv
                cross_kv_cache_block_offsets = Tensor(
                    name=f'cross_kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_cross_blocks_per_seq',
                         max_cross_blocks_per_seq_range),
                    ]))
                host_cross_kv_cache_block_offsets = Tensor(
                    name=f'host_cross_kv_cache_block_offsets',
                    dtype=trt.int32,
                    shape=[num_kv_cache_pools, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('batch_size_beam_width', [bb_range]),
                        ('kv', [2]),
                        ('max_cross_blocks_per_seq',
                         max_cross_blocks_per_seq_range),
                    ]))
                host_cross_kv_cache_pool_pointers = Tensor(
                    name=f'host_cross_kv_cache_pool_pointers',
                    dtype=trt.int64,
                    shape=[num_kv_cache_pools, 2],
                    dim_range=OrderedDict([
                        ('num_kv_cache_pools', [num_kv_cache_pools]),
                        ('num_pools', [2]),
                    ]))
                host_cross_kv_cache_pool_mapping = Tensor(
                    name=f"host_cross_kv_cache_pool_mapping",
                    dtype=trt.int32,
                    # 2: (Index of pool, Index of layer within pool)
                    shape=[num_pp_layers, 2],
                    dim_range=OrderedDict([
                        ('pools_mapping', [num_pp_layers]),
                        ('layer_cache_pool_locator', [2]),
                    ]))

                for i in layers_range:
                    past_key_value.append(None)

            kv_cache_params = KeyValueCacheParams(
                past_key_value=past_key_value,
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                cache_indirection=cache_indirection,
                kv_cache_block_offsets=kv_cache_block_offsets,
                host_kv_cache_block_offsets=host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
                cross_kv_cache_block_offsets=cross_kv_cache_block_offsets,
                host_cross_kv_cache_block_offsets=
                host_cross_kv_cache_block_offsets,
                host_cross_kv_cache_pool_pointers=
                host_cross_kv_cache_pool_pointers,
                host_cross_kv_cache_pool_mapping=
                host_cross_kv_cache_pool_mapping,
            )

            attention_params = AttentionParams(
                sequence_length=sequence_length,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                max_context_length=max_decoder_input_len,
                host_request_types=host_request_types,
                encoder_input_lengths=encoder_input_lengths,
                encoder_max_input_length=encoder_max_input_length,
                host_runtime_perf_knobs=runtime_perf_knobs,
                host_context_progress=context_progress)

        cross_kv_cache_gen = Tensor(name='cross_kv_cache_gen',
                                    dtype=trt.bool,
                                    shape=[1],
                                    dim_range=OrderedDict([
                                        ('boolean', [1]),
                                    ]))
        cross_kv_reuse = None
        num_heads = (self.num_heads + self.mapping.tp_size -
                     1) // self.mapping.tp_size
        cross_kv_out_dim = 2 * num_kv_heads * self.head_size
        if self.skip_cross_kv:
            if remove_input_padding:
                cross_kv_reuse = Tensor(
                    name="cross_kv_reuse",
                    dtype=self._dtype,
                    shape=[-1, cross_kv_out_dim],
                    dim_range=OrderedDict([
                        ("encoder_num_tokens", [encoder_num_tokens_range]),
                        ("encoder_kv_size", [cross_kv_out_dim]),
                    ]),
                )
            else:
                cross_kv_reuse = Tensor(
                    name="cross_kv_reuse",
                    dtype=self._dtype,
                    shape=[-1, -1, cross_kv_out_dim],
                    dim_range=OrderedDict([
                        ("batch_size_beam_width_encoder", [bb_range]),
                        ("encoder_input_len", [encoder_input_len_range]),
                        ("encoder_kv_size", [cross_kv_out_dim]),
                    ]),
                )

        language_adapter_routings = None
        if self.language_adapter_config:
            language_adapter_routings = Tensor(
                name="language_adapter_routings",
                dtype=trt.int32,
                shape=[-1, 1],
                dim_range=OrderedDict([('decoder_num_tokens_range',
                                        [decoder_num_tokens_range]),
                                       ('routing_dim', [1])]),
            )

        result = {
            'decoder_input_ids': input_ids,
            'encoder_output': encoder_output,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
            'use_cache': True,
            'attention_mask_params': attention_mask_params,
            'last_token_ids': last_token_ids,
            'kv_cache_params': kv_cache_params,
            'attention_params': attention_params,
            'hidden_states': hidden_states,
            'lora_params': lora_params,
            'cross_kv_cache_gen': cross_kv_cache_gen,
            'cross_kv_reuse': cross_kv_reuse,
            'language_adapter_routings': language_adapter_routings,
        }

        return result

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)

    def precompute_relative_attention_bias(self, build_config):
        if self.config.relative_attention and not self.use_implicit_relative_attention:
            relative_attention_bias_builder = torch.ops.tensorrt_llm.relative_attention_bias
            rel_attn_precomputed = torch.zeros(
                (self.config.num_attention_heads // self.mapping.tp_size,
                 build_config.max_seq_len + 1, build_config.max_seq_len + 1),
                dtype=str_dtype_to_torch(self.config.dtype),
                device='cuda')
            rel_attn_table = numpy_to_torch(
                self.rel_attn_table.raw_value).to('cuda')
            relative_attention_bias_builder(
                rel_attn_precomputed,
                rel_attn_table,
                self.config.num_attention_heads // self.mapping.tp_size,
                build_config.max_seq_len,
                self.config.num_buckets,
                False,
                self.config.max_distance,
            )
            for layer_idx in range(self.num_layers):
                self.transformer.layers[
                    layer_idx].self_attention.set_rel_attn_table(
                        build_config.max_seq_len, rel_attn_precomputed)


class WhisperEncoder(PretrainedModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self._dtype = self.config.dtype
        conv1 = Conv1d(config.n_mels,
                       config.hidden_size,
                       kernel_size=3,
                       padding=1,
                       dtype=self._dtype)
        conv2 = Conv1d(config.hidden_size,
                       config.hidden_size,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       dtype=self._dtype)
        self.transformer = EncDecTransformer(
            0,
            self.config.hidden_size,
            has_vocab_embeddings=False,
            max_position_embeddings=self.config.max_position_embeddings,
            has_position_embedding=self.config.has_position_embedding,
            layernorm_type=LayerNormType.LayerNorm,
            dtype=self.config.dtype,
            has_model_final_layernorm=True,
            is_decoder=False)
        encoder_layers = ModuleList([
            EncoderLayer(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.hidden_size * 4,
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_attention_heads,
                head_size=config.hidden_size // config.num_attention_heads,
                q_scaling=1.0,
                has_attention_qkvo_bias=True,
                has_mlp_bias=True,
                hidden_act='gelu',
                dtype=self._dtype) for _ in range(config.num_hidden_layers)
        ])
        self.transformer.assign_module(encoder_layers, "layers")
        self.transformer.assign_module(conv1, "conv1")
        self.transformer.assign_module(conv2, "conv2")
        self.max_audio_feature_seq_len = 3000
        self.downsample_factor = 2

    def forward(self,
                input_features: Tensor,
                input_lengths=None,
                position_ids=None):
        if default_net().plugin_config.remove_input_padding:
            # BXT,D -> 1,BxT,D -> 1,D,BxT
            input_features = unsqueeze(input_features, 0)
            input_features = transpose(input_features, 1, 2)
        x_type = input_features.dtype
        input_features = cast(input_features, self._dtype)
        x = self.transformer.conv1(input_features)
        x = gelu(x)
        x = self.transformer.conv2(x)
        x = cast(x, x_type)
        x = gelu(x)
        x = transpose(x, 2, 1)
        x = x + cast(self.transformer.position_embedding(position_ids), x.dtype)

        if default_net().plugin_config.remove_input_padding:
            #B,T,D -> BxT,D
            x = x.view([-1, self.config.hidden_size])
        hidden_states = x
        input_lengths = input_lengths // self.downsample_factor
        for encoder_layer in self.transformer.layers:
            hidden_states = encoder_layer(hidden_states,
                                          input_lengths=input_lengths)

        x = hidden_states
        x = self.transformer.ln_f(x)
        x.mark_output('encoder_output', self._dtype)
        return x

    def prepare_inputs(self, max_batch_size=16):

        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        min_feat_len, optimal_feat_len = 10, 1000  # 100ms, 10s
        inlen_range = [
            min_feat_len, optimal_feat_len, self.max_audio_feature_seq_len
        ]
        inlen_range_after_downsample = [
            min_feat_len // self.downsample_factor,
            optimal_feat_len // self.downsample_factor,
            self.max_audio_feature_seq_len // self.downsample_factor
        ]
        if not default_net().plugin_config.remove_input_padding:
            x = Tensor(name="input_features",
                       dtype=self._dtype,
                       shape=[-1, self.config.n_mels, -1],
                       dim_range=OrderedDict([
                           ("batch_size", [bs_range]),
                           ("feature_dim", [self.config.n_mels]),
                           ("feature_len_range", [inlen_range]),
                       ]))
            position_ids = Tensor(
                name='position_ids',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([('batch_size', [bs_range]),
                                       ('feature_len_downsample_range',
                                        [inlen_range_after_downsample])]),
            )
        else:
            batch_seqlen_range = [
                1,
                (self.max_audio_feature_seq_len * max_batch_size + 1) // 2,
                self.max_audio_feature_seq_len * max_batch_size,
            ]
            batch_seqlen_downsample_range = [
                1,
                (self.max_audio_feature_seq_len // self.downsample_factor *
                 max_batch_size + 1) // 2,
                self.max_audio_feature_seq_len // self.downsample_factor *
                max_batch_size,
            ]
            x = Tensor(name="input_features",
                       dtype=self._dtype,
                       shape=[-1, self.config.n_mels],
                       dim_range=OrderedDict([
                           ("batch_seqlen_range", [batch_seqlen_range]),
                           ("feature_dim", [self.config.n_mels]),
                       ]))
            position_ids = Tensor(
                name='position_ids',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_seqlen_downsample_range',
                                        [batch_seqlen_downsample_range])]),
            )
        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [bs_range])]),
        )

        return {
            'input_features': x,
            'input_lengths': input_lengths,
            'position_ids': position_ids
        }

    def precompute_relative_attention_bias(self, build_config):
        pass
