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
from typing import Optional

import tensorrt as trt

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     MLPType, PositionEmbeddingType, Tensor,
                                     assertion, gather_last_token_logits, gelu,
                                     recv, send, shape, transpose)
from tensorrt_llm.layers import (MLP, Attention, AttentionMaskType,
                                 AttentionParams, BertAttention, ColumnLinear,
                                 Conv1d, Embedding, FusedGatedMLP, GatedMLP,
                                 GroupNorm, KeyValueCacheParams, LayerNorm,
                                 PromptTuningEmbedding, RmsNorm)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.plugin.plugin import (current_all_reduce_helper,
                                        init_all_reduce_helper)

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


class EncDecEmbedding(Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position_embeddings=None,
                 has_position_embedding=False,
                 type_vocab_size=None,
                 has_embedding_layernorm=False,
                 has_embedding_scale=False,
                 layernorm_eps=1e-5,
                 layernorm_type=LayerNormType.LayerNorm,
                 dtype=None,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 mapping=Mapping()):
        super().__init__()

        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]
        self.use_prompt_tuning = use_prompt_tuning

        EmbeddingCls = PromptTuningEmbedding if use_prompt_tuning else Embedding
        self.vocab_embedding = EmbeddingCls(
            vocab_size,
            hidden_size,
            dtype=dtype,
            tp_size=mapping.tp_size if use_parallel_embedding else 1,
            tp_group=mapping.tp_group if use_parallel_embedding else None,
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
                tp_group=mapping.tp_group if use_parallel_embedding else None,
                sharding_dim=embedding_sharding_dim,
                tp_rank=mapping.tp_rank)

        self.token_type_embedding = None
        if type_vocab_size:
            self.token_type_embedding = Embedding(
                type_vocab_size,
                hidden_size,
                dtype=dtype,
                tp_size=mapping.tp_size if use_parallel_embedding else 1,
                tp_group=mapping.tp_group if use_parallel_embedding else None,
                sharding_dim=embedding_sharding_dim,
                tp_rank=mapping.tp_rank)

        # e.g. BART true, T5 false
        self.embedding_layernorm = None
        if has_embedding_layernorm:
            self.embedding_layernorm = ln_type(normalized_shape=hidden_size,
                                               eps=layernorm_eps,
                                               dtype=dtype)

        # e.g. BART true, T5 false
        self.embedding_scale = 1.0
        if has_embedding_scale:
            self.embedding_scale = math.sqrt(hidden_size)

        # Note: embedding offset in BART is not considered as a standard. For the specific case,
        # we just need to shrink its position embedding table by [offset:] during weight loading

    def forward(self,
                input_ids,
                position_ids=None,
                token_type_ids=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):
        # position_ids and token_type_ids are provided inputs
        # and should not be formulated determinisitically

        ptuning_args = []
        if self.use_prompt_tuning:
            ptuning_args = [
                prompt_embedding_table, prompt_tasks, prompt_vocab_size
            ]
        x = self.vocab_embedding(input_ids, *
                                 ptuning_args) * self.embedding_scale
        self.register_network_output('word_embeddings', x)

        if self.position_embedding:
            pos_emb = self.position_embedding(position_ids)
            self.register_network_output('position_embeddings', pos_emb)
            x = x + pos_emb
        if self.token_type_embedding:
            x = x + self.token_type_embedding(token_type_ids)

        if self.embedding_layernorm:
            x = self.embedding_layernorm(x)

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
                 num_buckets=0):
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
            num_buckets=num_buckets)

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
        )

        self.mlp_layernorm = ln_type(normalized_shape=hidden_size,
                                     eps=layernorm_eps,
                                     dtype=dtype)

        self.residual_scaling = residual_scaling

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                input_lengths=None,
                max_input_length=None):
        assert isinstance(hidden_states, Tensor)

        # self attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          input_lengths=input_lengths,
                                          max_input_length=max_input_length)

        self.register_network_output('attention_output', attention_output)

        hidden_states = residual + attention_output

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.attention_layernorm(hidden_states)

        # MLP
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        self.register_network_output('mlp_output', hidden_states)

        hidden_states = residual + hidden_states

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        return hidden_states


class DecoderLayer(Module):

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
                 num_buckets=0):
        super().__init__()

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART post, T5 pre
        self.layernorm_position = layernorm_position

        # e.g. BART q_scaling = 1.f, T5 q_scaling = 1.f/sqrt(head_size)
        self.self_attention = Attention(
            hidden_size,
            num_attention_heads,
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
            max_distance=max_distance,
            num_buckets=num_buckets,
            position_embedding_type=PositionEmbeddingType.relative
            if relative_attention else PositionEmbeddingType.learned_absolute)

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
            hidden_size,
            num_attention_heads,
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
            position_embedding_type=PositionEmbeddingType.learned_absolute)

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
        )

        self.mlp_layernorm = ln_type(normalized_shape=hidden_size,
                                     eps=layernorm_eps,
                                     dtype=dtype)

        self.residual_scaling = residual_scaling

    def forward(self,
                hidden_states: Tensor,
                encoder_output: Optional[Tensor] = None,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        assert isinstance(hidden_states, Tensor)

        if encoder_output:
            assert isinstance(encoder_output, Tensor)

        # self-attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        attention_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params)

        if use_cache:
            attention_output, presents_self = attention_output

        self.register_network_output('self_attention_output', attention_output)

        hidden_states = residual + attention_output

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        # cross attention
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.cross_attention_layernorm(hidden_states)

        attention_output = self.cross_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_output=encoder_output,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params)

        if use_cache:
            attention_output, presents_cross = attention_output

        self.register_network_output('cross_attention_output', attention_output)

        hidden_states = residual + attention_output

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.cross_attention_layernorm(hidden_states)

        # MLP
        residual = hidden_states * self.residual_scaling

        if self.layernorm_position == LayerNormPositionType.pre_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        self.register_network_output('mlp_output', hidden_states)

        hidden_states = residual + hidden_states

        if self.layernorm_position == LayerNormPositionType.post_layernorm:
            hidden_states = self.mlp_layernorm(hidden_states)

        if use_cache:
            return (hidden_states, presents_self, presents_cross)
        return hidden_states


class EncoderModel(Module, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 ffn_hidden_size,
                 vocab_size,
                 dtype,
                 head_size=None,
                 num_kv_heads=None,
                 max_position_embeddings=None,
                 has_position_embedding=False,
                 relative_attention=False,
                 max_distance=None,
                 num_buckets=None,
                 type_vocab_size=None,
                 has_embedding_layernorm=False,
                 has_embedding_scale=False,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 has_model_final_layernorm=False,
                 layernorm_eps=1e-5,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 hidden_act="relu",
                 mlp_type=MLPType.MLP,
                 residual_scaling=1.0,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 mapping=Mapping()):
        super().__init__()
        init_all_reduce_helper()
        self.mapping = mapping

        self.has_position_embedding = has_position_embedding
        self.has_token_type_embedding = type_vocab_size is not None

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART true, T5 false
        self.has_attention_qkvo_bias = has_attention_qkvo_bias
        self.has_mlp_bias = has_mlp_bias

        # e.g. BART false, T5 true
        self.has_model_final_layernorm = has_model_final_layernorm

        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype

        self.total_num_layers = num_layers
        self.num_layers = num_layers // self.mapping.pp_size

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = self.hidden_size // self.num_heads if head_size is None else head_size

        if self.mapping.is_first_pp_rank():
            self.embedding = EncDecEmbedding(
                vocab_size,
                hidden_size,
                max_position_embeddings=max_position_embeddings,
                has_position_embedding=has_position_embedding,
                type_vocab_size=type_vocab_size,
                has_embedding_layernorm=has_embedding_layernorm,
                has_embedding_scale=has_embedding_scale,
                layernorm_eps=layernorm_eps,
                layernorm_type=layernorm_type,
                dtype=dtype,
                use_prompt_tuning=use_prompt_tuning,
                use_parallel_embedding=use_parallel_embedding,
                embedding_sharding_dim=embedding_sharding_dim,
                mapping=self.mapping)

        self.encoder_layers = ModuleList([
            EncoderLayer(hidden_size=hidden_size,
                         ffn_hidden_size=ffn_hidden_size,
                         num_attention_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         head_size=self.head_size,
                         max_position_embeddings=max_position_embeddings,
                         q_scaling=q_scaling,
                         has_attention_qkvo_bias=has_attention_qkvo_bias,
                         has_mlp_bias=has_mlp_bias,
                         layernorm_position=layernorm_position,
                         layernorm_eps=layernorm_eps,
                         layernorm_type=layernorm_type,
                         hidden_act=hidden_act,
                         mlp_type=mlp_type,
                         mapping=self.mapping,
                         dtype=dtype,
                         residual_scaling=residual_scaling,
                         relative_attention=relative_attention,
                         max_distance=max_distance,
                         num_buckets=num_buckets)
            for _ in self.mapping.pp_layers(self.total_num_layers)
        ])

        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                self.final_layernorm = ln_type(normalized_shape=hidden_size,
                                               eps=layernorm_eps,
                                               dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                input_lengths=None,
                position_ids=None,
                token_type_ids=None,
                hidden_states=None,
                max_input_length=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):

        # In PP, layer 0 has ids as inputs, all other layers have hidden_states as inputs
        if self.mapping.is_first_pp_rank():
            hidden_states = self.embedding(input_ids, position_ids,
                                           token_type_ids,
                                           prompt_embedding_table, prompt_tasks,
                                           prompt_vocab_size)
            self.register_network_output('embedding_layer_output',
                                         hidden_states)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states=hidden_states,
                                          input_lengths=input_lengths,
                                          max_input_length=max_input_length)

        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                hidden_states = self.final_layernorm(hidden_states)
            hidden_states.mark_output('encoder_output', self._dtype)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())
            hidden_states.mark_output('hidden_states_output', self._dtype)

        return hidden_states

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       prompt_embedding_table_size: int = 0):
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
        use_custom_all_reduce = default_net(
        ).plugin_config.use_custom_all_reduce

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

        if use_custom_all_reduce and self.mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(
                self.mapping, False)

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
                                   ('batch_size_beam_width', bs_range),
                                   ('broadcast_dim', [1]),
                               ]))
            prompt_vocab_size = Tensor(name='prompt_vocab_size',
                                       dtype=trt.int32,
                                       shape=[1],
                                       dim_range=OrderedDict([('size', [1])]))

        return (input_ids, input_lengths, position_ids, token_type_ids,
                hidden_states, max_input_length, prompt_embedding_table, tasks,
                prompt_vocab_size)


class DecoderModel(Module, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 ffn_hidden_size,
                 encoder_num_heads,
                 encoder_hidden_size,
                 vocab_size,
                 dtype,
                 logits_dtype='float32',
                 head_size=None,
                 encoder_head_size=None,
                 num_kv_heads=None,
                 encoder_num_kv_heads=None,
                 max_position_embeddings=None,
                 has_position_embedding=False,
                 relative_attention=False,
                 max_distance=None,
                 num_buckets=None,
                 type_vocab_size=None,
                 has_embedding_layernorm=False,
                 has_embedding_scale=False,
                 q_scaling=1.0,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 has_model_final_layernorm=False,
                 layernorm_eps=1e-5,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 layernorm_type=LayerNormType.LayerNorm,
                 hidden_act="relu",
                 mlp_type=MLPType.MLP,
                 rescale_before_lm_head=False,
                 has_lm_head_bias=False,
                 residual_scaling=1.0,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 mapping=Mapping()):
        super().__init__()
        init_all_reduce_helper()
        self.mapping = mapping

        self.has_position_embedding = has_position_embedding  # TODO: remove dup codes
        self.has_token_type_embedding = type_vocab_size is not None
        self.rescale_before_lm_head = rescale_before_lm_head

        # e.g. BART regular, T5 RMS
        self.layernorm_type = layernorm_type
        ln_type = layernorm_map[layernorm_type]

        # e.g. BART true, T5 false
        self.has_attention_qkvo_bias = has_attention_qkvo_bias
        self.has_mlp_bias = has_mlp_bias

        # e.g. BART false, T5 true
        self.has_model_final_layernorm = has_model_final_layernorm

        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype

        # no quantization considered for now
        self._kv_dtype = self._dtype

        if isinstance(logits_dtype, str):
            self._logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self._logits_dtype = logits_dtype

        self.total_num_layers = num_layers
        self.num_layers = num_layers // self.mapping.pp_size

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = self.hidden_size // self.num_heads if head_size is None else head_size

        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_heads = encoder_num_heads
        if encoder_num_kv_heads is None or encoder_num_kv_heads <= 0:
            encoder_num_kv_heads = encoder_num_heads
        self.encoder_num_kv_heads = encoder_num_kv_heads
        self.encoder_head_size = self.encoder_hidden_size // self.num_heads if encoder_head_size is None else encoder_head_size

        self.has_position_embedding = has_position_embedding
        self.has_token_type_embedding = type_vocab_size is not None
        self.rescale_before_lm_head = rescale_before_lm_head

        if self.mapping.is_first_pp_rank():
            self.embedding = EncDecEmbedding(
                vocab_size,
                hidden_size,
                max_position_embeddings=max_position_embeddings,
                has_position_embedding=has_position_embedding,
                type_vocab_size=type_vocab_size,
                has_embedding_layernorm=has_embedding_layernorm,
                has_embedding_scale=has_embedding_scale,
                layernorm_eps=layernorm_eps,
                layernorm_type=layernorm_type,
                dtype=dtype,
                use_parallel_embedding=use_parallel_embedding,
                embedding_sharding_dim=embedding_sharding_dim,
                mapping=self.mapping)

        self.decoder_layers = ModuleList([
            DecoderLayer(hidden_size=hidden_size,
                         ffn_hidden_size=ffn_hidden_size,
                         num_attention_heads=num_heads,
                         num_kv_heads=self.num_kv_heads,
                         head_size=self.head_size,
                         max_position_embeddings=max_position_embeddings,
                         q_scaling=q_scaling,
                         has_attention_qkvo_bias=has_attention_qkvo_bias,
                         has_mlp_bias=has_mlp_bias,
                         layernorm_position=layernorm_position,
                         layernorm_eps=layernorm_eps,
                         layernorm_type=layernorm_type,
                         hidden_act=hidden_act,
                         mlp_type=mlp_type,
                         mapping=self.mapping,
                         dtype=dtype,
                         residual_scaling=residual_scaling,
                         relative_attention=relative_attention,
                         max_distance=max_distance,
                         num_buckets=num_buckets)
            for _ in self.mapping.pp_layers(self.total_num_layers)
        ])

        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                self.final_layernorm = ln_type(normalized_shape=hidden_size,
                                               eps=layernorm_eps,
                                               dtype=dtype)

            self.lm_head = ColumnLinear(
                hidden_size,
                vocab_size,
                bias=has_lm_head_bias,
                dtype=dtype,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                gather_output=True,
            )

    def forward(self,
                decoder_input_ids: Tensor,
                encoder_output: Tensor,
                position_ids=None,
                token_type_ids=None,
                use_cache=False,
                attention_mask=None,
                last_token_ids=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None):
        if self.mapping.is_first_pp_rank():
            assert isinstance(decoder_input_ids, Tensor)
        else:
            assert isinstance(hidden_states, Tensor)

        # In PP, layer 0 has ids as inputs, all other layers have hidden_states as inputs
        if self.mapping.is_first_pp_rank():
            hidden_states = self.embedding(decoder_input_ids, position_ids,
                                           token_type_ids)
            self.register_network_output('embedding_layer_output',
                                         hidden_states)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        kv_cache_params.fill_none_tensor_list(len(self.decoder_layers))

        if use_cache:
            presents = []

        for i, (decoder_layer, past, max_attention_window_size) in enumerate(
                zip(self.decoder_layers, kv_cache_params.past_key_value,
                    kv_cache_params.host_max_attention_window_sizes)):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=past,
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

            if use_cache:
                presents_self, presents_cross = hidden_states[1], hidden_states[
                    2]
                presents.append((presents_self, presents_cross))
                hidden_states = hidden_states[0]
            self.register_network_output(f'decoder_layer_{i}_output',
                                         hidden_states)

        if self.mapping.is_last_pp_rank():
            if self.has_model_final_layernorm:
                hidden_states = self.final_layernorm(hidden_states)

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
                present[1].mark_output(f'cross_present_key_value_{i}',
                                       self._kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, tuple(presents))
            return (hidden_states, tuple(presents))
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(
        self,
        max_batch_size,
        max_beam_width,
        max_decoder_input_len,
        max_new_tokens,
        max_encoder_input_len,
        gather_context_logits: bool = False,
        gather_generation_logits: bool = False,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        max_output_len = max_decoder_input_len + max_new_tokens

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
            1,
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
            0, (max_encoder_input_len + 1) // 2, max_encoder_input_len
        ]
        past_key_value = []
        sequence_length = None
        host_past_key_value_lengths = None
        attention_mask = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_custom_all_reduce = default_net(
        ).plugin_config.use_custom_all_reduce

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

        if use_custom_all_reduce and self.mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(
                self.mapping, False)

        layers_range = self.mapping.pp_layers(self.total_num_layers)

        if use_gpt_attention_plugin:
            host_max_attention_window_sizes = []
            for i in layers_range:
                host_attention_window_size_tensor = Tensor(
                    name=f'host_max_attention_window_size_{i}',
                    dtype=trt.int32,
                    shape=[1],
                    dim_range=OrderedDict([('scalar', [1])]))
                host_max_attention_window_sizes.append(
                    host_attention_window_size_tensor)
            host_sink_token_length = Tensor(name=f'host_sink_token_length',
                                            dtype=trt.int32,
                                            shape=[1],
                                            dim_range=OrderedDict([('scalar',
                                                                    [1])]))

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

            cross_kv_dim_range = OrderedDict([
                ('batch_size_beam_width', [bb_range]),
                ('kv', [2]),
                ('cross_num_heads', [encoder_num_kv_heads]),
                ('cross_past_key_len', [encoder_input_len_range]),
                ('cross_head_size', [encoder_head_size]),
            ])
            cross_kv = Tensor(
                name=f'cross_past_key_value_{i}',
                dtype=self._kv_dtype,
                shape=[-1, 2, encoder_num_kv_heads, -1, encoder_head_size],
                dim_range=cross_kv_dim_range)
            past_key_value.append((kv, cross_kv))

            # TODO: Remove this when TRT fix the named dimension
            if not remove_input_padding:
                assertion(
                    shape(
                        input_ids if self.mapping.is_first_pp_rank() else
                        hidden_states, 0) == shape(kv, 0), 'batch size')

            kv_cache_params = KeyValueCacheParams(
                past_key_value=past_key_value,
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                cache_indirection=cache_indirection)

            attention_params = AttentionParams(
                sequence_length=sequence_length,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                max_context_length=max_decoder_input_len,
                host_request_types=host_request_types,
                encoder_input_lengths=encoder_input_lengths,
                encoder_max_input_length=encoder_max_input_length,
            )

        return (input_ids, encoder_output, position_ids, token_type_ids, True,
                attention_mask, last_token_ids, kv_cache_params,
                attention_params, hidden_states)


class WhisperEncoder(Module):

    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int,
                 n_layer: int, dtype):
        super().__init__()
        self.n_mels = n_mels
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state,
                            n_state,
                            kernel_size=3,
                            stride=2,
                            padding=1)
        self.positional_embedding = Parameter(shape=(n_ctx, n_state),
                                              dtype=dtype)
        self.encoder_layers = ModuleList([
            EncoderLayer(hidden_size=n_state,
                         ffn_hidden_size=n_state * 4,
                         num_attention_heads=n_head,
                         num_kv_heads=n_head,
                         head_size=n_state // n_head,
                         max_position_embeddings=3000,
                         q_scaling=1.0,
                         has_attention_qkvo_bias=True,
                         has_mlp_bias=True,
                         hidden_act='gelu',
                         dtype=dtype) for _ in range(n_layer)
        ])

        self.ln_post = LayerNorm(n_state)
        self._dtype = dtype

    def forward(self, x: Tensor, input_lengths=None):

        x = self.conv1(x)
        x = gelu(x)
        x = self.conv2(x)
        x = gelu(x)
        x = transpose(x, 2, 1)
        x = x + self.positional_embedding.value

        hidden_states = x
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states,
                                          input_lengths=input_lengths)

        x = hidden_states
        x = self.ln_post(x)
        x.mark_output('output', self._dtype)
        return x

    def prepare_inputs(self, max_batch_size=16):

        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]

        x = Tensor(name="x",
                   dtype=self._dtype,
                   shape=[-1, self.n_mels, 3000],
                   dim_range=OrderedDict([
                       ("batch_size", [bs_range]),
                       ("feature_dim", [self.n_mels]),
                       ("feature_len_range", [3000]),
                   ]))
        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [bs_range])]),
        )
        return (x, input_lengths)
