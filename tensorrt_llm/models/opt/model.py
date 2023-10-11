# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import OrderedDict

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import Tensor, gather_last_token_logits
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, KeyValueCacheParams, LayerNorm,
                       PositionEmbeddingType)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin
from ..gpt.model import GPTEmbedding


class OPTDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 dtype=None,
                 hidden_act='relu',
                 pre_norm=False,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            attention_mask_type=AttentionMaskType.causal,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size)
        self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                        dtype=dtype)

        self.pre_norm = pre_norm

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        residual = hidden_states

        attention_input = hidden_states
        if self.pre_norm:
            attention_input = self.input_layernorm(hidden_states)

        # At this point the hidden_states object must be a Tensor.
        assert isinstance(attention_input, Tensor)

        attention_output = self.attention(attention_input,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output
        if not self.pre_norm:
            hidden_states = self.input_layernorm(hidden_states)

        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if not self.pre_norm:
            hidden_states = self.post_layernorm(hidden_states)

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class OPTModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype=None,
                 mapping=Mapping(),
                 pre_norm=True,
                 do_layer_norm_before=True,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0):
        super().__init__()
        self.do_layer_norm_before = do_layer_norm_before

        self.embedding = GPTEmbedding(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            dtype=dtype,
            use_prompt_tuning=use_prompt_tuning,
            tensor_parallel=mapping.tp_size if use_parallel_embedding else 1,
            tensor_parallel_group=mapping.tp_group
            if use_parallel_embedding else None,
            sharding_dim=embedding_sharding_dim,
            tp_rank=mapping.tp_rank)

        self.layers = ModuleList([
            OPTDecoderLayer(hidden_size=hidden_size,
                            num_attention_heads=num_heads,
                            max_position_embeddings=max_position_embeddings,
                            dtype=dtype,
                            hidden_act=hidden_act,
                            pre_norm=pre_norm,
                            tp_group=mapping.tp_group,
                            tp_size=mapping.tp_size) for _ in range(num_layers)
        ])

        if self.do_layer_norm_before:
            self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):

        hidden_states = self.embedding(input_ids, position_ids,
                                       prompt_embedding_table, prompt_tasks,
                                       prompt_vocab_size)

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        for layer, past in zip(self.layers, kv_cache_params.past_key_value):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)
            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if self.do_layer_norm_before:
            hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class OPTLMHeadModel(OPTModel, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 mapping=Mapping(),
                 pre_norm=True,
                 do_layer_norm_before=True,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 share_embedding_table=False):
        if share_embedding_table and mapping.tp_size > 1:
            if (not use_parallel_embedding) or (use_parallel_embedding and
                                                embedding_sharding_dim == 1):
                raise NotImplementedError(
                    'For multiple-processes cases, sharing the embedding table must set use_parallel_embedding=True and embedding_sharding_dim = 0'
                )

        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, dtype, mapping,
                         pre_norm, do_layer_norm_before, use_prompt_tuning,
                         use_parallel_embedding, embedding_sharding_dim)
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype
        self._dtype = self._kv_dtype
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tp_size = mapping.tp_size
        self._use_prompt_tuning = use_prompt_tuning

        share_weight = None
        if share_embedding_table:
            share_weight = self.embedding.vocab_embedding.weight

        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True,
                                    share_weight=share_weight)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):
        hidden_states = super().forward(input_ids, position_ids, use_cache,
                                        attention_mask, kv_cache_params,
                                        attention_params,
                                        prompt_embedding_table, prompt_tasks,
                                        prompt_vocab_size)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._kv_dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width,
                       prompt_embedding_table_size=32):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tp_size
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin

        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            num_heads,
            head_size,
            self._num_layers,
            self._kv_dtype,
            remove_input_padding,
            use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin)

        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        p_embedding_range = [
            1, prompt_embedding_table_size // 2, prompt_embedding_table_size
        ]
        num_tokens_range = [
            1, max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size)
        ]
        [1, 1, max_input_len]

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None
        if self._use_prompt_tuning:
            prompt_embedding_table = Tensor(name='prompt_embedding_table',
                                            dtype=self._dtype,
                                            shape=[-1, self._hidden_size],
                                            dim_range=OrderedDict([
                                                ('prompt_embedding_table_size',
                                                 [p_embedding_range]),
                                                ('hidden_size',
                                                 [self._hidden_size]),
                                            ]))
            if remove_input_padding:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size', [1]),
                                   ('input_len', [num_tokens_range]),
                               ]))
            else:
                tasks = Tensor(name='tasks',
                               dtype=trt.int32,
                               shape=[-1, 1],
                               dim_range=OrderedDict([
                                   ('batch_size', [bb_range]),
                                   ('input_len_for_task', [1]),
                               ]))
            prompt_vocab_size = Tensor(name='prompt_vocab_size',
                                       dtype=trt.int32,
                                       shape=[1],
                                       dim_range=OrderedDict([('size', [1])]))

        return (model_inputs['input_ids'], model_inputs['position_ids'], True,
                model_inputs['last_token_ids'], model_inputs['attention_mask'],
                KeyValueCacheParams(
                    past_key_value=model_inputs['past_key_value'],
                    host_past_key_value_lengths=model_inputs[
                        'host_past_key_value_lengths'],
                    cache_indirection=model_inputs['cache_indirection'],
                ),
                AttentionParams(
                    sequence_length=model_inputs['sequence_length'],
                    context_lengths=model_inputs['context_lengths'],
                    host_context_lengths=model_inputs['host_context_lengths'],
                    max_context_length=max_input_len,
                    host_request_types=model_inputs['host_request_types']),
                prompt_embedding_table, tasks, prompt_vocab_size)
