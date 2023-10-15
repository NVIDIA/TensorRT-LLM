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
from ...functional import (Tensor, gather_last_token_logits,
                           is_gated_activation, non_gated_version)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       LayerNorm, PositionEmbeddingType, PromptTuningEmbedding)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


def MLPFactory(hidden_size,
               ffn_hidden_size,
               hidden_act,
               bias=True,
               dtype=None,
               tp_group=None,
               tp_size=1,
               quant_mode=QuantMode(0),
               instance_id: int = 0):
    MLPClass = GatedMLP if is_gated_activation(hidden_act) else MLP
    hidden_act = non_gated_version(hidden_act)
    return MLPClass(hidden_size, ffn_hidden_size, hidden_act, bias, dtype,
                    tp_group, tp_size, quant_mode, instance_id)


class GPTEmbedding(Module):

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position_embeddings,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 dtype=None,
                 use_prompt_tuning=False,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 sharding_dim=0,
                 tp_rank=None):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.use_prompt_tuning = use_prompt_tuning

        EmbeddingCls = PromptTuningEmbedding if use_prompt_tuning else Embedding
        self.vocab_embedding = EmbeddingCls(vocab_size,
                                            hidden_size,
                                            dtype=dtype,
                                            tp_size=tensor_parallel,
                                            tp_group=tensor_parallel_group,
                                            sharding_dim=sharding_dim,
                                            tp_rank=tp_rank)

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.position_embedding = Embedding(max_position_embeddings,
                                                hidden_size,
                                                dtype=dtype)

    def forward(self,
                input_ids,
                position_ids,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None):
        args = []
        if self.use_prompt_tuning:
            args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size]
        x = self.vocab_embedding(input_ids, *args)
        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            x = x + self.position_embedding(position_ids)

        return x


class GPTDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers,
                 dtype=None,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.causal,
                 hidden_act='relu',
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 quant_mode=QuantMode(0),
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 multi_query_mode=False,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 instance_id: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.dtype = dtype
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_mask_type = attention_mask_type
        self.hidden_act = hidden_act
        self.position_embedding_type = position_embedding_type
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            1 if multi_query_mode else num_attention_heads,
            max_position_embeddings,
            num_layers,
            apply_query_key_layer_scaling,
            dtype=dtype,
            attention_mask_type=attention_mask_type,
            position_embedding_type=position_embedding_type,
            rotary_embedding_percentage=rotary_embedding_percentage,
            bias=bias,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache(),
            quant_mode=quant_mode,
            instance_id=2 * instance_id)

        if inter_size is None:
            inter_size = hidden_size * 4

        self.mlp = MLPFactory(hidden_size=hidden_size,
                              ffn_hidden_size=inter_size,
                              hidden_act=hidden_act,
                              dtype=dtype,
                              bias=bias,
                              tp_group=tp_group,
                              tp_size=tp_size,
                              quant_mode=quant_mode,
                              instance_id=2 * instance_id + 1)
        self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                        dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                workspace=None):

        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          workspace=workspace)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, workspace=workspace)

        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype=None,
                 mapping=Mapping(),
                 apply_query_key_layer_scaling=False,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0),
                 multi_query_mode=False,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0):
        super().__init__()
        self.mapping = mapping

        self.embedding = GPTEmbedding(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            position_embedding_type,
            dtype=dtype,
            use_prompt_tuning=use_prompt_tuning,
            tensor_parallel=mapping.tp_size if use_parallel_embedding else 1,
            tensor_parallel_group=mapping.tp_group
            if use_parallel_embedding else None,
            sharding_dim=embedding_sharding_dim,
            tp_rank=mapping.tp_rank)

        self.layers = ModuleList([
            GPTDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                num_layers=num_layers,
                dtype=dtype,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                attention_mask_type=AttentionMaskType.causal,
                hidden_act=hidden_act,
                position_embedding_type=position_embedding_type,
                rotary_embedding_percentage=rotary_embedding_percentage,
                multi_query_mode=multi_query_mode,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                inter_size=inter_size,
                bias=bias,
                quant_mode=quant_mode,
                instance_id=i,
            ) for i in range(num_layers)
        ])

        self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

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
                workspace=None):

        hidden_states = self.embedding(input_ids, position_ids,
                                       prompt_embedding_table, prompt_tasks,
                                       prompt_vocab_size)

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        for layer, past, pointer in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    kv_cache_block_pointers=[pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
                workspace=workspace)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTLMHeadModel(GPTModel, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 logits_dtype='float32',
                 mapping=Mapping(),
                 apply_query_key_layer_scaling=False,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0),
                 multi_query_mode=False,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0,
                 share_embedding_table=False):

        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype

        if share_embedding_table and mapping.tp_size > 1:
            if (not use_parallel_embedding) or (use_parallel_embedding and
                                                embedding_sharding_dim == 1):
                raise NotImplementedError(
                    'For multiple-processes cases, sharing the embedding table must set use_parallel_embedding=True and embedding_sharding_dim = 0'
                )

        self._dtype = self._kv_dtype
        self.quant_mode = quant_mode
        if quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('fp8')

        if isinstance(logits_dtype, str):
            self._logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self._logits_dtype = logits_dtype

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tp_size = mapping.tp_size
        self._multi_query_mode = multi_query_mode
        self._use_prompt_tuning = use_prompt_tuning

        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, dtype, mapping,
                         apply_query_key_layer_scaling, position_embedding_type,
                         rotary_embedding_percentage, inter_size, bias,
                         quant_mode, multi_query_mode, use_prompt_tuning,
                         use_parallel_embedding, embedding_sharding_dim)
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)

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
                prompt_vocab_size=None,
                workspace=None):

        hidden_states = super().forward(input_ids, position_ids, use_cache,
                                        attention_mask, kv_cache_params,
                                        attention_params,
                                        prompt_embedding_table, prompt_tasks,
                                        prompt_vocab_size, workspace)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._logits_dtype)

        if use_cache:
            if default_net().plugin_config.paged_kv_cache == False:
                for i, present in enumerate(presents):
                    present.mark_output(f'present_key_value_{i}',
                                        self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width: int = 1,
                       max_num_tokens: int = None,
                       prompt_embedding_table_size: int = 128,
                       gather_all_token_logits: bool = False):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tp_size
        num_heads_kv = 1 if self._multi_query_mode else num_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net(
        ).plugin_config.use_custom_all_reduce

        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            num_heads_kv,
            head_size,
            self._num_layers,
            self._kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            gather_all_token_logits=gather_all_token_logits,
            max_num_tokens=max_num_tokens)

        bb_range_cxt = [1, (max_batch_size + 1) // 2, max_batch_size]
        bb_range_gen = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        _p_embedding_range = [
            1, prompt_embedding_table_size // 2, prompt_embedding_table_size
        ]
        [1, (max_input_len + 1) // 2, max_input_len]
        _num_tokens_range = [
            1, max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size)
        ]

        enable_two_optimization_profiles = False
        if use_gpt_attention_plugin == False or use_gemm_plugin == False:
            use_in_flight_batching = use_gpt_attention_plugin and remove_input_padding and paged_kv_cache
            enable_two_optimization_profiles = not use_in_flight_batching
        if enable_two_optimization_profiles:
            bb_range = [bb_range_cxt, bb_range_gen]
            p_embedding_range = [_p_embedding_range, _p_embedding_range]
            num_tokens_range = [_num_tokens_range, _num_tokens_range]
        else:
            bb_range = [bb_range_gen]
            p_embedding_range = [_p_embedding_range]
            num_tokens_range = [_num_tokens_range]

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None
        if self._use_prompt_tuning:
            prompt_embedding_table = Tensor(
                name='prompt_embedding_table',
                dtype=self._dtype,
                shape=[-1, self._hidden_size],
                dim_range=OrderedDict([
                    ('prompt_embedding_table_size', p_embedding_range),
                    ('hidden_size', [self._hidden_size, self._hidden_size] if
                     enable_two_optimization_profiles else [self._hidden_size]),
                ]))
            if remove_input_padding:
                tasks = Tensor(
                    name='tasks',
                    dtype=trt.int32,
                    shape=[1, -1],
                    dim_range=OrderedDict([
                        ('batch_size_fake',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                        ('input_len_task', num_tokens_range),
                    ]))
            else:
                tasks = Tensor(
                    name='tasks',
                    dtype=trt.int32,
                    shape=[-1, 1],
                    dim_range=OrderedDict([
                        ('batch_size_beam_width', bb_range),
                        ('broadcast_dim',
                         [1, 1] if enable_two_optimization_profiles else [1]),
                    ]))
            prompt_vocab_size = Tensor(
                name='prompt_vocab_size',
                dtype=trt.int32,
                shape=[1],
                dim_range=OrderedDict([
                    ('size',
                     [1, 1] if enable_two_optimization_profiles else [1])
                ]))

        all_reduce_workspace = None
        if use_custom_all_reduce:
            # 3 (= buffer + signals_in + signals_out)
            workspace_size = 3 * self.mapping.tp_size
            all_reduce_workspace = Tensor(
                name='all_reduce_workspace',
                dtype=trt.int64,
                shape=[workspace_size],
                dim_range=OrderedDict([
                    ('all_reduce_size', [workspace_size, workspace_size]
                     if enable_two_optimization_profiles else [workspace_size])
                ]))

        return (model_inputs['input_ids'], model_inputs['position_ids'], True,
                model_inputs['last_token_ids'], model_inputs['attention_mask'],
                KeyValueCacheParams(
                    past_key_value=model_inputs['past_key_value'],
                    host_past_key_value_lengths=model_inputs[
                        'host_past_key_value_lengths'],
                    kv_cache_block_pointers=model_inputs[
                        'kv_cache_block_pointers_list'],
                    cache_indirection=model_inputs['cache_indirection'],
                ),
                AttentionParams(
                    sequence_length=model_inputs['sequence_length'],
                    context_lengths=model_inputs['context_lengths'],
                    host_context_lengths=model_inputs['host_context_lengths'],
                    max_context_length=max_input_len,
                    host_request_types=model_inputs['host_request_types']),
                prompt_embedding_table, tasks, prompt_vocab_size,
                all_reduce_workspace)
