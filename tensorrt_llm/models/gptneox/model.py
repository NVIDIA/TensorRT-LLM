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
import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor,
                           gather_last_token_logits)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin


class GPTNeoXDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers,
                 rotary_dim,
                 dtype=None,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.causal,
                 hidden_act='relu',
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.post_attention_layernorm = LayerNorm(normalized_shape=hidden_size,
                                                  dtype=dtype)

        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rotary_embedding_percentage=rotary_dim /
            (hidden_size // num_attention_heads),
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=True,
            tp_group=tp_group,
            tp_size=tp_size)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       tp_group=tp_group,
                       tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        if not default_net(
        ).plugin_config.layernorm_plugin and trt.__version__[:3] == '8.6':
            raise AssertionError(
                "You need to enable the LayerNorm plugin for GPT-NeoX with TensorRT 8.6. Please set plugin_config.layernorm_plugin"
            )
        residual = hidden_states

        input_layernorm_output = self.input_layernorm(hidden_states)
        post_attention_layernorm_output = self.post_attention_layernorm(
            hidden_states)

        attention_output = self.attention(input_layernorm_output,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          norm_before_bmm1=True)

        if use_cache:
            attention_output, presents = attention_output

        feed_forward_hidden_states = self.mlp(post_attention_layernorm_output)
        hidden_states = attention_output + feed_forward_hidden_states + residual
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTNeoXModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 mapping=Mapping(),
                 apply_query_key_layer_scaling=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0):
        super().__init__()
        self.vocab_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
            tp_size=mapping.tp_size if use_parallel_embedding else 1,
            tp_group=mapping.tp_group if use_parallel_embedding else None,
            sharding_dim=embedding_sharding_dim,
            tp_rank=mapping.rank)

        self.layers = ModuleList([
            GPTNeoXDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                num_layers=num_layers,
                rotary_dim=rotary_dim,
                dtype=dtype,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                attention_mask_type=AttentionMaskType.causal,
                hidden_act=hidden_act,
                position_embedding_type=position_embedding_type,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size) for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        hidden_states = self.vocab_embedding(input_ids)

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        for layer, past, max_attention_window_size in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.host_max_attention_window_sizes):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTNeoXForCausalLM(GPTNeoXModel, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 mapping=Mapping(),
                 apply_query_key_layer_scaling=False,
                 use_parallel_embedding=False,
                 embedding_sharding_dim=0):
        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tp_size = mapping.tp_size
        self._use_parallel_embedding = use_parallel_embedding
        self._embedding_sharding_dim = embedding_sharding_dim

        super().__init__(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            rotary_dim=rotary_dim,
            dtype=dtype,
            position_embedding_type=position_embedding_type,
            mapping=mapping,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            use_parallel_embedding=use_parallel_embedding,
            embedding_sharding_dim=embedding_sharding_dim)

        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                kv_cache_params=None,
                attention_params=None):
        hidden_states = super().forward(input_ids, position_ids, use_cache,
                                        kv_cache_params, attention_params)

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

    def prepare_inputs(self, max_batch_size, max_input_len, max_new_tokens,
                       use_cache, max_beam_width):
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

        return (
            model_inputs['input_ids'], model_inputs['position_ids'], True,
            model_inputs['last_token_ids'],
            KeyValueCacheParams(
                past_key_value=model_inputs['past_key_value'],
                host_past_key_value_lengths=model_inputs[
                    'host_past_key_value_lengths'],
                host_max_attention_window_sizes=model_inputs[
                    'host_max_attention_window_sizes'],
                host_sink_token_length=model_inputs['host_sink_token_length'],
                cache_indirection=model_inputs['cache_indirection'],
            ),
            AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types']))
