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
import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor,
                           gather_last_token_logits, gpt_attention)
from ...layers import (MLP, AttentionMaskType, AttentionParams, ColumnLinear,
                       Embedding, KeyValueCacheParams, LayerNorm, RowLinear)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


class GPTNeoXAttention(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 rotary_dim,
                 max_position_embeddings,
                 dtype=None,
                 multi_block_mode=False,
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 quant_mode=QuantMode(0),
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads // tp_size
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.position_embedding_type = position_embedding_type
        self.multi_block_mode = multi_block_mode
        self.multi_query_mode = False
        self.quant_mode = quant_mode

        if self.quant_mode.has_int8_kv_cache():
            self.kv_quantization_scale = Parameter(shape=(1, ), dtype='float32')
            self.kv_dequantization_scale = Parameter(shape=(1, ),
                                                     dtype='float32')
        else:
            self.register_parameter('kv_quantization_scale', None)
            self.register_parameter('kv_dequantization_scale', None)

        self.qkv = ColumnLinear(in_features=hidden_size,
                                out_features=hidden_size * 3,
                                bias=True,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False,
                                dtype=dtype)
        self.dense = RowLinear(in_features=hidden_size,
                               out_features=hidden_size,
                               bias=True,
                               dtype=dtype,
                               tp_group=tp_group,
                               tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        if not default_net().plugin_config.gpt_attention_plugin:
            raise ValueError(
                'GPT-NeoX RoPE is only supported with GPTAttention plugin')
        qkv = self.qkv(hidden_states)

        assert attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)

        context, past_key_value = gpt_attention(
            tensor=qkv,
            past_key_value=kv_cache_params.get_first_past_key_value(),
            sequence_length=attention_params.sequence_length,
            host_past_key_value_lengths=kv_cache_params.
            host_past_key_value_lengths,
            context_lengths=attention_params.context_lengths,
            cache_indirection=kv_cache_params.cache_indirection,
            host_request_types=attention_params.host_request_types,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            hidden_size_per_head=self.attention_head_size,
            q_scaling=1.0,
            rotary_embedding_dim=self.rotary_dim,
            position_embedding_type=self.position_embedding_type,
            multi_block_mode=self.multi_block_mode,
            kv_orig_quant_scale=self.kv_quantization_scale,
            kv_quant_orig_scale=self.kv_dequantization_scale,
            kv_cache_quant_mode=self.quant_mode,
            max_context_length=attention_params.max_context_length,
            host_context_lengths=attention_params.host_context_lengths)

        context = self.dense(context)

        if use_cache:
            return (context, past_key_value)

        return context


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

        self.attention = GPTNeoXAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            position_embedding_type=position_embedding_type,
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
                                          attention_params=attention_params)

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
        self.embedding = Embedding(
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
        hidden_states = self.embedding(input_ids)

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        for layer, past in zip(self.layers, kv_cache_params.past_key_value):
            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
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

        return (model_inputs['input_ids'], model_inputs['position_ids'], True,
                model_inputs['last_token_ids'],
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
                    host_request_types=model_inputs['host_request_types']))
