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
from ...functional import Tensor, gather_last_token_logits
from ...layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                       GatedMLP, RmsNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin


class BaichuanDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 position_embedding_type,
                 dtype=None,
                 hidden_act='silu',
                 mlp_hidden_size=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0):
        super().__init__()
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size,
                                       dtype=dtype)

        assert position_embedding_type is not None
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type=position_embedding_type,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank)
        if not mlp_hidden_size:
            mlp_hidden_size = hidden_size * 4
        self.mlp = GatedMLP(hidden_size=hidden_size,
                            ffn_hidden_size=mlp_hidden_size,
                            hidden_act=hidden_act,
                            dtype=dtype,
                            bias=False,
                            tp_group=tp_group,
                            tp_size=tp_size)
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                past_key_value=None,
                sequence_length=None,
                host_past_key_value_lengths=None,
                use_cache=False,
                cache_indirection=None,
                kv_cache_block_pointers=None,
                context_lengths=None,
                host_context_lengths=None,
                host_request_types=None,
                max_context_length: int = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            host_past_key_value_lengths=host_past_key_value_lengths,
            use_cache=use_cache,
            cache_indirection=cache_indirection,
            kv_cache_block_pointers=kv_cache_block_pointers,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            host_request_types=host_request_types,
            max_context_length=max_context_length)

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


class BaichuanModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 position_embedding_type,
                 dtype,
                 mlp_hidden_size=None,
                 mapping=Mapping()):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            BaichuanDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                position_embedding_type=position_embedding_type,
                dtype=dtype,
                hidden_act=hidden_act,
                mlp_hidden_size=mlp_hidden_size,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank) for _ in range(num_layers)
        ])

        self.ln_f = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                past_key_value=None,
                sequence_length=None,
                host_past_key_value_lengths=None,
                use_cache=False,
                attention_mask=None,
                cache_indirection=None,
                kv_cache_block_pointers=None,
                context_lengths=None,
                host_context_lengths=None,
                host_request_types=None,
                max_context_length: int = None):

        hidden_states = self.vocab_embedding(input_ids)

        if past_key_value is None:
            past_key_value = tuple([None] * len(self.layers))

        if use_cache:
            presents = []

        for layer, past, pointers in zip(self.layers, past_key_value,
                                         kv_cache_block_pointers):
            hidden_states = layer(
                hidden_states,
                past_key_value=past,
                sequence_length=sequence_length,
                host_past_key_value_lengths=host_past_key_value_lengths,
                use_cache=use_cache,
                attention_mask=attention_mask,
                cache_indirection=cache_indirection,
                kv_cache_block_pointers=pointers,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types,
                max_context_length=max_context_length)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class BaichuanForCausalLM(BaichuanModel, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 position_embedding_type,
                 dtype,
                 mlp_hidden_size=None,
                 mapping=Mapping()):
        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype
        self._num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = mapping.tp_size
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings,
                         position_embedding_type, dtype, mlp_hidden_size,
                         mapping)
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
                past_key_value=None,
                sequence_length=None,
                host_past_key_value_lengths=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                cache_indirection=None,
                kv_cache_block_pointers=None,
                context_lengths=None,
                host_context_lengths=None,
                host_request_types=None,
                max_context_length: int = None):
        hidden_states = super().forward(
            input_ids, position_ids, past_key_value, sequence_length,
            host_past_key_value_lengths, use_cache, attention_mask,
            cache_indirection, kv_cache_block_pointers, context_lengths,
            host_context_lengths, host_request_types, max_context_length)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._kv_dtype)

        if use_cache:
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
                       paged_kv_cache: bool = False,
                       tokens_per_block: int = 64):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads
        num_heads_kv = (self.num_kv_heads + self.tp_size - 1) // self.tp_size

        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin

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
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block)

        return (model_inputs['input_ids'], model_inputs['position_ids'],
                model_inputs['past_key_value'], model_inputs['sequence_length'],
                model_inputs['host_past_key_value_lengths'], True,
                model_inputs['last_token_ids'], model_inputs['attention_mask'],
                model_inputs['cache_indirection'],
                model_inputs['kv_cache_block_pointers_list'],
                model_inputs['context_lengths'],
                model_inputs['host_context_lengths'],
                model_inputs['host_request_types'], max_input_len)
