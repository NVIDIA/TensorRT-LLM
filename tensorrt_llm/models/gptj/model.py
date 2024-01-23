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
from ...functional import (PositionEmbeddingType, Tensor, allreduce,
                           gather_last_token_logits)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


class GPTJDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 hidden_act='relu',
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         dtype=dtype)

        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rotary_embedding_percentage=rotary_dim /
            (hidden_size // num_attention_heads),
            position_embedding_type=PositionEmbeddingType.rope_gptj,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            tp_group=None,
            tp_size=tp_size,
            quant_mode=quant_mode)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=hidden_size * 4,
                       hidden_act=hidden_act,
                       dtype=dtype,
                       tp_group=None,
                       tp_size=tp_size,
                       quant_mode=quant_mode)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        if not default_net(
        ).plugin_config.layernorm_plugin and trt.__version__[:3] == '8.6':
            raise AssertionError(
                "You need to enable the LayerNorm plugin for GPT-J with TensorRT 8.6. Please set plugin_config.layernorm_plugin"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output
        attention_output = attention_output

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attention_output + feed_forward_hidden_states
        if self.tp_size > 1:
            hidden_states = allreduce(hidden_states, self.tp_group)
        hidden_states = hidden_states + residual

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTJModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        super().__init__()
        self.mapping = mapping
        self.vocab_embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            GPTJDecoderLayer(hidden_size=hidden_size,
                             num_attention_heads=num_heads,
                             max_position_embeddings=max_position_embeddings,
                             rotary_dim=rotary_dim,
                             dtype=dtype,
                             hidden_act=hidden_act,
                             tp_group=mapping.tp_group,
                             tp_size=mapping.tp_size,
                             quant_mode=quant_mode) for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        hidden_states = self.vocab_embedding(input_ids)

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        for layer, past, pointer, host_pointer, max_attention_window_size in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
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
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GPTJForCausalLM(GPTJModel, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 rotary_dim,
                 dtype,
                 logits_dtype='float32',
                 mapping=Mapping(),
                 quant_mode=QuantMode(0)):
        if isinstance(dtype, str):
            self._dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._dtype = dtype
        self._kv_dtype = dtype
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
        super().__init__(num_layers, num_heads, hidden_size, vocab_size,
                         hidden_act, max_position_embeddings, rotary_dim, dtype,
                         mapping, quant_mode)
        self._vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        self.lm_head = ColumnLinear(hidden_size,
                                    self._vocab_size_padded,
                                    bias=True,
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
        hidden_states = super().forward(input_ids, use_cache, kv_cache_params,
                                        attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self._logits_dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_seq_len,
                       use_cache,
                       max_beam_width,
                       max_num_tokens: int = None):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''
        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads_kv = self._num_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net(
        ).plugin_config.use_custom_all_reduce

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=num_heads_kv,
            head_size=head_size,
            num_layers=self._num_layers,
            kv_dtype=self._kv_dtype,
            num_heads=self._num_heads,
            dtype=self._dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens)

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
                kv_cache_block_pointers=model_inputs[
                    'kv_cache_block_pointers_list'],
                host_kv_cache_block_pointers=model_inputs[
                    'host_kv_cache_block_pointers_list'],
                cache_indirection=model_inputs['cache_indirection'],
            ),
            AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types']))
