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

from tensorrt_llm.plugin.plugin import init_all_reduce_helper

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (Tensor, gather_last_token_logits, partial, recv,
                           send, unary)
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       PositionEmbeddingType, RmsNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin

log = partial(unary, op=trt.UnaryOperation.LOG)
ceil = partial(unary, op=trt.UnaryOperation.CEIL)


class QWenBlock(Module):

    def __init__(self,
                 layer_id,
                 hidden_size,
                 seq_length,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers,
                 dtype=None,
                 attention_mask_type=AttentionMaskType.causal,
                 apply_query_key_layer_scaling=False,
                 hidden_act='silu',
                 position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                 rotary_base=10000.0,
                 rotary_scaling=None,
                 quant_mode=QuantMode(0),
                 mlp_hidden_size=None,
                 neox_rotary_style=True,
                 bias=False,
                 tp_group=None,
                 tp_size=1,
                 rms_norm_eps=1e-06):
        super().__init__()
        self._layer_id = layer_id  # useful for debugging
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.mlp_hidden_size = mlp_hidden_size
        self.neox_rotary_style = neox_rotary_style
        self.bias = bias
        self.hidden_act = hidden_act
        self.dtype = dtype
        self.attention_mask_type = attention_mask_type
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.position_embedding_type = position_embedding_type

        self.ln_1 = RmsNorm(normalized_shape=hidden_size,
                            eps=rms_norm_eps,
                            dtype=dtype)

        self.attention = Attention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            dtype=self.dtype,
            attention_mask_type=self.attention_mask_type,
            position_embedding_type=self.position_embedding_type,
            rotary_embedding_base=rotary_base,
            rotary_embedding_scaling=rotary_scaling,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
            quant_mode=quant_mode,
            dense_bias=bias)
        if not mlp_hidden_size:
            mlp_hidden_size = hidden_size * 4

        self.mlp = GatedMLP(hidden_size=hidden_size,
                            ffn_hidden_size=mlp_hidden_size // 2,
                            hidden_act=hidden_act,
                            dtype=dtype,
                            bias=False,
                            tp_group=tp_group,
                            tp_size=tp_size,
                            quant_mode=quant_mode,
                            instance_id=2 * layer_id + 1)
        self.ln_2 = RmsNorm(normalized_shape=hidden_size,
                            eps=rms_norm_eps,
                            dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attention_output = self.attention(
            hidden_states,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class QWenModel(Module):

    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_size,
        seq_length,
        vocab_size,
        hidden_act,
        max_position_embeddings,
        dtype,
        mlp_hidden_size=None,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        neox_rotary_style=True,
        bias=False,
        rotary_base=10000.0,
        rotary_scaling=None,
        mapping=Mapping(),
        quant_mode=QuantMode(0),
        use_parallel_embedding=False,
        embedding_sharding_dim=0,
        rms_norm_eps=1e-06,
    ):
        super().__init__()
        self.mapping = mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
                dtype=dtype,
                tp_size=mapping.tp_size if use_parallel_embedding else 1,
                tp_group=mapping.tp_group if use_parallel_embedding else None,
                sharding_dim=embedding_sharding_dim,
                tp_rank=mapping.tp_rank)

        self.layers = ModuleList([
            QWenBlock(layer_id=i,
                      hidden_size=hidden_size,
                      seq_length=seq_length,
                      num_attention_heads=num_heads,
                      num_layers=num_layers,
                      max_position_embeddings=max_position_embeddings,
                      dtype=dtype,
                      hidden_act=hidden_act,
                      quant_mode=quant_mode,
                      mlp_hidden_size=mlp_hidden_size,
                      position_embedding_type=position_embedding_type,
                      rotary_base=rotary_base,
                      rotary_scaling=rotary_scaling,
                      neox_rotary_style=neox_rotary_style,
                      bias=bias,
                      tp_group=mapping.tp_group,
                      tp_size=mapping.tp_size,
                      rms_norm_eps=rms_norm_eps)
            for i in self.mapping.pp_layers(num_layers)
        ])

        self.ln_f = RmsNorm(normalized_shape=hidden_size,
                            eps=rms_norm_eps,
                            dtype=dtype)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None):

        if kv_cache_params.past_key_value is None:
            tuple([None] * len(self.layers))

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
        self.register_network_output(f"embd", hidden_states)

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

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class QWenForCausalLM(QWenModel, GenerationMixin):

    def __init__(
        self,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size,
        seq_length,
        vocab_size,
        hidden_act,
        max_position_embeddings,
        dtype,
        logits_dtype="float32",
        mlp_hidden_size=None,
        neox_rotary_style=True,
        rotary_base=10000.0,
        rotary_scaling=None,
        mapping=Mapping(),
        quant_mode=QuantMode(0),
        use_parallel_embedding=False,
        embedding_sharding_dim=0,
        rms_norm_eps=1e-06,
    ):
        init_all_reduce_helper()
        self.mapping = mapping
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype
        if isinstance(logits_dtype, str):
            self.logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self.logits_dtype = logits_dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = mapping.tp_size

        self.kv_dtype = self.dtype
        if quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')
        self.quant_mode = quant_mode
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim

        super().__init__(num_layers=num_layers,
                         num_heads=num_heads,
                         hidden_size=hidden_size,
                         seq_length=seq_length,
                         vocab_size=vocab_size,
                         hidden_act=hidden_act,
                         max_position_embeddings=max_position_embeddings,
                         dtype=dtype,
                         mlp_hidden_size=mlp_hidden_size,
                         neox_rotary_style=neox_rotary_style,
                         rotary_base=rotary_base,
                         rotary_scaling=rotary_scaling,
                         mapping=mapping,
                         quant_mode=quant_mode,
                         use_parallel_embedding=use_parallel_embedding,
                         embedding_sharding_dim=embedding_sharding_dim,
                         rms_norm_eps=rms_norm_eps)
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(hidden_size,
                                        vocab_size_padded,
                                        bias=False,
                                        dtype=dtype,
                                        tp_group=mapping.tp_group,
                                        tp_size=mapping.tp_size,
                                        gather_output=True)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None):
        hidden_states = super().forward(input_ids, position_ids, use_cache,
                                        kv_cache_params, attention_params,
                                        hidden_states)
        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids,
                default_net().plugin_config.remove_input_padding)

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output('logits', self.logits_dtype)
        else:
            hidden_states.mark_output('hidden_states_output', self.dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(self.mapping.pp_layers(self.num_layers),
                                  presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_new_tokens,
        use_cache,
        max_beam_width: int = 1,
        max_num_tokens: int = None,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads
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
            self.num_kv_heads,
            head_size,
            self.num_layers,
            self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_heads,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
        )

        return (
            model_inputs['input_ids'],
            model_inputs['position_ids'],
            True,
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
                host_request_types=model_inputs['host_request_types']),
            model_inputs['hidden_states_input'],
        )
