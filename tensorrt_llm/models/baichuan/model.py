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

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import Tensor, gather_last_token_logits
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       PromptTuningEmbedding, RmsNorm)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


class BaichuanDecoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 position_embedding_type,
                 num_kv_heads=None,
                 dtype=None,
                 attention_mask_type=AttentionMaskType.causal,
                 hidden_act='silu',
                 mlp_hidden_size=None,
                 tp_group=None,
                 tp_size=1,
                 tp_rank=0,
                 quant_mode=QuantMode(0)):
        super().__init__()
        # used for quantizing model
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.mlp_hidden_size = mlp_hidden_size
        self.attention_mask_type = attention_mask_type
        self.position_embedding_type = position_embedding_type
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size,
                                       dtype=dtype)

        assert position_embedding_type is not None
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=attention_mask_type,
            bias=False,
            position_embedding_type=position_embedding_type,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_mode=quant_mode)
        if not mlp_hidden_size:
            self.mlp_hidden_size = hidden_size * 4
        self.mlp = GatedMLP(hidden_size=hidden_size,
                            ffn_hidden_size=self.mlp_hidden_size,
                            hidden_act=hidden_act,
                            dtype=dtype,
                            bias=False,
                            tp_group=tp_group,
                            tp_size=tp_size,
                            quant_mode=quant_mode)
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

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
                 num_kv_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 position_embedding_type,
                 dtype,
                 mlp_hidden_size=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0),
                 use_prompt_tuning: bool = False):
        super().__init__()
        self.mapping = mapping
        self.num_layers = num_layers
        self.use_prompt_tuning = use_prompt_tuning
        EmbeddingCls = PromptTuningEmbedding if use_prompt_tuning else Embedding
        self.vocab_embedding = EmbeddingCls(vocab_size,
                                            hidden_size,
                                            dtype=dtype)

        self.layers = ModuleList([
            BaichuanDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                position_embedding_type=position_embedding_type,
                num_kv_heads=num_kv_heads,
                dtype=dtype,
                hidden_act=hidden_act,
                mlp_hidden_size=mlp_hidden_size,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                quant_mode=quant_mode) for _ in range(num_layers)
        ])

        self.ln_f = RmsNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
    ):
        ptuning_args = []
        if self.use_prompt_tuning:
            ptuning_args = [
                prompt_embedding_table, prompt_tasks, prompt_vocab_size
            ]

        hidden_states = self.vocab_embedding(
            input_ids,
            *ptuning_args,
        )

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
                attention_mask=attention_mask,
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


class BaichuanForCausalLM(BaichuanModel, GenerationMixin):

    def __init__(self,
                 num_layers,
                 num_heads,
                 num_kv_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 position_embedding_type,
                 dtype,
                 logits_dtype='float32',
                 mlp_hidden_size=None,
                 mapping=Mapping(),
                 quant_mode=QuantMode(0),
                 use_prompt_tuning: bool = False):
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype

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

        if isinstance(logits_dtype, str):
            self._logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self._logits_dtype = logits_dtype

        self.quant_mode = quant_mode

        super().__init__(num_layers, num_heads, num_kv_heads, hidden_size,
                         vocab_size, hidden_act, max_position_embeddings,
                         position_embedding_type, dtype, mlp_hidden_size,
                         mapping, quant_mode, use_prompt_tuning)
        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=True)

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
    ):
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
        lm_logits.mark_output('logits', self._logits_dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_seq_len,
        use_cache,
        max_beam_width,
        max_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
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

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_size=head_size,
            num_layers=self.num_layers,
            kv_dtype=self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_heads,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
            prompt_embedding_table_size=prompt_embedding_table_size,
        )

        return (
            model_inputs['input_ids'],
            model_inputs['position_ids'],
            True,
            model_inputs['last_token_ids'],
            model_inputs['attention_mask'],
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
            model_inputs['prompt_embedding_table'],
            model_inputs['tasks'],
            model_inputs['prompt_vocab_size'],
        )
