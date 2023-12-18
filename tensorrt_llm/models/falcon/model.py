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
from typing import Optional, Union

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (Tensor, allreduce, gather_last_token_logits, recv,
                           send)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm,
                       PositionEmbeddingType)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...quantization import QuantMode
from ..generation_mixin import GenerationMixin


class FalconDecoderLayer(Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings,
        num_attention_kv_heads=None,
        dtype=None,
        hidden_act='gelu',
        quant_mode=QuantMode(0),
        mlp_hidden_size=None,
        bias=True,
        use_alibi=True,
        new_decoder_architecture=False,
        parallel_attention=False,
        layernorm_epsilon=1e-5,
        tp_group=None,
        tp_size=1,
        tp_rank=0,
        layer_id=None,
    ):
        super().__init__()
        self._layer_id = layer_id  # useful for debugging
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_kv_heads = num_attention_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         eps=layernorm_epsilon,
                                         dtype=dtype)
        if use_alibi:
            # Note falcon models will also scale alibi with inv_sqrt_Dh
            position_embedding_type = PositionEmbeddingType.alibi_with_scale
        else:
            position_embedding_type = PositionEmbeddingType.rope_gpt_neox

        self.new_decoder_architecture = new_decoder_architecture
        self.parallel_attn = parallel_attention
        if self.is_parallel_attention:
            # Not to apply allreduce inside the Attention/MLP layers.
            # allreduce applies after those layer.
            tp_group = None
        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_attention_kv_heads,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=bias,
            position_embedding_type=position_embedding_type,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_mode=quant_mode,
            instance_id=2 * layer_id,
        )

        if mlp_hidden_size is None:
            mlp_hidden_size = hidden_size * 4

        if self.new_decoder_architecture:
            # Layernorm before MLP.
            self.mlp_layernorm = LayerNorm(normalized_shape=hidden_size,
                                           eps=layernorm_epsilon,
                                           dtype=dtype)
        else:
            self.mlp_layernorm = None
        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=hidden_act,
            dtype=dtype,
            bias=bias,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=quant_mode,
            instance_id=2 * layer_id + 1,
        )
        if self.is_parallel_attention:
            self.post_layernorm = None
        else:
            self.post_layernorm = LayerNorm(normalized_shape=hidden_size,
                                            dtype=dtype)

    @property
    def is_parallel_attention(self):
        return self.new_decoder_architecture or self.parallel_attn

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                all_reduce_workspace=None):
        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        if self.new_decoder_architecture:
            mlp_ln_output = self.mlp_layernorm(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        input_ln_output = hidden_states
        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          workspace=all_reduce_workspace)

        if use_cache:
            attention_output, presents = attention_output

        if not self.new_decoder_architecture:
            if self.parallel_attn:
                hidden_states = input_ln_output
            else:
                hidden_states = residual + attention_output
                residual = hidden_states
                hidden_states = self.post_layernorm(hidden_states)
        else:
            hidden_states = mlp_ln_output

        hidden_states = self.mlp(hidden_states, all_reduce_workspace)

        if self.is_parallel_attention:
            hidden_states = hidden_states + attention_output
            if self.tp_size > 1:
                hidden_states = allreduce(hidden_states, self.tp_group,
                                          all_reduce_workspace, self._layer_id)

        hidden_states = residual + hidden_states
        if use_cache:
            return hidden_states, presents
        return hidden_states


class FalconModel(Module):

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        vocab_size: int,
        hidden_act: int,
        max_position_embeddings: int,
        dtype: Optional[Union[str, trt.DataType]] = None,
        mapping: Mapping = Mapping(),
        num_kv_heads: Optional[int] = None,
        mlp_hidden_size: Optional[int] = None,
        bias: bool = True,
        quant_mode: QuantMode = QuantMode(0),
        use_alibi: bool = True,
        parallel_attention: bool = False,
        new_decoder_architecture: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.mapping = mapping

        # Falcon variants
        self.parallel_attention = parallel_attention
        self.new_decoder_architecture = new_decoder_architecture

        self.quant_mode = quant_mode
        assert isinstance(dtype, (str, trt.DataType))
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            self.dtype = dtype
        if self.quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')
        else:
            self.kv_dtype = self.dtype

        if self.mapping.is_first_pp_rank():
            self.embedding = Embedding(vocab_size, hidden_size, dtype=dtype)

        self.layers = ModuleList([
            FalconDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                dtype=dtype,
                bias=bias,
                quant_mode=self.quant_mode,
                hidden_act=hidden_act,
                num_attention_kv_heads=self.num_kv_heads,
                mlp_hidden_size=mlp_hidden_size,
                use_alibi=use_alibi,
                parallel_attention=parallel_attention,
                new_decoder_architecture=new_decoder_architecture,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                layer_id=i,
            ) for i in self.mapping.pp_layers(num_layers)
        ])

        if self.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(normalized_shape=hidden_size, dtype=dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                all_reduce_workspace=None):

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

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
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection),
                attention_params=attention_params,
                all_reduce_workspace=all_reduce_workspace)

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


class FalconForCausalLM(FalconModel, GenerationMixin):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 hidden_size: int,
                 vocab_size: int,
                 max_position_embeddings: int,
                 hidden_act: str = 'gelu',
                 dtype: Optional[Union[str, trt.DataType]] = None,
                 num_kv_heads: Optional[int] = None,
                 mlp_hidden_size: Optional[int] = None,
                 bias: bool = True,
                 quant_mode: QuantMode = QuantMode(0),
                 use_alibi: bool = True,
                 parallel_attention: bool = False,
                 new_decoder_architecture: bool = False,
                 logits_dtype: Union[str, trt.DataType] = 'float32',
                 mapping=Mapping()):
        super().__init__(num_layers=num_layers,
                         num_heads=num_heads,
                         hidden_size=hidden_size,
                         vocab_size=vocab_size,
                         hidden_act=hidden_act,
                         max_position_embeddings=max_position_embeddings,
                         dtype=dtype,
                         num_kv_heads=num_kv_heads,
                         mlp_hidden_size=mlp_hidden_size,
                         bias=bias,
                         quant_mode=quant_mode,
                         mapping=mapping,
                         use_alibi=use_alibi,
                         parallel_attention=parallel_attention,
                         new_decoder_architecture=new_decoder_architecture)

        # TODO: For compatibility to quantization modules. Remove it later.
        self._num_layers = num_layers

        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(
                hidden_size,
                vocab_size_padded,
                bias=False,
                dtype=dtype,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                gather_output=True,
            )
        if isinstance(logits_dtype, str):
            self.logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self.logits_dtype = logits_dtype

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                all_reduce_workspace=None):

        hidden_states = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            hidden_states=hidden_states,
            all_reduce_workspace=all_reduce_workspace)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states,
                last_token_ids,
                default_net().plugin_config.remove_input_padding,
            )

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
                return lm_logits, presents
            else:
                return hidden_states, presents
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            else:
                return hidden_states

    def prepare_inputs(self,
                       max_batch_size: int,
                       max_input_len: int,
                       max_new_tokens: int,
                       use_cache: bool,
                       max_beam_width: int = 1,
                       max_num_tokens: int = None):
        '''

        @brief: Prepare inputs Tensors for the model, the given sizes are used
        to determine the ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads

        plugin_config = default_net().plugin_config
        use_gpt_attention_plugin = plugin_config.gpt_attention_plugin
        remove_input_padding = plugin_config.remove_input_padding
        use_gemm_plugin = plugin_config.gemm_plugin
        paged_kv_cache = plugin_config.paged_kv_cache
        tokens_per_block = plugin_config.tokens_per_block
        use_custom_all_reduce = plugin_config.use_custom_all_reduce

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_new_tokens=max_new_tokens,
            num_kv_heads=self.num_kv_heads,
            head_size=head_size,
            num_layers=self.num_layers,
            kv_dtype=self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_heads,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens)

        return (
            model_inputs['input_ids'],
            model_inputs['position_ids'],
            use_cache,
            model_inputs['last_token_ids'],
            model_inputs['attention_mask'],
            KeyValueCacheParams(
                past_key_value=model_inputs['past_key_value'],
                host_past_key_value_lengths=model_inputs[
                    'host_past_key_value_lengths'],
                host_max_attention_window_sizes=model_inputs[
                    'host_max_attention_window_sizes'],
                kv_cache_block_pointers=model_inputs[
                    'kv_cache_block_pointers_list'],
                host_kv_cache_block_pointers=model_inputs[
                    'host_kv_cache_block_pointers_list'],
                cache_indirection=model_inputs['cache_indirection']),
            AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types']),
            model_inputs['hidden_states_input'],
            model_inputs['all_reduce_workspace'],
        )
