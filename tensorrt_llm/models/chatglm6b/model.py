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
import argparse

import numpy as np
import tensorrt as trt

from ..._common import default_net
from ..._utils import (pad_vocab_size, str_dtype_to_np, str_dtype_to_trt,
                       trt_dtype_to_np)
from ...functional import (PositionEmbeddingType, Tensor, concat,
                           gather_last_token_logits, shape)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm)
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin


class ChatGLM6BDecoderLayer(Module):

    def __init__(self, args):

        super().__init__()

        self.use_cache = args.use_cache

        self.input_layernorm = LayerNorm(
            normalized_shape=args.hidden_size,
            eps=args.layernorm_epsilon,
            dtype=args.dtype,
        )

        self.attention = Attention(
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_heads,
            num_kv_heads=args.num_heads,
            max_position_embeddings=args.max_seq_length,
            num_layers=args.num_layers,
            apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
            attention_mask_type=AttentionMaskType.bidirectional,
            bias=args.bias,
            dtype=args.dtype,
            position_embedding_type=PositionEmbeddingType.chatglm,
            use_int8_kv_cache=args.quant_mode.has_int8_kv_cache(),
            tp_group=args.mapping.tp_group,
            tp_size=args.mapping.tp_size,
            multi_block_mode=args.multi_block_mode,
            quant_mode=args.quant_mode,
        )

        self.mlp = MLP(
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            hidden_act=args.hidden_act,
            dtype=args.dtype,
            bias=args.bias,
            tp_group=args.mapping.tp_group,
            tp_size=args.mapping.tp_size,
        )

        self.post_layernorm = LayerNorm(
            normalized_shape=args.hidden_size,
            eps=args.layernorm_epsilon,
            dtype=args.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embedding: Tensor,
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):

        layernorm_output = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states=layernorm_output,
            attention_mask=None,
            use_cache=self.use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            encoder_output=None,
            workspace=None,
            position_embedding=position_embedding,
        )

        if self.use_cache:
            attention_output, presents = attention_output

        layernorm_input = layernorm_output * 7.484375 + attention_output

        layernorm_output = self.post_layernorm(layernorm_input)

        mlp_output = self.mlp(layernorm_output)

        output = layernorm_output * 7.484375 + mlp_output

        return (output, presents) if self.use_cache else output


class ChatGLM6BModel(Module):

    def __init__(self, args):

        super().__init__()

        self.use_cache = args.use_cache
        self.half_head_size = args.hidden_size // args.num_heads // 2

        self.embedding = Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.hidden_size,
            dtype=args.dtype,
        )

        # pre-compute weight of position embedding manually
        if isinstance(args.dtype, trt.DataType):
            np_dtype = trt_dtype_to_np(args.dtype)
        else:
            np_dtype = str_dtype_to_np(args.dtype)

        inv_freq = 10**(-1 / 16 *
                        np.arange(0, 64, 2, dtype=np.float32)).reshape(1, 32)
        valueTable = np.matmul(
            np.arange(args.max_seq_length, dtype=np.float32).reshape(-1, 1),
            np.tile(inv_freq, [1, 2]),
        ).reshape(args.max_seq_length, 64)

        self.position_embedding_cos = Embedding(
            num_embeddings=args.max_seq_length,
            embedding_dim=self.half_head_size,
            dtype=args.dtype,
        )
        self.position_embedding_sin = Embedding(
            num_embeddings=args.max_seq_length,
            embedding_dim=self.half_head_size,
            dtype=args.dtype,
        )

        self.position_embedding_cos.weight.value = np.cos(valueTable).astype(
            np_dtype)
        self.position_embedding_sin.weight.value = np.sin(valueTable).astype(
            np_dtype)

        self.layers = ModuleList(
            ChatGLM6BDecoderLayer(args) for _ in range(args.num_layers))

        self.final_layernorm = LayerNorm(
            normalized_shape=args.hidden_size,
            eps=args.layernorm_epsilon,
            dtype=args.dtype,
        )

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):

        batch_size = shape(input_ids, 0)
        input_len = shape(input_ids, 1)

        hidden_states = self.embedding(input_ids)

        position_embedding_cos = self.position_embedding_cos(position_ids)
        position_embedding_sin = self.position_embedding_sin(position_ids)

        position_embedding_cos0, position_embedding_cos1 = position_embedding_cos.split(
            1, dim=1)
        position_embedding_sin0, position_embedding_sin1 = position_embedding_sin.split(
            1, dim=1)

        position_embedding_cos0 = position_embedding_cos0.view(
            concat([batch_size, input_len, 1, self.half_head_size]))
        position_embedding_cos1 = position_embedding_cos1.view(
            concat([batch_size, input_len, 1, self.half_head_size]))
        position_embedding_sin0 = position_embedding_sin0.view(
            concat([batch_size, input_len, 1, self.half_head_size]))
        position_embedding_sin1 = position_embedding_sin1.view(
            concat([batch_size, input_len, 1, self.half_head_size]))

        position_embedding = [
            position_embedding_cos0, position_embedding_cos1,
            position_embedding_sin0, position_embedding_sin1
        ]

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = tuple([None] * len(self.layers))

        if self.use_cache:
            presents = []

        for layer, past_key_value, kv_cache_block_pointers in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers):
            layer_output = layer(
                hidden_states,
                position_embedding,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past_key_value],
                    kv_cache_block_pointers=[kv_cache_block_pointers],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
            )

            if self.use_cache:
                hidden_states = layer_output[0]
                presents.append(layer_output[1])

        hidden_states = self.final_layernorm(hidden_states)

        return (hidden_states,
                tuple(presents)) if self.use_cache else hidden_states


class ChatGLM6BHeadModel(ChatGLM6BModel, GenerationMixin):

    def __init__(self, **args):

        if "args" not in args.keys():
            argNamespace = argparse.Namespace()
            for key, value in args.items():
                argNamespace.__setattr__(key, value)
            # Other default values
            argNamespace.bias = True
            argNamespace.ffn_hidden_size = 16384
            argNamespace.layernorm_epsilon = 1.0e-5
            argNamespace.max_seq_length = argNamespace.max_position_embeddings
            argNamespace.multi_block_mode = False
            argNamespace.num_kv_heads = 32
            argNamespace.use_cache = True
            args = argNamespace
        else:
            args = args["args"]

        self.init(args)

    def init(self, args):

        super().__init__(args)

        if isinstance(args.dtype, str):
            self.kv_dtype = str_dtype_to_trt(args.dtype)
        else:
            assert isinstance(args.dtype, trt.DataType)
            self.kv_dtype = args.dtype
        self.dtype = self.kv_dtype

        if args.quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif args.quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.num_layers = args.num_layers
        self.tp_size = args.mapping.tp_size
        self.use_cache = args.use_cache

        self.lm_head = ColumnLinear(
            in_features=self.hidden_size,
            out_features=pad_vocab_size(args.vocab_size, self.tp_size),
            bias=False,
            dtype=self.dtype,
            tp_group=args.mapping.tp_group,
            tp_size=self.tp_size,
            gather_output=True,
        )

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,
        last_token_ids: Tensor = None,
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):

        hidden_states = super().forward(
            input_ids,
            position_ids,
            kv_cache_params,
            attention_params,
        )

        if self.use_cache:
            hidden_states, presents = hidden_states

        hidden_states = gather_last_token_logits(
            hidden_states, last_token_ids,
            default_net().plugin_config.remove_input_padding)

        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self.dtype)

        if self.use_cache and default_net(
        ).plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(
        self,
        max_batch_size: int = 0,
        max_input_len: int = 0,
        max_new_tokens: int = 0,
        use_cache: bool = True,
        max_beam_width: int = 1,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_new_tokens=max_new_tokens,
            num_kv_heads=self.num_kv_heads // self.tp_size,
            head_size=self.hidden_size // self.num_heads,
            num_layers=self.num_layers,
            kv_dtype=self.kv_dtype,
            remove_input_padding=default_net(
            ).plugin_config.remove_input_padding,
            use_gpt_attention_plugin=default_net().plugin_config.
            gpt_attention_plugin,
            use_gemm_plugin=default_net().plugin_config.gemm_plugin,
            is_chatglm6b=True,
        )

        return (model_inputs['input_ids'], model_inputs['position_ids'],
                model_inputs['last_token_ids'],
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
                    host_request_types=model_inputs['host_request_types'],
                ))
