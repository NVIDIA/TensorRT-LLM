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
from copy import deepcopy
from types import SimpleNamespace

import tensorrt as trt

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantMode

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (PositionEmbeddingType, Tensor, concat,
                           gather_last_token_logits, shape)
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm,
                       RmsNorm)
from ...module import Module, ModuleList
from ..generation_mixin import GenerationMixin


class ChatGLMParams:
    apply_query_key_layer_scaling: bool = None
    apply_residual_connection_post_layernorm: bool = None
    dtype: str = None
    enable_debug_output: bool = None
    ffn_hidden_size: int = None
    hidden_act: str = None
    hidden_size: int = None
    linear_bias: bool = None
    logits_dtype: str = None
    mapping: Mapping = None
    max_batch_size: int = None
    max_beam_width: int = None
    max_input_len: int = None
    max_num_tokens: int = None
    max_output_len: int = None
    max_seq_length: int = None
    model_name: str = None
    norm_epsilon: float = None
    num_heads: int = None
    num_kv_heads: int = None
    num_layers: int = None
    qkv_bias: bool = None
    quant_mode: QuantMode = None
    rmsnorm: bool = None
    rotary_embedding_scaling: float = None
    tokens_per_block: int = None
    use_cache: bool = None
    vocab_size: int = None

    # default values
    default_config = {}
    default_config["chatglm_6b"] = SimpleNamespace(
        apply_query_key_layer_scaling=False,
        apply_residual_connection_post_layernorm=False,
        dtype="float16",
        ffn_hidden_size=16384,
        hidden_act='gelu',
        hidden_size=4096,
        linear_bias=True,
        logits_dtype="float16",
        mapping=Mapping(),
        max_batch_size=256,
        max_beam_width=1,
        max_input_len=512,
        max_num_tokens=256 * 512,
        max_output_len=512,
        max_seq_length=2048,
        norm_epsilon=1.0e-5,
        num_heads=32,
        num_kv_heads=32,
        num_layers=28,
        qkv_bias=True,
        quant_mode=QuantMode(0),
        rmsnorm=False,
        rotary_embedding_scaling=1.0,
        use_cache=True,
        vocab_size=130528,
    )
    default_config["chatglm2_6b"] = SimpleNamespace(
        apply_query_key_layer_scaling=False,
        apply_residual_connection_post_layernorm=False,
        dtype="float16",
        ffn_hidden_size=13696,
        hidden_act='swiglu',
        hidden_size=4096,
        linear_bias=False,
        logits_dtype="float16",
        mapping=Mapping(),
        max_batch_size=256,
        max_beam_width=1,
        max_input_len=512,
        max_num_tokens=256 * 512,
        max_output_len=512,
        max_seq_length=32768,
        norm_epsilon=1.0e-5,
        num_heads=32,
        num_kv_heads=2,
        num_layers=28,
        qkv_bias=True,
        quant_mode=QuantMode(0),
        rmsnorm=True,
        rotary_embedding_scaling=1.0,
        use_cache=True,
        vocab_size=65024,
    )
    default_config["chatglm3_6b"] = default_config["chatglm2_6b"]
    default_config["chatglm3_6b_base"] = default_config["chatglm2_6b"]
    default_config["chatglm2_6b_32k"] = deepcopy(default_config["chatglm2_6b"])
    default_config["chatglm2_6b_32k"].rotary_embedding_scaling = 50.0
    default_config["chatglm3_6b_32k"] = default_config["chatglm2_6b_32k"]
    default_config["glm_10b"] = SimpleNamespace(
        apply_query_key_layer_scaling=False,
        apply_residual_connection_post_layernorm=False,
        dtype="float16",
        ffn_hidden_size=16384,
        hidden_act='gelu',
        hidden_size=4096,
        linear_bias=True,
        logits_dtype="float16",
        mapping=Mapping(),
        max_batch_size=256,
        max_beam_width=1,
        max_input_len=1024,
        max_num_tokens=256 * 1024,
        max_output_len=1024,
        max_seq_length=2048,
        norm_epsilon=1.0e-5,
        num_heads=32,
        num_kv_heads=32,
        num_layers=48,
        qkv_bias=True,
        quant_mode=QuantMode(0),
        rmsnorm=False,
        rotary_embedding_scaling=1.0,
        use_cache=True,
        vocab_size=50304,
    )
    default_config["glm_2b"] = deepcopy(default_config["glm_10b"])
    default_config["glm_2b"].num_layers = 36
    default_config["glm_2b"].num_heads = 32
    default_config["glm_2b"].hidden_size = 2048
    default_config["glm_2b"].ffn_hidden_size = 8192
    default_config["glm_10b_chinese"] = deepcopy(default_config["glm_10b"])
    default_config["glm_10b_chinese"].vocab_size = 50048

    def __init__(self, **args):

        for key, value in args.items():
            assert key in dir(
                self), f"{key} is not in configuration of ChatGLMHeadModel"
            if value is not None:
                self.__setattr__(key, value)
        assert self.model_name is not None, "model_name must be set for ChatGLMHeadModel"

        # fill other parameters as default values
        for key, value in self.default_config[self.model_name].__dict__.items():
            if self.__getattribute__(key) is None:
                self.__setattr__(key, value)

    def report(self):
        for key, value in self.__dict__.items():
            print(f"{key} = {value}")


class ChatGLMDecoderLayer(Module):

    def __init__(self, layer_id, config):

        super().__init__()

        rotary_embedding_scaling = None
        self.model_name = config.model_name
        self.rotary_embedding_base = 10000.0
        self.use_cache = config.use_cache

        # Save for Smooth Quantization
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.max_seq_length = config.max_seq_length
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_layers = config.num_layers
        self.tp_group = config.mapping.tp_group
        self.tp_size = config.mapping.tp_size
        self.hidden_act = config.hidden_act
        self.bias = config.qkv_bias
        self.dense_bias = config.linear_bias

        if self.model_name in ["chatglm_6b"]:
            self.alpha = (2 * config.num_layers)**0.5
            self.norm = LayerNorm
            self.attention_mask_type = AttentionMaskType.bidirectional
            self.position_embedding_type = PositionEmbeddingType.chatglm
        elif config.model_name in [
                "chatglm2_6b", "chatglm2_6b_32k", "chatglm3_6b",
                "chatglm3_6b_base", "chatglm3_6b_32k"
        ]:
            self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
            self.norm = RmsNorm if config.rmsnorm else LayerNorm
            self.attention_mask_type = AttentionMaskType.causal
            self.position_embedding_type = PositionEmbeddingType.rope_gptj
            if config.model_name in ["chatglm2_6b_32k", "chatglm3_6b_32k"]:
                self.rotary_embedding_base *= config.rotary_embedding_scaling
        elif config.model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
            self.norm = LayerNorm
            self.attention_mask_type = AttentionMaskType.bidirectionalglm
            self.position_embedding_type = PositionEmbeddingType.learned_absolute

        self.pre_norm = self.norm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            elementwise_affine=True,
            dtype=config.dtype,
        )

        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_seq_length,
            num_layers=config.num_layers,
            apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
            attention_mask_type=self.attention_mask_type,
            bias=config.qkv_bias,
            dtype=config.dtype,
            position_embedding_type=self.position_embedding_type,
            rotary_embedding_base=self.rotary_embedding_base,
            rotary_embedding_scaling=rotary_embedding_scaling,
            rotary_embedding_percentage=0.5,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.rank,
            quant_mode=config.quant_mode,
            q_scaling=1.0,
            cross_attention=False,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
            instance_id=layer_id * 2,
            dense_bias=config.linear_bias,
        )

        self.mlp = MLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            hidden_act=config.hidden_act,
            bias=config.linear_bias,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
            instance_id=layer_id * 2 + 1,
        )

        self.post_norm = self.norm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            elementwise_affine=True,
            dtype=config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor = None,  # only used in ChatGLM-6B
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):

        norm_output = self.pre_norm(hidden_states)

        attention_output = self.attention(
            hidden_states=norm_output,
            attention_mask=None,
            use_cache=self.use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            encoder_output=None,
            position_embedding=position_ids,
        )

        if self.use_cache:
            attention_output, presents = attention_output

        if self.model_name in ["chatglm_6b"]:
            residual = norm_output

            norm_input = residual * self.alpha + attention_output

            norm_output = self.post_norm(norm_input)

            mlp_output = self.mlp(norm_output)

            residual = norm_output

            output = residual * self.alpha + mlp_output

        elif self.model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
                "glm_2b",
                "glm_10b",
                "glm_10b_chinese",
        ]:
            residual = norm_output if self.apply_residual_connection_post_layernorm else hidden_states

            norm_input = residual + attention_output

            norm_output = self.post_norm(norm_input)

            mlp_output = self.mlp(norm_output)

            residual = norm_output if self.apply_residual_connection_post_layernorm else norm_input

            output = residual + mlp_output

        return (output, presents) if self.use_cache else output


class ChatGLMModel(Module):

    def __init__(self, config):

        super().__init__()

        self.model_name = config.model_name
        self.use_cache = config.use_cache
        if config.model_name in [
                "chatglm_6b",
                "glm_2b",
                "glm_10b",
                "glm_10b_chinese",
        ]:
            self.norm = LayerNorm
        elif config.model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            self.norm = RmsNorm
            self.hidden_size = config.hidden_size

        self.vocab_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            tp_size=1,  #config.mapping.tp_size,
            tp_group=None,  #config.mapping.tp_group,
            sharding_dim=0,
            tp_rank=0,  #config.mapping.rank,
            instance_id=config.num_layers * 2,
        )

        if config.model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            self.position_embedding = Embedding(
                config.max_seq_length + 1,
                config.hidden_size,
                dtype=config.dtype,
                tp_size=1,  #config.mapping.tp_size,
                tp_group=None,  #config.mapping.tp_group,
                sharding_dim=0,
                tp_rank=0,  #config.mapping.rank,
                instance_id=config.num_layers * 2,
            )
            self.block_embedding = Embedding(
                config.max_seq_length + 1,
                config.hidden_size,
                dtype=config.dtype,
                tp_size=1,  #config.mapping.tp_size,
                tp_group=None,  #config.mapping.tp_group,
                sharding_dim=0,
                tp_rank=0,  #config.mapping.rank,
                instance_id=config.num_layers * 2,
            )

        self.layers = ModuleList(
            ChatGLMDecoderLayer(i, config) for i in range(config.num_layers))

        self.final_norm = self.norm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            elementwise_affine=True,
            dtype=config.dtype,
        )

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,  # only used in ChatGLM-6B
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):

        hidden_states = self.vocab_embedding(input_ids)

        if self.model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            position_ids_list = position_ids.split(1, dim=1)
            position_embedding = self.position_embedding(position_ids_list[0])
            block_embedding = self.block_embedding(position_ids_list[1])
            position_embedding = position_embedding + block_embedding

            position_embedding = position_embedding.view(
                concat([
                    shape(input_ids, 0),
                    shape(input_ids, 1),
                    self.hidden_size,
                ]))

            hidden_states = hidden_states + position_embedding

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if self.use_cache:
            presents = []

        for layer, past, pointer, host_pointer, max_attention_window_size in zip(
                self.layers, kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_attention_window_sizes):
            layer_output = layer(
                hidden_states,
                position_ids,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=max_attention_window_size,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
            )

            if self.use_cache:
                hidden_states = layer_output[0]
                presents.append(layer_output[1])

        hidden_states = self.final_norm(hidden_states)

        return (hidden_states,
                tuple(presents)) if self.use_cache else hidden_states


class ChatGLMHeadModel(ChatGLMModel, GenerationMixin):

    def __init__(
        self,
        apply_query_key_layer_scaling: bool = None,
        apply_residual_connection_post_layernorm: bool = None,
        dtype: str = None,
        enable_debug_output: bool = None,
        ffn_hidden_size: int = None,
        hidden_act: str = None,
        hidden_size: int = None,
        linear_bias: bool = None,
        logits_dtype: str = None,
        mapping: Mapping = None,
        max_batch_size: int = None,
        max_beam_width: int = None,
        max_input_len: int = None,
        max_output_len: int = None,
        max_num_tokens: int = None,
        max_seq_length: int = None,
        model_name: str = None,
        norm_epsilon: float = None,
        num_heads: int = None,
        num_kv_heads: int = None,
        num_layers: int = None,
        qkv_bias: bool = None,
        quant_mode: QuantMode = None,
        rmsnorm: bool = None,
        rotary_embedding_scaling: float = None,
        tokens_per_block: int = None,
        use_cache: bool = None,
        vocab_size: int = None,
        max_position_embeddings: int = None,
    ):

        # for benchmark scripts
        if max_seq_length is None and max_position_embeddings is not None:
            max_seq_length = max_position_embeddings

        config = ChatGLMParams(
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            apply_residual_connection_post_layernorm=
            apply_residual_connection_post_layernorm,
            dtype=dtype,
            enable_debug_output=enable_debug_output,
            ffn_hidden_size=ffn_hidden_size,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            linear_bias=linear_bias,
            logits_dtype=logits_dtype,
            mapping=mapping,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_num_tokens=max_num_tokens,
            max_seq_length=max_seq_length,
            model_name=model_name,
            norm_epsilon=norm_epsilon,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            qkv_bias=qkv_bias,
            quant_mode=quant_mode,
            rmsnorm=rmsnorm,
            rotary_embedding_scaling=rotary_embedding_scaling,
            tokens_per_block=tokens_per_block,
            use_cache=use_cache,
            vocab_size=vocab_size,
        )

        super().__init__(config)

        if isinstance(config.dtype, str):
            self.kv_dtype = str_dtype_to_trt(config.dtype)
        else:
            assert isinstance(config.dtype, trt.DataType)
            self.kv_dtype = config.dtype
        self.dtype = self.kv_dtype

        if isinstance(config.logits_dtype, str):
            self.logits_dtype = str_dtype_to_trt(config.logits_dtype)
        else:
            assert isinstance(config.logits_dtype, trt.DataType)
            self.logits_dtype = config.logits_dtype

        if config.quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('int8')
        elif config.quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt('fp8')

        self.hidden_size = config.hidden_size
        self.mapping = config.mapping
        self.max_batch_size = config.max_batch_size
        self.max_beam_width = config.max_beam_width
        self.max_input_len = config.max_input_len
        self.max_num_tokens = config.max_num_tokens
        self.max_output_len = config.max_output_len
        self.max_seq_length = config.max_seq_length
        self.model_name = config.model_name
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_layers = config.num_layers
        self.tokens_per_block = config.tokens_per_block
        self.use_cache = config.use_cache

        self.lm_head = ColumnLinear(
            in_features=self.hidden_size,
            out_features=pad_vocab_size(config.vocab_size,
                                        self.mapping.tp_size),
            bias=False,
            dtype=self.dtype,
            tp_group=self.mapping.tp_group,
            tp_size=self.mapping.tp_size,
            gather_output=True,
        )

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,  # used in chatglm_6b / glm_*
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
        lm_logits.mark_output('logits', self.logits_dtype)

        if self.use_cache and default_net(
        ).plugin_config.paged_kv_cache == False:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self.kv_dtype)
            return (lm_logits, presents)

        return lm_logits

    def prepare_inputs(
        self,
        max_batch_size: int = None,
        max_input_len: int = None,
        max_output_len: int = None,
        use_cache: bool = True,
        max_beam_width: int = 1,
        gather_context_logits: bool = False,
        gather_generation_logits: bool = False,
        use_custom_all_reduce: bool = False,
    ):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        position_encoding_2d = (self.model_name in [
            "chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"
        ])

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_new_tokens=max_output_len,
            num_kv_heads=self.num_kv_heads,
            head_size=self.hidden_size // self.num_heads,
            num_layers=self.num_layers,
            kv_dtype=self.kv_dtype,
            remove_input_padding=default_net(
            ).plugin_config.remove_input_padding,
            use_gpt_attention_plugin=default_net().plugin_config.
            gpt_attention_plugin,
            use_gemm_plugin=default_net().plugin_config.gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=default_net().plugin_config.paged_kv_cache,
            tokens_per_block=self.tokens_per_block,
            gather_context_logits=gather_context_logits,
            gather_generation_logits=gather_generation_logits,
            dtype=self.kv_dtype,
            num_heads=self.num_heads,
            mapping=self.mapping,
            max_num_tokens=self.max_num_tokens,
            prompt_embedding_table_size=0,
            position_encoding_2d=position_encoding_2d,
        )

        return (
            model_inputs['input_ids'], model_inputs['position_ids'],
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
                host_request_types=model_inputs['host_request_types'],
            ))
