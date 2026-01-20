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

from typing import Optional, Union

import torch
from transformers import AutoModel

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import Tensor, concat, shape
from ...layers import (MLP, Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, KeyValueCacheParams, LayerNorm,
                       RmsNorm)
from ...mapping import Mapping
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import GLM_ARCH1_VERSIONS, GLM_ARCH2_VERSIONS, ChatGLMConfig
from .convert import load_weights_from_hf_model


class ChatGLMDecoderLayer(Module):

    def __init__(self, config: ChatGLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.chatglm_version = config.chatglm_version

        hidden_size = config.hidden_size
        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        layernorm_epsilon = config.norm_epsilon

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.alpha = (2 * config.num_hidden_layers)**0.5
        norm_cls = RmsNorm if config.rmsnorm else LayerNorm

        if config.chatglm_version == 'glm':
            attention_mask_type = AttentionMaskType.bidirectionalglm
        elif config.chatglm_version == 'chatglm':
            attention_mask_type = AttentionMaskType.bidirectional
        elif config.chatglm_version in GLM_ARCH2_VERSIONS:
            attention_mask_type = AttentionMaskType.causal

        self.input_layernorm = norm_cls(
            normalized_shape=hidden_size,
            eps=layernorm_epsilon,
            elementwise_affine=True,
            dtype=dtype,
        )

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
            attention_mask_type=attention_mask_type,
            bias=config.add_qkv_bias,
            dense_bias=config.add_bias_linear,
            dtype=config.dtype,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            rotary_embedding_percentage=config.rotary_pct,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_mode=config.quant_mode,
            q_scaling=1.0,
            cross_attention=False,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
            cp_rank=config.mapping.cp_rank,
            cp_size=config.mapping.cp_size,
            cp_group=config.mapping.cp_group,
        )

        mlp_hidden_size = hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=config.hidden_act,
            bias=config.add_bias_linear,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
        )

        self.post_layernorm = norm_cls(
            normalized_shape=hidden_size,
            eps=layernorm_epsilon,
            elementwise_affine=True,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        position_ids: Tensor = None,  # only used in ChatGLM-6B
        use_cache: bool = False,
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):
        norm_output = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states=norm_output,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            encoder_output=None,
            position_embedding=position_ids,
        )

        if use_cache:
            attention_output, presents = attention_output

        if self.chatglm_version == 'chatglm':
            residual = norm_output

            norm_input = residual * self.alpha + attention_output

            norm_output = self.post_layernorm(norm_input)

            mlp_output = self.mlp(norm_output)

            residual = norm_output

            output = residual * self.alpha + mlp_output

        else:
            residual = norm_output if self.apply_residual_connection_post_layernorm else hidden_states

            norm_input = residual + attention_output

            norm_output = self.post_layernorm(norm_input)

            mlp_output = self.mlp(norm_output)

            residual = norm_output if self.apply_residual_connection_post_layernorm else norm_input

            output = residual + mlp_output

        if use_cache:
            return (output, presents)
        return output


class ChatGLMModel(Module):

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.chatglm_version = config.chatglm_version
        norm_cls = RmsNorm if config.rmsnorm else LayerNorm

        self.vocab_embedding = Embedding(config.vocab_size,
                                         config.hidden_size,
                                         dtype=config.dtype)

        if config.chatglm_version == 'glm':
            self.position_embedding = Embedding(
                config.max_position_embeddings + 1,
                config.hidden_size,
                dtype=config.dtype,
            )
            self.block_embedding = Embedding(
                config.max_position_embeddings + 1,
                config.hidden_size,
                dtype=config.dtype,
            )

        self.layers = DecoderLayerList(ChatGLMDecoderLayer, config)

        self.ln_f = norm_cls(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            elementwise_affine=True,
            dtype=config.dtype,
        )

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,  # only used in ChatGLM-6B
        use_cache: bool = False,
        attention_mask: Tensor = None,
        kv_cache_params: KeyValueCacheParams = None,
        attention_params: AttentionParams = None,
    ):
        hidden_states = self.vocab_embedding(input_ids)

        if self.chatglm_version == 'glm':
            if default_net().plugin_config.remove_input_padding:
                position_ids_list = position_ids.split(1, dim=0)
            else:
                position_ids_list = position_ids.split(1, dim=1)

            position_embedding = self.position_embedding(position_ids_list[0])
            block_embedding = self.block_embedding(position_ids_list[1])
            position_embedding = position_embedding + block_embedding

            if default_net().plugin_config.remove_input_padding:
                position_embedding = position_embedding.view(
                    concat([
                        shape(position_embedding, 1),
                        shape(position_embedding, 2)
                    ]))
            else:
                position_embedding = position_embedding.view(
                    concat([
                        shape(position_embedding, 0),
                        shape(position_embedding, 2),
                        shape(position_embedding, 3),
                    ]))

            hidden_states = hidden_states + position_embedding

        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params,
                                    position_ids=position_ids)

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class ChatGLMForCausalLM(DecoderModelForCausalLM):
    config_class = ChatGLMConfig

    def __init__(self, config: ChatGLMConfig):
        transformer = ChatGLMModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        lm_head = ColumnLinear(config.hidden_size,
                               vocab_size_padded,
                               bias=False,
                               dtype=config.dtype,
                               tp_group=config.mapping.tp_group,
                               tp_size=config.mapping.tp_size,
                               gather_output=True)
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        ''' Create a LLaMAForCausalLM object from give parameters
        '''
        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        trust_remote_code = kwargs.pop('trust_remote_code', True)

        config = ChatGLMConfig.from_hugging_face(hf_model_or_dir,
                                                 dtype=dtype,
                                                 mapping=mapping,
                                                 quant_config=quant_config,
                                                 **kwargs)
        if config.chatglm_version == 'glm':
            device_map = 'cuda' if not load_model_on_cpu else 'cpu'
        else:
            device_map = 'auto' if not load_model_on_cpu else 'cpu'
        hf_model = AutoModel.from_pretrained(
            hf_model_or_dir,
            trust_remote_code=trust_remote_code,
            dtype='auto' if config.chatglm_version != 'glm' else getattr(
                torch, config.dtype),
            device_map=device_map)
        weights = load_weights_from_hf_model(hf_model, config)

        model = cls(config)
        model.load(weights)
        return model

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'auto',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        device: str = 'cuda',
        calib_dataset: str = 'cnn_dailymail',
        calib_batches: int = 512,
        calib_batch_size: int = 1,
        calib_max_seq_length: int = 512,
        random_seed: int = 1234,
        tokenizer_max_seq_length: int = 2048,
        **kwargs,
    ):
        if quant_config._requires_modelopt_quantization:
            # modelopt quantization flow
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=dtype,
                             mapping=mapping,
                             quant_config=quant_config,
                             device=device,
                             calib_dataset=calib_dataset,
                             calib_batches=calib_batches,
                             calib_batch_size=calib_batch_size,
                             calib_max_seq_length=calib_max_seq_length,
                             random_seed=random_seed,
                             tokenizer_max_seq_length=tokenizer_max_seq_length)
        elif quant_config._requires_calibration:
            # non-modelopt quantization flow
            from . import convert

            config = ChatGLMConfig.from_hugging_face(hf_model_dir,
                                                     dtype=dtype,
                                                     mapping=mapping,
                                                     quant_config=quant_config,
                                                     **kwargs)
            convert.quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             calib_dataset=calib_dataset,
                             device=device)
        else:
            raise ValueError(
                f"The quant_config ({quant_config}) does not require calibration, try {cls.__name__}.from_hugging_face instead."
            )

    def prepare_inputs(self, *args, **kwargs):
        """See `PretrainedModel.prepare_inputs` for the detailed parameter list.
        """
        if self.transformer.chatglm_version in GLM_ARCH1_VERSIONS:
            position_encoding_2d = True
        else:
            position_encoding_2d = False
        return super().prepare_inputs(*args,
                                      **kwargs,
                                      position_encoding_2d=position_encoding_2d)
