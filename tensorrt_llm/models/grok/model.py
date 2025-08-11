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

from ..._utils import pad_vocab_size
from ...functional import Tensor, recv, send
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, MoeConfig, PositionEmbeddingType, RmsNorm)
from ...lora_helper import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig, QuantConfig)


class GrokDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
            quant_mode=config.quant_mode,
            max_attn_value=config.max_attn_value)

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size
        self.post_attn_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                           eps=config.norm_epsilon,
                                           dtype=config.dtype)

        self.post_mlp_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                          eps=config.norm_epsilon,
                                          dtype=config.dtype)
        mlp_kwargs = {}
        assert config.moe_num_experts > 1, "Grok model is a MoE model."
        ClsMLP = MOE
        moe_config = MoeConfig(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            normalization_mode=config.moe_normalization_mode).validate()
        mlp_kwargs = {
            "moe_config": moe_config,
            "mapping": config.mapping,
        }
        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=mlp_hidden_size,
                          hidden_act=config.hidden_act,
                          dtype=config.dtype,
                          bias=config.mlp_bias,
                          tp_group=config.mapping.tp_group,
                          tp_size=config.mapping.tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents = attention_output

        attention_output = self.post_attn_layernorm(attention_output)
        hidden_states = residual + attention_output

        residual_attn = hidden_states

        # regular llama/mixtral layers
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual_attn + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GrokModel(Module):

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(GrokDecoderLayer, config)

        self.embedding_multiplier_scale = config.embedding_multiplier_scale

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
            hidden_states *= 78.38367176906169
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
            spec_decoding_params=spec_decoding_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GrokForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = GrokModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('mlp_bias', False)
        config.set_if_not_exist('attn_bias', False)
        config.set_if_not_exist('rotary_base', 10000.0)
        config.set_if_not_exist('rotary_scaling', None)
        config.set_if_not_exist('moe_num_experts', 0)
        config.set_if_not_exist('moe_top_k', 0)
        config.set_if_not_exist('moe_normalization_mode',
                                MoeConfig.ExpertScaleNormalizationMode.NONE)

    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir,
                          dtype='float16',
                          mapping: Optional[Mapping] = None,
                          **kwargs):
        from . import convert
        if mapping is None:
            mapping = Mapping()
        grok = convert.from_hugging_face(
            cls,
            hf_model_dir,
            dtype,
            mapping=mapping,
            quantization=kwargs.get('quantization', QuantConfig()),
            override_fields=kwargs.get('override_fields', {}),
            skip_loading_weights=kwargs.get('skip_loading_weights', False),
            preloaded_model=kwargs.get('preloaded_model', None))
        return grok

    def default_plugin_config(self, **kwargs):
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.set_weight_only_groupwise_quant_matmul_plugin()
        return plugin_config

    @classmethod
    def quantize(
        cls,
        hf_model_dir,
        output_dir,
        quant_config: QuantConfig,
        *,
        dtype='float16',
        mapping: Optional[Mapping] = None,
        calib_batches=512,
        calib_batch_size=1,
        random_seed=1234,
        tokenizer_max_seq_length=2048,
        **kwargs,
    ):
        pass

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config)
