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

import os
from typing import Optional

from ..._utils import pad_vocab_size
from ...functional import Tensor, non_gated_version, recv, send
from ...layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                       GatedMLP, PositionEmbeddingType, RmsNorm, SharedMoE)
from ...mapping import Mapping
from ...module import Module
from ...plugin import init_all_reduce_helper
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)
from .config import DeepSeekV1Config
from .convert import convert_deepseek, load_hf_deepseek


class DeepseekDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        ### Input layernorm in Deepseek v1 is same as Llama
        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        ### Deepseek v1 model with standard attention
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
            quant_mode=config.quant_mode)

        ClsMLP = GatedMLP
        moe_config = config.moe

        if moe_config.num_experts > 0 and layer_idx > 0:
            mlp_hidden_size = config.moe_intermediate_size
            hidden_act = config.hidden_act
            mlp_kwargs = {'moe_config': moe_config, 'mapping': config.mapping}
            if moe_config.shared_expert_intermediate_size > 0:
                ClsMLP = SharedMoE
                mlp_kwargs['use_shared_gate'] = False
                mlp_kwargs['use_side_stream'] = False
            else:
                ClsMLP = MOE
        else:
            ClsMLP = GatedMLP
            mlp_hidden_size = config.intermediate_size
            hidden_act = non_gated_version(
                config.hidden_act)  # back to non gated for dense layers
            mlp_kwargs = {}

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=mlp_hidden_size,
                          hidden_act=hidden_act,
                          dtype=config.dtype,
                          bias=False,
                          tp_group=config.mapping.tp_group,
                          tp_size=config.mapping.tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)

        ### Pose layernorm in Deepseek v1 is same as Llama             )
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
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


class DeepseekModel(Module):

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        init_all_reduce_helper()  # enable use_customer_all_reduce

        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(DeepseekDecoderLayer, config)

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
                prompt_vocab_size: Optional[Tensor] = None):

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
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


class DeepseekForCausalLM(DecoderModelForCausalLM):
    config_class = DeepSeekV1Config

    def __init__(self, config: PretrainedConfig):
        transformer = DeepseekModel(config)
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
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(cls,
                          model_dir,
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          override_fields={},
                          **kwargs):
        if mapping is None:
            mapping = Mapping()

        pretrained_config = DeepSeekV1Config.from_hugging_face(model_dir,
                                                               dtype=dtype,
                                                               mapping=mapping,
                                                               **kwargs)
        deepseek = cls.from_config(pretrained_config)
        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER") is None:

            custom_dict = {}

            rank_experts = mapping.ep_experts(pretrained_config.moe.num_experts)
            for index, module in enumerate(deepseek.transformer.layers):
                if index > 0:

                    module.mlp.shared_expert.fc.tllm_to_externel_key_dict = {
                        "fc": ["up_proj", "gate_proj"],
                        "shared_expert": "shared_experts"
                    }
                    module.mlp.shared_expert.proj.tllm_to_externel_key_dict = {
                        "shared_expert": "shared_experts"
                    }
                    module.mlp.fc.tllm_to_externel_key_dict = {
                        "fc": [
                            f"experts.{expert}.up_proj"
                            for expert in rank_experts
                        ] + [
                            f"experts.{expert}.gate_proj"
                            for expert in rank_experts
                        ]
                    }
                    module.mlp.proj.tllm_to_externel_key_dict = {
                        "proj": [
                            f"experts.{expert}.down_proj"
                            for expert in rank_experts
                        ]
                    }
                    module.mlp.router.tllm_to_externel_key_dict = {
                        "mlp": "mlp",
                        "router": "gate"
                    }

            loader = ModelWeightsLoader(model_dir, custom_dict)
            loader.generate_tllm_weights(deepseek)
            return deepseek
        else:

            hf_model = load_hf_deepseek(model_dir)
            weights = convert_deepseek(
                hf_model,
                pretrained_config,
                mapping=pretrained_config.mapping,
                dtype=pretrained_config.dtype,
                use_parallel_embedding=pretrained_config.use_parallel_embedding,
                sharding_dim=pretrained_config.embedding_sharding_dim)
            deepseek.load(weights)
            return deepseek
