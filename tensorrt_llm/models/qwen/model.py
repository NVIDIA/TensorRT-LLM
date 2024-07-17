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

from tensorrt_llm.lora_manager import LoraConfig, use_lora

from ..._utils import pad_vocab_size
from ...functional import Tensor, recv, send, sigmoid
from ...layers import (MLP, MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, RmsNorm, RowLinear)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)


class QWenDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
            dense_bias=False)

        ClsMLP = GatedMLP
        mlp_kwargs = {}
        if config.qwen_type == 'qwen2_moe':
            ClsMLP = MOE
            mlp_kwargs = {
                "moe_config": config.moe,
                "mapping": config.mapping,
            }

        if config.qwen_type == 'qwen2_moe':
            self.shared_expert = MLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.moe_shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                dtype=dtype,
                bias=False,
                tp_group=tp_group,
                tp_size=tp_size,
                quant_mode=config.quant_mode)
            self.shared_expert_gate = RowLinear(config.hidden_size,
                                                1,
                                                bias=False,
                                                dtype=dtype,
                                                tp_group=None,
                                                tp_size=1)

        # Qwen's real inter_size depends on qwen_type
        if self.config.qwen_type == 'qwen':
            intermediate_size = config.intermediate_size // 2
        elif self.config.qwen_type == 'qwen2_moe':
            intermediate_size = config.moe_intermediate_size
        else:
            intermediate_size = config.intermediate_size

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=intermediate_size,
                          hidden_act=config.hidden_act,
                          dtype=dtype,
                          bias=False,
                          tp_group=tp_group,
                          tp_size=tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
        )
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states

        hidden_states = self.post_layernorm(hidden_states)

        shared_output = None
        if self.config.qwen_type == 'qwen2_moe':
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = sigmoid(
                    self.shared_expert_gate(hidden_states)) * shared_output

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        if shared_output is not None:
            hidden_states = hidden_states + shared_output

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class QWenModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(QWenDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
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
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(hidden_states,
                                            use_cache=use_cache,
                                            attention_mask=attention_mask,
                                            kv_cache_params=kv_cache_params,
                                            attention_params=attention_params,
                                            lora_params=lora_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class QWenForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = QWenModel(config)
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
        if config.qwen_type == 'qwen':
            self.trtllm_modules_to_hf_modules = {
                "attn_qkv": "c_attn",
                "attn_dense": "attn.c_proj",
                "mlp_h_to_4h": "w2",
                "mlp_4h_to_h": "mlp.c_proj",
                "mlp_gate": "w1",
            }
        else:
            self.trtllm_modules_to_hf_modules = None
        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('rotary_base', 10000.0)
        config.set_if_not_exist('rotary_scaling', None)

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
