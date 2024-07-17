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

from transformers import AutoModelForCausalLM

from ..._utils import pad_vocab_size
from ...functional import PositionEmbeddingType, Tensor
from ...layers import (MLP, Attention, AttentionMaskType, Embedding, LayerNorm,
                       ParallelLMHead)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig, save_checkpoint)
from .convert import convert_hf_config, convert_hf_weights


class PhiDecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        self.input_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                         dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rotary_embedding_percentage=config.partial_rotary_factor,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=True,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode)

        self.mlp = MLP(hidden_size=config.hidden_size,
                       ffn_hidden_size=config.intermediate_size,
                       hidden_act=config.hidden_act,
                       dtype=config.dtype,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       quant_mode=config.quant_mode)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
    ):
        residual = hidden_states

        input_layernorm_output = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            norm_before_bmm1=True,
        )

        if use_cache:
            attention_output, presents = attention_output

        feed_forward_hidden_states = self.mlp(input_layernorm_output, )
        hidden_states = attention_output + feed_forward_hidden_states + residual
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class PhiModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.vocab_embedding = Embedding(num_embeddings=config.vocab_size,
                                         embedding_dim=config.hidden_size,
                                         dtype=config.dtype)

        self.layers = DecoderLayerList(PhiDecoderLayer, config)
        self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                              dtype=config.dtype)

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
    ):
        args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size
                ] if prompt_embedding_table is not None else []
        hidden_states = self.vocab_embedding(input_ids, *args)

        hidden_states = self.layers(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class PhiForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = PhiModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        lm_head = ParallelLMHead(config.hidden_size,
                                 vocab_size_padded,
                                 bias=True,
                                 dtype=config.dtype,
                                 tp_group=config.mapping.tp_group,
                                 tp_size=config.mapping.tp_size,
                                 gather_output=True)

        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('partial_rotary_factor', 0.4)
        config.set_if_not_exist('rotary_base', 10000.0)

    @classmethod
    def convert_hf_checkpoint(cls,
                              hf_model_dir: str,
                              dtype: Optional[str] = "float16",
                              output_dir: Optional[str] = None,
                              args=None):
        '''
        Convert Huggingface checkpoint to TRT-LLM checkpoint
        '''
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_dir,
                                                        torch_dtype="auto",
                                                        trust_remote_code=True)
        config = convert_hf_config(hf_model.config, dtype, args)
        weights = convert_hf_weights(hf_model, dtype, args)

        if output_dir:
            save_checkpoint(output_dir, config=config, weights=weights)

        return {"weights": weights, "config": config}
