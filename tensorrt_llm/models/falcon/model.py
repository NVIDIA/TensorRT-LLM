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

from ..._utils import pad_vocab_size
from ...functional import Tensor, allreduce, recv, send
from ...layers import (MLP, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, LayerNorm)
from ...mapping import Mapping
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import FalconConfig
from .convert import load_weights_from_hf_by_shard, load_weights_from_hf_model


class FalconDecoderLayer(Module):

    def __init__(self, config: FalconConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        hidden_size = config.hidden_size
        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        layernorm_epsilon = config.norm_epsilon

        self.input_layernorm = LayerNorm(normalized_shape=hidden_size,
                                         eps=layernorm_epsilon,
                                         dtype=dtype)

        self.new_decoder_architecture = config.new_decoder_architecture
        self.parallel_attn = config.parallel_attention
        self.num_ln_in_parallel_attn = config.num_ln_in_parallel_attn
        if self.num_ln_in_parallel_attn is None and self.new_decoder_architecture:
            self.num_ln_in_parallel_attn = 2
        if self.is_parallel_attention:
            # Not to apply allreduce inside the Attention/MLP layers.
            # allreduce applies after those layer.
            tp_group = None
        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            attention_mask_type=AttentionMaskType.causal,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=config.bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            quant_mode=config.quantization.quant_mode,
        )

        mlp_hidden_size = hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        if self.new_decoder_architecture and self.num_ln_in_parallel_attn == 2:
            # Layernorm before MLP.
            self.mlp_layernorm = LayerNorm(normalized_shape=hidden_size,
                                           eps=layernorm_epsilon,
                                           dtype=dtype)
        else:
            self.mlp_layernorm = None
        self.mlp = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=config.hidden_act,
            dtype=dtype,
            bias=config.bias,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quantization.quant_mode,
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
                attention_params=None):
        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        if self.new_decoder_architecture and self.num_ln_in_parallel_attn == 2:
            mlp_ln_output = self.mlp_layernorm(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        input_ln_output = hidden_states
        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output

        if not self.new_decoder_architecture:
            if self.parallel_attn:
                hidden_states = input_ln_output
            else:
                hidden_states = residual + attention_output
                residual = hidden_states
                hidden_states = self.post_layernorm(hidden_states)
        elif self.num_ln_in_parallel_attn == 2:
            hidden_states = mlp_ln_output

        if (self.new_decoder_architecture and self.parallel_attn
                and self.num_ln_in_parallel_attn == 1):
            hidden_states = input_ln_output

        hidden_states = self.mlp(hidden_states)

        if self.is_parallel_attention:
            hidden_states = hidden_states + attention_output
            if self.config.mapping.tp_size > 1:
                hidden_states = allreduce(hidden_states,
                                          self.config.mapping.tp_group)

        hidden_states = residual + hidden_states
        if use_cache:
            return hidden_states, presents
        return hidden_states


class FalconModel(Module):

    def __init__(self, config: FalconConfig):
        super().__init__()
        self.config = config
        if config.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(FalconDecoderLayer, config)
        if config.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None):
        if self.config.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)
        else:
            hidden_states = recv(hidden_states,
                                 self.config.mapping.prev_pp_rank())

        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.config.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states,
                                 self.config.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class FalconForCausalLM(DecoderModelForCausalLM):
    config_class = FalconConfig

    def __init__(self, config: FalconConfig):
        self.check_config(config)
        transformer = FalconModel(config)

        if config.mapping.is_last_pp_rank():
            vocab_size_padded = pad_vocab_size(config.vocab_size,
                                               config.mapping.tp_size)
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('bias', True)
        config.set_if_not_exist('new_decoder_architecture', False)
        config.set_if_not_exist('parallel_attention', False)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        ''' Create a FalconForCausalLM object from give parameters
        '''
        import transformers

        load_by_shard = kwargs.pop('load_by_shard', False)
        # load_model_on_cpu is ignored here, since specify target device_map will fail when workers > 1.

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = FalconConfig.from_hugging_face(hf_config_or_dir,
                                                dtype=dtype,
                                                mapping=mapping,
                                                quant_config=quant_config,
                                                **kwargs)

        if use_preloading:
            assert not load_by_shard
            weights = load_weights_from_hf_model(hf_model, config)
        elif load_by_shard:
            weights = load_weights_from_hf_by_shard(hf_model_dir, config)
        else:
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                hf_model_dir, dtype='auto')
            weights = load_weights_from_hf_model(hf_model, config)

        model = cls(config)
        model.load(weights)
        return model
