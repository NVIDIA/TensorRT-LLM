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
import copy
import os
from typing import Optional, Union

import safetensors
from transformers import AutoModelForCausalLM

from ..._utils import pad_vocab_size
from ...functional import Tensor
from ...layers import (MLP, Attention, AttentionMaskType, Embedding, LayerNorm,
                       ParallelLMHead)
from ...mapping import Mapping
from ...module import Module
from ...quantization import QuantAlgo
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig, QuantConfig)
from .config import PhiConfig
from .convert import load_weights_from_hf_model


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
            rotary_embedding_percentage=config.rotary_pct,
            position_embedding_type=config.position_embedding_type,
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
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir
        config = PhiConfig.from_hugging_face(hf_config_or_dir,
                                             dtype=dtype,
                                             mapping=mapping,
                                             quant_config=quant_config,
                                             **kwargs)
        if not use_preloading:
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, torch_dtype="auto", trust_remote_code=True)

        assert isinstance(hf_model, transformers.PreTrainedModel)

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
        DEFAULT_MODELOPT_FLOW = [
            QuantAlgo.W4A16_AWQ,
            QuantAlgo.FP8,
            QuantAlgo.W8A8_SQ_PER_CHANNEL,
        ]
        NATIVE_QUANT_FLOW = [QuantAlgo.W4A16, QuantAlgo.W8A16, None]

        config = PhiConfig.from_hugging_face(hf_model_dir,
                                             dtype=dtype,
                                             mapping=mapping,
                                             quant_config=quant_config,
                                             **kwargs)

        if quant_config.quant_algo in DEFAULT_MODELOPT_FLOW:
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=config.dtype,
                             mapping=config.mapping,
                             quant_config=config.quantization,
                             device=device,
                             calib_dataset=calib_dataset,
                             calib_batches=calib_batches,
                             calib_batch_size=calib_batch_size,
                             calib_max_seq_length=calib_max_seq_length,
                             random_seed=random_seed,
                             tokenizer_max_seq_length=tokenizer_max_seq_length)
        else:
            assert quant_config.quant_algo in NATIVE_QUANT_FLOW, f"Internal error: shall call Modelopt for this quantization {quant_config}"

            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, torch_dtype="auto", trust_remote_code=True)

            for rank in range(mapping.world_size):
                weights = load_weights_from_hf_model(hf_model, config)
                config = copy.deepcopy(config)
                config.set_rank(rank)
                safetensors.torch.save_file(
                    weights, os.path.join(output_dir,
                                          f'rank{rank}.safetensors'))
