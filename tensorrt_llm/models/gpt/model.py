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
from ...functional import (Tensor, is_gated_activation, non_gated_version, recv,
                           send)
from ...layers import (MLP, MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, LayerNorm, MoeConfig,
                       PositionEmbeddingType)
from ...lora_helper import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ...quantization import QuantMode
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import GPTConfig
from .convert import (load_hf_gpt, load_weights_from_hf_model,
                      load_weights_from_nemo)


def MLPFactory(hidden_size,
               ffn_hidden_size,
               hidden_act,
               bias=True,
               dtype=None,
               moe_config: MoeConfig = MoeConfig(),
               tp_group=None,
               tp_size=1,
               mapping=Mapping(),
               quant_mode=QuantMode(0),
               inner_layernorm=False,
               eps=1e-05):
    if moe_config.has_moe():
        return MOE(moe_config,
                   hidden_size,
                   ffn_hidden_size,
                   hidden_act,
                   mapping=mapping,
                   bias=bias,
                   dtype=dtype,
                   tp_group=tp_group,
                   tp_size=tp_size,
                   quant_mode=quant_mode)
    MLPClass = GatedMLP if is_gated_activation(hidden_act) else MLP
    hidden_act = non_gated_version(hidden_act)
    return MLPClass(
        hidden_size,
        ffn_hidden_size,
        hidden_act,
        bias,
        dtype,
        tp_group,
        tp_size,
        quant_mode,
        inner_layernorm=inner_layernorm,
        eps=eps,
    )


class GPTDecoderLayer(Module):

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank

        self.input_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                         eps=config.norm_epsilon,
                                         dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        inner_layernorm = config.inner_layernorm if hasattr(
            config, "inner_layernorm") else False
        attention_head_size = config.head_size if hasattr(config,
                                                          "head_size") else None
        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            q_scaling=config.q_scaling,
            apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            attention_head_size=attention_head_size,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_percentage=config.rotary_pct,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            bias=config.bias,
            tp_group=tp_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_mode=config.quant_mode,
            qk_layernorm=config.qk_layernorm,
            inner_layernorm=inner_layernorm,
            eps=config.norm_epsilon)

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size
        self.norm_before_bmm1 = config.norm_before_bmm1 if hasattr(
            config, "norm_before_bmm1") else False

        self.mlp = MLPFactory(hidden_size=config.hidden_size,
                              ffn_hidden_size=mlp_hidden_size,
                              hidden_act=config.hidden_act,
                              dtype=config.dtype,
                              bias=config.bias,
                              moe_config=config.moe,
                              tp_group=tp_group,
                              tp_size=tp_size,
                              mapping=config.mapping,
                              quant_mode=config.quant_mode,
                              inner_layernorm=inner_layernorm,
                              eps=config.norm_epsilon)

        self.post_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                        eps=config.norm_epsilon,
                                        dtype=config.dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None,
                spec_decoding_params=None):

        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
            norm_before_bmm1=self.norm_before_bmm1)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GPTModel(Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.mapping = config.mapping
        self.position_embedding_type = config.position_embedding_type
        if config.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

            self.embedding_scale = config.embedding_scale

            if config.position_embedding_type == PositionEmbeddingType.learned_absolute:
                self.position_embedding = Embedding(
                    num_embeddings=config.max_position_embeddings,
                    embedding_dim=config.hidden_size,
                    dtype=config.dtype)

        self.layers = DecoderLayerList(GPTDecoderLayer, config)

        if config.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  eps=config.norm_epsilon,
                                  dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None,
                lora_params=None,
                spec_decoding_params=None):
        if self.mapping.is_first_pp_rank():
            ptuning_args = [
                prompt_embedding_table, prompt_tasks, prompt_vocab_size
            ] if prompt_embedding_table is not None else []
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
            if self.embedding_scale is not None:
                hidden_states *= self.embedding_scale
            if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
                hidden_states = hidden_states + self.position_embedding(
                    position_ids)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers(hidden_states,
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


class GPTForCausalLM(DecoderModelForCausalLM):
    config_class = GPTConfig

    def __init__(self, config: GPTConfig):
        transformer = GPTModel(config)

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
        self.trtllm_modules_to_hf_modules = {
            "attn_q": "q_proj",
            "attn_k": "k_proj",
            "attn_v": "v_proj",
            "attn_dense": "o_proj",
            "mlp_h_to_4h": "c_fc",
            "mlp_4h_to_h": "c_proj",
        }
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
        import transformers

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        is_prequantized_to_fp8 = kwargs.pop('is_prequantized_to_fp8', False)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = GPTConfig.from_hugging_face(hf_config_or_dir,
                                             dtype=dtype,
                                             mapping=mapping,
                                             quant_config=quant_config,
                                             **kwargs)
        if is_prequantized_to_fp8:
            custom_dict = {'fc': 'up_proj'}
            loader = ModelWeightsLoader(hf_model_dir, custom_dict)
            model = cls(config)

            # This is to account for all layernorms in nemotron variants being NemotronLayerNorm-1P.
            def apply_layernorm_1p(weights):
                return weights + 1.0

            loader.update_key_mapping(model)
            tllm_weights = {}
            for tllm_key, _ in model.named_parameters():
                if config.gpt_variant == "nemotron" and (
                        'layernorm.weight' in tllm_key
                        or 'ln_f.weight' in tllm_key):
                    tllm_weights.update(
                        loader.load(tllm_key,
                                    preprocess=apply_layernorm_1p,
                                    custom_postprocess_kwargs={}))
                else:
                    tllm_weights.update(
                        loader.load(tllm_key, custom_postprocess_kwargs={}))
            loader.fill(tllm_weights)
        else:
            if not use_preloading:
                hf_model = load_hf_gpt(hf_model_dir, load_model_on_cpu)
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

            config = GPTConfig.from_hugging_face(hf_model_dir,
                                                 dtype=dtype,
                                                 mapping=mapping,
                                                 quant_config=quant_config,
                                                 **kwargs)
            convert.quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             device=device,
                             calib_dataset=calib_dataset)
        else:
            raise ValueError(
                f"The quant_config ({quant_config}) does not require calibration, try {cls.__name__}.from_hugging_face instead."
            )

    @classmethod
    def from_nemo(cls,
                  nemo_ckpt_dir: str,
                  dtype: str = 'auto',
                  mapping: Optional[Mapping] = None,
                  quant_config: Optional[QuantConfig] = None,
                  **kwargs):
        config = GPTConfig.from_nemo(nemo_ckpt_dir,
                                     dtype=dtype,
                                     mapping=mapping,
                                     quant_config=quant_config,
                                     **kwargs)

        weights = load_weights_from_nemo(nemo_ckpt_dir, config, **kwargs)

        model = cls(config)
        model.load(weights)
        return model

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
