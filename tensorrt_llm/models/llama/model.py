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

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import (AllReduceFusionOp, AllReduceFusionParams, Tensor,
                           non_gated_version, recv, send)
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, PositionEmbeddingType, RmsNorm)
from ...lora_manager import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ...plugin import init_all_reduce_helper
from ...quantization import W8A8_SQ_PLUGIN_LIST, QuantAlgo
from ..convert_utils import has_safetensors
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig, check_share_embedding)
from .config import LLaMAConfig
from .convert import (load_hf_llama, load_weights_from_hf_by_shard,
                      load_weights_from_hf_model,
                      load_weights_from_hf_safetensors,
                      load_weights_from_meta_ckpt)


class LLaMADecoderLayer(Module):

    def __init__(self, config: LLaMAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        self.local_layer_idx = layer_idx - layers_range[0]
        self.attention = Attention(
            local_layer_idx=self.local_layer_idx,
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
            quant_mode=config.quant_mode)

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        ClsMLP = GatedMLP
        mlp_kwargs = {}
        if config.moe.has_moe():
            ClsMLP = MOE
            mlp_kwargs = {
                "moe_config": config.moe,
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

        # Residual MLP that applies on pre-attention input
        # TODO: change to self.has_residual_mlp = self.config.residual_mlp after ModelOpt quantize config is updated
        self.has_residual_mlp = False
        if hasattr(self.config,
                   "residual_mlp") and self.config.residual_mlp is True:
            self.has_residual_mlp = True

        if self.has_residual_mlp:
            self.residual_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype)
            ClsMLP = GatedMLP  # TODO: may use FusedGatedMLP to further speedup
            self.residual_mlp = ClsMLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.
                hidden_size,  # residual mlp uses hidden_size
                hidden_act=non_gated_version(
                    config.hidden_act),  # back to non-gated
                dtype=config.dtype,
                bias=config.mlp_bias,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                quant_mode=config.quant_mode)

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None,
                next_layer_input_layernorm_args=None):
        assert not (
            default_net().plugin_config.reduce_fusion and self.has_residual_mlp
        ), "Custom all reduce and residual mlp can't be enabled at the same time."
        if default_net(
        ).plugin_config.reduce_fusion and self.local_layer_idx > 0:
            hidden_states, residual = hidden_states
        else:
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
            reduce_fusion_params=AllReduceFusionParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM
                if default_net().plugin_config.reduce_fusion else
                AllReduceFusionOp.NONE,
                residual=residual,
                norm_weight=self.post_layernorm.weight.value,
                eps=self.post_layernorm.eps))

        if use_cache:
            attention_output, presents = attention_output

        if self.has_residual_mlp:
            hidden_states = residual + attention_output
            residual_attn = hidden_states
            # arctic layer w/ residual mlp

            # residual mlp
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_mlp = residual_attn + hidden_states

            # parallel moe
            # parallel moe layers applies on PRE-ATTENTION input residual, therefore achieving pre-fetching and better parallelism
            hidden_states = self.post_layernorm(residual)
            hidden_states = self.mlp(hidden_states,
                                     lora_layer_params=lora_layer_params)
            hidden_states = residual_mlp + hidden_states
        else:
            if default_net().plugin_config.reduce_fusion:
                hidden_states, residual = attention_output
            else:
                hidden_states = residual + attention_output
                residual = hidden_states
                hidden_states = self.post_layernorm(hidden_states)
            if next_layer_input_layernorm_args is not None:
                hidden_states = self.mlp(
                    hidden_states,
                    lora_layer_params=lora_layer_params,
                    reduce_fusion_params=AllReduceFusionParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM
                        if default_net().plugin_config.reduce_fusion else
                        AllReduceFusionOp.NONE,
                        residual=residual,
                        norm_weight=next_layer_input_layernorm_args[0],
                        eps=next_layer_input_layernorm_args[1]))
            else:
                hidden_states = self.mlp(hidden_states,
                                         lora_layer_params=lora_layer_params)
                hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class LLaMAModel(Module):

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        init_all_reduce_helper()

        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(LLaMADecoderLayer, config)

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


class LLaMAForCausalLM(DecoderModelForCausalLM):
    config_class = LLaMAConfig

    def __init__(self, config: LLaMAConfig):
        transformer = LLaMAModel(config)
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

        load_by_shard = kwargs.pop('load_by_shard', False)
        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = LLaMAConfig.from_hugging_face(hf_config_or_dir,
                                               dtype=dtype,
                                               mapping=mapping,
                                               quant_config=quant_config,
                                               **kwargs)

        if use_preloading:
            assert not load_by_shard
            weights = load_weights_from_hf_model(hf_model, config)
        elif load_by_shard:
            weights = load_weights_from_hf_by_shard(hf_model_dir, config)
        elif has_safetensors(
                hf_model_dir) and not config.quant_mode.has_any_quant():
            weights = load_weights_from_hf_safetensors(hf_model_dir, config)
        else:
            hf_model = load_hf_llama(hf_model_dir, load_model_on_cpu)
            weights = load_weights_from_hf_model(hf_model, config)

        check_share_embedding(weights, config)
        model = LLaMAForCausalLM(config)
        model.load(weights)
        return model

    def default_plugin_config(self, **kwargs):
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.weight_only_groupwise_quant_matmul_plugin = 'auto'
        return plugin_config

    @classmethod
    def from_meta_ckpt(cls,
                       meta_ckpt_dir: str,
                       dtype: str = 'auto',
                       mapping: Optional[Mapping] = None,
                       quant_config: Optional[QuantConfig] = None,
                       **kwargs):
        config = LLaMAConfig.from_meta_ckpt(meta_ckpt_dir,
                                            dtype=dtype,
                                            mapping=mapping,
                                            quant_config=quant_config,
                                            **kwargs)

        weights = load_weights_from_meta_ckpt(meta_ckpt_dir, config)

        check_share_embedding(weights, config)
        model = LLaMAForCausalLM(config)
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
            QuantAlgo.W4A16_AWQ, QuantAlgo.FP8, QuantAlgo.W8A8_SQ_PER_CHANNEL,
            QuantAlgo.W4A8_AWQ
        ]
        config = LLaMAConfig.from_hugging_face(hf_model_dir,
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
            # non-modelopt, the legacy TRT-LLM native quantization algorithm:
            # sq, int4/int8 weights only, int8 kv cache
            NATIVE_QUANT_FLOW = [QuantAlgo.W4A16, QuantAlgo.W8A16, None
                                 ] + W8A8_SQ_PLUGIN_LIST
            is_valid_native_quant = (quant_config.quant_algo in NATIVE_QUANT_FLOW) and \
                (quant_config.kv_cache_quant_algo in [QuantAlgo.INT8, None])
            assert quant_config.quant_algo is not None or quant_config.kv_cache_quant_algo is not None, \
                "There is no point to call the quantize function if both quant_algo and kv_cache_quant_algo is None"
            assert is_valid_native_quant, f"Internal error: shall call Modelopt for this quantization {quant_config}"

            from . import convert
            convert.quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             device=device,
                             calib_dataset=calib_dataset)

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config)
