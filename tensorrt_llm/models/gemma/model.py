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
import math
from typing import TYPE_CHECKING, Any, Dict, Optional

from tensorrt_llm.models.gemma.convert import (QuantizeModifiers, Weights,
                                               load_gemma_weights_from_hf_model,
                                               non_modelopt_quantize_if_needed)
from tensorrt_llm.quantization.mode import (MODELOPT_FLOW_QUANTIZATIONS,
                                            QuantAlgo)

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import (AllReduceFusionOp, AllReduceParams, LayerNormType,
                           Tensor, cast, recv, send)
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       LoraParams, PositionEmbeddingType, RmsNorm)
from ...lora_helper import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig, save_checkpoint, save_config)
from .config import GemmaConfig

if TYPE_CHECKING:

    from .config import HfConfigOrDir


class GemmaDecoderLayer(Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        self.local_layer_idx = layer_idx - layers_range[0]

        q_scaling = 1.0
        max_attn_value = 0.0
        qk_layernorm = False
        is_sliding = False
        rotary_base = config.rotary_base
        rotary_base_local = None

        gemma2_config = config.gemma2_config()
        gemma3_config = config.gemma3_config()
        if gemma2_config:
            q_scaling = math.sqrt(
                gemma2_config.query_pre_attn_scalar) / math.sqrt(
                    config.head_size)
            max_attn_value = config.attn_logit_softcapping or 0.0
        elif gemma3_config:
            qk_layernorm = True
            q_scaling = math.sqrt(
                gemma3_config.query_pre_attn_scalar) / math.sqrt(
                    config.head_size)
            is_sliding = bool(
                (layer_idx + 1) % gemma3_config._sliding_window_pattern)
            rotary_base_local = config.rope_local_base_freq

        self.attention = Attention(
            local_layer_idx=self.local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attention_head_size=config.head_size,
            qk_layernorm=qk_layernorm,
            layernorm_type=LayerNormType.RmsNorm,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=rotary_base,
            rotary_embedding_base_local=rotary_base_local,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
            q_scaling=q_scaling,
            max_attn_value=max_attn_value,
            is_local=is_sliding,
        )

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            ffn_hidden_size=mlp_hidden_size,
                            hidden_act=config.hidden_act,
                            dtype=config.dtype,
                            bias=config.mlp_bias,
                            tp_group=config.mapping.tp_group,
                            tp_size=config.mapping.tp_size,
                            quant_mode=config.quant_mode)

        if self.config.inter_layernorms:
            self.pre_feedforward_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype)
            self.post_feedforward_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype)

        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None,
                lora_layer_params: Optional[LoraParams] = None,
                next_layer_input_layernorm_args=None):
        # assert not (
        #     default_net().plugin_config.reduce_fusion and self.has_residual_mlp
        # ), "Custom all reduce and residual mlp can't be enabled at the same time."
        if default_net(
        ).plugin_config.reduce_fusion and self.local_layer_idx > 0:
            hidden_states, residual = hidden_states  #FIXME:AN need to check if appropriate residual value is hidden state is pulled out.
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            norm_before_bmm1=True,
            lora_layer_params=lora_layer_params,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                if default_net().plugin_config.reduce_fusion else
                AllReduceFusionOp.NONE,
                residual=residual,
                norm_weight=self.pre_feedforward_layernorm.weight.value
                if self.config.inter_layernorms else None,
                norm_pre_residual_weight=self.post_layernorm.weight.value,
                eps=self.pre_feedforward_layernorm.eps
                if self.config.inter_layernorms else 1e-06,
            ),
        )

        if use_cache:
            attention_output, presents = attention_output

        if default_net().plugin_config.reduce_fusion:
            hidden_states, residual = attention_output
        else:
            if self.config.inter_layernorms:
                attention_output = self.post_layernorm(attention_output)
            hidden_states = residual + attention_output
            residual = hidden_states
            if self.config.inter_layernorms:
                hidden_states = self.pre_feedforward_layernorm(hidden_states)
            else:
                hidden_states = self.post_layernorm(hidden_states)

        if next_layer_input_layernorm_args is not None:
            hidden_states = self.mlp(
                hidden_states,
                lora_layer_params=lora_layer_params,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                    if default_net().plugin_config.reduce_fusion else
                    AllReduceFusionOp.NONE,
                    residual=residual,
                    norm_weight=next_layer_input_layernorm_args[0],
                    norm_pre_residual_weight=self.post_feedforward_layernorm.
                    weight.value,
                    eps=next_layer_input_layernorm_args[1]))
        else:
            hidden_states = self.mlp(hidden_states,
                                     lora_layer_params=lora_layer_params)

            if self.config.inter_layernorms:
                hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GemmaModel(Module):

    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()

        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(GemmaDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)
        self.hidden_size = config.hidden_size

    def forward(self,
                input_ids,
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
            hidden_states = cast(hidden_states * math.sqrt(self.hidden_size),
                                 hidden_states.dtype)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class GemmaForCausalLM(DecoderModelForCausalLM):
    config_class = GemmaConfig

    def __init__(self, config: GemmaConfig):
        transformer = GemmaModel(config)

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

    @staticmethod
    def _load_gemma_weights_from_hf(hf_model_dir: "HfConfigOrDir",
                                    trt_llm_config: GemmaConfig, *,
                                    load_model_on_cpu: bool) -> Weights:
        """`AutoModelForCausalLM.from_pretrained` will parse the correct gemma, whether Gemma or Gemma2 or future versions."""
        import transformers
        hf_gemma = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_dir,
            device_map="cpu" if load_model_on_cpu else "auto",
            dtype='auto',
        )
        weights = load_gemma_weights_from_hf_model(hf_gemma, trt_llm_config)
        del hf_gemma
        return weights

    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir: "HfConfigOrDir",
                          dtype='float16',
                          mapping: Optional[Mapping] = None,
                          quant_config: Optional[QuantConfig] = None,
                          load_model_on_cpu: bool = True,
                          **kwargs):
        config = GemmaConfig.from_hugging_face(hf_config_or_dir=hf_model_dir,
                                               dtype=dtype,
                                               mapping=mapping,
                                               quant_config=quant_config,
                                               **kwargs)
        model = GemmaForCausalLM(config)
        weights = cls._load_gemma_weights_from_hf(
            hf_model_dir, config, load_model_on_cpu=load_model_on_cpu)
        model.load(weights)
        return model

    NATIVE_QUANT_FLOW = {
        QuantAlgo.W8A16, QuantAlgo.W4A16,
        QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
        QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN,
        QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN,
        QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
    }

    @classmethod
    def assert_valid_quant_algo(cls, quant_algo: Optional[QuantAlgo]):
        allowed_quant_values = {
            None
        } | cls.NATIVE_QUANT_FLOW | MODELOPT_FLOW_QUANTIZATIONS
        assert quant_algo in allowed_quant_values, f"{quant_algo} isn't in the allowed `QuantAlgo` values for this model: {allowed_quant_values}"

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'float16',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        gemma_config_kwargs: Dict[str, Any] = None,
        **quantize_kwargs: Dict[str, Any],
    ):
        config = GemmaConfig.from_hugging_face(hf_model_dir,
                                               dtype=dtype,
                                               mapping=mapping,
                                               quant_config=quant_config,
                                               **(gemma_config_kwargs or {}))

        quant_algo = config.quantization.quant_algo
        if quant_algo is None and config.quantization.kv_cache_quant_algo is None:
            raise ValueError(
                "There is no point in calling `quantize()` if both `quant_algo` and `kv_cache_quant_algo` are `None`"
            )
        elif quant_algo in MODELOPT_FLOW_QUANTIZATIONS:
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=config.dtype,
                             mapping=config.mapping,
                             quant_config=config.quantization,
                             **quantize_kwargs)
        elif quant_algo in cls.NATIVE_QUANT_FLOW:
            save_config(config, output_dir=output_dir, log=True)
            for config in config.for_each_rank():
                hf_weights = cls._load_gemma_weights_from_hf(
                    hf_model_dir, config)
                ranked_weights = non_modelopt_quantize_if_needed(
                    hf_weights,
                    model_dir=hf_model_dir,
                    quantize_modifiers=QuantizeModifiers(),
                    trt_llm_config=config)
                save_checkpoint(
                    output_dir=output_dir,
                    weights=ranked_weights,
                    rank=config.mapping.rank,
                )
                del hf_weights
        else:
            cls.assert_valid_quant_algo(quant_algo)

    def use_lora(self, lora_config: LoraConfig) -> None:
        return use_lora(
            self, lora_config)  # Use the default trtllm->hf module mapping
