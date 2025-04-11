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
from typing import Optional, Union

import transformers

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import (AllReduceFusionOp, AllReduceParams, Tensor,
                           allgather, concat, constant, div, non_gated_version,
                           recv, send, unsqueeze)
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, FusedGatedMLP, GatedMLP,
                       PositionEmbeddingType, RmsNorm)
from ...lora_manager import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ...quantization.functional import fused_layernorm
from ..convert_utils import has_safetensors
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import LLaMAConfig
from .convert import (load_hf_llama, load_weights_from_deepcompressor,
                      load_weights_from_gptq, load_weights_from_hf_by_shard,
                      load_weights_from_hf_model,
                      load_weights_from_hf_safetensors,
                      load_weights_from_meta_ckpt)


class LLaMADecoderLayer(Module):

    def __init__(self, config: LLaMAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        layer_idx += config.layer_idx_offset
        self.config = config
        self.mapping = config.mapping

        if (self.config.use_input_layernorm_in_first_layer
                and self.layer_idx == 0) or self.layer_idx > 0:
            self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                           eps=config.norm_epsilon,
                                           dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        self.local_layer_idx = layer_idx - layers_range[0]
        self.is_last_local_layer = layer_idx == layers_range[-1]
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
            q_scaling=1.0 / config.attention_multiplier,
            quant_mode=config.quant_mode,
            cp_group=config.mapping.cp_group,
            cp_size=config.mapping.cp_size,
            cp_rank=config.mapping.cp_rank)

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
        assert not (
            default_net().plugin_config.reduce_fusion
            and default_net().plugin_config.user_buffer
            and default_net().plugin_config.pp_reduce_scatter
        ), "User buffer reduce fusion enabled with PP reduce scatter is not supported now."
        assert not (
            default_net().plugin_config.reduce_fusion
            and default_net().plugin_config.norm_quant_fusion
        ), "Reduce fusion and quant fusion can't be enabled at the same time."
        if default_net(
        ).plugin_config.reduce_fusion and self.local_layer_idx > 0:
            hidden_states, residual = hidden_states
        elif default_net(
        ).plugin_config.norm_quant_fusion and self.local_layer_idx > 0:
            hidden_states, residual = hidden_states
        else:
            residual = hidden_states
            if (self.config.use_input_layernorm_in_first_layer
                    and self.layer_idx == 0) or self.layer_idx > 0:
                hidden_states = self.input_layernorm(hidden_states)

        reduce_fusion_op = AllReduceFusionOp.NONE
        if default_net().plugin_config.reduce_fusion:
            if default_net().plugin_config.user_buffer:
                if self.config.quant_mode.has_fp8_qdq():
                    reduce_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
                elif self.config.quant_mode.has_nvfp4():
                    assert default_net(
                    ).plugin_config.gemm_plugin == "nvfp4", "UB with nvfp4 model must use nvfp4 gemm plugin"
                    reduce_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
                else:
                    assert False, "UB must enabled with fp8 or nvfp4 model"
            else:
                reduce_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

        reduce_fusion_scale = None
        if default_net().plugin_config.reduce_fusion and default_net(
        ).plugin_config.user_buffer:
            if isinstance(self.mlp, FusedGatedMLP):
                if self.config.quant_mode.has_fp8_qdq():
                    reduce_fusion_scale = constant(
                        self.mlp.fused_fc.activation_scaling_factor.raw_value.
                        copy())
                elif self.config.quant_mode.has_nvfp4():
                    reduce_fusion_scale = constant(
                        [1.0] / self.mlp.fused_fc.
                        activation_global_scaling_factor.raw_value)
            else:
                if self.config.quant_mode.has_fp8_qdq():
                    reduce_fusion_scale = constant(
                        self.mlp.fc.activation_scaling_factor.raw_value.copy())
                elif self.config.quant_mode.has_nvfp4():
                    reduce_fusion_scale = constant(
                        [1.0] /
                        self.mlp.fc.activation_global_scaling_factor.raw_value)
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
            all_reduce_params=AllReduceParams(
                fusion_op=reduce_fusion_op,
                residual=residual,
                norm_weight=self.post_layernorm.weight.value,
                scale=reduce_fusion_scale,
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
            elif default_net().plugin_config.norm_quant_fusion:
                hidden_states, residual_attn, act_per_block_scale = fused_layernorm(
                    input=attention_output,
                    normalized_shape=self.config.hidden_size,
                    residual=residual,
                    weight=self.post_layernorm.weight.value,
                    scale=div(
                        1, self.mlp.fc.activation_global_scaling_factor.value)
                    if self.mlp.fc.activation_global_scaling_factor.value else
                    None,
                    eps=self.post_layernorm.eps,
                    p_dtype=self.config.dtype)

                hidden_states, residual_attn = (
                    hidden_states, act_per_block_scale), residual_attn
                assert isinstance(hidden_states, tuple)
            else:
                hidden_states = residual + attention_output * self.config.residual_multiplier
                residual = hidden_states
                hidden_states = self.post_layernorm(hidden_states)
            if next_layer_input_layernorm_args is not None:
                #this is middle layer
                hidden_states = self.mlp(
                    hidden_states,
                    lora_layer_params=lora_layer_params,
                    all_reduce_params=AllReduceParams(
                        fusion_op=reduce_fusion_op,
                        residual=residual_attn
                        if default_net().plugin_config.norm_quant_fusion else
                        residual,
                        norm_weight=next_layer_input_layernorm_args[0],
                        scale=next_layer_input_layernorm_args[2],
                        eps=next_layer_input_layernorm_args[1]))
                if default_net().plugin_config.norm_quant_fusion:
                    hidden_states, residual, act_per_block_scale = fused_layernorm(
                        input=hidden_states,
                        normalized_shape=self.config.hidden_size,
                        residual=residual_attn,
                        weight=next_layer_input_layernorm_args[0],
                        scale=div(1, next_layer_input_layernorm_args[2])
                        if next_layer_input_layernorm_args[2] else None,
                        eps=next_layer_input_layernorm_args[1],
                        p_dtype=self.config.dtype)
                    hidden_states = (hidden_states,
                                     act_per_block_scale), residual
            else:
                if default_net(
                ).plugin_config.pp_reduce_scatter and self.is_last_local_layer and not self.mapping.is_last_pp_rank(
                ):
                    hidden_states = self.mlp(
                        hidden_states,
                        lora_layer_params=lora_layer_params,
                        last_local_layer_residual=residual)
                else:
                    if (default_net().plugin_config.reduce_fusion
                            and default_net().plugin_config.user_buffer):
                        hidden_states, residual = self.mlp(
                            hidden_states,
                            lora_layer_params=lora_layer_params,
                            all_reduce_params=AllReduceParams(
                                fusion_op=AllReduceFusionOp.LAST_PROCESS_FOR_UB,
                                residual=residual))
                    else:
                        hidden_states = self.mlp(
                            hidden_states, lora_layer_params=lora_layer_params)
                    hidden_states = residual + hidden_states * self.config.residual_multiplier
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class LLaMAModel(Module):

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()

        self.mapping = config.mapping
        self.vocab_size = config.vocab_size
        self.has_partial_lora_mask = config.has_partial_lora_mask
        self.hidden_size = config.hidden_size
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)
            self.embedding_multiplier = config.embedding_multiplier

        self.layers = DecoderLayerList(LLaMADecoderLayer, config)

        if config.fc_after_embed:
            self.fc = ColumnLinear(2 * config.hidden_size,
                                   config.hidden_size,
                                   bias=True,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)

        if self.mapping.is_last_pp_rank():
            self.ln_f = None
            if config.use_last_layernorm:
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
                hidden_states_for_embed=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
            hidden_states *= self.embedding_multiplier
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
            if default_net().plugin_config.pp_reduce_scatter:
                hidden_states = allgather(hidden_states,
                                          self.mapping.tp_group,
                                          gather_dim=0)
                # reshape to (-1, hidden_size)
                hidden_states = hidden_states.view(
                    concat([-1, self.hidden_size]))

        if hidden_states_for_embed is not None:
            hidden_states = concat([hidden_states, hidden_states_for_embed],
                                   dim=-1)
            hidden_states = self.fc(hidden_states)

        if lora_params is not None and self.has_partial_lora_mask:
            partial_lora_mask = input_ids > (self.vocab_size - 1)
            lora_params.partial_lora_mask = unsqueeze(partial_lora_mask, -1)

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
            if self.ln_f:
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
        quant_ckpt_path = kwargs.pop('quant_ckpt_path', None)
        use_autoawq = kwargs.pop('use_autoawq', None)
        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER"
                          ) is not None and not isinstance(
                              hf_model_or_dir, transformers.PreTrainedModel):
            if "vila" in hf_model_or_dir or "llava" in hf_model_or_dir:
                hf_model_or_dir = load_hf_llama(hf_model_or_dir,
                                                load_model_on_cpu)
            elif not load_by_shard and not has_safetensors(
                    hf_model_or_dir) and (
                        quant_config is None
                        or not quant_config.quant_mode.has_any_quant()):
                hf_model_or_dir = load_hf_llama(hf_model_or_dir,
                                                load_model_on_cpu)

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
        if config.remove_duplicated_kv_heads:
            config.num_key_value_heads = config.num_key_value_heads // 2
        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER") is None:
            custom_dict = {}
            model_name = hf_model.config.model_type if use_preloading else hf_model_or_dir
            if "llava" in model_name:
                custom_dict = {
                    "transformer": "language_model.model",
                    "lm_head": "language_model.lm_head"
                }
            elif "vila" in model_name:
                hf_model_dir += "/llm"
            elif "exaone" in model_name.lower():
                custom_dict = {
                    "transformer": "transformer",
                    "layers": "h",
                    "vocab_embedding": "wte",
                    "lm_head": "lm_head",
                    "ln_f": "ln_f",
                    "attention": "attn.attention",
                    "dense": "out_proj",
                    "gate": "c_fc_1",
                    "proj": "c_proj",
                    "fc": "c_fc_0",
                    "input_layernorm": "ln_1",
                    "post_layernorm": "ln_2",
                }
            elif config.tie_word_embeddings:
                custom_dict = {"lm_head": "model.embed_tokens"}

            if quant_ckpt_path is not None:
                hf_model_dir = quant_ckpt_path
            arg_dict = {"use_autoawq": True} if use_autoawq else {}

            loader = ModelWeightsLoader(hf_model_dir, custom_dict)
            model = cls(config)
            loader.generate_tllm_weights(model, arg_dict)
        else:
            if use_preloading:
                assert not load_by_shard
                weights = load_weights_from_hf_model(hf_model, config)
            elif load_by_shard:
                weights = load_weights_from_hf_by_shard(hf_model_dir, config)
            elif has_safetensors(
                    hf_model_dir) and not config.quant_mode.has_any_quant():
                weights = load_weights_from_hf_safetensors(hf_model_dir, config)
            elif quant_ckpt_path is not None:
                if quant_config.quant_mode.is_int4_weight_only():
                    weights = load_weights_from_gptq(quant_ckpt_path, config)
                elif quant_config.quant_mode.is_qserve_w4a8():
                    weights = load_weights_from_deepcompressor(
                        quant_ckpt_path, config)
                else:
                    raise ValueError(
                        "quant_ckpt_path should be specified only for GPTQ or QServe"
                    )
            else:
                hf_model = load_hf_llama(hf_model_dir, load_model_on_cpu)
                weights = load_weights_from_hf_model(hf_model, config)
            model = cls(config)
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

            config = LLaMAConfig.from_hugging_face(hf_model_dir,
                                                   dtype=dtype,
                                                   mapping=mapping,
                                                   quant_config=quant_config,
                                                   **kwargs)
            trust_remote_code = kwargs.pop("trust_remote_code", True)

            convert.quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             device=device,
                             calib_dataset=calib_dataset,
                             trust_remote_code=trust_remote_code,
                             calib_batches=calib_batches,
                             calib_max_seq_length=calib_max_seq_length)
        else:
            raise ValueError(
                f"The quant_config ({quant_config}) does not require calibration, try {cls.__name__}.from_hugging_face instead."
            )

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config)
