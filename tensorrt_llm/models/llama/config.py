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
import json
import math
import sys
from pathlib import Path
from typing import Optional, Union

from ...layers import MoeConfig
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class LLaMAConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 mlp_bias: bool = False,
                 attn_bias: bool = False,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 residual_mlp: bool = False,
                 disable_weight_only_quant_plugin: bool = False,
                 moe: Optional[Union[MoeConfig, dict]] = None,
                 remove_duplicated_kv_heads: bool = False,
                 embedding_multiplier: float = 1.0,
                 attention_multiplier: float = 1.0,
                 residual_multiplier: float = 1.0,
                 output_multiplier_scale: float = 1.0,
                 **kwargs):
        self.mlp_bias = mlp_bias
        self.attn_bias = attn_bias
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.residual_mlp = residual_mlp
        self.disable_weight_only_quant_plugin = disable_weight_only_quant_plugin
        if moe is None:
            # Legacy MOE config fields
            moe = MoeConfig(
                num_experts=kwargs.pop('moe_num_experts', 0),
                top_k=kwargs.pop('moe_top_k', 0),
                normalization_mode=kwargs.pop(
                    'moe_normalization_mode',
                    MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE))
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()
        self.remove_duplicated_kv_heads = remove_duplicated_kv_heads
        self.fc_after_embed = False
        self.use_input_layernorm_in_first_layer = True
        self.use_last_layernorm = True
        self.layer_idx_offset = 0
        self.embedding_multiplier = embedding_multiplier
        self.attention_multiplier = attention_multiplier
        self.residual_multiplier = residual_multiplier
        self.output_multiplier_scale = output_multiplier_scale
        self.has_partial_lora_mask = False

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in LLaMAConfig
        output['mlp_bias'] = self.mlp_bias
        output['attn_bias'] = self.attn_bias
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output['residual_mlp'] = self.residual_mlp
        output[
            'disable_weight_only_quant_plugin'] = self.disable_weight_only_quant_plugin
        output['fc_after_embed'] = self.fc_after_embed
        output[
            'use_input_layernorm_in_first_layer'] = self.use_input_layernorm_in_first_layer
        output['use_last_layernorm'] = self.use_last_layernorm
        output['layer_idx_offset'] = self.layer_idx_offset
        output['moe'] = self.moe.to_dict()
        return output

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        trust_remote_code = kwargs.pop('trust_remote_code', True)
        has_partial_lora_mask = False

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)
            if "vila" in hf_config_dir.lower():
                sys.path.append(hf_config_dir + "/../VILA")
                from llava.model import LlavaLlamaConfig  # noqa
                from llava.model import LlavaLlamaModel
                transformers.AutoConfig.register("llava_llama",
                                                 LlavaLlamaConfig)
                transformers.AutoModelForCausalLM.register(
                    LlavaLlamaConfig, LlavaLlamaModel)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=trust_remote_code)
            if hf_config.model_type == "llava":
                # LLaVA = Vision model + Llama LLM
                # We load a llava config and use its' text config as llama config
                from transformers import LlavaConfig
                hf_config = LlavaConfig.from_pretrained(
                    hf_config_dir).text_config
            if hf_config.model_type == "llava_next":
                from transformers import LlavaNextConfig
                hf_config = LlavaNextConfig.from_pretrained(
                    hf_config_dir).text_config
            if hf_config.model_type == "llava_llama":
                hf_config.llm_cfg["architecture"] = hf_config.llm_cfg[
                    "architectures"][0]
                hf_config.llm_cfg["dtype"] = hf_config.llm_cfg["torch_dtype"]
                hf_config = PretrainedConfig.from_dict(hf_config.llm_cfg)
            if hf_config.model_type == 'internlmxcomposer2':
                # InternLM-XComposer2 has a mask for partial lora
                # Therefore we need an additional flag for this mask
                has_partial_lora_mask = True
            if hf_config.model_type == 'mistral3':
                from transformers import Mistral3Config
                hf_config = Mistral3Config.from_pretrained(
                    hf_config_dir).text_config
                hf_config.architectures = ["MistralForCausalLM"]

        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        if hf_config.model_type == "exaone":
            hidden_act = hf_config.activation_function
            # NOTE
            # EXAONE also uses RMS norm but they represent as layer_norm_epsilon.
            norm_epsilon = getattr(hf_config, "layer_norm_epsilon", 1e-5)
        else:
            hidden_act = hf_config.hidden_act
            norm_epsilon = hf_config.rms_norm_eps
        head_dim = getattr(
            hf_config, "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads)
        head_size = getattr(hf_config, "kv_channels", head_dim)
        attn_bias = getattr(hf_config, 'bias', False) or getattr(
            hf_config, 'attention_bias', False)
        rotary_scaling = getattr(hf_config, "rope_scaling", None)
        rotary_base = getattr(hf_config, "rope_theta", 10000.0)
        residual_mlp = getattr(hf_config, "parallel_attn_mlp_res", False)
        disable_weight_only_quant_plugin = kwargs.pop(
            'disable_weight_only_quant_plugin', False)
        remove_duplicated_kv_heads = kwargs.pop('remove_duplicated_kv_heads',
                                                False)
        embedding_multiplier = getattr(hf_config, "embedding_multiplier", 1.0)
        attention_multiplier = getattr(hf_config, "attention_multiplier", 1.0)
        if attention_multiplier != 1.0:
            attention_multiplier *= math.sqrt(head_size)
        residual_multiplier = getattr(hf_config, "residual_multiplier", 1.0)
        output_multiplier_scale = 1.0 / getattr(hf_config, "logits_scaling",
                                                1.0)
        if hf_config.model_type in ["mixtral", "arctic", "granitemoe"]:
            # HF LLaMA-type models are implicitly using gated activation.
            # With our MoE implementation, we must make it explicit
            hidden_act = "swiglu"
            moe_normalization_mode = MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
        else:
            moe_normalization_mode = None
        moe_num_experts = getattr(hf_config, "num_local_experts", 0)
        moe_top_k = getattr(hf_config, "num_experts_per_tok", 0)
        moe_config = MoeConfig(num_experts=moe_num_experts,
                               top_k=moe_top_k,
                               normalization_mode=moe_normalization_mode)
        moe_config.validate()

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))
        tie_word_embeddings = getattr(hf_config, 'tie_word_embeddings', False)

        return cls(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            vocab_size=hf_config.vocab_size,
            position_embedding_type='rope_gpt_neox',
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act=hidden_act,
            norm_epsilon=norm_epsilon,
            attn_bias=attn_bias,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            residual_mlp=residual_mlp,
            disable_weight_only_quant_plugin=disable_weight_only_quant_plugin,
            moe=moe_config,
            mapping=mapping,
            quantization=quant_config,
            has_partial_lora_mask=has_partial_lora_mask,
            remove_duplicated_kv_heads=remove_duplicated_kv_heads,
            tie_word_embeddings=tie_word_embeddings,
            embedding_multiplier=embedding_multiplier,
            attention_multiplier=attention_multiplier,
            residual_multiplier=residual_multiplier,
            output_multiplier_scale=output_multiplier_scale,
            **kwargs)

    @classmethod
    def from_meta_ckpt(cls,
                       meta_ckpt_dir: str,
                       dtype: str = 'auto',
                       mapping: Optional[Mapping] = None,
                       quant_config: Optional[QuantConfig] = None,
                       **kwargs):

        with open(Path(meta_ckpt_dir, "params.json")) as fp:
            meta_config: dict = json.load(fp)

        n_embd = meta_config["dim"]
        n_head = meta_config["n_heads"]
        n_kv_head = meta_config.get("n_kv_heads", n_head)
        vocab_size = meta_config.get("vocab_size", 32000)

        # Reset vocab_size to 32000 for LLama v2 checkpoint.
        if vocab_size == -1:
            vocab_size = 32000

        if "hidden_dim" in meta_config:
            inter_size = meta_config["hidden_dim"]
        else:
            multiple_of = meta_config.get("multiple_of", 1)
            n_embd_ = int(4 * n_embd * 2 / 3)
            ffn_dim_multiplier = meta_config.get("ffn_dim_multiplier", 1)
            inter_size = multiple_of * (
                (int(n_embd_ * ffn_dim_multiplier) + multiple_of - 1) //
                multiple_of)

        dtype = infer_dtype(dtype, 'bfloat16')

        if meta_config.get('use_scaled_rope'):
            rotary_scaling = {"type": "llama3"}
        else:
            rotary_scaling = meta_config.get("rope_scaling")

        # meta checkpoint don't have vocab_size|hidden_act|rotary_base specified, use same default value as HF
        return cls(architecture="LlamaForCausalLM",
                   dtype=dtype,
                   num_hidden_layers=meta_config["n_layers"],
                   num_attention_heads=n_head,
                   hidden_size=n_embd,
                   intermediate_size=inter_size,
                   num_key_value_heads=n_kv_head,
                   vocab_size=vocab_size,
                   position_embedding_type='rope_gpt_neox',
                   max_position_embeddings=2048,
                   hidden_act='silu',
                   rotary_scaling=rotary_scaling,
                   rotary_base=meta_config.get('rope_theta', 10000),
                   norm_epsilon=meta_config["norm_eps"],
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
