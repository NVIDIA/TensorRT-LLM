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
from pathlib import Path
from typing import List, Optional, Union

from ...functional import LayerNormPositionType, LayerNormType, MLPType
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class MLLaMAConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 mlp_bias: bool = False,
                 attn_bias: bool = False,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 residual_mlp: bool = False,
                 disable_weight_only_quant_plugin: bool = False,
                 cross_attention: bool = True,
                 cross_attention_layers: List[int] = None,
                 vision_output_dim: int = 0,
                 has_position_embedding=False,
                 type_vocab_size=None,
                 rescale_before_lm_head=False,
                 layernorm_type=LayerNormType.RmsNorm,
                 layernorm_position=LayerNormPositionType.pre_layernorm,
                 has_attention_qkvo_bias=False,
                 has_mlp_bias=False,
                 has_model_final_layernorm=True,
                 model_type='MLLaMAForCausalLM',
                 skip_cross_kv=False,
                 mlp_type=MLPType.GatedMLP,
                 has_embedding_scale=False,
                 residual_scaling=1.0,
                 has_lm_head_bias=False,
                 num_buckets=None,
                 max_distance=0,
                 relative_attention=False,
                 **kwargs):
        self.mlp_bias = mlp_bias
        self.attn_bias = attn_bias
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.residual_mlp = residual_mlp
        self.disable_weight_only_quant_plugin = disable_weight_only_quant_plugin

        assert cross_attention
        self.cross_attention = cross_attention
        self.cross_attention_layers = cross_attention_layers
        assert vision_output_dim != 0
        self.vision_output_dim = vision_output_dim

        self.has_position_embedding = has_position_embedding
        self.type_vocab_size = type_vocab_size
        self.rescale_before_lm_head = rescale_before_lm_head
        self.layernorm_type = layernorm_type
        self.layernorm_position = layernorm_position
        self.has_attention_qkvo_bias = has_attention_qkvo_bias
        self.has_mlp_bias = has_mlp_bias
        self.has_model_final_layernorm = has_model_final_layernorm
        self.model_type = model_type
        self.skip_cross_kv = skip_cross_kv
        self.mlp_type = mlp_type
        self.has_embedding_scale = has_embedding_scale
        self.residual_scaling = residual_scaling
        self.has_lm_head_bias = has_lm_head_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention = relative_attention
        self.skip_cross_attn_blocks = True

        kwargs.pop('embed_vocab_size', None)
        kwargs.pop('num_kv_heads_per_layer', None)
        kwargs.pop('num_kv_heads_per_cross_attn_layer', None)
        super().__init__(**kwargs)

    @property
    def embed_vocab_size(self):
        return self.vocab_size + 8  #FIXME The vocab_size of embedding contains the special tokens for image

    @property
    def num_kv_heads_per_layer(self):
        num_kv_heads_per_layer = [
            self.num_key_value_heads for _ in range(self.num_hidden_layers)
        ]
        for layer_idx in self.cross_attention_layers:
            num_kv_heads_per_layer[layer_idx] = 0
        return num_kv_heads_per_layer

    @property
    def num_kv_heads_per_cross_attn_layer(self):
        num_kv_heads_per_cross_attn_layer = [
            0 for _ in range(self.num_hidden_layers)
        ]
        for layer_idx in self.cross_attention_layers:
            num_kv_heads_per_cross_attn_layer[
                layer_idx] = self.num_key_value_heads
        return num_kv_heads_per_cross_attn_layer

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in MLLaMAConfig
        output['mlp_bias'] = self.mlp_bias
        output['attn_bias'] = self.attn_bias
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output['residual_mlp'] = self.residual_mlp
        output[
            'disable_weight_only_quant_plugin'] = self.disable_weight_only_quant_plugin
        output['cross_attention'] = self.cross_attention
        output['cross_attention_layers'] = self.cross_attention_layers
        output['vision_output_dim'] = self.vision_output_dim
        output['embed_vocab_size'] = self.embed_vocab_size
        output['num_kv_heads_per_layer'] = self.num_kv_heads_per_layer
        output[
            'num_kv_heads_per_cross_attn_layer'] = self.num_kv_heads_per_cross_attn_layer
        output['skip_cross_attn_blocks'] = self.skip_cross_attn_blocks
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

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True)

        hf_text_config = hf_config.text_config
        hf_vision_config = hf_config.vision_config
        num_key_value_heads = getattr(hf_text_config, "num_key_value_heads",
                                      hf_text_config.num_attention_heads)

        hidden_act = hf_text_config.hidden_act if hasattr(
            hf_text_config, "hidden_act") else hf_text_config.hidden_activation
        norm_epsilon = hf_text_config.rms_norm_eps

        head_dim = getattr(
            hf_text_config, "head_dim",
            hf_text_config.hidden_size // hf_text_config.num_attention_heads)
        head_size = getattr(hf_text_config, "kv_channels", head_dim)
        attn_bias = getattr(hf_text_config, 'bias', False) or getattr(
            hf_text_config, 'attention_bias', False)
        rotary_scaling = getattr(hf_text_config, "rope_scaling", None)
        rotary_base = getattr(hf_text_config, "rope_theta", 10000.0)
        residual_mlp = getattr(hf_text_config, "parallel_attn_mlp_res", False)
        disable_weight_only_quant_plugin = kwargs.pop(
            'disable_weight_only_quant_plugin', False)

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        return cls(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            num_hidden_layers=hf_text_config.num_hidden_layers,
            num_attention_heads=hf_text_config.num_attention_heads,
            hidden_size=hf_text_config.hidden_size,
            intermediate_size=hf_text_config.intermediate_size,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            vocab_size=hf_text_config.vocab_size,
            position_embedding_type='rope_gpt_neox',
            max_position_embeddings=hf_text_config.max_position_embeddings,
            hidden_act=hidden_act,
            norm_epsilon=norm_epsilon,
            attn_bias=attn_bias,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            residual_mlp=residual_mlp,
            disable_weight_only_quant_plugin=disable_weight_only_quant_plugin,
            mapping=mapping,
            quantization=quant_config,
            cross_attention_layers=hf_text_config.cross_attention_layers,
            vision_output_dim=hf_vision_config.vision_output_dim,
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
        return cls(architecture="MLLaMAForCausalLM",
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
