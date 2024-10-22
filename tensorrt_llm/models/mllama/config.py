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

import torch

from ..._utils import torch_dtype_to_str
from ...mapping import Mapping
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
                 cross_attention_layers: List[int] = None,
                 vision_output_dim: int = 0,
                 **kwargs):
        self.mlp_bias = mlp_bias
        self.attn_bias = attn_bias
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.residual_mlp = residual_mlp
        self.disable_weight_only_quant_plugin = disable_weight_only_quant_plugin

        self.cross_attention = True
        self.cross_attention_layers = cross_attention_layers
        assert vision_output_dim != 0
        self.vision_output_dim = vision_output_dim

        super().__init__(**kwargs)
        self.embed_vocab_size = self.vocab_size + 8  #FIXME The vocab_size of embedding contains the special tokens for image

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
        output['embed_vocab_size'] = self.embed_vocab_size
        output['vision_output_dim'] = self.vision_output_dim
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

        if dtype == 'auto':
            dtype = getattr(hf_text_config, 'torch_dtype', None)
            if dtype is None:
                dtype = 'float16'
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == 'float32':
                dtype = 'float16'

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

        if dtype == 'auto':
            dtype = 'bfloat16'

        if meta_config.get('use_scaled_rope'):
            rotary_scaling = {"type": "llama3"}
        else:
            rotary_scaling = meta_config.get("rope_scaling")

        # meta checkpoint don't have vocab_size|hidden_act|rotary_base specified, use same default value as HF
        return cls(architecture="MLLaMAModel",
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
