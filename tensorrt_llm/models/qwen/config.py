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

import torch

from ..._utils import torch_dtype_to_str
from ...layers import MoeConfig
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class QWenConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 mlp_bias: bool = False,
                 attn_bias: bool = True,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 disable_weight_only_quant_plugin: bool = False,
                 moe: Optional[Union[MoeConfig, dict]] = None,
                 **kwargs):
        self.mlp_bias = mlp_bias
        self.attn_bias = attn_bias
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.disable_weight_only_quant_plugin = disable_weight_only_quant_plugin
        if moe is None:
            # Legacy MOE config fields
            moe = MoeConfig(num_experts=kwargs.pop('moe_num_experts', 0),
                            top_k=kwargs.pop('moe_top_k', 0),
                            normalization_mode=kwargs.pop(
                                'moe_normalization_mode',
                                MoeConfig.ExpertScaleNormalizationMode.NONE))
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in QWenConfig
        output['mlp_bias'] = self.mlp_bias
        output['attn_bias'] = self.attn_bias
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output[
            'disable_weight_only_quant_plugin'] = self.disable_weight_only_quant_plugin
        output['moe'] = self.moe.to_dict()
        return output

    @classmethod
    def from_hugging_face(cls,
                          hf_config_or_dir: Union[
                              str, 'transformers.PretrainedConfig'],
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          quant_config: Optional[QuantConfig] = None,
                          **kwargs) -> "QWenConfig":
        import transformers

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True)

        qwen_type = hf_config.model_type
        valid_types = ('qwen', 'qwen2', 'qwen2_moe')
        assert qwen_type in valid_types, f"Unsupported Qwen type: {qwen_type}, only {valid_types} are acceptable."
        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        head_size = getattr(hf_config, "kv_channels", head_dim)
        hidden_act = getattr(hf_config, "hidden_act", "silu")
        if qwen_type == "qwen2_moe":
            hidden_act = "swiglu"
        attn_bias = True  # All existing Qwen models have attn bias
        rotary_scaling = getattr(hf_config, "rope_scaling", None)
        disable_weight_only_quant_plugin = kwargs.pop(
            'disable_weight_only_quant_plugin', False)
        if qwen_type == "qwen":
            rms_norm_eps = hf_config.layer_norm_epsilon
            rotary_base = getattr(hf_config, "rotary_emb_base", 10000.0)
        else:
            rms_norm_eps = hf_config.rms_norm_eps
            rotary_base = getattr(hf_config, "rope_theta", 100000.0)

        moe_num_experts = getattr(hf_config, "num_experts", 0)
        moe_top_k = getattr(hf_config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", 0)
        moe_shared_expert_intermediate_size = getattr(
            hf_config, "shared_expert_intermediate_size", 0)
        moe_normalization_mode = MoeConfig.ExpertScaleNormalizationMode.NONE
        moe_config = MoeConfig(num_experts=moe_num_experts,
                               top_k=moe_top_k,
                               normalization_mode=moe_normalization_mode)
        moe_config.validate()

        if dtype == 'auto':
            dtype = getattr(hf_config, 'torch_dtype', None)
            if dtype is None:
                dtype = 'float16'
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == 'float32':
                dtype = 'float16'

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
            norm_epsilon=rms_norm_eps,
            attn_bias=attn_bias,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            disable_weight_only_quant_plugin=disable_weight_only_quant_plugin,
            qwen_type=qwen_type,
            moe_intermediate_size=moe_intermediate_size,
            moe_shared_expert_intermediate_size=
            moe_shared_expert_intermediate_size,
            moe=moe_config,
            mapping=mapping,
            quantization=quant_config,
            **kwargs)
