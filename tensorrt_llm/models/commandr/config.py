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

import transformers

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class CohereConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 output_multiplier_scale: float = 0.0625,
                 rotary_base: float = 10000.0,
                 attn_bias: bool = False,
                 **kwargs):
        self.output_multiplier_scale = output_multiplier_scale
        self.rotary_base = rotary_base
        self.attn_bias = attn_bias
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in CohereConfig
        output['output_multiplier_scale'] = self.output_multiplier_scale
        output['rotary_base'] = self.rotary_base
        output['attn_bias'] = self.attn_bias
        return output

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_or_dir, trust_remote_code=True)

        head_size = hf_config.hidden_size // hf_config.num_attention_heads

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        if hf_config.tie_word_embeddings:
            kwargs['use_parallel_embedding'] = True
            kwargs['embedding_sharding_dim'] = 0

        return CohereConfig(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_size=head_size,
            vocab_size=hf_config.vocab_size,
            position_embedding_type='rope_gptj',  # different rope type
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act=hf_config.hidden_act,
            norm_epsilon=hf_config.layer_norm_eps,
            output_multiplier_scale=hf_config.logit_scale,
            rotary_base=hf_config.rope_theta,
            attn_bias=hf_config.attention_bias,
            qk_layernorm=hf_config.use_qk_norm,
            mapping=mapping,
            quantization=quant_config,
            **kwargs)
