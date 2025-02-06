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

from ...layers import MoeConfig
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class DeepSeekV1Config(PretrainedConfig):

    def __init__(self,
                 *,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 moe: Optional[Union[MoeConfig, dict]] = None,
                 **kwargs):

        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        if isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in DeepSeekV1Config

        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
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

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=trust_remote_code)

        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        rotary_scaling = getattr(hf_config, "rope_scaling", None)
        rotary_base = getattr(hf_config, "rope_theta", 10000.0)
        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))
        moe_config = MoeConfig(
            num_experts=getattr(hf_config, 'n_routed_experts', 0),
            shared_expert_intermediate_size=getattr(hf_config,
                                                    'n_shared_experts', 0) *
            getattr(hf_config, "moe_intermediate_size", 0),
            top_k=getattr(hf_config, 'num_experts_per_tok', 0),
            normalization_mode=getattr(
                hf_config, 'moe_normalization_mode',
                MoeConfig.ExpertScaleNormalizationMode.NONE),
        )
        moe_config.validate()
        return cls(architecture=hf_config.architectures[0],
                   dtype=dtype,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   hidden_size=hf_config.hidden_size,
                   intermediate_size=hf_config.intermediate_size,
                   num_key_value_heads=num_key_value_heads,
                   vocab_size=hf_config.vocab_size,
                   position_embedding_type='rope_gpt_neox',
                   max_position_embeddings=hf_config.max_position_embeddings,
                   hidden_act='swiglu',
                   rotary_base=rotary_base,
                   rotary_scaling=rotary_scaling,
                   norm_epsilon=hf_config.rms_norm_eps,
                   mapping=mapping,
                   quantization=quant_config,
                   moe=moe_config,
                   moe_intermediate_size=hf_config.moe_intermediate_size,
                   **kwargs)
