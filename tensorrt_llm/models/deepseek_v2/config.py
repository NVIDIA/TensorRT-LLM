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

from transformers import AutoConfig

from ...layers import MoeConfig
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class DeepSeekV2Config(PretrainedConfig):

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
        assert isinstance(moe,
                          MoeConfig), "moe must be a MoeConfig or a dictionary"
        self.moe = moe.validate()
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in DeepSeekV2Config
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

            hf_config = AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=trust_remote_code)

        moe_routed_scaling_factor = hf_config.routed_scaling_factor
        moe_top_k = hf_config.num_experts_per_tok
        assert moe_routed_scaling_factor > 0, 'routed_scaling_factor should be greater than 0'
        if hf_config.topk_method == 'group_limited_greedy':
            if moe_top_k > 1 and hf_config.norm_topk_prob:
                moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED_RENORM
            else:
                moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED
        elif hf_config.topk_method == 'greedy':
            assert moe_routed_scaling_factor == 1.0, 'The combination of topk_method == greedy and routed_scaling_factor != 1.0 is not supported'
            if moe_top_k > 1 and hf_config.norm_topk_prob:
                moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
            else:
                moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.NONE
        else:
            raise AssertionError(
                f'Unsupported topk_method in hf_config: {hf_config.topk_method}'
            )

        rotary_scaling = None
        if hf_config.rope_scaling is not None:
            rotary_scaling = {
                'beta_fast':
                hf_config.rope_scaling['beta_fast'],
                'beta_slow':
                hf_config.rope_scaling['beta_slow'],
                'factor':
                hf_config.rope_scaling['factor'],
                'mscale':
                hf_config.rope_scaling['mscale'],
                'mscale_all_dim':
                hf_config.rope_scaling['mscale_all_dim'],
                'original_max_position_embeddings':
                hf_config.rope_scaling['original_max_position_embeddings'],
                'type':
                hf_config.rope_scaling['type']
            }

        moe_config = MoeConfig(
            num_experts=hf_config.n_routed_experts,
            shared_expert_intermediate_size=hf_config.n_shared_experts *
            hf_config.moe_intermediate_size,
            top_k=hf_config.num_experts_per_tok,
            normalization_mode=moe_renorm_mode,
            device_limited_n_group=hf_config.n_group,
            device_limited_topk_group=hf_config.topk_group,
            device_limited_routed_scaling_factor=hf_config.routed_scaling_factor
        )

        moe_config.validate()

        return cls(architecture=hf_config.architectures[0],
                   dtype=dtype,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   hidden_size=hf_config.hidden_size,
                   intermediate_size=hf_config.intermediate_size,
                   num_key_value_heads=hf_config.num_key_value_heads,
                   vocab_size=hf_config.vocab_size,
                   position_embedding_type='rope_gpt_neox',
                   max_position_embeddings=hf_config.max_position_embeddings,
                   hidden_act='swiglu',
                   norm_epsilon=hf_config.rms_norm_eps,
                   rotary_base=hf_config.rope_theta,
                   rotary_scaling=rotary_scaling,
                   moe_inter_size=hf_config.moe_intermediate_size,
                   moe=moe_config,
                   mapping=mapping,
                   quantization=quant_config,
                   kv_lora_rank=hf_config.kv_lora_rank,
                   q_lora_rank=hf_config.q_lora_rank,
                   qk_nope_head_dim=hf_config.qk_nope_head_dim,
                   qk_rope_head_dim=hf_config.qk_rope_head_dim,
                   v_head_dim=hf_config.v_head_dim,
                   topk_method=hf_config.topk_method,
                   first_k_dense_replace=hf_config.first_k_dense_replace,
                   moe_layer_freq=hf_config.moe_layer_freq,
                   scoring_func=hf_config.scoring_func,
                   **kwargs)
