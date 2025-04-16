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


class Phi3Config(PretrainedConfig):

    def __init__(self,
                 *,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 **kwargs):

        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in PhiConfig

        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling

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
        if hasattr(hf_config, "llm_config"):
            hf_config = hf_config.llm_config
        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        small_variant = hf_config.architectures[0] == "Phi3SmallForCausalLM"

        kwargs['rotary_pct'] = getattr(hf_config, "partial_rotary_factor", 1.0)
        kwargs['tie_word_embeddings'] = getattr(hf_config,
                                                "tie_word_embeddings", False)
        if small_variant:
            kwargs['gegelu_limit'] = getattr(hf_config, "gegelu_limit", None)
            kwargs['rotary_base'] = hf_config.rope_embedding_base
            kwargs['mup_attn_multiplier'] = getattr(hf_config,
                                                    "mup_attn_multiplier", None)
            kwargs['mup_embedding_multiplier'] = getattr(
                hf_config, "mup_embedding_multiplier", None)
            kwargs['mup_use_scaling'] = getattr(hf_config, "mup_use_scaling",
                                                None)
            kwargs['mup_width_multiplier'] = getattr(hf_config,
                                                     "mup_width_multiplier",
                                                     None)
            kwargs['blocksparse_block_size'] = getattr(
                hf_config, "blocksparse_block_size", None)
            kwargs['blocksparse_homo_head_pattern'] = getattr(
                hf_config, "blocksparse_homo_head_pattern", None)
            kwargs['blocksparse_num_local_blocks'] = getattr(
                hf_config, "blocksparse_num_local_blocks", None)
            kwargs['blocksparse_vertical_stride'] = getattr(
                hf_config, "blocksparse_vert_stride", None)
            kwargs['dense_attention_every_n_layers'] = getattr(
                hf_config, "dense_attention_every_n_layers", None)
            kwargs['norm_epsilon'] = hf_config.layer_norm_epsilon
        else:
            kwargs['rotary_base'] = hf_config.rope_theta
            kwargs['norm_epsilon'] = hf_config.rms_norm_eps
        moe_variant = hf_config.architectures[0] == "PhiMoEForCausalLM"
        if moe_variant:
            kwargs.update({
                'moe': {
                    'num_experts': hf_config.num_local_experts,
                    'top_k': hf_config.num_experts_per_tok,
                    'normalization_mode':
                    MoeConfig.ExpertScaleNormalizationMode.SPARSE_MIXER,
                    'sparse_mixer_epsilon': hf_config.router_jitter_noise,
                },
                'attention_bias': hf_config.attention_bias
            })

        kwargs['position_embedding_type'] = 'rope_gpt_neox'
        if hf_config.max_position_embeddings >= 128000:
            kwargs[
                'original_max_position_embeddings'] = hf_config.original_max_position_embeddings
            kwargs['position_embedding_type'] = "long_rope"
            kwargs['longrope_scaling_short_factors'] = hf_config.rope_scaling[
                "short_factor"]
            kwargs['longrope_scaling_long_factors'] = hf_config.rope_scaling[
                "long_factor"]
            if small_variant or moe_variant:
                kwargs['longrope_long_mscale'] = hf_config.rope_scaling[
                    "long_mscale"]
                kwargs['longrope_short_mscale'] = hf_config.rope_scaling[
                    "short_mscale"]

        return cls(architecture=hf_config.architectures[0],
                   dtype=dtype,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   hidden_size=hf_config.hidden_size,
                   intermediate_size=hf_config.intermediate_size,
                   num_key_value_heads=num_key_value_heads,
                   vocab_size=hf_config.vocab_size,
                   max_position_embeddings=hf_config.max_position_embeddings,
                   hidden_act="swiglu"
                   if hf_config.hidden_act == 'silu' else hf_config.hidden_act,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
