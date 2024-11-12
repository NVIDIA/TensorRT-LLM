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

from ...logger import logger
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class BaichuanConfig(PretrainedConfig):

    def __init__(self, model_version: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if model_version is None:
            model_version = BaichuanConfig.guess_model_version(self)
        self.model_version = model_version

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in BaichuanConfig
        output['model_version'] = self.model_version
        return output

    @staticmethod
    def guess_model_version(
        config: Union['transformers.PretrainedConfig',
                      'BaichuanConfig']) -> str:
        logger.warning(
            "Model version is not set, trying to guess from loaded config")
        size = '7' if config.num_attention_heads == 32 else '13'
        version = '1' if config.vocab_size == 64000 else '2'
        return f'v{version}_{size}b'

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

        model_version = kwargs.pop('model_version', None)
        if model_version is None:
            model_version = BaichuanConfig.guess_model_version(hf_config)

        if model_version == 'v1_7b' or model_version == 'v2_7b':
            position_embedding_type = 'rope_gpt_neox'
            max_position_embeddings = hf_config.max_position_embeddings
        else:
            position_embedding_type = 'alibi'
            max_position_embeddings = hf_config.model_max_length

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        return cls(architecture='BaichuanForCausalLM',
                   dtype=dtype,
                   vocab_size=hf_config.vocab_size,
                   max_position_embeddings=max_position_embeddings,
                   hidden_size=hf_config.hidden_size,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   num_key_value_heads=hf_config.num_attention_heads,
                   hidden_act=hf_config.hidden_act,
                   intermediate_size=hf_config.intermediate_size,
                   norm_epsilon=hf_config.rms_norm_eps,
                   position_embedding_type=position_embedding_type,
                   model_version=model_version,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
