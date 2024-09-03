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
from ...logger import logger
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class PhiConfig(PretrainedConfig):

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

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True)

        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        rotary_scaling = getattr(hf_config, "rope_scaling", None)
        rotary_base = getattr(hf_config, "rope_theta", 10000.0)
        if dtype == 'auto':
            dtype = getattr(hf_config, 'torch_dtype', None)
            if dtype is None:
                dtype = 'float16'
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == 'float32':
                dtype = 'float16'
        if dtype == 'bfloat16' and torch.cuda.get_device_properties(
                0).major < 8:
            logger.warning(
                "Pre SM 80 GPUs do not support bfloat16, fallback to float16")
            dtype = 'float16'

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
                   hidden_act=hf_config.hidden_act,
                   rotary_base=rotary_base,
                   rotary_scaling=rotary_scaling,
                   rotary_pct=hf_config.partial_rotary_factor,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
