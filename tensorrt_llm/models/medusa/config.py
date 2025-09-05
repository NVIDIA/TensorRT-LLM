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
from typing import Optional, Union

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..llama.config import LLaMAConfig
from ..modeling_utils import PretrainedConfig, QuantConfig
from ..qwen.config import QWenConfig


# Medusa-specific config is stored and retrieved from GenericMedusaConfig.
class MedusaConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 num_medusa_heads: int = 4,
                 num_medusa_layers: int = 1,
                 max_draft_len: int = 63,
                 **kwargs):

        model_type = str(kwargs.get('model_type', '')).lower()
        generic_medusa_config = QWenConfig if 'qwen' in model_type else LLaMAConfig
        self.config = generic_medusa_config(**kwargs)

        # Add objects
        self.config.num_medusa_heads = num_medusa_heads
        self.config.num_medusa_layers = num_medusa_layers
        self.config.max_draft_len = max_draft_len

    def to_dict(self):
        output = self.config.to_dict()
        output['num_medusa_heads'] = self.config.num_medusa_heads
        output['num_medusa_layers'] = self.config.num_medusa_layers
        output['max_draft_len'] = self.config.max_draft_len
        return output

    # Specialization to redirect accesses to self.config
    def __getattr__(self, name):
        return getattr(self.config, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

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
        speculative_config_or_dir = kwargs.pop('speculative_model_dir', None)
        speculative_config = kwargs.pop("speculative_config", None)

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=trust_remote_code)
        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        if hasattr(hf_config, "medusa"):
            # is modelOpt ckpt
            num_medusa_heads = hf_config.medusa["num_medusa_heads"]
            num_medusa_layers = hf_config.medusa["num_medusa_layers"]
        else:
            config_file = speculative_config_or_dir / "config.json"
            with open(config_file) as fp:
                config = json.load(fp)

            num_medusa_heads = speculative_config.num_medusa_heads if speculative_config is not None else config.get(
                'num_medusa_heads', None)
            num_medusa_layers = config.get('medusa_num_layers', None)

        return cls(architecture="MedusaForCausalLM",
                   dtype=dtype,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   hidden_size=hf_config.hidden_size,
                   intermediate_size=hf_config.intermediate_size,
                   num_key_value_heads=hf_config.num_key_value_heads,
                   vocab_size=hf_config.vocab_size,
                   position_embedding_type='rope_gpt_neox',
                   max_position_embeddings=hf_config.max_position_embeddings,
                   hidden_act=hf_config.hidden_act,
                   norm_epsilon=hf_config.rms_norm_eps,
                   mapping=mapping,
                   quantization=quant_config,
                   num_medusa_heads=num_medusa_heads,
                   num_medusa_layers=num_medusa_layers,
                   **kwargs)
