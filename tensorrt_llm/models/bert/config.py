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
import transformers

from ..._utils import torch_dtype_to_str
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class BERTConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 is_roberta: bool = False,
                 type_vocab_size,
                 pad_token_id=None,
                 num_labels=None,
                 **kwargs):
        self.is_roberta = is_roberta
        self.type_vocab_size = type_vocab_size
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output['is_roberta'] = self.is_roberta
        output['type_vocab_size'] = self.type_vocab_size
        output['pad_token_id'] = self.pad_token_id
        output['num_labels'] = self.num_labels

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

            hf_config = transformers.AutoConfig.from_pretrained(hf_config_dir)

        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        head_dim = getattr(
            hf_config, "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads)
        head_size = getattr(hf_config, "kv_channels", head_dim)
        num_labels = getattr(hf_config, "num_labels", None)

        if (hf_config.position_embedding_type == 'absolute'):
            position_embedding_type = 'learned_absolute'
        else:
            raise NotImplementedError(
                f"{hf_config.position_embedding_type} hasn't been supported")

        if hf_config.model_type == "bert":
            is_roberta = False
        else:
            is_roberta = True

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
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            vocab_size=hf_config.vocab_size,
            hidden_act=hf_config.hidden_act,
            logits_dtype='float32',
            norm_epsilon=hf_config.layer_norm_eps,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=hf_config.max_position_embeddings,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=hf_config.intermediate_size,
            head_size=head_size,
            quantization=quant_config,
            mapping=mapping,
            #BERT model args
            is_roberta=is_roberta,
            type_vocab_size=hf_config.type_vocab_size,
            pad_token_id=hf_config.pad_token_id,
            num_labels=num_labels,
            **kwargs)
