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

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class FalconConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 bias: bool = False,
                 parallel_attention: bool = False,
                 num_ln_in_parallel_attn: int | None = None,
                 new_decoder_architecture: bool = False,
                 rotary_base: float = 10000.0,
                 **kwargs):
        self.bias = bias
        self.parallel_attention = parallel_attention
        self.num_ln_in_parallel_attn = num_ln_in_parallel_attn
        self.new_decoder_architecture = new_decoder_architecture
        self.rotary_base = rotary_base
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in LLaMAConfig
        output['bias'] = self.bias
        output['parallel_attention'] = self.parallel_attention
        output['new_decoder_architecture'] = self.new_decoder_architecture
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

        # Falcon-7B config may not have num_kv_heads or n_head_kv.
        # Although Falcon-180B uses GQA (num_kv_heads=8), its config
        # has multi_query=True.
        if getattr(hf_config, 'multi_query', False) and not getattr(
                hf_config, 'new_decoder_architecture', False):
            hf_config.num_kv_heads = 1

        if hf_config.model_type == 'RefinedWeb':
            # Case 1. Falcon-40B / Falcon-40B-instruct
            # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
            hf_config.num_hidden_layers = hf_config.n_layer
            hf_config.num_attention_heads = hf_config.n_head
            hf_config.num_kv_heads = hf_config.n_head_kv
            hf_config.new_decoder_architecture = True
        elif hf_config.model_type == 'RefinedWebModel':
            # Case 2. Falcon-7B / Falcon-7B-instruct
            # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
            hf_config.num_hidden_layers = hf_config.n_layer
            hf_config.num_attention_heads = hf_config.n_head
            hf_config.num_kv_heads = 1 if hf_config.multi_query else hf_config.n_head
            hf_config.new_decoder_architecture = False
        elif hf_config.model_type != 'falcon':
            raise ValueError("Shouldn't reach here.")
        hf_config.model_type = 'falcon'

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        return cls(architecture='FalconForCausalLM',
                   dtype=dtype,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   num_key_value_heads=hf_config.num_kv_heads,
                   hidden_size=hf_config.hidden_size,
                   norm_epsilon=hf_config.layer_norm_epsilon,
                   vocab_size=hf_config.vocab_size,
                   position_embedding_type='alibi_with_scale'
                   if hf_config.alibi else 'rope_gpt_neox',
                   hidden_act='gelu',
                   bias=hf_config.bias,
                   parallel_attention=hf_config.parallel_attn,
                   num_ln_in_parallel_attn=getattr(hf_config,
                                                   'num_ln_in_parallel_attn',
                                                   None),
                   new_decoder_architecture=hf_config.new_decoder_architecture,
                   max_position_embeddings=getattr(hf_config,
                                                   'max_position_embeddings',
                                                   2048),
                   rotary_base=getattr(hf_config, 'rope_theta', 10000.0),
                   intermediate_size=getattr(hf_config, 'ffn_hidden_size',
                                             None),
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
