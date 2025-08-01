# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from transformers import LlamaConfig

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..llama.config import LLaMAConfig
from ..modeling_utils import QuantAlgo, QuantConfig


class EagleConfig(LLaMAConfig):

    def __init__(self,
                 *,
                 num_eagle_layers: int = 1,
                 max_draft_len: int = 63,
                 max_non_leaves_per_layer: int = 10,
                 **kwargs):
        self.num_eagle_layers = num_eagle_layers
        self.max_non_leaves_per_layer = max_non_leaves_per_layer
        self.max_draft_len = max_draft_len
        self.eagle_net_config = LLaMAConfig.from_dict(
            kwargs["eagle_net_config"])
        del kwargs["eagle_net_config"]
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in EagleConfig
        output['num_eagle_layers'] = self.num_eagle_layers
        output['max_non_leaves_per_layer'] = self.max_non_leaves_per_layer
        output['max_draft_len'] = self.max_draft_len
        output['eagle_net_config'] = self.eagle_net_config.to_dict()
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
        speculative_config_or_dir = kwargs.pop('speculative_model_dir', None)

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=trust_remote_code)
        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        hf_config = None
        hf_config_or_dir if speculative_config_or_dir is None else speculative_config_or_dir
        if hf_config_or_dir is not None:
            hf_config = LlamaConfig.from_pretrained(hf_config_or_dir)

            hf_config.model_type
            n_head = hf_config.num_attention_heads
            inter_size = hf_config.intermediate_size
            n_layer = hf_config.num_hidden_layers
            n_embd = hf_config.hidden_size
            n_kv_head = hf_config.num_key_value_heads
            rms_norm_eps = hf_config.rms_norm_eps
            vocab_size = hf_config.vocab_size
            rotary_scaling = hf_config.rope_scaling
            rotary_base = hf_config.rope_theta
            n_positions = hf_config.max_position_embeddings
            hidden_act = hf_config.hidden_act
            dtype = str(hf_config.torch_dtype)[6:] if dtype == 'auto' else dtype
            if hasattr(hf_config, 'head_dim'):
                head_dim = hf_config.head_dim
            else:
                head_dim = hf_config.n_embd // hf_config.n_head
            if hasattr(hf_config, 'head_size'):
                head_size = hf_config.head_size
            else:
                head_size = head_dim

            if speculative_config_or_dir is None:
                hf_config_eagle = hf_config.eagle
                n_head_eagle = hf_config_eagle['num_attention_heads']
                inter_size_eagle = hf_config_eagle['intermediate_size']
                n_layer_eagle = hf_config_eagle['num_hidden_layers']
                n_embd_eagle = hf_config_eagle['hidden_size']
                n_kv_head_eagle = hf_config_eagle['num_key_value_heads']
                rms_norm_eps_eagle = hf_config_eagle['rms_norm_eps']
                n_positions_eagle = hf_config_eagle['max_position_embeddings']
            else:
                hf_config_eagle = LlamaConfig.from_pretrained(
                    speculative_config_or_dir)
                n_head_eagle = hf_config_eagle.num_attention_heads
                inter_size_eagle = hf_config_eagle.intermediate_size
                n_layer_eagle = hf_config_eagle.num_hidden_layers
                n_embd_eagle = hf_config_eagle.hidden_size
                n_kv_head_eagle = hf_config_eagle.num_key_value_heads
                rms_norm_eps_eagle = hf_config_eagle.rms_norm_eps
                n_positions_eagle = hf_config_eagle.max_position_embeddings

        if rotary_scaling is not None:
            # assert use_gpt_attention_plugin, "RoPE scaling is only supported through GPT attention plugin."
            rotary_scaling = {
                "type": rotary_scaling["rope_type"],
            }
            rotary_scaling = rotary_scaling

        eagle_net_config = {
            'architecture': "LlamaForCausalLM",
            'dtype': dtype,
            'logits_dtype': 'float32',
            'num_hidden_layers': n_layer_eagle,
            'num_attention_heads': n_head_eagle,
            'hidden_size': n_embd_eagle,
            'intermediate_size': inter_size_eagle,
            'num_key_value_heads': n_kv_head_eagle,
            'vocab_size': vocab_size,
            'position_embedding_type': 'rope_gpt_neox',
            'max_position_embeddings': n_positions_eagle,
            'hidden_act': hidden_act,
            'rotary_base': rotary_base,
            'rotary_scaling': rotary_scaling,
            'norm_epsilon': rms_norm_eps_eagle,
            'quantization': {
                'quant_algo': None,
                'kv_cache_quant_algo': None,
            },
            'mapping': {
                'world_size': mapping.world_size,
                'tp_size': mapping.tp_size,
                'pp_size': mapping.pp_size,
            },
            'use_parallel_embedding': kwargs['use_parallel_embedding'],
            'embedding_sharding_dim': kwargs['embedding_sharding_dim'],
            'head_dim': head_dim,
            'head_size': head_size
        }

        config = {
            'architecture': 'EagleForCausalLM',
            'dtype': dtype,
            'logits_dtype': 'float32',
            'num_hidden_layers': n_layer,
            'num_attention_heads': n_head,
            'hidden_size': n_embd,
            'intermediate_size': inter_size,
            'num_key_value_heads': n_kv_head,
            'vocab_size': vocab_size,
            'position_embedding_type': 'rope_gpt_neox',
            'max_position_embeddings': n_positions,
            'hidden_act': hidden_act,
            'rotary_base': rotary_base,
            'rotary_scaling': rotary_scaling,
            'norm_epsilon': rms_norm_eps,
            'quantization': {
                'quant_algo': None,
                'kv_cache_quant_algo': None,
            },
            'mapping': {
                'world_size': mapping.world_size,
                'tp_size': mapping.tp_size,
                'pp_size': mapping.pp_size,
            },
            'use_parallel_embedding': kwargs['use_parallel_embedding'],
            'embedding_sharding_dim': kwargs['embedding_sharding_dim'],
            'num_eagle_layers': kwargs['speculative_config'].num_eagle_layers,
            'max_non_leaves_per_layer':
            kwargs['speculative_config'].max_non_leaves_per_layer,
            'eagle_net_config': eagle_net_config
        }
        if quant_config:
            config['quantization']['quant_algo'] = quant_config.quant_algo
            config['quantization'][
                'kv_cache_quant_algo'] = quant_config.kv_cache_quant_algo

            if quant_config.quant_algo == QuantAlgo.W4A16_GPTQ:
                config['quantization'].update({
                    "group_size": quant_config.group_size,
                    "has_zero_point": True,
                    "pre_quant_scale": False,
                    'quant_algo': QuantAlgo.W4A16_GPTQ
                })
        eagle_quant_config = {}
        try:
            with open(
                    str(speculative_config_or_dir) + '/' +
                    'hf_quant_config.json') as f:
                eagle_quant_config = json.load(f)
                if "lm_head" in eagle_quant_config['quantization'][
                        'exclude_modules']:
                    eagle_quant_config['quantization']['exclude_modules'] += [
                        f"eagle_nets.{i}.lm_head" for i in range(
                            kwargs['speculative_config'].num_eagle_layers)
                    ]
                config['quantization'].update(
                    eagle_quant_config['quantization'])
                config['eagle_net_config']['quantization'].update(
                    eagle_quant_config['quantization'])
        except IOError:
            pass

        return cls.from_dict(config)
