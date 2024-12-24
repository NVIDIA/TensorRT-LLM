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

GLM_VERSIONS = ['glm4', 'chatglm3', 'chatglm2', 'chatglm', 'glm']
GLM_ARCH1_VERSIONS = ['chatglm', 'glm']
GLM_ARCH2_VERSIONS = ['glm4', 'chatglm3', 'chatglm2']


class ChatGLMConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 chatglm_version: str = 'chatglm3',
                 add_bias_linear: bool = False,
                 add_qkv_bias: bool = True,
                 apply_query_key_layer_scaling: bool = False,
                 apply_residual_connection_post_layernorm: bool = False,
                 rmsnorm: bool = True,
                 rotary_pct: float = 0.5,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 **kwargs):
        self.chatglm_version = chatglm_version
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.rmsnorm = rmsnorm
        self.rotary_pct = rotary_pct
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in ChatGLMConfig
        output['chatglm_version'] = self.chatglm_version
        output['add_bias_linear'] = self.add_bias_linear
        output['add_qkv_bias'] = self.add_qkv_bias
        output[
            'apply_query_key_layer_scaling'] = self.apply_query_key_layer_scaling
        output[
            'apply_residual_connection_post_layernorm'] = self.apply_residual_connection_post_layernorm
        output['rmsnorm'] = self.rmsnorm
        output['rotary_pct'] = self.rotary_pct
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

        # load hugging face config
        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=trust_remote_code)

        logits_dtype = kwargs.pop('logits_dtype', 'float32')
        use_parallel_embedding = kwargs.pop('use_parallel_embedding', False)
        embedding_sharding_dim = kwargs.pop('embedding_sharding_dim', 0)
        chatglm_version = kwargs.pop('chatglm_version', None)

        # get chatglm version
        if chatglm_version is None:
            print("Inferring chatglm version from path...")
            for v in GLM_VERSIONS:
                if v in hf_config._name_or_path:
                    chatglm_version = v
                    break
            if 'glm_4' in hf_config._name_or_path.replace("-", "_"):
                chatglm_version = 'glm4'
        assert chatglm_version in GLM_VERSIONS
        print(f"Chatglm version: {chatglm_version}")

        if chatglm_version == 'glm':
            hf_config.num_kv_heads = hf_config.num_attention_heads
            hf_config.ffn_hidden_size = hf_config.hidden_size * 4
            hf_config.hidden_act = 'gelu'
            hf_config.layernorm_epsilon = 1e-5
            hf_config.max_position_embeddings = hf_config.max_sequence_length
            hf_config.add_bias_linear = True
            hf_config.add_qkv_bias = True
            hf_config.apply_query_key_layer_scaling = False
            hf_config.apply_residual_connection_post_layernorm = False
            hf_config.rmsnorm = False
            hf_config.rope_ratio = 1.0
        elif chatglm_version == 'chatglm':
            hf_config.num_kv_heads = hf_config.num_attention_heads
            hf_config.ffn_hidden_size = hf_config.inner_hidden_size
            hf_config.hidden_act = 'gelu'
            hf_config.max_position_embeddings = hf_config.max_sequence_length
            hf_config.add_bias_linear = True
            hf_config.add_qkv_bias = True
            hf_config.apply_query_key_layer_scaling = False
            hf_config.apply_residual_connection_post_layernorm = False
            hf_config.rmsnorm = False
            hf_config.rope_ratio = 1.0
        else:
            hf_config.vocab_size = hf_config.padded_vocab_size
            hf_config.num_kv_heads = hf_config.multi_query_group_num
            hf_config.hidden_act = 'swiglu'
            hf_config.max_position_embeddings = hf_config.seq_length
            hf_config.rmsnorm = getattr(hf_config, 'rmsnorm', 1.0)
            hf_config.rope_ratio = getattr(hf_config, 'rope_ratio', 1.0)

        if chatglm_version == 'glm':
            position_embedding_type = 'learned_absolute'
        elif chatglm_version == 'chatglm':
            position_embedding_type = 'chatglm'
        elif chatglm_version in GLM_ARCH2_VERSIONS:
            position_embedding_type = 'rope_gptj'

        rotary_base = 10000.0
        rotary_embedding_scaling = None
        if chatglm_version == 'chatglm2':
            if hf_config.rope_ratio > 1:
                rotary_embedding_scaling = {
                    'type': 'linear',
                    'factor': hf_config.rope_ratio
                }
        elif chatglm_version == 'chatglm3' or chatglm_version == 'glm4':
            rotary_base *= hf_config.rope_ratio

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        return cls(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            logits_dtype=logits_dtype,
            num_hidden_layers=hf_config.num_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_kv_heads,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.ffn_hidden_size,
            norm_epsilon=hf_config.layernorm_epsilon,
            vocab_size=hf_config.vocab_size,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=hf_config.max_position_embeddings,
            rotary_pct=0.5,
            rotary_base=rotary_base,
            rotary_scaling=rotary_embedding_scaling,
            hidden_act=hf_config.hidden_act,
            use_parallel_embedding=use_parallel_embedding,
            embedding_sharding_dim=embedding_sharding_dim,
            quantization=quant_config,
            mapping=mapping,
            chatglm_version=chatglm_version,
            add_bias_linear=hf_config.add_bias_linear,
            add_qkv_bias=hf_config.add_qkv_bias,
            apply_query_key_layer_scaling=False,
            apply_residual_connection_post_layernorm=hf_config.
            apply_residual_connection_post_layernorm,
            rmsnorm=hf_config.rmsnorm,
        )
