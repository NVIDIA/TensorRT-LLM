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
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import infer_dtype
from tensorrt_llm.models.modeling_utils import PretrainedConfig, QuantConfig
from tensorrt_llm.models.nemotron_nas.convert import \
    hf_block_configs_to_layer_configs
from tensorrt_llm.models.nemotron_nas.layer_config import (
    AttentionConfig, AttentionImplementation, DeciLayerConfig, FFNConfig)


class DeciConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 architecture: str = 'DeciLMForCausalLM',
                 dtype: str,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 vocab_size: int,
                 hidden_act: str = 'gelu',
                 logits_dtype: str = 'float32',
                 norm_epsilon: float = 0.00001,
                 position_embedding_type: Union[
                     PositionEmbeddingType,
                     str] = PositionEmbeddingType.rope_gpt_neox,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 max_position_embeddings: int,
                 num_key_value_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 mapping: Optional[Union[Mapping, dict]] = None,
                 quantization: Optional[Union[QuantConfig, dict]] = None,
                 use_parallel_embedding: bool = False,
                 embedding_sharding_dim: int = 0,
                 head_size: Optional[int] = None,
                 qk_layernorm: bool = False,
                 layer_configs: Optional[List[Union[DeciLayerConfig,
                                                    Dict[str,
                                                         Dict[str,
                                                              Any]]]]] = None,
                 block_configs: Optional[object] = None,
                 **kwargs):
        super().__init__(architecture=architecture,
                         dtype=dtype,
                         hidden_size=hidden_size,
                         num_hidden_layers=num_hidden_layers,
                         num_attention_heads=num_attention_heads,
                         vocab_size=vocab_size,
                         hidden_act=hidden_act,
                         logits_dtype=logits_dtype,
                         norm_epsilon=norm_epsilon,
                         position_embedding_type=position_embedding_type,
                         max_position_embeddings=max_position_embeddings,
                         num_key_value_heads=num_key_value_heads,
                         intermediate_size=intermediate_size,
                         mapping=mapping,
                         quantization=quantization,
                         use_parallel_embedding=use_parallel_embedding,
                         embedding_sharding_dim=embedding_sharding_dim,
                         head_size=head_size,
                         qk_layernorm=qk_layernorm,
                         **kwargs)

        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling

        if block_configs is not None:
            assert layer_configs is None
            self.layer_configs = hf_block_configs_to_layer_configs(
                block_configs,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size)
        elif layer_configs is not None:
            assert len(
                layer_configs
            ) == num_hidden_layers, f"num_hidden_layers ({num_hidden_layers}) must match len(layer_configs) ({len(layer_configs)})"

            self.layer_configs = self._ensure_layer_configs(layer_configs)
        else:
            self.layer_configs = None

        # HACK: this is needed for many parts of the code
        self.layer_types = [
            AttentionImplementation(
                self.get_layer_config(layer_idx).attention.impl).value
            for layer_idx in range(self.num_hidden_layers)
        ]

        # HACK: this is here since the runtime doesn't parse the layer_configs yet
        self.num_kv_heads_per_layer = []
        for layer_idx in range(self.num_hidden_layers):
            layer_config = self.get_layer_config(layer_idx)
            if layer_config.is_attention_layer:
                self.num_kv_heads_per_layer.append(
                    layer_config.attention.num_key_value_heads)

    def _ensure_layer_configs(
        self, layer_configs: List[Union[DeciLayerConfig, Dict[str, Any]]]
    ) -> List[DeciLayerConfig]:
        return [
            DeciLayerConfig.from_dict(c) if isinstance(c, dict) else c
            for c in layer_configs
        ]

    def to_dict(self):
        output = super().to_dict()
        if self.layer_configs is not None:
            output["layer_configs"] = [asdict(c) for c in self.layer_configs]
        return output

    def get_layer_config(self, layer_idx: int) -> DeciLayerConfig:
        if self.layer_configs is not None:
            conf = self.layer_configs[layer_idx]
        else:
            conf = DeciLayerConfig()

        attention_impl = conf.attention.impl
        num_key_value_heads = conf.attention.num_key_value_heads or self.num_key_value_heads
        ffn_impl = conf.ffn.impl
        intermediate_size = conf.ffn.intermediate_size or self.intermediate_size

        return DeciLayerConfig(
            attention=AttentionConfig(impl=attention_impl,
                                      num_key_value_heads=num_key_value_heads),
            ffn=FFNConfig(impl=ffn_impl, intermediate_size=intermediate_size))

    def get_layer_num_kv_heads(self, layer_idx) -> int:
        layer_config = self.get_layer_config(layer_idx)
        assert layer_config.is_attention_layer, f"Layer {layer_idx} is not an attention layer"
        return layer_config.attention.num_key_value_heads or self.num_key_value_heads

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            trust_remote_code: bool = True,
            **kwargs):
        import transformers

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_or_dir, trust_remote_code=trust_remote_code)

        assert hf_config.model_type in (
            "deci",
            "nemotron-nas"), f"Unsupported model type: {hf_config.model_type}"

        block_configs = getattr(hf_config, "block_configs", None)
        if block_configs is not None:
            layer_configs = hf_block_configs_to_layer_configs(
                block_configs,
                num_attention_heads=hf_config.num_attention_heads,
                hidden_size=hf_config.hidden_size)
        else:
            # older deci arch
            num_key_value_heads_per_layer = getattr(
                hf_config, "num_key_value_heads_per_layer", None)
            if num_key_value_heads_per_layer is not None:
                layer_configs = [
                    DeciLayerConfig(attention=AttentionConfig(
                        num_key_value_heads=num_key_value_heads))
                    for num_key_value_heads in num_key_value_heads_per_layer
                ]
            else:
                layer_configs = None

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        return cls(dtype=dtype,
                   hidden_size=hf_config.hidden_size,
                   hidden_act=hf_config.hidden_act,
                   intermediate_size=hf_config.intermediate_size,
                   num_attention_heads=hf_config.num_attention_heads,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_key_value_heads=hf_config.num_key_value_heads,
                   norm_epsilon=hf_config.rms_norm_eps,
                   rotary_scaling=hf_config.rope_scaling,
                   rotary_base=hf_config.rope_theta,
                   vocab_size=hf_config.vocab_size,
                   max_position_embeddings=hf_config.max_position_embeddings,
                   mapping=mapping,
                   quantization=quant_config,
                   layer_configs=layer_configs,
                   **kwargs)
