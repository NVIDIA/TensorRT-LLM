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

from ...layers import MoeConfig
from ...logger import logger
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class GPTConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 gpt_variant: str = 'gpt2',
                 bias: bool = True,
                 q_scaling: float = 1.0,
                 embedding_scale: Optional[float] = None,
                 apply_query_key_layer_scaling: bool = False,
                 rotary_pct: float = 1.0,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 inner_layernorm: bool = False,
                 norm_before_bmm1: bool = False,
                 moe: Optional[Union[MoeConfig, dict]] = None,
                 **kwargs):
        self.gpt_variant = gpt_variant
        self.bias = bias
        self.q_scaling = q_scaling
        self.embedding_scale = embedding_scale
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.rotary_pct = rotary_pct
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.inner_layernorm = inner_layernorm
        self.norm_before_bmm1 = norm_before_bmm1
        if moe is None:
            # Legacy MOE config fields
            moe = MoeConfig(
                num_experts=kwargs.pop('moe_num_experts', 0),
                top_k=kwargs.pop('moe_top_k', 0),
                normalization_mode=kwargs.pop(
                    'moe_normalization_mode',
                    MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE))
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in GPTConfig
        output['gpt_variant'] = self.gpt_variant
        output['bias'] = self.bias
        output['q_scaling'] = self.q_scaling
        output['embedding_scale'] = self.embedding_scale
        output[
            'apply_query_key_layer_scaling'] = self.apply_query_key_layer_scaling
        output['rotary_pct'] = self.rotary_pct
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output['inner_layernorm'] = self.inner_layernorm
        output['norm_before_bmm1'] = self.norm_before_bmm1
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

        from .convert import get_needed_padding

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_or_dir, trust_remote_code=trust_remote_code)

        gpt_variant = kwargs.pop('gpt_variant', None)
        if gpt_variant is None:
            logger.info("Inferring gpt variant from path...")
            for v in [
                    'starcoder2', 'starcoder', 'santacoder', 'gpt2',
                    'persimmon', 'fuyu', 'kosmos-2', 'jais', 'nemotron'
            ]:
                if v in hf_config._name_or_path or v == hf_config.model_type:
                    gpt_variant = v
                    break
        if gpt_variant == 'fuyu':
            gpt_variant = 'persimmon'

        assert gpt_variant in [
            'gpt2', 'santacoder', 'starcoder', 'starcoder2', 'persimmon',
            'kosmos-2', 'jais', 'nemotron'
        ]
        logger.info(f"Gpt variant: {gpt_variant}")

        if gpt_variant in ['starcoder2', 'nemotron', 'persimmon']:
            hf_config.n_embd = hf_config.hidden_size
            hf_config.n_inner = hf_config.intermediate_size
            hf_config.n_head = hf_config.num_attention_heads
            hf_config.n_kv_head = hf_config.num_key_value_heads if hasattr(
                hf_config, 'num_key_value_heads') else hf_config.n_head
            hf_config.n_layer = hf_config.num_hidden_layers
            hf_config.n_positions = hf_config.max_position_embeddings
            hf_config.activation_function = 'gelu' if gpt_variant == 'starcoder2' else 'squared-relu'
            if gpt_variant == "nemotron":
                hf_config.layer_norm_eps = hf_config.norm_eps
            hf_config.layer_norm_epsilon = hf_config.norm_epsilon if gpt_variant == 'starcoder2' else hf_config.layer_norm_eps
            hf_config.bias = hf_config.use_bias if gpt_variant == 'starcoder2' else gpt_variant != 'nemotron'
            hf_config.position_embedding_type = 'rope_gpt_neox'
            hf_config.rotary_base = hf_config.rope_theta
            hf_config.rotary_pct = getattr(
                hf_config, 'partial_rotary_factor',
                getattr(hf_config, 'rope_percent', 1.0))
            try:
                # only for persimmon, not starcoder2
                hf_config.vocab_size = hf_config.text_config.vocab_size
            except AttributeError:
                pass
        elif gpt_variant == "kosmos-2":
            hf_config.n_embd = hf_config.text_config.embed_dim
            hf_config.n_inner = hf_config.text_config.ffn_dim
            hf_config.n_head = hf_config.text_config.attention_heads
            hf_config.n_kv_head = hf_config.n_head
            hf_config.n_layer = hf_config.text_config.layers
            hf_config.n_positions = hf_config.text_config.max_position_embeddings
            hf_config.activation_function = hf_config.text_config.activation_function
            hf_config.layer_norm_epsilon = hf_config.text_config.layer_norm_eps
            hf_config.bias = True
            hf_config.vocab_size = hf_config.text_config.vocab_size
        else:
            if hf_config.n_inner is None:
                hf_config.n_inner = hf_config.n_embd * 4
            if gpt_variant in ['santacoder', 'starcoder']:
                hf_config.n_kv_head = 1
            else:
                hf_config.n_kv_head = hf_config.n_head

        if gpt_variant == 'jais':
            hf_config.q_scaling = (hf_config.n_embd // hf_config.n_head)**0.5
            if hasattr(hf_config, 'width_scale'):
                hf_config.logits_scale = hf_config.width_scale
            else:
                hf_config.logits_scale = hf_config.mup_output_alpha * hf_config.mup_width_scale

            if hasattr(hf_config, 'mup_embeddings_scale'):
                hf_config.embeddings_scale = hf_config.mup_embeddings_scale
            else:
                assert hasattr(hf_config, 'embeddings_scale')

            hf_config.n_inner += get_needed_padding(hf_config.n_inner,
                                                    mapping.tp_size)

        if gpt_variant == 'kosmos-2':
            if hf_config.text_config.scale_embedding:
                hf_config.embeddings_scale = hf_config.n_embd**0.5

        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

        return cls(architecture=hf_config.architectures[0],
                   dtype=dtype,
                   num_hidden_layers=hf_config.n_layer,
                   num_attention_heads=hf_config.n_head,
                   num_key_value_heads=hf_config.n_kv_head,
                   hidden_size=hf_config.n_embd,
                   intermediate_size=hf_config.n_inner,
                   norm_epsilon=hf_config.layer_norm_epsilon,
                   vocab_size=hf_config.vocab_size,
                   position_embedding_type=getattr(hf_config,
                                                   'position_embedding_type',
                                                   'learned_absolute'),
                   max_position_embeddings=hf_config.n_positions,
                   hidden_act=hf_config.activation_function,
                   gpt_variant=gpt_variant,
                   bias=getattr(hf_config, 'bias', True),
                   apply_query_key_layer_scaling=getattr(
                       hf_config, 'apply_query_key_layer_scaling', False),
                   rotary_pct=getattr(hf_config, 'rotary_pct', 1.0),
                   rotary_base=getattr(hf_config, 'rotary_base', 10000.0),
                   rotary_scaling=getattr(hf_config, 'rotary_scaling', None),
                   qk_layernorm=gpt_variant == 'persimmon',
                   inner_layernorm=gpt_variant == 'kosmos-2',
                   norm_before_bmm1=gpt_variant == 'kosmos-2',
                   q_scaling=getattr(hf_config, 'q_scaling', 1),
                   embedding_scale=getattr(hf_config, 'embeddings_scale', None),
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)

    @classmethod
    def from_nemo(cls,
                  nemo_ckpt_dir: str,
                  dtype: str = 'auto',
                  mapping: Optional[Mapping] = None,
                  quant_config: Optional[QuantConfig] = None,
                  **kwargs):
        import transformers

        from .convert import (UnpackedNemoCheckpointDir, cpu_map_location,
                              gpu_map_location, rename_keys)

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        nemo_rename_key = kwargs.pop('nemo_rename_key', [])
        layer_rename_config = {
            pattern.split(':')[0]: pattern.split(':')[1]
            for pattern in nemo_rename_key
        }

        unpacked_checkpoints_dir = UnpackedNemoCheckpointDir(
            nemo_ckpt_dir, load_checkpoints_to_cpu=load_model_on_cpu)
        nemo_model_config = unpacked_checkpoints_dir.model_config

        training_tp_size = nemo_model_config.get("tensor_model_parallel_size",
                                                 1)
        training_pp_size = nemo_model_config.get("pipeline_model_parallel_size",
                                                 1)

        checkpoints_paths = unpacked_checkpoints_dir.get_checkpoints_paths(
            training_tp_size,
            training_pp_size,
        )
        if unpacked_checkpoints_dir._load_checkpoints_to_cpu:
            map_location_fn = cpu_map_location
        else:
            map_location_fn = gpu_map_location
        model_00 = torch.load(checkpoints_paths[0][0],
                              map_location=map_location_fn)
        model_00 = rename_keys(model_00, layer_rename_config)
        vocab_size = model_00[
            "model.language_model.embedding.word_embeddings.weight"].shape[
                0] * training_tp_size
        del model_00

        hf_config = transformers.GPT2Config(
            vocab_size=vocab_size,
            n_positions=nemo_model_config['max_position_embeddings'],
            n_embd=nemo_model_config['hidden_size'],
            n_layer=nemo_model_config['num_layers'],
            n_head=nemo_model_config['num_attention_heads'],
            n_inner=nemo_model_config['ffn_hidden_size'],
            activation_function=nemo_model_config['activation'],
            layer_norm_epsilon=nemo_model_config['layernorm_epsilon'],
        )
        hf_config.n_kv_head = hf_config.n_head
        hf_config.bias = nemo_model_config['bias']
        hf_config.apply_query_key_layer_scaling = False

        hf_config.position_embedding_type = nemo_model_config.get(
            'position_embedding_type', 'learned_absolute')
        if hf_config.position_embedding_type == 'rope':
            hf_config.position_embedding_type = 'rope_gpt_neox'
        hf_config.rotary_base = nemo_model_config.get('rotary_base', 10000.0)
        hf_config.rotary_pct = nemo_model_config.get('rotary_percentage', 1.0)
        assert hf_config.rotary_pct >= 0 and hf_config.rotary_pct <= 1

        rotary_scaling_factor = nemo_model_config.get(
            'seq_len_interpolation_factor', None)
        if rotary_scaling_factor is None:
            hf_config.rotary_scaling = None
        else:
            assert rotary_scaling_factor > 1
            hf_config.rotary_scaling = {
                'type': 'linear',
                'factor': rotary_scaling_factor
            }

        if dtype == 'auto':
            dtype = nemo_model_config.get('precision', None)
            if dtype is None:
                dtype = 'float16'
            elif 'bf16' in dtype or 'bfloat16' in dtype:
                dtype = 'bfloat16'
            else:
                dtype = 'float16'
            logger.info(f"Specified dtype 'auto'; inferred dtype {dtype!r}.")

        return cls(architecture='GPTForCausalLM',
                   dtype=dtype,
                   num_hidden_layers=hf_config.n_layer,
                   num_attention_heads=hf_config.n_head,
                   num_key_value_heads=hf_config.n_kv_head,
                   hidden_size=hf_config.n_embd,
                   intermediate_size=hf_config.n_inner,
                   norm_epsilon=hf_config.layer_norm_epsilon,
                   vocab_size=hf_config.vocab_size,
                   position_embedding_type=hf_config.position_embedding_type,
                   max_position_embeddings=hf_config.n_positions,
                   hidden_act=hf_config.activation_function,
                   bias=hf_config.bias,
                   apply_query_key_layer_scaling=hf_config.
                   apply_query_key_layer_scaling,
                   rotary_pct=hf_config.rotary_pct,
                   rotary_base=hf_config.rotary_base,
                   rotary_scaling=hf_config.rotary_scaling,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
