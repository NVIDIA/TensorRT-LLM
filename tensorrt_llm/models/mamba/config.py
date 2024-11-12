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
import os
from enum import Enum
from typing import List, Optional, Union

import transformers

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class CheckpointType(str, Enum):
    mistral_inference = "mistral_inference"
    state_spaces = "state_spaces"
    hf = "hf"


def get_ckpt_type(model_path):
    hf_config = transformers.AutoConfig.from_pretrained(model_path,
                                                        trust_remote_code=True)
    if hasattr(hf_config, "ssm_cfg") and hf_config.ssm_cfg:
        return CheckpointType.state_spaces
    if os.path.exists(os.path.join(model_path, "params.json")):
        return CheckpointType.mistral_inference
    return CheckpointType.hf


class MambaConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 residual_in_fp32: bool = True,
                 pad_vocab_size_multiple: int = -1,
                 layer_types: List[str] = ["recurrent"],
                 **kwargs):
        self.residual_in_fp32 = residual_in_fp32
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.layer_types = layer_types
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in MambaConfig

        return output

    def update(self, data_dict):
        self.__dict__.update(data_dict)

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        ckpt_type = get_ckpt_type(hf_config_or_dir)

        mamba_version = 'Mamba1'
        if ckpt_type == CheckpointType.hf:
            if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
                hf_config = hf_config_or_dir
            else:
                hf_config_dir = str(hf_config_or_dir)

                hf_config = transformers.AutoConfig.from_pretrained(
                    hf_config_dir, trust_remote_code=True)

            dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

            vocab_size = hf_config.vocab_size
            pad_vocab_size_multiple = getattr(hf_config,
                                              "pad_vocab_size_multiple", 1)
            if vocab_size % pad_vocab_size_multiple != 0:
                vocab_size += pad_vocab_size_multiple - (
                    vocab_size % pad_vocab_size_multiple)
            return cls(architecture="MambaForCausalLM",
                       dtype=dtype,
                       num_hidden_layers=hf_config.num_hidden_layers,
                       num_attention_heads=mapping.world_size,
                       hidden_size=hf_config.hidden_size,
                       intermediate_size=hf_config.intermediate_size,
                       vocab_size=vocab_size,
                       mamba_version=mamba_version,
                       hidden_act=hf_config.hidden_act,
                       rms_norm=hf_config.rms_norm,
                       residual_in_fp32=hf_config.residual_in_fp32,
                       pad_vocab_size_multiple=pad_vocab_size_multiple,
                       rnn_hidden_size=hf_config.intermediate_size,
                       rnn_conv_dim_size=hf_config.intermediate_size,
                       state_size=hf_config.state_size,
                       conv_kernel=hf_config.conv_kernel,
                       use_bias=hf_config.use_bias,
                       mapping=mapping,
                       quantization=quant_config,
                       **kwargs)
        elif ckpt_type == CheckpointType.state_spaces:

            mamba_version = 'Mamba2'
            if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
                hf_config = hf_config_or_dir
            else:
                hf_config_dir = str(hf_config_or_dir)

                hf_config = transformers.AutoConfig.from_pretrained(
                    hf_config_dir, trust_remote_code=True)

            dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

            vocab_size = hf_config.vocab_size
            pad_vocab_size_multiple = getattr(hf_config,
                                              "pad_vocab_size_multiple", 1)
            if vocab_size % pad_vocab_size_multiple != 0:
                vocab_size += pad_vocab_size_multiple - (
                    vocab_size % pad_vocab_size_multiple)
            assert hasattr(hf_config,
                           'ssm_cfg') and hf_config.ssm_cfg['layer'] == 'Mamba2'
            config = json.load(
                open(os.path.join(hf_config_or_dir, 'config.json')))
            ssm_cfg = config.pop('ssm_cfg')
            cfg_to_mamba_cfg = {
                'd_model': 'hidden_size',
                'n_layer': 'num_hidden_layers',
                'fused_add_norm': None,
                'tie_embeddings': None,
            }
            ssm_cfg_to_mamba_cfg = {
                'd_state': 'state_size',
                'd_conv': 'conv_kernel',
                'bias': 'use_bias',
                'headdim': 'head_dim',
                'ngroups': 'n_groups',
                'chunk_size': 'chunk_size',
                'rmsnorm': 'ssm_rmsnorm',
            }
            for k in cfg_to_mamba_cfg:
                if k in config:
                    v = config.pop(k)
                    if cfg_to_mamba_cfg[k] is not None:
                        config[cfg_to_mamba_cfg[k]] = v
            for k in ssm_cfg_to_mamba_cfg:
                if k in ssm_cfg and ssm_cfg_to_mamba_cfg[k] is not None:
                    config[ssm_cfg_to_mamba_cfg[k]] = ssm_cfg[k]

            if 'expand' in config:
                expand = config['expand']
                hf_config.intermediate_size = expand * config['hidden_size']
            else:
                hf_config.intermediate_size = 2 * config['hidden_size']
            mamba2_default_cfg = {
                'n_groups': 1,
                'hidden_size': hf_config.d_model,
                'head_dim': 64,
                'chunk_size': 256,
                'state_size': 128,
            }
            hf_config.update(mamba2_default_cfg)

            conv_dim = hf_config.intermediate_size + 2 * hf_config.n_groups * hf_config.state_size
            ssm_rmsnorm = getattr(hf_config, "ssm_rmsnorm", hf_config.rms_norm)
            mamba2_cfg = {
                'rnn_head_size': hf_config.head_dim,
                'rnn_conv_dim_size': conv_dim,
                'ngroups': hf_config.n_groups,
                'chunk_size': hf_config.chunk_size,
                'ssm_rmsnorm': ssm_rmsnorm,
            }
            hf_config.update(mamba2_cfg)

            return cls(architecture="MambaForCausalLM",
                       dtype=dtype,
                       num_hidden_layers=hf_config.n_layer,
                       num_attention_heads=mapping.world_size
                       if mapping is not None else 1,
                       hidden_size=hf_config.hidden_size,
                       intermediate_size=hf_config.intermediate_size,
                       vocab_size=vocab_size,
                       mamba_version=mamba_version,
                       hidden_act=hf_config.hidden_act,
                       rms_norm=hf_config.rms_norm,
                       residual_in_fp32=hf_config.residual_in_fp32,
                       pad_vocab_size_multiple=pad_vocab_size_multiple,
                       rnn_hidden_size=hf_config.intermediate_size,
                       rnn_conv_dim_size=hf_config.rnn_conv_dim_size,
                       state_size=hf_config.state_size,
                       conv_kernel=hf_config.conv_kernel,
                       use_bias=hf_config.use_bias,
                       mapping=mapping,
                       quantization=quant_config,
                       rnn_head_size=hf_config.rnn_head_size,
                       ngroups=hf_config.ngroups,
                       chunk_size=hf_config.chunk_size,
                       ssm_rmsnorm=hf_config.ssm_rmsnorm,
                       **kwargs)
        elif ckpt_type == CheckpointType.mistral_inference:
            mamba_version = 'Mamba2'

            config = json.load(
                open(os.path.join(hf_config_or_dir, 'params.json')))
            cfg_to_mamba_cfg = {
                'dim': 'hidden_size',
                'n_layers': 'num_hidden_layers',
                'n_groups': 'n_groups',
                'fused_add_norm': None,
                'tie_embeddings': None,
                'model_type': None,
            }
            for k in cfg_to_mamba_cfg:
                if k in config:
                    v = config.pop(k)
                    if cfg_to_mamba_cfg[k] is not None:
                        config[cfg_to_mamba_cfg[k]] = v

            config['architecture'] = 'MambaForCuasualLM'
            config['dtype'] = dtype
            config['num_attention_heads'] = mapping.world_size

            hf_config = MambaConfig(**config)
            mamba2_default_cfg = {
                'n_groups': 8,
                'hidden_size': 4096,
                'head_dim': 64,
                'chunk_size': 256,
                'state_size': 128,
                'conv_kernel': 4,
                'use_bias': False
            }

            hf_config.update(mamba2_default_cfg)
            conv_dim = hf_config.intermediate_size + 2 * hf_config.n_groups * hf_config.state_size
            ssm_rmsnorm = getattr(hf_config, "ssm_rmsnorm", hf_config.rms_norm)
            mamba2_cfg = {
                'rnn_head_size': hf_config.head_dim,
                'rnn_conv_dim_size': conv_dim,
                'ngroups': hf_config.n_groups,
                'chunk_size': hf_config.chunk_size,
                'ssm_rmsnorm': ssm_rmsnorm,
            }
            hf_config.update(mamba2_cfg)

            if 'expand' in config:
                expand = config['expand']
                hf_config.intermediate_size = expand * hf_config.hidden_size
            else:
                hf_config.intermediate_size = 2 * hf_config.hidden_size
            vocab_size = hf_config.vocab_size
            pad_vocab_size_multiple = getattr(hf_config,
                                              "pad_vocab_size_multiple", 1)
            if vocab_size % pad_vocab_size_multiple != 0:
                vocab_size += pad_vocab_size_multiple - (
                    vocab_size % pad_vocab_size_multiple)

            return cls(
                architecture="MambaForCausalLM",
                dtype=dtype,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=mapping.world_size,
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                #    num_key_value_heads=num_key_value_heads,
                vocab_size=vocab_size,
                mamba_version=mamba_version,
                hidden_act=hf_config.hidden_act,
                rms_norm=hf_config.rms_norm,
                residual_in_fp32=hf_config.residual_in_fp32,
                pad_vocab_size_multiple=pad_vocab_size_multiple,
                rnn_hidden_size=hf_config.intermediate_size,
                rnn_conv_dim_size=hf_config.rnn_conv_dim_size,
                state_size=hf_config.state_size,
                conv_kernel=hf_config.conv_kernel,
                use_bias=hf_config.use_bias,
                mapping=mapping,
                quantization=quant_config,
                rnn_head_size=hf_config.rnn_head_size,
                ngroups=hf_config.n_groups,
                chunk_size=hf_config.chunk_size,
                ssm_rmsnorm=hf_config.ssm_rmsnorm,
                **kwargs)
        else:
            pass

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True)

        vocab_size = hf_config.vocab_size
        pad_vocab_size_multiple = getattr(hf_config, "pad_vocab_size_multiple",
                                          1)
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size %
                                                     pad_vocab_size_multiple)
