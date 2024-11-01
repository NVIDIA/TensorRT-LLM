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
import copy
import re
import time
from pathlib import Path
from typing import Union

import torch

import tensorrt_llm
from tensorrt_llm.models.convert_utils import (iterate_shard_files,
                                               load_state_dict)


def get_weight(config, prefix, dtype):
    return config[prefix + '.weight'].to(dtype).detach()


def get_bias(config, prefix, dtype):
    if (prefix + '.bias') in config:
        return config[prefix + '.bias'].to(dtype).detach()
    return None


def get_weight_and_bias(config, prefix, dtype_w, dtype_b):
    return get_weight(config, prefix,
                      dtype_w), get_bias(config, prefix, dtype_b)


def split(v, tp_size, idx, dim=0):
    assert v.shape[dim] % tp_size == 0
    split_size = v.shape[dim] // tp_size
    if tp_size == 1:
        return v
    return torch.split(v, split_size, dim=dim)[idx]


def rename_hf_to_tllm(name: str):
    """ Rename a HF parameter name by the corresponding TRT-LLM style name. """
    # remove model
    if 'model.' in name:
        name = name.replace('model.', '')

    # change layer name
    if 'embeddings.' in name:
        name = name.replace('embeddings', 'vocab_embedding')
    elif 'embedding.' in name:
        name = name.replace('embedding', 'vocab_embedding')
    norm_pattern = r'\d\.norm\.'
    if 'mixer.' in name:
        name = name.replace('mixer.', 'ssm.')
    elif re.search(norm_pattern, name):
        name = name.replace('norm.', 'input_layernorm.')
    elif 'norm_f.' in name:
        name = name.replace('norm_f.', 'ln_f.')

    # Parameter names in ssm layers
    if 'A_log' in name:
        name = name.replace('A_log', 'A')
    elif 'dt_proj.bias' in name:
        name = name.replace('dt_proj.bias', 'dt_bias')
    return name


def convert_hf_mamba(hf_mamba, dtype='float32'):
    weights = {}
    tik = time.time()

    model_params = dict(hf_mamba.named_parameters())
    dtype = getattr(torch, dtype)

    # Parameter names in mamba block
    for l in range(hf_mamba.config.num_hidden_layers):
        # ssm layer
        prefix = f'backbone.layers.{l}.mixer.'
        tllm_prex = f'backbone.layers.{l}.ssm.'
        for layer in ['conv1d', 'x_proj', 'dt_proj', 'out_proj']:
            dtype_b = torch.float32 if layer == 'dt_proj' else dtype
            weight, bias = get_weight_and_bias(model_params, prefix + layer,
                                               dtype, dtype_b)
            if layer == 'conv1d':
                weight = weight.unsqueeze(3)
            tllm_weight_name = tllm_prex + layer + '.weight'
            tllm_bias_name = tllm_prex + ('dt_bias' if layer == 'dt_proj' else
                                          layer + '.bias')
            weights[tllm_weight_name] = weight
            if bias is not None:
                weights[tllm_bias_name] = bias
        # in_proj
        weight, bias = get_weight_and_bias(model_params, prefix + 'in_proj',
                                           dtype, dtype)
        in_proj_weights = torch.split(weight, weight.size(0) // 2, dim=0)
        tllm_weight_name = tllm_prex + 'in_proj.weight'
        weights[tllm_weight_name.replace('proj', 'proj_x')] = in_proj_weights[0]
        weights[tllm_weight_name.replace('proj', 'proj_z')] = in_proj_weights[1]
        if bias is not None:
            in_proj_biases = torch.split(bias, bias.size(0) // 2, dim=0)
            tllm_bias_name = tllm_prex + 'in_proj.bias'
            weights[tllm_bias_name.replace('proj',
                                           'proj_x')] = in_proj_biases[0]
            weights[tllm_bias_name.replace('proj',
                                           'proj_x')] = in_proj_biases[1]

        # A and D
        Aparam = model_params[prefix + 'A_log'].float().detach()
        Aparam = Aparam.permute(1, 0).contiguous()
        weights[tllm_prex + 'A'] = -torch.exp(Aparam)
        weights[tllm_prex + 'D'] = model_params[prefix + 'D'].float().detach()
        # norm
        prefix = f'backbone.layers.{l}.norm'
        tllm_prex = f'backbone.layers.{l}.input_layernorm.'
        weight, bias = get_weight_and_bias(model_params, prefix, dtype, dtype)
        weights[tllm_prex + 'weight'] = weight
        if bias is not None:
            weights[tllm_prex + 'bias'] = bias

    # others
    for layer in ['backbone.embeddings', 'backbone.norm_f']:
        weight, bias = get_weight_and_bias(model_params, layer, dtype, dtype)
        layer = layer.replace('embeddings', 'vocab_embedding')
        layer = layer.replace('norm_f', 'ln_f')
        weights[layer + '.weight'] = weight
        if bias is not None:
            weights[layer + '.bias'] = bias
    weights['lm_head.weight'], _ = get_weight_and_bias(model_params,
                                                       'backbone.embeddings',
                                                       dtype, dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def convert_from_hf_checkpoint(mamba_config: dict, model_dir: Union[str, Path]):

    print('Loading weights from HF Mamba...')
    tik = time.time()

    tp_rank = mamba_config.mapping.tp_rank
    tp_size = mamba_config.mapping.tp_size
    d_inner = mamba_config.rnn_hidden_size
    d_state = mamba_config.state_size
    dtype = mamba_config.dtype
    mamba_version = mamba_config.mamba_version
    weights = {}
    if isinstance(dtype, str):
        dtype = tensorrt_llm.str_dtype_to_torch(dtype)

    for model_file in iterate_shard_files(model_dir, 0):
        # logger.debug(f'Loading file {str(model_file)}...')
        model_params = load_state_dict(model_file, dtype=dtype)
        for name, param in model_params.items():
            # logger.debug(f'Converting weight {name}...')
            tllm_name = rename_hf_to_tllm(name)
            param = param.detach().cpu()
            if 'A_log' in name:
                param = -torch.exp(param.float())
                if mamba_version == 'Mamba1':
                    param = param.permute(1, 0).contiguous()
            elif 'D' in name:
                param = param.float()
            elif 'dt_proj.bias' in name:
                param = param.float()
            elif 'dt_bias' in name:
                param = param.float()
            elif 'conv1d.weight' in name:
                param = param.unsqueeze(3)

            # split in_proj in Mamba1
            if 'in_proj' in name and mamba_version == 'Mamba1':
                in_proj_params = torch.split(param, param.size(0) // 2, dim=0)
                weights[tllm_name.replace('proj', 'proj_x')] = in_proj_params[0]
                weights[tllm_name.replace('proj', 'proj_z')] = in_proj_params[1]
            elif 'in_proj' in name and mamba_version == 'Mamba2':
                nheads = d_inner // mamba_config.rnn_head_size
                ngroups = mamba_config.ngroups

                in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt = torch.split(
                    param, [
                        d_inner, d_inner, ngroups * d_state, ngroups * d_state,
                        nheads
                    ],
                    dim=0)
                in_proj_z = split(in_proj_z, tp_size, tp_rank, dim=0)
                in_proj_x = split(in_proj_x, tp_size, tp_rank, dim=0)
                in_proj_b = split(in_proj_b, tp_size, tp_rank, dim=0)
                in_proj_c = split(in_proj_c, tp_size, tp_rank, dim=0)
                in_proj_dt = split(in_proj_dt, tp_size, tp_rank, dim=0)
                in_proj = torch.concat(
                    [in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt])
                weights[tllm_name] = in_proj.contiguous()
            elif 'conv1d' in name and mamba_version == 'Mamba2':
                ngroups = mamba_config.ngroups
                conv_x, conv_b, conv_c = torch.split(
                    param, [d_inner, ngroups * d_state, ngroups * d_state],
                    dim=0)
                conv_x = split(conv_x, tp_size, tp_rank, dim=0)
                conv_b = split(conv_b, tp_size, tp_rank, dim=0)
                conv_c = split(conv_c, tp_size, tp_rank, dim=0)
                conv = torch.concat([conv_x, conv_b, conv_c])
                weights[tllm_name] = conv.contiguous()
            elif any(keyword in name for keyword in (
                    'mixer.norm.weight',
                    'A_log',
                    'D',
                    'dt_proj.bias',
                    'dt_bias',
            )) and mamba_version == 'Mamba2':
                weights[tllm_name] = split(param, tp_size, tp_rank, dim=0)
            elif 'out_proj' in name and mamba_version == 'Mamba2':
                weights[tllm_name] = split(param, tp_size, tp_rank,
                                           dim=1).contiguous()
            else:
                weights[tllm_name] = param
        del model_params

    # lm_head
    emb = weights['backbone.vocab_embedding.weight']
    if 'lm_head.weight' not in weights or weights['lm_head.weight'].data_ptr(
    ) == emb.data_ptr():
        weights['lm_head.weight'] = copy.deepcopy(emb)
    if mamba_version == 'Mamba2':
        weights['lm_head.weight'] = split(weights['lm_head.weight'],
                                          tp_size,
                                          tp_rank,
                                          dim=0)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights
