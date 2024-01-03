# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import configparser
import time
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_np
from tensorrt_llm.models import BloomForCausalLM
from tensorrt_llm.quantization import QuantMode


def check_embedding_share(dir_path):
    share_embedding_table = False
    if Path(dir_path).exists():
        share_embedding_table = True
    return share_embedding_table


def get_att_time_mix_params(config, prefix, dtype):
    names = {
        'time_decay': torch.float32,
        'time_faaaa': torch.float32, 
        'time_mix_gate': dtype, 
        'time_mix_key': dtype, 
        'time_mix_value': dtype, 
        'time_mix_receptance': dtype
    }
    return [config[prefix + name].to(type).detach().cpu().numpy() for name, type in names.items()]


def get_ffn_time_mix_params(config, prefix, dtype):
    names = ['time_mix_key', 'time_mix_receptance']
    return [config[prefix + name].to(dtype).detach().cpu().numpy() for name in names]


def get_weight(config, prefix, dtype):
    return config[prefix + '.weight'].to(dtype).detach().cpu().numpy()


def get_bias(config, prefix, dtype):
    return config[prefix + '.bias'].to(dtype).detach().cpu().numpy()


def get_weight_and_bias(config, prefix, dtype):
    return get_weight(config, prefix, dtype), get_bias(config, prefix, dtype)


def load_from_hf_rwkv(tensorrt_llm_rwkv,
                       hf_rwkv,
                       rank=0,
                       tensor_parallel=1,
                       fp16=False,
                       use_parallel_embedding=False,
                       sharding_dim=0,
                       share_embedding_table=False):
    tensorrt_llm.logger.info('Loading weights from HF RWKV...')
    
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_rwkv, 'quant_mode', QuantMode(0))
    
    model_params = dict(hf_rwkv.named_parameters())
    dtype = torch.float16 if fp16 else torch.float32
    
    for l in range(hf_rwkv.config.num_hidden_layers):
        prefix = f'rwkv.blocks.{l}.'
        
        # TODO?
        # rwkv.blocks.0.pre_ln.weight torch.Size([2048])
        # rwkv.blocks.0.pre_ln.bias torch.Size([2048])

        # get the weight and bias for layernorm first
        ln_att_weight, ln_att_bias = get_weight_and_bias(
            model_params, prefix + 'ln1', dtype)
        tensorrt_llm_rwkv.layers[l].attention.ln_w.value = ln_att_weight
        tensorrt_llm_rwkv.layers[l].attention.ln_b.value = ln_att_bias
        ln_ffn_weight, ln_ffn_bias = get_weight_and_bias(
            model_params, prefix + 'ln2', dtype)
        tensorrt_llm_rwkv.layers[l].feed_forward.ln_w.value = ln_ffn_weight
        tensorrt_llm_rwkv.layers[l].feed_forward.ln_b.value = ln_ffn_bias
      
        # get weights for attention layer
        prefix += 'attention.'
        time_decay, time_first, time_mix_g, time_mix_k, time_mix_v, time_mix_r \
            = get_att_time_mix_params(model_params, prefix, dtype)
        att_k = get_weight(model_params, prefix + 'key', dtype)
        att_v = get_weight(model_params, prefix + 'value', dtype)
        att_r = get_weight(model_params, prefix + 'receptance', dtype)
        att_g = get_weight(model_params, prefix + 'gate', dtype)
        att_o = get_weight(model_params, prefix + 'output', dtype)
        lx_weight, lx_bias = get_weight_and_bias(
            model_params, prefix + 'ln_x', torch.float32)
        tensorrt_llm_rwkv.layers[l].attention.t_decay.value = time_decay
        tensorrt_llm_rwkv.layers[l].attention.t_first.value = time_first
        tensorrt_llm_rwkv.layers[l].attention.g_mix.value = time_mix_g.squeeze()
        tensorrt_llm_rwkv.layers[l].attention.k_mix.value = time_mix_k.squeeze()
        tensorrt_llm_rwkv.layers[l].attention.v_mix.value = time_mix_v.squeeze()
        tensorrt_llm_rwkv.layers[l].attention.r_mix.value = time_mix_r.squeeze()
        # TODO(Rinne): deal with tensor parallel
        tensorrt_llm_rwkv.layers[l].attention.kxw.weight.value = att_k
        tensorrt_llm_rwkv.layers[l].attention.vxw.weight.value = att_v
        tensorrt_llm_rwkv.layers[l].attention.rxw.weight.value = att_r
        tensorrt_llm_rwkv.layers[l].attention.gxw.weight.value = att_g
        tensorrt_llm_rwkv.layers[l].attention.oxw.weight.value = att_o
        tensorrt_llm_rwkv.layers[l].attention.lx_w.value = lx_weight
        tensorrt_llm_rwkv.layers[l].attention.lx_b.value = lx_bias
        
        # get weights for feed forward layer
        prefix = f'rwkv.blocks.{l}.feed_forward.'
        time_mix_k, time_mix_r = get_ffn_time_mix_params(model_params, prefix, dtype)
        ffn_k = get_weight(model_params, prefix + 'key', dtype)
        ffn_r = get_weight(model_params, prefix + 'receptance', dtype)
        ffn_v = get_weight(model_params, prefix + 'value', dtype)
        tensorrt_llm_rwkv.layers[l].feed_forward.k_mix.value = time_mix_k.squeeze()
        tensorrt_llm_rwkv.layers[l].feed_forward.r_mix.value = time_mix_r.squeeze()
        tensorrt_llm_rwkv.layers[l].feed_forward.kxw.weight.value = ffn_k.squeeze()
        tensorrt_llm_rwkv.layers[l].feed_forward.rxw.weight.value = ffn_r.squeeze()
        tensorrt_llm_rwkv.layers[l].feed_forward.vxw.weight.value = ffn_v
        
    # get weights for model output processing
    ln_out_weight, ln_out_bias = get_weight_and_bias(
            model_params, 'rwkv.ln_out', dtype)
    head_weight = get_weight(model_params, 'head', dtype)
    tensorrt_llm_rwkv.ln_out_w.value = ln_out_weight
    tensorrt_llm_rwkv.ln_out_b.value = ln_out_bias
    tensorrt_llm_rwkv.head_layer.weight.value = head_weight
    
    # get the word embedding weight
    embed_w = get_weight(model_params, 'rwkv.embeddings', dtype)
    if not share_embedding_table:
        # TODO(Rinne): deal with tp
        tensorrt_llm_rwkv.lm_head.weight.value = embed_w.copy()
    
    tensorrt_llm_rwkv.vocab_embedding.weight.value = embed_w
    
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    
        
    