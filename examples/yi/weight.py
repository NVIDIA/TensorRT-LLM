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
import time

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.mapping import Mapping

def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size,
                  -1)[:, None, :, :].expand(num_head, reps, head_size,
                                            v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone()

def load_from_hf_yi(tensorrt_llm_yi: tensorrt_llm.models.YiForCausalLM,
                       hf_yi,
                       mapping,
                       dtype):
    tensorrt_llm.logger.info('Loading weights from HF Yi...')
    tik = time.time()

    num_key_value_heads = tensorrt_llm_yi.num_key_value_heads
    mha_mode = (num_key_value_heads == tensorrt_llm_yi.num_attention_heads)

    model_params = dict(hf_yi.named_parameters())
    for l in range(hf_yi.config.num_hidden_layers):
        prefix = f'model.layers.{l}.self_attn.'
        q_weight = model_params[prefix + 'q_proj.weight']
        k_weight = model_params[prefix + 'k_proj.weight']
        v_weight = model_params[prefix + 'v_proj.weight']
        if not mha_mode:
            head_size = tensorrt_llm_yi.hidden_size // tensorrt_llm_yi.num_attention_heads
            if num_key_value_heads < mapping.tp_size:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_key_value_heads,
                                         mapping.tp_size)
                v_weight = dup_kv_weight(v_weight, num_key_value_heads,
                                         mapping.tp_size)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            qkv_weight = [q_weight, k_weight, v_weight]
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        model_params[prefix + 'qkv_proj.weight'] = qkv_weight
        
    torch_dtype = str_dtype_to_torch(dtype)
    layers_per_pipeline_stage = hf_yi.config.num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
    for k, v in model_params.items():
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
        if 'model.embed_tokens.weight' in k:
            if tensorrt_llm_yi.use_parallel_embedding:
                v = split(v, mapping.tp_size, mapping.tp_rank,
                          tensorrt_llm_yi.embedding_sharding_dim)
            if mapping.is_first_pp_rank():
                tensorrt_llm_yi.embed_tokens.weight.value = v
        elif 'model.norm.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_yi.norm.weight.value = v
        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_yi.lm_head.weight.value = np.ascontiguousarray(
                    split(v, mapping.tp_size, mapping.tp_rank))
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue
            idx = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
            if idx >= tensorrt_llm_yi.num_hidden_layers:
                continue
            if 'ln1.weight' in k:
                tensorrt_llm_yi.layers[idx].ln1.weight.value = v
            elif 'ln2.weight' in k:
                dst = tensorrt_llm_yi.layers[idx].ln2.weight
                dst.value = v
            elif 'self_attn.qkv_proj.weight' in k:
                dst = tensorrt_llm_yi.layers[idx].self_attn.qkv.weight
                if not mha_mode:
                    assert isinstance(v, list) and len(v) == 3
                    wq = split(v[0], mapping.tp_size, mapping.tp_rank)
                    wk = split(v[1], mapping.tp_size, mapping.tp_rank)
                    wv = split(v[2], mapping.tp_size, mapping.tp_rank)
                    split_v = np.concatenate((wq, wk, wv))
                else:
                    q_emb = v.shape[0] // 3
                    model_emb = v.shape[1]
                    v = v.reshape(3, q_emb, model_emb)
                    split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size),
                                              model_emb)
                dst.value = np.ascontiguousarray(split_v)
            elif 'self_attn.o_proj.weight' in k:
                dst = tensorrt_llm_yi.layers[idx].self_attn.dense.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.up_proj.weight' in k:
                dst = tensorrt_llm_yi.layers[idx].mlp.gate.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.down_proj.weight' in k:
                dst = tensorrt_llm_yi.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                dst.value = np.ascontiguousarray(split_v)
            elif 'mlp.gate_proj.weight' in k:
                dst = tensorrt_llm_yi.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                dst.value = np.ascontiguousarray(split_v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
